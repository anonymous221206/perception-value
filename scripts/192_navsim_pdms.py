#!/usr/bin/env python
"""Follow-up E2: PDMS of PDM-Closed on the reference, cheap and full objects of every NAVSIM token (addendum E2 in docs/followup_protocol.md).

Runs on the machine that holds the NAVSIM data, in its navsim environment (devkit v1.1, nuPlan devkit, maps):
  objects   reference = the token's annotations. cheap / full: eligible annotations (vehicle, pedestrian, bicycle)
            projected into CAM_F0 (annotation boxes are in the LiDAR frame of the camera extrinsics); Hungarian
            matching against the mode's detections at the operating threshold 0.25 (class-compatible, IoU >= 0.3);
            unmatched in-camera eligible annotations removed; unmatched detections added as static objects lifted on
            the ground plane (box bottom centre, class-median sizes of the navtest annotations, heading along the
            viewing ray, zero velocity); everything outside CAM_F0 unchanged.
  planner   PDM-Closed exactly as NAVSIM's MetricCacheProcessor builds it, on the modified scene.
  score     NAVSIM's pdm_score against the token's metric cache, default scoring parameters; the trajectory handed over
            as NAVSIM's Trajectory (8 poses at 0.5 s, ego-local).
Detections: this repository's 191 (data/cache/navsim/dets/<log>.pkl). Stages:
  --stage priors   class-median sizes and the ground-plane height from the navtest annotations
  --stage run      per-log rows -> $OUT/<log>.pkl (shardable: --shard i/n)
  --stage collect  every row -> $OUT/../pdms_rows_<tag>.csv
Environment: OPENSCENE_DATA_ROOT, NAVSIM_DEVKIT_ROOT, NUPLAN_MAPS_ROOT, DEEP_NAVSIM (folder with dets/ and the outputs),
NAVSIM_CACHE (metric cache folder), NAVSIM_SPLIT (test | mini).
"""
from __future__ import annotations

import argparse
import dataclasses
import lzma
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from scipy.optimize import linear_sum_assignment

from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import Annotations, SceneFilter, SensorConfig, Trajectory
from navsim.common.dataloader import SceneLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

DATA = Path(os.environ["OPENSCENE_DATA_ROOT"])
DEV = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
D = Path(os.environ["DEEP_NAVSIM"])
SPLIT = os.environ.get("NAVSIM_SPLIT", "test")
CACHE = Path(os.environ["NAVSIM_CACHE"])
TAG = os.environ.get("DEEP_TAG", SPLIT)
DETS, OUT = D / "dets", D / f"pdms_{TAG}"
W, H = 1920, 1080
OP_CONF = 0.25
BRANCHES = tuple(os.environ.get("DEEP_BRANCHES", "reference,cheap,full").split(","))
# gate E2-0 switches: the trajectory handover interval (the addendum's 0.5 s x 8 poses by default) and a token list
HANDOVER_S = float(os.environ.get("DEEP_HANDOVER_S", "0.5"))
TOKENS_FILE = os.environ.get("DEEP_TOKENS")         # CSV with log, token: score these tokens (reference branch only)
ELIGIBLE = ("vehicle", "pedestrian", "bicycle")
COMPAT = {"vehicle": {"vehicle"}, "pedestrian": {"person"}, "bicycle": {"cyclist", "person"}}
DET_TO_NAME = {"vehicle": "vehicle", "person": "pedestrian", "cyclist": "bicycle"}
SIGNS = np.array([[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], float)
EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8) if int(np.sum(SIGNS[i] != SIGNS[j])) == 1]
NEAR, CLAMP, IOU_THR, FP_MIN_AHEAD, FP_MAX_RANGE = 0.1, 1.25, 0.3, 0.5, 80.0


class Cam:
    """Projection / ground lift for OpenScene CAM_F0 (camera -> LiDAR frame R, t; OpenCV distortion)."""

    def __init__(self, K, D, R, t):
        import cv2
        self.K, self.D, self.R, self.t = (np.asarray(a, float) for a in (K, D, R, t))
        border = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1], [0, H / 2], [W - 1, H / 2], [W / 2, 0],
                           [W / 2, H - 1]], np.float64)
        und = cv2.undistortPoints(border.reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)
        self.bx, self.by = CLAMP * float(np.abs(und[:, 0]).max()), CLAMP * float(np.abs(und[:, 1]).max())

    def distort(self, xn, yn):
        k1, k2, p1, p2, k3 = self.D
        r2 = xn * xn + yn * yn
        rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        return self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]

    def project(self, box):
        """box (x, y, z, l, w, h, heading) in the LiDAR frame -> clipped image box or None (as nuplan_spec_real)."""
        x, y, z, l, w, h, yaw = box
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        pc = (np.array([x, y, z]) + (SIGNS * [l / 2, w / 2, h / 2]) @ Rz.T - self.t) @ self.R
        if (np.array([x, y, z]) - self.t) @ self.R[:, 2] <= 0:
            return None
        pts = [p for p in pc if p[2] > NEAR]
        for i, j in EDGES:
            zi, zj = pc[i, 2], pc[j, 2]
            if (zi - NEAR) * (zj - NEAR) < 0:
                a = (NEAR - zi) / (zj - zi)
                pts.append(pc[i] + a * (pc[j] - pc[i]))
        if not pts:
            return None
        P = np.asarray(pts)
        u, v = self.distort(np.clip(P[:, 0] / P[:, 2], -self.bx, self.bx), np.clip(P[:, 1] / P[:, 2], -self.by, self.by))
        x1, y1, x2, y2 = max(u.min(), 0.0), max(v.min(), 0.0), min(u.max(), W - 1.0), min(v.max(), H - 1.0)
        return None if x2 <= x1 or y2 <= y1 else (x1, y1, x2, y2)

    def ground_point(self, u, v, gz):
        import cv2
        xn, yn = cv2.undistortPoints(np.array([[[u, v]]], np.float64), self.K, self.D).reshape(2)
        d = self.R @ np.array([xn, yn, 1.0])
        if d[2] >= -1e-9:
            return None
        s = (gz - self.t[2]) / d[2]
        if s <= 0:
            return None
        g = self.t + s * d
        return g[:2], d[:2] / max(np.hypot(d[0], d[1]), 1e-9)


def iou(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])                    # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def branch_annotations(ann: Annotations, det, cam: Cam, gz, priors, tag):
    """The branch's objects as NAVSIM Annotations (rule in the module docstring)."""
    boxes, names = np.asarray(ann.boxes, float), list(ann.names)
    elig = [i for i, n in enumerate(names) if n in ELIGIBLE]
    proj = {i: cam.project(boxes[i]) for i in elig}
    inc = [i for i in elig if proj[i] is not None]
    dx, dc = np.asarray(det["xyxy"], float).reshape(-1, 4), np.asarray(det["coarse"]).astype(str)
    matched, used = set(), set()
    if inc and len(dx):
        M = iou([proj[i] for i in inc], dx)
        for a, i in enumerate(inc):
            M[a, [j for j in range(len(dx)) if dc[j] not in COMPAT[names[i]]]] = 0
        r, c = linear_sum_assignment(-M)
        for a, j in zip(r, c):
            if M[a, j] >= IOU_THR:
                matched.add(inc[a]); used.add(j)
    keep = [i for i in range(len(names)) if not (i in inc and i not in matched)]
    nb, nn = [boxes[i] for i in keep], [names[i] for i in keep]
    nv = [np.asarray(ann.velocity_3d)[i] for i in keep]
    nt = [ann.instance_tokens[i] for i in keep]; nk = [ann.track_tokens[i] for i in keep]
    for j in range(len(dx)):
        if j in used:
            continue
        b = dx[j]
        gp = cam.ground_point(0.5 * (b[0] + b[2]), b[3], gz)
        if gp is None:
            continue
        (gx, gy), ray = gp
        if gx < FP_MIN_AHEAD or np.hypot(gx, gy) > FP_MAX_RANGE:
            continue
        name = DET_TO_NAME[dc[j]]
        L, Wd, Ht = priors[name]
        cx, cy = gx + ray[0] * L / 2, gy + ray[1] * L / 2
        nb.append(np.array([cx, cy, gz + Ht / 2, L, Wd, Ht, float(np.arctan2(ray[1], ray[0]))]))
        nn.append(name); nv.append(np.zeros(3)); tok = f"fp:{tag}:{j}"; nt.append(tok); nk.append(tok)
    return Annotations(boxes=np.asarray(nb, np.float32).reshape(-1, 7), names=nn,
                       velocity_3d=np.asarray(nv, np.float32).reshape(-1, 3), instance_tokens=nt, track_tokens=nk)


def plan(proc: MetricCacheProcessor, scene):
    scenario = NavSimScenario(scene, map_root=os.environ["NUPLAN_MAPS_ROOT"], map_version="nuplan-maps-v1.0")
    pi, init = proc._get_planner_inputs(scenario)
    proc._pdm_closed.initialize(init)
    traj = proc._pdm_closed.compute_planner_trajectory(pi)
    ego = scenario.initial_ego_state
    ts = TrajectorySampling(time_horizon=4, interval_length=HANDOVER_S)
    states = [traj.get_state_at_time(TimePoint(int(ego.time_point.time_us + (k + 1) * HANDOVER_S * 1e6))).rear_axle
              for k in range(ts.num_poses)]
    rel = absolute_to_relative_poses([ego.rear_axle] + states)[1:]
    return Trajectory(np.array([[p.x, p.y, p.heading] for p in rel], np.float32), ts)


def thresholded(det):
    k = np.asarray(det["conf"], float) >= OP_CONF
    return {"xyxy": np.asarray(det["xyxy"], float)[k], "coarse": np.asarray(det["coarse"]).astype(str)[k]}


def run(args):
    si, sn = (int(x) for x in args.shard.split("/"))
    params = OmegaConf.load(DEV / "navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml")
    simulator, scorer = instantiate(params.simulator), instantiate(params.scorer)
    mcp = {p.parent.name: p for p in CACHE.glob("*/*/*/metric_cache.pkl")}
    gz = float((D / "ground_z.txt").read_text())
    priors = pickle.load(open(D / "size_priors.pkl", "rb"))
    OUT.mkdir(parents=True, exist_ok=True)
    if TOKENS_FILE:
        assert BRANCHES == ("reference",), "a token list scores the reference branch only"
        tl = pd.read_csv(TOKENS_FILE)
        by_log = {l: list(g.token) for l, g in tl.groupby("log")}
    else:
        by_log = None
    logs = sorted(by_log) if by_log else sorted(p.stem for p in DETS.glob("*.pkl"))
    for li, log in enumerate(logs):
        if li % sn != si or (OUT / f"{log}.pkl").exists():
            continue
        dets = None if by_log else pickle.load(open(DETS / f"{log}.pkl", "rb"))
        toks = [t for t in (by_log[log] if by_log else dets) if t in mcp]
        if not toks:
            continue
        flt = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                          log_names=[log], tokens=toks)
        sl = SceneLoader(DATA / "navsim_logs" / SPLIT, DATA / "sensor_blobs" / SPLIT, flt, SensorConfig.build_no_sensors())
        proc = MetricCacheProcessor(cache_path=None, force_feature_computation=False)
        rows = []
        for tok in sl.tokens:
            with lzma.open(mcp[tok], "rb") as fh:
                mc = pickle.load(fh)
            scene = sl.get_scene_from_token(tok)
            idx = scene.scene_metadata.num_history_frames - 1
            cam = Cam(*dets[tok]["cam"]) if dets else None
            row = {"log": log, "token": tok}
            for br in BRANCHES:
                try:
                    if br != "reference":
                        fr = dataclasses.replace(scene.frames[idx], annotations=branch_annotations(
                            scene.frames[idx].annotations, thresholded(dets[tok][br]), cam, gz, priors, br))
                        frames = list(scene.frames); frames[idx] = fr
                        sc = dataclasses.replace(scene, frames=frames)
                    else:
                        sc = scene
                    res = pdm_score(mc, plan(proc, sc), simulator.proposal_sampling, simulator, scorer)
                    row.update({f"{k}_{br}": float(v) for k, v in dataclasses.asdict(res).items()})
                except Exception as e:                                                       # noqa: BLE001
                    row[f"err_{br}"] = type(e).__name__
            rows.append(row)
        with open(OUT / f"{log}.pkl", "wb") as fh:
            pickle.dump(rows, fh)
        print(f"  [{li + 1}/{len(logs)}] {log}: {len(rows)} tokens", flush=True)
    print("DEEP NAVSIM PDMS SHARD DONE")


def collect():
    d = pd.DataFrame([r for f in sorted(OUT.glob("*.pkl")) for r in pickle.load(open(f, "rb"))])
    d.to_csv(D / f"pdms_rows_{TAG}.csv", index=False)
    print(len(d), "rows ->", D / f"pdms_rows_{TAG}.csv")


def priors():
    """Class-median box sizes and the ground-plane height (median bottom of vehicle boxes within 30 m), navtest."""
    f = yaml.safe_load((DEV / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml").read_text())
    acc, bottoms = {n: [] for n in ELIGIBLE}, []
    for log in sorted(set(f["log_names"])):
        for fr in pickle.load(open(DATA / "navsim_logs" / "test" / f"{log}.pkl", "rb")):
            b, n = np.asarray(fr["anns"]["gt_boxes"], float).reshape(-1, 7), np.asarray(fr["anns"]["gt_names"])
            for name in ELIGIBLE:
                acc[name] += [tuple(x) for x in b[n == name][:, 3:6]]
            v = b[(n == "vehicle") & (np.hypot(b[:, 0], b[:, 1]) <= 30)]
            bottoms += list(v[:, 2] - v[:, 5] / 2)
    pri = {n: tuple(np.median(np.asarray(v), 0)) for n, v in acc.items()}
    pickle.dump(pri, open(D / "size_priors.pkl", "wb"))
    (D / "ground_z.txt").write_text(f"{float(np.median(bottoms))}\n")
    print("size priors (l, w, h):", {k: tuple(round(x, 2) for x in v) for k, v in pri.items()},
          "ground z:", float(np.median(bottoms)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["priors", "run", "collect"], required=True)
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    {"priors": priors, "run": lambda: run(args), "collect": collect}[args.stage]()


if __name__ == "__main__":
    main()
