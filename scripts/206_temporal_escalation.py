#!/usr/bin/env python
"""Follow-up E10: escalation in time -- reuse the previous frame's detections (stale) or run the detector (fresh).

addendum E10 in docs/followup_protocol.md. One detector mode per cell; both branches feed the benchmark's braking
controller (`rap.planner`), with the geometry source applied as `rap.decision` applies it and the shared action history
of the all-stale run. Stages:

  tables     per-frame rows of both branches for dataset x mode x propagation x geometry, and gate E10-0 (the fresh
             branch equals the benchmark's tables) -> <work>/tables/*.pkl, <work>/gate_e10_0.json
  settings   P1's quantities per cell and population, unit (and log) bootstrap
             -> results/followup/E10/settings.csv, intervals.csv
  refresh    secondary: which frames to refresh at a 20% budget -> results/followup/E10/refresh_allocators.csv

Nothing under data/cache or results/final is written.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, egospeed, frames, geometry as G, kitti, planner as P, runmeta, runs, xl  # noqa: E402
from rap.cache import DetCache                                                                     # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                                  # noqa: E402
from rap.risk import RiskConfig                                                                    # noqa: E402
from rap.scoring import EPS                                                                        # noqa: E402

OUT = Path(RESULTS) / "followup" / "E10"
WORK = Path(CACHE) / "xl" / "e10"
MODES = {"nuScenes": ("ns_full_640", "ns_cheap_320"), "KITTI": ("full_640", "cheap_320")}
DETS = {"nuScenes": Path(CACHE) / "nusc_det_xl", "KITTI": Path(CACHE) / "det"}
PROPS = ("ego", "hold")
GEOMS = ("mono", "oracle")
MIN_RANGE = 0.5          # the lift's lower clip: a carried detection nearer than this has left the camera's view


def role(mode):
    return "cheap" if "320" in mode else "full"


# ------------------------------------------------------------------------------------------------ ego motion

def planar(R, t):
    return np.array([t[0], t[1], np.arctan2(R[1, 0], R[0, 0])])


def nusc_poses(db, by_name, seq):
    """Planar ego pose (x, y, yaw) and time [s] of each keyframe, indexed by frame."""
    toks = db.samples(by_name[seq])
    p = np.array([planar(*db._global_to_ego(tk)) for tk in toks])
    ts = np.array([db._t["sample"][tk]["timestamp"] / 1e6 for tk in toks])
    return p, ts


def kitti_poses(seq):
    """OXTS: Mercator position with the first frame's scale, heading from yaw; 10 Hz frame times."""
    ox = kitti.load_oxts(seq)
    lat, lon, yaw = ox[:, 0], ox[:, 1], ox[:, 5]
    er, s = 6378137.0, np.cos(lat[0] * np.pi / 180.0)
    x = s * er * lon * np.pi / 180.0
    y = s * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    return np.stack([x, y, yaw], 1), np.arange(len(ox)) * 0.1


def carry(geo, p_prev, p_cur):
    """Static-world propagation of (z, lat_min, lat_max, ttc) from ego(prev) into ego(cur); closing speed kept."""
    z, lo, hi, tt = geo
    if len(z) == 0:
        return geo
    dx, dy = p_prev[0] - p_cur[0], p_prev[1] - p_cur[1]
    c0, s0, c1, s1 = np.cos(p_prev[2]), np.sin(p_prev[2]), np.cos(p_cur[2]), np.sin(p_cur[2])

    def move(x, y):                                     # ego(prev) -> global -> ego(cur)
        gx, gy = c0 * x - s0 * y + dx, s0 * x + c0 * y + dy
        return c1 * gx + s1 * gy, -s1 * gx + c1 * gy

    zc, _ = move(z, 0.5 * (lo + hi))
    _, yl = move(z, lo)
    _, yr = move(z, hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        vrel = np.where(np.isfinite(tt) & (tt > 0), z / tt, 0.0)
        tc = np.where(vrel > 0, zc / vrel, np.inf)
    k = zc >= MIN_RANGE
    return zc[k], np.minimum(yl, yr)[k], np.maximum(yl, yr)[k], tc[k]


# ------------------------------------------------------------------------------------------------ tables

def build(dataset, mode, prop, geom_src, seqs, adapter, poses, cfg, pp, cp):
    rng = np.random.default_rng(0)
    thr = cfg.thr(role(mode))
    rows = []
    for s in seqs:
        geom, speeds = adapter.geometry(s), adapter.speeds(s)
        pose, ts = poses(s)
        c = DetCache(DETS[dataset] / mode / f"{s}.npz")
        prev = prev_fr = prev_act = None
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= thr
            geo_k = {kk: vv[k] for kk, vv in geo.items()}
            fresh = decision._apply_range_source(d["xyxy"][k].astype(float), geo_k, g, geom_src, rng, 0.0)
            ent = d["binent"][k]
            if prev is None:                            # the detector runs on a unit's first frame
                stale, ent_s, dt = fresh, ent, np.nan
            else:
                stale = prev[0] if prop == "hold" else carry(prev[0], pose[prev_fr], pose[fr])
                ent_s, dt = prev[1], float(ts[fr] - ts[prev_fr])
            a_f, _ = P.required_decel(*fresh, v, pp)
            a_s, _ = P.required_decel(*stale, v, pp)
            a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
            act_f, act_s = P.discrete_action(a_f, pp), P.discrete_action(a_s, pp)
            Jf = P.decision_cost(act_f, a_gt, prev_act, pp, cp)
            Js = P.decision_cost(act_s, a_gt, prev_act, pp, cp)
            rows.append({"seq": s, "frame": fr, "prev_frame": -1 if prev_fr is None else prev_fr,
                         "first": int(prev is None), "dt": dt, "v_ego": v,
                         "a_stale": a_s, "a_fresh": a_f, "a_gt": a_gt, "act_stale": act_s, "act_fresh": act_f,
                         "J_stale": Js["J"], "J_fresh": Jf["J"],
                         "collision_stale": Js["collision"], "collision_fresh": Jf["collision"],
                         "n_stale": int(len(stale[0])), "n_fresh": int(k.sum()),
                         "unc_stale": float(np.sum(ent_s)), "n_gt": int(len(g))})
            prev, prev_fr, prev_act = (fresh, ent), fr, act_s
    return pd.DataFrame(rows)


def reference(dataset, mode, geom_src):
    """The benchmark's per-frame requirement and action of this mode (gate E10-0)."""
    side = "cheap" if role(mode) == "cheap" else "full"
    if dataset == "nuScenes":
        t = pd.read_pickle(Path(CACHE) / "xl" / "tables850" / f"{geom_src}__ns_cheap_320tons_full_640.pkl")
    else:
        t = pd.read_pickle(runs.core_matrix() / f"KITTI__YOLOv8s__cheap_320tofull_640__{geom_src}.pkl")
    return t[["seq", "frame", f"a_{side}", f"act_{side}"]].rename(columns={f"a_{side}": "a_ref", f"act_{side}": "act_ref"})


def stage_tables(args):
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    work = Path(args.work)
    (work / "tables").mkdir(parents=True, exist_ok=True)
    gate = {}
    for dataset in args.datasets:
        t0 = time.time()
        if dataset == "nuScenes":
            from rap.nusc import NuScenesDB, make_adapter
            db = NuScenesDB(args.dataroot, "v1.0-trainval")
            adapter, by_name = make_adapter(db), {db.scene_name(s): s for s in db.scenes}
            seqs = sorted(p.stem for p in (DETS[dataset] / MODES[dataset][0]).glob("*.npz"))
            pc = {}
            poses = lambda s: pc.setdefault(s, nusc_poses(db, by_name, s))           # noqa: E731
        else:
            adapter, seqs, poses = decision.KittiAdapter, kitti.sequences(), kitti_poses
        print(f"  {dataset}: {len(seqs)} units [{time.time() - t0:.0f}s]", flush=True)
        for mode in MODES[dataset]:
            for geom_src in GEOMS:
                ref = reference(dataset, mode, geom_src)
                for prop in PROPS:
                    t0 = time.time()
                    d = build(dataset, mode, prop, geom_src, seqs, adapter, poses, cfg, pp, cp)
                    f = work / "tables" / f"{dataset}__{mode}__{prop}__{geom_src}.pkl"
                    d.to_pickle(f)
                    m = d.merge(ref, on=["seq", "frame"], how="outer", validate="one_to_one", indicator=True)
                    both = m._merge == "both"
                    key = f"{dataset}|{mode}|{prop}|{geom_src}"
                    gate[key] = {"frames": int(len(d)), "reference_frames": int(len(ref)),
                                 "unmatched": int((~both).sum()),
                                 "a_differ": int((m.a_fresh[both].to_numpy() != m.a_ref[both].to_numpy()).sum()),
                                 "act_differ": int((m.act_fresh[both].to_numpy() != m.act_ref[both].to_numpy()).sum())}
                    print(f"  {f.name}: {len(d)} frames, gate {gate[key]} [{time.time() - t0:.0f}s]", flush=True)
    ok = all(g["unmatched"] == 0 and g["a_differ"] == 0 and g["act_differ"] == 0 for g in gate.values())
    (work / "gate_e10_0.json").write_text(json.dumps({"pass": ok, "cells": gate}, indent=1))
    print("gate E10-0:", "pass" if ok else "FAIL")
    return ok


# ------------------------------------------------------------------------------------------------ settings

def kitti_split():
    """The benchmark's rule on the 21 sequences (configs/benchmark_splits.json): position mod 10."""
    seqs = kitti.sequences() or [f"{i:04d}" for i in range(21)]    # without the dataset: the 21 training sequences
    return {s: ("train" if i % 10 < 5 else "val" if i % 10 < 7 else "test") for i, s in enumerate(seqs)}


def populations(dataset):
    if dataset == "KITTI":
        sp = kitti_split()
        return {"kitti21": set(sp), "kitti_test6": {s for s, x in sp.items() if x == "test"}}
    sp = xl.scene_split()
    new, old = sp[sp.group == "new765"], sp[sp.group == "frozen85"]
    return {"new228_test": set(new[new.deep_split == "test"].scene), "new765": set(new.scene),
            "frozen85": set(old.scene), "pooled850": set(sp.scene)}


def one(d, log_of, nboot):
    v = (d.J_stale - d.J_fresh).to_numpy(float)
    c = d.J_stale.to_numpy(float)
    r = xl.describe(v, c, (0.2,))
    r["budget_binding"] = bool((v > EPS).sum() > max(int(round(0.2 * len(v))), 1))
    r["n_units"] = int(d.seq.nunique())
    r.update(xl.bootstrap_describe(v, c, d.seq.to_numpy(), nboot, np.random.default_rng(0), (0.2,)))
    if log_of is not None:
        r["n_logs"] = int(d.seq.map(log_of).nunique())
        lg = xl.bootstrap_describe(v, c, d.seq.map(log_of).to_numpy(), nboot, np.random.default_rng(0), (0.2,))
        r.update({k.replace("_lo", "_lo_log").replace("_hi", "_hi_log"): x for k, x in lg.items()})
    return r


def stage_settings(args):
    work = Path(args.work)
    gate = json.loads((work / "gate_e10_0.json").read_text())
    if not gate["pass"]:
        raise SystemExit("gate E10-0 failed: settings not computed")
    meta = xl.scene_meta(args.dataroot)
    log_nusc = dict(zip(meta.scene, meta.log))
    rows, dts = [], []
    for f in sorted((work / "tables").glob("*.pkl")):
        dataset, mode, prop, geom_src = f.stem.split("__")
        d = pd.read_pickle(f)
        d = d[d["first"] == 0]
        if prop == PROPS[0] and geom_src == GEOMS[0]:
            q = d.dt.quantile([0.0, 0.05, 0.5, 0.95, 1.0]).to_numpy()
            dts.append({"dataset": dataset, "mode": mode, "n": len(d), "dt_min": q[0], "dt_p05": q[1],
                        "dt_median": q[2], "dt_p95": q[3], "dt_max": q[4]})
        for pop, units in populations(dataset).items():
            r = one(d[d.seq.isin(units)], log_nusc if dataset == "nuScenes" else None, args.nboot)
            rows.append({"dataset": dataset, "mode": mode, "propagation": prop, "geometry": geom_src,
                         "population": pop, "primary": mode in ("ns_full_640", "full_640") and prop == "ego", **r})
            print(f"  {dataset} {mode} {prop} {geom_src} {pop}: harmed {r['harmed_share']:.3f} "
                  f"rho {r['rho']:.3f} all-fresh {r['all_full_reduction']:+.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.DataFrame(rows)
    s.columns = [c.replace("all_full", "all_fresh") for c in s.columns]
    s.to_csv(OUT / "settings.csv", index=False)
    pd.DataFrame(dts).to_csv(OUT / "intervals.csv", index=False)
    print("wrote", OUT / "settings.csv")


# ------------------------------------------------------------------------------------------------ refresh (secondary)

def stage_refresh(args):
    work = Path(args.work)
    if not json.loads((work / "gate_e10_0.json").read_text())["pass"]:
        raise SystemExit("gate E10-0 failed")
    d = pd.read_pickle(work / "tables" / "nuScenes__ns_full_640__ego__mono.pkl")
    d = d[d["first"] == 0].reset_index(drop=True)
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    d = d[d.seq.isin(set(new.scene))].reset_index(drop=True)
    split = dict(zip(new.scene, new.deep_split))
    # the benchmark's gate features of the stale list: the mode's features of the previous frame
    from rap.tables import feature_columns
    feats = xl.t92().g84.build_features(DETS["nuScenes"], "ns_full_640", RiskConfig())
    fcols = feature_columns(feats)
    feats = feats.rename(columns={"frame": "prev_frame"})
    d = d.merge(feats, on=["seq", "prev_frame"], how="left", validate="many_to_one")
    assert d[fcols].notna().all().all(), "stale features missing"
    xl.use_ego_table(Path(CACHE) / "xl" / "tables850" / "v_ego_causal.csv")
    d = egospeed.attach(d, "nuScenes")
    d["unit"] = d.seq
    v = (d.J_stale - d.J_fresh).to_numpy(float)
    fit = d.seq.map(split).isin(["train", "val"]).to_numpy()
    test = (d.seq.map(split) == "test").to_numpy()
    scores = xl.gate_scores(d, fcols + [egospeed.COL], v, fit, which=("gate_gbm",))
    scores.update({"uncertainty": d.unc_stale.to_numpy(float), "trivial_ego_speed": d[egospeed.COL].to_numpy(float)})
    c = dict(track="nuScenes", geometry="mono", system="brake", target="J", d=d, cheap="J_stale", full="J_fresh")
    p = xl.prepare(c, test, scores, (0.2,))
    draws, keep = xl.paired_draws([p], args.nboot, np.random.default_rng(0), (0.2,))
    rows = xl.paired_rows([p], draws, keep, (0.2,), {"mode": "ns_full_640", "propagation": "ego"})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "refresh_allocators.csv", index=False)
    for r in rows:
        print(f"  {r['signal']}: delta share {r['delta_share_all_cheap']:+.4f} "
              f"[{r['delta_share_lo_all_draws']:+.4f}, {r['delta_share_hi_all_draws']:+.4f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tables", "settings", "refresh"], required=True)
    ap.add_argument("--datasets", nargs="+", default=["KITTI", "nuScenes"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"e10_{args.stage}", vars(args))
    ok = {"tables": stage_tables, "settings": stage_settings, "refresh": stage_refresh}[args.stage](args)
    (run / "summary.json").write_text(json.dumps({"stage": args.stage, "ok": ok is not False}, indent=1))
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
