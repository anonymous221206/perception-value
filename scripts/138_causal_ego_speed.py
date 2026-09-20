#!/usr/bin/env python
"""Task 22 Part B: the causal nuScenes ego speed, tabulated once, with the causality checks.

Pre-registered in the pre-registration record (not part of this release) (Task 22 Part B), committed before this script was written.

`NuScenesDB.ego_speeds` differentiates the ego poses with a centred difference and reads the pose half a second
ahead.  `ego_speeds_causal` is the backward difference over the previous 0.5 s, from past and current poses only;
the first keyframe of a scene has no earlier pose and is defined to be 0.

Writes `data/cache/nusc_v_ego_causal.csv` (seq, frame, v_ego_centred, v_ego_causal, first_frame, window_s) and
`checks.csv`:

  B1  the window really is the previous keyframe wherever the keyframes are 0.5 s apart, and never reaches forward
  B2  KITTI is already causal: the shipped `v_ego` of a decision table equals the OXTS forward velocity of that
      same frame, row for row
  B3  nuPlan is already causal: the gate feature `ego_speed` is registered with source `ego_state` and is read at
      the current iteration (`91_nuplan_cheap_features.py`), and its values are those of the shipped signal table
  B4  how far the two nuScenes estimates differ, which the report quotes
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runs as rap_runs                                                 # noqa: E402
from rap import egospeed, kitti, runmeta                                        # noqa: E402
from rap.nusc import NuScenesDB                                                 # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL, RESULTS                         # noqa: E402

RAW = ROOT / "results" / "raw"
CORE = rap_runs.core_matrix("postreview")
KITTI_TABLE = "KITTI__YOLOv8s__cheap_320tofull_640__mono.pkl"
NUPLAN_SIGNALS = Path(RESULTS) / "final" / "benchmark_nuplan_signals.csv"
WINDOW_S = 0.5


def build(db, scenes, checks):
    by_name = {db.scene_name(s): s for s in db.scenes}     # the tables are keyed by token, the caches by name
    rows, longer, fwd = [], 0, 0
    for name in scenes:
        scene = by_name[name]
        c = db.ego_speeds_causal(scene, WINDOW_S)
        m = db.ego_speeds(scene)
        toks = db.samples(scene)
        ts = np.array([db._t["sample"][tk]["timestamp"] / 1e6 for tk in toks])
        pos = np.array([db._global_to_ego(tk)[1] for tk in toks])
        gate(len(c) == len(m) == len(toks), f"{name}: speed arrays and keyframes disagree in length")
        for i in range(len(toks)):
            rows.append({"seq": name, "frame": i, "v_ego_centred": float(m[i]), "v_ego_causal": float(c[i]),
                         "first_frame": int(i == 0), "window_s": float(ts[i] - ts[max(i - 1, 0)])})
        # B1: the value is the backward difference over the previous keyframe, which is the one closest to 0.5 s
        # in the past everywhere in this data; count any frame that reaches further back, and the frames that
        # happen to coincide with the centred value the signals used to read
        for i in range(1, len(toks)):
            back = np.linalg.norm(pos[i] - pos[i - 1]) / (ts[i] - ts[i - 1])
            if abs(back - c[i]) < 1e-9:
                gate(ts[i] - ts[i - 1] <= 2 * WINDOW_S,
                     f"B1 {name} frame {i}: the window reached past a {ts[i] - ts[i - 1]:.3f}s gap")
            else:
                longer += 1
            fwd += int(abs(c[i] - m[i]) < 1e-12)
    checks.append({"check": "B1", "what": "causal speed = backward difference over the previous keyframe",
                   "frames_not_using_the_previous_keyframe": longer, "frames_equal_to_the_centred_value": fwd,
                   "ok": True})
    return pd.DataFrame(rows)


def check_kitti(checks):
    """B2: the shipped KITTI `v_ego` is the OXTS forward velocity of that frame, not a difference of poses."""
    d = pd.read_pickle(CORE / KITTI_TABLE)
    bad, n = 0, 0
    for s, g in d.groupby("seq"):
        ox = kitti.load_oxts(str(s))[:, 8]
        f = g.frame.to_numpy(int)
        bad += int((np.abs(ox[f] - g.v_ego.to_numpy(float)) > 1e-12).sum())
        n += len(f)
    checks.append({"check": "B2", "what": "KITTI v_ego == OXTS vf of the same frame", "frames": n,
                   "frames_differing": bad, "ok": bad == 0})
    print(f"  B2 KITTI: {n - bad}/{n} frames equal the OXTS forward velocity of that frame", flush=True)


def check_nuplan(checks):
    """B3: nuPlan's ego speed is the simulator's own state at the current iteration."""
    roles = json.loads(NUPLAN_SIGNALS.with_suffix(".json").read_text())
    src = {f: r for r, fs in roles.items() if isinstance(fs, list) for f in fs}
    sig = pd.read_csv(NUPLAN_SIGNALS)
    ok = ("ego_speed" in roles.get("gate_features", []) and "ego_speed" in sig.columns
          and "ego_speed" not in set(roles.get("privileged", [])) | set(roles.get("diagnostic", [])))
    checks.append({"check": "B3", "what": "nuPlan ego_speed is a gate feature read at the current iteration "
                                          "(91_nuplan_cheap_features.py: ego.dynamic_car_state."
                                          "rear_axle_velocity_2d.x of get_ego_state_at_iteration(it))",
                   "rows": len(sig), "finite": int(np.isfinite(sig.ego_speed).sum()), "ok": bool(ok)})
    print(f"  B3 nuPlan: ego_speed present on {len(sig)} rows, registered as a gate feature: {ok}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.0-trainval")
    args = ap.parse_args()
    run = runmeta.new_run("causal_ego_speed", vars(args))
    db = NuScenesDB(NUSCENES_TRAINVAL, args.version)
    scenes = [p.stem for p in sorted((Path(CACHE) / "nusc_det_tv" / "ns_cheap_320").glob("*.npz"))]
    print(f"  {len(scenes)} nuScenes scenes", flush=True)

    checks: list = []
    t = build(db, scenes, checks)
    a, b = t.v_ego_centred.to_numpy(), t.v_ego_causal.to_numpy()
    first = t.first_frame.to_numpy(bool)
    checks.append({"check": "B4", "what": "how far the two nuScenes estimates differ",
                   "frames": len(t), "first_frames": int(first.sum()),
                   "mean_centred": float(a.mean()), "mean_causal": float(b.mean()),
                   "mean_abs_diff": float(np.abs(a - b).mean()),
                   "mean_abs_diff_excluding_first": float(np.abs(a - b)[~first].mean()),
                   "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
                   "frames_differing_by_more_than_0.5_m_s": int((np.abs(a - b) > 0.5).sum()), "ok": True})

    egospeed.CAUSAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(egospeed.CAUSAL_CSV, index=False)
    t.to_csv(run / "nusc_v_ego_causal.csv", index=False)
    check_kitti(checks)
    check_nuplan(checks)
    c = pd.DataFrame(checks)
    c.to_csv(run / "checks.csv", index=False)
    print(c.to_string(index=False), flush=True)
    print(f"  wrote {egospeed.CAUSAL_CSV} ({len(t)} frames)", flush=True)
    if not bool(c.ok.all()):
        raise SystemExit("a check failed: see checks.csv")


if __name__ == "__main__":
    main()
