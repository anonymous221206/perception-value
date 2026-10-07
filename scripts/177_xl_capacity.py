#!/usr/bin/env python
"""Follow-up E5: escalation across detector capacity, architecture and fine-tuning (X7).

For each pair of detection folders in one cache (176's output, with the COCO YOLOv8s folders linked in), at the two
operating points of the protocol -- the shared threshold 0.25, and equal box count (the new model's threshold set so
that its total box count on the 152 validation scenes equals the old model's at 0.25) -- this builds the braking
tables on the 228 test scenes with `decision.build` (both geometries, shared action history) and reports recall and
precision of both models, affected inputs, harmed share and harm ratio with 95% scene-bootstrap intervals, and the
all-full and oracle@20 reductions. Writes results/followup/E5/x7_upgrades.csv (or --out_csv).
Protocol: docs/followup_protocol.md, Section 8.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import decision, frames, geometry as G, planner as P, runmeta, xl      # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                   # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

from importlib import import_module                                             # noqa: E402
_pm = import_module("50_percep_metrics")

OUT = Path(RESULTS) / "followup" / "E5"
# (old folder, new folder, primary operating point): shared within a family, equal box count across families
PAIRS = [("yolov8s__ns_full_640", "yolov8m__ns_full_640", "shared"),
         ("yolov8s__ns_full_640", "yolov8l__ns_full_640", "shared"),
         ("yolov8s__ns_full_640", "yolov8x__ns_full_640", "shared"),
         ("yolov8m__ns_full_640", "yolov8l__ns_full_640", "shared"),
         ("yolov8l__ns_full_640", "yolov8x__ns_full_640", "shared"),
         ("yolo11s__ns_full_640", "yolo11m__ns_full_640", "shared"),
         ("yolov8s__ns_full_640", "yolo11s__ns_full_640", "equal_count"),
         ("yolov8m__ns_full_640", "yolo11m__ns_full_640", "equal_count"),
         ("yolov8s__ns_full_640", "rtdetrl__rt_full_640", "equal_count"),
         ("yolov8s__ns_cheap_320", "yolov8l__ns_full_640", "shared")]
FT_PAIRS = [("yolov8m_ft__ns_full_640", "yolov8l_ft__ns_full_640", "shared"),
            ("yolov8s_ft__ns_cheap_320", "yolov8s_ft__ns_full_640", "shared")]
OP = 0.25


def confs(det_dir, folder, scenes):
    out = []
    for s in scenes:
        c = DetCache(Path(det_dir) / folder / f"{s}.npz")
        out += [c.det(i)["conf"] for i in range(len(c.frames))]
    return np.concatenate(out) if out else np.zeros(0)


def equal_count_threshold(det_dir, old, new, val):
    """The new model's threshold whose total box count on the validation scenes equals the old model's at 0.25."""
    n_old = int((confs(det_dir, old, val) >= OP).sum())
    c = np.sort(confs(det_dir, new, val))[::-1]
    if n_old == 0 or n_old > len(c):
        return np.nan, n_old
    return float(c[n_old - 1]), n_old


def recall_precision(prim, role):
    n_gt, fn = prim[f"{role}_n_gt"].sum(), prim[f"{role}_fn"].sum()
    n_det, fp = prim[f"{role}_n_det"].sum(), prim[f"{role}_fp"].sum()
    return (1 - fn / n_gt if n_gt else np.nan), (1 - fp / n_det if n_det else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(CACHE / "xl_models"))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--finetuned", action="store_true", help="the fine-tuned pairs (x7_finetuned.csv)")
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_capacity" + ("_ft" if args.finetuned else ""), vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    val, test = sorted(new[new.deep_split == "val"].scene), sorted(new[new.deep_split == "test"].scene)
    det = Path(args.det)
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    nad = make_adapter(db)
    pp, cp = P.PlannerParams(), P.CostParams()
    rows, thr = [], {}
    for old, new_, primary in (FT_PAIRS if args.finetuned else PAIRS):
        if not ((det / old).exists() and (det / new_).exists()):
            print(f"  skip {old} -> {new_}: detections missing", flush=True)
            continue
        t_eq, n_old = equal_count_threshold(det, old, new_, val)
        thr[f"{old}->{new_}"] = {"equal_count_threshold": t_eq, "old_boxes_val_at_0.25": n_old}
        for op, t_full in (("shared", None), ("equal_count", t_eq)):
            cfg = RiskConfig(op_conf=OP, op_conf_full=t_full)
            prim = _pm.primitives(det, old, new_, test, cfg, nad, cfg.min_gt_height)
            r_old, p_old = recall_precision(prim, "cheap")
            r_new, p_new = recall_precision(prim, "full")
            for geo in ("mono", "oracle"):
                t0 = time.time()
                d = decision.build(det, old, new_, test, cfg, pp, cp, G.PRIMARY, range_source=geo, adapter=nad)
                v = (d.J_cheap - d.J_full).to_numpy(float)
                r = xl.describe(v, d.J_cheap.to_numpy(float), (0.2,))
                r.update(xl.bootstrap_describe(v, d.J_cheap.to_numpy(float), d.seq.to_numpy(), args.nboot,
                                               np.random.default_rng(0), (0.2,)))
                rows.append({"pair": f"{old} -> {new_}", "old": old, "new": new_, "operating_point": op,
                             "primary": op == primary, "full_threshold": OP if t_full is None else t_full,
                             "geometry": geo, "system": "brake", "recall_old": r_old, "recall_new": r_new,
                             "precision_old": p_old, "precision_new": p_new, "n_units": d.seq.nunique(),
                             "changed_decision_share": float((d.same_action == 0).mean()), **r})
                print(f"  {old} -> {new_} {op:11s} {geo:6s} recall {r_old:.3f}->{r_new:.3f} harmed "
                      f"{r['harmed_share']:.3f} [{r['harmed_share_lo']:.3f}, {r['harmed_share_hi']:.3f}] "
                      f"all-full {r['all_full_reduction']:+.3f} [{time.time() - t0:.0f}s]", flush=True)
    name = "x7_finetuned.csv" if args.finetuned else "x7_upgrades.csv"
    pd.DataFrame(rows).to_csv(OUT / name, index=False)
    pd.DataFrame(rows).to_csv(run / name, index=False)
    (OUT / (name.replace(".csv", "_thresholds.json"))).write_text(json.dumps(thr, indent=1))
    print("wrote", OUT / name)


if __name__ == "__main__":
    main()
