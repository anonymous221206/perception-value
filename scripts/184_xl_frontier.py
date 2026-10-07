#!/usr/bin/env python
"""Follow-up E5, X8 (secondary): the oracle's decision gain against compute along two escalation ladders.

From the all-YOLOv8s-320 operation on the 228 new test scenes (braking, both geometries, shared action history):
  resolution ladder   YOLOv8s at 288x480, 384x640, 544x960
  capacity ladder     YOLOv8s / v8m / v8l / v8x at 384x640
Each level's per-frame decision value is an escalation from 320 (`decision.build` with the 320 cache as CHEAP). For
budgets b = C_320 + alpha x C_640(YOLOv8s), alpha in {0.05, 0.1, 0.2, 0.3, 0.5, 1.0}, in ms and module mJ (x7_costs),
93's multi-fidelity oracle (upper-hull increments in order of gain per cost) gives the best achievable reduction of the
all-320 loss for each ladder, and for each single level alone. -> results/followup/E5/x8_frontier.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, frames, geometry as G, planner as P, runmeta, xl     # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                  # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402
from rap.risk import RiskConfig                                                # noqa: E402

OUT = Path(RESULTS) / "followup" / "E5"
BASE = "yolov8s__ns_cheap_320"
LADDERS = {"resolution": ["yolov8s__ns_mid_480", "yolov8s__ns_full_640", "yolov8s__ns_full_960"],
           "capacity": ["yolov8s__ns_full_640", "yolov8m__ns_full_640", "yolov8l__ns_full_640", "yolov8x__ns_full_640"]}
ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50, 1.00)


def main():
    frames.configure(type("A", (), {"frame": None})())
    run = runmeta.new_run("xl_frontier", {})
    t93 = xl.load_script("t93", "93_budget_allocation.py")
    det = Path(CACHE) / "xl_models"
    sp = xl.scene_split()
    test = sorted(sp[(sp.group == "new765") & (sp.deep_split == "test")].scene)
    db = NuScenesDB(NUSCENES_XL, "v1.0-trainval")
    nad = make_adapter(db)
    cost = pd.read_csv(OUT / "x7_costs.csv").set_index("engine")
    eng = lambda f: f.replace("__", "_")                                      # noqa: E731
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    levels = sorted({l for v in LADDERS.values() for l in v})
    rows = []
    for geo in ("mono", "oracle"):
        tabs = {l: decision.build(det, BASE, l, test, cfg, pp, cp, G.PRIMARY, range_source=geo, adapter=nad)
                for l in levels}
        base = tabs[levels[0]]
        for l in levels:
            assert np.array_equal(tabs[l].J_cheap.to_numpy(float), base.J_cheap.to_numpy(float))
        tot = float(base.J_cheap.sum())
        for unit, col in (("ms", "ms"), ("mJ", "mJ_module")):
            c320, c640 = cost.loc[eng(BASE), col], cost.loc["yolov8s_ns_full_640", col]
            for name, ladder in list(LADDERS.items()) + [(l, [l]) for l in levels]:
                gains = np.stack([(tabs[l].J_cheap - tabs[l].J_full).to_numpy(float) for l in ladder], 1)
                lc = np.array([cost.loc[eng(l), col] for l in ladder])
                inc = t93.prepare(gains, gains, lc)
                n = len(gains)
                for a in ALPHAS:
                    g, level, _ = t93.greedy(inc, n * a * c640, np.ones(n))
                    rows.append({"geometry": geo, "unit": unit, "ladder": name, "alpha": a,
                                 "budget_per_frame": c320 + a * c640, "oracle_reduction": g / tot,
                                 **{f"share_{l}": float((level == i + 1).mean()) for i, l in enumerate(ladder)}})
        print(f"  {geo} done", flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "x8_frontier.csv", index=False)
    d.to_csv(run / "x8_frontier.csv", index=False)
    print(d[(d.unit == "ms") & d.ladder.isin(LADDERS)].pivot_table(index=["geometry", "ladder"], columns="alpha",
                                                                    values="oracle_reduction").round(3).to_string())


if __name__ == "__main__":
    main()
