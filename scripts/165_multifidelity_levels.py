#!/usr/bin/env python
"""Task 32: the KITTI multi-fidelity levels on the shared action history.

The multi-fidelity allocator (93) escalates each input from the 320 detector to 384, 512 or 640. It took the loss of
level L from the CHEAP column of the L->640 table, which charged level L against its own previous action. Under the
shared history, every escalation from all-cheap operation is charged against the action the all-320 run took at the
previous frame. So level L is the FULL branch of a 320->L pair.

This script builds, on KITTI with monocular geometry and threshold 0.25, the pairs 320->384 and 320->512 with
`decision.build` and `65.build_b` at their default (shared) history, and keeps the per-frame losses. The 512 detections
live in another cache directory (`det512`), so the FULL branch's caches are opened through `cache_factory`.

The 320->640 pair is the core table (52, 65), carried into levels.pkl unchanged. Check: the 320 branch of each pair
equals the core 320->640 table's CHEAP column, bit for bit. 93 asserts in turn that levels.pkl's 320 and 640 columns
are exactly those of the cell it scores.

Writes levels.pkl and checks.json into the run directory; 93 reads it.
"""
from __future__ import annotations

import argparse, json, sys, time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import frames as rap_frames                                              # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap import runs as rap_runs                                                # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.paths import CACHE                                                     # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

LEVELS = {"384": ("det", "cheap_384"), "512": ("det512", "cheap_512")}


def main():
    ap = argparse.ArgumentParser()
    rap_frames.add_argument(ap)
    args = ap.parse_args()
    rap_frames.configure(args)
    m52, m65 = import_module("52_core_matrix"), import_module("65_planner_b_decision")
    B, P = m65.B, m52.P
    run = runmeta.new_run("multifidelity_levels", vars(args))
    ad = m52.decision.KittiAdapter
    seqs = [p.stem for p in sorted((CACHE / "det" / "cheap_320").glob("*.npz"))]
    cfg = RiskConfig(op_conf=0.25)
    core, planb = rap_runs.core_matrix("postreview"), rap_runs.planner_b()
    ref = pd.read_pickle(core / "KITTI__YOLOv8s__cheap_320tofull_640__mono.pkl")[["seq", "frame", "J_cheap", "J_full"]]
    refb = pd.read_pickle(planb / "planB__KITTI__YOLOv8s__cheap_320tofull_640__mono.pkl")[
        ["seq", "frame", "JB_cheap", "JB_full"]]
    out = ref.merge(refb, on=["seq", "frame"], validate="one_to_one")
    out["seq"] = out.seq.astype(str)
    checks, t_all = {}, time.time()
    for res, (ddir, mode) in LEVELS.items():
        t0 = time.time()

        def factory(path, role, ddir=ddir, mode=mode):
            path = Path(path)
            return DetCache(CACHE / ddir / mode / path.name) if role == "full" else DetCache(path)
        d = m52.decision.build(CACHE / "det", "cheap_320", mode, seqs, cfg, P.PlannerParams(), P.CostParams(),
                               m52.G.PRIMARY, range_source="mono", adapter=ad, cache_factory=factory)
        b = m65.build_b(CACHE / "det", "cheap_320", mode, seqs, cfg, ad, "mono", B.PARAMS_B["static_obstacles"],
                        B.COSTS_B["default"], cache_factory=factory)
        x = d[["seq", "frame", "J_cheap", "J_full"]].merge(b[["seq", "frame", "JB_cheap", "JB_full"]],
                                                           on=["seq", "frame"], validate="one_to_one")
        x["seq"] = x.seq.astype(str)
        j = out[["seq", "frame", "J_cheap", "JB_cheap"]].merge(x, on=["seq", "frame"], suffixes=("", "_l"),
                                                               how="left", validate="one_to_one")
        dc = float(np.max(np.abs(j.J_cheap - j.J_cheap_l)))
        dbc = float(np.max(np.abs(j.JB_cheap - j.JB_cheap_l)))
        checks[f"320 branch of 320->{res} = core CHEAP"] = {"n": len(x), "n_core": len(out), "max_diff_J": dc,
                                                             "max_diff_JB": dbc,
                                                             "pass": len(x) == len(out) and dc == 0 and dbc == 0}
        out = out.merge(x[["seq", "frame", "J_full", "JB_full"]].rename(
            columns={"J_full": f"J_{res}", "JB_full": f"JB_{res}"}), on=["seq", "frame"], validate="one_to_one")
        print(f"  320->{res}: {len(x)} frames [{time.time() - t0:.0f}s]", flush=True)
    out.to_pickle(run / "levels.pkl")
    ok = all(v["pass"] for v in checks.values())
    (run / "checks.json").write_text(json.dumps({"all_pass": ok, "core": core.name, "planner_b": planb.name,
                                                 "seconds": time.time() - t_all, "checks": checks}, indent=1))
    for k, v in checks.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  {v}", flush=True)
    print(f"  wrote {run}", flush=True)
    if not ok:
        raise SystemExit("checks failed")


if __name__ == "__main__":
    main()
