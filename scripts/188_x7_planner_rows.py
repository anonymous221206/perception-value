#!/usr/bin/env python
"""Follow-up E5, X7 planner rows (protocol Section 8): PKL's planner scored against the logged trajectory (74) for
each primary X7 pair, old detector as cheap and new as full, both geometries, on the 228 new test scenes.

Inputs: data/cache/xl/x7_planner/pairs/<old>@<t>__<new>@<t>/planC_vs_truth[_mono].csv (187 assemble, then 74).
V = ADE(old) - ADE(new) (and FDE); loss = the old detector's error. Intervals: 1,000 scene-bootstrap draws.
-> results/followup/E5/x7_planner.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import xl                                                               # noqa: E402
from rap.paths import CACHE, RESULTS                                             # noqa: E402

PAIRS = Path(CACHE) / "xl" / "x7_planner" / "pairs"
OUT = Path(RESULTS) / "followup" / "E5" / "x7_planner.csv"


def main():
    rows = []
    for d in sorted(PAIRS.iterdir()):
        old, new = d.name.split("__", 1)[0] + "__" + d.name.split("__")[1], "__".join(d.name.split("__")[2:])
        for geo in ("mono", "oracle"):
            f = d / ("planC_vs_truth.csv" if geo == "oracle" else f"planC_vs_truth_{geo}.csv")
            if not f.exists():
                continue
            t = pd.read_csv(f)
            for target in ("ade", "fde"):
                c, n = t[f"JC_{target}_cheap"].to_numpy(float), t[f"JC_{target}_full"].to_numpy(float)
                r = xl.describe(c - n, c, (0.2,))
                r.update(xl.bootstrap_describe(c - n, c, t.scene.to_numpy(), 1000, np.random.default_rng(0), (0.2,)))
                rows.append({"old": old, "new": new, "geometry": geo, "target": target, "n_scenes": t.scene.nunique(),
                             "mean_old": float(c.mean()), "mean_new": float(n.mean()), **r})
            print(f"  {d.name} {geo}: harmed {rows[-2]['harmed_share']:.3f} all-full {rows[-2]['all_full_reduction']:+.3f}",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print("wrote", OUT, len(rows))


if __name__ == "__main__":
    main()
