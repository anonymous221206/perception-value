#!/usr/bin/env python
"""Follow-up E3 (a): the uniform ladder on the 220 Bench2Drive routes (addendum E3 in docs/followup_protocol.md).

Reads the evaluator checkpoints route_<id>_lad_<branch>_r<rep>.json (198_b2d_dispatch.sh ladder) from --out_dir.
  ladder_runs.csv     one row per route x branch x repetition: status, driving score, route completion, penalty,
                      success, infraction counts by type
  uniform_ladder.csv  per branch: runs, routes, driving score, success rate, route completion, infractions by type
                      per run (means over route x repetition); per branch pair: the paired difference of route means
                      (each route's mean over its repetitions) with a 95% bootstrap interval by route (1,000 draws,
                      seed 0), on the routes present in both branches
Success follows Bench2Drive: status Completed or Perfect and no infraction other than min_speed_infractions.
"""
from __future__ import annotations

import argparse, json, re, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.paths import RESULTS                                                    # noqa: E402

OUT = Path(RESULTS) / "followup" / "E3"
BRANCHES = ["privileged", "cheap", "full", "u960"]
PAIRS = [("cheap", "privileged"), ("full", "cheap"), ("u960", "full"), ("u960", "cheap"), ("full", "privileged")]
METRICS = ["driving_score", "success", "route_completion"]


def runs(out_dir, kind="lad"):
    """One row per checkpoint route_<id>_<kind>_<branch>_r<rep>.json with a record (kind lad: ladder, rt: routing)."""
    rows = []
    for f in sorted(Path(out_dir).glob(f"route_*_{kind}_*_r*.json")):
        m = re.fullmatch(rf"route_(\d+)_{kind}_(\w+?)_r(\d)\.json", f.name)
        rec = json.loads(f.read_text())["_checkpoint"]["records"]
        if not m or not rec:
            continue
        r = rec[0]
        inf = {k: len(v) if isinstance(v, list) else int(v) for k, v in r["infractions"].items()}
        ok = r["status"] in ("Completed", "Perfect") and all(v == 0 for k, v in inf.items()
                                                                if k != "min_speed_infractions")
        rows.append({"route": int(m[1]), "branch": m[2], "rep": int(m[3]), "status": r["status"],
                     "driving_score": r["scores"]["score_composed"], "route_completion": r["scores"]["score_route"],
                     "penalty": r["scores"]["score_penalty"], "success": float(ok), **inf})
    return pd.DataFrame(rows)


def boot(diff, rng):
    n = len(diff)
    return np.percentile([diff[rng.integers(0, n, n)].mean() for _ in range(1000)], [2.5, 97.5]) if n else [np.nan] * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="folder holding the ladder checkpoints (copied from the simulation machine)")
    args = ap.parse_args()
    d = runs(args.out_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    d.sort_values(["route", "branch", "rep"]).to_csv(OUT / "ladder_runs.csv", index=False)
    inf_cols = [c for c in d.columns if c not in ("route", "branch", "rep", "status", "driving_score",
                                                  "route_completion", "penalty", "success")]
    rows = []
    for b in BRANCHES:
        g = d[d.branch == b]
        if g.empty:
            continue
        rows.append({"kind": "branch", "branch": b, "runs": len(g), "routes": g.route.nunique(),
                     **{m: g[m].mean() for m in METRICS},
                     **{f"status_{s}": int((g.status == s).sum()) for s in sorted(d.status.unique())},
                     **{f"inf_{c}_per_run": g[c].mean() for c in inf_cols}})
    rm = d.groupby(["branch", "route"])[METRICS].mean()
    for a, b in PAIRS:
        if a not in rm.index.get_level_values(0) or b not in rm.index.get_level_values(0):
            continue
        both = rm.loc[a].join(rm.loc[b], lsuffix="_a", rsuffix="_b", how="inner")
        row = {"kind": "pair", "branch": f"{a} - {b}", "routes": len(both)}
        for m in METRICS:
            diff = (both[f"{m}_a"] - both[f"{m}_b"]).to_numpy(float)
            lo, hi = boot(diff, np.random.default_rng(0))
            row.update({m: diff.mean() if len(diff) else np.nan, f"{m}_lo": lo, f"{m}_hi": hi})
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(OUT / "uniform_ladder.csv", index=False)
    print(s.T.to_string())


if __name__ == "__main__":
    main()
