#!/usr/bin/env python
"""Task 33: Task 30's outputs before (own action history, as first shipped) and after (shared history, deterministic R1).

Reads target_transform_control.csv, target_transform_heatmap.csv and seed_variation.csv from two results/final
directories and prints, side by side, every quantity Task 30's report reads:
- the pooled paired differences, against raw V and with the transform held fixed;
- the counts of per-cell intervals above and below zero against raw V, over the 40 cell-quota pairs;
- the per-cell entries its reading quotes: GBM planner-cell gains, KITTI mono trajectory rank(V);
- the pooled binary comparison at 20%;
- the seed table (min / median / max nDG at 20%, winning seeds), the shipped wins and the flag.

Writes the side-by-side tables to --out as CSV and markdown.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCH = {"R1_mlp_reg": "MLP", "R1_gbm_reg": "GBM", "R1_mlp_clf": "MLP", "R1_gbm_clf": "GBM"}
LBL = {"rankV": "rank(V)", "secdfV": "signed ECDF(V)", "Q": "Q", "G": "G", "V": "V"}


def rd(d, name):
    return pd.read_csv(Path(d) / name, keep_default_na=False, na_values=[""], float_precision="round_trip",
                       low_memory=False)


def iv(x, lo, hi):
    return "—" if not np.isfinite(x) else f"{x:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def pooled(ctl):
    p = ctl[(ctl.track == "pooled") & (ctl.kind == "regression")]
    rows = []
    for _, r in p.iterrows():
        base = r.pooled_against
        rows.append({"architecture": ARCH[r.architecture], "comparison": f"{LBL[r.label]} - {LBL[base]}",
                     "quota": r.quota, "value": r[f"minus_{base}"], "lo": r[f"minus_{base}_lo"],
                     "hi": r[f"minus_{base}_hi"]})
    b = ctl[(ctl.track == "pooled") & (ctl.kind == "binary")]
    for _, r in b.iterrows():
        rows.append({"architecture": ARCH[r.architecture], "comparison": f"binary {LBL[r.label]}>0 - V>0",
                     "quota": r.quota, "value": r.minus_V, "lo": r.minus_V_lo, "hi": r.minus_V_hi})
    return pd.DataFrame(rows)


def counts(heat):
    h = heat.assign(above=heat.minus_V_lo > 0, below=heat.minus_V_hi < 0)
    g = h.groupby(["architecture", "label"])[["above", "below"]].sum().reset_index()
    g["architecture"] = g.architecture.map(ARCH)
    g["label"] = g.label.map(LBL)
    return g


def seeds(sv):
    s = sv[sv.row_type == "spread"]
    return s[["track", "geometry", "system", "target", "signal", "ndg_min", "ndg_median", "ndg_max", "wins", "seeds",
              "shipped_win", "flag_shipped_win_not_robust"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="results/final directory before")
    ap.add_argument("--after", default=str(ROOT / "results" / "final"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = {k: rd(a.before, f"{k}.csv") for k in ("target_transform_control", "target_transform_heatmap", "seed_variation")}
    A = {k: rd(a.after, f"{k}.csv") for k in B}
    md = []

    pb, pa = pooled(B["target_transform_control"]), pooled(A["target_transform_control"])
    p = pb.merge(pa, on=["architecture", "comparison", "quota"], suffixes=("_before", "_after"), validate="one_to_one")
    p["excludes_zero_before"] = (p.lo_before > 0) | (p.hi_before < 0)
    p["excludes_zero_after"] = (p.lo_after > 0) | (p.hi_after < 0)
    p.to_csv(out / "pooled.csv", index=False)
    md += ["## Pooled over the ten core cells (paired difference in nDG)", "",
           "| architecture | comparison | quota | before | after |", "|---|---|---|---|---|"]
    for _, r in p.iterrows():
        flag = " **(interval call changes)**" if r.excludes_zero_before != r.excludes_zero_after else ""
        md.append(f"| {r.architecture} | {r.comparison} | {r.quota:.0%} | {iv(r.value_before, r.lo_before, r.hi_before)} | "
                  f"{iv(r.value_after, r.lo_after, r.hi_after)}{flag} |")

    cb, ca = counts(B["target_transform_heatmap"]), counts(A["target_transform_heatmap"])
    c = cb.merge(ca, on=["architecture", "label"], suffixes=("_before", "_after"))
    c.to_csv(out / "counts.csv", index=False)
    md += ["", "## Per-cell intervals against raw V, above / below zero, over the 40 cell-quota pairs", "",
           "| architecture | label | before | after |", "|---|---|---|---|"]
    for _, r in c.iterrows():
        md.append(f"| {r.architecture} | {r.label} | {r.above_before} / {r.below_before} | {r.above_after} / {r.below_after} |")

    k = ["architecture", "label", "track", "geometry", "system", "target", "quota"]
    h = B["target_transform_heatmap"].merge(A["target_transform_heatmap"], on=k, suffixes=("_before", "_after"))
    h.to_csv(out / "heatmap_cells.csv", index=False)
    sel = h[(h.architecture == "R1_gbm_reg") & np.isclose(h.quota, 0.2)]
    md += ["", "## GBM router at 20 %, per cell (paired difference to raw V)", "",
           "| cell | label | before | after |", "|---|---|---|---|"]
    for _, r in sel.sort_values(["label", "track", "geometry", "system"]).iterrows():
        md.append(f"| {r.track} {r.geometry} {r.system} | {LBL[r.label]} | "
                  f"{iv(r.minus_V_before, r.minus_V_lo_before, r.minus_V_hi_before)} | "
                  f"{iv(r.minus_V_after, r.minus_V_lo_after, r.minus_V_hi_after)} |")

    sb, sa = seeds(B["seed_variation"]), seeds(A["seed_variation"])
    ks = ["track", "geometry", "system", "target", "signal"]
    s = sb.merge(sa, on=ks, suffixes=("_before", "_after"), how="outer")
    s.to_csv(out / "seeds.csv", index=False)
    md += ["", "## Seed variation at 20 % (nDG min / median / max over six seeds; winning seeds)", "",
           "| cell | signal | before | wins | after | wins | shipped win before → after | flag before → after |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in s.sort_values(ks).iterrows():
        md.append(f"| {r.track} {r.geometry} {r.system} | {r.signal} | {r.ndg_min_before:.3f} / {r.ndg_median_before:.3f} / "
                  f"{r.ndg_max_before:.3f} | {r.wins_before} | {r.ndg_min_after:.3f} / {r.ndg_median_after:.3f} / "
                  f"{r.ndg_max_after:.3f} | {r.wins_after} | {r.shipped_win_before} → {r.shipped_win_after} | "
                  f"{r.flag_shipped_win_not_robust_before} → {r.flag_shipped_win_not_robust_after} |")
    (out / "before_after.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
