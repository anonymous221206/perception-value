#!/usr/bin/env python
"""Mechanism table: how detections change between CHEAP and FULL on frames where FULL helps, harms, or does nothing.

Input: the nuScenes oracle-geometry frame table `results/raw/20260913_211441_phase0g_eta_fde_oracle/joined_frames.pkl`
(3,376 CAM_FRONT keyframes).  Frames are grouped by the sign of the decision value of two decision makers:

  brake        `_dJ_longitudinal`       (braking controller)
  self control `_dJ_plannerC_path_dev`  (Planner C against its own plan on perfect perception)

with V = J(CHEAP) - J(FULL): helped V > 1e-9, harmed V < -1e-9, unaffected |V| <= 1e-9.  For each group the table
reports the mean FULL - CHEAP change in false positives, false negatives and detection count, the share of frames
where FULL adds false positives, and the share where FULL does not reduce false negatives.

Writes results/final/mechanism_table.csv.  Cached tier, stage C13; no dataset needed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap.paths import RAW, RESULTS                                              # noqa: E402

FRAMES = RAW / "20260913_211441_phase0g_eta_fde_oracle" / "joined_frames.pkl"
DECISIONS = {"brake": "_dJ_longitudinal", "self control (plannerC_path_dev)": "_dJ_plannerC_path_dev"}
EPS = 1e-9


def main():
    j = pd.read_pickle(FRAMES)
    d_fp = (j.full_fp - j.cheap_fp).to_numpy(float)
    d_fn = (j.full_fn - j.cheap_fn).to_numpy(float)
    d_n = (j.full_n_det - j.cheap_n_det).to_numpy(float)
    rows = []
    for name, col in DECISIONS.items():
        v = j[col].to_numpy(float)
        for group, m in (("helped", v > EPS), ("harmed", v < -EPS), ("unaffected", np.abs(v) <= EPS)):
            rows.append({"decision": name, "dJ_column": col, "group": group, "n_frames": int(m.sum()),
                         "mean_delta_fp": float(d_fp[m].mean()), "mean_delta_fn": float(d_fn[m].mean()),
                         "mean_delta_n_det": float(d_n[m].mean()),
                         "share_delta_fp_pos": float((d_fp[m] > 0).mean()),
                         "share_delta_fn_nonneg": float((d_fn[m] >= 0).mean())})
    df = pd.DataFrame(rows)
    out = Path(RESULTS) / "final" / "mechanism_table.csv"
    df.to_csv(out, index=False)
    for r in df.itertuples():
        print(f"  {r.decision:34s} {r.group:10s} n={r.n_frames:4d}  dFP {r.mean_delta_fp:+.3f}  dFN {r.mean_delta_fn:+.3f}  "
              f"dN {r.mean_delta_n_det:+.3f}  dFP>0 {100 * r.share_delta_fp_pos:.1f}%  dFN>=0 {100 * r.share_delta_fn_nonneg:.1f}%")
    print(f"  wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
