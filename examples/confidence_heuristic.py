#!/usr/bin/env python3
"""Example submission 2: a heuristic that reads only the CHEAP detections.

Score of an input = the summed confidence of the CHEAP detections just below the operating threshold (0.10 <= conf <
0.25): objects the cheap mode almost reported, which the full mode may confirm. It reads `rap.submission.inputs`
only -- the labels-free export of what an allocator may see at test time -- and enters the KITTI and nuScenes cells
(nuPlan's CHEAP track list carries no confidences).

    python examples/confidence_heuristic.py

`score_input` is also what the profiling harness times; confidence_cost_profile.json is its output on the reference
board (python scripts/151_profile_allocator.py examples/confidence_heuristic.py --out ...).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from rap import features, submission as S                                        # noqa: E402

OUT = ROOT / "examples" / "submissions" / "confidence.csv"
PROFILE = ROOT / "examples" / "submissions" / "confidence_cost_profile.json"
LOW, OP = 0.10, 0.25
PROVENANCE = {"near_threshold_conf_mass": "cheap_det"}      # the one feature, and what it is computed from


def score_input(inp):
    c = inp["detections"]["conf"]
    return float(c[(c >= LOW) & (c < OP)].sum())


if __name__ == "__main__":
    import evaluate_submission
    features.check_provenance(PROVENANCE)
    parts = []
    for track in ("KITTI", "nuScenes"):
        det = S.inputs(track)["detections"]
        by = {(s, int(f)): {"conf": g.conf.to_numpy(float)} for (s, f), g in det.groupby(["seq", "frame"])}
        none = {"conf": np.zeros(0)}
        for cell in S.cells():
            if cell[0] == track:
                r = S.rows(cell).drop(columns="unit").astype({"frame": int})
                r["score"] = [score_input({"detections": by.get((s, f), none)}) for s, f in zip(r.seq, r.frame)]
                parts.append(r)
    pd.concat(parts, ignore_index=True).to_csv(OUT, index=False)
    sys.exit(evaluate_submission.main([str(OUT), "--cost_profile", str(PROFILE)]))
