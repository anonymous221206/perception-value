#!/usr/bin/env python3
"""Example submission 1: a random allocator.

Every test input of every cell gets the same score. Equal scores are ties, and the benchmark evaluates every selection
in exact expectation over random tie-breaks, so this is the uniformly random allocator: its nDG is the benchmark's
`random` row, and its difference from random is exactly zero.

    python examples/random_allocator.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from rap import submission as S                                                  # noqa: E402

OUT = ROOT / "examples" / "submissions" / "random.csv"


def score_input(inp):
    return 0.0


if __name__ == "__main__":
    import evaluate_submission
    parts = []
    for cell in S.cells():
        r = S.rows(cell).drop(columns="unit")
        num = S.IDS[cell[0]][1]
        parts.append(r.astype({num: int}).assign(score=score_input(None)))
    sub = pd.concat(parts, ignore_index=True)
    for c in ("frame", "iteration"):                    # integers where present, empty for the other track
        sub[c] = sub[c].map(lambda x: "" if pd.isna(x) else str(int(x)))
    sub.fillna("").to_csv(OUT, index=False)
    sys.exit(evaluate_submission.main([str(OUT)]))
