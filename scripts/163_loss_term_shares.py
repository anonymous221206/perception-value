#!/usr/bin/env python
"""Task 32 (b): harm and benefit by loss term, as shares.

For every per-frame table of the braking controller (52, `Jterm_*`) and the trajectory controller (65, `JBterm_*`), on
the shared action history:
- each weighted loss term contributes `d_k,i = term_k(cheap)_i - term_k(full)_i` to `V_i`, and the terms add up to
  `V_i` (checked to 1e-9);
- **harm share** of term k: `sum_{i: V_i < 0} (-d_k,i) / sum_{i: V_i < 0} (-V_i)`, the part of the total harm that
  term carries;
- **benefit share**: the same over `V_i > 0`.

The shares of a set add up to 1. A term can have a negative share when it pulls against the rest in that set.

Terms:
- braking: safety shortfall, collision, excess braking, jerk;
- trajectory: collision, clearance, progress, acceleration, lateral, switching.

All inputs of every setting are used, as in Table 1. Writes results/final/loss_term_shares.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402

EPS = 1e-9
TERMS = {"brake": ("J", "Jterm", ["shortfall", "collision", "excess", "jerk"]),
         "traj": ("JB", "JBterm", ["collision", "clearance", "progress", "acceleration", "lateral", "switching"])}


def shares(d, system, setting):
    j, pre, terms = TERMS[system]
    v = (d[f"{j}_cheap"] - d[f"{j}_full"]).to_numpy(float)
    dk = {k: (d[f"{pre}_cheap_{k}"] - d[f"{pre}_full_{k}"]).to_numpy(float) for k in terms}
    resid = float(np.max(np.abs(v - sum(dk.values()))))
    assert resid < 1e-9, (setting, system, resid)
    rows = []
    for side, m, sign in (("harm", v < -EPS, -1.0), ("benefit", v > EPS, 1.0)):
        tot = float(sign * v[m].sum())
        for k in terms:
            part = float(sign * dk[k][m].sum())
            rows.append({"setting": setting, "system": system, "side": side, "term": k, "n_inputs": len(v),
                         "n_side": int(m.sum()), "total": tot, "term_sum": part,
                         "share": part / tot if tot > EPS else np.nan,
                         "n_side_term_nonzero": int((np.abs(dk[k][m]) > EPS).sum())})
    return rows


def main():
    frames.configure(None)
    core, planb = rap_runs.core_matrix("postreview"), rap_runs.planner_b()
    rows = []
    for f in sorted(core.glob("*.pkl")):
        d = pd.read_pickle(f)
        if "Jterm_cheap_jerk" not in d.columns:
            continue
        rows += shares(d, "brake", f.stem)
    for f in sorted(planb.glob("planB__*.pkl")):
        rows += shares(pd.read_pickle(f), "traj", f.stem.replace("planB__", ""))
    out = pd.DataFrame(rows)
    out["core_run"], out["planner_b_run"] = core.name, planb.name
    out.to_csv(ROOT / "results" / "final" / "loss_term_shares.csv", index=False)
    piv = out.pivot_table(index=["system", "setting"], columns=["side", "term"], values="share")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(piv.round(3).to_string())
    print("wrote", ROOT / "results" / "final" / "loss_term_shares.csv")


if __name__ == "__main__":
    main()
