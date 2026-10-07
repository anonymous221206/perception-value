#!/usr/bin/env python
"""Follow-up E6: setting-level decision value of Diffusion Planner on the nuPlan real-perception states, beside
PDM-Closed and IDM on the same states (addendum E6 in docs/followup_protocol.md).

Losses are the external track's (83's `costs`: safety = 10 x collision + 1.5 x clearance shortfall^2; scalar adds 0.1 x
mean log deviation). States where the planner failed in any branch are dropped from every comparison (counted).
Intervals: 1,000 log-bootstrap draws over all logs. Undefined (with the count) when fewer than 30 states are affected.
A/A noise floor (Diffusion Planner only, addendum amendment 1): the reference observation with sampling seed 1 against
seed 0, the share of states whose loss changes and its mean absolute change, beside the same for cheap against full.
Populations (amendment 2): `all` from raw.csv (after the rerun of the 180 states that failed on the history padding),
and `run1` from raw_run1.csv (the first run, 1,260 states), side by side.
-> results/followup/E6/settings.csv, aa_noise.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import xl                                                               # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402

OUT = Path(RESULTS) / "followup" / "E6"
FINAL = Path(RESULTS) / "final"
MIN_AFFECTED = 30


def main():
    x83 = xl.t92().x83
    rows, aa = [], []
    for population, fname in (("all", "raw.csv"), ("run1", "raw_run1.csv")):
        if not (OUT / fname).exists():
            continue
        dp = pd.read_csv(OUT / fname)
        ok = (dp[["ok_reference", "ok_cheap", "ok_full", "ok_reference_s1"]].fillna(0) == 1).all(axis=1)
        sources = {"diffusion_planner": dp[ok]}
        for p in ("pdm_closed", "idm"):
            sources[p] = pd.read_csv(FINAL / f"nuplan_real_perception_{p}_raw.csv")
        keep = set(zip(dp[ok].scenario, dp[ok].iteration))
        for planner, d in sources.items():
            d = d[[k in keep for k in zip(d.scenario, d.iteration)]].reset_index(drop=True)
            cc, cf = x83.costs(d, "cheap"), x83.costs(d, "full")
            for target in ("safety", "scalar_J"):
                v, c = cc[target] - cf[target], np.asarray(cc[target], float)
                r = xl.describe(v, c, (0.2,))
                r.update(xl.bootstrap_describe(v, c, d.log.to_numpy(), 1000, np.random.default_rng(0), (0.2,)))
                if r["affected"] < MIN_AFFECTED:
                    for k in ("rho", "rho_lo", "rho_hi"):
                        r[k] = np.nan
                    r["undefined_reason"] = f"{r['affected']} affected states (< {MIN_AFFECTED})"
                rows.append({"population": population, "planner": planner, "target": target, "states": len(d),
                             "logs": d.log.nunique(), "states_dropped_planner_failure": int((~ok).sum()), **r})
        d = dp[ok].reset_index(drop=True)
        for a, b in (("reference_s1", "reference"), ("cheap", "full")):
            ca, cb = x83.costs(d, a), x83.costs(d, b)
            for target in ("safety", "scalar_J"):
                v = ca[target] - cb[target]
                aa.append({"population": population, "pair": f"{a}-{b}", "target": target, "states": len(d),
                           "changed": int((v != 0).sum()), "changed_share": float((v != 0).mean()),
                           "mean_abs_change": float(np.abs(v).mean())})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "settings.csv", index=False)
    print(t[["population", "planner", "target", "states", "affected", "harmed_share", "harmed_share_lo",
             "harmed_share_hi", "rho", "all_full_reduction", "oracle20_reduction"]].round(3).to_string(index=False))
    aa = pd.DataFrame(aa)
    aa.to_csv(OUT / "aa_noise.csv", index=False)
    print(aa.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
