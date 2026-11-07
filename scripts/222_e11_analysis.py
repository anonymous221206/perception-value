#!/usr/bin/env python
"""Follow-up E11: gate E11-0 and the setting-level quantities of LTF's camera-resolution escalation on navtest.

addendum E11 in docs/followup_protocol.md. Reads 221's two score tables (NAVSIM v1.1 run_pdm_score, one per branch), copied to
--scores/{full,cheap}/*.csv, and the navtest token -> log map. V = PDMS(full) - PDMS(cheap) per token; the loss is
1 - PDMS, so the reductions are shares of the all-cheap mean shortfall (E2's convention). Sub-scores likewise.
Writes results/followup/E11/gate_e11_0.json, settings.csv, tokens.csv. --seed 1 / 2 (secondary): the tables in
full_s<seed>, cheap_s<seed>; writes means_seed<n>.json, settings_seed<n>.csv, tokens_seed<n>.csv (the gate is seed 0's).
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta, xl                                  # noqa: E402
from rap.paths import RESULTS                                # noqa: E402
from rap.scoring import EPS                                  # noqa: E402

OUT = Path(RESULTS) / "followup" / "E11"
PUBLISHED_LTF_PDMS = 83.8      # NAVSIM (Dauner et al., NeurIPS 2024 D&B), navtest, Latent TransFuser
SCORES = ["score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
          "time_to_collision_within_bound", "comfort"]          # E2's six (193), NAVSIM v1.1 column names


def table(folder):
    f = sorted(Path(folder).glob("*.csv"))[-1]
    d = pd.read_csv(f, index_col=0)
    return d[d.token != "average"].set_index("token")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--logs", required=True, help="CSV with token,log for navtest (E2's token table)")
    ap.add_argument("--nboot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0, choices=[0, 1, 2], help="LTF checkpoint seed; 0 is primary")
    args = ap.parse_args()
    run = runmeta.new_run("e11_analysis", vars(args))
    sfx, tag = ("", "") if args.seed == 0 else (f"_s{args.seed}", f"_seed{args.seed}")
    full, cheap = table(Path(args.scores) / f"full{sfx}"), table(Path(args.scores) / f"cheap{sfx}")
    log_of = pd.read_csv(args.logs).set_index("token")["log"]
    both = full.index.intersection(cheap.index)
    valid = full.loc[both, "valid"].astype(bool) & cheap.loc[both, "valid"].astype(bool)
    tok = both[valid.to_numpy()]
    OUT.mkdir(parents=True, exist_ok=True)
    g = {"tokens_full": int(len(full)), "tokens_cheap": int(len(cheap)), "valid_both": int(len(tok)),
         "mean_pdms_full": float(100 * full.loc[full.valid.astype(bool), "score"].mean()),
         "mean_pdms_cheap": float(100 * cheap.loc[cheap.valid.astype(bool), "score"].mean()),
         "published": PUBLISHED_LTF_PDMS}
    if args.seed == 0:
        g["pass"] = bool(abs(g["mean_pdms_full"] - PUBLISHED_LTF_PDMS) <= 1.0)
        (OUT / "gate_e11_0.json").write_text(json.dumps(g, indent=1))
        print("gate E11-0:", g)
        if not g["pass"]:
            sys.exit(1)
    else:
        (OUT / f"means{tag}.json").write_text(json.dumps(g, indent=1))
        print(f"seed {args.seed}:", g)
    units = log_of.reindex(tok).to_numpy()
    assert pd.notna(units).all(), "tokens without a log"
    rows = []
    for s in SCORES:
        c = 1.0 - cheap.loc[tok, s].to_numpy(float)
        v = (full.loc[tok, s] - cheap.loc[tok, s]).to_numpy(float)
        r = xl.describe(v, c, (0.2,))
        if r["affected"] < 30:
            r["rho"] = np.nan
        r.update(xl.bootstrap_describe(v, c, units, args.nboot, np.random.default_rng(0), (0.2,)))
        rows.append({"score": s, "n_logs": int(len(set(units))), **r})
        print(f"  {s}: affected {r['affected']} harmed {r['harmed_share']:.3f} rho {r['rho']:.3f} "
              f"all-full {r['all_full_reduction']:+.3f}")
    pd.DataFrame(rows).to_csv(OUT / f"settings{tag}.csv", index=False)
    pd.DataFrame({"token": tok, "log": units, "pdms_full": full.loc[tok, "score"].to_numpy(),
                  "pdms_cheap": cheap.loc[tok, "score"].to_numpy()}).to_csv(OUT / f"tokens{tag}.csv", index=False)
    rep = Path(args.scores) / "full_repeat"
    if args.seed == 0 and list(rep.glob("*.csv")):        # diagnostic, not registered: the same branch run twice
        r2 = table(rep)
        d = (r2.loc[tok, "score"] - full.loc[tok, "score"]).to_numpy(float)
        nz = np.abs(d[d != 0])
        g["repeat"] = {"tokens": int(len(tok)), "changed": int((d != 0).sum()),
                       "mean_pdms": float(100 * r2.loc[tok, "score"].mean()),
                       "abs_diff_q50_q90_q99_max": [float(x) for x in np.quantile(nz, [.5, .9, .99, 1.0])] if len(nz) else []}
        (OUT / "repeat_full.json").write_text(json.dumps(g["repeat"], indent=1))
        print("repeat of branch full:", g["repeat"])
    (run / "summary.json").write_text(json.dumps(g, indent=1))


if __name__ == "__main__":
    main()
