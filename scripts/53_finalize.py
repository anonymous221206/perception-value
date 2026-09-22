#!/usr/bin/env python
"""Jetson cost columns for core_matrix.csv, the run manifest, and two by-products.

The by-products, headline_table.csv and statistical_tests.csv, are computed from the post-review core matrix
(52_core_matrix.py --tag core_matrix_postreview). They are not shipped and no paper table uses them: the
robustness table is results/final/core_matrix.csv.
"""
from __future__ import annotations

import argparse, glob, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import budget, egospeed, runmeta, viz  # noqa: E402
from rap.paths import RAW, RESULTS              # noqa: E402

OUT = Path(RESULTS) / "final"
FIG = OUT / "figures"
QUOTAS = [0.10, 0.20, 0.30, 0.50]
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}


def eta(df, score, col, quota=0.20):
    pick = lambda s: budget.select_pooled(np.asarray(s, float), quota)
    allc = float(df[col[0]].sum())
    orc = budget.total_risk(df, pick((df[col[0]] - df[col[1]]).to_numpy()), col)
    tot = budget.total_risk(df, pick(np.asarray(score, float)), col)
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def per_seq_eta(df, score, col, quota=0.20, min_n=40):
    out = {}
    s = np.asarray(score, float)
    for g, idx in df.groupby("seq").groups.items():
        pos = df.index.get_indexer(idx)
        if len(pos) < min_n:
            continue
        sub = df.iloc[pos].reset_index(drop=True)
        v = eta(sub, s[pos], col, quota)
        if np.isfinite(v):
            out[g] = float(v)
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, float)
    n, order = len(p), np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for i, k in enumerate(order):
        running = max(running, (n - i) * p[k])
        adj[k] = min(running, 1.0)
    return adj


def latest(tag):
    """The core matrix is resolved through rap.runs, which prefers a corrected copy; other tags glob as before."""
    if tag == "core_matrix":
        return rap_runs.core_matrix("plain")
    d = sorted(glob.glob(str(RAW / f"*_{tag}")))
    return Path(d[-1]) if d else None


def statistical_tests(run, tables: Path, policies=None, write_final=True):
    """The per-sequence Wilcoxon tests, from the decision tables of one core-matrix run.

    `policies` restricts which policies are recomputed. `write_final=False` writes the run copy only: the Holm
    adjustment couples every row of the file, so a partial recomputation cannot be spliced into it without moving
    rows this part does not own (Task 22 Part B).
    """
    tests = []
    for pkl in sorted(tables.glob("*.pkl")):
        d = pd.read_pickle(pkl).reset_index(drop=True)
        parts = pkl.stem.split("__")
        ds, detname, geo = parts[0], parts[1], parts[-1]
        d = egospeed.attach(d, ds)
        for tname, col in TASKS.items():
            dj = (d[col[0]] - d[col[1]]).to_numpy()
            for pol, score in [("perception oracle", d["dE"].to_numpy()),
                               ("uncertainty", d.unc_sum.to_numpy()),
                               ("criticality", d.crit_sum.to_numpy()),
                               ("ego speed", d[egospeed.COL].to_numpy())]:
                if policies and pol not in policies:
                    continue
                a = per_seq_eta(d, score, col)
                b = per_seq_eta(d, dj, col)          # decision oracle, = 1 by construction
                keys = sorted(set(a) & set(b))
                if len(keys) < 5:
                    continue
                va = np.array([a[k] for k in keys])
                vb = np.array([b[k] for k in keys])
                w = stats.wilcoxon(vb, va, alternative="greater")
                tests.append({
                    "config": pkl.stem, "dataset": ds, "detector": detname,
                    "geometry": geo, "task": tname, "policy": pol,
                    "n_sequences": len(keys), "median_policy_eta": float(np.median(va)),
                    "iqr_policy_eta": float(np.percentile(va, 75) - np.percentile(va, 25)),
                    "n_seq_policy_above_0.5": int((va > 0.5).sum()),
                    "median_gap_to_oracle": float(np.median(vb - va)),
                    "wilcoxon_p": float(w.pvalue),
                })
    st = pd.DataFrame(tests)
    if not len(st):
        return st
    st["holm_p"] = holm(st.wilcoxon_p.to_numpy())
    st.to_csv(run / "statistical_tests.csv", index=False)
    if write_final:
        st.to_csv(OUT / "statistical_tests.csv", index=False)
    print(f"statistical_tests.csv: {len(st)} tests, "
          f"{int((st.holm_p < 0.05).sum())} significant after Holm"
          f"{'' if write_final else ' (run copy only)'}")
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="finalize")
    ap.add_argument("--tables", default=None,
                    help="the core-matrix run holding the decision tables (default: the newest *_core_matrix)")
    ap.add_argument("--tests_only", action="store_true",
                    help="write statistical_tests.csv and nothing else: no core matrix, headline or manifest")
    ap.add_argument("--policies", nargs="+", default=None, help="with --tests_only: recompute only these policies")
    ap.add_argument("--run_copy_only", action="store_true",
                    help="with --tests_only: write the run copy and leave results/final alone")
    ap.add_argument("--write_core_matrix", action="store_true",
                    help="also rewrite results/final/core_matrix.csv with the Jetson cost columns. Off by "
                         "default: 52_core_matrix.py owns that file, and this stage would change its schema")
    egospeed.add_argument(ap)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    egospeed.configure(args)
    run = runmeta.new_run(args.tag, vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    if args.tests_only:
        statistical_tests(run, Path(args.tables) if args.tables else latest("core_matrix"),
                          args.policies, not args.run_copy_only)
        print("wrote", run)
        return

    cm = pd.read_csv(OUT / "core_matrix.csv")

    # ---------------- Jetson costs ----------------
    prof = {}
    for d in sorted(glob.glob(str(RAW / "*_profile"))):
        f = Path(d) / "profile_summary.json"
        if f.exists():
            for k, v in json.loads(f.read_text()).items():
                if not k.startswith("_"):
                    prof[k] = v
    def cost(mode, key):
        return prof.get(mode, {}).get(key, np.nan)

    cm["cheap_e2e_ms"] = cm.cheap_mode.map(lambda m: cost(m, "lat_e2e_ms_median"))
    cm["full_e2e_ms"] = cm.full_mode.map(lambda m: cost(m, "lat_e2e_ms_median"))
    cm["cheap_gpu_ms"] = cm.cheap_mode.map(lambda m: cost(m, "lat_fwd_ms_median"))
    cm["full_gpu_ms"] = cm.full_mode.map(lambda m: cost(m, "lat_fwd_ms_median"))
    cm["cheap_mj"] = cm.cheap_mode.map(lambda m: cost(m, "energy_gpu_mj_per_frame"))
    cm["full_mj"] = cm.full_mode.map(lambda m: cost(m, "energy_gpu_mj_per_frame"))
    cm["compute_ratio"] = cm.full_gpu_ms / cm.cheap_gpu_ms
    cm["energy_ratio"] = cm.full_mj / cm.cheap_mj
    cm.to_csv(run / "core_matrix_with_costs.csv", index=False)
    if args.write_core_matrix:
        cm.to_csv(OUT / "core_matrix.csv", index=False)
    else:
        print("core_matrix.csv left alone (52_core_matrix.py owns it); the cost columns are in the run directory")

    # ---------------- headline table ----------------
    head = cm[["dataset", "detector", "cheap_mode", "full_mode", "task", "geometry",
               "n_sequences", "n_frames", "compute_ratio", "energy_ratio",
               "action_change_rate", "improved_same_action_rate", "harmful_full_rate",
               "corr_deltaE_deltaJ", "eta_random_20", "eta_uncertainty_20",
               "eta_criticality_20", "eta_perception_oracle_20",
               "eta_best_perception_metric_20",
               "eta_multimetric_perception_oracle_20", "top20_cross_task_overlap",
               "best_trivial_heuristic", "best_trivial_eta20"]].copy()
    head["oracle_gap_20"] = 1.0 - head.eta_perception_oracle_20
    head.to_csv(OUT / "headline_table.csv", index=False)
    print(f"headline_table.csv: {len(head)} rows")

    # ---------------- per-sequence statistics ----------------
    st = statistical_tests(run, Path(args.tables) if args.tables else latest("core_matrix"))

    # ---------------- run manifest ----------------
    man = []
    for d in sorted(glob.glob(str(RAW / "*"))):
        d = Path(d)
        cfgf, envf = d / "config.json", d / "environment.json"
        if not cfgf.exists():
            continue
        env = json.loads(envf.read_text()) if envf.exists() else {}
        man.append({"run": d.name, "tag": d.name.split("_", 2)[-1],
                    "git_commit": env.get("git_commit", ""),
                    "git_dirty": env.get("git_dirty", ""),
                    "torch": env.get("torch", ""), "gpu": env.get("gpu", ""),
                    "config": json.dumps(json.loads(cfgf.read_text()))[:400]})
    pd.DataFrame(man).to_csv(OUT / "run_manifest.csv", index=False)
    print(f"run_manifest.csv: {len(man)} runs")

    (run / "done.json").write_text(json.dumps({"rows": len(cm), "tests": len(st)}, indent=2))
    print("wrote", run)


if __name__ == "__main__":
    main()
