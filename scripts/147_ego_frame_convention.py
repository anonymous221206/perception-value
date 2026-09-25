#!/usr/bin/env python
"""Task 23 Phase 4: OLD (camera frame, as shipped) beside NEW (ego frame) for every registered quantity.

Pre-registered in the pre-registration record (not part of this release) (Task 23, "Quantities reported OLD -> NEW" and "Registered reading", and the
amendment's B, D and G5).  OLD is the snapshot of results/final taken before any Task 23 run
(`*_ego_frame_before/results_final`); NEW is results/final after the ego-frame regeneration.  Nothing is recomputed
from the per-frame tables here: every figure is read from the official files, so the comparison is the one a reader
of the two releases would make.

Writes
  results/final/ego_frame_convention.csv           long format: group, file, item, statistic, old, new, delta, note
  results/final/ego_frame_convention_detail.csv.gz the same columns: every row of the budget, energy, streaming,
                                                   skip-accounting and control tables (groups 7 and 8), whose
                                                   summaries are in the main file
  results/final/ego_frame_convention_reading.json  Q1-Q8 under OLD and NEW, the file inventory, and the gate summaries
A statistic empty in both files is not listed.

Groups: 0 the registered reading; 1 calibration cells; 2 nDG of every deployable signal; 3 diagnostic signals;
4 core_matrix.csv in full; 5 nuScenes-oracle sign agreement per gain; 6 mechanism table; 7 budgets, energy,
streaming and skip accounting; 8 controls; 9 every file of results/final (rows moved, largest change, identical);
B, D, G5 the amendment's measurements and gate, from their runs.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
OUT_CSV, OUT_DETAIL, OUT_JSON = ("ego_frame_convention.csv", "ego_frame_convention_detail.csv.gz",
                                 "ego_frame_convention_reading.json")
QUOTAS = (0.1, 0.2, 0.3, 0.5)
FIVE = ("affected", "harm_rate", "rho", "all_full_reduction", "oracle20_reduction")
GAINS = ("exact_FN", "FN_FP", "E5_combined", "E_risk")
LEARNED = ("gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf")
HARM_TOL, RHO_TOL = 0.05, 0.05                                                  # Task 22 Part C's tolerance
REGISTERED_SCHEMES = ("S0", "S1", "S2", "S3")


# ------------------------------------------------------------------------------------------------------ utilities

def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def text(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def same(a, b) -> bool:
    fa, fb = num(a), num(b)
    if fa is not None and fb is not None:
        return (fa == fb) or (np.isnan(fa) and np.isnan(fb))
    return str(a) == str(b)


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v)
    return "" if v is None else str(v)


def empty(v) -> bool:
    return v is None or (isinstance(v, str) and v == "") or (not isinstance(v, str) and pd.isna(v))


class Rows:
    def __init__(self):
        self.rows, self.detail = [], []

    def add(self, group, file, item, statistic, old, new, note="", detail=False):
        if empty(old) and empty(new):
            return
        fo, fn = num(old), num(new)
        delta = fn - fo if (fo is not None and fn is not None and not isinstance(old, str) and
                            not isinstance(new, str)) else ""
        (self.detail if detail else self.rows).append(
            {"group": group, "file": file, "item": item, "statistic": statistic, "old": fmt(old), "new": fmt(new),
             "delta": fmt(delta) if delta != "" else "", "note": note})


def keyed(old: pd.DataFrame, new: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    """Rows of the two files side by side on `keys` (suffixes _o / _n); a row present in one file only is kept."""
    o, n = old.copy(), new.copy()
    for d in (o, n):
        for k in keys:
            d[k] = d[k].astype(str).replace("nan", "")
    dup = o.duplicated(keys).any() or n.duplicated(keys).any()
    if dup:
        raise SystemExit(f"{name}: the keys {keys} do not identify rows")
    return o.merge(n, on=keys, how="outer", suffixes=("_o", "_n"), indicator=True)


def item_of(r, keys) -> str:
    return " | ".join(str(r[k]) for k in keys if str(r[k]) != "")


def side(lo, hi) -> str:
    lo, hi = num(lo), num(hi)
    if lo is None or hi is None or np.isnan(lo) or np.isnan(hi):
        return "undefined"
    return "above random" if lo > 0 else ("below random" if hi < 0 else "includes random")


def table_rows(R: Rows, group, name, old, new, keys, stats, note_fn=None, detail=False):
    m = keyed(old, new, keys, name)
    for _, r in m.iterrows():
        it = item_of(r, keys)
        if r["_merge"] != "both":
            R.add(group, name, it, "row", "present" if r["_merge"] == "left_only" else "absent",
                  "present" if r["_merge"] == "right_only" else "absent", "row in one file only", detail=detail)
            continue
        for s in stats:
            if f"{s}_o" in r:
                R.add(group, name, it, s, r[f"{s}_o"], r[f"{s}_n"], note_fn(r, s) if note_fn else "", detail=detail)
    return m


def wins_summary(R: Rows, group, name, old, new, by):
    """Per group of rows: how many are feasible and how many have a paired 95% interval above random."""
    def count(d):
        out = {}
        for key, x in d.groupby(by, dropna=False):
            feas = x.feasible.astype(str).str.lower() == "true" if "feasible" in x else pd.Series(True, x.index)
            out[key] = (int(feas.sum()), int((x.minus_random_lo > 0).sum()), int(len(x)))
        return out
    co, cn = count(old), count(new)
    for key in sorted(set(co) | set(cn), key=str):
        it = " | ".join(str(k) for k in (key if isinstance(key, tuple) else (key,)) if str(k) != "nan")
        a, b = co.get(key, (np.nan,) * 3), cn.get(key, (np.nan,) * 3)
        R.add(group, name, it, "rows", a[2], b[2])
        R.add(group, name, it, "feasible rows", a[0], b[0])
        R.add(group, name, it, "rows above random (paired 95% CI)", a[1], b[1])


# ------------------------------------------------------------------------------------------------------ groups 1-8

def g1_calibration(R, old, new):
    keys = ["cell", "scheme", "split"]
    o, n = old[old.split == "all"], new[new.split == "all"]
    m = keyed(o, n, keys, "calibration_cells.csv")
    moved = []
    for _, r in m[m._merge == "both"].iterrows():
        it = item_of(r, keys)
        dh = r.harm_rate_n - r.harm_rate_o
        dr = r.rho_n - r.rho_o
        out = abs(dh) > HARM_TOL or abs(dr) > RHO_TOL
        note = "outside Task 22 tolerance" if out else ""
        if out:
            moved.append(f"{r.cell} {r.scheme} (harmed {dh:+.3f}, rho {dr:+.3f})")
        for s in FIVE + ("harm_rate_lo", "harm_rate_hi", "rho_lo", "rho_hi"):
            R.add("1 calibration cells", "calibration_cells.csv", it, s, r[f"{s}_o"], r[f"{s}_n"], note)
    return moved


def g2_benchmark(R, name, old, new, deployable: bool, group):
    keys = ["track", "geometry", "system", "target", "split", "signal", "quota"]
    if "deployable" in old.columns:
        o = old[old.deployable.astype(str).str.lower() == str(deployable).lower()]
        n = new[new.deployable.astype(str).str.lower() == str(deployable).lower()]
    else:
        o, n = old, new
    o, n = o[o.quota.isin(QUOTAS)], n[n.quota.isin(QUOTAS)]
    m = keyed(o, n, keys, name)
    flips = []
    for _, r in m.iterrows():
        it = item_of(r, keys)
        if r["_merge"] != "both":
            R.add(group, name, it, "row", r["_merge"], "", "row in one file only")
            continue
        so, sn = side(r.minus_random_lo_o, r.minus_random_hi_o), side(r.minus_random_lo_n, r.minus_random_hi_n)
        note = ""
        if so != sn:
            note = ("win appears" if sn == "above random" else "win disappears" if so == "above random"
                    else "interval changes side")
            flips.append({"file": name, "item": it, "old": so, "new": sn, "note": note})
        for s in ("eta", "eta_lo", "eta_hi", "minus_random", "minus_random_lo", "minus_random_hi"):
            R.add(group, name, it, s, r[f"{s}_o"], r[f"{s}_n"], note)
        R.add(group, name, it, "paired 95% CI against random", so, sn, note)
    return flips


def g4_core(R, old, new):
    keys = ["dataset", "detector", "cheap_mode", "full_mode", "task", "geometry"]
    stats = [c for c in old.columns if c not in keys]
    table_rows(R, "4 core matrix", "core_matrix.csv", old, new, keys, stats)


def g5_sign(R, old, new):
    keys = ["cell", "scheme", "split"]
    sel = lambda d: d[(d.dataset == "nuScenes") & (d.geometry == "oracle") & (d.split == "all")]
    stats = [f"{g}_{s}" for g in GAINS for s in ("spearman", "sign_disagreement", "harmed_given_gain_pos",
                                                 "harmed_given_gain_pos_affected", "n_both_nonzero")]
    table_rows(R, "5 nuScenes oracle sign agreement", "calibration_cells.csv", sel(old), sel(new), keys, stats)


def g6_mechanism(R, old, new):
    keys = ["decision", "dJ_column", "group"]
    table_rows(R, "6 mechanism table", "mechanism_table.csv", old, new, keys, [c for c in old.columns if c not in keys])


BUDGET_KEYS = ["track", "geometry", "system", "target", "unit", "budget_level", "signal"]
BUDGET_STATS = ["feasible", "escalated_frac", "eta", "eta_lo", "eta_hi", "minus_random", "minus_random_lo",
                "minus_random_hi"]
MULTI_KEYS = ["system", "unit", "budget_level", "signal"]
MULTI_STATS = ["feasible", "eta", "eta_lo", "eta_hi", "gain", "oracle_gain", "share_384", "share_512", "share_640"]
GROUP7 = {
    "benchmark_budget_two_level.csv": (BUDGET_KEYS, BUDGET_STATS),
    "benchmark_budget_two_level_1thread.csv": (BUDGET_KEYS, BUDGET_STATS),
    "benchmark_budget_routers.csv": (BUDGET_KEYS, BUDGET_STATS),
    "benchmark_budget_multifidelity.csv": (MULTI_KEYS, MULTI_STATS),
    "benchmark_budget_multifidelity_1thread.csv": (MULTI_KEYS, MULTI_STATS),
    "energy_module_budget_two_level.csv": (BUDGET_KEYS + ["table", "convention"], BUDGET_STATS),
    "energy_module_budget_multifidelity.csv": (MULTI_KEYS + ["table", "convention"], MULTI_STATS),
    "energy_module_claims.csv": (["claim", "convention", "budget_level", "quantity"], ["value", "holds"]),
    "fig_budget_curves.csv": (["row_type", "track", "geometry", "system", "target", "signal", "budget_level"],
                              ["feasible", "eta", "eta_lo", "eta_hi", "minus_random_lo", "minus_random_hi"]),
}


def g7_budgets(R, before):
    G = "7 budgets, energy, streaming, skip accounting"
    for name, (keys, stats) in GROUP7.items():
        if (before / name).exists() and (FINAL / name).exists():
            o, n = read(before / name), read(FINAL / name)
            table_rows(R, G, name, o, n, keys, stats, detail=name != "energy_module_claims.csv")
            if "minus_random_lo" in o.columns and name != "fig_budget_curves.csv":
                wins_summary(R, G, name, o, n, [k for k in ("track", "geometry", "system", "target", "unit",
                                                            "budget_level", "table", "convention") if k in o.columns])
    # streaming controllers: the pooled table and every row's nDG - C
    o, n = read(before / "streaming_controllers.csv"), read(FINAL / "streaming_controllers.csv")
    keys = ["section", "signal", "rate_target", "variant", "policy"]
    table_rows(R, "7 budgets, energy, streaming, skip accounting", "streaming_controllers.csv",
               o[o.section == "pooled"], n[n.section == "pooled"], keys,
               ["ndg_minus_C", "ndg_minus_C_lo", "ndg_minus_C_hi", "rows_improved_over_C", "rows_rate_miss_gt_5pp",
                "median_abs_rate_dev", "reading"])
    keys = ["section", "signal", "rate_target", "variant", "policy", "track", "geometry", "system", "target"]
    table_rows(R, "7 budgets, energy, streaming, skip accounting", "streaming_controllers.csv",
               o[o.section == "row"], n[n.section == "row"], keys, ["ndg", "ndg_C", "ndg_minus_C", "rate_realised"],
               detail=True)
    keys = ["section", "rate_target", "policy", "W", "gP", "gI", "capacity"]
    table_rows(R, "7 budgets, energy, streaming, skip accounting", "streaming_controllers.csv",
               o[o.section == "hyperparameters"], n[n.section == "hyperparameters"], keys,
               ["val_median_abs_dev", "val_mean_ndg_minus_C", "chosen", "qualifies"])
    # skip accounting: the evaluation and the shares, every numeric column
    o, n = read(before / "skip_accounting.csv"), read(FINAL / "skip_accounting.csv")
    for sec, keys in (("evaluation", ["section", "track", "geometry", "system", "target", "unit", "budget_level"]),
                      ("shares", ["section", "track", "geometry", "system", "target", "signal", "unit", "budget_level",
                                  "regime", "denominator"])):
        oo, nn = o[o.section == sec].dropna(axis=1, how="all"), n[n.section == sec].dropna(axis=1, how="all")
        keys = [k for k in keys if k in oo.columns]
        stats = [c for c in oo.columns if c not in keys and c in nn.columns]
        table_rows(R, "7 budgets, energy, streaming, skip accounting", "skip_accounting.csv", oo, nn, keys, stats,
                   detail=sec == "evaluation")


def g8_controls(R, before):
    G = "8 controls"
    # statistics hardening: harm, the raw-gain win counts per quota, and leave-one-unit-out influence
    o, n = read(before / "statistics_hardening.csv"), read(FINAL / "statistics_hardening.csv")
    keys = ["section", "track", "geometry", "system", "target", "split"]
    table_rows(R, G, "statistics_hardening.csv", o[o.section == "harm"], n[n.section == "harm"], keys,
               ["n_inputs", "n_affected", "n_harmed", "harmed_over_affected", "harmed_over_all_inputs"])
    keys = ["section", "track", "geometry", "system", "target", "signal", "quota"]
    table_rows(R, G, "statistics_hardening.csv", o[o.section == "paired_gain"], n[n.section == "paired_gain"], keys,
               ["delta_vs_random", "delta_share_all_cheap", "delta_lo_all_draws", "delta_hi_all_draws",
                "beats_random_all_draws", "beats_random_filtered", "ndg"], detail=True)
    for q in QUOTAS:
        for col in ("beats_random_all_draws", "beats_random_filtered"):
            def wins(d):
                x = d[(d.section == "paired_gain") & np.isclose(d.quota, q)]
                return int((x[col].astype(str).str.lower() == "true").sum()), int(x[col].notna().sum())
            (wo, no), (wn, nn) = wins(o), wins(n)
            R.add(G, "statistics_hardening.csv", f"raw-gain paired rows at {q:.0%}", f"wins ({col})",
                  f"{wo} of {no}", f"{wn} of {nn}")
    table_rows(R, G, "statistics_hardening.csv", o[o.section == "influence"], n[n.section == "influence"], keys,
               ["ndg_full", "ndg_min_leave_one_out", "ndg_max_leave_one_out", "most_influential_unit",
                "ndg_change_without_that_unit"], detail=True)
    for q in QUOTAS:
        def spread(d):
            x = d[(d.section == "influence") & np.isclose(d.quota, q)]
            return float((x.ndg_max_leave_one_out - x.ndg_min_leave_one_out).median())
        R.add(G, "statistics_hardening.csv", f"leave-one-unit-out influence at {q:.0%}",
              "median nDG range over left-out units", spread(o), spread(n))
    # consumer transfer: the summaries and every entry's transfer regret
    o, n = read(before / "consumer_transfer.csv"), read(FINAL / "consumer_transfer.csv")
    keys = ["group", "track", "geometry", "trained_for", "evaluated_on", "signal", "quota"]
    summ = lambda d: d[d.trained_for == "SUMMARY"]
    entries = lambda d: d[d.trained_for != "SUMMARY"]
    table_rows(R, G, "consumer_transfer.csv", summ(o), summ(n), keys,
               ["n_entries", "median_transfer_regret", "off_diag_beating_random_all_draws",
                "off_diag_beating_random_filtered", "diag_beating_random_all_draws", "n_diag_entries"])
    table_rows(R, G, "consumer_transfer.csv", entries(o), entries(n), keys,
               ["ndg", "ndg_diag", "transfer_regret", "beats_random_all_draws", "beats_random_filtered"], detail=True)
    # objective swap: every comparison
    o, n = read(before / "objective_swap.csv"), read(FINAL / "objective_swap.csv")
    keys = ["section", "pool", "track", "geometry", "system", "target", "objective", "quota"]
    cmp = lambda d: d[d.section == "compare"]
    table_rows(R, G, "objective_swap.csv", cmp(o), cmp(n), keys,
               ["e_dec_defined", "argmax_dec", "argmax_perc", "argmax_differs", "kendall_tau", "regret_ndg",
                "regret_gain_lo", "regret_gain_hi", "perc_winner_worse_than_random"])
    # target swap: every row, and the pooled summary
    o, n = read(before / "benchmark_target_swap.csv"), read(FINAL / "benchmark_target_swap.csv")
    keys = ["track", "geometry", "system", "target", "g_variant", "arch", "setting", "quota"]
    table_rows(R, G, "benchmark_target_swap.csv", o, n, keys,
               ["eta_V", "V_minus_random_lo", "V_minus_random_hi", "eta_G", "G_minus_random_lo", "G_minus_random_hi",
                "diff", "diff_lo", "diff_hi", "sign_agree_train"], detail=True)
    jo = json.loads((before / "benchmark_target_swap_summary.json").read_text())
    jn = json.loads((FINAL / "benchmark_target_swap_summary.json").read_text())
    for label in jo["pooled"]:
        R.add(G, "benchmark_target_swap_summary.json", f"pooled {label}", "reading",
              jo["pooled"][label]["reading"], jn["pooled"][label]["reading"])
        for arch, v in jo["pooled"][label]["per_architecture"].items():
            w = jn["pooled"][label]["per_architecture"][arch]
            for s in ("mean_diff_V_minus_G", "ci_lo", "ci_hi", "cells_V_gt_G", "cells_diff_ci_above_0",
                      "cells_diff_ci_below_0"):
                R.add(G, "benchmark_target_swap_summary.json", f"pooled {label} {arch}", s, v[s], w[s])


# ------------------------------------------------------------------------------------------------------ group 9

# columns that identify a row in the results/final tables; a file's rows are aligned on those it has, when they are
# unique in both versions, so a table whose rows were only reordered is not reported as moved
KEY_NAMES = ("section", "group", "cell", "claim", "dataset", "track", "geometry", "system", "target", "system_note",
             "detector", "cheap_mode", "full_mode", "task", "split", "scheme", "decision", "dJ_column", "pair", "n",
             "unit", "table", "convention", "quantity", "row_type", "variant", "policy", "signal", "arch", "g_variant",
             "setting", "trained_for", "evaluated_on", "objective", "pool", "metric", "label", "reference",
             "statistic", "figure", "t_cheap", "t_full", "rate_target", "budget_level", "quota")


def align(x: pd.DataFrame, y: pd.DataFrame):
    """y's rows in x's order on the key columns, when they identify the rows of both; else y as it is."""
    keys = [k for k in KEY_NAMES if k in x.columns]
    if not keys or x.duplicated(keys).any() or y.duplicated(keys).any():
        return y, keys, False
    kx, ky = x[keys].apply(tuple, axis=1), y[keys].apply(tuple, axis=1)
    if set(kx) != set(ky):
        return y, keys, False
    pos = {k: i for i, k in enumerate(ky)}
    order = [pos[k] for k in kx]
    return y.iloc[order].reset_index(drop=True), keys, order != list(range(len(y)))


# Outputs of later tasks, which this inventory is not about: it reports what moving the lift to the ego frame did to
# the tables that existed then. They would otherwise appear here as "only in NEW".
AFTER_TASK_23 = ("benchmark_decision_values.csv.gz", "benchmark_bootstrap_plans.json", "submission_path_g1.csv",
                 "published_objective_labels.csv", "published_objective_routers.csv", "published_objective_arbiter.csv",
                 "published_objective_reading.json", "paper_evidence_pack.csv", "claims_check.csv",
                 "cost_registry.json", "cached_analyses_selection.csv", "cached_analyses_benefit_harm.csv",
                 "cached_analyses_overhead_tolerance.csv", "cached_analyses_gap_accounting.csv",
                 "cached_analyses_figure_data.csv", "target_transform_control.csv", "target_transform_heatmap.csv",
                 "seed_variation.csv", "figures/target_transform_heatmap.pdf", "figures/target_transform_heatmap.png",
                 # Tasks 32-35: the shared action history, its significance table and loss-term shares, the loss
                 # sensitivity, and Figure 1's frames
                 "shared_history_effect.csv", "significance_table.csv", "loss_term_shares.csv", "loss_sensitivity.csv",
                 "fig_gallery/figure1.json", "fig_gallery/figure1_scene-0055_03.json", "fig_gallery/figure1_scene-0055_03.jpg",
                 "fig_gallery/figure1_scene-0065_24.json", "fig_gallery/figure1_scene-0065_24.jpg")


def inventory(R, before: Path) -> list[dict]:
    names = sorted({p.relative_to(before).as_posix() for p in before.rglob("*") if p.is_file()} |
                   {p.relative_to(FINAL).as_posix() for p in FINAL.rglob("*") if p.is_file()})
    names = [x for x in names if x not in (OUT_CSV, OUT_DETAIL, OUT_JSON) and x not in AFTER_TASK_23]
    out = []
    for name in names:
        a, b = before / name, FINAL / name
        rec = {"file": name}
        if not a.exists() or not b.exists():
            rec["status"] = "only in NEW" if b.exists() else "only in OLD"
        elif a.read_bytes() == b.read_bytes():
            rec["status"] = "byte-identical"
        elif name.endswith((".csv", ".csv.gz")):
            x, y = text(a), text(b)
            if list(x.columns) != list(y.columns) or len(x) != len(y):
                rec.update(status="structure differs", rows_old=len(x), rows_new=len(y))
            else:
                y, keys, reordered = align(x, y)
                diff = np.zeros(x.shape, bool)
                big = (0.0, "", "", "", "")
                for j, c in enumerate(x.columns):
                    for i in np.flatnonzero((x[c] != y[c]).to_numpy()):
                        if same(x.at[i, c], y.at[i, c]):
                            continue
                        diff[i, j] = True
                        fo, fn = num(x.at[i, c]), num(y.at[i, c])
                        d = abs(fn - fo) if (fo is not None and fn is not None and np.isfinite(fo) and
                                             np.isfinite(fn)) else 0.0
                        if d >= big[0]:
                            cols = keys or [k for k in x.columns if num(x.at[i, k]) is None]
                            lab = " | ".join(v for v in x.iloc[i][cols].tolist() if v)[:160]
                            big = (d, c, lab, x.at[i, c], y.at[i, c])
                if not diff.any():
                    rec.update(status="rows reordered only", rows=len(x))
                    out.append(rec)
                    R.add("9 every file", name, "file", "status", "", rec["status"])
                    R.add("9 every file", name, "file", "rows", "", rec["rows"])
                    continue
                rec.update(status="values moved", rows=len(x), reordered=bool(reordered), rows_moved=int(diff.any(1).sum()),
                           cells_moved=int(diff.sum()), largest_abs_change=big[0], largest_column=big[1],
                           largest_row=big[2], largest_old=big[3], largest_new=big[4])
        else:
            rec["status"] = "bytes differ"
        out.append(rec)
        R.add("9 every file", name, "file", "status", "", rec["status"])
        for k in ("rows", "reordered", "rows_moved", "cells_moved", "largest_abs_change", "largest_column", "largest_row",
                  "largest_old", "largest_new", "rows_old", "rows_new"):
            if k in rec:
                R.add("9 every file", name, "file", k, "", rec[k])
    return out


# ------------------------------------------------------------------------------------------------------ the reading

def readings(d: Path) -> dict:
    cal = read(d / "calibration_cells.csv")
    cal = cal[cal.split == "all"]
    reg = cal[cal.scheme.isin(REGISTERED_SCHEMES)]
    out = {}

    q1 = reg[(reg.dataset == "nuScenes") | (reg.moderate_gap_kitti.astype(str).str.lower() == "true")]
    fail = q1[(q1.harm_rate < 0.20) | (q1.rho < 0.20)]
    out["Q1"] = {"holds": bool(len(fail) == 0), "rows": int(len(q1)),
                 "min_harmed": float(q1.harm_rate.min()), "min_rho": float(q1.rho.min()),
                 "failing": [f"{r.cell} {r.scheme} harmed {r.harm_rate:.3f} rho {r.rho:.3f}" for r in fail.itertuples()]}

    q2 = reg[(reg.dataset == "nuScenes") & (reg.geometry == "oracle")]
    bad, worst_dis, worst_rho = [], 1.0, -1.0
    for g in GAINS:
        for r in q2.itertuples():
            dis, rho = getattr(r, f"{g}_sign_disagreement"), getattr(r, f"{g}_spearman")
            worst_dis, worst_rho = min(worst_dis, dis), max(worst_rho, rho)
            if not (dis >= 0.25 and rho < 0.2):
                bad.append(f"{r.cell} {r.scheme} {g}: disagreement {dis:.3f}, Spearman {rho:+.3f}")
    out["Q2"] = {"holds": not bad, "min_sign_disagreement": float(worst_dis), "max_spearman": float(worst_rho),
                 "failing": bad}

    wins = []
    for name in ("benchmark_table.csv", "benchmark_table_routers.csv"):
        t = read(d / name)
        t = t[(t.track == "nuScenes") & (t.split == "test") & np.isclose(t.quota, 0.2) &
              (t.deployable.astype(str).str.lower() == "true")]
        for r in t[t.minus_random_lo > 0].itertuples():
            wins.append(f"{name}: {r.geometry} {r.system} {r.target} {r.signal} nDG {r.eta:.3f} "
                        f"[{r.minus_random_lo:+.3f}, {r.minus_random_hi:+.3f}] vs random")
    out["Q3"] = {"holds": not wins, "nuScenes_rows_above_random_at_20": wins}

    j = json.loads((d / "benchmark_target_swap_summary.json").read_text())
    excl = [f"{lab} {a}: [{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]" for lab, p in j["pooled"].items()
            for a, v in p["per_architecture"].items() if not (v["ci_lo"] <= 0 <= v["ci_hi"])]
    out["Q4"] = {"holds": not excl, "readings": {lab: p["reading"] for lab, p in j["pooled"].items()},
                 "pooled_intervals_excluding_zero": excl}

    t = read(d / "causal_threshold.csv")
    bc = t[(t.track == "pooled") & (t.variant == "V1") & (t.signal == "all_learned") & (t.policy == "B-C") &
           np.isclose(t.rate_target, 0.2)].iloc[0]
    straddles = bool(bc.ndg_lo < -0.05 < bc.ndg_hi)
    st = read(d / "streaming_controllers.csv")
    p = st[(st.section == "pooled") & (st.signal == "all_learned") & np.isclose(st.rate_target, 0.2) &
           st.policy.isin(["D", "E"])]
    rec = {r.policy: str(r.reading) for r in p.itertuples()}
    out["Q5"] = {"holds": straddles and all(v == "does not recover" for v in rec.values()) and len(rec) == 2,
                 "B_minus_C": [float(bc.ndg), float(bc.ndg_lo), float(bc.ndg_hi)],
                 "B_minus_C_reading": "inconclusive" if straddles else
                 ("streaming holds" if bc.ndg_lo > -0.05 else "streaming costs"),
                 "controllers": rec}

    b = t[np.isclose(t.budget_level, 0.5)]
    cells = []
    for key, x in b.groupby(["track", "geometry", "system", "target"], dropna=False):
        uni = x[(x.variant == "baseline") & (x.signal == "uniform_full")].gain_share_all_cheap
        alloc = x[(x.variant == "V1") & (x.feasible.astype(str).str.lower() == "true")].gain_share_all_cheap
        if len(uni) != 1:
            continue
        best = float(alloc.max()) if len(alloc) else np.nan
        cells.append({"cell": " ".join(str(k) for k in key if str(k) != "nan"), "uniform_full": float(uni.iloc[0]),
                      "best_feasible_allocator": best,
                      "uniform_stronger": bool(not np.isfinite(best) or float(uni.iloc[0]) > best)})
    n_uni = sum(c["uniform_stronger"] for c in cells)
    out["Q6"] = {"holds": n_uni >= 8, "uniform_stronger_cells": n_uni, "cells": len(cells),
                 "allocator_wins": [c["cell"] for c in cells if not c["uniform_stronger"]], "per_cell": cells}

    # "every scheme" of the reading: S0-S3; S4 is reported but does not enter the reading (Task 2 pre-registration)
    k7a = cal[cal.cell.str.startswith(("KITTI mono Y8 320->640", "KITTI oracle Y8 320->640"))]
    k7 = k7a[k7a.scheme.isin(REGISTERED_SCHEMES)]
    fail7 = k7[k7.rho > 0.2]
    out["Q7"] = {"holds": bool(len(fail7) == 0), "schemes": sorted(k7.scheme.unique().tolist()),
                 "max_rho": float(k7.rho.max()), "max_rho_S4_reported_only": float(k7a[k7a.scheme == "S4"].rho.max()),
                 "failing": [f"{r.cell} {r.scheme} rho {r.rho:.3f}" for r in fail7.itertuples()]}

    o = read(d / "objective_swap.csv")
    o = o[(o.section == "compare") & (o.pool == "headline") & (o.e_dec_defined.astype(str).str.lower() == "true")]
    per = {}
    for obj, x in o.groupby("objective"):
        per[obj] = {"argmax_differs": int((x.argmax_differs.astype(str).str.lower() == "true").sum()),
                    "comparisons": int(len(x))}
    out["Q8"] = {"holds": all(v["argmax_differs"] >= 26 for v in per.values()) and len(per) == 2, "per_variant": per}
    return out


# ------------------------------------------------------------------------------------------------------ B, D, G5

def gates(R) -> dict:
    out = {}
    for part, files in (("b", ("b_variant_difference.csv", "b_accuracy_vs_reference.csv")),
                        ("d", ("d_bias_paired.csv", "d_bias_per_mode.csv")),
                        ("g5", ("g5_oracle_tables.csv", "g5_submissions.csv")),
                        ("g4", ("g4_primitives.csv",))):
        try:
            run = rap_runs.latest(f"ego_frame_gate_{part}")
        except SystemExit:
            continue
        out[part] = run.name
        for f in files:
            if not (run / f).exists():
                continue
            t = read(run / f)
            keys = [c for c in t.columns if t[c].dtype == object or c in ("band", "mode", "mode_role")]
            for _, r in t.iterrows():
                it = " | ".join(str(r[k]) for k in keys if str(r[k]) != "nan")
                for c in t.columns:
                    if c not in keys:
                        R.add(part.upper(), f"{run.name}/{f}", it, c, "", r[c])
    return out


def repo_relative(d) -> str:
    """A directory as a path inside the repository, so no machine's absolute path is written into a result."""
    try:
        return str(Path(d).resolve().relative_to(ROOT))
    except ValueError:
        return str(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default=None, help="the snapshot of results/final taken before Task 23")
    ap.add_argument("--after", default=None, help="testing only: a directory standing in for results/final")
    ap.add_argument("--run_only", action="store_true", help="write the run copies only, not results/final")
    args = ap.parse_args()
    global FINAL
    FINAL = Path(args.after) if args.after else FINAL
    before = Path(args.before) if args.before else sorted(RAW.glob("*_ego_frame_before"))[-1] / "results_final"
    run = runmeta.new_run("ego_frame_convention", {"before": str(before)})
    R = Rows()

    r_old, r_new = readings(before), readings(FINAL)
    for q in r_old:
        R.add("0 registered reading", "", q, "holds", r_old[q]["holds"], r_new[q]["holds"],
              "" if r_old[q]["holds"] == r_new[q]["holds"] else "REVERSES")
        for k, v in r_old[q].items():
            if k != "holds" and not isinstance(v, (list, dict)):
                R.add("0 registered reading", "", q, k, v, r_new[q][k])

    moved = g1_calibration(R, read(before / "calibration_cells.csv"), read(FINAL / "calibration_cells.csv"))
    flips = []
    for name in ("benchmark_table.csv", "benchmark_table_routers.csv"):
        flips += g2_benchmark(R, name, read(before / name), read(FINAL / name), True, "2 deployable nDG")
    flips += g2_benchmark(R, "benchmark_table.csv", read(before / "benchmark_table.csv"),
                          read(FINAL / "benchmark_table.csv"), False, "3 diagnostic signals")
    g4_core(R, read(before / "core_matrix.csv"), read(FINAL / "core_matrix.csv"))
    g5_sign(R, read(before / "calibration_cells.csv"), read(FINAL / "calibration_cells.csv"))
    g6_mechanism(R, read(before / "mechanism_table.csv"), read(FINAL / "mechanism_table.csv"))
    g7_budgets(R, before)
    g8_controls(R, before)
    inv = inventory(R, before)
    gate_runs = gates(R)

    df = pd.DataFrame(R.rows)
    dd = pd.DataFrame(R.detail)
    outs = [run] if args.run_only else [FINAL, run]
    for d in outs:
        df.to_csv(d / OUT_CSV, index=False)
        dd.to_csv(d / OUT_DETAIL, index=False, compression={"method": "gzip", "mtime": 0})
    summary = {"before": repo_relative(before), "after": repo_relative(FINAL),
               "reading_old": r_old, "reading_new": r_new,
               "reversed": [q for q in r_old if r_old[q]["holds"] != r_new[q]["holds"]],
               "calibration_outside_task22_tolerance": moved, "ci_side_changes": flips,
               "inventory": inv, "gate_runs": gate_runs}
    for d in outs:
        (d / OUT_JSON).write_text(json.dumps(summary, indent=1, default=float))
    print(f"  wrote {outs[0] / OUT_CSV} ({len(df)} rows), {OUT_DETAIL} ({len(dd)} rows) and {OUT_JSON}")
    for q in r_old:
        print(f"  {q}: old {'holds' if r_old[q]['holds'] else 'fails'}, new {'holds' if r_new[q]['holds'] else 'fails'}")
    print(f"  calibration cells outside Task 22's tolerance: {len(moved)}; CI side changes: {len(flips)}")
    print(f"  files: " + ", ".join(f"{k} {v}" for k, v in pd.Series([x['status'] for x in inv]).value_counts().items()))


if __name__ == "__main__":
    main()
