#!/usr/bin/env python
"""Task 22 Part A: the OLD -> NEW figure table, and C4 (nothing outside nuScenes moved).

Pre-registered in the pre-registration record (not part of this release) (Task 22 Part A).  `136_class_error_fix.py` corrects the nuScenes class labels in
the per-frame tables; the official stages are then re-run against the corrected tables.  This script compares the
regenerated official files with copies taken before the fix and writes:

  results/final/class_error_fix.csv   every figure the task asks for, old value beside new
  <run>/c4_identity.csv               per file: rows that moved, split into nuScenes and everything else (C4)

Every reported figure is a row of the CSV, so `reproduce.py --verify` covers it.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

FINAL = Path(RESULTS) / "final"
GAINS = ("exact_FN", "FN_FP", "E5_combined", "E_risk")
ARCHS = ("gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf")
LABELS = ("primary", "dE_exact", "dE_E6_risk_weighted")
PER_MODE = ("S1", "S2", "S3", "S4")
# Every official file a stage of this part rewrites, with the column that names the unit (None: all rows are
# nuScenes).  Files that no re-run touches are not listed: comparing them to their own copy would prove nothing.
# `calibration_pr_curves.csv` is one of those, and C1 is what covers it -- its inputs are fn, fp, n_det and n_gt.
IDENTITY = [("core_matrix.csv", "dataset"), ("calibration_cells.csv", "dataset"),
            ("calibration_thresholds.csv", "cell"), ("calibration_sweep.csv", "cell"),
            ("calibration_brake_vs_plan.csv", None), ("phase0f_planning_metric_eta.csv", None),
            ("benchmark_table.csv", "track"), ("benchmark_cells.csv", "track"),
            ("benchmark_self_agreement.csv", "track"), ("benchmark_target_swap.csv", "track"),
            ("statistics_hardening.csv", "track")]


def rows_as_text(f: Path) -> pd.DataFrame:
    """The file as strings, so the comparison is on what was written, not on a re-parse."""
    return pd.read_csv(f, dtype=str, keep_default_na=False)


def is_nusc(v) -> bool:
    return str(v).startswith("nuScenes")


def check_identity(before: Path, checks: list):
    """C4: outside nuScenes every regenerated official file must be byte-identical to the shipped one."""
    for name, col in IDENTITY:
        old_f, new_f = before / name, FINAL / name
        if not old_f.exists() or not new_f.exists():
            checks.append({"file": name, "status": "missing", "ok": False})
            continue
        a, b = rows_as_text(old_f), rows_as_text(new_f)
        rec = {"file": name, "rows_old": len(a), "rows_new": len(b),
               "columns_equal": list(a.columns) == list(b.columns)}
        if len(a) != len(b) or not rec["columns_equal"]:
            rec.update(status="shape differs", ok=False)
            checks.append(rec)
            continue
        nusc = a[col].map(is_nusc).to_numpy() if col else np.ones(len(a), bool)
        differs = (a.to_numpy() != b.to_numpy()).any(1)
        rec.update(rows_nuScenes=int(nusc.sum()), rows_other=int((~nusc).sum()),
                   nuScenes_rows_changed=int((differs & nusc).sum()),
                   other_rows_changed=int((differs & ~nusc).sum()),
                   status="ok", ok=bool((differs & ~nusc).sum() == 0))
        checks.append(rec)
        print(f"  C4 {name:38s} nuScenes changed {rec['nuScenes_rows_changed']:5d}/{rec['rows_nuScenes']:<5d}"
              f"  other changed {rec['other_rows_changed']}/{rec['rows_other']}"
              f"  {'ok' if rec['ok'] else 'FAILED'}", flush=True)


def cal(df, system, scheme, geometry="oracle"):
    m = df[(df.dataset == "nuScenes") & (df.geometry == geometry) & (df.scheme == scheme)
           & (df.split == "all") & (df.system == system)]
    assert len(m) == 1, (system, scheme, geometry, len(m))
    return m.iloc[0]


def add(rows, group, figure, cell, statistic, old, new, source):
    num = isinstance(old, (int, float, np.floating, np.integer)) and isinstance(new, (int, float, np.floating, np.integer))
    rows.append({"group": group, "figure": figure, "cell": cell, "statistic": statistic,
                 "old": old, "new": new, "delta": (float(new) - float(old)) if num else "",
                 "source": source})


def figures(before: Path, rows: list):
    co, cn = pd.read_csv(before / "calibration_cells.csv"), pd.read_csv(FINAL / "calibration_cells.csv")
    src = "calibration_cells.csv"

    # ---- 1: the E5 row of the gain-vs-decision table, oracle geometry, S0
    for system, label in (("brake", "q_brake"), ("plan", "q_plan")):
        o, n = cal(co, system, "S0"), cal(cn, system, "S0")
        for stat, col in (("sign disagreement", "E5_combined_sign_disagreement"),
                          ("Spearman rho with V", "E5_combined_spearman"),
                          ("P(V<0 | gain>0) among affected", "E5_combined_harmed_given_gain_pos_affected"),
                          ("P(V<0 | gain>0) all frames", "E5_combined_harmed_given_gain_pos"),
                          ("frames with gain and V both non-zero", "E5_combined_n_both_nonzero")):
            add(rows, "1 gain-vs-decision (E5_combined)", "E5 row, oracle geometry, S0",
                f"nuScenes oracle {label}", stat, float(o[col]), float(n[col]), src)

    # ---- 2: the sign-disagreement range across the four gains
    for system, label in (("brake", "q_brake"), ("plan", "q_plan")):
        for scope, schemes in (("S0", ("S0",)), ("per-mode S1-S4", PER_MODE)):
            for f, agg in (("min", np.min), ("max", np.max)):
                o = agg([float(cal(co, system, s)[f"{g}_sign_disagreement"]) for s in schemes for g in GAINS])
                n = agg([float(cal(cn, system, s)[f"{g}_sign_disagreement"]) for s in schemes for g in GAINS])
                add(rows, "2 sign-disagreement range over the four gains", f"{scope}, oracle geometry",
                    f"nuScenes oracle {label}", f"{f} over gains", float(o), float(n), src)

    # ---- 3: the scatter counts, and the overall range at S0
    for system, label in (("brake", "q_brake"), ("plan", "q_plan")):
        o, n = cal(co, system, "S0"), cal(cn, system, "S0")
        add(rows, "3 scatter and overall share", "scatter, E5_combined, oracle S0", f"nuScenes oracle {label}",
            "frames plotted (gain and V both non-zero)", float(o.E5_combined_n_both_nonzero),
            float(n.E5_combined_n_both_nonzero), src)
        add(rows, "3 scatter and overall share", "scatter, E5_combined, oracle S0", f"nuScenes oracle {label}",
            "share with opposite signs", float(o.E5_combined_sign_disagreement),
            float(n.E5_combined_sign_disagreement), src)
    for f, agg in (("min", np.min), ("max", np.max)):
        o = agg([float(cal(co, s, "S0")[f"{g}_sign_disagreement"]) for s in ("brake", "plan") for g in GAINS])
        n = agg([float(cal(cn, s, "S0")[f"{g}_sign_disagreement"]) for s in ("brake", "plan") for g in GAINS])
        add(rows, "3 scatter and overall share", "both consumers, four gains, oracle S0", "nuScenes oracle",
            f"{f} sign disagreement", float(o), float(n), src)

    # ---- 4: the target swap
    so = json.loads((before / "benchmark_target_swap_summary.json").read_text())
    sn = json.loads((FINAL / "benchmark_target_swap_summary.json").read_text())
    ssrc = "benchmark_target_swap_summary.json"
    for lab in LABELS:
        for a in ARCHS:
            po, pn = so["pooled"][lab]["per_architecture"][a], sn["pooled"][lab]["per_architecture"][a]
            for stat, k in (("pooled mean V - G", "mean_diff_V_minus_G"), ("CI low", "ci_lo"), ("CI high", "ci_hi")):
                add(rows, "4 target swap (C19)", f"pooled test, G = {lab}", a, stat, float(po[k]), float(pn[k]), ssrc)
        for stat, fn in (("architectures whose CI includes zero (of 6)",
                          lambda s: sum(s["pooled"][lab]["per_architecture"][a]["ci_lo"] < 0 <
                                        s["pooled"][lab]["per_architecture"][a]["ci_hi"] for a in ARCHS)),
                         ("architectures leaning to V (of 6)",
                          lambda s: sum(s["pooled"][lab]["per_architecture"][a]["mean_diff_V_minus_G"] > 0 for a in ARCHS)),
                         ("architectures leaning to G (of 6)",
                          lambda s: sum(s["pooled"][lab]["per_architecture"][a]["mean_diff_V_minus_G"] < 0 for a in ARCHS)),
                         ("cells where V wins significantly (of 72)",
                          lambda s: sum(s["pooled"][lab]["per_architecture"][a]["cells_diff_ci_above_0"] for a in ARCHS)),
                         ("cells where G wins significantly (of 72)",
                          lambda s: sum(s["pooled"][lab]["per_architecture"][a]["cells_diff_ci_below_0"] for a in ARCHS))):
            add(rows, "4 target swap (C19)", f"pooled test, G = {lab}", "all six architectures", stat,
                int(fn(so)), int(fn(sn)), ssrc)
        add(rows, "4 target swap (C19)", f"pooled test, G = {lab}", "all six architectures",
            "architectures leaning to G, named",
            ",".join(a for a in ARCHS if so["pooled"][lab]["per_architecture"][a]["mean_diff_V_minus_G"] < 0) or "none",
            ",".join(a for a in ARCHS if sn["pooled"][lab]["per_architecture"][a]["mean_diff_V_minus_G"] < 0) or "none",
            ssrc)
    add(rows, "4 target swap (C19)", "registered reading", "primary G", "reading",
        so["reading_primary"], sn["reading_primary"], ssrc)

    to, tn = pd.read_csv(before / "benchmark_target_swap.csv"), pd.read_csv(FINAL / "benchmark_target_swap.csv")

    def core10(t, arch):
        m = t[(t.g_variant == "primary") & (t.setting == "quota") & np.isclose(t.quota, 0.2) & (t.track != "nuPlan")]
        m = m[m.arch == arch]
        assert len(m) == 10, (arch, len(m))
        return float(m["diff"].mean())
    for a in ("gate_ridge", "gate_gbm"):
        add(rows, "4 target swap (C19)", "the 10 core cells, 20% quota, primary G", a,
            "mean V - G over the core cells", core10(to, a), core10(tn, a), "benchmark_target_swap.csv")

    # ---- 5: the multi-metric diagnostic on the four nuScenes rows
    mo, mn = pd.read_csv(before / "core_matrix.csv"), pd.read_csv(FINAL / "core_matrix.csv")
    for geom in ("mono", "oracle"):
        for task in ("longitudinal", "lateral"):
            def pick(t):
                m = t[(t.dataset == "nuScenes") & (t.geometry == geom) & (t.task == task)]
                assert len(m) == 1
                return m.iloc[0]
            o, n = pick(mo), pick(mn)
            for stat, col in (("multi-metric perception oracle, nDG@20%", "eta_multimetric_perception_oracle_20"),
                              ("best single perception metric, nDG@20%", "eta_best_perception_metric_20"),
                              ("best single perception metric", "best_perception_metric"),
                              ("exact perception oracle, nDG@20%", "eta_perception_oracle_20"),
                              ("Spearman(dE, dJ)", "corr_deltaE_deltaJ")):
                v_o, v_n = o[col], n[col]
                add(rows, "5 robustness table (nuScenes rows)", "core matrix", f"nuScenes {geom} {task}",
                    stat, float(v_o) if col != "best_perception_metric" else str(v_o),
                    float(v_n) if col != "best_perception_metric" else str(v_n), "core_matrix.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, help="a directory of the official files as they were before the fix")
    ap.add_argument("--fix_run", default=None, help="the 136 run directory, whose checks.csv is carried into the report")
    args = ap.parse_args()
    run = runmeta.new_run("class_error_figures", vars(args))
    before = Path(args.before)

    checks: list = []
    check_identity(before, checks)
    c4 = pd.DataFrame(checks)
    c4.to_csv(run / "c4_identity.csv", index=False)

    rows: list = []
    figures(before, rows)
    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "class_error_fix.csv", index=False)
    df.to_csv(run / "class_error_fix.csv", index=False)
    print(f"  wrote {FINAL / 'class_error_fix.csv'} ({len(df)} figures)")

    if args.fix_run:
        ch = pd.read_csv(Path(args.fix_run) / "checks.csv")
        print(f"  C1-C3 from {Path(args.fix_run).name}: {int(ch.ok.sum())}/{len(ch)} passed")
    ok = bool(c4.ok.all())
    print(f"  C4: {int(c4.ok.sum())}/{len(c4)} files unchanged outside nuScenes")
    if not ok:
        raise SystemExit("C4 failed: see c4_identity.csv")


if __name__ == "__main__":
    main()
