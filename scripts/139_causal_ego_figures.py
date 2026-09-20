#!/usr/bin/env python
"""Task 22 Part B: the OLD -> NEW figure table, the sanity gate and C4.

Pre-registered in the pre-registration record (not part of this release) (Task 22 Part B). `138_causal_ego_speed.py` tabulates the causal nuScenes ego
speed; the stages that score ego speed as an allocation signal are then re-run. This script compares the
regenerated official files with the copies taken before the change and writes:

  results/final/causal_ego_speed.csv   every figure the task asks for, old value beside new
  <run>/sanity_and_c4.csv              the gate (the centred column through the new plumbing reproduces the
                                       shipped files) and C4 (nothing outside nuScenes moved)
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

FINAL = Path(RESULTS) / "final"
# (file, the column naming the unit; None: every row is nuScenes)
FILES = [("statistics_hardening.csv", "track"), ("objective_swap.csv", "track"), ("core_matrix.csv", "dataset")]
CELLS = [("oracle", "brake"), ("mono", "brake"), ("oracle", "plan_ade"), ("mono", "plan_ade"),
         ("oracle", "plan_fde"), ("mono", "plan_fde")]


def text(f: Path) -> pd.DataFrame:
    return pd.read_csv(f, dtype=str, keep_default_na=False)


def is_nusc(v) -> bool:
    return str(v).startswith("nuScenes")


def compare_files(a_dir: Path, b_dir: Path, label: str, rows: list, nusc_must_be_equal: bool):
    """Row-level comparison of two directories of official files, split into nuScenes and everything else."""
    for name, col in FILES:
        a, b = a_dir / name, b_dir / name
        if not a.exists() or not b.exists():
            rows.append({"check": label, "file": name, "status": "missing", "ok": False})
            continue
        x, y = text(a), text(b)
        if len(x) != len(y) or list(x.columns) != list(y.columns):
            rows.append({"check": label, "file": name, "status": "shape differs", "ok": False})
            continue
        nusc = x[col].map(is_nusc).to_numpy() if col else np.ones(len(x), bool)
        differs = (x.to_numpy() != y.to_numpy()).any(1)
        ok = bool((differs & ~nusc).sum() == 0 and (not nusc_must_be_equal or (differs & nusc).sum() == 0))
        rows.append({"check": label, "file": name, "rows": len(x), "rows_nuScenes": int(nusc.sum()),
                     "nuScenes_rows_changed": int((differs & nusc).sum()),
                     "other_rows_changed": int((differs & ~nusc).sum()), "status": "ok", "ok": ok})
        print(f"  {label:16s} {name:28s} nuScenes changed {int((differs & nusc).sum()):4d}/{int(nusc.sum()):<5d}"
              f" other changed {int((differs & ~nusc).sum())}/{int((~nusc).sum())}  {'ok' if ok else 'FAILED'}",
              flush=True)


def add(rows, group, figure, cell, statistic, old, new):
    num = isinstance(old, (int, float, np.integer, np.floating)) and not isinstance(old, bool)
    rows.append({"group": group, "figure": figure, "cell": cell, "statistic": statistic, "old": old, "new": new,
                 "delta": (float(new) - float(old)) if num and np.isfinite(float(old)) else ""})


def hardening(before: Path, rows: list, new_dir: Path = FINAL, group=None, label="C15 statistics hardening"):
    def load(f):
        d = pd.read_csv(f)
        m = d[(d.section == "paired_gain") & (d.signal == "trivial_ego_speed")].copy()
        m["ndg_random"] = m.gain_random / m.oracle_prize
        return m
    o, n = load(before / "statistics_hardening.csv"), load(new_dir / "statistics_hardening.csv")
    g1 = group or "1 held-out nDG@20%, ego speed"
    g2 = group or "2 realised gain vs random, every draw kept"

    def pick(t, geom, system, quota=0.2):
        m = t[(t.track == "nuScenes") & (t.geometry == geom) & (t.system == system) & np.isclose(t.quota, quota)]
        assert len(m) == 1, (geom, system, len(m))
        return m.iloc[0]
    for geom, system in CELLS:
        a, b = pick(o, geom, system), pick(n, geom, system)
        add(rows, g1, label, f"nuScenes {geom} {system}",
            "nDG@20% of ego speed", float(a.ndg), float(b.ndg))
        add(rows, g1, label, f"nuScenes {geom} {system}",
            "nDG@20% of random", float(a.ndg_random), float(b.ndg_random))
        add(rows, g1, label, f"nuScenes {geom} {system}",
            "ego speed above random", bool(a.ndg > a.ndg_random), bool(b.ndg > b.ndg_random))

    def wins(t):
        w = t[t.beats_random_all_draws.astype(str).str.lower() == "true"]
        return sorted(f"{r.track} {r.geometry} {r.system} @{int(round(r.quota * 100))}%" for r in w.itertuples())
    wo, wn = wins(o), wins(n)
    add(rows, g2, label, "all tracks",
        "rows where ego speed beats random", len(wo), len(wn))
    add(rows, g2, label, "all tracks",
        "which rows", "; ".join(wo), "; ".join(wn))
    for q in (0.1, 0.2, 0.3, 0.5):
        a = sum(1 for r in o[np.isclose(o.quota, q)].itertuples()
                if str(r.beats_random_all_draws).lower() == "true")
        b = sum(1 for r in n[np.isclose(n.quota, q)].itertuples()
                if str(r.beats_random_all_draws).lower() == "true")
        add(rows, g2, label, "all tracks",
            f"rows won at the {int(q * 100)}% quota", int(a), int(b))


def objective(before: Path, rows: list, new_dir: Path = FINAL, group="3 objective swap (C17) headline"):
    def head(f):
        d = pd.read_csv(f)
        c = d[(d.section == "compare") & (d.pool == "headline") & (d.e_dec_defined == True)]   # noqa: E712
        return {obj: g for obj, g in c.groupby("objective")}
    o, n = head(before / "objective_swap.csv"), head(new_dir / "objective_swap.csv")
    stats = [("comparisons", lambda g: int(len(g))),
             ("argmax differs", lambda g: int(g.argmax_differs.sum())),
             ("Kendall tau between the rankings, median", lambda g: float(g.kendall_tau.median())),
             ("selection regret, median nDG", lambda g: float(g.regret_ndg.median())),
             ("selection regret, median share of the all-cheap loss",
              lambda g: float(g.regret_share_cheap.median())),
             ("the E_perc winner is worse than random on decision value",
              lambda g: int(g.perc_winner_worse_than_random.sum())),
             ("regret intervals excluding zero",
              lambda g: int(((g.regret_gain_lo > 0) | (g.regret_gain_hi < 0)).sum()))]
    for obj in sorted(o):
        for name, fn in stats:
            add(rows, group, "13 defined cells x 4 budgets", obj, name,
                fn(o[obj]), fn(n[obj]))


def core(before: Path, rows: list, new_dir: Path = FINAL, group="4 trivial heuristics of the core matrix"):
    o, n = pd.read_csv(before / "core_matrix.csv"), pd.read_csv(new_dir / "core_matrix.csv")
    for geom in ("mono", "oracle"):
        for task in ("longitudinal", "lateral"):
            def pick(t):
                m = t[(t.dataset == "nuScenes") & (t.geometry == geom) & (t.task == task)]
                assert len(m) == 1
                return m.iloc[0]
            a, b = pick(o), pick(n)
            add(rows, group, "core matrix", f"nuScenes {geom} {task}",
                "best trivial heuristic", str(a.best_trivial_heuristic), str(b.best_trivial_heuristic))
            add(rows, group, "core matrix", f"nuScenes {geom} {task}",
                "its nDG@20%", float(a.best_trivial_eta20), float(b.best_trivial_eta20))


def per_sequence(run_dir: Path, rows: list):
    """The per-sequence ego-speed policy, from the two 53 runs (the official file is left alone; see the log)."""
    a = run_dir / "sanity_centred" / "statistical_tests.csv"
    b = run_dir / "causal_full_statistical_tests.csv"
    if not (a.exists() and b.exists()):
        return
    o, n = pd.read_csv(a), pd.read_csv(b)
    m = o.merge(n, on=["config", "task", "policy"], suffixes=("_o", "_n"))
    m = m[(m.policy == "ego speed") & (m.dataset_o == "nuScenes")]
    for _, r in m.iterrows():
        cell = f"{r['config'].split('__')[0]} {r['config'].split('__')[-1]} {r['task']}"
        add(rows, "5 per-sequence ego-speed policy", "53 statistical tests (run copies only)", cell,
            "median per-sequence nDG", float(r["median_policy_eta_o"]), float(r["median_policy_eta_n"]))
        add(rows, "5 per-sequence ego-speed policy", "53 statistical tests (run copies only)", cell,
            "sequences above 0.5", int(r["n_seq_policy_above_0.5_o"]), int(r["n_seq_policy_above_0.5_n"]))
        add(rows, "5 per-sequence ego-speed policy", "53 statistical tests (run copies only)", cell,
            "Wilcoxon p (oracle > policy)", float(r["wilcoxon_p_o"]), float(r["wilcoxon_p_n"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the 138 run directory, holding before/ and sanity_centred/")
    args = ap.parse_args()
    run = runmeta.new_run("causal_ego_figures", vars(args))
    rd = Path(args.run)

    checks: list = []
    compare_files(rd / "before", rd / "sanity_centred", "sanity centred", checks, nusc_must_be_equal=True)
    compare_files(rd / "before", FINAL, "C4 causal", checks, nusc_must_be_equal=False)
    c = pd.DataFrame(checks)
    c.to_csv(run / "sanity_and_c4.csv", index=False)

    rows: list = []
    hardening(rd / "before", rows)
    objective(rd / "before", rows)
    core(rd / "before", rows)
    per_sequence(rd, rows)
    drop = rd / "sensitivity_drop_first"
    if (drop / "statistics_hardening.csv").exists():
        # the pre-registered sensitivity: "old" here is the reported causal run, "new" the variant in which a
        # scene's first frame is put out of reach of any quota instead of being given a speed of 0
        g = "6 sensitivity: a scene's first frame dropped, not zeroed"
        hardening(FINAL, rows, drop, group=g, label="reported causal -> first frame dropped")
        objective(FINAL, rows, drop, group=g)
        core(FINAL, rows, drop, group=g)
    df = pd.DataFrame(rows)
    df.to_csv(FINAL / "causal_ego_speed.csv", index=False)
    df.to_csv(run / "causal_ego_speed.csv", index=False)
    print(f"  wrote {FINAL / 'causal_ego_speed.csv'} ({len(df)} figures)")
    print(f"  sanity + C4: {int(c.ok.sum())}/{len(c)} passed")
    if not bool(c.ok.all()):
        raise SystemExit("a check failed: see sanity_and_c4.csv")


if __name__ == "__main__":
    main()
