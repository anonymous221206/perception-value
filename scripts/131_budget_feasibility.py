#!/usr/bin/env python
"""Task 19 Part A: report infeasible measured budgets instead of a 0% escalation.

Pre-registered in the pre-registration record (not part of this release), Task 19 Part A, committed before this script was written.

Every measured-budget table charged an allocator the share max((b - C_c - C_S) / C_f, 0).  When its overhead C_S
exceeds the headroom b - C_c it cannot run within the budget at all, yet the row read as a 0% escalation with nDG 0.
The rule, `rap.budget.infeasible`: feasible = C_S <= b - C_c; an infeasible row keeps its budget, costs and overhead,
and what it would have achieved (nDG, gain, shares, intervals) is NaN.

The tables 93 writes need hardware to regenerate, so this stage derives them from 93's own run outputs, which
`93_budget_allocation.py` now flags itself when it writes (so the transformation is idempotent):

  results/raw/*_benchmark_budget/          -> benchmark_budget_two_level.csv, benchmark_budget_multifidelity.csv
  results/raw/*_benchmark_budget_1thread/  -> benchmark_budget_two_level_1thread.csv, ..._multifidelity_1thread.csv
  results/raw/*_benchmark_budget_routers/  -> benchmark_budget_routers.csv
  benchmark_budget_routers.csv             -> fig_budget_curves.csv (118 part C)

The stages that write the other budget tables (C12 120, C14 124, C18 128, C19 122) apply the rule themselves.

  --flag_shipped         one-off: apply 120's, 122's and 124's rule to their shipped outputs without re-running them
                         (124 refits and drifts; the file must change only in its infeasible rows)
  --validate_against D   compare every output with the copy in D (the files before this change): rows that stay
                         feasible must be equal in every pre-existing column, exactly; writes the diff to the run
"""
from __future__ import annotations

import argparse, importlib.util, io, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import budget, runmeta                                                  # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
KEYS = ["track", "geometry", "system", "target", "g_variant", "arch", "setting", "section", "variant", "signal", "policy",
        "unit", "budget_level", "quota", "rate_target", "row_type"]


def _latest(tag):
    """The newest run of a tag (exact tag, so *_benchmark_budget does not pick up *_benchmark_budget_1thread)."""
    d = [p for p in sorted(RAW.glob(f"*_{tag}")) if p.name.split("_", 2)[-1] == tag]
    if not d:
        raise SystemExit(f"no results/raw/*_{tag} run found")
    return d[-1]


def read(path):
    """Exact floats, and nuPlan's geometry label "n/a" kept as text rather than parsed as a missing value."""
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""])


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def derive():
    """93's five tables from its run outputs, and the figure export from the routers table."""
    out = {}
    for tag, sfx in (("benchmark_budget", ""), ("benchmark_budget_1thread", "_1thread")):
        run = _latest(tag)
        costs = json.loads((run / f"benchmark_budget_overheads{sfx}.json").read_text())["costs"]
        kc = costs["KITTI"]["cheap"]
        out[f"benchmark_budget_two_level{sfx}.csv"] = budget.flag_two_level(read(run / f"benchmark_budget_two_level{sfx}.csv"))
        out[f"benchmark_budget_multifidelity{sfx}.csv"] = budget.flag_multifidelity(
            read(run / f"benchmark_budget_multifidelity{sfx}.csv"), {"ms": kc["ms"], "mJ": kc["mJ"]})
        print(f"  {run.name}: two-level and multi-fidelity tables", flush=True)
    run = _latest("benchmark_budget_routers")
    out["benchmark_budget_routers.csv"] = budget.flag_two_level(read(run / "benchmark_budget_routers.csv"))
    print(f"  {run.name}: routers table", flush=True)
    # 118 reads the routers table back from CSV with pandas' default parser (which can land one ULP off, and turns
    # "n/a" into a missing value); the figure data are derived through exactly that path, so they match 118's output
    out["fig_budget_curves.csv"] = budget.budget_curves(pd.read_csv(io.StringIO(out["benchmark_budget_routers.csv"].to_csv(index=False))))
    return out


def flag_shipped():
    """One-off for the outputs of stages that are not re-run here: C12, C14 and C19."""
    t122 = _load("t122", "122_target_swap.py")
    t124 = _load("t124", "124_causal_threshold.py")
    ov = json.loads((FINAL / "benchmark_budget_overheads_routers.json").read_text())["overheads"]
    return {"benchmark_budget_nuplan_real.csv": budget.flag_two_level(read(FINAL / "benchmark_budget_nuplan_real.csv")),
            "benchmark_target_swap.csv": t122.flag_budget(read(FINAL / "benchmark_target_swap.csv"), ov),
            "causal_threshold.csv": t124.flag_budget(read(FINAL / "causal_threshold.csv"))}


def _same(a, b):
    """Exact equality of two parsed values, NaN equal to NaN; text compared as text."""
    fa = pd.api.types.is_number(a) and not isinstance(a, bool)
    fb = pd.api.types.is_number(b) and not isinstance(b, bool)
    if fa and fb:
        return (np.isnan(a) and np.isnan(b)) or a == b
    if pd.isna(a) and pd.isna(b):
        return True
    return str(a) == str(b)


def validate(name, new, old):
    """Rows whose flag is true must be unchanged in every pre-existing column; return the rows that changed."""
    assert len(new) == len(old), (name, len(new), len(old))
    flag = new["feasible"].astype(str).str.lower() if "feasible" in new.columns else pd.Series("", index=new.index)
    changed = []
    for i in range(len(new)):
        diff = {c: (old.at[i, c], new.at[i, c]) for c in old.columns if not _same(old.at[i, c], new.at[i, c])}
        if diff:
            changed.append({"file": name, "row": i, "feasible": flag.iat[i],
                            **{k: new.at[i, k] for k in KEYS if k in new.columns},
                            "changed": json.dumps({c: [str(x), str(y)] for c, (x, y) in diff.items()})})
    ch = pd.DataFrame(changed)
    if not len(ch):
        return ch, ch
    # figure medians are aggregates over cells: they change wherever a cell turned infeasible, and are listed separately
    median = ch["row_type"].eq("median_over_cells") if "row_type" in ch.columns else pd.Series(False, index=ch.index)
    return ch, ch[(ch.feasible != "false") & ~median]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag_shipped", action="store_true")
    ap.add_argument("--validate_against", default=None)
    args = ap.parse_args()
    run = runmeta.new_run("budget_feasibility", vars(args))
    outs = flag_shipped() if args.flag_shipped else derive()
    summary = {}
    for name, df in outs.items():
        if args.validate_against:
            ch, bad = validate(name, df, read(Path(args.validate_against) / name))
            ch.to_csv(run / f"changed_rows__{name}", index=False)
            summary[name] = {"rows": int(len(df)), "rows_changed": int(len(ch)),
                             "changed_rows_not_flagged_infeasible": int(len(bad))}
            print(f"  {name}: {len(ch)} rows changed, {len(bad)} of them not flagged infeasible", flush=True)
            if len(bad):
                (run / "validation_summary.json").write_text(json.dumps(summary, indent=1))
                raise SystemExit(f"validation failed for {name}: a row that stays feasible changed; nothing written")
        feas = df["feasible"].astype(str).str.lower() if "feasible" in df.columns else pd.Series(dtype=str)
        print(f"  {name}: {int((feas == 'false').sum())} rows infeasible of {len(df)}", flush=True)
    if args.validate_against:
        (run / "validation_summary.json").write_text(json.dumps(summary, indent=1))
    for name, df in outs.items():
        df.to_csv(FINAL / name, index=False)
        df.to_csv(run / name, index=False)
    print(f"  wrote {len(outs)} files to {FINAL}", flush=True)


if __name__ == "__main__":
    main()
