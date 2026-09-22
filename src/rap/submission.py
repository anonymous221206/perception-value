"""The submission path (Task 25): validate a third party's allocator scores and score them the way the benchmark
scores its own signals.

A submission is a CSV with one row per test input of each cell it enters:

  track, geometry, system, target      the cell, spelled as in results/final/benchmark_table.csv
                                       (nuPlan's geometry is the literal "n/a")
  seq, frame                           KITTI and nuScenes: sequence / scene name and frame index
  scenario, iteration                  nuPlan: scenario token and iteration
  score                                one ranking, scored at every quota                      (form "ranking")
  score_q10, score_q20, score_q30, score_q50
                                       one score column per budget level                        (form "per-budget")

Nothing else. Higher scores are escalated first. Scores are compared after rounding to 9 decimals, as the benchmark
does; equal scores are ties, and every selection is evaluated in exact expectation over random tie-breaks, so the
order of the rows changes nothing. A cell whose scores are all equal is the random allocator, and is scored by the
benchmark's closed form for it.

`score` reproduces a table's own paired unit bootstrap: the resamples a cell gets depend on the order in which the
official script visits cells (every table draws from one `default_rng(0)`), so the plan names the table whose draws a
submission shares (`benchmark_table` by default; `routers_r1`, `router_r2:<dataset>`, `benchmark_budget`).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import scoring
from .budget import infeasible
from .paths import REPO, RESULTS

FINAL = Path(RESULTS) / "final"
INPUTS = Path(REPO) / "data" / "submission_inputs"
VALUES = FINAL / "benchmark_decision_values.csv.gz"
PLANS = FINAL / "benchmark_bootstrap_plans.json"
CELL = ("track", "geometry", "system", "target")
IDS = {"KITTI": ("seq", "frame"), "nuScenes": ("seq", "frame"), "nuPlan": ("scenario", "iteration")}
PER_BUDGET = {0.10: "score_q10", 0.20: "score_q20", 0.30: "score_q30", 0.50: "score_q50"}
ALLOWED = set(CELL) | {"seq", "frame", "scenario", "iteration", "score"} | set(PER_BUDGET.values())
NBOOT = 1000
MAX_LISTED = 10


class SubmissionError(ValueError):
    """A submission that does not follow the schema; the message names the offending rows."""


def cell_id(key) -> str:
    return "|".join(str(k) for k in key)


# ------------------------------------------------------------------------------------------------ the benchmark side

@lru_cache(maxsize=1)
def decision_values() -> pd.DataFrame:
    """Every cell's inputs with unit, split and decision value, in the order the benchmark scores them.

    This file holds V for the test inputs too: the pre-escalation rule of docs/SUBMITTING.md is honour-based."""
    d = pd.read_csv(VALUES, dtype={"track": str, "geometry": str, "system": str, "target": str, "seq": str,
                                   "scenario": str, "unit": str, "split": str},
                    keep_default_na=False, na_values=[""], float_precision="round_trip")
    return d


def cells() -> list[tuple]:
    d = decision_values()
    return [tuple(k) for k in d[list(CELL)].drop_duplicates().itertuples(index=False)]


def rows(cell, split: str = "test") -> pd.DataFrame:
    """The identifiers (and unit) of a cell's inputs on a split: what a submission must cover. No labels."""
    d = decision_values()
    m = np.logical_and.reduce([d[k].astype(str) == str(v) for k, v in zip(CELL, cell)]) & (d.split == split)
    ids = list(IDS[cell[0]])
    return d.loc[m, list(CELL) + ids + ["unit"]].reset_index(drop=True)


def inputs(track: str) -> dict:
    """Everything an allocator may read at test time, for every unit of the frozen split, and nothing else.

    KITTI and nuScenes: `frames` (unit, split, `prev_frame` -- the previous CHEAP frame of the unit, -1 for the first
    -- ego speed, image reference, CHEAP image statistics) and `detections` (every CHEAP detection at confidence
    >= 0.10 with its uncertainty summaries and its ego-frame monocular geometry), plus `calibration` (intrinsics and
    camera -> ego transform per unit). nuPlan: `states` (the cheap-side gate features and the CHEAP track list per
    state). No reference object, FULL output or decision value is in these files; on training units a submission may
    read those too (docs/SUBMITTING.md), from the datasets or from `decision_values()`."""
    read = lambda n: pd.read_csv(INPUTS / n, keep_default_na=False, na_values=[""], float_precision="round_trip",
                                 dtype={"seq": str, "unit": str, "split": str, "scenario": str, "coarse": str})
    if track in ("KITTI", "nuScenes"):
        cal = json.loads((INPUTS / "calibration.json").read_text())[track]
        return {"frames": read(f"{track.lower()}_frames.csv.gz"),
                "detections": read(f"{track.lower()}_detections.csv.gz"), "calibration": cal}
    if track == "nuPlan":
        return {"states": read("nuplan_states.csv.gz")}
    raise KeyError(track)


@lru_cache(maxsize=1)
def plans() -> dict:
    return json.loads(PLANS.read_text())


def plan_rng(plan: str, cell, nboot: int = NBOOT) -> np.random.Generator:
    """`default_rng(0)` advanced past every draw the official script made before this cell's scored evaluation."""
    events = plans()[plan]
    rng = np.random.default_rng(0)
    target = cell_id(cell)
    for e in events:
        if e["cell"] == target and e.get("scored"):
            return rng
        for _ in range(nboot):
            rng.integers(0, e["n_units"], e["n_units"])
    raise KeyError(f"cell {target} is not scored in plan {plan!r}")


# ------------------------------------------------------------------------------------------------ validation

def load(path) -> pd.DataFrame:
    """Read a submission CSV without letting pandas reinterpret identifiers or round scores."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _listed(df, mask, what):
    bad = df[mask]
    lines = [f"  line {i + 2}: " + ", ".join(f"{c}={bad.at[i, c]!r}" for c in bad.columns if c in ALLOWED)
             for i in bad.index[:MAX_LISTED]]
    more = f"\n  ... and {len(bad) - MAX_LISTED} more" if len(bad) > MAX_LISTED else ""
    return f"{what} ({len(bad)} rows):\n" + "\n".join(lines) + more


def validate(sub: pd.DataFrame) -> tuple[str, dict]:
    """Check a submission against the schema and the frozen test split; return its form and its rows per cell.

    Raises SubmissionError naming every offending row (the first ten of each kind)."""
    errors = []
    unknown = [c for c in sub.columns if c not in ALLOWED]
    if unknown:
        raise SubmissionError(f"unknown columns {unknown}; allowed: {sorted(ALLOWED)}")
    missing_cell = [c for c in CELL if c not in sub.columns]
    if missing_cell:
        raise SubmissionError(f"missing cell columns {missing_cell}")
    has_score, per = "score" in sub.columns, [c for c in PER_BUDGET.values() if c in sub.columns]
    if has_score and per:
        raise SubmissionError("give either `score` or the four per-budget columns, not both")
    if not has_score and len(per) != len(PER_BUDGET):
        raise SubmissionError(f"missing score columns: need `score`, or all of {list(PER_BUDGET.values())} "
                              f"(found {per})")
    form = "ranking" if has_score else "per-budget"
    score_cols = ["score"] if has_score else list(PER_BUDGET.values())
    sub = sub.reset_index(drop=True)
    known = {cell_id(c): c for c in cells()}
    sub["_cell"] = sub[list(CELL)].astype(str).agg("|".join, axis=1)
    bad_cell = ~sub._cell.isin(known)
    if bad_cell.any():
        errors.append(_listed(sub, bad_cell, "rows naming no benchmark cell (see benchmark_table.csv for the spelling)"))
    for c in score_cols:
        v = pd.to_numeric(sub[c], errors="coerce")
        bad = v.isna() | ~np.isfinite(v.fillna(0).to_numpy(float))
        if bad.any():
            errors.append(_listed(sub, bad, f"non-numeric or non-finite `{c}`"))
    out = {}
    for cid, g in sub[~bad_cell].groupby("_cell", sort=False):
        cell = known[cid]
        ids = list(IDS[cell[0]])
        miss_cols = [c for c in ids if c not in g.columns]
        if miss_cols:
            errors.append(f"cell {cid}: missing identifier columns {miss_cols}")
            continue
        other = [c for c in ("seq", "frame", "scenario", "iteration") if c not in ids and c in g.columns
                 and (g[c] != "").any()]
        if other:
            errors.append(_listed(g, (g[other] != "").any(axis=1), f"cell {cid}: identifiers of another track {other}"))
        num = ids[1]
        iv = pd.to_numeric(g[num], errors="coerce")
        bad = iv.isna() | (iv != np.floor(iv.fillna(0)))
        if bad.any():
            errors.append(_listed(g, bad, f"cell {cid}: `{num}` is not an integer"))
            continue
        g = g.assign(**{num: iv.astype(int)})
        dup = g.duplicated(ids, keep=False)
        if dup.any():
            errors.append(_listed(g, dup, f"cell {cid}: duplicate inputs"))
            continue
        ref = rows(cell, "test")
        ref[num] = ref[num].astype(int)
        j = ref.merge(g, on=ids, how="outer", indicator=True, suffixes=("", "_sub"))
        miss = j[j._merge == "left_only"]
        if len(miss):
            errors.append(f"cell {cid}: {len(miss)} test inputs missing, e.g. " +
                          ", ".join(f"({r[ids[0]]}, {r[ids[1]]})" for _, r in miss.head(MAX_LISTED).iterrows()))
        test_keys = set(zip(ref[ids[0]].astype(str), ref[ids[1]].astype(int)))
        extra = pd.Series([(a, int(b)) not in test_keys for a, b in zip(g[ids[0]].astype(str), g[ids[1]])],
                          index=g.index)
        if extra.any():
            errors.append(_listed(g, extra, f"cell {cid}: rows that are not test inputs of this cell (training or "
                                            "validation inputs, or unknown frames)"))
        if not len(miss) and not extra.any():
            out[cell] = g
    if errors:
        raise SubmissionError("the submission does not follow the schema:\n" + "\n".join(errors))
    return form, out


# ------------------------------------------------------------------------------------------------ scoring

def _cell_arrays(cell):
    d = decision_values()
    m = np.logical_and.reduce([d[k].astype(str) == str(v) for k, v in zip(CELL, cell)]) & (d.split == "test")
    t = d[m].reset_index(drop=True)
    ids = list(IDS[cell[0]])
    t[ids[1]] = t[ids[1]].astype(int)
    return t, ids


def _aligned(cell, g, cols):
    """The submission's scores in the benchmark's own row order."""
    t, ids = _cell_arrays(cell)
    j = t[ids].merge(g[ids + cols], on=ids, how="left", validate="one_to_one")
    return t, {c: pd.to_numeric(j[c], errors="raise").to_numpy(float) for c in cols}


def _all_tied(s) -> bool:
    r = np.round(np.asarray(s, float), 9)
    return bool((r == r[0]).all())


def score_cell(cell, g, form: str, plan: str = "benchmark_table", nboot: int = NBOOT) -> list[dict]:
    """Selection-budget rows: nDG at the benchmark's quotas, paired with random inside each bootstrap draw."""
    cols = ["score"] if form == "ranking" else list(PER_BUDGET.values())
    t, sc = _aligned(cell, g, cols)
    v, units = t.V.to_numpy(float), t.unit.to_numpy()
    scores = {"random": None}
    for c, s in sc.items():
        scores[c] = None if _all_tied(s) else s               # an all-tied ranking is the random allocator, exactly
    ks, prize0, point, draws, dropped = scoring.evaluate(v, units, scores, nboot, plan_rng(plan, cell, nboot))
    rows_ = []
    for qi, q in enumerate(scoring.QUOTAS):
        c = cols[0] if form == "ranking" else PER_BUDGET[q]
        p = point[c]
        dr = draws[c][:, qi] - draws["random"][:, qi]
        lo, hi = scoring.ci(draws[c][:, qi])
        dlo, dhi = scoring.ci(dr)
        rows_.append({**dict(zip(CELL, cell)), "track_kind": "selection budget", "form": form, "plan": plan,
                      "score_column": c, "all_tied": scores[c] is None, "quota": q, "k": int(ks[qi]),
                      "ndg": float(p["eta"][qi]), "ndg_lo": lo, "ndg_hi": hi,
                      "minus_random": float(np.nanmean(dr)), "minus_random_lo": dlo, "minus_random_hi": dhi,
                      "interval_excludes_zero": bool(np.isfinite(dlo) and (dlo > 0 or dhi < 0)),
                      # an all-tied submission is the random allocator itself: no comparison with random exists
                      "p_le_random": np.nan if scores[c] is None else float(np.mean(dr[np.isfinite(dr)] <= 0)),
                      "gain": float(p["gain"][qi]), "prize": float(prize0[qi]),
                      "reduction_frac": float(p["gain"][qi] / t.J_cheap.to_numpy(float).sum()),
                      "tie_frac": float(p["tie"][qi]), "responsive_frac": float(p["resp"][qi]),
                      "boot_dropped": int(dropped[qi]), "ndg_defined": bool(prize0[qi] > scoring.EPS),
                      "n_units": int(len(np.unique(units))), "n_frames": int(len(v))})
    return rows_


def detector_costs(track: str) -> dict:
    """C_c and C_f per unit, as `benchmark_budget_overheads.json` records them from the shipped profiles."""
    costs = json.loads((FINAL / "benchmark_budget_overheads.json").read_text())["costs"][track]
    return {"cheap": costs["cheap"], "640": costs["640"]}


def budget_cell(cell, g, form: str, profile: dict, nboot: int = NBOOT) -> list[dict]:
    """Measured-budget rows: 93's two-level cascade with the allocator's own cost charged on every input, then the
    feasibility rule of 131 (an allocator whose overhead exceeds the headroom b - C_c cannot run)."""
    cols = ["score"] if form == "ranking" else list(PER_BUDGET.values())
    t, sc = _aligned(cell, g, cols)
    v, units = t.V.to_numpy(float), t.unit.to_numpy()
    rng = plan_rng("benchmark_budget", cell, nboot)
    uniq = np.unique(units)
    idx = [np.flatnonzero(units == u) for u in uniq]
    draws = [np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(nboot)]
    cost = detector_costs(cell[0])
    out = []

    def gain_at(s, frac, tt=None):
        vv = v if tt is None else v[tt]
        k = int(np.floor(frac * len(vv) + 1e-9))
        if k <= 0:
            return 0.0
        if s is None:
            return k / len(vv) * vv.sum()
        return float(scoring.topk_expect(s if tt is None else s[tt], [vv], [k])[0][0][0])
    for unit in ("ms", "mJ"):
        o = profile.get(unit, {}).get(cell[0]) if isinstance(profile.get(unit), dict) else profile.get(unit)
        if o is None:
            continue
        c0, c1 = cost["cheap"][unit], cost["640"][unit]
        for f in scoring.QUOTAS:
            budget = c0 + f * c1
            s = sc[cols[0] if form == "ranking" else PER_BUDGET[f]]
            s = None if _all_tied(s) else s
            prize = gain_at(v, f)
            prize_t = np.array([gain_at(v, f, tt) for tt in draws])
            ok = prize_t > max(scoring.EPS, 0.25 * prize)
            rand_t = np.array([gain_at(None, f, tt) for tt in draws]) / np.where(ok, prize_t, 1)
            frac = max((budget - c0 - o) / c1, 0.0)
            row = {**dict(zip(CELL, cell)), "track_kind": "measured budget", "form": form, "plan": "benchmark_budget",
                   "unit": unit, "budget_level": f, "budget_per_frame": budget, "cheap_cost": c0, "full_cost": c1,
                   "overhead": float(o), "cost_route": profile.get("route", "unstated"),
                   "reference_platform": (profile.get("device_check") or {}).get("reference_platform"),
                   "n_frames": int(len(v)), "n_units": int(len(uniq)), "boot_dropped": int((~ok).sum())}
            if bool(infeasible(budget, c0, o)):
                row.update(feasible=False, note=(f"infeasible: allocator overhead {o:.4f} {unit} per input exceeds "
                                                 f"the budget headroom b - C_c = {budget - c0:.4f} {unit}, so it "
                                                 "cannot run within this budget"))
                out.append(row)
                continue
            g_ = gain_at(s, frac)
            e_t = np.array([gain_at(s, frac, tt) for tt in draws]) / np.where(ok, prize_t, 1)
            e_t, dr = np.where(ok, e_t, np.nan), np.where(ok, e_t - rand_t, np.nan)
            lo, hi = scoring.ci(e_t)
            dlo, dhi = scoring.ci(dr)
            row.update(feasible=True, escalated_frac=frac, escalations_lost_to_overhead=f - frac,
                       ndg=g_ / prize if prize > scoring.EPS else np.nan, ndg_lo=lo, ndg_hi=hi,
                       minus_random=float(np.nanmean(dr)), minus_random_lo=dlo, minus_random_hi=dhi,
                       interval_excludes_zero=bool(np.isfinite(dlo) and (dlo > 0 or dhi < 0)))
            out.append(row)
    return out


def load_profile(path) -> dict:
    """A cost profile: {"ms": <per input> or {track: ...}, "mJ": ..., "route": "harness" | "profiled by us",
    "device_check": {..., "reference_platform": bool}} -- what scripts/151_profile_allocator.py writes. A track or
    unit given as null is not scored under measured budgets."""
    p = json.loads(Path(path).read_text())
    for unit in ("ms", "mJ"):
        v = p.get(unit)
        vals = [x for x in (v.values() if isinstance(v, dict) else [v]) if x is not None]
        if any(not np.isfinite(float(x)) or float(x) < 0 for x in vals):
            raise SubmissionError(f"cost profile: `{unit}` must be finite and non-negative")
    if p.get("ms") is None and p.get("mJ") is None:
        raise SubmissionError("cost profile: give at least one of `ms` and `mJ` per input")
    return p


def score(sub: pd.DataFrame, plan: str = "benchmark_table", nboot: int = NBOOT, profile: dict | None = None,
          cells_only=None) -> tuple[str, pd.DataFrame]:
    form, per_cell = validate(sub)
    out = []
    for cell, g in per_cell.items():
        if cells_only and cell_id(cell) not in cells_only:
            continue
        out += score_cell(cell, g, form, plan, nboot)
        if profile is not None:
            out += budget_cell(cell, g, form, profile, nboot)
    return form, pd.DataFrame(out)
