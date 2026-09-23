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
submission shares. By default each track uses the plan of the table the paper reports it from: `benchmark_table` for
KITTI and nuScenes, `nuplan_real` (benchmark_table_nuplan_real.csv) for nuPlan; `routers_r1` and
`router_r2:<dataset>` are the router tables' own. Measured budgets use `benchmark_budget` and `nuplan_real_budget`.

nuPlan is the real-perception track (Task 5 labels, scripts/120_nuplan_real_allocation.py); nDG is undefined where the
test split has fewer than MIN_AFFECTED affected inputs or the oracle's saving at the quota is not above 1e-9. Detector
costs, and the energy convention an allocator's cost must be measured under, come from the versioned cost registry
(results/final/cost_registry.json, scripts/156_cost_registry.py); its version is written into every scored row.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import scoring
from .budget import flag_two_level
from .paths import REPO, RESULTS

FINAL = Path(RESULTS) / "final"
INPUTS = Path(REPO) / "data" / "submission_inputs"
VALUES = FINAL / "benchmark_decision_values.csv.gz"
PLANS = FINAL / "benchmark_bootstrap_plans.json"
REGISTRY = FINAL / "cost_registry.json"
CELL = ("track", "geometry", "system", "target")
IDS = {"KITTI": ("seq", "frame"), "nuScenes": ("seq", "frame"), "nuPlan": ("scenario", "iteration")}
PER_BUDGET = {0.10: "score_q10", 0.20: "score_q20", 0.30: "score_q30", 0.50: "score_q50"}
ALLOWED = set(CELL) | {"seq", "frame", "scenario", "iteration", "score"} | set(PER_BUDGET.values())
NBOOT = 1000
MAX_LISTED = 10
MIN_AFFECTED = 10     # 120's rule; every KITTI and nuScenes test cell has at least 46 affected inputs, so it binds on nuPlan only
DEFAULT_PLAN = {"KITTI": "benchmark_table", "nuScenes": "benchmark_table", "nuPlan": "nuplan_real"}
BUDGET_PLAN = {"KITTI": "benchmark_budget", "nuScenes": "benchmark_budget", "nuPlan": "nuplan_real_budget"}


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
                                   "scenario": str, "unit": str, "split": str, "labels": str, "source_run": str},
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
    raise SubmissionError(f"cell {target} is not scored in plan {plan!r} (the default plan of its track is "
                          f"{DEFAULT_PLAN[cell[0]]!r})")


@lru_cache(maxsize=1)
def registry() -> dict:
    """The cost registry, checked against its own content hash."""
    r = json.loads(REGISTRY.read_text())
    body = {k: v for k, v in r.items() if k != "content_sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True, default=float).encode()).hexdigest() != r["content_sha256"]:
        raise RuntimeError(f"{REGISTRY} does not match its content_sha256: edited by hand?")
    return r


def cell_labels(cell) -> dict:
    """Where a cell's decision values come from: the label set and the run it was read from."""
    d = decision_values()
    m = np.logical_and.reduce([d[k].astype(str) == str(v) for k, v in zip(CELL, cell)])
    lab = d.loc[m, ["labels", "source_run"]].drop_duplicates()
    assert len(lab) == 1, (cell, lab)
    return lab.iloc[0].to_dict()


def undefined(prize: float, n_affected: int) -> str:
    """Why nDG is undefined at a quota ("" if it is defined): 120's rule."""
    if n_affected < MIN_AFFECTED:
        return f"near-zero oracle: {n_affected} affected states on this split (< {MIN_AFFECTED})"
    if prize <= scoring.EPS:
        return "oracle prize is zero at this quota"
    return ""


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


def score_cell(cell, g, form: str, plan: str | None = None, nboot: int = NBOOT) -> list[dict]:
    """Selection-budget rows: nDG at the benchmark's quotas, paired with random inside each bootstrap draw."""
    plan = plan or DEFAULT_PLAN[cell[0]]
    cols = ["score"] if form == "ranking" else list(PER_BUDGET.values())
    t, sc = _aligned(cell, g, cols)
    v, units = t.V.to_numpy(float), t.unit.to_numpy()
    n_aff = int((np.abs(v) > scoring.EPS).sum())
    scores = {"random": None}
    for c, s in sc.items():
        scores[c] = None if _all_tied(s) else s               # an all-tied ranking is the random allocator, exactly
    ks, prize0, point, draws, dropped = scoring.evaluate(v, units, scores, nboot, plan_rng(plan, cell, nboot))
    ident = {**cell_labels(cell), "registry_version": registry()["version"]}
    rows_ = []
    for qi, q in enumerate(scoring.QUOTAS):
        c = cols[0] if form == "ranking" else PER_BUDGET[q]
        p = point[c]
        why = undefined(prize0[qi], n_aff)
        ok = not why
        dr = draws[c][:, qi] - draws["random"][:, qi]
        lo, hi = scoring.ci(draws[c][:, qi])
        dlo, dhi = scoring.ci(dr)
        na = lambda x: x if ok else np.nan                   # noqa: E731  what an undefined nDG cannot have
        rows_.append({**dict(zip(CELL, cell)), **ident, "track_kind": "selection budget", "form": form, "plan": plan,
                      "score_column": c, "all_tied": scores[c] is None, "quota": q, "k": int(ks[qi]),
                      "ndg_defined": ok, "undefined_reason": why,
                      "ndg": na(float(p["eta"][qi])), "ndg_lo": na(lo), "ndg_hi": na(hi),
                      "minus_random": na(float(np.nanmean(dr))), "minus_random_lo": na(dlo), "minus_random_hi": na(dhi),
                      "interval_excludes_zero": bool(ok and np.isfinite(dlo) and (dlo > 0 or dhi < 0)),
                      # an all-tied submission is the random allocator itself: no comparison with random exists
                      "p_le_random": np.nan if (scores[c] is None or not ok) else float(np.mean(dr[np.isfinite(dr)] <= 0)),
                      "gain": float(p["gain"][qi]), "prize": float(prize0[qi]),
                      "reduction_frac": float(p["gain"][qi] / t.J_cheap.to_numpy(float).sum()),
                      "tie_frac": float(p["tie"][qi]), "responsive_frac": float(p["resp"][qi]),
                      "boot_dropped": int(dropped[qi]), "n_affected": n_aff,
                      "n_units": int(len(np.unique(units))), "n_frames": int(len(v))})
    return rows_


def detector_costs(track: str) -> dict:
    """C_c and C_f per input (ms, mJ) from the cost registry, with the energy convention they are measured under."""
    reg = registry()
    det = reg["detectors"][track]
    return {"cheap": {u: det["cheap"][u] for u in ("ms", "mJ")}, "640": {u: det["640"][u] for u in ("ms", "mJ")},
            "convention": reg["energy_convention"]["name"], "version": reg["version"],
            "timing_boundary": det["cheap"]["timing_boundary"]}


def measured_share(budget: float, cheap: float, full: float, overhead: float) -> float:
    """The share of inputs an allocator can escalate under a measured budget: 93's `max((b - C_c - o) / C_f, 0)`."""
    return max((budget - cheap - overhead) / full, 0.0)


def measured_k(share: float, n: int) -> int:
    """The number of inputs that share escalates among n: 93's `floor(share n + 1e-9)`."""
    return int(np.floor(share * n + 1e-9))


def _profile_value(profile, unit, track):
    v = profile.get(unit)
    return v.get(track) if isinstance(v, dict) else v


def budget_cell(cell, g, form: str, profile: dict, nboot: int = NBOOT) -> list[dict]:
    """Measured-budget rows: 93's two-level cascade with the allocator's own cost charged on every input, the
    detector costs of the cost registry, then the feasibility rule of 131 (an allocator whose overhead exceeds the
    headroom b - C_c cannot run)."""
    cols = ["score"] if form == "ranking" else list(PER_BUDGET.values())
    t, sc = _aligned(cell, g, cols)
    v, units = t.V.to_numpy(float), t.unit.to_numpy()
    n_aff = int((np.abs(v) > scoring.EPS).sum())
    plan = BUDGET_PLAN[cell[0]]
    rng = plan_rng(plan, cell, nboot)
    uniq = np.unique(units)
    idx = [np.flatnonzero(units == u) for u in uniq]
    draws = [np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(nboot)]
    cost = detector_costs(cell[0])
    ident = {**cell_labels(cell), "registry_version": cost["version"], "cost_convention": cost["convention"],
             "timing_boundary": cost["timing_boundary"]}
    out = []

    def gain_at(s, frac, tt=None):
        vv = v if tt is None else v[tt]
        k = measured_k(frac, len(vv))
        if k <= 0:
            return 0.0
        if s is None:
            return k / len(vv) * vv.sum()
        return float(scoring.topk_expect(s if tt is None else s[tt], [vv], [k])[0][0][0])
    for unit in ("ms", "mJ"):
        o = _profile_value(profile, unit, cell[0])
        if o is None:
            continue
        c0, c1 = cost["cheap"][unit], cost["640"][unit]
        for f in scoring.QUOTAS:
            budget = c0 + f * c1
            s = sc[cols[0] if form == "ranking" else PER_BUDGET[f]]
            s = None if _all_tied(s) else s
            prize = gain_at(v, f)
            prize_t = np.array([gain_at(v, f, tt) for tt in draws])
            okb = prize_t > max(scoring.EPS, 0.25 * prize)
            rand_t = np.array([gain_at(None, f, tt) for tt in draws]) / np.where(okb, prize_t, 1)
            why = undefined(prize, n_aff)
            ok = not why
            frac = measured_share(budget, c0, c1, o)
            g_ = gain_at(s, frac)
            e_t = np.array([gain_at(s, frac, tt) for tt in draws]) / np.where(okb, prize_t, 1)
            e_t, dr = np.where(okb, e_t, np.nan), np.where(okb, e_t - rand_t, np.nan)
            lo, hi = scoring.ci(e_t)
            dlo, dhi = scoring.ci(dr)
            na = lambda x: x if ok else np.nan               # noqa: E731
            out.append({**dict(zip(CELL, cell)), **ident, "track_kind": "measured budget", "form": form, "plan": plan,
                        "unit": unit, "budget_level": f, "budget_per_frame": budget, "cheap_cost": c0, "full_cost": c1,
                        "overhead": float(o),
                        "overhead_provenance": (profile.get("provenance") or {}).get(cell[0], "as stated by the profile"),
                        "cost_route": profile.get("route", "unstated"),
                        "reference_platform": (profile.get("device_check") or {}).get("reference_platform"),
                        "feasible": True, "escalated_frac": frac, "escalations_lost_to_overhead": f - frac,
                        "k": measured_k(frac, len(v)),
                        "ndg_defined": ok, "undefined_reason": why, "gain": g_, "prize": prize,
                        "ndg": na(g_ / prize if prize > scoring.EPS else np.nan), "ndg_lo": na(lo), "ndg_hi": na(hi),
                        "minus_random": na(float(np.nanmean(dr))), "minus_random_lo": na(dlo),
                        "minus_random_hi": na(dhi),
                        "interval_excludes_zero": bool(ok and np.isfinite(dlo) and (dlo > 0 or dhi < 0)),
                        "boot_dropped": int((~okb).sum()), "n_affected": n_aff, "n_frames": int(len(v)),
                        "n_units": int(len(uniq))})
    if not out:
        return out
    # 131's rule, as the official tables apply it: blank what an infeasible row cannot have achieved
    df = flag_two_level(pd.DataFrame(out).rename(columns={"ndg": "eta", "ndg_lo": "eta_lo", "ndg_hi": "eta_hi"}))
    df.loc[~df.feasible.astype(bool), "interval_excludes_zero"] = False
    df.loc[~df.feasible.astype(bool), "k"] = np.nan
    df = df.rename(columns={"eta": "ndg", "eta_lo": "ndg_lo", "eta_hi": "ndg_hi"})
    return df.to_dict("records")


def load_profile(path) -> dict:
    """A cost profile: {"ms": <per input> or {track: ...}, "mJ": ..., "rails": [...], "route": "harness" |
    "profiled by us", "device_check": {..., "reference_platform": bool}} -- what scripts/151_profile_allocator.py
    writes. A track or unit given as null is not scored under measured budgets."""
    return check_profile(json.loads(Path(path).read_text()))


def check_profile(p: dict) -> dict:
    """A profile's energy must be measured under the registry's convention (the rails its detector costs use)."""
    for unit in ("ms", "mJ"):
        v = p.get(unit)
        vals = [x for x in (v.values() if isinstance(v, dict) else [v]) if x is not None]
        if any(not np.isfinite(float(x)) or float(x) < 0 for x in vals):
            raise SubmissionError(f"cost profile: `{unit}` must be finite and non-negative")
    if p.get("ms") is None and p.get("mJ") is None:
        raise SubmissionError("cost profile: give at least one of `ms` and `mJ` per input")
    mj = p.get("mJ")
    if mj is not None and not (isinstance(mj, dict) and all(x is None for x in mj.values())):
        conv = registry()["energy_convention"]
        rails = p.get("rails")
        if rails is None or sorted(rails) != sorted(conv["rails"]):
            raise SubmissionError(
                f"cost profile: `mJ` is measured on rails {rails}, but the detector costs it is compared with "
                f"(cost registry {registry()['version']}) are on the {conv['name']} convention, rails {conv['rails']}: "
                f"{conv['definition']}. Profile on those rails (scripts/151_profile_allocator.py does by default), or "
                "give `mJ` as null to score latency only")
    return p


def score(sub: pd.DataFrame, plan: str | None = None, nboot: int = NBOOT, profile: dict | None = None,
          cells_only=None, selection: bool = True) -> tuple[str, pd.DataFrame]:
    """Score a submission. `plan` None uses each track's default plan (DEFAULT_PLAN); `selection` False scores the
    measured budgets only."""
    form, per_cell = validate(sub)
    if profile is not None:
        check_profile(profile)
    out = []
    for cell, g in per_cell.items():
        if cells_only and cell_id(cell) not in cells_only:
            continue
        if selection:
            out += score_cell(cell, g, form, plan, nboot)
        if profile is not None:
            out += budget_cell(cell, g, form, profile, nboot)
    return form, pd.DataFrame(out)
