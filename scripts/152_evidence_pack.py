#!/usr/bin/env python
"""Task 26: the evidence pack -- every quantity a write-up may quote, traceable to a file, a cell, a split and a
budget -- and a pass/fail check of the statements the write-up currently makes.

Fits nothing and re-runs no experiment: it reads results/final (and, for the leave-one-test-unit-out ranges of 3A and
the total-decision-value intervals of 3B, the saved per-frame scores and outcomes those tables were built from).

  results/final/paper_evidence_pack.csv   one row per quantity (columns: PACK_COLUMNS)
  results/final/claims_check.csv          claim_id, statement, verdict (PASS / FAIL / UNVERIFIABLE), value_now,
                                          artifact, note
  docs/iclr_evidence_pack.md              the summary: counts, every FAIL with its value, every UNVERIFIABLE with
                                          its reason, tables 3A and 3B, the artifact disagreements, sections 4-5

Every claim is also evaluated on the camera-frame artifacts as first shipped
(results/raw/20260921_152927_ego_frame_before/results_final), and the note says whether it held there: that is how a
statement that was true when written but moved with Task 23 is told apart from one that never held.
"""
from __future__ import annotations

import argparse, json, sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta, scoring                                                 # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

FINAL = Path(RESULTS) / "final"
BEFORE = ROOT / "results" / "raw" / "20260921_152927_ego_frame_before" / "results_final"
DOC = ROOT / "docs" / "iclr_evidence_pack.md"
QUOTAS = (0.1, 0.2, 0.3, 0.5)
CORE = ("KITTI", "nuScenes")
PACK_COLUMNS = ["quantity_id", "description", "artifact_file", "track", "geometry", "system", "target", "split",
                "units_scope", "quota_or_budget", "signal_or_policy", "estimate", "ci_lo", "ci_hi", "interval_method",
                "n_units", "n_affected", "denominator", "notes"]
LEARNED = ("gate_ridge", "gate_gbm", "gate_gbm_batched", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf",
           "R2_cnn_clf")
ROUTERS = ("R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "R2_cnn_clf")
DIAGNOSTIC_EXCLUDE = ("oracle",)

# interval methods, named once
IM_92 = ("paired unit bootstrap, 1000 draws, default_rng(0) in 92's cell order; draws whose oracle prize is below 25% "
         "of the full-sample prize dropped (boot_dropped); ties in exact expectation; 95% percentile")
IM_92_DIFF = IM_92.replace("95% percentile", "95% percentile of nDG(signal) - nDG(random) within each draw")
IM_103 = IM_92.replace("92's cell order", "103's cell order (routers_r1)")
IM_107 = IM_92.replace("92's cell order", "107's order, one stream per dataset (router_r2)")
IM_93 = IM_92.replace("92's cell order", "93's cell order (measured budgets)")
IM_CAL = ("unit bootstrap of per-unit counts, 1000 draws, default_rng(0) in 102's cell order, no filter; 95% "
          "percentile (needs 50 finite draws)")
IM_NR = "unit bootstrap over logs (116), 95% percentile; finite draws reported in the artifact"
IM_ALLDRAWS = ("paired unit bootstrap of realised gain minus random, 1000 draws, every draw kept (125); ties in exact "
               "expectation; 95% percentile")
IM_NONE = "none (a count or a point value)"

TARGET_LABEL = {
    ("brake", "J"): "q_brake: braking controller loss",
    ("traj", "JB"): "q_traj: rollout trajectory controller loss (Planner B)",
    ("plan_ade", "JC_ade"): "q_plan ADE: learned planner (Planner C) against the real trajectory, average displacement",
    ("plan_fde", "JC_fde"): "q_plan FDE: learned planner (Planner C) against the real trajectory, final displacement",
    ("pdm_closed", "safety"): "PDM-Closed safety loss", ("pdm_closed", "scalar_J"): "PDM-Closed scalar loss",
    ("idm", "safety"): "IDM safety loss", ("idm", "scalar_J"): "IDM scalar loss"}
SELF_REF = "self-referenced control: Planner C path deviation against its own ground-truth-conditioned output (Phase 0F)"
NUPLAN_DIAG = "HISTORICAL transported detection-profile diagnostic (not the official nuPlan result)"
NUPLAN_REAL = "official real-perception nuPlan result"


# ------------------------------------------------------------------------------------------------ artifacts

class Art:
    """One results/final directory, read losslessly (nuPlan's geometry is the literal "n/a")."""

    def __init__(self, root: Path, label: str):
        self.root, self.label = Path(root), label

    @lru_cache(maxsize=None)
    def csv(self, name: str) -> pd.DataFrame:
        f = self.root / name
        if not f.exists():
            raise FileNotFoundError(f"{self.label}: {name} not present")
        return pd.read_csv(f, keep_default_na=False, na_values=[""], float_precision="round_trip", low_memory=False)

    @lru_cache(maxsize=None)
    def json(self, name: str):
        f = self.root / name
        if not f.exists():
            raise FileNotFoundError(f"{self.label}: {name} not present")
        return json.loads(f.read_text())


NEW = Art(FINAL, "now")
OLD = Art(BEFORE, "before Task 23 (camera frame, as first shipped)")


def pct(x, nd=1):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}%"


def f_(x, nd=3, sign=False):
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci_(lo, hi, nd=2, sign=False):
    return f"[{f_(lo, nd, sign)}, {f_(hi, nd, sign)}]"


def r(x, nd):
    """Round as the statements are written (half away from zero on the printed value)."""
    return float(f"{x:.{nd}f}")


def range_ok(lo, hi, a, b, scale=100.0, nd=0, nd_hi=None):
    """A stated range "a–b" matches [lo, hi] if each end is the value rounded to the precision it is stated at, either
    to the nearest or outward (floor for the lower end, ceiling for the upper): both conventions occur in the
    statements."""
    import math
    nd_hi = nd if nd_hi is None else nd_hi
    los = {math.floor(lo * scale * 10 ** nd) / 10 ** nd, r(lo * scale, nd)}
    his = {math.ceil(hi * scale * 10 ** nd_hi) / 10 ** nd_hi, r(hi * scale, nd_hi)}
    return any(abs(x - a) < 1e-9 for x in los) and any(abs(x - b) < 1e-9 for x in his)


def cell_key(row):
    get = row.get if isinstance(row, dict) else (lambda k: getattr(row, k))
    return f"{get('track')} {get('geometry')} {get('system')} {get('target')}"


def is_core(df):
    return df.track.isin(CORE)


# ------------------------------------------------------------------------------------------------ claims

CLAIMS = []


def claim(cid, statement):
    def deco(fn):
        CLAIMS.append((cid, statement, fn))
        return fn
    return deco


def res(ok, value, artifact, note=""):
    return {"ok": ok, "value": value, "artifact": artifact, "note": note}


def cal_S(A, scheme="S0", split="all"):
    c = A.csv("calibration_cells.csv")
    return c[(c.scheme == scheme) & (c.split == split)]


@claim("H1", "On the core track, escalation harms 35–52 % of affected inputs in every row, over all units, at the "
             "shared threshold")
def h1(A):
    c = cal_S(A)
    h = c.harm_rate.to_numpy(float)
    mono = c[c.geometry == "mono"].harm_rate.to_numpy(float)
    ok = bool(((np.round(100 * mono) >= 35) & (np.round(100 * mono) <= 52)).all())
    orc = c[c.geometry == "oracle"]
    val = (f"mono rows (10): {pct(mono.min())}–{pct(mono.max())}, all within the stated bounds: {ok}; oracle rows: " +
           ", ".join(f"{x.cell.replace(' Y8 320->640', '')} {pct(x.harm_rate)}" for x in orc.itertuples()) +
           f"; all 14 rows {pct(h.min())}–{pct(h.max())}")
    return res(ok, val, "calibration_cells.csv (scheme S0, split all)",
               "Ambiguous: 'every row' holds on the pre-Task-23 tables only for the monocular (deployed) rows -- "
               "with the oracle rows it failed there too -- so it is read as the 10 mono fidelity-pair rows and "
               "checked as a bound (every row inside 35–52 %). The oracle rows are reported beside it. core_matrix.csv's "
               "harmful_full_rate is a share of ALL frames and is not this quantity.")


@claim("H2", "Under oracle geometry the braking controller harms 34–51 % of affected inputs on every KITTI fidelity "
             "pair and 48.5 % on nuScenes")
def h2(A):
    g = A.csv("reference_geometry_sweep.csv")
    k = g[g.section.isna() & (g.system == "q_brake") & g.pair.notna() & ~g.pair.astype(str).str.startswith("summary")]
    kh = k.reference_harmed.to_numpy(float)
    ns = cal_S(A)
    ns = ns[(ns.dataset == "nuScenes") & (ns.geometry == "oracle") & (ns.system == "brake")].harm_rate.iloc[0]
    ok = bool(((np.round(100 * kh) >= 34) & (np.round(100 * kh) <= 51)).all() and r(100 * ns, 1) == 48.5)
    val = ("KITTI oracle: " + ", ".join(f"{p} {pct(v)}" for p, v in zip(k.pair, kh)) + f"; nuScenes oracle {pct(ns)}")
    return res(ok, val, "reference_geometry_sweep.csv (reference = oracle geometry, S0, all units); "
                        "calibration_cells.csv (nuScenes oracle q_brake, S0, all)")


@claim("H3", "Under oracle geometry the trajectory controller's harm ratio is below 0.06 on four of the five fidelity "
             "pairs")
def h3(A):
    g = A.csv("reference_geometry_sweep.csv")
    k = g[g.section.isna() & (g.system == "q_traj") & g.pair.notna() & ~g.pair.astype(str).str.startswith("summary")]
    rho = k.reference_rho.to_numpy(float)
    n = int((rho < 0.06).sum())
    return res(n == 4 and len(rho) == 5, f"{n} of {len(rho)} below 0.06: " +
               ", ".join(f"{p} {v:.4f}" for p, v in zip(k.pair, rho)), "reference_geometry_sweep.csv (q_traj, reference)")


@claim("H4", "Outside the KITTI 320→640 pairs, the core-track harm ratio ranges 0.21–0.82, reaching 0.88 for the "
             "learned planner under final displacement; in the KITTI 320→640 cells it is 0.004–0.19")
def h4(A):
    c = cal_S(A)
    in320 = c.cell.str.contains("KITTI") & c.cell.str.contains("Y8 320->640")
    out, inn = c[~in320].rho.to_numpy(float), c[in320].rho.to_numpy(float)
    b = A.csv("benchmark_cells.csv")
    fde = b[(b.system == "plan_fde") & (b.split == "all")]
    fmax = float(fde.destroyed_benefit_D.max())
    ok = (range_ok(out.min(), out.max(), 0.21, 0.82, 1, 2) and r(fmax, 2) == 0.88
          and range_ok(inn.min(), inn.max(), 0.004, 0.19, 1, 3, 2))
    val = (f"outside 320→640: {out.min():.3f}–{out.max():.3f} (max: {c[~in320].loc[c[~in320].rho.idxmax(), 'cell']}); "
           f"learned planner FDE (all units): " + ", ".join(f"{x.geometry} {x.destroyed_benefit_D:.3f}" for x in fde.itertuples())
           + f"; KITTI 320→640 cells: {inn.min():.4f}–{inn.max():.3f}")
    return res(ok, val, "calibration_cells.csv (rho, S0, all); benchmark_cells.csv (destroyed_benefit_D of plan_fde, all)",
               "harm ratio = sum of harm / sum of benefit over all units (rho = D). The calibration cells carry the "
               "planner under ADE only; FDE is read from benchmark_cells.csv.")


@claim("H5", "On nuScenes braking, helpful escalations reduce the all-cheap loss by 25.0 % and harmful ones raise it "
             "by 15.1 %, giving a harm ratio of 0.60 and a net 10.0 % for uniform escalation")
def h5(A):
    b = A.csv("benchmark_cells.csv")
    x = b[(b.track == "nuScenes") & (b.geometry == "oracle") & (b.system == "brake") & (b.split == "all")].iloc[0]
    jc = (x.gross_benefit - x.gross_harm) / x.all_full_reduction          # the all-cheap loss the shares are of
    gb, gh = x.gross_benefit / jc, x.gross_harm / jc
    ok = (r(100 * gb, 1) == 25.0 and r(100 * gh, 1) == 15.1 and r(x.destroyed_benefit_D, 2) == 0.60
          and r(100 * x.all_full_reduction, 1) == 10.0)
    return res(ok, f"benefit {pct(gb)}, harm {pct(gh)}, ratio {x.destroyed_benefit_D:.3f}, net {pct(x.all_full_reduction)} "
                   f"(nuScenes oracle braking, all units)", "benchmark_cells.csv (nuScenes oracle brake J, split all)",
               "The statement names neither geometry nor population; oracle geometry over all units is the row whose "
               "camera-frame values it quoted. All-cheap loss recovered as (benefit - harm) / all_full_reduction.")


@claim("H6", "Mean harm over cells is 39.4 % of affected inputs and 8.9 % of all inputs on the test splits")
def h6(A):
    s = A.csv("statistics_hardening.csv")
    h = s[(s.section == "harm") & (s.split == "test")]
    a, b = float(h.harmed_over_affected.mean()), float(h.harmed_over_all_inputs.mean())
    ok = r(100 * a, 1) == 39.4 and r(100 * b, 1) == 8.9
    return res(ok, f"{pct(a)} of affected, {pct(b)} of all inputs (mean over {len(h)} cells, test split)",
               "statistics_hardening.csv (section harm, split test)",
               "nuPlan cells here are the official real-perception ones.")


def _sign_range(A, schemes, system):
    c = A.csv("calibration_cells.csv")
    c = c[(c.dataset == "nuScenes") & (c.geometry == "oracle") & (c.split == "all") & c.scheme.isin(schemes)
          & (c.system == system)]
    cols = [f"{g}_sign_disagreement" for g in ("exact_FN", "FN_FP", "E5_combined", "E_risk")]
    v = c[cols].to_numpy(float).ravel()
    return float(np.nanmin(v)), float(np.nanmax(v))


@claim("S1", "On nuScenes oracle geometry at the shared threshold, the four perception-level gains disagree in sign "
             "with decision value on 34–43 % of affected braking frames and 49–52 % of planner frames")
def s1(A):
    b, p = _sign_range(A, ["S0"], "brake"), _sign_range(A, ["S0"], "plan")
    ok = range_ok(*b, 34, 43) and range_ok(*p, 49, 52)
    return res(ok, f"braking {pct(b[0])}–{pct(b[1])}, planner {pct(p[0])}–{pct(p[1])}",
               "calibration_cells.csv (*_sign_disagreement, nuScenes oracle, S0, all)",
               "Planner = q_plan ADE. Sign disagreement is over frames where both the gain and V are non-zero.")


@claim("S2", "Under the per-mode operating points those ranges are 29–45 % and 49–53 %")
def s2(A):
    sch = ["S1", "S2", "S3", "S4"]
    b, p = _sign_range(A, sch, "brake"), _sign_range(A, sch, "plan")
    ok = range_ok(*b, 29, 45) and range_ok(*p, 49, 53)
    return res(ok, f"braking {pct(b[0])}–{pct(b[1])}, planner {pct(p[0])}–{pct(p[1])} (S1–S4)",
               "calibration_cells.csv (nuScenes oracle, schemes S1–S4, all)",
               "'Per-mode operating points' read as schemes S1–S4 together.")


@claim("S3", "Harmed frames gain about twice as many false positives as helped frames while recovering fewer misses, "
             "and on 45.5 % of harmed frames escalation recovers no miss")
def s3(A):
    m = A.csv("mechanism_table.csv")
    m = m[m.decision == "brake"].set_index("group")
    ratio = m.loc["harmed", "mean_delta_fp"] / m.loc["helped", "mean_delta_fp"]
    fewer = -m.loc["harmed", "mean_delta_fn"] < -m.loc["helped", "mean_delta_fn"]
    share = m.loc["harmed", "share_delta_fn_nonneg"]
    ok = bool(1.75 <= ratio <= 2.25 and fewer and r(100 * share, 1) == 45.5)
    return res(ok, f"FP gain ratio harmed/helped {ratio:.2f} ({m.loc['harmed', 'mean_delta_fp']:.2f} vs "
                   f"{m.loc['helped', 'mean_delta_fp']:.2f}); misses recovered {-m.loc['harmed', 'mean_delta_fn']:.2f} vs "
                   f"{-m.loc['helped', 'mean_delta_fn']:.2f}; no miss recovered on {pct(share)} of harmed frames "
                   f"({int(m.loc['harmed', 'n_frames'])} harmed, {int(m.loc['helped', 'n_frames'])} helped)",
               "mechanism_table.csv (decision brake: nuScenes oracle, all units)",
               "'About twice' checked as a ratio in [1.75, 2.25].")


@claim("S4", "Of the 153 frames to which both downstream systems respond, 80 are helped under one and harmed under "
             "the other")
def s4(A):
    b = A.csv("calibration_brake_vs_plan.csv")
    x = b[(b.geometry == "oracle") & (b.scheme == "S0") & (b.split == "all")].iloc[0]
    ok = int(x.both_nonzero) == 153 and int(x.opposite_sign) == 80
    mono = b[(b.geometry == "mono") & (b.scheme == "S0") & (b.split == "all")].iloc[0]
    return res(ok, f"oracle: {int(x.opposite_sign)} of {int(x.both_nonzero)} (mono: {int(mono.opposite_sign)} of "
                   f"{int(mono.both_nonzero)})", "calibration_brake_vs_plan.csv (S0, all units)",
               "Braking controller against the learned planner (ADE) on nuScenes planner frames.")


def _bt(A, **kw):
    t = A.csv("benchmark_table.csv")
    m = np.ones(len(t), bool)
    for k, v in kw.items():
        m &= (t[k] == v).to_numpy() if not isinstance(v, (list, tuple)) else t[k].isin(v).to_numpy()
    return t[m]


@claim("D1", "No diagnostic signal approaches the oracle; the best are the risk-weighted gain at 0.562 [0.28, 0.75] on "
             "braking and PKL at 0.395 [0.14, 0.57] on the planner")
def d1(A):
    out, ok = [], True
    for system, want, val in (("brake", "dE_E6_risk_weighted", (0.562, 0.28, 0.75)), ("plan_ade", "PKL", (0.395, 0.14, 0.57))):
        t = _bt(A, track="nuScenes", geometry="oracle", system=system, split="all", quota=0.2)
        t = t[(~t.deployable.astype(str).eq("True")) & ~t.signal.isin(DIAGNOSTIC_EXCLUDE)]
        best = t.loc[t.eta.idxmax()]
        out.append(f"{system}: best {best.signal} {best.eta:.3f} {ci_(best.eta_lo, best.eta_hi)}")
        ok &= best.signal == want and (r(best.eta, 3), r(best.eta_lo, 2), r(best.eta_hi, 2)) == val
    return res(bool(ok), "; ".join(out) + " (nuScenes oracle, 20%, split all)",
               "benchmark_table.csv (nuScenes oracle, split all, quota 0.2, non-deployable signals)",
               "Split 'all' is the population whose camera-frame point values the statement quoted; its intervals match "
               "no shipped table even before Task 23 (they differ in the second decimal), so they come from an "
               "unshipped or earlier artifact.")


@claim("D2", "Under monocular geometry every diagnostic collapses toward random on braking: risk-weighted 0.165, PKL "
             "0.087, TIP 0.068, against 0.103 for random")
def d2(A):
    t = _bt(A, track="nuScenes", geometry="mono", system="brake", split="all", quota=0.2).set_index("signal")
    want = {"dE_E6_risk_weighted": 0.165, "PKL": 0.087, "TIP": 0.068, "random": 0.103}
    ok = all(r(t.loc[s, "eta"], 3) == v for s, v in want.items())
    sep = [s for s in ("dE_E6_risk_weighted", "PKL", "TIP", "dE_exact") if t.loc[s, "minus_random_lo"] > 0]
    return res(ok, ", ".join(f"{s} {t.loc[s, 'eta']:.3f}" for s in want) +
               (f"; paired interval against random excludes zero for {', '.join(sep)}" if sep else ""),
               "benchmark_table.csv (nuScenes mono brake, split all, 20%)",
               "The statement's random value 0.103 matches neither this table's exact-expectation random before or "
               "after Task 23; the table's random is quoted.")


@claim("D3", "Exact perception gain never exceeds nDG 0.204 across the 16 configurations and is negative in seven; the "
             "multi-metric diagnostic exceeds 0.5 only in the two aggressive-gap KITTI longitudinal cells, and falls "
             "below the exact gain in three configurations")
def d3(A):
    c = A.csv("core_matrix.csv")
    e, mm = c.eta_perception_oracle_20.to_numpy(float), c.eta_multimetric_perception_oracle_20.to_numpy(float)
    above = c[mm > 0.5]
    lbl = lambda x: f"{x.dataset} {x.cheap_mode} {x.task} {x.geometry}"            # noqa: E731
    two = (len(above) == 2 and set(above.dataset) == {"KITTI"} and set(above.task) == {"longitudinal"}
           and set(above.cheap_mode) == {"cheap_320"})
    ok = r(e.max(), 3) <= 0.204 and int((e < 0).sum()) == 7 and two and int((mm < e).sum()) == 3
    return res(bool(ok), f"max exact {e.max():.3f} ({lbl(c.iloc[int(np.argmax(e))])}); negative in {int((e < 0).sum())}; "
                         f"multi-metric > 0.5 in {len(above)}: " + "; ".join(lbl(x) for x in above.itertuples()) +
               f"; multi-metric below exact in {int((mm < e).sum())}",
               "core_matrix.csv (eta_perception_oracle_20, eta_multimetric_perception_oracle_20; 16 configurations, all units)")


# ---- allocator wins

def wins(A, quota=None, official_nuplan=True):
    """Deployable rows on test whose paired interval against random lies above zero (minus_random_lo > 0)."""
    parts = []
    t = A.csv("benchmark_table.csv")
    t = t[(t.split == "test") & t.deployable.astype(str).eq("True") & (t.signal != "random") & t.quota.notna()]
    parts.append(t[is_core(t) if official_nuplan else np.ones(len(t), bool)].assign(source="benchmark_table.csv"))
    rt = A.csv("benchmark_table_routers.csv")
    rt = rt[rt.quota.notna()]
    parts.append(rt[is_core(rt) if official_nuplan else np.ones(len(rt), bool)].assign(source="benchmark_table_routers.csv"))
    if official_nuplan:
        n = A.csv("benchmark_table_nuplan_real.csv")
        n = n[(n.split == "test") & n.deployable.astype(str).eq("True") & (n.signal != "random") & n.quota.notna()]
        parts.append(n.assign(source="benchmark_table_nuplan_real.csv"))
    d = pd.concat(parts, ignore_index=True)
    if quota is not None:
        d = d[np.isclose(d.quota.astype(float), quota)]
    d = d.assign(win=d.minus_random_lo > 0, lose=d.minus_random_hi < 0, learned=d.signal.isin(LEARNED))
    return d


def wlist(d):
    lvl = lambda x: getattr(x, "quota", getattr(x, "budget_level", float("nan")))    # noqa: E731
    return "; ".join(f"{cell_key(x)} {x.signal}@{lvl(x):g} {x.minus_random:+.3f} {ci_(x.minus_random_lo, x.minus_random_hi, sign=True)}"
                     for x in d.itertuples())


@claim("A1", "At the 20 % quota, learned allocators are the only deployable signals with a paired interval above "
             "random, on KITTI's trajectory controller under oracle geometry and on the two nuPlan planners")
def a1(A):
    w = wins(A, 0.2)
    w = w[w.win]
    cells = set(zip(w.track, w.geometry, w.system))
    want = {("KITTI", "oracle", "traj"), ("nuPlan", "n/a", "pdm_closed"), ("nuPlan", "n/a", "idm")}
    ok = bool(w.learned.all() and want <= set(zip(w[w.learned].track, w[w.learned].geometry, w[w.learned].system)))
    extra = sorted(cells - want)
    return res(ok, f"all {len(w)} winning rows learned: {bool(w.learned.all())}; learned wins on the three named settings: "
                   f"{want <= cells}; wins also on: {', '.join(' '.join(c) for c in extra) or 'none'}. Rows: " + wlist(w),
               "benchmark_table.csv + benchmark_table_routers.csv (KITTI, nuScenes) + benchmark_table_nuplan_real.csv "
               "(nuPlan, official), split test, quota 0.2",
               "Read as: every deployable win at 20% is a learned allocator, and there are learned wins on the three "
               "named settings. The list is not exhaustive, before or after Task 23: learned wins also occur on other "
               "cells (named in the value). The routers table's and the benchmark table's own nuPlan rows are the "
               "historical transported detection-profile diagnostic and are not used here.")


@claim("A2", "Under deployed monocular geometry no signal in the headline set separates from random, and only two "
             "router variants do, both on KITTI braking")
def a2(A):
    w = wins(A, 0.2)
    w = w[w.win & (w.geometry == "mono")]
    head = w[w.source == "benchmark_table.csv"]
    rout = w[w.signal.isin(ROUTERS)]
    ok = len(head) == 0 and len(rout) == 2 and set(zip(rout.track, rout.system)) == {("KITTI", "brake")}
    return res(bool(ok), f"headline set: {len(head)} winning rows{': ' + wlist(head) if len(head) else ''}; routers: "
                         f"{len(rout)}: " + wlist(rout), "benchmark_table.csv (deployable signals) and "
               "benchmark_table_routers.csv, mono cells, test, 20%",
               "'Headline set' read as the deployable signals of benchmark_table.csv (uncertainty, cheap-side "
               "criticality, ridge and gradient-boosted gates). Quota read as 20%.")


def _sh(A):
    s = A.csv("statistics_hardening.csv")
    return s[s.section == "paired_gain"]


BASELINES = ("uncertainty", "criticality_cheap", "trivial_ego_speed", "trivial_n_det", "trivial_area_max",
             "trivial_risk_cheap")


@claim("A3", "On every nuScenes cell, ego speed is below random, and no deployable baseline separates from random")
def a3(A):
    s = _sh(A)
    out, ok = [], True
    for q in QUOTAS:
        x = s[(s.track == "nuScenes") & np.isclose(s.quota.astype(float), q)]
        ego = x[x.signal == "trivial_ego_speed"]
        below = int((ego.delta_vs_random < 0).sum())
        sep = x[x.signal.isin(BASELINES) & (x.beats_random_filtered.astype(str) == "True")]
        out.append(f"{q:.0%}: ego speed below random in {below} of {len(ego)} cells; baselines separating: "
                   f"{len(sep)}" + (f" ({'; '.join(f'{cell_key(z)} {z.signal}' for z in sep.itertuples())})" if len(sep) else ""))
        if np.isclose(q, 0.2):
            ok = below == len(ego) == 6 and len(sep) == 0
    return res(bool(ok), " | ".join(out), "statistics_hardening.csv (paired_gain, nuScenes, test)",
               "Checked at 20% (verdict) and reported at every quota. 'Separates' = the official filtered paired "
               "interval (beats_random_filtered); deployable baselines = uncertainty, cheap-side criticality and the "
               "four trivial baselines.")


@claim("A4", "Ego speed has the highest point estimate among deployable baselines on all four KITTI cells but "
             "separates from random on none")
def a4(A):
    s = _sh(A)
    x = s[(s.track == "KITTI") & np.isclose(s.quota.astype(float), 0.2) & s.signal.isin(BASELINES)]
    out, top, sep = [], 0, 0
    for key, g in x.groupby(["geometry", "system", "target"]):
        best = g.loc[g.ndg.idxmax()]
        e = g[g.signal == "trivial_ego_speed"].iloc[0]
        top += best.signal == "trivial_ego_speed"
        sep += str(e.beats_random_filtered) == "True"
        out.append(f"{' '.join(key)}: best {best.signal} {best.ndg:.3f}, ego speed {e.ndg:.3f} "
                   f"({'separates' if str(e.beats_random_filtered) == 'True' else 'does not separate'})")
    return res(top == 4 and sep == 0, "; ".join(out), "statistics_hardening.csv (paired_gain, KITTI, test, 20%)",
               "Quota read as 20%.")


@claim("A5", "Across the four quotas, 25 deployable rows beat random on test: 23 learned gates, plus cheap-side "
             "criticality on nuScenes mono braking at 30 % and on PDM-Closed safety at 50 %. Each detection-list router "
             "variant adds 9–16 wins; the pixel router adds none")
def a5(A):
    w = wins(A)
    w = w[w.win]
    base = w[~w.signal.isin(ROUTERS)]
    gates = base[base.signal.str.startswith("gate_")]
    other = base[~base.signal.str.startswith("gate_")]
    per = {s: int((w.signal == s).sum()) for s in ROUTERS}
    want_other = {("nuScenes", "mono", "brake", "criticality_cheap", 0.3), ("nuPlan", "n/a", "pdm_closed", "criticality_cheap", 0.5)}
    got_other = {(x.track, x.geometry, x.system, x.signal, round(float(x.quota), 2)) for x in other.itertuples()}
    r1 = [per[s] for s in ROUTERS if s.startswith("R1_")]
    ok = (len(base) == 25 and len(gates) == 23 and got_other == want_other and min(r1) == 9 and max(r1) == 16
          and per["R2_cnn_clf"] == 0)
    return res(bool(ok), f"{len(base)} non-router deployable wins: {len(gates)} gate rows, others: " + (wlist(other) or "none")
               + "; router wins: " + ", ".join(f"{s} {n}" for s, n in per.items()),
               "benchmark_table.csv + benchmark_table_routers.csv (KITTI, nuScenes) + benchmark_table_nuplan_real.csv, "
               "test, 10/20/30/50%",
               "Wins = minus_random_lo > 0 (the official filtered paired interval). nuPlan from the official "
               "real-perception table; the PDM-Closed criticality row of the statement is read against it.")


@claim("A6", "On realised gain with every draw kept, nine of the thirteen deployable wins at 20 % survive; the four "
             "that do not are the two PDM-Closed safety gates, the scalar-loss gradient-boosted gate and IDM's router. "
             "The oracle-prize filter adds 7 to 20 wins across the four quotas and removes none")
def a6(A):
    s = _sh(A)
    dep = s[s.signal.isin(LEARNED + BASELINES)]
    f = lambda x: x.astype(str) == "True"                                         # noqa: E731
    x = dep[np.isclose(dep.quota.astype(float), 0.2)]
    filt, alld = x[f(x.beats_random_filtered)], x[f(x.beats_random_filtered) & f(x.beats_random_all_draws)]
    lost = x[f(x.beats_random_filtered) & ~f(x.beats_random_all_draws)]
    adds, removes = [], []
    for q in QUOTAS:                                      # the filter's effect, over every signal of the table
        y = s[s.quota.notna() & np.isclose(s.quota.astype(float), q)]
        adds.append(int((f(y.beats_random_filtered) & ~f(y.beats_random_all_draws)).sum()))
        removes.append(int((~f(y.beats_random_filtered) & f(y.beats_random_all_draws)).sum()))
    ok = len(filt) == 13 and len(alld) == 9 and min(adds) == 7 and max(adds) == 20 and sum(removes) == 0
    return res(bool(ok), f"at 20%: {len(filt)} filtered wins, {len(alld)} survive with every draw kept; not surviving: "
                         + "; ".join(f"{cell_key(z)} {z.signal}" for z in lost.itertuples())
               + f"; filter adds {adds} and removes {removes} at 10/20/30/50%",
               "statistics_hardening.csv (paired_gain, test)",
               "Deployable = learned allocators and target-free baselines (uncertainty, cheap-side criticality, trivial "
               "baselines); the filter's added and removed wins are counted over every signal of the table (the reading "
               "that reproduces the statement's 7–20 on the pre-Task-23 table). nuPlan cells are the official "
               "real-perception ones.")


@claim("A7", "Removing the single dominant nuPlan log leaves the gates' ratio at 0.99 while 93 % of the realised gain "
             "disappears (162.9 → 12.0)")
def a7(A):
    s = A.csv("statistics_hardening.csv")
    x = s[(s.section == "influence") & (s.track == "nuPlan") & (s.system == "pdm_closed") & (s.target == "safety")
          & s.signal.isin(["gate_ridge", "gate_gbm"])]
    out, ok = [], True
    for z in x.itertuples():
        lost = 1 - z.gain_without_that_unit / z.gain_full
        out.append(f"{z.signal}: nDG {z.ndg_full:.3f} -> {z.ndg_without_that_unit:.3f} without {z.most_influential_unit}; "
                   f"gain {z.gain_full:.1f} -> {z.gain_without_that_unit:.1f} ({pct(lost, 0)} lost)")
        ok &= r(z.ndg_full, 2) == 0.99 and z.ndg_without_that_unit >= 0.99 and r(z.gain_full, 1) == 162.9 \
            and r(z.gain_without_that_unit, 1) == 12.0 and r(100 * lost, 0) == 93
    return res(bool(ok) and len(x) > 0, "; ".join(out), "statistics_hardening.csv (influence, nuPlan PDM-Closed safety, 20%)",
               "'Leaves the ratio at 0.99' read as: nDG 0.99 with every log and at least 0.99 without the dominant one.")


# ---- costs and budgets

def _ov(A, name):
    return A.json(name)


@claim("C1", "The feature-based gates spend 18–19 % of a full pass on feature extraction; at a 20 % latency budget the "
             "ridge gate is infeasible and the batched gradient-boosted gate escalates at most 1.7 % of inputs, and "
             "neither beats random")
def c1(A):
    shares = {}
    for name in ("benchmark_budget_overheads.json", "benchmark_budget_overheads_routers.json"):
        o = _ov(A, name)
        shares[name] = {t: o["overheads"][t]["features_ms"] / o["costs"][t]["640"]["ms"] for t in CORE}
    b = A.csv("benchmark_budget_routers.csv")
    b = b[is_core(b) & (b.unit == "ms") & np.isclose(b.budget_level, 0.2)]
    ridge = b[b.signal == "gate_ridge"]
    bat = b[b.signal == "gate_gbm_batched"]
    win = b[b.signal.isin(["gate_ridge", "gate_gbm_batched"]) & (b.minus_random_lo > 0)]
    s_r = shares["benchmark_budget_overheads_routers.json"]
    ok = (all(18 <= r(100 * v, 0) <= 19 for v in s_r.values()) and not ridge.feasible.astype(str).eq("True").any()
          and r(100 * bat.escalated_frac.max(), 1) <= 1.7 and len(win) == 0)
    disagree = any(abs(shares["benchmark_budget_overheads.json"][t] - s_r[t]) > 0.005 for t in CORE)
    return res(bool(ok), "feature share of a full pass: " + "; ".join(
        f"{n.replace('benchmark_budget_overheads', 'overheads')}: " + ", ".join(f"{t} {pct(v)}" for t, v in s.items())
        for n, s in shares.items()) + f"; ridge feasible in {int(ridge.feasible.astype(str).eq('True').sum())} of {len(ridge)} "
        f"core cells at 20% ms; batched GBM escalates at most {pct(bat.escalated_frac.max(), 2)}; winning rows {len(win)}",
        "benchmark_budget_overheads_routers.json, benchmark_budget_overheads.json, benchmark_budget_routers.csv "
        "(core cells, ms, 20%)",
        "ARTIFACT DISAGREEMENT: the two overhead files measured the feature extraction at different times and give "
        "different shares; the routers file is the one the router budget table uses." if disagree else "")


@claim("C2", "The detection-list router costs 0.65 ms, about 3.5 % of a full pass, still escalates 16.5 % of inputs at "
             "the 20 % latency budget, and is the only learned allocator on the core track that beats random there")
def c2(A):
    o = _ov(A, "benchmark_budget_overheads_routers.json")
    ov = o["overheads"]
    ms = {t: ov[f"r1_features_ms_{t}"] + ov["r1_mlp_predict_ms"] for t in CORE}
    share = {t: ms[t] / o["costs"][t]["640"]["ms"] for t in CORE}
    b = A.csv("benchmark_budget_routers.csv")
    b = b[is_core(b) & (b.unit == "ms") & np.isclose(b.budget_level, 0.2)]
    esc = b[b.signal == "R1_mlp_reg"].groupby("track").escalated_frac.first()
    win = b[b.signal.isin(LEARNED) & (b.minus_random_lo > 0)]
    ok = (all(r(v, 2) == 0.65 for v in ms.values()) and all(r(100 * v, 1) in (3.4, 3.5, 3.6) for v in share.values())
          and r(100 * esc.get("KITTI", np.nan), 1) == 16.5 and len(win) > 0 and win.signal.str.startswith("R1_mlp").all())
    return res(bool(ok), ", ".join(f"{t} {ms[t]:.3f} ms = {pct(share[t])} of FULL, escalates {pct(esc.get(t, np.nan))}"
                                   for t in CORE) + "; learned rows beating random at 20% ms: " + (wlist(win) or "none"),
               "benchmark_budget_overheads_routers.json; benchmark_budget_routers.csv (core cells, ms, 20%)",
               "'The detection-list router' read as the MLP variants (R1_mlp_reg, R1_mlp_clf), which share the cost.")


@claim("C3", "The pixel router spends 26–50 % of a full pass on resize and upload and beats random nowhere, even at no "
             "cost")
def c3(A):
    o = _ov(A, "benchmark_budget_overheads_routers.json")
    ov = o["overheads"]
    pre = {t: ov[f"r2_pre_ms_{t}"] / o["costs"][t]["640"]["ms"] for t in CORE}
    tot = {t: (ov[f"r2_pre_ms_{t}"] + ov["r2_trt_ms"]) / o["costs"][t]["640"]["ms"] for t in CORE}
    rt = A.csv("benchmark_table_routers.csv")
    w = rt[(rt.signal == "R2_cnn_clf") & (rt.minus_random_lo > 0)]
    lo, hi = min(tot.values()), max(tot.values())
    ok = range_ok(lo, hi, 26, 50) and len(w) == 0
    return res(bool(ok), "resize + upload + TensorRT: " + ", ".join(f"{t} {pct(v)}" for t, v in tot.items()) +
               "; resize + upload alone: " + ", ".join(f"{t} {pct(v)}" for t, v in pre.items()) +
               f"; winning R2 rows at any quota under a frame quota: {len(w)}",
               "benchmark_budget_overheads_routers.json; benchmark_table_routers.csv (R2, every quota)",
               "The 26–50 % range matches resize + upload + TensorRT inference, not resize and upload alone.")


@claim("C4", "Uniform full fidelity is infeasible below 28.6 % of a full pass on KITTI and nuPlan and 34.3 % on "
             "nuScenes; where feasible, at the 50 % latency budget it is ahead of the best allocator in seven of the ten "
             "core cells")
def c4(A):
    o = _ov(A, "benchmark_budget_overheads.json")["costs"]
    thr = {t: 1 - o[t]["cheap"]["ms"] / o[t]["640"]["ms"] for t in ("KITTI", "nuPlan", "nuScenes")}
    c = A.csv("causal_threshold.csv")
    c = c[is_core(c) & np.isclose(c.budget_level.astype(float), 0.5)]
    uni = c[c.signal == "uniform_full"].set_index(["track", "geometry", "system", "target"]).gain_share_all_cheap
    alloc = c[(c.policy == "B") & c.variant.isin(["V1", "V2"]) & c.feasible.astype(str).isin(["True"])]
    best = alloc.groupby(["track", "geometry", "system", "target"]).gain_share_all_cheap.max()
    j = pd.concat([uni.rename("uniform"), best.rename("best")], axis=1)
    ahead = int((j.uniform > j.best).sum())
    ok = r(100 * thr["KITTI"], 1) == 28.6 and r(100 * thr["nuScenes"], 1) == 34.3 and ahead == 7
    return res(bool(ok), ", ".join(f"{t} {pct(v)}" for t, v in thr.items()) + f"; at 50% ms uniform full ahead in "
               f"{ahead} of {len(j)} core cells: " + "; ".join(f"{' '.join(k)} {pct(x.uniform, 2)} vs {pct(x.best, 2)}"
                                                              for k, x in j.iterrows()),
               "benchmark_budget_overheads.json (costs); causal_threshold.csv (budget 0.5; uniform_full and policy B, "
               "V1, feasible rows)",
               "Compared as loss reduction against the all-cheap loss (the uniform row's nDG is 1 by construction). "
               "'Best allocator' = the best feasible streaming allocator (policy B) over both variants V1 and V2: the "
               "reading that reproduces the statement's seven on the pre-Task-23 table (V1 alone gives eight there).")


@claim("C5", "Charged as a skipping router, the pixel router escalates 83.5 % of KITTI inputs at the 50 % latency "
             "budget against 23.9 % under the cascade, and over the 38 cell-unit-budget combinations with a positive "
             "share it beats random in none and loses in three")
def c5(A):
    s = A.csv("skip_accounting.csv")
    sh = s[(s.section == "shares") & (s.track == "KITTI") & (s.unit == "ms") & np.isclose(s.budget_level.astype(float), 0.5)].iloc[0]
    ev = s[(s.section == "evaluation") & (s.share_skipping.astype(float) > 0)]
    b, l = int((ev.beats_random.astype(str) == "True").sum()), int((ev.loses_to_random.astype(str) == "True").sum())
    e = A.csv("energy_module_skip_accounting.csv")
    e = e[(e.section == "evaluation") & (e.share_skipping.astype(float) > 0)]
    alt = "; ".join(f"{cv}: {len(g)} combinations, beats {int((g.beats_random.astype(str) == 'True').sum())}, loses "
                    f"{int((g.loses_to_random.astype(str) == 'True').sum())}" for cv, g in e.groupby("convention"))
    ok = (r(100 * sh.share_skipping, 1) == 83.5 and r(100 * sh.share_cascade, 1) == 23.9 and len(ev) == 38 and b == 0
          and l == 3)
    return res(bool(ok), f"KITTI 50% ms: skipping {pct(sh.share_skipping)}, cascade {pct(sh.share_cascade)}; "
                         f"{len(ev)} combinations with a positive share, beats random in {b}, loses in {l}",
               "skip_accounting.csv (shares; evaluation)",
               f"energy_module_skip_accounting.csv, by energy convention: {alt}. No shipped table gives 38 "
               "combinations, before or after Task 23.")


# ---- target swap, transfer, objective swap

@claim("T1", "Refitting on perception-gain labels gives a pooled mean paired difference of +0.12 [−0.09, +0.27] for the "
             "ridge gate and +0.09 [−0.09, +0.21] for the gradient-boosted gate, with every interval containing zero "
             "for all six architectures and all labels; four of six lean to the decision-value target; the "
             "decision-value target wins significantly in 8 of 72 pooled pairs and the perception-gain target in 5; "
             "over the ten core cells the gate means are +0.007 and −0.011")
def t1(A):
    s = A.json("benchmark_target_swap_summary.json")["pooled"]
    pa = s["primary"]["per_architecture"]
    rg, gb = pa["gate_ridge"], pa["gate_gbm"]
    allzero = all(v["ci_lo"] <= 0 <= v["ci_hi"] for g in s.values() for v in g["per_architecture"].values())
    lean = sum(v["mean_diff_V_minus_G"] > 0 for v in pa.values())
    vsig = sum(v["cells_diff_ci_above_0"] for v in pa.values())
    gsig = sum(v["cells_diff_ci_below_0"] for v in pa.values())
    t = A.csv("benchmark_target_swap.csv")
    t = t[(t.g_variant == "primary") & (t.setting == "quota") & np.isclose(t.quota.astype(float), 0.2) & is_core(t)]
    core = {a: float(t[t.arch == a]["diff"].mean()) for a in ("gate_ridge", "gate_gbm")}
    ok = ((r(rg["mean_diff_V_minus_G"], 2), r(rg["ci_lo"], 2), r(rg["ci_hi"], 2)) == (0.12, -0.09, 0.27)
          and (r(gb["mean_diff_V_minus_G"], 2), r(gb["ci_lo"], 2), r(gb["ci_hi"], 2)) == (0.09, -0.09, 0.21)
          and allzero and lean == 4 and vsig == 8 and gsig == 5
          and r(core["gate_ridge"], 3) == 0.007 and r(core["gate_gbm"], 3) == -0.011)
    return res(bool(ok), f"ridge {rg['mean_diff_V_minus_G']:+.2f} {ci_(rg['ci_lo'], rg['ci_hi'], sign=True)}, GBM "
                         f"{gb['mean_diff_V_minus_G']:+.2f} {ci_(gb['ci_lo'], gb['ci_hi'], sign=True)}; every interval "
                         f"contains zero: {allzero}; lean to V: {lean} of 6; significant V {vsig}, G {gsig} of 72; core-10 "
                         f"gate means {core['gate_ridge']:+.3f}, {core['gate_gbm']:+.3f}",
               "benchmark_target_swap_summary.json (pooled, primary G); benchmark_target_swap.csv (primary, 20%, core cells)",
               "Pooled over 12 cells (10 core + 2 PDM-Closed) at 20%, V minus G.")


@claim("T2", "Median transfer regret is −0.002 over the 528 off-diagonal entries, off-diagonal entries beat random 18 of "
             "132 against 9 of 78 on the diagonal at 20 %, and the gradient-boosted gate fitted for PDM-Closed reaches "
             "0.94 on IDM where the gate fitted for IDM reaches 0.00")
def t2(A):
    c = A.csv("consumer_transfer.csv")
    e = c[(c.trained_for != "SUMMARY") & c.signal.notna() & (c.signal != "all")]
    off = e[e.diagonal.astype(str) == "False"]
    med = float(off.transfer_regret.median())
    q2 = e[np.isclose(e.quota.astype(float), 0.2)]
    f = lambda x: x.astype(str) == "True"                                         # noqa: E731
    o2, d2 = q2[q2.diagonal.astype(str) == "False"], q2[q2.diagonal.astype(str) == "True"]
    n = e[(e.track == "nuPlan") & (e.signal == "gate_gbm") & np.isclose(e.quota.astype(float), 0.2)]
    pdm = n[(n.trained_for.str.startswith("pdm_closed")) & (n.evaluated_on.str.startswith("idm"))].ndg.max()
    idm = n[(n.trained_for.str.startswith("idm")) & (n.evaluated_on.str.startswith("idm"))].ndg.max()
    ok = (len(off) == 528 and r(med, 3) == -0.002 and int(f(o2.beats_random_all_draws).sum()) == 18 and len(o2) == 132
          and int(f(d2.beats_random_all_draws).sum()) == 9 and len(d2) == 78 and r(pdm, 2) == 0.94 and r(idm, 2) == 0.00)
    return res(bool(ok), f"median regret {med:+.4f} over {len(off)} off-diagonal entries; at 20% off-diagonal beating "
                         f"random {int(f(o2.beats_random_filtered).sum())} of {len(o2)} (filtered; "
                         f"{int(f(o2.beats_random_all_draws).sum())} with every draw), diagonal "
                         f"{int(f(d2.beats_random_filtered).sum())} of {len(d2)} (filtered; "
                         f"{int(f(d2.beats_random_all_draws).sum())} with every draw); GBM gate PDM-Closed -> IDM "
                         f"{pdm:.2f}, IDM -> IDM {idm:.2f}",
               "consumer_transfer.csv (entries; 20% for the counts)",
               "The statement does not say which interval the counts use; the every-draw counts are the ones that "
               "reproduce it on the pre-Task-23 table (18 and 9). Both are reported.")


@claim("T3", "Ranking signals by realised perception gain rather than decision value changes the winner in 42 of 52 "
             "cell-budget pairs under one gain and 40 of 52 under the other, with median Kendall tau +0.17 and +0.01; "
             "in 16 and 22 of the 52 the perception-selected signal does not beat random on decision value; the median "
             "regret is 0.10 of nDG, and 4 of 52 differences exclude zero under the first gain and none under the second")
def t3(A):
    o = A.csv("objective_swap.csv")
    o = o[(o.section == "compare") & (o.pool == "headline") & (o.e_dec_defined.astype(str) == "True")]
    out, got = [], {}
    for obj in ("E_perc_dE", "E_perc_risk"):
        x = o[o.objective == obj]
        excl = int(((x.regret_gain_lo > 0) | (x.regret_gain_hi < 0)).sum())
        got[obj] = (len(x), int((x.argmax_differs.astype(str) == "True").sum()), float(x.kendall_tau.median()),
                    int((x.perc_winner_worse_than_random.astype(str) == "True").sum()), float((-x.regret_ndg).median()), excl)
        n, d, tau, worse, reg, ex = got[obj]
        out.append(f"{obj}: {d} of {n} winners change, median tau {tau:+.3f}, perception pick not beating random {worse}, "
                   f"median regret {reg:.3f}, regret intervals excluding zero {ex}")
    a, b = got["E_perc_dE"], got["E_perc_risk"]
    ok = (a[0] == b[0] == 52 and a[1] == 42 and b[1] == 40 and r(a[2], 2) == 0.17 and r(b[2], 2) == 0.01
          and a[3] == 16 and b[3] == 22 and r(a[4], 2) == 0.10 and a[5] == 4 and b[5] == 0)
    return res(bool(ok), "; ".join(out), "objective_swap.csv (compare, headline pool, E_dec defined)",
               "Regret = nDG(decision winner) - nDG(perception winner) on decision value. 'The median regret' is read "
               "as the first gain's (E_perc_dE), which reproduces 0.10 on the pre-Task-23 table; pooled over both it was 0.11.")


# ---- robustness, streaming, nuPlan, planner

@claim("R1", "Requiring a KITTI detection to persist for three frames removes 26–32 % of detections and moves harmed "
             "shares by at most 7 points")
def r1(A):
    p = A.csv("persistence_sweep.csv")
    p = p[p.section.isna() & p.n.notna() & ~p.pair.astype(str).str.startswith("summary")]
    one = p[p.n == 1].set_index(["pair", "geometry", "system"])
    three = p[p.n == 3].set_index(["pair", "geometry", "system"])
    red = pd.concat([1 - three.dets_per_frame_cheap / one.dets_per_frame_cheap,
                     1 - three.dets_per_frame_full / one.dets_per_frame_full])
    dh = (three.harmed - one.harmed).abs()
    ok = range_ok(red.min(), red.max(), 26, 32) and r(100 * dh.max(), 0) <= 7
    return res(bool(ok), f"detections removed {pct(red.min())}–{pct(red.max())} (CHEAP and FULL); largest move of the "
                         f"harmed share {100 * dh.max():.1f} points ({' '.join(dh.idxmax())})",
               "persistence_sweep.csv (n = 3 against n = 1, all units)")


@claim("R2", "Streaming: a causal running cap costs −0.10 [−0.18, −0.02] against the offline ranking, an adaptive "
             "threshold −0.08, the token bucket matches the cap, rate misses fall from 63 to 12, and a randomly ordered "
             "cap costs 0.01")
def r2(A):
    s = A.csv("streaming_controllers.csv")
    p = s[(s.section == "pooled") & np.isclose(s.rate_target.astype(float), 0.2) & (s.variant == "V1")]
    g = lambda sig, pol: p[(p.signal == sig) & (p.policy == pol)].iloc[0]            # noqa: E731
    B, D, E, RB = g("all_learned", "B"), g("all_learned", "D"), g("all_learned", "E"), g("random", "B")
    rows = s[(s.section == "row") & np.isclose(s.rate_target.astype(float), 0.2) & (s.variant == "V1")
             & s.signal.isin(LEARNED) & (s.pooled_cell.astype(str) == "True")]
    miss = {pol: int((rows[rows.policy == pol].rate_abs_dev > 0.05).sum()) for pol in ("B", "D", "E")}
    same = abs(B.ndg_minus_C - E.ndg_minus_C) < 1e-12
    ok = ((r(B.ndg_minus_C, 2), r(B.ndg_minus_C_lo, 2), r(B.ndg_minus_C_hi, 2)) == (-0.10, -0.18, -0.02)
          and r(D.ndg_minus_C, 2) == -0.08 and same and miss["B"] == 63 and miss["D"] == 12
          and r(abs(RB.ndg_minus_C), 2) == 0.01)
    return res(bool(ok), f"cap (B) {B.ndg_minus_C:+.3f} {ci_(B.ndg_minus_C_lo, B.ndg_minus_C_hi, sign=True)}; adaptive "
                         f"threshold (D) {D.ndg_minus_C:+.3f}; token bucket (E) {E.ndg_minus_C:+.3f} (equal to B: {same}); "
                         f"rows missing the rate by > 5 points: B {miss['B']}, D {miss['D']}, E {miss['E']}; random under "
                         f"the cap {RB.ndg_minus_C:+.3f}",
               "streaming_controllers.csv (pooled and rows, 20%, V1, learned signals)",
               "Policies as in the artifact: B causal running cap, D adaptive threshold, E token bucket, C offline "
               "ranking.")


@claim("E1", "PDM-Closed harms 44 % of the states whose safety loss changes and 38 % under the scalar loss (20 and 39 "
             "of 1,440), with harm ratios 0.33 and 0.35, and an oracle at 20 % capacity reduces the safety loss by "
             "14.7 % against 9.8 % for uniform escalation")
def e1(A):
    c = A.csv("nuplan_real_perception_cells.csv")
    c = c[(c.planner == "pdm_closed") & (c.split == "all") & (c.variant == "primary")].set_index("loss")
    s, j = c.loc["safety"], c.loc["scalar_J"]
    hs, hj = int(round(s.harm_rate * s.affected)), int(round(j.harm_rate * j.affected))
    ok = (r(100 * s.harm_rate, 0) == 44 and r(100 * j.harm_rate, 0) == 38 and hs == 20 and hj == 39 and int(s.states) == 1440
          and r(s.rho, 2) == 0.33 and r(j.rho, 2) == 0.35 and r(100 * s.oracle20_reduction, 1) == 14.7
          and r(100 * s.all_full_reduction, 1) == 9.8)
    return res(bool(ok), f"safety {pct(s.harm_rate)} ({hs} of {int(s.affected)} affected, {int(s.states)} states), scalar "
                         f"{pct(j.harm_rate)} ({hj} of {int(j.affected)}); ratios {s.rho:.3f}, {j.rho:.3f}; oracle@20 "
                         f"{pct(s.oracle20_reduction)}, uniform {pct(s.all_full_reduction)}",
               "nuplan_real_perception_cells.csv (PDM-Closed, primary, all logs)", NUPLAN_REAL)


@claim("E2", "IDM's safety loss changes on 5 of 1,440 states; dropping the unmatched detections leaves PDM-Closed's "
             "safety loss changing on 6 states instead of 45")
def e2(A):
    c = A.csv("nuplan_real_perception_cells.csv")
    c = c[(c.split == "all") & (c.loss == "safety")].set_index(["planner", "variant"])
    idm, nofp, prim = c.loc[("idm", "primary")], c.loc[("pdm_closed", "nofp")], c.loc[("pdm_closed", "primary")]
    ok = int(idm.affected) == 5 and int(idm.states) == 1440 and int(nofp.affected) == 6 and int(prim.affected) == 45
    return res(bool(ok), f"IDM {int(idm.affected)} of {int(idm.states)}; PDM-Closed without unmatched detections "
                         f"{int(nofp.affected)}, primary {int(prim.affected)}",
               "nuplan_real_perception_cells.csv (all logs; variants primary, nofp)", NUPLAN_REAL)


@claim("P1", "The learned planner has non-zero decision value on 1,226 of 2,655 frames, of which 630 (51.4 %) are "
             "harmed; the oracle at 20 % reduces the loss by 4.60 % against 0.81 % for uniform escalation")
def p1(A):
    c = cal_S(A)
    x = c[(c.dataset == "nuScenes") & (c.geometry == "oracle") & (c.system == "plan")].iloc[0]
    ok = (int(x.affected) == 1226 and int(x.n_frames) == 2655 and int(x.harmed) == 630 and r(100 * x.harm_rate, 1) == 51.4
          and r(100 * x.oracle20_reduction, 2) == 4.60 and r(100 * x.all_full_reduction, 2) == 0.81)
    return res(bool(ok), f"{int(x.affected)} of {int(x.n_frames)}, {int(x.harmed)} harmed ({pct(x.harm_rate)}); oracle@20 "
                         f"{pct(x.oracle20_reduction, 2)}, uniform {pct(x.all_full_reduction, 2)}",
               "calibration_cells.csv (nuScenes oracle q_plan ADE, S0, all units)",
               "The statement names no geometry; oracle geometry is the row whose camera-frame values it quoted.")


def run_claims():
    rows = []
    for cid, statement, fn in CLAIMS:
        now = fn(NEW)
        try:
            old = fn(OLD)
            before = (f"Before Task 23 (camera frame, as first shipped): {'held' if old['ok'] else 'did not hold'} "
                      f"({old['value']}).")
        except Exception as e:                                                   # noqa: BLE001
            before = f"Before Task 23: not evaluable ({type(e).__name__}: {e})."
        verdict = "UNVERIFIABLE" if now["ok"] is None else ("PASS" if now["ok"] else "FAIL")
        rows.append({"claim_id": cid, "statement": statement, "verdict": verdict, "value_now": now["value"],
                     "artifact": now["artifact"], "note": " ".join(x for x in (now["note"], before) if x)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ the pack

class Pack:
    def __init__(self):
        self.rows = []

    def add(self, qid, **kw):
        row = {c: "" for c in PACK_COLUMNS}
        row.update(quantity_id=qid, **kw)
        for c in ("estimate", "ci_lo", "ci_hi"):
            if isinstance(row[c], (float, np.floating)) and not np.isfinite(row[c]):
                row[c] = ""
        self.rows.append(row)

    def frame(self):
        d = pd.DataFrame(self.rows, columns=PACK_COLUMNS)
        dup = d.quantity_id.duplicated(keep=False)
        assert not dup.any(), d[dup].quantity_id.tolist()[:10]
        return d


def tgt(system, target):
    return TARGET_LABEL.get((system, target), f"{system} {target}")


def nuplan_note(track, source_is_real):
    if track != "nuPlan":
        return ""
    return NUPLAN_REAL if source_is_real else NUPLAN_DIAG


def scope(split):
    return {"all": "SETTING: all units (train, validation and test); not an allocator result",
            "test": "ALLOCATOR: frozen test split"}.get(split, split)


def pack_harm(P):
    c = NEW.csv("calibration_cells.csv")
    sysmap = {"brake": ("brake", "J"), "traj": ("traj", "JB"), "plan": ("plan_ade", "JC_ade")}
    for x in c.itertuples():
        s, t = sysmap[x.system]
        base = dict(artifact_file="calibration_cells.csv", track=x.dataset, geometry=x.geometry, system=x.system,
                    target=tgt(s, t), split=x.split, units_scope=scope(x.split), quota_or_budget="",
                    signal_or_policy=f"uniform escalation, operating-point scheme {x.scheme} (t_c {x.t_cheap:g}, t_f {x.t_full:g})",
                    n_units=int(x.n_units), n_affected=int(x.affected))
        q = f"cal|{x.cell}|{x.scheme}|{x.split}"
        P.add(q + "|harm_rate", description=f"harm rate: harmed among affected inputs ({x.cell}, {x.scheme})",
              estimate=x.harm_rate, ci_lo=x.harm_rate_lo, ci_hi=x.harm_rate_hi, interval_method=IM_CAL,
              denominator="affected inputs (V != 0)", notes=f"{int(x.harmed)} harmed of {int(x.affected)} affected, "
              f"{int(x.n_frames)} inputs", **base)
        P.add(q + "|harm_ratio", description=f"harm ratio rho = sum of harm / sum of benefit ({x.cell}, {x.scheme})",
              estimate=x.rho, ci_lo=x.rho_lo, ci_hi=x.rho_hi, interval_method=IM_CAL,
              denominator="gross benefit of escalation (sum of positive V)", notes="", **base)
        for col, what in (("all_full_reduction", "uniform escalation (all FULL)"), ("oracle20_reduction", "oracle at 20%")):
            P.add(q + f"|{col}", description=f"loss reduction, {what} ({x.cell}, {x.scheme})", estimate=getattr(x, col),
                  interval_method=IM_NONE, denominator="all-cheap loss", notes="", **base)
        for m in ("cheap", "full"):
            for k in ("precision", "recall", "det_per_frame"):
                P.add(q + f"|{m}_{k}", description=f"{m.upper()} {k.replace('_', ' ')} at t = "
                      f"{x.t_cheap if m == 'cheap' else x.t_full:g} ({x.cell}, {x.scheme})",
                      estimate=getattr(x, f"{m}_{k}"), interval_method=IM_NONE,
                      denominator={"precision": "detections", "recall": "reference objects",
                                   "det_per_frame": "inputs"}[k], notes="", **base)
        if x.dataset == "nuScenes":
            for g in ("exact_FN", "FN_FP", "E5_combined", "E_risk"):
                P.add(q + f"|sign_disagreement_{g}", description=f"share of frames where perception gain {g} and V "
                      f"disagree in sign ({x.cell}, {x.scheme})", estimate=getattr(x, f"{g}_sign_disagreement"),
                      interval_method=IM_NONE, denominator="frames where both the gain and V are non-zero",
                      notes=f"n = {getattr(x, f'{g}_n_both_nonzero'):g}", **base)
    # the 14 benchmark cells (nuPlan here: the transported diagnostic), all quotas of the oracle
    b = NEW.csv("benchmark_cells.csv")
    for x in b.itertuples():
        base = dict(artifact_file="benchmark_cells.csv", track=x.track, geometry=x.geometry, system=x.system,
                    target=tgt(x.system, x.target), split=x.split, units_scope=scope(x.split),
                    signal_or_policy="uniform escalation", n_units=int(x.n_units),
                    n_affected=int(round(x.affected_share * x.n_frames)))
        note = nuplan_note(x.track, False)
        q = f"bcell|{cell_key(x._asdict())}|{x.split}"
        P.add(q + "|harmed_among_affected", description="harmed among affected inputs", estimate=x.harmed_among_affected,
              interval_method=IM_NONE, denominator="affected inputs", notes=note, quota_or_budget="", **base)
        P.add(q + "|harm_ratio", description="harm ratio D = gross harm / gross benefit", estimate=x.destroyed_benefit_D,
              interval_method=IM_NONE, denominator="gross benefit", notes=note, quota_or_budget="", **base)
        P.add(q + "|affected_share", description="affected share", estimate=x.affected_share, interval_method=IM_NONE,
              denominator="all inputs", notes=f"{int(x.n_frames)} inputs. {note}", quota_or_budget="", **base)
        P.add(q + "|all_full_reduction", description="loss reduction, uniform escalation", estimate=x.all_full_reduction,
              interval_method=IM_NONE, denominator="all-cheap loss", notes=note, quota_or_budget="", **base)
        for qq in (10, 20, 30, 50):
            P.add(q + f"|oracle{qq}_reduction", description=f"loss reduction, oracle at {qq}%",
                  estimate=getattr(x, f"oracle{qq}_reduction"), interval_method=IM_NONE, denominator="all-cheap loss",
                  notes=note, quota_or_budget=f"quota {qq}%", **base)
    s = NEW.csv("statistics_hardening.csv")
    for x in s[s.section == "harm"].itertuples():
        base = dict(artifact_file="statistics_hardening.csv", track=x.track, geometry=x.geometry, system=x.system,
                    target=tgt(x.system, x.target), split=x.split, units_scope=scope(x.split), quota_or_budget="",
                    signal_or_policy="uniform escalation", n_affected=int(x.n_affected), interval_method=IM_NONE)
        note = nuplan_note(x.track, True)
        q = f"harmden|{cell_key(x._asdict())}|{x.split}"
        P.add(q + "|over_affected", description="harmed inputs as a share of affected inputs", estimate=x.harmed_over_affected,
              denominator="affected inputs", notes=f"{int(x.n_harmed)} of {int(x.n_affected)}. {note}", **base)
        P.add(q + "|over_all", description="harmed inputs as a share of all inputs", estimate=x.harmed_over_all_inputs,
              denominator="all inputs", notes=f"{int(x.n_harmed)} of {int(x.n_inputs)}. {note}", **base)
    n = NEW.csv("nuplan_real_perception_cells.csv")
    for x in n.itertuples():
        base = dict(artifact_file="nuplan_real_perception_cells.csv", track="nuPlan", geometry="n/a", system=x.planner,
                    target=f"{x.planner} {x.loss}", split=x.split, units_scope=scope(x.split), quota_or_budget="",
                    signal_or_policy=f"uniform escalation, perception variant {x.variant}", n_units=int(x.logs),
                    n_affected=int(x.affected), interval_method=IM_NR)
        note = NUPLAN_REAL + ("" if x.variant == "primary" else f" (sensitivity variant {x.variant})")
        q = f"nuplan_real|{x.planner}|{x.loss}|{x.variant}|{x.split}"
        P.add(q + "|harm_rate", description="harm rate among affected states", estimate=x.harm_rate, ci_lo=x.harm_rate_lo,
              ci_hi=x.harm_rate_hi, denominator="affected states", notes=f"{x.v_neg:g} of {x.affected:g}. {note}", **base)
        P.add(q + "|harm_ratio", description="harm ratio", estimate=x.rho, ci_lo=x.rho_lo, ci_hi=x.rho_hi,
              denominator="gross benefit", notes=note, **base)
        P.add(q + "|all_full_reduction", description="loss reduction, uniform escalation", estimate=x.all_full_reduction,
              ci_lo=x.all_full_reduction_lo, ci_hi=x.all_full_reduction_hi, denominator="all-cheap loss", notes=note, **base)
        P.add(q + "|oracle20_reduction", description="loss reduction, oracle at 20%", estimate=x.oracle20_reduction,
              ci_lo=x.oracle20_reduction_lo, ci_hi=x.oracle20_reduction_hi, denominator="all-cheap loss", notes=note,
              **{**base, "quota_or_budget": "quota 20%"})


def pack_ndg(P):
    for name, im, im_d, real in (("benchmark_table.csv", IM_92, IM_92_DIFF, False),
                                 ("benchmark_table_routers.csv", IM_103, IM_103, False),
                                 ("benchmark_table_nuplan_real.csv", IM_92, IM_92_DIFF, True)):
        t = NEW.csv(name)
        t = t[t.quota.notna()]
        for x in t.itertuples():
            split = getattr(x, "split", "test")
            if name == "benchmark_table_routers.csv" and x.signal == "R2_cnn_clf":
                im = im_d = IM_107
            elif name == "benchmark_table_routers.csv":
                im = im_d = IM_103
            kind = ("deployable" if str(x.deployable) == "True" else "diagnostic (needs FULL or reference)")
            k_, tie_, na_ = getattr(x, "k", np.nan), getattr(x, "tie_frac", np.nan), getattr(x, "na_reason", "")
            note = " ".join(z for z in (nuplan_note(x.track, real),
                                        f"k = {k_:g}; ties in {tie_:.3f} of the selection" if np.isfinite(k_) else "",
                                        na_ if isinstance(na_, str) else "") if z)
            base = dict(artifact_file=name, track=x.track, geometry=x.geometry, system=x.system,
                        target=tgt(x.system, x.target), split=split, units_scope=scope(split),
                        quota_or_budget=f"quota {x.quota:.0%}", signal_or_policy=f"{x.signal} ({kind})",
                        n_units=int(x.n_units) if np.isfinite(getattr(x, "n_units", np.nan)) else "",
                        n_affected=int(x.n_affected) if np.isfinite(getattr(x, "n_affected", np.nan)) else "")
            q = f"ndg|{name.replace('.csv', '')}|{cell_key(x._asdict())}|{split}|{x.signal}|{x.quota:g}"
            P.add(q, description="nDG", estimate=x.eta, ci_lo=x.eta_lo, ci_hi=x.eta_hi, interval_method=im,
                  denominator="oracle prize at the same quota",
                  notes=f"{note} boot_dropped {getattr(x, 'boot_dropped', np.nan):g}".strip(), **base)
            P.add(q + "|minus_random", description="nDG minus random (paired)", estimate=x.minus_random,
                  ci_lo=x.minus_random_lo, ci_hi=x.minus_random_hi, interval_method=im_d,
                  denominator="oracle prize at the same quota", notes=note, **base)
            if "gain" in t.columns and np.isfinite(getattr(x, "gain", np.nan)):
                P.add(q + "|gain", description="realised gain (loss units)", estimate=x.gain, interval_method=IM_NONE,
                      denominator="none (loss units)", notes=f"oracle prize {x.prize:g}. {note}", **base)
    s = NEW.csv("statistics_hardening.csv")
    for x in s[(s.section == "paired_gain") & s.quota.notna()].itertuples():
        base = dict(artifact_file="statistics_hardening.csv", track=x.track, geometry=x.geometry, system=x.system,
                    target=tgt(x.system, x.target), split="test", units_scope=scope("test"),
                    quota_or_budget=f"quota {x.quota:.0%}", signal_or_policy=x.signal, n_units=int(x.n_test_units))
        q = f"gain_all_draws|{cell_key(x._asdict())}|{x.signal}|{x.quota:g}"
        P.add(q, description="realised gain minus random, every bootstrap draw kept (loss units)",
              estimate=x.delta_vs_random, ci_lo=x.delta_lo_all_draws, ci_hi=x.delta_hi_all_draws, interval_method=IM_ALLDRAWS,
              denominator="none (loss units)", notes=f"filtered interval [{x.delta_lo_filtered:g}, {x.delta_hi_filtered:g}], "
              f"{x.draws_dropped_by_filter:g} draws dropped by the filter. {nuplan_note(x.track, True)}", **base)


def pack_budgets(P):
    for name, im in (("benchmark_budget_two_level.csv", IM_93), ("benchmark_budget_routers.csv", IM_93),
                     ("benchmark_budget_multifidelity.csv", IM_93), ("benchmark_budget_nuplan_real.csv", IM_93)):
        b = NEW.csv(name)
        real = "nuplan_real" in name
        for i, x in enumerate(b.itertuples()):
            track = getattr(x, "track", "KITTI")
            feas = str(getattr(x, "feasible", ""))
            note = " ".join(z for z in (f"feasible {feas}", f"overhead {getattr(x, 'overhead', np.nan):g} {x.unit} per input",
                                         f"escalated {getattr(x, 'escalated_frac', np.nan):g}",
                                         nuplan_note(track, real) if track == "nuPlan" else "",
                                         str(x.note) if isinstance(getattr(x, "note", None), str) else "") if z)
            geometry, system, target = (getattr(x, "geometry", ""), getattr(x, "system", ""), getattr(x, "target", ""))
            if "multifidelity" in name:
                geometry, note = "mono", "KITTI mono multi-fidelity cascade 320 -> {384, 512, 640}. " + note
            P.add(f"budget|{name.replace('.csv', '')}|{i}", description=f"nDG under a measured {x.unit} budget",
                  artifact_file=name, track=track, geometry=geometry, system=system, target=tgt(system, target),
                  split="test", units_scope=scope("test"), quota_or_budget=f"{x.unit} budget level {x.budget_level:g} of "
                  f"a full pass ({x.budget_per_frame:g} {x.unit} per input)", signal_or_policy=x.signal,
                  estimate=getattr(x, "eta", np.nan), ci_lo=getattr(x, "eta_lo", np.nan), ci_hi=getattr(x, "eta_hi", np.nan),
                  interval_method=im, n_units="", n_affected="", denominator="oracle prize at the same budget, no overhead",
                  notes=note)


def pack_task24(P):
    f = FINAL / "published_objective_routers.csv"
    if not f.exists():
        return
    t = pd.read_csv(f, keep_default_na=False, na_values=[""], float_precision="round_trip")
    for x in t.itertuples():
        v = f" ({x.variant})" if isinstance(x.variant, str) and x.variant else ""
        base = dict(artifact_file="published_objective_routers.csv", track=x.track, geometry=x.geometry, system=x.system,
                    target=tgt(x.system, x.target), split="test", units_scope=scope("test"),
                    quota_or_budget=f"quota {x.quota:.0%}", signal_or_policy=f"{x.architecture} trained on {x.objective}{v}",
                    n_units=int(x.n_units), denominator="oracle prize at the same quota")
        q = f"task24|{cell_key(x)}|{x.architecture}|{x.objective}{v}|{x.quota:g}"
        P.add(q, description="nDG", estimate=x.ndg, ci_lo=x.ndg_lo, ci_hi=x.ndg_hi, interval_method=IM_92.replace(
            "92's cell order", "149's own sequence"), notes="Task 24: published objective on this benchmark's "
            "architecture" if x.objective != "V" else "V-trained counterpart (shipped scores)", **base)
        P.add(q + "|minus_random", description="nDG minus random (paired)", estimate=x.minus_random,
              ci_lo=x.minus_random_lo, ci_hi=x.minus_random_hi, interval_method=IM_92_DIFF.replace("92's cell order",
              "149's own sequence"), notes="", **base)
        if x.objective != "V":
            P.add(q + "|minus_V", description="nDG minus the V-trained counterpart (paired)", estimate=x.minus_V,
                  ci_lo=x.minus_V_lo, ci_hi=x.minus_V_hi, interval_method=IM_92_DIFF.replace("92's cell order",
                  "149's own sequence").replace("nDG(random)", "nDG(V-trained counterpart)"), notes="", **base)


def pack_misc(P):
    # streaming controllers: pooled
    s = NEW.csv("streaming_controllers.csv")
    for x in s[s.section == "pooled"].itertuples():
        P.add(f"stream|{x.signal}|{x.policy}|{x.variant}|{x.rate_target:g}", description="nDG minus the offline ranking "
              "(pooled over cells and signals)", artifact_file="streaming_controllers.csv", track="pooled", split="test",
              units_scope=scope("test"), quota_or_budget=f"rate {x.rate_target:.0%}",
              signal_or_policy=f"{x.signal}, policy {x.policy} ({ {'A': 'frozen threshold', 'B': 'causal running cap', 'D': 'adaptive threshold', 'E': 'token bucket'}.get(x.policy, x.policy)})",
              estimate=x.ndg_minus_C, ci_lo=x.ndg_minus_C_lo, ci_hi=x.ndg_minus_C_hi,
              interval_method="paired unit bootstrap, 1000 draws, units resampled once per dataset per draw; official "
                              "25% prize filter", denominator="oracle prize at the same rate", n_units="",
              n_affected="", notes=f"{x.n_rows:g} cell x signal rows, {x.rows_improved_over_C:g} improved over C")
    # skipping accounting
    k = NEW.csv("skip_accounting.csv")
    for i, x in enumerate(k[k.section == "evaluation"].itertuples()):
        P.add(f"skip|{i}", description="pixel router charged as a skipping router: nDG minus random",
              artifact_file="skip_accounting.csv", track=x.track, geometry=x.geometry, system=x.system,
              target=tgt(x.system, x.target), split="test", units_scope=scope("test"),
              quota_or_budget=f"{x.unit} budget level {x.budget_level:g}", signal_or_policy="R2_cnn_clf (skipping)",
              estimate=x.ndg_minus_random, ci_lo=x.gain_minus_random_lo, ci_hi=x.gain_minus_random_hi,
              interval_method="paired unit bootstrap of gain minus random (loss units; the interval is on GAIN, the "
                              "estimate on nDG)", denominator=str(x.denominator), n_units=int(x.n_units), n_affected="",
              notes=f"share escalated {x.share_skipping:g}, regime {x.regime}, beats {x.beats_random}, loses {x.loses_to_random}")
    # target swap pooled
    ts = NEW.json("benchmark_target_swap_summary.json")["pooled"]
    for g, v in ts.items():
        for a, z in v["per_architecture"].items():
            P.add(f"target_swap|{g}|{a}", description="pooled mean paired difference nDG(V-target) - nDG(G-target) at 20%",
                  artifact_file="benchmark_target_swap_summary.json", track="pooled (10 core + 2 PDM-Closed)", split="test",
                  units_scope=scope("test"), quota_or_budget="quota 20%", signal_or_policy=f"{a}, G = {g}",
                  estimate=z["mean_diff_V_minus_G"], ci_lo=z["ci_lo"], ci_hi=z["ci_hi"],
                  interval_method="paired unit bootstrap over the pooled cells", denominator="oracle prize",
                  notes=f"{z['cells_diff_ci_above_0']} cells V significantly ahead, {z['cells_diff_ci_below_0']} G ahead")
    # consumer transfer and objective swap summaries
    c = NEW.csv("consumer_transfer.csv")
    for x in c[c.trained_for == "SUMMARY"].itertuples():
        P.add(f"transfer|{x.group}", description="median transfer regret, off-diagonal, 20%",
              artifact_file="consumer_transfer.csv", track=x.track, geometry=x.geometry, split="test",
              units_scope=scope("test"), quota_or_budget="quota 20%", signal_or_policy="all transferred signals",
              estimate=x.median_transfer_regret, interval_method=IM_NONE, denominator="nDG",
              notes=f"{x.n_entries:g} entries; off-diagonal beating random {x.off_diag_beating_random_filtered:g} "
                    f"(filtered), diagonal {x.diag_beating_random_all_draws:g} of {x.n_diag_entries:g} (every draw)")
    o = NEW.csv("objective_swap.csv")
    for i, x in enumerate(o[o.section == "compare"].itertuples()):
        P.add(f"objswap|{x.objective}|{x.pool}|{cell_key(x._asdict())}|{x.quota:g}", description="regret of choosing by "
              "perception gain, on decision value", artifact_file="objective_swap.csv", track=x.track, geometry=x.geometry,
              system=x.system, target=tgt(x.system, x.target), split="test", units_scope=scope("test"),
              quota_or_budget=f"quota {x.quota:.0%}", signal_or_policy=f"{x.objective}, pool {x.pool}: decision winner "
              f"{x.argmax_dec}, perception winner {x.argmax_perc}", estimate=x.regret_gain, ci_lo=x.regret_gain_lo,
              ci_hi=x.regret_gain_hi, interval_method="paired unit bootstrap, 1000 draws, every draw kept",
              denominator="none (loss units)", notes=f"regret in nDG {x.regret_ndg:g}; Kendall tau {x.kendall_tau:g}")
    # robustness
    p = NEW.csv("persistence_sweep.csv")
    for i, x in enumerate(p[p.section.isna() & p.n.notna() & ~p.pair.astype(str).str.startswith("summary")].itertuples()):
        P.add(f"persistence|{x.pair}|{x.geometry}|{x.system}|n{x.n:g}", description=f"harm rate with detections required "
              f"to persist {x.n:g} frames", artifact_file="persistence_sweep.csv", track="KITTI", geometry=x.geometry,
              system=x.system, target=x.system, split="all", units_scope=scope("all"), signal_or_policy="uniform escalation",
              estimate=x.harmed, interval_method=IM_NONE, denominator="affected inputs", n_affected=int(x.affected),
              notes=f"rho {x.rho:g}; detections per frame CHEAP {x.dets_per_frame_cheap:g}, FULL {x.dets_per_frame_full:g}")
    g = NEW.csv("reference_geometry_sweep.csv")
    for x in g[g.section.isna() & g.mono_affected.notna()].itertuples():
        for geo, pre in (("mono", "mono"), ("oracle", "reference")):
            P.add(f"refgeo|{x.pair}|{x.system}|{geo}", description="harm rate, mono lift against reference geometry",
                  artifact_file="reference_geometry_sweep.csv", track="KITTI", geometry=geo, system=x.system,
                  target=x.system, split="all", units_scope=scope("all"), signal_or_policy="uniform escalation",
                  estimate=getattr(x, f"{pre}_harmed"), interval_method=IM_NONE, denominator="affected inputs",
                  n_affected=int(getattr(x, f"{pre}_affected")), notes=f"rho {getattr(x, f'{pre}_rho'):g}")
    # mechanism counts
    m = NEW.csv("mechanism_table.csv")
    for x in m.itertuples():
        for col in ("mean_delta_fp", "mean_delta_fn", "share_delta_fp_pos", "share_delta_fn_nonneg"):
            P.add(f"mechanism|{x.decision}|{x.group}|{col}", description=f"{col} on {x.group} frames",
                  artifact_file="mechanism_table.csv", track="nuScenes", geometry="oracle",
                  system=x.decision, target=SELF_REF if x.decision.startswith("self_control") else tgt("brake", "J"),
                  split="all", units_scope=scope("all"), signal_or_policy="uniform escalation", estimate=getattr(x, col),
                  interval_method=IM_NONE, denominator=f"{x.group} frames" if col.startswith("share") else "per frame",
                  n_affected=int(x.n_frames), notes="")
    # the external track: historical transported planners
    e = NEW.csv("phase0g_external_planner_summary.csv")
    for x in e.itertuples():
        for col in ("harmful_among_affected", "all_full_reduction", "oracle20_reduction"):
            P.add(f"external|{x.planner}|{x.cost}|{col}", description=f"{col}", artifact_file="phase0g_external_planner_summary.csv",
                  track="nuPlan", geometry="n/a", system=x.planner, target=f"{x.planner} {x.cost}", split="all",
                  units_scope=scope("all"), signal_or_policy="uniform escalation", estimate=getattr(x, col),
                  interval_method=IM_NONE, denominator="affected states" if col.startswith("harm") else "all-cheap loss",
                  n_affected=int(x.v_pos + x.v_neg), notes=f"{NUPLAN_DIAG}; {x.n_states:g} states")
    # the self-referenced control of Phase 0F
    f = NEW.csv("phase0f_planning_metric_eta.csv")
    for x in f.itertuples():
        for q in QUOTAS:
            qq = int(round(100 * q))
            P.add(f"phase0f|{x.task}|{x.variant}|{x.signal}|{qq}", description="nDG (Phase 0F)",
                  artifact_file="phase0f_planning_metric_eta.csv", track="nuScenes", geometry=x.variant,
                  system=x.task, target=SELF_REF if "plan" in str(x.task).lower() else str(x.task), split="all",
                  units_scope=scope("all"), quota_or_budget=f"quota {qq}%", signal_or_policy=x.signal,
                  estimate=getattr(x, f"eta_{qq}"), ci_lo=getattr(x, f"eta_{qq}_lo"), ci_hi=getattr(x, f"eta_{qq}_hi"),
                  interval_method="unit bootstrap (Phase 0F), oracle-prize filter as in 92", denominator="oracle prize",
                  n_units=int(x.n_scenes), notes=f"scene-normalised {x.scene_normalised}")


# ------------------------------------------------------------------------------------------------ 3A and 3B

def table_3a():
    """Every allocator variant on the ten core cells: nDG, paired interval, realised gain (every draw kept), LOO range."""
    bt, rt, sh = NEW.csv("benchmark_table.csv"), NEW.csv("benchmark_table_routers.csv"), NEW.csv("statistics_hardening.csv")
    loo = loo_ranges()
    rows = []
    for sig in ("gate_ridge", "gate_gbm") + ROUTERS:
        src = bt[(bt.split == "test")] if sig.startswith("gate") else rt
        x = src[(src.signal == sig) & is_core(src) & src.quota.notna()]
        for z in x.itertuples():
            k = (z.track, z.geometry, z.system, z.target)
            g = sh[(sh.section == "paired_gain") & (sh.signal == sig) & (sh.track == z.track) & (sh.geometry == z.geometry)
                   & (sh.system == z.system) & (sh.target == z.target) & np.isclose(sh.quota.astype(float), z.quota)]
            g = g.iloc[0] if len(g) else None
            lo_, hi_ = loo.get(k + (sig, round(float(z.quota), 2)), (np.nan, np.nan))
            rows.append({"cell": " ".join(k), "signal": sig, "quota": z.quota, "ndg": z.eta,
                         "minus_random": z.minus_random, "minus_random_lo": z.minus_random_lo,
                         "minus_random_hi": z.minus_random_hi, "beats_random": bool(z.minus_random_lo > 0),
                         "gain": g.gain if g is not None else np.nan,
                         "gain_minus_random": g.delta_vs_random if g is not None else np.nan,
                         "gain_lo_all_draws": g.delta_lo_all_draws if g is not None else np.nan,
                         "gain_hi_all_draws": g.delta_hi_all_draws if g is not None else np.nan,
                         "loo_ndg_min": lo_, "loo_ndg_max": hi_,
                         "source": "benchmark_table.csv" if sig.startswith("gate") else "benchmark_table_routers.csv"})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def loo_ranges():
    """Leave-one-test-unit-out nDG range per core cell x allocator x quota, from the saved per-frame scores.

    The influence section of statistics_hardening.csv has it at 20% only; this recomputes every quota with the
    benchmark's own scorer and is checked against that section at 20% (a mismatch stops the script)."""
    from rap import runs as rap_runs
    r1, r2 = rap_runs.latest("routers_r1"), rap_runs.latest("router_r2")
    gates = sorted((ROOT / "results" / "raw").glob("*_budget_gate_scores_ego"))[-1]
    out = {}
    for fz in sorted(r1.glob("scores__*.npz")):
        _, track, geometry, system, target = fz.stem.split("__")
        if track not in CORE:
            continue
        z = np.load(fz, allow_pickle=False)
        d = pd.DataFrame({"seq": z["seq"].astype(str), "frame": z["frame"].astype(int), "unit": z["unit"].astype(str),
                          "split": z["split"].astype(str), "V": z["V"]})
        for a in ROUTERS[:4]:
            d[a] = z[a]
        g = np.load(gates / f"gates__{track}__{geometry}__{system}__{target}.npz", allow_pickle=False)
        gk = pd.DataFrame({"seq": g["key_seq"].astype(str), "frame": g["key_frame"].astype(int),
                           "gate_ridge": g["gate_ridge"], "gate_gbm": g["gate_gbm"]})
        d = d.merge(gk, on=["seq", "frame"], how="left", validate="one_to_one")
        f2 = r2 / fz.name
        if f2.exists():
            z2 = np.load(f2, allow_pickle=False)
            d = d.merge(pd.DataFrame({"seq": z2["seq"].astype(str), "frame": z2["frame"].astype(int),
                                      "R2_cnn_clf": z2["R2_cnn_clf"].astype(float)}), on=["seq", "frame"], how="left")
        t = d[d.split == "test"].reset_index(drop=True)
        v, units = t.V.to_numpy(float), t.unit.to_numpy()
        geo = "n/a" if geometry == "na" else geometry
        for sig in [c for c in ("gate_ridge", "gate_gbm") + ROUTERS if c in t.columns]:
            s = t[sig].to_numpy(float)
            for qi, q in enumerate(scoring.QUOTAS):
                vals = []
                for u in np.unique(units):
                    m = units != u
                    k = scoring.quota_k(int(m.sum()))[qi]
                    prize = scoring.topk_expect(v[m], [v[m]], [k])[0][0][0]
                    if prize <= scoring.EPS:
                        continue
                    vals.append(scoring.topk_expect(s[m], [v[m]], [k])[0][0][0] / prize)
                out[(track, geo, system, target, sig, round(q, 2))] = (min(vals), max(vals)) if vals else (np.nan, np.nan)
    # the check against the shipped influence section at 20%
    inf = NEW.csv("statistics_hardening.csv")
    inf = inf[(inf.section == "influence") & is_core(inf)]
    bad = []
    for z in inf.itertuples():
        key = (z.track, z.geometry, z.system, z.target, z.signal, 0.2)
        if key in out and np.isfinite(z.ndg_min_leave_one_out):
            lo_, hi_ = out[key]
            if not (np.isclose(lo_, z.ndg_min_leave_one_out, atol=1e-9) and np.isclose(hi_, z.ndg_max_leave_one_out, atol=1e-9)):
                bad.append((key, lo_, hi_, z.ndg_min_leave_one_out, z.ndg_max_leave_one_out))
    if bad:
        raise SystemExit(f"leave-one-unit-out recomputation does not reproduce statistics_hardening.csv at 20%: {bad[:5]}")
    return out


def table_3b():
    """Cells where FULL beats CHEAP on precision AND recall yet the total decision value over affected inputs is < 0."""
    c = NEW.csv("calibration_cells.csv")
    hit = c[(c.full_precision > c.cheap_precision) & (c.full_recall > c.cheap_recall) & (c.all_full_reduction < 0)]
    rows = []
    if len(hit):
        tot = total_v_intervals(hit)
    for x in hit.itertuples():
        t = tot.get((x.cell, x.scheme, x.split), {})
        rows.append({"cell": x.cell, "scheme": x.scheme, "split": x.split, "units_scope": scope(x.split),
                     "cheap_precision": x.cheap_precision, "full_precision": x.full_precision,
                     "cheap_recall": x.cheap_recall, "full_recall": x.full_recall,
                     "total_V": t.get("total", np.nan), "total_V_lo": t.get("lo", np.nan), "total_V_hi": t.get("hi", np.nan),
                     "net_share_of_all_cheap": x.all_full_reduction, "harm_ratio": x.rho, "harm_ratio_lo": x.rho_lo,
                     "harm_ratio_hi": x.rho_hi, "n_affected": int(x.affected),
                     "total_V_interval_excludes_zero": bool(np.isfinite(t.get("hi", np.nan)) and t["hi"] < 0),
                     "reading": ("statistical: the interval of the total decision value lies below zero"
                                 if np.isfinite(t.get("hi", np.nan)) and t["hi"] < 0 else
                                 "counterexample by point estimate only: the interval includes zero")})
    return pd.DataFrame(rows)


def total_v_intervals(hit):
    """Sum of V over the cell's frames (= over its affected inputs), with a unit bootstrap (1000 draws, rng 0, per-unit
    sums reweighted -- 102's scheme), from the per-mode outcome tables 102 composed the cell from."""
    import importlib.util
    from rap import runs as rap_runs
    s = importlib.util.spec_from_file_location("t102", ROOT / "scripts" / "102_calibration_cells.py")
    t102 = importlib.util.module_from_spec(s)
    s.loader.exec_module(t102)
    S = t102.Store(rap_runs.latest("calibration_outcomes"), rap_runs.latest("calibration_plan"))
    thr = NEW.csv("calibration_thresholds.csv")
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    test = {"nuScenes": set(splits["nuscenes"]["test"]), "KITTI": set(splits["kitti"]["test"])}
    cells = {c[0]: c for c in t102.CELLS}
    rng = np.random.default_rng(0)
    out = {}
    for x in hit.itertuples():
        th = thr[(thr.cell == x.cell) & (thr.scheme == x.scheme)].iloc[0]
        d = t102.cell_frame(S, cells[x.cell], float(th.t_cheap), float(th.t_full))
        if x.split == "test":
            d = d[d.unit.isin(test[x.dataset])]
        v, units = d.V.to_numpy(float), d.unit.to_numpy()
        aff = np.abs(v) > t102.EPS
        assert int(aff.sum()) == int(x.affected), (x.cell, x.scheme, x.split, int(aff.sum()), x.affected)
        uniq, inv = np.unique(units, return_inverse=True)
        per = np.bincount(inv, v, len(uniq))
        W = np.stack([np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)) for _ in range(1000)])
        bt = W @ per
        out[(x.cell, x.scheme, x.split)] = {"total": float(v.sum()), "lo": float(np.percentile(bt, 2.5)),
                                            "hi": float(np.percentile(bt, 97.5))}
    return out


# ------------------------------------------------------------------------------------------------ sections 4 and 5

def section4():
    f = FINAL / "published_objective_routers.csv"
    if not f.exists():
        return None
    return {"routers": pd.read_csv(f, keep_default_na=False, na_values=[""], float_precision="round_trip"),
            "labels": pd.read_csv(FINAL / "published_objective_labels.csv") if (FINAL / "published_objective_labels.csv").exists() else None,
            "reading": json.loads((FINAL / "published_objective_reading.json").read_text())
            if (FINAL / "published_objective_reading.json").exists() else None}


def section5():
    f = FINAL / "submission_path_g1.csv"
    if not f.exists():
        return None
    return pd.read_csv(f)


# ------------------------------------------------------------------------------------------------ the document

def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, x in df.iterrows():
        cells = []
        for c in cols:
            v = x[c]
            if c in fmt:
                v = fmt[c](v)
            elif isinstance(v, (float, np.floating)):
                v = "n/a" if not np.isfinite(v) else f"{v:.3f}"
            cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_doc(claims, t3a, t3b, s4, s5, pack):
    n = claims.verdict.value_counts().to_dict()
    L = ["# The evidence pack and the claims check (Task 26)", "",
         "Generated by `scripts/152_evidence_pack.py` from the final artifacts; nothing was refit or re-run. "
         f"`results/final/paper_evidence_pack.csv` holds {len(pack)} quantities, each with its file, cell, split, "
         "population (`units_scope`), budget, interval method and denominator; `results/final/claims_check.csv` holds "
         "the verdicts below. Every claim was also evaluated on the camera-frame artifacts as first shipped "
         "(`results/raw/20260921_152927_ego_frame_before/results_final`); the note of each row says whether it held "
         "there.", "",
         f"**{n.get('PASS', 0)} of {len(claims)} statements pass, {n.get('FAIL', 0)} fail, "
         f"{n.get('UNVERIFIABLE', 0)} are unverifiable.**", "",
         "## Every FAIL, with the value on the final tables", ""]
    for x in claims[claims.verdict == "FAIL"].itertuples():
        L += [f"**{x.claim_id}.** {x.statement}", "", f"* now: {x.value_now}", f"* artifact: `{x.artifact}`",
              f"* note: {x.note}", ""]
    L += ["## Every UNVERIFIABLE, with the reason", ""]
    u = claims[claims.verdict == "UNVERIFIABLE"]
    L += ([f"* **{x.claim_id}.** {x.note}" for x in u.itertuples()] or ["None."]) + [""]
    L += ["## The statements that pass", ""]
    L += [f"* **{x.claim_id}.** {x.value_now} (`{x.artifact}`)" for x in claims[claims.verdict == "PASS"].itertuples()] + [""]
    L += ["## Where two artifacts disagree", ""]
    dis = claims[claims.note.str.contains("DISAGREEMENT")]
    L += ([f"* **{x.claim_id}.** {x.note}" for x in dis.itertuples()] or ["None found among the checked statements."])
    L += ["* nuPlan: `benchmark_table.csv`, `benchmark_table_routers.csv`, `benchmark_cells.csv` and the budget tables "
          "carry the historical transported detection-profile diagnostic under the same cell names as the official "
          "real-perception results (`benchmark_table_nuplan_real.csv`, `nuplan_real_perception_cells.csv`, "
          "`statistics_hardening.csv`). The pack labels every such row; a write-up must not quote the former as the "
          "nuPlan result.", ""]
    L += ["## 3A. Learnability under each geometry: every allocator variant, every core cell and quota", "",
          "nDG on the frozen test split; the paired interval of nDG minus random (official filter); the realised gain "
          "minus random in loss units with every bootstrap draw kept; the leave-one-test-unit-out nDG range "
          "(recomputed from the saved scores at every quota and checked against the shipped 20% values).", ""]
    if len(t3a):
        t = t3a.copy()
        t["quota"] = t.quota.map(lambda q: f"{q:.0%}")
        t["nDG − random [95%]"] = [f"{a:+.3f} {ci_(b, c, sign=True)}" for a, b, c in zip(t.minus_random, t.minus_random_lo, t.minus_random_hi)]
        t["gain − random, all draws"] = [f"{a:+.2f} {ci_(b, c, sign=True)}" for a, b, c in zip(t.gain_minus_random, t.gain_lo_all_draws, t.gain_hi_all_draws)]
        t["LOO range"] = [f"[{f_(a)}, {f_(b)}]" for a, b in zip(t.loo_ndg_min, t.loo_ndg_max)]
        L += [md_table(t, ["cell", "signal", "quota", "ndg", "nDG − random [95%]", "beats_random", "gain", "gain − random, all draws", "LOO range"]), ""]
    L += ["**Was the headline router variant registered before any test score?** The pre-registration of the routers "
          "(the pre-registration record (not part of this release), 2026-09-14, \"Task 2 pre-registration: lightweight routers\") fixes four detection-list "
          "variants and the pixel router and reports all of them; it does not name a headline variant. Its reading "
          "rule, verbatim: \"A router beats random in a cell at a quota if the paired 95% lower bound is above zero. "
          "[...] Reported against the 65-feature gates: the number of cells where any router beats random at 20% under "
          "a frame quota; the same count under the measured ms budget at 20%.\" and its targets: \"V (regression, "
          "score = prediction) and 1[V>0] (classification, score = P(V>0)). This gives four rows: R1_mlp_reg, "
          "R1_mlp_clf, R1_gbm_reg, R1_gbm_clf.\" Any single variant presented as the headline was therefore chosen "
          "after the test scores were seen; the table above reports all of them.", ""]
    L += ["## 3B. Detection quality against downstream value", "",
          "Cells (fidelity pair × system × geometry × operating-point scheme × population) where FULL has higher "
          "precision AND higher recall than CHEAP, yet the total decision value over the affected inputs is negative "
          "(uniform escalation raises the loss). The total's interval is a unit bootstrap (1000 draws) of the "
          "per-frame V the calibration table was built from.", ""]
    if len(t3b):
        t = t3b.copy()
        t["P cheap → full"] = [f"{a:.3f} → {b:.3f}" for a, b in zip(t.cheap_precision, t.full_precision)]
        t["R cheap → full"] = [f"{a:.3f} → {b:.3f}" for a, b in zip(t.cheap_recall, t.full_recall)]
        t["total V [95%]"] = [f"{a:+.2f} {ci_(b, c, sign=True)}" for a, b, c in zip(t.total_V, t.total_V_lo, t.total_V_hi)]
        t["harm ratio [95%]"] = [f"{a:.2f} {ci_(b, c)}" for a, b, c in zip(t.harm_ratio, t.harm_ratio_lo, t.harm_ratio_hi)]
        L += [md_table(t, ["cell", "scheme", "split", "P cheap → full", "R cheap → full", "total V [95%]",
                           "harm ratio [95%]", "n_affected", "total_V_interval_excludes_zero", "reading"]), ""]
    else:
        L += ["No such cell.", ""]
    L += ["## 4. The published routing objectives (Task 24)", ""]
    if s4 is None:
        L += ["**Pending**: Task 24 has not landed (`results/final/published_objective_routers.csv` does not exist). "
              "Nothing from it is quoted here, and no older number stands in for it.", ""]
    else:
        L += section4_md(s4)
    L += ["## 5. The submission path (Task 25)", ""]
    if s5 is None:
        L += ["**Pending**: Task 25 has not landed (`results/final/submission_path_g1.csv` does not exist).", ""]
    else:
        L += section5_md(s5)
    DOC.write_text("\n".join(L) + "\n")


def section4_md(s4):
    r = s4["routers"]
    L = ["Definitions as transcribed (full text in `docs/iclr_published_objectives.md`): **Q** = ORIC of Qiu et al. "
         "(SEC 2024), ORIC_i = (|E|+1)(mAPC_s − mAPC_w), per-class mean AP@0.5 (101-point) over the frame and |E| = 1000 "
         "context frames of its own set, trained as its ordinal rank (MORIC) with weighted MSE or as 1[ORIC>0]; **G** = "
         "ΔAP of Geng et al. (NAIC '26), the change of the set-wide pooled AP@0.5 when one frame is escalated, × N, "
         "trained as its ECDF (MORIC) or as OffloadBin = 1[ΔAP>0], plus the paper's budget-adaptive arbiter (Eq. 8).", "",
         "**What this is.** The published *objectives* on *our* architectures (R1 ×4, R2), with our features, our fit "
         "units and our scorer. The only piece of a published *method* reimplemented is Geng et al.'s budget-adaptive "
         "arbiter (their selection rule between a conditioned and a skipping estimator); neither paper's full pipeline "
         "(their detectors, datasets, features or estimators) is reproduced.", ""]
    if s4["labels"] is not None:
        L += ["Correlation of each label with the perception-gain labels already in use (fit units):", "",
              md_table(s4["labels"], list(s4["labels"].columns)[:10]), ""]
    L += [f"All {len(r)} rows are in `results/final/published_objective_routers.csv` and in the pack (`task24|...`). Per "
          "cell and quota, every architecture: nDG with the paired interval of nDG minus random, under the V-trained "
          "target (`*_reg`: V; `*_clf` and R2: 1[V>0]) and under Q and G, and the paired difference published minus "
          "V-trained. `*` marks an interval excluding zero.", ""]
    t = r.copy()
    t["arch"] = t.architecture + np.where(t.variant.astype(str).str.len() > 0, " (budget-adaptive)", "")
    fmt = lambda a, lo, hi: (f"{a:+.3f} [{lo:+.2f}, {hi:+.2f}]" + ("*" if np.isfinite(lo) and (lo > 0 or hi < 0) else "")
                             if np.isfinite(a) else "n/a")                    # noqa: E731
    lines = ["| cell | quota | architecture | V-trained nDG (− random) | Q nDG (− random) | Q − V | G nDG (− random) | G − V |",
             "|---|---|---|---|---|---|---|---|"]
    for (cell, q, arch), g in t.groupby([t.track + " " + t.geometry + " " + t.system + " " + t.target, "quota", "arch"], sort=False):
        row = {o: g[g.objective == o].iloc[0] if (g.objective == o).any() else None for o in ("V", "Q", "G")}
        v = t[(t.track + " " + t.geometry + " " + t.system + " " + t.target == cell) & (t.quota == q)
              & (t.architecture == ("R1_gbm_reg" if "adaptive" in arch else arch)) & (t.objective == "V")]
        v = v.iloc[0] if len(v) else None
        cellf = lambda z: (f"{z.ndg:.3f} ({fmt(z.minus_random, z.minus_random_lo, z.minus_random_hi)})" if z is not None else "")  # noqa: E731
        dif = lambda z: (fmt(z.minus_V, z.minus_V_lo, z.minus_V_hi) if z is not None else "")  # noqa: E731
        lines.append(f"| {cell} | {q:.0%} | {arch} | {cellf(v)} | {cellf(row['Q'])} | {dif(row['Q'])} | {cellf(row['G'])} | {dif(row['G'])} |")
    L += lines + [""]
    if s4["reading"] is not None:
        L += ["Registered reading: " + json.dumps(s4["reading"])[:2000], ""]
    return L


def section5_md(g1):
    bad = int((g1.differing > 0).sum())
    return ["What the release has (docs/SUBMITTING.md, `evaluate_submission.py`, `rap.submission`): a CSV with one row "
            "per test input of each cell entered; identifiers per track (KITTI and nuScenes `seq, frame`; nuPlan "
            "`scenario, iteration`); either one `score` column (a ranking, every quota) or `score_q10 … score_q50` "
            "(budget-conditioned; supported, and the scored file records the form). Validation rejects unknown columns, "
            "non-finite scores, duplicates, and any row that is not exactly a test input of the cell, naming the rows.",
            "",
            "Legal pre-escalation inputs: the five provenance labels of `rap.features.LEGAL_SOURCES` (CHEAP detections, "
            "CHEAP image, previous CHEAP frames, calibration, ego state), exported labels-free in "
            "`data/submission_inputs/`; the checker enforces the file's form, not the provenance of the scores "
            "(`rap.features.check_provenance` checks declarations; the rule is honour-based, as SUBMITTING.md says).", "",
            "Tracks: selection budget (quotas 10/20/30/50 %, any CPU); measured budget (ms and mJ, needs a cost profile "
            "from the provided harness on a device passing `environment/check_device.py`, or code profiled by us; the "
            "feasibility rule of `rap.budget.infeasible`).", "",
            f"G1: {len(g1)} signal × field comparisons against the official tables (`benchmark_table.csv`, "
            f"`benchmark_table_routers.csv`, and `benchmark_budget_two_level.csv` for the measured track), {bad} with any "
            "difference, no tolerance (`results/final/submission_path_g1.csv`). On its first run G1 found the shipped "
            "nuScenes R2 rows of `benchmark_table_routers.csv` a few ulps from their own computation (a lossy CSV re-read "
            "inside 107); they were regenerated from the saved scores, and no printed number changed "
            "(`docs/iclr_submission_path.md`).", ""]


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_3a_loo", action="store_true", help="skip the leave-one-unit-out recomputation")
    args = ap.parse_args()
    run = runmeta.new_run("evidence_pack", vars(args))
    claims = run_claims()
    P = Pack()
    pack_harm(P)
    pack_ndg(P)
    pack_budgets(P)
    pack_misc(P)
    pack_task24(P)
    t3a = table_3a() if not args.no_3a_loo else pd.DataFrame()
    t3b = table_3b()
    for x in t3a.itertuples():
        P.add(f"3A|{x.cell}|{x.signal}|{x.quota:g}", description="3A: leave-one-test-unit-out nDG range",
              artifact_file="saved per-frame scores and V (runs routers_r1_ego, router_r2_ego, budget_gate_scores_ego)",
              track=x.cell.split()[0], geometry=x.cell.split()[1], system=x.cell.split()[2],
              target=tgt(x.cell.split()[2], x.cell.split()[3]), split="test", units_scope=scope("test"),
              quota_or_budget=f"quota {x.quota:.0%}", signal_or_policy=x.signal, estimate=x.ndg, ci_lo=x.loo_ndg_min,
              ci_hi=x.loo_ndg_max, interval_method="range of nDG over leave-one-test-unit-out subsets (not a CI)",
              denominator="oracle prize at the same quota", notes=f"source {x.source}")
    for x in t3b.itertuples():
        P.add(f"3B|{x.cell}|{x.scheme}|{x.split}", description="3B: total decision value over affected inputs, FULL "
              "better on precision and recall", artifact_file="calibration_cells.csv + per-frame calibration outcomes",
              track=x.cell.split()[0], geometry=x.cell.split()[1], system=x.cell.split()[-1], target=x.cell.split()[-1],
              split=x.split, units_scope=x.units_scope, signal_or_policy=f"uniform escalation, scheme {x.scheme}",
              estimate=x.total_V, ci_lo=x.total_V_lo, ci_hi=x.total_V_hi,
              interval_method="unit bootstrap of the per-unit sums of V, 1000 draws, default_rng(0)",
              n_affected=x.n_affected, denominator="none (loss units)", notes=x.reading)
    pack = P.frame()
    pack.to_csv(FINAL / "paper_evidence_pack.csv", index=False)
    claims.to_csv(FINAL / "claims_check.csv", index=False)
    for f in ("paper_evidence_pack.csv", "claims_check.csv"):
        (run / f).write_bytes((FINAL / f).read_bytes())
    write_doc(claims, t3a, t3b, section4(), section5(), pack)
    print(claims.verdict.value_counts().to_string())
    print(f"wrote {len(pack)} pack rows, {len(claims)} claims, 3A {len(t3a)} rows, 3B {len(t3b)} rows, {DOC}")


if __name__ == "__main__":
    main()
