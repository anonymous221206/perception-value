#!/usr/bin/env python
"""Task 29: four analyses on cached scores (pre-registered in docs/iclr_cached_analyses_prereg.md, committed first).

  --part a   selection by objective: descriptive test-set disagreement of the optimum sets (P1, P2, 127's pool), and
             selection on validation / evaluation on test with V1 scores     -> cached_analyses_selection.csv
  --part b   benefit captured and harm incurred, with the gap identity        -> cached_analyses_benefit_harm.csv
  --part c   the overheads a ranking can afford under the registry's budgets  -> cached_analyses_overhead_tolerance.csv
  --part d   the gap between the zero-cost oracle and a causal allocator      -> cached_analyses_gap_accounting.csv
  --part fig the three-panel figure data, from the four files                 -> cached_analyses_figure_data.csv

Nothing is refit. Scores: the official test scores of the shipped runs, and the V1 scores (train-only fits) of
`*_streaming_v1_scores`. The bootstrap resamples units once per dataset per draw, shared by every cell of the
dataset, every draw kept. New files only; no existing file of results/final is written.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta, scoring, submission as S                               # noqa: E402
from rap.budget import infeasible                                               # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

FINAL = Path(RESULTS) / "final"
NBOOT = 1000
EPS, QUOTAS = scoring.EPS, scoring.QUOTAS
MIN_AFFECTED = S.MIN_AFFECTED
DATASETS = ("nuScenes", "KITTI", "nuPlan")
P1 = ["random", "uncertainty", "criticality_cheap", "trivial_ego_speed"]
LEARNED = ["gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf"]
P2 = P1 + LEARNED
DIAG = ["dE_exact", "dE_E1_fn_only", "dE_E6_risk_weighted", "PKL", "TIP"]
OLD = P2 + ["R2_cnn_clf"] + DIAG
OBJECTIVES = {"E_perc_dE": "dE_E1_fn_only", "E_perc_risk": "dE_E6_risk_weighted"}
DEPLOYABLE_B = ["uncertainty", "criticality_cheap", "trivial_ego_speed", *LEARNED, "R2_cnn_clf"]
CHARGED = ["uncertainty", "criticality_cheap", "gate_ridge", "gate_gbm", "gate_gbm_batched",
           "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "R2_cnn_clf"]
ALPHAS = (0.10, 0.20, 0.30, 0.50)
D_ALPHAS = (0.20, 0.50)
OVERHEADS = "routers"
TOL = 1e-9


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ cells

def _col(d, name):
    return pd.to_numeric(d[name], errors="coerce").to_numpy(float)


def _aligned(z, keys, names):
    ida, idb = ("scenario", "iteration") if "scenario" in z.files else ("seq", "frame")
    src = pd.DataFrame({ida: z[ida].astype(str), idb: z[idb].astype(int)})
    for n in names:
        if n in z.files:
            src[n] = np.asarray(z[n], float)
    m = keys.merge(src, on=[ida, idb], how="left", validate="one_to_one")
    return {n: m[n].to_numpy(float) for n in names if n in m.columns}


def build_cells(t92, t120, t130):
    """The 14 cells with official scores (all rows where cached; NaN elsewhere), V1 scores and stream order."""
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    r1, r2 = rap_runs.latest("routers_r1"), rap_runs.latest("router_r2")
    gates, nr = rap_runs.latest("budget_gate_scores"), rap_runs.latest("nuplan_real_allocation")
    v1run = rap_runs.latest("streaming_v1_scores")
    meta = pd.read_csv(ROOT / "configs" / "benchmark_nuplan_scenarios.csv")[["scenario", "t0"]]
    cells = []
    t120.register_features()
    gens = [("core", g) for g in (t92.nuscenes_cells, t92.kitti_cells)] + [("nuplan", t120.cells)]
    for kind, gen in gens:
        for c in gen(splits):
            d = c["d"].reset_index(drop=True)
            v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            key = (c["track"], c["geometry"], c["system"], c["target"])
            ck = f"{c['track']}__{'na' if c['geometry'] == 'n/a' else c['geometry']}__{c['system']}__{c['target']}"
            sc = {}
            if kind == "core":
                keys = d[["seq", "frame"]].astype({"seq": str, "frame": int})
                stem = f"scores__{c['track']}__{c['geometry']}__{c['system']}__{c['target']}.npz"
                z = np.load(r1 / stem, allow_pickle=False)
                assert np.array_equal(z["V"], v), stem
                sc.update(_aligned(z, keys, LEARNED[2:]))
                if (r2 / stem).exists():
                    sc.update(_aligned(np.load(r2 / stem, allow_pickle=False), keys, ["R2_cnn_clf"]))
                zg = np.load(gates / f"gates__{ck}.npz", allow_pickle=False)
                assert np.array_equal(zg["V"], v) and np.array_equal(zg["key_seq"], d.seq.astype(str).to_numpy().astype(str))
                sc.update({g: np.asarray(zg[g], float) for g in ("gate_ridge", "gate_gbm")})
                cols = dict(c["cols"])
                for s in ("uncertainty", "criticality_cheap", "dE_exact", "dE_E1_fn_only", "dE_E6_risk_weighted",
                          "PKL", "TIP"):
                    if cols.get(s) in d.columns:
                        sc[s] = _col(d, cols[s])
                sc["trivial_ego_speed"] = _col(d, "v_ego_causal")
                order = np.lexsort((d.frame.to_numpy(int), d.seq.astype(str).to_numpy()))
            else:
                keys = d[["scenario", "iteration"]].astype({"scenario": str, "iteration": int})
                z = np.load(nr / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
                assert np.array_equal(z["V"], v)
                sc.update(_aligned(z, keys, LEARNED))
                sc["criticality_cheap"] = _col(d, "nr_crit_cheap_sum")
                sc["trivial_ego_speed"] = _col(d, "nr_ego_speed")
                for s in ("dE_E1_fn_only", "dE_E6_risk_weighted"):
                    sc[s] = _col(d, s)
                t0 = d[["scenario"]].merge(meta, on="scenario", how="left", validate="many_to_one").t0.to_numpy(float)
                order = np.lexsort((d.iteration.to_numpy(int), t0))
            zv = np.load(v1run / f"v1__{ck}.npz", allow_pickle=False)
            for k, arr in t130._keys(d).items():
                assert np.array_equal(zv[f"key_{k}"], arr), (ck, k)
            assert np.array_equal(zv["V"], v)
            split = d.split.to_numpy()
            cells.append(dict(key=key, ck=ck, track=c["track"], geometry=c["geometry"], system=c["system"],
                              target=c["target"], dataset=c["track"], d=d, v_all=v, official=sc,
                              v1={a: np.asarray(zv[a], float) for a in LEARNED}, order=order,
                              units_all=d.unit.astype(str).to_numpy(), jc=d[c["cheap"]].to_numpy(float),
                              tr=split == "train", va=split == "val", te=split == "test"))
    for c in cells:
        te = c["te"]
        c["v"] = c["v_all"][te]
        c["n"] = int(te.sum())
        c["units_test"] = c["units_all"][te]
        c["uniq"] = np.array(sorted(set(c["units_test"])))
        c["idx"] = [np.flatnonzero(c["units_test"] == u) for u in c["uniq"]]
        c["upos"] = {u: i for i, u in enumerate(c["uniq"])}
    return cells, {"routers_r1": r1.name, "router_r2": r2.name, "gates": gates.name, "nuplan_real": nr.name,
                   "v1": v1run.name}


class Draws:
    """Units resampled once per dataset per draw, shared by the cells of that dataset; every draw kept."""

    def __init__(self, cells, nboot=NBOOT, seed=0):
        rng = np.random.default_rng(seed)
        self.units = {ds: sorted(set().union(*[set(c["uniq"]) for c in cells if c["dataset"] == ds]) or [])
                      for ds in DATASETS}
        self.pick = {ds: np.zeros((nboot, len(self.units[ds])), int) for ds in DATASETS}
        for b in range(nboot):
            for ds in DATASETS:
                if self.units[ds]:
                    self.pick[ds][b] = rng.integers(0, len(self.units[ds]), len(self.units[ds]))
        self.nboot = nboot

    def rows(self, c, b):
        """(test row indices, per-unit counts) of cell c in draw b."""
        us = [self.units[c["dataset"]][i] for i in self.pick[c["dataset"]][b]]
        pos = [c["upos"][u] for u in us if u in c["upos"]]
        counts = np.bincount(pos, minlength=len(c["uniq"])).astype(float)
        t = np.concatenate([c["idx"][p] for p in pos]) if pos else np.zeros(0, int)
        return t, counts


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def gains(score, values, ks):
    """topk_expect for several value vectors; score None = random (every input tied, the closed form)."""
    ks = np.asarray(ks, int)
    n = len(values[0])
    if score is None:
        return [ks / n * np.asarray(w, float).sum() for w in values]
    out = scoring.topk_expect(np.nan_to_num(score, nan=-np.inf), values, np.maximum(ks, 1))[0]
    return [np.where(ks > 0, o, 0.0) for o in out]


def defined(values, k):
    """The benchmark's rule for one value vector at capacities k: enough affected inputs, a positive prize."""
    n_aff = int((np.abs(values) > EPS).sum())
    prize = gains(values, [values], k)[0]
    return (n_aff >= MIN_AFFECTED) & (prize > EPS), n_aff, prize


def base(c):
    return dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"])


# ------------------------------------------------------------------------------------------------ A

def part_a(cells):
    rows = []
    pools = {"P1": P1, "P2": P2, "old": OLD}
    # A2: descriptive test-set disagreement, official test scores
    for c in cells:
        te, v = c["te"], c["v"]
        ks = scoring.quota_k(c["n"])
        ok_dec, _, _ = defined(v, ks)
        vals = {"E_dec": v, **{o: c["official"][col][te] for o, col in OBJECTIVES.items()}}
        eta = {}
        for o, val in vals.items():
            prize = gains(val, [val], ks)[0]
            for s in OLD:
                x = None if s == "random" else c["official"].get(s)
                if s != "random" and (x is None or not np.isfinite(x[te]).any()):
                    continue
                g = gains(None if s == "random" else x[te], [val], ks)[0]
                eta[(o, s)] = np.where(prize > EPS, g / np.where(prize > EPS, prize, 1), np.nan)
        for o in OBJECTIVES:
            for pname, pool in pools.items():
                for qi, q in enumerate(QUOTAS):
                    have = [s for s in pool if (("E_dec", s) in eta and (o, s) in eta
                                                and np.isfinite(eta[("E_dec", s)][qi]) and np.isfinite(eta[(o, s)][qi]))]
                    if not have:
                        continue
                    dec = np.array([eta[("E_dec", s)][qi] for s in have])
                    per = np.array([eta[(o, s)][qi] for s in have])
                    opt_d = [s for s, x in zip(have, dec) if x >= dec.max() - TOL]
                    opt_p = [s for s, x in zip(have, per) if x >= per.max() - TOL]
                    tau = kendalltau(dec, per).correlation if len(have) > 1 else np.nan
                    rows.append({"section": "test_disagreement", **base(c), "objective": o, "pool": pname,
                                 "quota": q, "e_dec_defined_test": bool(ok_dec[qi]), "n_signals": len(have),
                                 "optimal_dec": "|".join(opt_d), "optimal_perc": "|".join(opt_p),
                                 "no_shared_optimum": not set(opt_d) & set(opt_p),
                                 "argmax_dec": have[int(np.argmax(dec))], "argmax_perc": have[int(np.argmax(per))],
                                 "kendall_tau": tau})
    # A3: selection on validation, evaluation on test (V1 scores for learned signals)
    draws = Draws(cells)
    sel = []
    for ci_, c in enumerate(cells):
        va, te = c["va"], c["te"]
        vv, vt = c["v_all"][va], c["v"]
        ks_va, ks_te = scoring.quota_k(int(va.sum())), scoring.quota_k(c["n"])
        ok_dec_va, aff_dec_va, _ = defined(vv, ks_va)
        ok_dec_te, _, _ = defined(vt, ks_te)

        def score(s, mask):
            if s == "random":
                return None
            x = c["v1"][s] if s in LEARNED else c["official"].get(s)
            if x is None or not np.isfinite(x[mask]).any():
                return "missing"
            return x[mask]
        for o, col in OBJECTIVES.items():
            pv = c["official"][col][va]
            ok_p_va, aff_p_va, _ = defined(pv, ks_va)
            for pname, pool in (("P1", P1), ("P2", P2)):
                have = [s for s in pool if not isinstance(score(s, va), str)]
                for qi, q in enumerate(QUOTAS):
                    r = {"section": "validation_selection", **base(c), "objective": o, "pool": pname, "quota": q,
                         "n_val": int(va.sum()), "n_affected_val_dec": aff_dec_va, "n_affected_val_perc": aff_p_va}
                    why = ("decision value undefined on the validation units" if not ok_dec_va[qi] else
                           "perception gain undefined on the validation units" if not ok_p_va[qi] else
                           "decision value undefined on the test units" if not ok_dec_te[qi] else "")
                    if why:
                        rows.append({**r, "included": False, "drop_reason": why})
                        continue
                    k = ks_va[qi:qi + 1]
                    val_dec = {s: gains(score(s, va), [vv], k)[0][0] for s in have}
                    val_per = {s: gains(score(s, va), [pv], k)[0][0] for s in have}
                    w_dec = next(s for s in have if val_dec[s] >= max(val_dec.values()) - TOL)
                    w_per = next(s for s in have if val_per[s] >= max(val_per.values()) - TOL)
                    kt = ks_te[qi:qi + 1]
                    g_dec = gains(score(w_dec, te), [vt], kt)[0][0]
                    g_per = gains(score(w_per, te), [vt], kt)[0][0]
                    tot = c["jc"][te].sum()
                    sel.append(dict(ci=ci_, o=o, pool=pname, qi=qi, w_dec=w_dec, w_per=w_per, row=len(rows)))
                    rows.append({**r, "included": True, "drop_reason": "", "selected_dec": w_dec,
                                 "selected_perc": w_per, "same_winner": w_dec == w_per,
                                 "gain_test_dec_selected": g_dec, "gain_test_perc_selected": g_per,
                                 "diff_gain": g_per - g_dec, "all_cheap_loss": tot, "diff_share": (g_per - g_dec) / tot})
    # the draws: one pass, every selection of every cell
    per_draw = {i: np.full(NBOOT, np.nan) for i in range(len(sel))}
    for b in range(NBOOT):
        rowsets = {}
        for i, s_ in enumerate(sel):
            c = cells[s_["ci"]]
            if s_["ci"] not in rowsets:
                rowsets[s_["ci"]] = draws.rows(c, b)[0]
            t = rowsets[s_["ci"]]
            if not len(t):
                continue
            vt, te = c["v"][t], c["te"]
            kt = scoring.quota_k(len(t))[s_["qi"]:s_["qi"] + 1]

            def sc(s):
                if s == "random":
                    return None
                x = c["v1"][s] if s in LEARNED else c["official"][s]
                return x[te][t]
            if s_["w_dec"] == s_["w_per"]:
                per_draw[i][b] = 0.0
                continue
            d_ = gains(sc(s_["w_per"]), [vt], kt)[0][0] - gains(sc(s_["w_dec"]), [vt], kt)[0][0]
            per_draw[i][b] = d_ / c["jc"][te][t].sum()
    for i, s_ in enumerate(sel):
        lo, hi = ci(per_draw[i])
        tot = rows[s_["row"]]["all_cheap_loss"]
        rows[s_["row"]].update(diff_share_lo=lo, diff_share_hi=hi, diff_gain_lo=lo * tot, diff_gain_hi=hi * tot,
                               note_gain_interval="share interval x the test all-cheap loss")
    # pooled over cells: mean share-scale D, per pool, objective and quota
    for o in OBJECTIVES:
        for pname in ("P1", "P2"):
            for qi, q in enumerate(QUOTAS):
                idx = [i for i, s_ in enumerate(sel) if s_["o"] == o and s_["pool"] == pname and s_["qi"] == qi]
                if not idx:
                    continue
                point = float(np.mean([rows[sel[i]["row"]]["diff_share"] for i in idx]))
                mat = np.vstack([per_draw[i] for i in idx])
                lo, hi = ci(np.nanmean(mat, axis=0))
                rows.append({"section": "validation_selection_pooled", "track": "pooled", "objective": o, "pool": pname,
                             "quota": q, "n_cells": len(idx), "diff_share": point, "diff_share_lo": lo,
                             "diff_share_hi": hi, "n_same_winner": int(sum(rows[sel[i]["row"]]["same_winner"] for i in idx)),
                             "cells": "; ".join(f"{cells[sel[i]['ci']]['ck']}" for i in idx)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ B

def part_b(cells):
    draws = Draws(cells)
    rows, stash = [], []
    for ci_, c in enumerate(cells):
        te, v = c["te"], c["v"]
        ks = scoring.quota_k(c["n"])
        sigs = ["random"] + [s for s in DEPLOYABLE_B if s in c["official"] and np.isfinite(c["official"][s][te]).any()]
        for s in sigs:
            stash.append((ci_, s))
    point, boot = {}, {k: np.full((NBOOT, len(QUOTAS), 6), np.nan) for k in stash}

    def quantities(score, v, jc, ks):
        vp, vn = np.maximum(v, 0.0), np.maximum(-v, 0.0)
        B, H, G = gains(score, [vp, vn, v], ks)
        O = gains(v, [v], ks)[0]
        Ball = vp.sum()
        lhs = O - G
        rhs = (Ball - B) + H - (Ball - O)
        assert np.all(np.abs(lhs - rhs) <= TOL), (lhs, rhs)
        tot = jc.sum()
        return np.stack([B, H, Ball - B, Ball - O, G, O], axis=1) / tot
    for ci_, s in stash:
        c = cells[ci_]
        sc = None if s == "random" else np.nan_to_num(c["official"][s][c["te"]], nan=-np.inf)
        point[(ci_, s)] = quantities(sc, c["v"], c["jc"][c["te"]], scoring.quota_k(c["n"]))
    for b in range(NBOOT):
        rs = {}
        for ci_, s in stash:
            c = cells[ci_]
            if ci_ not in rs:
                rs[ci_] = draws.rows(c, b)[0]
            t = rs[ci_]
            sc = None if s == "random" else np.nan_to_num(c["official"][s][c["te"]], nan=-np.inf)[t]
            boot[(ci_, s)][b] = quantities(sc, c["v"][t], c["jc"][c["te"]][t], scoring.quota_k(len(t)))
    names = ["benefit", "harm", "missed_benefit", "budget_forced", "gain", "oracle_gain"]
    for ci_, s in stash:
        c = cells[ci_]
        for qi, q in enumerate(QUOTAS):
            r = {**base(c), "signal": s, "quota": q, "k": int(scoring.quota_k(c["n"])[qi]), "n_frames": c["n"],
                 "all_cheap_loss": float(c["jc"][c["te"]].sum()), "identity_checked": True}
            for j, nm in enumerate(names):
                r[f"{nm}_share"] = float(point[(ci_, s)][qi, j])
                r[f"{nm}_share_lo"], r[f"{nm}_share_hi"] = ci(boot[(ci_, s)][:, qi, j])
                if s != "random" and nm in ("benefit", "harm", "missed_benefit", "gain"):
                    dr = boot[(ci_, s)][:, qi, j] - boot[(ci_, "random")][:, qi, j]
                    r[f"{nm}_minus_random"] = float(point[(ci_, s)][qi, j] - point[(ci_, "random")][qi, j])
                    r[f"{nm}_minus_random_lo"], r[f"{nm}_minus_random_hi"] = ci(dr)
            rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ C

def capacity_intervals(budget, c0, c1, n, o_max):
    """[(k, o_lo, o_hi)]: the overheads in [0, o_max] that the scorer's rule turns into capacity k, checked with it."""
    out = []
    k0 = S.measured_k(S.measured_share(budget, c0, c1, 0.0), n)
    for k in range(0, k0 + 1):
        # k(o) >= k  <=>  o <= (b - C_c) - C_f (k - 1e-9) / n  (floor(share n + 1e-9) >= k)
        top = min((budget - c0) - c1 * (k - 1e-9) / n, o_max) if k > 0 else o_max
        bot = max((budget - c0) - c1 * (k + 1 - 1e-9) / n, 0.0) if k < k0 else 0.0
        if top < bot:
            continue
        mid = (top + bot) / 2
        assert S.measured_k(S.measured_share(budget, c0, c1, mid), n) == k, (k, bot, top)
        out.append((k, bot, top))
    return sorted(out, key=lambda x: x[1])


def merge(ivs):
    ivs = sorted(ivs)
    out = []
    for lo, hi in ivs:
        if out and lo <= out[-1][1] + 1e-12:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def fmt(ivs):
    return "; ".join(f"[{lo:.6g}, {hi:.6g}]" for lo, hi in ivs) if ivs else "none"


def part_c(cells, run):
    reg = S.registry()
    ovr = reg["allocators"][OVERHEADS]["tracks"]
    rows, curves = [], []
    for c in cells:
        te, v, n = c["te"], c["v"], c["n"]
        det = S.detector_costs(c["track"])
        allk = np.arange(n + 1)
        for s in CHARGED:
            src = "gate_gbm" if s == "gate_gbm_batched" else s
            if src not in c["official"] or s not in ovr[c["track"]]:
                continue
            x = c["official"][src][te]
            if not np.isfinite(x).any():
                continue
            g_all = gains(np.nan_to_num(x, nan=-np.inf), [v], allk)[0]
            for unit in ("ms", "mJ"):
                c0, c1 = det["cheap"][unit], det["640"][unit]
                o_meas = ovr[c["track"]][s][unit]
                for a in ALPHAS:
                    b = c0 + a * c1
                    o_max = b - c0
                    k0 = S.measured_k(S.measured_share(b, c0, c1, 0.0), n)
                    rnd = k0 / n * v.sum()
                    ff_feasible = c1 <= b
                    ivs = capacity_intervals(b, c0, c1, n, o_max)
                    beat_r = merge([(lo, hi) for k, lo, hi in ivs if g_all[k] >= rnd - TOL])
                    beat_f = merge([(lo, hi) for k, lo, hi in ivs if g_all[k] >= v.sum() - TOL]) if ff_feasible else None
                    feas_meas = not bool(infeasible(b, c0, o_meas))
                    k_meas = S.measured_k(S.measured_share(b, c0, c1, o_meas), n)
                    inside = lambda iv: any(lo - 1e-12 <= o_meas <= hi + 1e-12 for lo, hi in iv)   # noqa: E731
                    rows.append({**base(c), "signal": s, "unit": unit, "budget_level": a, "budget_per_frame": b,
                                 "cheap_cost": c0, "full_cost": c1, "headroom": o_max, "n_frames": n, "k_zero_overhead": k0,
                                 "gain_random": rnd, "gain_zero_overhead": float(g_all[k0]),
                                 "overheads_at_least_random": fmt(beat_r),
                                 "full_fidelity_feasible": ff_feasible, "gain_full_fidelity": float(v.sum()),
                                 "overheads_at_least_full_fidelity": fmt(beat_f) if ff_feasible else "not feasible (C_f > b)",
                                 "max_overhead_at_least_random": max((hi for _, hi in beat_r), default=np.nan),
                                 "measured_overhead": o_meas, "measured_feasible": feas_meas,
                                 "measured_capacity": k_meas if feas_meas else np.nan,
                                 "measured_gain": float(g_all[k_meas]) if feas_meas else np.nan,
                                 "measured_in_at_least_random": inside(beat_r) if feas_meas else False,
                                 "measured_in_at_least_full_fidelity": (inside(beat_f) if (ff_feasible and feas_meas)
                                                                        else np.nan),
                                 "overhead_provenance": ovr[c["track"]][s]["provenance"],
                                 "all_cheap_loss": float(c["jc"][te].sum())})
                    if c["track"] != "nuPlan" and a == 0.20 and unit == "ms":
                        for k, lo, hi in ivs:
                            curves.append({**base(c), "signal": s, "o_lo": lo, "o_hi": hi, "capacity": k,
                                           "gain_share": float(g_all[k] / c["jc"][te].sum()),
                                           "random_share": rnd / c["jc"][te].sum(), "measured_overhead": o_meas})
        print(f"  C {c['ck']}", flush=True)
    df = pd.DataFrame(rows)
    # the paired interval at the measured overhead: rap.submission.budget_cell must equal the shipped rows
    df = df.merge(check_measured(cells), on=["track", "geometry", "system", "target", "signal", "unit", "budget_level"],
                  how="left", validate="one_to_one")
    pd.DataFrame(curves).to_csv(run / "overhead_curves_core_20.csv", index=False)
    return df, pd.DataFrame(curves)


def check_measured(cells):
    reg = S.registry()
    rd = lambda n: pd.read_csv(FINAL / n, float_precision="round_trip", keep_default_na=False, na_values=[""],  # noqa: E731
                               low_memory=False)                                      # lossless: compared bit for bit
    core = lambda df: df[df.track != "nuPlan"]            # noqa: E731  their nuPlan rows are the transported track
    ship = {"ms": pd.concat([core(rd("benchmark_budget_routers.csv")), rd("benchmark_budget_nuplan_real.csv")],
                            ignore_index=True),
            "mJ": pd.concat([core(rd("energy_module_budget_two_level.csv").query("table == 'routers'")),
                             rd("energy_module_budget_nuplan_real.csv")], ignore_index=True)}
    ship["mJ"] = ship["mJ"][ship["mJ"].convention == reg["energy_convention"]["name"]]
    fields = ("feasible", "overhead", "escalated_frac", "eta", "eta_lo", "eta_hi", "minus_random", "minus_random_lo",
              "minus_random_hi")
    out = []
    ovr = reg["allocators"][OVERHEADS]["tracks"]
    for c in cells:
        rows = S.rows(c["key"], "test")
        idn = list(S.IDS[c["track"]])
        rows[idn[1]] = rows[idn[1]].astype(int)
        d = c["d"][c["te"]].reset_index(drop=True)
        ids = d[idn].astype({idn[0]: str, idn[1]: int})
        for s in CHARGED:
            src = "gate_gbm" if s == "gate_gbm_batched" else s
            if src not in c["official"] or s not in ovr[c["track"]]:
                continue
            x = c["official"][src][c["te"]]
            if not np.isfinite(x).any():
                continue
            g = ids.assign(score=np.nan_to_num(x, nan=-1e300))
            prof = {"ms": {c["track"]: ovr[c["track"]][s]["ms"]}, "mJ": {c["track"]: ovr[c["track"]][s]["mJ"]},
                    "rails": reg["energy_convention"]["rails"]}
            sc = pd.DataFrame(S.budget_cell(c["key"], g, "ranking", prof)).rename(
                columns={"ndg": "eta", "ndg_lo": "eta_lo", "ndg_hi": "eta_hi"})
            for unit in ("ms", "mJ"):
                sh = ship[unit]
                sh = sh[(sh.track == c["track"]) & (sh.geometry.replace("", "n/a").astype(str) == c["geometry"])
                        & (sh.system == c["system"]) & (sh.target == c["target"]) & (sh.signal == s) & (sh.unit == unit)]
                for a in ALPHAS:
                    mine = sc[(sc.unit == unit) & np.isclose(sc.budget_level, a)]
                    off = sh[np.isclose(sh.budget_level, a)]
                    assert len(mine) == 1 and len(off) == 1, (c["ck"], s, unit, a, len(mine), len(off))
                    m, o = mine.iloc[0], off.iloc[0]
                    for f in fields:
                        x1, x2 = float(m.get(f, np.nan)), float(o[f])
                        if not ((x1 == x2) or (np.isnan(x1) and np.isnan(x2))):
                            raise SystemExit(f"C: the measured-overhead row differs from the shipped table: {c['ck']} "
                                             f"{s} {unit} {a} {f}: {x1!r} vs {x2!r}")
                    out.append({**base(c), "signal": s, "unit": unit, "budget_level": a,
                                "measured_ndg": float(m.eta), "measured_minus_random": float(m.get("minus_random", np.nan)),
                                "measured_minus_random_lo": float(m.get("minus_random_lo", np.nan)),
                                "measured_minus_random_hi": float(m.get("minus_random_hi", np.nan)),
                                "measured_matches_shipped": True})
        print(f"  C check {c['ck']}", flush=True)
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ D

def part_d(cells, t130):
    reg = S.registry()
    ovr = reg["allocators"][OVERHEADS]["tracks"]
    use = [c for c in cells if c["track"] != "nuPlan" or c["system"] == "pdm_closed"]
    draws = Draws(use)
    items, rows = [], []
    for ci_, c in enumerate(use):
        te, va, v, n = c["te"], c["va"], c["v"], c["n"]
        det = S.detector_costs(c["track"])
        c0, c1 = det["cheap"]["ms"], det["640"]["ms"]
        idx_units = t130.streams(c, te)
        # streams() orders units by name, as c["uniq"]
        assert all(np.array_equal(np.sort(a), np.sort(c["idx"][i])) for i, a in enumerate(idx_units))
        tot = c["jc"][te].sum()
        for s in LEARNED:
            o = ovr[c["track"]][s]["ms"]
            s_te = t130.prep(c["v1"][s][te])
            for a in D_ALPHAS:
                b = c0 + a * c1
                k0 = S.measured_k(S.measured_share(b, c0, c1, 0.0), n)
                feas = not bool(infeasible(b, c0, o))
                f_o = S.measured_share(b, c0, c1, o)
                k_o = S.measured_k(f_o, n)
                O0 = gains(v, [v], [k0])[0][0]
                r = {**base(c), "signal": s, "budget_level": a, "unit": "ms", "budget_per_frame": b, "overhead": o,
                     "overhead_provenance": ovr[c["track"]][s]["provenance"], "n_frames": n, "k_zero_overhead": k0,
                     "feasible": feas, "all_cheap_loss": tot, "O_0_share": O0 / tot}
                # the official scorer's top-k gain of the same V1 scores at zero overhead
                ids = c["d"][te][list(S.IDS[c["track"]])].reset_index(drop=True)
                ids = ids.astype({S.IDS[c["track"]][0]: str, S.IDS[c["track"]][1]: int})
                bc = S.budget_cell(c["key"], ids.assign(score=np.nan_to_num(c["v1"][s][te], nan=-1e300)), "ranking",
                                   {"ms": 0.0, "mJ": None}, nboot=1)
                g_off = [x for x in bc if x["unit"] == "ms" and np.isclose(x["budget_level"], a)][0]["gain"]
                g0 = gains(s_te, [v], [k0])[0][0]
                assert g0 == g_off, (c["ck"], s, a, g0, g_off)
                r["G_rank_zero_overhead_equals_scorer"] = True
                if not feas:
                    r.update(note=f"infeasible: overhead {o:.4f} ms exceeds the headroom {b - c0:.4f} ms")
                    rows.append(r)
                    continue
                cal = t130.Calib(c["v1"][s][va])
                tau, p = cal(f_o)
                ntie = t130.NTIE if t130.has_ties(s_te, c["v1"][s][va]) else 1
                keys = [np.random.default_rng(t130.stable_seed("tie", c["ck"], s, f_o, rr)).random(n) for rr in range(ntie)]
                ug, rate, _ = t130.policy_units("B", s_te, keys, f_o, cal, tau, p, idx_units, v, None)
                Oo = gains(v, [v], [k_o])[0][0]
                Gr = gains(s_te, [v], [k_o])[0][0]
                Gs = float(ug.sum())
                terms = np.array([O0 - Oo, Oo - Gr, Gr - Gs])
                assert abs(terms.sum() - (O0 - Gs)) <= TOL
                r.update(share_after_overhead=f_o, k_after_overhead=k_o, stream_rate_realised=rate,
                         O_o_share=Oo / tot, G_rank_share=Gr / tot, G_stream_share=Gs / tot,
                         gap_share=(O0 - Gs) / tot, term_overhead_share=terms[0] / tot, term_ranking_share=terms[1] / tot,
                         term_causal_share=terms[2] / tot, term_causal_sign=("positive" if terms[2] > TOL else
                                                                             "negative" if terms[2] < -TOL else "zero"),
                         tau=tau, tie_p=p, tie_draws=ntie)
                items.append((ci_, s, S.measured_share(b, c0, c1, 0.0), len(rows), ug, k0, k_o, f_o))
                rows.append(r)
        print(f"  D {c['ck']}", flush=True)
    boots = {i: np.full((NBOOT, 4), np.nan) for i in range(len(items))}
    for bb in range(NBOOT):
        cache = {}
        for i, (ci_, s, share0, ri, ug, k0, k_o, f_o) in enumerate(items):
            c = use[ci_]
            if ci_ not in cache:
                cache[ci_] = draws.rows(c, bb)
            t, counts = cache[ci_]
            if not len(t):
                continue
            vt = c["v"][t]
            nt = len(t)
            k0t, kot = S.measured_k(share0, nt), S.measured_k(f_o, nt)
            O0 = gains(vt, [vt], [k0t])[0][0]
            Oo = gains(vt, [vt], [kot])[0][0]
            Gr = gains(t130.prep(c["v1"][s][c["te"]])[t], [vt], [kot])[0][0]
            Gs = float(ug @ counts)
            terms = np.array([O0 - Oo, Oo - Gr, Gr - Gs])
            assert abs(terms.sum() - (O0 - Gs)) <= TOL
            boots[i][bb] = np.r_[O0 - Gs, terms] / c["jc"][c["te"]][t].sum()
    for i, it in enumerate(items):
        ri = it[3]
        for j, nm in enumerate(("gap", "term_overhead", "term_ranking", "term_causal")):
            rows[ri][f"{nm}_share_lo"], rows[ri][f"{nm}_share_hi"] = ci(boots[i][:, j])
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ figure data

def part_fig(run):
    sel = pd.read_csv(FINAL / "cached_analyses_selection.csv", low_memory=False)
    a = sel[(sel.section == "test_disagreement") & (sel.pool == "P1") & sel.e_dec_defined_test.astype(bool)]
    pa = a[["track", "geometry", "system", "target", "objective", "quota", "optimal_dec", "optimal_perc",
            "no_shared_optimum", "kendall_tau"]].assign(panel="a")
    curves = sorted((ROOT / "results" / "raw").glob("*_cached_analyses_c*/overhead_curves_core_20.csv"))[-1]
    pb = pd.read_csv(curves).assign(panel="b")
    g = pd.read_csv(FINAL / "cached_analyses_gap_accounting.csv")
    pc = g[np.isclose(g.budget_level, 0.20)].assign(panel="c")
    df = pd.concat([pa, pb, pc], ignore_index=True).drop(columns=["registry_version"], errors="ignore")
    return df, curves.parent.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["a", "b", "c", "d", "fig"])
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"cached_analyses_{args.part}", vars(args))
    t0 = time.time()
    name = {"a": "selection", "b": "benefit_harm", "c": "overhead_tolerance", "d": "gap_accounting",
            "fig": "figure_data"}[args.part]
    out = FINAL / f"cached_analyses_{name}.csv"
    if args.part == "fig":
        df, src = part_fig(run)
        (run / "sources.json").write_text(json.dumps({"curves_run": src}, indent=1))
    else:
        t130 = _load("t130", "130_streaming_controllers.py")
        cells, runs = build_cells(t130.t92, t130.t120, t130)
        (run / "sources.json").write_text(json.dumps({**runs, "registry": S.registry()["version"]}, indent=1))
        print(f"  {len(cells)} cells [{time.time() - t0:.0f}s]", flush=True)
        if args.part == "a":
            df = part_a(cells)
        elif args.part == "b":
            df = part_b(cells)
        elif args.part == "c":
            df, _ = part_c(cells, run)
        else:
            df = part_d(cells, t130)
    df.insert(0, "registry_version", S.registry()["version"])
    df.to_csv(run / out.name, index=False)
    if out.exists() and args.part != "fig":
        # a new file of this task: only this script writes it, so a re-run replaces its own earlier output
        print(f"  replacing {out.name} (written by an earlier run of this script)", flush=True)
    df.to_csv(out, index=False)
    print(f"  wrote {out} ({len(df)} rows) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
