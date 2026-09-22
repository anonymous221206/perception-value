#!/usr/bin/env python
"""Task 20: causal streaming allocation with rate controllers, on frozen saved scores.

Pre-registered in the pre-registration record (not part of this release), Task 20, committed before this script
was run.  Script 124 showed that a threshold frozen before the test stream (policy A) reproduces the official
whole-split ranking (C), while adding a causal running cap (B: at most floor(1 + k t) escalations after t inputs of a
unit) costs about half the decision value.  Two causal controllers try to recover that loss while hitting the target
rate:

  D  adaptive threshold: A's threshold, then a PI update of the escalation share the threshold is calibrated to,
     k_t = clip(k + gP (k - r_W) + gI (k - r_cum), 0, 1), with r_W the realised rate over the last W inputs of the unit
     and r_cum over all its inputs so far; B's cap still applies
  E  token bucket: one token at the start of a unit, k tokens accrue per input up to a capacity; escalate when the
     score clears A's threshold and a whole token is available (capacity infinite is exactly B)

W, gP, gI and the capacity are chosen per rate on the validation units only and frozen before the test stream.

Numeric hygiene.  Nothing is refit in this stage.  The scores it reads:
  official test scores    R1 `*_routers_r1`, nuPlan `*_nuplan_real_allocation`, core gates `*_budget_gate_scores`
  V1 scores (models fit on the train units only, as 124's V1)   `*_streaming_v1_scores`, written once by --fit_once
Ties at a threshold are broken by keys drawn from a stable per-row seed (124 used Python's per-process string hash).

  --fit_once   write the V1 scores (the only fitting this task does), then stop
  (default)    sanity gate against 124's shipped pooled figures, then everything else:
               results/final/streaming_controllers.csv
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time, zlib
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t124 = _load("t124", "124_causal_threshold.py")
t120, t92 = t124.t120, t124.t92

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
EPS = t92.EPS
RATES = (0.10, 0.20, 0.30, 0.50)
LEARNED = t124.LEARNED
NBOOT, NRAND, NTIE = 1000, 32, 32
POOLED_RATE = 0.20
D_GRID = [(w, gp, gi) for w in (10, 30, 100) for gp in (0.25, 0.5, 1.0) for gi in (0.0, 0.5, 1.0)]
E_GRID = [1, 2, 3, 5, 10, 20, np.inf]
SHIPPED = {"B_minus_C_point": -0.089, "B_minus_C_ci": (-0.14, -0.02), "A_minus_C_point": 0.012}
KEYCOLS = ("seq", "frame", "scenario", "iteration")


def reference() -> dict:
    """124's pooled figures the saved scores must reproduce, at the same roundings.  Camera frame: the shipped values
    above.  Any other frame (Task 23): the same figures from that frame's own 124 run -- the pooled B - C row, and
    A - C as the mean nDG of policy A (V1) minus that of policy C over the pooled cells and learned signals."""
    if frames.current() == "camera":
        return SHIPPED
    t = pd.read_csv(rap_runs.latest("causal_threshold") / "causal_threshold.csv")
    at = np.isclose(t.rate_target, POOLED_RATE)
    bc = t[at & (t.track == "pooled") & (t.variant == "V1") & (t.signal == "all_learned") & (t.policy == "B-C")].iloc[0]
    x = t[at & t.signal.isin(LEARNED) & (t.track != "pooled") & t.budget_level.isna()]
    x = x[(x.track != "nuPlan") | (x.system == "pdm_closed")]
    a_c = x[(x.variant == "V1") & (x.policy == "A")].ndg.mean() - x[x.policy == "C"].ndg.mean()
    return {"B_minus_C_point": round(float(bc.ndg), 3), "B_minus_C_ci": (round(float(bc.ndg_lo), 2), round(float(bc.ndg_hi), 2)),
            "A_minus_C_point": round(float(a_c), 3)}


def _latest(tag):
    """The newest run of `tag` in the current lift frame (rap.runs)."""
    return rap_runs.latest(tag)


def stable_seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def cell_key(c):
    return f"{c['track']}__{'na' if c['geometry'] == 'n/a' else c['geometry']}__{c['system']}__{c['target']}"


def _keys(d):
    return {k: d[k].astype(str).to_numpy().astype(str) for k in KEYCOLS if k in d.columns}


# ------------------------------------------------------------------------------------------------ cells and scores

def load_cells():
    """124's 14 cells, with the official core gate scores from the saved Task 19 set instead of a refit."""
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t120.register_features()
    gate_run = _latest("budget_gate_scores")
    cells = []
    for gen in (t124.core_cells(splits), t124.nuplan_cells(splits, nr_fcols)):
        for c in gen:
            d = c["d"]
            if c["track"] != "nuPlan":
                z = np.load(gate_run / f"gates__{cell_key(c)}.npz", allow_pickle=False)
                for k, v in _keys(d).items():
                    assert np.array_equal(z[f"key_{k}"], v), f"saved gate scores misaligned on {k}"
                assert np.array_equal(z["V"], c["v_all"])
                for a in ("gate_ridge", "gate_gbm"):
                    c["official"][a] = np.asarray(z[a], float)
            split = d.split.to_numpy()
            te = split == "test"
            n = int(te.sum())
            pos = np.full(len(d), -1)
            pos[te] = np.arange(n)
            c.update(tr=split == "train", va=split == "val", te=te, n=n, v=c["v_all"][te],
                     units_all=d.unit.astype(str).to_numpy())
            c["n_aff"] = int((np.abs(c["v"]) > EPS).sum())
            c["undefined"] = c["track"] == "nuPlan" and c["n_aff"] < t124.MIN_AFFECTED
            cells.append(c)
    return cells, gate_run.name


def fit_once(cells):
    run = runmeta.new_run("streaming_v1_scores", {"note": "V1 models (train units only), fit once for Task 20"})
    for c in cells:
        sc = t124.learned_scores(c, c["tr"])
        np.savez_compressed(run / f"v1__{cell_key(c)}.npz", V=c["v_all"],
                            **{f"key_{k}": v for k, v in _keys(c["d"]).items()}, **{a: np.asarray(sc[a], float) for a in LEARNED})
        print(f"  fit once: {cell_key(c)}", flush=True)
    print(f"  wrote {run}", flush=True)


def attach_v1(cells):
    run = _latest("streaming_v1_scores")
    for c in cells:
        z = np.load(run / f"v1__{cell_key(c)}.npz", allow_pickle=False)
        for k, v in _keys(c["d"]).items():
            assert np.array_equal(z[f"key_{k}"], v), f"saved V1 scores misaligned on {k}"
        assert np.array_equal(z["V"], c["v_all"])
        c["v1"] = {a: np.asarray(z[a], float) for a in LEARNED}
    return run.name


def streams(c, mask):
    """Positions (within `mask`) of each unit's inputs, in 124's timestamp order."""
    pos = np.full(len(c["d"]), -1)
    pos[mask] = np.arange(int(mask.sum()))
    order = [i for i in c["order"] if mask[i]]
    units = c["units_all"]
    out = {}
    for i in order:
        out.setdefault(units[i], []).append(pos[i])
    return [np.array(v, int) for _, v in sorted(out.items())]


# ------------------------------------------------------------------------------------------------ thresholds

class Calib:
    """124's `calibrate` for any escalation share, in O(1) per query."""

    def __init__(self, scores):
        s = np.round(np.asarray(scores, float), 9)
        self.sd = np.sort(s[np.isfinite(s)])[::-1]
        self.n = len(self.sd)
        if self.n:
            first = np.searchsorted(-self.sd, -self.sd, side="left")
            last = np.searchsorted(-self.sd, -self.sd, side="right")
            self.n_gt, self.n_eq = first, last - first

    def __call__(self, k):
        if self.n == 0 or k <= 0:
            return np.inf, 0.0
        if k >= 1:
            return -np.inf, 1.0
        kk = k * self.n
        idx = int(np.ceil(kk)) - 1
        return float(self.sd[idx]), float(np.clip((kk - self.n_gt[idx]) / max(self.n_eq[idx], 1), 0.0, 1.0))


def prep(scores):
    """Scores as every threshold rule compares them: NaN to -inf, rounded to 9 decimals (124's `mask_A`)."""
    return np.round(np.nan_to_num(np.asarray(scores, float), nan=-np.inf), 9)


def passes(s, key, tau, p):
    return s > tau or (s == tau and key < p)


def run_B(s, keys, k, tau, p, idx_units):
    m = np.zeros(len(s), bool)
    for idx in idx_units:
        c = 0
        for t, i in enumerate(idx, 1):
            if passes(s[i], keys[i], tau, p) and c < int(np.floor(1 + k * t)):
                m[i] = True
                c += 1
    return m


def run_A(s, keys, tau, p):
    return (s > tau) | ((s == tau) & (keys < p))


def run_D(s, keys, k, cal, idx_units, w, gp, gi):
    m = np.zeros(len(s), bool)
    for idx in idx_units:
        c, win, wsum = 0, deque(), 0
        for t, i in enumerate(idx, 1):
            if t == 1:
                kt = k
            else:
                r_w, r_cum = wsum / len(win), c / (t - 1)
                kt = min(max(k + gp * (k - r_w) + gi * (k - r_cum), 0.0), 1.0)
            tau, p = cal(kt)
            esc = passes(s[i], keys[i], tau, p) and c < int(np.floor(1 + k * t))
            if esc:
                m[i] = True
                c += 1
            win.append(int(esc))
            wsum += int(esc)
            if len(win) > w:
                wsum -= win.popleft()
    return m


def run_E(s, keys, k, tau, p, idx_units, capacity):
    """Token bucket in hundredths of a token, so the arithmetic is exact for k in {0.1, 0.2, 0.3, 0.5}."""
    m = np.zeros(len(s), bool)
    kk = int(round(100 * k))
    cap = None if not np.isfinite(capacity) else int(round(100 * capacity))
    for idx in idx_units:
        tok = 100
        for i in idx:
            tok += kk
            if cap is not None:
                tok = min(tok, cap)
            if tok >= 100 and passes(s[i], keys[i], tau, p):
                m[i] = True
                tok -= 100
    return m


def has_ties(test_s, cal_s):
    a = np.round(np.asarray(test_s, float), 9)
    b = np.round(np.asarray(cal_s, float), 9)
    return bool(np.isin(a[np.isfinite(a)], b[np.isfinite(b)]).any())


# ------------------------------------------------------------------------------------------------ evaluation helpers

def unit_gains(v, mask, idx_units):
    return np.array([v[idx][mask[idx]].sum() for idx in idx_units]), np.array([mask[idx].mean() for idx in idx_units])


def prize_at(v, k):
    kn = max(int(round(k * len(v))), 1)
    return float(t92.topk_expect(v, [v], [kn])[0][0][0])


def topk_gain(score, v, k):
    kn = max(int(round(k * len(v))), 1)
    return float(t92.topk_expect(np.nan_to_num(score, nan=-np.inf), [v], [kn])[0][0][0])


def policy_units(policy, s, keys_list, k, cal, tauA, pA, idx_units, v, hp):
    """Mean per-unit gain, mean realised rate and the mean per-unit rate vector over the tie draws."""
    gs, rs, ur = [], [], []
    for keys in keys_list:
        if policy == "A":
            m = run_A(s, keys, tauA, pA)
        elif policy == "B":
            m = run_B(s, keys, k, tauA, pA, idx_units)
        elif policy == "D":
            m = run_D(s, keys, k, cal, idx_units, *hp)
        else:
            m = run_E(s, keys, k, tauA, pA, idx_units, hp)
        g, r = unit_gains(v, m, idx_units)
        gs.append(g)
        rs.append(m.mean())
        ur.append(r)
    return np.mean(gs, axis=0), float(np.mean(rs)), np.mean(ur, axis=0)


# ------------------------------------------------------------------------------------------------ sanity gate (124's own procedure)

def sanity(cells):
    """124's V1 A and B at 20% with its tie seeds and its filtered joint bootstrap, from the saved scores."""
    k = POOLED_RATE
    series = {}
    for ci, c in enumerate(cells):
        v = c["v"]
        idx_units = streams(c, c["te"])
        c["idx_test"], c["uniq_test"] = idx_units, sorted(set(c["units_all"][c["te"]]))
        for s in LEARNED:
            off = np.asarray(c["official"][s], float)[c["te"]]
            series[(ci, s, "C")] = ("topk", np.nan_to_num(off, nan=-np.inf))
            tau, p = t124.calibrate(c["v1"][s][c["va"]], k)
            rng = np.random.default_rng(t124.seed_of(c["system"], c["target"], c["geometry"], s, k, "V1"))
            mA = t124.mask_A(c["v1"][s][c["te"]], tau, p, rng)
            unit_id = c["units_all"][c["te"]]
            stream = np.concatenate(idx_units)
            mB = t124.apply_cap(mA, k, unit_id, stream)
            for pol, m in (("A", mA), ("B", mB)):
                series[(ci, s, pol)] = np.array([v[idx][m[idx]].sum() for idx in idx_units])
    pooled_cells = [i for i, c in enumerate(cells) if c["track"] != "nuPlan" or c["system"] == "pdm_closed"]
    point = {}
    for (ci, s, pol), val in series.items():
        c = cells[ci]
        prize = prize_at(c["v"], k)
        g = topk_gain(val[1], c["v"], k) if pol == "C" else float(val.sum())
        point[(ci, s, pol)] = g / prize if prize > EPS else np.nan
    a_minus_c = float(np.mean([point[(ci, s, "A")] for ci in pooled_cells for s in LEARNED])
                      - np.mean([point[(ci, s, "C")] for ci in pooled_cells for s in LEARNED]))
    draws = joint_draws(cells, series, k, filtered=True, rng=np.random.default_rng(0))
    per_signal = [np.nanmean(np.array([draws[(ci, s, "B")] - draws[(ci, s, "C")] for ci in pooled_cells]), axis=0)
                  for s in LEARNED]
    pooled = np.nanmean(np.array(per_signal), axis=0)
    lo, hi = t92.ci(pooled)
    res = {"B_minus_C_point": float(np.nanmean(pooled)), "B_minus_C_lo": lo, "B_minus_C_hi": hi, "A_minus_C_point": a_minus_c}
    ref = reference()
    res["match_B_point_3dp"] = round(res["B_minus_C_point"], 3) == ref["B_minus_C_point"]
    res["match_B_ci_2dp"] = (round(lo, 2), round(hi, 2)) == ref["B_minus_C_ci"]
    res["match_A_point_2dp"] = round(a_minus_c, 2) == round(ref["A_minus_C_point"], 2)
    res["passed"] = bool(res["match_B_point_3dp"] and res["match_B_ci_2dp"] and res["match_A_point_2dp"])
    res["ties_at_threshold_rows"] = int(sum(
        has_ties(cells[ci]["v1"][s][cells[ci]["te"]], cells[ci]["v1"][s][cells[ci]["va"]]) for ci in pooled_cells for s in LEARNED))
    return res


def joint_draws(cells, series, k, filtered, rng):
    """124's joint bootstrap: units resampled once per dataset per draw and shared by every cell and series.

    `filtered` applies the benchmark's 25% prize filter (124); otherwise every draw is kept and the statistic is
    normalised by the full-sample prize, so no draw can be dropped (Task 20's pre-registered form).
    """
    datasets = ["nuScenes", "KITTI", "nuPlan"]
    units_of = {ds: sorted(set().union(*[set(c["uniq_test"]) for c in cells if c["dataset"] == ds])) for ds in datasets}
    upos = [{u: i for i, u in enumerate(c["uniq_test"])} for c in cells]
    prize0 = [prize_at(c["v"], k) for c in cells]
    out = {key: np.full(NBOOT, np.nan) for key in series}
    by_cell = {}
    for key in series:
        by_cell.setdefault(key[0], []).append(key)
    for b in range(NBOOT):
        pick = {ds: rng.integers(0, len(units_of[ds]), len(units_of[ds])) for ds in datasets}
        for ci, c in enumerate(cells):
            us = [u for u in (units_of[c["dataset"]][i] for i in pick[c["dataset"]]) if u in upos[ci]]
            if not us:
                continue
            counts = np.bincount([upos[ci][u] for u in us], minlength=len(c["uniq_test"])).astype(float)
            t = np.concatenate([c["idx_test"][upos[ci][u]] for u in us])
            vt = c["v"][t]
            kn = max(int(round(k * len(t))), 1)
            if filtered:
                pz = float(t92.topk_expect(vt, [vt], [kn])[0][0][0])
                if not pz > max(EPS, 0.25 * prize0[ci]):
                    continue
                den = pz
            else:
                if not prize0[ci] > EPS:
                    continue
                den = prize0[ci]
            for key in by_cell.get(ci, ()):
                val = series[key]
                g = float(t92.topk_expect(val[1][t], [vt], [kn])[0][0][0]) if isinstance(val, tuple) else float(val @ counts)
                out[key][b] = g / den
    return out


# ------------------------------------------------------------------------------------------------ hyperparameters on validation units

def choose(cells, k, policy, grid):
    """Per rate, across the 12 pooled cells and six learned signals, on the validation units only.

    Among configurations whose median |realised rate - k| on validation is at most 2 points, take the highest mean
    (nDG - C) on validation; if none qualifies, the smallest median deviation, ties to the higher mean value.
    """
    pooled = [c for c in cells if c["track"] != "nuPlan" or c["system"] == "pdm_closed"]
    table = []
    for hp in grid:
        devs, vals = [], []
        for c in pooled:
            va = c["va"]
            v = c["v_all"][va]
            idx_units = streams(c, va)
            prize = prize_at(v, k)
            for s in LEARNED:
                sv = prep(c["v1"][s][va])
                cal = Calib(sv)
                tau, p = cal(k)
                keys = np.random.default_rng(stable_seed("val", cell_key(c), s, k)).random(len(sv))
                m = (run_D(sv, keys, k, cal, idx_units, *hp) if policy == "D" else
                     run_E(sv, keys, k, tau, p, idx_units, hp))
                devs.append(abs(m.mean() - k))
                if prize > EPS:
                    vals.append((v[m].sum() - topk_gain(sv, v, k)) / prize)
        table.append({"policy": policy, "rate_target": k, "hp": hp, "val_median_abs_dev": float(np.median(devs)),
                      "val_mean_ndg_minus_C": float(np.mean(vals)) if vals else np.nan})
    ok = [r for r in table if r["val_median_abs_dev"] <= 0.02]
    best = (max(ok, key=lambda r: r["val_mean_ndg_minus_C"]) if ok else
            min(table, key=lambda r: (r["val_median_abs_dev"], -r["val_mean_ndg_minus_C"])))
    for r in table:
        r["chosen"] = r is best
        r["qualifies"] = r in ok
    return best["hp"], table


# ------------------------------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit_once", action="store_true")
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    t_all = time.time()
    cells, gate_run = load_cells()
    if args.fit_once:
        fit_once(cells)
        return
    run = runmeta.new_run("streaming_controllers", vars(args))
    v1_run = attach_v1(cells)
    rows = []

    # ---- sanity gate
    san, ref = sanity(cells), reference()
    rows.append({"section": "sanity", "signal": "all_learned", "rate_target": POOLED_RATE, "variant": "V1",
                 "shipped_B_minus_C_point": ref["B_minus_C_point"], "shipped_B_minus_C_lo": ref["B_minus_C_ci"][0],
                 "shipped_B_minus_C_hi": ref["B_minus_C_ci"][1], "shipped_A_minus_C_point": ref["A_minus_C_point"],
                 **{k: v for k, v in san.items()}, "gate_scores_run": gate_run, "v1_scores_run": v1_run})
    print(f"  sanity: B - C {san['B_minus_C_point']:+.4f} [{san['B_minus_C_lo']:+.4f}, {san['B_minus_C_hi']:+.4f}], "
          f"A - C {san['A_minus_C_point']:+.4f}; passed {san['passed']}", flush=True)
    if not san["passed"]:
        pd.DataFrame(rows).to_csv(run / "streaming_controllers_SANITY_FAILED.csv", index=False)
        raise SystemExit("sanity gate failed: the saved scores do not reproduce 124's shipped pooled figures; stopping")

    # ---- hyperparameters, validation units only
    chosen = {}
    for k in RATES:
        for policy, grid in (("D", D_GRID), ("E", E_GRID)):
            hp, table = choose(cells, k, policy, grid)
            chosen[(policy, k)] = hp
            for r in table:
                hpv = r.pop("hp")
                rows.append({"section": "hyperparameters", **r,
                             **({"W": hpv[0], "gP": hpv[1], "gI": hpv[2]} if policy == "D" else {"capacity": hpv})})
            print(f"  chosen {policy} at {k:.0%}: {hp}", flush=True)

    # ---- test streams
    series, meta = {}, {}
    for ci, c in enumerate(cells):
        v, te, n = c["v"], c["te"], c["n"]
        for k in RATES:
            prize = prize_at(v, k)
            kn = max(int(round(k * n)), 1)
            for s in ["random"] + LEARNED:
                if s == "random":
                    # uniform scores: A's threshold is 1 - k and D's is 1 - k_t exactly; ties have probability zero
                    runs = [(np.random.default_rng(stable_seed("score", cell_key(c), k, r)).random(n), None) for r in range(NRAND)]
                    keysets = [np.zeros(n)]
                    series[(ci, s, k, "C")] = ("topk", np.zeros(n))                 # all tied: k/n of the total
                    gC = float(t92.topk_expect(np.zeros(n), [v], [kn])[0][0][0])
                else:
                    sv, cs = prep(c["v1"][s][te]), c["v1"][s][c["va"]]
                    runs = [(sv, Calib(cs))]
                    ntie = NTIE if has_ties(sv, cs) else 1
                    keysets = [np.random.default_rng(stable_seed("tie", cell_key(c), s, k, r)).random(n) for r in range(ntie)]
                    off = np.nan_to_num(np.asarray(c["official"][s], float)[te], nan=-np.inf)
                    series[(ci, s, k, "C")] = ("topk", off)
                    gC = float(t92.topk_expect(off, [v], [kn])[0][0][0])
                for pol in ("A", "B", "D", "E"):
                    gsum, rates, urates = [], [], []
                    for sc, cal in runs:
                        if cal is None:
                            cal = (lambda kt: (1.0 - kt, 0.0))                     # noqa: E731
                        tauA, pA = cal(k)
                        g_u, r, ur = policy_units(pol, sc, keysets, k, cal, tauA, pA, c["idx_test"], v, chosen.get((pol, k)))
                        gsum.append(g_u); rates.append(r); urates.append(ur)
                    g_u, ur = np.mean(gsum, axis=0), np.mean(urates, axis=0)
                    series[(ci, s, k, pol)] = g_u
                    meta[(ci, s, k, pol)] = dict(rate=float(np.mean(rates)), umin=float(ur.min()), umax=float(ur.max()),
                                                 ties=len(keysets), draws=len(runs), prize=prize, gain=float(g_u.sum()), gainC=gC)
        print(f"  streams {cell_key(c)} [{time.time() - t_all:.0f}s]", flush=True)

    # ---- paired cluster bootstrap, every draw kept
    by_rate = {}
    for k in RATES:
        sub = {(ci, s, pol): series[(ci, s, kk, pol)] for (ci, s, kk, pol) in series if kk == k}
        by_rate[k] = joint_draws(cells, sub, k, filtered=False, rng=np.random.default_rng(stable_seed("bootstrap", k)))
    pooled_cells = [i for i, c in enumerate(cells) if c["track"] != "nuPlan" or c["system"] == "pdm_closed"]
    for (ci, s, k, pol), mt in meta.items():
        c = cells[ci]
        dr = by_rate[k][(ci, s, pol)] - by_rate[k][(ci, s, "C")]
        lo, hi = t92.ci(dr)
        defined = not c["undefined"] and mt["prize"] > EPS
        ndg = mt["gain"] / mt["prize"] if defined else np.nan
        ndgC = mt["gainC"] / mt["prize"] if defined else np.nan
        rows.append({"section": "row", "track": c["track"], "geometry": c["geometry"], "system": c["system"],
                     "target": c["target"], "signal": s, "policy": pol, "rate_target": k, "variant": "V1",
                     "n_test_frames": c["n"], "n_test_units": len(c["uniq_test"]), "n_affected": c["n_aff"],
                     "ndg_defined": defined, "pooled_cell": ci in pooled_cells, "gain": mt["gain"], "gain_C": mt["gainC"],
                     "oracle_prize": mt["prize"], "ndg": ndg, "ndg_C": ndgC, "ndg_minus_C": ndg - ndgC if defined else np.nan,
                     "ndg_minus_C_lo": lo if defined else np.nan, "ndg_minus_C_hi": hi if defined else np.nan,
                     "rate_realised": mt["rate"], "rate_abs_dev": abs(mt["rate"] - k), "rate_unit_min": mt["umin"],
                     "rate_unit_max": mt["umax"], "tie_draws": mt["ties"], "score_draws": mt["draws"],
                     "hyperparameters": str(chosen.get((pol, k), ""))})

    # ---- pooled: 12 cells x 6 learned signals (and random for reference)
    for k in RATES:
        for pol in ("A", "B", "D", "E"):
            for group, sigs in (("all_learned", LEARNED), ("random", ["random"])):
                keys = [(ci, s) for ci in pooled_cells for s in sigs]
                arr = np.array([by_rate[k][(ci, s, pol)] - by_rate[k][(ci, s, "C")] for ci, s in keys])
                per_draw = np.nanmean(arr, axis=0)
                lo, hi = t92.ci(per_draw)
                pts = [r for r in rows if r["section"] == "row" and r["policy"] == pol and r["rate_target"] == k
                       and r["pooled_cell"] and r["signal"] in sigs]
                point = float(np.mean([r["ndg_minus_C"] for r in pts]))
                med_dev = float(np.median([r["rate_abs_dev"] for r in pts]))
                row = {"section": "pooled", "signal": group, "policy": pol, "rate_target": k, "variant": "V1",
                       "n_rows": len(pts), "ndg_minus_C": point, "ndg_minus_C_lo": lo, "ndg_minus_C_hi": hi,
                       "ndg_minus_C_boot_mean": float(np.nanmean(per_draw)),
                       "rows_improved_over_C": int(sum(r["ndg_minus_C"] > 0 for r in pts)),
                       "rows_rate_miss_gt_5pp": int(sum(r["rate_abs_dev"] > 0.05 for r in pts)),
                       "median_abs_rate_dev": med_dev, "draws_kept": int(np.isfinite(per_draw).sum())}
                if pol in ("D", "E") and group == "all_learned" and np.isclose(k, POOLED_RATE):
                    rec = (hi >= 0) and med_dev <= 0.02
                    row["reading"] = "recovers the causal loss" if rec else "does not recover"
                rows.append(row)
    df = pd.DataFrame(rows)
    for out in (FINAL / "streaming_controllers.csv", run / "streaming_controllers.csv"):
        df.to_csv(out, index=False)
    p = df[(df.section == "pooled") & (df.signal == "all_learned")]
    print(p[["policy", "rate_target", "ndg_minus_C", "ndg_minus_C_lo", "ndg_minus_C_hi", "rows_improved_over_C",
             "rows_rate_miss_gt_5pp", "median_abs_rate_dev", "reading"]].to_string(index=False))
    print(f"  wrote {FINAL / 'streaming_controllers.csv'} ({len(df)} rows) in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
