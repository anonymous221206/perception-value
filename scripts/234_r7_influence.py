#!/usr/bin/env python
"""Follow-up round 3, R7: influence, tail and selection audit of the XL claims (addendum R7 in docs/followup_protocol.md).

  --stage a        no fit: settings with scenes removed, day / night, without the two logs; the other braking rows
  --stage b        refit and save per-frame scores; gate R7-0 (X2, R1 and R3 test rows reproduced exactly)
  --stage cdg      comparators (C), influence (D) and capped decision value (G), from the saved scores
  --stage e        fit-set sensitivity of gate_gbm on V
  --stage f        20 repeated log-grouped 5-fold out-of-fold runs over the 765 scenes
  --stage h        loss weights, exact recombination from the Jterm columns
  --stage reading  the registered reading of the seven claims
"""
from __future__ import annotations

import argparse, gzip, importlib.util, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta, xl                                                       # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402
from rap.scoring import EPS                                                        # noqa: E402

OUT = Path(RESULTS) / "followup" / "R7"
WORK = Path(CACHE) / "xl" / "r7"
TABLES = Path(CACHE) / "xl" / "tables850"
REB = Path(RESULTS) / "followup"
LOGS = ("n015-2018-11-21-19-11-29+0800", "n015-2018-11-21-19-38-26+0800")
TARGETS = ("V", "fn_gain", "risk_gain", "Q", "G")
LEARNERS = ("gate_gbm", "gate_ridge", "R1_gbm_reg", "R1_mlp_reg")
NBOOT = 1000
NPROC = 6


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def meta():
    sp = xl.scene_split()
    m = pd.read_csv(Path(CACHE) / "xl" / "scene_meta.csv")
    sp = sp.merge(m[["scene", "log"]], on="scene", how="left", validate="one_to_one")
    assert sp.log.notna().all()
    return sp


def populations(sp):
    new = sp[sp.group == "new765"]
    return {"pooled850": set(sp.scene), "new765": set(new.scene), "new228_test": set(new[new.deep_split == "test"].scene),
            "frozen85": set(sp[sp.group == "frozen85"].scene)}


def table(geometry):
    return pd.read_pickle(TABLES / f"{geometry}__{xl.PAIR}.pkl")


def nboot_describe(v, c, units):
    r = xl.describe(v, c, (0.2,))
    r.update(xl.bootstrap_describe(v, c, units, NBOOT, np.random.default_rng(0), (0.2,)))
    return r


def describe_two(v, c, seq, log):
    """Scene-level intervals and the log-level ones beside them (rule 6)."""
    r = nboot_describe(v, c, seq)
    lg = xl.bootstrap_describe(v, c, log, NBOOT, np.random.default_rng(0), (0.2,))
    r.update({k + "_log": x for k, x in lg.items()})
    return r


# ------------------------------------------------------------------------------------------------ A

def _a_task(t):
    label, v, c, seq, log = t
    return {**label, "n_scenes": len(set(seq)), "n_logs": len(set(log)), **describe_two(v, c, seq, log)}


def drop_rows(x, label, sp, jc, jf, tasks, names):
    """The registered subsets of one population (ranked on that population)."""
    v = (x[jc] - x[jf]).to_numpy(float)
    x = x.assign(_v=v)
    sv = x.groupby("seq")._v.sum().sort_values()
    neg, pos = list(sv.index[:5]), list(sv.index[::-1][:5])
    night = set(sp[sp.night == 1].scene)
    log_of = dict(zip(sp.scene, sp.log))
    two = set(sp[sp.log.isin(LOGS)].scene)
    subsets = {"all": set()}
    for m in (1, 3, 5):
        subsets[f"minus {m} most negative"] = set(neg[:m])
        subsets[f"minus {m} most positive"] = set(pos[:m])
    subsets["minus both 5"] = set(neg) | set(pos)
    subsets["minus the two logs"] = two
    for s, keep in (("day only", ~x.seq.isin(night)), ("night only", x.seq.isin(night))):
        subsets[s] = None
    for name, rem in subsets.items():
        if name == "day only":
            y = x[~x.seq.isin(night)]
        elif name == "night only":
            y = x[x.seq.isin(night)]
        else:
            y = x[~x.seq.isin(rem)]
        if y.empty:
            continue
        seq = y.seq.to_numpy()
        tasks.append(({**label, "subset": name}, y._v.to_numpy(float), y[jc].to_numpy(float), seq,
                      np.array([log_of[s] for s in seq])))
    for rank, s in enumerate(neg, 1):
        names.append({**label, "side": "most negative", "rank": rank, "scene": s, "sum_V": float(sv[s]), "log": log_of[s],
                      "night": int(s in night)})
    for rank, s in enumerate(pos, 1):
        names.append({**label, "side": "most positive", "rank": rank, "scene": s, "sum_V": float(sv[s]), "log": log_of[s],
                      "night": int(s in night)})


def other_rows(sp):
    """The other braking rows on their own populations, each with the stored all-full it must reproduce."""
    pops = populations(sp)
    up = pd.read_csv(REB / "E5" / "x7_upgrades.csv")
    ft = pd.read_csv(REB / "E5" / "x7_finetuned.csv")
    stored = {}
    for t in (up, ft):
        for r in t[t.primary & (t.geometry == "mono") & (t.system == "brake")].itertuples():
            stored[(r.old, r.new)] = r.all_full_reduction
    for f in sorted((Path(CACHE) / "xl" / "r5").glob("pair__*.pkl")):
        d = pd.read_pickle(f)
        key = [k for k in stored if f.stem == f"pair__{k[0]}__{k[1]}"][0]
        yield f"E5 {key[0]} -> {key[1]}", "new228_test", d.J_cheap.to_numpy(float), d.J_full.to_numpy(float), d.seq.to_numpy(), stored[key]
    e7 = pd.read_csv(REB / "E7" / "settings.csv")
    s = e7[e7.primary == True].iloc[0]                                              # noqa: E712
    d = pd.read_pickle(Path(CACHE) / "xl" / "e7" / "detected__shared__voxel.pkl")
    d = d[d.seq.isin(set(sp[sp.official == "val"].scene))]
    yield "E7 LiDAR primary (PointPillar -> VoxelNeXt, detected, shared 0.25)", "official_val150", d.J_cheap.to_numpy(float), \
        d.J_full.to_numpy(float), d.seq.to_numpy(), s.all_full_reduction
    e10 = pd.read_csv(REB / "E10" / "settings.csv")
    m = e10[(e10.dataset == "nuScenes") & (e10["mode"] == "ns_full_640") & (e10.propagation == "ego") & (e10.geometry == "mono")
            & (e10.population == "new228_test")].iloc[0]
    d = pd.read_pickle(Path(CACHE) / "xl" / "e10" / "tables" / "nuScenes__ns_full_640__ego__mono.pkl")
    d = d[(d["first"] == 0) & d.seq.isin(pops["new228_test"])]
    yield "E10 reuse primary (640, ego carry, mono)", "new228_test", d.J_stale.to_numpy(float), d.J_fresh.to_numpy(float), \
        d.seq.to_numpy(), m.all_fresh_reduction


def stage_a(args):
    sp = meta()
    pops = populations(sp)
    tasks, names = [], []
    tabs = {"mono": table("mono"), "oracle": table("oracle")}
    for geo, system, jc, jf in (("mono", "brake", "J_cheap", "J_full"), ("oracle", "brake", "J_cheap", "J_full"),
                                ("mono", "plan_ade", "JC_ade_cheap", "JC_ade_full"),
                                ("mono", "plan_fde", "JC_fde_cheap", "JC_fde_full")):
        d = tabs[geo]
        d = d[d[jc].notna()]
        for pop, units in pops.items():
            drop_rows(d[d.seq.isin(units)], {"geometry": geo, "system": system, "population": pop}, sp, jc, jf, tasks, names)
    print(f"{len(tasks)} settings rows", flush=True)
    with Pool(NPROC) as p:
        rows = p.map(_a_task, tasks, chunksize=2)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "partA_settings.csv", index=False)
    pd.DataFrame(names).to_csv(OUT / "partA_scenes.csv", index=False)
    log_of = dict(zip(sp.scene, sp.log))
    tasks2 = []
    for name, pop, jc, jf, seq, stored in other_rows(sp):
        v = jc - jf
        tot = jc.sum()
        assert abs(v.sum() / tot - stored) < 1e-9, (name, v.sum() / tot, stored)
        s = pd.Series(v).groupby(seq).sum()
        c = pd.Series(jc).groupby(seq).sum()
        drop = (s.sum() - s) / (c.sum() - c)
        top = (drop - v.sum() / tot).abs().idxmax()
        for sub, keep in (("all", np.ones(len(v), bool)), (f"minus {top}", seq != top)):
            tasks2.append(({"row": name, "population": pop, "subset": sub, "removed": "" if sub == "all" else top,
                            "removed_sum_V": 0.0 if sub == "all" else float(s[top])}, v[keep], jc[keep], seq[keep],
                           np.array([log_of[x] for x in seq[keep]])))
    with Pool(NPROC) as p:
        rows = p.map(_a_task, tasks2, chunksize=1)
    pd.DataFrame(rows).to_csv(OUT / "partA_other_rows.csv", index=False)


# ------------------------------------------------------------------------------------------------ B

GATE_X2 = ["gain", "gain_random", "delta_vs_random", "delta_lo_all_draws", "delta_hi_all_draws", "delta_share_all_cheap",
           "delta_share_lo_all_draws", "delta_share_hi_all_draws", "ndg", "delta_lo_filtered", "delta_hi_filtered"]


def router_scores(X, y, fit, name):
    m = xl.t103().models()[name][1]()
    if name == "R1_gbm_reg":
        with threadpool_limits(limits=1, user_api="openmp"):
            m.fit(X[fit], y[fit])
            return m.predict(X)
    m.fit(X[fit], y[fit])
    return m.predict(X)


def stage_b(args):
    t227 = _load("t227", "227_r1_target_swap.py")
    t231 = _load("t231", "231_r3_gate_cost.py")
    a = argparse.Namespace(det=str(t227.DET), dataroot=args.dataroot)
    cells_, log_of = t227.setup(a)
    mat = xl.r1_matrix(a.det)
    tg = pd.read_csv(REB / "R1" / "targets.csv", dtype={"seq": str})
    scores, memo = {}, {}
    for c in cells_:
        t0 = time.time()
        d = c["d"]
        fit = d.split.isin(["train", "val"]).to_numpy()
        lab = "brake" if c["system"] == "brake" else "plan"
        arrays = t227.target_arrays(d, tg, lab)
        arrays["V"] = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        X1 = xl.r1_rows(d, mat)
        cell = f"{c['geometry']}_{c['system']}"
        out = {}
        for learner in LEARNERS:
            for variant, t, key in [("raw", t, t) for t in TARGETS] + [("cdf", t, f"{t}_cdf") for t in ("Q", "G")]:
                mk = (lab, learner, variant, t) if t != "V" else None
                if mk in memo:
                    out[(learner, variant, t)] = memo[mk]
                    continue
                y = arrays[key].copy()
                assert np.isfinite(y[fit]).all()
                s = (xl.gate_scores(d, c["fcols"], y, fit, which=(learner,))[learner] if learner.startswith("gate")
                     else router_scores(X1, y, fit, learner))
                out[(learner, variant, t)] = s
                if mk is not None:
                    memo[mk] = s
        if c["system"] == "brake":
            groups = t231.group_columns(c["fcols"])
            sel = json.loads((REB / "R3" / "selected.json").read_text())
            v = arrays["V"]
            for learner in t231.LEARNERS:
                _, s = t231.fit_predict(d, t231.subset_columns(sel[learner]["subset"], groups), v, fit, learner)
                out[("R3", learner, sel[learner]["subset"])] = s
        scores[cell] = {"keys": d[["seq", "frame", "split"]].astype({"seq": str, "frame": int}).reset_index(drop=True), "s": out}
        print(f"  {cell}: {len(out)} series [{time.time() - t0:.0f}s]", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "scores.pkl", "wb") as fh:
        pickle.dump(scores, fh, protocol=pickle.HIGHEST_PROTOCOL)
    # one file per signal: a column per cell whose values differ (non-V targets: one per frame set)
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted({k for c in scores.values() for k in c["s"]}, key=str)
    for k in names:
        frames_ = None
        for cell, c in scores.items():
            if k not in c["s"]:
                continue
            if k[0] != "R3" and k[2] != "V" and cell.startswith("oracle"):
                continue
            if k[0] != "R3" and k[2] != "V" and cell == "mono_plan_fde":
                continue
            col = cell if (k[0] == "R3" or k[2] == "V") else ("brake" if cell == "mono_brake" else "plan")
            f = c["keys"].assign(**{col: c["s"][k]})
            frames_ = f if frames_ is None else frames_.merge(f.drop(columns="split"), on=["seq", "frame"], how="outer")
        fn = "__".join(map(str, k))
        frames_.sort_values(["seq", "frame"]).to_csv(OUT / f"scores_{fn}.csv.gz", index=False, float_format="%.17g")
    gate(cells_, log_of, scores, t231)


def gate(cells_, log_of, scores, t231):
    x3 = pd.read_csv(REB / "E1" / "x3_budget_sweep.csv", float_precision="round_trip")
    sw = pd.read_csv(REB / "R1" / "swap.csv", float_precision="round_trip")
    fails, n = [], {"x2": 0, "r1": 0, "r3": 0}
    for unit in ("scene", "log"):
        prep = []
        for c in cells_:
            d = c["d"] if unit == "scene" else c["d"].assign(unit=c["d"].seq.map(log_of))
            s = scores[f"{c['geometry']}_{c['system']}"]["s"]
            series = {f"{l}|{v}|{t}": s[(l, v, t)] for (l, v, t) in [k for k in s if k[0] != "R3"]}
            prep.append(xl.prepare(dict(c, d=d), (d.split == "test").to_numpy(), series, xl.SWEEP))
        draws, keep = xl.paired_draws(prep, NBOOT, np.random.default_rng(0), xl.SWEEP)
        rows = pd.DataFrame(xl.paired_rows(prep, draws, keep, xl.SWEEP))
        for r in rows.itertuples():
            l, variant, t = r.signal.split("|")
            if variant == "raw" and t == "V":
                m = x3[(x3.signal == l) & (x3.geometry == r.geometry) & (x3.system == r.system) & (x3.quota == r.quota)
                       & (x3.bootstrap_unit == unit)]
                assert len(m) == 1
                for k in GATE_X2:
                    n["x2"] += 1
                    a_, b_ = float(getattr(r, k)), float(m.iloc[0][k])
                    if not ((np.isnan(a_) and np.isnan(b_)) or a_ == b_):
                        fails.append(f"X2 {unit} {r.signal} {r.geometry} {r.system} {r.quota} {k}: {a_!r} {b_!r}")
            if l.startswith("gate"):
                m = sw[(sw.signal == r.signal) & (sw.geometry == r.geometry) & (sw.system == r.system) & (sw.quota == r.quota)
                       & (sw.bootstrap_unit == unit)]
                assert len(m) == 1, r.signal
                for k in GATE_X2:
                    n["r1"] += 1
                    a_, b_ = float(getattr(r, k)), float(m.iloc[0][k])
                    if not ((np.isnan(a_) and np.isnan(b_)) or a_ == b_):
                        fails.append(f"R1 {unit} {r.signal} {r.geometry} {r.system} {r.quota} {k}: {a_!r} {b_!r}")
    costs = pd.read_csv(REB / "R3" / "feature_costs.csv")
    tt = pd.read_csv(REB / "R3" / "test.csv", float_precision="round_trip")
    tt = tt[tt.chosen & tt.feasible]
    for c in cells_:
        if c["system"] != "brake":
            continue
        d = c["d"]
        test = (d.split == "test").to_numpy()
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        tot = float(d[c["cheap"]].to_numpy(float)[test].sum())
        for r in tt[tt.geometry == c["geometry"]].itertuples():
            s = scores[f"{c['geometry']}_brake"]["s"][("R3", r.learner, r.subset)]
            f = t231.frac_at(r.budget, r.C_c, r.C_f, r.C_S)
            g, k = t231.gain_at(s[test], v[test], f)
            gr, _ = t231.gain_at(None, v[test], r.budget)
            x = t231.boot(s[test], v[test], d.unit.astype(str).to_numpy()[test], f, r.budget, NBOOT, np.random.default_rng(0))
            lo, hi = xl.ci(x)
            for a_, b_, nm in ((g, r.gain, "gain"), (k, r.k, "k"), (g - gr, r.vs_random, "vs_random"), (lo, r.vs_random_lo, "lo"),
                               (hi, r.vs_random_hi, "hi")):
                n["r3"] += 1
                if float(a_) != float(b_):
                    fails.append(f"R3 {c['geometry']} {r.learner} {r.unit} {r.budget} {nm}: {a_!r} {b_!r}")
    rep = {"comparisons": n, "differing": len(fails), "pass": not fails, "first_differences": fails[:20]}
    (OUT / "gate_r7_0.json").write_text(json.dumps(rep, indent=1))
    print("gate R7-0:", {k: rep[k] for k in ("comparisons", "differing", "pass")}, flush=True)
    if fails:
        print("\n".join(fails[:20]), flush=True)
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ C, D, G

class Cell:
    """The primary cell's test frames, with bootstrap index draws per unit (paired: one set of draws for everything)."""

    def __init__(self, geometry="mono"):
        with open(WORK / "scores.pkl", "rb") as fh:
            sc = pickle.load(fh)[f"{geometry}_brake"]
        sp = meta()
        self.log_of = dict(zip(sp.scene, sp.log))
        t = table(geometry).astype({"seq": str, "frame": int})
        d = sc["keys"].merge(t[["seq", "frame", "J_cheap", "J_full"]], on=["seq", "frame"], how="left", validate="one_to_one")
        self.d, self.s = d, sc["s"]
        self.v_all = (d.J_cheap - d.J_full).to_numpy(float)
        self.test = (d.split == "test").to_numpy()
        self.fit = d.split.isin(["train", "val"]).to_numpy()
        self.seq = d.seq.to_numpy()
        self.log = np.array([self.log_of[x] for x in self.seq])
        self.j = d.J_cheap.to_numpy(float)
        aff = np.abs(self.v_all[self.fit]) > EPS
        self.cap = float(np.percentile(np.abs(self.v_all[self.fit][aff]), 95))
        self.vcap_all = np.clip(self.v_all, -self.cap, self.cap)


def k_round(n, q=0.2):
    return max(int(round(q * n)), 1)


def k_floor(n, q):
    return int(np.floor(q * n + 1e-9))


def gain(s, v, k):
    return float(xl.gains(s, v, np.array([k]))[0]) if k > 0 else 0.0


def inclusion(s, k):
    """Each frame's chance of being in the top k under the benchmark's tie rule."""
    s = np.round(np.asarray(s, float), 9)
    o = np.argsort(-s, kind="stable")
    cut = s[o][k - 1]
    p = (s > cut).astype(float)
    tie = s == cut
    above = int((s > cut).sum())
    p[tie] = (k - above) / tie.sum()
    return p


def value(row, s_map, v, j, seq):
    """A row's quantities on one frame set (point)."""
    n = len(v)
    k = row["kfun"](n)
    g = gain(s_map[row["signal"]], v, k)
    kr = row["krand"](n)
    out = {"k": k, "gain": g, "tot": float(j.sum()), "rand_rate": kr / n * v.sum(), "rand_count": k / n * v.sum()}
    if row.get("ref") is not None:
        out["gain_ref"] = gain(s_map[row["ref"]], v, k)
    return out


def quantities(q):
    r = {"realized_share": q["gain"] / q["tot"], "over_random_rate_share": (q["gain"] - q["rand_rate"]) / q["tot"],
         "over_random_count_share": (q["gain"] - q["rand_count"]) / q["tot"]}
    if "gain_ref" in q:
        r["D_share"] = (q["gain_ref"] - q["gain"]) / q["tot"]
    return r


def intervals(row, s_map, v, j, seq, units, tot_point, rng_seed=0):
    uniq, idx = xl.unit_index(units)
    rng = np.random.default_rng(rng_seed)
    keys = None
    for b in range(NBOOT):
        t = np.concatenate([idx[u] for u in uniq[rng.integers(0, len(uniq), len(uniq))]])
        q = quantities(dict(value(row, {k: x[t] for k, x in s_map.items()}, v[t], j[t], seq[t]), tot=tot_point))
        if keys is None:
            keys = {k: np.full(NBOOT, np.nan) for k in q}
        for k, x in q.items():
            keys[k][b] = x
    out = {}
    for k, x in keys.items():
        out[f"{k}_lo"], out[f"{k}_hi"] = xl.ci(x)
    return out


def rows_for(cell):
    """Part C's rows, with the D_t rows of R1 (gate_gbm, raw) as `ref`."""
    costs = pd.read_csv(REB / "R3" / "feature_costs.csv")
    t231 = _load("t231", "231_r3_gate_cost.py")
    c_c, c_f = t231.engines(costs, "ms")
    sel = json.loads((REB / "R3" / "selected.json").read_text())
    rows = [{"row": "X2 / R1 gate_gbm on V", "signal": ("gate_gbm", "raw", "V"), "kfun": k_round, "krand": k_round, "budget": "20%"}]
    for t in TARGETS[1:]:
        rows.append({"row": f"R1 gate_gbm on {t}", "signal": ("gate_gbm", "raw", t), "kfun": k_round, "krand": k_round,
                     "budget": "20%", "ref": ("gate_gbm", "raw", "V")})
    for l in ("gate_gbm", "gate_ridge"):
        sub = sel[l]["subset"]
        f = t231.frac_at(0.2, c_c, c_f, t231.subset_cost(costs, sub, l))
        rows.append({"row": f"R3 {l} ({sub})", "signal": ("R3", l, sub), "kfun": (lambda n, f=f: k_floor(n, f)),
                     "krand": (lambda n: k_floor(n, 0.2)), "budget": "20% latency"})
    return rows


def scene_contrib(s, v, k, seq):
    return pd.Series(inclusion(s, k) * v).groupby(seq).sum()


def stage_cdg(args):
    cell = Cell("mono")
    rows = rows_for(cell)
    t = cell.test
    v, vc, j, seq, log = cell.v_all[t], cell.vcap_all[t], cell.j[t], cell.seq[t], cell.log[t]
    s_map = {k: x[t] for k, x in cell.s.items()}
    sp = meta()
    two_test = sorted(set(seq) & set(sp[sp.log.isin(LOGS)].scene))
    print("test scenes of the two logs:", two_test, "capped at", cell.cap, flush=True)
    sets = {"all": set(), "minus scene-1051": {"scene-1051"}, "minus scene-1081": {"scene-1081"},
            "minus both": {"scene-1051", "scene-1081"}, "minus test scenes of the two logs": set(two_test)}
    res, drop, conc = [], [], []
    full_map = s_map
    for row in rows:
        ask = "D_share" if row.get("ref") else "over_random_rate_share"
        s_map = {k: full_map[k] for k in (row["signal"], row.get("ref")) if k is not None}
        for vname, vv in (("V", v), ("capped V", vc)):
            base = quantities(value(row, s_map, vv, j, seq))
            # drop one scene: the whole range, and the claim's most influential scene
            dd = {}
            for u in sorted(set(seq)):
                m = seq != u
                dd[u] = quantities(value(row, {k: x[m] for k, x in s_map.items()}, vv[m], j[m], seq[m]))
            dq = pd.DataFrame(dd).T
            for q in dq.columns:
                drop.append({"row": row["row"], "decision_value": vname, "quantity": q, "all_scenes": base[q], "min": dq[q].min(),
                             "scene_at_min": dq[q].idxmin(), "max": dq[q].max(), "scene_at_max": dq[q].idxmax()})
            worst = dq[ask].idxmin() if base[ask] >= 0 else dq[ask].idxmax()
            for name, rem in list(sets.items()) + [(f"minus the most influential scene ({worst})", {worst})]:
                m = ~np.isin(seq, sorted(rem))
                sm = {k: x[m] for k, x in s_map.items()}
                q = value(row, sm, vv[m], j[m], seq[m])
                r = {"row": row["row"], "budget": row["budget"], "decision_value": vname, "set": name, "n_frames": int(m.sum()),
                     "escalated": q["k"], "random_at_rate_escalations": row["krand"](int(m.sum())), "gain": q["gain"],
                     "all_cheap_loss": q["tot"], **quantities(q)}
                r.update(intervals(row, sm, vv[m], j[m], seq[m], seq[m], q["tot"]))
                lg = intervals(row, sm, vv[m], j[m], seq[m], log[m], q["tot"])
                r.update({k + "_log": x for k, x in lg.items()})
                res.append(r)
            print(f"  {row['row']} {vname}: {base}", flush=True)
        # concentration of the gain (and of D_t) in the top scenes
        k = row["kfun"](len(v))
        ct = scene_contrib(s_map[row["signal"]], v, k, seq)
        if row.get("ref"):
            ct = scene_contrib(s_map[row["ref"]], v, k, seq) - ct
        o = ct.sort_values(ascending=False)
        total = float(ct.sum())
        conc.append({"row": row["row"], "what": "D_t" if row.get("ref") else "realized gain", "total": total,
                     "negative_scene_sum": float(o[o < 0].sum()),
                     **{f"top{m}_share": float(o.head(m).sum() / total) for m in (1, 3, 5)}, "top5_scenes": " ".join(o.index[:5]),
                     "scene_1051": float(ct.get("scene-1051", 0.0)), "scene_1081": float(ct.get("scene-1081", 0.0))})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res).to_csv(OUT / "partC_D_G_rows.csv", index=False)
    pd.DataFrame(drop).to_csv(OUT / "partD_drop_one.csv", index=False)
    pd.DataFrame(conc).to_csv(OUT / "partD_concentration.csv", index=False)
    # secondary of G: gate_gbm trained on capped V
    t227 = _load("t227", "227_r1_target_swap.py")
    a = argparse.Namespace(det=str(t227.DET), dataroot=args.dataroot)
    cells_, _ = t227.setup(a)
    c = next(x for x in cells_ if x["geometry"] == "mono" and x["system"] == "brake")
    d = c["d"]
    fit = d.split.isin(["train", "val"]).to_numpy()
    tt = (d.split == "test").to_numpy()
    assert (d.seq.astype(str).to_numpy() == cell.seq).all()
    sc = xl.gate_scores(d, c["fcols"], cell.vcap_all, fit, which=("gate_gbm",))["gate_gbm"]
    row = dict(rows[0], signal="capped")
    m = {"capped": sc[tt]}
    out = []
    for vname, vv in (("V", v), ("capped V", vc)):
        q = value(row, m, vv, j, seq)
        r = {"row": "gate_gbm trained on capped V", "decision_value": vname, "cap": cell.cap, "escalated": q["k"], "gain": q["gain"], **quantities(q)}
        r.update(intervals(row, m, vv, j, seq, seq, q["tot"]))
        r.update({k + "_log": x for k, x in intervals(row, m, vv, j, seq, log, q["tot"]).items()})
        out.append(r)
    pd.DataFrame(out).to_csv(OUT / "partG_trained_capped.csv", index=False)


# ------------------------------------------------------------------------------------------------ E

def stage_e(args):
    t227 = _load("t227", "227_r1_target_swap.py")
    a = argparse.Namespace(det=str(t227.DET), dataroot=args.dataroot)
    cells_, log_of = t227.setup(a)
    c = next(x for x in cells_ if x["geometry"] == "mono" and x["system"] == "brake")
    d = c["d"]
    sp = meta()
    v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    j = d[c["cheap"]].to_numpy(float)
    seq = d.seq.astype(str).to_numpy()
    log = np.array([log_of[x] for x in seq])
    tv = d.split.isin(["train", "val"]).to_numpy()
    test = (d.split == "test").to_numpy()
    two = set(sp[sp.log.isin(LOGS)].scene)
    fits = {"537 scenes": tv, "537 minus scene-1049, 1050, 1080": tv & ~np.isin(seq, ["scene-1049", "scene-1050", "scene-1080"]),
            "537 minus every scene of the two logs": tv & ~np.isin(seq, sorted(two)), "385 scenes": (d.split == "train").to_numpy()}
    row = {"signal": "s", "kfun": k_round, "krand": k_round}
    rows = []
    for name, fit in fits.items():
        s = xl.gate_scores(d, c["fcols"], v, fit, which=("gate_gbm",))["gate_gbm"]
        for pop, m in (("228 test", test), ("228 test minus the two logs", test & ~np.isin(seq, sorted(two)))):
            mp = {"s": s[m]}
            q = value(row, mp, v[m], j[m], seq[m])
            r = {"fit": name, "fit_scenes": len(set(seq[fit])), "fit_frames": int(fit.sum()), "population": pop,
                 "scenes": len(set(seq[m])), "escalated": q["k"], "gain": q["gain"], "all_cheap_loss": q["tot"], **quantities(q)}
            r.update(intervals(row, mp, v[m], j[m], seq[m], seq[m], q["tot"]))
            r.update({k + "_log": x for k, x in intervals(row, mp, v[m], j[m], seq[m], log[m], q["tot"]).items()})
            rows.append(r)
            print(f"  {name} | {pop}: {r['realized_share']:+.3f} over random {r['over_random_rate_share']:+.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "partE_fit_sets.csv", index=False)


# ------------------------------------------------------------------------------------------------ F

def stage_f(args):
    t227 = _load("t227", "227_r1_target_swap.py")
    t231 = _load("t231", "231_r3_gate_cost.py")
    a = argparse.Namespace(det=str(t227.DET), dataroot=args.dataroot)
    cells_, log_of = t227.setup(a)
    c = next(x for x in cells_ if x["geometry"] == "mono" and x["system"] == "brake")
    d = c["d"]
    tg = pd.read_csv(REB / "R1" / "targets.csv", dtype={"seq": str})
    arrays = t227.target_arrays(d, tg, "brake")
    arrays["V"] = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    v, j = arrays["V"], d[c["cheap"]].to_numpy(float)
    seq = d.seq.astype(str).to_numpy()
    log = np.array([log_of[x] for x in seq])
    logs = sorted(set(log))
    keep2 = ~np.isin(log, LOGS)
    costs = pd.read_csv(REB / "R3" / "feature_costs.csv")
    c_c, c_f = t231.engines(costs, "ms")
    sub = json.loads((REB / "R3" / "selected.json").read_text())["gate_gbm"]["subset"]
    crit = t231.subset_columns(sub, t231.group_columns(c["fcols"]))
    f_crit = t231.frac_at(0.2, c_c, c_f, t231.subset_cost(costs, sub, "gate_gbm"))
    rows = []
    for seed in range(20):
        t0 = time.time()
        order = np.random.default_rng(seed).permutation(len(logs))
        fold_of = {logs[i]: p % 5 for p, i in enumerate(order)}
        fold = np.array([fold_of[x] for x in log])
        pooled = {t: np.full(len(v), np.nan) for t in TARGETS}
        pooled["crit"] = np.full(len(v), np.nan)
        for kf in range(5):
            fit, held = fold != kf, fold == kf
            for t in TARGETS:
                pooled[t][held] = xl.gate_scores(d, c["fcols"], arrays[t], fit, which=("gate_gbm",))["gate_gbm"][held]
            pooled["crit"][held] = t231.fit_predict(d, crit, v, fit, "gate_gbm")[1][held]
        for pop, m in (("765 scenes", np.ones(len(v), bool)), ("765 minus the two logs", keep2)):
            n, tot = int(m.sum()), float(j[m].sum())
            k = k_round(n)
            gs = {t: gain(pooled[t][m], v[m], k) for t in TARGETS}
            rnd = k / n * v[m].sum()
            r = {"seed": seed, "population": pop, "n_frames": n, "all_cheap_loss": tot, "gain_V_share": gs["V"] / tot,
                 "over_random_V_share": (gs["V"] - rnd) / tot}
            r.update({f"D_{t}": (gs["V"] - gs[t]) / tot for t in TARGETS[1:]})
            kc = k_floor(n, f_crit)
            gc = gain(pooled["crit"][m], v[m], kc)
            r.update(crit_gain_share=gc / tot, crit_over_random_share=(gc - k_floor(n, 0.2) / n * v[m].sum()) / tot, crit_k=kc)
            rows.append(r)
        print(f"  seed {seed}: {rows[-2]['gain_V_share']:+.3f} [{time.time() - t0:.0f}s]", flush=True)
        pd.DataFrame(rows).to_csv(WORK / "partF_partial.csv", index=False)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "partF_oof.csv", index=False)
    cols = ["gain_V_share", "over_random_V_share", "D_fn_gain", "D_risk_gain", "D_Q", "D_G", "crit_gain_share", "crit_over_random_share"]
    s = []
    for pop, g in t.groupby("population"):
        for q in cols:
            s.append({"population": pop, "quantity": q, "mean": g[q].mean(), "min": g[q].min(), "max": g[q].max(),
                      "repeats_above_0": int((g[q] > 0).sum()), "share_above_0": float((g[q] > 0).mean())})
    pd.DataFrame(s).to_csv(OUT / "partF_summary.csv", index=False)


# ------------------------------------------------------------------------------------------------ H

def stage_h(args):
    sp = meta()
    log_of = dict(zip(sp.scene, sp.log))
    pops = populations(sp)
    d = table("mono")
    for t in ("cheap", "full"):
        chk = d[[f"Jterm_{t}_{k}" for k in ("shortfall", "collision", "excess", "jerk")]].sum(axis=1) - d[f"J_{t}"]
        assert chk.abs().max() < 1e-9
    B0, C0 = 0.12, 6.0

    def J(x, t, b, c):
        return (x[f"Jterm_{t}_shortfall"] + x[f"Jterm_{t}_jerk"] + b / B0 * x[f"Jterm_{t}_excess"]
                + c / C0 * x[f"Jterm_{t}_collision"]).to_numpy(float)
    grid = [("over-braking", b, 6.0) for b in (0, 0.03, 0.06, 0.12, 0.24, 0.48)]
    grid += [("collision", 0.12, c) for c in (0, 3, 6, 12)]
    grid += [("joint", b, c) for b in (0, 0.48) for c in (0, 12)]
    tasks, cross = [], []
    for pop in ("pooled850", "new228_test"):
        x = d[d.seq.isin(pops[pop])]
        seq = x.seq.to_numpy()
        lg = np.array([log_of[s] for s in seq])
        for fam, b, c in grid:
            jc, jf = J(x, "cheap", b, c), J(x, "full", b, c)
            tasks.append(({"population": pop, "family": fam, "over_braking_weight": b, "collision_weight": c}, jc - jf, jc, seq, lg))

        def rho(b):
            v = J(x, "cheap", b, 6.0) - J(x, "full", b, 6.0)
            return -v[v < -EPS].sum() / v[v > EPS].sum()
        lo, hi = 0.0, 0.48
        rl, rh = rho(lo), rho(hi)
        if (rl - 1) * (rh - 1) < 0:
            for _ in range(60):
                mid = (lo + hi) / 2
                if (rho(mid) - 1) * (rl - 1) > 0:
                    lo = mid
                else:
                    hi = mid
            w = (lo + hi) / 2
        else:
            w = np.nan
        cross.append({"population": pop, "rho_at_0": rl, "rho_at_0.48": rh, "weight_where_rho_crosses_1": w})
    with Pool(NPROC) as p:
        rows = p.map(_a_task, tasks, chunksize=1)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "partH_weights.csv", index=False)
    pd.DataFrame(cross).to_csv(OUT / "partH_rho_crossing.csv", index=False)
    print(pd.DataFrame(cross).to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ reading

def stage_reading(args):
    c = pd.read_csv(OUT / "partC_D_G_rows.csv")
    f = pd.read_csv(OUT / "partF_summary.csv")
    f = f[f.population == "765 scenes"].set_index("quantity")
    c = c[c.decision_value == "V"]
    rows = []
    claims = [("X2 primary: gate_gbm over random at 20%", "X2 / R1 gate_gbm on V", "over_random_rate_share", "over_random_V_share"),
              ("R1 D_t fn_gain", "R1 gate_gbm on fn_gain", "D_share", "D_fn_gain"), ("R1 D_t risk_gain", "R1 gate_gbm on risk_gain", "D_share", "D_risk_gain"),
              ("R1 D_t Q", "R1 gate_gbm on Q", "D_share", "D_Q"), ("R1 D_t G", "R1 gate_gbm on G", "D_share", "D_G"),
              ("R3 primary: chosen gate_gbm subset over random, 20% latency", "R3 gate_gbm (crit)", "over_random_rate_share", "crit_over_random_share")]
    for name, row, q, fq in claims:
        x = c[c.row == row]
        a = x[x.set == "all"].iloc[0]
        m = x[x.set.str.startswith("minus the most influential")].iloc[0]
        rep = int(f.loc[fq, "repeats_above_0"])
        same = m[f"{q}_lo"] > 0
        point = m[q] > 0
        reading = ("robust" if same and rep >= 18 else "tail-dependent" if point and rep >= 11 else "fragile")
        rows.append({"claim": name, "value": a[q], "lo": a[f"{q}_lo"], "hi": a[f"{q}_hi"], "lo_log": a[f"{q}_lo_log"], "hi_log": a[f"{q}_hi_log"],
                     "removed": m.set, "value_after": m[q], "lo_after": m[f"{q}_lo"], "hi_after": m[f"{q}_hi"], "lo_after_log": m[f"{q}_lo_log"],
                     "hi_after_log": m[f"{q}_hi_log"], "repeats_above_0_of_20": rep, "reading": reading})
    a = pd.read_csv(OUT / "partA_settings.csv")
    a = a[(a.geometry == "mono") & (a.system == "brake") & (a.population == "pooled850")].set_index("subset")
    al, ne, po = a.loc["all"], a.loc["minus 1 most negative"], a.loc["minus 1 most positive"]
    excl = lambda r: r.all_full_reduction_lo > 0 or r.all_full_reduction_hi < 0
    rows.append({"claim": "all-full is not resolved on the 850 scenes", "value": al.all_full_reduction, "lo": al.all_full_reduction_lo,
                 "hi": al.all_full_reduction_hi, "lo_log": al.all_full_reduction_lo_log, "hi_log": al.all_full_reduction_hi_log,
                 "removed": "1 most negative scene / 1 most positive scene", "value_after": ne.all_full_reduction, "lo_after": ne.all_full_reduction_lo,
                 "hi_after": ne.all_full_reduction_hi, "lo_after_log": ne.all_full_reduction_lo_log, "hi_after_log": ne.all_full_reduction_hi_log,
                 "repeats_above_0_of_20": np.nan, "value_after_positive_side": po.all_full_reduction, "lo_after_positive_side": po.all_full_reduction_lo,
                 "hi_after_positive_side": po.all_full_reduction_hi, "reading": "fragile" if excl(ne) or excl(po) else "robust"})
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "reading.csv", index=False)
    print(r[["claim", "value", "lo", "hi", "value_after", "lo_after", "hi_after", "repeats_above_0_of_20", "reading"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["a", "b", "cdg", "e", "f", "h", "reading"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    runmeta.new_run(f"r7_{args.stage}", vars(args))
    {"a": stage_a, "b": stage_b, "cdg": stage_cdg, "e": stage_e, "f": stage_f, "h": stage_h, "reading": stage_reading}[args.stage](args)


if __name__ == "__main__":
    main()
