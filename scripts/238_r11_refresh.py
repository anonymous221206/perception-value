#!/usr/bin/env python
"""Follow-up round 3, R11: refresh scheduling with real staleness (addendum R11 in docs/followup_protocol.md).  A new cell type, separate from E10b.

Before any current-frame perception a policy either runs the 640 detector or reuses the last detected list carried to this
frame with ego motion (E10's `ego`).  Maximum age 10 frames, then a forced refresh, charged to the budget.

  --stage tables   per unit, the action at t from the list of frame t - a, a = 0..10; gate R11-0 (a = 0, 1 equal E10's branches)
  --stage oracle   dynamic programming over (t, age) with a shared multiplier, refresh shares 10, 20, 30, 50% -> oracle.csv
  --stage policies periodic, random, age x speed, gate_gbm with age, the sequential learner; scored by replay -> policies.csv, primary.csv
"""
from __future__ import annotations

import argparse, importlib.util, json, os, pickle, sys, time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):                                   # before numpy, as 236
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""                                             # CPU only, as 236

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, egospeed, geometry as G, kitti, planner as P, predict, runmeta, xl   # noqa: E402
from rap.cache import DetCache                                                                 # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                              # noqa: E402
from rap.risk import RiskConfig                                                                # noqa: E402
from rap.scoring import EPS                                                                    # noqa: E402

OUT = Path(RESULTS) / "followup" / "R11"
WORK = Path(CACHE) / "xl" / "r11"
AGES = 11
BUDGETS = (0.10, 0.20, 0.30, 0.50)
NBOOT, NPROC, SEEDS_RANDOM, GRID = 1000, 6, 20, 41
CREDIT = (-3, 5)
FORBIDDEN = ("decision_margin", "boundary_distance", "a_cheap", "a_full", "a_gt", "act_")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


t206 = _load("t206", "206_temporal_escalation.py")
DATASETS = {"KITTI": "full_640", "nuScenes": "ns_full_640"}
_C = {}


def _setup(ds):
    if ds == "nuScenes":
        from rap.nusc import NuScenesDB, make_adapter
        db = NuScenesDB(str(NUSCENES_XL), "v1.0-trainval")
        by = {db.scene_name(s): s for s in db.scenes}
        _C.update(adapter=make_adapter(db), poses=lambda s: t206.nusc_poses(db, by, s))
    else:
        _C.update(adapter=decision.KittiAdapter, poses=t206.kitti_poses)


def _unit(t):
    """Requirement and action of every (t, age) of one unit, as E10's build with a list carried over `age` frames."""
    ds, s, geom_src = t
    cfg, pp = RiskConfig(), P.PlannerParams()
    mode = DATASETS[ds]
    geom, speeds = _C["adapter"].geometry(s), _C["adapter"].speeds(s)
    pose, _ = _C["poses"](s)
    c = DetCache(t206.DETS[ds] / mode / f"{s}.npz")
    thr = cfg.thr(t206.role(mode))
    rng = np.random.default_rng(0)
    n = len(c.frames)
    a_ = np.full((n, AGES), np.nan)
    act = np.full((n, AGES), -1, np.int8)
    a_gt, v_ = np.zeros(n), np.zeros(n)
    lists, frs = [], []
    for i, fr in enumerate(c.frames):
        fr = int(fr)
        v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
        g = geom[geom["frame"] == fr]
        g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
        d, geo = c.det(i), c.geo(i)
        k = d["conf"] >= thr
        fresh = decision._apply_range_source(d["xyxy"][k].astype(float), {x: y[k] for x, y in geo.items()}, g, geom_src, rng, 0.0)
        lists.append(fresh)
        frs.append(fr)
        for age in range(min(i, AGES - 1) + 1):
            cur = fresh if age == 0 else t206.carry(lists[i - age], pose[frs[i - age]], pose[fr])
            a_[i, age] = P.required_decel(*cur, v, pp)[0]
            act[i, age] = P.discrete_action(a_[i, age], pp)
        a_gt[i] = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)[0]
        v_[i] = v
    return s, {"frames": np.array(frs), "a": a_, "act": act, "a_gt": a_gt, "v": v_}


def table_file(ds, geom):
    return WORK / f"table_{ds}_{geom}.pkl"


def stage_tables(args):
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    gate = {}
    fails = 0
    for ds, mode in DATASETS.items():
        _setup(ds)
        seqs = sorted(p.stem for p in (t206.DETS[ds] / mode).glob("*.npz")) if ds == "nuScenes" else kitti.sequences()
        for geom in ("mono", "oracle"):
            t0 = time.time()
            with Pool(3 if ds == "nuScenes" else NPROC) as p:                       # the nuScenes database is forked: three fit in memory
                tab = dict(p.map(_unit, [(ds, s, geom) for s in seqs], chunksize=4))
            table_file(ds, geom).write_bytes(pickle.dumps(tab, protocol=pickle.HIGHEST_PROTOCOL))
            e = pd.read_pickle(Path(CACHE) / "xl" / "e10" / "tables" / f"{ds}__{mode}__ego__{geom}.pkl")
            n, bad = 0, 0
            for s, g in e.groupby("seq", sort=False):
                u = tab[str(s)]
                assert (u["frames"] == g.frame.to_numpy(int)).all()
                ns = g["first"].to_numpy(int) == 0
                bad += int((u["a"][:, 0] != g.a_fresh.to_numpy(float)).sum() + (u["act"][:, 0] != g.act_fresh.to_numpy(int)).sum())
                bad += int((u["a"][ns, 1] != g.a_stale.to_numpy(float)[ns]).sum() + (u["act"][ns, 1] != g.act_stale.to_numpy(int)[ns]).sum())
                bad += int((u["a_gt"] != g.a_gt.to_numpy(float)).sum())
                n += 3 * len(g) + 2 * int(ns.sum())
            gate[f"{ds}|{geom}"] = {"compared": n, "differ": bad}
            fails += bad
            print(f"  {ds} {geom}: {gate[f'{ds}|{geom}']} [{time.time() - t0:.0f}s]", flush=True)
    rep = {"cells": gate, "pass": fails == 0}
    (OUT / "gate_r11_0.json").write_text(json.dumps(rep, indent=1))
    print("gate R11-0:", rep, flush=True)
    if fails:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ costs

def costs(u):
    """base[t, age] and the command, the two parts of the benchmark's loss; the jerk term joins them at a transition."""
    pp, cp = P.PlannerParams(), P.CostParams()
    cmd_ = np.array([P.commanded_decel(a, pp) for a in range(3)] + [np.nan])
    c = cmd_[u["act"]]                                               # -1 -> nan
    short, exc = np.maximum(0.0, u["a_gt"][:, None] - c), np.maximum(0.0, c - u["a_gt"][:, None])
    base = cp.lam_risk * short ** 2 + cp.lam_brake * exc ** 2 + cp.lam_collision * (short >= cp.collision_a - 1e-9)
    return base, c, cp.lam_jerk


def dp(u, lam):
    """Least of loss + lam * refreshes; returns (loss, refreshes, z)."""
    base, c, wj = u["cost"]
    n = len(base)
    val = np.full(AGES, np.inf)
    val[0] = base[0, 0] + lam
    back = np.zeros(n, np.int8)
    nr = np.zeros((n, AGES), np.int32)
    nr[0, 0] = 1
    for t in range(1, n):
        new = np.full(AGES, np.inf)
        ok = np.isfinite(val)
        tr = val + wj * np.abs(c[t, 0] - np.nan_to_num(c[t - 1])) + base[t, 0]
        a = int(np.argmin(np.where(ok, tr, np.inf)))
        new[0] = tr[a] + lam
        nr[t, 0] = nr[t - 1, a] + 1
        back[t] = a
        h = t if t < AGES else AGES - 1
        new[1:h + 1] = (val[:h] + wj * np.abs(c[t, 1:h + 1] - np.nan_to_num(c[t - 1, :h])) + base[t, 1:h + 1])
        nr[t, 1:h + 1] = nr[t - 1, :h]
        val = new
    a = int(np.nanargmin(val))
    return float(val[a]) - lam * int(nr[n - 1, a]), int(nr[n - 1, a]), (a, back)


def path(u, end):
    """The refresh decisions behind dp's result."""
    a, back = end
    n = len(back)
    z = np.zeros(n, bool)
    for t in range(n - 1, -1, -1):
        if a == 0:
            z[t] = True
            a = int(back[t])
        else:
            a -= 1
    return z


def solve(units, k):
    """Shared multiplier: refreshes at most k. Returns (feasible loss, count, lam, dual bound, per-unit z)."""
    def run(lam):
        r = [dp(u, lam) for u in units]
        return sum(x[0] for x in r), sum(x[1] for x in r), r
    lo, hi = 0.0, 1.0
    l0, c0, r0 = run(0.0)
    if c0 <= k:
        return l0, c0, 0.0, l0, [path(u, x[2]) for u, x in zip(units, r0)]
    while True:
        l1, c1, r1 = run(hi)
        if c1 <= k or hi > 1e7:
            break
        lo, hi = hi, hi * 4
    if c1 > k:
        return np.nan, c1, hi, np.nan, None                          # below the forced refreshes
    for _ in range(40):
        m = 0.5 * (lo + hi)
        l_, c_, r_ = run(m)
        if c_ > k:
            lo = m
        else:
            hi, l1, c1, r1 = m, l_, c_, r_
    return l1, c1, hi, l1 + hi * (c1 - k), [path(u, x[2]) for u, x in zip(units, r1)]


def replay(u, z):
    """Loss of a refresh schedule with its true staleness and its own action history; z[0] is True."""
    base, c, wj = u["cost"]
    n, loss, age, prev = len(z), 0.0, 0, None
    for t in range(n):
        age = 0 if z[t] else age + 1
        assert age < AGES
        loss += base[t, age] + (0.0 if prev is None else wj * abs(c[t, age] - prev))
        prev = c[t, age]
    return float(loss)


def minimal(n):
    z = np.zeros(n, bool)
    z[np.arange(0, n, AGES)] = True
    return z


def periodic(n, q):
    p = min(max(int(round(1 / q)), 1), AGES)
    z = np.zeros(n, bool)
    z[np.arange(0, n, p)] = True
    return z


def load_units(ds, geom="mono"):
    tab = pickle.loads(table_file(ds, geom).read_bytes())
    for u in tab.values():
        u["cost"] = costs(u)
    if ds == "KITTI":
        sp = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["kitti"]
        split = {s: k for k in ("train", "val", "test") for s in sp[k]}
        groups = {s: s for s in tab}
    else:
        x = xl.scene_split()
        new = x[x.group == "new765"]
        split = dict(zip(new.scene, new.deep_split))
        m = xl.scene_meta(str(NUSCENES_XL))
        groups = dict(zip(m.scene, m.log))
    return tab, split, groups




def oracle_rows(ds, tab, split, pops):
    rows = []
    sol = {}
    for pop, names in pops.items():
        units = [tab[s] for s in names]
        n = sum(len(u["v"]) for u in units)
        mn = sum(replay(u, minimal(len(u["v"]))) for u in units)
        for q in BUDGETS:
            k = max(int(round(q * n)), 1)
            t0 = time.time()
            loss, cnt, lam, dual, z = solve(units, k)
            per = sum(replay(u, periodic(len(u["v"]), q)) for u in units)
            pc = sum(int(periodic(len(u["v"]), q).sum()) for u in units)
            if z is not None:
                rp = sum(replay(u, zz) for u, zz in zip(units, z))
                assert abs(rp - loss) < 1e-6, (rp, loss)
            rows.append({"dataset": ds, "population": pop, "budget": q, "k": k, "frames": n, "units": len(units), "minimal_loss": mn,
                         "oracle_loss": loss, "oracle_refreshes": cnt, "multiplier": lam, "dual_bound_loss": dual, "gap": loss - dual,
                         "oracle_gain": mn - loss, "periodic_gain": mn - per, "periodic_refreshes": pc})
            sol[(pop, q)] = z
            print(f"  {ds} {pop} {q}: oracle {mn - loss:.2f}, periodic {mn - per:.2f}, refreshes {cnt} of {k} [{time.time() - t0:.0f}s]", flush=True)
    return rows, sol


def stage_oracle(args):
    runmeta.new_run("r11_oracle", vars(args))
    rows = []
    for ds in DATASETS:
        tab, split, _ = load_units(ds)
        names = lambda k: sorted(s for s in tab if split.get(s) == k)    # noqa: E731
        pops = {"train": names("train"), "val": names("val"), "test": names("test")}
        if ds == "KITTI":
            pops["all 21"] = sorted(tab)
        r, sol = oracle_rows(ds, tab, split, pops)
        rows += r
        (WORK / f"oracle_{ds}.pkl").write_bytes(pickle.dumps({"names": pops, "z": sol}, protocol=pickle.HIGHEST_PROTOCOL))
    pd.DataFrame(rows).to_csv(OUT / "oracle.csv", index=False)


# ------------------------------------------------------------------------------------------------ policies

def features(ds):
    from rap.tables import feature_columns
    f = xl.t92().g84.build_features(t206.DETS[ds], DATASETS[ds], RiskConfig())
    cols = feature_columns(f)
    assert not [c for c in cols if any(x in c for x in FORBIDDEN)]
    return f.assign(seq=f.seq.astype(str), frame=f.frame.astype(int)), cols


def attach(tab, feats, cols, ds):
    if ds == "nuScenes":
        xl.use_ego_table(Path(CACHE) / "xl" / "tables850" / "v_ego_causal.csv")
    fr = pd.concat([pd.DataFrame({"seq": s, "frame": u["frames"], "v_ego": u["v"]}) for s, u in tab.items()], ignore_index=True)
    e = egospeed.attach(fr.copy(), ds)
    fj = fr.merge(feats, on=["seq", "frame"], how="left", validate="one_to_one")
    X = np.nan_to_num(fj[cols].to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    v = e[egospeed.COL].to_numpy(float)
    i = 0
    for s, u in tab.items():
        n = len(u["v"])
        u["X"], u["vc"] = X[i:i + n], v[i:i + n]
        i += n


def gbm_params():
    p = predict.make_model("gbm", "reg", 0).get_params()
    return {k: p[k] for k in ("max_depth", "max_iter", "learning_rate", "min_samples_leaf", "l2_regularization", "max_bins", "early_stopping", "random_state")}


def state_rows(u, ts, ages, credit=None):
    """Features of the decision at t with a list of the given age: the list's features, age, causal ego speed, budget left."""
    x = [u["X"][ts - ages], ages[:, None].astype(float), u["vc"][ts][:, None]]
    if credit is not None:
        x.append(credit[:, None].astype(float))
    return np.hstack(x)


def run_policy(u, score, tau, q):
    """A causal run: forced at age 10; otherwise refresh if score(t, age, credit) > tau and the running cap allows."""
    n = len(u["v"])
    z = np.zeros(n, bool)
    z[0] = True
    used, age = 1, 0
    for t in range(1, n):
        age += 1
        if age >= AGES:
            z[t], age = True, 0
            used += 1
            continue
        c = int(np.clip(np.floor(q * (t + 1)) - used, *CREDIT))
        if score(t, age, c) > tau and used < int(np.floor(1 + q * (t + 1))):
            z[t], age = True, 0
            used += 1
    return z


def tau_on_val(units, score, q, k, grid):
    best = None
    mn = sum(replay(u, minimal(len(u["v"]))) for u in units)
    for tau in grid:
        zs = [run_policy(u, score(u), tau, q) for u in units]
        cnt = sum(int(z.sum()) for z in zs)
        g = mn - sum(replay(u, z) for u, z in zip(units, zs))
        key = (cnt <= k, g if cnt <= k else -cnt)
        if best is None or key > best[0]:
            best = (key, float(tau), cnt, g)
    return best


def bootstrap(a, b, groups, seed=0):
    """Interval of the sum of a minus the sum of b over group draws."""
    s = pd.Series(a - b).groupby(np.asarray(groups)).sum().to_numpy()
    r = np.random.default_rng(seed)
    return xl.ci(np.array([s[r.integers(0, len(s), len(s))].sum() for _ in range(NBOOT)]))


def stage_policies(args):
    runmeta.new_run("r11_policies", vars(args))
    rows, prim = [], []
    for ds in DATASETS:
        t0 = time.time()
        tab, split, grp = load_units(ds)
        feats, cols = features(ds)
        attach(tab, feats, cols, ds)
        names = {k: sorted(s for s in tab if split.get(s) == k) for k in ("train", "val", "test")}
        sol = pickle.loads((WORK / f"oracle_{ds}.pkl").read_bytes())
        units = {k: [tab[s] for s in v] for k, v in names.items()}
        # gate_gbm with age added: V of every (t, age) of the training units, list's features + speed + age
        Xg, yg = [], []
        for u in units["train"]:
            base, c, wj = u["cost"]
            n = len(base)
            for age in range(1, AGES):
                ts = np.arange(age, n)
                if not len(ts):
                    continue
                Xg.append(state_rows(u, ts, np.full(len(ts), age)))
                yg.append(base[ts, age] - base[ts, 0])
        gate = predict.make_model("gbm", "reg", 0)
        with threadpool_limits(limits=1, user_api="openmp"):
            gate.fit(np.vstack(Xg), np.concatenate(yg))
        for u in units["val"] + units["test"]:
            n = len(u["v"])
            S = np.full((n, AGES), -np.inf)
            for age in range(1, AGES):
                ts = np.arange(age, n)
                if len(ts):
                    S[ts, age] = gate.predict(state_rows(u, ts, np.full(len(ts), age)))
            u["S_gate"] = S
            u["S_speed"] = np.arange(AGES)[None, :] * u["vc"][:, None]
        for q in BUDGETS:
            # the sequential learner: imitates the oracle's decisions on the training units' visited states
            zs = sol["z"].get(("train", q))
            if zs is None:
                print(f"  {ds} {q}: infeasible on train", flush=True)
                continue
            Xs, ys = [], []
            for u, z in zip(units["train"], zs):
                n = len(z)
                age, used, ages, cred = 0, 1, np.zeros(n, int), np.zeros(n, int)
                ts = []
                for t in range(1, n):
                    age += 1
                    if age >= AGES:
                        age, used = 0, used + 1                      # forced: no decision here
                        continue
                    ts.append(t)
                    ages[t], cred[t] = age, np.clip(np.floor(q * (t + 1)) - used, *CREDIT)
                    if z[t]:
                        age, used = 0, used + 1
                ts = np.array(ts, int)
                if len(ts):
                    Xs.append(state_rows(u, ts, ages[ts], cred[ts]))
                    ys.append(z[ts].astype(int))
            y = np.concatenate(ys)
            m = HistGradientBoostingClassifier(**gbm_params())
            with threadpool_limits(limits=1, user_api="openmp"):
                m.fit(np.vstack(Xs), y)
            creds = np.arange(CREDIT[0], CREDIT[1] + 1)
            every = units["val"] + units["test"]
            for u in every:
                u["S_seq"] = np.full((len(u["v"]), AGES, len(creds)), -np.inf)
            for age in range(1, AGES):                                # one batch per age and credit: a single predict call each
                parts = [(u, np.arange(age, len(u["v"]))) for u in every if len(u["v"]) > age]
                base_ = np.vstack([state_rows(u, ts, np.full(len(ts), age)) for u, ts in parts])
                for ic, cv in enumerate(creds):
                    s_ = m.decision_function(np.hstack([base_, np.full((len(base_), 1), float(cv))]))
                    i = 0
                    for u, ts in parts:
                        u["S_seq"][ts, age, ic] = s_[i:i + len(ts)]
                        i += len(ts)
            pol = {"age x speed": (lambda u: (lambda t, a, c, S=u["S_speed"]: S[t, a]), "S_speed"),
                   "gate_gbm with age": (lambda u: (lambda t, a, c, S=u["S_gate"]: S[t, a]), "S_gate"),
                   "sequential learner": (lambda u: (lambda t, a, c, S=u["S_seq"]: S[t, a, c - CREDIT[0]]), "S_seq")}
            nv = sum(len(u["v"]) for u in units["val"])
            ktest = max(int(round(q * sum(len(u["v"]) for u in units["test"]))), 1)
            test = units["test"]
            ntest = sum(len(u["v"]) for u in test)
            gr = [grp[s] for s in names["test"]]
            mn = np.array([replay(u, minimal(len(u["v"]))) for u in test])
            tot_cost = float(mn.sum())
            res = {"periodic": ([periodic(len(u["v"]), q) for u in test], None)}
            for pn, (fn, key) in pol.items():
                vals = np.concatenate([u[key][np.isfinite(u[key])] for u in units["val"]])
                grid = np.unique(np.quantile(vals, np.linspace(0, 1, GRID)))
                b = tau_on_val(units["val"], fn, q, max(int(round(q * nv)), 1), grid)
                res[pn] = ([run_policy(u, fn(u), b[1], q) for u in test], b[1])
            # random: Bernoulli share found on the count only
            rng0 = lambda s: np.random.default_rng(s)                    # noqa: E731

            def rand_sched(p, seed):
                r = rng0(seed)
                out = []
                for u in test:
                    n = len(u["v"])
                    z = np.zeros(n, bool)
                    z[0] = True
                    age = 0
                    for t in range(1, n):
                        age += 1
                        if age >= AGES or r.random() < p:
                            z[t], age = True, 0
                    out.append(z)
                return out
            lo, hi = 0.0, 1.0
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                c = np.mean([sum(int(z.sum()) for z in rand_sched(mid, s)) for s in range(3)])
                lo, hi = (mid, hi) if c < ktest else (lo, mid)
            per_unit = {}
            for pn, (zs_, tau) in res.items():
                g = mn - np.array([replay(u, z) for u, z in zip(test, zs_)])
                per_unit[pn] = g
                c = sum(int(z.sum()) for z in zs_)
                lo_, hi_ = bootstrap(g, np.zeros(len(g)), gr)
                rows.append({"dataset": ds, "budget": q, "policy": pn, "tau": tau, "refreshes": c, "k": ktest, "within_budget": c <= ktest, "share_refreshed": c / ntest,
                             "gain": float(g.sum()), "gain_share": float(g.sum()) / tot_cost, "gain_lo": lo_ / tot_cost, "gain_hi": hi_ / tot_cost})
            g = np.mean([mn - np.array([replay(u, z) for u, z in zip(test, rand_sched(lo, s))]) for s in range(SEEDS_RANDOM)], 0)
            per_unit["random"] = g
            c = float(np.mean([sum(int(z.sum()) for z in rand_sched(lo, s)) for s in range(SEEDS_RANDOM)]))
            rows.append({"dataset": ds, "budget": q, "policy": "random", "tau": lo, "refreshes": c, "k": ktest, "within_budget": c <= ktest, "share_refreshed": c / ntest,
                         "gain": float(g.sum()), "gain_share": float(g.sum()) / tot_cost})
            zo = sol["z"].get(("test", q))
            if zo is not None:
                go = mn - np.array([replay(u, z) for u, z in zip(test, zo)])
                rows.append({"dataset": ds, "budget": q, "policy": "oracle (hindsight)", "refreshes": sum(int(z.sum()) for z in zo), "k": ktest, "share_refreshed": sum(int(z.sum()) for z in zo) / ntest,
                             "gain": float(go.sum()), "gain_share": float(go.sum()) / tot_cost})
            for other in ("periodic", "random", "gate_gbm with age", "age x speed"):
                d = per_unit["sequential learner"] - per_unit[other]
                x_lo, x_hi = bootstrap(per_unit["sequential learner"], per_unit[other], gr)
                prim.append({"dataset": ds, "budget": q, "comparison": f"sequential learner - {other}", "units": len(set(gr)), "difference": float(d.sum()), "difference_share": float(d.sum()) / tot_cost,
                             "lo": x_lo / tot_cost, "hi": x_hi / tot_cost, "reading": "sequential better" if x_lo > 0 else ("sequential worse" if x_hi < 0 else "not resolved")})
            print(f"  {ds} {q} [{time.time() - t0:.0f}s]", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "policies.csv", index=False)
    p = pd.DataFrame(prim)
    p.to_csv(OUT / "primary.csv", index=False)
    print(t.to_string(), "\n", p.to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("tables", "oracle", "policies"))
    args = ap.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
