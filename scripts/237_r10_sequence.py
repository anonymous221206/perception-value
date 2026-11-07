#!/usr/bin/env python
"""Follow-up round 3, R10: what the additive, fixed-history evaluation misses, and a decision-mask scorer
(addendum R10 in docs/followup_protocol.md).

  --stage gate    gate R10-0: 150's G1' with its one write under results/final/ redirected, compared with the stored file; the
                  shipped examples' scored files byte-identical
  --stage replay  Part A: replayed gain of a mask against the additive sum -> replay.csv
  --stage oracle  Part B: sequence oracle by dynamic programming with a shared multiplier -> sequence_oracle.csv
  --stage masks   Part C: six learned signals and R9's chosen learner under policies A and B as masks -> masks_xl.csv, masks_frozen.csv
  --stage masks_frozen  the frozen part of Part C alone, in a fresh process (masks execs it)

The loss of a braking action depends on the action at t and at t-1 only, so J(b | p) for branch b at t and branch p at t-1 is
a 2 x 2 table per frame, and a mask replays exactly with its own history.  This covers the controller's action memory only.
"""
from __future__ import annotations

import argparse, filecmp, importlib.util, json, os, pickle, subprocess, sys, tempfile, time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):                                   # before numpy, as 236
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""                                             # CPU only, as 236: forked workers cannot initialize CUDA

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import planner as P, runmeta, scoring, submission as S, xl               # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402

OUT = Path(RESULTS) / "followup" / "R10"
WORK = Path(CACHE) / "xl" / "r10"
REB = Path(RESULTS) / "followup"
NBOOT, NPROC = 1000, 6
BUDGETS = (0.10, 0.20, 0.30, 0.50)
LEARNED = ("gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf")
EPS = scoring.EPS


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


# ------------------------------------------------------------------------------------------------ gate

def stage_gate(args):
    t150 = _load("t150", "150_submission_tables.py")
    final = str(t150.FINAL)
    tmp = Path(tempfile.mkdtemp(prefix="r10_", dir=str(WORK.parent if WORK.parent.exists() else "/tmp")))
    orig = pd.DataFrame.to_csv

    def redirected(self, path_or_buf=None, *a, **k):
        if path_or_buf is not None and str(path_or_buf).startswith(final):
            path_or_buf = tmp / Path(path_or_buf).name
        return orig(self, path_or_buf, *a, **k)
    pd.DataFrame.to_csv = redirected
    sys.argv = ["150", "--stage", "g1"]
    t0 = time.time()
    code = 0
    try:
        t150.main()
    except SystemExit as e:
        code = e.code
    pd.DataFrame.to_csv = orig
    stored = t150.FINAL / "submission_path_g1.csv"
    same = filecmp.cmp(tmp / "submission_path_g1.csv", stored, shallow=False)
    ex = {}
    for name in ("random", "confidence"):
        sub = ROOT / "examples" / "submissions" / f"{name}.csv"
        o = tmp / f"{name}_scored.csv"
        cmd = [sys.executable, str(ROOT / "evaluate_submission.py"), str(sub), "--out", str(o)]
        prof = sub.with_name(f"{name}_cost_profile.json")
        if prof.exists():
            cmd += ["--cost_profile", str(prof)]
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        ex[name] = filecmp.cmp(o, sub.with_name(f"{name}_scored.csv"), shallow=False)
    g = pd.read_csv(tmp / "submission_path_g1.csv")
    rep = {"g1_exit": code, "g1_file_identical": same, "examples_identical": ex, "compared": int(g.rows.sum()), "differing": int(g.differing.sum()),
           "structural": int(g.structural.fillna(0).sum()), "seconds": time.time() - t0}
    rep["pass"] = bool(same and code in (0, None) and all(ex.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_r10_0.json").write_text(json.dumps(rep, indent=1))
    print("gate R10-0:", rep, flush=True)
    if not rep["pass"]:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ tables

def j_table(d):
    """J[t, b, p]: the loss at t of branch b (0 cheap, 1 full) after branch p at t-1; rows sorted by unit and frame."""
    pp, cp = P.PlannerParams(), P.CostParams()
    cmd = np.array([P.commanded_decel(a, pp) for a in range(3)])
    ac = np.stack([d.act_cheap.to_numpy(int), d.act_full.to_numpy(int)], 1)
    agt = d.a_gt.to_numpy(float)
    n = len(d)
    J = np.zeros((n, 2, 2))
    first = np.r_[True, d.seq.to_numpy()[1:] != d.seq.to_numpy()[:-1]]
    for b in (0, 1):
        c = cmd[ac[:, b]]
        short, exc = np.maximum(0.0, agt - c), np.maximum(0.0, c - agt)
        base = cp.lam_risk * short ** 2 + cp.lam_brake * exc ** 2 + cp.lam_collision * (short >= cp.collision_a - 1e-9)
        for p in (0, 1):
            prev = np.r_[0.0, cmd[ac[:-1, p]]]
            jerk = np.where(first, 0.0, np.abs(c - prev))
            J[:, b, p] = base + cp.lam_jerk * jerk
    return J, first


def load_xl(geometry):
    t = pd.read_pickle(Path(CACHE) / "xl" / "tables850" / f"{geometry}__{xl.PAIR}.pkl")
    t = t[t.J_cheap.notna()].astype({"seq": str, "frame": int}).sort_values(["seq", "frame"]).reset_index(drop=True)
    J, first = j_table(t)
    assert np.abs(J[:, 0, 0] - t.J_cheap.to_numpy(float)).max() < 1e-9
    assert np.abs(J[:, 1, 0] - t.J_full.to_numpy(float)).max() < 1e-9
    return t, J, first


def load_kitti(geometry):
    t92 = xl.t92()
    t = pd.read_pickle(Path(t92.CORE) / f"KITTI__YOLOv8s__cheap_320tofull_640__{geometry}.pkl")
    t = t.assign(seq=t.seq.astype(str), frame=t.frame.astype(int))
    t = t.sort_values(["seq", "frame"]).reset_index(drop=True)
    J, first = j_table(t)
    assert np.abs(J[:, 0, 0] - t.J_cheap.to_numpy(float)).max() < 1e-9
    assert np.abs(J[:, 1, 0] - t.J_full.to_numpy(float)).max() < 1e-9
    return t, J, first


def replay_loss(z, J, first):
    z = np.asarray(z, int)
    p = np.r_[0, z[:-1]]
    p[first] = 0
    return J[np.arange(len(z)), z, p]


def additive(z, J):
    return (J[:, 0, 0] - J[:, 1, 0]) * z


def topk_mask(score, k, seed):
    """Top k with ties broken by a seeded permutation."""
    rng = np.random.default_rng(seed)
    pm = rng.permutation(len(score))
    o = pm[np.argsort(-np.round(score[pm], 9), kind="stable")]
    z = np.zeros(len(score), int)
    z[o[:k]] = 1
    return z


def by_unit(x, units):
    return pd.Series(x).groupby(units).sum()


def unit_ci(a, b, units):
    """Interval of sum(a) - sum(b) over unit draws, seed 0."""
    d = by_unit(a - b, units).to_numpy()
    r = np.random.default_rng(0)
    return xl.ci(np.array([d[r.integers(0, len(d), len(d))].sum() for _ in range(NBOOT)]))


# ------------------------------------------------------------------------------------------------ A

def stage_replay(args):
    runmeta.new_run("r10_replay", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    t234 = _load("t234", "234_r7_influence.py")
    sc = pickle.loads((Path(CACHE) / "xl" / "r7" / "scores.pkl").read_bytes())
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    pops_xl = {"pooled850": set(sp.scene), "new228_test": set(new[new.deep_split == "test"].scene)}
    t124 = _load("t124", "124_causal_threshold.py")
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    ktest = set(splits["kitti"]["test"])
    kscores = {c["geometry"]: c for c in t124.core_cells(splits) if c["track"] == "KITTI" and c["system"] == "brake"}
    rows = []

    def one(track, geo, pop, tt, J, first, gate):
        seq = tt.seq.to_numpy()
        n, tot = len(tt), float(J[:, 0, 0].sum())
        k = max(int(round(0.2 * n)), 1)
        v = J[:, 0, 0] - J[:, 1, 0]
        sets = {"all-full": [np.ones(n, int)], "oracle at 20%": [topk_mask(v, k, s) for s in range(20)],
                "random at 20%": [topk_mask(np.zeros(n), k, s) for s in range(20)]}
        if gate is not None:
            sets["gate_gbm at 20%"] = [topk_mask(gate, k, 0)]
        for name, zs in sets.items():
            add = np.mean([additive(z, J) for z in zs], 0)
            rep = np.mean([J[:, 0, 0] - replay_loss(z, J, first) for z in zs], 0)
            lo, hi = unit_ci(rep, add, seq)
            a, r = float(add.sum()), float(rep.sum())
            rows.append({"track": track, "geometry": geo, "population": pop, "policy": name, "frames": n, "units": len(set(seq)),
                         "escalated": int(zs[0].sum()), "all_cheap_loss": tot, "additive_gain": a, "replayed_gain": r, "difference": r - a,
                         "difference_lo": lo, "difference_hi": hi, "relative_difference": (r - a) / abs(a) if abs(a) > EPS else np.nan,
                         "additive_share": a / tot, "replayed_share": r / tot})
    for geo in ("mono", "oracle"):
        t, J, first = load_xl(geo)
        g = sc[f"{geo}_brake"]
        gk = g["keys"].assign(s=g["s"][("gate_gbm", "raw", "V")], split=g["keys"].split)
        for pop, scenes in pops_xl.items():
            m = t.seq.isin(scenes).to_numpy()
            tt = t[m].reset_index(drop=True)
            J_, first_ = J[m], np.r_[True, tt.seq.to_numpy()[1:] != tt.seq.to_numpy()[:-1]]
            gate = None
            if pop == "new228_test":
                gate = tt[["seq", "frame"]].merge(gk[["seq", "frame", "s"]], on=["seq", "frame"], how="left", validate="one_to_one").s.to_numpy(float)
                assert np.isfinite(gate).all()
            one("nuScenes XL", geo, pop, tt, J_, first_, gate)
        t, J, first = load_kitti(geo)
        c = kscores[geo]
        for pop, seqs in (("all 21", set(t.seq)), ("test", ktest)):
            m = t.seq.isin(seqs).to_numpy()
            tt = t[m].reset_index(drop=True)
            J_, first_ = J[m], np.r_[True, tt.seq.to_numpy()[1:] != tt.seq.to_numpy()[:-1]]
            gate = None
            if pop == "test":
                fit = c["d"].split.isin(["train", "val"]).to_numpy()                # no stored frame scores for the gate: the benchmark's refit
                g = t124.gate_scores(c["d"], c["fcols"], c["v_all"], fit)["gate_gbm"]
                d = c["d"].assign(seq=c["d"].seq.astype(str), frame=c["d"].frame.astype(int), s=np.asarray(g, float))
                gate = tt[["seq", "frame"]].merge(d[["seq", "frame", "s"]], on=["seq", "frame"], how="left", validate="one_to_one").s.to_numpy(float)
            one("KITTI", geo, pop, tt, J_, first_, gate)
        print(f"  {geo}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "replay.csv", index=False)
    print(t[["track", "geometry", "population", "policy", "additive_share", "replayed_share", "difference", "difference_lo", "difference_hi", "relative_difference"]].to_string(), flush=True)


# ------------------------------------------------------------------------------------------------ B

def scene_curve(J):
    """F[m]: least loss of one unit with exactly m escalations (dynamic programming over t, previous branch, count)."""
    n = len(J)
    f = np.full((n + 2, 2), np.inf)                                 # f[m, last branch]
    f[0, 0] = J[0, 0, 0]
    f[1, 1] = J[0, 1, 0]
    for t in range(1, n):
        g = np.full_like(f, np.inf)
        g[:, 0] = np.minimum(f[:, 0] + J[t, 0, 0], f[:, 1] + J[t, 0, 1])
        g[1:, 1] = np.minimum(f[:-1, 0] + J[t, 1, 0], f[:-1, 1] + J[t, 1, 1])
        f = g
    return f.min(1)[:n + 1]


def knapsack(curves, k):
    """Least total loss with at most k escalations over the units, exact."""
    best = np.full(k + 1, np.inf)
    best[0] = 0.0
    for F in curves:
        new = np.full(k + 1, np.inf)
        for m in range(min(len(F), k + 1)):
            new[m:] = np.minimum(new[m:], best[:k + 1 - m] + F[m])
        best = new
    return float(best.min()), int(best.argmin())


def lagrange(F, w, k, iters=60):
    """F: units x (max count + 1), padded with inf; w: copies of each unit. min over lam >= 0 duality: the best dual value and its count."""
    lo, hi = 0.0, float(np.nanmax(np.where(np.isfinite(F), F, np.nan))) + 1.0
    m = np.arange(F.shape[1])

    def at(lam):
        x = F + lam * m
        a = x.argmin(1)
        return float((w * x[np.arange(len(F)), a]).sum()), float((w * a).sum())
    v0, c0 = at(0.0)
    if c0 <= k:
        return v0, c0, 0.0
    for _ in range(iters):
        lam = 0.5 * (lo + hi)
        v, c = at(lam)
        if c > k:
            lo = lam
        else:
            hi = lam
    v, c = at(hi)
    return v - hi * k, c, hi                                        # dual bound on the least loss


def _pop(t):
    label, tt, J, first, groups = t
    seq = tt.seq.to_numpy()
    uniq, starts = np.unique(seq, return_index=True)
    order = np.argsort(starts)
    uniq, starts = uniq[order], starts[order]
    ends = np.r_[starts[1:], len(seq)]
    curves = [scene_curve(J[a:b]) for a, b in zip(starts, ends)]
    F = np.full((len(curves), max(len(c) for c in curves)), np.inf)
    for i, c in enumerate(curves):
        F[i, :len(c)] = c
    v = J[:, 0, 0] - J[:, 1, 0]
    tot, n = float(J[:, 0, 0].sum()), len(v)
    rows = []
    for q in BUDGETS:
        k = max(int(round(q * n)), 1)
        add = float(xl.gains(v, v, np.array([k]))[0])
        zs = [topk_mask(v, k, s) for s in range(20)]
        add_rep = float(np.mean([tot - replay_loss(z, J, first).sum() for z in zs]))
        exact, used = knapsack(curves, k)
        dual, cnt, lam = lagrange(F, np.ones(len(F)), k)
        rec = {**label, "budget": q, "k": k, "units": len(uniq), "frames": n, "all_cheap_loss": tot, "additive_oracle": add, "additive_oracle_replayed": add_rep,
               "sequence_oracle": tot - exact, "sequence_escalated": used, "lagrange_bound": tot - dual, "multiplier": lam,
               "duality_gap": exact - dual, "difference": (tot - exact) - add, "difference_share": ((tot - exact) - add) / tot,
               "relative_difference": ((tot - exact) - add) / abs(add) if abs(add) > EPS else np.nan}
        idx = {u: np.flatnonzero(seq == u) for u in uniq}
        for unit, grp in (("", uniq), ("_log", np.array([groups[u] for u in uniq]))):
            r = np.random.default_rng(0)
            gs = np.unique(grp)
            mem = [np.flatnonzero(grp == g) for g in gs]
            x = np.full(NBOOT, np.nan)
            for b in range(NBOOT):
                w = np.zeros(len(uniq))
                for i in r.integers(0, len(gs), len(gs)):
                    w[mem[i]] += 1
                # frames of the draw, in unit copies
                nt = int((w * (ends - starts)).sum())
                kt = max(int(round(q * nt)), 1)
                tt_ = np.concatenate([np.repeat(v[idx[u]], int(c)) for u, c in zip(uniq, w) if c])
                d_, _, _ = lagrange(F, w, kt)
                tot_ = float((w * F[:, 0]).sum())
                x[b] = (tot_ - d_) - float(xl.gains(tt_, tt_, np.array([kt]))[0])
            rec[f"difference_lo{unit}"], rec[f"difference_hi{unit}"] = xl.ci(x)
        rows.append(rec)
        print(f"  {label} {q}: {rec['sequence_oracle']:.2f} vs {add:.2f}", flush=True)
    return rows


def stage_oracle(args):
    runmeta.new_run("r10_oracle", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    t234 = _load("t234", "234_r7_influence.py")
    sp = t234.meta()
    log_of = dict(zip(sp.scene, sp.log))
    new = sp[sp.group == "new765"]
    test = set(new[new.deep_split == "test"].scene)
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    tasks = []
    for geo in ("mono", "oracle"):
        t, J, first = load_xl(geo)
        for pop, keep in (("pooled850", np.ones(len(t), bool)), ("new228_test", t.seq.isin(test).to_numpy())):
            tt = t[keep].reset_index(drop=True)[["seq", "frame"]]
            tasks.append(({"track": "nuScenes XL", "geometry": geo, "population": pop}, tt, J[keep],
                          np.r_[True, tt.seq.to_numpy()[1:] != tt.seq.to_numpy()[:-1]], log_of))
        t, J, first = load_kitti(geo)
        for pop, keep in (("all 21", np.ones(len(t), bool)), ("test", t.seq.isin(set(splits["kitti"]["test"])).to_numpy())):
            tt = t[keep].reset_index(drop=True)[["seq", "frame"]]
            tasks.append(({"track": "KITTI", "geometry": geo, "population": pop}, tt, J[keep],
                          np.r_[True, tt.seq.to_numpy()[1:] != tt.seq.to_numpy()[:-1]], {s: s for s in set(tt.seq)}))
    with Pool(min(NPROC, len(tasks))) as p:
        rows = [r for x in p.map(_pop, tasks, chunksize=1) for r in x]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "sequence_oracle.csv", index=False)
    print(t.to_string(), flush=True)


# ------------------------------------------------------------------------------------------------ C

def policy_masks(s_val, s_test, q, units, stream, t124):
    tau, p = t124.calibrate(s_val, q)
    a = t124.mask_A(s_test, tau, p, np.random.default_rng(0))
    return {"A": a, "B": t124.apply_cap(a, q, units, stream)}


def stage_masks(args):
    runmeta.new_run("r10_masks", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    t124 = _load("t124", "124_causal_threshold.py")
    t236 = _load("t236", "236_r9_direct.py")
    t103 = t124.t103
    t234 = t236.t234
    sel = json.loads((REB / "R9" / "selected.json").read_text())
    by = {(r["cell"], r["budget"]): r for r in sel if r["variant"] == "V"}
    # XL
    cells_, log_of = t236.xl_setup(args)
    mat = xl.r1_matrix(str(t236.DET))
    tasks = []
    for c in cells_:
        for q in BUDGETS:
            r = by[(t236.cname(c), q)]
            tasks.append({"cell": t236.cname(c), "variant": "V", "setting": r["setting"], "epoch": int(r["epoch"]), "learner": r["learner"], "fit": ["train"],
                          "ev": ["val", "test"], "tag": f"{t236.cname(c)}|{q}"})
    dir_ = t236.refits(tasks)
    rows = []
    for c in cells_:
        d = c["d"]
        name = t236.cname(c)
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        fit = (d.split == "train").to_numpy()
        va, te = (d.split == "val").to_numpy(), (d.split == "test").to_numpy()
        s = {**xl.gate_scores(d, c["fcols"], v, fit), **t103.fit_score(xl.r1_rows(d, mat), v, fit)}
        seq = d.seq.astype(str).to_numpy()[te]
        log = np.array([log_of[x] for x in seq])
        fr = d.frame.to_numpy(int)[te]
        stream = np.lexsort((fr, seq))
        vt = v[te]
        tot = float(d[c["cheap"]].to_numpy(float)[te].sum())
        split = d.split.to_numpy()[np.isin(d.split.to_numpy(), ["val", "test"])]
        for q in BUDGETS:
            items = {n: (s[n][va], s[n][te]) for n in LEARNED}
            x = dir_[f"{name}|{q}"]
            items["R9 chosen"] = (x[split == "val"], x[split == "test"])
            for n, (sv, st) in items.items():
                for pol, z in policy_masks(sv, st, q, seq, stream, t124).items():
                    z = z.astype(float)
                    r = S.mask_stats(vt, seq, z, q, NBOOT, np.random.default_rng(0))
                    rl = S.mask_stats(vt, log, z, q, NBOOT, np.random.default_rng(0))
                    viol, vu = S.cap_violations(z, seq, stream, q)
                    rows.append({"cell": name, "signal": n, "policy": pol, **r, "share": r["gain"] / tot, "over_random_share": r["gain_minus_random"] / tot,
                                 "over_random_lo": r["gain_minus_random_lo"] / tot, "over_random_hi": r["gain_minus_random_hi"] / tot,
                                 "over_random_lo_log": rl["gain_minus_random_lo"] / tot, "over_random_hi_log": rl["gain_minus_random_hi"] / tot,
                                 "cap_violations": viol, "cap_units": vu, "all_cheap_loss": tot})
        print(f"  {name}", flush=True)
    pd.DataFrame(rows).to_csv(OUT / "masks_xl.csv", index=False)
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--stage", "masks_frozen", "--dataroot", args.dataroot])   # fresh process: a pool forked after the fits above deadlocks


def stage_masks_frozen(args):
    """The frozen cells, through the submission path."""
    runmeta.new_run("r10_masks_frozen", vars(args))
    t124 = _load("t124", "124_causal_threshold.py")
    t236 = _load("t236", "236_r9_direct.py")
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t124.t120.register_features()
    fr = list(t124.core_cells(splits)) + [c for c in t124.nuplan_cells(splits, nr_fcols) if c["system"] == "pdm_closed"]
    fs = pd.read_csv(REB / "R9" / "frozen_selected.csv")
    tasks = [{"cell": r.cell, "variant": "V", "setting": r.setting, "epoch": int(r.epoch), "learner": r.chosen, "fit": ["train"], "ev": ["val", "test"],
              "tag": f"{r.cell}|{r.budget}"} for r in fs.itertuples()]
    dir_ = t236.refits(tasks)
    subs = {}
    for i, c in enumerate(fr):
        d = c["d"]
        name = f"frozen{i:02d}"
        fit = (d.split == "train").to_numpy()
        va, te = (d.split == "val").to_numpy(), (d.split == "test").to_numpy()
        s = t124.learned_scores(c, fit)
        ids = ("seq", "frame") if c["track"] != "nuPlan" else ("scenario", "iteration")
        pos = {int(j): n for n, j in enumerate(np.flatnonzero(te))}
        stream = np.array([pos[int(j)] for j in c["order"] if te[j]], int)
        units = d.unit.astype(str).to_numpy()[te]
        split = d.split.to_numpy()[np.isin(d.split.to_numpy(), ["val", "test"])]
        base = {"track": c["track"], "geometry": c["geometry"], "system": c["system"], "target": c["target"],
                ids[0]: d[ids[0]].astype(str).to_numpy()[te], ids[1]: d[ids[1]].to_numpy()[te]}
        for n in list(LEARNED) + ["R9 chosen"]:
            for pol in ("A", "B"):
                col = dict(base)
                for q in BUDGETS:
                    if n == "R9 chosen":
                        x = dir_[f"{name}|{q}"]
                        sv, st = x[split == "val"], x[split == "test"]
                    else:
                        sv, st = s[n][va], s[n][te]
                    col[f"z_q{int(round(q * 100))}"] = policy_masks(sv, st, q, units, stream, t124)[pol].astype(int)
                subs.setdefault((n, pol, c["track"]), []).append(pd.DataFrame(col))
        print(f"  {name}", flush=True)
    out = []
    for (n, pol, track), parts in subs.items():
        sub = pd.concat(parts, ignore_index=True).astype(str)
        form, r = S.score(sub, causal=True)
        out.append(r.assign(signal=n, policy=pol))
        if (n, pol) == ("R9 chosen", "B"):
            f = OUT / f"frozen_mask_example_{track}.csv"
            sub.to_csv(f, index=False)
            subprocess.run([sys.executable, str(ROOT / "evaluate_submission.py"), str(f), "--causal", "--out", str(OUT / f"frozen_mask_example_{track}_scored.csv")],
                           check=True, cwd=str(ROOT))
    pd.concat(out, ignore_index=True).to_csv(OUT / "masks_frozen.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("gate", "replay", "oracle", "masks", "masks_frozen"))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
