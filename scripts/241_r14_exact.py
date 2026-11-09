#!/usr/bin/env python
"""Follow-up round 3, R14 (post hoc, a sensitivity): scorers trained directly on the exact budgeted objective
(docs/followup_protocol.md).  Allocators only; no column that reads a distance to an action threshold enters a model.

Every fit is a job (a JSON spec, run by `--stage worker` from a queue file; a result is one pickle named by the spec's hash),
so the same code runs on the first device and on the second machines.  Stages, in order:

  prep        the arrays of the six XL cells, the crit subset and the ladder rungs -> data/cache/xl/r14/data/*.npz
  q_select    the selection fits (385 -> 152)                       -> queues
  select      validation J of every setting, the choice, gate R14-0 (iii) -> val_settings.csv, selected.json, optim_trace.csv
  gate        gate R14-0 (i), (ii) and the (iii) summary            -> gate_r14_0.json (exit 1 on a failure)
  q_test      the chosen settings refit on the 537                   -> queues
  test        the 228 test scenes                                    -> primary.csv, cells_budgets.csv, influence.csv, capped.csv
  q_sec       repeats, causal folds, crit selection, ladder          -> queues;  q_sec2: crit refit and folds
  secondary   repeats.csv, causal.csv, crit.csv, profile_exact_*.json, ladder_capacity.csv
  frozen_prep / q_frozen2 / frozen   the 10 core and 2 PDM-Closed cells through evaluate_submission.py -> frozen_*
  worker      --queue FILE --procs N: run every job of a queue whose dependencies exist (claims by lock file)
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, pickle, platform, socket, subprocess, sys, time

BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _k in BLAS_ENV:                                                                 # before numpy: one BLAS thread per worker
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import predict, runmeta, xl                                                # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                   # noqa: E402
from rap.scoring import EPS                                                         # noqa: E402

REB = Path(RESULTS) / "followup"
OUT = REB / "R14"
WORK = Path(os.environ.get("R14_WORK", Path(CACHE) / "xl" / "r14"))
DATA, JOBS, QUEUES = WORK / "data", WORK / "jobs", WORK / "queues"
NBOOT = 1000
CELLS = ("mono_brake", "mono_plan_ade", "mono_plan_fde", "oracle_brake", "oracle_plan_ade", "oracle_plan_fde")
PRIMARY = ("mono_brake", "mono_plan_ade")
BUDGETS = tuple(xl.SWEEP)
LIN_SET = [("ridge", 0.3), ("ridge", 1.0), ("zero", 0.3), ("zero", 1.0)]
MLP_SET = [(1e-3, "random"), (1e-3, "mlp_V"), (3e-4, "random"), (3e-4, "mlp_V")]
REPEAT_ROWS = [("mono_brake 20%", "mono_brake", 0.2), ("mono_brake 5%", "mono_brake", 0.05), ("mono_plan_ade 20%", "mono_plan_ade", 0.2)]
CRIT_RATE = 0.0944490
FROZEN_Q = (0.10, 0.20, 0.30, 0.50)
FORBIDDEN = ("decision_margin", "boundary_distance", "a_cheap", "a_full", "a_gt", "act_")
SECOND_LEARNERS = ("mlp_V", "exact_mlp", "topk_mlp", "pair_mlp")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


C = _load("t241c", "241_r14_core.py")
C.EPS_V = EPS


def check_columns(cols):
    bad = [c for c in cols if any(f in c for f in FORBIDDEN)]
    assert not bad, bad


def cap_of(v):
    return float(np.percentile(np.abs(v[np.abs(v) > EPS]), 95))


def k_of(n, q, rule):
    return C.k_round(n, q) if rule == "round" else int(np.floor(q * n + 1e-9))


# ------------------------------------------------------------------------------------------------ jobs

def job_id(job):
    return hashlib.blake2b(json.dumps(job, sort_keys=True).encode(), digest_size=10).hexdigest()


def J_(job):
    """A job with its id and the machine it belongs to."""
    j = dict(job)
    j.setdefault("deps", [])
    return dict(j, id=job_id(j))


def result_path(jid):
    return JOBS / f"{jid}.pkl"


def done(jid):
    return result_path(jid).exists()


def load_result(jid):
    return pickle.loads(result_path(jid).read_bytes())


_DCACHE = {}


def data(name):
    if name not in _DCACHE:
        z = np.load(DATA / f"{name}.npz", allow_pickle=False)
        _DCACHE.clear()
        _DCACHE[name] = {k: z[k] for k in z.files}
    return _DCACHE[name]


def mask(spec, z):
    m = np.isin(z["split"], spec["split"])
    if "fold" not in spec:
        return m
    seed, kf, side = spec["fold"]
    logs = sorted(set(z["log"][m]))
    order = np.random.default_rng(seed).permutation(len(logs))
    fold_of = {logs[i]: p % 5 for p, i in enumerate(order)}
    f = np.array([fold_of.get(x, -1) for x in z["log"]])
    return m & ((f != kf) if side == "out" else (f == kf))


def bench_ridge_alpha(X, v):
    m = predict.make_model("linear", "reg", 0)
    m.fit(X, v)
    return float(m.named_steps["m"].alpha_), m


def run_job(job):
    """One fit. Result: {"scores": {epoch: {eval: array}}, "info": {...}, "weights": {...}}."""
    t0 = time.time()
    z = data(job["data"])
    X, v, seq = z["X"], z["v"].astype(float), z["seq"]
    fit = mask(job["fit"], z)
    evals = {k: mask(s, z) for k, s in job["evals"].items()}
    if job.get("variant", "V") == "capped":
        c = cap_of(v[fit])
        v = np.clip(v, -c, c)
    p = job.get("params", {})
    L = job["learner"]
    info, weights = {"n_fit": int(fit.sum())}, None
    if L in ("exact_lin", "ridge", "mlp_V", "exact_mlp"):
        Zf, *Ze, (mu, sd) = C.zscore(X[fit], *[X[m] for m in evals.values()])
        Ze = dict(zip(evals, Ze))
        info["clipped_fit_values"] = int((np.abs((X[fit] - mu) / sd) > C.CLIP).sum())
    if L in ("exact_lin", "ridge"):
        alpha, bench = bench_ridge_alpha(X[fit], v[fit])
        th, b0 = C.ridge_theta(Zf, v[fit], alpha)
        info.update(alpha=alpha, ridge_theta=th, ridge_intercept=b0)
        if L == "ridge":
            sc = {-1: {k: Ze[k] @ th + b0 for k in evals}}
            if p.get("bench"):
                sc[-2] = {k: bench.predict(X[m]) for k, m in evals.items()}
        else:
            k = k_of(int(fit.sum()), p["budget"], p["k_rule"])
            mu0 = th / np.linalg.norm(th) if p["init"] == "ridge" else np.zeros_like(th)
            r = C.fit_exact_lin(Zf, v[fit], k, mu0, p["sigma0"], iters=p.get("iters", C.ITERS), pop=p.get("pop", C.POP),
                                keep_trace=p.get("trace", False))
            _, ties = C.topk_mask(Zf @ r["theta"], k)
            info.update(theta=r["theta"], J_fit=r["J"], seed=r["seed"], J_seeds=r["J_seeds"], k_fit=k, ties_fit=ties, trace=r["trace"])
            sc = {-1: {kk: Ze[kk] @ r["theta"] for kk in evals}}
    elif L == "gate_gbm":
        from threadpoolctl import threadpool_limits
        m = predict.make_model("gbm", "reg", 0)
        with threadpool_limits(limits=1, user_api="openmp"):
            m.fit(X[fit], v[fit])
        sc = {-1: {k: m.predict(X[mm]) for k, mm in evals.items()}}
    elif L in ("mlp_V", "exact_mlp"):
        scale = float(np.abs(v[fit][np.abs(v[fit]) > EPS]).mean())
        init = None
        if L == "exact_mlp" and p["init"] is not None:
            ir = load_result(p["init"]["job"])
            ep = p["init"]["epoch"]
            if ep == "val_best":
                ev, kq, rule = p["init"]["eval"], p["init"]["budget"], p["init"]["k_rule"]
                ve = v[evals[ev]] if ev in evals else None
                assert ve is not None
                ep = best_epoch({e: d[ev] for e, d in ir["scores"].items()}, ve, k_of(len(ve), kq, rule))
            init = ir["weights"][int(ep)]
            info["init_epoch"] = int(ep)
        keep = p.get("keep")
        out = C.train_mlp(Zf, v[fit], seq[fit], {k: Ze[k] for k in evals}, "V" if L == "mlp_V" else "exact", p.get("lr", 1e-3),
                          epochs=p.get("epochs", C.EPOCHS), alpha=p.get("budget"), init=init, scale=scale, keep_weights=keep is not None,
                          m_samples=p.get("M", C.M_SAMPLES))
        if keep is not None:
            out, W = out
            weights = {e: w for e, w in W.items() if keep == "all" or e in keep}
        sc = out
        import torch
        info["torch"] = torch.__version__
    else:                                                                           # R9's learners (R15)
        t236 = _load("t236", "236_r9_direct.py")
        st = json.loads(p["setting"])
        names = list(evals)
        Xe = np.vstack([X[evals[k]] for k in names])
        o = t236.score_setting(X[fit], v[fit], seq[fit], Xe, st, epochs=p.get("epochs"))
        cut = np.cumsum([0] + [int(evals[k].sum()) for k in names])
        sc = {e: {k: s[cut[i]:cut[i + 1]] for i, k in enumerate(names)} for e, s in o.items()}
    info.update(seconds=time.time() - t0, host=socket.gethostname(), machine=platform.machine(), numpy=np.__version__)
    return {"scores": sc, "info": info, "weights": weights}


def best_epoch(scores_by_epoch, v, k):
    """The earliest epoch with the largest J."""
    best, be = -np.inf, None
    for e in sorted(scores_by_epoch):
        j = C.J(scores_by_epoch[e], v, k)
        if j > best:
            best, be = j, e
    return be


def _worker(args):
    qfile, wi = args
    jobs = json.loads(Path(qfile).read_text())
    JOBS.mkdir(parents=True, exist_ok=True)
    n = len(jobs)
    while True:
        pending = progressed = False
        for i in range(n):
            j = jobs[i]
            if done(j["id"]):
                continue
            pending = True
            lock = JOBS / f"{j['id']}.lock"
            if lock.exists() or not all(done(d) for d in j["deps"]):
                continue
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, f"{socket.gethostname()} {os.getpid()} {time.time()}".encode())
            os.close(fd)
            t0 = time.time()
            try:
                r = run_job(j)
            except Exception as e:                                                  # leave the lock: a failed job is looked at by hand
                print(f"  FAILED {j['id']} {j['learner']} {j.get('tag')}: {e!r}", flush=True)
                continue
            tmp = JOBS / f".{j['id']}.{socket.gethostname()}.{os.getpid()}.tmp"
            tmp.write_bytes(pickle.dumps(r, protocol=4))
            os.replace(tmp, result_path(j["id"]))
            print(f"  {j['id']} {j['learner']} {j.get('tag', '')} [{time.time() - t0:.0f}s]", flush=True)
            progressed = True
            break
        if not pending:
            return wi
        if not progressed:
            time.sleep(20)


def stage_worker(args):
    with Pool(args.procs) as p:
        p.map(_worker, [(args.queue, i) for i in range(args.procs)], chunksize=1)
    left = [j for j in json.loads(Path(args.queue).read_text()) if not done(j["id"])]
    print(f"queue {args.queue}: {len(left)} jobs not done", flush=True)
    return 0 if not left else 1


def stage_left(args):
    """Exit 1 if a job of the queue has no result."""
    left = [j for j in json.loads(Path(args.queue).read_text()) if not done(j["id"])]
    print(f"{Path(args.queue).name}: {len(left)} jobs without a result", flush=True)
    return 1 if left else 0


def write_queues(name, jobs):
    QUEUES.mkdir(parents=True, exist_ok=True)
    seen, uniq = set(), []
    for j in jobs:
        if j["id"] not in seen:
            seen.add(j["id"])
            uniq.append(j)
    for mach in ("first", "second"):
        q = [j for j in uniq if (j["learner"] in SECOND_LEARNERS or j.get("machine") == "second") == (mach == "second")]
        (QUEUES / f"{name}_{mach}.json").write_text(json.dumps(q, indent=0))
        print(f"{name}_{mach}: {len(q)} jobs, {sum(not done(j['id']) for j in q)} not done", flush=True)


# ------------------------------------------------------------------------------------------------ job builders

TR, VA, TE, TV = {"split": ["train"]}, {"split": ["val"]}, {"split": ["test"]}, {"split": ["train", "val"]}


def lin_job(cell, fit, evals, budget, init, sig, variant="V", rule="round", trace=False, tag="", **kw):
    return J_({"data": cell, "fit": fit, "evals": evals, "variant": variant, "learner": "exact_lin", "tag": tag,
               "params": {"budget": budget, "k_rule": rule, "init": init, "sigma0": sig, "trace": trace}, **kw})


def ridge_job(cell, fit, evals, variant="V", bench=False, tag="", **kw):
    return J_({"data": cell, "fit": fit, "evals": evals, "variant": variant, "learner": "ridge", "tag": tag, "params": {"bench": bench}, **kw})


def gbm_job(cell, fit, evals, variant="V", tag="", **kw):
    return J_({"data": cell, "fit": fit, "evals": evals, "variant": variant, "learner": "gate_gbm", "tag": tag, **kw})


def mlpv_job(cell, fit, evals, variant="V", epochs=200, keep="all", tag="", **kw):
    return J_({"data": cell, "fit": fit, "evals": evals, "variant": variant, "learner": "mlp_V", "tag": tag,
               "params": {"lr": 1e-3, "epochs": epochs, "keep": keep}, **kw})


EXACT_M, EXACT_EPOCHS = 256, 400                                                   # gate R14-0 (iii) fix, 2026-11-09 (addendum)


def emlp_job(cell, fit, evals, budget, lr, init_job, init_epoch, variant="V", rule="round", epochs=EXACT_EPOCHS, tag="", init_eval="val", keep=None):
    init = None
    if init_job is not None:
        init = {"job": init_job["id"], "epoch": init_epoch}
        if init_epoch == "val_best":
            init.update(eval=init_eval, budget=budget, k_rule=rule)
    return J_({"data": cell, "fit": fit, "evals": evals, "variant": variant, "learner": "exact_mlp", "tag": tag,
               "params": {"budget": budget, "k_rule": rule, "lr": lr, "init": init, "epochs": epochs, "M": EXACT_M, **({"keep": keep} if keep else {})},
               "deps": [init_job["id"]] if init_job is not None else []})


def select_jobs(cell, variant, budgets, rule="round", trace=True, evals=None, exact_mlp=True, lin_machine=None):
    """Every setting of the four learners fit on the training scenes, scored on validation and on the fit frames."""
    ev = evals or {"val": VA, "fit": TR}
    jobs = {"ridge": ridge_job(cell, TR, ev, variant, tag=f"sel {cell} {variant}")}
    mv = mlpv_job(cell, TR, ev, variant, tag=f"sel {cell} {variant}")
    jobs["mlp_V"] = mv
    for b in budgets:
        for init, sig in LIN_SET:
            jobs[("exact_lin", b, init, sig)] = lin_job(cell, TR, ev, b, init, sig, variant, rule, trace, tag=f"sel {cell} {variant} {b}",
                                                        **({"machine": lin_machine} if lin_machine else {}))
        for lr, init in (MLP_SET if exact_mlp else []):
            jobs[("exact_mlp", b, lr, init)] = emlp_job(cell, TR, ev, b, lr, mv if init == "mlp_V" else None, "val_best", variant, rule,
                                                        tag=f"sel {cell} {variant} {b}")
    return jobs


# ------------------------------------------------------------------------------------------------ prep

def xl_cells(args):
    t236 = _load("t236", "236_r9_direct.py")
    cells_ = t236.fill_cell_frames(args)
    return t236, {t236.cname(c): c for c in cells_}


def stage_prep(args):
    runmeta.new_run("r14_prep", vars(args))
    DATA.mkdir(parents=True, exist_ok=True)
    t236, cells_ = xl_cells(args)
    t234 = t236.t234
    log_of = dict(zip(t234.meta().scene, t234.meta().log))
    w9 = Path(CACHE) / "xl" / "r9"
    for name in CELLS:
        c = cells_[name]
        d = c["d"]
        check_columns(c["fcols"])
        z = np.load(w9 / f"data__{name}.npz", allow_pickle=False)
        X = t236.matrix(d, c["fcols"])
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        assert np.array_equal(z["X"], X) and np.array_equal(z["v"], v) and (z["seq"] == d.seq.astype(str).to_numpy()).all()
        seq = np.asarray(d.seq.astype(str), dtype=str)
        arr = dict(X=X, v=v, seq=seq, split=np.asarray(d.split.astype(str), dtype=str), log=np.array([log_of[s] for s in seq], dtype=str),
                   frame=d.frame.to_numpy(int), j=d[c["cheap"]].to_numpy(float), cols=np.array(c["fcols"], dtype=str))
        np.savez(DATA / f"{name}.npz", **arr)
        if name == "mono_brake":
            t231 = _load("t231", "231_r3_gate_cost.py")
            sub = json.loads((REB / "R3" / "selected.json").read_text())["gate_gbm"]["subset"]
            assert sub == "crit"
            use = t231.subset_columns(sub, t231.group_columns(c["fcols"]))
            check_columns(use)
            np.savez(DATA / "mono_brake_crit.npz", **dict(arr, X=t236.matrix(d, use), cols=np.array(use, dtype=str)))
        print(f"  {name}: {X.shape}", flush=True)
    base = dict(np.load(DATA / "mono_brake.npz", allow_pickle=False))
    for r in ("L0", "L1", "L2", "L3"):
        z = np.load(w9 / f"data__ladder_{r}.npz", allow_pickle=False)
        assert np.array_equal(z["v"], base["v"]) and (z["seq"] == base["seq"]).all()
        np.savez(DATA / f"ladder_{r}.npz", **dict(base, X=z["X"], cols=z["cols"]))
    print("prep done", flush=True)


# ------------------------------------------------------------------------------------------------ select

def all_select_jobs():
    out = {}
    for cell in CELLS:
        out[(cell, "V")] = select_jobs(cell, "V", BUDGETS)
        out[(cell, "capped")] = select_jobs(cell, "capped", (0.2,))
    return out


def stage_q_select(args):
    jobs = [j for d in all_select_jobs().values() for j in d.values()]
    write_queues("select", jobs)


def stage_select(args):
    run = runmeta.new_run("r14_select", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    rows, trace, sel, gate = [], [], [], []
    for (cell, variant), jobs in all_select_jobs().items():
        z = data(cell)
        v = z["v"].astype(float)
        tr, va = mask(TR, z), mask(VA, z)
        if variant == "capped":
            c = cap_of(v[tr])
            v = np.clip(v, -c, c)
        vt, vv = v[tr], v[va]
        budgets = BUDGETS if variant == "V" else (0.2,)
        rr = load_result(jobs["ridge"]["id"])
        mv = load_result(jobs["mlp_V"]["id"])
        for b in budgets:
            kt, kv = C.k_round(len(vt), b), C.k_round(len(vv), b)
            base = {"cell": cell, "variant": variant, "budget": b, "k_fit": kt, "k_val": kv}
            s = rr["scores"][-1]
            rows.append({**base, "learner": "ridge_V", "setting": json.dumps({"alpha": rr["info"]["alpha"]}), "epoch": -1,
                         "J_fit": C.J(s["fit"], vt, kt), "J_val": C.J(s["val"], vv, kv),
                         "ties_fit": C.topk_mask(s["fit"], kt)[1], "ties_val": C.topk_mask(s["val"], kv)[1]})
            for e, s in sorted(mv["scores"].items()):
                rows.append({**base, "learner": "mlp_V", "setting": "{}", "epoch": e, "J_fit": C.J(s["fit"], vt, kt), "J_val": C.J(s["val"], vv, kv),
                             "ties_fit": C.topk_mask(s["fit"], kt)[1], "ties_val": C.topk_mask(s["val"], kv)[1]})
            for init, sig in LIN_SET:
                r = load_result(jobs[("exact_lin", b, init, sig)]["id"])
                s = r["scores"][-1]
                rows.append({**base, "learner": "exact_lin", "setting": json.dumps({"init": init, "sigma0": sig}), "epoch": -1,
                             "J_fit": r["info"]["J_fit"], "J_val": C.J(s["val"], vv, kv), "seed": r["info"]["seed"],
                             "J_fit_seeds": json.dumps(r["info"]["J_seeds"]), "ties_fit": r["info"]["ties_fit"], "ties_val": C.topk_mask(s["val"], kv)[1]})
                assert abs(r["info"]["J_fit"] - C.J(s["fit"], vt, kt)) < 1e-9
                for sd, tr_ in (r["info"]["trace"] or {}).items():
                    for it, jmu, jmax, jel, jbest, sgm in tr_:
                        trace.append({"cell": cell, "variant": variant, "budget": b, "init": init, "sigma0": sig, "seed": sd, "iter": it,
                                      "J_mu": jmu, "J_pop_max": jmax, "J_elite_mean": jel, "J_best_so_far": jbest, "sigma_mean": sgm})
            for lr, init in MLP_SET:
                r = load_result(jobs[("exact_mlp", b, lr, init)]["id"])
                for e, s in sorted(r["scores"].items()):
                    rows.append({**base, "learner": "exact_mlp", "setting": json.dumps({"lr": lr, "init": init}), "epoch": e,
                                 "J_fit": C.J(s["fit"], vt, kt), "J_val": C.J(s["val"], vv, kv), "init_epoch": r["info"].get("init_epoch"),
                                 "ties_fit": C.topk_mask(s["fit"], kt)[1], "ties_val": C.topk_mask(s["val"], kv)[1]})
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "val_settings.csv", index=False)
    pd.DataFrame(trace).to_csv(OUT / "optim_trace.csv", index=False)
    for (cell, variant, b, learner), g in val.groupby(["cell", "variant", "budget", "learner"], sort=False):
        r = g.iloc[int(np.argmax(g.J_val.to_numpy()))].to_dict()                     # first maximum: enumeration order, earliest epoch
        sel.append({k: (None if isinstance(x, float) and np.isnan(x) else x) for k, x in r.items()})
    (OUT / "selected.json").write_text(json.dumps(sel, indent=1, default=float))
    S = pd.DataFrame(sel).set_index(["cell", "variant", "budget", "learner"])
    for (cell, variant, b) in sorted({k[:3] for k in S.index}):
        for a, ref in (("exact_lin", "ridge_V"), ("exact_mlp", "mlp_V")):
            x, y = S.loc[(cell, variant, b, a)], S.loc[(cell, variant, b, ref)]
            gate.append({"cell": cell, "variant": variant, "budget": b, "learner": a, "comparator": ref, "J_fit": x.J_fit, "J_fit_comparator": y.J_fit,
                         "setting": x.setting, "epoch": x.epoch, "pass": bool(x.J_fit >= y.J_fit)})
    g = pd.DataFrame(gate)
    g.to_csv(OUT / "gate_iii.csv", index=False)
    g.to_csv(run / "gate_iii.csv", index=False)
    print(g.to_string(index=False), flush=True)
    print("gate (iii):", "pass" if g["pass"].all() else f"FAIL at {int((~g['pass']).sum())} rows", flush=True)


# ------------------------------------------------------------------------------------------------ gate

def stored_scores(learner):
    """X2's stored frame scores (R7 wrote them from X2's refit), one column per cell."""
    return pd.read_csv(REB / "R7" / f"scores_{learner}__raw__V.csv.gz", dtype={"seq": str}, float_precision="round_trip")


def aligned(cell, tab):
    z = data(cell)
    k = pd.DataFrame({"seq": z["seq"].astype(str), "frame": z["frame"].astype(int)})
    j = k.merge(tab[["seq", "frame", cell]].astype({"frame": int}), on=["seq", "frame"], how="left", validate="one_to_one")
    assert j[cell].notna().all()
    return j[cell].to_numpy(float)


def x2_row(cell, signal):
    t = pd.read_csv(REB / "E1" / "x2_allocators.csv", float_precision="round_trip")
    g, s = cell.split("_", 1)
    r = t[(t.geometry == g) & (t.system == s) & (t.signal == signal) & (t.bootstrap_unit == "scene")]
    assert len(r) == 1
    return r.iloc[0]


def stage_gate(args):
    run = runmeta.new_run("r14_gate", vars(args))
    t234 = _load("t236", "236_r9_direct.py").t234
    rep = {"checks": []}
    # (i) the R14 code's realized gain of X2's stored gate_gbm test scores
    z = data("mono_brake")
    te = mask(TE, z)
    v = z["v"][te]
    s = aligned("mono_brake", stored_scores("gate_gbm"))[te]
    g = gain(s, v, C.k_round(len(v), 0.2))
    x = x2_row("mono_brake", "gate_gbm")
    rep["checks"].append({"what": "(i) gate_gbm stored scores, mono_brake 20%, gain", "ours": g, "stored": float(x.gain), "diff": g - float(x.gain),
                          "share_ours": g / z["j"][te].sum(), "pass": abs(g - float(x.gain)) < 1e-9})
    # (ii) ridge_V against the stored gate_ridge: the benchmark form through this code first, then the R14 form on z
    p2 = []
    with Pool(6) as pool:
        res = pool.map(run_job, [ridge_job(c, TV, {"test": TE, "fit": TV}, bench=True, tag="gate ii") for c in CELLS], chunksize=1)
    for cell, r in zip(CELLS, res):
        z = data(cell)
        te = mask(TE, z)
        v, jt = z["v"][te], z["j"][te]
        k = C.k_round(len(v), 0.2)
        stored = aligned(cell, stored_scores("gate_ridge"))
        x = x2_row(cell, "gate_ridge")
        gb, gz = gain(r["scores"][-2]["test"], v, k), gain(r["scores"][-1]["test"], v, k)
        p2.append({"cell": cell, "alpha": r["info"]["alpha"], "clipped_fit_values": r["info"]["clipped_fit_values"],
                   "bench_form_max_score_diff_vs_stored": float(np.abs(r["scores"][-2]["test"] - stored[te]).max()),
                   "gain_bench_form": gb, "gain_stored": float(x.gain), "diff_bench_form": gb - float(x.gain),
                   "gain_ridge_V_on_z": gz, "diff_ridge_V_vs_stored": gz - float(x.gain), "share_ridge_V": gz / jt.sum(),
                   "share_stored": float(x.gain) / jt.sum(), "spearman_ridge_V_vs_stored": float(pd.Series(r["scores"][-1]["test"]).corr(pd.Series(stored[te]), method="spearman"))})
    p2 = pd.DataFrame(p2)
    p2.to_csv(OUT / "gate_ii_ridge.csv", index=False)
    rep["checks"].append({"what": "(ii) benchmark-form ridge through R14 code equals stored gate_ridge (gain, 6 cells)",
                          "max_abs_diff": float(p2.diff_bench_form.abs().max()), "pass": bool(p2.diff_bench_form.abs().max() < 1e-9)})
    rep["ii_ridge_V_differences"] = "inputs differ: z clipped at +-10 vs StandardScaler on the +-1e6 matrix; alpha is the benchmark's (same fit frames); see gate_ii_ridge.csv"
    g3 = pd.read_csv(OUT / "gate_iii.csv")
    rep["checks"].append({"what": "(iii) J_fit(exact) >= J_fit(regression) at F385, every cell, variant and budget", "rows": len(g3),
                          "failing": int((~g3["pass"]).sum()), "pass": bool(g3["pass"].all())})
    rep["pass"] = all(c["pass"] for c in rep["checks"])
    (OUT / "gate_r14_0.json").write_text(json.dumps(rep, indent=1, default=float))
    (run / "gate_r14_0.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float), flush=True)
    print(p2.to_string(index=False), flush=True)
    if not rep["pass"]:
        raise SystemExit(1)


def passed():
    """Learners that pass gate R14-0: (i) and (ii) pass, and every (iii) row of the learner passes (addendum, second run, 2026-11-09)."""
    rep = json.loads((OUT / "gate_r14_0.json").read_text())
    if not all(c["pass"] for c in rep["checks"][:2]):
        return set()
    g3 = pd.read_csv(OUT / "gate_iii.csv")
    return {a for a, g in g3.groupby("learner") if bool(g["pass"].all())}


def mlp_on():
    return "exact_mlp" in passed()


def gain(s, v, k):
    return float(xl.gains(s, v, np.array([k]))[0]) if k > 0 else 0.0


# ------------------------------------------------------------------------------------------------ test

def selected():
    return {(r["cell"], r["variant"], r["budget"], r["learner"]): r for r in json.loads((OUT / "selected.json").read_text())}


def refit_jobs(cell, variant, budgets, S, fit=TV, evals=None, rule="round", tag="test", lin_machine=None, keep_exact=False):
    """The chosen settings refit on `fit`; mlp_V once with every epoch that a budget or an init needs."""
    ev = evals or {"test": TE, "fit": fit}
    kw = {"machine": lin_machine} if lin_machine else {}
    out = {"ridge": ridge_job(cell, fit, ev, variant, tag=f"{tag} {cell} {variant}", **kw)}
    need = sorted({int(S[(cell, variant, b, "mlp_V")]["epoch"]) for b in budgets})
    mv = mlpv_job(cell, fit, ev, variant, epochs=max(need), keep=need, tag=f"{tag} {cell} {variant}")
    out["mlp_V"] = mv
    for b in budgets:
        r = S[(cell, variant, b, "exact_lin")]
        st = json.loads(r["setting"])
        out[("exact_lin", b)] = lin_job(cell, fit, ev, b, st["init"], st["sigma0"], variant, rule, tag=f"{tag} {cell} {variant} {b}", **kw)
        if not mlp_on():
            continue
        r = S[(cell, variant, b, "exact_mlp")]
        st = json.loads(r["setting"])
        e_v = int(S[(cell, variant, b, "mlp_V")]["epoch"])
        out[("exact_mlp", b)] = emlp_job(cell, fit, ev, b, st["lr"], mv if st["init"] == "mlp_V" else None, e_v, variant, rule,
                                         epochs=int(r["epoch"]), tag=f"{tag} {cell} {variant} {b}", keep=[int(r["epoch"])] if keep_exact else None)
    return out


def all_test_jobs():
    S = selected()
    return {(cell, variant): refit_jobs(cell, variant, BUDGETS if variant == "V" else (0.2,), S) for cell in CELLS for variant in ("V", "capped")}


def stage_q_test(args):
    assert "exact_lin" in passed()
    write_queues("test", [j for d in all_test_jobs().values() for j in d.values()])


def test_scores(cell, variant, budget, jobs, S, ev="test"):
    """{learner: scores} of one cell, variant and budget."""
    out = {"ridge_V": load_result(jobs["ridge"]["id"])["scores"][-1][ev],
           "mlp_V": load_result(jobs["mlp_V"]["id"])["scores"][int(S[(cell, variant, budget, "mlp_V")]["epoch"])][ev],
           "exact_lin": load_result(jobs[("exact_lin", budget)]["id"])["scores"][-1][ev]}
    if ("exact_mlp", budget) not in jobs:
        return out
    r = load_result(jobs[("exact_mlp", budget)]["id"])
    out["exact_mlp"] = r["scores"][int(S[(cell, variant, budget, "exact_mlp")]["epoch"])][ev]
    return out


def two_sided(x):
    x = x[np.isfinite(x)]
    return float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))


def draws(units, n=NBOOT, seed=0):
    uniq = np.array(sorted(set(units)))
    idx = {u: np.flatnonzero(units == u) for u in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[u] for u in uniq[rng.integers(0, len(uniq), len(uniq))]]) for _ in range(n)]


_DRAWS = {}


def draws_c(units):
    key = (len(units), hashlib.blake2b(np.asarray(units).astype(str).tobytes(), digest_size=8).hexdigest())
    if key not in _DRAWS:
        if len(_DRAWS) >= 4:
            _DRAWS.clear()
        _DRAWS[key] = draws(units)
    return _DRAWS[key]


def kq(n, q, rule):
    return k_of(n, q, rule)


def contrast(sa, sb, v, j, seq, log, q, rule=("round", "round")):
    """a against b at budget q: realized shares of the all-cheap loss, over matched-count random, paired intervals; scene and log."""
    tot, n = float(j.sum()), len(v)
    ka, kb = kq(n, q[0], rule[0]), kq(n, q[1], rule[1])
    ga, gb = gain(sa, v, ka), gain(sb, v, kb)
    r = {"frames": n, "k_a": ka, "k_b": kb, "all_cheap_loss": tot, "gain_a": ga, "gain_b": gb, "share_a": ga / tot, "share_b": gb / tot,
         "delta_share": (ga - gb) / tot, "a_over_random_share": (ga - ka / n * v.sum()) / tot, "b_over_random_share": (gb - kb / n * v.sum()) / tot}
    for unit, units in (("scene", seq), ("log", log)):
        x, ra, rb, xa, xb = (np.full(NBOOT, np.nan) for _ in range(5))
        for b, t in enumerate(draws_c(units)):
            nt = len(t)
            ka_, kb_ = kq(nt, q[0], rule[0]), kq(nt, q[1], rule[1])
            a, bb = gain(sa[t], v[t], ka_), gain(sb[t], v[t], kb_)
            vs = v[t].sum()
            x[b], ra[b], rb[b] = (a - bb) / tot, a / tot, bb / tot
            xa[b], xb[b] = (a - ka_ / nt * vs) / tot, (bb - kb_ / nt * vs) / tot
        sfx = "" if unit == "scene" else "_log"
        r[f"delta_lo{sfx}"], r[f"delta_hi{sfx}"] = xl.ci(x)
        r[f"delta_p{sfx}"] = two_sided(x)
        for nm, arr in (("share_a", ra), ("share_b", rb), ("a_over_random", xa), ("b_over_random", xb)):
            r[f"{nm}_lo{sfx}"], r[f"{nm}_hi{sfx}"] = xl.ci(arr)
    return r


def cell_test(cell):
    z = data(cell)
    te = mask(TE, z)
    return z["v"][te].astype(float), z["j"][te].astype(float), z["seq"][te].astype(str), z["log"][te].astype(str)


def r9_scores():
    t236 = _load("t236", "236_r9_direct.py")
    s9 = pickle.loads((Path(CACHE) / "xl" / "r9" / "scores.pkl").read_bytes())
    sel9 = {(r["cell"], r["variant"], r["budget"]): r for r in json.loads((REB / "R9" / "selected.json").read_text())}
    return lambda cell, b: (sel9[(cell, "V", b)]["learner"], s9[t236.tag_of(sel9[(cell, "V", b)])])


def stage_test(args):
    runmeta.new_run("r14_test", vars(args))
    assert "exact_lin" in passed()
    S, T = selected(), all_test_jobs()
    gb_tab = stored_scores("gate_gbm")
    r9 = r9_scores()
    rows = []
    for cell in CELLS:
        v, j, seq, log = cell_test(cell)
        te = mask(TE, data(cell))
        sg = aligned(cell, gb_tab)[te]
        for b in BUDGETS:
            sc = test_scores(cell, "V", b, T[(cell, "V")], S)
            l9, s9 = r9(cell, b)
            for a, bn, sb in (("exact_lin", "ridge_V", sc["ridge_V"]), ("exact_mlp", "mlp_V", sc["mlp_V"]), ("exact_lin", "gate_gbm", sg),
                              ("exact_mlp", "gate_gbm", sg), ("exact_lin", f"R9 chosen ({l9})", s9), ("exact_mlp", f"R9 chosen ({l9})", s9)):
                if a not in sc:
                    continue
                rows.append({"cell": cell, "budget": b, "a": a, "b": bn, "setting_a": S[(cell, "V", b, a)]["setting"], "epoch_a": S[(cell, "V", b, a)]["epoch"],
                             **contrast(sc[a], sb, v, j, seq, log, (b, b))})
            print(f"  {cell} {b}: lin {[r['delta_share'] for r in rows if (r['cell'], r['budget'], r['a'], r['b']) == (cell, b, 'exact_lin', 'ridge_V')][0]:+.4f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cells_budgets.csv", index=False)
    p = t[(t.budget == 0.2) & t.cell.isin(PRIMARY) & (((t.a == "exact_lin") & (t.b == "ridge_V")) | ((t.a == "exact_mlp") & (t.b == "mlp_V")))].copy()
    p["endpoint"] = ["Δ_lin" if a == "exact_lin" else "Δ_mlp" for a in p.a]
    p["holm_family"] = len(p)
    p["holm_rejects"] = xl.holm(p.delta_p.to_numpy(), alpha=0.05)
    p["reading"] = np.where(p.holm_rejects, np.where(p.delta_share > 0, "exact better", "regression better"), "not resolved")
    if not mlp_on():                                   # addendum, gate R14-0 second run: Δ_mlp not tested
        p = pd.concat([p, pd.DataFrame([{"cell": c, "budget": 0.2, "a": "exact_mlp", "b": "mlp_V", "endpoint": "Δ_mlp",
                                         "reading": "not tested: exact_mlp failed gate R14-0 (iii)"} for c in PRIMARY])], ignore_index=True)
    p.to_csv(OUT / "primary.csv", index=False)
    print(p[["endpoint", "cell", "delta_share", "delta_lo", "delta_hi", "delta_p", "delta_lo_log", "delta_hi_log", "holm_rejects", "reading"]].to_string(index=False), flush=True)
    # R7's influence rows on the two primaries
    t234 = _load("t236", "236_r9_direct.py").t234
    rows = []
    for cell in PRIMARY:
        v, j, seq, log = cell_test(cell)
        sc = test_scores(cell, "V", 0.2, T[(cell, "V")], S)
        for a, bn in (("exact_lin", "ridge_V"), ("exact_mlp", "mlp_V")):
            if a not in sc:
                continue
            sa, sb = sc[a], sc[bn]
            two = set(seq[np.isin(log, t234.LOGS)])
            drop = {}
            for s in sorted(set(seq)):
                m = seq != s
                k = C.k_round(int(m.sum()), 0.2)
                drop[s] = (gain(sa[m], v[m], k) - gain(sb[m], v[m], k)) / float(j[m].sum())
            k = C.k_round(len(v), 0.2)
            base = (gain(sa, v, k) - gain(sb, v, k)) / float(j.sum())
            worst = max(drop, key=lambda s: abs(drop[s] - base))
            sets = [("all", set()), ("minus scene-1051", {"scene-1051"}), ("minus scene-1081", {"scene-1081"}), ("minus both", {"scene-1051", "scene-1081"}),
                    ("minus the two logs' test scenes", two), (f"minus most influential ({worst})", {worst})]
            for name, rem in sets:
                m = ~np.isin(seq, sorted(rem))
                rows.append({"cell": cell, "a": a, "b": bn, "subset": name, "drop_one_min": min(drop.values()), "drop_one_max": max(drop.values()),
                             **contrast(sa[m], sb[m], v[m], j[m], seq[m], log[m], (0.2, 0.2))})
    pd.DataFrame(rows).to_csv(OUT / "influence.csv", index=False)
    # capped V at 20%: chosen and compared on capped V; the V-trained pair on capped V; the capped-trained pair on V
    rows = []
    for cell in CELLS:
        v, j, seq, log = cell_test(cell)
        z = data(cell)
        c = cap_of(z["v"][mask(TV, z)])
        vc = np.clip(v, -c, c)
        scc = test_scores(cell, "capped", 0.2, T[(cell, "capped")], S)
        scv = test_scores(cell, "V", 0.2, T[(cell, "V")], S)
        for a, bn in (("exact_lin", "ridge_V"), ("exact_mlp", "mlp_V")):
            if a not in scc:
                continue
            for lab, sc, ev, vv in (("trained on capped V", scc, "capped V", vc), ("trained on V", scv, "capped V", vc), ("trained on capped V", scc, "V", v)):
                rows.append({"cell": cell, "a": a, "b": bn, "training": lab, "evaluated_on": ev, "cap": c, **contrast(sc[a], sc[bn], vv, j, seq, log, (0.2, 0.2))})
    pd.DataFrame(rows).to_csv(OUT / "capped.csv", index=False)


# ------------------------------------------------------------------------------------------------ secondary

def repeat_jobs():
    S = selected()
    out = {}
    for seed in range(20):
        for kf in range(5):
            fit, held = {"split": ["train", "val", "test"], "fold": [seed, kf, "out"]}, {"split": ["train", "val", "test"], "fold": [seed, kf, "in"]}
            ev = {"held": held}
            for cell in sorted({c for _, c, _ in REPEAT_ROWS}):
                bs = [b for _, c, b in REPEAT_ROWS if c == cell]
                need = sorted({int(S[(cell, "V", b, "mlp_V")]["epoch"]) for b in bs})
                out[("ridge", cell, seed, kf)] = ridge_job(cell, fit, ev, tag=f"rep {cell} {seed} {kf}", machine="second")
                mv = mlpv_job(cell, fit, ev, epochs=max(need), keep=need, tag=f"rep {cell} {seed} {kf}")
                out[("mlp_V", cell, seed, kf)] = mv
                for b in bs:
                    r = S[(cell, "V", b, "exact_lin")]
                    st = json.loads(r["setting"])
                    out[("exact_lin", cell, b, seed, kf)] = lin_job(cell, fit, ev, b, st["init"], st["sigma0"], tag=f"rep {cell} {b} {seed} {kf}", machine="second")
                    if not mlp_on():
                        continue
                    r = S[(cell, "V", b, "exact_mlp")]
                    st = json.loads(r["setting"])
                    out[("exact_mlp", cell, b, seed, kf)] = emlp_job(cell, fit, ev, b, st["lr"], mv if st["init"] == "mlp_V" else None,
                                                                     int(S[(cell, "V", b, "mlp_V")]["epoch"]), epochs=int(r["epoch"]), tag=f"rep {cell} {b} {seed} {kf}")
    return out


def causal_jobs():
    """exact_lin and ridge_V, mono_brake 20%: out-of-fold scores on the 537 for the threshold (R13's F537 folds)."""
    S = selected()
    r = S[("mono_brake", "V", 0.2, "exact_lin")]
    st = json.loads(r["setting"])
    out = {}
    for kf in range(5):
        fit, held = {"split": ["train", "val"], "fold": [0, kf, "out"]}, {"split": ["train", "val"], "fold": [0, kf, "in"]}
        out[("exact_lin", kf)] = lin_job("mono_brake", fit, {"held": held}, 0.2, st["init"], st["sigma0"], tag=f"causal {kf}")
        out[("ridge", kf)] = ridge_job("mono_brake", fit, {"held": held}, tag=f"causal {kf}")
    return out


def crit_select_jobs():
    return select_jobs("mono_brake_crit", "V", (CRIT_RATE,), rule="floor", trace=False, exact_mlp=mlp_on())


def crit_refit_jobs():
    S = crit_selected()
    out = refit_jobs("mono_brake_crit", "V", (CRIT_RATE,), S, rule="floor", tag="crit", keep_exact=True)
    for kf in range(5):
        fit, held = {"split": ["train", "val"], "fold": [0, kf, "out"]}, {"split": ["train", "val"], "fold": [0, kf, "in"]}
        f = refit_jobs("mono_brake_crit", "V", (CRIT_RATE,), S, fit=fit, evals={"held": held}, rule="floor", tag=f"crit oof {kf}")
        out.update({(k if isinstance(k, str) else k[0], "oof", kf): x for k, x in f.items()})
    return out


def crit_selected():
    f = OUT / "crit_selected.json"
    if not f.exists():
        jobs = crit_select_jobs()
        z = data("mono_brake_crit")
        v = z["v"].astype(float)
        vt, vv = v[mask(TR, z)], v[mask(VA, z)]
        kt, kv = k_of(len(vt), CRIT_RATE, "floor"), k_of(len(vv), CRIT_RATE, "floor")
        rows = []
        r = load_result(jobs["ridge"]["id"])["scores"][-1]
        rows.append({"learner": "ridge_V", "setting": "{}", "epoch": -1, "J_fit": C.J(r["fit"], vt, kt), "J_val": C.J(r["val"], vv, kv)})
        for e, s in sorted(load_result(jobs["mlp_V"]["id"])["scores"].items()):
            rows.append({"learner": "mlp_V", "setting": "{}", "epoch": e, "J_fit": C.J(s["fit"], vt, kt), "J_val": C.J(s["val"], vv, kv)})
        for init, sig in LIN_SET:
            r = load_result(jobs[("exact_lin", CRIT_RATE, init, sig)]["id"])
            rows.append({"learner": "exact_lin", "setting": json.dumps({"init": init, "sigma0": sig}), "epoch": -1, "J_fit": r["info"]["J_fit"],
                         "J_val": C.J(r["scores"][-1]["val"], vv, kv)})
        for lr, init in (MLP_SET if mlp_on() else []):
            r = load_result(jobs[("exact_mlp", CRIT_RATE, lr, init)]["id"])
            for e, s in sorted(r["scores"].items()):
                rows.append({"learner": "exact_mlp", "setting": json.dumps({"lr": lr, "init": init}), "epoch": e, "J_fit": C.J(s["fit"], vt, kt),
                             "J_val": C.J(s["val"], vv, kv)})
        t = pd.DataFrame(rows).assign(cell="mono_brake_crit", variant="V", budget=CRIT_RATE, k_fit=kt, k_val=kv)
        t.to_csv(OUT / "crit_val.csv", index=False)
        sel = [g.iloc[int(np.argmax(g.J_val.to_numpy()))].to_dict() for _, g in t.groupby("learner", sort=False)]
        f.write_text(json.dumps(sel, indent=1, default=float))
    return {(r["cell"], r["variant"], r["budget"], r["learner"]): r for r in json.loads(f.read_text())}


def ladder_jobs():
    return {r: ridge_job(f"ladder_{r}", TV, {"test": TE}, tag=f"ladder {r}") for r in ("L0", "L1", "L2", "L3")}


def stage_q_sec(args):
    jobs = list(repeat_jobs().values()) + list(causal_jobs().values()) + list(crit_select_jobs().values()) + list(ladder_jobs().values())
    write_queues("sec", jobs)


def stage_q_sec2(args):
    write_queues("sec2", list(crit_refit_jobs().values()))


def form_rows(name, s_test, s_oof, v, j, seq, log, frame, q, rule):
    """rank, A and B of one scorer on the test stream: threshold from out-of-fold scores (R13 F537)."""
    t124 = _load("t124", "124_causal_threshold.py")
    n, tot = len(v), float(j.sum())
    tau, p = t124.calibrate(s_oof, q)
    stream = np.lexsort((frame, seq))
    A = t124.mask_A(s_test, tau, p, np.random.default_rng(0))
    B = t124.apply_cap(A, q, seq, stream)
    k = kq(n, q, rule)
    out = []
    for form, m in (("rank", None), ("A", A), ("B", B)):
        g = gain(s_test, v, k) if m is None else float(v[m].sum())
        u = k if m is None else int(m.sum())
        r = {"scorer": name, "form": form, "rate": q, "k": k, "used": u, "tau": tau, "gain": g, "realized_share": g / tot, "over_random_share": (g - u / n * v.sum()) / tot}
        for unit, units in (("scene", seq), ("log", log)):
            x, y = np.full(NBOOT, np.nan), np.full(NBOOT, np.nan)
            for b, t in enumerate(draws_c(units)):
                nt = len(t)
                if m is None:
                    kk = kq(nt, q, rule)
                    gg, uu = gain(s_test[t], v[t], kk), kk
                else:
                    gg, uu = float(v[t][m[t]].sum()), int(m[t].sum())
                x[b], y[b] = gg / tot, (gg - uu / nt * v[t].sum()) / tot
            sfx = "" if unit == "scene" else "_log"
            r[f"realized_lo{sfx}"], r[f"realized_hi{sfx}"] = xl.ci(x)
            r[f"over_random_lo{sfx}"], r[f"over_random_hi{sfx}"] = xl.ci(y)
        out.append(r)
    return out


def oof_of(jobs_by_fold, key, cell, ev="held", epoch=-1):
    z = data(cell)
    tv = mask(TV, z)
    o = np.full(len(z["v"]), np.nan)
    for kf in range(5):
        h = mask({"split": ["train", "val"], "fold": [0, kf, "in"]}, z)
        o[h] = load_result(jobs_by_fold[kf]["id"])["scores"][epoch][ev]
    assert np.isfinite(o[tv]).all()
    return o[tv]


def stage_secondary(args):
    runmeta.new_run("r14_secondary", vars(args))
    S = selected()
    T = all_test_jobs()
    # 8. repeats
    R = repeat_jobs()
    t234 = _load("t236", "236_r9_direct.py").t234
    rows = []
    for name, cell, b in REPEAT_ROWS:
        z = data(cell)
        v, j = z["v"].astype(float), z["j"].astype(float)
        keep2 = ~np.isin(z["log"], t234.LOGS)
        e_v = int(S[(cell, "V", b, "mlp_V")]["epoch"])
        e_x = int(S[(cell, "V", b, "exact_mlp")]["epoch"]) if mlp_on() else None
        for seed in range(20):
            sc = {k: np.full(len(v), np.nan) for k in ("ridge_V", "exact_lin", "mlp_V") + (("exact_mlp",) if mlp_on() else ())}
            for kf in range(5):
                h = mask({"split": ["train", "val", "test"], "fold": [seed, kf, "in"]}, z)
                sc["ridge_V"][h] = load_result(R[("ridge", cell, seed, kf)]["id"])["scores"][-1]["held"]
                sc["mlp_V"][h] = load_result(R[("mlp_V", cell, seed, kf)]["id"])["scores"][e_v]["held"]
                sc["exact_lin"][h] = load_result(R[("exact_lin", cell, b, seed, kf)]["id"])["scores"][-1]["held"]
                if "exact_mlp" in sc:
                    sc["exact_mlp"][h] = load_result(R[("exact_mlp", cell, b, seed, kf)]["id"])["scores"][e_x]["held"]
            for pop, m in (("765 scenes", np.ones(len(v), bool)), ("765 minus the two logs", keep2)):
                k, tot = C.k_round(int(m.sum()), b), float(j[m].sum())
                g = {x: gain(s[m], v[m], k) for x, s in sc.items()}
                for a, bn in (("exact_lin", "ridge_V"), ("exact_mlp", "mlp_V")):
                    if a not in sc:
                        continue
                    rows.append({"row": name, "a": a, "b": bn, "seed": seed, "population": pop, "k": k, "all_cheap_loss": tot, "share_a": g[a] / tot,
                                 "share_b": g[bn] / tot, "delta_share": (g[a] - g[bn]) / tot})
    t = pd.DataFrame(rows)
    u = t.groupby(["row", "a", "b", "population"], sort=False).agg(mean=("delta_share", "mean"), min=("delta_share", "min"), max=("delta_share", "max"),
                                                                    repeats_above_0=("delta_share", lambda x: int((x > 0).sum())), repeats=("delta_share", "size")).reset_index()
    u["reading"] = np.where(u.repeats_above_0 >= 18, "exact ahead", np.where(u.repeats_above_0 <= 2, "regression ahead", "no consistent difference"))
    t.to_csv(OUT / "repeats.csv", index=False)
    u.to_csv(OUT / "repeats_summary.csv", index=False)
    print(u.to_string(index=False), flush=True)
    # 6. causal forms, exact_lin and ridge_V, mono_brake 20%, F537
    CJ = causal_jobs()
    z = data("mono_brake")
    te = mask(TE, z)
    v, j, seq, log, frame = z["v"][te], z["j"][te], z["seq"][te].astype(str), z["log"][te].astype(str), z["frame"][te]
    sc = test_scores("mono_brake", "V", 0.2, T[("mono_brake", "V")], S)
    rows = []
    for name, key in (("exact_lin", "exact_lin"), ("ridge_V", "ridge")):
        oof = oof_of({kf: CJ[(key, kf)] for kf in range(5)}, key, "mono_brake")
        rows += form_rows(name, sc[name], oof, v, j, seq, log, frame, 0.2, "round")
    r13 = pd.read_csv(REB / "R13" / "partA_rows.csv")
    r13 = r13[(r13.fit == "F537") & r13.signal.isin(["V", "crit_ms"])]
    for _, r in r13.iterrows():
        rows.append({"scorer": f"R13 gate_gbm {r.signal} (stored)", **{k: r[k] for k in ("form", "rate", "k", "used", "gain", "realized_share", "over_random_share",
                                                                                         "realized_lo", "realized_hi", "over_random_lo", "over_random_hi",
                                                                                         "realized_lo_log", "realized_hi_log", "over_random_lo_log", "over_random_hi_log")}})
    pd.DataFrame(rows).to_csv(OUT / "causal.csv", index=False)
    # 7. crit: ranking, A, B at F537 with out-of-fold thresholds; measured cost
    CS = crit_selected()
    CR = crit_refit_jobs()
    zc = data("mono_brake_crit")
    rows = []
    for name in ("exact_lin", "exact_mlp", "ridge_V"):
        if name == "exact_mlp" and not mlp_on():
            continue
        key = {"ridge_V": "ridge"}.get(name, name)
        jk = key if name == "ridge_V" else (key, CRIT_RATE)
        ep = -1 if name in ("exact_lin", "ridge_V") else int(CS[("mono_brake_crit", "V", CRIT_RATE, name)]["epoch"])
        st = load_result(CR[jk]["id"])["scores"][ep]["test"]
        oof = oof_of({kf: CR[(key, "oof", kf)] for kf in range(5)}, key, "mono_brake_crit", epoch=ep)
        rows += [dict(r, setting=CS[("mono_brake_crit", "V", CRIT_RATE, name)]["setting"], epoch=ep) for r in form_rows(name, st, oof, v, j, seq, log, frame, CRIT_RATE, "floor")]
    n = len(v)
    for r in rows:
        if r["form"] == "rank":
            r["over_random20_floor_share"] = (r["gain"] - t234.k_floor(n, 0.2) / n * v.sum()) / float(j.sum())
    for _, r in r13[r13.signal == "crit_ms"].iterrows():
        rows.append({"scorer": "R13 gate_gbm crit_ms (stored)", **{k: r[k] for k in ("form", "rate", "k", "used", "gain", "realized_share", "over_random_share", "realized_lo", "realized_hi")}})
    crit = pd.DataFrame(rows)
    prof = profile_crit(CS, CR)
    crit["profile_ms"] = crit.scorer.map(prof)
    crit.to_csv(OUT / "crit.csv", index=False)
    # 9. ladder: ridge_V on the four rungs
    L = ladder_jobs()
    l9 = pd.read_csv(REB / "R9" / "ladder.csv")
    rows = []
    vt, jt, seqt, logt = cell_test("mono_brake")
    k = C.k_round(len(vt), 0.2)
    prize = gain(vt, vt, k)
    for r, jb in L.items():
        s = load_result(jb["id"])["scores"][-1]["test"]
        g = gain(s, vt, k)
        rows.append({"learner": "ridge_V", "rung": r, "input_dim": int(data(f"ladder_{r}")["X"].shape[1]), "gain": g, "share": g / jt.sum(), "ndg": g / prize,
                     "oracle_share": prize / jt.sum(), "source": "R14"})
    for _, r in l9.iterrows():
        rows.append({"learner": f"{r.learner} ({r.chosen})", "rung": r.rung, "input_dim": r.input_dim, "gain": r.gain, "share": r.share, "ndg": r.ndg,
                     "oracle_share": r.oracle_share, "source": "R9 ladder.csv"})
    pd.DataFrame(rows).to_csv(OUT / "ladder_capacity.csv", index=False)


def profile_crit(CS, CR):
    """151_profile_allocator.py on the chosen crit models, plain numpy, as R9's crit stage."""
    t231 = _load("t231", "231_r3_gate_cost.py")
    zc = data("mono_brake_crit")
    cols = [str(c) for c in zc["cols"]]
    out = {}
    for name in ("exact_lin", "exact_mlp") if mlp_on() else ("exact_lin",):
        r = load_result(CR[(name, CRIT_RATE)]["id"])
        z = zc
        fit = mask(TV, z)
        mu, sd = z["X"][fit].mean(0), z["X"][fit].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        if name == "exact_lin":
            spec = {"kind": "lin", "mu": mu, "sd": sd, "theta": r["info"]["theta"]}
        else:
            ep = int(CS[("mono_brake_crit", "V", CRIT_RATE, "exact_mlp")]["epoch"])
            spec = {"kind": "mlp", "mu": mu, "sd": sd, "nets": [[np.asarray(a, np.float32) for a in w] for w in r["weights"][ep]]}
            assert len(spec["nets"]) == len(C.SEEDS)
        f = WORK / f"alloc_{name}.pkl"
        f.write_bytes(pickle.dumps(dict(spec, subset="crit", cols=cols), protocol=pickle.HIGHEST_PROTOCOL))
        o = OUT / f"profile_{name}.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "241b_r14_allocator.py"), "--check", str(f)], check=True, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        subprocess.run([sys.executable, str(ROOT / "scripts" / "151_profile_allocator.py"), str(ROOT / "scripts" / "241b_r14_allocator.py"), "--tracks", "nuScenes",
                        "--frames", "400", "--out", str(o)], check=True, env={**{k: x for k, x in os.environ.items() if k not in BLAS_ENV}, "R14_ALLOC": str(f)})
        out[name] = float(json.loads(o.read_text())["ms"]["nuScenes"])
    print("profiles (ms):", out, flush=True)
    return out


# ------------------------------------------------------------------------------------------------ frozen

def frozen_meta():
    return json.loads((WORK / "frozen_meta.json").read_text())


def stage_frozen_prep(args):
    """The 12 frozen cells (R9's construction), their arrays, and the selection queue."""
    runmeta.new_run("r14_frozen_prep", vars(args))
    t236 = _load("t236", "236_r9_direct.py")
    t124 = _load("t124", "124_causal_threshold.py")
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t124.t120.register_features()
    cells_ = list(t124.core_cells(splits)) + [c for c in t124.nuplan_cells(splits, nr_fcols) if c["system"] == "pdm_closed"]
    assert len(cells_) == 12
    DATA.mkdir(parents=True, exist_ok=True)
    meta = {}
    for i, c in enumerate(cells_):
        d = c["d"]
        check_columns(c["fcols"])
        ids = ("seq", "frame") if c["track"] != "nuPlan" else ("scenario", "iteration")
        seq = np.asarray(d[ids[0]].astype(str), dtype=str)
        name = f"frozen{i:02d}"
        X = t236.matrix(d, c["fcols"])
        z9 = np.load(Path(CACHE) / "xl" / "r9" / f"data__{name}.npz", allow_pickle=False)
        assert np.array_equal(z9["X"], X) and np.array_equal(z9["v"], np.asarray(c["v_all"], float))
        np.savez(DATA / f"{name}.npz", X=X, v=np.asarray(c["v_all"], float), seq=seq, split=np.asarray(d.split.astype(str), dtype=str), log=seq,
                 id1=d[ids[1]].to_numpy(), cols=np.array(c["fcols"], dtype=str))
        meta[name] = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"], ids=list(ids))
    (WORK / "frozen_meta.json").write_text(json.dumps(meta, indent=1))
    write_queues("frozen1", [j for name in meta for j in frozen_select_jobs(name).values()])


def frozen_select_jobs(name):
    return select_jobs(name, "V", FROZEN_Q, trace=False, exact_mlp=mlp_on(), lin_machine="second")   # the second machine's CPUs


def frozen_selected():
    f = OUT / "frozen_val.csv"
    rows = []
    for name in frozen_meta():
        jobs = frozen_select_jobs(name)
        z = data(name)
        v = z["v"].astype(float)
        vt, vv = v[mask(TR, z)], v[mask(VA, z)]
        for b in FROZEN_Q:
            kt, kv = C.k_round(len(vt), b), C.k_round(len(vv), b)
            base = {"cell": name, "variant": "V", "budget": b, "k_fit": kt, "k_val": kv}
            s = load_result(jobs["ridge"]["id"])["scores"][-1]
            rows.append({**base, "learner": "ridge_V", "setting": "{}", "epoch": -1, "J_fit": C.J(s["fit"], vt, kt), "J_val": C.J(s["val"], vv, kv)})
            for e, s in sorted(load_result(jobs["mlp_V"]["id"])["scores"].items()):
                rows.append({**base, "learner": "mlp_V", "setting": "{}", "epoch": e, "J_fit": C.J(s["fit"], vt, kt), "J_val": C.J(s["val"], vv, kv)})
            for init, sig in LIN_SET:
                r = load_result(jobs[("exact_lin", b, init, sig)]["id"])
                rows.append({**base, "learner": "exact_lin", "setting": json.dumps({"init": init, "sigma0": sig}), "epoch": -1, "J_fit": r["info"]["J_fit"],
                             "J_val": C.J(r["scores"][-1]["val"], vv, kv)})
            for lr, init in (MLP_SET if mlp_on() else []):
                r = load_result(jobs[("exact_mlp", b, lr, init)]["id"])
                for e, s in sorted(r["scores"].items()):
                    rows.append({**base, "learner": "exact_mlp", "setting": json.dumps({"lr": lr, "init": init}), "epoch": e, "J_fit": C.J(s["fit"], vt, kt),
                                 "J_val": C.J(s["val"], vv, kv)})
    t = pd.DataFrame(rows)
    t.to_csv(f, index=False)
    sel = {}
    for k, g in t.groupby(["cell", "variant", "budget", "learner"], sort=False):
        sel[k] = g.iloc[int(np.argmax(g.J_val.to_numpy()))].to_dict()
    return sel


def frozen_refit_jobs(S):
    return {name: refit_jobs(name, "V", FROZEN_Q, S, tag="frozen", lin_machine="second") for name in frozen_meta()}


def stage_q_frozen2(args):
    S = frozen_selected()
    pd.DataFrame(list(S.values())).to_csv(OUT / "frozen_selected.csv", index=False)
    write_queues("frozen2", [j for d in frozen_refit_jobs(S).values() for j in d.values()])


def stage_frozen(args):
    runmeta.new_run("r14_frozen", vars(args))
    S = {(r["cell"], r["variant"], r["budget"], r["learner"]): r for r in pd.read_csv(OUT / "frozen_selected.csv").to_dict("records")}
    R = frozen_refit_jobs(S)
    meta = frozen_meta()
    subs = {k: [] for k in ("exact_lin", "ridge_V", "mlp_V") + (("exact_mlp",) if mlp_on() else ())}
    for name, m in meta.items():
        z = data(name)
        te = mask(TE, z)
        base = {"track": m["track"], "geometry": m["geometry"], "system": m["system"], "target": m["target"], m["ids"][0]: z["seq"][te], m["ids"][1]: z["id1"][te]}
        cols = {k: dict(base) for k in subs}
        for q in FROZEN_Q:
            sc = test_scores(name, "V", q, R[name], S)
            for k in subs:
                cols[k][f"score_q{int(round(q * 100))}"] = np.round(sc[k], 9)
        for k in subs:
            subs[k].append(pd.DataFrame(cols[k]))
    for name, parts in subs.items():
        for track, g in pd.concat(parts, ignore_index=True).groupby("track", sort=False):
            g = g.dropna(axis=1, how="all")
            f = OUT / f"frozen_submission_{name}_{track}.csv"
            g.to_csv(f, index=False, float_format="%.9f")
            subprocess.run([sys.executable, str(ROOT / "evaluate_submission.py"), str(f), "--out", str(OUT / f"frozen_scored_{name}_{track}.csv")], check=True, cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--queue")
    ap.add_argument("--procs", type=int, default=6)
    args = ap.parse_args()
    rc = globals()[f"stage_{args.stage}"](args)
    raise SystemExit(rc or 0)


if __name__ == "__main__":
    main()
