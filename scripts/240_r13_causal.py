#!/usr/bin/env python
"""Follow-up round 3, R13 (post hoc, a sensitivity): causal rules at a matched fit set, and the direct learner under log-grouped folds
(addendum R13 in docs/followup_protocol.md).

  --stage a        ranking and policies A, B at F385 and F537 on the 228; gate R13-0 -> partA_*.csv, gate_r13_0.json
  --stage b        out of fold by log with a causal threshold, seeds 0-19 -> partB_repeats.csv, partB_summary.csv
  --stage c        the direct learner under the same folds, ranking -> partC_repeats.csv, partC_summary.csv, partC_time.json
  --stage c_export the arrays stage c needs, for a machine without the 850-scene caches; c_check: that machine against R9's stored rows
  --stage reading  the registered readings -> reading.csv
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):                                   # before numpy, as 236
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import predict, runmeta, submission as S, xl                               # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                   # noqa: E402

REB = Path(RESULTS) / "followup"
OUT = REB / "R13"
WORK = Path(CACHE) / "xl" / "r13"
NPROC, NBOOT, SEEDS = int(os.environ.get("R13_PROC", 3)), 1000, 20
SIGNALS = ("V", "fn_gain", "risk_gain", "Q", "G", "crit_ms", "crit_mJ")
TG = ("fn_gain", "risk_gain", "Q", "G")
FORMS = ("rank", "A", "B")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


t124 = _load("t124", "124_causal_threshold.py")
t227 = _load("t227", "227_r1_target_swap.py")
t231 = _load("t231", "231_r3_gate_cost.py")
t236 = _load("t236", "236_r9_direct.py")
t234 = t236.t234
LOGS = t234.LOGS
_C = {}


def rates():
    t = pd.read_csv(REB / "R3" / "test.csv")
    t = t[(t.geometry == "mono") & (t.learner == "gate_gbm") & (t.subset == "crit") & (t.budget == 0.2)].set_index("unit")
    ms, mj = float(t.escalated_frac["ms"]), float(t.escalated_frac["mJ"])
    assert round(ms, 7) == 0.0944490 and round(mj, 7) == 0.1600907, (ms, mj)
    return {**{s: 0.2 for s in SIGNALS[:5]}, "crit_ms": ms, "crit_mJ": mj}, t


def kfun(sig, n, rt):
    return t234.k_round(n) if sig in SIGNALS[:5] else t234.k_floor(n, rt[sig])


def setup(args):
    a = argparse.Namespace(det=str(t227.DET), dataroot=args.dataroot)
    cells_, log_of = t227.setup(a)
    c = next(x for x in cells_ if x["geometry"] == "mono" and x["system"] == "brake")
    d = c["d"]
    tg = pd.read_csv(REB / "R1" / "targets.csv", dtype={"seq": str})
    arrays = t227.target_arrays(d, tg, "brake")
    arrays["V"] = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    sub = json.loads((REB / "R3" / "selected.json").read_text())["gate_gbm"]["subset"]
    assert sub == "crit", sub
    crit = t231.subset_columns(sub, t231.group_columns(c["fcols"]))
    t236.check_columns(c["fcols"])
    t236.check_columns(crit)
    seq = d.seq.astype(str).to_numpy()
    _C.update(c=c, d=d, arrays=arrays, crit=crit, seq=seq, log=np.array([log_of[x] for x in seq]), frame=d.frame.to_numpy(int), v=arrays["V"], j=d[c["cheap"]].to_numpy(float),
              X=t236.matrix(d, c["fcols"]), Xc=t236.matrix(d, crit), split=d.split.to_numpy(), cells=cells_)


def fit_one(sig, fit):
    """gate_gbm on one signal: one OpenMP thread, X2's model and seed."""
    tgt = "V" if sig.startswith("crit") else sig
    X = _C["Xc"] if sig.startswith("crit") else _C["X"]
    m = predict.make_model("gbm", "reg", 0)
    m.fit(X[fit], _C["arrays"][tgt][fit])
    return m.predict(X)


def _fit_task(t):
    key, sig, fit = t
    t0 = time.time()
    s = fit_one(sig, fit)
    print(f"  {key} {sig} [{time.time() - t0:.0f}s]", flush=True)
    return key, sig, s


def policy(sv, st, q, units, stream):
    tau, p = t124.calibrate(sv, q)
    a = t124.mask_A(st, tau, p, np.random.default_rng(0))
    return {"A": a, "B": t124.apply_cap(a, q, units, stream)}, tau, p


def draws_of(units, n=NBOOT):
    uniq, idx = xl.unit_index(units)
    rng = np.random.default_rng(0)
    return uniq, idx, [rng.integers(0, len(uniq), len(uniq)) for _ in range(n)]


def level(sc, z, sigs, v, units, rt):
    """Per draw: gain and used of every (fit, signal, form); rank is ranked again on the draw."""
    uniq, idx, dr = draws_of(units)
    pos = {u: i for i, u in enumerate(uniq)}
    uid = np.array([pos[u] for u in units])
    nu = np.bincount(uid, minlength=len(uniq)).astype(float)
    vu = np.bincount(uid, v, minlength=len(uniq))
    G, U = {}, {}
    for key, m in z.items():
        G[key] = np.bincount(uid, m * v, minlength=len(uniq))
        U[key] = np.bincount(uid, m.astype(float), minlength=len(uniq))
    out = {}
    N, VS = np.zeros(NBOOT), np.zeros(NBOOT)
    for b, ix in enumerate(dr):
        N[b], VS[b] = nu[ix].sum(), vu[ix].sum()
    for key in z:
        out[key] = (np.array([G[key][ix].sum() for ix in dr]), np.array([U[key][ix].sum() for ix in dr]))
    for f, s in sc.items():
        for sig in sigs:
            g, k = np.zeros(NBOOT), np.zeros(NBOOT)
            for b, ix in enumerate(dr):
                t = np.concatenate([idx[uniq[u]] for u in ix])
                k[b] = kfun(sig, len(t), rt)
                g[b] = t234.gain(s[sig][t], v[t], int(k[b]))
            out[(f, sig, "rank")] = (g, k)
    return out, N, VS


def stage_a(args):
    runmeta.new_run("r13_a", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    setup(args)
    rt, t3 = rates()
    seq, log, split, v, j, frame = _C["seq"], _C["log"], _C["split"], _C["v"], _C["j"], _C["frame"]
    tr, va, te = split == "train", split == "val", split == "test"
    tv = tr | va
    fl = sorted(set(log[tv]))
    order = np.random.default_rng(0).permutation(len(fl))
    fold_of = {fl[i]: p % 5 for p, i in enumerate(order)}
    fold = np.array([fold_of.get(x, -1) for x in log])
    fits = {"crit_ms": "crit", **{s: s for s in SIGNALS[:5]}}
    tasks = [(("F385", -1), s, tr) for s in fits] + [(("F537", -1), s, tv) for s in fits]
    tasks += [(("F537", kf), s, tv & (fold != kf)) for kf in range(5) for s in fits]
    with Pool(NPROC) as p:
        res = p.map(_fit_task, tasks, chunksize=1)
    s_all = {"F385": {}, "F537": {}}
    oof = {s: np.full(len(v), np.nan) for s in fits}
    for (f, kf), sig, s in res:
        if kf < 0:
            s_all[f][sig] = s
        else:
            m = tv & (fold == kf)
            oof[sig][m] = s[m]
    for f in s_all:
        s_all[f]["crit_mJ"] = s_all[f]["crit_ms"]
    oof["crit_mJ"] = oof["crit_ms"]
    assert np.isfinite(np.concatenate([oof[s][tv] for s in SIGNALS])).all()
    sq, lg, vt, jt, fr = seq[te], log[te], v[te], j[te], frame[te]
    stream = np.lexsort((fr, sq))
    tot = float(jt.sum())
    n = len(vt)
    sc = {f: {s: s_all[f][s][te] for s in SIGNALS} for f in s_all}
    z, th = {}, []
    for f in s_all:
        for sig in SIGNALS:
            sv = s_all["F385"][sig][va] if f == "F385" else oof[sig][tv]
            ms, tau, pp = policy(sv, sc[f][sig], rt[sig], sq, stream)
            th.append({"fit": f, "signal": sig, "rate": rt[sig], "tau": tau, "tie_p": pp, "threshold_frames": len(sv)})
            for fm in ("A", "B"):
                z[(f, sig, fm)] = ms[fm].astype(float)
    # gate R13-0
    point = {(f, sig): t234.gain(sc[f][sig], vt, kfun(sig, n, rt)) for f in sc for sig in SIGNALS}
    rd = lambda p: pd.read_csv(p, float_precision="round_trip")
    r7 = rd(REB / "R7" / "partC_D_G_rows.csv")
    r7 = r7[(r7.decision_value == "V") & (r7.set == "all")].set_index("row")
    e = rd(REB / "R7" / "partE_fit_sets.csv")
    e = e[(e.fit == "385 scenes") & (e.population == "228 test")].iloc[0]
    mx = rd(REB / "R10" / "masks_xl.csv")
    mx = mx[(mx.cell == "mono_brake") & (mx.signal == "gate_gbm") & (mx.quota == 0.2)].set_index("policy")
    chk = [("F537 rank V", point[("F537", "V")], r7.gain["X2 / R1 gate_gbm on V"]), ("F537 rank crit_ms", point[("F537", "crit_ms")], t3.gain["ms"]), ("F537 rank crit_mJ", point[("F537", "crit_mJ")], t3.gain["mJ"]),
           ("F385 rank V", point[("F385", "V")], e.gain), ("F385 A V", float((z[("F385", "V", "A")] * vt).sum()), mx.gain["A"]), ("F385 B V", float((z[("F385", "V", "B")] * vt).sum()), mx.gain["B"])]
    for t in TG:
        chk.append((f"F537 rank D {t}", (point[("F537", "V")] - point[("F537", t)]) / tot, r7.D_share[f"R1 gate_gbm on {t}"]))
    rep = {"checks": [{"what": a, "ours": float(b), "stored": float(c), "diff": float(b - c)} for a, b, c in chk]}
    rep["pass"] = all(abs(b - c) < 1e-6 for _, b, c in chk)
    (OUT / "gate_r13_0.json").write_text(json.dumps(rep, indent=1))
    print("gate R13-0:", rep["pass"], [(x["what"], round(x["diff"], 9)) for x in rep["checks"]], flush=True)
    if not rep["pass"]:
        raise SystemExit(1)
    pd.DataFrame(th).to_csv(OUT / "partA_thresholds.csv", index=False)
    fm = pd.DataFrame({"seq": sq, "frame": fr})
    for f in sc:
        for sig in SIGNALS:
            if sig != "crit_mJ":
                fm[f"s_{f}_{sig}"] = sc[f][sig]
            for x in ("A", "B"):
                fm[f"z_{f}_{sig}_{x}"] = z[(f, sig, x)].astype(int)
    fm.to_csv(OUT / "partA_frames.csv.gz", index=False, float_format="%.17g")
    # intervals
    L = {"": (level(sc, z, SIGNALS, vt, sq, rt)), "_log": (level(sc, z, SIGNALS, vt, lg, rt))}
    rows, dts = [], []
    ci = xl.ci
    pgain = {}
    for f in sc:
        for sig in SIGNALS:
            k = kfun(sig, n, rt)
            pgain[(f, sig, "rank")], used_r = point[(f, sig)], k
            for fm_ in FORMS:
                key = (f, sig, fm_)
                if fm_ == "rank":
                    g, u, viol, cu = point[(f, sig)], k, np.nan, np.nan
                else:
                    zz = z[key]
                    g, u = float((zz * vt).sum()), int(zz.sum())
                    viol, cu = S.cap_violations(zz, sq, stream, rt[sig])
                    pgain[key] = g
                r = {"fit": f, "signal": sig, "form": fm_, "rate": rt[sig], "k": k, "used": u, "within_budget": bool(u <= k), "cap_violations": viol, "cap_units": cu, "gain": g,
                     "realized_share": g / tot, "over_random_share": (g - u / n * vt.sum()) / tot, "all_cheap_loss": tot}
                if fm_ != "rank":
                    r["minus_rank_share"] = (g - point[(f, sig)]) / tot
                for tag, (o, N, VS) in L.items():
                    gg, uu = o[key]
                    x = ci(gg / tot)
                    r["realized_lo" + tag], r["realized_hi" + tag] = x
                    x = ci((gg - uu / N * VS) / tot)
                    r["over_random_lo" + tag], r["over_random_hi" + tag] = x
                    if fm_ != "rank":
                        x = ci((gg - o[(f, sig, "rank")][0]) / tot)
                        r["minus_rank_lo" + tag], r["minus_rank_hi" + tag] = x
                rows.append(r)
    for f in sc:
        for fm_ in FORMS:
            for t in TG:
                r = {"fit": f, "form": fm_, "target": t, "D_share": (pgain[(f, "V", fm_)] - pgain[(f, t, fm_)]) / tot, "used_V": (z[(f, "V", fm_)].sum() if fm_ != "rank" else kfun("V", n, rt)),
                     "used_target": (z[(f, t, fm_)].sum() if fm_ != "rank" else kfun(t, n, rt))}
                for tag, (o, N, VS) in L.items():
                    r["D_lo" + tag], r["D_hi" + tag] = ci((o[(f, "V", fm_)][0] - o[(f, t, fm_)][0]) / tot)
                dts.append(r)
    pd.DataFrame(rows).to_csv(OUT / "partA_rows.csv", index=False)
    pd.DataFrame(dts).to_csv(OUT / "partA_Dt.csv", index=False)
    # influence, B at F537
    claims = {"V": (z[("F537", "V", "B")], None), "crit_ms": (z[("F537", "crit_ms", "B")], None)}
    claims.update({f"D {t}": (z[("F537", "V", "B")], z[("F537", t, "B")]) for t in TG})
    inf = []
    for name, (a, b) in claims.items():
        contrib = a * vt - (0 if b is None else b * vt)

        def val(m):
            return float(contrib[m].sum() / jt[m].sum())
        base = val(np.ones(n, bool))
        vals = pd.Series({u: val(sq != u) for u in sorted(set(sq))})
        w = vals.idxmin()
        m = sq != w
        tt = float(jt[m].sum())
        r = {"claim": name, "value_all": base, "scene": w, "value_after": float(vals[w]), "drop_one_min": float(vals.min()), "drop_one_max": float(vals.max())}
        for tag, un in (("", sq[m]), ("_log", lg[m])):
            uniq, idx, dr = draws_of(un)
            pos = {u: i for i, u in enumerate(uniq)}
            cu = np.bincount(np.array([pos[u] for u in un]), contrib[m], minlength=len(uniq))
            r["lo_after" + tag], r["hi_after" + tag] = xl.ci(np.array([cu[ix].sum() for ix in dr]) / tt)
        inf.append(r)
    pd.DataFrame(inf).to_csv(OUT / "partA_influence.csv", index=False)
    print(pd.DataFrame(rows)[["fit", "signal", "form", "k", "used", "realized_share", "realized_lo", "realized_hi", "minus_rank_share"]].round(3).to_string(index=False), flush=True)


def outer_folds(seed, log=None):
    """R7 Part F's folds over the logs of the 765 scenes; `log` gives another cell's frames (the fold follows the log)."""
    logs = sorted(set(_C["log"]))
    order = np.random.default_rng(seed).permutation(len(logs))
    fold_of = {logs[i]: p % 5 for p, i in enumerate(order)}
    return logs, np.array([fold_of[x] for x in (_C["log"] if log is None else log)])


def _b_task(t):
    seed, kf = t
    t0 = time.time()
    log, seq, frame = _C["log"], _C["seq"], _C["frame"]
    _, fold = outer_folds(seed)
    fit_all, held = fold != kf, fold == kf
    fl = sorted(set(log[fit_all]))
    rng = np.random.default_rng([1000 + seed, kf])
    inner = set(np.array(fl)[rng.permutation(len(fl))[:int(round(len(fl) / 5))]])
    im = np.isin(log, sorted(inner))
    fit = fit_all & ~im
    rt = _C["rt"]
    h = np.flatnonzero(held)
    sq, fr = seq[h], frame[h]
    stream = np.lexsort((fr, sq))
    out = {}
    for sig in SIGNALS[:6]:
        s = fit_one(sig, fit)
        ms, _, _ = policy(s[im & fit_all], s[h], rt[sig], sq, stream)
        out[sig] = (s[h].copy(), ms["A"].astype(float), ms["B"].astype(float))
    print(f"  b seed {seed} fold {kf} [{time.time() - t0:.0f}s]", flush=True)
    return seed, kf, h, out


def stage_b(args):
    runmeta.new_run("r13_b", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    setup(args)
    rt, _ = rates()
    _C["rt"] = rt
    v, j, log = _C["v"], _C["j"], _C["log"]
    keep2 = ~np.isin(log, LOGS)
    rows = []
    with Pool(NPROC) as p:
        by = {}
        for seed, kf, h, out in p.imap_unordered(_b_task, [(s, k) for s in range(SEEDS) for k in range(5)], chunksize=1):
            by.setdefault(seed, []).append((h, out))
            if len(by[seed]) < 5:
                continue
            S_, Z = {}, {}
            for sig in SIGNALS[:6]:
                a = np.full(len(v), np.nan); za = np.zeros(len(v)); zb = np.zeros(len(v))
                for hh, o in by[seed]:
                    a[hh], za[hh], zb[hh] = o[sig]
                S_[sig], Z[sig] = a, {"A": za, "B": zb}
            for pop, m in (("765 scenes", np.ones(len(v), bool)), ("765 minus the two logs", keep2)):
                n, tot, sv = int(m.sum()), float(j[m].sum()), float(v[m].sum())
                for fm in FORMS:
                    gs, us = {}, {}
                    for sig in SIGNALS[:6]:
                        if fm == "rank":
                            k = kfun(sig, n, rt)
                            gs[sig], us[sig] = t234.gain(S_[sig][m], v[m], k), k
                        else:
                            gs[sig], us[sig] = float((Z[sig][fm][m] * v[m]).sum()), float(Z[sig][fm][m].sum())
                    r = {"seed": seed, "population": pop, "form": fm, "n_frames": n, "all_cheap_loss": tot}
                    for sig in ("V", "crit_ms"):
                        r[f"realized_{sig}"] = gs[sig] / tot
                        r[f"over_random_{sig}"] = (gs[sig] - us[sig] / n * sv) / tot
                        r[f"used_share_{sig}"] = us[sig] / n
                    r.update({f"D_{t}": (gs["V"] - gs[t]) / tot for t in TG})
                    rows.append(r)
            print(f"seed {seed} done, {len(rows)} rows", flush=True)
            pd.DataFrame(rows).to_csv(WORK / "partB_partial.csv", index=False)
    t = pd.DataFrame(rows).sort_values(["seed", "population", "form"])
    t.to_csv(OUT / "partB_repeats.csv", index=False)
    cols = [c for c in t.columns if c.split("_")[0] in ("realized", "over", "used", "D")]
    s = []
    for (pop, fm), g in t.groupby(["population", "form"], sort=False):
        for q in cols:
            s.append({"population": pop, "form": fm, "quantity": q, "mean": g[q].mean(), "min": g[q].min(), "max": g[q].max(), "repeats_above_0": int((g[q] > 0).sum()), "repeats": len(g)})
    pd.DataFrame(s).to_csv(OUT / "partB_summary.csv", index=False)


ROWS = [("mono_brake 20%", "mono_brake", 0.2), ("mono_brake 5%", "mono_brake", 0.05), ("mono_plan_ade 20%", "mono_plan_ade", 0.2)]


def _c_task(t):
    what, cell, seed, kf, row = t
    t0 = time.time()
    X, v, seq, split = t236.load_data(cell)
    _, fold = outer_folds(seed, _C["logs"][cell])
    fit, held = fold != kf, fold == kf
    if what == "gate":
        m = predict.make_model("gbm", "reg", 0)
        m.fit(X[fit], v[fit])
        s = m.predict(X[held])
    else:
        st = json.loads(row["setting"])
        out = t236.score_setting(X[fit], v[fit], seq[fit], X[held], st, epochs=None if row["epoch"] < 0 else int(row["epoch"]))
        s = out[int(row["epoch"])]
    return what, cell, row["name"] if row else None, seed, kf, s, time.time() - t0


def c_inputs(args):
    """All-cheap losses, V and logs of Part C's cells; written once where the 850-scene caches are (`--stage c_export`)."""
    return WORK / "c_inputs.npz"


def stage_c_export(args):
    WORK.mkdir(parents=True, exist_ok=True)
    setup(args)
    cells_ = {f"{c['geometry']}_{c['system']}": c for c in _C["cells"]}
    log_of = dict(zip(_C["seq"], _C["log"]))
    out = {"log": _C["log"]}
    for cell in ("mono_brake", "mono_plan_ade"):
        c = cells_[cell]
        d = c["d"]
        X, v, seq, split = t236.load_data(cell)
        assert (d.seq.astype(str).to_numpy() == seq.astype(str)).all() and np.array_equal(v, (d[c["cheap"]] - d[c["full"]]).to_numpy(float))
        out[f"j_{cell}"], out[f"v_{cell}"], out[f"log_{cell}"] = d[c["cheap"]].to_numpy(float), v, np.array([log_of[x] for x in d.seq.astype(str)])
    np.savez(c_inputs(args), **out)


def _check_task(t):
    what, cell, row = t
    X, v, seq, split = t236.load_data(cell)
    fit, te = np.isin(split, ["train", "val"]), split == "test"
    if what == "gate":
        m = predict.make_model("gbm", "reg", 0)
        m.fit(X[fit], v[fit])
        s = m.predict(X[te])
    else:
        s = t236.score_setting(X[fit], v[fit], seq[fit], X[te], json.loads(row["setting"]), epochs=None if row["epoch"] < 0 else int(row["epoch"]))[int(row["epoch"])]
    return what, cell, row["budget"] if row else None, s


def stage_c_check(args):
    """This machine against the Jetson's stored R9 rows, on the 537 -> 228 refit, before Part C is trusted here."""
    OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((REB / "R9" / "selected.json").read_text())
    p = pd.read_csv(REB / "R9" / "cells_budgets.csv").set_index(["cell", "budget"])
    z = np.load(c_inputs(args), allow_pickle=False)
    tasks = [("gate", c, None) for c in ("mono_brake", "mono_plan_ade")]
    rows = []
    for _, cell, b in ROWS:
        r = next(x for x in sel if x["cell"] == cell and x["variant"] == "V" and x["budget"] == b)
        tasks.append(("direct", cell, dict(r, name=cell)))
    with Pool(NPROC) as pool:
        res = pool.map(_check_task, tasks, chunksize=1)
    gs = {(w, c, b): s for w, c, b, s in res}
    for _, cell, b in ROWS:
        v, split = z[f"v_{cell}"], t236.load_data(cell)[3]
        te, k = split == "test", t234.k_round(int((split == "test").sum()), b)
        tot = float(z[f"j_{cell}"][te].sum())
        g, d = t234.gain(gs[("gate", cell, None)], v[te], k) / tot, t234.gain(gs[("direct", cell, b)], v[te], k) / tot
        q = p.loc[(cell, b)]
        rows.append({"cell": cell, "budget": b, "gate_here": g, "gate_kit": float(q.share_gate), "gate_diff": g - float(q.share_gate), "direct_here": d, "direct_kit": float(q.share_direct), "direct_diff": d - float(q.share_direct)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "partC_machine_check.csv", index=False)
    print(t.to_string(index=False), flush=True)


def stage_c(args):
    runmeta.new_run("r13_c", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    sel = json.loads((REB / "R9" / "selected.json").read_text())
    sets = []
    for name, cell, b in ROWS:
        r = next(x for x in sel if x["cell"] == cell and x["variant"] == "V" and x["budget"] == b)
        sets.append({"name": name, "cell": cell, "budget": b, "setting": r["setting"], "epoch": int(r["epoch"]), "learner": r["learner"]})
        print(" ", sets[-1], flush=True)
    assert [s["learner"] for s in sets] == ["pair_mlp", "cs_gbm", "topk_mlp"]
    z = np.load(c_inputs(args), allow_pickle=False)
    J, V = {}, {}
    for cell in {s["cell"] for s in sets}:
        J[cell], V[cell] = z[f"j_{cell}"], z[f"v_{cell}"]
        assert np.array_equal(t236.load_data(cell)[1], V[cell])
    _C["log"] = z["log"].astype(str)
    _C["logs"] = {cell: z[f"log_{cell}"].astype(str) for cell in V}
    assert all(set(x) == set(_C["log"]) for x in _C["logs"].values())

    def tasks(seed, rows):
        out = [("gate", cell, seed, kf, None) for cell in sorted({s["cell"] for s in rows}) for kf in range(5)]
        return out + [("direct", s["cell"], seed, kf, s) for s in rows for kf in range(5)]
    timing = {}
    seeds_of = {s["name"]: SEEDS for s in sets}
    res = {}
    with Pool(NPROC) as p:
        for x in p.imap_unordered(_c_task, tasks(0, sets), chunksize=1):
            res.setdefault((x[0], x[1], x[2], x[3]), {})[x[4]] = x[5]
            timing.setdefault((x[0], x[1], x[2]), []).append(x[6])
        for s in sets:
            sec = sum(timing[("direct", s["cell"], s["name"])]) + sum(timing[("gate", s["cell"], None)])
            ext = sec * SEEDS / NPROC / 3600
            seeds_of[s["name"]] = 10 if ext > 4 else SEEDS
            s["seconds_seed0"], s["extrapolated_hours"], s["seeds"] = sec, ext, seeds_of[s["name"]]
            print(f"  {s['name']}: seed 0 {sec:.0f}s, extrapolated {ext:.1f} h, seeds {seeds_of[s['name']]}", flush=True)
        (OUT / "partC_time.json").write_text(json.dumps(sets, indent=1))

        def collect(seed):
            rows = [s for s in sets if seed < seeds_of[s["name"]]]
            return tasks(seed, rows)
        ts = [t for seed in range(1, SEEDS) for t in collect(seed)]
        for x in p.imap_unordered(_c_task, ts, chunksize=1):
            res.setdefault((x[0], x[1], x[2], x[3]), {})[x[4]] = x[5]
            print(f"  c {x[0]} {x[2]} seed {x[3]} fold {x[4]} [{x[6]:.0f}s]", flush=True)
    rows = []
    for s in sets:
        v, j = V[s["cell"]], J[s["cell"]]
        keep2 = ~np.isin(_C["logs"][s["cell"]], LOGS)
        for seed in range(seeds_of[s["name"]]):
            _, fold = outer_folds(seed, _C["logs"][s["cell"]])
            a, b = np.full(len(v), np.nan), np.full(len(v), np.nan)
            for kf in range(5):
                h = fold == kf
                a[h] = res[("direct", s["cell"], s["name"], seed)][kf]
                b[h] = res[("gate", s["cell"], None, seed)][kf]
            for pop, m in (("765 scenes", np.ones(len(v), bool)), ("765 minus the two logs", keep2)):
                k, tot = t234.k_round(int(m.sum()), s["budget"]), float(j[m].sum())
                gd, gg = t234.gain(a[m], v[m], k), t234.gain(b[m], v[m], k)
                rows.append({"row": s["name"], "learner": s["learner"], "seed": seed, "population": pop, "k": k, "all_cheap_loss": tot, "direct_share": gd / tot, "gate_share": gg / tot, "delta_share": (gd - gg) / tot})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "partC_repeats.csv", index=False)
    g = t.groupby(["row", "learner", "population"], sort=False)
    u = g.agg(mean=("delta_share", "mean"), min=("delta_share", "min"), max=("delta_share", "max"), repeats_above_0=("delta_share", lambda x: int((x > 0).sum())), repeats=("delta_share", "size"),
              direct_mean=("direct_share", "mean"), gate_mean=("gate_share", "mean")).reset_index()
    u.to_csv(OUT / "partC_summary.csv", index=False)
    print(u.round(4).to_string(index=False), flush=True)


def stage_reading(args):
    A = pd.read_csv(OUT / "partA_rows.csv")
    D = pd.read_csv(OUT / "partA_Dt.csv")
    I = pd.read_csv(OUT / "partA_influence.csv").set_index("claim")
    B = pd.read_csv(OUT / "partB_repeats.csv")
    C = pd.read_csv(OUT / "partC_summary.csv")
    b = B[(B.population == "765 scenes") & (B.form == "B")]
    b2 = B[(B.population == "765 minus the two logs") & (B.form == "B")]
    rows = []
    for name, col in [("V", "realized_V"), ("crit_ms", "realized_crit_ms")] + [(f"D {t}", f"D_{t}") for t in TG]:
        r = I.loc[name]
        rep, rep2 = int((b[col] > 0).sum()), int((b2[col] > 0).sum())
        lab = "robust" if r.lo_after > 0 and rep >= 18 else "tail-dependent" if r.value_after > 0 and rep >= 11 else "fragile"
        rows.append({"reading": "claim under B at F537", "item": name, "value_all": r.value_all, "scene_removed": r.scene, "value_after": r.value_after, "lo_after": r.lo_after, "hi_after": r.hi_after,
                     "lo_after_log": r.lo_after_log, "hi_after_log": r.hi_after_log, "repeats_above_0": rep, "repeats_above_0_minus_two_logs": rep2, "label": lab})
    for f in ("F385", "F537"):
        for sig in ("V", "crit_ms"):
            for fm in ("A", "B"):
                r = A[(A.fit == f) & (A.signal == sig) & (A.form == fm)].iloc[0]
                rows.append({"reading": "cost of the causal rule", "item": f"{sig} {fm} {f}", "value_all": r.minus_rank_share, "lo_after": r.minus_rank_lo, "hi_after": r.minus_rank_hi, "lo_after_log": r.minus_rank_lo_log,
                             "hi_after_log": r.minus_rank_hi_log, "label": "costs gain" if r.minus_rank_hi < 0 else "no resolved cost"})
    for _, r in C[C.population == "765 scenes"].iterrows():
        hi, lo = (18, 2) if r.repeats == 20 else (9, 1)
        rows.append({"reading": "Part C", "item": r.row, "value_all": r["mean"], "repeats_above_0": r.repeats_above_0, "label": "direct ahead" if r.repeats_above_0 >= hi else "regression ahead" if r.repeats_above_0 <= lo else "no consistent difference"})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "reading.csv", index=False)
    print(t[["reading", "item", "value_all", "value_after", "lo_after", "hi_after", "repeats_above_0", "label"]].round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("a", "b", "c_export", "c_check", "c", "reading"))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
