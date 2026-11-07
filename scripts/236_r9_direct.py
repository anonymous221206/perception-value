#!/usr/bin/env python
"""Follow-up round 3, R9: baselines trained directly on the budgeted objective, and an information ladder
(addendum R9 in docs/followup_protocol.md).  Allocators only: no detector is trained or re-weighted, and no column that reads a
distance to an action threshold enters any model.

  --stage select   XL cells: every setting fit on the 385 training scenes, realized gain on the 152 validation scenes, per
                   cell and budget -> val_settings.csv, selected.json (also on capped V and on R8's primary variant)
  --stage test     the chosen settings refit on the 537 scenes and scored once on the 228 -> primary.csv, cells_budgets.csv,
                   learners_descriptive.csv, influence.csv, capped.csv, r8_variant.csv
  --stage crit     R3's chosen subset under the measured 20% latency budget, costs from 151_profile_allocator.py -> crit.csv
  --stage frozen   the 10 core cells and 2 PDM-Closed cells through evaluate_submission.py, per-budget form -> frozen*.csv
  --stage causal   policies A and B of docs/iclr_causal_threshold.md, variant V1 -> causal.csv
  --stage ladder   rungs L0-L3 (saved), then ladder_fit in a fresh process -> ladder.csv, ladder_deltas.csv
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, os, pickle, subprocess, sys, time

BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
for _k in BLAS_ENV:                                                                 # before numpy: one BLAS thread in each of the six workers
    os.environ.setdefault(_k, "1")

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

os.environ["CUDA_VISIBLE_DEVICES"] = ""                                             # CPU only: forked workers cannot initialize CUDA
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import geometry as G, planner as P, predict, runmeta, xl                 # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402
from rap.risk import RiskConfig                                                   # noqa: E402
from rap.scoring import EPS                                                        # noqa: E402

OUT = Path(RESULTS) / "followup" / "R9"
WORK = Path(CACHE) / "xl" / "r9"
REB = Path(RESULTS) / "followup"
DET = Path(CACHE) / "nusc_det_xl"
NBOOT, NPROC = 1000, 6
PRIMARY_CELL, ADE_CELL = "mono_brake", "mono_plan_ade"
LAM_Q = (None, 50, 75, 90, 95)
EPS_M = (0.0, 0.01, 0.1)
TEMPS = (0.1, 0.3, 1.0)
SEEDS = (0, 1, 2, 3, 4)
EPOCHS, EVERY, BATCH, PAIRS, LR, WD, CLIP = 200, 10, 2048, 8192, 1e-3, 1e-4, 10.0
FROZEN_Q = (0.10, 0.20, 0.30, 0.50)
FORBIDDEN = ("decision_margin", "boundary_distance", "a_cheap", "a_full", "a_gt", "act_")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


t234 = _load("t234", "234_r7_influence.py")


def check_columns(cols):
    bad = [c for c in cols if any(f in c for f in FORBIDDEN)]
    assert not bad, bad


def matrix(d, cols):
    return np.nan_to_num(d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)


def affected(v):
    return np.abs(v) > EPS


def cap_of(v):
    return float(np.percentile(np.abs(v[affected(v)]), 95))


# ------------------------------------------------------------------------------------------------ learners

def clf():
    """The gate's hyperparameters on the classifier."""
    p = predict.make_model("gbm", "reg", 0).get_params()
    keep = ("max_depth", "max_iter", "learning_rate", "min_samples_leaf", "l2_regularization", "max_bins", "early_stopping",
            "random_state")
    return HistGradientBoostingClassifier(**{k: p[k] for k in keep})


def fit_cs(X, v, seqs, setting):
    """cs_gbm on the fit frames: y = 1[V > lam], weight |V - lam| + eps."""
    lam = 0.0 if setting["lam_q"] is None else float(np.percentile(v[v > EPS], setting["lam_q"]))
    eps = setting["eps_m"] * float(np.median(np.abs(v[affected(v)])))
    y = (v > lam).astype(int)
    if len(set(y)) < 2:
        return None
    m = clf()
    with threadpool_limits(limits=1, user_api="openmp"):
        m.fit(X, y, sample_weight=np.abs(v - lam) + eps)
    return m


def batches(seqs, rng):
    """Whole scenes in shuffled order, cut when a batch reaches BATCH frames; a short remainder joins the last batch."""
    idx = pd.Series(np.arange(len(seqs))).groupby(seqs).apply(lambda x: x.to_numpy()).to_dict()
    names = sorted(idx)
    out, cur, n = [], [], 0
    for i in rng.permutation(len(names)):
        cur.append(idx[names[i]])
        n += len(idx[names[i]])
        if n >= BATCH:
            out.append(np.concatenate(cur))
            cur, n = [], 0
    if cur:
        if out:
            out[-1] = np.concatenate([out[-1]] + cur)
        else:
            out.append(np.concatenate(cur))
    return out


def fit_mlp(X, v, seqs, X_eval, setting, keep_net=False):
    """Five seeds of the routers' MLP; scores on X_eval averaged over the seeds at every checkpoint.

    setting: kind (topk | pair), T and alpha (topk), epochs (the last checkpoint wanted).  Returns {epoch: scores}.
    """
    import torch
    torch.set_num_threads(1)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    tr = torch.as_tensor(np.clip((X - mu) / sd, -CLIP, CLIP), dtype=torch.float32)
    ev = torch.as_tensor(np.clip((X_eval - mu) / sd, -CLIP, CLIP), dtype=torch.float32)
    scale = float(np.abs(v[affected(v)]).mean())
    vt = torch.as_tensor(v / scale, dtype=torch.float32)
    sums, nets = {}, []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        gen = torch.Generator().manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                                  torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
        for epoch in range(1, setting["epochs"] + 1):
            for b in batches(seqs, rng):
                b = torch.as_tensor(b)
                s = net(tr[b]).squeeze(1)
                if setting["kind"] == "topk":
                    th = torch.quantile(s.detach(), 1 - setting["alpha"])
                    loss = -(torch.sigmoid((s - th) / setting["T"]) * vt[b]).mean()
                else:
                    i = torch.randint(0, len(b), (PAIRS,), generator=gen)
                    j = torch.randint(0, len(b), (PAIRS,), generator=gen)
                    dv = vt[b][i] - vt[b][j]
                    m = dv != 0
                    if not m.any():
                        continue
                    loss = (dv[m].abs() * torch.nn.functional.softplus(-dv[m].sign() * (s[i][m] - s[j][m]))).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % EVERY == 0:
                with torch.no_grad():
                    sums[epoch] = sums.get(epoch, 0.0) + net(ev).squeeze(1).numpy().astype(np.float64)
        if keep_net:
            nets.append([p.detach().numpy().copy() for p in net.parameters()])
    out = {e: x / len(SEEDS) for e, x in sums.items()}
    return (out, {"mu": mu, "sd": sd, "nets": nets}) if keep_net else out


def score_setting(X_fit, v_fit, seq_fit, X_eval, setting, epochs=None):
    """{epoch: scores} (epoch -1 for cs_gbm)."""
    if setting["learner"] == "cs_gbm":
        m = fit_cs(X_fit, v_fit, seq_fit, setting)
        return {-1: m.decision_function(X_eval)} if m is not None else {}
    s = dict(setting, kind=setting["learner"].split("_")[0], epochs=epochs or EPOCHS)
    return fit_mlp(X_fit, v_fit, seq_fit, X_eval, s)


def settings(budgets):
    out = [{"learner": "cs_gbm", "lam_q": l, "eps_m": e} for l in LAM_Q for e in EPS_M]
    out += [{"learner": "topk_mlp", "T": t, "alpha": a} for t in TEMPS for a in budgets]
    return out + [{"learner": "pair_mlp"}]


def key(s):
    return json.dumps({k: x for k, x in s.items() if k not in ("epoch",)}, sort_keys=True)


def data_file(cell):
    return WORK / f"data__{cell}.npz"


def save_data(cell, X, v, seq, split, cols):
    np.savez(data_file(cell), X=X, v=v, seq=seq.astype(str), split=split.astype(str), cols=np.array(cols, dtype=str))


def load_data(cell, sub=None):
    z = np.load(data_file(cell), allow_pickle=False)
    X = z["X"]
    if sub is not None:
        X = X[:, sub]
    return X, z["v"], z["seq"], z["split"]


def ks_of(n, budgets, floor):
    return np.array([t234.k_floor(n, b) if floor else t234.k_round(n, b) for b in budgets])


def _select_task(t):
    """Val gains of one setting: fit on train, realized gain on val at every budget it may serve."""
    t0 = time.time()
    X, v, seq, split = load_data(t["cell"], t.get("sub"))
    tr, va = split == "train", split == "val"
    if t["variant"] == "capped":
        c = cap_of(v[tr])
        v = np.clip(v, -c, c)
    sc = score_setting(X[tr], v[tr], seq[tr], X[va], t["setting"])
    rows = []
    budgets = t["budgets"]
    use = [t["setting"]["alpha"]] if t["setting"]["learner"] == "topk_mlp" else budgets
    ks = ks_of(int(va.sum()), use, t.get("floor", False))
    for e, s in sc.items():
        g = xl.gains(s, v[va], ks)
        for b, x in zip(use, g):
            rows.append({"cell": t["cell"], "variant": t["variant"], "budget": b, "learner": t["setting"]["learner"],
                         "setting": key(t["setting"]), "epoch": e, "val_gain": float(x)})
    print(f"  {t['cell']} {t['variant']} {key(t['setting'])}: [{time.time() - t0:.0f}s]", flush=True)
    return rows


def choose(val):
    """The maximum per cell, variant and budget; ties to the first in enumeration order."""
    rows = {}
    for r in val.to_dict("records"):
        k = (r["cell"], r["variant"], r["budget"])
        if k not in rows or r["val_gain"] > rows[k]["val_gain"]:
            rows[k] = r
    return rows


def _refit_task(t):
    """Refit on `fit`, score `ev`; the masks come from the split column."""
    X, v, seq, split = load_data(t["cell"], t.get("sub"))
    fit, ev = np.isin(split, t["fit"]), np.isin(split, t["ev"])
    if t["variant"] == "capped":
        c = cap_of(v[fit])
        v = np.clip(v, -c, c)
    s = json.loads(t["setting"])
    if t["learner"] == "gate_gbm":
        m = predict.make_model("gbm", "reg", 0)
        with threadpool_limits(limits=1, user_api="openmp"):
            m.fit(X[fit], v[fit])
        return t["tag"], m.predict(X[ev])
    out = score_setting(X[fit], v[fit], seq[fit], X[ev], s, epochs=None if t["epoch"] < 0 else int(t["epoch"]))
    return t["tag"], out[int(t["epoch"])]


def refits(tasks):
    with Pool(NPROC) as p:
        return dict(p.map(_refit_task, tasks, chunksize=1))


def xl_setup(args):
    t227 = _load("t227", "227_r1_target_swap.py")
    a = argparse.Namespace(det=str(t227.DET), dataroot=args.dataroot)
    cells_, log_of = t227.setup(a)
    return cells_, log_of


def cname(c):
    return f"{c['geometry']}_{c['system']}"


def stage_select(args):
    run = runmeta.new_run("r9_select", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    cells_, _ = xl_setup(args)
    names = []
    for c in cells_:
        d, cols = c["d"], c["fcols"]
        check_columns(cols)
        assert len(cols) == 65
        save_data(cname(c), matrix(d, cols), (d[c["cheap"]] - d[c["full"]]).to_numpy(float), d.seq.to_numpy(), d.split.to_numpy(), cols)
        names.append(cname(c))
        if cname(c) == PRIMARY_CELL:
            t235 = _load("t235", "235_r8_reference.py")
            r8 = t235.vtable(*t235.PRIMARY, "mono").astype({"seq": str, "frame": int})
            j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(r8[["seq", "frame", "J_cheap", "J_full"]], on=["seq", "frame"],
                                                                            how="left", validate="one_to_one")
            assert j.J_cheap.notna().all()
            save_data("r8_b4", matrix(d, cols), (j.J_cheap - j.J_full).to_numpy(float), d.seq.to_numpy(), d.split.to_numpy(), cols)
            np.savez(WORK / "r8_b4_jcheap.npz", j=j.J_cheap.to_numpy(float))
            names.append("r8_b4")
    tasks = []
    for name in names:
        for s in settings(list(xl.SWEEP)):
            tasks.append({"cell": name, "variant": "V", "setting": s, "budgets": list(xl.SWEEP)})
        if name != "r8_b4":
            for s in settings([0.2]):
                tasks.append({"cell": name, "variant": "capped", "setting": s, "budgets": [0.2]})
    print(f"{len(tasks)} settings", flush=True)
    with Pool(NPROC) as p:
        rows = [r for x in p.imap(_select_task, tasks, chunksize=1) for r in x]
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "val_settings.csv", index=False)
    ch = choose(val)
    sel = [dict(r) for r in ch.values()]
    (OUT / "selected.json").write_text(json.dumps(sel, indent=1))
    pd.DataFrame(sel).to_csv(run / "selected.csv", index=False)
    print(pd.DataFrame(sel)[["cell", "variant", "budget", "learner", "epoch"]].to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ test

def tag_of(r):
    return hashlib.blake2b(f"{r['cell']}|{r['variant']}|{r['setting']}|{r['epoch']}".encode(), digest_size=8).hexdigest()


def two_sided(x):
    x = x[np.isfinite(x)]
    return float(min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))


def test_scores():
    """Every refit the test stage needs, once each: scores of the 228 (with gate_gbm on V and on capped V)."""
    f = WORK / "scores.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    sel = json.loads((OUT / "selected.json").read_text())
    val = pd.read_csv(OUT / "val_settings.csv")
    rows = list(sel)
    # each learner's own best at 20% (descriptive)
    for (cell, variant), g in val[val.budget == 0.2].groupby(["cell", "variant"]):
        for l, h in g.groupby("learner"):
            rows.append(h.sort_values("val_gain", ascending=False, kind="stable").iloc[0].to_dict())
    tasks = {}
    for r in rows:
        t = {"cell": r["cell"], "variant": r["variant"], "setting": r["setting"], "epoch": int(r["epoch"]), "learner": r["learner"],
             "fit": ["train", "val"], "ev": ["test"], "tag": tag_of(r)}
        tasks[t["tag"]] = t
    for cell in sorted({r["cell"] for r in rows}):
        for variant in ("V", "capped") if cell != "r8_b4" else ("V",):
            tasks[f"gate|{cell}|{variant}"] = {"cell": cell, "variant": variant, "setting": "{}", "epoch": -1, "learner": "gate_gbm",
                                               "fit": ["train", "val"], "ev": ["test"], "tag": f"gate|{cell}|{variant}"}
    print(f"{len(tasks)} refits", flush=True)
    s = refits(list(tasks.values()))
    f.write_bytes(pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL))
    return s


def draws(units, n, seed=0):
    uniq = np.array(sorted(set(units)))
    idx = {u: np.flatnonzero(units == u) for u in uniq}
    rng = np.random.default_rng(seed)
    return [np.concatenate([idx[u] for u in uniq[rng.integers(0, len(uniq), len(uniq))]]) for _ in range(n)]


def delta_row(sd, sg, v, j, seq, log, q, floor=(False, False), vsrandom=True):
    """Direct against gate_gbm at budget q: points as shares of the all-cheap loss, paired intervals; scene and log."""
    tot, n = float(j.sum()), len(v)
    kd, kg = t234.k_floor(n, q[0]) if floor[0] else t234.k_round(n, q[0]), t234.k_floor(n, q[1]) if floor[1] else t234.k_round(n, q[1])
    gd, gg = t234.gain(sd, v, kd), t234.gain(sg, v, kg)
    rnd = kd / n * v.sum()
    prize = t234.gain(v, v, kd)
    r = {"frames": n, "k_direct": kd, "k_gate": kg, "all_cheap_loss": tot, "gain_direct": gd, "gain_gate": gg, "share_direct": gd / tot,
         "share_gate": gg / tot, "delta_share": (gd - gg) / tot, "direct_over_random_share": (gd - rnd) / tot,
         "gate_over_random_share": (gg - kg / n * v.sum()) / tot, "ndg_direct": gd / prize if prize > EPS else np.nan,
         "ndg_gate": gg / prize if prize > EPS else np.nan}
    for unit, units in (("scene", seq), ("log", log)):
        x, xr = np.full(NBOOT, np.nan), np.full(NBOOT, np.nan)
        for b, t in enumerate(draws(units, NBOOT)):
            nt = len(t)
            kd_ = t234.k_floor(nt, q[0]) if floor[0] else t234.k_round(nt, q[0])
            kg_ = t234.k_floor(nt, q[1]) if floor[1] else t234.k_round(nt, q[1])
            a = t234.gain(sd[t], v[t], kd_)
            x[b] = (a - t234.gain(sg[t], v[t], kg_)) / tot
            xr[b] = (a - kd_ / nt * v[t].sum()) / tot
        sfx = "" if unit == "scene" else "_log"
        r[f"delta_lo{sfx}"], r[f"delta_hi{sfx}"] = xl.ci(x)
        r[f"delta_p{sfx}"] = two_sided(x)
        r[f"direct_over_random_lo{sfx}"], r[f"direct_over_random_hi{sfx}"] = xl.ci(xr)
    return r


def test_frames(cell):
    """(v, all-cheap, seq, log) of the 228 test scenes for a cell."""
    X, v, seq, split = load_data(cell)
    log_of = dict(zip(t234.meta().scene, t234.meta().log))
    te = split == "test"
    if cell == "r8_b4":
        jc = np.load(WORK / "r8_b4_jcheap.npz")["j"]
    else:
        col = {"brake": "J_cheap", "plan_ade": "JC_ade_cheap", "plan_fde": "JC_fde_cheap"}[cell.split("_", 1)[1]]
        jc = _CF[cell][col].to_numpy(float)                       # the data file keeps the cell's frame order
    return v[te], jc[te], seq[te], np.array([log_of[s] for s in seq[te]])


_CF = {}


def fill_cell_frames(args):
    cells_, _ = xl_setup(args)
    for c in cells_:
        _CF[cname(c)] = c["d"]
    return cells_


def stage_test(args):
    runmeta.new_run("r9_test", vars(args))
    fill_cell_frames(args)
    sc = test_scores()
    sel = json.loads((OUT / "selected.json").read_text())
    by = {(r["cell"], r["variant"], r["budget"]): r for r in sel}
    data = {}

    def get(cell):
        if cell not in data:
            data[cell] = test_frames(cell)
        return data[cell]
    rows = []
    for (cell, variant, b), r in by.items():
        if variant != "V":
            continue
        v, j, seq, log = get(cell)
        rows.append({"cell": cell, "budget": b, "chosen": r["learner"], "setting": r["setting"], "epoch": r["epoch"], "val_gain": r["val_gain"],
                     **delta_row(sc[tag_of(r)], sc[f"gate|{cell}|V"], v, j, seq, log, (b, b))})
        print(f"  {cell} {b}: {rows[-1]['delta_share']:+.4f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cells_budgets.csv", index=False)
    # the two primaries, Holm
    p = t[(t.budget == 0.2) & t.cell.isin([PRIMARY_CELL, ADE_CELL])].copy()
    p["holm_rejects"] = xl.holm(p.delta_p.to_numpy(), alpha=0.05)
    p["reading"] = np.where(p.holm_rejects, np.where(p.delta_share > 0, "direct better", "regression better"), "not resolved")
    p.to_csv(OUT / "primary.csv", index=False)
    print(p[["cell", "chosen", "delta_share", "delta_lo", "delta_hi", "delta_p", "holm_rejects", "reading"]].to_string(index=False), flush=True)
    # each learner's own validation-best (descriptive)
    val = pd.read_csv(OUT / "val_settings.csv")
    rows = []
    for (cell, variant), g in val[(val.budget == 0.2) & (val.variant == "V")].groupby(["cell", "variant"]):
        v, j, seq, log = get(cell)
        for l, h in g.groupby("learner"):
            r = h.sort_values("val_gain", ascending=False, kind="stable").iloc[0].to_dict()
            rows.append({"cell": cell, "learner": l, "setting": r["setting"], "epoch": r["epoch"],
                         **delta_row(sc[tag_of(r)], sc[f"gate|{cell}|V"], v, j, seq, log, (0.2, 0.2))})
    pd.DataFrame(rows).to_csv(OUT / "learners_descriptive.csv", index=False)
    # capped: chosen and compared on capped V; the V-trained pair evaluated on capped V beside it
    rows = []
    for cell in sorted({k[0] for k in by if k[1] == "capped"}):
        v, j, seq, log = get(cell)
        X, va, _, split = load_data(cell)
        c = cap_of(va[np.isin(split, ["train", "val"])])
        vc = np.clip(v, -c, c)
        rc, rv = by[(cell, "capped", 0.2)], by[(cell, "V", 0.2)]
        for lab, r, g in (("trained on capped V", rc, "capped"), ("trained on V", rv, "V")):
            rows.append({"cell": cell, "training": lab, "evaluated_on": "capped V", "cap": c, "chosen": r["learner"], "setting": r["setting"],
                         **delta_row(sc[tag_of(r)], sc[f"gate|{cell}|{g}"], vc, j, seq, log, (0.2, 0.2))})
        rows.append({"cell": cell, "training": "trained on capped V", "evaluated_on": "V", "cap": c, "chosen": rc["learner"], "setting": rc["setting"],
                     **delta_row(sc[tag_of(rc)], sc[f"gate|{cell}|capped"], v, j, seq, log, (0.2, 0.2))})
    pd.DataFrame(rows).to_csv(OUT / "capped.csv", index=False)
    # R8's primary variant
    rows = []
    for b in xl.SWEEP:
        r = by[("r8_b4", "V", b)]
        v, j, seq, log = get("r8_b4")
        rows.append({"cell": "lead_b4 ref+ctrl mono", "budget": b, "chosen": r["learner"], "setting": r["setting"],
                     **delta_row(sc[tag_of(r)], sc["gate|r8_b4|V"], v, j, seq, log, (b, b))})
    pd.DataFrame(rows).to_csv(OUT / "r8_variant.csv", index=False)
    # R7's influence rows on the two primaries
    rows = []
    for cell in (PRIMARY_CELL, ADE_CELL):
        v, j, seq, log = get(cell)
        r = by[(cell, "V", 0.2)]
        sd, sg = sc[tag_of(r)], sc[f"gate|{cell}|V"]
        two = set(seq[np.isin(log, t234.LOGS)])
        sets = [("all", set()), ("minus scene-1051", {"scene-1051"}), ("minus scene-1081", {"scene-1081"}),
                ("minus both", {"scene-1051", "scene-1081"}), ("minus the two logs' test scenes", two)]
        drop = {}
        for s in sorted(set(seq)):
            m = seq != s
            n = int(m.sum())
            k = t234.k_round(n)
            drop[s] = (t234.gain(sd[m], v[m], k) - t234.gain(sg[m], v[m], k)) / float(j[m].sum())
        base = (t234.gain(sd, v, t234.k_round(len(v))) - t234.gain(sg, v, t234.k_round(len(v)))) / float(j.sum())
        worst = max(drop, key=lambda s: abs(drop[s] - base))
        sets.append((f"minus most influential ({worst})", {worst}))
        for name, rem in sets:
            m = ~np.isin(seq, sorted(rem))
            rows.append({"cell": cell, "subset": name, "drop_one_min": min(drop.values()), "drop_one_max": max(drop.values()),
                         **delta_row(sd[m], sg[m], v[m], j[m], seq[m], log[m], (0.2, 0.2))})
    pd.DataFrame(rows).to_csv(OUT / "influence.csv", index=False)


# ------------------------------------------------------------------------------------------------ crit

def np_gbm(m):
    out = []
    for p in m._predictors:
        nd = p[0].nodes
        out.append([(bool(a), int(b), float(c), int(d), int(e), float(f)) for a, b, c, d, e, f in
                    zip(nd["is_leaf"], nd["feature_idx"], nd["num_threshold"], nd["left"], nd["right"], nd["value"])])
    return {"kind": "gbm", "base": float(np.ravel(m._baseline_prediction)[0]), "trees": out}


def stage_crit(args):
    t231 = _load("t231", "231_r3_gate_cost.py")
    runmeta.new_run("r9_crit", vars(args))
    fill_cell_frames(args)
    costs = pd.read_csv(REB / "R3" / "feature_costs.csv")
    c_c, c_f = t231.engines(costs, "ms")
    sel = json.loads((REB / "R3" / "selected.json").read_text())
    subset = sel["gate_gbm"]["subset"]
    cs_r3 = float(t231.subset_cost(costs, subset, "gate_gbm"))
    f_r3 = t231.frac_at(0.2, c_c, c_f, cs_r3)
    X, v, seq, split = load_data(PRIMARY_CELL)
    cols = list(np.load(data_file(PRIMARY_CELL))["cols"])
    groups = t231.group_columns(cols)
    use = t231.subset_columns(subset, groups)
    check_columns(use)
    sub = [cols.index(c) for c in use]
    tasks = [{"cell": PRIMARY_CELL, "sub": sub, "variant": "V", "setting": s, "budgets": [f_r3], "floor": True}
             for s in settings([f_r3])]
    with Pool(NPROC) as p:
        rows = [r for x in p.imap(_select_task, tasks, chunksize=1) for r in x]
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "crit_val.csv", index=False)
    r = list(choose(val).values())[0]
    print("crit chosen:", r, flush=True)
    fit, te = np.isin(split, ["train", "val"]), split == "test"
    Xs = X[:, sub]
    s = json.loads(r["setting"])
    # chosen model, with the object the allocator file evaluates
    if r["learner"] == "cs_gbm":
        m = fit_cs(Xs[fit], v[fit], seq[fit], s)
        sd, spec = m.decision_function(Xs[te]), np_gbm(m)
    else:
        out, sp = fit_mlp(Xs[fit], v[fit], seq[fit], Xs[te], dict(s, kind=r["learner"].split("_")[0], epochs=int(r["epoch"])), keep_net=True)
        sd, spec = out[int(r["epoch"])], dict(sp, kind="mlp")
    g = predict.make_model("gbm", "reg", 0)
    g.fit(Xs[fit], v[fit])
    sg = g.predict(Xs[te])
    prof = {}
    for name, sp in (("direct", spec), ("gate_gbm", np_gbm(g))):
        f = WORK / f"alloc_{name}.pkl"
        f.write_bytes(pickle.dumps(dict(sp, subset=subset, cols=use), protocol=pickle.HIGHEST_PROTOCOL))
        o = OUT / f"profile_crit_{name}.json"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "236b_r9_allocator.py"), "--check", str(f)], check=True,
                       env=dict(os.environ, PYTHONPATH=str(ROOT / "src")))
        subprocess.run([sys.executable, str(ROOT / "scripts" / "151_profile_allocator.py"), str(ROOT / "scripts" / "236b_r9_allocator.py"),
                        "--tracks", "nuScenes", "--frames", "400", "--out", str(o)], check=True,
                       env={**{k: x for k, x in os.environ.items() if k not in BLAS_ENV}, "R9_ALLOC": str(f)})
        prof[name] = float(json.loads(o.read_text())["ms"]["nuScenes"])
    print("profiles (ms):", prof, flush=True)
    vt, jt, seqt, logt = test_frames(PRIMARY_CELL)
    n = len(vt)
    rows = []
    for lab, csd, csg in (("measured with 151 (per input, plain numpy)", prof["direct"], prof["gate_gbm"]),
                          ("gate at R3's batched C_S, direct at its 151 cost", prof["direct"], cs_r3)):
        fd, fg = t231.frac_at(0.2, c_c, c_f, csd), t231.frac_at(0.2, c_c, c_f, csg)
        rows.append({"costs": lab, "subset": subset, "chosen": r["learner"], "setting": r["setting"], "epoch": r["epoch"], "C_S_direct_ms": csd,
                     "C_S_gate_ms": csg, "frac_direct": fd, "frac_gate": fg,
                     **delta_row(sd, sg, vt, jt, seqt, logt, (fd, fg), floor=(True, True))})
    t = pd.DataFrame(rows)
    # over random at floor(0.2 n), R3's reading: direct_over_random in delta_row uses the matched count, so restate it
    t["direct_over_random20_share"] = [(g_ - t234.k_floor(n, 0.2) / n * vt.sum()) / jt.sum() for g_ in t.gain_direct]
    t.to_csv(OUT / "crit.csv", index=False)
    print(t.T.to_string(), flush=True)


# ------------------------------------------------------------------------------------------------ frozen

def stage_frozen(args):
    runmeta.new_run("r9_frozen", vars(args))
    t124 = _load("t124", "124_causal_threshold.py")
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr_fcols, _ = t124.t120.register_features()
    cells_ = list(t124.core_cells(splits)) + [c for c in t124.nuplan_cells(splits, nr_fcols) if c["system"] == "pdm_closed"]
    assert len(cells_) == 12, len(cells_)                                           # 10 core + the 2 PDM-Closed cells
    tasks, meta = [], {}
    for i, c in enumerate(cells_):
        d = c["d"]
        check_columns(c["fcols"])
        ids = ("seq", "frame") if c["track"] != "nuPlan" else ("scenario", "iteration")
        seq = d[ids[0]].astype(str).to_numpy()
        name = f"frozen{i:02d}"
        save_data(name, matrix(d, c["fcols"]), c["v_all"], seq, d.split.to_numpy(), c["fcols"])
        meta[name] = dict(track=c["track"], geometry=c["geometry"], system=c["system"], target=c["target"], ids=ids,
                          id0=seq, id1=d[ids[1]].to_numpy(), split=d.split.to_numpy())
        for s in settings(list(FROZEN_Q)):
            tasks.append({"cell": name, "variant": "V", "setting": s, "budgets": list(FROZEN_Q)})
    print(f"{len(tasks)} settings", flush=True)
    with Pool(NPROC) as p:
        rows = [r for x in p.imap(_select_task, tasks, chunksize=1) for r in x]
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "frozen_val.csv", index=False)
    ch = choose(val)
    ref = {}
    for (cell, variant, b), r in ch.items():
        tag = f"{cell}|{tag_of(r)}"
        ref.setdefault(tag, {"cell": cell, "variant": "V", "setting": r["setting"], "epoch": int(r["epoch"]), "learner": r["learner"],
                             "fit": ["train", "val"], "ev": ["test"], "tag": tag})
    for name in meta:
        ref[f"gate|{name}"] = {"cell": name, "variant": "V", "setting": "{}", "epoch": -1, "learner": "gate_gbm", "fit": ["train", "val"],
                               "ev": ["test"], "tag": f"gate|{name}"}
    sc = refits(list(ref.values()))
    subs = {"direct": [], "gate_gbm": []}
    for name, m in meta.items():
        te = m["split"] == "test"
        base = {"track": m["track"], "geometry": m["geometry"], "system": m["system"], "target": m["target"],
                m["ids"][0]: m["id0"][te], m["ids"][1]: m["id1"][te]}
        a = dict(base)
        b_ = dict(base)
        for q in FROZEN_Q:
            r = ch[(name, "V", q)]
            a[f"score_q{int(round(q * 100))}"] = np.round(sc[f"{name}|{tag_of(r)}"], 9)
            b_[f"score_q{int(round(q * 100))}"] = np.round(sc[f"gate|{name}"], 9)
        subs["direct"].append(pd.DataFrame(a))
        subs["gate_gbm"].append(pd.DataFrame(b_))
    pd.DataFrame([dict(cell=k[0], budget=k[2], **{**meta[k[0]], "id0": None, "id1": None, "split": None, "ids": None}, chosen=r["learner"],
                       setting=r["setting"], epoch=r["epoch"], val_gain=r["val_gain"]) for k, r in ch.items()]).drop(
        columns=["id0", "id1", "split", "ids"]).to_csv(OUT / "frozen_selected.csv", index=False)
    for name, parts in subs.items():
        for track, g in pd.concat(parts, ignore_index=True).groupby("track", sort=False):
            g = g.dropna(axis=1, how="all")
            f = OUT / f"frozen_submission_{name}_{track}.csv"
            g.to_csv(f, index=False, float_format="%.9f")
            subprocess.run([sys.executable, str(ROOT / "evaluate_submission.py"), str(f), "--out",
                            str(OUT / f"frozen_scored_{name}_{track}.csv")], check=True, cwd=str(ROOT))


# ------------------------------------------------------------------------------------------------ causal

def stage_causal(args):
    runmeta.new_run("r9_causal", vars(args))
    t124 = _load("t124", "124_causal_threshold.py")
    fill_cell_frames(args)
    sel = json.loads((OUT / "selected.json").read_text())
    by = {(r["cell"], r["variant"], r["budget"]): r for r in sel}
    tasks = []
    for cell in ("mono_brake", "mono_plan_ade", "mono_plan_fde", "oracle_brake", "oracle_plan_ade", "oracle_plan_fde"):
        r = by[(cell, "V", 0.2)]
        for lab, ln, st, ep in (("direct", r["learner"], r["setting"], int(r["epoch"])), ("gate_gbm", "gate_gbm", "{}", -1)):
            tasks.append({"cell": cell, "variant": "V", "setting": st, "epoch": ep, "learner": ln, "fit": ["train"], "ev": ["val", "test"],
                          "tag": f"{cell}|{lab}"})
    s = refits(tasks)
    rows = []
    for cell in sorted({t["cell"] for t in tasks}):
        X, v_all, seq_all, split = load_data(cell)
        ev = np.isin(split, ["val", "test"])
        va, te = split[ev] == "val", split[ev] == "test"
        v, j, seq, log = test_frames(cell)
        frame = _CF[cell].frame.to_numpy()[split == "test"]
        stream = np.lexsort((frame, seq))
        res = {}
        for lab in ("direct", "gate_gbm"):
            x = s[f"{cell}|{lab}"]
            tau, p = t124.calibrate(x[va], 0.2)
            rng = np.random.default_rng(0)
            ma = t124.mask_A(x[te], tau, p, rng)
            mb = t124.apply_cap(ma, 0.2, seq, stream)
            res[lab] = {"A": ma, "B": mb, "tau": tau}
        tot = float(j.sum())
        for pol in ("A", "B"):
            md, mg = res["direct"][pol], res["gate_gbm"][pol]
            r = {"cell": cell, "policy": pol, "rate_direct": float(md.mean()), "rate_gate": float(mg.mean()),
                 "share_direct": float(v[md].sum()) / tot, "share_gate": float(v[mg].sum()) / tot,
                 "delta_share": float(v[md].sum() - v[mg].sum()) / tot,
                 "direct_over_random_share": float(v[md].sum() - md.mean() * v.sum()) / tot}
            for unit, units in (("scene", seq), ("log", log)):
                x = np.full(NBOOT, np.nan)
                y = np.full(NBOOT, np.nan)
                for b, t in enumerate(draws(units, NBOOT)):
                    x[b] = (v[t][md[t]].sum() - v[t][mg[t]].sum()) / tot
                    y[b] = (v[t][md[t]].sum() - md[t].mean() * v[t].sum()) / tot
                sfx = "" if unit == "scene" else "_log"
                r[f"delta_lo{sfx}"], r[f"delta_hi{sfx}"] = xl.ci(x)
                r[f"direct_over_random_lo{sfx}"], r[f"direct_over_random_hi{sfx}"] = xl.ci(y)
            rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "causal.csv", index=False)
    print(pd.DataFrame(rows).iloc[:, :9].to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ ladder

def full_features():
    """The same extractor on the full cache, with the cheap cache's image statistics (the full cache stores none)."""
    from rap import features as F
    from rap.cache import DetCache
    rows = []
    op = RiskConfig().op_conf
    for f in sorted((DET / "ns_full_640").glob("*.npz")):
        c, cc = DetCache(f), DetCache(DET / "ns_cheap_320" / f.name)
        for i, fr in enumerate(c.frames):
            sc = cc.scalars(i)
            img = {k: x for k, x in sc.items() if k.startswith(("img_bright", "img_dark", "img_lap", "img_edge", "img_contrast", "motion"))}
            rows.append({"seq": f.stem, "frame": int(fr), **F.frame_features(c.det(i), c.geo(i), img, sc.get("img_w", 1600.0) * sc.get("img_h", 900.0),
                                                                            G.PRIMARY, op)})
    return pd.DataFrame(rows)


def reference_counts(d, dataroot):
    from rap.nusc import NuScenesDB, make_adapter
    nad = make_adapter(NuScenesDB(dataroot, "v1.0-trainval"))
    pp, cfg = P.PlannerParams(), RiskConfig()
    out = []
    for s, g_ in d[["seq", "frame", "v_ego"]].groupby("seq", sort=False):
        geom = nad.geometry(s)
        for fr, v_ego in zip(g_.frame.to_numpy(int), g_.v_ego.to_numpy(float)):
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            m = P.in_corridor(g["lat_min"], g["lat_max"], pp.corridor_half_w) & (g["long_near"] <= pp.max_range)
            r = {"seq": s, "frame": int(fr), "ref_count": int(m.sum()), "ref_nearest": float(pp.max_range), "ref_speed": 0.0, "ref_closing": 0.0}
            if m.any():
                h = g[m]
                i = int(np.argmin(h["long_near"]))
                rr = float(h["range_rate"][i]) if bool(h["has_rate"][i]) else np.nan
                r.update(ref_nearest=float(h["long_near"][i]),
                         ref_speed=float(P.lead_speed(h["long_near"][i:i + 1], h["ttc"][i:i + 1], v_ego, np.array([rr]))[0]),
                         ref_closing=float(-rr) if np.isfinite(rr) else float(v_ego))
            out.append(r)
    return pd.DataFrame(out)


def stage_ladder(args):
    run = runmeta.new_run("r9_ladder", vars(args))
    t183 = _load("t183", "183_xl_temporal.py")
    from rap.nusc import NuScenesDB
    cells_ = fill_cell_frames(args)
    c = [x for x in cells_ if cname(x) == PRIMARY_CELL][0]
    d = c["d"]
    cols = c["fcols"]
    keys = d[["seq", "frame"]].astype({"seq": str, "frame": int})
    feats, _ = xl.features(str(DET))
    f = feats.astype({"seq": str, "frame": int}).reset_index(drop=True)
    kk = f[["seq", "frame"]].assign(_row=np.arange(len(f)))
    H = t183.history_matrix(f[["seq", "frame"]], matrix(f, cols))
    row = keys.merge(kk, on=["seq", "frame"], how="left", validate="one_to_one")._row.to_numpy(int)
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    vtab = pd.read_csv(t234.TABLES / "v_ego_causal.csv").astype({"seq": str, "frame": int})
    e = keys.merge(t183.ego_dynamics(db, sorted(keys.seq.unique()), vtab), on=["seq", "frame"], how="left", validate="one_to_one")
    L0 = matrix(d, cols)
    assert (H[row][:, -len(cols):] == L0).all()
    L1 = np.hstack([H[row], e[["ego_acc", "yaw_rate"]].to_numpy(float)])
    ff = full_features().astype({"seq": str, "frame": int})
    assert sorted(c_ for c_ in ff.columns if c_.startswith("feat_")) == sorted(cols)
    fj = keys.merge(ff, on=["seq", "frame"], how="left", validate="one_to_one")
    assert fj[cols[0]].notna().all()
    Lf = matrix(fj, cols)
    L2 = np.hstack([L1, Lf, Lf - L0])
    rc = reference_counts(d.astype({"seq": str, "frame": int}), args.dataroot)
    rj = keys.merge(rc, on=["seq", "frame"], how="left", validate="one_to_one")
    L3 = np.hstack([L2, rj[["ref_count", "ref_nearest", "ref_speed", "ref_closing"]].to_numpy(float)])
    v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    seq, split = d.seq.astype(str).to_numpy(), d.split.astype(str).to_numpy()
    rungs = {"L0": L0, "L1": L1, "L2": L2, "L3": L3}
    for n, X in rungs.items():
        save_data(f"ladder_{n}", X, v, seq, split, [str(i) for i in range(X.shape[1])])
    print({n: X.shape[1] for n, X in rungs.items()}, flush=True)
    # a fresh process for the fits: workers forked from the two NuScenesDB copies here exhaust the Jetson's memory
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve()), "--stage", "ladder_fit", "--dataroot", args.dataroot])


def stage_ladder_fit(args):
    run = runmeta.new_run("r9_ladder_fit", vars(args))
    fill_cell_frames(args)
    rungs = {n: load_data(f"ladder_{n}")[0] for n in ("L0", "L1", "L2", "L3")}
    print({n: X.shape[1] for n, X in rungs.items()}, flush=True)
    sel = json.loads((OUT / "selected.json").read_text())
    r = [x for x in sel if x["cell"] == PRIMARY_CELL and x["variant"] == "V" and x["budget"] == 0.2][0]
    tasks = []
    for n in rungs:
        tasks.append({"cell": f"ladder_{n}", "variant": "V", "setting": "{}", "epoch": -1, "learner": "gate_gbm", "fit": ["train", "val"],
                      "ev": ["test"], "tag": f"gate_gbm|{n}"})
        tasks.append({"cell": f"ladder_{n}", "variant": "V", "setting": r["setting"], "epoch": int(r["epoch"]), "learner": r["learner"],
                      "fit": ["train", "val"], "ev": ["test"], "tag": f"direct|{n}"})
    s = refits(tasks)
    (WORK / "ladder_scores.pkl").write_bytes(pickle.dumps(s, protocol=pickle.HIGHEST_PROTOCOL))
    vt, j, seqt, log = test_frames(PRIMARY_CELL)
    tot, n, k = float(j.sum()), len(vt), t234.k_round(len(vt))
    prize = t234.gain(vt, vt, k)
    aff = affected(vt)
    rows, drows = [], []
    for ln in ("gate_gbm", "direct"):
        for rn in rungs:
            x = s[f"{ln}|{rn}"]
            g = t234.gain(x, vt, k)
            rows.append({"learner": ln, "chosen": r["learner"] if ln == "direct" else "gate_gbm", "rung": rn, "input_dim": rungs[rn].shape[1],
                         "deployable": rn in ("L0", "L1"), "gain": g, "share": g / tot, "ndg": g / prize,
                         "spearman_affected": float(spearmanr(x[aff], vt[aff]).statistic), "oracle_gain": prize, "oracle_share": prize / tot})
        for rn in ("L1", "L2", "L3"):
            sub = delta_row(s[f"{ln}|{rn}"], s[f"{ln}|L0"], vt, j, seqt, log, (0.2, 0.2))
            drows.append({"learner": ln, "delta": f"{rn} - L0", "value_share": sub["delta_share"], "lo": sub["delta_lo"], "hi": sub["delta_hi"],
                          "lo_log": sub["delta_lo_log"], "hi_log": sub["delta_hi_log"]})
    t, dt = pd.DataFrame(rows), pd.DataFrame(drows)
    read = {}
    for ln in ("gate_gbm", "direct"):
        l0 = t[(t.learner == ln) & (t.rung == "L0")].iloc[0]
        l3 = t[(t.learner == ln) & (t.rung == "L3")].iloc[0]
        d2 = dt[(dt.learner == ln) & (dt.delta == "L2 - L0")].iloc[0]
        half_gap = 0.5 * (l0.oracle_share - l0.share)
        info = (d2.lo > 0) and d2.value_share > half_gap
        arch = l3.gain < 0.5 * l3.oracle_gain
        read[ln] = "mainly information" if info else ("mainly architecture or noise" if arch else "mixed")
    t["reading"] = t.learner.map(read)
    t.to_csv(OUT / "ladder.csv", index=False)
    dt.to_csv(OUT / "ladder_deltas.csv", index=False)
    print(t.to_string(index=False), "\n", dt.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("select", "test", "crit", "frozen", "causal", "ladder", "ladder_fit"))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
