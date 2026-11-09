#!/usr/bin/env python
"""Follow-up round 3, R15 (post hoc, a sensitivity): allocators fitted on the new nuScenes scenes, tested on the frozen-85 scenes whose
logs never enter the fit (docs/followup_protocol.md).  Fits run as 241's jobs.

  --stage prep    populations, the arrays of T and F (standard XL tables)     -> populations.csv, data/cache/xl/r14/data/r15_*.npz
  --stage gate    gate R15-0                                                   -> gate_r15_0.json (exit 1 on a failure)
  --stage q_cv    five log-grouped folds inside F: every setting, and the out-of-fold scores the causal threshold needs -> queues
  --stage cv      the choice inside F                                          -> cv_selected.json, cv_values.csv
  --stage q_fit   the chosen settings fit on all of F                          -> queues
  --stage test    T, once                                                      -> primary.csv, allocators.csv, budgets.csv, causal.csv,
                                                                                  crit.csv, drop_one_log.csv, power.csv
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta, xl                                                         # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                   # noqa: E402


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


R = _load("t241", "241_r14_exact.py")
C = R.C
REB = Path(RESULTS) / "followup"
OUT = REB / "R15"
TABLES = Path(CACHE) / "xl" / "tables850"
DET = Path(CACHE) / "nusc_det_xl"
CELLS = ("mono_brake", "oracle_brake", "mono_plan_ade", "oracle_plan_ade")
ALL_CELLS = ("mono_brake", "mono_plan_ade", "mono_plan_fde", "oracle_brake", "oracle_plan_ade", "oracle_plan_fde")
BUDGETS = tuple(xl.SWEEP)
FI, TE = {"split": ["fit"]}, {"split": ["test"]}
CRIT_RATE = R.CRIT_RATE


def name(cell):
    return f"r15_{cell}"


def fold(kf, side):
    return {"split": ["fit"], "fold": [0, kf, side]}


# ------------------------------------------------------------------------------------------------ prep

def populations():
    sp = xl.scene_split()
    m = pd.read_csv(Path(CACHE) / "xl" / "scene_meta.csv")
    s = sp.merge(m[["scene", "log"]], on="scene", how="left", validate="one_to_one")
    assert s.log.notna().all()
    shared = set(s[s.group == "frozen85"].log) & set(s[s.group == "new765"].log)
    s["role"] = np.where(s.group == "frozen85", "T", "F")
    s["excluded_reason"] = np.where(s.log.isin(shared), "log also has scenes in the other group", "")
    s.loc[s.log.isin(shared), "role"] = "excluded"
    return s[["scene", "log", "group", "role", "excluded_reason", "official", "deep_split"]], sorted(shared)


def stage_prep(args):
    runmeta.new_run("r15_prep", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    R.DATA.mkdir(parents=True, exist_ok=True)
    pop, shared = populations()
    pop.to_csv(OUT / "populations.csv", index=False)
    print("shared logs:", shared, "\n", pop.groupby(["group", "role"]).size(), flush=True)
    split_of = {r.scene: ("fit" if r.role == "F" else "test") for r in pop.itertuples() if r.role != "excluded"}
    log_of = dict(zip(pop.scene, pop.log))
    xl.use_ego_table(TABLES / "v_ego_causal.csv")
    feats, fcols = xl.features(str(DET))
    R.check_columns(fcols)
    t236 = _load("t236", "236_r9_direct.py")
    t231 = _load("t231", "231_r3_gate_cost.py")
    for c in xl.cells(TABLES, feats, fcols, split_of):
        cell = f"{c['geometry']}_{c['system']}"
        d = c["d"]
        assert len(c["fcols"]) == 65
        seq = np.asarray(d.seq.astype(str), dtype=str)
        arr = dict(X=t236.matrix(d, c["fcols"]), v=(d[c["cheap"]] - d[c["full"]]).to_numpy(float), seq=seq, split=np.asarray(d.split.astype(str), dtype=str),
                   log=np.array([log_of[x] for x in seq], dtype=str), frame=d.frame.to_numpy(int), j=d[c["cheap"]].to_numpy(float), cols=np.array(c["fcols"], dtype=str))
        np.savez(R.DATA / f"{name(cell)}.npz", **arr)
        if cell == "mono_brake":
            sub = json.loads((REB / "R3" / "selected.json").read_text())["gate_gbm"]["subset"]
            use = t231.subset_columns(sub, t231.group_columns(c["fcols"]))
            R.check_columns(use)
            np.savez(R.DATA / f"{name(cell)}_crit.npz", **dict(arr, X=t236.matrix(d, use), cols=np.array(use, dtype=str)))
        print(f"  {cell}: {len(d)} frames, fit {int((arr['split'] == 'fit').sum())}, test {int((arr['split'] == 'test').sum())}", flush=True)


# ------------------------------------------------------------------------------------------------ gate

def stage_gate(args):
    run = runmeta.new_run("r15_gate", vars(args))
    pop = pd.read_csv(OUT / "populations.csv")
    rep = {"checks": []}
    f_logs, t_logs = set(pop[pop.role == "F"].log), set(pop[pop.role == "T"].log)
    rep["checks"].append({"what": "(i) no log in both F and T", "shared": sorted(f_logs & t_logs), "F_logs": len(f_logs), "T_logs": len(t_logs),
                          "F_scenes": int((pop.role == "F").sum()), "T_scenes": int((pop.role == "T").sum()),
                          "excluded": pop[pop.role == "excluded"].scene.tolist(), "pass": not (f_logs & t_logs)})
    rel = pd.read_csv(Path(RESULTS) / "final" / "benchmark_decision_values.csv.gz", float_precision="round_trip", low_memory=False)
    rel = rel[rel.track == "nuScenes"]
    rows = []
    for cell in ALL_CELLS:
        z = R.data(name(cell))
        t = z["split"] == "test"
        g, s = cell.split("_", 1)
        r = rel[(rel.geometry == g) & (rel.system == s)][["seq", "frame", "V"]].astype({"seq": str, "frame": int})
        k = pd.DataFrame({"seq": z["seq"][t].astype(str), "frame": z["frame"][t].astype(int), "v": z["v"][t]})
        j = k.merge(r, on=["seq", "frame"], how="left", validate="one_to_one")
        rt = r[r.seq.isin(set(k.seq))]
        rows.append({"cell": cell, "T_frames": len(k), "released_frames_of_T_scenes": len(rt), "missing_in_released": int(j.V.isna().sum()),
                     "differences": int((j.V.to_numpy(float) != j.v.to_numpy(float)).sum()), "max_abs_diff": float(np.nanmax(np.abs(j.V - j.v)))})
    rep["checks"].append({"what": "(ii) V on T equals the released V (benchmark_decision_values.csv.gz)", "cells": rows,
                          "pass": all(r["differences"] == 0 and r["missing_in_released"] == 0 and r["T_frames"] == r["released_frames_of_T_scenes"] for r in rows)})
    jb = R.gbm_job(name("mono_brake"), FI, {"test": TE}, tag="r15 gate iii")
    with Pool(2) as p:
        a, b = p.map(R.run_job, [jb, jb], chunksize=1)
    d = float(np.abs(a["scores"][-1]["test"] - b["scores"][-1]["test"]).max())
    rep["checks"].append({"what": "(iii) gate_gbm refit on F twice: identical scores", "max_abs_diff": d, "pass": d == 0.0})
    rep["pass"] = all(c["pass"] for c in rep["checks"])
    for f in (OUT / "gate_r15_0.json", run / "gate_r15_0.json"):
        f.write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float), flush=True)
    if not rep["pass"]:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ CV inside F

def r9_settings(budgets):
    t236 = _load("t236", "236_r9_direct.py")
    return t236.settings(list(budgets)), t236.key


def lin_settings():
    return R.LIN_SET


def cv_jobs():
    sets, key = r9_settings(BUDGETS)
    out = {}
    for cell in CELLS:
        for kf in range(5):
            ev = {"held": fold(kf, "in")}
            fit = fold(kf, "out")
            out[(cell, "gate_gbm", "{}", kf)] = R.gbm_job(name(cell), fit, ev, tag=f"r15 cv {cell} {kf}")
            out[(cell, "ridge_V", "{}", kf)] = R.ridge_job(name(cell), fit, ev, tag=f"r15 cv {cell} {kf}")
            for s in sets:
                k = key(s)
                out[(cell, s["learner"], k, kf)] = R.J_({"data": name(cell), "fit": fit, "evals": ev, "variant": "V", "learner": s["learner"],
                                                        "tag": f"r15 cv {cell} {kf}", "params": {"setting": k}, "machine": "second"})
            for b in BUDGETS:
                for init, sig in R.LIN_SET:
                    out[(cell, "exact_lin", json.dumps({"budget": b, "init": init, "sigma0": sig}), kf)] = R.lin_job(
                        name(cell), fit, ev, b, init, sig, tag=f"r15 cv {cell} {b} {kf}", machine="second")
    return out


def stage_q_cv(args):
    """Two queues: the exact_lin rows only once R14's gate has passed (R14's code and settings fixed)."""
    assert json.loads((OUT / "gate_r15_0.json").read_text())["pass"]
    jobs = cv_jobs()
    R.write_queues("r15_cv", [j for k, j in jobs.items() if k[1] != "exact_lin"])
    if (R.OUT / "gate_r14_0.json").exists() and "exact_lin" in R.passed():          # exact_lin passes gate R14-0 (addendum R14, second run)
        R.write_queues("r15_cv_exact", [j for k, j in jobs.items() if k[1] == "exact_lin"])
    else:
        print("R14's gate has not passed: no exact_lin queue", flush=True)


def oof(jobs, cell, learner, k):
    """Pooled out-of-fold scores over F, per epoch."""
    z = R.data(name(cell))
    f = z["split"] == "fit"
    out = {}
    for kf in range(5):
        h = R.mask(fold(kf, "in"), z)
        for e, s in R.load_result(jobs[(cell, learner, k, kf)]["id"])["scores"].items():
            out.setdefault(e, np.full(len(z["v"]), np.nan))[h] = s["held"]
    for e in out:
        assert np.isfinite(out[e][f]).all()
    return {e: x[f] for e, x in out.items()}


def stage_cv(args):
    runmeta.new_run("r15_cv", vars(args))
    J = cv_jobs()
    sets, key = r9_settings(BUDGETS)
    rows = []
    for cell in CELLS:
        z = R.data(name(cell))
        v = z["v"][z["split"] == "fit"].astype(float)
        for s in sets:
            k = key(s)
            sc = oof(J, cell, s["learner"], k)
            use = [s["alpha"]] if s["learner"] == "topk_mlp" else BUDGETS
            for b in use:
                kk = C.k_round(len(v), b)
                for e, x in sorted(sc.items()):
                    rows.append({"cell": cell, "budget": b, "learner": s["learner"], "setting": k, "epoch": e, "cv_gain": R.gain(x, v, kk)})
        for b in BUDGETS:
            kk = C.k_round(len(v), b)
            for init, sig in R.LIN_SET:
                k = json.dumps({"budget": b, "init": init, "sigma0": sig})
                x = oof(J, cell, "exact_lin", k)[-1]
                rows.append({"cell": cell, "budget": b, "learner": "exact_lin", "setting": json.dumps({"init": init, "sigma0": sig}), "epoch": -1,
                             "cv_gain": R.gain(x, v, kk), "cv_J": C.J(x, v, kk)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cv_values.csv", index=False)
    sel = []
    for (cell, b, l), g in t.groupby(["cell", "budget", "learner"], sort=False):
        crit = "cv_J" if l == "exact_lin" else "cv_gain"
        sel.append(dict(g.iloc[int(np.argmax(g[crit].to_numpy()))].to_dict(), criterion=crit))
    s9 = pd.DataFrame([r for r in sel if r["learner"] != "exact_lin"])
    for (cell, b), g in s9.groupby(["cell", "budget"], sort=False):            # R9's choice among its three learners, order cs, topk, pair
        g = g.set_index("learner").loc[[x for x in ("cs_gbm", "topk_mlp", "pair_mlp") if x in set(g.learner)]]
        r = g.iloc[int(np.argmax(g.cv_gain.to_numpy()))]
        sel.append(dict(r.to_dict(), learner="R9_chosen", chosen=r.name, criterion="cv_gain"))
    (OUT / "cv_selected.json").write_text(json.dumps(sel, indent=1, default=float))
    print(pd.DataFrame(sel)[["cell", "budget", "learner", "setting", "epoch"]].to_string(index=False), flush=True)


def selected():
    return {(r["cell"], r["budget"], r["learner"]): r for r in json.loads((OUT / "cv_selected.json").read_text())}


def fit_jobs():
    S = selected()
    out = {}
    for cell in CELLS:
        ev = {"test": TE}
        out[(cell, "gate_gbm")] = R.gbm_job(name(cell), FI, ev, tag=f"r15 fit {cell}")
        out[(cell, "ridge_V")] = R.ridge_job(name(cell), FI, ev, tag=f"r15 fit {cell}")
        for b in BUDGETS:
            for l in ("cs_gbm", "topk_mlp", "pair_mlp"):
                r = S[(cell, b, l)]
                out[(cell, l, b)] = R.J_({"data": name(cell), "fit": FI, "evals": ev, "variant": "V", "learner": l, "tag": f"r15 fit {cell} {b}",
                                          "params": {"setting": r["setting"], "epochs": None if int(r["epoch"]) < 0 else int(r["epoch"])}})
            st = json.loads(S[(cell, b, "exact_lin")]["setting"])
            out[(cell, "exact_lin", b)] = R.lin_job(name(cell), FI, ev, b, st["init"], st["sigma0"], tag=f"r15 fit {cell} {b}")
    out["crit"] = R.gbm_job(name("mono_brake") + "_crit", FI, {"test": TE}, tag="r15 fit crit")
    return out


def stage_q_fit(args):
    R.write_queues("r15_fit", list(fit_jobs().values()))


# ------------------------------------------------------------------------------------------------ test

def test_arrays(cell):
    z = R.data(name(cell))
    t = z["split"] == "test"
    return z["v"][t].astype(float), z["j"][t].astype(float), z["seq"][t].astype(str), z["log"][t].astype(str), z["frame"][t]


def scores(cell, b, F, S):
    out = {"gate_gbm": R.load_result(F[(cell, "gate_gbm")]["id"])["scores"][-1]["test"],
           "ridge_V": R.load_result(F[(cell, "ridge_V")]["id"])["scores"][-1]["test"],
           "exact_lin": R.load_result(F[(cell, "exact_lin", b)]["id"])["scores"][-1]["test"]}
    for l in ("cs_gbm", "topk_mlp", "pair_mlp"):
        out[l] = R.load_result(F[(cell, l, b)]["id"])["scores"][int(S[(cell, b, l)]["epoch"])]["test"]
    out["R9_chosen"] = out[S[(cell, b, "R9_chosen")]["chosen"]]
    return out


def rank_row(s, v, j, seq, log, q, rule="round"):
    """Realized and over matched-count random of one ranking, with scene and log intervals (241's form_rows, rank form)."""
    return R.form_rows("x", s, s, v, j, seq, log, np.zeros(len(v), int), q, rule)[0]


def stage_test(args):
    runmeta.new_run("r15_test", vars(args))
    assert json.loads((OUT / "gate_r15_0.json").read_text())["pass"]
    S, F = selected(), fit_jobs()
    pop = pd.read_csv(OUT / "populations.csv")
    tscenes = set(pop[pop.role == "T"].scene)
    rows = []
    for cell in CELLS:
        v, j, seq, log, frame = test_arrays(cell)
        for b in BUDGETS:
            sc = scores(cell, b, F, S)
            for l, s in sc.items():
                r = rank_row(s, v, j, seq, log, b)
                r.pop("scorer")
                st = S.get((cell, b, l), {})
                rows.append({"cell": cell, "budget": b, "allocator": l, "setting": st.get("setting", "{}"), "epoch": st.get("epoch", -1),
                             "chosen": st.get("chosen"), "T_scenes": len(set(seq)), "T_logs": len(set(log)), "T_frames": len(v), **r})
            print(f"  {cell} {b}: gate_gbm over random {rows[-len(sc)]['over_random_share']:+.4f}", flush=True)
    t = pd.DataFrame(rows)
    t["reading"] = np.where(t.over_random_lo > 0, "carries over", np.where(t.over_random_hi < 0, "fails", "not resolved"))
    t.to_csv(OUT / "budgets.csv", index=False)
    p = t[(t.cell == "mono_brake") & (t.budget == 0.2) & (t.allocator == "gate_gbm")]
    p.to_csv(OUT / "primary.csv", index=False)
    print(p.T.to_string(), flush=True)
    a = t[t.budget == 0.2].copy()
    sig = pd.read_csv(Path(RESULTS) / "final" / "significance_table.csv", float_precision="round_trip")
    o = sig[(sig.track == "nuScenes") & (sig.signal == "gate_gbm") & (sig.quota == 0.2) & (sig.system.isin(["brake", "plan_ade"]))]
    orient = [{"cell": f"{r.geometry}_{r.system}", "budget": 0.2, "allocator": "gate_gbm, frozen benchmark row (its own fit, published; orientation only)",
               "over_random_share": r.delta_share_all_cheap, "over_random_lo": r.delta_share_lo_all_draws, "over_random_hi": r.delta_share_hi_all_draws}
              for r in o.itertuples()]
    a = pd.concat([a, pd.DataFrame(orient)], ignore_index=True)
    off = pop[pop.role == "T"].official.value_counts().to_dict()
    a["T_official_train_scenes"] = off.get("train", 0)
    a.to_csv(OUT / "allocators.csv", index=False)
    # causal forms A, B at 20%: threshold from F's out-of-fold scores of the chosen setting
    CV = cv_jobs()
    rows = []
    for cell in CELLS:
        v, j, seq, log, frame = test_arrays(cell)
        sc = scores(cell, 0.2, F, S)
        for l in ("gate_gbm", "ridge_V", "exact_lin", "cs_gbm", "topk_mlp", "pair_mlp"):
            if l in ("gate_gbm", "ridge_V"):
                k, e = "{}", -1
            elif l == "exact_lin":
                st = json.loads(S[(cell, 0.2, l)]["setting"])
                k, e = json.dumps({"budget": 0.2, "init": st["init"], "sigma0": st["sigma0"]}), -1
            else:
                k, e = S[(cell, 0.2, l)]["setting"], int(S[(cell, 0.2, l)]["epoch"])
            o_ = oof(CV, cell, l, k)[e]
            rows += [dict(r, cell=cell, allocator=l) for r in R.form_rows(l, sc[l], o_, v, j, seq, log, frame, 0.2, "round")]
    pd.DataFrame(rows).to_csv(OUT / "causal.csv", index=False)
    # crit subset at the measured 20% latency budget
    v, j, seq, log, frame = test_arrays("mono_brake")
    s = R.load_result(F["crit"]["id"])["scores"][-1]["test"]
    r = rank_row(s, v, j, seq, log, CRIT_RATE, "floor")
    n = len(v)
    r["over_random20_floor_share"] = (r["gain"] - int(np.floor(0.2 * n + 1e-9)) / n * v.sum()) / float(j.sum())
    pd.DataFrame([dict(r, scorer="gate_gbm crit")]).to_csv(OUT / "crit.csv", index=False)
    # drop one log, primary row
    sc = scores("mono_brake", 0.2, F, S)["gate_gbm"]
    rows = []
    for lg in sorted(set(log)):
        m = log != lg
        k = C.k_round(int(m.sum()), 0.2)
        g = R.gain(sc[m], v[m], k)
        rows.append({"log_removed": lg, "scenes_removed": len(set(seq[~m])), "frames": int(m.sum()), "over_random_share": (g - k / m.sum() * v[m].sum()) / float(j[m].sum()),
                     "realized_share": g / float(j[m].sum())})
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "drop_one_log.csv", index=False)
    # power context: median half-width of the scene-level interval over the 20% rows
    a20 = t[t.budget == 0.2]
    hw = (a20.over_random_hi - a20.over_random_lo) / 2
    pd.DataFrame([{"rows": len(a20), "median_half_width_over_random_scene": float(hw.median()), "min": float(hw.min()), "max": float(hw.max()),
                   "primary_half_width": float((p.over_random_hi - p.over_random_lo).iloc[0] / 2), "T_scenes": len(tscenes)}]).to_csv(OUT / "power.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
