#!/usr/bin/env python
"""Follow-up E1: allocators on the nuScenes track (P2 and the budget sweep of X3), and gate (a).

  --gate     the pipeline on the frozen 85 scenes against results/final (gate (a) of the protocol):
             - 170's tables against the joined tables the benchmark read (`62`'s joined_frames.pkl);
             - 92's cell descriptives and official nDG rows of the gates and heuristics, 103's R1 rows, 107's R2 rows
               (R2 from its released per-frame scores);
             - 125's every-draw paired intervals, with every random-number stream consumed as those stages consumed it.
             Writes results/followup/E1/gate_a.json; exits non-zero if anything differs.
  (default)  the follow-up population: allocators fitted on the new train + validation scenes, scored once on the 228
             new test scenes, every deployable signal, quotas 5-50%; intervals by scene and by log.

Protocol: docs/followup_protocol.md (Sections 3-5 and 7).
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, runmeta, xl                                              # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402
from rap.scoring import QUOTAS, evaluate                                         # noqa: E402

FINAL = Path(RESULTS) / "final"


def frozen_split():
    s = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["nuscenes"]
    return {u: k for k in ("train", "val", "test") for u in s[k]}


def rd(name):
    return pd.read_csv(FINAL / name, keep_default_na=False, na_values=[""], float_precision="round_trip",
                       low_memory=False)


def cmp(a, b, tol=0.0):
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= tol


# ------------------------------------------------------------------------------------------------ gate (a)

def gate(args):
    t92, t103 = xl.t92(), xl.t103()
    report, fails = {}, []
    split_of = frozen_split()
    tables = Path(args.tables)

    # 1. the tables
    tcmp = {}
    for geom in xl.GEOMETRIES:
        ref = pd.read_pickle(rap_runs.latest(f"phase0g_eta_fde_{geom}") / "joined_frames.pkl")
        new = pd.read_pickle(tables / f"{geom}__{xl.PAIR}.pkl")
        common = [c for c in new.columns if c in ref.columns]
        r = ref[common].sort_values(["seq", "frame"]).reset_index(drop=True)
        n = new[common].sort_values(["seq", "frame"]).reset_index(drop=True)
        diff = [c for c in common if not r[c].equals(n[c])]
        tcmp[geom] = {"frames_ref": len(ref), "frames_new": len(new), "columns_compared": len(common),
                      "columns_only_in_ref": [c for c in ref.columns if c not in new.columns],
                      "columns_differing": diff}
        if diff or len(ref) != len(new):
            fails.append(f"tables {geom}: {diff}")
    report["tables"] = tcmp

    feats, fcols = xl.features(CACHE / "nusc_det_tv")
    cells = list(xl.cells(tables, feats, fcols, split_of))
    assert [(c["geometry"], c["system"]) for c in cells] == [(g, s) for g in xl.GEOMETRIES for s, _, _ in xl.SYSTEMS]
    mat = xl.r1_matrix(CACHE / "nusc_det_tv")
    r2run = Path([str(p) for p in rap_runs.frame_runs("router_r2")][-1])

    bt = rd("benchmark_table.csv")
    bc = rd("benchmark_cells.csv")
    br = rd("benchmark_table_routers.csv")
    rng92, rng103, rng107 = (np.random.default_rng(0) for _ in range(3))
    n92 = n103 = n107 = nc = 0
    prepared, fitted = [], []
    for c in cells:
        d = c["d"]
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        fit = d.split.isin(["train", "val"]).to_numpy()
        test = (d.split == "test").to_numpy()
        gates = xl.gate_scores(d, c["fcols"], v_all, fit)
        cols = xl.column_scores(d)
        r1 = t103.fit_score(xl.r1_rows(d, mat), v_all, fit)
        r2 = xl.r2_scores(d, r2run / f"scores__nuScenes__{c['geometry']}__{c['system']}__{c['target']}.npz")
        key = (c["track"], c["geometry"], c["system"], c["target"])
        sel = lambda t: ((t.track == key[0]) & (t.geometry == key[1]) & (t.system == key[2]) & (t.target == key[3]))
        # 92: descriptives and evaluate, test then all; plan cells evaluate twice per split (self-agreement)
        for split in ("test", "all"):
            m = test if split == "test" else np.ones(len(d), bool)
            v, units = v_all[m], d.unit.to_numpy()[m]
            ref = bc[sel(bc) & (bc.split == split)].iloc[0]
            mine = t92.describe(v, d[c["cheap"]].to_numpy(float)[m])
            for k, x in mine.items():
                nc += 1
                if not cmp(x, ref[k]):
                    fails.append(f"describe {key} {split} {k}: {x} vs {ref[k]}")
            scores = {"random": None, "uncertainty": cols["uncertainty"][m],
                      "criticality_cheap": cols["criticality_cheap"][m]}
            if split == "test":
                scores.update({g: s[m] for g, s in gates.items()})
            ks, prize0, point, draws, dropped = evaluate(v, units, scores, 1000, rng92)
            if c["system"].startswith("plan_"):
                evaluate(v, units, {"PKL": None}, 1000, rng92)        # 92's self-agreement call: stream only
            if split != "test":
                continue
            for s in scores:
                for qi, q in enumerate(QUOTAS):
                    r = bt[sel(bt) & (bt.split == "test") & (bt.signal == s) & (bt.quota == q)].iloc[0]
                    lo, hi = t92.ci(draws[s][:, qi])
                    for k, x in (("eta", point[s]["eta"][qi]), ("eta_lo", lo), ("eta_hi", hi),
                                 ("boot_dropped", dropped[qi])):
                        n92 += 1
                        if not cmp(x, r[k]):
                            fails.append(f"92 {key} {s} {q} {k}: {x} vs {r[k]}")
        # 103: one evaluate per cell on test
        ks, prize0, point, draws, dropped = evaluate(v_all[test], d.unit.to_numpy()[test],
                                                     {"random": None, **{k: s[test] for k, s in r1.items()}}, 1000, rng103)
        for s in r1:
            for qi, q in enumerate(QUOTAS):
                r = br[sel(br) & (br.signal == s) & (br.quota == q)].iloc[0]
                lo, hi = t92.ci(draws[s][:, qi])
                for k, x in (("eta", point[s]["eta"][qi]), ("eta_lo", lo), ("eta_hi", hi)):
                    n103 += 1
                    if not cmp(x, r[k]):
                        fails.append(f"103 {key} {s} {q} {k}: {x} vs {r[k]}")
        # 107: one evaluate per head on test, its own stream
        if r2:
            ks, prize0, point, draws, dropped = evaluate(v_all[test], d.unit.to_numpy()[test],
                                                         {"random": None, "R2_cnn_clf": r2["R2_cnn_clf"][test]}, 1000, rng107)
            for qi, q in enumerate(QUOTAS):
                r = br[sel(br) & (br.signal == "R2_cnn_clf") & (br.quota == q)].iloc[0]
                lo, hi = t92.ci(draws["R2_cnn_clf"][:, qi])
                for k, x in (("eta", point["R2_cnn_clf"]["eta"][qi]), ("eta_lo", lo), ("eta_hi", hi)):
                    n107 += 1
                    if not cmp(x, r[k]):
                        fails.append(f"107 {key} R2 {q} {k}: {x} vs {r[k]}")
        prepared.append(xl.prepare(c, test, {**gates, **r1, **r2, **cols}, QUOTAS))
    # 125: every-draw paired intervals; KITTI (6 units) and nuPlan (9 logs) advance the stream after nuScenes
    draws, keep = xl.paired_draws(prepared, 1000, np.random.default_rng(0), QUOTAS, extra=(6, 9))
    rows = pd.DataFrame(xl.paired_rows(prepared, draws, keep, QUOTAS))
    sig = rd("significance_table.csv")
    sig = sig[sig.track == "nuScenes"]
    n125 = 0
    for r in sig.itertuples():
        m = rows[(rows.geometry == r.geometry) & (rows.system == r.system) & (rows.target == r.target)
                 & (rows.signal == r.signal) & (rows.quota == r.quota)]
        if m.empty:
            fails.append(f"125 missing {r.geometry} {r.system} {r.signal} {r.quota}")
            continue
        m = m.iloc[0]
        for k in ("gain", "gain_random", "delta_vs_random", "delta_lo_all_draws", "delta_hi_all_draws",
                  "delta_share_all_cheap", "delta_share_lo_all_draws", "delta_share_hi_all_draws", "ndg",
                  "delta_lo_filtered", "delta_hi_filtered"):
            n125 += 1
            if not cmp(m[k], getattr(r, k)):
                fails.append(f"125 {r.geometry} {r.system} {r.signal} {r.quota} {k}: {m[k]} vs {getattr(r, k)}")
        n125 += 1
        if bool(m["beats_random_all_draws"]) != bool(r.beats_random_all_draws):
            fails.append(f"125 {r.geometry} {r.system} {r.signal} {r.quota} beats_random")
    report["comparisons"] = {"describe": nc, "92_official": n92, "103_R1": n103, "107_R2": n107,
                             "125_significance": n125}
    report["differing"] = fails[:200]
    report["n_differing"] = len(fails)
    report["result"] = "pass" if not fails else "fail"
    report["r2_run"] = r2run.name
    (xl.OUT / "gate_a.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: report[k] for k in ("comparisons", "n_differing", "result")}, indent=1))
    for f in fails[:20]:
        print("  ", f)
    return 0 if not fails else 1


# ------------------------------------------------------------------------------------------------ follow-up population

TRIPLE = ["gate_gbm", "R1_mlp_reg", "R1_gbm_reg"]          # the claims under test (P2), braking, monocular geometry
OOF = ["gate_gbm", "R1_gbm_reg", "gate_ridge"]


def xl_split(sp):
    new = sp[sp.group == "new765"]
    return dict(zip(new.scene, new.deep_split))


def signals_for(c, mat, r2dir, fit):
    d = c["d"]
    v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    out = {**xl.gate_scores(d, c["fcols"], v_all, fit), **xl.t103().fit_score(xl.r1_rows(d, mat), v_all, fit)}
    if r2dir is not None:
        out.update(xl.r2_scores(d, Path(r2dir) / f"scores__nuScenes__{c['geometry']}__{c['system']}__{c['target']}.npz"))
    out.update(xl.column_scores(d))
    return v_all, out


def official_rows(cells_, scored, quotas, rng):
    """nDG with the official 25% prize filter (rap.scoring.evaluate generalised to the sweep), one stream."""
    rows = []
    for c, (v_all, sc) in zip(cells_, scored):
        d = c["d"]
        test = (d.split == "test").to_numpy()
        ks, prize0, point, draws, dropped = xl.evaluate_q(v_all[test], d.unit.to_numpy()[test],
                                                          {k: s[test] for k, s in sc.items()}, 1000, rng, quotas)
        for s in sc:
            for qi, q in enumerate(quotas):
                lo, hi = xl.ci(draws[s][:, qi])
                rows.append({"geometry": c["geometry"], "system": c["system"], "target": c["target"], "signal": s,
                             "quota": q, "ndg_official": float(point[s]["eta"][qi]), "ndg_lo_filtered": lo,
                             "ndg_hi_filtered": hi, "ndg_draws_dropped_by_filter": int(dropped[qi])})
    return pd.DataFrame(rows)


def interval_label(lo, hi):
    return np.where(lo > 0, "above random", np.where(hi < 0, "below random", np.where(pd.notna(lo), "includes 0", "n/a")))


def xl_run(args, run):
    sp = xl.scene_split()
    meta = xl.scene_meta(args.dataroot)
    log_of = dict(zip(meta.scene, meta.log))
    split_of = xl_split(sp)
    xl.use_ego_table(Path(args.tables) / "v_ego_causal.csv")
    feats, fcols = xl.features(args.det)
    mat = xl.r1_matrix(args.det)
    cells_ = list(xl.cells(args.tables, feats, fcols, split_of))
    quotas = xl.SWEEP
    scored, prep_scene, prep_log = [], [], []
    for c in cells_:
        d = c["d"]
        t0 = time.time()
        fit = d.split.isin(["train", "val"]).to_numpy()
        test = (d.split == "test").to_numpy()
        v_all, sc = signals_for(c, mat, args.r2, fit)
        scored.append((v_all, sc))
        prep_scene.append(xl.prepare(c, test, sc, quotas))
        cl = dict(c, d=d.assign(unit=d.seq.map(log_of)))
        prep_log.append(xl.prepare(cl, test, sc, quotas))
        print(f"  {c['geometry']:6s} {c['system']:8s} fitted {len(sc)} signals, test {int(test.sum())} frames "
              f"[{time.time() - t0:.0f}s]", flush=True)
    off = official_rows(cells_, scored, quotas, np.random.default_rng(0))
    rows = []
    for unit, prep in (("scene", prep_scene), ("log", prep_log)):
        draws, keep = xl.paired_draws(prep, 1000, np.random.default_rng(0), quotas)
        r = pd.DataFrame(xl.paired_rows(prep, draws, keep, quotas, {"bootstrap_unit": unit}))
        # one-sided bootstrap p of "not above random": the share of draws at or below zero
        p = []
        for ci_, c in enumerate(prep):
            for s in c["series"]:
                for qi, q in enumerate(quotas):
                    x = draws[(ci_, s)][:, qi]
                    p.append(float(np.mean(x[np.isfinite(x)] <= 0)))
        r["p_le_random"] = p
        rows.append(r)
    t = pd.concat(rows, ignore_index=True)
    t = t.merge(off, on=["geometry", "system", "target", "signal", "quota"], how="left", validate="many_to_one")
    t["interval"] = interval_label(t.delta_lo_all_draws.astype(float), t.delta_hi_all_draws.astype(float))
    t["population"] = "new228_test"
    t["train_scenes"] = int(sp[(sp.group == "new765") & sp.deep_split.isin(["train", "val"])].shape[0])
    t["fit_units"] = "new train + validation"
    # Holm over the three claims (scene bootstrap, 20%)
    t["claim_under_test"] = (t.geometry == "mono") & (t.system == "brake") & t.signal.isin(TRIPLE) & (t.quota == 0.2)
    t["holm_above_random"] = np.nan
    for unit in ("scene", "log"):
        m = t.claim_under_test & (t.bootstrap_unit == unit)
        t.loc[m, "holm_above_random"] = xl.holm(t.loc[m, "p_le_random"].to_numpy())
    t.to_csv(xl.OUT / "x3_budget_sweep.csv", index=False)
    t[t.quota == 0.2].to_csv(xl.OUT / "x2_allocators.csv", index=False)
    t.to_csv(run / "x3_budget_sweep.csv", index=False)
    print(t[(t.quota == 0.2) & (t.bootstrap_unit == "scene")].pivot_table(
        index="signal", columns=["geometry", "system"], values="delta_share_all_cheap").round(3).to_string())

    # planner exposure: planner cells on the official-validation test scenes alone (secondary)
    offval = set(sp[sp.official == "val"].scene)
    pe = []
    for c, (v_all, sc) in zip(cells_, scored):
        if not c["system"].startswith("plan_"):
            continue
        d = c["d"]
        test = ((d.split == "test") & d.seq.isin(offval)).to_numpy()
        if not test.any():
            continue
        p = xl.prepare(c, test, sc, (0.2,))
        dr, kp = xl.paired_draws([p], 1000, np.random.default_rng(0), (0.2,))
        r = pd.DataFrame(xl.paired_rows([p], dr, kp, (0.2,), {"population": "new228_test_official_val",
                                                               "bootstrap_unit": "scene"}))
        pe.append(r)
    if pe:
        pd.concat(pe).to_csv(xl.OUT / "x2_planner_official_val.csv", index=False)

    # log-grouped out-of-fold check (secondary): five folds by log over the 765 new scenes, braking, monocular
    oof_rows = []
    c = next(c for c in xl.cells(args.tables, feats, fcols, split_of, geometries=("mono",)) if c["system"] == "brake")
    d = c["d"]
    v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    logs = sorted(set(d.seq.map(log_of)))
    fold_of_log = {l: i % 5 for i, l in enumerate(logs)}
    fold = d.seq.map(log_of).map(fold_of_log).to_numpy()
    X1 = xl.r1_rows(d, mat)
    pooled = {s: np.full(len(d), np.nan) for s in OOF}
    for k in range(5):
        fit, held = fold != k, fold == k
        g = xl.gate_scores(d, c["fcols"], v_all, fit, which=("gate_gbm", "gate_ridge"))
        r1 = xl.t103().fit_score(X1, v_all, fit)
        for s in OOF:
            pooled[s][held] = (g if s.startswith("gate") else r1)[s][held]
    allm = np.ones(len(d), bool)
    cl = dict(c, d=d.assign(unit=d.seq.map(log_of)))
    p = xl.prepare(cl, allm, pooled, (0.2,))
    dr, kp = xl.paired_draws([p], 1000, np.random.default_rng(0), (0.2,))
    oof = pd.DataFrame(xl.paired_rows([p], dr, kp, (0.2,), {"population": "new765_log_oof", "bootstrap_unit": "log"}))
    oof.to_csv(xl.OUT / "x2_log_oof.csv", index=False)
    print(oof[["signal", "delta_share_all_cheap", "delta_share_lo_all_draws", "delta_share_hi_all_draws", "ndg"]]
          .to_string(index=False))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--tables", required=True, help="170's output folder")
    ap.add_argument("--det", default=str(CACHE / "nusc_det_xl"), help="detection cache of the follow-up scenes")
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--r2", default=None, help="a 107 run on the follow-up scenes (R2 scores); omitted if absent")
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_allocators" + ("_gate" if args.gate else ""), vars(args))
    if args.gate:
        rc = gate(args)
        (run / "gate_a.json").write_text((xl.OUT / "gate_a.json").read_text())
        raise SystemExit(rc)
    raise SystemExit(xl_run(args, run))


if __name__ == "__main__":
    main()
