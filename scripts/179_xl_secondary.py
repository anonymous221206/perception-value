#!/usr/bin/env python
"""Follow-up E1, secondary analyses on cached values (protocol Section 7):

  sign       sign disagreement between the four perception gains of the paper's sign-agreement table and decision
             value, both geometries, braking and the planner (ADE)        -> x1_sign_disagreement.csv
             (gains as 102 defines them: weighted error counts, CHEAP minus FULL)
  valsel     the validation-selected comparison (158's part A3 with the target-free pool): each objective picks its
             winner on the 152 validation scenes, the winners are compared once on test by realized decision gain
                                                                        -> x2_validation_selected.csv
  mechanism  braking: added false positive the controller considers and recovered miss it needs, on harmed and on
             helped frames (118's per-object rebuild)                   -> x1_mechanism.csv

With --frozen the same code runs on the frozen 85 scenes and prints its values beside the released ones.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, percep_metrics as PM, planner as P, runmeta, xl          # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                   # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL, NUSCENES_XL, RESULTS               # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402
from rap.scoring import EPS, QUOTAS                                             # noqa: E402

GAINS = {"exact_FN": {"fn": 1.0}, "FN_FP": PM.METRICS["E2_fn_fp"],
         "E5_combined": PM.METRICS["E5_combined"], "E_risk": PM.METRICS["E6_risk_weighted"]}
FINAL = Path(RESULTS) / "final"


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def gain(d, w):
    return (sum(x * d[f"cheap_{k}"] for k, x in w.items()) - sum(x * d[f"full_{k}"] for k, x in w.items())).to_numpy(float)


def part_sign(tables, pops):
    rows = []
    for geom in xl.GEOMETRIES:
        t = pd.read_pickle(Path(tables) / f"{geom}__{xl.PAIR}.pkl")
        for system, cc, cf in (("brake", "J_cheap", "J_full"), ("plan_ade", "JC_ade_cheap", "JC_ade_full")):
            for pop, scenes in pops.items():
                d = t[t.seq.isin(scenes) & t[cc].notna()]
                v = (d[cc] - d[cf]).to_numpy(float)
                r = {"population": pop, "geometry": geom, "system": system, "n_frames": len(d),
                     "n_affected": int((np.abs(v) > EPS).sum())}
                for g, w in GAINS.items():
                    x = gain(d, w)
                    both = (np.abs(x) > EPS) & (np.abs(v) > EPS)
                    aff = np.abs(v) > EPS
                    gp = (x > EPS) & aff
                    r[f"{g}_sign_disagreement"] = float(np.mean(np.sign(x[both]) != np.sign(v[both]))) if both.any() else np.nan
                    r[f"{g}_n_both_nonzero"] = int(both.sum())
                    r[f"{g}_spearman_affected"] = float(stats.spearmanr(x[aff], v[aff]).correlation) if aff.sum() > 2 else np.nan
                    r[f"{g}_harmed_given_gain_pos_affected"] = float(np.mean(v[gp] < -EPS)) if gp.any() else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def part_valsel(tables, split_of, det):
    t158 = _load("t158", "158_cached_analyses.py")
    xl.use_ego_table(Path(tables) / "v_ego_causal.csv") if (Path(tables) / "v_ego_causal.csv").exists() else None
    feats, fcols = xl.features(det)
    rows, per = [], {}
    rng_cells = []
    for c in xl.cells(tables, feats, fcols, split_of):
        d = c["d"]
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        va, te = (d.split == "val").to_numpy(), (d.split == "test").to_numpy()
        sc = {"random": None, **xl.column_scores(d)}
        for o, col in t158.OBJECTIVES.items():
            pv = d[col].to_numpy(float)
            for qi, q in enumerate(QUOTAS):
                k_va, k_te = QUOTAS_K(va.sum(), qi), QUOTAS_K(te.sum(), qi)
                ok_dec, _, _ = t158.defined(v_all[va], k_va)
                ok_p, _, _ = t158.defined(pv[va], k_va)
                ok_te, _, _ = t158.defined(v_all[te], k_te)
                base = {"geometry": c["geometry"], "system": c["system"], "target": c["target"], "objective": o,
                        "pool": "P1", "quota": q}
                if not (ok_dec[0] and ok_p[0] and ok_te[0]):
                    rows.append({**base, "included": False}); continue
                pick = lambda s, m: None if sc[s] is None else sc[s][m]
                val_dec = {s: t158.gains(pick(s, va), [v_all[va]], k_va)[0][0] for s in sc}
                val_per = {s: t158.gains(pick(s, va), [pv[va]], k_va)[0][0] for s in sc}
                w_dec = next(s for s in sc if val_dec[s] >= max(val_dec.values()) - t158.TOL)
                w_per = next(s for s in sc if val_per[s] >= max(val_per.values()) - t158.TOL)
                g_dec = t158.gains(pick(w_dec, te), [v_all[te]], k_te)[0][0]
                g_per = t158.gains(pick(w_per, te), [v_all[te]], k_te)[0][0]
                tot = float(d[c["cheap"]].to_numpy(float)[te].sum())
                rows.append({**base, "included": True, "selected_dec": w_dec, "selected_perc": w_per,
                             "same_winner": w_dec == w_per, "gain_test_dec_selected": g_dec,
                             "gain_test_perc_selected": g_per, "diff_share": (g_per - g_dec) / tot,
                             "all_cheap_loss": tot})
                rng_cells.append((len(rows) - 1, d, te, v_all, sc, w_dec, w_per, qi, c["cheap"]))
    # every-draw scene bootstrap of each difference, one stream, units resampled once per draw for all cells
    units = sorted(set().union(*[set(d.seq[te]) for _, d, te, *_ in rng_cells])) if rng_cells else []
    rng = np.random.default_rng(0)
    draws = np.full((1000, len(rng_cells)), np.nan)
    pre = []
    for (_, d, te, v_all, sc, w_dec, w_per, qi, cheap) in rng_cells:
        seq = d.seq.to_numpy()[te]
        pre.append(({u: np.flatnonzero(seq == u) for u in set(seq)}, v_all[te], d[cheap].to_numpy(float)[te],
                    None if sc[w_dec] is None else sc[w_dec][te], None if sc[w_per] is None else sc[w_per][te],
                    w_dec == w_per, qi))
    for b in range(1000):
        pick = [units[i] for i in rng.integers(0, len(units), len(units))]
        for j, (idx, vt, jc, s_dec, s_per, same, qi) in enumerate(pre):
            t = np.concatenate([idx[u] for u in pick if u in idx]) if any(u in idx for u in pick) else np.zeros(0, int)
            if not len(t):
                continue
            if same:
                draws[b, j] = 0.0; continue
            k = QUOTAS_K(len(t), qi)
            g = lambda s: t158.gains(None if s is None else s[t], [vt[t]], k)[0][0]
            draws[b, j] = (g(s_per) - g(s_dec)) / jc[t].sum()
    out = pd.DataFrame(rows)
    for j, (ri, *_rest) in enumerate(rng_cells):
        lo, hi = xl.ci(draws[:, j])
        out.loc[ri, "diff_share_lo"], out.loc[ri, "diff_share_hi"] = lo, hi
    pooled = []
    for o in t158.OBJECTIVES:
        for qi, q in enumerate(QUOTAS):
            js = [j for j, (ri, *_r) in enumerate(rng_cells) if out.loc[ri, "objective"] == o and out.loc[ri, "quota"] == q]
            if not js:
                continue
            lo, hi = xl.ci(np.nanmean(draws[:, js], axis=1))
            pooled.append({"geometry": "pooled", "objective": o, "pool": "P1", "quota": q, "n_cells": len(js),
                           "diff_share": float(np.mean([out.loc[rng_cells[j][0], "diff_share"] for j in js])),
                           "diff_share_lo": lo, "diff_share_hi": hi, "included": True})
    return pd.concat([out, pd.DataFrame(pooled)], ignore_index=True)


def QUOTAS_K(n, qi):
    return np.array([max(int(round(QUOTAS[qi] * n)), 1)])


def part_mechanism(tables, det, dataroot, pops):
    t118 = _load("t118", "118_figure_exports.py")
    t118.DET = Path(det)
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    db = NuScenesDB(dataroot, "v1.0-trainval")
    adapter = make_adapter(db)
    rows = []
    for pop, scenes in pops.items():
        seqs = sorted(scenes)
        for s in seqs:
            adapter.geometry(s)
        for geom in xl.GEOMETRIES:
            jt = pd.read_pickle(Path(tables) / f"{geom}__{xl.PAIR}.pkl").set_index(["seq", "frame"])
            fr = t118.rebuild(db, adapter, seqs, cfg, pp, cp, geom)
            bad = sum(abs(fr[k]["J_full"] - jt.loc[k, "J_full"]) > EPS or abs(fr[k]["J_cheap"] - jt.loc[k, "J_cheap"]) > EPS
                      for k in fr)
            assert bad == 0, f"rebuild differs from the table on {bad} frames"
            V = jt.loc[list(fr), "J_cheap"] - jt.loc[list(fr), "J_full"]
            keep = V[V.abs() > EPS]
            stat = {-1: {"n": 0, "fp": 0, "miss": 0}, 1: {"n": 0, "fp": 0, "miss": 0}}
            for (s, f_), v in keep.items():
                r, _ = t118.object_rows(fr[(s, f_)], pp, cfg, dict(scene=s, frame=int(f_), sample_token="", system="brake",
                                                                    geometry=geom, V=float(v), sign_V=int(np.sign(v))))
                sg = int(np.sign(v))
                stat[sg]["n"] += 1
                stat[sg]["fp"] += any(x["type"] == "fp_added_by_full" and x["considered_by_controller"] for x in r)
                stat[sg]["miss"] += any(x["type"] == "miss_recovered_by_full" and x["considered_by_controller"] for x in r)
            for sg, name in ((-1, "harmed"), (1, "helped")):
                n = stat[sg]["n"]
                rows.append({"population": pop, "geometry": geom, "system": "brake", "frames": name, "n_frames": n,
                             "share_fp_added_considered": stat[sg]["fp"] / n if n else np.nan,
                             "share_miss_recovered_considered": stat[sg]["miss"] / n if n else np.nan})
            print(rows[-2], rows[-1], flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--det", default=str(CACHE / "nusc_det_xl"))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--parts", nargs="+", default=["sign", "valsel", "mechanism"])
    ap.add_argument("--frozen", action="store_true", help="run on the frozen 85 scenes (a check against results/final)")
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_secondary" + ("_frozen" if args.frozen else ""), vars(args))
    sp = xl.scene_split()
    if args.frozen:
        split = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["nuscenes"]
        split_of = {u: k for k in ("train", "val", "test") for u in split[k]}
        pops = {"frozen85": set(split_of)}
        args.det = str(CACHE / "nusc_det_tv")
        args.dataroot = str(NUSCENES_TRAINVAL)
    else:
        new = sp[sp.group == "new765"]
        split_of = dict(zip(new.scene, new.deep_split))
        pops = {"new228_test": set(new[new.deep_split == "test"].scene), "new765": set(new.scene),
                "frozen85": set(sp[sp.group == "frozen85"].scene)}
    dest = run if args.frozen else xl.OUT
    if "sign" in args.parts:
        s = part_sign(args.tables, pops)
        s.to_csv(dest / "x1_sign_disagreement.csv", index=False)
        print(s.to_string())
    if "valsel" in args.parts:
        v = part_valsel(args.tables, split_of, args.det)
        v.to_csv(dest / "x2_validation_selected.csv", index=False)
        print(v.to_string())
    if "mechanism" in args.parts:
        m = part_mechanism(args.tables, args.det, args.dataroot,
                           pops if args.frozen else {"new228_test": pops["new228_test"]})
        m.to_csv(dest / "x1_mechanism.csv", index=False)
        print(m.to_string())


if __name__ == "__main__":
    main()
