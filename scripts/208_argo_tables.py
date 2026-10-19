#!/usr/bin/env python
"""Follow-up E8a: Argoverse 1.1 decision tables, gate E8-0, setting-level quantities and allocators.

addendum E8 in docs/followup_protocol.md. The tables are 170's (`decision.build` on the shared action history and
`add_perception_gain`) with `rap.argo`'s adapter on 207's detections; the quantities are 171's and the allocators
172's (`rap.xl`), with five folds by log in place of a train / test split. Stages:

  gate         E8-0: recall of the 640 mode on reference boxes >= 30 px high, lift range error -> <out>/gate_e8_0.json
  tables       <work>/<geometry>__ns_cheap_320to<full>.pkl for the three pairs and both geometries
  settings     <out>/x1_settings.csv, x1_heterogeneity.csv (by city)
  allocators   <out>/x2_allocators.csv (log out-of-fold, 20%), x2_transfer.csv (fitted on nuScenes, secondary)
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import argo, decision, egospeed, frames, geometry as G, planner as P, runmeta, xl  # noqa: E402
from rap.cache import DetCache                                                             # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS, DATASETS                          # noqa: E402
from rap.risk import RiskConfig, match                                                     # noqa: E402
from rap.scoring import EPS                                                                # noqa: E402

DATASET = "Argoverse1"
AV1_ROOT = DATASETS / "argoverse1" / "argoverse-tracking" / "val"
DET = Path(CACHE) / "argo1_det"
WORK = Path(CACHE) / "xl" / "e8a"
OUT = Path(RESULTS) / "followup" / "E8"
CHEAP = "ns_cheap_320"
FULLS = ("ns_full_640", "ns_mid_480", "ns_full_960")
PRIMARY = "ns_full_640"
SIGNALS = [s for s in xl.SIGNALS if s != "R2_cnn_clf"]


def units(root):
    logs = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    return logs, [argo.unit_name(l) for l in logs]


def city_of():
    t = json.loads((Path(CACHE) / "cam_to_ego_argo.json").read_text())[DATASET]
    return {u: e["city"] for u, e in t.items()}


# ------------------------------------------------------------------------------------------------ gate E8-0

def stage_gate(args, ad, seqs):
    cfg = RiskConfig()
    hit = tot = 0
    rel = []
    for s in seqs:
        geom = ad.geometry(s)
        c = DetCache(DET / PRIMARY / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            g = geom[geom["frame"] == int(fr)]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= cfg.thr("full")
            if not len(g):
                continue
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
            best, dm = match(gt, np.array(["v"] * len(gt)), d["xyxy"][k].astype(float), d["conf"][k],
                             d["coarse"][k], cfg)
            big = (g["y2"] - g["y1"]) >= 30
            hit += int((best[big] >= cfg.iou_thr).sum()); tot += int(big.sum())
            if dm.any():
                from rap.mono import box_iou
                iou = box_iou(d["xyxy"][k][dm].astype(float), gt)
                zt = g["long_near"][iou.argmax(1)].astype(float)
                rel.append((geo["z"][k][dm] - zt) / np.maximum(zt, 1e-6))
    rel = np.concatenate(rel) if rel else np.zeros(0)
    r = {"recall_640_ge30px": hit / tot if tot else float("nan"), "n_ref_ge30px": tot,
         "lift_rel_error_median": float(np.median(rel)) if len(rel) else float("nan"),
         "lift_rel_error_p25_p75": [float(np.percentile(rel, 25)), float(np.percentile(rel, 75))] if len(rel) else [],
         "n_matched": int(len(rel))}
    r["pass"] = bool(r["recall_640_ge30px"] >= 0.5 and abs(r["lift_rel_error_median"]) <= 0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_e8_0.json").write_text(json.dumps(r, indent=1))
    print("gate E8-0:", r)
    return r["pass"]


def require_gate():
    f = OUT / "gate_e8_0.json"
    if not f.exists() or not json.loads(f.read_text())["pass"]:
        raise SystemExit("gate E8-0 has not passed")


# ------------------------------------------------------------------------------------------------ tables

def stage_tables(args, ad, seqs):
    require_gate()
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    for fm in FULLS:
        for geo in ("oracle", "mono"):
            t0 = time.time()
            d = decision.build(DET, CHEAP, fm, seqs, cfg, pp, cp, G.PRIMARY, range_source=geo, adapter=ad)
            d = decision.add_perception_gain(d, DET, CHEAP, fm, seqs, cfg, adapter=ad)
            f = work / f"{geo}__{CHEAP}to{fm}.pkl"
            d.to_pickle(f)
            print(f"  {f.name}: {len(d)} frames, {d.seq.nunique()} logs [{time.time() - t0:.0f}s]", flush=True)


# ------------------------------------------------------------------------------------------------ settings

def one(d, nboot):
    v = (d.J_cheap - d.J_full).to_numpy(float)
    c = d.J_cheap.to_numpy(float)
    r = xl.describe(v, c, (0.2,))
    r["budget_binding"] = bool((v > EPS).sum() > max(int(round(0.2 * len(v))), 1))
    r["n_units"] = int(d.seq.nunique())
    r.update(xl.bootstrap_describe(v, c, d.seq.to_numpy(), nboot, np.random.default_rng(0), (0.2,)))
    return r


def stage_settings(args, ad, seqs):
    require_gate()
    city = city_of()
    rows, het = [], []
    for fm in FULLS:
        for geo in ("mono", "oracle"):
            d = pd.read_pickle(Path(args.work) / f"{geo}__{CHEAP}to{fm}.pkl")
            r = one(d, args.nboot)
            rows.append({"dataset": DATASET, "pair": f"{CHEAP}to{fm}", "geometry": geo, "system": "brake",
                         "population": "av1_val24", "primary": fm == PRIMARY, **r})
            print(f"  {fm} {geo}: affected {r['affected']} harmed {r['harmed_share']:.3f} "
                  f"[{r['harmed_share_lo']:.3f}, {r['harmed_share_hi']:.3f}] rho {r['rho']:.3f} "
                  f"all-full {r['all_full_reduction']:+.3f} oracle@20 {r['oracle20_reduction']:+.3f}", flush=True)
            if fm == PRIMARY:
                for cty, g in d.groupby(d.seq.map(city)):
                    het.append({"pair": f"{CHEAP}to{fm}", "geometry": geo, "city": cty, **one(g, args.nboot)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "x1_settings.csv", index=False)
    pd.DataFrame(het).to_csv(OUT / "x1_heterogeneity.csv", index=False)
    print("wrote", OUT / "x1_settings.csv")


# ------------------------------------------------------------------------------------------------ allocators

def cell(work, geo, feats):
    d = pd.read_pickle(Path(work) / f"{geo}__{CHEAP}to{PRIMARY}.pkl")
    d = d.merge(feats, on=["seq", "frame"], how="inner", validate="one_to_one")
    d = egospeed.attach(d, DATASET)
    d["unit"] = d.seq
    return dict(track=DATASET, geometry=geo, system="brake", target="J", d=d, cheap="J_cheap", full="J_full")


def stage_allocators(args, ad, seqs):
    require_gate()
    feats, fcols = xl.features(DET)
    mat = xl.r1_matrix(DET)
    fold_of = {u: i % 5 for i, u in enumerate(sorted(seqs))}
    rows = []
    for geo in ("mono", "oracle"):
        c = cell(args.work, geo, feats)
        d = c["d"]
        v = (d.J_cheap - d.J_full).to_numpy(float)
        fold = d.seq.map(fold_of).to_numpy()
        X1 = xl.r1_rows(d, mat)
        pooled = {s: np.full(len(d), np.nan) for s in SIGNALS}
        for k in range(5):
            fit, held = fold != k, fold == k
            sc = {**xl.gate_scores(d, fcols, v, fit), **xl.t103().fit_score(X1, v, fit)}
            for s, x in sc.items():
                pooled[s][held] = x[held]
        pooled.update(xl.column_scores(d))
        p = xl.prepare(c, np.ones(len(d), bool), pooled, (0.2,))
        dr, kp = xl.paired_draws([p], args.nboot, np.random.default_rng(0), (0.2,))
        rows += xl.paired_rows([p], dr, kp, (0.2,), {"population": "av1_val24_log_oof", "bootstrap_unit": "log"})
    t = pd.DataFrame(rows)
    t["interval"] = np.where(t.delta_lo_all_draws > 0, "above random",
                             np.where(t.delta_hi_all_draws < 0, "below random", "includes 0"))
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "x2_allocators.csv", index=False)
    print(t[["geometry", "signal", "delta_share_all_cheap", "delta_share_lo_all_draws", "delta_share_hi_all_draws",
             "interval"]].to_string(index=False))
    transfer(args, feats, fcols, mat)


def transfer(args, feats, fcols, mat):
    """Secondary: gates and R1 routers fitted on E1's new nuScenes train + validation scenes, scored on AV1."""
    sp = xl.scene_split()
    split_of = {s: x for s, x in zip(sp.scene, sp.deep_split) if s in set(sp[sp.group == "new765"].scene)}
    xl.use_ego_table(Path(CACHE) / "xl" / "tables850" / "v_ego_causal.csv")
    nf, nfc = xl.features(Path(CACHE) / "nusc_det_xl")
    assert list(nfc) == list(fcols), "feature columns differ between the datasets"
    nmat = xl.r1_matrix(Path(CACHE) / "nusc_det_xl")
    rows = []
    for c_ns in xl.cells(Path(CACHE) / "xl" / "tables850", nf, nfc, split_of):
        if c_ns["system"] != "brake":
            continue
        dn = c_ns["d"][c_ns["d"].split.isin(["train", "val"])].reset_index(drop=True)
        c = cell(args.work, c_ns["geometry"], feats)
        da = c["d"]
        both = pd.concat([dn.assign(_ns=1), da.assign(_ns=0)], ignore_index=True)
        v = (both.J_cheap - both.J_full).to_numpy(float)
        fit = both._ns.to_numpy() == 1
        X1 = np.vstack([xl.r1_rows(dn, nmat), xl.r1_rows(da, mat)])
        sc = {**xl.gate_scores(both, fcols, v, fit), **xl.t103().fit_score(X1, v, fit)}
        sc = {s: x[~fit] for s, x in sc.items()}
        p = xl.prepare(c, np.ones(len(da), bool), sc, (0.2,))
        dr, kp = xl.paired_draws([p], args.nboot, np.random.default_rng(0), (0.2,))
        rows += xl.paired_rows([p], dr, kp, (0.2,), {"population": "av1_val24", "fit": "nuScenes new train + val",
                                                      "bootstrap_unit": "log"})
    pd.DataFrame(rows).to_csv(OUT / "x2_transfer.csv", index=False)
    print("wrote", OUT / "x2_transfer.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "tables", "settings", "allocators"], required=True)
    ap.add_argument("--root", default=str(AV1_ROOT))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"e8a_{args.stage}", vars(args))
    logs, seqs = units(args.root)
    ad = argo.make_adapter(args.root, logs)
    ok = {"gate": stage_gate, "tables": stage_tables, "settings": stage_settings,
          "allocators": stage_allocators}[args.stage](args, ad, seqs)
    (run / "summary.json").write_text(json.dumps({"stage": args.stage, "ok": ok is not False}, indent=1))
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
