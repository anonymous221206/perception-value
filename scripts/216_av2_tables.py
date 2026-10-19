#!/usr/bin/env python
"""Follow-up E8b: Argoverse 2 gate E8-0 (on both image versions), decision tables, setting-level quantities, allocators.

addendum E8 in docs/followup_protocol.md (E8b rules). 208's stages on 215's detections; the allocators are fitted on the
140 fitting logs and scored once on the 150 validation logs. Stages:

  gate         E8-0 on the raw and the undistorted images; the version with the higher 640 recall is used
               -> results/followup/E8/av2/gate_e8_0.json
  tables       <work>/<geometry>__ns_cheap_320to<full>.pkl
  settings     av2/x1_settings.csv (validation logs; fitting logs secondary)
  allocators   av2/x2_allocators.csv, av2/x2_transfer.csv (fitted on nuScenes -> AV2 val; AV2 fitting logs -> AV1)
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import argo, argo2, decision, egospeed, frames, geometry as G, planner as P, runmeta, xl  # noqa: E402
from rap.cache import DetCache                                                                    # noqa: E402
from rap.mono import box_iou                                                                      # noqa: E402
from rap.paths import CACHE, RESULTS, DATASETS                                              # noqa: E402
from rap.risk import RiskConfig, match                                                            # noqa: E402
from rap.scoring import EPS                                                                       # noqa: E402

DATASET = "Argoverse2"
TABLES = DATASETS / "av2_deep" / "av2_tables"
IMAGES = DATASETS / "av2_deep" / "sensor"
WORK = Path(CACHE) / "xl" / "e8b"
OUT = Path(RESULTS) / "followup" / "E8" / "av2"
CHEAP, PRIMARY = "ns_cheap_320", "ns_full_640"
FULLS = ("ns_full_640", "ns_mid_480", "ns_full_960")
SIGNALS = [s for s in xl.SIGNALS if s != "R2_cnn_clf"]


def t208():
    s = importlib.util.spec_from_file_location("t208", ROOT / "scripts" / "208_argo_tables.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def setup():
    sl = argo2.logs(TABLES)
    split = {argo.unit_name(l): sp for sp, l in sl}
    return sl, split, argo2.make_adapter(TABLES, IMAGES, sl)


def det_dir():
    g = json.loads((OUT / "gate_e8_0.json").read_text())
    if not g["pass"]:
        raise SystemExit("gate E8-0 has not passed")
    return Path(CACHE) / f"argo2_det_{g['version']}"


def recall_and_lift(det, ad, seqs):
    cfg = RiskConfig()
    hit = tot = 0
    rel = []
    for s in seqs:
        geom = ad.geometry(s)
        c = DetCache(det / PRIMARY / f"{s}.npz")
        for i, fr in enumerate(c.frames):
            g = geom[geom["frame"] == int(fr)]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            if not len(g):
                continue
            d, geo = c.det(i), c.geo(i)
            k = d["conf"] >= cfg.thr("full")
            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
            best, dm = match(gt, np.array(["v"] * len(gt)), d["xyxy"][k].astype(float), d["conf"][k],
                             d["coarse"][k], cfg)
            big = (g["y2"] - g["y1"]) >= 30
            hit += int((best[big] >= cfg.iou_thr).sum()); tot += int(big.sum())
            if dm.any():
                iou = box_iou(d["xyxy"][k][dm].astype(float), gt)
                zt = g["long_near"][iou.argmax(1)].astype(float)
                rel.append((geo["z"][k][dm] - zt) / np.maximum(zt, 1e-6))
    rel = np.concatenate(rel) if rel else np.zeros(1)
    return {"recall_640_ge30px": hit / max(tot, 1), "n_ref_ge30px": tot,
            "lift_rel_error_median": float(np.median(rel)), "n_matched": int(len(rel))}


def stage_gate(args, sl, split, ad):
    val = [u for u, sp in split.items() if sp == "val"]
    res = {v: recall_and_lift(Path(CACHE) / f"argo2_det_{v}", ad, val) for v in ("raw", "undist")}
    best = max(res, key=lambda v: res[v]["recall_640_ge30px"])
    r = res[best]
    out = {"versions": res, "version": best,
           "pass": bool(r["recall_640_ge30px"] >= 0.5 and abs(r["lift_rel_error_median"]) <= 0.2)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_e8_0.json").write_text(json.dumps(out, indent=1))
    print("gate E8-0 (AV2):", out)
    return out["pass"]


def stage_tables(args, sl, split, ad):
    det = det_dir()
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    seqs = sorted(split)
    for fm in FULLS:
        for geo in ("oracle", "mono"):
            t0 = time.time()
            d = decision.build(det, CHEAP, fm, seqs, cfg, pp, cp, G.PRIMARY, range_source=geo, adapter=ad)
            d = decision.add_perception_gain(d, det, CHEAP, fm, seqs, cfg, adapter=ad)
            d["split"] = d.seq.map(split)
            d.to_pickle(work / f"{geo}__{CHEAP}to{fm}.pkl")
            print(f"  {geo} {fm}: {len(d)} frames [{time.time() - t0:.0f}s]", flush=True)


def stage_settings(args, sl, split, ad):
    det_dir()
    one = t208().one
    rows = []
    for fm in FULLS:
        for geo in ("mono", "oracle"):
            d = pd.read_pickle(Path(args.work) / f"{geo}__{CHEAP}to{fm}.pkl")
            for pop, sp in (("av2_val150", "val"), ("av2_fit140", "train")):
                r = one(d[d.split == sp], args.nboot)
                rows.append({"dataset": DATASET, "pair": f"{CHEAP}to{fm}", "geometry": geo, "system": "brake",
                             "population": pop, "primary": fm == PRIMARY and sp == "val", **r})
                print(f"  {fm} {geo} {pop}: harmed {r['harmed_share']:.3f} rho {r['rho']:.3f} "
                      f"all-full {r['all_full_reduction']:+.3f}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "x1_settings.csv", index=False)


def cell(work, geo, feats):
    d = pd.read_pickle(Path(work) / f"{geo}__{CHEAP}to{PRIMARY}.pkl")
    d = d.merge(feats, on=["seq", "frame"], how="inner", validate="one_to_one")
    d = egospeed.attach(d, DATASET)
    d["unit"] = d.seq
    return dict(track=DATASET, geometry=geo, system="brake", target="J", d=d, cheap="J_cheap", full="J_full")


def stage_allocators(args, sl, split, ad):
    det = det_dir()
    feats, fcols = xl.features(det)
    mat = xl.r1_matrix(det)
    rows = []
    for geo in ("mono", "oracle"):
        c = cell(args.work, geo, feats)
        d = c["d"]
        v = (d.J_cheap - d.J_full).to_numpy(float)
        fit, test = (d.split == "train").to_numpy(), (d.split == "val").to_numpy()
        sc = {**xl.gate_scores(d, fcols, v, fit), **xl.t103().fit_score(xl.r1_rows(d, mat), v, fit)}
        sc.update(xl.column_scores(d))
        p = xl.prepare(c, test, sc, (0.2,))
        dr, kp = xl.paired_draws([p], args.nboot, np.random.default_rng(0), (0.2,))
        rows += xl.paired_rows([p], dr, kp, (0.2,), {"population": "av2_val150", "fit": "av2_fit140",
                                                      "bootstrap_unit": "log"})
    t = pd.DataFrame(rows)
    t["interval"] = np.where(t.delta_lo_all_draws > 0, "above random",
                             np.where(t.delta_hi_all_draws < 0, "below random", "includes 0"))
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "x2_allocators.csv", index=False)
    print(t[["geometry", "signal", "delta_share_all_cheap", "delta_share_lo_all_draws", "delta_share_hi_all_draws",
             "interval"]].to_string(index=False))
    transfer(args, det, feats, fcols, mat)


def transfer(args, det, feats, fcols, mat):
    """Secondary: nuScenes (E1's new train + validation) -> AV2 validation; AV2 fitting logs -> AV1."""
    sp = xl.scene_split()
    split_of = {s: x for s, x in zip(sp.scene, sp.deep_split) if s in set(sp[sp.group == "new765"].scene)}
    xl.use_ego_table(Path(CACHE) / "xl" / "tables850" / "v_ego_causal.csv")
    nf, nfc = xl.features(Path(CACHE) / "nusc_det_xl")
    assert list(nfc) == list(fcols)
    nmat = xl.r1_matrix(Path(CACHE) / "nusc_det_xl")
    m208 = t208()
    af, afc = xl.features(m208.DET)
    amat = xl.r1_matrix(m208.DET)
    rows = []
    for c_ns in xl.cells(Path(CACHE) / "xl" / "tables850", nf, nfc, split_of):
        if c_ns["system"] != "brake":
            continue
        geo = c_ns["geometry"]
        c2 = cell(args.work, geo, feats)
        d2 = c2["d"]
        pairs = (("nuScenes new train + val", c_ns["d"][c_ns["d"].split.isin(["train", "val"])], nmat,
                  dict(c2, d=d2[d2.split == "val"].reset_index(drop=True)), mat, "av2_val150"),
                 ("av2_fit140", d2[d2.split == "train"], mat, m208.cell(m208.WORK, geo, af), amat, "av1_val24"))
        for fitname, dfit, mfit, ctest, mtest, pop in pairs:
            dfit = dfit.reset_index(drop=True)
            dt = ctest["d"]
            both = pd.concat([dfit.assign(_f=1), dt.assign(_f=0)], ignore_index=True)
            v = (both.J_cheap - both.J_full).to_numpy(float)
            fit = both._f.to_numpy() == 1
            X1 = np.vstack([xl.r1_rows(dfit, mfit), xl.r1_rows(dt, mtest)])
            sc = {**xl.gate_scores(both, fcols, v, fit), **xl.t103().fit_score(X1, v, fit)}
            sc = {s: x[~fit] for s, x in sc.items()}
            p = xl.prepare(ctest, np.ones(len(dt), bool), sc, (0.2,))
            dr, kp = xl.paired_draws([p], args.nboot, np.random.default_rng(0), (0.2,))
            rows += xl.paired_rows([p], dr, kp, (0.2,), {"population": pop, "fit": fitname, "bootstrap_unit": "log"})
    pd.DataFrame(rows).to_csv(OUT / "x2_transfer.csv", index=False)
    print("wrote", OUT / "x2_transfer.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate", "tables", "settings", "allocators"], required=True)
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"e8b_{args.stage}", vars(args))
    sl, split, ad = setup()
    ok = {"gate": stage_gate, "tables": stage_tables, "settings": stage_settings,
          "allocators": stage_allocators}[args.stage](args, sl, split, ad)
    (run / "summary.json").write_text(json.dumps({"stage": args.stage, "ok": ok is not False}, indent=1))
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
