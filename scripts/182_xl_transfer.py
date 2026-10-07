#!/usr/bin/env python
"""Follow-up E1, X6 (secondary): does an allocator fitted elsewhere transfer to the new test scenes?

  frozen85->new   fitted on the frozen 85 scenes' train + validation units, scored on the 228 new test scenes
  boston->sg      fitted on the new train + validation scenes in Boston, scored on the new test scenes in Singapore
  sg->boston      the reverse
  kitti->nusc     braking: fitted on every KITTI frame (YOLOv8s 320->640, same geometry) on the feature columns both
                  tracks share, scored on the 228 new test scenes
Signals: the two gates and the gradient-boosted / MLP routers on V (103's detection list, normalised by image size).
Realized gain against random and nDG at 20%, every-draw scene bootstrap; the in-domain fit (new train + validation)
beside each. -> x6_transfer.csv
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, runmeta, xl                                              # noqa: E402
from rap.paths import CACHE                                                      # noqa: E402

SIGNALS = ("gate_ridge", "gate_gbm", "R1_gbm_reg", "R1_mlp_reg")


def scores_for(c, mat, v, fit, fcols=None):
    d = c["d"]
    g = xl.gate_scores(d, fcols or c["fcols"], v, fit)
    r = xl.t103().fit_score(xl.r1_rows(d, mat), v, fit)
    return {s: (g if s.startswith("gate") else r)[s] for s in SIGNALS}


def evaluate(c, sc, test, setting):
    p = xl.prepare(c, test, sc, (0.2,))
    dr, kp = xl.paired_draws([p], 1000, np.random.default_rng(0), (0.2,))
    return pd.DataFrame(xl.paired_rows([p], dr, kp, (0.2,), {"transfer": setting, "bootstrap_unit": "scene"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(Path(CACHE) / "xl" / "tables850"))
    ap.add_argument("--det", default=str(Path(CACHE) / "nusc_det_xl"))
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_transfer", vars(args))
    sp = xl.scene_split()
    xl.use_ego_table(Path(args.tables) / "v_ego_causal.csv")
    feats, fcols = xl.features(args.det)
    mat = xl.r1_matrix(args.det)
    new, old = sp[sp.group == "new765"], sp[sp.group == "frozen85"]
    test = set(new[new.deep_split == "test"].scene)
    loc = dict(zip(sp.scene, sp.location))
    # one split map over the frozen 85 (train/val as fit units) and the new scenes; the fit mask chooses per setting
    split_of = {**{s: ("old_" + k) for s, k in zip(old.scene, old.deep_split)}, **dict(zip(new.scene, new.deep_split))}
    out = []
    for c in xl.cells(args.tables, feats, fcols, split_of):
        d = c["d"]
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        te = d.seq.isin(test).to_numpy()
        boston = d.seq.map(loc).eq("boston-seaport").to_numpy()
        settings = {"in_domain": d.split.isin(["train", "val"]).to_numpy(),
                    "frozen85->new": d.split.isin(["old_train", "old_val"]).to_numpy()}
        for name, fit in settings.items():
            out.append(evaluate(c, scores_for(c, mat, v, fit), te, name))
        trv = d.split.isin(["train", "val"]).to_numpy()
        for name, fit, tmask in (("boston->sg", trv & boston, te & ~boston), ("sg->boston", trv & ~boston, te & boston),
                                 ("in_domain sg", trv, te & ~boston), ("in_domain boston", trv, te & boston)):
            out.append(evaluate(c, scores_for(c, mat, v, fit), tmask, name))
        print(f"  {c['geometry']} {c['system']}", flush=True)
    # KITTI -> nuScenes braking on shared feature columns (KITTI: every frame of the mono and oracle cells)
    t92 = xl.t92()
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    kmat = xl.t103().frame_matrix(Path(CACHE) / "det", "cheap_320", "KITTI")
    for kc in t92.kitti_cells(splits):
        if kc["system"] != "brake":
            continue
        shared = [f for f in kc["fcols"] if f in fcols]
        nc = next(c for c in xl.cells(args.tables, feats, fcols, split_of, geometries=(kc["geometry"],))
                  if c["system"] == "brake")
        kd, nd = kc["d"], nc["d"]
        kv = (kd[kc["cheap"]] - kd[kc["full"]]).to_numpy(float)
        Xk = np.nan_to_num(kd[shared].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        Xn = np.nan_to_num(nd[shared].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        sc = {}
        for name, mdl in xl.GATES.items():
            m = xl.predict.make_model(mdl, "reg", 0).fit(Xk, kv)
            sc[name] = m.predict(Xn)
        R1k = xl.r1_rows(kd.astype({"seq": str}), kmat)
        R1n = xl.r1_rows(nd, mat)
        allfit = np.ones(len(kv), bool)
        for name, (task, make) in xl.t103().models().items():
            if name not in SIGNALS:
                continue
            mdl = make().fit(R1k, kv)
            sc[name] = mdl.predict(R1n)
        te = nd.seq.isin(test).to_numpy()
        r = evaluate(nc, sc, te, "kitti->nusc")
        r["shared_feature_columns"] = len(shared)
        out.append(r)
    t = pd.concat(out, ignore_index=True)
    t.to_csv(xl.OUT / "x6_transfer.csv", index=False)
    t.to_csv(run / "x6_transfer.csv", index=False)
    print(t[(t.system == "brake")][["geometry", "transfer", "signal", "delta_share_all_cheap", "delta_share_lo_all_draws",
                                     "delta_share_hi_all_draws", "ndg"]].to_string(index=False))


if __name__ == "__main__":
    main()
