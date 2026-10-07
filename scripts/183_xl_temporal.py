#!/usr/bin/env python
"""Follow-up E1, temporal-context router (secondary): 103's detection-list router with a short history.

Input per frame: 103's 225-dim CHEAP detection list for frames t-3 .. t of the same scene (zeros before the first
frame), plus the causal ego acceleration (backward difference of the causal ego speed) and yaw rate (backward
difference of the ego heading) -- information available before escalating frame t. Models: 103's GBM and MLP on V
(regression) and 1[V > 0], fit on the new train + validation scenes, scored once on the 228 new test scenes at 20%,
every-draw scene bootstrap against random; 103's single-frame routers beside them. -> x6_temporal.csv
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, runmeta, xl                                              # noqa: E402
from rap.nusc import NuScenesDB                                                  # noqa: E402
from rap.paths import CACHE, NUSCENES_XL                                           # noqa: E402

HIST = 3


def ego_dynamics(db, scenes, vtab):
    """Causal ego acceleration and yaw rate per (scene, frame)."""
    by_name = {db.scene_name(s): s for s in db.scenes}
    rows = []
    v = vtab.set_index(["seq", "frame"]).v_ego_causal
    for name in scenes:
        toks = db.samples(by_name[name])
        ts = np.array([db._t["sample"][tk]["timestamp"] / 1e6 for tk in toks])
        yaw = np.array([np.arctan2(R[1, 0], R[0, 0]) for R, _ in (db._global_to_ego(tk) for tk in toks)])
        for i in range(len(toks)):
            dt = ts[i] - ts[i - 1] if i else np.nan
            acc = (v[(name, i)] - v[(name, i - 1)]) / dt if i else 0.0
            dy = np.angle(np.exp(1j * (yaw[i] - yaw[i - 1]))) / dt if i else 0.0
            rows.append({"seq": name, "frame": i, "ego_acc": float(acc), "yaw_rate": float(dy)})
    return pd.DataFrame(rows)


def history_matrix(keys, M):
    """[x_{t-3}, x_{t-2}, x_{t-1}, x_t] per row of (seq, frame) keys, zeros where the scene has no such frame."""
    k = keys.reset_index(drop=True)
    pos = {(s, f): i for i, (s, f) in enumerate(zip(k.seq, k.frame))}
    blocks = []
    for lag in range(HIST, -1, -1):
        idx = np.array([pos.get((s, f - lag), -1) for s, f in zip(k.seq, k.frame)])
        b = np.zeros((len(k), M.shape[1]))
        b[idx >= 0] = M[idx[idx >= 0]]
        blocks.append(b)
    return np.hstack(blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(Path(CACHE) / "xl" / "tables850"))
    ap.add_argument("--det", default=str(Path(CACHE) / "nusc_det_xl"))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_temporal", vars(args))
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    vtab = pd.read_csv(Path(args.tables) / "v_ego_causal.csv").astype({"seq": str, "frame": int})
    xl.use_ego_table(Path(args.tables) / "v_ego_causal.csv")
    feats, fcols = xl.features(args.det)
    keys, M = xl.r1_matrix(args.det)
    H = history_matrix(keys, M)
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    dyn = ego_dynamics(db, sorted(new.scene), vtab)
    out = []
    for c in xl.cells(args.tables, feats, fcols, split_of):
        d = c["d"]
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        fit, te = d.split.isin(["train", "val"]).to_numpy(), (d.split == "test").to_numpy()
        kk = keys.assign(_row=np.arange(len(keys)))
        j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(kk, on=["seq", "frame"], how="left", validate="one_to_one")
        e = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(dyn, on=["seq", "frame"], how="left", validate="one_to_one")
        X = np.hstack([H[j._row.to_numpy(int)], e[["ego_acc", "yaw_rate"]].to_numpy(float)])
        temporal = {f"{k}_temporal": s for k, s in xl.t103().fit_score(X, v, fit).items()}
        single = xl.t103().fit_score(xl.r1_rows(d, (keys, M)), v, fit)
        p = xl.prepare(c, te, {**single, **temporal}, (0.2,))
        dr, kp = xl.paired_draws([p], 1000, np.random.default_rng(0), (0.2,))
        out.append(pd.DataFrame(xl.paired_rows([p], dr, kp, (0.2,), {"input_dim": X.shape[1]})))
        print(f"  {c['geometry']} {c['system']}", flush=True)
    t = pd.concat(out, ignore_index=True)
    t.to_csv(xl.OUT / "x6_temporal.csv", index=False)
    t.to_csv(run / "x6_temporal.csv", index=False)
    print(t[["geometry", "system", "signal", "delta_share_all_cheap", "delta_share_lo_all_draws",
             "delta_share_hi_all_draws", "ndg"]].to_string(index=False))


if __name__ == "__main__":
    main()
