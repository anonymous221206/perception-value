#!/usr/bin/env python
"""Follow-up E1: the learning curve (P3, X4).

Allocators are fitted on the first n of the 385 new training scenes (n in 48, 96, 192, 385; five nested orderings,
seeds 0-4; training scenes only, no validation scenes) and scored once on the 228 new test scenes at 20%.
Primary: gate_gbm and R1_gbm_reg on braking under monocular geometry, endpoint = value at 385 minus the mean over
orderings at 48, paired over test scenes within each of 1,000 scene-bootstrap draws, for realized gain (share of the
all-cheap loss) and for nDG. Secondary: the other cells and learners, and the MLP routers over seeds 0-4 (braking,
monocular). Writes results/followup/E1/x4_learning_curve.csv. Protocol: docs/followup_protocol.md, Section 5.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, runmeta, xl                                              # noqa: E402
from rap.paths import CACHE                                                      # noqa: E402
from rap.scoring import EPS                                                      # noqa: E402

SIZES = (48, 96, 192, 385)
ORDERINGS = range(5)
PRIMARY = ("gate_gbm", "R1_gbm_reg")
Q = 0.2


def mlp(task, seed):
    m = (MLPRegressor if task == "reg" else MLPClassifier)(hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=300,
                                                            random_state=seed)
    return Pipeline([("s", StandardScaler()), ("m", m)])


def fit_all(c, X1, v, fit, mlp_seeds):
    d = c["d"]
    out = {**xl.gate_scores(d, c["fcols"], v, fit), **xl.t103().fit_score(X1, v, fit)}
    for seed in mlp_seeds:
        if seed == 0:
            continue                                     # seed 0 is fit_score's own MLP
        for task in ("reg", "clf"):
            y = v if task == "reg" else (v > EPS).astype(int)
            m = mlp(task, seed).fit(X1[fit], y[fit])
            out[f"R1_mlp_{task}_seed{seed}"] = m.predict(X1) if task == "reg" else m.predict_proba(X1)[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--det", default=str(CACHE / "nusc_det_xl"))
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_learning_curve", vars(args))
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    train = sorted(new[new.deep_split == "train"].scene)
    assert len(train) == 385
    order = {o: list(np.random.default_rng(o).permutation(train)) for o in ORDERINGS}
    xl.use_ego_table(Path(args.tables) / "v_ego_causal.csv")
    feats, fcols = xl.features(args.det)
    mat = xl.r1_matrix(args.det)

    rows, endpoint = [], []
    for c in xl.cells(args.tables, feats, fcols, split_of):
        d = c["d"]
        primary_cell = c["geometry"] == "mono" and c["system"] == "brake"
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        test = (d.split == "test").to_numpy()
        vt, units = v[test], d.seq.to_numpy()[test]
        tot = float(d[c["cheap"]].to_numpy(float)[test].sum())
        X1 = xl.r1_rows(d, mat)
        k = max(int(round(Q * len(vt))), 1)
        prize = xl.gains(vt, vt, [k])[0]
        rnd = k / len(vt) * vt.sum()
        scores = {}                                       # (n, ordering) -> {signal: test scores}
        for n in SIZES:
            for o in (ORDERINGS if n < 385 else [0]):
                t0 = time.time()
                fit = d.seq.isin(set(order[o][:n])).to_numpy()
                sc = fit_all(c, X1, v, fit, range(5) if primary_cell else [0])
                scores[(n, o)] = {s: x[test] for s, x in sc.items()}
                for s, x in scores[(n, o)].items():
                    g = xl.gains(x, vt, [k])[0]
                    rows.append({"signal": s, "cell": f"{c['geometry']} {c['system']} {c['target']}",
                                 "geometry": c["geometry"], "system": c["system"], "train_scenes": n,
                                 "ordering_seed": o, "quota": Q, "ndg": g / prize if prize > EPS else np.nan,
                                 "gain_share_all_cheap": g / tot, "delta_share_all_cheap": (g - rnd) / tot,
                                 "primary": primary_cell and s in PRIMARY})
                print(f"  {c['geometry']:6s} {c['system']:8s} n={n:3d} ordering {o} [{time.time() - t0:.0f}s]", flush=True)
        # every-draw bootstrap: per draw, the gain at each (n, ordering) and the endpoint 385 - mean(48)
        uniq = np.array(sorted(set(units)))
        idx = {u: np.flatnonzero(units == u) for u in uniq}
        rng = np.random.default_rng(0)
        sigs = list(scores[(385, 0)])
        bt = {key: {s: np.full(args.nboot, np.nan) for s in sigs} for key in scores}
        bn = {key: {s: np.full(args.nboot, np.nan) for s in sigs} for key in scores}
        for b in range(args.nboot):
            t = np.concatenate([idx[u] for u in uniq[rng.integers(0, len(uniq), len(uniq))]])
            vb = vt[t]
            kb = max(int(round(Q * len(t))), 1)
            pz = xl.gains(vb, vb, [kb])[0]
            tb = float(d[c["cheap"]].to_numpy(float)[test][t].sum())
            rb = kb / len(t) * vb.sum()
            for key, sc in scores.items():
                for s in sigs:
                    g = xl.gains(sc[s][t], vb, [kb])[0]
                    bt[key][s][b] = (g - rb) / tb
                    bn[key][s][b] = g / pz if pz > EPS else np.nan
        for r in rows[-len(scores) * len(sigs):]:
            key = (r["train_scenes"], r["ordering_seed"])
            r["delta_share_lo"], r["delta_share_hi"] = xl.ci(bt[key][r["signal"]])
            r["ndg_lo"], r["ndg_hi"] = xl.ci(bn[key][r["signal"]])
        for s in sigs:
            e_share = bt[(385, 0)][s] - np.mean([bt[(48, o)][s] for o in ORDERINGS], axis=0)
            e_ndg = bn[(385, 0)][s] - np.mean([bn[(48, o)][s] for o in ORDERINGS], axis=0)
            p385 = [r for r in rows if r["signal"] == s and r["geometry"] == c["geometry"] and r["system"] == c["system"]]
            pt = lambda n, f: np.mean([r[f] for r in p385 if r["train_scenes"] == n])
            lo, hi = xl.ci(e_share)
            nlo, nhi = xl.ci(e_ndg)
            endpoint.append({"signal": s, "cell": f"{c['geometry']} {c['system']} {c['target']}",
                             "geometry": c["geometry"], "system": c["system"], "train_scenes": "385-48",
                             "ordering_seed": "mean", "quota": Q,
                             "delta_share_all_cheap": pt(385, "delta_share_all_cheap") - pt(48, "delta_share_all_cheap"),
                             "delta_share_lo": lo, "delta_share_hi": hi,
                             "ndg": pt(385, "ndg") - pt(48, "ndg"), "ndg_lo": nlo, "ndg_hi": nhi,
                             "gain_share_all_cheap": pt(385, "gain_share_all_cheap") - pt(48, "gain_share_all_cheap"),
                             "primary": primary_cell and s in PRIMARY})
    out = pd.concat([pd.DataFrame(rows), pd.DataFrame(endpoint)], ignore_index=True)
    out.to_csv(xl.OUT / "x4_learning_curve.csv", index=False)
    out.to_csv(run / "x4_learning_curve.csv", index=False)
    print(pd.DataFrame(endpoint).query("primary")[["signal", "delta_share_all_cheap", "delta_share_lo",
                                                    "delta_share_hi", "ndg", "ndg_lo", "ndg_hi"]].to_string(index=False))


if __name__ == "__main__":
    main()
