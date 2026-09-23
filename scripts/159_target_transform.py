#!/usr/bin/env python
"""Task 30 (POST-HOC: decided after Task 24's test results were seen): training objective against target transform
for the R1 regression routers, and training-seed variation of the headline learned allocators.

  --part 1   R1_mlp_reg and R1_gbm_reg fitted on raw V (refit asserted equal to the shipped scores), rank(V),
             signed ECDF of V, and Q and G in their shipped CDF forms (refits asserted equal to Task 24's); the binary
             targets (V > 0, Q > 0, G > 0) scored from their caches as a separate comparison. Ten core cells, the
             benchmark's scorer on routers_r1's bootstrap draws.        -> results/final/target_transform_control.csv
                                                                           results/final/target_transform_heatmap.csv
  --part heatmap   the heatmaps of part 1 (architecture x cell, paired difference to raw V; 20 % large, the other
             quotas as small multiples)                             -> results/final/figures/target_transform_heatmap.{pdf,png}
  --part 2   R1 x 4 and gate_gbm refitted with seeds 1-5 beside the shipped seed 0 (a model whose fit does not depend
             on the seed is shown to be so and not refitted); 20 %.  -> results/final/seed_variation.csv

Architectures, features, hyperparameters, fit units (train + val), loss and preprocessing are the shipped ones (103's
`models`, 92's gate); only the label (part 1) or the seed (part 2) changes. New files only.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import predict, runmeta, scoring, submission as S                      # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

FINAL = Path(RESULTS) / "final"
NBOOT = 1000
LABEL_RUN = ROOT / "results" / "raw" / "20260922_193146_published_objective_labels"
REG = ("R1_mlp_reg", "R1_gbm_reg")
CLF = ("R1_mlp_clf", "R1_gbm_clf")
LABELS = ("V", "rankV", "secdfV", "Q", "G")
SEEDS = (0, 1, 2, 3, 4, 5)
POST_HOC = "post-hoc (decided after Task 24's test results were seen)"


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rank_v(v):
    """rank(V): ordinal ranks / n on the fit units, ties as 148's Q-MORIC (`argsort(argsort)`, quicksort)."""
    t148 = sys.modules["t148"]
    return t148.q_moric(np.asarray(v, float))


def secdf_v(v):
    """Signed ECDF: F+(V) for V > 0, 0 for V = 0, F-(V) - 1 for V < 0; F+ and F- the empirical CDFs (right-continuous,
    as 148's `g_moric`) of the positive and of the negative fit-unit values."""
    v = np.asarray(v, float)
    out = np.zeros(len(v))
    pos, neg = np.sort(v[v > 0]), np.sort(v[v < 0])
    if len(pos):
        out[v > 0] = np.searchsorted(pos, v[v > 0], side="right") / len(pos)
    if len(neg):
        out[v < 0] = np.searchsorted(neg, v[v < 0], side="right") / len(neg) - 1.0
    return out


def mlp(task, seed):
    cls = MLPRegressor if task == "reg" else MLPClassifier
    return Pipeline([("s", StandardScaler()), ("m", cls(hidden_layer_sizes=(64, 64), alpha=1e-4, max_iter=300,
                                                        random_state=seed))])


def cells(t103, t92):
    """The ten core cells: Task 24's label frame (all splits), the R1 input matrix, the gate features."""
    mats = {"nuScenes": t103.frame_matrix(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "nuScenes"),
            "KITTI": t103.frame_matrix(Path(CACHE) / "det", "cheap_320", "KITTI")}
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    gate = {}
    for gen in (t92.nuscenes_cells, t92.kitti_cells):
        for c in gen(splits):
            gate[(c["track"], c["geometry"], c["system"], c["target"])] = (c["d"].reset_index(drop=True), c["fcols"])
    out = []
    for f in sorted(LABEL_RUN.glob("labels__*.npz")):
        _, track, geometry, system, target = f.stem.split("__")
        z = np.load(f, allow_pickle=False)
        lab = pd.DataFrame({k: z[k] for k in z.files}).astype({"seq": str, "frame": int})
        keys, M = mats[track]
        j = lab[["seq", "frame"]].merge(keys.assign(_row=np.arange(len(keys))), on=["seq", "frame"], how="left",
                                        validate="one_to_one")
        assert j._row.notna().all()
        d, fcols = gate[(track, geometry, system, target)]
        jd = lab[["seq", "frame"]].merge(d[["seq", "frame"]].astype({"seq": str, "frame": int}).assign(_d=np.arange(len(d))),
                                         on=["seq", "frame"], how="left", validate="one_to_one")
        assert jd._d.notna().all()
        dd = d.iloc[jd._d.to_numpy(int)].reset_index(drop=True)
        Xg = np.nan_to_num(dd[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64),
                           nan=0.0, posinf=1e6, neginf=-1e6)
        out.append(dict(key=(track, geometry, system, target), lab=lab, X=M[j._row.to_numpy(int)], Xg=Xg,
                        fit=lab.split.isin(["train", "val"]).to_numpy(), test=(lab.split == "test").to_numpy()))
    return out


def evaluate(c, scores, plan):
    """The benchmark's scorer on the plan's draws for this cell: (ks, prize, point, draws, dropped)."""
    lab, te = c["lab"], c["test"]
    cell = c["key"]
    d = S.decision_values()
    m = np.logical_and.reduce([d[k].astype(str) == str(x) for k, x in zip(S.CELL, cell)]) & (d.split == "test")
    ref = d[m].reset_index(drop=True)
    t = lab[te].reset_index(drop=True)
    # the scorer's row order is the decision-values file's; align every score to it
    j = ref[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(
        t[["seq", "frame"]].assign(_i=np.arange(len(t))), on=["seq", "frame"], how="left", validate="one_to_one")
    assert j._i.notna().all()
    order = j._i.to_numpy(int)
    assert np.array_equal(ref.V.to_numpy(float), t.V.to_numpy(float)[order])
    sc = {k: (None if s is None else np.asarray(s, float)[te][order]) for k, s in scores.items()}
    return scoring.evaluate(ref.V.to_numpy(float), ref.unit.to_numpy(), sc, NBOOT, S.plan_rng(plan, cell, NBOOT))


def ci(x):
    return scoring.ci(np.asarray(x, float))


# ------------------------------------------------------------------------------------------------ part 1

def part1(run):
    t103 = _load("t103", "103_routers_r1.py")
    _load("t148", "148_published_objective_labels.py")
    t92 = t103.t92
    off_r1 = rap_runs.latest("routers_r1")
    pub_r1 = sorted((ROOT / "results" / "raw").glob("*_published_objective_r1"))[-1]
    rt = pd.read_csv(FINAL / "benchmark_table_routers.csv", float_precision="round_trip")
    rows, checks, pooled_draws = [], [], {}
    for c in cells(t103, t92):
        t0 = time.time()
        track, geometry, system, target = c["key"]
        stem = f"{track}__{geometry}__{system}__{target}"
        lab, X, fit = c["lab"], c["X"], c["fit"]
        v = lab.V.to_numpy(float)
        zv = np.load(off_r1 / f"scores__{stem}.npz", allow_pickle=False)
        zp = np.load(pub_r1 / f"r1__{stem}.npz", allow_pickle=False)
        assert (zv["seq"].astype(str) == lab.seq.to_numpy()).all() and (zp["seq"].astype(str) == lab.seq.to_numpy()).all()
        targets = {"V": v[fit], "rankV": rank_v(v[fit]), "secdfV": secdf_v(v[fit]),
                   "Q": lab.Q_moric.to_numpy(float)[fit], "G": lab.G_moric.to_numpy(float)[fit]}
        scores = {"random": None}
        for a in REG:
            make = t103.models()[a][1]
            for lbl, y in targets.items():
                s = make().fit(X[fit], y).predict(X)
                if lbl in ("V", "Q", "G"):
                    ref = zv[a] if lbl == "V" else zp[f"{a}__{lbl}"]
                    diff = float(np.max(np.abs(s - ref)))
                    checks.append({"cell": stem, "model": a, "label": lbl, "max_abs_diff_vs_cached": diff,
                                   "cached_run": off_r1.name if lbl == "V" else pub_r1.name})
                    if diff != 0.0:
                        raise SystemExit(f"part 1: the {a} refit on {lbl} does not reproduce the cached scores "
                                         f"({stem}, max |diff| {diff:.3g})")
                scores[f"{a}|{lbl}"] = s
        for a in CLF:                                     # the binary targets, from their caches
            scores[f"{a}|V"] = np.asarray(zv[a], float)
            for o in ("Q", "G"):
                scores[f"{a}|{o}"] = np.asarray(zp[f"{a}__{o}"], float)
        ks, prize, point, draws, dropped = evaluate(c, scores, "routers_r1")
        # the raw-V rows must be benchmark_table_routers.csv's own
        for a in REG + CLF:
            off = rt[(rt.track == track) & (rt.geometry == geometry) & (rt.system == system) & (rt.target == target)
                     & (rt.signal == a)].sort_values("quota")
            mine = point[f"{a}|V"]["eta"]
            assert np.array_equal(off.eta.to_numpy(float), np.asarray(mine, float)), (stem, a)
        for arch, labels in [(a, LABELS) for a in REG] + [(a, ("V", "Q", "G")) for a in CLF]:
            for lbl in labels:
                name = f"{arch}|{lbl}"
                for qi, q in enumerate(scoring.QUOTAS):
                    dr = draws[name][:, qi] - draws["random"][:, qi]
                    r = {"status": POST_HOC, "track": track, "geometry": geometry, "system": system, "target": target,
                         "architecture": arch, "kind": "regression" if arch in REG else "binary", "label": lbl,
                         "quota": q, "k": int(ks[qi]), "ndg": float(point[name]["eta"][qi]),
                         "minus_random": float(np.nanmean(dr)), "boot_dropped": int(dropped[qi])}
                    r["minus_random_lo"], r["minus_random_hi"] = ci(dr)
                    for base in ("V",) + (("rankV", "secdfV") if (arch in REG and lbl in ("Q", "G")) else ()):
                        if lbl == base:
                            continue
                        dd = draws[name][:, qi] - draws[f"{arch}|{base}"][:, qi]
                        r[f"minus_{base}"] = float(point[name]["eta"][qi] - point[f"{arch}|{base}"]["eta"][qi])
                        r[f"minus_{base}_lo"], r[f"minus_{base}_hi"] = ci(dd)
                        pooled_draws.setdefault((arch, lbl, base, q), []).append((dd, r[f"minus_{base}"]))
                    rows.append(r)
        print(f"  part 1 {stem} [{time.time() - t0:.0f}s]", flush=True)
    for (arch, lbl, base, q), items in pooled_draws.items():
        mat = np.vstack([d_ for d_, _ in items])
        lo, hi = ci(np.nanmean(mat, axis=0))
        rows.append({"status": POST_HOC, "track": "pooled", "architecture": arch,
                     "kind": "regression" if arch in REG else "binary", "label": lbl, "quota": q,
                     "pooled_against": base, "n_cells": len(items), f"minus_{base}": float(np.mean([p for _, p in items])),
                     f"minus_{base}_lo": lo, f"minus_{base}_hi": hi,
                     "note": "mean over the ten core cells of the paired difference; interval from the draw-wise mean "
                             "(each cell on its own routers_r1 draws)"})
    pd.DataFrame(checks).to_csv(run / "refit_checks.csv", index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ part 2

def part2(run):
    t103 = _load("t103", "103_routers_r1.py")
    t92 = t103.t92
    off_r1 = rap_runs.latest("routers_r1")
    gate_run = rap_runs.latest("budget_gate_scores")
    rows, seed_check = [], []
    q20 = scoring.QUOTAS.index(0.20)
    for c in cells(t103, t92):
        t0 = time.time()
        track, geometry, system, target = c["key"]
        stem = f"{track}__{geometry}__{system}__{target}"
        lab, X, Xg, fit = c["lab"], c["X"], c["Xg"], c["fit"]
        v = lab.V.to_numpy(float)
        zv = np.load(off_r1 / f"scores__{stem}.npz", allow_pickle=False)
        zg = np.load(gate_run / f"gates__{stem}.npz", allow_pickle=False)
        jg = lab[["seq", "frame"]].merge(pd.DataFrame({"seq": zg["key_seq"].astype(str), "frame": zg["key_frame"].astype(int),
                                                       "_g": np.arange(len(zg["V"]))}), on=["seq", "frame"], how="left",
                                         validate="one_to_one")
        shipped_gate = np.asarray(zg["gate_gbm"], float)[jg._g.to_numpy(int)]
        scores = {"R1": {}, "gate": {}}
        for a in ("R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "gate_gbm"):
            task = "clf" if a.endswith("_clf") else "reg"
            y = v if task == "reg" else (v > scoring.EPS).astype(int)
            Xa = Xg if a == "gate_gbm" else X
            per = {}
            for seed in SEEDS:
                if task == "clf" and len(np.unique(y[fit])) < 2:
                    per[seed] = np.zeros(len(v))
                    continue
                m = mlp(task, seed) if a.startswith("R1_mlp") else predict.make_model("gbm", task, seed)
                m.fit(Xa[fit], y[fit])
                per[seed] = m.predict(Xa) if task == "reg" else m.predict_proba(Xa)[:, 1]
                if seed == 0:
                    ref = shipped_gate if a == "gate_gbm" else np.asarray(zv[a], float)
                    diff = float(np.max(np.abs(per[0] - ref)))
                    seed_check.append({"cell": stem, "model": a, "check": "seed 0 refit vs shipped", "max_abs_diff": diff})
                    if diff != 0.0:
                        raise SystemExit(f"part 2: the seed-0 refit of {a} does not reproduce the shipped scores "
                                         f"({stem}, max |diff| {diff:.3g})")
                if seed == 1 and not a.startswith("R1_mlp"):
                    same = bool(np.array_equal(per[0], per[1]))
                    seed_check.append({"cell": stem, "model": a, "check": "seed 1 fit equals seed 0 fit",
                                       "max_abs_diff": float(np.max(np.abs(per[0] - per[1]))), "identical": same})
                    if same:                               # the fit does not depend on the seed: no further refits
                        per.update({s: per[0] for s in SEEDS[2:]})
                        break
            for seed in SEEDS:
                scores["gate" if a == "gate_gbm" else "R1"][f"{a}|seed{seed}"] = per[seed]
        for group, plan in (("R1", "routers_r1"), ("gate", "benchmark_table")):
            ks, prize, point, draws, dropped = evaluate(c, {"random": None, **scores[group]}, plan)
            for name in scores[group]:
                a, seed = name.split("|seed")
                dr = draws[name][:, q20] - draws["random"][:, q20]
                lo, hi = ci(dr)
                rows.append({"status": POST_HOC, "track": track, "geometry": geometry, "system": system, "target": target,
                             "signal": a, "seed": int(seed), "shipped_seed": int(seed) == 0, "quota": 0.20, "plan": plan,
                             "ndg": float(point[name]["eta"][q20]), "minus_random": float(np.nanmean(dr)),
                             "minus_random_lo": lo, "minus_random_hi": hi, "win": bool(np.isfinite(lo) and lo > 0),
                             "loss": bool(np.isfinite(hi) and hi < 0)})
        print(f"  part 2 {stem} [{time.time() - t0:.0f}s]", flush=True)
    df = pd.DataFrame(rows)
    spread = (df.groupby(["track", "geometry", "system", "target", "signal"])
              .agg(ndg_min=("ndg", "min"), ndg_median=("ndg", "median"), ndg_max=("ndg", "max"),
                   wins=("win", "sum"), seeds=("seed", "nunique")).reset_index())
    ship = df[df.shipped_seed][["track", "geometry", "system", "target", "signal", "win"]].rename(columns={"win": "shipped_win"})
    spread = spread.merge(ship, on=["track", "geometry", "system", "target", "signal"])
    spread["flag_shipped_win_not_robust"] = spread.shipped_win & (spread.wins < 4)
    spread["status"] = POST_HOC
    pd.DataFrame(seed_check).to_csv(run / "seed_checks.csv", index=False)
    return pd.concat([df.assign(row_type="seed"), spread.assign(row_type="spread")], ignore_index=True)


def heatmap():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.read_csv(FINAL / "target_transform_heatmap.csv")
    h["cell"] = h.track.str.replace("nuScenes", "nuSc.") + " " + h.geometry + "\n" + h.system
    cells_ = list(dict.fromkeys(h.cell))
    rows_ = [(a, l) for a in REG for l in LABELS if l != "V"]
    fig = plt.figure(figsize=(13, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.45)
    lim = float(np.nanmax(np.abs(h.minus_V)))
    for i, q in enumerate((0.20, 0.10, 0.30, 0.50)):
        ax = fig.add_subplot(gs[0, :]) if i == 0 else fig.add_subplot(gs[1, i - 1])
        x = h[np.isclose(h.quota, q)]
        M = np.array([[x[(x.architecture == a) & (x.label == l) & (x.cell == c)].minus_V.mean() for c in cells_]
                      for a, l in rows_])
        S_ = np.array([[bool(((x.architecture == a) & (x.label == l) & (x.cell == c) &
                              ((x.minus_V_lo > 0) | (x.minus_V_hi < 0))).any()) for c in cells_] for a, l in rows_])
        im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        if i == 0:
            for r in range(M.shape[0]):
                for cc in range(M.shape[1]):
                    ax.text(cc, r, f"{M[r, cc]:+.2f}" + ("*" if S_[r, cc] else ""), ha="center", va="center", fontsize=7)
            ax.set_xticks(range(len(cells_)))
            ax.set_xticklabels(cells_, fontsize=8)
        else:
            ax.set_xticks([])
        ax.set_yticks(range(len(rows_)))
        ax.set_yticklabels([f"{a.replace('R1_', '').replace('_reg', '')} {l}" for a, l in rows_] if i in (0, 1) else [],
                           fontsize=8)
        ax.set_title(f"{int(q * 100)} %: nDG(label) - nDG(raw V)" + (" (* paired 95 % interval excludes 0)" if i == 0 else ""),
                     fontsize=9)
    fig.colorbar(im, ax=fig.axes, shrink=0.6)
    fig.suptitle("R1 regression routers: target transform and objective against raw V, ten core cells (post-hoc)",
                 fontsize=10)
    out = FINAL / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"target_transform_heatmap.{ext}", dpi=150, bbox_inches="tight")
    print(f"  wrote {out}/target_transform_heatmap.{{pdf,png}}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["1", "2", "heatmap"])
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    if args.part == "heatmap":
        heatmap()
        return
    run = runmeta.new_run(f"target_transform_part{args.part}", vars(args))
    t0 = time.time()
    if args.part == "1":
        df = part1(run)
        heat = df[(df.track != "pooled") & (df.kind == "regression") & df.minus_V.notna()][
            ["architecture", "label", "track", "geometry", "system", "target", "quota", "minus_V", "minus_V_lo",
             "minus_V_hi"]].assign(status=POST_HOC)
        outs = {"target_transform_control.csv": df, "target_transform_heatmap.csv": heat}
    else:
        outs = {"seed_variation.csv": part2(run)}
    for name, x in outs.items():
        x.to_csv(run / name, index=False)
        x.to_csv(FINAL / name, index=False)
        print(f"  wrote {FINAL / name} ({len(x)} rows)", flush=True)
    secs = time.time() - t0
    (run / "timing.json").write_text(json.dumps({"part": args.part, "seconds": round(secs)}, indent=1))
    print(f"  part {args.part} took {secs:.0f}s", flush=True)


if __name__ == "__main__":
    main()
