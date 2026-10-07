#!/usr/bin/env python
"""Follow-up E1, X5 (secondary): multi-fidelity allocation on the follow-up nuScenes scenes.

93's multi-fidelity allocator, unchanged, on braking at four YOLOv8s fidelities (320 cheap; 480, 640, 960 as
escalation levels, each an escalation from all-320 operation on the shared action history): the oracle and one ridge
and one gradient-boosted gate per level (fit on the new train + validation scenes) take the upper-hull increments of
their predicted per-level gains in order of gain per cost until the budget is spent; single-level comparators send
every escalated input to 640. Budgets b = C_320 + alpha x C_640 per frame, alpha in {0.05, 0.1, 0.2, 0.3, 0.5}, in ms
and in module mJ from x7_costs.csv; gates are charged with 93's overhead of three level models, or free. 228 test
scenes, both geometries; the multi-fidelity oracle's gain is the denominator. -> x5_multifidelity.csv
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, runmeta, xl                                              # noqa: E402
from rap.paths import CACHE, RESULTS                                             # noqa: E402
from rap.scoring import EPS                                                      # noqa: E402

ALPHAS = (0.05, 0.10, 0.20, 0.30, 0.50)
LEVELS = {"480": "ns_cheap_320tons_mid_480", "640": "ns_cheap_320tons_full_640", "960": "ns_cheap_320tons_full_960"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(Path(CACHE) / "xl" / "tables850"))
    ap.add_argument("--det", default=str(Path(CACHE) / "nusc_det_xl"))
    ap.add_argument("--costs", default=str(Path(RESULTS) / "followup" / "E5" / "x7_costs.csv"))
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_multifidelity", vars(args))
    t93 = xl.load_script("t93", "93_budget_allocation.py")
    ov = json.loads((Path(RESULTS) / "final" / "benchmark_budget_overheads.json").read_text())["overheads"]
    cost = pd.read_csv(args.costs).set_index("engine")
    C = {lvl: {"ms": cost.loc[f"yolov8s_{m}", "ms"], "mJ": cost.loc[f"yolov8s_{m}", "mJ_module"]}
         for lvl, m in (("cheap", "ns_cheap_320"), ("480", "ns_mid_480"), ("640", "ns_full_640"), ("960", "ns_full_960"))}
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    xl.use_ego_table(Path(args.tables) / "v_ego_causal.csv")
    feats, fcols = xl.features(args.det)
    rng = np.random.default_rng(0)
    rows = []
    for geom in xl.GEOMETRIES:
        t = {lvl: pd.read_pickle(Path(args.tables) / f"{geom}__{p}.pkl").set_index(["seq", "frame"]) for lvl, p in LEVELS.items()}
        base = t["640"]
        base = base[base.index.get_level_values(0).isin(set(split_of))]
        for lvl in ("480", "960"):
            assert np.array_equal(t[lvl].loc[base.index, "J_cheap"].to_numpy(float), base.J_cheap.to_numpy(float)), \
                f"{lvl}: the 320 branch differs from the 320->640 table"
        d = base.reset_index().merge(feats, on=["seq", "frame"], how="inner", validate="one_to_one")
        d["split"] = d.seq.map(split_of)
        gains = np.stack([(t[l].loc[list(zip(d.seq, d.frame)), "J_cheap"].to_numpy(float)
                           - t[l].loc[list(zip(d.seq, d.frame)), "J_full"].to_numpy(float)) for l in ("480", "640", "960")], 1)
        X = np.nan_to_num(d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
        fit, m = d.split.isin(["train", "val"]).to_numpy(), (d.split == "test").to_numpy()
        preds = {"oracle": gains, **t93.level_predictions(X, fit, gains)}
        gt = gains[m]
        n = len(gt)
        units = d.seq.to_numpy()[m]
        uniq = np.unique(units)
        wdraw = []
        for _ in range(args.nboot):
            w = np.zeros(n)
            for i in rng.integers(0, len(uniq), len(uniq)):
                w[units == uniq[i]] += 1
            wdraw.append(w)
        unc = pd.to_numeric(d["unc_sum"], errors="coerce").fillna(-np.inf).to_numpy()[m]
        w1 = np.ones(n)
        for unit in ("ms", "mJ"):
            lvl_cost = np.array([C["480"][unit], C["640"][unit], C["960"][unit]]) - 0.0
            incs = {name: t93.prepare(P[m], gt, lvl_cost) for name, P in preds.items()}
            for f in ALPHAS:
                budget = C["cheap"][unit] + f * C["640"][unit]
                o_or, _, lp = t93.greedy(incs["oracle"], n * f * C["640"][unit], w1)
                o_t = np.array([t93.greedy(incs["oracle"], w.sum() * f * C["640"][unit], w)[0] for w in wdraw])
                ok = o_t > max(EPS, 0.25 * o_or)
                key = dict(geometry=geom, system="brake", unit=unit, budget_level=f, budget_per_frame=budget, n_frames=n)
                for name in preds:
                    for charged in ((False, True) if name != "oracle" else (False,)):
                        o = 0.0
                        if charged:
                            o_ms, o_mj, _ = t93.signal_overhead(ov, "nuScenes", name, n_models=3)
                            o = o_ms if unit == "ms" else o_mj
                        extra = max(budget - C["cheap"][unit] - o, 0.0)
                        g, level, _ = t93.greedy(incs[name], n * extra, w1)
                        e_t = np.array([t93.greedy(incs[name], w.sum() * extra, w)[0] for w in wdraw])
                        e_t = np.where(ok, e_t / np.where(ok, o_t, 1), np.nan)
                        lo, hi = xl.ci(e_t)
                        rows.append({**key, "signal": f"{name} (multi-fidelity)", "overhead_charged": charged,
                                     "overhead": o, "eta_vs_multifidelity_oracle": g / o_or if o_or > EPS else np.nan,
                                     "eta_lo": lo, "eta_hi": hi, "gain": g, "oracle_gain": o_or,
                                     "share_480": float((level == 1).mean()), "share_640": float((level == 2).mean()),
                                     "share_960": float((level == 3).mean())})
                for name, sc in (("random", None), ("uncertainty", unc), ("gate_gbm", preds["gate_gbm"][m, 1]),
                                 ("oracle", gt[:, 1])):
                    frac = f
                    k = int(np.floor(frac * n + 1e-9))
                    g = (k / n * gt[:, 1].sum() if sc is None else
                         float(xl.gains(sc, gt[:, 1], [max(k, 1)])[0]) if k > 0 else 0.0)
                    rows.append({**key, "signal": f"{name} (640 only)", "overhead_charged": False, "overhead": 0.0,
                                 "eta_vs_multifidelity_oracle": g / o_or if o_or > EPS else np.nan, "gain": g,
                                 "oracle_gain": o_or, "share_640": k / n})
            print(f"  {geom} {unit} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(xl.OUT / "x5_multifidelity.csv", index=False)
    out.to_csv(run / "x5_multifidelity.csv", index=False)
    print(out[(out.budget_level == 0.2)][["geometry", "unit", "signal", "overhead_charged",
                                          "eta_vs_multifidelity_oracle"]].to_string(index=False))


if __name__ == "__main__":
    main()
