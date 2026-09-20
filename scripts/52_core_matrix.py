#!/usr/bin/env python
"""Phase 0E Stages 9-17: the core matrix across every configuration.

One row per (dataset, detector, fidelity pair, geometry, task). Everything the final
verdict rests on is computed here so the whole claim can be audited from one CSV.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap.paths import DATASETS as _DS, MODELS as _MD                      # noqa: E402
from rap import budget, decision, egospeed, geometry as G, planner as P, predict, runmeta  # noqa: E402
from rap import percep_metrics as PM                                             # noqa: E402
from rap.cache import DetCache                                                   # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import CACHE, RESULTS                                             # noqa: E402
from rap.risk import RiskConfig                                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module                                              # noqa: E402
_pm = import_module("50_percep_metrics")

QUOTAS = [0.10, 0.20, 0.30, 0.50]
TASKS = {"longitudinal": ("J_cheap", "J_full"), "lateral": ("Jlat_cheap", "Jlat_full")}
NOCHK = lambda c: None


TIE_SEEDS = 8


def eta(df, score, col, quota=0.20):
    """Share of the oracle's achievable reduction captured, with ties averaged out.

    Count-valued signals such as an unweighted dE tie on most of the selected set at a 20%
    quota, so a single tie-break -- random or not -- is one draw from a wide distribution.
    """
    def pick(s, seed):
        return budget.select_pooled(np.asarray(s, float), quota, seed=seed)

    allc = float(df[col[0]].sum())
    dj = (df[col[0]] - df[col[1]]).to_numpy()
    sc = np.asarray(score, float)
    orc = float(np.mean([budget.total_risk(df, pick(dj, sd), col) for sd in range(TIE_SEEDS)]))
    tot = float(np.mean([budget.total_risk(df, pick(sc, sd), col) for sd in range(TIE_SEEDS)]))
    return (allc - tot) / (allc - orc) if allc - orc > 1e-12 else np.nan


def topk_overlap(a, b, frac):
    k = int(round(frac * len(a)))
    ia = set(np.argsort(-np.asarray(a), kind="stable")[:k])
    ib = set(np.argsort(-np.asarray(b), kind="stable")[:k])
    return len(ia & ib) / max(k, 1)


def trivial_heuristics(d):
    z = d.feat_min_z_corridor.to_numpy() if "feat_min_z_corridor" in d else np.zeros(len(d))
    v = d[egospeed.COL].to_numpy(float)          # the speed an allocator could compute (Task 22 Part B)
    return {
        "ego speed": v,
        "empty-frame": d.empty_cheap.to_numpy().astype(float),
        "few detections": -d.n_cheap.to_numpy().astype(float),
        "n detections": d.n_cheap.to_numpy().astype(float),
        "low mean confidence": -d.conf_mean.to_numpy(),
        "uncertainty": d.unc_sum.to_numpy(),
        "criticality": d.crit_sum.to_numpy(),
        "closest object": -z,
        "speed x empty": v * (1 + 3 * d.empty_cheap.to_numpy()),
        "speed + detections": stats.rankdata(v) + stats.rankdata(d.n_cheap),
        "speed x closest": v * np.clip(1 / np.maximum(z, 1.0), 0, 1),
    }


def spec_rows(d, ds, detname, cm, fm, geo, skip_multimetric=False):
    """The core-matrix rows of one (dataset, detector, pair, geometry) from its per-frame table."""
    rows = []
    d = egospeed.attach(d, ds)
    jl = (d.J_cheap - d.J_full).to_numpy()
    jt = (d.Jlat_cheap - d.Jlat_full).to_numpy()
    overlap20 = topk_overlap(jl, jt, 0.20)
    heur = trivial_heuristics(d)

    for tname, col in TASKS.items():
        dj = (d[col[0]] - d[col[1]]).to_numpy()
        act = "same_action" if tname == "longitudinal" else "lat_same_action"
        improved = d["dE"] > 0
        best_metric = max(PM.METRICS, key=lambda m: eta(d, d[f"dE_{m}"], col))
        hh = {k: eta(d, v, col) for k, v in heur.items()}
        bh = max(hh, key=lambda k: hh[k])
        mm = np.nan
        if not skip_multimetric:
            feats = _pm.oracle_features(d)
            dd2 = d.copy()
            for c in feats:
                dd2[c] = pd.to_numeric(dd2[c], errors="coerce").fillna(0.0)
            dd2["_dJ"] = dj
            mm = eta(dd2, predict.loso(dd2, feats, "gbm", "_dJ", "reg",
                                       checker=NOCHK).pred, col)
        r = {
            "dataset": ds, "detector": detname, "cheap_mode": cm, "full_mode": fm,
            "task": tname, "geometry": geo,
            "perception_metric": "E1_fn_only", "best_perception_metric": best_metric,
            "n_sequences": int(d.seq.nunique()), "n_frames": len(d),
            "action_change_rate": float((d[act] == 0).mean()),
            "improved_same_action_rate": float((improved & (d[act] == 1)).sum() /
                                               max(improved.sum(), 1)),
            "harmful_full_rate": float((dj < -1e-9).mean()),
            "corr_deltaE_deltaJ": float(stats.spearmanr(d["dE"], dj).correlation),
            "eta_random_20": float(np.mean([eta(d, np.random.default_rng(k).random(len(d)), col)
                                            for k in range(6)])),
            "eta_uncertainty_20": eta(d, d.unc_sum, col),
            "eta_criticality_20": eta(d, d.crit_sum, col),
            "eta_perception_oracle_20": eta(d, d["dE"], col),
            "eta_best_perception_metric_20": eta(d, d[f"dE_{best_metric}"], col),
            "eta_multimetric_perception_oracle_20": mm,
            "eta_decision_oracle_20": 1.0,
            "top20_cross_task_overlap": overlap20,
            "best_trivial_heuristic": bh, "best_trivial_eta20": hh[bh],
        }
        for q in QUOTAS:
            r[f"eta_perception_oracle_{int(q*100)}"] = eta(d, d["dE"], col, q)
        rows.append(r)
        print(f"   {tname:13s} corr={r['corr_deltaE_deltaJ']:+.3f} "
              f"etaE={r['eta_perception_oracle_20']:+.3f} "
              f"multi={mm if np.isnan(mm) else round(mm,3)} "
              f"best-trivial={bh} {hh[bh]:+.3f} overlap20={overlap20:.3f}")
    return rows


def rows_only(args, run):
    """Recompute the rows of some datasets from stored per-frame tables, and keep the rest of the shipped CSV.

    Task 22 Part A: only the nuScenes class error changed, so the decision tables are not rebuilt and the rows of the
    other datasets are carried over unchanged.
    """
    tables = Path(args.rows_only)
    want = set(args.datasets or [])
    rows = []
    for f in sorted(tables.glob("*.pkl")):
        ds, detname, pair, geo = f.stem.split("__")
        if want and ds not in want:
            continue
        cm, fm = pair.split("to", 1)
        d = pd.read_pickle(f)
        rows += spec_rows(d, ds, detname, cm, fm, geo, args.skip_multimetric)
        print(f"  rows {f.stem}: {len(d)} frames", flush=True)
    m = pd.DataFrame(rows)
    out = Path(RESULTS) / "final"
    if args.merge_into:
        # The rows of the datasets that were not recomputed are spliced in as text, character for character.  A
        # read_csv/to_csv round trip drops the 17th significant digit of a float64, which would make every untouched
        # row differ although no value moved; C4 asks for byte identity, so the old lines are carried, not re-emitted.
        lines = Path(args.merge_into).read_text().splitlines()
        header, cols = lines[0], lines[0].split(",")
        ki = [cols.index(k) for k in ("dataset", "detector", "cheap_mode", "full_mode", "task", "geometry")]
        kof = lambda line: tuple(next(csv.reader([line]))[i] for i in ki)
        fresh = {kof(l): l for l in m[cols].to_csv(index=False).splitlines()[1:]}
        body, used = [], set()
        for l in lines[1:]:
            k = kof(l)
            body.append(fresh[k] if k in fresh else l)
            used.add(k) if k in fresh else None
        body += [l for k, l in fresh.items() if k not in used]
        text = "\n".join([header] + body) + "\n"
        (out / "core_matrix.csv").write_text(text)
        (run / "core_matrix.csv").write_text(text)
        n = len(body)
    else:
        m.to_csv(out / "core_matrix.csv", index=False)
        m.to_csv(run / "core_matrix.csv", index=False)
        n = len(m)
    print(f"  wrote {out / 'core_matrix.csv'} ({n} rows; {len(rows)} recomputed)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="core_matrix")
    ap.add_argument("--skip_multimetric", action="store_true")
    ap.add_argument("--rows_only", default=None,
                    help="a run directory of per-frame tables: recompute only the row statistics from them")
    ap.add_argument("--datasets", nargs="+", default=None, help="with --rows_only: which datasets to recompute")
    ap.add_argument("--merge_into", default=None,
                    help="with --rows_only: keep every row of this CSV whose dataset is not recomputed")
    egospeed.add_argument(ap)
    args = ap.parse_args()
    egospeed.configure(args)
    run = runmeta.new_run(args.tag, vars(args))
    if args.rows_only:
        return rows_only(args, run)
    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()

    kseqs = [p.stem for p in sorted((Path(CACHE / "det") / "cheap_320").glob("*.npz"))]
    db = NuScenesDB(str(_DS / "nuscenes/trainval"), "v1.0-trainval")
    nad = make_adapter(db)
    nseqs = [p.stem for p in sorted((Path(CACHE / "nusc_det_tv") / "ns_cheap_320").glob("*.npz"))]

    SPECS = [
        ("KITTI", "YOLOv8s", "cheap_320", "full_640", CACHE / "det", kseqs,
         decision.KittiAdapter, "mono"),
        ("KITTI", "YOLOv8s", "cheap_320", "full_640", CACHE / "det", kseqs,
         decision.KittiAdapter, "oracle"),
        ("KITTI", "YOLOv8s", "cheap_384", "full_640", CACHE / "det", kseqs,
         decision.KittiAdapter, "mono"),
        ("KITTI", "YOLOv8s", "cheap_512", "full_640", CACHE / "det512", kseqs,
         decision.KittiAdapter, "mono"),
        ("KITTI", "RT-DETR-l", "rt_cheap_320", "rt_full_640", CACHE / "rtdetr_kitti",
         kseqs, decision.KittiAdapter, "mono"),
        ("KITTI", "RT-DETR-l", "rt_mid_480", "rt_full_640", CACHE / "rtdetr_kitti_mid",
         kseqs, decision.KittiAdapter, "mono"),
        ("nuScenes", "YOLOv8s", "ns_cheap_320", "ns_full_640", CACHE / "nusc_det_tv",
         nseqs, nad, "mono"),
        ("nuScenes", "YOLOv8s", "ns_cheap_320", "ns_full_640", CACHE / "nusc_det_tv",
         nseqs, nad, "oracle"),
    ]

    rows, tables = [], {}
    for ds, detname, cm, fm, dd, seqs, ad, geo in SPECS:
        dd = Path(dd)
        if not (dd / cm).exists():
            print(f"skip {ds}/{detname}/{cm}: no cache"); continue
        key = f"{ds}|{detname}|{cm}->{fm}|{geo}"
        d = decision.build(dd, cm, fm, seqs, cfg, pp, cp, G.PRIMARY,
                           range_source=geo, adapter=ad)
        d = decision.add_perception_gain(d, dd, cm, fm, seqs, cfg, adapter=ad)
        prim = _pm.primitives(dd, cm, fm, seqs, cfg, ad, cfg.min_gt_height)
        d = d.merge(prim, on=["seq", "frame"], how="left", validate="one_to_one")
        tables[key] = d
        print(f"{key}: {len(d)} frames")

        rows += spec_rows(d, ds, detname, cm, fm, geo, args.skip_multimetric)

    out = Path(RESULTS) / "final"
    out.mkdir(parents=True, exist_ok=True)
    m = pd.DataFrame(rows)
    m.to_csv(out / "core_matrix.csv", index=False)
    m.to_csv(run / "core_matrix.csv", index=False)
    for k, v in tables.items():
        v.to_pickle(run / (k.replace("|", "__").replace("->", "to").replace("/", "_") + ".pkl"))
    print(f"\nwrote {out/'core_matrix.csv'} ({len(m)} rows)")
    print("\nwrote", run)


if __name__ == "__main__":
    main()
