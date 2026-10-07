#!/usr/bin/env python
"""Follow-up E1: setting-level quantities (P1, X1), heterogeneity, concentration, the resolution ladder, gate (b).

From 170's tables over all 850 scenes. Populations: the 228 new test scenes, all 765 new scenes, the frozen 85 scenes,
the pooled 850; each quantity with a 95% bootstrap interval by scene and by log (1,000 draws, seed 0, every draw kept).
Writes results/followup/E1/x1_settings.csv, x1_heterogeneity.csv, x1_concentration.csv, and gate (b) into
gates.json. Protocol: docs/followup_protocol.md, Sections 5-7.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, runmeta, xl                                              # noqa: E402
from rap.scoring import EPS                                                      # noqa: E402
from rap.paths import NUSCENES_XL                                                  # noqa: E402

PAIRS = {"ns_cheap_320tons_full_640": "320→640", "ns_cheap_320tons_mid_480": "320→480",
         "ns_cheap_320tons_full_960": "320→960"}
EXPLORATORY_UNIFORM_640 = -0.181      # Section 2, item 1 of the protocol (informational gate (b))


def populations(sp):
    new, old = sp[sp.group == "new765"], sp[sp.group == "frozen85"]
    return {"new228_test": set(new[new.deep_split == "test"].scene), "new765": set(new.scene),
            "frozen85": set(old.scene), "pooled850": set(sp.scene)}


def one(d, cheap, full, log_of, nboot=1000):
    v = (d[cheap] - d[full]).to_numpy(float)
    c = d[cheap].to_numpy(float)
    r = xl.describe(v, c, (0.2,))
    k20 = max(int(round(0.2 * len(v))), 1)
    r["budget_binding"] = bool((v > EPS).sum() > k20)
    r["n_units"] = int(d.seq.nunique())
    r["n_logs"] = int(d.seq.map(log_of).nunique())
    r.update(xl.bootstrap_describe(v, c, d.seq.to_numpy(), nboot, np.random.default_rng(0), (0.2,)))
    lg = xl.bootstrap_describe(v, c, d.seq.map(log_of).to_numpy(), nboot, np.random.default_rng(0), (0.2,))
    r.update({k.replace("_lo", "_lo_log").replace("_hi", "_hi_log"): x for k, x in lg.items()})
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True)
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run("xl_settings", vars(args))
    sp = xl.scene_split()
    meta = xl.scene_meta(args.dataroot)
    sp = sp.merge(meta, on="scene", how="left", validate="one_to_one")
    log_of = dict(zip(sp.scene, sp.log))
    pops = populations(sp)
    offval = set(sp[sp.official == "val"].scene)
    tables = Path(args.tables)

    rows = []
    for pair, label in PAIRS.items():
        for geom in xl.GEOMETRIES:
            f = tables / f"{geom}__{pair}.pkl"
            if not f.exists():
                print("  missing", f.name)
                continue
            t = pd.read_pickle(f)
            for system, cc, cf in xl.SYSTEMS:
                if cc not in t:
                    continue
                for pop, scenes in pops.items():
                    variants = [(pop, scenes)]
                    if system.startswith("plan_"):
                        variants.append((pop + "_official_val", scenes & offval))
                    for name, sc in variants:
                        d = t[t.seq.isin(sc) & t[cc].notna()].reset_index(drop=True)
                        if d.empty:
                            continue
                        r = one(d, cc, cf, log_of, args.nboot)
                        rows.append({"population": name, "geometry": geom, "system": system,
                                     "target": cc.replace("_cheap", ""), "pair": label,
                                     "primary": label == "320→640" and not name.endswith("_official_val"), **r})
                        print(f"  {label} {geom:6s} {system:8s} {name:28s} aff {r['affected']:5d} harmed "
                              f"{r['harmed_share']:.3f} [{r['harmed_share_lo']:.3f}, {r['harmed_share_hi']:.3f}] "
                              f"all-full {r['all_full_reduction']:+.4f}", flush=True)
    x1 = pd.DataFrame(rows)
    x1.to_csv(xl.OUT / "x1_settings.csv", index=False)
    x1.to_csv(run / "x1_settings.csv", index=False)

    # heterogeneity and concentration: braking, both geometries, new and frozen scenes, the primary pair
    het, conc = [], []
    v_cut = None
    for geom in xl.GEOMETRIES:
        t = pd.read_pickle(tables / f"{geom}__{xl.PAIR}.pkl").merge(
            sp[["scene", "location", "deep_split", "night", "rain", "calibration", "group"]].rename(columns={"scene": "seq"}),
            on="seq", how="left", validate="many_to_one")
        if v_cut is None:      # ego-speed terciles of the pooled 850 scenes' frames (the controller's v_ego)
            v_cut = np.quantile(t.v_ego.to_numpy(float), [1 / 3, 2 / 3])
        t["speed_tercile"] = np.digitize(t.v_ego.to_numpy(float), v_cut)
        for grp, gname in (("new765", "new765"), ("frozen85", "frozen85")):
            g = t[t.group == grp]
            for by in ("location", "deep_split", "speed_tercile", "night", "rain", "calibration"):
                for val, d in g.groupby(by):
                    d = d.reset_index(drop=True)
                    r = one(d, "J_cheap", "J_full", log_of, args.nboot)
                    het.append({"group": gname, "geometry": geom, "system": "brake", "by": by, "value": val, **r})
            # concentration over scenes
            v = (g.J_cheap - g.J_full).to_numpy(float)
            per = pd.DataFrame({"seq": g.seq.to_numpy(), "b": np.maximum(v, 0), "h": np.maximum(-v, 0)}).groupby("seq").sum()
            for frac in (0.05, 0.10):
                k = max(int(np.ceil(frac * len(per))), 1)
                conc.append({"group": gname, "geometry": geom, "system": "brake", "top_share_of_scenes": frac,
                             "n_scenes": len(per), "k": k,
                             "benefit_share": float(per.b.nlargest(k).sum() / per.b.sum()) if per.b.sum() > EPS else np.nan,
                             "harm_share": float(per.h.nlargest(k).sum() / per.h.sum()) if per.h.sum() > EPS else np.nan})
    pd.DataFrame(het).to_csv(xl.OUT / "x1_heterogeneity.csv", index=False)
    pd.DataFrame(conc).to_csv(xl.OUT / "x1_concentration.csv", index=False)
    (run / "speed_terciles.json").write_text(json.dumps({"cut_points_mps": [float(x) for x in v_cut]}))

    # gate (b), informational
    r = x1[(x1.population == "new228_test") & (x1.geometry == "mono") & (x1.system == "brake") & (x1.pair == "320→640")]
    g = json.loads((xl.OUT / "gates.json").read_text())
    if len(r):
        r = r.iloc[0]
        b = {"status": "recorded", "informational": True,
             "this_repository": float(r.all_full_reduction),
             "interval": [float(r.all_full_reduction_lo), float(r.all_full_reduction_hi)],
             "harmed_share": float(r.harmed_share), "affected": int(r.affected),
             "exploratory_value": EXPLORATORY_UNIFORM_640,
             "difference": float(r.all_full_reduction) - EXPLORATORY_UNIFORM_640}
        old = {k: v for k, v in g.get("b", {}).items() if k != "date"}
        # the date of first recording is kept when a rerun reproduces the same values (cached-tier reruns compare files)
        date = g["b"]["date"] if old == b else pd.Timestamp.now().strftime("%Y-%m-%d")
        g["b"] = {"status": b.pop("status"), "informational": b.pop("informational"), "date": date, **b}
        (xl.OUT / "gates.json").write_text(json.dumps(g, indent=1))
        print("gate (b):", json.dumps(g["b"]))


if __name__ == "__main__":
    main()
