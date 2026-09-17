#!/usr/bin/env python
"""Task 21: two realism controls on the KITTI core track -- detection persistence and reference geometry.

Pre-registered in the pre-registration record (not part of this release), Task 21.  Reads the per-frame tables of
`134_realism_outcomes.py` (the latest results/raw/*_realism_outcomes run) and computes, per setting and over all units:

  affected   inputs with |V| > 1e-9
  harmed     share of affected inputs with V < 0
  rho        sum of |V| over V < 0 divided by sum of V over V > 0
  all_full   sum of V as a share of the all-cheap loss
  oracle20   the exact top-20% expectation of V (never selecting V < 0 while non-negative inputs remain), as a share
             of the all-cheap loss -- the definitions of `102_calibration_cells.py`

Sanity gates, before any reading: every n = 1 setting reproduces the shipped values to 3 decimals, from
results/final/calibration_cells.csv (scheme S0, all units) where the setting is there, and from the shipped per-frame
tables otherwise (RT-DETR-l 320 -> 640).  If a gate fails the script stops and writes only the sanity table.

Outputs: results/final/persistence_sweep.csv, results/final/reference_geometry_sweep.csv
"""
from __future__ import annotations

import importlib.util, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

_s = importlib.util.spec_from_file_location("t92", ROOT / "scripts" / "92_benchmark_table.py")
t92 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(t92)

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
EPS = 1e-9
PAIR_LABEL = {"Y8_320": "YOLOv8s 320->640", "Y8_384": "YOLOv8s 384->640", "Y8_512": "YOLOv8s 512->640",
              "RT_320": "RT-DETR-l 320->640", "RT_480": "RT-DETR-l 480->640"}
CALIB_CELL = {("Y8_320", "mono"): "KITTI mono Y8 320->640", ("Y8_384", "mono"): "KITTI mono Y8 384->640",
              ("Y8_512", "mono"): "KITTI mono Y8 512->640", ("RT_480", "mono"): "KITTI mono RT-DETR 480->640",
              ("Y8_320", "oracle"): "KITTI oracle Y8 320->640"}
SHIPPED_TABLES = {"RT_320": ("20260913_133004_core_matrix_postreview/KITTI__RT-DETR-l__rt_cheap_320tort_full_640__mono.pkl",
                             "20260912_111225_planner_b_static_fixed/planB__KITTI__RTDETRl__rt_cheap_320tort_full_640__mono.pkl")}
SYSTEMS = {"q_brake": ("J_cheap", "J_full"), "q_traj": ("JB_cheap", "JB_full")}
QUANT = ("affected", "harmed", "rho", "all_full", "oracle20")


def _latest(tag):
    d = [p for p in sorted(RAW.glob(f"*_{tag}")) if p.name.split("_", 2)[-1] == tag]
    if not d:
        raise SystemExit(f"no results/raw/*_{tag} run found")
    return d[-1]


def five(jc, jf):
    jc, jf = np.asarray(jc, float), np.asarray(jf, float)
    v = jc - jf
    aff, neg, pos = np.abs(v) > EPS, v < -EPS, v > EPS
    tot = float(jc.sum())
    k20 = max(int(round(0.2 * len(v))), 1)
    prize = float(t92.topk_expect(v, [v], [k20])[0][0][0])
    return {"affected": int(aff.sum()), "harmed": float(neg.sum() / aff.sum()) if aff.any() else np.nan,
            "rho": float(-v[neg].sum() / v[pos].sum()) if pos.any() else np.nan,
            "all_full": float(v.sum() / tot) if tot > EPS else np.nan,
            "oracle20": float(prize / tot) if tot > EPS else np.nan}


def reading_A(base, n3):
    h, r = n3["harmed"] / base["harmed"], n3["rho"] / base["rho"]
    if h <= 0.5 and r <= 0.5:
        return "removes"
    if h <= 2 / 3 or r <= 2 / 3:
        return "mitigates"
    return "no material change"


def reading_B(mono, ref):
    return ("geometry-driven" if ref["harmed"] <= 0.5 * mono["harmed"] and ref["rho"] <= 0.5 * mono["rho"]
            else "persists without lifting error")


def majority(labels):
    s = pd.Series(labels).value_counts()
    return s.index[0] if s.iloc[0] > len(labels) / 2 else "no majority (" + ", ".join(f"{k} {v}" for k, v in s.items()) + ")"


def main():
    run_in = _latest("realism_outcomes")
    run = runmeta.new_run("realism_controls", {"outcomes_run": run_in.name})
    tabs = {}
    for f in sorted(run_in.glob("outcomes__*.csv.gz")):
        _, pair, geo, n = f.name[: -len(".csv.gz")].split("__")
        tabs[(pair, geo, int(n[1:]))] = pd.read_csv(f)
    stats = {}
    for (pair, geo, n), t in tabs.items():
        for sysname, (c, fcol) in SYSTEMS.items():
            stats[(pair, geo, n, sysname)] = {**five(t[c], t[fcol]), "n_frames": len(t),
                                             "dets_per_frame_cheap": float(t.n_cheap.mean()),
                                             "dets_per_frame_full": float(t.n_full.mean())}

    # ---- sanity gates: n = 1 against the shipped values, 3 decimals
    calib = pd.read_csv(FINAL / "calibration_cells.csv", low_memory=False)
    calib = calib[(calib.scheme == "S0") & (calib.split == "all")]
    san = []
    for (pair, geo, n, sysname), st in stats.items():
        if n != 1:
            continue
        if (pair, geo) in CALIB_CELL:
            r = calib[calib.cell == f"{CALIB_CELL[(pair, geo)]} {sysname}"]
            assert len(r) == 1, (pair, geo, sysname)
            r = r.iloc[0]
            ref = {"affected": r.affected, "harmed": r.harm_rate, "rho": r.rho, "all_full": r.all_full_reduction,
                   "oracle20": r.oracle20_reduction}
            src = "calibration_cells.csv S0 all"
        elif geo == "mono" and pair in SHIPPED_TABLES:
            core, planb = SHIPPED_TABLES[pair]
            a = pd.read_pickle(RAW / core)
            b = pd.read_pickle(RAW / planb)
            cols = SYSTEMS[sysname]
            x = a if sysname == "q_brake" else b
            ref = five(x[cols[0]], x[cols[1]])
            src = "shipped per-frame tables"
        else:
            continue                                   # a new setting (Part B): nothing shipped to reproduce
        ok = all((st[q] == ref[q]) if q == "affected" else (round(st[q], 3) == round(float(ref[q]), 3)) for q in QUANT)
        san.append({"section": "sanity", "pair": PAIR_LABEL[pair], "geometry": geo, "system": sysname, "n": 1,
                    "reference": src, **{q: st[q] for q in QUANT}, **{f"shipped_{q}": ref[q] for q in QUANT},
                    "reproduced_3dp": bool(ok)})
    sanity = pd.DataFrame(san)
    sanity.to_csv(run / "realism_sanity.csv", index=False)
    print(sanity[["pair", "geometry", "system", "affected", "shipped_affected", "harmed", "shipped_harmed", "rho",
                  "shipped_rho", "reproduced_3dp"]].to_string(index=False))
    if not sanity.reproduced_3dp.all():
        raise SystemExit("sanity gate failed: n = 1 does not reproduce the shipped values; stopping before any reading")

    # ---- A: persistence
    rows, labels = [], []
    for (pair, geo, n, sysname), st in sorted(stats.items()):
        if (pair, geo) not in (tuple((p, "mono") for p in PAIR_LABEL) + (("Y8_320", "oracle"),)):
            continue
        base = stats[(pair, geo, 1, sysname)]
        row = {"pair": PAIR_LABEL[pair], "geometry": geo, "system": sysname, "n": n, "latency_added_s": 0.1 * (n - 1),
               **st, "harmed_ratio_to_n1": st["harmed"] / base["harmed"], "rho_ratio_to_n1": st["rho"] / base["rho"]}
        if n == 3:
            row["reading"] = reading_A(base, st)
            labels.append(row["reading"])
        rows.append(row)
    pers = pd.DataFrame(rows)
    pers = pd.concat([pers, pd.DataFrame([{"pair": "summary over settings", "geometry": "all", "system": "all", "n": 3,
                                           "reading": majority(labels), "settings": len(labels)}]),
                      sanity.assign(reading=np.nan)], ignore_index=True)

    # ---- B: reference geometry for every pair
    rows = {"q_brake": [], "q_traj": []}
    brows = []
    for pair in PAIR_LABEL:
        for sysname in SYSTEMS:
            mono, ref = stats[(pair, "mono", 1, sysname)], stats[(pair, "oracle", 1, sysname)]
            lab = reading_B(mono, ref)
            rows[sysname].append(lab)
            brows.append({"pair": PAIR_LABEL[pair], "system": sysname,
                          **{f"mono_{q}": mono[q] for q in QUANT}, **{f"reference_{q}": ref[q] for q in QUANT},
                          "harmed_ratio": ref["harmed"] / mono["harmed"], "rho_ratio": ref["rho"] / mono["rho"],
                          "reading": lab})
    for sysname, labs in rows.items():
        brows.append({"pair": "summary over pairs", "system": sysname, "reading": majority(labs), "pairs": len(labs)})
    refg = pd.concat([pd.DataFrame(brows), sanity[sanity.geometry == "oracle"].assign(reading=np.nan)], ignore_index=True)

    for name, df in (("persistence_sweep.csv", pers), ("reference_geometry_sweep.csv", refg)):
        df.to_csv(FINAL / name, index=False)
        df.to_csv(run / name, index=False)
    print(pers[pers.n == 3][["pair", "geometry", "system", "harmed", "rho", "harmed_ratio_to_n1", "rho_ratio_to_n1",
                             "reading"]].to_string(index=False))
    print(refg[["pair", "system", "harmed_ratio", "rho_ratio", "reading"]].to_string(index=False))


if __name__ == "__main__":
    main()
