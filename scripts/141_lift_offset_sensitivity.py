#!/usr/bin/env python
"""Task 22 Part C: the five quantities with and without the camera-to-ego translation, and the reading.

Pre-registered in the pre-registration record (not part of this release) (Task 22 Part C).  Reads the per-frame tables of `140_lift_offset_outcomes.py`
and computes, per setting and over all units, the definitions of `102_calibration_cells.py`:

  affected   inputs with |V| > 1e-9
  harmed     share of affected inputs with V < 0
  rho        sum of |V| over V < 0 divided by sum of V over V > 0
  all_full   sum of V as a share of the all-cheap loss
  oracle20   the exact top-20% expectation of V as a share of the all-cheap loss

Sanity gate, before any reading: with the flag **off**, every setting reproduces the shipped values to 3 decimals,
from `calibration_cells.csv` (scheme S0, all units).  A failure stops the run and writes only the sanity table.

Registered reading: "robust" if `harmed` moves by at most 5 percentage points and `rho` by at most 0.05 in absolute
value in every setting; otherwise the settings that move more are listed.

Writes results/final/lift_offset_sensitivity.csv.  No official output is changed.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

_s = importlib.util.spec_from_file_location("t92", ROOT / "scripts" / "92_benchmark_table.py")
t92 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(t92)

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
EPS = 1e-9
SYSTEMS = {"q_brake": ("J_cheap", "J_full"), "q_traj": ("JB_cheap", "JB_full")}
CELL = {("Y8_320", "mono"): "KITTI mono Y8 320->640", ("Y8_384", "mono"): "KITTI mono Y8 384->640",
        ("Y8_512", "mono"): "KITTI mono Y8 512->640", ("RT_480", "mono"): "KITTI mono RT-DETR 480->640",
        ("Y8_320", "oracle"): "KITTI oracle Y8 320->640", ("NS_320", "mono"): "nuScenes mono Y8 320->640",
        ("NS_320", "oracle"): "nuScenes oracle Y8 320->640",
        ("RT_320", "mono"): "KITTI mono RT-DETR 320->640"}
# The one setting of the sign-variation tables with no calibration row: its gate reads the shipped per-frame tables.
SHIPPED = {("RT_320", "mono"): (str(rap_runs.core_matrix("postreview") / "KITTI__RT-DETR-l__rt_cheap_320tort_full_640__mono.pkl"),
                                str(rap_runs.planner_b() / "planB__KITTI__RTDETRl__rt_cheap_320tort_full_640__mono.pkl"))}
LABEL = {"Y8_320": "YOLOv8s 320->640", "Y8_384": "YOLOv8s 384->640", "Y8_512": "YOLOv8s 512->640",
         "RT_320": "RT-DETR-l 320->640", "RT_480": "RT-DETR-l 480->640", "NS_320": "YOLOv8s 320->640"}
HARM_TOL, RHO_TOL = 0.05, 0.05
# (base flag, alternative flag) as 140 names its tables in each frame.  Camera frame: Task 22 Part C, base = the
# camera-frame lift, alternative = the translation-only shift.  Ego frame (Task 23, C27 restated): base = the ego-frame
# lift now shipped, alternative = the camera frame, so each delta is the cost of the old convention.
FLAGS = {"camera": ("off", "on"), "ego": ("ego", "camera")}
QPLAN_COST = ("nuScenes q_plan is Planner C's ADE/FDE against the real trajectory and needs the submissions and "
              "rasters rebuilt: submissions ~15 min, Planner C oracle + mono ~2.6 h in 12 chunks, then "
              "62_planning_metric_eta.py for both variants ~1.1 h (measured 33 + 32 min on 2026-09-20) -- about "
              "4 hours, or about 9 hours if the PKL/TIP gains are refreshed too (24 chunks, ~5 h). Not run.")


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


def latest(tag):
    """The newest run of `tag` in the current lift frame (rap.runs)."""
    return rap_runs.latest(tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outcomes", default=None)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run_in = Path(args.outcomes) if args.outcomes else latest("lift_offset_outcomes")
    run = runmeta.new_run("lift_offset_sensitivity", {"outcomes_run": run_in.name})

    stats = {}
    for f in sorted(run_in.glob("outcomes__*.csv.gz")):
        _, pair, geo, flag = f.name[: -len(".csv.gz")].split("__")
        t = pd.read_csv(f)
        for sysname, (c, fcol) in SYSTEMS.items():
            if t[c].isna().all():
                continue
            stats[(pair, geo, sysname, flag)] = {**five(t[c], t[fcol]), "n_frames": len(t)}

    base, alt = FLAGS[frames.current()]
    # ---- sanity gate: the base convention against the shipped values, 3 decimals
    calib = pd.read_csv(FINAL / "calibration_cells.csv", low_memory=False)
    calib = calib[(calib.scheme == "S0") & (calib.split == "all")]
    san = []
    for (pair, geo, sysname, flag), st in sorted(stats.items()):
        if flag != base:
            continue
        r = calib[calib.cell == f"{CELL.get((pair, geo), '')} {sysname}"]
        if len(r) == 1:
            r = r.iloc[0]
            ref = {"affected": r.affected, "harmed": r.harm_rate, "rho": r.rho,
                   "all_full": r.all_full_reduction, "oracle20": r.oracle20_reduction}
            source = "calibration_cells.csv S0 all"
        elif (pair, geo) in SHIPPED:
            core, planb = SHIPPED[(pair, geo)]
            a = pd.read_pickle(RAW / core)
            b = pd.read_pickle(RAW / planb)
            t = a[["seq", "frame", "J_cheap", "J_full"]].merge(
                b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], validate="one_to_one")
            c, fcol = SYSTEMS[sysname]
            ref = five(t[c], t[fcol])
            source = "the shipped per-frame tables"
        else:
            continue
        row = {"setting": f"{pair} {geo} {sysname}", "source": source}
        ok = True
        for k, v in ref.items():
            got = st[k]
            same = (int(got) == int(v)) if k == "affected" else bool(abs(float(got) - float(v)) < 5e-4)
            row[f"{k}_shipped"], row[f"{k}_recomputed"], row[f"{k}_ok"] = v, got, same
            ok &= same
        row["ok"] = ok
        san.append(row)
    s = pd.DataFrame(san)
    s.to_csv(run / "sanity.csv", index=False)
    print(f"  sanity: {int(s.ok.sum())}/{len(s)} settings reproduce the shipped values to 3 decimals", flush=True)
    if not bool(s.ok.all()):
        print(s[~s.ok].to_string(index=False))
        raise SystemExit("the sanity gate failed: see sanity.csv")

    # ---- the sensitivity table
    rows = []
    for (pair, geo, sysname, flag) in sorted(stats):
        if flag != base:
            continue
        a, b = stats[(pair, geo, sysname, base)], stats[(pair, geo, sysname, alt)]
        row = {"dataset": "nuScenes" if pair.startswith("NS") else "KITTI", "pair": LABEL[pair],
               "geometry": geo, "system": sysname, "cell": CELL.get((pair, geo), f"{pair} {geo}"),
               "n_frames": a["n_frames"]}
        for k in ("affected", "harmed", "rho", "all_full", "oracle20"):
            row[f"{k}_{base}"], row[f"{k}_{alt}"] = a[k], b[k]
            row[f"{k}_delta"] = b[k] - a[k]
        row["within_tolerance"] = bool(abs(row["harmed_delta"]) <= HARM_TOL and abs(row["rho_delta"]) <= RHO_TOL)
        rows.append(row)
    d = pd.DataFrame(rows)
    moved = d[~d.within_tolerance]
    reading = "robust" if len(moved) == 0 else "sensitive in " + "; ".join(
        f"{r.cell} {r.system} (harmed {r.harmed_delta:+.3f}, rho {r.rho_delta:+.3f})" for r in moved.itertuples())
    d.to_csv(FINAL / "lift_offset_sensitivity.csv", index=False)
    d.to_csv(run / "lift_offset_sensitivity.csv", index=False)
    (run / "reading.json").write_text(json.dumps(
        {"reading": reading, "tolerance": {"harmed_points": HARM_TOL, "rho_abs": RHO_TOL},
         "settings": len(d), "settings_outside_tolerance": len(moved), "q_plan": QPLAN_COST,
         "outcomes_run": run_in.name}, indent=1))
    pd.set_option("display.width", 200)
    print(d[["cell", "system", f"harmed_{base}", f"harmed_{alt}", "harmed_delta", f"rho_{base}", f"rho_{alt}",
             "rho_delta"]].round(4).to_string(index=False))
    print(f"\n  reading: {reading}")
    print(f"  q_plan: {QPLAN_COST}")
    print(f"  wrote {FINAL / 'lift_offset_sensitivity.csv'} ({len(d)} settings)")


if __name__ == "__main__":
    main()
