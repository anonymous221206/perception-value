#!/usr/bin/env python
"""Follow-up E5 / X5: measured cost of every engine, with the frozen track's timing boundary and energy convention.

Each detector is profiled by 01_profile_jetson.py exactly as the frozen nuScenes profile was run (TensorRT, real
CAM_FRONT frames, 250 repetitions, 40 warm-up, 3 interleaved rounds): ms = the end-to-end median (image decoding
included), mJ = module rails (GPU + SOC + CPU) over idle x the end-to-end median (133's `profile_energy`).
Writes results/followup/E5/x7_costs.csv. Protocol: docs/followup_protocol.md, Sections 7-8.
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, subprocess, sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.detect import MODES                                                    # noqa: E402
from rap.paths import MODELS, NUSCENES_TRAINVAL, RESULTS                           # noqa: E402

RAW = ROOT / "results" / "raw"
OUT = Path(RESULTS) / "followup" / "E5"
IMAGES = str(NUSCENES_TRAINVAL / "samples" / "CAM_FRONT")
# (weights, engine stem or None, modes)
DEFAULT = [("yolov8s.pt", None, ["ns_cheap_320", "ns_mid_480", "ns_full_640", "ns_full_960"]),
           ("yolov8m.pt", None, ["ns_full_640"]), ("yolov8l.pt", None, ["ns_full_640"]),
           ("yolov8x.pt", None, ["ns_full_640"]), ("yolo11s.pt", None, ["ns_full_640"]),
           ("yolo11m.pt", None, ["ns_full_640"]), ("rtdetr-l.pt", "rtdetrl", ["rt_full_640"])]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default=None, help="JSON list of [weights, engine_stem, [modes]] (default: E5's set)")
    ap.add_argument("--out_csv", default=str(OUT / "x7_costs.csv"))
    args = ap.parse_args()
    specs = json.loads(args.specs) if args.specs else DEFAULT
    t133 = _load("t133", "133_energy_module_budgets.py")
    rows = []
    for weights, stem, modes in specs:
        before = set(glob.glob(str(RAW / "*_profile")))
        cmd = [str(ROOT / "scripts" / "py"), str(ROOT / "scripts" / "01_profile_jetson.py"), "--backend", "trt",
               "--weights", str(MODELS / weights), "--images", IMAGES, "--reps", "250", "--warmup", "40",
               "--rounds", "3", "--modes", *modes] + (["--engine_stem", stem] if stem else [])
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        run = Path(sorted(set(glob.glob(str(RAW / "*_profile"))) - before)[-1])
        summ = json.loads((run / "profile_summary.json").read_text())
        for m in modes:
            rows.append({"engine": f"{stem or Path(weights).stem}_{m}", "weights": weights, "mode": m,
                         "input_hw": f"{MODES[m].net_h}x{MODES[m].net_w}",
                         "ms": float(summ[m]["lat_e2e_ms_median"]),
                         "mJ_module": t133.profile_energy(run.name, m, t133.MODULE),
                         "mJ_gpu": float(summ[m]["energy_gpu_mj_per_frame"]), "run": run.name})
            print(rows[-1], flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("wrote", args.out_csv)


if __name__ == "__main__":
    main()
