#!/usr/bin/env python
"""Task 25: the profiling harness for the measured-budget track -- an allocator's own cost per input, measured the way
93_budget_allocation.py measured the benchmark's allocators.

    python scripts/151_profile_allocator.py ALLOCATOR.py --tracks KITTI nuScenes --out PROFILE.json

ALLOCATOR.py defines `score_input(inp) -> float`, called once per input with exactly what docs/SUBMITTING.md allows
at test time (`rap.submission.inputs`):

  KITTI / nuScenes   inp = {"frame": {column: value} of the frames file,
                            "detections": {column: numpy array} of this frame's CHEAP detections,
                            "previous": the same two for the previous CHEAP frame of the unit, or None,
                            "calibration": the unit's calibration entry}
  nuPlan             inp = {"state": {column: value} of nuplan_states.csv.gz}

The measurement is 93's, unchanged (`_median_ms`, `_rails_during`, `signal_overhead`'s energy rule):

  latency  median single-input wall time over three passes of the first --frames test inputs of each track, inputs
           prepared before the clock starts (the benchmark times feature extraction and inference, not file reading)
  energy   latency x the power over idle of --rails (default CPU, as 93 charges its CPU workloads; add GPU for an
           allocator that runs on the GPU), with idle and busy each sampled for 8 s every 50 ms

The profile carries the output of environment/check_device.py (docs/SUBMITTING.md, measured track, route 1); a
profile measured off the reference platform is written but marked so, and is not a measured-track result.
"""
from __future__ import annotations

import argparse, importlib.util, json, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import power, submission as S                                          # noqa: E402

TRACKS = ("KITTI", "nuScenes", "nuPlan")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def prepared_inputs(track, n):
    """The first n test inputs of a track, in file order, as `score_input` receives them."""
    inp = S.inputs(track)
    if track == "nuPlan":
        st = inp["states"]
        st = st[st.split == "test"].head(n)
        return [{"state": r} for r in st.to_dict("records")]
    fr, det, cal = inp["frames"], inp["detections"], inp["calibration"]
    by = {k: {c: g[c].to_numpy() for c in g.columns} for k, g in det.groupby(["seq", "frame"], sort=False)}
    empty = {c: det[c].to_numpy()[:0] for c in det.columns}
    rows = {(r["seq"], int(r["frame"])): r for r in fr.to_dict("records")}
    out = []
    for r in fr[fr.split == "test"].head(n).to_dict("records"):
        key = (r["seq"], int(r["frame"]))
        prev = (r["seq"], int(r["prev_frame"]))
        out.append({"frame": r, "detections": by.get(key, empty),
                    "previous": ({"frame": rows[prev], "detections": by.get(prev, empty)} if prev in rows else None),
                    "calibration": cal.get(r["unit"])})
    return out


def device_check():
    f = ROOT / "environment" / "check_device.py"
    if not f.exists():
        return {"available": False, "note": "environment/check_device.py is not in this checkout"}
    p = subprocess.run([sys.executable, str(f)], capture_output=True, text=True)
    return {"available": True, "exit_code": p.returncode, "reference_platform": p.returncode == 0,
            "stdout": p.stdout, "stderr": p.stderr}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("allocator", help="a Python file defining score_input(inp) -> float")
    ap.add_argument("--tracks", nargs="+", default=["KITTI", "nuScenes"], choices=TRACKS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=400, help="inputs timed per track (93 uses 400)")
    ap.add_argument("--rails", nargs="+", default=["CPU"], help="rails whose power over idle is charged (93: CPU)")
    args = ap.parse_args()
    t93 = _load("t93", ROOT / "scripts" / "93_budget_allocation.py")
    alloc = _load("allocator", Path(args.allocator).resolve())
    check = device_check()
    prof = {"route": "harness", "harness": "scripts/151_profile_allocator.py", "allocator": Path(args.allocator).name,
            "frames": args.frames, "rails": args.rails, "power_available": power.AVAILABLE,
            "ms": {}, "mJ": {}, "power_mw_over_idle": {}, "device_check": check}
    for track in args.tracks:
        items = prepared_inputs(track, args.frames)
        prof["ms"][track] = t93._median_ms(alloc.score_input, items)
        if power.AVAILABLE:
            i = {"k": 0}

            def work():
                alloc.score_input(items[i["k"] % len(items)])
                i["k"] += 1
            idle = t93._rails_during(lambda: time.sleep(0.02), 8.0)
            busy = t93._rails_during(work, 8.0)
            mw = float(sum(busy.get(r, np.nan) - idle.get(r, np.nan) for r in args.rails))
            prof["power_mw_over_idle"][track] = mw
            prof["mJ"][track] = prof["ms"][track] * max(mw, 0.0) / 1e3 if np.isfinite(mw) else None
        else:
            prof["mJ"][track] = None                 # no rails: latency only, the energy budgets are not scored
        print(f"  {track}: {prof['ms'][track]:.4f} ms, {prof['mJ'][track]} mJ per input", flush=True)
    if not prof["mJ"] or all(v is None for v in prof["mJ"].values()):
        prof["mJ"] = None
    Path(args.out).write_text(json.dumps(prof, indent=1))
    ref = check.get("reference_platform")
    print(f"wrote {args.out}" + ("" if ref else "  (NOT measured on the reference platform: see device_check)"))


if __name__ == "__main__":
    main()
