#!/usr/bin/env python
"""Task 23: the ego-frame per-frame tables with the pre-fix nuScenes class labels, for C25 re-expressed in the ego frame.

Pre-registered in the pre-registration record (not part of this release) (Task 23: "C25 by substituting the pre-fix cheap_cls/full_cls (frame-independent,
G4) into the ego-frame tables and re-running the stages that read E3/E5").  The class error is a property of the image
matching, which the frame cannot reach, so the pre-fix `cls` columns of the camera-frame tables can be carried into
the ego-frame tables row for row.  E3 and the three E5 variants are then recombined with `percep_metrics`' weights.

  core_matrix_ego            -> core_matrix_ego_prefix             (the two nuScenes pickles; KITTI copied unchanged)
  calibration_outcomes_ego   -> calibration_outcomes_ego_prefix    (the 32 nuScenes outcome files)

Checks, each stops the run: the rows align on (seq, frame); every column other than cls and the four affected gains
is identical to the ego-frame table; recombining the ego-frame primitives reproduces the ego-frame dE_E1/E2/E4/E6.
"""
from __future__ import annotations

import argparse, json, shutil, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import percep_metrics as PM, runmeta, runs as rap_runs                 # noqa: E402

RAW = ROOT / "results" / "raw"
PREFIX_CORE = RAW / "20260912_071140_core_matrix"                 # camera frame, before the class-error fix
PREFIX_OUTCOMES = RAW / "20260914_103252_calibration_outcomes"
NUSC = "nuScenes__YOLOv8s__ns_cheap_320tons_full_640__{g}.pkl"
AFFECTED = ("dE_E3_class_aware", "dE_E5_combined", "dE_E5_fp_heavy", "dE_E5_loc_heavy")


def gate(ok, msg):
    if not ok:
        raise SystemExit(f"CHECK FAILED: {msg}")


def recombine(d):
    comb = lambda w, tag: sum(v * d[f"{tag}_{k}"].to_numpy(float) for k, v in w.items())
    return {f"dE_{n}": comb(w, "cheap") - comb(w, "full") for n, w in PM.METRICS.items()}


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run = runmeta.new_run("class_error_prefix_tables", vars(args))
    stamp = "_".join(run.name.split("_")[:2])
    log = []

    src = rap_runs.core_matrix("plain", "ego")
    out = RAW / f"{stamp}_core_matrix_ego_prefix"
    shutil.copytree(src, out)
    for g in ("oracle", "mono"):
        e, old = pd.read_pickle(out / NUSC.format(g=g)), pd.read_pickle(PREFIX_CORE / NUSC.format(g=g))
        gate((e.seq.astype(str).to_numpy() == old.seq.astype(str).to_numpy()).all()
             and (e.frame.to_numpy() == old.frame.to_numpy()).all(), f"{g}: rows do not align")
        rc = recombine(e)
        for k in ("dE_E1_fn_only", "dE_E2_fn_fp", "dE_E4_localization", "dE_E6_risk_weighted"):
            gate(np.array_equal(rc[k], e[k].to_numpy(float)), f"{g}: recombination does not reproduce {k}")
        p = e.copy()
        for tag in ("cheap", "full"):
            p[f"{tag}_cls"] = old[f"{tag}_cls"].to_numpy()
        for k, v in recombine(p).items():
            if k in AFFECTED:
                p[k] = v
        moved = [c for c in p.columns if not p[c].astype(str).equals(e[c].astype(str))]
        gate(set(moved) <= {"cheap_cls", "full_cls", *AFFECTED}, f"{g}: unexpected columns moved: {moved}")
        p.to_pickle(out / NUSC.format(g=g))
        log.append({"file": f"{out.name}/{NUSC.format(g=g)}", "columns_moved": moved})
        print(f"  {out.name}/{NUSC.format(g=g)}: {moved}", flush=True)

    src = rap_runs.latest("calibration_outcomes", "ego")
    out = RAW / f"{stamp}_calibration_outcomes_ego_prefix"
    shutil.copytree(src, out)
    n = 0
    for f in sorted((out / "outcomes").glob("nusc_*.npz")):
        z = dict(np.load(f, allow_pickle=False))
        old = np.load(PREFIX_OUTCOMES / "outcomes" / f.name, allow_pickle=False)
        gate(np.array_equal(z["frame"], old["frame"]) and np.array_equal(z["seq"], old["seq"]), f"{f.name}: rows")
        for tag in ("cheap", "full"):
            z[f"{tag}_cls"] = old[f"{tag}_cls"].astype(z[f"{tag}_cls"].dtype)
        np.savez_compressed(f, **z)
        n += 1
    (out / "TASK23_PREFIX.md").write_text(
        "Ego-frame calibration outcomes with the pre-fix nuScenes class labels, for C25 re-expressed in the ego frame "
        "(Task 23). Only cheap_cls and full_cls of the 32 nuScenes files differ from the ego-frame run this copies; "
        "102_calibration_cells.py recombines the gains from them.\n")
    log.append({"file": f"{out.name}/outcomes/nusc_*.npz", "files": n})
    print(f"  {out.name}: {n} nuScenes outcome files with the pre-fix cls", flush=True)
    (run / "prefix_tables.json").write_text(json.dumps(log, indent=1))
    print("wrote", run)


if __name__ == "__main__":
    main()
