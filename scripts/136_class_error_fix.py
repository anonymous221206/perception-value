#!/usr/bin/env python
"""Task 22 Part A: corrected nuScenes class labels in the per-frame tables.

Pre-registered in the pre-registration record (not part of this release) (Task 22 Part A), committed before this script was written.

Until 2026-09-20 the reference class of a nuScenes object was `TYPE_TO_COARSE.get(type, "vehicle")` over the category
prefix, so every object was labelled `vehicle` and every correctly detected pedestrian or cyclist counted as a class
error.  `rap.geometry.coarse_classes` now reads the coarse class the geometry carries, and `rap.nusc.coarse_class`
maps the full category.  Only the `cls` primitive changes, and with it E3 and the three E5 variants, on nuScenes.

This script recomputes `cls` for every nuScenes mode and threshold the shipped per-frame tables use, writes corrected
copies of those tables to new run directories, and checks that nothing else moves:

  C1  the recomputation reproduces the shipped fn, fp, loc, crit_fn, n_det and n_gt exactly
  C2  recombining the shipped primitives with the shipped weights reproduces the shipped dE_E1, dE_E2, dE_E4, dE_E6
  C3  (--check_end_to_end) the unchanged generator path reproduces a corrected outcome file exactly

Outputs, all new: results/raw/<ts>_core_matrix_classfix/, <ts>_core_matrix_postreview_classfix/,
<ts>_calibration_outcomes/ (the corrected copy the later stages read), and a summary JSON.
"""
from __future__ import annotations

import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import geometry as G, percep_metrics as PM, runmeta                    # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                   # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                  # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

RAW = ROOT / "results" / "raw"
CORE_RUNS = ("20260912_071140_core_matrix", "20260913_133004_core_matrix_postreview")
OUTCOMES_RUN = "20260914_103252_calibration_outcomes"
NUSC_PICKLE = "nuScenes__YOLOv8s__ns_cheap_320tons_full_640__{geom}.pkl"
DET_DIR, MODES = "nusc_det_tv", {"cheap": "ns_cheap_320", "full": "ns_full_640"}
PRIMITIVES = ("fn", "fp", "loc", "cls", "crit_fn", "n_det", "n_gt")
UNCHANGED = ("fn", "fp", "loc", "crit_fn", "n_det", "n_gt")       # every primitive except the class error
E_CHANGED = ("E3_class_aware", "E5_combined", "E5_fp_heavy", "E5_loc_heavy")
E_UNCHANGED = ("E1_fn_only", "E2_fn_fp", "E4_localization", "E6_risk_weighted")


def frame_primitives(adapter, seqs, mode, threshold, cfg):
    """Every per-frame primitive of one nuScenes mode at one threshold, under the corrected class labels."""
    rows = []
    for s in seqs:
        geom = adapter.geometry(s)
        crit = G.criticality_for(geom, G.PRIMARY)
        cache = DetCache(Path(CACHE) / DET_DIR / mode / f"{s}.npz")
        for i, fr in enumerate(cache.frames):
            fr = int(fr)
            sel = geom["frame"] == fr
            g, cg = geom[sel], crit[sel]
            keep = (g["y2"] - g["y1"]) >= cfg.min_gt_height
            g, cg = g[keep], cg[keep]
            gt = (np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4)))
            L = PM.frame_losses(gt, G.coarse_classes(g), cg, cache.det(i), cfg, op_conf=threshold)
            rows.append({"seq": s, "frame": fr, **{k: L[k] for k in PRIMITIVES}})
    return pd.DataFrame(rows)


class Recompute:
    """Per (mode, threshold) corrected primitives, computed once and reused by every table that needs them."""

    def __init__(self, adapter, seqs):
        self.adapter, self.seqs, self.cache, self.checks = adapter, seqs, {}, []

    def get(self, mode, threshold):
        key = (mode, float(threshold))            # the exact threshold: S1 and S3 carry more than six decimals
        if key not in self.cache:
            t0 = time.time()
            self.cache[key] = frame_primitives(self.adapter, self.seqs, mode, key[1], RiskConfig())
            print(f"  primitives {mode} at {key_str(key[1])}: {len(self.cache[key])} frames "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        return self.cache[key]

    def check_unchanged(self, where, mode, threshold, stored: pd.DataFrame, tag: str):
        """C1: every primitive except `cls` must match what the shipped table stores."""
        got = self.get(mode, threshold)
        assert len(got) == len(stored), (where, len(got), len(stored))
        assert np.array_equal(got["seq"].astype(str).to_numpy(), stored["seq"].astype(str).to_numpy()) and \
            np.array_equal(got["frame"].to_numpy(int), stored["frame"].to_numpy(int)), f"frame order differs: {where}"
        bad = {}
        for k in UNCHANGED:
            col = f"{tag}_{k}"
            if col not in stored.columns:
                continue
            a, b = got[k].to_numpy(float), stored[col].to_numpy(float)
            if not same_array(a, b):
                bad[col] = int((a != b).sum())
        self.checks.append({"check": "C1", "where": where, "mode": mode, "threshold": key_str(threshold),
                            "columns_compared": len([k for k in UNCHANGED if f"{tag}_{k}" in stored.columns]),
                            "columns_differing": json.dumps(bad), "ok": not bad})
        return not bad


def key_str(t):
    return repr(float(t))


def same_array(a, b) -> bool:
    """Exact equality, counting NaN equal to NaN: on nuScenes the Planner B columns are all NaN."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        na, nb = np.isnan(a), np.isnan(b)
        return bool(np.array_equal(na, nb)) and bool(np.array_equal(a[~na], b[~nb]))
    return bool(np.array_equal(a, b))


def combine(d: pd.DataFrame, weights: dict, tag: str) -> np.ndarray:
    return sum(w * d[f"{tag}_{k}"].to_numpy(float) for k, w in weights.items())


def recombine_gains(d: pd.DataFrame) -> dict:
    """dE_<metric> = E(CHEAP) - E(FULL) from the primitives, with `percep_metrics`' own weights."""
    return {f"dE_{name}": combine(d, w, "cheap") - combine(d, w, "full") for name, w in PM.METRICS.items()}


def check_arithmetic(where, d: pd.DataFrame, checks: list):
    """C2: the recombination reproduces the shipped gains that do not involve the class error."""
    got, bad = recombine_gains(d), {}
    for name in E_UNCHANGED:
        col = f"dE_{name}"
        if col in d.columns and not same_array(got[col], d[col].to_numpy(float)):
            bad[col] = int((got[col] != d[col].to_numpy(float)).sum())
    checks.append({"check": "C2", "where": where, "metrics": len(E_UNCHANGED),
                   "columns_differing": json.dumps(bad), "ok": not bad})
    return not bad


def patch_core_matrix(rec: Recompute, run_in: Path, run_out: Path, summary: list):
    shutil.copytree(run_in, run_out, dirs_exist_ok=True)
    for geom in ("oracle", "mono"):
        f = run_out / NUSC_PICKLE.format(geom=geom)
        d = pd.read_pickle(f)
        old = d.copy()
        for tag, mode in MODES.items():
            assert rec.check_unchanged(f"{run_in.name}/{geom}", mode, 0.25, d, tag), "C1 failed"
            d[f"{tag}_cls"] = rec.get(mode, 0.25)["cls"].to_numpy(float)
        assert check_arithmetic(f"{run_in.name}/{geom}", old, rec.checks), "C2 failed"
        for col, v in recombine_gains(d).items():
            if col in d.columns:
                d[col] = v
        moved = {c: int((old[c].to_numpy(float) != d[c].to_numpy(float)).sum()) for c in d.columns
                 if pd.api.types.is_numeric_dtype(d[c]) and not same_array(old[c].to_numpy(float), d[c].to_numpy(float))}
        d.to_pickle(f)
        summary.append({"file": f"{run_out.name}/{f.name}", "rows": len(d), "columns_changed": sorted(moved),
                        "rows_changed_per_column": moved})
        print(f"  {run_out.name}/{f.name}: changed {sorted(moved)}", flush=True)


def patch_outcomes(rec: Recompute, run_in: Path, run_out: Path, summary: list):
    shutil.copytree(run_in, run_out, dirs_exist_ok=True)
    # The file name rounds the threshold to six decimals, but the S1 and S3 operating points come off a bisection
    # (0.498046875 on nuScenes), so the exact thresholds are read from the run's own index.
    idx = pd.read_csv(run_in / "outcome_index.csv")
    thr = {str(r.file): (float(r.t_cheap), float(r.t_full)) for r in idx.itertuples()}
    files = sorted((run_out / "outcomes").glob("nusc_*.npz"))
    for f in files:
        t_cheap, t_full = thr[f.name]
        z = dict(np.load(f, allow_pickle=False))
        d = pd.DataFrame({k: v for k, v in z.items()})
        changed = {}
        for tag, threshold in (("cheap", t_cheap), ("full", t_full)):
            assert rec.check_unchanged(f"{f.name}", MODES[tag], threshold, d, tag), "C1 failed"
            new = rec.get(MODES[tag], threshold)["cls"].to_numpy(float)
            old = z[f"{tag}_cls"].astype(float)
            changed[f"{tag}_cls"] = int((old != new).sum())
            z[f"{tag}_cls"] = new.astype(z[f"{tag}_cls"].dtype)
        np.savez_compressed(f, **z)
        summary.append({"file": f"{run_out.name}/outcomes/{f.name}", "rows": len(d), "rows_changed_per_column": changed})
    print(f"  {run_out.name}: {len(files)} nuScenes outcome files corrected", flush=True)


def check_end_to_end(run_out: Path, checks: list):
    """C3: 100's own generator path, with the fixed code, reproduces the corrected outcome file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("m100", ROOT / "scripts" / "100_calibration_outcomes.py")
    m100 = importlib.util.module_from_spec(spec)
    sys.modules["m100"] = m100
    spec.loader.exec_module(m100)
    m100._init("nuScenes")
    tmp = run_out / "check_end_to_end"
    tmp.mkdir(exist_ok=True)
    for key in ("nusc_oracle", "nusc_mono"):
        m100.run_task(key, 0.25, 0.25, tmp)
        name = f"{key}__c{0.25:.6f}__f{0.25:.6f}.npz"
        a = dict(np.load(tmp / name, allow_pickle=False))
        b = dict(np.load(run_out / "outcomes" / name, allow_pickle=False))
        bad = sorted(k for k in b if not same_array(a[k], b[k]))
        checks.append({"check": "C3", "where": name, "columns": len(b), "columns_differing": json.dumps(bad),
                       "ok": not bad})
        print(f"  C3 {name}: {'identical' if not bad else 'DIFFERS ' + str(bad)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check_end_to_end", action="store_true", help="also run C3 (rebuilds two outcome files)")
    args = ap.parse_args()
    run = runmeta.new_run("class_error_fix", vars(args))
    db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
    adapter = make_adapter(db)
    seqs = [p.stem for p in sorted((Path(CACHE) / DET_DIR / MODES["cheap"]).glob("*.npz"))]
    print(f"  {len(seqs)} nuScenes scenes", flush=True)
    rec, summary = Recompute(adapter, seqs), []
    stamp = run.name.split("_")[0] + "_" + run.name.split("_")[1]

    outs = {}
    for name in CORE_RUNS:
        out = RAW / f"{stamp}_{name.split('_', 2)[-1]}_classfix"
        patch_core_matrix(rec, RAW / name, out, summary)
        outs[name] = out.name
    out = RAW / f"{stamp}_calibration_outcomes"
    patch_outcomes(rec, RAW / OUTCOMES_RUN, out, summary)
    outs[OUTCOMES_RUN] = out.name
    if args.check_end_to_end:
        check_end_to_end(out, rec.checks)

    checks = pd.DataFrame(rec.checks)
    checks.to_csv(run / "checks.csv", index=False)
    ok = bool(checks.ok.all())
    (run / "class_error_fix_tables.json").write_text(json.dumps(
        {"corrected_runs": outs, "checks_passed": ok, "checks": rec.checks, "files": summary}, indent=1, default=str))
    print(f"  checks: {int(checks.ok.sum())}/{len(checks)} passed; corrected runs: {outs}", flush=True)
    if not ok:
        raise SystemExit("a check failed: see checks.csv")


if __name__ == "__main__":
    main()
