#!/usr/bin/env python
"""Task 22 Part C, data stage: per-frame outcomes with and without the camera-to-ego translation in the lift.

Pre-registered in the pre-registration record (not part of this release) (Task 22 Part C), committed before this script was written.  Sensitivity only:
no official output is touched.

The monocular lift measures range along the camera optical axis and lateral extent about the camera axis, and the
planners read those as ego-frame quantities.  `mono.apply_cam_offset` applies the calibrated translation, off by
default.  Applying it to the cached geometry is exact (see `mono.apply_cam_offset`); check L1 below asserts it
against re-running the lift.

For every setting the per-frame table is rebuilt twice, `off` and `on`, through the unchanged `decision.build` and
`65.build_b`, and written to results/raw/<run>/outcomes__<setting>__<geometry>__<off|on>.csv.gz.
`141_lift_offset_sensitivity.py` computes every reported quantity from these tables.
"""
from __future__ import annotations

import argparse, importlib.util, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import decision, geometry as G, kitti, mono, planner as P, runmeta      # noqa: E402
from rap.cache import DetCache                                                   # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                   # noqa: E402
from rap.risk import RiskConfig                                                  # noqa: E402

_s = importlib.util.spec_from_file_location("b65", ROOT / "scripts" / "65_planner_b_decision.py")
b65 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(b65)

KITTI_PAIRS = {"Y8_320": ("det", "cheap_320", "full_640"), "Y8_384": ("det", "cheap_384", "full_640"),
               "Y8_512": ("det512", "cheap_512", "full_640"),
               "RT_320": ("rtdetr_kitti", "rt_cheap_320", "rt_full_640"),
               "RT_480": ("rtdetr_kitti_mid", "rt_mid_480", "rt_full_640")}
KITTI_SETTINGS = [(p, "mono") for p in KITTI_PAIRS] + [("Y8_320", "oracle")]
# (file flag, lift frame) of the two tables built per setting.  In the camera frame this is Task 22 Part C exactly --
# the camera-frame lift against the translation-only shift, which gate G2 of Task 23 proved the frame-aware cache
# reproduces byte for byte.  In the ego frame (Task 23, C27 restated) the base is the ego-frame lift and the
# alternative the camera frame, so the same 14 settings measure the cost of the old convention.
FLAGS = {"camera": (("off", "camera"), ("on", "task22")), "ego": (("ego", "ego"), ("camera", "camera"))}
NUSC_PAIR = ("nusc_det_tv", "ns_cheap_320", "ns_full_640")


def gate(ok, message: str) -> None:
    """A pre-registered check that must stop the run. `assert` would vanish under `python -O`."""
    if not ok:
        raise RuntimeError(message)


def kitti_offsets(seqs):
    """cam2 -> IMU translation per sequence: (forward, left) in metres."""
    return {s: tuple(float(v) for v in kitti.load_calib(s).cam_to_imu[:2, 3]) for s in seqs}


def nusc_offsets(db, names):
    by = {db.scene_name(s): s for s in db.scenes}
    out = {}
    for n in names:
        _, t = db._cam_to_ego(db.samples(by[n])[0])
        out[n] = (float(t[0]), float(t[1]))
    return out


def factory(offsets):
    """A cache factory whose `geo` is the cached geometry shifted into the ego frame."""
    if offsets is None:
        return None

    class Shifted(DetCache):
        def __init__(self, path: Path, tag: str):
            super().__init__(path)
            self.off = offsets[path.stem]

        def geo(self, i: int) -> dict:
            return mono.apply_cam_offset(super().geo(i), self.off)

    return lambda path, tag: Shifted(Path(path), tag)


def check_lift(seqs, offsets, checks):
    """L1: shifting the cached geometry equals re-running the lift with `cam_offset` set."""
    worst, n = 0.0, 0
    for s in seqs[:3]:
        c = DetCache(Path(CACHE) / "det" / "cheap_320" / f"{s}.npz")
        calib = kitti.load_calib(s)
        prev = None
        for i in range(min(len(c), 40)):
            d = c.det(i)
            a = mono.apply_cam_offset(mono.predicted_geometry(d, prev, calib), offsets[s])
            b = mono.predicted_geometry(d, prev, calib, cam_offset=offsets[s])
            for k in a:
                worst = max(worst, float(np.max(np.abs(a[k] - b[k]))) if len(a[k]) else 0.0)
            n += len(d["conf"])
            prev = d
    checks.append({"check": "L1", "what": "shifting the cached geometry == the lift with cam_offset",
                   "detections": n, "max_abs_diff": worst, "ok": worst == 0.0})
    gate(worst == 0.0, f"L1 failed: shifting the cached geometry differs from the lift by {worst:g}")
    print(f"  L1: {n} detections, max |diff| {worst:g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None)
    ap.add_argument("--skip_nuscenes", action="store_true")
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    flags = FLAGS[frames.current()]
    run = Path(args.resume) if args.resume else runmeta.new_run("lift_offset_outcomes", vars(args))
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    pb, cb = b65.B.PARAMS_B["static_obstacles"], b65.B.COSTS_B["default"]
    kseqs = [p.stem for p in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]
    koff = kitti_offsets(kseqs)
    checks: list = []
    check_lift(kseqs, koff, checks)
    print(f"  KITTI offsets: forward {min(v[0] for v in koff.values()):.3f}-{max(v[0] for v in koff.values()):.3f} m, "
          f"lateral {min(v[1] for v in koff.values()):+.3f}-{max(v[1] for v in koff.values()):+.3f} m", flush=True)

    for pair, geo in KITTI_SETTINGS:
        dd, cm, fm = KITTI_PAIRS[pair]
        dd = Path(CACHE) / dd
        for flag, fr in flags:
            out = run / f"outcomes__{pair}__{geo}__{flag}.csv.gz"
            if out.exists():
                print(f"  skip {out.name}", flush=True)
                continue
            t0 = time.time()
            fac = lambda path, tag, fr=fr: DetCache(Path(path), frame=fr)
            a = decision.build(dd, cm, fm, kseqs, cfg, pp, cp, G.PRIMARY, range_source=geo,
                               adapter=decision.KittiAdapter, cache_factory=fac)
            b = b65.build_b(dd, cm, fm, kseqs, cfg, decision.KittiAdapter, geo, pb, cb, cache_factory=fac)
            t = a[["seq", "frame", "v_ego", "J_cheap", "J_full", "n_cheap", "n_full"]].merge(
                b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], validate="one_to_one")
            t.to_csv(out, index=False)
            print(f"  {out.name}: {len(t)} frames [{time.time() - t0:.0f}s]", flush=True)

    if not args.skip_nuscenes:
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        ad = make_adapter(db)
        nseqs = [p.stem for p in sorted((Path(CACHE) / NUSC_PAIR[0] / NUSC_PAIR[1]).glob("*.npz"))]
        noff = nusc_offsets(db, nseqs)
        print(f"  nuScenes offsets: forward {min(v[0] for v in noff.values()):.3f}-"
              f"{max(v[0] for v in noff.values()):.3f} m, lateral {min(v[1] for v in noff.values()):+.3f}-"
              f"{max(v[1] for v in noff.values()):+.3f} m", flush=True)
        dd = Path(CACHE) / NUSC_PAIR[0]
        for geo in ("mono", "oracle"):
            for flag, fr in flags:
                out = run / f"outcomes__NS_320__{geo}__{flag}.csv.gz"
                if out.exists():
                    print(f"  skip {out.name}", flush=True)
                    continue
                t0 = time.time()
                fac = lambda path, tag, fr=fr: DetCache(Path(path), frame=fr)
                a = decision.build(dd, NUSC_PAIR[1], NUSC_PAIR[2], nseqs, cfg, pp, cp, G.PRIMARY,
                                   range_source=geo, adapter=ad, cache_factory=fac)
                t = a[["seq", "frame", "v_ego", "J_cheap", "J_full", "n_cheap", "n_full"]].copy()
                t["JB_cheap"] = t["JB_full"] = np.nan          # Planner B is a KITTI track
                t.to_csv(out, index=False)
                print(f"  {out.name}: {len(t)} frames [{time.time() - t0:.0f}s]", flush=True)

    c = pd.DataFrame(checks)
    c.to_csv(run / "checks.csv", index=False)
    if not bool(c.ok.all()):
        raise SystemExit("a check failed: see checks.csv")
    print("wrote", run)


if __name__ == "__main__":
    main()
