#!/usr/bin/env python
"""Task 21, data stage [D]: per-frame braking and Planner B outcomes under two realism controls, KITTI.

Pre-registered in the pre-registration record (not part of this release), Task 21.  Cached detections only; the
KITTI labels, calibration and OXTS files are needed for the reference geometry, the planners' ego speed and the
evaluator, as in `52_core_matrix.py`.

  persistence   the same filter on the CHEAP and FULL lists before both controllers: a detection at frame t (conf at
                the operating threshold) passes only if a detection of the same coarse class, also at the operating
                threshold, overlaps it at image IoU >= 0.3 in each of the previous n - 1 frames of the sequence and
                mode.  A frame with fewer than n - 1 predecessors checks the ones that exist (the first frame keeps
                every detection).  n = 1 is the benchmark.
  reference     `decision._apply_range_source(..., "oracle")`, the existing control, for every KITTI pair

Every configuration writes one table, results/raw/<run>/outcomes__<pair>__<geometry>__n<n>.csv.gz, with J (braking
controller) and JB (Planner B, `static_obstacles`) for CHEAP and FULL per frame.  `135_realism_controls.py` computes
every reported quantity from these tables.  Configurations already written are skipped (--resume).
"""
from __future__ import annotations

import argparse, importlib.util, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, geometry as G, planner as P, runmeta                   # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.mono import box_iou                                                    # noqa: E402
from rap.paths import CACHE                                                     # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

_s = importlib.util.spec_from_file_location("b65", ROOT / "scripts" / "65_planner_b_decision.py")
b65 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(b65)

PAIRS = {"Y8_320": ("det", "cheap_320", "full_640"), "Y8_384": ("det", "cheap_384", "full_640"),
         "Y8_512": ("det512", "cheap_512", "full_640"), "RT_320": ("rtdetr_kitti", "rt_cheap_320", "rt_full_640"),
         "RT_480": ("rtdetr_kitti_mid", "rt_mid_480", "rt_full_640")}
IOU_PERSIST = 0.3


class PersistentDetCache(DetCache):
    """A detection cache whose frames keep only detections that persisted over the previous n - 1 frames."""

    def __init__(self, path, n: int, thr: float):
        super().__init__(path)
        self.n, self.thr, self._keep = n, thr, {}

    def keep(self, i: int) -> np.ndarray:
        if i not in self._keep:
            d = DetCache.det(self, i)
            k = d["conf"] >= self.thr
            if self.n > 1 and k.any():
                idx = np.flatnonzero(k)
                ok = np.ones(len(idx), bool)
                for lag in range(1, self.n):
                    j = i - lag
                    if j < 0:                                  # whatever history exists
                        break
                    p = DetCache.det(self, j)
                    pk = p["conf"] >= self.thr
                    if not pk.any():
                        ok[:] = False
                        break
                    iou = box_iou(d["xyxy"][idx].astype(float), p["xyxy"][pk].astype(float))
                    same = d["coarse"][idx][:, None] == p["coarse"][pk][None, :]
                    ok &= ((iou >= IOU_PERSIST) & same).any(axis=1)
                k[idx[~ok]] = False
            self._keep[i] = k
        return self._keep[i]

    def det(self, i: int) -> dict:
        d, m = DetCache.det(self, i), self.keep(i)
        return {key: (val[m] if isinstance(val, np.ndarray) and len(val) == len(m) else val) for key, val in d.items()}

    def geo(self, i: int) -> dict:
        g, m = DetCache.geo(self, i), self.keep(i)
        return {key: val[m] for key, val in g.items()}


def factory(n, cfg):
    if n == 1:
        return None                                            # the benchmark's own DetCache
    return lambda path, role: PersistentDetCache(path, n, cfg.thr(role))


def configurations():
    out = [(pair, "mono", n) for pair in PAIRS for n in (1, 2, 3)]
    out += [("Y8_320", "oracle", n) for n in (1, 2, 3)]
    out += [(pair, "oracle", 1) for pair in PAIRS if pair != "Y8_320"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="an existing run directory to complete")
    args = ap.parse_args()
    run = Path(args.resume) if args.resume else runmeta.new_run("realism_outcomes", vars(args))
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    pb, cb = b65.B.PARAMS_B["static_obstacles"], b65.B.COSTS_B["default"]
    seqs = [p.stem for p in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]
    for pair, geo, n in configurations():
        out = run / f"outcomes__{pair}__{geo}__n{n}.csv.gz"
        if out.exists():
            print(f"  skip {out.name}", flush=True)
            continue
        dd, cm, fm = PAIRS[pair]
        dd = Path(CACHE) / dd
        t0 = time.time()
        fac = factory(n, cfg)
        a = decision.build(dd, cm, fm, seqs, cfg, pp, cp, G.PRIMARY, range_source=geo,
                           adapter=decision.KittiAdapter, cache_factory=fac)
        b = b65.build_b(dd, cm, fm, seqs, cfg, decision.KittiAdapter, geo, pb, cb, cache_factory=fac)
        t = a[["seq", "frame", "v_ego", "J_cheap", "J_full", "n_cheap", "n_full"]].merge(
            b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], validate="one_to_one")
        t.to_csv(out, index=False)
        print(f"  {out.name}: {len(t)} frames, dets kept per frame {t.n_cheap.mean():.2f} / {t.n_full.mean():.2f} "
              f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
