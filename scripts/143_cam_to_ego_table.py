#!/usr/bin/env python
"""Task 23: tabulate every unit's camera -> ego transform and intrinsics once, for the frame-aware detection cache.

Pre-registered in the pre-registration record (not part of this release) (Task 23).  `rap.cache.DetCache` re-lifts cached boxes in the ego frame on read;
it needs, per KITTI sequence and nuScenes scene, the rotation R and translation t taking camera coordinates to ego
coordinates, the intrinsics, the lift's camera height and the frame interval.  Reading them here once means no stage
has to load the nuScenes database just to lift a box.

  KITTI     R from `Calib.cam_to_imu`; t = cam2's centre in the IMU frame (the rectified-cam0 origin plus the rotated
            cam2 baseline -K^-1 P2[:, 3]).  `t_task22` is the rectified-cam0 origin Task 22 Part C used, kept for G2.
            P2 as `kitti.load_calib` reads it, so the re-lift needs no KITTI calibration file (the release's cached
            tier runs without the datasets).
  nuScenes  R, t from CAM_FRONT's calibrated_sensor of the scene's first sample, with the intrinsics of that sample,
            as `40_nusc_detect.py` used them.  `t_task22` = t.

Writes data/cache/cam_to_ego.json.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import kitti, mono, runmeta                                            # noqa: E402
from rap.nusc import FRAME_DT as NUSC_DT, NuScenesDB                            # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                  # noqa: E402

OUT = Path(CACHE) / "cam_to_ego.json"
NUSC_CAM_H = 1.51           # as scripts/40_nusc_detect.py


def lst(a):
    return [[float(v) for v in row] for row in np.asarray(a, float)] if np.ndim(a) == 2 else [float(v) for v in a]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_nuscenes", action="store_true")
    args = ap.parse_args()
    run = runmeta.new_run("cam_to_ego_table", vars(args))
    table = {"KITTI": {}, "nuScenes": {}}
    for s in sorted(p.stem for p in (Path(CACHE) / "det" / "cheap_320").glob("*.npz")):
        c = kitti.load_calib(s)
        K = c.P2[:3, :3]
        T = c.cam_to_imu
        c2 = -np.linalg.solve(K, c.P2[:, 3])
        table["KITTI"][s] = {"R": lst(T[:3, :3]), "t": lst(T[:3, :3] @ c2 + T[:3, 3]), "t_task22": lst(T[:3, 3]),
                             "P2": lst(c.P2), "cam_h": mono.CAMERA_HEIGHT, "dt": kitti.FRAME_DT}
    if not args.skip_nuscenes:
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        by = {db.scene_name(sc): sc for sc in db.scenes}
        for n in sorted(p.stem for p in (Path(CACHE) / "nusc_det_tv" / "ns_cheap_320").glob("*.npz")):
            tok = db.samples(by[n])[0]
            R, t = db._cam_to_ego(tok)
            c = db.calib(tok)
            table["nuScenes"][n] = {"R": lst(R), "t": lst(t), "t_task22": lst(t), "K": lst(c.P2[:3, :3]),
                                    "cam_h": NUSC_CAM_H, "dt": NUSC_DT}
    OUT.write_text(json.dumps(table, indent=1))
    (run / "cam_to_ego.json").write_text(json.dumps(table, indent=1))
    print(f"  wrote {OUT}: {len(table['KITTI'])} KITTI sequences, {len(table['nuScenes'])} nuScenes scenes")


if __name__ == "__main__":
    main()
