#!/usr/bin/env python
"""Task 23 Phase 0: the camera -> ego transform behind the monocular lift, measured before anything changes.

Pre-registered in the pre-registration record (not part of this release) (Task 23), which this script's numbers feed.  Nothing official is read or written.

  A  Decompose camera -> ego for every KITTI sequence and nuScenes scene into the axis permutation (camera x right,
     y down, z forward -> ego x forward, y left, z up) and the residual mounting rotation; report roll, pitch and yaw.
  B  Synthetic: a car on the road plane at 10, 25 and 50 m, straight ahead of the camera and at either image edge,
     projected through the true camera.  The shipped lift runs on its 2D box, and four conventions turn the result
     into ego coordinates.  Their error against the known truth isolates what the residual rotation does.
  C  Empirical: every CHEAP detection matched (IoU >= 0.5) to a reference object, its lifted range and lateral centre
     against the reference geometry (which is ego-frame on both tracks), binned by range, under the same four
     conventions.  This says which convention the data supports, not just which one the algebra prefers.

Conventions (x = forward range to the near face, y = lateral centre, both ego frame):
  S  shipped: the camera-frame lift read as ego-frame
  T  translation only: Task 22 Part C's cam_offset, with KITTI's cam2 baseline offset included
  F  full rigid transform of the fused 3D point; lateral extent from the transformed corner rays
  G  the ground cue re-derived in the ego frame: the bottom-centre ray rotated into the ego frame and intersected with
     the road plane (as the nuPlan track's 114 already does); the height cue transformed rigidly; the two fused with
     the shipped weight rule, measured on the horizon-corrected image row
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import geometry as G, kitti, mono, runmeta                              # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.mono import box_iou                                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                  # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

PERM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)      # camera (x right, y down, z fwd) -> ego (x fwd, y left, z up)
CAR_W, CAR_H = 1.95, mono.HEIGHT_PRIOR["vehicle"]                # the prior's height, so the height cue is unbiased
DISTS = (10.0, 25.0, 50.0)
BINS = [(0, 10), (10, 25), (25, 50), (50, 1e9)]
THR, IOU = 0.25, 0.5


class Cam:
    """One camera: intrinsics, true camera -> ego rotation and translation, the lift's camera height, image width."""

    def __init__(self, name, fx, fy, cx, cy, R, t, cam_h, width, calib):
        self.name, self.fx, self.fy, self.cx, self.cy = name, fx, fy, cx, cy
        self.R, self.t, self.cam_h, self.W, self.calib = R, t, cam_h, width, calib
        self.z_road = t[2] - cam_h                   # the road plane sits cam_h below the camera, as the lift assumes

    def residual_rpy(self):
        """roll, pitch, yaw [deg] of R @ PERM^T about ego x, y, z; pitch > 0 = optical axis tilted up."""
        yaw, pitch, roll = Rotation.from_matrix(self.R @ PERM.T).as_euler("zyx", degrees=True)
        return roll, -pitch, yaw                    # a rotation about +y tilts +x down, so flip it to "up is positive"

    def project(self, P):
        pc = (self.R.T @ (np.asarray(P, float) - self.t).T).T
        return np.stack([self.fx * pc[:, 0] / pc[:, 2] + self.cx, self.fy * pc[:, 1] / pc[:, 2] + self.cy], 1)


def kitti_cam(seq):
    c = kitti.load_calib(seq)
    K = c.P2[:3, :3]
    c2 = -np.linalg.solve(K, c.P2[:, 3])            # cam2's centre in the rectified cam0 frame: the ~6 cm baseline
    T = c.cam_to_imu
    t = T[:3, :3] @ c2 + T[:3, 3]
    import cv2
    img = cv2.imread(str(kitti.image_path(seq, int(kitti.frame_ids(seq)[0]))))
    return Cam(f"KITTI {seq}", c.fx, c.fy, c.cx, c.cy, T[:3, :3], t, mono.CAMERA_HEIGHT, img.shape[1], c)


def convert(cam, det, geo, how):
    """(x, y_centre) in the ego frame for each detection under one convention."""
    x1, y1, x2, y2 = det["xyxy"].astype(float).T
    z = geo["z"].astype(float)
    uc = 0.5 * (x1 + x2)
    if how == "S":
        return z, 0.5 * (geo["lat_min"] + geo["lat_max"])
    if how == "T":
        return z + cam.t[0], 0.5 * (geo["lat_min"] + geo["lat_max"]) + cam.t[1]

    def rigid(depth, u):
        p = np.stack([(u - cam.cx) * depth / cam.fx, (y2 - cam.cy) * depth / cam.fy, depth], 1)
        return (cam.R @ p.T).T + cam.t

    if how == "F":
        pc, pl, pr = rigid(z, uc), rigid(z, x1), rigid(z, x2)
        return pc[:, 0], 0.5 * (pl[:, 1] + pr[:, 1])
    # G: ground cue from the rotated ray, height cue rigidly, fused on the horizon-corrected row
    ray = np.stack([(uc - cam.cx) / cam.fx, (y2 - cam.cy) / cam.fy, np.ones_like(uc)], 1)
    d = (cam.R @ ray.T).T
    down = d[:, 2] < -1e-9
    s = np.where(down, (cam.z_road - cam.t[2]) / np.where(down, d[:, 2], -1.0), np.inf)
    Pg = cam.t + s[:, None] * d
    Pg[~down] = np.nan
    zh = np.clip(geo["z_height"].astype(float), 0.5, 200.0)
    Ph = rigid(zh, uc)
    dv = cam.fy * (-d[:, 2]) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)   # pixels below the ego horizon
    w = np.clip((dv - 5.0) / 25.0, 0.0, 1.0)
    w = np.where(np.isfinite(Pg[:, 0]), w, 0.0)
    Pf = w[:, None] * np.nan_to_num(Pg) + (1 - w[:, None]) * Ph
    x = np.clip(Pf[:, 0], 0.5, 200.0)
    zf = (cam.R.T @ (Pf - cam.t).T).T[:, 2]                                   # its camera depth, for the corners
    pl, pr = rigid(zf, x1), rigid(zf, x2)
    return x, 0.5 * (pl[:, 1] + pr[:, 1])


def synthetic(cam):
    """Part B: a car on the road, straight ahead and at either image edge, at each distance."""
    rows = []
    for d in DISTS:
        for pos, u_frac in (("axis", None), ("left edge", -0.9), ("right edge", 0.9)):
            if u_frac is None:
                y = cam.t[1]
            else:
                u = cam.cx + u_frac * cam.W / 2
                y = cam.t[1] - (d - cam.t[0]) * (u - cam.cx) / cam.fx
            corners = np.array([[d, y + sy * CAR_W / 2, cam.z_road + sz * CAR_H] for sy in (-1, 1) for sz in (0, 1)])
            uv = cam.project(corners)
            box = np.array([[uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()]])
            det = {"xyxy": box, "coarse": np.array(["vehicle"])}
            geo = mono.predicted_geometry(det, None, cam.calib, cam.cam_h)
            for how in "STFG":
                x, yc = convert(cam, det, geo, how)
                rows.append({"camera": cam.name, "dist": d, "position": pos, "convention": how,
                             "range_err": float(x[0] - d), "lat_err": float(yc[0] - y)})
    return rows


def empirical(cam, det_dir, mode, seq, geom, cfg_min_h, pick_frames=None):
    """Part C: matched CHEAP detections of one sequence, their lift against the reference geometry."""
    c = DetCache(Path(det_dir) / mode / f"{seq}.npz")
    rows = []
    for i, fr in enumerate(c.frames):
        fr = int(fr)
        g = geom[geom["frame"] == fr]
        g = g[(g["y2"] - g["y1"]) >= cfg_min_h]
        if not len(g):
            continue
        det, geo = c.det(i), c.geo(i)
        k = det["conf"] >= THR
        if not k.any():
            continue
        d = {"xyxy": det["xyxy"][k].astype(float), "coarse": det["coarse"][k]}
        gk = {kk: vv[k] for kk, vv in geo.items()}
        gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
        iou = box_iou(d["xyxy"], gt)
        m = iou.max(1) >= IOU
        if not m.any():
            continue
        best = iou.argmax(1)[m]
        ref_x, ref_y = g["long_near"][best].astype(float), g["lat_cen"][best].astype(float)
        dm = {"xyxy": d["xyxy"][m], "coarse": d["coarse"][m]}
        gm = {kk: vv[m] for kk, vv in gk.items()}
        for how in "STFG":
            x, yc = convert(cam, dm, gm, how)
            for a, b, rx, ry in zip(x, yc, ref_x, ref_y):
                rows.append({"camera": cam.name, "convention": how, "ref_x": rx, "range_err": a - rx,
                             "lat_err": b - ry})
    return rows


def summarise(df):
    out = []
    for conv in "STFG":
        for lo, hi in BINS:
            s = df[(df.convention == conv) & (df.ref_x >= lo) & (df.ref_x < hi)]
            if not len(s):
                continue
            out.append({"convention": conv, "range_bin": f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}", "n": len(s),
                        "range_bias": float(s.range_err.median()), "range_mae": float(s.range_err.abs().median()),
                        "rel_range_bias": float((s.range_err / s.ref_x).median()),
                        "lat_bias": float(s.lat_err.median()), "lat_mae": float(s.lat_err.abs().median())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_nuscenes", action="store_true")
    args = ap.parse_args()
    run = runmeta.new_run("frame_transform_audit", vars(args))
    rpy, syn, emp = [], [], []

    kseqs = [p.stem for p in sorted((Path(CACHE) / "det" / "cheap_320").glob("*.npz"))]
    for s in kseqs:
        cam = kitti_cam(s)
        r, p, y = cam.residual_rpy()
        rpy.append({"track": "KITTI", "unit": s, "roll": r, "pitch": p, "yaw": y,
                    "tx": cam.t[0], "ty": cam.t[1], "tz": cam.t[2]})
        syn += [{"track": "KITTI", **x} for x in synthetic(cam)]
        emp += [{"track": "KITTI", **x} for x in empirical(cam, Path(CACHE) / "det", "cheap_320", s,
                                                          G.sequence_geometry(s), RiskConfig().min_gt_height)]
    print(f"  KITTI: {len(kseqs)} sequences", flush=True)

    if not args.skip_nuscenes:
        from rap.nusc import NuScenesDB, make_adapter
        db = NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval")
        ad = make_adapter(db)
        by = {db.scene_name(s): s for s in db.scenes}
        names = [p.stem for p in sorted((Path(CACHE) / "nusc_det_tv" / "ns_cheap_320").glob("*.npz"))]
        for n in names:
            tok = db.samples(by[n])[0]
            R, t = db._cam_to_ego(tok)
            c = db.calib(tok)
            cam = Cam(f"nuScenes {n}", c.fx, c.fy, c.cx, c.cy, R, t, 1.51, 1600, c)
            r, p, y = cam.residual_rpy()
            rpy.append({"track": "nuScenes", "unit": n, "roll": r, "pitch": p, "yaw": y,
                        "tx": t[0], "ty": t[1], "tz": t[2]})
            syn += [{"track": "nuScenes", **x} for x in synthetic(cam)]
            emp += [{"track": "nuScenes", **x} for x in empirical(cam, Path(CACHE) / "nusc_det_tv", "ns_cheap_320", n,
                                                                 ad.geometry(n), RiskConfig().min_gt_height)]
        print(f"  nuScenes: {len(names)} scenes", flush=True)

    A, B, C = pd.DataFrame(rpy), pd.DataFrame(syn), pd.DataFrame(emp)
    A.to_csv(run / "A_residual_rotation.csv", index=False)
    B.to_csv(run / "B_synthetic.csv", index=False)
    C.to_csv(run / "C_empirical_rows.csv.gz", index=False)
    out = {}
    for tr, a in A.groupby("track"):
        out[tr] = {k: {"min": float(a[k].min()), "median": float(a[k].median()), "max": float(a[k].max())}
                   for k in ("roll", "pitch", "yaw", "tx", "ty", "tz")}
    (run / "A_summary.json").write_text(json.dumps(out, indent=1))
    Bs = (B.groupby(["track", "position", "dist", "convention"])
            .agg(range_err_min=("range_err", "min"), range_err_median=("range_err", "median"),
                 range_err_max=("range_err", "max"), lat_err_median=("lat_err", "median"),
                 lat_err_absmax=("lat_err", lambda v: float(np.abs(v).max()))).reset_index())
    Bs.to_csv(run / "B_summary.csv", index=False)
    Cs = pd.concat([summarise(c).assign(track=tr) for tr, c in C.groupby("track")], ignore_index=True)
    Cs.to_csv(run / "C_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(json.dumps({tr: {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in d.items()}
                      for tr, d in out.items()}, indent=1))
    print(Bs.round(3).to_string(index=False))
    print(Cs.round(3).to_string(index=False))
    print("wrote", run)


if __name__ == "__main__":
    main()
