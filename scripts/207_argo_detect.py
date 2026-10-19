#!/usr/bin/env python
"""Follow-up E8: detection on Argoverse ring_front_center, and the camera table the cache re-lifts with.

addendum E8 in docs/followup_protocol.md. 176's per-frame path (TensorRT engine, `TwoFidelityDetector.detect`, the
monocular lift with the previous frame's detections of the same mode), on each log's decision frames, the images
undistorted first (`rap.argo.undistort_maps`). Writes
  data/cache/cam_to_ego_argo.json          per log: intrinsics, camera height above the ground, frame interval,
                                           camera -> ego (read by rap.cache.Camera), and the ground-offset record
  data/cache/argo1_det/<mode>/<unit>.npz   one file per log and mode
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import argo, cache, mono, runmeta              # noqa: E402
from rap.detect import MODES, TwoFidelityDetector       # noqa: E402
from rap.kitti import Calib                             # noqa: E402
from rap.paths import CACHE, MODELS, DATASETS     # noqa: E402

AV1_ROOT = DATASETS / "argoverse1" / "argoverse-tracking" / "val"
TABLE = Path(CACHE) / "cam_to_ego_argo.json"
MODES_E8 = ("ns_cheap_320", "ns_mid_480", "ns_full_640", "ns_full_960")


def camera_table(root, logs):
    """The camera entries of Argoverse 1 and the ground offset they use (the addendum's camera-height rule)."""
    L = {l: argo.AV1Log(root, l) for l in logs}
    z = np.concatenate([L[l].ground_offset_samples() for l in logs])
    offset = float(-np.median(z))
    ent = {}
    for l, g in L.items():
        ent[argo.unit_name(l)] = {"log": l, "city": g.city, "K": g.K.tolist(), "cam_h": float(g.t_cam[2] + offset),
                                  "dt": argo.AV1_DT, "R": g.R_cam.tolist(), "t": g.t_cam.tolist(),
                                  "t_task22": [float(g.t_cam[0]), float(g.t_cam[1])],
                                  "frames": len(g.frames), "dropped": g.dropped}
    return ent, {"ground_offset_m": offset, "n_boxes": int(len(z))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(AV1_ROOT))
    ap.add_argument("--modes", nargs="+", default=list(MODES_E8))
    ap.add_argument("--out", default=str(Path(CACHE) / "argo1_det"))
    ap.add_argument("--only", default=None, help="comma-separated log ids (checks)")
    args = ap.parse_args()
    run = runmeta.new_run("argo1_detect", vars(args))
    logs = sorted(p.name for p in Path(args.root).iterdir() if p.is_dir())
    tab = json.loads(TABLE.read_text()) if TABLE.exists() else {}
    if "Argoverse1" not in tab:
        ent, rec = camera_table(args.root, logs)
        tab["Argoverse1"], tab.setdefault("_records", {})["Argoverse1"] = ent, rec
        TABLE.write_text(json.dumps(tab, indent=1))
        print(f"  camera table: {len(ent)} logs, ground offset {rec['ground_offset_m']:.3f} m "
              f"({rec['n_boxes']} boxes)", flush=True)
    if args.only:
        logs = [l for l in logs if l in set(args.only.split(","))]
    det = TwoFidelityDetector(str(MODELS / "yolov8s.pt"), backend="trt")
    summary = {}
    for mname in args.modes:
        mode, outdir = MODES[mname], Path(args.out) / mname
        nf, t0 = 0, time.time()
        for l in logs:
            u = argo.unit_name(l)
            if (outdir / f"{u}.npz").exists():
                continue
            g = argo.AV1Log(args.root, l)
            e = tab["Argoverse1"][u]
            m1, m2 = argo.undistort_maps(g)
            P2 = np.zeros((3, 4)); P2[:3, :3] = np.asarray(e["K"], float)
            calib = Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))
            dets, geos, scal, prev = [], [], {}, None
            for i in range(len(g.frames)):
                img = cv2.remap(cv2.imread(str(g.image_path(i))), m1, m2, cv2.INTER_LINEAR)
                d, _ = det.detect(img, mode)
                geos.append(mono.predicted_geometry(d, prev, calib, e["cam_h"], e["dt"]))
                dets.append(d); prev = d
                for k in ("n_cand", "n_cand_raw", "n_post"):
                    scal.setdefault(k, []).append(d[k])
                scal.setdefault("img_h", []).append(img.shape[0]); scal.setdefault("img_w", []).append(img.shape[1])
            cache.save(outdir / f"{u}.npz", list(range(len(g.frames))), dets, geos, scal)
            nf += len(g.frames)
        summary[mname] = {"frames": nf, "seconds": round(time.time() - t0, 1)}
        print(f"  {mname}: {nf} frames [{time.time() - t0:.0f}s]", flush=True)
    (run / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
