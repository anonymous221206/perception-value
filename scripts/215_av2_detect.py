#!/usr/bin/env python
"""Follow-up E8b: detection on Argoverse 2 ring_front_center (214's copy), and its camera table.

addendum E8 in docs/followup_protocol.md (E8b rules). 207's per-frame path, with the four modes run frame by frame so
each image is decoded once, the hood rows cropped, and the images undistorted or not (`--version raw|undist`; gate
E8-0 picks one). Writes
  data/cache/cam_to_ego_argo.json                 the "Argoverse2" entries (one per log) and the ground-offset record
  data/cache/argo2_det[_<version>]/<mode>/<unit>.npz
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import argo, argo2, cache, mono, runmeta       # noqa: E402
from rap.detect import MODES, TwoFidelityDetector       # noqa: E402
from rap.kitti import Calib                             # noqa: E402
from rap.paths import CACHE, MODELS, DATASETS     # noqa: E402

TABLES = DATASETS / "av2_deep" / "av2_tables"
IMAGES = DATASETS / "av2_deep" / "sensor"
TABLE = Path(CACHE) / "cam_to_ego_argo.json"
MODES_E8 = ("ns_cheap_320", "ns_mid_480", "ns_full_640", "ns_full_960")


def camera_table(split_logs):
    L = {l: argo2.AV2Log(TABLES, IMAGES, sp, l) for sp, l in split_logs}
    z = np.concatenate([L[l].ground_offset_samples() for sp, l in split_logs if sp == "val"])
    offset = float(-np.median(z))
    ent = {}
    for (sp, l) in split_logs:
        g = L[l]
        ent[argo.unit_name(l)] = {"log": l, "split": sp, "K": g.K.tolist(), "cam_h": float(g.t_cam[2] + offset),
                                  "dt": argo2.AV2_DT, "R": g.R_cam.tolist(), "t": g.t_cam.tolist(),
                                  "t_task22": [float(g.t_cam[0]), float(g.t_cam[1])], "frames": len(g.frames)}
    assert len(ent) == len(split_logs), "unit names collide"
    return ent, {"ground_offset_m": offset, "n_boxes": int(len(z))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=["raw", "undist"], required=True)
    ap.add_argument("--modes", nargs="+", default=list(MODES_E8))
    ap.add_argument("--splits", nargs="+", default=["val", "train"])
    ap.add_argument("--out", default=None, help="default data/cache/argo2_det_<version>")
    args = ap.parse_args()
    run = runmeta.new_run("argo2_detect", vars(args))
    split_logs = argo2.logs(TABLES)
    tab = json.loads(TABLE.read_text()) if TABLE.exists() else {}
    if "Argoverse2" not in tab:
        ent, rec = camera_table(split_logs)
        tab["Argoverse2"], tab.setdefault("_records", {})["Argoverse2"] = ent, rec
        TABLE.write_text(json.dumps(tab, indent=1))
        print(f"  camera table: {len(ent)} logs, ground offset {rec['ground_offset_m']:.3f} m", flush=True)
    out = Path(args.out or Path(CACHE) / f"argo2_det_{args.version}")
    det = TwoFidelityDetector(str(MODELS / "yolov8s.pt"), backend="trt")
    modes = [MODES[m] for m in args.modes]
    t0, nf = time.time(), 0
    for sp, l in split_logs:
        if sp not in args.splits:
            continue
        u = argo.unit_name(l)
        if all((out / m.name / f"{u}.npz").exists() for m in modes):
            continue
        g = argo2.AV2Log(TABLES, IMAGES, sp, l)
        e = tab["Argoverse2"][u]
        maps = argo2.undistort_maps(g) if args.version == "undist" else None
        P2 = np.zeros((3, 4)); P2[:3, :3] = np.asarray(e["K"], float)
        calib = Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))
        acc = {m.name: ([], [], {}, None) for m in modes}
        for i in range(len(g.frames)):
            img = argo2.load_image(g, i, maps)
            for m in modes:
                dets, geos, scal, prev = acc[m.name]
                d, _ = det.detect(img, m)
                geos.append(mono.predicted_geometry(d, prev, calib, e["cam_h"], e["dt"]))
                dets.append(d)
                for k in ("n_cand", "n_cand_raw", "n_post"):
                    scal.setdefault(k, []).append(d[k])
                scal.setdefault("img_h", []).append(img.shape[0]); scal.setdefault("img_w", []).append(img.shape[1])
                acc[m.name] = (dets, geos, scal, d)
        for m in modes:
            dets, geos, scal, _ = acc[m.name]
            cache.save(out / m.name / f"{u}.npz", list(range(len(g.frames))), dets, geos, scal)
        nf += len(g.frames)
        print(f"  {sp} {u}: {len(g.frames)} frames, {nf} in all [{time.time() - t0:.0f}s]", flush=True)
    (run / "summary.json").write_text(json.dumps({"frames": nf, "seconds": round(time.time() - t0, 1)}, indent=1))


if __name__ == "__main__":
    main()
