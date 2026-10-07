#!/usr/bin/env python
"""Follow-up E5: run further detectors (other capacities, architectures, fine-tuned weights) over nuScenes keyframes.

Exactly 40_nusc_detect's per-frame path (TensorRT engine, `TwoFidelityDetector.detect`, the monocular lift with the
previous frame's detections of the same detector), for one detector and one mode per call, over the scenes of a split
list. Writes <out>/<name>/<scene>.npz, where <name> is --name (default <weights stem>__<mode>), so that
`decision.build` can pair any two folders of one cache. Protocol: docs/followup_protocol.md, Section 8.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap import cache, mono, runmeta                     # noqa: E402
from rap.detect import MODES, TwoFidelityDetector        # noqa: E402
from rap.nusc import FRAME_DT, NuScenesDB                # noqa: E402
from rap.paths import CACHE, MODELS                      # noqa: E402

CAM_HEIGHT = 1.51   # as 40_nusc_detect


def scene_index(dataroot, version, want):
    """{scene: (image paths in temporal order, CAM_FRONT intrinsics of its first keyframe)} -- built in a child process
    (below), so the full nuScenes tables (about 10 GB resident on this board) are gone before any engine is loaded."""
    db = NuScenesDB(dataroot, version)
    out = {}
    for sc in db.scenes:
        name = db.scene_name(sc)
        if name in want:
            toks = db.samples(sc)
            out[name] = ([str(db.image_path(t)) for t in toks], db.calib(toks[0]).P2[:3, :3].tolist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", required=True)
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--weights", required=True, help="e.g. yolov8m.pt (looked up in the model folder if relative)")
    ap.add_argument("--engine_stem", default=None, help="engine file stem if it differs from the weights' stem")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--scenes", required=True, help="CSV with 'scene' and optionally 'deep_split', 'group'")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], help="deep_split values of new scenes to run")
    ap.add_argument("--out", default=str(CACHE / "xl_models"))
    ap.add_argument("--only", default=None, help="comma-separated scene names (checks)")
    args = ap.parse_args()

    w = Path(args.weights)
    w = w if w.is_absolute() else MODELS / w
    name = args.name or f"{w.stem}__{args.mode}"
    run = runmeta.new_run("xl_detect_models", vars(args))
    sp = pd.read_csv(args.scenes)
    sp = sp[(sp.group == "new765") & sp.deep_split.isin(args.splits)]
    want = set(sp.scene) if not args.only else set(sp.scene) & set(args.only.split(","))
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        index = ex.submit(scene_index, args.dataroot, args.version, want).result()
    from rap.kitti import Calib
    det = TwoFidelityDetector(str(w), backend="trt", engine_stem=args.engine_stem)
    mode = MODES[args.mode]
    outdir = Path(args.out) / name
    tot_f, tot_t, kept = 0, 0.0, []
    for sname in sorted(index):
        if (outdir / f"{sname}.npz").exists():
            kept.append(sname); continue
        paths, K = index[sname]
        P2 = np.zeros((3, 4)); P2[:3, :3] = np.asarray(K, float)
        calib = Calib(P2=P2, R_rect=np.eye(4), Tr_velo_cam=np.eye(4), Tr_imu_velo=np.eye(4))
        dets, geos, scal, prev = [], [], {}, None
        t0 = time.time()
        for path in paths:
            img = cv2.imread(path)
            d, _ = det.detect(img, mode)
            g = mono.predicted_geometry(d, prev, calib, CAM_HEIGHT, FRAME_DT)
            dets.append(d); geos.append(g); prev = d
            for k in ("n_cand", "n_cand_raw", "n_post"):
                scal.setdefault(k, []).append(d[k])
            scal.setdefault("img_h", []).append(img.shape[0]); scal.setdefault("img_w", []).append(img.shape[1])
        cache.save(outdir / f"{sname}.npz", list(range(len(paths))), dets, geos, scal)
        tot_f += len(paths); tot_t += time.time() - t0; kept.append(sname)
    (run / "summary.json").write_text(json.dumps({"name": name, "scenes": len(kept), "frames": tot_f,
                                                  "seconds": tot_t}, indent=1))
    print(f"{name}: {len(kept)} scenes, {tot_f} new frames in {tot_t:.0f}s -> {outdir}")


if __name__ == "__main__":
    main()
