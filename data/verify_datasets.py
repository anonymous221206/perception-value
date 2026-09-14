#!/usr/bin/env python
"""Check that the datasets are laid out where the pipeline expects them (see data/README.md).

Only needed for `reproduce.py --tier full`; the cached tier needs no dataset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rap.paths import DATA as KITTI, DATASETS, MODELS                          # noqa: E402

CHECKS = {
    "nuScenes v1.0-trainval metadata": DATASETS / "nuscenes/trainval/v1.0-trainval/sample.json",
    "nuScenes CAM_FRONT keyframes (trainval blob 01)": DATASETS / "nuscenes/trainval/samples/CAM_FRONT",
    "nuScenes map expansion v1.3 (PKL/TIP rasters)": DATASETS / "nuscenes/trainval/maps/expansion",
    "KITTI tracking left colour images": KITTI / "training/image_02",
    "KITTI tracking labels": KITTI / "training/label_02",
    "KITTI tracking calibration": KITTI / "training/calib",
    "KITTI tracking OXTS": KITTI / "training/oxts",
    "nuPlan v1.1 mini split databases": DATASETS / "nuplan/nuplan-v1.1/splits/mini",
    "nuPlan maps v1.0": DATASETS / "nuplan/nuplan-maps-v1.0",
    "nuPlan CAM_F0 scenario-window images (scripts/112)": DATASETS / "nuplan/sensor_blobs_cam_f0",
    "YOLOv8s weights": MODELS / "yolov8s.pt",
    "RT-DETR-l weights": MODELS / "rtdetr-l.pt",
}

if __name__ == "__main__":
    missing = 0
    for name, p in CHECKS.items():
        ok = p.exists()
        missing += not ok
        print(f"  [{'ok' if ok else 'MISSING'}] {name}: {p}")
    sys.exit(1 if missing else 0)
