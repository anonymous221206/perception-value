"""Canonical paths. Scripts import from here, so no script hard-codes a machine path.

Datasets and model weights are located through environment variables (see configs/paths.env.example); the
defaults point inside the repository:

    RAP_DATASETS  datasets root holding nuscenes/, kitti_tracking/ and nuplan/     (default: <repo>/datasets)
    RAP_MODELS    detector weights (yolov8s.pt, rtdetr-l.pt) and engines/          (default: <repo>/models)
    RAP_KITTI     KITTI tracking root, if not under RAP_DATASETS/kitti_tracking

nuScenes v1.0-trainval is read from RAP_DATASETS/nuscenes/trainval (NUSCENES_TRAINVAL).
"""
from pathlib import Path
import os

REPO = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("RAP_DATASETS", REPO / "datasets"))
MODELS = Path(os.environ.get("RAP_MODELS", REPO / "models"))
DATA = Path(os.environ.get("RAP_KITTI", DATASETS / "kitti_tracking"))

KITTI_IMAGES = DATA / "training" / "image_02"
KITTI_LABELS = DATA / "training" / "label_02"
KITTI_CALIB = DATA / "training" / "calib"
KITTI_OXTS = DATA / "training" / "oxts"
NUSCENES_TRAINVAL = DATASETS / "nuscenes" / "trainval"

CACHE = REPO / "data" / "cache"
RESULTS = REPO / "results"
RAW = RESULTS / "raw"
PROCESSED = RESULTS / "processed"
FIGURES = RESULTS / "figures"
DOCS = REPO / "docs"

for _p in (CACHE, RAW, PROCESSED, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
