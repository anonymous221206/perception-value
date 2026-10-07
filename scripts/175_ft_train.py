#!/usr/bin/env python
"""Follow-up E5: fine-tune a COCO YOLOv8 detector on the new nuScenes training scenes (recipe fixed in the protocol).

Run on the training machine, inside a folder that holds:
  labels/{train,val}/*.txt and images_{train,val}.txt   (174's output)
  --images  the folder of nuScenes CAM_FRONT keyframes
It links images/{train,val}/<file> to the dataset files (YOLO finds labels by replacing 'images' with 'labels'),
writes data.yaml with COCO's 80 class names, and trains with the recipe of docs/followup_protocol.md, Section 8:
Ultralytics 8.3.40, input 640, 15 epochs, batch 16 (fallback 8, nominal batch 64), SGD lr0 0.01 momentum 0.937
weight decay 5e-4, 3 warm-up epochs, default augmentation, seed 0, deterministic; best.pt by Ultralytics' detection
fitness on the validation scenes.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with 174's labels and image lists")
    ap.add_argument("--images", required=True, help="folder of CAM_FRONT keyframes")
    ap.add_argument("--weights", required=True, help="COCO weights, e.g. yolov8m.pt")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    import ultralytics
    from ultralytics import YOLO
    from ultralytics.utils import yaml_load
    assert ultralytics.__version__ == "8.3.40", ultralytics.__version__

    root = Path(args.root).resolve()
    for split in ("train", "val"):
        d = root / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        for name in (root / f"images_{split}.txt").read_text().split():
            link = d / name
            if not link.exists():
                os.symlink(Path(args.images).resolve() / name, link)
    names = yaml_load(Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml")["names"]
    data = root / "data.yaml"
    data.write_text(json.dumps({"path": str(root), "train": "images/train", "val": "images/val", "names": names}))

    stem = Path(args.weights).stem
    run = f"{stem}_ft_s{args.seed}_b{args.batch}"
    YOLO(args.weights).train(data=str(data), imgsz=640, epochs=args.epochs, batch=args.batch, nbs=64,
                             optimizer="SGD", lr0=0.01, momentum=0.937, weight_decay=5e-4, warmup_epochs=3,
                             seed=args.seed, deterministic=True, device=args.device, workers=args.workers,
                             project=str(root / "runs"), name=run, exist_ok=False, plots=False)
    print("done", root / "runs" / run / "weights" / "best.pt")


if __name__ == "__main__":
    sys.exit(main())
