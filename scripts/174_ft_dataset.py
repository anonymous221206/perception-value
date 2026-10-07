#!/usr/bin/env python
"""Follow-up E5: the fine-tuning dataset (YOLO format) from the benchmark's own reference boxes.

Labels are exactly the boxes the benchmark matches detections against (`NuScenesDB.sequence_geometry`: projected 3D
boxes of the evaluated categories with at least one lidar or radar point), kept when at least `RiskConfig.min_gt_height`
(10 px) high, with COCO class ids so the detector keeps its 80-class head:

  human.pedestrian -> person (0); vehicle.bicycle -> bicycle (1); vehicle.car -> car (2);
  vehicle.motorcycle -> motorcycle (3); vehicle.bus -> bus (5); vehicle.truck / trailer / construction -> truck (7)

Writes <out>/labels/{train,val}/<image stem>.txt (class cx cy w h, normalised by 1600 x 900), <out>/images_{split}.txt
(the CAM_FRONT file names) and <out>/summary.json. The images themselves are the dataset's files, linked on the training
machine by 175. Protocol: docs/followup_protocol.md, Section 8.
"""
from __future__ import annotations

import argparse, json, sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.nusc import IMG_H, IMG_W, NuScenesDB                                   # noqa: E402
from rap.paths import NUSCENES_TRAINVAL                                        # noqa: E402
from rap.risk import RiskConfig                                                # noqa: E402

COCO_ID = [("human.pedestrian", 0), ("vehicle.bicycle", 1), ("vehicle.car", 2), ("vehicle.motorcycle", 3),
           ("vehicle.bus", 5), ("vehicle.truck", 7), ("vehicle.trailer", 7), ("vehicle.construction", 7)]


def coco_id(category: str) -> int:
    for prefix, i in COCO_ID:
        if category.startswith(prefix):
            return i
    raise ValueError(f"category outside the evaluated set: {category}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default=str(NUSCENES_TRAINVAL))
    ap.add_argument("--split_csv", default=str(ROOT / "results" / "followup" / "E1" / "scene_split.csv"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    sp = pd.read_csv(args.split_csv)
    sp = sp[sp.group == "new765"]
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    by_name = {db.scene_name(s): s for s in db.scenes}
    cat_of_inst = {tok: db._t["category"][r["category_token"]]["name"] for tok, r in db._t["instance"].items()}
    minh = RiskConfig().min_gt_height
    summary = {}
    for split in ("train", "val"):
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        names = sorted(sp[sp.deep_split == split].scene)
        images, boxes, classes = [], 0, Counter()
        for name in names:
            scene = by_name[name]
            g = db.sequence_geometry(scene)
            inv = {i: t for t, i in db._track_map.get(name, {}).items()}
            toks = db.samples(scene)
            for fi, tk in enumerate(toks):
                img = db.image_path(tk)
                rows = g[(g["frame"] == fi) & ((g["y2"] - g["y1"]) >= minh)]
                lines = []
                for r in rows:
                    c = coco_id(cat_of_inst[inv[int(r["track_id"])]])
                    x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
                    lines.append(f"{c} {(x1 + x2) / 2 / IMG_W:.6f} {(y1 + y2) / 2 / IMG_H:.6f} "
                                 f"{(x2 - x1) / IMG_W:.6f} {(y2 - y1) / IMG_H:.6f}")
                    classes[c] += 1
                (out / "labels" / split / f"{img.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
                images.append(img.name)
                boxes += len(lines)
        (out / f"images_{split}.txt").write_text("\n".join(images) + "\n")
        summary[split] = {"scenes": len(names), "images": len(images), "boxes": boxes,
                          "boxes_per_class": {str(k): v for k, v in sorted(classes.items())}}
        print(split, summary[split], flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
