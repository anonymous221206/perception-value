#!/usr/bin/env python
"""Follow-up E1: the pixel router R2 on the follow-up scenes, by 107's own stages.

107's image cache, training (MobileNetV2 at 128 x 128, one head per nuScenes cell, masked BCE on 1[V > 0], AdamW,
30 epochs, seed 0, train + validation frames) and TensorRT export run unchanged; only their inputs are redirected:
the image cache is built from the follow-up detection cache and data root, and the labels are the follow-up cells
(new train / validation / test scenes). Scores land in the run folder as scores__nuScenes__<cell>.npz, which 172 reads
with --r2 <run>. Stages, each its own process as in 107:  --stage cache | train | export.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames, runmeta, xl                                              # noqa: E402
from rap.paths import CACHE, NUSCENES_XL                                           # noqa: E402

t107 = xl.load_script("t107", "107_router_r2.py")
CACHE_DIR = Path(CACHE) / "xl" / "router_r2"


def stage_cache(det, dataroot):
    import cv2
    from rap.cache import DetCache
    from rap.nusc import NuScenesDB
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = CACHE_DIR / "nuScenes.npz"
    if out.exists():
        print("  exists", out); return
    db = NuScenesDB(dataroot, "v1.0-trainval")
    name_to_scene = {db.scene_name(s): s for s in db.scenes}
    seqs, frames_, imgs = [], [], []
    t0 = time.time()
    for f in sorted((Path(det) / "ns_cheap_320").glob("*.npz")):
        c = DetCache(f)
        toks = db.samples(name_to_scene[f.stem])
        for fr in c.frames:
            fr = int(fr)
            im = cv2.imread(str(db.image_path(toks[fr])), cv2.IMREAD_COLOR)
            assert im is not None
            imgs.append(cv2.resize(im, (t107.SIDE, t107.SIDE), interpolation=cv2.INTER_AREA)[:, :, ::-1])
            seqs.append(f.stem); frames_.append(fr)
    np.savez_compressed(out, seq=np.array(seqs, "U32"), frame=np.array(frames_), img=np.stack(imgs))
    print(f"  {len(imgs)} images cached in {time.time() - t0:.0f}s -> {out}")


def make_labels(tables, det):
    def labels(dataset, splits):
        assert dataset == "nuScenes"
        sp = xl.scene_split()
        new = sp[sp.group == "new765"]
        xl.use_ego_table(Path(tables) / "v_ego_causal.csv")
        feats, fcols = xl.features(det)
        cells = list(xl.cells(tables, feats, fcols, dict(zip(new.scene, new.deep_split))))
        full = max(cells, key=lambda c: len(c["d"]))
        base = full["d"][["seq", "frame", "unit", "split"]].astype({"seq": str, "frame": int, "unit": str}).copy()
        heads = []
        for c in cells:
            name = f"{c['geometry']}__{c['system']}__{c['target']}"
            d = c["d"][["seq", "frame"]].astype({"seq": str, "frame": int}).copy()
            d[f"V__{name}"] = (c["d"][c["cheap"]] - c["d"][c["full"]]).to_numpy(float)
            d[f"Jc__{name}"] = c["d"][c["cheap"]].to_numpy(float)
            base = base.merge(d, on=["seq", "frame"], how="left", validate="one_to_one")
            heads.append(name)
        return base, heads
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cache", "train", "export"])
    ap.add_argument("--tables", default=str(Path(CACHE) / "xl" / "tables850"))
    ap.add_argument("--det", default=str(Path(CACHE) / "nusc_det_xl"))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--run", default=None)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    t107.CACHE_DIR = CACHE_DIR
    t107.labels = make_labels(args.tables, args.det)
    if args.stage == "cache":
        return stage_cache(args.det, args.dataroot)
    a = argparse.Namespace(dataset="nuScenes", run=args.run, reuse_pt=None, tag="xl_router_r2", objective="V",
                           label_run=None, fit_units="trainval", no_official=True)
    if args.stage == "train":
        run = runmeta.new_run("xl_router_r2", vars(args))
        a.run = str(run)
        t107.stage_train(a)
        print("RUN", run)
    else:
        t107.stage_export(a)


if __name__ == "__main__":
    main()
