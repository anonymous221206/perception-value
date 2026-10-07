#!/usr/bin/env python
"""Follow-up E2: the benchmark's nuScenes engines on the CAM_F0 image of every NAVSIM token (addendum E2 in docs/followup_protocol.md).

YOLOv8s TensorRT FP16, 192x320 (cheap) and 384x640 (full), letterboxed from 1920x1080, decoding as the benchmark
(candidate floor 0.10; the operating threshold 0.25 is applied by the scorer). Writes
data/cache/navsim/dets/<log>.pkl = {token: {"cheap": det, "full": det, "cam": (K, D, R, t)}}, det = {xyxy, conf,
coarse, binent} in image pixels.
"""
from __future__ import annotations

import argparse, pickle, sys, time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                   # noqa: E402
from rap.detect import MODES, TwoFidelityDetector         # noqa: E402
from rap.paths import MODELS, NAVSIM_ROOT                 # noqa: E402

OUT = ROOT / "data" / "cache" / "navsim"
MODES_USED = {"cheap": "ns_cheap_320", "full": "ns_full_640"}
KEYS = ("xyxy", "conf", "coarse", "binent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blobs", default=str(NAVSIM_ROOT / "sensor_blobs"))
    ap.add_argument("--splits", nargs="+", default=["test", "dev_train", "dev_val"])
    args = ap.parse_args()
    run = runmeta.new_run("navsim_detect", vars(args))
    tok = pd.read_pickle(OUT / "tokens.pkl")
    tok = tok[tok.split.isin(args.splits)]
    det = TwoFidelityDetector(str(MODELS / "yolov8s.pt"), backend="trt")
    (OUT / "dets").mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    for log, g in tok.groupby("log"):
        f = OUT / "dets" / f"{log}.pkl"
        if f.exists():
            continue
        out = {}
        for r in g.itertuples():
            img = cv2.imread(str(Path(args.blobs) / r.folder / r.image))
            assert img is not None, r.image
            e = {"cam": (r.K, r.D, r.R, r.t)}
            for br, m in MODES_USED.items():
                d, _ = det.detect(img, MODES[m])
                e[br] = {k: np.asarray(d[k]) for k in KEYS}
            out[r.token] = e
            n += 1
        pickle.dump(out, open(f, "wb"))
        print(f"  {log}: {len(g)} tokens  [{n} done, {time.time() - t0:.0f}s]", flush=True)
    (run / "summary.txt").write_text(f"{n} tokens in {time.time() - t0:.0f}s\n")


if __name__ == "__main__":
    main()
