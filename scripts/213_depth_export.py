#!/usr/bin/env python
"""Follow-up E9, on the training machine: Depth Anything V2 (metric, outdoor) to ONNX for the Jetson's TensorRT, and the
PyTorch FP32 reference of gate E9-0. addendum E9 in docs/followup_protocol.md.

  export      <out>/<size>_<h>x<w>.onnx for each model size and each dataset's input shape, static shape, opset 16
              (the Jetson's TensorRT 8.5 has no LayerNormalization importer, so the layer norms are exported decomposed)
  reference   PyTorch FP32 depth on the gate frames (input prepared exactly as 212 prepares it) -> <out>/ref_<size>_<dataset>.npz
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np
import torch
import transformers

MODELS = {"small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
          "large": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"}
REVISIONS = {"small": "fd2c22027eaf20374204f14099b8341e1925ad39",    # the revisions E9 ran with (export_meta.json)
             "large": "4eab4cf1983c2801c515804005214de56a4b67cc"}
SHAPES = {"nuScenes": (294, 518), "KITTI": (154, 518)}
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).predicted_depth


def load(size):
    from transformers import AutoModelForDepthEstimation
    m = AutoModelForDepthEstimation.from_pretrained(MODELS[size], revision=REVISIONS[size]).eval()
    # transformers < 4.45 ignores the checkpoint's metric settings and builds the relative head (ReLU, no 80 m scale)
    assert getattr(m.config, "depth_estimation_type", None) == "metric" and m.config.max_depth == 80, "relative head"
    return Wrap(m)


def prepare(img_bgr, h, w):
    """212's input: RGB, bilinear resize to (h, w), /255, ImageNet normalisation, NCHW float32."""
    import cv2
    x = cv2.resize(img_bgr[:, :, ::-1], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["export", "reference"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--images", help="reference: JSON {dataset: [image paths]} of the gate frames")
    ap.add_argument("--opset", type=int, default=16)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for size in MODELS:
        net = load(size)
        meta[size] = {"repo": MODELS[size], "revision": getattr(net.m.config, "_commit_hash", None), "opset": args.opset,
                      "transformers": transformers.__version__}
        for ds, (h, w) in SHAPES.items():
            if args.stage == "export":
                f = out / f"{size}_{h}x{w}.onnx"
                torch.onnx.export(net, torch.zeros(1, 3, h, w), str(f), opset_version=args.opset, input_names=["images"],
                                  output_names=["depth"], do_constant_folding=True)
                print("wrote", f)
            else:
                import cv2
                paths = json.loads(Path(args.images).read_text())[ds]
                with torch.no_grad():
                    d = np.stack([net(torch.from_numpy(prepare(cv2.imread(p), h, w)))[0].numpy() for p in paths])
                np.savez_compressed(out / f"ref_{size}_{ds}.npz", depth=d.astype(np.float32), paths=np.array(paths))
                print("reference", size, ds, d.shape)
    (out / f"{args.stage}_meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
