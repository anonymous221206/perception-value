#!/usr/bin/env python
"""Follow-up E5, X7 planner rows (protocol Section 8): PKL planner inputs for every detector of the X7 pairs.

E1's test rasters (`data/cache/xl/planner_d/test/chunk_{variant}_n60cXX.npz`, written by 70 over the 850 scenes) hold,
per frame, the ground-truth raster and the YOLOv8s 192x320 / 384x640 rasters at 0.25. A raster's five channels are the
road, road divider and lane divider (map, identical for every detector), the object channel and the ego box. Here only
the object channel is redrawn, from one detector's submission (60, `--roles full --op_conf_full <thr>`), with 70's
own path (devkit `load_prediction`, `add_center_dist`, `filter_eval_boxes`, PKL's `get_other_objs` / `raster_render`);
the other four channels are copied from the ground-truth raster. Frames: the 228 new test scenes of E1's split.

  render   --mode <xl_models folder> --thr <t> --subs <dir of 60's output>
           -> data/cache/xl/x7_planner/rasters/<mode>@<thr>/<variant>_cXX.npz (sample_token, packed)
  check    redraws E1's own YOLOv8s 384x640 rasters from E1's submissions; must be byte-identical (gate for this script)
  assemble --old <key> --new <key> (key = <mode>@<thr>; yolov8s__ns_cheap_320@0.25 and yolov8s__ns_full_640@0.25 are E1's)
           -> data/cache/xl/x7_planner/pairs/<old>__<new>/test/chunk_{variant}_cXX.npz, the layout 74 reads
"""
from __future__ import annotations

import argparse, importlib.util, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
_s = importlib.util.spec_from_file_location("t70", ROOT / "scripts" / "70_planner_d_data.py")
t70 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(t70)
L, NuScenes, DetectionBox, config_factory = t70.L, t70.NuScenes, t70.DetectionBox, t70.config_factory
from planning_centric_metrics.planning_kl import get_other_objs, raster_render, samp2ego   # noqa: E402
from rap.paths import CACHE, NUSCENES_XL                                           # noqa: E402

E1 = Path(CACHE) / "xl" / "planner_d" / "test"
OUT = Path(CACHE) / "xl" / "x7_planner"
SPLIT = ROOT / "results" / "followup" / "E1" / "scene_split.csv"
VARIANTS = ("oracle", "mono")
E1_KEYS = {"yolov8s__ns_cheap_320@0.25": "packed_cheap", "yolov8s__ns_full_640@0.25": "packed_full"}
CELLS = 5 * t70.NX * t70.NY
OBJ = 3                                                     # raster_render's channel order: road, divs, objects, ego


def test_scenes():
    sp = pd.read_csv(SPLIT)
    return set(sp[(sp.group == "new765") & (sp.deep_split == "test")].scene)


def e1_chunks(variant):
    return sorted(E1.glob(f"chunk_{variant}_n60c*.npz"))


def unpack(p):
    return np.unpackbits(p)[:CELLS].reshape(5, t70.NX, t70.NY)


def redraw(gt_packed, samp, nusc, boxes):
    """The ground-truth raster with its object channel replaced by `boxes` (70's `render` minus the map query)."""
    ego = samp2ego(samp, nusc)
    lobjs, lws = get_other_objs(boxes, np.array([ego["x"], ego["y"], ego["hcos"], ego["hsin"]]))
    empty = {k: [] for k in t70.LAYERS + t70.LINES}
    obj = raster_render(empty, [ego["l"], ego["w"]], lobjs, lws, t70.NX, t70.NY, t70.LAYERS, t70.LINES,
                        t70.BX, t70.DX).astype(np.uint8)[OBJ]
    r = unpack(gt_packed).copy()
    r[OBJ] = obj
    return np.packbits(r.astype(bool).reshape(-1))


def predictions(nusc, path, tokens):
    dcfg = config_factory("detection_cvpr_2019")
    p, _ = L.load_prediction(str(path), dcfg.max_boxes_per_sample, DetectionBox, verbose=False)
    p = t70.subset(p, tokens)
    L.add_center_dist(nusc, p)
    return L.filter_eval_boxes(nusc, p, dcfg.class_range, verbose=False)


def render(nusc, sub_of, outdir, limit=0, compare=None):
    keep = test_scenes()
    outdir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        files = e1_chunks(variant)[: limit or None]
        toks = []
        for f in files:
            z = np.load(f)
            toks += [t for t, s in zip(z["sample_token"], z["scene"]) if s in keep]
        preds = predictions(nusc, sub_of(variant), toks)
        for f in files:
            o = outdir / f"{variant}_{f.stem.split('_')[-1]}.npz"
            if o.exists() and compare is None:
                continue
            z = np.load(f)
            m = np.array([s in keep for s in z["scene"]], bool)
            st = z["sample_token"][m]
            packed = np.stack([redraw(g, nusc.get("sample", t), nusc, preds[t]) for t, g in zip(st, z["packed_gt"][m])]) \
                if m.any() else np.zeros((0, z["packed_gt"].shape[1]), np.uint8)
            if compare is not None:
                same = np.array_equal(packed, z[compare][m])
                print(f"  {variant} {f.name}: {m.sum()} frames, identical to E1's {compare}: {same}", flush=True)
                if not same:
                    return False
                continue
            np.savez_compressed(o, sample_token=st, packed=packed)
            print(f"  {variant} {f.name}: {m.sum()} test frames", flush=True)
    return True


def assemble(old, new):
    keep = test_scenes()
    d = OUT / "pairs" / f"{old}__{new}" / "test"
    d.mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        for f in e1_chunks(variant):
            z = np.load(f)
            m = np.array([s in keep for s in z["scene"]], bool)
            out = {k: z[k][m] for k in ("sample_token", "scene", "target", "ego_v", "packed_gt")}
            for role, key in (("cheap", old), ("full", new)):
                if key in E1_KEYS:
                    out[f"packed_{role}"] = z[E1_KEYS[key]][m]
                else:
                    r = np.load(OUT / "rasters" / key / f"{variant}_{f.stem.split('_')[-1]}.npz")
                    assert np.array_equal(r["sample_token"], out["sample_token"]), (key, f.name)
                    out[f"packed_{role}"] = r["packed"]
            np.savez_compressed(d / f"chunk_{variant}_{f.stem.split('_')[-1]}.npz", **out)
    print("assembled", d.parent)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["render", "check", "assemble"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--mode"); ap.add_argument("--thr", type=float); ap.add_argument("--subs")
    ap.add_argument("--old"); ap.add_argument("--new")
    ap.add_argument("--limit", type=int, default=0, help="E1 chunks (check)")
    args = ap.parse_args()
    if args.action == "assemble":
        return assemble(args.old, args.new)
    nusc = NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    if args.action == "check":
        subs = Path(CACHE) / "xl" / "nusc_submissions"
        ok = render(nusc, lambda v: subs / f"{v}__ns_full_640.json", OUT / "check", args.limit, compare="packed_full")
        print("X7_RASTER_CHECK", "identical" if ok else "DIFFER")
        sys.exit(0 if ok else 1)
    key = f"{args.mode}@{args.thr:g}"
    render(nusc, lambda v: Path(args.subs) / f"{v}__{args.mode}.json", OUT / "rasters" / key)
    print("rendered", key)


if __name__ == "__main__":
    main()
