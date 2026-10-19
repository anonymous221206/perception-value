#!/usr/bin/env python
"""Follow-up E7, planner rows (secondary; addendum E7 in docs/followup_protocol.md): PKL's planner on the LiDAR boxes,
scored against the logged trajectory, on E1's 228 new test scenes (the planner's test frames).

The boxes are 210's, kept as for 211's braking rows (the eight traffic classes, score >= 0.25, CAM_FRONT's field of
view), written as a standard nuScenes detection submission (global frame, the LIDAR_TOP keyframe's ego pose). From
there the benchmark's path is unchanged: 187's object-channel redraw (devkit `load_prediction`, `filter_eval_boxes`,
PKL's `raster_render`) on E1's test rasters, and 74's PKL planner against the logged trajectory. LiDAR boxes are metric,
so the lift's two submission geometries do not apply; the rasters take E1's `oracle` chunks as their base (map, ego
and ground-truth channels, identical in both).

  subs      -> data/cache/xl/e7_planner/subs/<model>.json
  render    --model m -> data/cache/xl/e7_planner/rasters/<model>/oracle_cXX.npz
  assemble  --old m --new m -> data/cache/xl/e7_planner/pairs/<old>__<new>/test/chunk_oracle_cXX.npz (74's layout)
  rows      -> results/followup/E7/planner.csv (V = ADE / FDE of the cheap detector minus the full one's)
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


from rap import xl                                                                # noqa: E402
from rap.nusc import NuScenesDB                                                   # noqa: E402
from rap.nusc_submission import DEFAULT_ATTR                                      # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                 # noqa: E402

OUT = Path(CACHE) / "xl" / "e7_planner"
RES = Path(RESULTS) / "followup" / "E7" / "planner.csv"
MODELS = ("pp", "voxel", "transfusion")
PAIRS = (("pp", "voxel"), ("pp", "transfusion"))
VARIANT = "oracle"


def stage_subs(args):
    t187, t211 = _load("t187", "187_x7_planner_rasters.py"), _load("t211", "211_lidar_tables.py")
    nusc = t187.NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    keep = t187.test_scenes()
    (OUT / "subs").mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        results = {}
        for scene in sorted(keep):
            z = np.load(t211.DET / m / f"{scene}.npz")
            for k, tok in enumerate(z["tokens"]):
                tok = str(tok)
                sd = nusc.get("sample_data", nusc.get("sample", tok)["data"]["LIDAR_TOP"])
                pose = nusc.get("ego_pose", sd["ego_pose_token"])
                Rg = Rotation.from_quat(np.roll(pose["rotation"], -1))
                Rc, tc = db._cam_to_ego(tok)
                K = db.calib(tok).P2[:3, :3]
                s = slice(int(z["offsets"][k]), int(z["offsets"][k + 1]))
                boxes = []
                for b, sc, nm in zip(z["boxes"][s], z["score"][s], z["name"][s]):
                    nm = str(nm)
                    if nm not in t211.CLASSES or sc < t211.THR or not t211.in_view(t211.footprint(b)[1], Rc, tc, K):
                        continue
                    x, y, zc, l, w, h, yaw, vx, vy = map(float, b)
                    q = (Rg * Rotation.from_euler("z", yaw)).as_quat()           # x, y, z, w
                    v = Rg.as_matrix()[:2, :2] @ np.array([vx, vy])
                    boxes.append({"sample_token": tok, "translation": [float(c) for c in Rg.apply([x, y, zc]) +
                                                                        np.array(pose["translation"])],
                                  "size": [w, l, h], "rotation": [float(q[3]), float(q[0]), float(q[1]), float(q[2])],
                                  "velocity": [float(v[0]), float(v[1])], "detection_name": nm,
                                  "detection_score": float(sc), "attribute_name": DEFAULT_ATTR.get(nm, "")})
                results[tok] = boxes
        sub = {"meta": {"use_camera": False, "use_lidar": True, "use_radar": False, "use_map": False,
                        "use_external": False}, "results": results}
        (OUT / "subs" / f"{m}.json").write_text(json.dumps(sub))
        print(f"  {m}: {len(results)} keyframes, {sum(len(v) for v in results.values())} boxes", flush=True)


def stage_render(args):
    t187 = _load("t187", "187_x7_planner_rasters.py")
    nusc = t187.NuScenes(version="v1.0-trainval", dataroot=args.dataroot, verbose=False)
    keep = t187.test_scenes()
    out = OUT / "rasters" / args.model
    out.mkdir(parents=True, exist_ok=True)
    files = t187.e1_chunks(VARIANT)
    toks = []
    for f in files:
        z = np.load(f)
        toks += [t for t, s in zip(z["sample_token"], z["scene"]) if s in keep]
    preds = t187.predictions(nusc, OUT / "subs" / f"{args.model}.json", toks)
    for f in files:
        o = out / f"{VARIANT}_{f.stem.split('_')[-1]}.npz"
        if o.exists():
            continue
        z = np.load(f)
        m = np.array([s in keep for s in z["scene"]], bool)
        st = z["sample_token"][m]
        packed = np.stack([t187.redraw(g, nusc.get("sample", t), nusc, preds[t]) for t, g in zip(st, z["packed_gt"][m])]) \
            if m.any() else np.zeros((0, z["packed_gt"].shape[1]), np.uint8)
        np.savez_compressed(o, sample_token=st, packed=packed)
        print(f"  {args.model} {f.name}: {m.sum()} test frames", flush=True)
    (out / ".done").touch()


def stage_assemble(args):
    t187 = _load("t187", "187_x7_planner_rasters.py")
    keep = t187.test_scenes()
    d = OUT / "pairs" / f"{args.old}__{args.new}" / "test"
    d.mkdir(parents=True, exist_ok=True)
    for f in t187.e1_chunks(VARIANT):
        z = np.load(f)
        m = np.array([s in keep for s in z["scene"]], bool)
        out = {k: z[k][m] for k in ("sample_token", "scene", "target", "ego_v", "packed_gt")}
        for role, key in (("cheap", args.old), ("full", args.new)):
            r = np.load(OUT / "rasters" / key / f"{VARIANT}_{f.stem.split('_')[-1]}.npz")
            assert np.array_equal(r["sample_token"], out["sample_token"]), (key, f.name)
            out[f"packed_{role}"] = r["packed"]
        np.savez_compressed(d / f"chunk_{VARIANT}_{f.stem.split('_')[-1]}.npz", **out)
    print("assembled", d.parent)


def stage_rows(args):
    meta = xl.scene_meta(args.dataroot)
    log_of = dict(zip(meta.scene, meta.log))
    rows = []
    for old, new in PAIRS:
        f = OUT / "pairs" / f"{old}__{new}" / "planC_vs_truth.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f)
        for target in ("ade", "fde"):
            c, n = t[f"JC_{target}_cheap"].to_numpy(float), t[f"JC_{target}_full"].to_numpy(float)
            r = xl.describe(c - n, c, (0.2,))
            r.update(xl.bootstrap_describe(c - n, c, t.scene.to_numpy(), 1000, np.random.default_rng(0), (0.2,)))
            lg = xl.bootstrap_describe(c - n, c, t.scene.map(log_of).to_numpy(), 1000, np.random.default_rng(0), (0.2,))
            r.update({k.replace("_lo", "_lo_log").replace("_hi", "_hi_log"): y for k, y in lg.items()})
            rows.append({"cheap": old, "full": new, "target": target, "n_scenes": t.scene.nunique(),
                         "n_logs": t.scene.map(log_of).nunique(), "mean_cheap": float(c.mean()),
                         "mean_full": float(n.mean()), **r})
            print(f"  {old} -> {new} {target}: affected {r['affected']} harmed {r['harmed_share']:.3f} "
                  f"all-full {r['all_full_reduction']:+.3f}", flush=True)
    RES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(RES, index=False)
    print("wrote", RES, len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["subs", "render", "assemble", "rows"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--old", choices=MODELS); ap.add_argument("--new", choices=MODELS)
    args = ap.parse_args()
    {"subs": stage_subs, "render": stage_render, "assemble": stage_assemble, "rows": stage_rows}[args.stage](args)


if __name__ == "__main__":
    main()
