#!/usr/bin/env python
"""Follow-up E9: escalation of the range estimator -- the ground-plane lift or Depth Anything V2's metric depth.

addendum E9 in docs/followup_protocol.md. The detections are the benchmark's cached ones; only the range and lateral extent
of each kept detection change. Stages (Jetson unless noted):

  gate_frames  the 50 gate frames per dataset -> <work>/gate_frames.json, images copied to <work>/gate_images/
               (213 --stage reference runs on them on the training machine)
  infer        --dataset --size: per detection of the 640 and 320 caches, the median depth of its central region
               -> <work>/<dataset>/<size>/<unit>.npz; the gate frames' per-box ranges -> <work>/gate_trt_<size>_<dataset>.npz
  gate         E9-0: TensorRT FP16 against 213's PyTorch FP32 reference -> results/followup/E9/gate_e9_0.json
  tables       per cell (cheap source -> full source) x mode x dataset -> <work>/tables/*.pkl
  settings     results/followup/E9/settings.csv, range_error.csv
  cost         results/followup/E9/costs.csv (latency and module energy per frame, Jetson)
"""
from __future__ import annotations

import argparse, json, shutil, sys, threading, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import cache, decision, frames, kitti, mono, planner as P, runmeta, xl              # noqa: E402
from rap.cache import DetCache                                                               # noqa: E402
from rap.mono import box_iou                                                                 # noqa: E402
from rap.paths import CACHE, MODELS, NUSCENES_XL, RESULTS                                    # noqa: E402
from rap.risk import RiskConfig                                                              # noqa: E402
from rap.scoring import EPS                                                                  # noqa: E402

WORK = Path(CACHE) / "xl" / "e9"
OUT = Path(RESULTS) / "followup" / "E9"
ENGINES = MODELS / "depth" / "engines"
SHAPES = {"nuScenes": (294, 518), "KITTI": (154, 518)}
DETS = {"nuScenes": Path(CACHE) / "nusc_det_xl", "KITTI": Path(CACHE) / "det"}
MODES = {"nuScenes": ("ns_full_640", "ns_cheap_320"), "KITTI": ("full_640", "cheap_320")}
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
CELLS = (("lift", "small"), ("lift", "large"), ("small", "large"), ("lift", "oracle"))


def role(mode):
    return "cheap" if "320" in mode else "full"


def prepare(img_bgr, h, w):
    """213's input: RGB, bilinear resize, /255, ImageNet normalisation, NCHW float32."""
    import cv2
    x = cv2.resize(img_bgr[:, :, ::-1], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    return ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def region_median(depth, xyxy):
    """Median depth over the middle 50% of the box's width, rows 50-90% of its height from the top."""
    H, W = depth.shape
    out = np.full(len(xyxy), np.nan, np.float32)
    for i, (x1, y1, x2, y2) in enumerate(np.asarray(xyxy, float)):
        w, h = x2 - x1, y2 - y1
        c0, c1 = int(np.floor(x1 + 0.25 * w)), int(np.ceil(x1 + 0.75 * w))
        r0, r1 = int(np.floor(y1 + 0.5 * h)), int(np.ceil(y1 + 0.9 * h))
        c0, r0 = min(max(c0, 0), W - 1), min(max(r0, 0), H - 1)
        c1, r1 = min(max(c1, c0 + 1), W), min(max(r1, r0 + 1), H)
        out[i] = np.median(depth[r0:r1, c0:c1])
    return out


def units_and_images(dataset):
    """{unit: [image path per cached frame]}, in the cache's frame order."""
    if dataset == "KITTI":
        return {s: [str(kitti.image_path(s, int(f))) for f in DetCache(DETS["KITTI"] / "full_640" / f"{s}.npz").frames]
                for s in kitti.sequences()}
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:   # frees the tables at once
        return ex.submit(_nusc_images, str(NUSCENES_XL)).result()


def _nusc_images(root):
    from rap.nusc import NuScenesDB
    db = NuScenesDB(root, "v1.0-trainval")
    want = {p.stem for p in (DETS["nuScenes"] / "ns_full_640").glob("*.npz")}
    return {db.scene_name(s): [str(db.image_path(t)) for t in db.samples(s)] for s in db.scenes
            if db.scene_name(s) in want}


def gate_frames(index):
    keys = [(u, i) for u in sorted(index) for i in range(len(index[u]))]
    return [keys[int(round(p))] for p in np.linspace(0, len(keys) - 1, 50)]


def stage_gate_frames(args):
    work = Path(args.work)
    (work / "gate_images").mkdir(parents=True, exist_ok=True)
    out = {}
    for ds in SHAPES:
        idx = units_and_images(ds)
        sel = gate_frames(idx)
        names = []
        for u, i in sel:
            src = Path(idx[u][i])
            dst = work / "gate_images" / f"{ds}__{u}__{i:04d}{src.suffix}"
            shutil.copy(src, dst)
            names.append(dst.name)
        out[ds] = {"frames": [[u, i] for u, i in sel], "images": names}
    (work / "gate_frames.json").write_text(json.dumps(out, indent=1))
    print("wrote", work / "gate_frames.json")


def engine(size, dataset):
    from rap.trt import TRTModule
    h, w = SHAPES[dataset]
    return TRTModule(ENGINES / f"{size}_{h}x{w}.engine")


def depth_of(net, img, h, w):
    import cv2, torch
    x = torch.from_numpy(prepare(img, h, w)).cuda()
    d = net(x).float().reshape(h, w).cpu().numpy()
    return cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)


def stage_infer(args):
    import cv2
    ds, size = args.dataset, args.size
    h, w = SHAPES[ds]
    idx = units_and_images(ds)
    net = engine(size, ds)
    out = Path(args.work) / ds / size
    out.mkdir(parents=True, exist_ok=True)
    gsel = {(u, i) for u, i in gate_frames(idx)}
    gate_rows = []
    t0, nf = time.time(), 0
    for u in sorted(idx):
        f = out / f"{u}.npz"
        caches = {m: DetCache(DETS[ds] / m / f"{u}.npz") for m in MODES[ds]}
        if f.exists() and not (gsel & {(u, i) for i in range(len(idx[u]))}):
            continue
        res = {m: [] for m in MODES[ds]}
        for i, p in enumerate(idx[u]):
            depth = depth_of(net, cv2.imread(p), h, w)
            for m, c in caches.items():
                res[m].append(region_median(depth, c.det(i)["xyxy"]))
            if (u, i) in gsel:
                gate_rows.append((u, i, res[MODES[ds][0]][-1]))
            nf += 1
        np.savez_compressed(f, **{m: (np.concatenate(v) if v else np.zeros(0, np.float32)) for m, v in res.items()})
        if nf and nf % 2000 < len(idx[u]):
            print(f"  {ds} {size}: {nf} frames [{time.time() - t0:.0f}s]", flush=True)
    np.savez_compressed(Path(args.work) / f"gate_trt_{size}_{ds}.npz",
                        units=np.array([r[0] for r in gate_rows]), frames=np.array([r[1] for r in gate_rows]),
                        offsets=np.concatenate([[0], np.cumsum([len(r[2]) for r in gate_rows])]),
                        z=np.concatenate([r[2] for r in gate_rows]) if gate_rows else np.zeros(0))
    print(f"  {ds} {size}: {nf} frames in {time.time() - t0:.0f}s", flush=True)


def stage_gate(args):
    import cv2
    work = Path(args.work)
    gf = json.loads((work / "gate_frames.json").read_text())
    res = {}
    for ds in SHAPES:
        frames_ = [tuple(x) for x in gf[ds]["frames"]]
        for size in ("small", "large"):
            ref = np.load(work / f"ref_{size}_{ds}.npz")
            trt = np.load(work / f"gate_trt_{size}_{ds}.npz")
            pos = {(str(u), int(i)): k for k, (u, i) in enumerate(zip(trt["units"], trt["frames"]))}
            rel = []
            for j, (u, i) in enumerate(frames_):
                img = cv2.imread(str(work / "gate_images" / gf[ds]["images"][j]))
                d = cv2.resize(ref["depth"][j], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                zr = region_median(d, DetCache(DETS[ds] / MODES[ds][0] / f"{u}.npz").det(i)["xyxy"])
                k = pos[(u, i)]
                zt = trt["z"][int(trt["offsets"][k]):int(trt["offsets"][k + 1])]
                rel.append(np.abs(zt - zr) / np.maximum(zr, 1e-3))
            rel = np.concatenate(rel)
            res[f"{ds}|{size}"] = {"boxes": int(len(rel)), "median_rel_diff": float(np.median(rel)),
                                   "p95_rel_diff": float(np.percentile(rel, 95))}
    ok = all(r["median_rel_diff"] < 0.01 for r in res.values())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_e9_0.json").write_text(json.dumps({"pass": ok, "cells": res}, indent=1))
    print("gate E9-0:", "pass" if ok else "FAIL", res)
    return ok


def require_gate():
    f = OUT / "gate_e9_0.json"
    if not f.exists() or not json.loads(f.read_text())["pass"]:
        raise SystemExit("gate E9-0 has not passed")


# ------------------------------------------------------------------------------------------------ tables

def ranged(src, d, geo, k, g, cam, zdep, rng):
    """(range, lat_min, lat_max, ttc) of the kept detections under one range source."""
    bx = d["xyxy"][k].astype(float)
    geo_k = {kk: vv[k] for kk, vv in geo.items()}
    if src in ("lift", "oracle"):
        return decision._apply_range_source(bx, geo_k, g, "mono" if src == "lift" else "oracle", rng, 0.0)
    zc = np.clip(zdep[src][k].astype(float), 0.5, 200.0)
    K = cam.calib
    x1, y1, x2, y2 = bx.T if len(bx) else (np.zeros(0),) * 4
    uc = 0.5 * (x1 + x2)
    yb = (y2 - K.cy) * zc / K.fy
    xs = [(x - K.cx) * zc / K.fx for x in (uc, x1, x2)]
    z = mono._ego_x(cam.R, cam.t, xs[0], yb, zc)
    yl, yr = mono._ego_y(cam.R, cam.t, xs[1], yb, zc), mono._ego_y(cam.R, cam.t, xs[2], yb, zc)
    return z, np.minimum(yl, yr), np.maximum(yl, yr), geo_k["ttc"].copy()


def adapter_for(ds):
    if ds == "KITTI":
        return decision.KittiAdapter, kitti.sequences()
    from rap.nusc import NuScenesDB, make_adapter
    db = NuScenesDB(NUSCENES_XL, "v1.0-trainval")
    return make_adapter(db), sorted(p.stem for p in (DETS[ds] / MODES[ds][0]).glob("*.npz"))


def stage_tables(args):
    require_gate()
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    work = Path(args.work)
    (work / "tables").mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        ad, seqs = adapter_for(ds)
        for mode in MODES[ds]:
            thr = cfg.thr(role(mode))
            for cs, fs in CELLS:
                t0, rows, err = time.time(), [], []
                rng = np.random.default_rng(0)
                for s in seqs:
                    p = DETS[ds] / mode / f"{s}.npz"
                    c, cam = DetCache(p), cache.camera_for(p)
                    zall = {sz: np.load(work / ds / sz / f"{s}.npz")[mode] for sz in ("small", "large")}
                    geom, speeds = ad.geometry(s), ad.speeds(s)
                    prev = None
                    for i, fr in enumerate(c.frames):
                        fr = int(fr)
                        v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
                        g = geom[geom["frame"] == fr]
                        g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
                        d, geo = c.det(i), c.geo(i)
                        sl = c._slice(i)
                        k = d["conf"] >= thr
                        zdep = {sz: z[sl] for sz, z in zall.items()}
                        rc = ranged(cs, d, geo, k, g, cam, zdep, rng)
                        rf = ranged(fs, d, geo, k, g, cam, zdep, rng)
                        a_c, _ = P.required_decel(*rc, v, pp)
                        a_f, _ = P.required_decel(*rf, v, pp)
                        a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                        act_c, act_f = P.discrete_action(a_c, pp), P.discrete_action(a_f, pp)
                        Jc = P.decision_cost(act_c, a_gt, prev, pp, cp)
                        Jf = P.decision_cost(act_f, a_gt, prev, pp, cp)
                        rows.append({"seq": s, "frame": fr, "v_ego": v, "a_cheap": a_c, "a_full": a_f, "a_gt": a_gt,
                                     "act_cheap": act_c, "act_full": act_f, "J_cheap": Jc["J"], "J_full": Jf["J"],
                                     "n_kept": int(k.sum())})
                        prev = act_c
                        if (cs, fs) == CELLS[0] and k.any() and len(g):
                            bx = d["xyxy"][k].astype(float)
                            gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
                            iou = box_iou(bx, gt)
                            m = iou.max(1) >= 0.5
                            if m.any():
                                zt = g["long_near"][iou.argmax(1)[m]].astype(float)
                                for src in ("lift", "small", "large"):
                                    z = ranged(src, d, geo, k, g, cam, zdep, rng)[0][m]
                                    err.append(pd.DataFrame({"source": src, "rel": (z - zt) / np.maximum(zt, 1e-6),
                                                             "seq": s}))
                f = work / "tables" / f"{ds}__{mode}__{cs}to{fs}.pkl"
                pd.DataFrame(rows).to_pickle(f)
                if err:
                    pd.concat(err).to_pickle(work / "tables" / f"{ds}__{mode}__range_error.pkl")
                print(f"  {f.name}: {len(rows)} frames [{time.time() - t0:.0f}s]", flush=True)


def populations(ds):
    if ds == "KITTI":
        seqs = kitti.sequences() or [f"{i:04d}" for i in range(21)]    # without the dataset: the 21 training sequences
        return {"kitti21": set(seqs), "kitti_test6": {s for i, s in enumerate(seqs) if i % 10 >= 7}}
    sp = xl.scene_split()
    new, old = sp[sp.group == "new765"], sp[sp.group == "frozen85"]
    return {"new228_test": set(new[new.deep_split == "test"].scene), "new765": set(new.scene),
            "frozen85": set(old.scene), "pooled850": set(sp.scene)}


def stage_settings(args):
    require_gate()
    work = Path(args.work)
    meta = xl.scene_meta(NUSCENES_XL)
    log_of = dict(zip(meta.scene, meta.log))
    rows, errs = [], []
    for f in sorted((work / "tables").glob("*to*.pkl")):
        ds, mode, cell = f.stem.split("__")
        d = pd.read_pickle(f)
        for pop, units in populations(ds).items():
            x = d[d.seq.isin(units)]
            v = (x.J_cheap - x.J_full).to_numpy(float)
            c = x.J_cheap.to_numpy(float)
            r = xl.describe(v, c, (0.2,))
            r["n_units"] = int(x.seq.nunique())
            r.update(xl.bootstrap_describe(v, c, x.seq.to_numpy(), args.nboot, np.random.default_rng(0), (0.2,)))
            if ds == "nuScenes":
                lg = xl.bootstrap_describe(v, c, x.seq.map(log_of).to_numpy(), args.nboot, np.random.default_rng(0),
                                           (0.2,))
                r.update({k.replace("_lo", "_lo_log").replace("_hi", "_hi_log"): y for k, y in lg.items()})
            rows.append({"dataset": ds, "mode": mode, "cell": cell, "population": pop,
                         "primary": cell == "lifttosmall" and "640" in mode, **r})
            print(f"  {ds} {mode} {cell} {pop}: harmed {r['harmed_share']:.3f} rho {r['rho']:.3f} "
                  f"all-full {r['all_full_reduction']:+.3f}", flush=True)
    for f in sorted((work / "tables").glob("*__range_error.pkl")):
        ds, mode, _ = f.stem.split("__")
        e = pd.read_pickle(f)
        for pop, units in populations(ds).items():
            for src, g in e[e.seq.isin(units)].groupby("source"):
                q = g.rel.quantile([0.25, 0.5, 0.75]).to_numpy()
                errs.append({"dataset": ds, "mode": mode, "population": pop, "source": src, "n": len(g),
                             "rel_err_p25": q[0], "rel_err_median": q[1], "rel_err_p75": q[2],
                             "abs_rel_err_median": float(g.rel.abs().median())})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "settings.csv", index=False)
    pd.DataFrame(errs).to_csv(OUT / "range_error.csv", index=False)
    print("wrote", OUT / "settings.csv")


# ------------------------------------------------------------------------------------------------ cost

def stage_cost(args):
    """Per frame, end to end (image decoding, preprocessing, engine, resize, box medians), 250 repetitions after 40
    warm-up, 3 rounds (the first discarded); module energy = sum over rails of (mean - idle mean) x median ms."""
    import cv2
    from rap import power
    rows = []
    for ds in SHAPES:
        h, w = SHAPES[ds]
        idx = units_and_images(ds)
        u = sorted(idx)[0]
        c = DetCache(DETS[ds] / MODES[ds][0] / f"{u}.npz")
        paths = idx[u]
        idle = sample_rails(power, 5.0)
        for size in ("small", "large"):
            net = engine(size, ds)
            for rnd in range(3):
                ms, samples, stop = [], [], threading.Event()

                def poll():
                    while not stop.is_set():
                        samples.append(power.read_power_mw())
                        time.sleep(0.05)
                th = threading.Thread(target=poll)
                for k in range(40):
                    depth_of(net, cv2.imread(paths[k % len(paths)]), h, w)
                th.start()
                for k in range(250):
                    t0 = time.perf_counter()
                    img = cv2.imread(paths[k % len(paths)])
                    region_median(depth_of(net, img, h, w), c.det(k % len(c))["xyxy"])
                    ms.append((time.perf_counter() - t0) * 1e3)
                stop.set(); th.join()
                mean = {r: float(np.mean([s[r] for s in samples])) for r in idle}
                rows.append({"dataset": ds, "size": size, "round": rnd, "ms_median": float(np.median(ms)),
                             "mJ": sum(mean[r] - idle[r] for r in idle) * float(np.median(ms)) / 1e3})
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "costs_rounds.csv", index=False)
    t[t["round"] > 0].groupby(["dataset", "size"])[["ms_median", "mJ"]].median().reset_index().to_csv(
        OUT / "costs.csv", index=False)
    print(t.to_string(index=False))


def sample_rails(power, seconds):
    s, t0 = [], time.time()
    while time.time() - t0 < seconds:
        s.append(power.read_power_mw()); time.sleep(0.05)
    return {r: float(np.mean([x[r] for x in s])) for r in s[0]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["gate_frames", "infer", "gate", "tables", "settings", "cost"], required=True)
    ap.add_argument("--dataset", choices=list(SHAPES))
    ap.add_argument("--size", choices=["small", "large"])
    ap.add_argument("--datasets", nargs="+", default=["KITTI", "nuScenes"])
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--nboot", type=int, default=1000)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"e9_{args.stage}", vars(args))
    ok = {"gate_frames": stage_gate_frames, "infer": stage_infer, "gate": stage_gate, "tables": stage_tables,
          "settings": stage_settings, "cost": stage_cost}[args.stage](args)
    (run / "summary.json").write_text(json.dumps({"stage": args.stage, "ok": ok is not False}, indent=1))
    if ok is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
