#!/usr/bin/env python
"""Task 23 Phase 2: the gates before any regenerated figure is read, and measurements B and D.

Pre-registered in the pre-registration record (not part of this release) (Task 23, and its amendment of 2026-09-21 17:10).  Parts, each its own invocation:

  g1geo     G1, geometry: in every official cache, the ego-frame code path with the identity transform reproduces the
            stored camera-frame geometry bitwise; G3 (ttc invariant) is asserted inside every re-lift, identity and ego
  b         B: the rejected variant F (camera-frame lift, then SE(3) on the fused and corner points) against the chosen
            lift, over every cached detection; max and p99 by range band; the revisit rule
  d         D: near-range bias against the reference geometry, cheap and full, camera and ego, paired and per mode;
            the symmetry criterion
  g2        G2: rotation = identity with Task 22's translation reproduces run 20260920_133624_lift_offset_outcomes
  tables    G1 (tables) / G4 / G5: compare two runs' per-frame tables (see `--help`)

Every part writes a CSV into this run and exits non-zero when its gate fails.
"""
from __future__ import annotations

import argparse, gzip, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, frames, geometry as G, planner as P, runmeta          # noqa: E402
from rap.cache import GEO_ARRAYS, DetCache, camera_for                          # noqa: E402
from rap.mono import box_iou                                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                  # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

RAW = ROOT / "results" / "raw"
# every cache mode an official stage reads, by track
KITTI_MODES = [("det", "cheap_320"), ("det", "cheap_384"), ("det", "full_640"), ("det512", "cheap_512"),
               ("det512", "full_640"), ("rtdetr_kitti", "rt_cheap_320"), ("rtdetr_kitti", "rt_full_640"),
               ("rtdetr_kitti_mid", "rt_mid_480"), ("rtdetr_kitti_mid", "rt_full_640")]
NUSC_MODES = [("nusc_det_tv", "ns_cheap_320"), ("nusc_det_tv", "ns_full_640")]
# the geometry-bearing pairs: (track, label, cheap dir/mode, full dir/mode)
PAIRS = [("KITTI", "Y8 320->640", ("det", "cheap_320"), ("det", "full_640")),
         ("KITTI", "Y8 384->640", ("det", "cheap_384"), ("det", "full_640")),
         ("KITTI", "Y8 512->640", ("det512", "cheap_512"), ("det512", "full_640")),
         ("KITTI", "RT 320->640", ("rtdetr_kitti", "rt_cheap_320"), ("rtdetr_kitti", "rt_full_640")),
         ("KITTI", "RT 480->640", ("rtdetr_kitti_mid", "rt_mid_480"), ("rtdetr_kitti_mid", "rt_full_640")),
         ("nuScenes", "Y8 320->640", ("nusc_det_tv", "ns_cheap_320"), ("nusc_det_tv", "ns_full_640"))]
B_BANDS = [(0, 15), (15, 30), (30, 50)]
D_BANDS = [(0, 15), (15, 30), (30, 50), (50, 1e9)]
B_Z, B_LAT = 0.25, 0.10                  # the revisit rule's materiality, pre-registered
D_TOL, D_MIN_N = 0.25, 100               # the symmetry criterion, pre-registered
THR, IOU = RiskConfig().op_conf, 0.5


def units(track):
    base = Path(CACHE) / ("det" if track == "KITTI" else "nusc_det_tv") / ("cheap_320" if track == "KITTI" else "ns_cheap_320")
    return [p.stem for p in sorted(base.glob("*.npz"))]


def band(x, bands):
    out = np.full(len(x), "", dtype=object)
    for lo, hi in bands:
        out[(x >= lo) & (x < hi)] = f"{lo:g}-{hi:g}" if hi < 1e8 else f">{lo:g}"
    return out


def gate(ok, msg):
    if not ok:
        raise SystemExit(f"GATE FAILED: {msg}")


# ----------------------------------------------------------------------------------------------- G1 geometry, G3

def part_g1geo(run):
    rows = []
    for track, modes in (("KITTI", KITTI_MODES), ("nuScenes", NUSC_MODES)):
        for d, m in modes:
            n_det = n_units = bad = 0
            for u in units(track):
                f = Path(CACHE) / d / m / f"{u}.npz"
                cam = DetCache(f, frame="camera")
                idn = DetCache(f, frame="identity")              # G3 asserted inside the re-lift
                DetCache(f, frame="ego")                         # and under the real transform
                for k in GEO_ARRAYS:
                    if cam.z[f"geo_{k}"].tobytes() != idn.z[f"geo_{k}"].tobytes():
                        bad += 1
                n_det += len(cam.z["conf"])
                n_units += 1
            rows.append({"track": track, "mode": f"{d}/{m}", "units": n_units, "detections": n_det,
                         "fields_differing_bitwise": bad, "g3_ttc_asserted_identity_and_ego": True, "ok": bad == 0})
            print(f"  G1 {track:8s} {d}/{m:13s} units {n_units:3d} detections {n_det:7d} "
                  f"fields differing {bad}  {'ok' if bad == 0 else 'FAILED'}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(run / "g1_geometry.csv", index=False)
    gate(bool(t.ok.all()), "G1 (geometry): the identity transform does not reproduce the stored lift; see g1_geometry.csv")


# ----------------------------------------------------------------------------------------------- B

def variant_f(cam, det, geo_cam):
    """The rejected variant: the camera-frame lift, then the full SE(3) on the fused point and the corner points."""
    R, t = cam.transform("ego")
    c = cam.calib
    x1, y1, x2, y2 = det["xyxy"].astype(np.float64).T
    z = geo_cam["z"].astype(np.float64)
    yb = (y2 - c.cy) * z / c.fy
    def ego(X, Y, Z, row):
        return R[row, 0] * X + R[row, 1] * Y + R[row, 2] * Z + t[row]
    zf = ego((0.5 * (x1 + x2) - c.cx) * z / c.fx, yb, z, 0)
    yl, yr = ego((x1 - c.cx) * z / c.fx, yb, z, 1), ego((x2 - c.cx) * z / c.fx, yb, z, 1)
    return zf, np.minimum(yl, yr), np.maximum(yl, yr)


def part_b(run):
    diffs = []
    for track, modes in (("KITTI", KITTI_MODES), ("nuScenes", NUSC_MODES)):
        for d, m in modes:
            for u in units(track):
                f = Path(CACHE) / d / m / f"{u}.npz"
                c, e = DetCache(f, frame="camera"), DetCache(f, frame="ego")
                cam = camera_for(f)
                for i in range(len(c)):
                    det = c.det(i)
                    if not len(det["conf"]):
                        continue
                    zf, lo_f, hi_f = variant_f(cam, det, c.geo_camera(i))
                    g = e.geo(i)
                    diffs.append(pd.DataFrame({"track": track, "mode": f"{d}/{m}", "z": g["z"].astype(float),
                                               "dz": np.abs(g["z"] - zf), "dlat_min": np.abs(g["lat_min"] - lo_f),
                                               "dlat_max": np.abs(g["lat_max"] - hi_f)}))
    D = pd.concat(diffs, ignore_index=True)
    D["band"] = band(D.z.to_numpy(), B_BANDS)
    D = D[D.band != ""]
    rows = []
    for (tr, b), s in D.groupby(["track", "band"]):
        r = {"track": tr, "band": b, "n": len(s)}
        for k in ("dz", "dlat_min", "dlat_max"):
            r[f"{k}_max"], r[f"{k}_p99"] = float(s[k].max()), float(np.percentile(s[k], 99))
        r["material"] = bool(r["dz_p99"] > B_Z or r["dlat_min_p99"] > B_LAT or r["dlat_max_p99"] > B_LAT)
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(run / "b_variant_difference.csv", index=False)
    print(t.round(3).to_string(index=False), flush=True)
    material = t[t.material]
    if material.empty:
        print("  B: no band is material on either track; the lift stands", flush=True)
        (run / "b_reading.json").write_text(json.dumps({"material_bands": 0, "decision_1": "stands"}, indent=1))
        return
    # where material: the chosen lift must be at least as accurate as F against the reference, cheap and full alike
    acc = accuracy_vs_reference(run)
    verdict = []
    for r in material.itertuples():
        for mode in ("cheap", "full"):
            a = acc[(acc.track == r.track) & (acc.band == r.band) & (acc.mode_role == mode)]
            ok = bool(len(a)) and float(a.mae_chosen.iloc[0]) <= float(a.mae_f.iloc[0])
            verdict.append({"track": r.track, "band": r.band, "mode": mode, "ok": ok,
                            "mae_chosen": float(a.mae_chosen.iloc[0]) if len(a) else np.nan,
                            "mae_f": float(a.mae_f.iloc[0]) if len(a) else np.nan})
    v = pd.DataFrame(verdict)
    v.to_csv(run / "b_revisit_check.csv", index=False)
    print(v.round(3).to_string(index=False), flush=True)
    (run / "b_reading.json").write_text(json.dumps(
        {"material_bands": int(len(material)), "decision_1": "stands" if bool(v.ok.all()) else "revisit"}, indent=1))
    gate(bool(v.ok.all()), "B: in a material band the chosen lift is less accurate than F; decision 1 is revisited")


def reference(track, u, ad=None):
    return G.sequence_geometry(u) if track == "KITTI" else ad.geometry(u)


def matched(det, geo, g, thr=THR, with_iou=False):
    """(detection index, reference index) of kept detections whose best reference IoU is >= 0.5 -- the same rule
    `decision._apply_range_source` uses to give a detection reference geometry."""
    k = np.flatnonzero(det["conf"] >= thr)
    if not len(k) or not len(g):
        e = np.zeros(0, int)
        return (e, e, np.zeros(0)) if with_iou else (e, e)
    gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
    iou = box_iou(det["xyxy"][k].astype(float), gt)
    ok = iou.max(1) >= IOU
    out = (k[ok], iou.argmax(1)[ok])
    return out + (iou.max(1)[ok],) if with_iou else out


def accuracy_vs_reference(run):
    """B's revisit check: median |range error| of the chosen lift and of F, matched detections, by band and mode."""
    rows = []
    ad = nusc_adapter()
    for track, _, cheap, full in (PAIRS[0], PAIRS[-1]):
        for role, (d, m) in (("cheap", cheap), ("full", full)):
            for u in units(track):
                f = Path(CACHE) / d / m / f"{u}.npz"
                c, e, cam = DetCache(f, frame="camera"), DetCache(f, frame="ego"), camera_for(f)
                geom = reference(track, u, ad)
                min_h = RiskConfig().min_gt_height
                for i, fr in enumerate(c.frames):
                    g = geom[geom["frame"] == int(fr)]
                    g = g[(g["y2"] - g["y1"]) >= min_h]
                    det = c.det(i)
                    di, gi = matched(det, None, g)
                    if not len(di):
                        continue
                    zf, _, _ = variant_f(cam, {"xyxy": det["xyxy"][di]}, {"z": c.geo_camera(i)["z"][di]})
                    ref = g["long_near"][gi].astype(float)
                    rows.append(pd.DataFrame({"track": track, "mode_role": role, "ref": ref,
                                              "err_chosen": np.abs(e.geo(i)["z"][di] - ref), "err_f": np.abs(zf - ref)}))
    A = pd.concat(rows, ignore_index=True)
    A["band"] = band(A.ref.to_numpy(), B_BANDS)
    t = (A[A.band != ""].groupby(["track", "band", "mode_role"])
         .agg(n=("ref", "size"), mae_chosen=("err_chosen", "median"), mae_f=("err_f", "median")).reset_index())
    t.to_csv(run / "b_accuracy_vs_reference.csv", index=False)
    return t


_AD = None


def nusc_adapter():
    global _AD
    if _AD is None:
        from rap.nusc import NuScenesDB, make_adapter
        _AD = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
    return _AD


# ----------------------------------------------------------------------------------------------- D

def part_d(run):
    ad = nusc_adapter()
    min_h = RiskConfig().min_gt_height
    paired, own = [], []
    for track, label, cheap, full in PAIRS:
        for u in units(track):
            geom = reference(track, u, ad)
            cc = {role: (DetCache(Path(CACHE) / d / m / f"{u}.npz", frame="camera"),
                         DetCache(Path(CACHE) / d / m / f"{u}.npz", frame="ego"))
                  for role, (d, m) in (("cheap", cheap), ("full", full))}
            fc = cc["full"][0]
            for i, fr in enumerate(cc["cheap"][0].frames):
                fr = int(fr)
                j = fc.index.get(fr)
                if j is None:
                    continue
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= min_h]
                if not len(g):
                    continue
                per = {}
                for role, idx in (("cheap", i), ("full", j)):
                    cam, ego = cc[role]
                    det = cam.det(idx)
                    di, gi, sc = matched(det, None, g, with_iou=True)
                    zc, ze = cam.geo(idx)["z"], ego.geo(idx)["z"]
                    best = {}
                    for a, b, s in zip(di, gi, sc):          # the best-IoU detection per reference object
                        if b not in best or s > best[b][0]:
                            best[b] = (s, a)
                    per[role] = {b: (float(zc[a]), float(ze[a])) for b, (_, a) in best.items()}
                    for b, (c_, e_) in per[role].items():
                        ref = float(g["long_near"][b])
                        own.append((track, label, role, ref, c_ - ref, e_ - ref))
                for b in set(per["cheap"]) & set(per["full"]):
                    ref = float(g["long_near"][b])
                    (cc_, ce_), (fc_, fe_) = per["cheap"][b], per["full"][b]
                    paired.append((track, label, ref, cc_ - ref, ce_ - ref, fc_ - ref, fe_ - ref))
        print(f"  D {track} {label}: {sum(1 for p in paired if p[0] == track and p[1] == label)} paired objects",
              flush=True)
    Pd = pd.DataFrame(paired, columns=["track", "pair", "ref", "cheap_cam", "cheap_ego", "full_cam", "full_ego"])
    Od = pd.DataFrame(own, columns=["track", "pair", "mode", "ref", "err_cam", "err_ego"])
    Pd["band"], Od["band"] = band(Pd.ref.to_numpy(), D_BANDS), band(Od.ref.to_numpy(), D_BANDS)
    rows = []
    for (tr, pr, b), s in Pd.groupby(["track", "pair", "band"]):
        r = {"track": tr, "pair": pr, "band": b, "n_paired": len(s)}
        for k in ("cheap_cam", "full_cam", "cheap_ego", "full_ego"):
            r[f"bias_{k}"] = float(s[k].median())
        r["asym_cam"] = r["bias_full_cam"] - r["bias_cheap_cam"]
        r["asym_ego"] = r["bias_full_ego"] - r["bias_cheap_ego"]
        tested = b in ("0-15", "15-30") and len(s) >= D_MIN_N
        r["tested"], r["ok"] = tested, (not tested) or abs(r["asym_ego"]) <= D_TOL
        rows.append(r)
    t = pd.DataFrame(rows)
    o = (Od.groupby(["track", "pair", "mode", "band"])
           .agg(n=("ref", "size"), bias_cam=("err_cam", "median"), bias_ego=("err_ego", "median"),
                mae_cam=("err_cam", lambda v: float(np.median(np.abs(v)))),
                mae_ego=("err_ego", lambda v: float(np.median(np.abs(v))))).reset_index())
    t.to_csv(run / "d_bias_paired.csv", index=False)
    o.to_csv(run / "d_bias_per_mode.csv", index=False)
    print(t.round(3).to_string(index=False), flush=True)
    gate(bool(t.ok.all()), "D: the near-range bias is not symmetric between the cheap and full modes; see d_bias_paired.csv")


# ----------------------------------------------------------------------------------------------- G2

G2_RUN = RAW / "20260920_133624_lift_offset_outcomes"
G2_KITTI = {"Y8_320": ("det", "cheap_320", "full_640"), "Y8_384": ("det", "cheap_384", "full_640"),
            "Y8_512": ("det512", "cheap_512", "full_640"), "RT_320": ("rtdetr_kitti", "rt_cheap_320", "rt_full_640"),
            "RT_480": ("rtdetr_kitti_mid", "rt_mid_480", "rt_full_640")}


def part_g2(run):
    import importlib.util
    s = importlib.util.spec_from_file_location("b65", ROOT / "scripts" / "65_planner_b_decision.py")
    b65 = importlib.util.module_from_spec(s)
    s.loader.exec_module(b65)
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    pb, cb = b65.B.PARAMS_B["static_obstacles"], b65.B.COSTS_B["default"]
    factory = lambda fr: (lambda path, tag: DetCache(Path(path), frame=fr))
    rows = []
    kseqs = units("KITTI")
    settings = [(p, "mono") for p in G2_KITTI] + [("Y8_320", "oracle")]
    for pair, geo in settings:
        dd, cm, fm = G2_KITTI[pair]
        for flag, fr in (("off", "camera"), ("on", "task22")):
            a = decision.build(Path(CACHE) / dd, cm, fm, kseqs, cfg, pp, cp, G.PRIMARY, range_source=geo,
                               adapter=decision.KittiAdapter, cache_factory=factory(fr))
            b = b65.build_b(Path(CACHE) / dd, cm, fm, kseqs, cfg, decision.KittiAdapter, geo, pb, cb,
                            cache_factory=factory(fr))
            t = a[["seq", "frame", "v_ego", "J_cheap", "J_full", "n_cheap", "n_full"]].merge(
                b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], validate="one_to_one")
            rows.append(compare_text(t, f"outcomes__{pair}__{geo}__{flag}.csv.gz", run))
    ad = nusc_adapter()
    nseqs = units("nuScenes")
    for geo in ("mono", "oracle"):
        for flag, fr in (("off", "camera"), ("on", "task22")):
            a = decision.build(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640", nseqs, cfg, pp, cp,
                               G.PRIMARY, range_source=geo, adapter=ad, cache_factory=factory(fr))
            t = a[["seq", "frame", "v_ego", "J_cheap", "J_full", "n_cheap", "n_full"]].copy()
            t["JB_cheap"] = t["JB_full"] = np.nan
            rows.append(compare_text(t, f"outcomes__NS_320__{geo}__{flag}.csv.gz", run))
    t = pd.DataFrame(rows)
    t.to_csv(run / "g2_task22.csv", index=False)
    print(t.to_string(index=False), flush=True)
    gate(bool(t.identical.all()), "G2: the new code with Task 22's translation does not reproduce its tables")


def compare_text(t, name, run):
    new = t.to_csv(index=False)
    with gzip.open(G2_RUN / name, "rt") as fh:
        old = fh.read()
    with gzip.open(run / name, "wt") as fh:
        fh.write(new)
    print(f"  G2 {name:40s} {'identical' if new == old else 'DIFFERS'}", flush=True)
    return {"file": name, "rows": len(t), "identical": new == old}


# ----------------------------------------------------------------------------------------------- tables

# the per-frame tables a regenerating stage writes; summaries, manifests and run metadata are not per-frame tables
PER_FRAME = ("*.pkl", "outcomes__*.csv.gz", "*__c*__f*.npz", "mono__*.json", "oracle__*.json")


def part_tables(run, a, b, kind, label):
    """G1 (tables) building block: every per-frame table the regenerated run `b` wrote, against the shipped `a`."""
    a, b = Path(a), Path(b)
    rows = []
    files = sorted({f for pat in PER_FRAME for f in b.glob(pat)})
    for f in files:
        rel = f.relative_to(b)
        g = a / rel
        if not g.exists():
            rows.append({"file": str(rel), "status": "absent from the shipped run", "identical": False})
            continue
        rows.append({"file": str(rel), "status": "compared", "identical": same(g, f, kind)})
    if not rows:
        rows.append({"file": "(none)", "status": "no per-frame table found in the regenerated run", "identical": False})
    t = pd.DataFrame(rows)
    t.to_csv(run / f"{label}.csv", index=False)
    print(f"  {label}: {int(t.identical.sum())}/{len(t)} files identical", flush=True)
    return t


def same(f, g, kind):
    if f.suffix == ".pkl":
        x, y = pd.read_pickle(f), pd.read_pickle(g)
        return list(x.columns) == list(y.columns) and x.to_csv(index=False) == y.to_csv(index=False)
    if f.suffix == ".npz":
        x, y = np.load(f, allow_pickle=False), np.load(g, allow_pickle=False)
        return sorted(x.files) == sorted(y.files) and all(x[k].tobytes() == y[k].tobytes() and x[k].dtype == y[k].dtype
                                                          for k in x.files)
    if f.name.endswith(".gz"):
        return gzip.open(f).read() == gzip.open(g).read()
    return f.read_bytes() == g.read_bytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["g1geo", "b", "d", "g2", "tables"])
    ap.add_argument("--a", help="tables: the shipped run (or directory)")
    ap.add_argument("--b", help="tables: the regenerated run (or directory)")
    ap.add_argument("--label", default="tables")
    args = ap.parse_args()
    run = runmeta.new_run(f"ego_frame_gate_{args.part}", vars(args))
    if args.part == "g1geo":
        part_g1geo(run)
    elif args.part == "b":
        part_b(run)
    elif args.part == "d":
        part_d(run)
    elif args.part == "g2":
        part_g2(run)
    else:
        t = part_tables(run, args.a, args.b, "exact", args.label)
        gate(bool(t.identical.all()), f"{args.label}: files differ; see {args.label}.csv")
    print("wrote", run)


if __name__ == "__main__":
    main()
