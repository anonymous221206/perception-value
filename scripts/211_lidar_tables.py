#!/usr/bin/env python
"""Follow-up E7: braking decisions from metric LiDAR boxes (210's detections), and the setting-level quantities.

addendum E7 in docs/followup_protocol.md. Per keyframe and detector: the traffic-class boxes at the operating point inside
CAM_FRONT's field of view (`rap.nusc`'s rule for the reference, applied to the detected boxes too), their footprint and
velocity TTC, the benchmark's braking controller on the shared action history. Stages:

  tables     <work>/<geometry>__<op>__<full>.pkl (geometry detected / oracle; op shared / equalcount; full voxel =
             VoxelNeXt, primary, or transfusion = TransFusion-L, secondary; cheap is CenterPoint-PointPillar)
  settings   results/followup/E7/settings.csv, detection.csv
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import planner as P, runmeta, xl                                        # noqa: E402
from rap.nusc import IMG_H, IMG_W, NuScenesDB, make_adapter                     # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                               # noqa: E402
from rap.scoring import EPS                                                     # noqa: E402

DET = Path(CACHE) / "e7_lidar"            # 210's <model>/<scene>.npz, copied from the training machine
WORK = Path(CACHE) / "xl" / "e7"
OUT = Path(RESULTS) / "followup" / "E7"
CLASSES = {"car", "truck", "construction_vehicle", "bus", "trailer", "motorcycle", "bicycle", "pedestrian"}
THR = 0.25
MATCH_M = 2.0


def in_view(corners_ego, Rc, tc, K):
    """`rap.nusc.sequence_geometry`'s rule: >= 4 of 8 corners in front of the camera, clipped box >= 4 px."""
    cam = (Rc.T @ (corners_ego - tc).T).T
    front = cam[:, 2] > 0.5
    if front.sum() < 4:
        return False
    uv = (K @ cam[front].T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1 = np.clip(uv.min(0), 0, [IMG_W - 1, IMG_H - 1])
    x2, y2 = np.clip(uv.max(0), 0, [IMG_W - 1, IMG_H - 1])
    return (x2 - x1) >= 4 and (y2 - y1) >= 4


def footprint(b):
    x, y, z, l, w, h, yaw = b[:7]
    c, s = np.cos(yaw), np.sin(yaw)
    xy = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    foot = np.stack([[x + c * a - s * bb, y + s * a + c * bb] for a, bb in xy])
    corners = np.vstack([np.c_[foot, np.full(4, z - h / 2)], np.c_[foot, np.full(4, z + h / 2)]])
    return foot, corners


def detections(db, model, scene, frame_of):
    """Per frame: list of (score, name, near, lat_min, lat_max, vx, cx, cy) inside the field of view."""
    z = np.load(DET / model / f"{scene}.npz")
    out = {}
    for k, tok in enumerate(z["tokens"]):
        s = slice(int(z["offsets"][k]), int(z["offsets"][k + 1]))
        Rc, tc = db._cam_to_ego(tok)
        K = db.calib(tok).P2[:3, :3]
        rows = []
        for b, sc, nm in zip(z["boxes"][s], z["score"][s], z["name"][s]):
            if str(nm) not in CLASSES:
                continue
            foot, corners = footprint(b)
            if not in_view(corners, Rc, tc, K):
                continue
            rows.append((float(sc), str(nm), foot[:, 0].min(), foot[:, 1].min(), foot[:, 1].max(), float(b[7]),
                         float(b[0]), float(b[1])))
        out[frame_of[str(tok)]] = rows
    return out


def branch(rows, thr, g, v, geometry):
    """(range, lat_min, lat_max, ttc) of the kept boxes; oracle: matched boxes take the reference geometry."""
    k = [r for r in rows if r[0] >= thr]
    if not k:
        e = np.zeros(0)
        return e, e, e, e, 0, 0
    near = np.array([r[2] for r in k]); lo = np.array([r[3] for r in k]); hi = np.array([r[4] for r in k])
    closing = v - np.array([r[5] for r in k])
    tt = np.where(closing > 1e-3, near / np.maximum(closing, 1e-9), np.inf)
    matched = np.zeros(len(k), bool)
    taken = np.zeros(len(g), bool)
    for i in np.argsort([-r[0] for r in k]):
        if not len(g):
            break
        d = np.hypot(g["long_cen"] - k[i][6], g["lat_cen"] - k[i][7])
        d[taken] = np.inf
        j = int(np.argmin(d))
        if d[j] <= MATCH_M:
            taken[j] = matched[i] = True
            if geometry == "oracle":
                near[i], lo[i], hi[i], tt[i] = g["long_near"][j], g["lat_min"][j], g["lat_max"][j], g["ttc"][j]
    return near, lo, hi, tt, int(matched.sum()), int(taken.sum())


def stage_tables(args):
    for m in ("pp", "voxel"):                          # gate E7-0 (210 --stage eval), copied with the detections
        g = DET / m / "gate_e7_0.json"
        if not g.exists() or not json.loads(g.read_text())["pass"]:
            raise SystemExit(f"gate E7-0 has not passed for {m}")
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    ad = make_adapter(db)
    by_name = {db.scene_name(s): s for s in db.scenes}
    scenes = sorted(p.stem for p in (DET / "pp").glob("*.npz"))
    pp, cp = P.PlannerParams(), P.CostParams()
    fulls = [m for m in ("voxel", "transfusion") if (DET / m).is_dir()]
    dets = {}
    for m in ["pp"] + fulls:
        for s in scenes:
            frame_of = {t: i for i, t in enumerate(db.samples(by_name[s]))}
            dets[(m, s)] = detections(db, m, s, frame_of)
    # equal box count (secondary): the full threshold whose count over the scored keyframes equals the cheap one's
    n_cheap = sum(sum(r[0] >= THR for r in rows) for s in scenes for rows in dets[("pp", s)].values())
    Path(args.work).mkdir(parents=True, exist_ok=True)
    rec = {"n_cheap_boxes": int(n_cheap)}
    for full in fulls:
        sv = np.sort([r[0] for s in scenes for rows in dets[(full, s)].values() for r in rows])[::-1]
        rec[f"{full}_threshold_equalcount"] = float(sv[n_cheap - 1]) if n_cheap <= len(sv) else float(sv[-1])
    (Path(args.work) / "operating_points.json").write_text(json.dumps(rec, indent=1))
    for full, op in [(f, o) for f in fulls for o in ("shared", "equalcount")]:
        tc_, tf_ = THR, (THR if op == "shared" else rec[f"{full}_threshold_equalcount"])
        for geometry in ("detected", "oracle"):
            t0 = time.time()
            rows = []
            for s in scenes:
                geom, speeds = ad.geometry(s), ad.speeds(s)
                prev = None
                for fr in sorted(dets[("pp", s)]):
                    g = geom[geom["frame"] == fr]
                    v = float(speeds[fr])
                    bc = branch(dets[("pp", s)][fr], tc_, g, v, geometry)
                    bf = branch(dets[(full, s)][fr], tf_, g, v, geometry)
                    a_c, _ = P.required_decel(*bc[:4], v, pp)
                    a_f, _ = P.required_decel(*bf[:4], v, pp)
                    a_gt, _ = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)
                    act_c, act_f = P.discrete_action(a_c, pp), P.discrete_action(a_f, pp)
                    Jc = P.decision_cost(act_c, a_gt, prev, pp, cp)
                    Jf = P.decision_cost(act_f, a_gt, prev, pp, cp)
                    rows.append({"seq": s, "frame": fr, "v_ego": v, "a_cheap": a_c, "a_full": a_f, "a_gt": a_gt,
                                 "act_cheap": act_c, "act_full": act_f, "J_cheap": Jc["J"], "J_full": Jf["J"],
                                 "n_cheap": len(bc[0]), "n_full": len(bf[0]), "tp_cheap": bc[4], "tp_full": bf[4],
                                 "n_gt": int(len(g))})
                    prev = act_c
            d = pd.DataFrame(rows)
            d.to_pickle(Path(args.work) / f"{geometry}__{op}__{full}.pkl")
            print(f"  {geometry} {op} {full}: {len(d)} frames, {d.seq.nunique()} scenes [{time.time() - t0:.0f}s]", flush=True)


def populations():
    """The two registered populations, then the registered strata of the primary one: E1's groups and split."""
    sp = xl.scene_split()
    val = sp[sp.official == "val"]
    pops = {"official_val150": set(val.scene),
            "new228_test": set(sp[(sp.group == "new765") & (sp.deep_split == "test")].scene)}
    for col in ("group", "deep_split"):
        for k, g in val.groupby(col):
            pops[f"official_val150|{col}={k}"] = set(g.scene)
    return sp, pops


def stage_settings(args):
    sp, pops = populations()
    meta = xl.scene_meta(args.dataroot)
    log_of = dict(zip(meta.scene, meta.log))
    rows, det = [], []
    for f in sorted(Path(args.work).glob("*__*__*.pkl")):
        geometry, op, full = f.stem.split("__")
        d = pd.read_pickle(f)
        for pop, units in pops.items():
            x = d[d.seq.isin(units)]
            v = (x.J_cheap - x.J_full).to_numpy(float)
            c = x.J_cheap.to_numpy(float)
            r = xl.describe(v, c, (0.2,))
            r["n_units"], r["n_logs"] = int(x.seq.nunique()), int(x.seq.map(log_of).nunique())
            r.update(xl.bootstrap_describe(v, c, x.seq.to_numpy(), args.nboot, np.random.default_rng(0), (0.2,)))
            lg = xl.bootstrap_describe(v, c, x.seq.map(log_of).to_numpy(), args.nboot, np.random.default_rng(0), (0.2,))
            r.update({k.replace("_lo", "_lo_log").replace("_hi", "_hi_log"): y for k, y in lg.items()})
            exposure = "outside the detectors' training split" if pop.startswith("official_val150") else \
                f"{int(sp[sp.scene.isin(units)].official.eq('train').sum())} scenes in the detectors' training split"
            rows.append({"full": full, "geometry": geometry, "operating_point": op, "population": pop,
                         "exposure": exposure, "primary": full == "voxel" and geometry == "detected" and op == "shared"
                         and pop == "official_val150", **r})
            print(f"  {full} {geometry} {op} {pop}: harmed {r['harmed_share']:.3f} rho {r['rho']:.3f} "
                  f"all-full {r['all_full_reduction']:+.3f}", flush=True)
            if geometry == "detected":
                ng = int(x.n_gt.sum())
                det.append({"full": full, "operating_point": op, "population": pop,
                            "recall_cheap": x.tp_cheap.sum() / ng, "recall_full": x.tp_full.sum() / ng,
                            "precision_cheap": x.tp_cheap.sum() / max(x.n_cheap.sum(), 1),
                            "precision_full": x.tp_full.sum() / max(x.n_full.sum(), 1), "n_reference": ng})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "settings.csv", index=False)
    pd.DataFrame(det).to_csv(OUT / "detection.csv", index=False)
    cost = [{"model": m, **{k: v for k, v in json.loads((DET / m / "summary.json").read_text()).items() if k != "model"},
             "name": json.loads((DET / m / "summary.json").read_text())["model"]} for m in ("pp", "voxel", "transfusion")]
    pd.DataFrame(cost).to_csv(OUT / "costs.csv", index=False)       # latency over 200 keyframes, A5000 shared as it is
    print("wrote", OUT / "settings.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tables", "settings"], required=True)
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--nboot", type=int, default=1000)
    args = ap.parse_args()
    run = runmeta.new_run(f"e7_{args.stage}", vars(args))
    {"tables": stage_tables, "settings": stage_settings}[args.stage](args)
    (run / "summary.json").write_text(json.dumps({"stage": args.stage}, indent=1))


if __name__ == "__main__":
    main()
