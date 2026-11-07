#!/usr/bin/env python
"""Follow-up round 2, R6: validity checks on XL (addendum R6 in docs/followup_protocol.md).

  --stage frames       decision.build's frame loop on all 850 scenes (YOLOv8s 320 -> 640, threshold 0.25, shared
                       action history, monocular and oracle geometry), keeping per-object state, with the standard
                       reference (asserted equal to the XL tables) and the completed reference (the boxes the
                       visibility rule excludes put back: no LiDAR or radar point, under 10 px high)
                       -> data/cache/xl/r6/frames_<geometry>.pkl, objects of the harmed frames
  --stage completeness harmed share and rho of braking V with the completed reference (primary: mono, 228 test; also
                       all 850); the share of added false positives the controller considers that overlap an
                       excluded annotation -> reference_completeness.csv
  --stage attribution  harmed frames (V < 0) of the primary cell: cumulative interventions on the full branch, and
                       each intervention alone -> attribution.csv, attribution_single.csv
  --stage thresholds   S1-S3 operating points from the 152 validation scenes (the paper's rules, 100's stage B)
  --stage tables       control tables: S1-S3 and the persistence filter, both geometries, all 850 scenes
  --stage controls     S1-S3, persistence, safety-only loss; both geometries; 228 test and 850 -> controls.csv
  --stage audit        100 harmed frames (seed 0) of the primary cell on the 228 test scenes: image copy and JSON in
                       the gallery format, a cheap | full image pair with the controller's actions -> audit/,
                       audit_frames.csv
"""
from __future__ import annotations

import argparse, importlib.util, json, pickle, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import decision, geometry as G, planner as P, runmeta, xl              # noqa: E402
from rap import percep_metrics as PM                                            # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.mono import box_iou                                                    # noqa: E402
from rap.nusc import NuScenesDB                                                 # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                               # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402
from rap.tables import match_detection_counts                                   # noqa: E402

OUT = Path(RESULTS) / "followup" / "R6"
WORK = Path(CACHE) / "xl" / "r6"
DET = Path(CACHE) / "nusc_det_xl"
CHEAP, FULL = "ns_cheap_320", "ns_full_640"
TABLES = Path(CACHE) / "xl" / "tables850"
EPS = 1e-9
GRID05 = [round(0.10 + 0.05 * i, 2) for i in range(13)]
GRID01 = [round(0.10 + 0.01 * i, 2) for i in range(61)]
CAUSES = ("added detection without a real object", "recovered real object misplaced by the lift",
          "geometry change of a jointly detected object", "other")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


t118 = _load("t118", "118_figure_exports.py")


def populations():
    """All 850 scenes, and the 228 new test scenes."""
    sp = xl.scene_split()
    test = set(sp[(sp.group == "new765") & (sp.deep_split == "test")].scene)
    return sorted(sp.scene), test


class Geometries:
    """Standard and completed reference per scene from one database, each with its own instance map (the database's
    track map is rebuilt by every call). Completed: min_pts 0 and min_px 1 (a box must still cover a pixel of the
    image in each direction; boxes outside the image clip to a line)."""

    def __init__(self, db):
        self.db, self.std, self.comp, self.inst_std, self.inst_comp, self.speed = db, {}, {}, {}, {}, {}

    def load(self, seq):
        if seq in self.std:
            return
        name = self.db_name(seq)
        self.std[seq] = self.db.sequence_geometry(name)
        self.inst_std[seq] = {v: k for k, v in self.db._track_map[seq].items()}
        self.comp[seq] = self.db.sequence_geometry(name, min_pts=0, min_px=1)
        self.inst_comp[seq] = {v: k for k, v in self.db._track_map[seq].items()}
        self.speed[seq] = self.db.ego_speeds(name)

    def db_name(self, seq):
        if not hasattr(self, "_by_name"):
            self._by_name = {self.db.scene_name(s): s for s in self.db.scenes}
        return self._by_name[seq]


def frame_objects(fr_std, fr_comp, inst_std, inst_comp, cfg):
    """Reference boxes of one frame: standard (>= 10 px tall, at least one point) and completed (every box), with the
    completed boxes the standard rule excludes flagged."""
    g = fr_std[(fr_std["y2"] - fr_std["y1"]) >= cfg.min_gt_height]
    std_tokens = {inst_std[int(t)] for t in g["track_id"]}
    gc = fr_comp
    excluded = np.array([inst_comp[int(t)] not in std_tokens for t in gc["track_id"]], bool)
    return g, gc, excluded


def boxes_of(g):
    return np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float) if len(g) else np.zeros((0, 4))


def stage_frames(args):
    """decision.build's loop (118's rebuild), extended: per frame the standard and completed requirements, the J of
    both branches under each, and the objects of every frame harmed under the standard reference."""
    seqs, test = populations()
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    geo = Geometries(db)
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    WORK.mkdir(parents=True, exist_ok=True)
    for geometry in xl.GEOMETRIES:
        rng = np.random.default_rng(0)
        table = pd.read_pickle(TABLES / f"{geometry}__{xl.PAIR}.pkl").set_index(["seq", "frame"])
        rows, harmed = [], {}
        t0 = time.time()
        for si, s in enumerate(seqs):
            geo.load(s)
            gs, gcomp, speeds = geo.std[s], geo.comp[s], geo.speed[s]
            c, f = DetCache(DET / CHEAP / f"{s}.npz"), DetCache(DET / FULL / f"{s}.npz")
            prev_c = None
            for i, fr in enumerate(c.frames):
                fr = int(fr)
                j = f.index[fr]
                v = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
                g, gc, excluded = frame_objects(gs[gs["frame"] == fr], gcomp[gcomp["frame"] == fr],
                                                geo.inst_std[s], geo.inst_comp[s], cfg)
                gt, gct = boxes_of(g), boxes_of(gc)
                out = {}
                for tag, cache, idx in (("cheap", c, i), ("full", f, j)):
                    d, gk = cache.det(idx), cache.geo(idx)
                    k = d["conf"] >= cfg.thr(tag)
                    geo_k = {kk: vv[k] for kk, vv in gk.items()}
                    bx = d["xyxy"][k].astype(float)
                    z, lo, hi, tt = decision._apply_range_source(bx, geo_k, g, geometry, rng, 0.0)
                    a, a_obj = P.required_decel(z, lo, hi, tt, v, pp)
                    det_gt = t118.match_pairs(gt, bx, d["conf"][k].astype(float), cfg)[0]
                    det_gc = t118.match_pairs(gct, bx, d["conf"][k].astype(float), cfg)[0]
                    out[tag] = dict(xyxy=bx, conf=d["conf"][k].astype(float), coarse=d["coarse"][k], z=z, lo=lo, hi=hi,
                                    ttc=tt, a=a, a_obj=a_obj, det_gt=det_gt, det_gc=det_gc)
                a_gt = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], v, pp)[0]
                a_gc = P.required_decel(gc["long_near"], gc["lat_min"], gc["lat_max"], gc["ttc"], v, pp)[0]
                act_c, act_f = P.discrete_action(out["cheap"]["a"], pp), P.discrete_action(out["full"]["a"], pp)
                Jc, Jf = P.decision_cost(act_c, a_gt, prev_c, pp, cp)["J"], P.decision_cost(act_f, a_gt, prev_c, pp, cp)["J"]
                Jc2 = P.decision_cost(act_c, a_gc, prev_c, pp, cp)["J"]
                Jf2 = P.decision_cost(act_f, a_gc, prev_c, pp, cp)["J"]
                tr = table.loc[(s, fr)]
                assert abs(Jc - tr.J_cheap) < EPS and abs(Jf - tr.J_full) < EPS, (geometry, s, fr)
                # added false positives the controller considers: full detections matching no standard box, with no
                # cheap detection at IoU >= 0.5, in the corridor and within range (mono geometry of the full branch)
                of, oc = out["full"], out["cheap"]
                cons = t118.corridor(of["lo"], of["hi"], of["z"], pp)[1]
                fp = of["det_gt"] < 0
                near_cheap = (box_iou(of["xyxy"], oc["xyxy"]).max(1) >= cfg.iou_thr) if len(oc["xyxy"]) and len(of["xyxy"]) \
                    else np.zeros(len(of["xyxy"]), bool)
                added = fp & ~near_cheap & cons
                exc = gct[excluded]
                ov = (box_iou(of["xyxy"][added], exc).max(1) >= 0.3) if added.any() and len(exc) else np.zeros(int(added.sum()), bool)
                rows.append({"seq": s, "frame": fr, "v_ego": v, "a_gt": a_gt, "a_gt_completed": a_gc, "act_cheap": act_c,
                             "act_full": act_f, "prev_cheap": -1 if prev_c is None else prev_c, "J_cheap": Jc, "J_full": Jf,
                             "J_cheap_completed": Jc2, "J_full_completed": Jf2, "n_ref": len(g), "n_ref_completed": len(gc),
                             "n_excluded": int(excluded.sum()), "added_fp_considered": int(added.sum()),
                             "added_fp_considered_on_excluded": int(ov.sum())})
                if Jc - Jf < 0:
                    harmed[(s, fr)] = dict(v=v, a_gt=a_gt, prev_c=prev_c, act_c=act_c, act_f=act_f, J_cheap=Jc, J_full=Jf,
                                           g=g, gc=gc, excluded=excluded, out=out, sample_token=tr.sample_token)
                prev_c = act_c
            if (si + 1) % 100 == 0:
                print(f"  {geometry} {si + 1} / {len(seqs)} scenes [{time.time() - t0:.0f}s]", flush=True)
        d = pd.DataFrame(rows)
        assert len(d) == len(table), (len(d), len(table))
        d.to_pickle(WORK / f"frames_{geometry}.pkl")
        with open(WORK / f"harmed_{geometry}.pkl", "wb") as fh:
            pickle.dump(harmed, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"{geometry}: {len(d)} frames (J equal to the tables), {len(harmed)} harmed [{time.time() - t0:.0f}s]",
              flush=True)


# ------------------------------------------------------------------------------------------------ statistics

def setting_rows(v, cheap, units, label, nboot=1000):
    r = xl.describe(v, cheap, (0.2,))
    r.update(xl.bootstrap_describe(v, cheap, units, nboot, np.random.default_rng(0), (0.2,)))
    return {**label, "n_scenes": len(set(units)), **r}


def ratio_ci(num, den, units, nboot=1000):
    """A pooled share sum(num) / sum(den) with its scene-bootstrap interval (1,000 draws, seed 0)."""
    df = pd.DataFrame({"u": units, "n": num, "d": den}).groupby("u")[["n", "d"]].sum()
    n, d = df.n.to_numpy(float), df.d.to_numpy(float)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(nboot):
        i = rng.integers(0, len(df), len(df))
        draws.append(n[i].sum() / d[i].sum() if d[i].sum() > 0 else np.nan)
    lo, hi = xl.ci(np.array(draws))
    return (n.sum() / d.sum() if d.sum() > 0 else np.nan), lo, hi


def stage_completeness(args):
    _, test = populations()
    d = pd.read_pickle(WORK / "frames_mono.pkl")
    rows = []
    for pop, m in (("new228_test", d.seq.isin(test)), ("all850", np.ones(len(d), bool))):
        x = d[m]
        for ref, (jc, jf) in (("standard", ("J_cheap", "J_full")), ("completed", ("J_cheap_completed", "J_full_completed"))):
            v = (x[jc] - x[jf]).to_numpy(float)
            r = setting_rows(v, x[jc].to_numpy(float), x.seq.to_numpy(), {"population": pop, "reference": ref})
            r["reading"] = ("not a labelling artefact" if (r["harmed_share"] >= 0.30 and r["rho"] > 0) else
                            "fails: harmed share below 30%" if r["harmed_share"] < 0.30 else "fails: rho not above 0") \
                if ref == "completed" else ""
            rows.append(r)
        sh, lo, hi = ratio_ci(x.added_fp_considered_on_excluded.to_numpy(float), x.added_fp_considered.to_numpy(float),
                              x.seq.to_numpy())
        rows.append({"population": pop, "reference": "added FP overlap", "added_fp_considered": int(x.added_fp_considered.sum()),
                     "added_fp_considered_on_excluded": int(x.added_fp_considered_on_excluded.sum()),
                     "share_overlapping_excluded": sh, "share_lo": lo, "share_hi": hi,
                     "frames_with_excluded": int((x.n_excluded > 0).sum()), "excluded_boxes": int(x.n_excluded.sum())})
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "reference_completeness.csv", index=False)
    cols = [c for c in ("population", "reference", "affected", "harmed_share", "harmed_share_lo", "harmed_share_hi", "rho",
                        "rho_lo", "rho_hi", "reading", "share_overlapping_excluded", "share_lo", "share_hi") if c in t]
    print(t[cols].to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ attribution

def j_full(o, keep, z, lo, hi, tt, fr, pp, cp):
    a = P.required_decel(z[keep], lo[keep], hi[keep], tt[keep], fr["v"], pp)[0]
    return P.decision_cost(P.discrete_action(a, pp), fr["a_gt"], fr["prev_c"], pp, cp)["J"]


def interventions(fr, pp, cp):
    """The three interventions on the full branch, as (name, keep mask, geometry) after each cumulative step, and each
    one alone. Matching against the completed reference (118's greedy assignment, IoU >= 0.5)."""
    of, oc = fr["out"]["full"], fr["out"]["cheap"]
    n = len(of["conf"])
    base = (np.ones(n, bool), of["z"].copy(), of["lo"].copy(), of["hi"].copy(), of["ttc"].copy())
    gc = fr["gc"]
    cheap_refs = set(oc["det_gc"][oc["det_gc"] >= 0].tolist())
    cheap_of_ref = {int(r): k for k, r in enumerate(oc["det_gc"]) if r >= 0}

    def step1(state):
        keep, z, lo, hi, tt = (x.copy() for x in state)
        keep &= of["det_gc"] >= 0
        return keep, z, lo, hi, tt

    def step2(state):
        keep, z, lo, hi, tt = (x.copy() for x in state)
        for k in range(n):
            r = int(of["det_gc"][k])
            if keep[k] and r >= 0 and r not in cheap_refs:
                z[k], lo[k], hi[k], tt[k] = gc["long_near"][r], gc["lat_min"][r], gc["lat_max"][r], gc["ttc"][r]
        return keep, z, lo, hi, tt

    def step3(state):
        keep, z, lo, hi, tt = (x.copy() for x in state)
        for k in range(n):
            r = int(of["det_gc"][k])
            if keep[k] and r >= 0 and r in cheap_refs:
                kc = cheap_of_ref[r]
                z[k], lo[k], hi[k], tt[k] = oc["z"][kc], oc["lo"][kc], oc["hi"][kc], oc["ttc"][kc]
        return keep, z, lo, hi, tt

    return base, (step1, step2, step3)


def stage_attribution(args):
    _, test = populations()
    pp, cp = P.PlannerParams(), P.CostParams()
    with open(WORK / "harmed_mono.pkl", "rb") as fh:
        harmed = pickle.load(fh)
    rows, single = [], []
    for (s, frn), fr in sorted(harmed.items()):
        base, steps = interventions(fr, pp, cp)
        J0 = j_full(fr["out"]["full"], *base, fr, pp, cp)
        assert abs(J0 - fr["J_full"]) < EPS, (s, frn)
        state, cause, vs = base, CAUSES[3], []
        for name, st in zip(CAUSES[:3], steps):
            state = st(state)
            v = fr["J_cheap"] - j_full(fr["out"]["full"], *state, fr, pp, cp)
            vs.append(v)
            if v >= 0 and cause == CAUSES[3]:
                cause = name
        alone = [fr["J_cheap"] - j_full(fr["out"]["full"], *st(base), fr, pp, cp) for st in steps]
        rec = {"seq": s, "frame": frn, "sample_token": fr["sample_token"], "test228": s in test,
               "V": fr["J_cheap"] - fr["J_full"], "cause": cause, "V_after_1": vs[0], "V_after_2": vs[1], "V_after_3": vs[2],
               "V_alone_1": alone[0], "V_alone_2": alone[1], "V_alone_3": alone[2]}
        rows.append(rec)
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "attribution.csv", index=False)
    summ = []
    for pop, m in (("new228_test", a.test228), ("all850", np.ones(len(a), bool))):
        x = a[m]
        for c in CAUSES:
            summ.append({"population": pop, "table": "cumulative (first step after which V >= 0)", "cause": c,
                         "frames": int((x.cause == c).sum()), "share": float((x.cause == c).mean()), "harmed_frames": len(x)})
        for k, c in enumerate(CAUSES[:3], 1):
            summ.append({"population": pop, "table": "each intervention alone (descriptive)", "cause": c,
                         "frames": int((x[f"V_alone_{k}"] >= 0).sum()), "share": float((x[f"V_alone_{k}"] >= 0).mean()),
                         "harmed_frames": len(x)})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "attribution_single.csv", index=False)
    print(s.to_string(index=False), flush=True)


# ------------------------------------------------------------------------------------------------ controls

def val_scenes():
    sp = xl.scene_split()
    return sorted(sp[(sp.group == "new765") & (sp.deep_split == "val")].scene)


def stage_thresholds(args):
    """100's stage B rules on the 152 validation scenes: S1 count-matched (full threshold matching cheap's detections
    per frame at 0.25), S2 F1-optimal on the 0.05 grid per mode, S3 precision-matched (lowest full threshold on the
    0.01 grid whose pooled precision reaches cheap's at 0.25). Pooled precision / recall with the standard reference."""
    from rap.geometry import coarse_classes
    from rap.nusc import make_adapter
    units = val_scenes()
    ad = make_adapter(NuScenesDB(args.dataroot, "v1.0-trainval"))
    cfg = RiskConfig()
    curves = {}
    for mode in (CHEAP, FULL):
        arr = []
        for s in units:
            geom = ad.geometry(s)
            c = DetCache(DET / mode / f"{s}.npz")
            for i, fr in enumerate(c.frames):
                g = geom[geom["frame"] == int(fr)]
                g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
                gt = boxes_of(g)
                det = c.det(i)
                row = []
                for t in GRID01:
                    L = PM.frame_losses(gt, coarse_classes(g), np.zeros(len(gt)), det, cfg, op_conf=t)
                    row.append((L["n_det"], L["n_det"] - L["fp"], L["fn"], L["n_gt"]))
                arr.append(row)
        a = np.asarray(arr, float).sum(0)
        det, tp, fn, ngt = a.T
        prec = np.where(det > 0, tp / np.maximum(det, 1), np.nan)
        rec = np.where(ngt > 0, (ngt - fn) / np.maximum(ngt, 1), np.nan)
        f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
        curves[mode] = pd.DataFrame({"threshold": GRID01, "precision": prec, "recall": rec, "f1": f1})
    s2 = {}
    for m, c in curves.items():
        g5 = c[c.threshold.isin(GRID05)]
        s2[m] = float(g5.loc[g5.f1.idxmax(), "threshold"])
    s1 = match_detection_counts([DetCache(DET / CHEAP / f"{s}.npz") for s in units],
                                [DetCache(DET / FULL / f"{s}.npz") for s in units], 0.25)
    pc = float(curves[CHEAP][curves[CHEAP].threshold == 0.25].precision.iloc[0])
    ok = curves[FULL][curves[FULL].precision >= pc].sort_values("threshold")
    s3, flag = (float(ok.threshold.iloc[0]), False) if len(ok) else (0.70, True)
    th = {"S1": [0.25, float(s1)], "S2": [s2[CHEAP], s2[FULL]], "S3": [0.25, s3], "S3_target_precision": pc,
          "S3_none_reached_target": flag, "units": "152 new validation scenes", "frames": int(sum(len(DetCache(
              DET / CHEAP / f"{s}.npz").frames) for s in units))}
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "thresholds.json").write_text(json.dumps(th, indent=1))
    pd.concat([c.assign(mode=m) for m, c in curves.items()]).to_csv(WORK / "val_curves.csv", index=False)
    print("thresholds:", th, flush=True)


class PersistentCache:
    """A detection cache whose kept detections must reappear in the same mode's two previous keyframes (same coarse
    class, image IoU >= 0.3). A scene's first two keyframes have no such history; their detections are kept."""

    def __init__(self, path, thr, iou=0.3):
        self._c, self.frames, self.index = DetCache(path), None, None
        self.frames, self.index, self.thr, self.iou = self._c.frames, self._c.index, thr, iou

    def _keep(self, i):
        d = self._c.det(i)
        keep = np.ones(len(d["conf"]), bool)
        if i < 2:
            return keep
        for back in (1, 2):
            p = self._c.det(i - back)
            pk = p["conf"] >= self.thr
            if not pk.any():
                return np.zeros(len(keep), bool)
            iou = box_iou(d["xyxy"].astype(float), p["xyxy"][pk].astype(float))
            same = d["coarse"][:, None] == p["coarse"][pk][None, :]
            keep &= (np.where(same, iou, 0.0) >= self.iou).any(1)
        return keep

    @staticmethod
    def _sub(d, k):
        """Filter the per-detection arrays; scalar entries (n_cand, n_cand_raw, n_post) pass unchanged."""
        return {kk: (vv[k] if isinstance(vv, np.ndarray) and vv.ndim >= 1 and len(vv) == len(k) else vv)
                for kk, vv in d.items()}

    def det(self, i):
        return self._sub(self._c.det(i), self._keep(i))

    def geo(self, i):
        return self._sub(self._c.geo(i), self._keep(i))


def stage_tables(args):
    from rap.nusc import make_adapter
    seqs, _ = populations()
    th = json.loads((WORK / "thresholds.json").read_text())
    nad = make_adapter(NuScenesDB(args.dataroot, "v1.0-trainval"))
    pp, cp = P.PlannerParams(), P.CostParams()
    specs = {f"{k}": RiskConfig(op_conf=th[k][0], op_conf_full=th[k][1]) for k in ("S1", "S2", "S3")}
    for geometry in xl.GEOMETRIES:
        for name, cfg in list(specs.items()) + [("persistence", RiskConfig())]:
            f = WORK / f"table_{name}_{geometry}.pkl"
            if f.exists():
                continue
            t0 = time.time()
            fac = (lambda path, role, c=cfg: PersistentCache(path, c.thr(role))) if name == "persistence" else None
            d = decision.build(DET, CHEAP, FULL, seqs, cfg, pp, cp, G.PRIMARY, range_source=geometry, adapter=nad,
                               cache_factory=fac)
            d.to_pickle(f)
            print(f"  {f.name}: {len(d)} frames [{time.time() - t0:.0f}s]", flush=True)


def stage_controls(args):
    _, test = populations()
    rows = []
    for geometry in xl.GEOMETRIES:
        std = pd.read_pickle(TABLES / f"{geometry}__{xl.PAIR}.pkl")
        variants = {"standard (shared 0.25)": (std, "J_cheap", "J_full")}
        safe = std.assign(Js_c=std.Jterm_cheap_shortfall + std.Jterm_cheap_collision,
                          Js_f=std.Jterm_full_shortfall + std.Jterm_full_collision)
        variants["safety-only loss (shortfall + collision)"] = (safe, "Js_c", "Js_f")
        for name in ("S1", "S2", "S3", "persistence"):
            variants[name] = (pd.read_pickle(WORK / f"table_{name}_{geometry}.pkl"), "J_cheap", "J_full")
        for name, (d, jc, jf) in variants.items():
            for pop, m in (("new228_test", d.seq.isin(test)), ("all850", np.ones(len(d), bool))):
                x = d[m]
                v = (x[jc] - x[jf]).to_numpy(float)
                rows.append(setting_rows(v, x[jc].to_numpy(float), x.seq.to_numpy(),
                                         {"geometry": geometry, "control": name, "population": pop}))
    t = pd.DataFrame(rows)
    th = json.loads((WORK / "thresholds.json").read_text())
    t["t_cheap"] = t.control.map(lambda c: th[c][0] if c in th else 0.25)
    t["t_full"] = t.control.map(lambda c: th[c][1] if c in th else 0.25)
    t.to_csv(OUT / "controls.csv", index=False)
    print(t[["geometry", "control", "population", "affected", "harmed_share", "rho", "all_full_reduction"]].to_string(index=False))


# ------------------------------------------------------------------------------------------------ audit

def stage_audit(args):
    import cv2
    _, test = populations()
    pp = P.PlannerParams()
    db = NuScenesDB(args.dataroot, "v1.0-trainval")
    with open(WORK / "harmed_mono.pkl", "rb") as fh:
        harmed = pickle.load(fh)
    att = pd.read_csv(OUT / "attribution.csv")
    cause = {(r.seq, int(r.frame)): r.cause for r in att.itertuples()}
    keys = sorted(k for k in harmed if k[0] in test)
    pick = sorted(keys[i] for i in np.random.default_rng(0).choice(len(keys), 100, replace=False))
    aud = OUT / "audit"
    aud.mkdir(parents=True, exist_ok=True)
    name = lambda a: P.ACTION_NAMES[int(a)]
    rows = []
    for s, frn in pick:
        fr = harmed[(s, frn)]
        src = db.image_path(fr["sample_token"])
        stem = f"{s}_{frn:02d}"
        shutil.copy2(src, aud / f"{stem}{src.suffix}")

        def boxes(mode):
            o = fr["out"][mode]
            lat, cons = t118.corridor(o["lo"], o["hi"], o["z"], pp)
            return [{"det_index": k, "xyxy": [float(x) for x in o["xyxy"][k]], "class": str(o["coarse"][k]),
                     "conf": float(o["conf"][k]), "matched_reference": bool(o["det_gt"][k] >= 0),
                     "matched_completed_reference": bool(o["det_gc"][k] >= 0), "z_forward_m": float(o["z"][k]),
                     "lat_min_m": float(o["lo"][k]), "lat_max_m": float(o["hi"][k]), "in_corridor": bool(lat[k]),
                     "considered_by_controller": bool(cons[k]), "a_req_obj": float(o["a_obj"][k])} for k in range(len(o["conf"]))]
        gc = fr["gc"]
        gts = [{"xyxy_projected": [float(gc[c][i]) for c in ("x1", "y1", "x2", "y2")], "type": str(gc["type"][i]),
                "z_forward_m": float(gc["long_near"][i]), "lat_min_m": float(gc["lat_min"][i]),
                "lat_max_m": float(gc["lat_max"][i]), "excluded_by_visibility_rule": bool(fr["excluded"][i])}
               for i in range(len(gc))]
        rec = {"scene": s, "frame": frn, "sample_token": fr["sample_token"], "system": "brake", "geometry": "mono",
               "image_copy": f"{stem}{src.suffix}", "V": fr["J_cheap"] - fr["J_full"], "J_cheap": fr["J_cheap"],
               "J_full": fr["J_full"], "ego_speed_mps": fr["v"],
               "action": {"cheap": name(fr["act_c"]), "full": name(fr["act_f"]),
                          "reference": name(P.discrete_action(fr["a_gt"], pp))},
               "a_req": {"cheap": float(fr["out"]["cheap"]["a"]), "full": float(fr["out"]["full"]["a"]),
                         "reference": float(fr["a_gt"])},
               "cheap_boxes": boxes("cheap"), "full_boxes": boxes("full"), "reference_boxes_completed": gts,
               "cause_from_attribution": cause[(s, frn)],
               "conventions": "boxes in original image pixels; z_forward_m = near-face range; detections in the monocular "
                              "geometry; reference boxes are the completed reference, with the standard rule's exclusions "
                              "flagged"}
        (aud / f"{stem}.json").write_text(json.dumps(rec, indent=1))
        img = cv2.imread(str(src))
        pair = []
        for mode, colour in (("cheap", (0, 165, 255)), ("full", (255, 128, 0))):
            im = img.copy()
            for b in gts:
                x1, y1, x2, y2 = map(int, b["xyxy_projected"])
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255) if b["excluded_by_visibility_rule"] else (0, 200, 0), 1)
            for b in rec[f"{mode}_boxes"]:
                x1, y1, x2, y2 = map(int, b["xyxy"])
                cv2.rectangle(im, (x1, y1), (x2, y2), colour, 3 if b["considered_by_controller"] else 1)
            cv2.putText(im, f"{mode}: {rec['action'][mode]} (a {rec['a_req'][mode]:.2f})  reference: "
                            f"{rec['action']['reference']} (a {rec['a_req']['reference']:.2f})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            pair.append(im)
        cv2.imwrite(str(aud / f"{stem}_pair.jpg"), np.hstack(pair), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rows.append({"frame_id": stem, "scene": s, "frame": frn, "sample_token": fr["sample_token"], "V": rec["V"],
                     "cause": cause[(s, frn)], "label": ""})
    pd.DataFrame(rows).to_csv(OUT / "audit_frames.csv", index=False)
    (aud / "README.txt").write_text(
        "R6 manual audit: 100 harmed frames (V < 0) of the primary cell (braking, monocular, 320 -> 640, threshold 0.25) "
        "on the 228 new test scenes, drawn at random with numpy default_rng(0).\n"
        "<scene>_<frame>.jpg: the image; <scene>_<frame>.json: the frame in the gallery format; <scene>_<frame>_pair.jpg: "
        "cheap (left, orange) | full (right, blue) detections, thick when the controller considers them; reference boxes "
        "green, red when the standard visibility rule excludes them; the actions on top.\n"
        "Label each row of audit_frames.csv: phantom, real_misplaced, real_missing_from_reference, real_correct, unclear.\n")
    print(f"audit: {len(rows)} frames -> {aud}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["frames", "completeness", "attribution", "thresholds", "tables",
                                                       "controls", "audit"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    runmeta.new_run(f"r6_{args.stage}", vars(args))
    {"frames": stage_frames, "completeness": stage_completeness, "attribution": stage_attribution,
     "thresholds": stage_thresholds, "tables": stage_tables, "controls": stage_controls,
     "audit": stage_audit}[args.stage](args)


if __name__ == "__main__":
    main()
