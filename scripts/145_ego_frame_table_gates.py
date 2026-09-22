#!/usr/bin/env python
"""Task 23 Phase 2, second half: G4 and G5, on the ego-frame per-frame tables, before any figure is computed from them.

Pre-registered in the pre-registration record (not part of this release) (Task 23 and its amendment).  Run after the per-frame tables have been regenerated in
the ego frame (52, 65, 100, 134, 60) and before any stage reads them.

  G4  the class primitives fn, fp, loc, cls, crit_fn, n_det, n_gt -- and the detection counts -- are identical in the
      camera-frame and ego-frame tables, on every mode and every threshold of the calibration outcomes and at the
      operating point of the core-matrix tables: matching is done in the image, so the frame cannot reach them
  G5  oracle geometry moves only through unmatched detections.  A frame is *touched* when, in either mode at the
      table's threshold, a kept detection has no reference box at IoU >= 0.5.  The controllers carry one piece of
      state, the previous action: braking and lateral actions depend on the current frame alone, so their previous
      action can change only right after a touched frame; Planner B's candidate depends on the previous candidate as
      well, and its tables store the candidates, so that chain is followed exactly.  Every frame that is neither
      touched nor downstream of one through the carried state must be byte-identical.  Planner B's JB in the
      calibration and realism tables, which store no candidates, is reported under the braking rule, not gated.  The oracle submissions are
      checked box by box: every field but the translation is frame-independent, and the translation may change only
      for a detection without a matched annotation.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import geometry as G, runmeta, runs as rap_runs                        # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.mono import box_iou                                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                  # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

RAW = ROOT / "results" / "raw"
PRIM = ("fn", "fp", "loc", "cls", "crit_fn", "n_det", "n_gt")
CFG = RiskConfig()


def gate(ok, msg):
    if not ok:
        raise SystemExit(f"GATE FAILED: {msg}")


# ------------------------------------------------------------------------------------------------------------ G4

def g4(run):
    rows = []
    cam_o, ego_o = rap_runs.latest("calibration_outcomes", "camera") / "outcomes", \
        rap_runs.latest("calibration_outcomes", "ego") / "outcomes"
    for f in sorted(ego_o.glob("*.npz")):
        a, b = np.load(cam_o / f.name, allow_pickle=False), np.load(f, allow_pickle=False)
        cols = [f"{t}_{p}" for t in ("cheap", "full") for p in PRIM] + ["n_cheap", "n_full"]
        bad = [c for c in cols if c in a.files and a[c].tobytes() != b[c].tobytes()]
        rows.append({"table": f"calibration_outcomes/{f.name}", "columns": len(cols), "differing": json.dumps(bad),
                     "ok": not bad})
    cam_c, ego_c = rap_runs.core_matrix("plain", "camera"), rap_runs.core_matrix("plain", "ego")
    for f in sorted(ego_c.glob("*.pkl")):
        a, b = pd.read_pickle(cam_c / f.name), pd.read_pickle(f)
        # the registered primitives and the detection counts; the dE columns are reported, not gated
        cols = [c for c in [f"{t}_{p}" for t in ("cheap", "full") for p in PRIM] + ["n_cheap", "n_full"]
                if c in a.columns]
        bad = [c for c in cols if a[c].to_numpy().tobytes() != b[c].to_numpy().tobytes()]
        info = [c for c in a.columns if c.startswith("dE") and a[c].to_numpy().tobytes() != b[c].to_numpy().tobytes()]
        rows.append({"table": f"core_matrix/{f.name}", "columns": len(cols), "differing": json.dumps(bad),
                     "dE_columns_differing_not_gated": json.dumps(info), "ok": not bad})
    t = pd.DataFrame(rows)
    t.to_csv(run / "g4_primitives.csv", index=False)
    print(f"  G4: {int(t.ok.sum())}/{len(t)} tables with identical primitives and counts", flush=True)
    gate(bool(t.ok.all()), "G4: a class primitive or detection count moved with the frame; see g4_primitives.csv")


# ------------------------------------------------------------------------------------------------------------ G5

class Touch:
    """Per (cache, unit, threshold): which frames have a kept detection without a reference match."""

    def __init__(self):
        self.memo, self.ad = {}, None

    def adapter(self):
        if self.ad is None:
            from rap.nusc import NuScenesDB, make_adapter
            self.ad = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
        return self.ad

    def geom(self, dataset, unit):
        return G.sequence_geometry(unit) if dataset == "KITTI" else self.adapter().geometry(unit)

    def frames(self, dataset, det_dir, mode, unit, thr, keep=None):
        key = (str(det_dir), mode, unit, round(float(thr), 9), keep is not None)
        if key in self.memo:
            return self.memo[key]
        c = DetCache(Path(det_dir) / mode / f"{unit}.npz", frame="camera")
        geom = self.geom(dataset, unit)
        out = {}
        for i, fr in enumerate(c.frames):
            fr = int(fr)
            d = c.det(i)
            k = d["conf"] >= thr
            if keep is not None:
                k &= keep(i)
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= CFG.min_gt_height]
            bx = d["xyxy"][k].astype(float)
            if not len(bx):
                out[fr] = False
            elif not len(g):
                out[fr] = True
            else:
                gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
                out[fr] = bool((box_iou(bx, gt).max(1) < 0.5).any())
        self.memo[key] = out
        return out


def excluded(df, touched, state_cols=None):
    """Frames G5 excludes: touched ones, and those the carried state can reach.

    Braking and lateral actions depend on the current frame alone, so the state can differ only right after a touched
    frame.  With `state_cols` (Planner B's stored candidates of the two runs), the chain is followed exactly: a frame is
    reached when the previous frame's candidate differs between the runs."""
    ex = np.zeros(len(df), bool)
    via_state = np.zeros(len(df), bool)
    seqs, frs = df.seq.astype(str).to_numpy(), df.frame.to_numpy(int)
    t = np.array([touched[s].get(f, False) for s, f in zip(seqs, frs)])
    for i in range(len(df)):
        ex[i] = t[i]
        if i and seqs[i] == seqs[i - 1]:
            if state_cols is None:
                carried = t[i - 1]
            else:
                a, b = state_cols
                carried = bool(a[i - 1] != b[i - 1])
            if carried and not t[i]:
                via_state[i] = True
                ex[i] = True
    return ex, t, via_state


def compare(name, cam, ego, cols, touched, state=None, gated=True):
    keys = ["seq", "frame"]
    cam = cam.astype({"seq": str}).reset_index(drop=True)
    ego = ego.astype({"seq": str}).reset_index(drop=True)
    assert (cam[keys].to_numpy() == ego[keys].to_numpy()).all(), name
    st = None if state is None else (cam[state].to_numpy(), ego[state].to_numpy())
    ex, t, via = excluded(cam, touched, st)
    keep = ~ex
    diff = np.zeros(len(cam), bool)
    for c in cols:
        a, b = cam[c].to_numpy(float), ego[c].to_numpy(float)
        diff |= ~((a == b) | (np.isnan(a) & np.isnan(b)))
    strict = ~t
    row = {"table": name, "frames": len(cam), "touched": int(t.sum()), "touched_share": float(t.mean()),
           "excluded_via_state": int(via.sum()), "compared": int(keep.sum()),
           "differing_among_compared": int((diff & keep).sum()),
           "differing_among_untouched_strict": int((diff & strict).sum()),
           "moved_frames_total": int(diff.sum())}
    row["ok"] = row["differing_among_compared"] == 0
    row["strict_ok"] = row["differing_among_untouched_strict"] == 0
    row["gated"] = gated
    print(f"  G5 {'' if gated else '(reported) '}{name:55s} touched {row['touched_share']:.1%}  via state {row['excluded_via_state']:4d}  "
          f"compared {row['compared']:5d}  differing {row['differing_among_compared']}  "
          f"(strict: {row['differing_among_untouched_strict']})  {'ok' if row['ok'] else 'FAILED'}", flush=True)
    return row


def g5(run):
    T, rows = Touch(), []
    kseqs = sorted(p.stem for p in (Path(CACHE) / "det" / "cheap_320").glob("*.npz"))
    nseqs = sorted(p.stem for p in (Path(CACHE) / "nusc_det_tv" / "ns_cheap_320").glob("*.npz"))

    def touched_pair(dataset, det_dir, cm, fm, units, tc, tf, keep=None):
        out = {}
        for u in units:
            a = T.frames(dataset, det_dir, cm, u, tc, None if keep is None else keep(u, "cheap"))
            b = T.frames(dataset, det_dir, fm, u, tf, None if keep is None else keep(u, "full"))
            out[u] = {f: a.get(f, False) or b.get(f, False) for f in set(a) | set(b)}
        return out

    # core-matrix tables at the operating point: braking and lateral, with the stored actions as the carried state
    cam_c, ego_c = rap_runs.core_matrix("plain", "camera"), rap_runs.core_matrix("plain", "ego")
    for f, dataset, det_dir, cm, fm, units in (
            ("KITTI__YOLOv8s__cheap_320tofull_640__oracle.pkl", "KITTI", Path(CACHE) / "det", "cheap_320", "full_640", kseqs),
            ("nuScenes__YOLOv8s__ns_cheap_320tons_full_640__oracle.pkl", "nuScenes", Path(CACHE) / "nusc_det_tv",
             "ns_cheap_320", "ns_full_640", nseqs)):
        a, b = pd.read_pickle(cam_c / f), pd.read_pickle(ego_c / f)
        tch = touched_pair(dataset, det_dir, cm, fm, units, CFG.thr("cheap"), CFG.thr("full"))
        rows.append(compare(f"core_matrix/{f} braking", a, b, ["J_cheap", "J_full"], tch))
        rows.append(compare(f"core_matrix/{f} lateral", a, b, ["Jlat_cheap", "Jlat_full"], tch))
    # Planner B at the operating point: the stored candidates carry the state
    cam_b, ego_b = rap_runs.planner_b("camera"), rap_runs.planner_b("ego")
    f = "planB__KITTI__YOLOv8s__cheap_320tofull_640__oracle.pkl"
    a, b = pd.read_pickle(cam_b / f), pd.read_pickle(ego_b / f)
    tch = touched_pair("KITTI", Path(CACHE) / "det", "cheap_320", "full_640", kseqs, CFG.thr("cheap"), CFG.thr("full"))
    for mode in ("cheap", "full"):
        rows.append(compare(f"planner_b/{f} {mode}", a, b, [f"JB_{mode}"], tch, state=f"planB_{mode}"))
    # calibration outcomes, every oracle threshold: braking and lateral exactly; Planner B columns have no stored
    # candidates here, so they are checked where the 0.25 table above checks them and reported, not gated, elsewhere
    cam_o, ego_o = rap_runs.latest("calibration_outcomes", "camera"), rap_runs.latest("calibration_outcomes", "ego")
    idx = pd.read_csv(cam_o / "outcome_index.csv")
    for r in idx[idx.spec.str.endswith("_oracle")].itertuples():
        if r.spec.startswith("nusc"):
            dataset, det_dir, cm, fm, units = "nuScenes", Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "ns_full_640", nseqs
        else:
            dataset, det_dir, cm, fm, units = "KITTI", Path(CACHE) / "det", "cheap_320", "full_640", kseqs
        za, zb = np.load(cam_o / "outcomes" / r.file), np.load(ego_o / "outcomes" / r.file)
        a = pd.DataFrame({k: za[k] for k in za.files})
        b = pd.DataFrame({k: zb[k] for k in zb.files})
        tch = touched_pair(dataset, det_dir, cm, fm, units, r.t_cheap, r.t_full)
        rows.append(compare(f"calibration_outcomes/{r.file} braking", a, b, ["J_cheap", "J_full"], tch))
        rows.append(compare(f"calibration_outcomes/{r.file} lateral", a, b, ["Jlat_cheap", "Jlat_full"], tch))
        if r.spec.startswith("kitti"):
            rows.append(compare(f"calibration_outcomes/{r.file} Planner B", a, b, ["JB_cheap", "JB_full"], tch,
                                gated=False))
    # realism controls (134), oracle geometry: braking exactly; persistence settings use their own kept detections
    s134 = importlib.util.spec_from_file_location("r134", ROOT / "scripts" / "134_realism_outcomes.py")
    r134 = importlib.util.module_from_spec(s134)
    s134.loader.exec_module(r134)
    cam_r, ego_r = rap_runs.latest("realism_outcomes", "camera"), rap_runs.latest("realism_outcomes", "ego")
    for f in sorted(ego_r.glob("outcomes__*__oracle__n*.csv.gz")):
        _, pair, _, nn = f.name[: -len(".csv.gz")].split("__")
        n = int(nn[1:])
        dd, cm, fm = r134.PAIRS[pair]

        def keep(u, role, dd=dd, cm=cm, fm=fm, n=n):
            if n == 1:
                return None
            c = r134.PersistentDetCache(Path(CACHE) / dd / (cm if role == "cheap" else fm) / f"{u}.npz", n,
                                        CFG.thr(role))
            return c.keep
        a, b = pd.read_csv(cam_r / f.name, dtype={"seq": str}), pd.read_csv(f, dtype={"seq": str})
        tch = touched_pair("KITTI", Path(CACHE) / dd, cm, fm, kseqs, CFG.thr("cheap"), CFG.thr("full"), keep)
        rows.append(compare(f"realism_outcomes/{f.name} braking", a, b, ["J_cheap", "J_full"], tch))
        rows.append(compare(f"realism_outcomes/{f.name} Planner B", a, b, ["JB_cheap", "JB_full"], tch, gated=False))
    t = pd.DataFrame(rows)
    t.to_csv(run / "g5_oracle_tables.csv", index=False)
    g5_submissions(run, T)
    gate(bool(t[t.gated].ok.all()), "G5: an oracle-geometry value moved on a frame with no unmatched detection; see g5_oracle_tables.csv")


def g5_submissions(run, T):
    """Oracle submissions: every field but the translation is frame-independent, and the translation of a matched
    (annotation-copied) box cannot move."""
    from rap import nusc_submission as NS
    cam_d, ego_d = Path(CACHE) / "nusc_submissions", Path(CACHE) / "nusc_submissions_ego"
    rows = []
    for mode in ("ns_cheap_320", "ns_full_640"):
        f = f"oracle__{mode}.json"
        if not (ego_d / f).exists():
            continue
        a, b = json.loads((cam_d / f).read_text())["results"], json.loads((ego_d / f).read_text())["results"]
        n = moved = other = matched_moved = lifted = 0
        for tok in a:
            gb = b.get(tok, [])
            other += len(a[tok]) != len(gb)
            for x, y in zip(a[tok], gb):
                n += 1
                is_lifted = list(x["size"]) == [float(v) for v in NS.SIZE_PRIOR.get(x["detection_name"], ())]
                lifted += is_lifted
                other += any(x[k] != y[k] for k in x if k != "translation")
                if x["translation"] != y["translation"]:
                    moved += 1
                    matched_moved += not is_lifted          # an annotation-copied box must never move
        rows.append({"submission": f, "boxes": n, "lifted_boxes": lifted, "translations_moved": moved,
                     "matched_boxes_moved": matched_moved, "fields_other_than_translation_moved": other,
                     "ok": other == 0 and matched_moved == 0})
        print(f"  G5 submissions {f}: {n} boxes, {lifted} lifted, {moved} translations moved, "
              f"{matched_moved} of them matched, {other} other fields moved", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(run / "g5_submissions.csv", index=False)
    gate(bool(len(t)) and bool(t.ok.all()), "G5 (submissions): a matched box or a non-translation field moved")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["g4", "g5"])
    args = ap.parse_args()
    run = runmeta.new_run(f"ego_frame_gate_{args.part}", vars(args))
    (g4 if args.part == "g4" else g5)(run)
    print("wrote", run)


if __name__ == "__main__":
    main()
