#!/usr/bin/env python
"""Task 24, Phase B: the published routing objectives as per-frame labels on the ego-frame tables.

Pre-registered in the pre-registration record (not part of this release) (Task 24); definitions and deviations in docs/iclr_published_objectives.md.
Stages, each its own process (the nuScenes tables and the official reward code never share one):

  --stage dump     per track, the cached CHEAP (weak) and FULL (strong) detections and the benchmark's reference
                   objects (>= 10 px tall, coarse class) of every frame of every cell  ->  boxes__<track>.pkl
  --stage labels   per cell, on the fit units (train + val) and on the test units separately, each with its context
                   drawn from its own set:
                     Q  ORIC, Qiu et al.: the official `reward.compute_orie` (edgeml-object-detection@859f702) on
                        `lib/metrics.box_correct` outputs, |E| = 1000, global RNG seeded 42, detections conf >= 0.10
                     G  dataset-wide dAP, Geng et al.: the official `proxy_metrics.compute_dataset_wide_oric`
                        (bgt-ada@6669ab0), context_size = 0 (primary) and 1000 (the code's default, sensitivity),
                        detections conf >= 0.3, `oric_allpoint`
                   and the training targets on the fit set: Q-MORIC (Qiu's ordinal rank), G-MORIC (Geng's ECDF),
                   1[Q > 0], OffloadBin = 1[G > 0]  ->  labels__<track>__<geometry>__<system>__<target>.npz
  --stage checks   exactness of both official AP computations against independent implementations, the no-test-leak
                   assertions, and the label correlations  ->  checks.csv, correlations.csv,
                   results/final/published_objective_labels.csv
"""
from __future__ import annotations

import argparse, contextlib, io, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

QIU = ROOT / "third_party" / "edgeml-object-detection"
GENG = ROOT / "third_party" / "bgt-ada"
TRACKS = {"KITTI": ("det", "cheap_320", "full_640"), "nuScenes": ("nusc_det_tv", "ns_cheap_320", "ns_full_640")}
CLASS_ID = {"vehicle": 0, "person": 1, "cyclist": 2}
Q_FLOOR, G_CONF = 0.10, 0.30               # the cache floor (Qiu's outputs go to 0.001); Geng's own threshold
N_ENSEMBLE, G_CONTEXT = 1000, 1000
SEED = 42
DE_LABELS = ("dE", "dE_E5_combined", "dE_E6_risk_weighted")    # dE is the benchmark's dE_exact


def cells():
    """Every KITTI and nuScenes cell, with its frames, split, unit and V, from the shipped R1 run (the cell's rows)."""
    run = rap_runs.latest("routers_r1")
    out = {}
    for f in sorted(run.glob("scores__*.npz")):
        _, track, geometry, system, target = f.stem.split("__")
        if track not in TRACKS:
            continue
        z = np.load(f, allow_pickle=False)
        out[(track, geometry, system, target)] = pd.DataFrame(
            {"seq": z["seq"].astype(str), "frame": z["frame"].astype(int), "unit": z["unit"].astype(str),
             "split": z["split"].astype(str), "V": z["V"].astype(float)})
    return run, out


# ------------------------------------------------------------------------------------------------ dump

def stage_dump(run):
    from rap.cache import DetCache
    from rap import decision
    from rap.risk import RiskConfig
    cfg = RiskConfig()
    _, cs = cells()
    for track, (det_dir, cm, fm) in TRACKS.items():
        keys = sorted({(s, f) for (t, *_), d in cs.items() if t == track for s, f in zip(d.seq, d.frame)})
        if track == "KITTI":
            adapter = decision.KittiAdapter
        else:
            from rap.nusc import NuScenesDB, make_adapter
            from rap.paths import NUSCENES_TRAINVAL
            adapter = make_adapter(NuScenesDB(NUSCENES_TRAINVAL, "v1.0-trainval"))
        from rap.geometry import coarse_classes
        boxes = {}
        for seq in sorted({s for s, _ in keys}):
            c = DetCache(Path(CACHE) / det_dir / cm / f"{seq}.npz", frame="camera")    # boxes only: no re-lift
            f = DetCache(Path(CACHE) / det_dir / fm / f"{seq}.npz", frame="camera")
            geom = adapter.geometry(seq)
            for s2, fr in keys:
                if s2 != seq:
                    continue
                g = geom[geom["frame"] == fr]
                g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
                rec = {"gt_xyxy": np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(np.float64) if len(g)
                       else np.zeros((0, 4)), "gt_cls": coarse_classes(g) if len(g) else np.zeros(0, "U8")}
                for tag, cache in (("weak", c), ("strong", f)):
                    d = cache.det(cache.index[fr])
                    rec[f"{tag}_xyxy"] = d["xyxy"].astype(np.float64)
                    rec[f"{tag}_conf"] = d["conf"].astype(np.float64)
                    rec[f"{tag}_cls"] = d["coarse"].astype("U8")
                boxes[(seq, int(fr))] = rec
        with open(run / f"boxes__{track}.pkl", "wb") as fh:
            pickle.dump(boxes, fh, protocol=pickle.HIGHEST_PROTOCOL)
        n_det = sum(len(r["weak_conf"]) + len(r["strong_conf"]) for r in boxes.values())
        n_gt = sum(len(r["gt_cls"]) for r in boxes.values())
        print(f"  {track}: {len(boxes)} frames, {n_det} detections (weak + strong), {n_gt} reference objects", flush=True)


# ------------------------------------------------------------------------------------------------ labels

def _qiu():
    sys.path.insert(0, str(QIU))
    import reward
    from lib.metrics import box_correct
    return reward, box_correct


def _geng():
    """bgt-ada's `proxy_metrics` and `metrics`, unchanged. The package's `__init__` is not run: it imports
    `features.py`, whose Python 3.9 annotations (`tuple[...]` in a signature) fail on this 3.8 environment and which
    nothing here uses; the three modules used (`proxy_metrics`, `metrics`, `error_decomposition`) run on 3.8 as they are."""
    import importlib, types
    if "src" not in sys.modules:
        pkg = types.ModuleType("src")
        pkg.__path__ = [str(GENG / "src")]
        sys.modules["src"] = pkg
    return importlib.import_module("src.proxy_metrics"), importlib.import_module("src.metrics")


def qiu_inputs(boxes, keys, box_correct):
    """`lib/data.set_data`'s per-image tuples, built from our arrays instead of its text files."""
    iouv = np.array([0.5])
    weak, strong, labels = [], [], []
    for k in keys:
        r = boxes[k]
        gcls = np.array([CLASS_ID[c] for c in r["gt_cls"]], dtype=float)
        lab = np.concatenate((gcls[:, None], r["gt_xyxy"]), axis=1) if len(gcls) else None
        out = []
        for tag in ("weak", "strong"):
            m = r[f"{tag}_conf"] >= Q_FLOOR
            xyxy, conf = r[f"{tag}_xyxy"][m], r[f"{tag}_conf"][m]
            cls = np.array([CLASS_ID[c] for c in r[f"{tag}_cls"][m]], dtype=float)
            correct = np.zeros((len(conf), 1), dtype=bool)
            if lab is not None and len(conf):
                correct = box_correct(np.concatenate((xyxy, conf[:, None], cls[:, None]), axis=1), lab, iouv)
            out.append((correct, conf, cls))
        weak.append(out[0]); strong.append(out[1])
        labels.append(gcls if len(gcls) else np.array([]))
    return weak, strong, labels


def labels_q(boxes, keys, reward, box_correct):
    weak, strong, labels = qiu_inputs(boxes, keys, box_correct)
    np.random.seed(SEED)                               # the official code draws E from the unseeded global RNG
    out = np.zeros(len(keys))
    with contextlib.redirect_stdout(io.StringIO()):     # compute_orie prints one line per image
        for i in range(len(keys)):
            out[i] = reward.compute_orie(i, weak, strong, labels, N_ENSEMBLE)
    return np.where(np.isnan(out), 0.0, out)           # reward.py:86


def _ltwh(b):
    """bgt-ada's box convention: (left, top, width, height) (`metrics._compute_iou_vectorized`)."""
    return [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]


def geng_raw(boxes, keys):
    raw = []
    for seq, fr in keys:
        r = boxes[(seq, fr)]
        det = {}
        for tag in ("weak", "strong"):
            m = r[f"{tag}_conf"] >= G_CONF
            det[tag] = [tuple(_ltwh(b)) + (float(c), str(k))
                        for b, c, k in zip(r[f"{tag}_xyxy"][m], r[f"{tag}_conf"][m], r[f"{tag}_cls"][m])]
        raw.append({"frame_id": int(fr), "video_name": str(seq), "edge_dets": det["weak"], "cloud_dets": det["strong"],
                    "gt_objs": [{"bbox": _ltwh(b), "class": str(c)} for b, c in zip(r["gt_xyxy"], r["gt_cls"])]})
    return raw


def labels_g(boxes, keys, proxy_metrics, context_size):
    res = proxy_metrics.compute_dataset_wide_oric(geng_raw(boxes, keys), quiet=True, context_size=context_size,
                                                  context_draws=1)
    return np.array([res.get((str(s), int(f)), {}).get("oric_allpoint", 0.0) for s, f in keys], dtype=float)


def q_moric(values):
    """Qiu et al. `regression.py`: (argsort(argsort(r)) + 1) / n over the training rewards."""
    return (np.argsort(np.argsort(values, kind="quicksort"), kind="quicksort") + 1) / len(values)


def g_moric(values):
    """Geng et al. `prepare_transforms._apply_moric` with the reference fitted on the same (training) values."""
    ref = np.sort(np.asarray(values, dtype=np.float64))
    return np.searchsorted(ref, values, side="right") / max(len(ref), 1)


def stage_labels(run, dump_run):
    reward, box_correct = _qiu()
    proxy_metrics, _ = _geng()
    r1_run, cs = cells()
    boxes = {}
    for t in TRACKS:
        with open(dump_run / f"boxes__{t}.pkl", "rb") as fh:
            boxes[t] = pickle.load(fh)
    memo = {}
    for (track, geometry, system, target), d in cs.items():
        t0 = time.time()
        out = d.copy()
        for col in ("Q_oric", "G_dap", "G_dap_ctx1000", "Q_moric", "G_moric", "Q_bin", "G_bin",
                    "G_dap_train", "G_moric_train", "G_bin_train"):
            out[col] = np.nan
        for set_name, mask in (("fit", d.split.isin(["train", "val"]).to_numpy()), ("test", (d.split == "test").to_numpy())):
            keys = [(s, int(f)) for s, f in zip(d.seq[mask], d.frame[mask])]
            sig = (track, set_name, tuple(keys))
            if sig not in memo:
                q = labels_q(boxes[track], keys, reward, box_correct)
                g = labels_g(boxes[track], keys, proxy_metrics, 0)
                g1k = (labels_g(boxes[track], keys, proxy_metrics, G_CONTEXT) if len(keys) - 1 > G_CONTEXT else g.copy())
                memo[sig] = (q, g, g1k)
            q, g, g1k = memo[sig]
            out.loc[mask, "Q_oric"], out.loc[mask, "G_dap"], out.loc[mask, "G_dap_ctx1000"] = q, g, g1k
            out.loc[mask, "Q_bin"], out.loc[mask, "G_bin"] = (q > 0).astype(float), (g > 0).astype(float)
            if set_name == "fit":                              # training targets exist on the fit units only
                out.loc[mask, "Q_moric"], out.loc[mask, "G_moric"] = q_moric(q), g_moric(g)
        # G on the train units alone, for the budget arbiter's train-only estimators (no validation unit in the context)
        mask = (d.split == "train").to_numpy()
        keys = [(s, int(f)) for s, f in zip(d.seq[mask], d.frame[mask])]
        sig = (track, "train", tuple(keys))
        if sig not in memo:
            memo[sig] = (None, labels_g(boxes[track], keys, proxy_metrics, 0), None)
        gtr = memo[sig][1]
        out.loc[mask, "G_dap_train"], out.loc[mask, "G_moric_train"] = gtr, g_moric(gtr)
        out.loc[mask, "G_bin_train"] = (gtr > 0).astype(float)
        np.savez_compressed(run / f"labels__{track}__{geometry}__{system}__{target}.npz",
                            **{c: (out[c].to_numpy().astype("U40") if out[c].dtype == object else out[c].to_numpy())
                               for c in out.columns})
        fit = out.split.isin(["train", "val"])
        print(f"  {track:8s} {geometry:6s} {system:9s} {target:7s}: {len(out)} frames; fit Q>0 {out.Q_bin[fit].mean():.3f} "
              f"G>0 {out.G_bin[fit].mean():.3f} Q=0 {(out.Q_oric[fit] == 0).mean():.3f} G=0 {(out.G_dap[fit] == 0).mean():.3f}"
              f" [{time.time() - t0:.0f}s]", flush=True)
    (run / "labels_meta.json").write_text(json.dumps(
        {"routers_r1_run": r1_run.name, "dump_run": dump_run.name, "q_floor": Q_FLOOR, "g_conf": G_CONF,
         "n_ensemble": N_ENSEMBLE, "g_context_sensitivity": G_CONTEXT, "seed": SEED,
         "qiu_commit": "859f70240aa090359ba827b374b68ca7821b7d55",
         "geng_commit": "6669ab0089a04fbe6257ebdc2601de13ed0e5398"}, indent=1))


# ------------------------------------------------------------------------------------------------ checks

def pooled_ap50(frames_dets, gt):
    """Independent implementation: one PR curve over all frames and classes, greedy by confidence, all-point AP."""
    dets = [(c, k, b, cl) for k, lst in frames_dets.items() for (b, c, cl) in lst]
    dets.sort(key=lambda x: -x[0])
    total = sum(len(v[1]) for v in gt.values())
    used = {k: np.zeros(len(v[1]), bool) for k, v in gt.items()}
    tp = np.zeros(len(dets))
    for i, (c, k, b, cl) in enumerate(dets):
        g_xyxy, g_cls = gt[k]
        if not len(g_cls):
            continue
        x1 = np.maximum(b[0], g_xyxy[:, 0]); y1 = np.maximum(b[1], g_xyxy[:, 1])
        x2 = np.minimum(b[2], g_xyxy[:, 2]); y2 = np.minimum(b[3], g_xyxy[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        union = (b[2] - b[0]) * (b[3] - b[1]) + (g_xyxy[:, 2] - g_xyxy[:, 0]) * (g_xyxy[:, 3] - g_xyxy[:, 1]) - inter
        iou = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
        ok = (g_cls == cl) & ~used[k]
        if ok.any():
            j = int(np.argmax(np.where(ok, iou, -1.0)))
            if iou[j] >= 0.5:
                tp[i] = 1; used[k][j] = True
    ctp = np.cumsum(tp); prec = ctp / np.arange(1, len(dets) + 1); rec = ctp / max(total, 1)
    r = np.concatenate(([0.0], rec)); p = np.concatenate(([1.0], prec))
    p = np.maximum.accumulate(p[::-1])[::-1]
    return float(np.sum((r[1:] - r[:-1]) * p[1:]))


def stage_checks(run, label_run, dump_run):
    proxy_metrics, gm = _geng()
    reward, box_correct = _qiu()
    _, cs = cells()
    boxes = {}
    for t in TRACKS:
        with open(dump_run / f"boxes__{t}.pkl", "rb") as fh:
            boxes[t] = pickle.load(fh)
    rows, corr = [], []
    rng = np.random.default_rng(0)
    done = set()
    for (track, geometry, system, target), d in cs.items():
        lab = np.load(label_run / f"labels__{track}__{geometry}__{system}__{target}.npz", allow_pickle=False)
        cell = f"{track} {geometry} {system} {target}"
        fit = np.isin(lab["split"], ["train", "val"])
        # (2) no test quantity in a fit: training targets exist on fit units only
        rows.append({"check": "targets only on fit units", "cell": cell,
                     "ok": bool(np.isnan(lab["Q_moric"][~fit]).all() and np.isnan(lab["G_moric"][~fit]).all()
                                and np.isfinite(lab["Q_moric"][fit]).all() and np.isfinite(lab["G_moric"][fit]).all())})
        # (3) correlations with the perception-gain labels, fit units
        core = pd.read_pickle(rap_runs.core_matrix("postreview") /
                              (f"KITTI__YOLOv8s__cheap_320tofull_640__{geometry}.pkl" if track == "KITTI"
                               else f"nuScenes__YOLOv8s__ns_cheap_320tons_full_640__{geometry}.pkl"))
        core = core.astype({"seq": str, "frame": int})
        j = pd.DataFrame({"seq": lab["seq"], "frame": lab["frame"], "fit": fit, "Q": lab["Q_oric"], "G": lab["G_dap"],
                          "G1k": lab["G_dap_ctx1000"]}).merge(core[["seq", "frame", *DE_LABELS]], on=["seq", "frame"],
                                                              how="left", validate="one_to_one")
        from scipy.stats import spearmanr
        for part, m in (("fit", j.fit.to_numpy()), ("test", ~j.fit.to_numpy())):
            for a in ("Q", "G"):
                for b in DE_LABELS:
                    corr.append({"cell": cell, "units": part, "label": a, "against": "dE_exact" if b == "dE" else b,
                                 "spearman": float(spearmanr(j[a][m], j[b][m]).correlation), "n": int(m.sum())})
            corr.append({"cell": cell, "units": part, "label": "Q", "against": "G",
                         "spearman": float(spearmanr(j.Q[m], j.G[m]).correlation), "n": int(m.sum())})
            corr.append({"cell": cell, "units": part, "label": "G", "against": "G context 1000 (code default)",
                         "spearman": float(spearmanr(j.G[m], j.G1k[m]).correlation), "n": int(m.sum())})
        # (1) exactness, once per distinct frame set
        for set_name, m in (("fit", fit), ("test", ~fit)):
            keys = [(str(s), int(f)) for s, f in zip(lab["seq"][m], lab["frame"][m])]
            sig = (track, set_name, tuple(keys))
            if sig in done:
                continue
            done.add(sig)
            b = boxes[track]
            # G: the official base AP against the independent pooled AP, and 50 swaps recomputed from scratch
            raw = geng_raw(b, keys)
            # the independent path reads the dumped corner boxes directly, not the official code's inputs
            gt = {k: (b[k]["gt_xyxy"], b[k]["gt_cls"].astype(str)) for k in keys}

            def dets(tag):
                out = {}
                for k in keys:
                    mm = b[k][f"{tag}_conf"] >= G_CONF
                    out[k] = list(zip(b[k][f"{tag}_xyxy"][mm], b[k][f"{tag}_conf"][mm], b[k][f"{tag}_cls"][mm].astype(str)))
                return out
            weak, strong = dets("weak"), dets("strong")
            all_edge = sorted([(dd, r["video_name"], r["frame_id"]) for r in raw for dd in r["edge_dets"]],
                              key=lambda x: x[0][4], reverse=True)
            gt_by = {(r["video_name"], r["frame_id"]): r["gt_objs"] for r in raw}
            total_gt = sum(len(v) for v in gt_by.values())
            base_off = gm._run_greedy_matching(gm._precompute_detection_ious(all_edge, gt_by), gt_by, total_gt, 0.5)["ap_allpoint"]
            base_ind = pooled_ap50(weak, gt)
            rows.append({"check": "G base AP: official vs independent", "cell": cell, "set": set_name,
                         "official": base_off, "independent": base_ind, "abs_diff": abs(base_off - base_ind),
                         "ok": abs(base_off - base_ind) < 1e-12})
            g = lab["G_dap"][m]
            n_set = len(keys)
            all_conf = np.concatenate([np.asarray([c for _, c, _ in v], float) for v in weak.values() if v] or [np.zeros(0)])
            worst = worst_ap = 0.0
            differing = tied = flips = 0
            for i in rng.choice(n_set, size=min(50, n_set), replace=False):
                k = keys[i]
                # as first written: the swapped frame keeps its place, so at equal confidence frame order decides
                sw = dict(weak); sw[k] = strong[k]
                d_stable = (pooled_ap50(sw, gt) - base_ind) * n_set
                worst = max(worst, abs(d_stable - g[i]))
                # the official order (`proxy_metrics._merge_swapped_precomputed`): FULL detections of the swapped
                # frame come before CHEAP detections of equal confidence
                sw2 = {k: strong[k]}; sw2.update((kk, v) for kk, v in weak.items() if kk != k)
                ap_swap = pooled_ap50(sw2, gt)
                worst_ap = max(worst_ap, abs(ap_swap - (base_ind + g[i] / n_set)))
                if abs(d_stable - g[i]) > 1e-9:
                    differing += 1
                    own = np.asarray([c for _, c, _ in weak[k]], float)
                    tied += any((all_conf == c).sum() > (own == c).sum() for _, c, _ in strong[k])
                    flips += (d_stable > 0) != (g[i] > 0)
            rows.append({"check": "G dAP: 50 swaps recomputed independently", "cell": cell, "set": set_name,
                         "abs_diff": worst, "ok": worst < 1e-9, "frames_differing": differing,
                         "differing_with_cross_frame_conf_tie": tied, "offloadbin_flips": flips,
                         "note": "stable re-sort: at equal confidence the swapped frame keeps its frame-order place; "
                                 "reported, not a stop (see docs/iclr_published_objectives.md, Phase B)"})
            rows.append({"check": "G dAP: 50 swaps, official tie order, AP_swap = base + dAP/N", "cell": cell,
                         "set": set_name, "abs_diff": worst_ap, "ok": worst_ap < 1e-12,
                         "note": "FULL detections of the swapped frame before CHEAP detections of equal confidence, as "
                                 "the official merge orders them; compared in AP units, as registered"})
            # Q: replay the seeded ensembles and recompute mAPC with an independent per-class AP (101-point)
            wq, sq, lq = qiu_inputs(b, keys, box_correct)
            np.random.seed(SEED)
            n = len(keys); ne = min(N_ENSEMBLE, n - 1)
            pick = set(rng.choice(n, size=min(50, n), replace=False).tolist())
            worst = 0.0
            q = lab["Q_oric"][m]
            for i in range(n):
                idx = np.arange(n - 1)
                if i < n - 1:
                    idx[i:] += 1
                idx = np.random.permutation(idx)[:ne]
                if i not in pick:
                    continue
                labels_all = np.concatenate([lq[s] for s in idx] + [lq[i]]).astype(int)
                vals = []
                for det in (wq, sq):
                    parts = [wq[s] for s in idx] + [det[i]]
                    tp = np.concatenate([p[0] for p in parts])[:, 0]
                    conf = np.concatenate([p[1] for p in parts]); cls = np.concatenate([p[2] for p in parts])
                    o = np.argsort(-conf)                     # the official order, ties included
                    tp, conf, cls = tp[o], conf[o], cls[o]
                    aps = []
                    for c in np.unique(labels_all):
                        sel = cls == c
                        n_l = int((labels_all == c).sum())
                        if not sel.any():
                            aps.append(0.0); continue
                        t = tp[sel].astype(float)
                        ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
                        recall, precision = ctp / (n_l + 1e-16), ctp / (ctp + cfp)
                        mrec = np.concatenate(([0.0], recall, [1.0])); mpre = np.concatenate(([1.0], precision, [0.0]))
                        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
                        x = np.linspace(0, 1, 101)
                        aps.append(float(np.trapz(np.interp(x, mrec, mpre), x)))
                    vals.append(np.mean(aps) if aps else np.nan)
                ind = (vals[1] - vals[0]) * (ne + 1)
                ind = 0.0 if np.isnan(ind) else ind
                worst = max(worst, abs(ind - q[i]))
            rows.append({"check": "Q ORIC: 50 images recomputed independently", "cell": cell, "set": set_name,
                         "abs_diff": worst, "ok": worst < 1e-9})
    c = pd.DataFrame(rows); k = pd.DataFrame(corr)
    c.to_csv(run / "checks.csv", index=False); k.to_csv(run / "correlations.csv", index=False)
    k.to_csv(Path(RESULTS) / "final" / "published_objective_labels.csv", index=False)
    print(c.groupby("check").ok.agg(["sum", "count"]).to_string())
    stable = c.check == "G dAP: 50 swaps recomputed independently"
    if stable.any() and not bool(c[stable].ok.all()):
        print("  the stable-order swap check differs from the official labels at confidence ties: "
              f"{int(c[stable].frames_differing.sum())} of the sampled frames, all with a cross-frame tie: "
              f"{int(c[stable].differing_with_cross_frame_conf_tie.sum())}; OffloadBin flips: "
              f"{int(c[stable].offloadbin_flips.sum())}; max |diff| {c[stable].abs_diff.max():.3g} (dAP x N units)")
    if not bool(c[~stable].ok.all()):
        raise SystemExit("a Phase B check failed: see checks.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dump", "labels", "checks"])
    ap.add_argument("--dump_run", default=None)
    ap.add_argument("--label_run", default=None)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"published_objective_{args.stage}", vars(args))
    if args.stage == "dump":
        stage_dump(run)
    elif args.stage == "labels":
        stage_labels(run, Path(args.dump_run) if args.dump_run else rap_runs.latest("published_objective_dump"))
    else:
        stage_checks(run, Path(args.label_run) if args.label_run else rap_runs.latest("published_objective_labels"),
                     Path(args.dump_run) if args.dump_run else rap_runs.latest("published_objective_dump"))
    print("wrote", run)


if __name__ == "__main__":
    main()
