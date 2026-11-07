#!/usr/bin/env python
"""Follow-up round 2, R1: training-target swap on XL (addendum R1 in docs/followup_protocol.md).

The X2 allocators gate_gbm (primary) and gate_ridge (secondary), with X2's features, hyperparameters, seeds and fit
units (the 385 + 152 new train / validation scenes), trained on another target and scored once by decision value on
the 228 new test scenes. Only the target changes. Targets: V (as in X2), false-negative gain (dE_E1_fn_only), risk-
weighted gain (dE_E6_risk_weighted), ORIC's offloading reward (Q, Qiu et al.) and the AP-swap objective of budget-
adaptive routing (G, Geng et al.); Q and G per frame as in the frozen published-objective study (148).

  --stage dump    the XL frames' cheap / full detections and reference objects, as 148's dump -> boxes.pkl
  --stage labels  Q and G per label set (the brake cells' frames, the planner cells' frames), each on the fit units
                  and on the test units separately, 148's functions -> labels__<set>.npz
  --stage targets every target per frame -> results/followup/R1/targets.csv
  --stage gate    gate R1-0: the V-trained gate_gbm, refit here, reproduces X2's test rows (every cell, quota and
                  bootstrap unit of x3_budget_sweep.csv) exactly -> gate_r1_0.json
  --stage swap    every learner x target x variant x cell x budget x bootstrap unit -> swap.csv
"""
from __future__ import annotations

import argparse, importlib.util, json, pickle, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta, xl                                                       # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402

OUT = Path(RESULTS) / "followup" / "R1"
WORK = Path(CACHE) / "xl" / "r1_target_swap"
DET = Path(CACHE) / "nusc_det_xl"
CHEAP, FULL = "ns_cheap_320", "ns_full_640"
TABLES = Path(CACHE) / "xl" / "tables850"


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def xl_frames():
    """The new 765 scenes' frames of the brake cells and of the planner cells, with split (X2's cell frames)."""
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    j = pd.read_pickle(TABLES / f"mono__{xl.PAIR}.pkl")
    j = j[j.seq.isin(set(split_of))].copy()
    j["split"] = j.seq.map(split_of)
    brake = j[j.J_cheap.notna()][["seq", "frame", "split"]].reset_index(drop=True)
    plan = j[j.JC_ade_cheap.notna()][["seq", "frame", "split"]].reset_index(drop=True)
    return {"brake": brake, "plan": plan}


def stage_dump(args):
    """148's dump on the XL frames: cheap (weak) and full (strong) boxes from the XL caches, and the benchmark's
    reference objects (>= 10 px tall, coarse class) from the XL dataroot."""
    from rap.cache import DetCache
    from rap.geometry import coarse_classes
    from rap.nusc import NuScenesDB, make_adapter
    from rap.risk import RiskConfig
    cfg = RiskConfig()
    keys = sorted({(s, int(f)) for d in xl_frames().values() for s, f in zip(d.seq, d.frame)})
    adapter = make_adapter(NuScenesDB(args.dataroot, "v1.0-trainval"))
    boxes = {}
    by_seq = {}
    for s, f in keys:
        by_seq.setdefault(s, []).append(f)
    t0 = time.time()
    for i, seq in enumerate(sorted(by_seq)):
        c = DetCache(DET / CHEAP / f"{seq}.npz", frame="camera")
        fu = DetCache(DET / FULL / f"{seq}.npz", frame="camera")
        geom = adapter.geometry(seq)
        for fr in by_seq[seq]:
            g = geom[geom["frame"] == fr]
            g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
            rec = {"gt_xyxy": np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(np.float64) if len(g)
                   else np.zeros((0, 4)), "gt_cls": coarse_classes(g) if len(g) else np.zeros(0, "U8")}
            for tag, cache in (("weak", c), ("strong", fu)):
                d = cache.det(cache.index[fr])
                rec[f"{tag}_xyxy"] = d["xyxy"].astype(np.float64)
                rec[f"{tag}_conf"] = d["conf"].astype(np.float64)
                rec[f"{tag}_cls"] = d["coarse"].astype("U8")
            boxes[(seq, int(fr))] = rec
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} / {len(by_seq)} scenes [{time.time() - t0:.0f}s]", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "boxes.pkl", "wb") as fh:
        pickle.dump(boxes, fh, protocol=pickle.HIGHEST_PROTOCOL)
    n_det = sum(len(r["weak_conf"]) + len(r["strong_conf"]) for r in boxes.values())
    n_gt = sum(len(r["gt_cls"]) for r in boxes.values())
    print(f"dump: {len(boxes)} frames of {len(by_seq)} scenes, {n_det} detections (weak + strong), {n_gt} reference "
          f"objects", flush=True)


# ------------------------------------------------------------------------------------------------ labels

def stage_labels(args):
    """148's labels_q and labels_g, unchanged, on each label set: fit units and test units separately, each its own
    context (Q: |E| = 1000, global RNG seeded 42 at each call, detections conf >= 0.10; G: conf >= 0.30, all-point
    AP, context_size as given)."""
    t148 = _load("t148", "148_published_objective_labels.py")
    reward, box_correct = t148._qiu()
    proxy_metrics, _ = t148._geng()
    with open(WORK / "boxes.pkl", "rb") as fh:
        boxes = pickle.load(fh)
    for name, d in xl_frames().items():
        out = WORK / f"labels__{name}.npz"
        if out.exists():
            continue
        Q, G = np.full(len(d), np.nan), np.full(len(d), np.nan)
        for set_name, mask in (("fit", d.split.isin(["train", "val"]).to_numpy()), ("test", (d.split == "test").to_numpy())):
            keys = [(s, int(f)) for s, f in zip(d.seq[mask], d.frame[mask])]
            t0 = time.time()
            G[mask] = t148.labels_g(boxes, keys, proxy_metrics, args.g_context)
            t1 = time.time()
            Q[mask] = t148.labels_q(boxes, keys, reward, box_correct)
            print(f"  {name} {set_name}: {len(keys)} frames, G [{t1 - t0:.0f}s], Q [{time.time() - t1:.0f}s]", flush=True)
        np.savez_compressed(out, seq=d.seq.to_numpy().astype("U40"), frame=d.frame.to_numpy(int),
                            split=d.split.to_numpy().astype("U8"), Q_oric=Q, G_dap=G)
    (WORK / "labels_meta.json").write_text(json.dumps(
        {"g_context": args.g_context, "q_floor": t148.Q_FLOOR, "g_conf": t148.G_CONF, "n_ensemble": t148.N_ENSEMBLE,
         "seed": t148.SEED, "qiu_commit": "859f70240aa090359ba827b374b68ca7821b7d55",
         "geng_commit": "6669ab0089a04fbe6257ebdc2601de13ed0e5398"}, indent=1))


# ------------------------------------------------------------------------------------------------ targets

TARGETS = ("V", "fn_gain", "risk_gain", "Q", "G")
OTHER = TARGETS[1:]
LEARNERS = ("gate_gbm", "gate_ridge")
COLUMN = {"fn_gain": "dE_E1_fn_only", "risk_gain": "dE_E6_risk_weighted"}


def label_frame(name):
    z = np.load(WORK / f"labels__{name}.npz", allow_pickle=False)
    return pd.DataFrame({"seq": z["seq"].astype(str), "frame": z["frame"].astype(int), f"Q_{name}": z["Q_oric"],
                         f"G_{name}": z["G_dap"]})


def stage_targets(args):
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    log_of = dict(zip(*[xl.scene_meta(args.dataroot)[c] for c in ("scene", "log")]))
    t = None
    for geom in xl.GEOMETRIES:
        j = pd.read_pickle(TABLES / f"{geom}__{xl.PAIR}.pkl")
        j = j[j.seq.isin(set(split_of))]
        v = j[["seq", "frame"]].copy()
        for system, cc, cf in xl.SYSTEMS:
            v[f"V_{system}_{geom}"] = j[cc] - j[cf]
        if t is None:
            t = j[["seq", "frame", COLUMN["fn_gain"], COLUMN["risk_gain"]]].rename(
                columns={c: k for k, c in COLUMN.items()}).merge(v, on=["seq", "frame"], validate="one_to_one")
        else:
            # the perception-level targets do not depend on the lift's geometry
            chk = j[["seq", "frame", COLUMN["fn_gain"], COLUMN["risk_gain"]]].merge(
                t[["seq", "frame", "fn_gain", "risk_gain"]], on=["seq", "frame"], validate="one_to_one")
            assert np.allclose(chk[COLUMN["fn_gain"]], chk.fn_gain) and np.allclose(chk[COLUMN["risk_gain"]], chk.risk_gain)
            t = t.merge(v, on=["seq", "frame"], validate="one_to_one")
    t = t[t.V_brake_mono.notna()].copy()
    t.insert(2, "split", t.seq.map(split_of))
    t.insert(3, "log", t.seq.map(log_of))
    for name in ("brake", "plan"):
        t = t.merge(label_frame(name), on=["seq", "frame"], how="left", validate="one_to_one")
    t148 = _load("t148", "148_published_objective_labels.py")
    for name in ("brake", "plan"):                                  # published CDF forms, on the fit units only
        fit = t.split.isin(["train", "val"]) & t[f"Q_{name}"].notna()
        t[f"Q_moric_{name}"], t[f"G_moric_{name}"] = np.nan, np.nan
        t.loc[fit, f"Q_moric_{name}"] = t148.q_moric(t.loc[fit, f"Q_{name}"].to_numpy(float))
        t.loc[fit, f"G_moric_{name}"] = t148.g_moric(t.loc[fit, f"G_{name}"].to_numpy(float))
    OUT.mkdir(parents=True, exist_ok=True)
    t.sort_values(["seq", "frame"]).to_csv(OUT / "targets.csv", index=False)
    print(f"targets: {len(t)} frames, columns {list(t.columns)}", flush=True)


# ------------------------------------------------------------------------------------------------ models and draws

def rank_v(y):
    """159's rank transform (ordinal ranks / n, ties as 148's Q-MORIC), on the fit units."""
    return (np.argsort(np.argsort(y, kind="quicksort"), kind="quicksort") + 1) / len(y)


def setup(args):
    """X2's cells, features and ego table (172's xl_run), and the per-frame targets."""
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    xl.use_ego_table(TABLES / "v_ego_causal.csv")
    feats, fcols = xl.features(args.det)
    cells_ = list(xl.cells(TABLES, feats, fcols, split_of))
    log_of = dict(zip(*[xl.scene_meta(args.dataroot)[c] for c in ("scene", "log")]))
    return cells_, log_of


def target_arrays(d, tg, label_set):
    """Full-length arrays per target name (raw), aligned with the cell's rows."""
    j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(tg, on=["seq", "frame"], how="left",
                                                                     validate="one_to_one")
    out = {"fn_gain": j.fn_gain.to_numpy(float), "risk_gain": j.risk_gain.to_numpy(float),
           "Q": j[f"Q_{label_set}"].to_numpy(float), "G": j[f"G_{label_set}"].to_numpy(float),
           "Q_cdf": j[f"Q_moric_{label_set}"].to_numpy(float), "G_cdf": j[f"G_moric_{label_set}"].to_numpy(float)}
    return out


def fit_series(c, tg, learners, memo):
    """Scores of every learner x variant x target on one cell. Non-V targets do not depend on the cell's geometry or
    controller, so their fits are shared between cells with the same frames (memo); V is the cell's own."""
    d = c["d"]
    fit = d.split.isin(["train", "val"]).to_numpy()
    label_set = "brake" if c["system"] == "brake" else "plan"
    arrays = target_arrays(d, tg, label_set)
    arrays["V"] = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    combos = [("raw", t, t) for t in TARGETS] + [("rank", t, t) for t in TARGETS] + [("cdf", t, f"{t}_cdf") for t in ("Q", "G")]
    series = {}
    for learner in learners:
        for variant, t, key in combos:
            y = arrays[key].copy()
            assert np.isfinite(y[fit]).all(), (c["system"], key)
            if variant == "rank":
                y[fit] = rank_v(y[fit])
            mk = (label_set, learner, variant, t) if t != "V" else None
            if mk is not None and mk in memo:
                series[f"{learner}|{variant}|{t}"] = memo[mk]
                continue
            s = xl.gate_scores(d, c["fcols"], y, fit, which=(learner,))[learner]
            if mk is not None:
                memo[mk] = s
            series[f"{learner}|{variant}|{t}"] = s
    return series


def two_sided_p(x):
    x = x[np.isfinite(x)]
    return float(min(1.0, 2 * min(np.mean(x <= 0), np.mean(x >= 0)))) if len(x) else np.nan


# ------------------------------------------------------------------------------------------------ gate R1-0

GATE_COLS = ["gain", "gain_random", "delta_vs_random", "delta_lo_all_draws", "delta_hi_all_draws",
             "delta_share_all_cheap", "delta_share_lo_all_draws", "delta_share_hi_all_draws", "ndg",
             "delta_lo_filtered", "delta_hi_filtered"]


def stage_gate(args):
    cells_, log_of = setup(args)
    x3 = pd.read_csv(xl.OUT / "x3_budget_sweep.csv", float_precision="round_trip")
    x3 = x3[x3.signal == "gate_gbm"]
    report, fails, n = {}, [], 0
    for unit in ("scene", "log"):
        prep = []
        for c in cells_:
            d = c["d"] if unit == "scene" else c["d"].assign(unit=c["d"].seq.map(log_of))
            fit = d.split.isin(["train", "val"]).to_numpy()
            test = (d.split == "test").to_numpy()
            v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
            sc = xl.gate_scores(d, c["fcols"], v_all, fit, which=("gate_gbm",))
            prep.append(xl.prepare(dict(c, d=d), test, sc, xl.SWEEP))
        draws, keep = xl.paired_draws(prep, 1000, np.random.default_rng(0), xl.SWEEP)
        rows = pd.DataFrame(xl.paired_rows(prep, draws, keep, xl.SWEEP))
        for r in rows.itertuples():
            m = x3[(x3.geometry == r.geometry) & (x3.system == r.system) & (x3.quota == r.quota) & (x3.bootstrap_unit == unit)]
            assert len(m) == 1, (r.geometry, r.system, r.quota, unit)
            m = m.iloc[0]
            for k in GATE_COLS:
                a, b = float(getattr(r, k)), float(m[k])
                n += 1
                if not ((np.isnan(a) and np.isnan(b)) or a == b):
                    fails.append(f"{unit} {r.geometry} {r.system} {r.quota} {k}: {a!r} vs {b!r}")
    report = {"comparisons": n, "differing": len(fails), "pass": not fails, "first_differences": fails[:20],
              "compared": "gate_gbm refit on V (X2's features, model, seed and fit units) against x3_budget_sweep.csv: "
                          + ", ".join(GATE_COLS) + "; every cell, quota and bootstrap unit"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_r1_0.json").write_text(json.dumps(report, indent=1))
    print("gate R1-0:", {k: report[k] for k in ("comparisons", "differing", "pass")}, flush=True)
    if fails:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ swap

def stage_swap(args):
    cells_, log_of = setup(args)
    tg = pd.read_csv(OUT / "targets.csv", dtype={"seq": str})
    memo, scored = {}, []
    for c in cells_:
        t0 = time.time()
        scored.append(fit_series(c, tg, LEARNERS, memo))
        print(f"  {c['geometry']:6s} {c['system']:8s} {len(scored[-1])} series [{time.time() - t0:.0f}s]", flush=True)
    rows = []
    for unit in ("scene", "log"):
        prep = []
        for c, series in zip(cells_, scored):
            d = c["d"] if unit == "scene" else c["d"].assign(unit=c["d"].seq.map(log_of))
            prep.append(xl.prepare(dict(c, d=d), (d.split == "test").to_numpy(), series, xl.SWEEP))
        draws, keep = xl.paired_draws(prep, 1000, np.random.default_rng(0), xl.SWEEP)
        base = pd.DataFrame(xl.paired_rows(prep, draws, keep, xl.SWEEP, {"bootstrap_unit": unit}))
        for ci_, p in enumerate(prep):
            for s in p["series"]:
                learner, variant, t = s.split("|")
                ref = f"{learner}|{'raw' if variant == 'cdf' else variant}|V"
                g = xl.gains(p["series"][s], p["v"], p["ks"])
                gv = xl.gains(p["series"][ref], p["v"], p["ks"])
                for qi, q in enumerate(xl.SWEEP):
                    r = base[(base.geometry == p["key"][1]) & (base.system == p["key"][2]) & (base.signal == s)
                             & (base.quota == q)].iloc[0].to_dict()
                    r.update(learner=learner, variant=variant, target=t, comparison_with=ref)
                    if t != "V":
                        x = draws[(ci_, ref)][:, qi] - draws[(ci_, s)][:, qi]      # gain(V) - gain(t), random cancels
                        lo, hi = xl.ci(x)
                        dd = float(gv[qi] - g[qi])
                        r.update(D=dd, D_lo=lo, D_hi=hi, D_share_all_cheap=dd / p["tot"], D_share_lo=lo / p["tot"],
                                 D_share_hi=hi / p["tot"], p_two_sided=two_sided_p(x))
                    rows.append(r)
    s = pd.DataFrame(rows)
    s["primary"] = ((s.geometry == "mono") & (s.system == "brake") & (s.learner == "gate_gbm") & (s.variant == "raw")
                    & (s.quota == 0.2) & (s.bootstrap_unit == "scene") & (s.target != "V"))
    s["holm_rejects"] = np.nan
    m = s.primary
    s.loc[m, "holm_rejects"] = xl.holm(s.loc[m, "p_two_sided"].to_numpy(), alpha=0.05)
    s["reading"] = ""
    s.loc[m, "reading"] = np.where(s.loc[m, "holm_rejects"].astype(bool),
                                   np.where(s.loc[m, "D"] > 0, "V-trained better", "t-trained better"), "not resolved")
    s.to_csv(OUT / "swap.csv", index=False)
    print(s[m][["target", "D_share_all_cheap", "D_share_lo", "D_share_hi", "p_two_sided", "holm_rejects", "reading"]]
          .to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["dump", "labels", "targets", "gate", "swap"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    ap.add_argument("--det", default=str(DET))
    ap.add_argument("--g_context", type=int, default=1000, help="G context_size (addendum default 3; 0 = every frame of the set)")
    args = ap.parse_args()
    runmeta.new_run(f"r1_{args.stage}", vars(args))
    {"dump": stage_dump, "labels": stage_labels, "targets": stage_targets, "gate": stage_gate,
     "swap": stage_swap}[args.stage](args)


if __name__ == "__main__":
    main()
