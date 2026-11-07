#!/usr/bin/env python
"""Follow-up round 2, R3: what the gate's features cost (addendum R3 in docs/followup_protocol.md).

The gate's 65 features in their four existing groups (confidence statistics, uncertainty, scene complexity,
criticality from the cheap detections' monocular geometry); the 15 non-empty subsets; no new feature.

  --stage profile   (Jetson, exclusive) one session: an idle baseline; three interleaved rounds of every subset's
                    extractor and every fitted model's inference (gate_gbm batched and per input, gate_ridge), each
                    40 warm-up and 250 timed calls cycling over 200 fixed nuScenes frames, rails sampled during the
                    timed calls; then the cheap and full engines with 01_profile_jetson.py as x7_costs.csv was measured
                    -> feature_costs.csv (round 0 discarded; median of the kept rounds; module mJ as 133)
  --stage select    each subset fit on the 385 training scenes; realized gain on the 152 validation scenes at the 20%
                    latency budget with the subset's C_S charged; the maximum, ties toward the lower C_S
                    -> subsets_val.csv, selected.json
  --stage test      the chosen subset refit on the 537 scenes and scored once on the 228 test scenes; against random
                    (scene bootstrap); budgets; the cost-gain table of all 15 subsets (descriptive) -> test.csv
  --stage frozen    the subset chosen on XL, without retuning, on the frozen 85 nuScenes scenes and on KITTI braking:
                    fit on each track's non-test units, scored once on its test split -> frozen_kitti.csv
"""
from __future__ import annotations

import argparse, glob, importlib.util, itertools, json, subprocess, sys, threading, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import features as F, geometry as G, power, predict, runmeta, xl       # noqa: E402
from rap.cache import DetCache                                                  # noqa: E402
from rap.paths import CACHE, MODELS, NUSCENES_TRAINVAL, RESULTS                 # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

OUT = Path(RESULTS) / "followup" / "R3"
WORK = Path(CACHE) / "xl" / "r3"
RAW = ROOT / "results" / "raw"
TABLES = Path(CACHE) / "xl" / "tables850"
DET = Path(CACHE) / "nusc_det_xl"
CHEAP = "ns_cheap_320"
GROUPS = ("conf", "unc", "complex", "crit")
GROUP_NAME = {"conf": "confidence statistics", "unc": "uncertainty", "complex": "scene complexity",
              "crit": "criticality"}
SUBSETS = ["+".join(c) for r in range(1, 5) for c in itertools.combinations(GROUPS, r)]
LEARNERS = {"gate_gbm": "gbm", "gate_ridge": "linear"}
OP = RiskConfig().op_conf
N_FRAMES, REPS, WARMUP, ROUNDS = 200, 250, 40, 3
BUDGETS = (0.10, 0.20, 0.30, 0.50)
NBOOT = 1000
IMAGES = str(NUSCENES_TRAINVAL / "samples" / "CAM_FRONT")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def group_columns(fcols):
    reg = F.registry()
    out = {g: [c for c in fcols if reg[c][0] == g] for g in GROUPS}
    assert sum(len(v) for v in out.values()) == len(fcols)
    return out


def subset_columns(subset, groups):
    return [c for g in GROUPS if g in subset.split("+") for c in groups[g]]


# ------------------------------------------------------------------------------------------------ data

def xl_cells():
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    xl.use_ego_table(TABLES / "v_ego_causal.csv")
    feats, fcols = xl.features(str(DET))
    cells_ = [c for c in xl.cells(TABLES, feats, fcols, split_of) if c["system"] == "brake"]
    return cells_, fcols


def fit_predict(d, cols, v, fit, learner):
    X = np.nan_to_num(d[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64), nan=0.0, posinf=1e6,
                      neginf=-1e6)
    m = predict.make_model(LEARNERS[learner], "reg", 0)          # X2's model and seed (rap.xl.gate_scores)
    m.fit(X[fit], v[fit])
    return m, m.predict(X)


# ------------------------------------------------------------------------------------------------ profile

def profile_frames():
    """200 fixed nuScenes frames: the first 200 frames of the 228 XL test scenes in (scene, frame) order, as the
    gate receives them at the frozen track's timing boundary (93's `_frames`: the cheap detections, their monocular
    geometry, the image statistics the cheap pipeline computes, the image area)."""
    sp = xl.scene_split()
    test = sorted(sp[(sp.group == "new765") & (sp.deep_split == "test")].scene)
    out = []
    for s in test:
        c = DetCache(DET / CHEAP / f"{s}.npz")
        for i in range(len(c.frames)):
            sc = c.scalars(i)
            img = {k: v for k, v in sc.items()
                   if k.startswith(("img_bright", "img_dark", "img_lap", "img_edge", "img_contrast", "motion"))}
            out.append((c.det(i), c.geo(i), img, sc.get("img_w", 1600.0) * sc.get("img_h", 900.0)))
            if len(out) == N_FRAMES:
                return out
    return out


def extractor(subset, cols):
    """The subset's groups only, in frame_features' order, as the model's input vector."""
    gs = set(subset.split("+"))

    def fn(it):
        det, geo, img, area = it
        keep = det["conf"] >= OP
        f = {}
        if "conf" in gs:
            f.update(F.confidence_features(det["conf"][keep]))
        if "unc" in gs:
            f.update(F.uncertainty_features(det, keep, OP))
        if "complex" in gs:
            f.update(F.complexity_features(det, keep, img, area))
        if "crit" in gs:
            f.update(F.criticality_features(det, keep, geo, G.PRIMARY))
        return np.nan_to_num(np.array([[f[c] for c in cols]], np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    return fn


class Rails(threading.Thread):
    def __init__(self, period=0.05):
        super().__init__(daemon=True)
        self.period, self.samples, self._halt = period, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append(power.read_power_mw())
            time.sleep(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        keys = self.samples[0].keys() if self.samples else []
        return {f"{k}_mw_mean": float(np.nanmean([s.get(k, np.nan) for s in self.samples])) for k in keys}


def timed(fn, items, per_call_frames=1):
    for i in range(WARMUP):
        fn(items[i % len(items)])
    rails = Rails()
    rails.start()
    ts = []
    for i in range(REPS):
        t = time.perf_counter()
        fn(items[i % len(items)])
        ts.append((time.perf_counter() - t) * 1e3 / per_call_frames)
    r = rails.stop()
    return {"lat_e2e_ms_median": float(np.median(ts)), "lat_e2e_ms_mean": float(np.mean(ts)),
            "lat_e2e_ms_p95": float(np.percentile(ts, 95)), **r}


def stage_profile(args):
    run = runmeta.new_run("r3_profile", vars(args))
    cells_, fcols = xl_cells()
    groups = group_columns(fcols)
    c = next(c for c in cells_ if c["geometry"] == "mono")
    d = c["d"]
    fit = (d.split == "train").to_numpy()
    v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    frames = profile_frames()
    items = {}
    for s in SUBSETS:
        cols = subset_columns(s, groups)
        ex = extractor(s, cols)
        X = np.vstack([ex(it) for it in frames])
        items[f"extract|{s}"] = (ex, frames, 1)
        for learner in LEARNERS:
            m, _ = fit_predict(d, cols, v, fit, learner)
            rows = [X[i:i + 1] for i in range(len(X))]
            items[f"infer|{learner}|{s}|per_input"] = (m.predict, rows, 1)
            if learner == "gate_gbm":
                items[f"infer|{learner}|{s}|batched"] = (m.predict, [X], len(X))
    rows = []
    idle = Rails()
    idle.start()
    time.sleep(10)
    idle_base = idle.stop()
    for rnd in range(ROUNDS):
        for name, (fn, its, per) in items.items():
            r = timed(fn, its, per)
            r.update(mode=name, round=rnd)
            rows.append(r)
        print(f"  round {rnd}: {len(items)} items", flush=True)
    (run / "profile_rows.json").write_text(json.dumps(rows, indent=1))
    (run / "idle_baseline.json").write_text(json.dumps(idle_base, indent=1))
    # the cheap and full engines in the same session, exactly as x7_costs.csv was measured (178)
    before = set(glob.glob(str(RAW / "*_profile")))
    cmd = [str(ROOT / "scripts" / "py"), str(ROOT / "scripts" / "01_profile_jetson.py"), "--backend", "trt",
           "--weights", str(MODELS / "yolov8s.pt"), "--images", IMAGES, "--reps", "250", "--warmup", "40",
           "--rounds", "3", "--modes", "ns_cheap_320", "ns_full_640"]
    subprocess.run(cmd, check=True)
    eng = Path(sorted(set(glob.glob(str(RAW / "*_profile"))) - before)[-1])
    t133 = _load("t133", "133_energy_module_budgets.py")
    summ = json.loads((eng / "profile_summary.json").read_text())
    out = []
    for name in items:
        kept = [r for r in rows if r["mode"] == name and r["round"] > 0]
        ms = float(np.median([r["lat_e2e_ms_median"] for r in kept]))
        mj = float(np.median([sum(r[f"{k}_mw_mean"] - idle_base[f"{k}_mw_mean"] for k in t133.MODULE
                                  if f"{k}_mw_mean" in r) * r["lat_e2e_ms_median"] / 1e3 for r in kept]))
        kind, *rest = name.split("|")
        out.append({"item": name, "kind": kind, "learner": rest[0] if kind == "infer" else "",
                    "subset": rest[-2] if kind == "infer" else rest[0], "call": rest[-1] if kind == "infer" else "",
                    "ms": ms, "mJ_module": mj, "run": run.name})
    for m in ("ns_cheap_320", "ns_full_640"):
        out.append({"item": f"engine|{m}", "kind": "engine", "learner": "", "subset": "", "call": m,
                    "ms": float(summ[m]["lat_e2e_ms_median"]), "mJ_module": t133.profile_energy(eng.name, m, t133.MODULE),
                    "run": eng.name})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(OUT / "feature_costs.csv", index=False)
    print(pd.DataFrame(out)[["item", "ms", "mJ_module"]].to_string(index=False), flush=True)


def subset_cost(costs, subset, learner, unit="ms", call=None):
    """C_S = the subset's extraction + its model's inference (gate_gbm batched by default, gate_ridge per input)."""
    call = call or ("batched" if learner == "gate_gbm" else "per_input")
    col = "ms" if unit == "ms" else "mJ_module"
    e = costs[(costs.kind == "extract") & (costs.subset == subset)][col].iloc[0]
    i = costs[(costs.kind == "infer") & (costs.subset == subset) & (costs.learner == learner) & (costs.call == call)][col].iloc[0]
    return float(e + i)


def engines(costs, unit="ms"):
    col = "ms" if unit == "ms" else "mJ_module"
    g = lambda m: float(costs[(costs.kind == "engine") & (costs.call == m)][col].iloc[0])
    return g("ns_cheap_320"), g("ns_full_640")


def frac_at(alpha, c_c, c_f, c_s):
    """93's measured-budget rule: b = C_c + alpha * C_f, every input charged C_c + C_S, an escalation C_f more."""
    b = c_c + alpha * c_f
    return (b - c_c - c_s) / c_f


def gain_at(score, v, frac):
    k = int(np.floor(max(frac, 0.0) * len(v) + 1e-9))
    if k <= 0:
        return 0.0, 0
    if score is None:
        return k / len(v) * v.sum(), k
    return float(xl.gains(score, v, np.array([k]))[0]), k


# ------------------------------------------------------------------------------------------------ select / test

def stage_select(args):
    runmeta.new_run("r3_select", vars(args))
    costs = pd.read_csv(OUT / "feature_costs.csv")
    cells_, fcols = xl_cells()
    groups = group_columns(fcols)
    c_c, c_f = engines(costs, "ms")
    rows, sel = [], {}
    for c in cells_:
        if c["geometry"] != "mono":
            continue
        d = c["d"]
        fit, val = (d.split == "train").to_numpy(), (d.split == "val").to_numpy()
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        for learner in LEARNERS:
            for s in SUBSETS:
                _, sc = fit_predict(d, subset_columns(s, groups), v, fit, learner)
                cs = subset_cost(costs, s, learner)
                f = frac_at(0.2, c_c, c_f, cs)
                feasible = f > 0
                g, k = gain_at(sc[val], v[val], f) if feasible else (np.nan, 0)
                rows.append({"learner": learner, "subset": s, "groups": len(s.split("+")), "C_S_ms": cs,
                             "escalated_frac": f, "feasible": feasible, "k_val": k, "val_gain": g,
                             "val_all_cheap_loss": float(d[c["cheap"]].to_numpy(float)[val].sum())})
            r = pd.DataFrame([x for x in rows if x["learner"] == learner and x["feasible"]])
            best = r.sort_values(["val_gain", "C_S_ms"], ascending=[False, True]).iloc[0]
            sel[learner] = {"subset": best.subset, "C_S_ms": float(best.C_S_ms), "val_gain": float(best.val_gain)}
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "subsets_val.csv", index=False)
    (OUT / "selected.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1), flush=True)


def boot(score, v, units, frac, frac_r, nboot, rng):
    uniq = sorted(set(units))
    idx = {u: np.flatnonzero(units == u) for u in uniq}
    out = np.full(nboot, np.nan)
    for b in range(nboot):
        t = np.concatenate([idx[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
        out[b] = gain_at(score[t], v[t], frac)[0] - gain_at(None, v[t], frac_r)[0]
    return out


def stage_test(args):
    runmeta.new_run("r3_test", vars(args))
    costs = pd.read_csv(OUT / "feature_costs.csv")
    sel = json.loads((OUT / "selected.json").read_text())
    cells_, fcols = xl_cells()
    groups = group_columns(fcols)
    rows = []
    for c in cells_:
        d = c["d"]
        fit, test = d.split.isin(["train", "val"]).to_numpy(), (d.split == "test").to_numpy()
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        units = d.unit.astype(str).to_numpy()[test]
        tot = float(d[c["cheap"]].to_numpy(float)[test].sum())
        for learner in LEARNERS:
            for s in SUBSETS:
                chosen = s == sel[learner]["subset"]
                _, sc = fit_predict(d, subset_columns(s, groups), v, fit, learner)
                for unit in ("ms", "mJ"):
                    c_c, c_f = engines(costs, unit)
                    cs = subset_cost(costs, s, learner, unit)
                    for a in BUDGETS:
                        f = frac_at(a, c_c, c_f, cs)
                        row = {"geometry": c["geometry"], "learner": learner, "subset": s, "chosen": chosen,
                               "unit": unit, "budget": a, "C_c": c_c, "C_f": c_f, "C_S": cs, "escalated_frac": f,
                               "feasible": f > 0, "all_cheap_loss": tot}
                        if f > 0:
                            g, k = gain_at(sc[test], v[test], f)
                            gr, kr = gain_at(None, v[test], a)
                            row.update(k=k, gain=g, k_random=kr, gain_random=gr, vs_random=g - gr,
                                       vs_random_share=(g - gr) / tot)
                            if chosen:
                                x = boot(sc[test], v[test], units, f, a, NBOOT, np.random.default_rng(0))
                                lo, hi = xl.ci(x)
                                row.update(vs_random_lo=lo, vs_random_hi=hi, vs_random_share_lo=lo / tot,
                                           vs_random_share_hi=hi / tot, resolved=bool(lo > 0 or hi < 0))
                        row["primary"] = bool(chosen and learner == "gate_gbm" and c["geometry"] == "mono" and unit == "ms"
                                              and a == 0.2)
                        row["descriptive"] = not chosen
                        rows.append(row)
        print(f"  {c['geometry']} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "test.csv", index=False)
    print(t[t.primary][["subset", "C_S", "escalated_frac", "gain", "gain_random", "vs_random_share",
                        "vs_random_share_lo", "vs_random_share_hi"]].to_string(index=False), flush=True)


def stage_frozen(args):
    """The chosen subset on the frozen tracks' braking cells (shipped R1 run's cells: monocular, J), fit on each track's
    non-test units, scored once on its test split, at the 20% latency budget with the track's frozen engine costs
    (results/final/cost_registry.json) and this session's C_S."""
    runmeta.new_run("r3_frozen", vars(args))
    costs = pd.read_csv(OUT / "feature_costs.csv")
    sel = json.loads((OUT / "selected.json").read_text())
    t92 = xl.t92()
    reg = json.loads((Path(RESULTS) / "final" / "cost_registry.json").read_text())["detectors"]
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    rows = []
    for track, gen in (("nuScenes", t92.nuscenes_cells), ("KITTI", t92.kitti_cells)):
        c = next(c for c in gen(splits) if c["geometry"] == "mono" and c["system"] == "brake")
        d = c["d"]
        groups = group_columns(c["fcols"])
        fit, test = (d.split != "test").to_numpy(), (d.split == "test").to_numpy()
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        units = d.unit.astype(str).to_numpy()[test]
        tot = float(d[c["cheap"]].to_numpy(float)[test].sum())
        c_c, c_f = float(reg[track]["cheap"]["ms"]), float(reg[track]["640"]["ms"])
        for learner in LEARNERS:
            s = sel[learner]["subset"]
            _, sc = fit_predict(d, subset_columns(s, groups), v, fit, learner)
            cs = subset_cost(costs, s, learner)
            f = frac_at(0.2, c_c, c_f, cs)
            row = {"track": track, "learner": learner, "subset": s, "C_c": c_c, "C_f": c_f, "C_S": cs,
                   "escalated_frac": f, "feasible": f > 0, "all_cheap_loss": tot}
            if f > 0:
                g, k = gain_at(sc[test], v[test], f)
                gr, kr = gain_at(None, v[test], 0.2)
                x = boot(sc[test], v[test], units, f, 0.2, NBOOT, np.random.default_rng(0))
                lo, hi = xl.ci(x)
                row.update(k=k, gain=g, gain_random=gr, vs_random=g - gr, vs_random_lo=lo, vs_random_hi=hi,
                           vs_random_share=(g - gr) / tot, vs_random_share_lo=lo / tot, vs_random_share_hi=hi / tot)
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "frozen_kitti.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["profile", "select", "test", "frozen"])
    args = ap.parse_args()
    {"profile": stage_profile, "select": stage_select, "test": stage_test, "frozen": stage_frozen}[args.stage](args)


if __name__ == "__main__":
    main()
