#!/usr/bin/env python
"""Task 25: what the submission path reads, and the gate that it scores exactly as the benchmark does.

  --stage values   results/final/benchmark_decision_values.csv.gz: every cell's inputs (identifiers, unit, split,
                   J_cheap, J_full, V) in the order 92 scores them, from 92's own cell builders; and
                   results/final/benchmark_bootstrap_plans.json: for each official table, the sequence of unit
                   resamples its script draws from `default_rng(0)` (cell, split, number of units), so a submission
                   shares the resamples of the table it is compared with
  --stage inputs   data/submission_inputs/: for every unit of the frozen split, exactly what an allocator may read
                   at test time -- CHEAP detections with their ego-frame monocular geometry, the previous CHEAP frame
                   (by `prev_frame`), calibration, ego speed -- and nothing else (no reference objects, no FULL
                   output, no decision value)
  --stage g1       gate G1: every deployable signal of benchmark_table.csv (random, uncertainty, criticality_cheap,
                   gate_ridge, gate_gbm) and of benchmark_table_routers.csv (R1 x 4, R2), written out as submission
                   files and scored through `evaluate_submission.py`'s library, must reproduce the official rows
                   exactly -- every compared field, every cell and quota; no tolerance. The measured-budget path is
                   held to benchmark_budget_two_level.csv the same way, for random and gate_gbm charged 93's
                   overheads
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta, submission as S                                        # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

FINAL = Path(RESULTS) / "final"
INPUTS = ROOT / "data" / "submission_inputs"
DEPLOYABLE = ("random", "uncertainty", "criticality_cheap", "gate_ridge", "gate_gbm")
ROUTERS = ("R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "R2_cnn_clf")
FIELDS_TABLE = ("k", "eta", "eta_lo", "eta_hi", "minus_random", "minus_random_lo", "minus_random_hi", "p_le_random",
                "gain", "prize", "reduction_frac", "tie_frac", "responsive_frac", "boot_dropped", "n_units", "n_frames")
FIELDS_ROUTERS = ("k", "eta", "eta_lo", "eta_hi", "minus_random", "minus_random_lo", "minus_random_hi", "p_le_random",
                  "tie_frac", "responsive_frac", "boot_dropped", "n_units", "n_frames")
BUDGET_FIELDS = ("feasible", "overhead", "escalated_frac", "eta", "eta_lo", "eta_hi", "minus_random", "minus_random_lo",
                 "minus_random_hi", "boot_dropped")
LOW = -1e300          # a missing signal value (the benchmark ranks it at -inf) as a finite score, same order and ties


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def cell_list(t92, splits):
    return [c for gen in (t92.nuscenes_cells, t92.kitti_cells, t92.nuplan_cells) for c in gen(splits)]


def ident(c):
    d = c["d"]
    if c["track"] == "nuPlan":
        return {"seq": "", "frame": "", "scenario": d.scenario.astype(str).to_numpy(), "iteration": d.iteration.astype(int).to_numpy()}
    return {"seq": d.seq.astype(str).to_numpy(), "frame": d.frame.astype(int).to_numpy(), "scenario": "", "iteration": ""}


# ------------------------------------------------------------------------------------------------ values, plans

def stage_values(run, t92, splits):
    parts, plans = [], {"benchmark_table": [], "routers_r1": [], "benchmark_budget": [],
                        "router_r2:nuScenes": [], "router_r2:KITTI": []}
    for c in cell_list(t92, splits):
        d = c["d"]
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        key = {k: c[k] for k in S.CELL}
        parts.append(pd.DataFrame({**key, **ident(c), "unit": d.unit.astype(str).to_numpy(), "split": d.split.to_numpy(),
                                   "J_cheap": d[c["cheap"]].to_numpy(float), "J_full": d[c["full"]].to_numpy(float),
                                   "V": v}))
        cid = S.cell_id(tuple(key.values()))
        n_test = int(len(np.unique(d.unit.to_numpy()[(d.split == "test").to_numpy()])))
        n_all = int(len(np.unique(d.unit.to_numpy())))
        plan_cell = c["track"] == "nuScenes" and c["system"].startswith("plan_")
        # 92: per split (test, all) one evaluate, then S(M) on the same units for the nuScenes planner cells
        for split, n in (("test", n_test), ("all", n_all)):
            plans["benchmark_table"].append({"cell": cid, "split": split, "n_units": n, "scored": split == "test"})
            if plan_cell:
                plans["benchmark_table"].append({"cell": cid, "split": f"{split} S(M)", "n_units": n, "scored": False})
        # 103 and 93: one draw sequence per cell, on test
        plans["routers_r1"].append({"cell": cid, "split": "test", "n_units": n_test, "scored": True})
        plans["benchmark_budget"].append({"cell": cid, "split": "test", "n_units": n_test, "scored": True})
        # 107: one rng per dataset, one evaluate per head on that head's test rows
        if c["track"] in ("nuScenes", "KITTI"):
            plans[f"router_r2:{c['track']}"].append({"cell": cid, "split": "test", "n_units": n_test, "scored": True})
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(S.VALUES, index=False, compression={"method": "gzip", "mtime": 0})
    S.PLANS.write_text(json.dumps(plans, indent=1))
    for f in (S.VALUES, S.PLANS):
        (run / f.name).write_bytes(f.read_bytes())
    print(f"  wrote {S.VALUES} ({len(df)} rows, {df.groupby(list(S.CELL)).ngroups} cells) and {S.PLANS}", flush=True)


# ------------------------------------------------------------------------------------------------ inputs

def stage_inputs(run, t92, splits):
    from rap.cache import DetCache, DET_ARRAYS, GEO_ARRAYS
    from rap import decision, egospeed
    INPUTS.mkdir(parents=True, exist_ok=True)
    split_of = {t: {u: k for k in ("train", "val", "test") for u in splits[t.lower()][k]} for t in ("KITTI", "nuScenes", "nuPlan")}
    for t, m in split_of.items():
        assert m and set(m.values()) == {"train", "val", "test"}, (t, len(m))
    token = pd.read_csv(Path(CACHE) / "nusc_token_map.csv").astype({"seq": str, "frame": int})
    manifest = {}
    for track, (det_dir, mode) in {"KITTI": ("det", "cheap_320"), "nuScenes": ("nusc_det_tv", "ns_cheap_320")}.items():
        frames_rows, det_rows = [], []
        for f in sorted((Path(CACHE) / det_dir / mode).glob("*.npz")):
            seq = f.stem
            if seq not in split_of[track]:
                continue
            c = DetCache(f)                                   # ego-frame geometry, re-lifted on read
            if track == "KITTI":
                speeds = decision.KittiAdapter.speeds(seq)
            prev = -1
            for i, fr in enumerate(c.frames):
                fr = int(fr)
                d, g, sc = c.det(i), c.geo(i), c.scalars(i)
                rec = {"seq": seq, "frame": fr, "unit": seq, "split": split_of[track][seq], "prev_frame": prev,
                       "n_detections": int(len(d["conf"]))}
                if track == "KITTI":
                    rec["ego_speed_mps"] = float(speeds[fr]) if fr < len(speeds) else float(speeds[-1])
                    rec["image"] = f"training/image_02/{seq}/{fr:06d}.png"
                rec.update({f"frame_{k}": v for k, v in sc.items()})
                frames_rows.append(rec)
                for j in range(len(d["conf"])):
                    det_rows.append({"seq": seq, "frame": fr, "det": j,
                                     "x1": float(d["xyxy"][j, 0]), "y1": float(d["xyxy"][j, 1]),
                                     "x2": float(d["xyxy"][j, 2]), "y2": float(d["xyxy"][j, 3]),
                                     "coarse": str(d["coarse"][j]),
                                     **{k: float(d[k][j]) for k in DET_ARRAYS if k != "xyxy"},
                                     **{f"geo_{k}": float(g[k][j]) for k in GEO_ARRAYS}})
                prev = fr
        fr_df = pd.DataFrame(frames_rows)
        if track == "nuScenes":
            v = egospeed.table()[["seq", "frame", "v_ego_causal"]].rename(columns={"v_ego_causal": "ego_speed_mps"})
            fr_df = fr_df.merge(v, on=["seq", "frame"], how="left", validate="one_to_one")
            fr_df = fr_df.merge(token[["seq", "frame", "sample_token"]], on=["seq", "frame"], how="left",
                                validate="one_to_one")
        for name, df in ((f"{track.lower()}_frames.csv.gz", fr_df), (f"{track.lower()}_detections.csv.gz", pd.DataFrame(det_rows))):
            df.to_csv(INPUTS / name, index=False, compression={"method": "gzip", "mtime": 0})
            manifest[name] = {"rows": int(len(df)), "columns": list(df.columns)}
    # calibration: the intrinsics and camera -> ego transforms every re-lift uses
    cam = json.loads((Path(CACHE) / "cam_to_ego.json").read_text())
    (INPUTS / "calibration.json").write_text(json.dumps(cam, indent=1))
    manifest["calibration.json"] = {"units": {k: len(v) for k, v in cam.items()}}
    # nuPlan: the legal cheap-side features (gate features) and the CHEAP track list, per state
    sig = pd.read_csv(FINAL / "benchmark_nuplan_signals.csv")
    roles = json.loads((FINAL / "benchmark_nuplan_signals.json").read_text())
    keep = ["scenario", "log", "iteration"] + list(roles["gate_features"])
    st = sig[keep].rename(columns={"log": "unit"})
    st["split"] = st.unit.map(split_of["nuPlan"])
    trk = sorted((ROOT / "results" / "raw").glob("*_nuplan_track_lists/nuplan_track_features.npz"))[-1]
    z = np.load(trk, allow_pickle=False)
    tl = pd.DataFrame(z["X"], columns=[f"track{i // 10:02d}_{n}" for i in range(z["X"].shape[1])
                                       for n in [("presence", "x", "y", "length", "width", "vehicle", "pedestrian",
                                                  "bicycle", "static", "area")[i % 10]]][: z["X"].shape[1]])
    tl.insert(0, "scenario", z["scenario"].astype(str)); tl.insert(1, "iteration", z["iteration"].astype(int))
    st = st.merge(tl, on=["scenario", "iteration"], how="left", validate="one_to_one")
    st.to_csv(INPUTS / "nuplan_states.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    manifest["nuplan_states.csv.gz"] = {"rows": int(len(st)), "columns": list(st.columns)}
    (INPUTS / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (run / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print("  wrote", INPUTS, {k: v.get("rows") for k, v in manifest.items()}, flush=True)


# ------------------------------------------------------------------------------------------------ G1

def _as_submission(cells_scores):
    parts = []
    for c, s in cells_scores:
        idn = ident(c)
        m = (c["d"].split == "test").to_numpy()
        parts.append(pd.DataFrame({**{k: c[k] for k in S.CELL},
                                   **{k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in idn.items()},
                                   "score": np.where(np.isfinite(s), s, LOW)}))
    df = pd.concat(parts, ignore_index=True)
    for c in ("seq", "frame", "scenario", "iteration"):
        df[c] = df[c].astype(str)
    return df


def _compare(ours, official, fields, label):
    key = list(S.CELL) + ["quota"]
    o = official.copy()
    for k in S.CELL:
        o[k] = o[k].astype(str)
    j = ours.rename(columns={"ndg": "eta", "ndg_lo": "eta_lo", "ndg_hi": "eta_hi"}).merge(
        o, on=key, how="left", suffixes=("", "_official"), validate="one_to_one")
    rows = []
    for f in fields:
        a, b = j[f].to_numpy(float), j[f"{f}_official"].to_numpy(float)
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        rows.append({"signal": label, "field": f, "rows": int(len(j)), "differing": int((~same).sum()),
                     "max_abs_diff": float(np.nanmax(np.abs(a - b))) if (~same).any() else 0.0})
    return rows


def g1_budget(run, sigs):
    """The measured-budget path against 93's two-level table: random and gate_gbm, each charged the overhead 93
    charged it (`signal_overhead` on the shipped overheads), as a cost profile."""
    t93 = _load("t93", "93_budget_allocation.py")
    ov = json.loads((FINAL / "benchmark_budget_overheads.json").read_text())["overheads"]
    two = pd.read_csv(FINAL / "benchmark_budget_two_level.csv", keep_default_na=False, na_values=[""],
                      float_precision="round_trip")
    rows = []
    for s in ("random", "gate_gbm"):
        prof = {u: {t: t93.signal_overhead(ov, t, s)[i] for t in ("KITTI", "nuScenes", "nuPlan")}
                for i, u in enumerate(("ms", "mJ"))}
        f = run / f"g1__{s}.csv"
        _, sc = S.score(S.load(f), plan="benchmark_table", profile=prof)
        ours = sc[sc.track_kind == "measured budget"].rename(columns={"ndg": "eta", "ndg_lo": "eta_lo",
                                                                      "ndg_hi": "eta_hi"})
        for fld in BUDGET_FIELDS:                     # an allocator infeasible everywhere never gets these columns
            if fld not in ours.columns:
                ours[fld] = np.nan
        off = two[two.signal == s].copy()
        for k in S.CELL:
            off[k] = off[k].astype(str)
        j = ours.merge(off, on=list(S.CELL) + ["unit", "budget_level"], how="left", suffixes=("", "_official"),
                       validate="one_to_one")
        assert len(j) == len(off), (s, len(j), len(off))
        for fld in BUDGET_FIELDS:
            a, b = j[fld].to_numpy(float), j[f"{fld}_official"].to_numpy(float)
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            rows.append({"signal": f"{s} (measured budget)", "field": fld, "rows": int(len(j)),
                         "differing": int((~same).sum()),
                         "max_abs_diff": float(np.nanmax(np.abs(a - b))) if (~same).any() else 0.0})
        print(f"  G1 {s} (measured budget): {len(j)} rows scored", flush=True)
    return rows


def stage_g1(run, t92, splits):
    tab = pd.read_csv(FINAL / "benchmark_table.csv", keep_default_na=False, na_values=[""], float_precision="round_trip")
    rt = pd.read_csv(FINAL / "benchmark_table_routers.csv", keep_default_na=False, na_values=[""],
                     float_precision="round_trip")
    cells = cell_list(t92, splits)
    sigs = {s: [] for s in DEPLOYABLE}
    for c in cells:
        d = c["d"]
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        gates = t92.gate_predictions(d, c["fcols"], v_all)
        for s in DEPLOYABLE:
            if s == "random":
                sigs[s].append((c, np.zeros(len(d))[(d.split == "test").to_numpy()]))
            elif s.startswith("gate_"):
                sigs[s].append((c, gates[s][(d.split == "test").to_numpy()]))
            elif s in c["cols"]:
                sigs[s].append((c, pd.to_numeric(d[c["cols"][s]], errors="coerce").fillna(-np.inf).to_numpy()[
                    (d.split == "test").to_numpy()]))
    rows = []
    for s, cs in sigs.items():
        sub = _as_submission(cs)
        f = run / f"g1__{s}.csv"
        sub.to_csv(f, index=False)
        _, sc = S.score(S.load(f), plan="benchmark_table")
        off = tab[(tab.split == "test") & (tab.signal == s) & tab.quota.notna()]
        rows += _compare(sc[sc.track_kind == "selection budget"], off, FIELDS_TABLE, s)
        print(f"  G1 {s}: {len(sc)} rows scored", flush=True)
    # routers: R1 from the shipped R1 run (its own draw sequence), R2 per dataset
    r1 = rap_runs.latest("routers_r1")
    r2 = rap_runs.latest("router_r2")
    by = {(c["track"], c["geometry"].replace("/", ""), c["system"], c["target"]): c for c in cells}
    for name in ROUTERS:
        runs_ = [("routers_r1", r1)] if name.startswith("R1_") else [("router_r2:nuScenes", r2), ("router_r2:KITTI", r2)]
        for plan, rdir in runs_:
            cs = []
            for fz in sorted(rdir.glob("scores__*.npz")):
                _, track, geometry, system, target = fz.stem.split("__")
                if plan.startswith("router_r2") and track != plan.split(":")[1]:
                    continue
                z = np.load(fz, allow_pickle=False)
                if name not in z.files:
                    continue
                c = by[(track, geometry, system, target)]
                d = c["d"]
                test = (d.split == "test").to_numpy()
                idc = ("scenario", "iteration") if track == "nuPlan" else ("seq", "frame")
                zz = pd.DataFrame({idc[0]: z[idc[0]].astype(str), idc[1]: z[idc[1]].astype(int), "s": z[name].astype(float)})
                k = d.loc[test, list(idc)].astype({idc[0]: str, idc[1]: int}).reset_index(drop=True)
                jj = k.merge(zz, on=list(idc), how="left", validate="one_to_one")
                assert jj.s.notna().all(), (name, fz.name)
                cs.append((c, jj.s.to_numpy(float)))
            if not cs:
                continue
            sub = _as_submission(cs)
            f = run / f"g1__{name}__{plan.replace(':', '_')}.csv"
            sub.to_csv(f, index=False)
            _, sc = S.score(S.load(f), plan=plan)
            off = rt[(rt.signal == name) & rt.quota.notna()]
            ours = sc[sc.track_kind == "selection budget"]
            off = off.merge(ours[list(S.CELL)].drop_duplicates().astype(str), on=list(S.CELL))
            rows += _compare(ours, off, FIELDS_ROUTERS, f"{name} ({plan})")
            print(f"  G1 {name} ({plan}): {len(ours)} rows scored", flush=True)
    rows += g1_budget(run, sigs)
    g = pd.DataFrame(rows)
    g.to_csv(run / "g1.csv", index=False)
    g.to_csv(FINAL / "submission_path_g1.csv", index=False)
    bad = g[g.differing > 0]
    print(f"  G1: {len(g)} signal x field comparisons, {int(g.rows.sum() // max(len(set(g.field)), 1))} rows; "
          f"{len(bad)} with a difference", flush=True)
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit("G1 failed: the submission path does not reproduce the official rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["values", "inputs", "g1"])
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"submission_{args.stage}", vars(args))
    t92 = _load("t92", "92_benchmark_table.py")
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    {"values": stage_values, "inputs": stage_inputs, "g1": stage_g1}[args.stage](run, t92, splits)
    print("wrote", run)


if __name__ == "__main__":
    main()
