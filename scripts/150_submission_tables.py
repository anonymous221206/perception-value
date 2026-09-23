#!/usr/bin/env python
"""Tasks 25 and 28: what the submission path reads, and the gate that it scores exactly as the paper's tables do.

  --stage values   results/final/benchmark_decision_values.csv.gz: every cell's inputs (identifiers, unit, split,
                   J_cheap, J_full, V, and where the labels come from) in the order the official scripts score them:
                   KITTI and nuScenes from 92's own cell builders, nuPlan from the real-perception track exactly as
                   120 builds it (Task 5 labels; Task 28 -- the transported track is no longer a submission cell); and
                   results/final/benchmark_bootstrap_plans.json: for each official table, the sequence of unit
                   resamples its script draws from `default_rng(0)` (cell, split, number of units), so a submission
                   shares the resamples of the table it is compared with (nuPlan: 120's stream, `nuplan_real`
                   and `nuplan_real_budget`)
  --stage inputs   data/submission_inputs/: for every unit of the frozen split, exactly what an allocator may read
                   at test time -- CHEAP detections with their ego-frame monocular geometry, the previous CHEAP frame
                   (by `prev_frame`), calibration, ego speed; nuPlan: the real CHEAP branch's 18 gate features and
                   25-track list (119) -- and nothing else (no reference objects, no FULL output, no decision value)
  --stage g1       gate G1' (Task 28): random and every shipped deployable signal, written out as submission files
                   and scored through `evaluate_submission.py`'s library, must reproduce the tables the paper uses
                   exactly -- every compared field, every cell and quota, no tolerance: benchmark_table.csv and
                   benchmark_table_routers.csv (core rows), benchmark_table_nuplan_real.csv, the latency budgets
                   (benchmark_budget_two_level / _routers core ms rows, benchmark_budget_nuplan_real ms rows,
                   benchmark_budget_multifidelity) and the energy budgets (energy_module_budget_two_level /
                   _multifidelity / _nuplan_real, module convention), each charged from the cost registry (156).
                   Before any number is compared, the scored rows and the canonical rows must agree on track, label
                   run, cost convention, detector costs and registry version; a mismatch stops the gate
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


def core_cells(t92, splits):
    return [c for gen in (t92.nuscenes_cells, t92.kitti_cells) for c in gen(splits)]


def cell_list(t92, t120, splits):
    """The submission cells in the order the official scripts visit them: 92's KITTI and nuScenes cells, then 120's
    real-perception nuPlan cells (the transported nuPlan track of 92 is not a submission cell)."""
    return core_cells(t92, splits) + list(t120.cells(splits))


def labels_of(t92, c):
    """(labels, source_run): which label set a cell's decision values are, and the run they are read from."""
    if c["track"] == "nuPlan":
        return "Task 5 real perception, primary", Path(c["run"]).name
    if c["track"] == "nuScenes":
        return "nuScenes closed-loop decision tables", t92.NUSC_JOINED[c["geometry"]].parent.name
    if c["system"] == "traj":
        return "KITTI Planner B decision tables", t92.PLANB.name
    return "KITTI core decision matrix", t92.CORE.name


def ident(c):
    d = c["d"]
    if c["track"] == "nuPlan":
        return {"seq": "", "frame": "", "scenario": d.scenario.astype(str).to_numpy(), "iteration": d.iteration.astype(int).to_numpy()}
    return {"seq": d.seq.astype(str).to_numpy(), "frame": d.frame.astype(int).to_numpy(), "scenario": "", "iteration": ""}


# ------------------------------------------------------------------------------------------------ values, plans

def stage_values(run, t92, t120, splits):
    parts, plans = [], {"benchmark_table": [], "routers_r1": [], "benchmark_budget": [],
                        "router_r2:nuScenes": [], "router_r2:KITTI": [], "nuplan_real": [], "nuplan_real_budget": []}
    for c in cell_list(t92, t120, splits):
        d = c["d"]
        v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        key = {k: c[k] for k in S.CELL}
        labels, source = labels_of(t92, c)
        parts.append(pd.DataFrame({**key, **ident(c), "unit": d.unit.astype(str).to_numpy(), "split": d.split.to_numpy(),
                                   "J_cheap": d[c["cheap"]].to_numpy(float), "J_full": d[c["full"]].to_numpy(float),
                                   "V": v, "labels": labels, "source_run": source}))
        cid = S.cell_id(tuple(key.values()))
        n_test = int(len(np.unique(d.unit.to_numpy()[(d.split == "test").to_numpy()])))
        n_all = int(len(np.unique(d.unit.to_numpy())))
        if c["track"] == "nuPlan":
            # 120: one rng for the table, per cell one evaluate on test then one on all; one rng for the budgets,
            # per cell the test draws
            plans["nuplan_real"] += [{"cell": cid, "split": "test", "n_units": n_test, "scored": True},
                                     {"cell": cid, "split": "all", "n_units": n_all, "scored": False}]
            plans["nuplan_real_budget"].append({"cell": cid, "split": "test", "n_units": n_test, "scored": True})
            continue
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
        plans[f"router_r2:{c['track']}"].append({"cell": cid, "split": "test", "n_units": n_test, "scored": True})
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(S.VALUES, index=False, compression={"method": "gzip", "mtime": 0})
    S.PLANS.write_text(json.dumps(plans, indent=1))
    for f in (S.VALUES, S.PLANS):
        (run / f.name).write_bytes(f.read_bytes())
    t = df[df.split == "test"]
    aff = t.assign(a=t.V.abs() > S.scoring.EPS).groupby(list(S.CELL), sort=False).a.sum()
    print(f"  wrote {S.VALUES} ({len(df)} rows, {df.groupby(list(S.CELL)).ngroups} cells) and {S.PLANS}\n"
          f"  affected test inputs per nuPlan cell: " +
          ", ".join(f"{k[2]} {k[3]} {int(n)}" for k, n in aff.items() if k[0] == "nuPlan"), flush=True)


# ------------------------------------------------------------------------------------------------ inputs

def stage_inputs(run, t92, t120, splits):
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
    # nuPlan: the real CHEAP branch's legal features (120's 18 gate features) and its 25-track list (119), per state
    fr = sorted((ROOT / "results" / "raw").glob("*_nuplan_real_features"))[-1]
    sig = pd.read_csv(fr / "nuplan_real_signals.csv")
    roles = json.loads((fr / "roles.json").read_text())
    assert list(roles["gate_features"]) == list(t120.GATE), "the features run's gate features are not 120's"
    keep = ["scenario", "log", "iteration"] + list(t120.GATE)
    st = sig[keep].rename(columns={"log": "unit"})
    st["split"] = st.unit.map(split_of["nuPlan"])
    z = np.load(fr / "nuplan_real_track_features.npz", allow_pickle=False)
    names = ("presence", "x", "y", "length", "width", "vehicle", "pedestrian", "bicycle", "static", "area")
    assert z["X"].shape[1] == 10 * 25, z["X"].shape
    tl = pd.DataFrame(z["X"], columns=[f"track{i // 10:02d}_{names[i % 10]}" for i in range(z["X"].shape[1])])
    tl.insert(0, "scenario", z["scenario"].astype(str)); tl.insert(1, "iteration", z["iteration"].astype(int))
    st = st.merge(tl, on=["scenario", "iteration"], how="left", validate="one_to_one")
    assert st.split.notna().all() and st.track00_presence.notna().all()
    st.to_csv(INPUTS / "nuplan_states.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    manifest["nuplan_states.csv.gz"] = {"rows": int(len(st)), "columns": list(st.columns),
                                        "source_run": fr.name, "branch": "real CHEAP perception (Task 5, 119)"}
    (INPUTS / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    (run / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print("  wrote", INPUTS, {k: v.get("rows") for k, v in manifest.items()}, flush=True)


# ------------------------------------------------------------------------------------------------ G1'

R1 = ROUTERS[:4]
NR_SIGNALS = ("random", "criticality_cheap", "gate_ridge", "gate_gbm", *R1)
FIELDS_NR = FIELDS_TABLE + ("n_affected", "ndg_defined", "undefined_reason")
BUDGET_FIELDS_NR = BUDGET_FIELDS + ("escalations_lost_to_overhead", "gain", "prize", "n_affected", "ndg_defined",
                                    "undefined_reason")
MF_FIELDS = ("feasible", "overhead", "gain", "share_640")
EGO_FIELDS = ("eta", "gain", "prize")
MF_CELLS = {"brake": ("KITTI", "mono", "brake", "J"), "traj": ("KITTI", "mono", "traj", "JB")}
STRING_FIELDS = ("undefined_reason",)
# A shipped signal whose test scores are all equal is, to the schema, the random allocator (scored by its closed form,
# no tie statistics); 120 scored such a signal through the tie expectation instead, so the official tables give the
# same constant ranking tie_frac NaN (random) or 1.0 (the signal) and responsive_frac apart in the last bits. Those two
# diagnostics cannot be reproduced from the scores alone: counted as structural, never as matched.
ALL_TIED_FIELDS = ("tie_frac", "responsive_frac")
ALL_TIED_REASON = "all-tied official signal: the schema scores it as the random allocator (tie statistics differ)"


class IdentityError(SystemExit):
    """The scored rows and the canonical rows do not describe the same thing: never a numeric difference."""


def read_final(name):
    return pd.read_csv(FINAL / name, keep_default_na=False, na_values=[""], float_precision="round_trip",
                       low_memory=False)


def _col(d, name):
    return pd.to_numeric(d[name], errors="coerce").fillna(-np.inf).to_numpy(float)


def official_signals(t92, t133, cells):
    """Every shipped deployable signal on every cell's test inputs, from the runs the canonical tables were built
    from: 92's gate refits (checked bit for bit against the saved gate scores 133 charged), 103's and 107's router
    runs, 120's nuPlan run, and the cells' own uncertainty, criticality and ego-speed columns."""
    prov, levels, gate_run = t133.load_gate_scores()
    r1_run, r2_run = rap_runs.latest("routers_r1"), rap_runs.latest("router_r2")
    nr_run = rap_runs.latest("nuplan_real_allocation")
    sig, gates_equal = {}, True
    for c in cells:
        d = c["d"]
        key = tuple(c[k] for k in S.CELL)
        m = (d.split == "test").to_numpy()
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        out = {"random": np.zeros(int(m.sum()))}
        if c["track"] == "nuPlan":
            z = np.load(nr_run / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
            assert np.array_equal(z["V"], v_all) and (z["scenario"].astype(str) == d.scenario.astype(str).to_numpy()).all()
            out.update({s: np.asarray(z[s], float)[m] for s in ("gate_ridge", "gate_gbm", *R1)})
            out["criticality_cheap"] = _col(d, "nr_crit_cheap_sum")[m]
            out["ego_speed"] = _col(d, "nr_ego_speed")[m]
        else:
            g = t92.gate_predictions(d, c["fcols"], v_all)
            saved = prov[key](d)
            eq = all(np.array_equal(g[s], saved[s]) for s in ("gate_ridge", "gate_gbm"))
            gates_equal &= eq
            print(f"  gates {S.cell_id(key)}: refit {'==' if eq else '!='} saved scores ({gate_run})", flush=True)
            out.update({s: g[s][m] for s in ("gate_ridge", "gate_gbm")})
            out.update({f"{s} (saved)": saved[s][m] for s in ("gate_ridge", "gate_gbm")})
            for s in ("uncertainty", "criticality_cheap"):
                if s in c["cols"]:
                    out[s] = _col(d, c["cols"][s])[m]
            out["ego_speed"] = _col(d, "v_ego_causal")[m]
            idc = ("seq", "frame")
            k = d.loc[m, list(idc)].astype({"seq": str, "frame": int}).reset_index(drop=True)
            geom = c["geometry"].replace("/", "")
            for rdir, names in ((r1_run, R1), (r2_run, ("R2_cnn_clf",))):
                fz = rdir / f"scores__{c['track']}__{geom}__{c['system']}__{c['target']}.npz"
                if not fz.exists():
                    continue
                z = np.load(fz, allow_pickle=False)
                zz = pd.DataFrame({"seq": z["seq"].astype(str), "frame": z["frame"].astype(int)})
                for n in names:
                    if n in z.files:
                        zz[n] = np.asarray(z[n], float)
                jj = k.merge(zz, on=list(idc), how="left", validate="one_to_one")
                for n in names:
                    if n in jj.columns:
                        assert jj[n].notna().all(), (n, fz.name)
                        out[n] = jj[n].to_numpy(float)
        out["gate_gbm_batched"] = out["gate_gbm"]                # the same ranking, charged batched inference
        if "gate_gbm (saved)" in out:
            out["gate_gbm_batched (saved)"] = out["gate_gbm (saved)"]
        sig[key] = out
    # the multi-fidelity table's "gate_gbm (640 only)" ranks by the 640-level GBM of 93's level predictions
    d, _, m, _ = t133.t93.multi_fidelity_frames()
    for system, key in MF_CELLS.items():
        c = next(c for c in cells if tuple(c[k] for k in S.CELL) == key)
        lp = pd.DataFrame({"seq": d.seq.astype(str).to_numpy()[m], "frame": d.frame.astype(int).to_numpy()[m],
                           "s": levels[system]["gate_gbm"][m, 2]})
        k = c["d"].loc[(c["d"].split == "test").to_numpy(), ["seq", "frame"]].astype({"seq": str, "frame": int})
        jj = k.reset_index(drop=True).merge(lp, on=["seq", "frame"], how="left", validate="one_to_one")
        assert jj.s.notna().all()
        sig[key]["gate_gbm (640-level)"] = jj.s.to_numpy(float)
    return sig, {"gate_scores_run": gate_run, "gate_refit_equals_saved": bool(gates_equal),
                 "routers_r1_run": r1_run.name, "router_r2_run": r2_run.name, "nuplan_real_run": nr_run.name}


def write_submission(run, name, cells, sig, signal, keys=None):
    """One official signal as a submission file (the benchmark's -inf as LOW), over the cells that have it."""
    parts = []
    for c in cells:
        key = tuple(c[k] for k in S.CELL)
        if (keys is not None and key not in keys) or signal not in sig[key]:
            continue
        idn = ident(c)
        m = (c["d"].split == "test").to_numpy()
        s = sig[key][signal]
        parts.append(pd.DataFrame({**dict(zip(S.CELL, key)),
                                   **{k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in idn.items()},
                                   "score": np.where(np.isfinite(s), s, LOW)}))
    if not parts:
        return None
    df = pd.concat(parts, ignore_index=True)
    for c in ("seq", "frame", "scenario", "iteration"):
        df[c] = df[c].astype(str)
    f = run / f"g1__{name}.csv"
    df.to_csv(f, index=False)
    return f


def profile_for(signal, overheads, units=("ms", "mJ")):
    """The cost profile the official tables charged `signal`: the registry's overheads of that measurement."""
    reg = S.registry()
    tr = reg["allocators"][overheads]["tracks"]
    base = signal.split(" (")[0]
    prof = {"route": f"cost registry {reg['version']}, overheads '{overheads}'",
            "rails": reg["energy_convention"]["rails"], "provenance": {}}
    for u in ("ms", "mJ"):
        prof[u] = {t: tr[t][base][u] for t in tr if base in tr[t]} if u in units else None
    prof["provenance"] = {t: tr[t][base]["provenance"] for t in tr if base in tr[t]}
    return prof


def assert_identity(table, ours, off, checks):
    """Before a number is compared: same registry version, track, label run, cost convention and detector costs."""
    reg = S.registry()
    bad = []
    vers = set(ours.registry_version.astype(str))
    if vers != {EXPECTED_VERSION} or reg["version"] != EXPECTED_VERSION:
        bad.append(f"registry version: scored {sorted(vers)}, registry {reg['version']}, gate started on "
                   f"{EXPECTED_VERSION}")
    allowed = set(reg["tables"][table]["tracks"])
    for name, df in (("scored", ours), ("canonical", off)):
        if "track" in df.columns and not set(df.track.astype(str)) <= allowed:
            bad.append(f"track: {name} rows on {sorted(set(df.track.astype(str)) - allowed)}, but the registry "
                       f"reports {table} for {sorted(allowed)} only")
    for ours_col, off_col in checks:
        a, b = ours[ours_col], off[off_col] if off_col in off.columns else None
        if b is None:
            bad.append(f"{off_col}: not a column of {table}")
            continue
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            same = (a.to_numpy(float) == b.to_numpy(float))
        else:
            same = a.astype(str).to_numpy() == b.astype(str).to_numpy()
        if not same.all():
            i = int(np.flatnonzero(~same)[0])
            bad.append(f"{ours_col} vs {table}.{off_col}: {int((~same).sum())} rows differ, e.g. "
                       f"{a.iloc[i]!r} against {b.iloc[i]!r}")
    if bad:
        raise IdentityError(f"G1' identity check failed for {table}:\n  " + "\n  ".join(bad))


def _field_rows(table, label, j, fields):
    rows = []
    tied = (j.all_tied.astype(bool).to_numpy() if "all_tied" in j.columns and not label.startswith("random")
            else np.zeros(len(j), bool))
    for f in fields:
        if f in STRING_FIELDS:
            a, b = j[f].fillna("").astype(str).to_numpy(), j[f"{f}_official"].fillna("").astype(str).to_numpy()
            same, diff = a == b, 0.0
        else:
            a, b = j[f].astype(float).to_numpy(), j[f"{f}_official"].astype(float).to_numpy()
            same = (a == b) | (np.isnan(a) & np.isnan(b))
        structural = tied if f in ALL_TIED_FIELDS else np.zeros(len(j), bool)
        if f not in STRING_FIELDS:
            bad = ~same & ~structural
            diff = float(np.nanmax(np.abs(a - b)[bad])) if bad.any() else 0.0
        rows.append({"table": table, "signal": label, "field": f, "rows": int(len(j)),
                     "differing": int((~same & ~structural).sum()), "max_abs_diff": diff,
                     "structural": int(structural.sum()), "structural_reason": ALL_TIED_REASON if structural.any() else ""})
    return rows


def _join(ours, off, keys, table, label):
    o = off.copy()
    for k in keys:
        if k in o.columns and k in S.CELL:
            o[k] = o[k].astype(str)
    j = ours.merge(o, on=keys, how="left", suffixes=("", "_official"), validate="one_to_one", indicator=True)
    if (j._merge != "both").any() or len(j) != len(off):
        raise IdentityError(f"G1' {table} {label}: {int((j._merge != 'both').sum())} scored rows have no canonical "
                            f"row and {len(off) - int((j._merge == 'both').sum())} canonical rows no scored row")
    return j


def g1_selection(run, cells, sig, rows):
    tab = read_final("benchmark_table.csv")
    rt = read_final("benchmark_table_routers.csv")
    nr = read_final("benchmark_table_nuplan_real.csv")
    core = {tuple(c[k] for k in S.CELL) for c in cells if c["track"] != "nuPlan"}
    nup = {tuple(c[k] for k in S.CELL) for c in cells if c["track"] == "nuPlan"}
    key = list(S.CELL) + ["quota"]
    ren = {"ndg": "eta", "ndg_lo": "eta_lo", "ndg_hi": "eta_hi"}
    for s in DEPLOYABLE + ("ego_speed",) + R1:
        f = write_submission(run, f"sel__{s}", cells, sig, s, keys=(core | nup) if not s.startswith("R1_") else nup)
        _, sc = S.score(S.load(f))                                   # every track on its own default plan
        sc = sc.rename(columns=ren)
        oc, on = sc[sc.track != "nuPlan"], sc[sc.track == "nuPlan"]
        if s == "ego_speed":
            st = read_final("statistics_hardening.csv")
            st = st[(st.section == "paired_gain") & (st.signal == "trivial_ego_speed")].rename(
                columns={"ndg": "eta", "oracle_prize": "prize"})
            for part, table in ((oc, "statistics_hardening.csv (core)"), (on, "statistics_hardening.csv (nuPlan)")):
                off = st[st.track.isin(set(part.track))]
                assert_identity("benchmark_table.csv" if "core" in table else "benchmark_table_nuplan_real.csv",
                                part, off, [])
                j = _join(part, off, key, table, s)
                assert (j.n_frames == j.n_test_frames).all()
                rows += _field_rows(table, "ego_speed (point values)", j, EGO_FIELDS)
            print(f"  G1' ego_speed: {len(sc)} rows", flush=True)
            continue
        if len(oc):
            off = tab[(tab.split == "test") & (tab.signal == s) & tab.quota.notna() & tab.track.isin(["KITTI", "nuScenes"])]
            assert_identity("benchmark_table.csv", oc, off, [])
            rows += _field_rows("benchmark_table.csv", s, _join(oc, off, key, "benchmark_table.csv", s), FIELDS_TABLE)
        if len(on):
            off = nr[(nr.split == "test") & (nr.signal == s) & nr.quota.notna()]
            j = _join(on, off, key, "benchmark_table_nuplan_real.csv", s)
            assert_identity("benchmark_table_nuplan_real.csv", j, j, [("labels", "labels_official"),
                                                                       ("source_run", "features_run")])
            rows += _field_rows("benchmark_table_nuplan_real.csv", s, j, FIELDS_NR)
        print(f"  G1' {s}: {len(sc)} selection rows", flush=True)
    # the router tables, on their own plans (core rows; nuPlan R1 is benchmark_table_nuplan_real.csv, above)
    for name in ROUTERS:
        plans = ["routers_r1"] if name.startswith("R1_") else ["router_r2:nuScenes", "router_r2:KITTI"]
        for plan in plans:
            ks = {k for k in core if plan == "routers_r1" or k[0] == plan.split(":")[1]}
            f = write_submission(run, f"sel__{name}__{plan.replace(':', '_')}", cells, sig, name, keys=ks)
            if f is None:
                continue
            _, sc = S.score(S.load(f), plan=plan)
            sc = sc.rename(columns=ren)
            off = rt[(rt.signal == name) & rt.quota.notna() & rt.track.isin(sorted({k[0] for k in ks}))]
            off = off.merge(sc[list(S.CELL)].drop_duplicates().astype(str), on=list(S.CELL))
            assert_identity("benchmark_table_routers.csv", sc, off, [])
            rows += _field_rows("benchmark_table_routers.csv", f"{name} ({plan})",
                                _join(sc, off, key, "benchmark_table_routers.csv", name), FIELDS_ROUTERS)
            print(f"  G1' {name} ({plan}): {len(sc)} rows", flush=True)


def _budget_scored(run, cells, sig, signal, keys, overheads, units, tag):
    f = write_submission(run, f"bud__{tag}", cells, sig, signal, keys=keys)
    if f is None:
        return None
    prof = profile_for(signal, overheads, units)
    _, sc = S.score(S.load(f), profile=prof, selection=False)
    sc = sc[sc.track_kind == "measured budget"].rename(columns={"ndg": "eta", "ndg_lo": "eta_lo", "ndg_hi": "eta_hi"})
    for fld in BUDGET_FIELDS_NR + ("k",):          # an allocator infeasible everywhere never gets these columns
        if fld not in sc.columns:
            sc[fld] = np.nan
    return sc


def g1_budgets(run, cells, sig, rows, gates_equal):
    reg = S.registry()
    conv = reg["energy_convention"]["name"]
    two, rou = read_final("benchmark_budget_two_level.csv"), read_final("benchmark_budget_routers.csv")
    nrb = read_final("benchmark_budget_nuplan_real.csv")
    e_two, e_nr = read_final("energy_module_budget_two_level.csv"), read_final("energy_module_budget_nuplan_real.csv")
    mf, e_mf = read_final("benchmark_budget_multifidelity.csv"), read_final("energy_module_budget_multifidelity.csv")
    core = {tuple(c[k] for k in S.CELL) for c in cells if c["track"] != "nuPlan"}
    nup = {tuple(c[k] for k in S.CELL) for c in cells if c["track"] == "nuPlan"}
    bkey = list(S.CELL) + ["unit", "budget_level"]
    cost_checks = [("cheap_cost", "cheap_cost_official"), ("full_cost", "full_cost_official"),
                   ("budget_per_frame", "budget_per_frame_official"), ("cost_convention", "_convention")]
    mf_rows = {}

    def compare(table, label, sc, off, fields):
        off = off.assign(_convention=off.convention if "convention" in off.columns else conv)
        j = _join(sc, off, bkey, table, label)
        assert_identity(table, j, j, cost_checks + ([("labels", "labels_official")] if "labels" in off.columns else []))
        rows.extend(_field_rows(table, label, j, fields))

    plan_core = {"primary": DEPLOYABLE, "1thread": DEPLOYABLE,
                 "routers": DEPLOYABLE + ("gate_gbm_batched",) + ROUTERS}
    for overheads, signals in plan_core.items():
        units = ("mJ",) if overheads == "1thread" else ("ms", "mJ")
        for s in signals:
            sv = f"{s} (saved)" if (not gates_equal and f"{s} (saved)" in next(iter(sig.values()))) else s
            sc = _budget_scored(run, cells, sig, s, core, overheads, units, f"{s}__{overheads}")
            if sc is None:
                continue
            ms, mj = sc[sc.unit == "ms"], sc[sc.unit == "mJ"]
            if len(ms):
                table, src = (("benchmark_budget_two_level.csv", two) if overheads == "primary"
                              else ("benchmark_budget_routers.csv", rou))
                off = src[(src.signal == s) & (src.unit == "ms") & src.track.isin(["KITTI", "nuScenes"])]
                compare(table, f"{s} ({overheads})", ms, off, BUDGET_FIELDS)
            # energy: 133 charged the saved gate scores; a refit that is not bit-identical is scored separately
            if sv != s:
                sce = _budget_scored(run, cells, sig, sv, core, overheads, ("mJ",), f"{s}__{overheads}__saved")
                mj = sce[sce.unit == "mJ"]
            off = e_two[(e_two.signal == s) & (e_two.table == overheads) & (e_two.convention == conv)
                        & e_two.track.isin(["KITTI", "nuScenes"])]
            compare("energy_module_budget_two_level.csv", f"{s} ({overheads})", mj, off, BUDGET_FIELDS)
            if overheads != "routers" and s in ("random", "uncertainty"):
                mf_rows[(s, overheads)] = sc
            print(f"  G1' {s} ({overheads}): {len(sc)} budget rows", flush=True)
    # nuPlan real track: the routers measurement, nuScenes feature time as the proxy (registry provenance)
    for s in NR_SIGNALS + ("gate_gbm_batched",):
        sc = _budget_scored(run, cells, sig, s, nup, "routers", ("ms", "mJ"), f"{s}__nuplan_real")
        compare("benchmark_budget_nuplan_real.csv", s, sc[sc.unit == "ms"],
                nrb[(nrb.signal == s) & (nrb.unit == "ms")], BUDGET_FIELDS_NR)
        compare("energy_module_budget_nuplan_real.csv", s, sc[sc.unit == "mJ"],
                e_nr[(e_nr.signal == s) & (e_nr.convention == conv)], BUDGET_FIELDS_NR)
        print(f"  G1' {s} (nuPlan real): {len(sc)} budget rows", flush=True)
    # multi-fidelity tables: the single-level ("640 only") rows are two-level cascades on KITTI mono
    for overheads, units in (("primary", ("ms", "mJ")), ("1thread", ("mJ",))):
        mf_rows[("gate_gbm", overheads)] = _budget_scored(run, cells, sig, "gate_gbm (640-level)", set(MF_CELLS.values()),
                                                          overheads, units, f"gate_gbm_640level__{overheads}")
    for (s, overheads), sc in mf_rows.items():
        for system, key in MF_CELLS.items():
            part = sc[np.logical_and.reduce([sc[k].astype(str) == v for k, v in zip(S.CELL, key)])].copy()
            part["system"] = system
            part["share_640"] = np.where(part.feasible.astype(bool), part.k / part.n_frames, np.nan)
            for table, src in (("benchmark_budget_multifidelity.csv", mf), ("energy_module_budget_multifidelity.csv", e_mf)):
                if table.startswith("benchmark") and overheads != "primary":
                    continue
                off = src[(src.signal == f"{s} (640 only)") & (src.system == system)]
                if "table" in src.columns:          # energy: the registry's convention, this overhead measurement
                    off = off[(off.table == overheads) & (off.convention == conv)]
                else:                               # latency: ms rows (the mJ rows are the superseded convention)
                    off = off[off.unit == "ms"]
                p = part[part.unit.isin(set(off.unit))].drop(columns=["track", "signal"], errors="ignore")
                j = p.merge(off, on=["system", "unit", "budget_level"], how="outer", suffixes=("", "_official"),
                            validate="one_to_one", indicator=True)
                if (j._merge != "both").any():
                    raise IdentityError(f"G1' {table} {s}: scored and canonical single-level rows do not pair up")
                assert_identity(table, j.assign(track="KITTI"), j.assign(track="KITTI"),
                                [("budget_per_frame", "budget_per_frame_official"), ("n_frames", "n_frames_official")])
                rows.extend(_field_rows(table, f"{s} (640 only, {overheads})", j, MF_FIELDS))
    print("  G1' multi-fidelity single-level rows compared", flush=True)


def stage_g1(run, t92, t120, splits):
    global EXPECTED_VERSION
    EXPECTED_VERSION = S.registry()["version"]
    t133 = _load("t133", "133_energy_module_budgets.py")
    cells = cell_list(t92, t120, splits)
    # the export is what the canonical builders produce now: same labels, same run, same decision values
    for c in cells:
        key = tuple(c[k] for k in S.CELL)
        lab, src = labels_of(t92, c)
        got = S.cell_labels(key)
        v = S.decision_values()
        mm = np.logical_and.reduce([v[k].astype(str) == str(x) for k, x in zip(S.CELL, key)])
        if got != {"labels": lab, "source_run": src} or not np.array_equal(
                v.loc[mm, "V"].to_numpy(float), (c["d"][c["cheap"]] - c["d"][c["full"]]).to_numpy(float)):
            raise IdentityError(f"G1' identity: the exported cell {S.cell_id(key)} ({got}) is not what the canonical "
                                f"builder gives now ({lab}, {src}); re-run --stage values")
    sig, runs_used = official_signals(t92, t133, cells)
    rows = []
    g1_selection(run, cells, sig, rows)
    g1_budgets(run, cells, sig, rows, runs_used["gate_refit_equals_saved"])
    g = pd.DataFrame(rows)
    g.insert(0, "registry_version", EXPECTED_VERSION)
    g.to_csv(run / "g1.csv", index=False)
    g.to_csv(FINAL / "submission_path_g1.csv", index=False)
    g["structural"] = g.structural.fillna(0).astype(int)
    per = (g.assign(matched=g.rows - g.differing - g.structural).groupby("table", sort=False)
           .agg(signals=("signal", "nunique"), fields=("field", "nunique"), compared=("rows", "sum"),
                matched=("matched", "sum"), differing=("differing", "sum"), structural=("structural", "sum")))
    (run / "g1_runs.json").write_text(json.dumps({"registry_version": EXPECTED_VERSION, **runs_used}, indent=1))
    print(per.to_string())
    print(f"  runs: {runs_used}")
    bad = g[g.differing > 0]
    (run / "g1_per_table.csv").write_text(per.to_csv())
    print(f"  G1': {len(g)} table x signal x field comparisons, {int(per.compared.sum())} compared values, "
          f"{int(per.differing.sum())} differing, {int(per.structural.sum())} structural ({ALL_TIED_REASON})",
          flush=True)
    if len(bad):
        print(bad.to_string(index=False))
        raise SystemExit("G1' failed: the submission path does not reproduce the paper's tables")


EXPECTED_VERSION = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["values", "inputs", "g1"])
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"submission_{args.stage}", vars(args))
    t92 = _load("t92", "92_benchmark_table.py")
    t120 = _load("t120", "120_nuplan_real_allocation.py")
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    {"values": stage_values, "inputs": stage_inputs, "g1": stage_g1}[args.stage](run, t92, t120, splits)
    print("wrote", run)


if __name__ == "__main__":
    main()
