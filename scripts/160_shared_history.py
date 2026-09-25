#!/usr/bin/env python
"""Task 32, Phase 0: what a shared action history does to the sign of decision value, measured before anything is
regenerated.

The braking controller's loss charges the change from the previous action (jerk), the lateral controller's the switch
of corridor, and the trajectory controller both plans and scores against its previous plan. Every per-frame table so
far gave each branch its own history: V_i compared an all-CHEAP sequence with an all-FULL one at frame i, not one
escalation. Here every setting of the sign table is rebuilt from the cached detections (`decision.build` and
`65.build_b` with `history_variants`) and scored three ways:

  own         each branch against its own previous action (the shipped tables; checked to reproduce them exactly)
  shared      both branches against the previous action of the all-CHEAP run (None at a unit's first frame): the
              single escalation of input i from all-CHEAP operation; the trajectory controller re-plans the FULL
              branch with that history
  memoryless  the shared plans, scored without any action-change term (sensitivity only)

Settings: the core matrix (every detector pair and geometry of 52), the trajectory controller on every KITTI pair of
65, oracle geometry on every other KITTI pair (the reference-geometry control), and the calibration schemes S1-S3 of
`calibration_thresholds.csv` run directly at each (t_cheap, t_full) -- composing a cell from single-threshold runs,
as 100 does, is no longer valid, because the FULL loss now depends on the CHEAP branch's previous action. The learned
planner and the nuPlan planners carry no action history; their rows are the same under every variant.

Writes results/final/shared_history_effect.csv (new) and, in the run directory, every rebuilt per-frame table.
"""
from __future__ import annotations

import argparse, json, sys, time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import frames as rap_frames                                              # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta, scoring                                                # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL, RESULTS                        # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

FINAL = Path(RESULTS) / "final"
EPS = scoring.EPS
VARIANTS = ("own", "shared", "memoryless")
PAIRS = {  # label -> (dataset, detector, det_dir, cheap, full)
    "Y8 320->640": ("KITTI", "YOLOv8s", "det", "cheap_320", "full_640"),
    "Y8 384->640": ("KITTI", "YOLOv8s", "det", "cheap_384", "full_640"),
    "Y8 512->640": ("KITTI", "YOLOv8s", "det512", "cheap_512", "full_640"),
    "RT 320->640": ("KITTI", "RT-DETR-l", "rtdetr_kitti", "rt_cheap_320", "rt_full_640"),
    "RT 480->640": ("KITTI", "RT-DETR-l", "rtdetr_kitti_mid", "rt_mid_480", "rt_full_640"),
    "nuSc Y8 320->640": ("nuScenes", "YOLOv8s", "nusc_det_tv", "ns_cheap_320", "ns_full_640"),
}
CALIB_PAIR = {"Y8 320->640": "Y8 320->640", "Y8 384->640": "Y8 384->640", "Y8 512->640": "Y8 512->640",
              "RT-DETR 480->640": "RT 480->640"}
PLANNER_B = ("static_obstacles", "default")          # the benchmark's Planner B preset (100, 92)


def settings():
    """(setting id, pair label, geometry, scheme, t_cheap, t_full, systems)."""
    out = []
    for pair in PAIRS:
        ds = PAIRS[pair][0]
        for g in ("mono", "oracle"):
            sys_ = ["brake", "lateral"] + (["traj"] if ds == "KITTI" else [])
            out.append((f"{pair}|{g}|S0", pair, g, "S0", 0.25, 0.25, sys_))
    th = pd.read_csv(FINAL / "calibration_thresholds.csv")
    th = th[th.scheme.isin(["S1", "S2", "S3"])]
    for (cell, scheme), r in th.set_index(["cell", "scheme"]).iterrows():
        ds, geo = cell.split()[0], cell.split()[1]
        system = cell.split()[-1].replace("q_", "")
        if system == "plan":
            continue                                   # the learned planner: no action history
        pair = " ".join(cell.split()[2:-1])
        pair = "nuSc Y8 320->640" if ds == "nuScenes" else CALIB_PAIR[pair]
        out.append((f"{pair}|{geo}|{scheme}", pair, geo, scheme, float(r.t_cheap), float(r.t_full), [system]))
    # one build per (pair, geometry, scheme): merge the systems of the calibration rows
    merged = {}
    for sid, pair, geo, scheme, tc, tf, sys_ in out:
        if sid in merged:
            assert (merged[sid][3], merged[sid][4]) == (tc, tf), f"{sid}: two threshold pairs"
            merged[sid][5].extend(s for s in sys_ if s not in merged[sid][5])
        else:
            merged[sid] = [pair, geo, scheme, tc, tf, list(sys_)]
    return [(sid, *v) for sid, v in merged.items()]


class Builders:
    def __init__(self):
        self.m52 = import_module("52_core_matrix")
        self.m65 = import_module("65_planner_b_decision")
        self._nusc = None
        self.kseqs = [p.stem for p in sorted((CACHE / "det" / "cheap_320").glob("*.npz"))]
        self.nseqs = [p.stem for p in sorted((CACHE / "nusc_det_tv" / "ns_cheap_320").glob("*.npz"))]

    def adapter(self, ds):
        if ds == "KITTI":
            return self.m52.decision.KittiAdapter
        if self._nusc is None:
            self._nusc = self.m52.make_adapter(self.m52.NuScenesDB(str(NUSCENES_TRAINVAL), "v1.0-trainval"))
        return self._nusc

    def run(self, pair, geo, tc, tf, systems):
        ds, _, det_dir, cm, fm = PAIRS[pair]
        cfg = RiskConfig(op_conf=tc, op_conf_full=None if tc == tf else tf)
        ad = self.adapter(ds)
        seqs = self.kseqs if ds == "KITTI" else self.nseqs
        dd = CACHE / det_dir
        P = self.m52.P
        d = self.m52.decision.build(dd, cm, fm, seqs, cfg, P.PlannerParams(), P.CostParams(), self.m52.G.PRIMARY,
                                    range_source=geo, adapter=ad, history_variants=True, history="own")
        if "traj" in systems:
            B = self.m65.B
            b = self.m65.build_b(dd, cm, fm, seqs, cfg, ad, geo, B.PARAMS_B[PLANNER_B[0]], B.COSTS_B[PLANNER_B[1]],
                                 history_variants=True, history="own")
            keep = ["seq", "frame", "JB_cheap", "JB_full", "same_planB", "planB_cheap", "planB_full",
                    "planB_full_shared", "JB_full_shared", "same_planB_shared", "JB_cheap_ml", "JB_full_ml"]
            d = d.merge(b[keep], on=["seq", "frame"], how="left", validate="one_to_one")
        d["seq"] = d.seq.astype(str)
        return d


def variant_columns(system):
    """(cheap, full, same-action) column per variant."""
    if system == "brake":
        return {"own": ("J_cheap", "J_full", "same_action"), "shared": ("J_cheap", "J_full_shared", "same_action"),
                "memoryless": ("J_cheap_ml", "J_full_ml", "same_action")}
    if system == "lateral":
        return {"own": ("Jlat_cheap", "Jlat_full", "lat_same_action"),
                "shared": ("Jlat_cheap", "Jlat_full_shared", "lat_same_action"),
                "memoryless": ("Jlat_cheap_ml", "Jlat_full_ml", "lat_same_action")}
    return {"own": ("JB_cheap", "JB_full", "same_planB"), "shared": ("JB_cheap", "JB_full_shared", "same_planB_shared"),
            "memoryless": ("JB_cheap_ml", "JB_full_ml", "same_planB_shared")}


def metrics(cheap, full, same=None):
    cheap, full = np.asarray(cheap, float), np.asarray(full, float)
    v = cheap - full
    n = len(v)
    aff = np.abs(v) > EPS
    harm = v < -EPS
    pos, neg = v[v > EPS].sum(), -v[harm].sum()
    tot = cheap.sum()
    prize = scoring.topk_expect(v, [v], [max(int(round(0.2 * n)), 1)])[0][0][0]
    return {"n": n, "affected": int(aff.sum()), "harmed": int(harm.sum()),
            "harmed_share_affected": float(harm.sum() / max(aff.sum(), 1)), "harmed_share_all": float(harm.mean()),
            "rho": float(neg / pos) if pos > EPS else np.nan, "sum_V": float(v.sum()),
            "all_full_reduction": float(v.sum() / tot) if tot > EPS else np.nan,
            "oracle20_reduction": float(prize / tot) if tot > EPS else np.nan,
            "affected_same_action": (int((aff & (np.asarray(same) == 1)).sum()) if same is not None else np.nan)}


def check_own(sid, pair, geo, scheme, d):
    """The own-history columns must equal the shipped per-frame tables (core matrix and Planner B, S0)."""
    if scheme != "S0":
        return None
    ds, det, _, cm, fm = PAIRS[pair]
    name = f"{ds}__{det}__{cm}to{fm}__{geo}.pkl"
    res = {}
    core = rap_runs.core_matrix("postreview") / name
    if core.exists():
        c = pd.read_pickle(core).astype({"seq": str})
        j = d.merge(c, on=["seq", "frame"], suffixes=("", "_ship"), validate="one_to_one")
        assert len(j) == len(d) == len(c), (sid, len(j), len(d), len(c))
        for col in ("J_cheap", "J_full", "Jlat_cheap", "Jlat_full"):
            res[col] = float(np.max(np.abs(j[col].to_numpy(float) - j[f"{col}_ship"].to_numpy(float))))
    pb = rap_runs.planner_b() / f"planB__{ds}__{det.replace('-', '')}__{cm}to{fm}__{geo}.pkl"
    if "JB_cheap" in d.columns and pb.exists():
        b = pd.read_pickle(pb).astype({"seq": str})
        j = d.merge(b[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], suffixes=("", "_ship"),
                    validate="one_to_one")
        assert len(j) == len(b), (sid, len(j), len(b))
        for col in ("JB_cheap", "JB_full"):
            res[col] = float(np.max(np.abs(j[col].to_numpy(float) - j[f"{col}_ship"].to_numpy(float))))
    return res or None                                 # None: no shipped table for this setting


def no_history_rows():
    """The learned planner and the nuPlan planners: identical under every variant, from the shipped values."""
    d = pd.read_csv(FINAL / "benchmark_decision_values.csv.gz", dtype={"geometry": str}, keep_default_na=False,
                    na_values=[""])
    rows = []
    for (tr, g, s, t), x in d.groupby(["track", "geometry", "system", "target"]):
        if not (s.startswith("plan_") or tr == "nuPlan"):
            continue
        m = metrics(x.J_cheap, x.J_full)
        for var in VARIANTS:
            rows.append({"setting": f"{tr}|{g}|S0", "pair": "nuSc Y8 320->640" if tr == "nuScenes" else "nuPlan real",
                         "geometry": g, "scheme": "S0", "t_cheap": 0.25, "t_full": 0.25,
                         "system": f"{s} ({t})", "variant": var, **m,
                         "note": "no action history: the same under every variant"})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="setting ids to run (debug); default: all")
    rap_frames.add_argument(ap)
    args = ap.parse_args()
    rap_frames.configure(args)
    run = runmeta.new_run("shared_history_phase0", vars(args))
    todo = settings()
    if args.only:
        todo = [t for t in todo if t[0] in args.only]
    print(f"  {len(todo)} builds", flush=True)
    bld = Builders()
    rows, checks = [], {}
    t_all = time.time()
    for sid, pair, geo, scheme, tc, tf, systems in todo:
        t0 = time.time()
        d = bld.run(pair, geo, tc, tf, systems)
        d.to_pickle(run / (sid.replace("|", "__").replace("->", "to").replace(" ", "_") + ".pkl"))
        chk = check_own(sid, pair, geo, scheme, d)
        if chk is not None:
            checks[sid] = chk
            bad = {k: v for k, v in chk.items() if v != 0.0}
            if bad:
                raise SystemExit(f"{sid}: the own-history columns do not reproduce the shipped tables: {bad}")
        for system in systems:
            for var, (cc, fc, sc) in variant_columns(system).items():
                rows.append({"setting": sid, "pair": pair, "geometry": geo, "scheme": scheme, "t_cheap": tc,
                             "t_full": tf, "system": system, "variant": var,
                             **metrics(d[cc], d[fc], d[sc])})
        print(f"  {sid}: {len(d)} frames, {systems} [{time.time() - t0:.0f}s; reproduces shipped: "
              f"{'n/a' if chk is None else (not any(chk.values()))}]", flush=True)
        pd.DataFrame(rows).to_csv(run / "shared_history_effect.partial.csv", index=False)
    rows += no_history_rows()
    df = pd.DataFrame(rows)
    df.to_csv(run / "shared_history_effect.csv", index=False)
    (run / "own_history_checks.json").write_text(json.dumps(checks, indent=1))
    if not args.only:
        df.to_csv(FINAL / "shared_history_effect.csv", index=False)
    print(f"  wrote {len(df)} rows in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
