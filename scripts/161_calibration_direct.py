#!/usr/bin/env python
"""Task 32: the calibration cells and sweep on the shared action history.

100 builds each mode at one threshold and 102 composes a cell at (t_c, t_f) from the CHEAP columns of the t_c run and
the FULL columns of the t_f run, because no quantity coupled the two modes. Under the shared action history the FULL
loss is charged against the CHEAP branch's previous action, so a FULL loss at t_f depends on t_c. This script supplies
the shared-history values 102 needs:

  braking     The braking action is a threshold rule on the perceived requirement: it does not depend on the history,
              only its loss does. Each mode is built once per threshold at (t, t) keeping the per-frame actions, and a
              cell at (t_c, t_f) is composed exactly:
                  J_full(t_c, t_f)[i] = decision_cost(act_full(t_f)[i], a_gt[i], act_cheap(t_c)[i-1])
              (none at a sequence's first frame, as in decision.build). Every scheme S0-S4 and the 5 x 5 sweep.
              Checked bit for bit against the direct shared-history builds of Phase 0 (every S0-S3 pair).
  trajectory  Planner B re-plans with the previous plan, so its FULL action depends on the history. Schemes S0-S3
              come from the direct builds of Phase 0 (`*_shared_history_phase0`, the `JB_full_shared` column, run at
              exactly the registered (t_c, t_f)); S0 is also checked against 65's new shared-history table. Its S4 and
              sweep are not recomputed; 102 labels them.

Thresholds are the registered ones (`calibration_thresholds.csv`; S4 chosen from the per-mode losses, unchanged).
KITTI builds run in a pool of --workers processes, nuScenes in this process so its database is loaded once.
"""
from __future__ import annotations

import argparse, json, sys, time
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import frames as rap_frames                                              # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap import runs as rap_runs                                                # noqa: E402
from rap import planner as P                                                    # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL, RESULTS                        # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

SWEEP = [0.15, 0.25, 0.35, 0.45, 0.55]
KEEP = ["act_cheap", "act_full", "a_gt", "J_cheap", "J_full"]
# calibration spec -> Phase 0 setting prefix (160's PAIRS label with its file-name spelling)
PHASE0 = {"nusc_oracle": "nuSc_Y8_320to640__oracle", "nusc_mono": "nuSc_Y8_320to640__mono",
          "kitti_y8_320_mono": "Y8_320to640__mono", "kitti_y8_320_oracle": "Y8_320to640__oracle",
          "kitti_y8_384_mono": "Y8_384to640__mono", "kitti_y8_512_mono": "Y8_512to640__mono",
          "kitti_rt_480_mono": "RT_480to640__mono"}
_W = {}


def _init(dataset):
    m52 = import_module("52_core_matrix")
    _W.update(m100=import_module("100_calibration_outcomes"), m52=m52)
    _W["adapter"] = (m52.make_adapter(m52.NuScenesDB(str(NUSCENES_TRAINVAL), "v1.0-trainval")) if dataset == "nuScenes"
                     else m52.decision.KittiAdapter)
    seqdir = (CACHE / "nusc_det_tv" / "ns_cheap_320") if dataset == "nuScenes" else (CACHE / "det" / "cheap_320")
    _W["seqs"] = [p.stem for p in sorted(seqdir.glob("*.npz"))]


def task(spec, t, outdir):
    """Both modes of one spec at threshold t: the per-frame actions and losses (shared history)."""
    t0 = time.time()
    m52 = _W["m52"]
    _, det_dir, cm, fm, geo, _, _ = _W["m100"].SPECS[spec]
    cfg = RiskConfig(op_conf=t, op_conf_full=t)
    d = m52.decision.build(CACHE / det_dir, cm, fm, _W["seqs"], cfg, m52.P.PlannerParams(), m52.P.CostParams(),
                           m52.G.PRIMARY, range_source=geo, adapter=_W["adapter"])
    name = f"{spec}__t{t:.6f}.npz"
    np.savez_compressed(outdir / name, seq=d.seq.astype(str).to_numpy().astype("U32"), frame=d.frame.to_numpy(int),
                        **{c: d[c].to_numpy() for c in KEEP})
    return spec, t, name, time.time() - t0


def load_mode(run, spec, t):
    idx = pd.read_csv(run / "mode_index.csv")
    r = idx[(idx.spec == spec) & np.isclose(idx.t, t, atol=1e-9)]
    assert len(r) == 1, (spec, t, len(r))
    z = np.load(run / "modes" / r.file.iloc[0], allow_pickle=False)
    d = pd.DataFrame({k: z[k] for k in z.files})
    d["seq"] = d.seq.astype(str)
    return d


def compose_brake(c, f):
    """Braking losses at (t_c, t_f) on the shared history, from the (t_c, t_c) and (t_f, t_f) mode builds."""
    assert (c.seq.to_numpy() == f.seq.to_numpy()).all() and (c.frame.to_numpy() == f.frame.to_numpy()).all()
    assert np.array_equal(c.a_gt.to_numpy(), f.a_gt.to_numpy())
    pp, cp = P.PlannerParams(), P.CostParams()
    first = np.r_[True, c.seq.to_numpy()[1:] != c.seq.to_numpy()[:-1]]
    prev = np.r_[-1, c.act_cheap.to_numpy()[:-1]]
    jf = np.array([P.decision_cost(int(a), float(g), None if fst else int(p), pp, cp)["J"]
                   for a, g, p, fst in zip(f.act_full.to_numpy(), f.a_gt.to_numpy(), prev, first)])
    return pd.DataFrame({"seq": c.seq, "frame": c.frame, "J_cheap": c.J_cheap.to_numpy(), "J_full": jf})


def needed(th, specs):
    want = {s: set(SWEEP) for s in specs}
    for _, r in th.iterrows():
        spec = r.spec
        if spec in want:
            want[spec] |= {float(r.t_cheap), float(r.t_full)}
    return want


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    rap_frames.add_argument(ap)
    args = ap.parse_args()
    rap_frames.configure(args)
    m100, m102 = import_module("100_calibration_outcomes"), import_module("102_calibration_cells")
    cell_spec = {c[0]: c[2] for c in m102.CELLS}
    th = pd.read_csv(Path(RESULTS) / "final" / "calibration_thresholds.csv")
    th["spec"] = th.cell.map(cell_spec)
    th["system"] = th.cell.str.split().str[-1].str.replace("q_", "", regex=False)
    p0 = rap_runs.runs("shared_history_phase0")[-1]
    run = runmeta.new_run("calibration_direct", {**vars(args), "phase0": p0.name,
                                                 "thresholds": "results/final/calibration_thresholds.csv"})
    (run / "modes").mkdir()
    th.to_csv(run / "thresholds_used.csv", index=False)
    brake = th[th.system == "brake"]
    want = needed(brake, sorted(brake.spec.unique()))
    jobs = [(s, t) for s in sorted(want) for t in sorted(want[s])]
    print(f"  {len(jobs)} mode builds over {len(want)} specs", flush=True)

    t_all, rows = time.time(), []
    kitti = [j for j in jobs if m100.SPECS[j[0]][0] == "KITTI"]
    nusc = [j for j in jobs if m100.SPECS[j[0]][0] == "nuScenes"]
    with get_context("spawn").Pool(args.workers, initializer=_init, initargs=("KITTI",)) as pool:
        futs = [pool.apply_async(task, (s, t, run / "modes")) for s, t in kitti]
        _init("nuScenes")
        for s, t in nusc:
            rows.append(task(s, t, run / "modes"))
            print(f"  {rows[-1][2]} [{rows[-1][3]:.0f}s]", flush=True)
        for f in futs:
            rows.append(f.get())
            print(f"  {rows[-1][2]} [{rows[-1][3]:.0f}s]", flush=True)
    pd.DataFrame(rows, columns=["spec", "t", "file", "seconds"]).to_csv(run / "mode_index.csv", index=False)
    t_build = time.time() - t_all

    # braking: compose every (t_c, t_f) a table needs
    pairs = {(r.spec, float(r.t_cheap), float(r.t_full)) for _, r in brake.iterrows()}
    pairs |= {(s, tc, tf) for s in want for tc in SWEEP for tf in SWEEP}
    out, index = run / "cells", []
    out.mkdir()
    for spec, tc, tf in sorted(pairs):
        d = compose_brake(load_mode(run, spec, tc), load_mode(run, spec, tf))
        name = f"brake__{spec}__c{tc:.6f}__f{tf:.6f}.npz"
        np.savez_compressed(out / name, seq=d.seq.to_numpy().astype("U32"), frame=d.frame.to_numpy(int),
                            Jc=d.J_cheap.to_numpy(), Jf=d.J_full.to_numpy())
        index.append({"system": "brake", "spec": spec, "t_cheap": tc, "t_full": tf, "file": name,
                      "source": "composed from mode builds"})

    # trajectory S0-S3: Phase 0's direct builds
    traj = th[(th.system == "traj") & th.scheme.isin(["S0", "S1", "S2", "S3"])]
    for _, r in traj.iterrows():
        d = pd.read_pickle(p0 / f"{PHASE0[r.spec]}__{r.scheme}.pkl")
        name = f"traj__{r.spec}__c{r.t_cheap:.6f}__f{r.t_full:.6f}.npz"
        np.savez_compressed(out / name, seq=d.seq.astype(str).to_numpy().astype("U32"), frame=d.frame.to_numpy(int),
                            Jc=d.JB_cheap.to_numpy(float), Jf=d.JB_full_shared.to_numpy(float))
        index.append({"system": "traj", "spec": r.spec, "t_cheap": float(r.t_cheap), "t_full": float(r.t_full),
                      "file": name, "source": f"{p0.name}/{PHASE0[r.spec]}__{r.scheme}.pkl"})
    idx = pd.DataFrame(index).drop_duplicates(["system", "spec", "t_cheap", "t_full"])
    idx.to_csv(run / "cell_index.csv", index=False)

    # checks
    checks = {}
    # (1) composed braking = Phase 0's direct shared-history builds, every S0-S3 pair
    for _, r in brake[brake.scheme.isin(["S0", "S1", "S2", "S3"])].iterrows():
        d0 = pd.read_pickle(p0 / f"{PHASE0[r.spec]}__{r.scheme}.pkl")
        d0["seq"] = d0.seq.astype(str)
        z = np.load(out / f"brake__{r.spec}__c{r.t_cheap:.6f}__f{r.t_full:.6f}.npz")
        same_rows = (z["seq"].astype(str) == d0.seq.to_numpy()).all() and (z["frame"] == d0.frame.to_numpy()).all()
        dc = float(np.max(np.abs(z["Jc"] - d0.J_cheap.to_numpy(float))))
        df = float(np.max(np.abs(z["Jf"] - d0.J_full_shared.to_numpy(float))))
        checks[f"brake {r.cell} {r.scheme}"] = {"rows": bool(same_rows), "max_diff_cheap": dc, "max_diff_full": df,
                                                "pass": bool(same_rows and dc == 0.0 and df == 0.0)}
    # (2) the (t, t) mode builds are shared-history builds: J_full at (t, t) = composition at (t, t)
    for spec in want:
        m = load_mode(run, spec, 0.25)
        d = compose_brake(m, m)
        df = float(np.max(np.abs(d.J_full.to_numpy() - m.J_full.to_numpy())))
        checks[f"brake {spec} (0.25, 0.25) self"] = {"max_diff_full": df, "pass": df == 0.0}
    # (3) trajectory S0 from Phase 0 = 65's new shared-history table
    try:
        pb = rap_runs.planner_b()
        m100s = m100.SPECS
        for spec in sorted(traj.spec.unique()):
            new = pd.read_pickle(pb / m100s[spec][6])
            new["seq"] = new.seq.astype(str)
            d0 = pd.read_pickle(p0 / f"{PHASE0[spec]}__S0.pkl")
            d0["seq"] = d0.seq.astype(str)
            j = d0[["seq", "frame", "JB_cheap", "JB_full_shared"]].merge(
                new[["seq", "frame", "JB_cheap", "JB_full"]], on=["seq", "frame"], suffixes=("_p0", "_new"),
                validate="one_to_one")
            dc = float(np.max(np.abs(j.JB_cheap_p0 - j.JB_cheap_new)))
            df = float(np.max(np.abs(j.JB_full_shared - j.JB_full)))
            checks[f"traj {spec} S0 vs {pb.name}"] = {"n": len(j), "n_phase0": len(d0), "max_diff_cheap": dc,
                                                      "max_diff_full": df,
                                                      "pass": bool(len(j) == len(d0) and dc == 0 and df == 0)}
    except Exception as e:                                   # noqa: BLE001
        checks["traj S0 vs 65"] = {"pass": False, "error": repr(e)}
    ok = all(v["pass"] for v in checks.values())
    (run / "checks.json").write_text(json.dumps({"all_pass": ok, "build_seconds": t_build, "checks": checks},
                                                indent=1))
    for k, v in checks.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}  {v}", flush=True)
    print(f"  {len(jobs)} builds in {t_build:.0f}s; {len(idx)} cells; all_pass={ok}; {run}", flush=True)
    if not ok:
        raise SystemExit("checks failed")


if __name__ == "__main__":
    main()
