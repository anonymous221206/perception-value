#!/usr/bin/env python3
"""Reproduce the results, stage by stage.

    python reproduce.py --tier cached --verify     every table and document from the shipped intermediate caches
                                                  (CPU only, no dataset, any machine); regenerated outputs are
                                                  compared with the shipped ones
    python reproduce.py --tier full --list         the complete pipeline from raw data, in order
    python reproduce.py --tier full --from K1      resume the full pipeline at a stage
    python reproduce.py --tier full --only B2      run one stage

Before running: `source configs/paths.env` (see configs/paths.env.example).  Scripts run through
environment/bin/py-edge or environment/bin/py-nuplan.  Hardware-dependent stages (engines, profiling, detection,
measured budgets) warn on any machine other than the reference platform and refuse to overwrite shipped caches
unless RAP_ALLOW_OTHER_DEVICE=1 (see src/rap/device.py).
"""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EDGE, NUPLAN = "environment/bin/py-edge", "environment/bin/py-nuplan"

# (id, description, interpreter, command, outputs compared by --verify)
CACHED = [
    ("C1", "external planners: per-planner summary and transfer (Track B, detection profile)", EDGE,
     "scripts/83_external_transfer.py --nboot 400",
     ["results/final/phase0g_external_planner_summary.csv", "results/final/phase0g_external_planner_transfer.csv",
      "results/final/phase0g_external_planner_stats.json"]),
    ("C2", "benchmark table: every cell, every baseline, one protocol", EDGE, "scripts/92_benchmark_table.py",
     ["results/final/benchmark_table.csv", "results/final/benchmark_cells.csv", "results/final/benchmark_self_agreement.csv"]),
    ("C3", "benchmark tables as markdown", EDGE, "scripts/94_benchmark_markdown.py", ["docs/benchmark_tables.md"]),
    ("C4", "per-mode operating-point calibration cells and reading", EDGE, "scripts/102_calibration_cells.py",
     ["results/final/calibration_thresholds.csv", "results/final/calibration_cells.csv", "results/final/calibration_sweep.csv",
      "results/final/calibration_brake_vs_plan.csv", "results/final/calibration_reading.json"]),
    ("C5", "calibration tables as markdown", EDGE, "scripts/109_calibration_markdown.py", ["docs/calibration_tables.md"]),
    ("C6", "detection-list routers (R1) on the frame-quota benchmark", EDGE, "scripts/103_routers_r1.py",
     ["results/final/benchmark_table_routers.csv"]),
    ("C7", "router tables as markdown", EDGE, "scripts/110_routers_markdown.py", ["docs/routers_tables.md"]),
    ("C8", "gate specification, generated from the code", EDGE, "scripts/105_gate_spec.py", ["docs/gate_spec.md"]),
    ("C9", "nuPlan real perception: recall and data checks", EDGE, "scripts/116_nuplan_real_cells.py --stage checks",
     ["results/final/nuplan_real_perception_recall.csv"]),
    ("C10", "nuPlan real perception: cells and pre-registered reading", EDGE, "scripts/116_nuplan_real_cells.py --stage cells",
     ["results/final/nuplan_real_perception_cells.csv", "results/final/nuplan_real_perception_checks.json"]),
    ("C11", "nuPlan real perception tables as markdown", EDGE, "scripts/117_nuplan_real_markdown.py",
     ["docs/nuplan_real_perception_tables.md"]),
    ("C12", "nuPlan allocation track on real-perception decision values (quotas and measured budgets)", EDGE,
     "scripts/120_nuplan_real_allocation.py",
     ["results/final/benchmark_table_nuplan_real.csv", "results/final/benchmark_budget_nuplan_real.csv"]),
]

# the full pipeline from raw data, in the order the results were produced; D = needs datasets, H = hardware-dependent
FULL = [
    ("A1", "[H] build FP16 TensorRT engines for every perception mode", EDGE, "scripts/00_build_engines.py"),
    ("A2", "[H] profile latency and energy of every mode", EDGE, "scripts/01_profile_jetson.py"),
    ("A3", "[H,D] KITTI detection at every fidelity (YOLOv8s, RT-DETR-l; options: --help)", EDGE, "scripts/02_run_detection.py"),
    ("A4", "[D] CHEAP/FULL pair selection on evidence", EDGE, "scripts/02b_mode_selection.py"),
    ("A5", "[H,D] nuScenes CAM_FRONT detection", EDGE,
     "scripts/40_nusc_detect.py --version v1.0-trainval --dataroot $RAP_DATASETS/nuscenes/trainval --out data/cache/nusc_det_tv"),
    ("B1", "[D] per-object mechanism table (KITTI)", EDGE, "scripts/08_mechanism.py"),
    ("B2", "fit the detector miss model transported to nuPlan", EDGE, "scripts/81_fit_detector_model.py"),
    ("B3", "[D] perception-metric robustness", EDGE, "scripts/50_percep_metrics.py"),
    ("B4", "[D] temporal replay", EDGE, "scripts/51_temporal.py --dataset kitti --tag temporal_kitti"),
    ("B5", "[D] core decision matrix (braking controller, lateral task)", EDGE, "scripts/52_core_matrix.py --tag core_matrix_postreview"),
    ("B6", "headline table, statistics and figures", EDGE, "{py} scripts/53_finalize.py && {py} scripts/54_figures.py"),
    ("B7", "[D] Planner B rollouts", EDGE, "scripts/65_planner_b_decision.py --params static_obstacles --tag planner_b_static_fixed --kitti_only"),
    ("C1", "[D] nuScenes detection submissions (oracle and mono geometry)", EDGE, "scripts/60_build_submissions.py"),
    ("C2", "[D] PKL and TIP on the submissions, in 6 chunks per metric and geometry", EDGE,
     "for m in pkl tip; do for v in oracle mono; do for c in 0 1 2 3 4 5; do {py} scripts/61_run_planning_metric.py "
     "--metric $m --variant $v --bsz 1 --nworkers 1 --nchunks 6 --chunk $c --tag ${m}_${v}_c$c --skip_existing || exit 1; done; done; done"),
    ("C3", "[D] Planner C (PKL's planner), in 6 chunks per geometry", EDGE,
     "for v in oracle mono; do for c in 0 1 2 3 4 5; do {py} scripts/66_planner_c_pkl_planner.py --variant $v --bsz 1 "
     "--nworkers 1 --nchunks 6 --chunk $c --tag planner_c_${v}_c$c --skip_existing || exit 1; done; done"),
    ("C4", "[D] Planner C against the real trajectory", EDGE,
     "{py} scripts/74_plannerC_vs_truth.py --variant oracle && {py} scripts/74_plannerC_vs_truth.py --variant mono"),
    ("C5", "planning-metric allocation (oracle and mono geometry) and figures", EDGE,
     "{py} scripts/62_planning_metric_eta.py --variant oracle --nboot 400 --tag phase0g_eta_fde_oracle && "
     "{py} scripts/62_planning_metric_eta.py --variant mono --nboot 400 --tag phase0g_eta_fde_mono && {py} scripts/63_phase0f_figures.py"),
    ("D1", "[D] Planner D raster cache and ego-velocity fix", EDGE,
     "{py} scripts/70_planner_d_data.py --split train && {py} scripts/70_planner_d_data.py --split val && "
     "{py} scripts/70_planner_d_data.py --split test && {py} scripts/70b_fix_ego_velocity.py"),
    ("D2", "[H] Planner D training: two variants x three seeds", EDGE,
     "for v in gt aug; do for s in 0 1 2; do {py} scripts/71_planner_d_train.py --variant $v --seed $s || exit 1; done; done"),
    ("D3", "Planner D evaluation of every checkpoint", EDGE,
     "for f in data/cache/planner_d/planner_d_*_best.pt; do {py} scripts/72_planner_d_eval.py --ckpt $f || exit 1; done"),
    ("D4", "planner transfer and cross-target measurement", EDGE,
     "{py} scripts/73_planner_transfer.py && for v in oracle mono; do s=$([ $v = mono ] && echo _mono); "
     "{py} scripts/75_cross_target_measurement.py --brake_table results/raw/20260913_133004_core_matrix_postreview/"
     "nuScenes__YOLOv8s__ns_cheap_320tons_full_640__$v.pkl --plan_csv data/cache/planner_d/planC_vs_truth$s.csv "
     "--label $v --nboot 400 --tag cross_target_tierobust_$v || exit 1; done"),
    ("E1", "[D] one untouched official nuPlan simulation per planner (environment check)", EDGE, "bash scripts/80_nuplan_smoke.sh"),
    ("E2", "[D] nuPlan counterfactual with the transported detection profile, IDM and PDM-Closed", NUPLAN,
     "{py} scripts/82_nuplan_counterfactual.py --planner idm --scenarios 60 --per_scenario 24 && "
     "{py} scripts/82_nuplan_counterfactual.py --planner pdm_closed --scenarios 60 --per_scenario 24"),
    ("E3", "external planner summaries, 2x2 decomposition, deployable gate and its checks", EDGE,
     "{py} scripts/83_external_transfer.py --nboot 400 && {py} scripts/85_external_2x2.py && "
     "{py} scripts/84_deployable_gate.py --nboot 400 && {py} scripts/86_gate_checks.py"),
    ("F1", "[D] frozen benchmark splits", EDGE, "scripts/90_benchmark_splits.py"),
    ("F2", "[D] nuPlan allocation signals (detection profile)", NUPLAN, "scripts/91_nuplan_cheap_features.py"),
    ("F3", "benchmark table and markdown", EDGE, "{py} scripts/92_benchmark_table.py && {py} scripts/94_benchmark_markdown.py"),
    ("G1", "[D] calibration: precision/recall curves, thresholds, per-mode outcomes", EDGE, "scripts/100_calibration_outcomes.py --workers 3"),
    ("G2", "[D] calibration: q_plan boxes and planning", EDGE,
     "{py} scripts/101_calibration_plan.py --stage boxes && {py} scripts/101_calibration_plan.py --stage plan"),
    ("G3", "calibration cells and markdown", EDGE, "{py} scripts/102_calibration_cells.py && {py} scripts/109_calibration_markdown.py"),
    ("H1", "[D] nuPlan R1 track lists", NUPLAN, "scripts/104_nuplan_track_lists.py"),
    ("H2", "R1 routers", EDGE, "scripts/103_routers_r1.py"),
    ("H3", "[H,D] R2 pixel router: image cache, training, TensorRT export and scoring", EDGE,
     "{py} scripts/107_router_r2.py --stage cache && {py} scripts/107_router_r2.py --stage train && "
     "{py} scripts/107_router_r2.py --stage export"),
    ("H4", "[H] measured latency and energy budgets with routers, batch-timing note, tables", EDGE,
     "{py} scripts/93_budget_allocation.py --routers --tag benchmark_budget_routers && "
     "{py} scripts/95_gate_inference_batch_timing.py && {py} scripts/110_routers_markdown.py && {py} scripts/105_gate_spec.py"),
    ("K1", "[D] nuPlan CAM_F0: archive directories, fetch plan, Range fetch (needs configs/nuplan_camera_urls.txt)", EDGE,
     "{py} scripts/112_nuplan_fetch_cam_f0.py --stage dirs && {py} scripts/112_nuplan_fetch_cam_f0.py --stage plan && "
     "{py} scripts/112_nuplan_fetch_cam_f0.py --stage fetch"),
    ("K2", "[H,D] YOLOv8s 320/640 on the nuPlan images", EDGE, "scripts/113_nuplan_detect.py"),
    ("K3", "[D] projection, matching, branch tracks and checks", NUPLAN, "scripts/114_nuplan_project_match.py"),
    ("K4", "real-perception checks", EDGE, "scripts/116_nuplan_real_cells.py --stage checks"),
    ("K5", "[D] reference and identity branches (stops if Track B is not reproduced)", NUPLAN,
     "{py} scripts/115_nuplan_real_counterfactual.py --planner idm --phase ref && "
     "{py} scripts/115_nuplan_real_counterfactual.py --planner pdm_closed --phase ref"),
    ("K6", "[D] real-perception branches", NUPLAN,
     "{py} scripts/115_nuplan_real_counterfactual.py --planner idm --phase branches && "
     "{py} scripts/115_nuplan_real_counterfactual.py --planner pdm_closed --phase branches"),
    ("K7", "real-perception cells, reading and tables", EDGE,
     "{py} scripts/116_nuplan_real_cells.py --stage cells && {py} scripts/117_nuplan_real_markdown.py"),
    ("L1", "[D] figure data export (BEV objects, gallery, budget curves)", EDGE, "scripts/118_figure_exports.py"),
    ("M1", "[D] nuPlan allocation inputs from the real CHEAP branch", NUPLAN, "scripts/119_nuplan_real_features.py"),
    ("M2", "nuPlan allocation track on real-perception decision values", EDGE, "scripts/120_nuplan_real_allocation.py"),
]


def render(py: str, cmd: str) -> str:
    """A bare `scripts/...` command runs under the stage's interpreter; `{py}` marks it inside compound commands."""
    return f"{py} {cmd}" if cmd.startswith("scripts/") else cmd.replace("{py}", py)


def compare(a: Path, b: Path) -> str:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return "identical" if a.read_bytes() == b.read_bytes() else "differs (byte comparison; install pandas for a numeric one)"
    if a.suffix == ".csv":
        x, y = pd.read_csv(a), pd.read_csv(b)
        if list(x.columns) != list(y.columns) or x.shape != y.shape:
            return f"DIFFERS: shape or columns {x.shape} vs {y.shape}"
        # row order is not part of a result (a script may append rows it keeps from an earlier run): sort by the
        # text columns, then by every column, before comparing
        keys = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])] + [c for c in x.columns if pd.api.types.is_numeric_dtype(x[c])]
        x = x.sort_values(keys, na_position="first", kind="mergesort").reset_index(drop=True)
        y = y.sort_values(keys, na_position="first", kind="mergesort").reset_index(drop=True)
        bad = []
        for c in x.columns:
            if pd.api.types.is_numeric_dtype(x[c]) and pd.api.types.is_numeric_dtype(y[c]):
                ok = np.isclose(x[c].to_numpy(float), y[c].to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)
                if not ok.all():
                    bad.append(f"{c} ({int((~ok).sum())} values)")
            elif not x[c].astype(str).equals(y[c].astype(str)):
                bad.append(f"{c} (text)")
        return "identical" if not bad else "DIFFERS: " + ", ".join(bad[:8])
    if a.suffix == ".json":
        def norm(o):
            if isinstance(o, float):
                return "nan" if o != o else round(o, 9)
            if isinstance(o, dict):
                return {k: norm(v) for k, v in o.items() if k != "seconds"}
            if isinstance(o, list):
                return [norm(v) for v in o]
            return o
        return "identical" if norm(json.loads(a.read_text())) == norm(json.loads(b.read_text())) else "DIFFERS"
    return "identical" if a.read_text() == b.read_text() else "DIFFERS"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=["cached", "full"], default="cached")
    ap.add_argument("--list", action="store_true", help="print the stages and exit")
    ap.add_argument("--only", help="run a single stage id")
    ap.add_argument("--from", dest="start", help="start at this stage id")
    ap.add_argument("--verify", action="store_true", help="cached tier: compare regenerated outputs with the shipped ones")
    args = ap.parse_args()
    stages = CACHED if args.tier == "cached" else [(s[0], s[1], s[2], s[3], []) for s in FULL]
    if args.list:
        for sid, desc, py, cmd, _ in stages:
            print(f"{sid:5s} {desc}\n      " + render(py, cmd))
        return
    ids = [s[0] for s in stages]
    if args.only:
        stages = [s for s in stages if s[0] == args.only]
    elif args.start:
        stages = stages[ids.index(args.start):]
    if not stages:
        sys.exit("no such stage; see --list")
    backup = ROOT / "results" / ".shipped_outputs"
    if args.verify:
        for _, _, _, _, outs in stages:
            for o in outs:
                if (ROOT / o).exists() and not (backup / o).exists():
                    (backup / o).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ROOT / o, backup / o)
    report = []
    for sid, desc, py, cmd, outs in stages:
        line = render(py, cmd)
        print(f"\n[{time.strftime('%H:%M:%S')}] {sid}: {desc}\n  {line}", flush=True)
        t0 = time.time()
        rc = subprocess.call(line, shell=True, cwd=ROOT, executable="/bin/bash")
        entry = {"stage": sid, "returncode": rc, "seconds": round(time.time() - t0)}
        if rc != 0:
            report.append(entry)
            print(f"  stage {sid} failed (exit {rc}); stopping", flush=True)
            break
        if args.verify:
            entry["outputs"] = {o: compare(ROOT / o, backup / o) if (backup / o).exists() else "no shipped copy"
                                for o in outs}
            for o, r in entry["outputs"].items():
                print(f"  {r:10s} {o}")
        report.append(entry)
    (ROOT / "results" / f"reproduce_{args.tier}_report.json").write_text(json.dumps(report, indent=1))
    failed = [r for r in report if r["returncode"] != 0 or any(v.startswith("DIFFERS") for v in r.get("outputs", {}).values())]
    print(f"\n{len(report)} stage(s) run; {'all outputs reproduced' if not failed else f'{len(failed)} stage(s) failed or differ'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
