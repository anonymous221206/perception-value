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
    ("C13", "mechanism table: detection changes on frames where FULL helps, harms or does nothing", EDGE,
     "scripts/123_mechanism_table.py", ["results/final/mechanism_table.csv"]),
    ("C14", "causal streaming allocation: a threshold frozen before the test stream", EDGE,
     "scripts/124_causal_threshold.py", ["results/final/causal_threshold.csv"]),
    ("C15", "statistics hardening: raw-gain intervals, leave-one-unit-out influence, trivial baselines", EDGE,
     "scripts/125_statistics_hardening.py", ["results/final/statistics_hardening.csv"]),
    ("C16", "consumer transfer matrix: an allocator trained for one consumer, scored against another", EDGE,
     "scripts/126_consumer_transfer.py", ["results/final/consumer_transfer.csv"]),
    ("C17", "objective swap: which allocator each evaluation objective selects", EDGE,
     "scripts/127_objective_swap.py", ["results/final/objective_swap.csv"]),
    ("C18", "skipping cost accounting for the pixel router, beside the cascade", EDGE,
     "scripts/128_skip_accounting.py", ["results/final/skip_accounting.csv"]),
    ("C19", "target swap: the same allocators trained on perception gain instead of decision value", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/122_target_swap.py --reuse_gscores results/raw/20260922_131440_target_swap_gscores_nuplan",
     ["results/final/benchmark_target_swap.csv", "results/final/benchmark_target_swap_summary.json"]),
    ("C20", "measured budgets: allocators whose overhead exceeds the budget headroom marked infeasible", EDGE,
     "scripts/131_budget_feasibility.py",
     ["results/final/benchmark_budget_two_level.csv", "results/final/benchmark_budget_two_level_1thread.csv",
      "results/final/benchmark_budget_multifidelity.csv", "results/final/benchmark_budget_multifidelity_1thread.csv",
      "results/final/benchmark_budget_routers.csv", "results/final/fig_budget_curves.csv"]),
    ("C21", "energy conventions: the saved gate scores reproduce every shipped budget run output (check S)", EDGE,
     "scripts/133_energy_module_budgets.py --check_scores", ["results/final/energy_module_score_check.json"]),
    ("C22", "energy conventions: every energy-budget result under one rail convention, and the claims", EDGE,
     "scripts/133_energy_module_budgets.py",
     ["results/final/energy_module_budget_two_level.csv", "results/final/energy_module_budget_multifidelity.csv",
      "results/final/energy_module_budget_nuplan_real.csv", "results/final/energy_module_skip_accounting.csv",
      "results/final/energy_module_costs.json", "results/final/energy_module_claims.csv"]),
    ("C23", "causal streaming allocation with rate controllers (adaptive threshold, token bucket) on saved scores", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/130_streaming_controllers.py", ["results/final/streaming_controllers.csv"]),
    ("C24", "realism controls on the KITTI core track: detection persistence and reference geometry for every pair", EDGE,
     "scripts/135_realism_controls.py", ["results/final/persistence_sweep.csv", "results/final/reference_geometry_sweep.csv"]),
    ("C25", "the nuScenes class-error fix: every figure it moves, old beside new, and the identity check outside nuScenes",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/137_class_error_figures.py "
           "--before results/raw/20260922_145941_class_error_fix_ego/before "
           "--fix_run results/raw/20260920_090509_class_error_fix", ["results/final/class_error_fix.csv"]),
    ("C26", "the causal nuScenes ego speed: every figure it moves, the sanity gate and the first-frame sensitivity",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/139_causal_ego_figures.py "
           "--run results/raw/20260922_134139_causal_ego_speed_ego --no_sanity", ["results/final/causal_ego_speed.csv"]),
    ("C27", "the cost of the camera-frame convention: the five sign-variation quantities, ego frame against camera frame",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/141_lift_offset_sensitivity.py",
     ["results/final/lift_offset_sensitivity.csv"]),
    ("C28", "the lift in the ego frame: every registered quantity old (camera frame, as first shipped) beside new",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/147_ego_frame_convention.py",
     ["results/final/ego_frame_convention.csv", "results/final/ego_frame_convention_detail.csv.gz",
      "results/final/ego_frame_convention_reading.json"]),
    ("C29", "the published routing objectives: the labels checked against the official code, and their correlation "
     "with the perception-gain labels", EDGE,
     "bash environment/setup_third_party.sh edgeml-object-detection bgt-ada && PYTHONHASHSEED=0 {py} "
     "scripts/148_published_objective_labels.py --stage checks --label_run results/raw/20260922_193146_published_objective_labels --dump_run results/raw/20260922_192602_published_objective_dump",
     ["results/final/published_objective_labels.csv"]),
    ("C30", "the published objectives on R1 and R2: the refits, the budget-adaptive arbiter and the scoring", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/149_published_objective_routers.py --stage r1 --label_run results/raw/20260922_193146_published_objective_labels && "
     "PYTHONHASHSEED=0 {py} scripts/149_published_objective_routers.py --stage score --label_run results/raw/20260922_193146_published_objective_labels "
     "--r1_run $(ls -d results/raw/*_published_objective_r1 | tail -1) "
     "--r2_q results/raw/20260922_224841_router_r2_ego_pubQ --r2_g results/raw/20260922_231042_router_r2_ego_pubG "
     "--r2_arbiter results/raw/20260922_233230_router_r2_ego_arbG --dump_run results/raw/20260922_192602_published_objective_dump",
     ["results/final/published_objective_routers.csv", "results/final/published_objective_arbiter.csv",
      "results/final/published_objective_reading.json"]),
    ("C30b", "the cost registry: detector costs, energy convention, timing boundaries and allocator overheads, "
     "versioned (Task 28)", EDGE, "PYTHONHASHSEED=0 {py} scripts/156_cost_registry.py",
     ["results/final/cost_registry.json"]),
    ("C31", "the submission path: every cell's decision values and each official table's bootstrap plan", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/150_submission_tables.py --stage values",
     ["results/final/benchmark_decision_values.csv.gz", "results/final/benchmark_bootstrap_plans.json"]),
    ("C32", "gate G1': every official signal scored through the submission path against the tables the paper uses, "
     "and the pixel router's rows rebuilt from its saved scores", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/107_router_r2.py --stage rescore --dataset nuScenes --run results/raw/20260922_102405_router_r2_ego "
     "--tag router_r2_ego && PYTHONHASHSEED=0 {py} scripts/107_router_r2.py --stage rescore --dataset KITTI "
     "--run results/raw/20260922_102405_router_r2_ego --tag router_r2_ego && PYTHONHASHSEED=0 {py} scripts/150_submission_tables.py --stage g1",
     ["results/final/benchmark_table_routers.csv", "results/final/submission_path_g1.csv"]),
    ("C33", "the two example submissions, end to end through the CLI", EDGE,
     "PYTHONHASHSEED=0 {py} examples/random_allocator.py && PYTHONHASHSEED=0 {py} examples/confidence_heuristic.py",
     ["examples/submissions/random_scored.csv", "examples/submissions/confidence_scored.csv"]),
    ("C34", "the evidence pack and the claims check: every quantity a write-up may quote, and the verdicts", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/152_evidence_pack.py",
     ["results/final/paper_evidence_pack.csv", "results/final/claims_check.csv"]),
    ("C35", "four analyses on cached scores (Task 29): selection by objective, benefit and harm, overhead tolerance, "
     "and where the gap to the oracle goes; and the figure data", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/158_cached_analyses.py --part a && PYTHONHASHSEED=0 {py} scripts/158_cached_analyses.py "
     "--part b && PYTHONHASHSEED=0 {py} scripts/158_cached_analyses.py --part c && PYTHONHASHSEED=0 {py} "
     "scripts/158_cached_analyses.py --part d && PYTHONHASHSEED=0 {py} scripts/158_cached_analyses.py --part fig",
     ["results/final/cached_analyses_selection.csv", "results/final/cached_analyses_benefit_harm.csv",
      "results/final/cached_analyses_overhead_tolerance.csv", "results/final/cached_analyses_gap_accounting.csv",
      "results/final/cached_analyses_figure_data.csv"]),
    ("C36", "target transform against objective for the R1 regression routers (Task 30 part 1, post hoc; refits small "
     "CPU models) and its heatmap", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/159_target_transform.py --part 1 && PYTHONHASHSEED=0 {py} "
     "scripts/159_target_transform.py --part heatmap",
     ["results/final/target_transform_control.csv", "results/final/target_transform_heatmap.csv"]),
    ("C37", "training-seed variation of the learned allocators (Task 30 part 2, post hoc; six seeds, about 20 minutes)",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/159_target_transform.py --part 2",
     ["results/final/seed_variation.csv"]),
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
    ("B6", "Jetson cost columns, run manifest and figures (plus unshipped by-products)", EDGE, "{py} scripts/53_finalize.py && {py} scripts/54_figures.py"),
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
    ("D4", "cross-target measurement: braking controller against Planner C, oracle and mono geometry", EDGE,
     "for v in oracle mono; do s=$([ $v = mono ] && echo _mono); "
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
    ("F3", "[H] benchmark table, measured-cost budgets (default threads and OMP_NUM_THREADS=1), markdown", EDGE,
     "{py} scripts/92_benchmark_table.py && {py} scripts/93_budget_allocation.py && "
     "OMP_NUM_THREADS=1 {py} scripts/93_budget_allocation.py --tag benchmark_budget_1thread --suffix _1thread && "
     "{py} scripts/94_benchmark_markdown.py"),
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
    ("N1", "[H,D] every power rail over idle during the allocator workloads (then cached stages C20-C22)", EDGE,
     "scripts/132_allocator_rails.py"),
    ("N2", "save the core gate scores and multi-fidelity level predictions once (read by C21, C22)", EDGE,
     "scripts/133_energy_module_budgets.py --save_gate_scores"),
    ("N3", "fit the train-only (V1) allocator scores once for the streaming controllers (read by C23)", EDGE,
     "scripts/130_streaming_controllers.py --fit_once"),
    ("N4", "[D] KITTI per-frame braking and Planner B outcomes under persistence and reference geometry (read by C24)",
     EDGE, "scripts/134_realism_outcomes.py"),
    ("N5", "[D] recompute the nuScenes class primitives under the corrected coarse labels and check C1-C3 (read by C25)",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/136_class_error_fix.py --check_end_to_end"),
    ("N6", "[D] tabulate the causal nuScenes ego speed and check KITTI and nuPlan use no future data (read by C26)",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/138_causal_ego_speed.py"),
    ("N7", "[D] KITTI and nuScenes per-frame outcomes, ego-frame lift against camera frame (read by C27)",
     EDGE, "PYTHONHASHSEED=0 {py} scripts/140_lift_offset_outcomes.py"),
    ("N8", "[D] every unit's camera -> ego transform and intrinsics, read by every ego-frame re-lift "
           "(data/cache/cam_to_ego.json; ships with the release)", EDGE, "scripts/143_cam_to_ego_table.py"),
    ("N9", "[D] the gates of the ego-frame lift: G1 geometry, G2, B, D; then, after the ego-frame tables, G4 and G5",
     EDGE, "{py} scripts/144_ego_frame_gates.py g1geo && {py} scripts/144_ego_frame_gates.py g2 && "
           "{py} scripts/144_ego_frame_gates.py b && {py} scripts/144_ego_frame_gates.py d && "
           "{py} scripts/145_ego_frame_table_gates.py g4 && {py} scripts/145_ego_frame_table_gates.py g5"),
    ("N11", "[D] the CHEAP and FULL detections and reference objects of every cell, for the published objectives", EDGE,
     "scripts/148_published_objective_labels.py --stage dump"),
    ("N12", "the published-objective labels themselves (Q and G per frame; about three hours)", EDGE,
     "PYTHONHASHSEED=0 {py} scripts/148_published_objective_labels.py --stage labels"),
    ("N13", "[H] R2 refit on each published objective and on the arbiter's train-only target, and exported", EDGE,
     "for o in Q G; do {py} scripts/107_router_r2.py --stage train --dataset nuScenes --objective $o "
     "--label_run results/raw/20260922_193146_published_objective_labels --tag router_r2_ego_pub$o; done   # then --stage export per dataset, see docs/iclr_published_objectives.md"),
    ("N14", "[D] the labels-free inputs a submission may read", EDGE,
     "scripts/150_submission_tables.py --stage inputs"),
    ("N15", "[H] the cost profile of the confidence example, measured by the harness on the reference board", EDGE,
     "scripts/151_profile_allocator.py examples/confidence_heuristic.py --out examples/submissions/confidence_cost_profile.json"),
    ("N10", "the pre-fix nuScenes class labels in the ego-frame tables, C25's before (read by C25)", EDGE,
     "scripts/146_class_error_prefix_tables.py"),
]


def render(py: str, cmd: str) -> str:
    """A bare `scripts/...` command runs under the stage's interpreter; `{py}` marks it inside compound commands."""
    return f"{py} {cmd}" if cmd.startswith("scripts/") else cmd.replace("{py}", py)


def committed_output(path: str):
    """The committed bytes of a shipped output when this is a git checkout, otherwise None."""
    try:
        r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=ROOT, capture_output=True, check=False)
    except OSError:
        return None
    return r.stdout if r.returncode == 0 else None


def compare(a: Path, b: Path) -> str:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return "identical" if a.read_bytes() == b.read_bytes() else "differs (byte comparison; install pandas for a numeric one)"
    if a.name.endswith((".csv", ".csv.gz")):             # pandas reads the gzip one by its extension
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
        def strip(o):
            """Timing fields are not results."""
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k != "seconds"}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o

        def diffs(u, v, path="", out=None):
            """The paths at which two parsed JSON documents differ, with what differs there.

            NaN equals NaN in the same position, and numbers compare at rtol 1e-9, atol 1e-12, as in the CSV branch.
            A boolean equals the number it stands for: a writer that passes a NumPy boolean through
            `json.dumps(default=float)` writes 0.0 or 1.0 where the same value elsewhere is written false or true,
            which is a difference of representation, not of result.  Key sets, list lengths, text and nulls must
            match exactly.  `u` is the regenerated document, `v` the shipped one.
            """
            out = [] if out is None else out
            if isinstance(u, dict) and isinstance(v, dict):
                for k in sorted(set(u) ^ set(v), key=str):
                    out.append(f"{path}.{k} (key only in the {'regenerated' if k in u else 'shipped'} file)")
                for k in sorted(set(u) & set(v), key=str):
                    diffs(u[k], v[k], f"{path}.{k}", out)
            elif isinstance(u, list) and isinstance(v, list):
                if len(u) != len(v):
                    out.append(f"{path} (list length {len(u)} vs {len(v)})")
                else:
                    for i, (x, y) in enumerate(zip(u, v)):
                        diffs(x, y, f"{path}[{i}]", out)
            elif isinstance(u, (bool, int, float)) and isinstance(v, (bool, int, float)):
                if not np.isclose(float(u), float(v), rtol=1e-9, atol=1e-12, equal_nan=True):
                    out.append(f"{path} ({u!r} vs {v!r})")
            elif type(u) is not type(v) or u != v:
                out.append(f"{path} ({repr(u)[:40]} vs {repr(v)[:40]})")
            return out

        bad = diffs(strip(json.loads(a.read_text())), strip(json.loads(b.read_text())))
        return "identical" if not bad else f"DIFFERS: {len(bad)} field(s): " + "; ".join(bad[:8])
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
                committed = committed_output(o)
                if (backup / o).exists():
                    # a reference taken earlier is stale if outputs were regenerated in place before it was taken,
                    # for example by `--only C9` and then `--only C10`: both write the nuPlan check file
                    if committed is not None and (backup / o).read_bytes() != committed:
                        (backup / o).write_bytes(committed)
                        print(f"  stale reference for {o} replaced with the committed version", flush=True)
                elif committed is not None:
                    (backup / o).parent.mkdir(parents=True, exist_ok=True)
                    (backup / o).write_bytes(committed)
                elif (ROOT / o).exists():
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
