# Scripts

Scripts are numbered in pipeline order. `python reproduce.py --tier full --list` prints the exact order and
arguments. Run each script through `environment/bin/py-edge`; scripts marked **nuPlan** run through
`environment/bin/py-nuplan`. Every script documents its purpose in its module docstring, and `--help` lists its
options.

Markers:
* **H**: hardware-dependent. Warns on any machine other than the reference platform, and refuses to overwrite
  shipped caches unless `RAP_ALLOW_OTHER_DEVICE=1` (see `src/rap/device.py`).
* **D**: needs the raw datasets (see `data/README.md`).

## Detection and profiling

| script | what | marker |
|---|---|---|
| `00_build_engines.py` | ONNX export and FP16 TensorRT engine per perception mode | H |
| `01_profile_jetson.py` | latency and energy per mode | H |
| `02_run_detection.py` | KITTI detection at every fidelity | H, D |
| `02b_mode_selection.py` | CHEAP/FULL pair chosen on evidence | D |
| `40_nusc_detect.py` | nuScenes CAM_FRONT detection | H, D |

## Decision value and perception metrics

| script | what | marker |
|---|---|---|
| `08_mechanism.py` | per-object detection outcomes by range, size and criticality | D |
| `50_percep_metrics.py` | perception-metric robustness | D |
| `51_temporal.py` | temporal replay | D |
| `52_core_matrix.py` | braking and lateral decision tables for every configuration | D |
| `53_finalize.py`, `54_figures.py` | headline table, statistics, figures | |
| `99_synthetic_pipeline_check.py` | end-to-end plumbing check with simulated detections | |

## Planning-aware metrics and planners

| script | what | marker |
|---|---|---|
| `60_build_submissions.py`, `61_run_planning_metric.py` | nuScenes submissions, PKL and TIP | D |
| `62_planning_metric_eta.py`, `63_phase0f_figures.py` | allocation by planning-aware metrics | |
| `65_planner_b_decision.py` | Planner B rollouts | D |
| `66_planner_c_pkl_planner.py`, `74_plannerC_vs_truth.py` | Planner C (PKL's planner) against its own output and the real trajectory | D |
| `70_planner_d_data.py`, `70b_fix_ego_velocity.py`, `71_planner_d_train.py`, `72_planner_d_eval.py` | Planner D | D |
| `73_planner_transfer.py`, `75_cross_target_measurement.py` | planner transfer, cross-target measurement | |

## External planners on nuPlan

| script | what | marker |
|---|---|---|
| `80_nuplan_smoke.sh` | one untouched official simulation per planner | nuPlan, D |
| `81_fit_detector_model.py` | measured detector miss model | |
| `82_nuplan_counterfactual.py` | per-state counterfactual with the transported detection profile | nuPlan, D |
| `83_external_transfer.py`, `85_external_2x2.py` | planner summaries and transfer | |
| `84_deployable_gate.py`, `86_gate_checks.py` | deployable gate and its checks | |

## Benchmark

| script | what | marker |
|---|---|---|
| `90_benchmark_splits.py` | frozen splits | D |
| `91_nuplan_cheap_features.py` | nuPlan allocation signals (detection profile) | nuPlan, D |
| `92_benchmark_table.py`, `94_benchmark_markdown.py` | benchmark table | |
| `93_budget_allocation.py`, `95_gate_inference_batch_timing.py` | measured latency and energy budgets | H |

## Operating points and routers

| script | what | marker |
|---|---|---|
| `100_calibration_outcomes.py`, `101_calibration_plan.py` | per-mode operating points | D |
| `102_calibration_cells.py`, `109_calibration_markdown.py` | calibration cells and reading | |
| `103_routers_r1.py` | detection-list routers | |
| `104_nuplan_track_lists.py` | nuPlan R1 inputs | nuPlan, D |
| `107_router_r2.py` | pixel router, trained and exported to TensorRT | H, D |
| `105_gate_spec.py`, `110_routers_markdown.py` | generated documentation | |

## Real perception on nuPlan

| script | what | marker |
|---|---|---|
| `112_nuplan_fetch_cam_f0.py` | Range fetch of the CAM_F0 scenario-window images | D |
| `113_nuplan_detect.py` | YOLOv8s 320/640 on those images | H, D |
| `114_nuplan_project_match.py` | projection, matching, branch tracks, checks | nuPlan, D |
| `115_nuplan_real_counterfactual.py` | PDM-Closed and IDM on real-perception branches | nuPlan, D |
| `116_nuplan_real_cells.py`, `117_nuplan_real_markdown.py` | cells, reading, tables | |
| `119_nuplan_real_features.py` | allocation inputs from the real CHEAP branch | nuPlan, D |
| `120_nuplan_real_allocation.py` | allocation track on real-perception decision values | |

## Figure data

| script | what | marker |
|---|---|---|
| `118_figure_exports.py` | BEV object table, gallery frames, budget curves | D |

## Earlier exploratory phases

`legacy/` holds the first exploratory analyses (tables, kill tests, cross-task checks on KITTI and nuScenes mini).
Later scripts supersede them, and no result in `results/final/` depends on them.
