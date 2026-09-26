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
| `53_finalize.py`, `54_figures.py` | Jetson cost columns, run manifest, figures (53's headline and statistics tables are unshipped by-products) | |
| `123_mechanism_table.py` | FP, FN and detection-count changes on frames where FULL helps, harms or does nothing | |

## Planning-aware metrics and planners

| script | what | marker |
|---|---|---|
| `60_build_submissions.py`, `61_run_planning_metric.py` | nuScenes submissions, PKL and TIP | D |
| `62_planning_metric_eta.py`, `63_phase0f_figures.py` | allocation by planning-aware metrics | |
| `65_planner_b_decision.py` | Planner B rollouts | D |
| `66_planner_c_pkl_planner.py`, `74_plannerC_vs_truth.py` | Planner C (PKL's planner) against its own output and the real trajectory | D |
| `70_planner_d_data.py`, `70b_fix_ego_velocity.py`, `71_planner_d_train.py`, `72_planner_d_eval.py` | Planner D | D |
| `75_cross_target_measurement.py` | cross-target measurement (braking controller against Planner C) | |

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

## Robustness, controls and corrections

Each of these answers one question asked of the benchmark after its first results, and writes its own table.

| script | what | marker |
|---|---|---|
| `122_target_swap.py` | the same allocators trained on perception gain instead of decision value | |
| `124_causal_threshold.py` | causal streaming allocation with a threshold frozen before the test stream | |
| `125_statistics_hardening.py` | raw-gain intervals, leave-one-unit-out influence, trivial baselines | |
| `126_consumer_transfer.py` | consumer transfer matrix across the consumers of each track | |
| `127_objective_swap.py` | which allocator a decision-value and a perception-gain objective each select | |
| `128_skip_accounting.py` | the pixel router under skipping cost accounting, beside the cascade | |
| `129_target_swap_audit.py` | audit of the target swap: label, refit, degeneracy and bootstrap checks | |
| `130_streaming_controllers.py` | causal streaming allocation with rate controllers on frozen saved scores | |
| `131_budget_feasibility.py` | marks allocators whose overhead exceeds the budget headroom as infeasible in every measured-budget table | |
| `132_allocator_rails.py` | every power rail over idle while the gate, R1 and R2 workloads run | H, D |
| `133_energy_module_budgets.py` | every energy-budget result under one rail convention for detectors and allocators | |
| `134_realism_outcomes.py` | KITTI per-frame outcomes under detection persistence and reference geometry | D |
| `135_realism_controls.py` | persistence and reference-geometry sweeps, sanity gates and readings | |
| `136_class_error_fix.py` | recomputes the nuScenes class primitives under the corrected coarse labels; checks C1–C3 | D |
| `137_class_error_figures.py` | every figure that fix moves, old beside new, and the identity check outside nuScenes | |
| `138_causal_ego_speed.py` | tabulates the causal nuScenes ego speed; checks that KITTI and nuPlan use no future data | D |
| `139_causal_ego_figures.py` | every figure the causal ego speed moves, the sanity gate and the first-frame sensitivity | |
| `140_lift_offset_outcomes.py` | KITTI and nuScenes per-frame outcomes, ego-frame lift against camera frame (Task 22: translation only) | D |
| `141_lift_offset_sensitivity.py` | the five sign-variation quantities per setting, the sanity gate and the reading | |
| `142_frame_transform_audit.py` | the audit behind the ego-frame pre-registration: residual rotation, lift variants against the reference | D |
| `143_cam_to_ego_table.py` | every unit's camera -> ego transform and intrinsics, `data/cache/cam_to_ego.json` | D |
| `144_ego_frame_gates.py` | gates G1 (geometry and tables), G2, and measurements B and D of the ego-frame lift | D (g1geo: none) |
| `145_ego_frame_table_gates.py` | gates G4 (class primitives) and G5 (oracle geometry) on the ego-frame tables | D |
| `146_class_error_prefix_tables.py` | the pre-fix nuScenes class labels in the ego-frame tables, C25's before | |
| `147_ego_frame_convention.py` | every registered quantity old (camera frame) beside new (ego frame), and the reading | |

## The published objectives, the submission path and the evidence pack

| script | what | marker |
|---|---|---|
| `148_published_objective_labels.py` | Task 24: the CHEAP/FULL detection dump, the published objectives as per-frame labels (Qiu's ORIC, Geng's ΔAP, from the official code), and their checks | D (dump) |
| `149_published_objective_routers.py` | Task 24: R1 and R2 refit on those labels, the budget-adaptive arbiter, and the scoring against the V-trained counterparts | |
| `150_submission_tables.py` | Tasks 25 and 28: the decision values and bootstrap plans the submission path needs (nuPlan from the real-perception track), the labels-free inputs export, and gate G1' against the tables the paper uses | D (inputs) |
| `151_profile_allocator.py` | Task 25: measures an allocator's own per-input latency and energy for the measured-budget track, on the cost registry's rails | H |
| `152_evidence_pack.py` | Task 26: the evidence pack, the claims check, and the 3A/3B tables | |
| `156_cost_registry.py` | Task 28: the versioned cost registry every measured-budget score is charged against | |
| `157_doc_staleness.py` | Task 28: which documents quote numbers the results no longer hold (reads the git history; writes no compared output) | |
| `158_cached_analyses.py` | Task 29: selection by objective, benefit and harm, overhead tolerance, gap accounting, figure data | |
| `159_target_transform.py` | Task 30 (post hoc): R1 regression on rank and signed-ECDF targets, training-seed variation, heatmap | |
| `160_shared_history.py` | Task 32 Phase 0: decision values under own, shared and memoryless action history, every setting of the sign table | D |
| `161_calibration_direct.py` | Task 32: calibration cells on the shared action history (braking composed exactly from per-threshold actions) | D |
| `162_significance_table.py` | Task 32: one significance table for every deployable signal | |
| `163_loss_term_shares.py` | Task 32: harm and benefit by loss term, as shares | |
| `164_old_new_inventory.py` | compares two `results/final` directories file by file (e.g. a regenerated tree against the shipped one); Task 32's inventory is in `results/raw/*_shared_history_before/inventory/` | |
| `165_multifidelity_levels.py` | Task 32: the multi-fidelity levels 320->384 and 320->512 on the shared action history | D |
| `167_loss_sensitivity.py` | Task 35 (post hoc): loss-weight and corridor sensitivity; part A exact, parts B and C rebuilt | D (B, C) |

## Figure data

`paper_figures/` holds the scripts that draw the paper's figures from release artifacts; see its README.

| script | what | marker |
|---|---|---|
| `118_figure_exports.py` | BEV object table, gallery frames, budget curves | D |
