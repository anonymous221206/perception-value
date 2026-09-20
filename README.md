# Allocating perception compute by decision value: code release

This repository contains the code, intermediate caches and final results for an evaluation of when extra
perception compute changes a downstream decision.

The perception pair is YOLOv8s at 320 and 640 input resolution. Its effect on decisions is measured on three
tracks:

| track | data | downstream decision makers |
|---|---|---|
| nuScenes | nuScenes | braking controller; PKL's published planner |
| KITTI | KITTI tracking | braking controller; Planner B |
| nuPlan | nuPlan mini | published PDM-Closed and IDM planners, first under a transported detection profile, then under real detections |

Together these tracks form the **DEEP** benchmark. Allocation signals are scored under frame quotas and under
measured latency and energy budgets.

### Paper terms and repository names

| paper | repository |
|---|---|
| q_brake | system `brake`, target `J` |
| q_traj (Planner B) | system `traj`, target `JB` |
| q_plan (Planner C) | system `plan_ade`, target `JC_ade` |
| self control | `plannerC_path_dev` |
| detection-list router | `R1_*` |
| pixel router | `R2_cnn_clf` |
| gates | `gate_ridge`, `gate_gbm` |

## Repository layout

```
README.md, LICENSE, reproduce.py   entry points
configs/        frozen splits, benchmark scenario list, paths.env.example
data/           dataset instructions and helpers; data/cache/ = shipped intermediate caches
docs/           result reports and generated tables
environment/    environment pins, setup scripts, device check, Python wrappers (bin/)
results/final/  every final table, check file and figure-data export
results/raw/    shipped intermediate runs that later stages read
scripts/        the pipeline, numbered in order (index: scripts/README.md); scripts/legacy/ = early exploration
src/rap/        library code
tests/          unit tests and negative controls
third_party/    PROVENANCE.md; prior-work code is cloned here at pinned commits
```

## 1. Reproduce every table from the shipped caches (any machine, CPU only)

`data/cache/` and `results/raw/` ship the detections, decision tables, planner outputs and nuPlan branch results
that the paper's tables are computed from. The statistics on top of them are deterministic, so on any machine:

```bash
conda create -n rap python=3.8 && conda activate rap
pip install -r environment/requirements-cached.txt
cp configs/paths.env.example configs/paths.env    # defaults: datasets/ and models/ here, the active environment
source configs/paths.env
python reproduce.py --tier cached --verify
```

`--verify` regenerates each table and compares it with the shipped file, taken from the git commit when the
repository is a git checkout.
* Numbers are compared to 1e-9 relative (1e-12 absolute). NaN matches NaN in the same position, and a JSON
  boolean matches the number it stands for. Text, keys, list lengths and nulls must match exactly.
* A difference names the columns, or the JSON fields, that differ.
* The nuPlan check file declared by C10 is written by C9 and completed by C10, so a difference reported for it
  under C10 can come from either stage.

**Known differences.** `--verify` can report the following. Both appear on CPUs other than the reference platform
and leave the paper's numbers unchanged.
* C2 differs at about 1e-6;
* in C12, `R1_gbm_clf` on PDM-Closed safety changes at the 30–50% quotas; the 20% cells used in the paper are
  unchanged.

**C14 used to differ on every run, and no longer does.** Its per-cell bootstrap seed came from `hash()`, which
Python salts per process unless `PYTHONHASHSEED` is set, so each run drew a different sample and 71 to 95 of its
3,464 rows moved. The seed is now a stable digest and two full runs are byte-identical; the refits, which the
earlier note here blamed, are deterministic on this platform. Fixing it moved 72 rows, and among the figures
`docs/iclr_causal_threshold.md` quotes only the pooled lower bound, from −0.143 to −0.142, and one row of its 20%
table. That report's section 3 means and win counts are not stored in `causal_threshold.csv`, so `--verify` still
does not cover them. Whether the refits are reproducible on a *different* platform is untested.

C19 refits its models as C14 does, and reuses the saved G-target scores of `results/raw/20260915_214330_target_swap`
wherever the label is unchanged, as its pre-registration requires; its two outputs were byte-identical in five runs
on the reference platform, two of them with BLAS and OpenMP pinned to 1 and to 4 threads. It has not been run on a
second platform. `docs/iclr_target_swap.md` names the one quoted figure that another platform could move.

`python -m pytest tests` runs the unit tests. Tests that need KITTI files are skipped when the dataset is absent;
tests that need PyTorch or OpenCV are skipped in this CPU environment.

| result | file(s) | stage |
|---|---|---|
| benchmark table (frame quotas, all tracks, all baselines) | `results/final/benchmark_table.csv`, `benchmark_cells.csv`, `docs/benchmark_tables.md` | C2–C3 |
| per-mode operating-point calibration | `results/final/calibration_*.csv/json`, `docs/calibration_tables.md` | C4–C5 |
| detection-list and pixel routers | `results/final/benchmark_table_routers.csv`, `docs/routers_tables.md` | C6–C7 |
| measured latency and energy budgets | `results/final/benchmark_budget_two_level*.csv`, `benchmark_budget_routers.csv` | full tier (F3, H4) |
| nuPlan with real perception | `results/final/nuplan_real_perception_*.csv/json`, overlays, `docs/nuplan_real_perception_tables.md` | C9–C11 |
| nuPlan allocation on real-perception decision values | `results/final/benchmark_table_nuplan_real.csv`, `benchmark_budget_nuplan_real.csv` | C12 |
| mechanism table (detection changes where FULL helps or harms) | `results/final/mechanism_table.csv` | C13 |
| causal streaming allocation (frozen threshold, causal cap) | `results/final/causal_threshold.csv`, `docs/iclr_causal_threshold.md` | C14 |
| statistics hardening (raw gain, influence, trivial baselines) | `results/final/statistics_hardening.csv`, `docs/iclr_statistics.md` | C15 |
| consumer transfer matrix | `results/final/consumer_transfer.csv`, `docs/iclr_consumer_transfer.md` | C16 |
| objective swap (decision-value vs perception-gain selection) | `results/final/objective_swap.csv`, `docs/iclr_objective_swap.md` | C17 |
| skipping cost accounting for the pixel router | `results/final/skip_accounting.csv`, `docs/iclr_skip_accounting.md` | C18 |
| target swap: allocators trained on perception gain instead of decision value | `results/final/benchmark_target_swap.csv`, `benchmark_target_swap_summary.json`, `docs/iclr_target_swap.md` | C19 |
| target-swap audit (no compared output; its record is `results/raw/20260915_221911_target_swap_audit/`) | run `scripts/129_target_swap_audit.py` after C19 | — |
| figure data (BEV objects, gallery, budget curves) | `results/final/fig_*` | full tier (L1); budget curves also C20 |
| measured budgets with infeasible allocators marked (overhead above the budget headroom) | `results/final/benchmark_budget_*.csv`, `fig_budget_curves.csv`, `docs/iclr_budget_feasibility.md` | C20 (C12, C14, C18, C19 apply the same rule) |
| energy budgets under one rail convention for detectors and allocators, and the claims they bear on | `results/final/energy_module_*`, `allocator_rails.json`, `docs/iclr_energy_conventions.md` | C21, C22 (rails: full tier N1) |
| exact formulas: monocular lifting, reference geometry, controllers, perception gains, ego speed | `docs/iclr_formulas.md` | — |
| causal streaming with rate controllers (adaptive threshold, token bucket) on saved scores | `results/final/streaming_controllers.csv`, `docs/iclr_streaming_controllers.md` | C23 (V1 scores: full tier N3) |
| realism controls on KITTI: detection persistence (n = 1, 2, 3) and reference geometry for every pair | `results/final/persistence_sweep.csv`, `reference_geometry_sweep.csv`, `docs/iclr_realism_controls.md` | C24 (per-frame tables: full tier N4) |
| the nuScenes class-error fix: every figure it moves, old beside new | `results/final/class_error_fix.csv`, `docs/iclr_class_error_fix.md` | C25 (corrected primitives: full tier N5) |
| a causal nuScenes ego speed: the allocation signal stops reading a future pose | `results/final/causal_ego_speed.csv`, `docs/iclr_causal_ego_speed.md` | C26 (speed table: full tier N6) |
| the camera-to-ego translation in the monocular lift, as a sensitivity | `results/final/lift_offset_sensitivity.csv`, `docs/iclr_lift_offset_sensitivity.md` | C27 (per-frame tables: full tier N7) |

Reports that interpret these tables: `docs/iclr_budget_feasibility.md`, `docs/iclr_calibration.md`, `docs/iclr_causal_ego_speed.md`, `docs/iclr_causal_threshold.md`, `docs/iclr_class_error_fix.md`, `docs/iclr_consumer_transfer.md`, `docs/iclr_energy_conventions.md`, `docs/iclr_formulas.md`, `docs/iclr_idm_route_fix.md`, `docs/iclr_lift_offset_sensitivity.md`, `docs/iclr_nuplan_real_allocation.md`, `docs/iclr_nuplan_real_perception.md`, `docs/iclr_objective_swap.md`, `docs/iclr_phase0g_external_planners.md`, `docs/iclr_realism_controls.md`, `docs/iclr_routers.md`, `docs/iclr_skip_accounting.md`, `docs/iclr_statistics.md`, `docs/iclr_streaming_controllers.md`, `docs/iclr_target_swap.md`. The other files in `docs/` are generated tables (`*_tables.md`, `gate_spec.md`).

**IDM route input.** The first nuPlan runs handed IDM a route it could not start from in 58% of states. All
IDM-dependent results shipped here were regenerated after the fix.
* `docs/iclr_idm_route_fix.md` documents the bug, the checks and every changed number.
* `--no_route_fix` on `scripts/82_nuplan_counterfactual.py` and `scripts/115_nuplan_real_counterfactual.py`
  reproduces the original behaviour.

## 2. Full reproduction from raw data

1. Datasets. Download nuScenes (v1.0-trainval metadata, blob 01, map expansion), KITTI tracking and nuPlan
   (v1.1 mini split, maps) and arrange them as described in `data/README.md`. Then run
   `python data/verify_datasets.py`.
2. Weights: `bash data/download_models.sh`.
3. Prior-work code at pinned commits: `bash environment/setup_third_party.sh`.
4. Environments.
   * Main: Python 3.8 with NVIDIA's Jetson PyTorch and JetPack's TensorRT (`environment/requirements-edge.txt`).
   * nuPlan simulation: `bash environment/setup_nuplan_env.sh`.
5. `python reproduce.py --tier full --list` prints every stage in order. Run them all with
   `python reproduce.py --tier full`, or resume at a stage with `--from <id>`.

Stage markers: `[D]` needs datasets; `[H]` is hardware-dependent. The full pipeline takes several days on the
reference device. The nuPlan planner runs alone take about 8 hours.

## Hardware: read before running hardware-dependent stages

All latency, energy and TensorRT detection results were produced on this reference platform:

| component | version |
|---|---|
| device | NVIDIA Jetson AGX Xavier (MAXN) |
| JetPack / L4T | 5.1.6 / R35.6.5 |
| CUDA | 11.4 |
| cuDNN | 8.6 |
| TensorRT | 8.5.2.2 |
| PyTorch | 2.1.0a0+nv23.6 |

`python environment/check_device.py` compares your machine with it. On any other machine:

* engine build, profiling, detection and budget stages print a warning;
* stages that would overwrite shipped caches stop unless `RAP_ALLOW_OTHER_DEVICE=1` is set;
* latency, energy and allocator overheads will not match the paper;
* FP16 TensorRT detections can differ slightly, and so can every table computed from re-run detections.

Use the cached tier to reproduce the paper's numbers exactly.

## Third-party code and data

* **Third-party code.** nuplan-devkit, tuPlan Garage (PDM-Closed), PKL (planning-centric-metrics) and TIP are
  cloned at the commits in `third_party/PROVENANCE.md` and used unmodified, except for the deviations listed
  there.
* **Detector weights.** The public Ultralytics YOLOv8s and RT-DETR-l releases.
* **Dataset-derived files.** Caches, result tables, overlays and gallery images remain under their datasets'
  licenses: nuScenes and nuPlan CC BY-NC-SA 4.0, KITTI CC BY-NC-SA 3.0.

## License

The code is released under the MIT license (see `LICENSE`).
