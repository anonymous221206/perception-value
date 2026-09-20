# Allocating perception compute by decision value

Code, intermediate caches and results for an evaluation of **when extra perception compute changes a downstream
decision** — and of whether that change can be predicted well enough to allocate compute by it.

The headline pair is YOLOv8s at 320 pixels of input (cheap) against 640 (full). KITTI additionally carries
YOLOv8s at 384 and 512, and RT-DETR-l at 320 and 480, against the same full model — controls on the size of the
fidelity gap and on the detector family. Escalating a frame from cheap to full is scored by what it does to a
decision, not by what it does to a detection metric. Three tracks carry the measurement:

| track | data | downstream decision makers |
|---|---|---|
| nuScenes | nuScenes trainval | braking controller; PKL's published planner |
| KITTI | KITTI tracking | braking controller; Planner B, a rollout planner |
| nuPlan | nuPlan mini | published PDM-Closed and IDM planners, first under a transported detection profile, then under real detections |

Together they form the **DEEP** benchmark. Allocation signals are scored under frame quotas and under measured
latency and energy budgets on an NVIDIA Jetson AGX Xavier.

## Contents

```
reproduce.py      every stage, in order, in two tiers
configs/          frozen splits, the benchmark scenario list, paths.env.example
data/             dataset instructions and helpers; data/cache/ holds the shipped intermediate caches (214 MB)
docs/             result reports and generated tables
environment/      pins, setup scripts, a device check, and interpreter wrappers in environment/bin/
results/final/    every final table, check file and figure-data export (18 MB)
results/raw/      the intermediate runs that later stages read (188 MB)
scripts/          the pipeline, numbered in run order; the index is scripts/README.md,
                  and scripts/legacy/ holds early exploration that no stage calls
src/rap/          library code
tests/            unit tests and negative controls
third_party/      PROVENANCE.md; prior-work code is cloned here at pinned commits
```

## Requirements

Two paths through the repository, with different requirements.

| | cached tier | full tier |
|---|---|---|
| what it does | recomputes every table from the shipped caches | rebuilds everything from the datasets |
| hardware | any CPU | the reference Jetson for latency, energy and TensorRT stages |
| datasets | none | nuScenes, KITTI tracking, nuPlan mini |
| Python | 3.8, `environment/requirements-cached.txt` | 3.8, `environment/requirements-edge.txt`, plus a separate nuPlan environment |
| time | about three hours | several days |

Run the stages with a plain interpreter. Some internal invariants are `assert` statements, which `python -O`
removes. The pre-registered gates raise explicitly and hold either way, but the rest of the safety net does not.

## Quick start: reproduce every table from the shipped caches

`data/cache/` and `results/raw/` ship the detections, decision tables, planner outputs and nuPlan branch results
that every table in the paper is computed from. The statistics on top of them are deterministic, so no dataset,
GPU or download is needed:

```bash
conda create -n rap python=3.8 && conda activate rap
pip install -r environment/requirements-cached.txt
cp configs/paths.env.example configs/paths.env     # defaults: datasets/ and models/ here, the active environment
source configs/paths.env

python reproduce.py --tier cached --list           # the 27 stages, C1 to C27, in order
python reproduce.py --tier cached --verify         # run them all and compare against the shipped files
```

Useful flags: `--only <id>` for one stage, `--from <id>` to resume. `python -m pytest tests` runs the unit tests;
those needing KITTI files, PyTorch or OpenCV skip themselves when those are absent.

Roughly three hours on the reference device, dominated by the bootstraps in C14, C19, C22 and C23.

## What `--verify` checks

Each stage is re-run and its outputs compared with the shipped files, read from the git commit when the repository
is a checkout.

* Numbers must agree to 1e-9 relative and 1e-12 absolute. NaN matches NaN in the same position, and a JSON boolean
  matches the number it stands for. Text, keys, list lengths and nulls must match exactly.
* Row order is not treated as a result: tables are sorted before comparison.
* A reported difference names the columns, or the JSON fields, that differ.
* The nuPlan check file that C10 declares is written by C9 and completed by C10, so a difference reported under
  C10 can come from either stage.

**Expected differences on a machine other than the reference platform.** Neither changes a reported figure.

| stage | what differs | effect |
|---|---|---|
| C2 | about 1e-6 | none on any figure the paper quotes |
| C12 | `R1_gbm_clf` on PDM-Closed safety, at the 30–50% quotas | the 20% cells the paper uses are unchanged |

C14 and C19 fit models inside the stage. Both are byte-identical across runs on the reference platform; whether
the fits are reproducible on a different one is untested, and `docs/iclr_causal_threshold.md` and
`docs/iclr_target_swap.md` say which of their quoted figures another platform could move.

## Results index

| result | file(s) | stage |
|---|---|---|
| benchmark table (frame quotas, all tracks, all baselines) | `results/final/benchmark_table.csv`, `benchmark_cells.csv`, `docs/benchmark_tables.md` | C2–C3 |
| per-mode operating-point calibration | `results/final/calibration_*.csv/json`, `docs/calibration_tables.md` | C4–C5 |
| detection-list and pixel routers | `results/final/benchmark_table_routers.csv`, `docs/routers_tables.md` | C6–C7 |
| nuPlan with real perception | `results/final/nuplan_real_perception_*.csv/json`, overlays, `docs/nuplan_real_perception_tables.md` | C9–C11 |
| nuPlan allocation on real-perception decision values | `results/final/benchmark_table_nuplan_real.csv`, `benchmark_budget_nuplan_real.csv` | C12 |
| mechanism table (detection changes where FULL helps or harms) | `results/final/mechanism_table.csv` | C13 |
| causal streaming allocation (frozen threshold, causal cap) | `results/final/causal_threshold.csv`, `docs/iclr_causal_threshold.md` | C14 |
| statistics hardening (raw gain, influence, trivial baselines) | `results/final/statistics_hardening.csv`, `docs/iclr_statistics.md` | C15 |
| consumer transfer matrix | `results/final/consumer_transfer.csv`, `docs/iclr_consumer_transfer.md` | C16 |
| objective swap (decision-value vs perception-gain selection) | `results/final/objective_swap.csv`, `docs/iclr_objective_swap.md` | C17 |
| skipping cost accounting for the pixel router | `results/final/skip_accounting.csv`, `docs/iclr_skip_accounting.md` | C18 |
| target swap (allocators trained on perception gain instead of decision value) | `results/final/benchmark_target_swap.csv`, `benchmark_target_swap_summary.json`, `docs/iclr_target_swap.md` | C19 |
| measured budgets, allocators over the budget headroom marked infeasible | `results/final/benchmark_budget_*.csv`, `fig_budget_curves.csv`, `docs/iclr_budget_feasibility.md` | C20 (C12, C14, C18, C19 apply the same rule) |
| energy budgets under one rail convention, and the claims they bear on | `results/final/energy_module_*`, `allocator_rails.json`, `docs/iclr_energy_conventions.md` | C21–C22 (rail measurement: N1) |
| causal streaming with rate controllers (adaptive threshold, token bucket) | `results/final/streaming_controllers.csv`, `docs/iclr_streaming_controllers.md` | C23 (V1 scores: N3) |
| realism controls on KITTI (detection persistence, reference geometry) | `results/final/persistence_sweep.csv`, `reference_geometry_sweep.csv`, `docs/iclr_realism_controls.md` | C24 (per-frame tables: N4) |
| the nuScenes class-error fix: every figure it moves, old beside new | `results/final/class_error_fix.csv`, `docs/iclr_class_error_fix.md` | C25 (corrected primitives: N5) |
| a causal nuScenes ego speed: the allocation signal stops reading a future pose | `results/final/causal_ego_speed.csv`, `docs/iclr_causal_ego_speed.md` | C26 (speed table: N6) |
| the camera-to-ego translation in the monocular lift, as a sensitivity | `results/final/lift_offset_sensitivity.csv`, `docs/iclr_lift_offset_sensitivity.md` | C27 (per-frame tables: N7) |
| measured latency and energy budgets | `results/final/benchmark_budget_two_level*.csv`, `benchmark_budget_routers.csv` | full tier (F3, H4) |
| figure data (BEV objects, gallery, budget curves) | `results/final/fig_*` | full tier (L1); budget curves also C20 |
| exact formulas (monocular lifting, reference geometry, controllers, perception gains, ego speed) | `docs/iclr_formulas.md` | — |
| target-swap audit (writes no compared output) | `results/raw/20260915_221911_target_swap_audit/`; run `scripts/129_target_swap_audit.py` after C19 | — |

Reports that interpret these tables: `docs/iclr_budget_feasibility.md`, `iclr_calibration.md`,
`iclr_causal_ego_speed.md`, `iclr_causal_threshold.md`, `iclr_class_error_fix.md`, `iclr_consumer_transfer.md`,
`iclr_energy_conventions.md`, `iclr_formulas.md`, `iclr_idm_route_fix.md`, `iclr_lift_offset_sensitivity.md`,
`iclr_nuplan_real_allocation.md`, `iclr_nuplan_real_perception.md`, `iclr_objective_swap.md`,
`iclr_phase0g_external_planners.md`, `iclr_realism_controls.md`, `iclr_routers.md`, `iclr_skip_accounting.md`,
`iclr_statistics.md`, `iclr_streaming_controllers.md`, `iclr_target_swap.md`. The rest of `docs/` is generated
tables (`*_tables.md`, `gate_spec.md`).

### Naming: paper and repository

| paper | repository |
|---|---|
| q_brake | system `brake`, target `J` |
| q_traj (Planner B) | system `traj`, target `JB` |
| q_plan (Planner C) | system `plan_ade`, target `JC_ade` |
| self control | `plannerC_path_dev` |
| detection-list router | `R1_*` |
| pixel router | `R2_cnn_clf` |
| gates | `gate_ridge`, `gate_gbm` |

## Full reproduction from raw data

1. **Datasets.** Download nuScenes (v1.0-trainval metadata, blob 01, map expansion), KITTI tracking and nuPlan
   (v1.1 mini split, maps) and arrange them as `data/README.md` describes, then run `python data/verify_datasets.py`.
2. **Weights.** `bash data/download_models.sh` fetches the public Ultralytics YOLOv8s and RT-DETR-l releases.
3. **Prior-work code.** `bash environment/setup_third_party.sh` clones it at the pinned commits.
4. **Environments.** Python 3.8 with NVIDIA's Jetson PyTorch and JetPack's TensorRT
   (`environment/requirements-edge.txt`); the nuPlan simulation environment comes from
   `bash environment/setup_nuplan_env.sh`.
5. **Run.** `python reproduce.py --tier full --list` prints all 51 stages in order; `--tier full` runs them, and
   `--from <id>` resumes.

Stage markers: `[D]` needs datasets, `[H]` is hardware-dependent. The full pipeline takes several days on the
reference device, of which the nuPlan planner runs alone are about eight hours.

## Reference platform

Every latency, energy and TensorRT detection result was produced here:

| component | version |
|---|---|
| device | NVIDIA Jetson AGX Xavier (MAXN) |
| JetPack / L4T | 5.1.6 / R35.6.5 |
| CUDA | 11.4 |
| cuDNN | 8.6 |
| TensorRT | 8.5.2.2 |
| PyTorch | 2.1.0a0+nv23.6 |

`python environment/check_device.py` compares your machine with it. Elsewhere, engine build, profiling, detection
and budget stages warn; stages that would overwrite shipped caches stop unless `RAP_ALLOW_OTHER_DEVICE=1` is set;
latency, energy and allocator overheads will not match the paper; and FP16 TensorRT detections can differ slightly,
along with every table computed from re-run detections. Use the cached tier to reproduce the paper's numbers
exactly.

## Corrections carried in this release

Each was found after the first results were produced, and each has a report that gives the old value beside the
new one and the checks that bound what moved.

| correction | what it touched | report |
|---|---|---|
| IDM was handed a route it could not start from in 58% of states | every IDM-dependent nuPlan result, all regenerated; `--no_route_fix` on `scripts/82_*.py` and `scripts/115_*.py` restores the old behaviour | `docs/iclr_idm_route_fix.md` |
| nuScenes reference objects were all labelled `vehicle`, so correctly detected pedestrians and cyclists counted as class errors | the `cls` primitive, and through it E3 and the three E5 variants, on nuScenes only | `docs/iclr_class_error_fix.md` |
| the nuScenes ego-speed signal was a centred difference and read a pose half a second in the future | the ego-speed allocation signal; the decision values themselves are unchanged by construction | `docs/iclr_causal_ego_speed.md` |
| the causal-threshold bootstrap was seeded from `hash()`, which Python salts per process | C14's per-row numbers varied between runs; the seed is now a stable digest | `docs/iclr_causal_threshold.md` |

The monocular lift measures range in the camera frame rather than the ego frame. That is a modelling choice rather
than a defect, so it is shipped as a default-off flag and measured: `docs/iclr_lift_offset_sensitivity.md`.

## Third-party code and data

* **Code.** nuplan-devkit, tuPlan Garage (PDM-Closed), PKL (planning-centric-metrics) and TIP are cloned at the
  commits in `third_party/PROVENANCE.md` and used unmodified, except for the deviations listed there.
* **Detector weights.** The public Ultralytics YOLOv8s and RT-DETR-l releases.
* **Dataset-derived files.** Caches, result tables, overlays and gallery images remain under their datasets'
  licenses: nuScenes and nuPlan CC BY-NC-SA 4.0, KITTI CC BY-NC-SA 3.0.

## License

The code is released under the MIT license; see `LICENSE`.
