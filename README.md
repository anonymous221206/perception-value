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
data/submission_inputs/  what an allocator may read at test time, labels-free (10 MB; docs/SUBMITTING.md)
docs/             result reports and generated tables
evaluate_submission.py   score your own allocator against the frozen test split (docs/SUBMITTING.md)
examples/         two runnable submissions and their scored output
environment/      pins, setup scripts, a device check, and interpreter wrappers in environment/bin/
results/final/    every final table, check file and figure-data export (42 MB)
results/raw/      the intermediate runs that later stages read (430 MB)
scripts/          the pipeline, numbered in run order; the index is scripts/README.md, and
                  scripts/paper_figures/ draws the paper's figures from release artifacts
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
| time | about four hours | several days |

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

python reproduce.py --tier cached --list           # the 41 stages, C1 to C40 (with C30b), in order
python reproduce.py --tier cached --verify         # run them all and compare against the shipped files
```

Useful flags: `--only <id>` for one stage, `--from <id>` to resume. `python -m pytest tests` runs the unit tests;
those needing KITTI files, PyTorch or OpenCV skip themselves when those are absent.

Roughly four hours on the reference device, dominated by the bootstraps in C14, C19, C22, C23 and C35 and the
seed refits in C37.

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

C6, C14, C19, C36 and C37 fit models inside the stage; C6 fits the gradient-boosted routers with OpenMP on one thread
(Task 33). C36 and C37 first check that a refit reproduces the shipped scores bit for bit; they fit the gradient-boosted models with OpenMP on one thread, because the multithreaded fit varies in the last bits from run to run (up to 2e-15 on the reference platform). Both are byte-identical across runs on the reference platform; whether
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
| the cost of the camera-frame convention: the same settings, ego frame against camera frame | `results/final/lift_offset_sensitivity.csv`, `docs/iclr_lift_offset_sensitivity.md` | C27 (per-frame tables: N7) |
| the monocular lift in the ego frame: every registered quantity old beside new, the gates and the reading | `results/final/ego_frame_convention*.csv/json`, `docs/iclr_ego_frame_convention.md` | C28 (transform table: N8; gates: N9; C25's pre-fix tables: N10) |
| the published routing objectives (Qiu et al.'s ORIC, Geng et al.'s ΔAP) on these routers | `results/final/published_objective_*.csv/json`, `docs/iclr_published_objectives.md` | C29–C30 (detections: N11; labels: N12; R2 refits: N13) |
| the cost registry: detector costs, energy convention, timing boundaries, allocator overheads, versioned | `results/final/cost_registry.json`, `docs/iclr_submission_sync.md` | C30b |
| the submission path: score your own allocator, on the tracks the paper reports | `results/final/benchmark_decision_values.csv.gz`, `benchmark_bootstrap_plans.json`, `submission_path_g1.csv`, `docs/SUBMITTING.md`, `docs/iclr_submission_path.md`, `docs/iclr_submission_sync.md` | C31–C33 (inputs export: N14; the example's cost profile: N15) |
| the evidence pack and the claims check | `results/final/paper_evidence_pack.csv`, `claims_check.csv`, `docs/iclr_evidence_pack.md` | C34 |
| four analyses on cached scores: selection by objective, benefit and harm, overhead tolerance, gap accounting | `results/final/cached_analyses_*.csv`, `docs/iclr_cached_analyses_prereg.md`, `docs/iclr_cached_analyses.md` | C35 |
| target transform against objective for the R1 regression routers, and training-seed variation (post hoc) | `results/final/target_transform_*.csv`, `seed_variation.csv`, `figures/target_transform_heatmap.pdf`, `docs/iclr_target_transform.md` | C36–C37 |
| decision values with a shared action history: the change, its effect under own, shared and memoryless history, and every moved quantity old beside new | `results/final/shared_history_effect.csv`, `docs/iclr_shared_history.md` | full tier (G2b); the regenerated tables: every stage |
| one significance table: every deployable signal, cell and quota, paired interval of realised gain against random | `results/final/significance_table.csv`, `docs/significance_table.md` | C38 |
| harm and benefit by loss term | `results/final/loss_term_shares.csv`, `docs/iclr_shared_history.md` | C39 |
| every downstream loss as the code computes it: terms, weights, thresholds | `docs/iclr_loss_definitions.md` | — |
| sensitivity of the sign variation to the loss weights and the corridor (post hoc) | `results/final/loss_sensitivity.csv`, `docs/iclr_loss_sensitivity.md`, `docs/loss_sensitivity_tables.md` | C40 (rebuilt variants: P2) |
| Figure 1's two frames, exported by name | `results/final/fig_gallery/figure1*` | full tier (L1) |
| measured latency and energy budgets | `results/final/benchmark_budget_two_level*.csv`, `benchmark_budget_routers.csv` | full tier (F3, H4) |
| figure data (BEV objects, gallery, budget curves) | `results/final/fig_*` | full tier (L1); budget curves also C20 |
| exact formulas (monocular lifting, reference geometry, controllers, perception gains, ego speed) | `docs/iclr_formulas.md` | — |
| target-swap audit (writes no compared output) | `results/raw/20260925_162632_target_swap_audit_ego/`; run `scripts/129_target_swap_audit.py` after C19 | — |

Reports that interpret these tables: `docs/iclr_budget_feasibility.md`, `iclr_calibration.md`,
`iclr_causal_ego_speed.md`, `iclr_causal_threshold.md`, `iclr_class_error_fix.md`, `iclr_consumer_transfer.md`,
`iclr_ego_frame_convention.md`, `iclr_energy_conventions.md`, `iclr_formulas.md`, `iclr_idm_route_fix.md`,
`iclr_lift_offset_sensitivity.md`,
`iclr_nuplan_real_allocation.md`, `iclr_nuplan_real_perception.md`, `iclr_objective_swap.md`,
`iclr_phase0g_external_planners.md`, `iclr_realism_controls.md`, `iclr_routers.md`, `iclr_skip_accounting.md`,
`iclr_statistics.md`, `iclr_streaming_controllers.md`, `iclr_target_swap.md`, `iclr_submission_sync.md`,
`iclr_cached_analyses.md`, `iclr_target_transform.md`, `iclr_shared_history.md`, `iclr_loss_definitions.md`,
`iclr_loss_sensitivity.md`. The rest of `docs/` is generated tables (`*_tables.md`, `significance_table.md`,
`gate_spec.md`).

**Paper figures.** `scripts/paper_figures/` holds the scripts that draw figures from release artifacts (usage in its
README): the qualitative gallery (`render_gallery.py`, 16 JPEGs) and Figure 1 (`render_fig1.py`) from
`results/final/fig_gallery/`, and TikZ or data files for the gap, bird's-eye-view and target-transform figures
(`make_fig_gap.py`, `make_fig_bev_data.py`, `make_fig_transform.py`).

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

### Task numbers in the reports

The reports and script headers refer to the development tasks by number, which is also how their pre-registrations
are indexed. The pre-registration records themselves are not part of this release; each report says what was
committed before which result.

| task | what | report |
|---|---|---|
| 1 | per-mode operating points | `iclr_calibration.md` |
| 2 | detection-list and pixel routers | `iclr_routers.md` |
| 3, 5 | nuPlan with real perception: feasibility, then the run | `iclr_nuplan_real_perception.md` |
| 6 | figure data export | — |
| 7 | nuPlan allocation on real-perception decision values | `iclr_nuplan_real_allocation.md` |
| 9 | target swap | `iclr_target_swap.md` |
| 11 | causal streaming allocation, frozen threshold | `iclr_causal_threshold.md` |
| 12 | statistics hardening | `iclr_statistics.md` |
| 13 | consumer transfer (Part A), objective swap (Part D) | `iclr_consumer_transfer.md`, `iclr_objective_swap.md` |
| 16 | skipping cost accounting | `iclr_skip_accounting.md` |
| 19 | infeasible budgets (A), energy conventions (B), exact formulas (C) | `iclr_budget_feasibility.md`, `iclr_energy_conventions.md`, `iclr_formulas.md` |
| 20 | streaming rate controllers | `iclr_streaming_controllers.md` |
| 21 | realism controls | `iclr_realism_controls.md` |
| 22 | class-error fix (A), causal ego speed (B), lift-offset sensitivity (C) | `iclr_class_error_fix.md`, `iclr_causal_ego_speed.md`, `iclr_lift_offset_sensitivity.md` |
| 23 | the monocular lift in the ego frame | `iclr_ego_frame_convention.md` |
| 24 | published routing objectives | `iclr_published_objectives.md` |
| 25, 28 | the submission path, then on the tracks the paper reports | `iclr_submission_path.md`, `iclr_submission_sync.md` |
| 26 | evidence pack and claims check | `iclr_evidence_pack.md` |
| 29 | four analyses on cached scores | `iclr_cached_analyses.md` |
| 30 | target transform and training-seed variation (post hoc) | `iclr_target_transform.md` |
| 32 | decision values with a shared action history; significance table, harm by loss term, loss definitions | `iclr_shared_history.md`, `iclr_loss_definitions.md` |
| 33 | deterministic R1 router fits | `iclr_shared_history.md` |
| 34 | the records that read the regenerated tables in full, on the shared history | `iclr_shared_history.md` |
| 35 | loss-weight and corridor sensitivity (post hoc); Figure 1's frames | `iclr_loss_sensitivity.md` |

## Full reproduction from raw data

1. **Datasets.** Download nuScenes (v1.0-trainval metadata, blob 01, map expansion), KITTI tracking and nuPlan
   (v1.1 mini split, maps) and arrange them as `data/README.md` describes, then run `python data/verify_datasets.py`.
2. **Weights.** `bash data/download_models.sh` fetches the public Ultralytics YOLOv8s and RT-DETR-l releases.
3. **Prior-work code.** `bash environment/setup_third_party.sh` clones it at the pinned commits.
4. **Environments.** Python 3.8 with NVIDIA's Jetson PyTorch and JetPack's TensorRT
   (`environment/requirements-edge.txt`); the nuPlan simulation environment comes from
   `bash environment/setup_nuplan_env.sh`.
5. **Run.** `python reproduce.py --tier full --list` prints all 64 stages in order; `--tier full` runs them, and
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
| the monocular lift reported camera-frame geometry, which every controller and planner read as ego-frame | every table built on the lift, all regenerated in the ego frame; two registered readings reverse and are restated; `--frame camera` selects the old convention | `docs/iclr_ego_frame_convention.md` |

Reports written before the ego-frame correction say so in a note under their title; their figures are superseded by
the ego-frame files in `results/final/`, and `docs/iclr_ego_frame_convention.md` gives every registered quantity
old beside new. The ego-frame range estimate carries a near-range bias of about
+1.2 m (0–15 m), which the camera frame had cancelled; it is reported there, not corrected.

## Submitting an allocator of your own

`docs/SUBMITTING.md` has the schema, the rules and the two budget tracks. In short:

```bash
python examples/random_allocator.py          # writes a submission and scores it
python evaluate_submission.py my_scores.csv [--cost_profile my_profile.json]
```

`rap.submission.inputs(track)` returns exactly what an allocator may read at test time, labels-free
(`data/submission_inputs/`), and `scripts/151_profile_allocator.py` measures an allocator's own per-input cost for the
measured-budget track, on the energy rails of the cost registry (`results/final/cost_registry.json`, stage C30b),
whose version every scored row carries. nuPlan is the real-perception track. Gate G1' (stage C32) checks that this
path reproduces every row of the tables the paper uses (selection, latency and energy) exactly, with no
tolerance: `results/final/submission_path_g1.csv`, `docs/iclr_submission_sync.md`.

## Third-party code and data

* **Code.** nuplan-devkit, tuPlan Garage (PDM-Closed), PKL (planning-centric-metrics) and TIP are cloned at the
  commits in `third_party/PROVENANCE.md` and used unmodified, except for the deviations listed there.
* **Detector weights.** The public Ultralytics YOLOv8s and RT-DETR-l releases.
* **Dataset-derived files.** Caches, result tables, overlays and gallery images remain under their datasets'
  licenses: nuScenes and nuPlan CC BY-NC-SA 4.0, KITTI CC BY-NC-SA 3.0.

## License

The code is released under the MIT license; see `LICENSE`.
