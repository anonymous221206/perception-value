# Submitting an allocator

This page is for scoring **your own** allocator on the benchmark, on the same frozen test split and with the same
scorer as every signal in `results/final/benchmark_table.csv`. You need a CPU and the cached tier's environment
(`environment/requirements-cached.txt`); no dataset and no GPU.

```bash
python examples/random_allocator.py            # writes examples/submissions/random.csv and scores it
python examples/confidence_heuristic.py        # the same for a heuristic that reads the CHEAP detections
python evaluate_submission.py my_allocator.csv [--cost_profile my_profile.json]
```

## What an allocator does here

Each **cell** is a pair of perception modes, CHEAP and FULL, feeding one downstream decision system, with one decision
loss (the columns `track, geometry, system, target` of `benchmark_table.csv`; 14 cells: KITTI 4, nuScenes 6,
nuPlan 4). For every input, the decision value `V = J(CHEAP) - J(FULL)` is how much loss escalating that input to FULL
saves. An allocator scores every input **before** escalation; the benchmark escalates the top-scored inputs up to a
budget and reports the **nDG**: the share of the budget-constrained oracle's saving that the allocator realises.

## The rules

**1. The split is frozen.** `configs/benchmark_splits.json` fixes, with its generating rule, which units (KITTI and
nuScenes sequences / scenes, nuPlan logs) are train, validation and test. A submission is scored on test units only.
No quantity computed from a test unit may enter anything you fit or choose: model weights, normalisation statistics,
feature selection, hyperparameters, early stopping, thresholds.

**2. At test time, only pre-escalation information.** On a test input an allocator may read the input itself, the
CHEAP output for it, earlier frames of the same unit and their CHEAP outputs, the calibration and the ego vehicle's own
state -- nothing produced by FULL, no reference object, no decision value, no future frame. In the code these are the
five provenance labels of `rap.features.LEGAL_SOURCES` (`cheap_det`, `cheap_image`, `cheap_prev`, `calib`,
`ego_state`); every feature of the benchmark's own gates is registered with one of them, and `assert_no_leakage`
rejects any column that is unregistered or carries another source (`tests/test_no_leakage.py`, `docs/gate_spec.md`).
On **training** units anything is allowed, including FULL outputs, reference objects and decision values.

**3. How to check your own features against rule 2.** Two steps, both in `examples/confidence_heuristic.py`:

* compute every test-time quantity through a function `score_input(inp)` that receives one input as
  `rap.submission.inputs` exports it (below). Those files hold only the five legal sources, so what `score_input`
  computes can carry nothing else -- the check is structural, not a declaration. The profiling harness calls the same
  function, so the code that is timed is the code that is scored;
* declare what each of your features is computed from and pass the declaration to
  `rap.features.check_provenance({"my_feature": "cheap_det", ...})`, which applies the rule `assert_no_leakage`
  applies to ours. Like our registry, this checks a declaration; it cannot see what your code actually read.

**4. The pre-escalation rule is honour-based.** The shipped tables contain the decision value of **every** input,
test inputs included: `results/final/benchmark_decision_values.csv.gz` (which the scorer needs) and the tables of
`results/final/`. Nothing in this repository can stop an allocator from reading them, or the reference objects of the
public datasets, at test time. The evaluator checks the form of a submission, not where its scores came from. What
makes a result checkable is the code: publish the allocator with the result, so that anyone can re-run `score_input`
on `data/submission_inputs/` and obtain the same file.

## What a submission may read: `rap.submission.inputs`

```python
from rap import submission as S
kitti = S.inputs("KITTI")      # {"frames": DataFrame, "detections": DataFrame, "calibration": {unit: ...}}
nusc = S.inputs("nuScenes")    # the same
nuplan = S.inputs("nuPlan")    # {"states": DataFrame}
S.cells()                      # the 14 cells
S.rows(cell, "test")           # the inputs a submission must score in that cell (identifiers and unit, no labels)
S.decision_values()            # the labels, for fitting on training units (rule 1)
```

The files are in `data/submission_inputs/` (written by `scripts/150_submission_tables.py --stage inputs`; columns in
`MANIFEST.json`), for every unit of every split:

| file | one row per | columns |
|---|---|---|
| `kitti_frames.csv.gz`, `nuscenes_frames.csv.gz` | frame | `seq, frame, unit, split, prev_frame` (the previous CHEAP frame of the unit, -1 for the first), `n_detections`, ego speed (`ego_speed_mps`), the image reference, and the CHEAP detector's per-frame statistics (`frame_*`: candidate counts, image statistics) |
| `kitti_detections.csv.gz`, `nuscenes_detections.csv.gz` | CHEAP detection at confidence >= 0.10 | `seq, frame, det`, box `x1, y1, x2, y2` (pixels), `coarse` class, `conf, entropy, margin, binent`, and the monocular geometry in the ego frame (`geo_*`: range, lateral extent and height in m, time to collision in s) |
| `calibration.json` | unit | intrinsics and the camera-to-ego transform the monocular lift uses |
| `nuplan_states.csv.gz` | planner state | `scenario, iteration, unit, split`, the cheap-side gate features of the nuPlan cells and the CHEAP track list (up to 25 tracks, 10 values each) |

The operating threshold of the CHEAP detector is 0.25; detections between 0.10 and 0.25 are what it almost reported.
Images are not redistributed: an allocator that reads pixels opens the dataset image named in the frames file.

## The submission file

A CSV with one row per test input of each cell it enters. A submission may enter any subset of the 14 cells, but a
cell it enters must be covered exactly.

| column | |
|---|---|
| `track, geometry, system, target` | the cell, spelled as in `benchmark_table.csv` (nuPlan's geometry is the literal `n/a`) |
| `seq, frame` | KITTI and nuScenes: sequence / scene name, frame index (integer) |
| `scenario, iteration` | nuPlan: scenario token, iteration (integer) |
| `score` | **ranking form**: one score per input, used at every budget |
| `score_q10, score_q20, score_q30, score_q50` | **per-budget form**: one score column per budget level, for budget-conditioned allocators; each budget is scored on its own column |

No other column. Leave the other track's identifier columns empty (or omit them). The scored output records which
form was used (`form`: `ranking` or `per-budget`).

**Higher scores are escalated first.** Scores are rounded to 9 decimals before ranking, as the benchmark does.

**Ties and row order.** Inputs with equal scores are a tie, and every selection is evaluated in exact expectation over
a uniformly random tie-break: when the budget cuts through a tie group, each member counts with the fraction of the
group that fits. The order of the rows therefore changes nothing, and a submission cannot break ties in its favour by
sorting. A cell in which all scores are equal is the random allocator and is scored by the benchmark's closed form
for it (`k/n` of the cell's total value); its difference from random is zero and `p_le_random` is left empty.

**Validation.** Before anything is scored, the evaluator checks the columns, that the cell exists, that identifiers
have the right type, that scores are finite (there is no missing score; give a very low finite score instead), that
no input appears twice, and that the rows of each cell are exactly its test inputs -- no missing input, no training or
validation input, no unknown frame. Every error names the offending rows by their line in the file; nothing is scored
until the file is valid.

## How it is scored

`evaluate_submission.py` scores through `rap.scoring`, the scorer of `scripts/92_benchmark_table.py` moved into the
library unchanged; `scripts/150_submission_tables.py --stage g1` checks that it reproduces every official row exactly
(`results/final/submission_path_g1.csv`).

**Selection budget** (quota of inputs; any CPU). For each quota `q` in 10, 20, 30, 50 % the budget is
`k = round(q n)` inputs (at least one) of the cell's `n` test inputs.

* `ndg`: the allocator's expected saving over its top `k` inputs, divided by the saving of the `k` inputs with the
  largest decision value (the oracle's). Undefined (empty, `ndg_defined` false) when that oracle saving is not above
  1e-9 -- a cell where escalation saves nothing at that quota.
* `gain`: the realised saving in the cell's own loss units (summed over the escalated test inputs); `prize` is the
  oracle's, and `reduction_frac` the saving as a share of the cell's total CHEAP loss.
* `ndg_lo, ndg_hi` and `minus_random_lo, minus_random_hi`: 95 % percentile intervals of a paired bootstrap over
  **units** (1000 resamples of whole sequences / scenes / logs): in each resample the allocator and random are scored
  on the same inputs. A resample whose oracle saving falls below a quarter of the full-sample one is dropped
  (`boot_dropped`); an interval needs at least 50 surviving resamples.
* `interval_excludes_zero`: whether the paired interval of nDG minus random excludes zero. `p_le_random` is the share
  of resamples in which the allocator does not beat random.

The resamples are the ones the official table drew for that cell (`--plan`, default `benchmark_table`; also
`routers_r1`, `router_r2:nuScenes`, `router_r2:KITTI`), replayed from `results/final/benchmark_bootstrap_plans.json`,
so a submission and the official rows it is compared with see identical resamples.

**Measured budget** (latency and energy; needs the allocator's cost on the reference platform). CHEAP runs on every
input, the allocator's own cost `o` is charged on every input, and escalated inputs additionally run FULL. With the
detector costs `C_c, C_f` measured on the reference board (`benchmark_budget_overheads.json`), the budget at level `f`
is `b = C_c + f C_f` per input, in ms or in mJ, and the allocator can escalate a share `(b - C_c - o) / C_f` of the
inputs (`k = floor(share n)`); nDG divides its saving by the oracle's at the same level with no overhead, so an allocator's own cost shows up as escalations lost. An allocator whose overhead exceeds the headroom, `o > b - C_c`, cannot run within the
budget: the row is marked infeasible and no nDG is reported (`rap.budget.infeasible`, the rule every official budget
table applies). Everything else is scored as in the selection track, on the resamples of
`benchmark_budget_two_level.csv`.

The cost profile is a JSON file, passed with `--cost_profile`:

For example:

```json
{"ms": {"KITTI": 0.012, "nuScenes": 0.015}, "mJ": {"KITTI": 0.004, "nuScenes": 0.005},
 "route": "harness", "device_check": {"reference_platform": true, "stdout": "..."}}
```

Costs are per input, either one number or one per track; a track or unit that is absent or `null` is not scored under
measured budgets. The scored rows carry `cost_route` and `reference_platform`.

There are two accepted ways to obtain the profile, and no others:

1. **Run the provided harness on a device that matches the reference platform**: `python
   scripts/151_profile_allocator.py my_allocator.py --out my_profile.json`, where `my_allocator.py` defines the
   `score_input(inp)` of rule 3. The harness measures latency and energy exactly as the benchmark measured its own
   allocators, runs `environment/check_device.py`, and stores that check's output in the profile. Submit the profile
   together with that output; a profile whose check did not pass is not a measured-track result.
2. **Submit the allocator's code for us to profile** with the same harness on the reference board.

## Citing a result

Report, for every number: the cell (`track | geometry | system | target`), the quota or budget level, the nDG and its
paired interval against random (and whether it excludes zero), the track (selection budget, or measured budget with
its unit, ms or mJ, and cost route), and whether the allocator is budget-conditioned (the `form`: `ranking` or
`per-budget`). Give the `plan` if it is not `benchmark_table`, the submission's `submission_sha256` from the scored
file, and the repository commit. Compare with the official rows of the same plan -- `benchmark_table.csv` for the
default plan -- not with a different table's intervals.
