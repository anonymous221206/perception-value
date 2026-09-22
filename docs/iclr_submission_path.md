# The submission path (Task 25)

What was added so that a third party can score an allocator of their own, the evidence that it scores exactly as the
benchmark does, the cost-profile policy as implemented, and the places where writing the protocol down showed that it
was under-specified. No experiment was added and no reported number changed.

## The interface

| piece | where |
|---|---|
| CLI | `evaluate_submission.py SUBMISSION.csv [--cost_profile P.json] [--plan ...] [--cells ...]` |
| schema and rules | `docs/SUBMITTING.md`; enforced by `rap.submission.validate` |
| scorer | `rap.scoring`: `topk_expect`, `quota_k`, `evaluate`, `ci`, `QUOTAS`, `EPS`, moved verbatim out of `scripts/92_benchmark_table.py`, which imports them back (every `t92.*` name still resolves) |
| what the scorer needs | `results/final/benchmark_decision_values.csv.gz` (every cell's inputs, unit, split, `J_cheap`, `J_full`, `V`, in the order 92 scores them, built by 92's own cell builders) and `results/final/benchmark_bootstrap_plans.json` (below) -- `scripts/150_submission_tables.py --stage values` |
| what a submission may read | `data/submission_inputs/` and `rap.submission.inputs(track)`: CHEAP detections with their ego-frame monocular geometry, the previous CHEAP frame, calibration, ego speed; nuPlan's cheap-side gate features and CHEAP track list -- `150 --stage inputs` |
| profiling harness | `scripts/151_profile_allocator.py`: 93's latency and energy measurement applied to a third party's `score_input` |
| examples | `examples/random_allocator.py`, `examples/confidence_heuristic.py`, outputs in `examples/submissions/` |
| third-party feature check | `rap.features.check_provenance`, the rule of `assert_no_leakage` for declared features |

A submission is one row per test input of each cell it enters: the four cell columns, the track's identifiers, and
either `score` (one ranking, every quota) or `score_q10 ... score_q50` (budget-conditioned). The scored file records
the form. Scores are rounded to 9 decimals and ranked high to low; ties are evaluated in exact expectation, so row
order changes nothing; a cell whose scores are all equal is scored as the random allocator.

## G1: the submission path reproduces the official numbers

`scripts/150_submission_tables.py --stage g1` writes every official signal out as a submission file, scores it through
`rap.submission` (the library `evaluate_submission.py` calls) and compares every field of every row with the official
table. No tolerance: a field passes only if the two values are equal bit for bit (NaN matching NaN).

| what | signals | compared |
|---|---|---|
| `benchmark_table.csv`, test split | random, uncertainty, criticality_cheap, gate_ridge, gate_gbm | 16 fields, 4 quotas, 14 cells |
| `benchmark_table_routers.csv` | R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf (plan `routers_r1`), R2_cnn_clf (plan `router_r2:<dataset>`) | 13 fields, 4 quotas, 14 and 10 cells |
| `benchmark_budget_two_level.csv` (measured track) | random and gate_gbm, each charged the overhead 93 charged it | 10 fields, 2 units, 4 levels, 14 cells |

**Result: 178 signal × field comparisons, 10,152 compared rows, 0 differences** (`results/final/submission_path_g1.csv`).

**What G1 found on its first run.** Every row reproduced except R2 on nuScenes, where nine fields differed in the last
bits (≤ 1.1e-16). Replaying 107's own arithmetic on its saved TensorRT scores reproduced the same differences, so the
submission path was not the cause: the shipped nuScenes R2 values were exactly the computed ones read back once
through pandas' default float parser, which is not round-trip. The KITTI export had re-read the table that way when it
merged its own rows, before that read was made lossless in Task 23. They were regenerated from the saved scores
(`107 --stage rescore`: no GPU, no refit, and each shipped value checked value by value against both the recomputed
one and its lossy re-read); `benchmark_table_routers.csv` changed in exactly those 24 rows, in the last bits, and no
number changes at any printed precision. G1 then passed with no tolerance.

## G2: no official output changes

`reproduce.py --tier cached --verify` on a fresh clone, with the new stages added; recorded with the run in
the pre-registration record (not part of this release). The only official file that changed at all is `benchmark_table_routers.csv`, in the last bits of its
24 nuScenes R2 rows (G1 above) -- inside the verifier's tolerance and below any reported precision.

## The cost-profile policy, as implemented

A measured-track submission gives its per-input latency (ms) and energy (mJ), one number or one per track. The
scorer applies 93's two-level cascade unchanged: budget `b = C_c + f C_f` with the shipped detector costs, escalated
share `max((b - C_c - o) / C_f, 0)`, `k = floor(share n + 1e-9)`, nDG relative to the oracle at the same level with no
overhead, and the feasibility rule of `rap.budget.infeasible` (`o > b - C_c + 1e-9` marks the row infeasible and
blanks what it would have achieved). The bootstrap resamples are those of `benchmark_budget_two_level.csv`.

The profile must come from one of the two routes of `docs/SUBMITTING.md`: the harness run on a device that passes
`environment/check_device.py`, with that check's output stored in the profile, or code submitted for us to profile.
The harness is 93's measurement, reused rather than re-implemented (`_median_ms`, `_rails_during`, and the energy rule
of `signal_overhead`): the median single-input wall time over three passes of the first 400 test inputs per track,
inputs prepared before timing; energy is that latency times the power over idle of the named rails, sampled for 8 s
idle and 8 s busy every 50 ms, CPU by default as 93 charged its CPU workloads. The scored rows carry `cost_route` and
`reference_platform`; the scorer does not reject a profile from another device, it labels it.

## What the examples produce

`examples/random_allocator.py` writes a submission that gives every test input the same score and scores it: the
scorer treats it as the random allocator (an all-tied cell is scored by the benchmark's closed form), its nDG equals
`benchmark_table.csv`'s `random` row in all 14 cells at all four quotas, and its paired interval against random is
exactly zero. `examples/confidence_heuristic.py` scores the KITTI and nuScenes cells by the summed confidence of the
CHEAP detections just below the operating threshold, reading only `rap.submission.inputs`, and is scored with the cost
profile the harness measured for it on the reference board.

## Where the protocol was under-specified

1. **"Score through `rap.budget.evaluate`" names the wrong scorer.** `rap.budget.evaluate` is the Phase-0 scorer (64
   seeded random tie-breaks, no bootstrap). Every official table is scored by 92's `evaluate`: exact expectation over
   ties and a paired bootstrap over units. The submission path uses 92's, moved into `rap.scoring`.
2. **A cell's bootstrap resamples are not a property of the cell.** Each official script draws its resamples from one
   `default_rng(0)` in the order it visits cells: 92 scores split `test` and split `all` for every cell, and the S(M)
   sensitivity for the nuScenes planner cells, all from the same stream; 103 (R1), 107 (R2, one stream per dataset)
   and 93 (measured budgets) each have their own. The same scores therefore get different intervals in different
   tables. Exact reproduction needs the stream replayed up to the cell (`benchmark_bootstrap_plans.json`, `--plan`);
   a submission's interval is comparable only with the official rows of the same plan.
3. **The two budget tracks round the same nominal budget differently.** The quota track escalates
   `k = round(q n)` inputs (Python's round-half-to-even, at least one); the measured track `k = floor(f n + 1e-9)`.
   At the same nominal level with zero overhead they can differ by one input and in the oracle they divide by: on
   nuScenes (n = 955) at 10 %, 96 against 95 inputs, and random's nDG is 0.0741 in `benchmark_table.csv` and 0.0733 in
   `benchmark_budget_two_level.csv`. Nothing was changed; the submission path reproduces each table's own rule.
4. **Random has two numerical forms.** The benchmark scores random by its closed form `k/n sum V`; an all-tied
   ranking through the tie expectation gives the same number up to the last bits (a cumulative sum against a sum).
   The schema therefore defines an all-tied cell as the random allocator and scores it by the closed form.
5. **Missing scores.** The benchmark ranks a missing signal value at minus infinity; the schema requires finite scores
   (a missing score is a submission error, not a silent bottom rank). G1 writes the benchmark's minus infinity as
   `-1e300`, which gives the same order and the same ties.
6. **When nDG is undefined, and when a resample is dropped,** was written only in code: nDG is undefined where the
   oracle's saving at the quota is not above 1e-9, and a bootstrap resample is dropped when its oracle saving falls
   below a quarter of the full-sample one; an interval needs 50 surviving resamples. Both are now in the schema.
7. **"Exactly the test units" is really "exactly the test inputs".** A cell is scored on frames (nuPlan: planner
   states), resampled by unit. Coverage is therefore checked per input, and an input of a test unit that a cell does
   not score is an error, not a padding row.
8. **No generic profiling harness existed.** The benchmark's own overheads were measured inline in 93 for its own
   workloads, nuPlan's by a nuScenes proxy (its track features were never timed). The harness had to be written; it
   reuses 93's functions rather than restating them. Which rails an allocator is charged on is a choice the protocol
   does not make: 93 charges CPU for CPU workloads and CPU + GPU for the TensorRT router, while the detector costs it
   is compared with are GPU-rail energy (Task 19 Part B measured how much this convention matters). The harness
   defaults to CPU and records the rails used.
9. **The per-budget form under measured budgets.** A budget-conditioned allocator gives one column per nominal level;
   under measured budgets its own overhead lowers the share it can escalate. The scorer uses the column of the nominal
   level; an allocator that wants to condition on the effective share would need that share, which depends on its
   own measured cost. The spec does not say which is meant.
10. **What a submission may read is partly not recoverable.** The CHEAP detector's per-class score vector was never
    cached (only entropy, margin and binary-entropy summaries are), nuPlan's pre-escalation information exists only in
    derived form (the 18 legal gate features and the 25-track CHEAP list), and images are not redistributed. An
    allocator needing more must re-run CHEAP from the datasets.
11. **The pre-escalation rule cannot be enforced.** The decision value of every test input is shipped (the scorer
    needs it) and the datasets' reference objects are public. `docs/SUBMITTING.md` says so, and makes the rule
    checkable instead: test-time features computed only through `score_input` on the labels-free export, and code
    published with a result.
