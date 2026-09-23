# The nuScenes class-error mapping: the defect, the fix, and every figure it moves

> **Pre-Task-23 run.** The numbers in this document come from the run before the ego-frame convention (Task 23, 2026-09-22) and are kept as the record of that run. `results/final/` now holds the ego-frame run; where the two differ, `results/final/` is current (Task 28 audit, `scripts/157_doc_staleness.py`).

> **Frame convention.** The figures quoted in this report were computed in the camera frame, before Task 23. The shipped `results/final/class_error_fix.csv` re-expresses the same comparison on the ego-frame tables: the pre-fix class labels substituted into them (`scripts/146_class_error_prefix_tables.py`), with checks C1–C4 passing. See `docs/iclr_ego_frame_convention.md`.

Pre-registration: the pre-registration record (not part of this release), 2026-09-20 08:55 (Task 22 Part A), committed before the fix was written,
with two corrections logged at 09:05 before any figure was read. Every number below is a row of
`results/final/class_error_fix.csv`, old value beside new, so `reproduce.py --verify` covers the report itself.

## The defect

The reference class of an object was `TYPE_TO_COARSE.get(type, "vehicle")`, a lookup table written for KITTI's
type names (`Car`, `Pedestrian`, `Cyclist`, …). nuScenes geometry stored `type` as the **category prefix** —
`vehicle` or `human` — so the lookup missed on every nuScenes object and the default applied: every reference
object was labelled `vehicle`. A correctly detected pedestrian, bicycle or motorcycle therefore counted as a
**class error**.

It reaches exactly one primitive, `cls`, and through it the perception losses whose weights include it: E3
(class-aware) and the three E5 variants. E1, E2, E4, E6, the false-negative count, the matching itself (which is
class-agnostic) and every KITTI and nuPlan quantity are untouched. In the nuScenes decision table, 59 of 65
columns come back byte-identical; the six that move are `cheap_cls`, `full_cls` and the four affected gains.

How large the defect was, on the 3,376-frame nuScenes table:

| quantity | before | after |
|---|---|---|
| mean class errors per frame, CHEAP | 0.345 | 0.016 |
| mean class errors per frame, FULL | 0.733 | 0.029 |
| frames whose `cheap_cls` changes | — | 557 / 3,376 |
| frames whose `full_cls` changes | — | 991 / 3,376 |
| mean `dE_E5_combined` | −0.051 | **+0.136** |
| mean `dE_E3_class_aware` | +0.417 | +0.791 |

About 95% of the class errors the pipeline counted on nuScenes were an artefact of the mapping, and they were
counted more often against FULL than against CHEAP — which is why the mean combined gain changed sign.

## The fix

* `GEOM_FIELDS` gains a `coarse` field, filled where the geometry is built: KITTI keeps `TYPE_TO_COARSE[type]`;
  nuScenes maps the **full** category, which `rap/nusc.py` now keeps — `human.*` → person, `vehicle.bicycle` and
  `vehicle.motorcycle` → cyclist, every other `vehicle.*` → vehicle.
* `rap.geometry.coarse_classes(geom)` reads that field, and the three consumers
  (`50_percep_metrics.py`, `100_calibration_outcomes.py`, `rap/tables.py`) stopped mapping `type` themselves.
* The detector's own coarse classes are unchanged, and so is the class-agnostic matching.

`136_class_error_fix.py` then recomputes `cls` for every nuScenes mode and threshold the shipped tables use and
writes **corrected copies** of the per-frame tables to new run directories. The shipped runs are not overwritten.

## Checks, run before any figure was read

| check | what it asserts | result |
|---|---|---|
| C1 | the recomputation reproduces the shipped `fn`, `fp`, `loc`, `crit_fn`, `n_det`, `n_gt` exactly, at every nuScenes mode and threshold it touches | 72 / 72 |
| C2 | recombining the shipped primitives with the shipped weights reproduces `dE_E1`, `dE_E2`, `dE_E4`, `dE_E6` exactly | 4 / 4 |
| C3 | the unchanged generator path (`100`'s `run_task`, both geometries at S0), run with the fixed code, reproduces the corrected outcome file exactly | 2 / 2 |
| C4 | every KITTI and nuPlan row of every regenerated official file is byte-identical to the shipped one | 11 / 11 files, 0 rows moved outside nuScenes |

C1 failed on the first attempt and the cause was in the checking script: outcome files are named with the
threshold rounded to six decimals, while the S1 and S3 operating points come off a bisection (nuScenes S1 FULL is
`0.498046875`), so the recomputation was filtering at a slightly different threshold. The exact values are now read
from the run's own `outcome_index.csv`. No figure was read from that run and its outputs were deleted.

## OLD → NEW

### 1. The E5 row of the gain-vs-decision table (oracle geometry, S0, frames where the gain and V are both non-zero)

| cell | statistic | old | new |
|---|---|---|---|
| nuScenes oracle q_brake | sign disagreement | 38.4% | **35.6%** |
| nuScenes oracle q_brake | Spearman ρ with V | +0.093 | +0.099 |
| nuScenes oracle q_brake | P(V<0 \| gain>0) among affected | 32.4% | 32.1% |
| nuScenes oracle q_brake | frames plotted | 289 | 289 |
| nuScenes oracle q_plan | sign disagreement | 48.6% | **49.3%** |
| nuScenes oracle q_plan | Spearman ρ with V | +0.019 | +0.019 |
| nuScenes oracle q_plan | P(V<0 \| gain>0) among affected | 49.9% | 50.6% |
| nuScenes oracle q_plan | frames plotted | 1,212 | 1,212 |

### 2. Sign-disagreement range across the four gains (nuScenes, oracle geometry)

| cell | scheme | old | new |
|---|---|---|---|
| q_brake | S0 | 34.3 – 42.6% | 34.3 – 42.6% |
| q_plan | S0 | 48.6 – 51.7% | 48.9 – 51.7% |
| q_brake | per-mode S1–S4 | 29.2 – 44.8% | 29.2 – 44.8% |
| q_plan | per-mode S1–S4 | 49.2 – 52.5% | 49.1 – 52.5% |

The E5 column is no longer the minimum on the planner at S0, which is the only reason the q_plan range moves at
all. The ranges the paper quotes — 34–43% on braking and 49–52% on the planner at S0, 29–45% and 49–53% under the
per-mode operating points — are the same to the digit reported.

### 3. The scatter, and the overall share

| quantity | old | new |
|---|---|---|
| q_brake scatter: frames, share with opposite signs | 289, 38% | 289, **36%** |
| q_plan scatter: frames, share with opposite signs | 1,212, 49% | 1,212, **49%** |
| overall, both consumers × four gains, S0 | 34 – 52% | 34 – 52% |

### 4. The target swap (C19), pooled test, primary G = `dE_E5_combined`

| architecture | old mean V − G [95% CI] | new mean V − G [95% CI] |
|---|---|---|
| gate_ridge | +0.118 [−0.080, +0.275] | +0.123 [−0.093, +0.271] |
| gate_gbm | +0.106 [−0.058, +0.217] | +0.085 [−0.090, +0.213] |
| R1_mlp_reg | +0.040 [−0.083, +0.141] | +0.024 [−0.090, +0.132] |
| R1_mlp_clf | +0.117 [−0.037, +0.197] | +0.142 [−0.023, +0.235] |
| R1_gbm_reg | −0.069 [−0.186, +0.131] | −0.090 [−0.197, +0.085] |
| R1_gbm_clf | +0.013 [−0.070, +0.164] | **−0.014** [−0.098, +0.126] |

| statistic | old | new |
|---|---|---|
| architectures whose CI includes zero, under all three G labels | 6 / 6, 6 / 6, 6 / 6 | 6 / 6, 6 / 6, 6 / 6 |
| architectures leaning to V (primary G) | 5 of 6 | **4 of 6** |
| architectures leaning to G, named (primary G) | R1_gbm_reg | **R1_gbm_reg, R1_gbm_clf** |
| cells where V wins significantly (primary G) | 8 / 72 | 8 / 72 |
| cells where G wins significantly (primary G) | 4 / 72 | **5 / 72** |
| mean V − G over the 10 core cells, gate_ridge | +0.000 | **+0.007** |
| mean V − G over the 10 core cells, gate_gbm | +0.014 | **−0.011** |
| registered reading | architecture-driven | architecture-driven |

The two secondary labels (`dE_exact`, `dE_E6_risk_weighted`) are unchanged to the last digit, which is the expected
result and a check in its own right: 122 reused its saved G-target scores for 36 of 42 cell × label pairs and refit
only the six nuScenes cells whose primary label moved.

### 5. The multi-metric diagnostic on the four nuScenes rows of the robustness table

| row | multi-metric nDG@20%, old | new |
|---|---|---|
| nuScenes mono, longitudinal | +0.151 | **+0.309** |
| nuScenes mono, lateral | +0.111 | +0.121 |
| nuScenes oracle, longitudinal | +0.434 | **+0.492** |
| nuScenes oracle, lateral | −0.054 | **+0.012** |

The exact perception oracle (`dE`), the Spearman correlation with ΔJ and the identity of the best single metric are
unchanged in all four rows; the best single metric's own nDG moves only where that metric is an E5 variant
(mono lateral, E5_loc_heavy: +0.023 → +0.144; oracle lateral: +0.309 → +0.327).

## What else moved, and what did not

* `benchmark_table.csv`: 192 of 828 nuScenes rows. **Only** the four affected signals moved —
  `dE_E3_class_aware`, `dE_E5_combined`, `dE_E5_fp_heavy`, `dE_E5_loc_heavy`. `random`, `uncertainty`, both
  criticality signals, `dE_exact`, E1, E2, E4, E6, PKL, TIP, both gates and the oracle are identical.
  The largest move is nuScenes mono `plan_ade`, where the corrected E3 goes from +0.051 to +0.407 and E5_combined
  from +0.038 to +0.278 — both still with 95% intervals that contain zero, and still below `dE_E6_risk_weighted`.
* `calibration_cells.csv`: all 40 nuScenes rows (each carries the four gain columns).
* In a tree that also carries Task 22 Part B (the causal ego speed), this stage reports 246 of 1,320 nuScenes rows
  changed in `statistics_hardening.csv` rather than 216: the shipped file then carries both corrections, and the
  ego-speed change moves 30 further rows. Nothing outside nuScenes moves either way, and no figure of this report
  is affected.
* `statistics_hardening.csv` 216 / 1,320 nuScenes rows, `benchmark_target_swap.csv` 186 / 474,
  `phase0f_planning_metric_eta.csv` 14 / 80, `core_matrix.csv` 4 / 4.
* Byte-identical, nuScenes included: `benchmark_cells.csv`, `benchmark_self_agreement.csv`,
  `calibration_thresholds.csv`, `calibration_sweep.csv`, `calibration_brake_vs_plan.csv`. Harm rate, ρ, the
  operating points and the sweep are functions of J alone, so the class error never touched them.
* `calibration_pr_curves.csv` is not regenerated and does not move: its inputs are `fn`, `fp`, `n_det` and `n_gt`,
  which C1 proves unchanged at every threshold.
* Two Phase 0E by-products, `headline_table.csv` and `statistical_tests.csv`, do not move either — the latter's
  four policies are `dE`, uncertainty, criticality and ego speed. Neither is part of this release: no stage reads
  them and they had already drifted from `core_matrix.csv` before any of this work.
* The Task 9 **audit** (`129_target_swap_audit.py`) was re-run against the regenerated table, because its exploratory
  statistics quote the primary G label. It still reproduces 122's pooled CIs exactly and still finds no computation
  error (nDG identical in all 84 rows, 252 / 252 recomputed nDG values equal). One exploratory number changes in a
  way worth stating: restricted to the 618 bootstrap draws in which all 12 cells survive the prize rule, the two
  gate intervals used to sit just above zero (+0.005 to +0.284 and +0.009 to +0.231) and now contain it
  (−0.009 to +0.294 and −0.006 to +0.231). The one place where the null reading was marginal is no longer marginal.

## What this changes for the paper

**Nothing in the reading, and one number in each of three tables.** The defect inflated the class-error term on
nuScenes by about a factor of twenty, and it inflated it more on FULL than on CHEAP, so the combined E5 gain had the
wrong mean sign. After the correction:

* better perception by the combined measure still disagrees with the decision on **36% of affected braking frames
  and 49% of affected planner frames**, against 38% and 49% before;
* the Spearman correlations stay within ±0.10 on braking and ±0.02 on the planner;
* the target swap's registered reading stays **architecture-driven**, with every pooled interval still containing
  zero under all three labels; the lean of one architecture (R1_gbm_clf) flips from V to G, which is exactly the
  kind of movement the "architecture-driven" reading predicts;
* the multi-metric diagnostic rises in all four nuScenes rows and is no longer negative in any of them. It remains
  a diagnostic: it is fit on the perception gains with leave-one-scene-out and is not deployable.

## Runs and code

| artefact | where it is in this release |
|---|---|
| the corrected per-frame tables | in place in `20260912_071140_core_matrix` and `20260913_133004_core_matrix_postreview`; each run carries a `CLASS_ERROR_FIX.md` saying so |
| the corrected per-mode outcomes (32 nuScenes files) | in place in `20260914_103252_calibration_outcomes` |
| the corrected planning metric, both geometries | in place in `20260913_211441_phase0g_eta_fde_oracle` and `20260913_214436_phase0g_eta_fde_mono` |
| C1–C3, and the pre-fix copy of every file this part changed or checked | `20260920_090509_class_error_fix` (`checks.csv`, `before/`: the 11 files C4 compares, the target-swap summary, and `calibration_tables.md`, the one generated table that moved) |
| figures and C4 | `results/final/class_error_fix.csv`; stage **C25** recomputes both |

This release ships the corrected tables in the run directories that first produced them, rather than a second copy
of every table: they are the tables the reported numbers come from, and each directory says so. Their pre-fix values
are preserved as data, in `results/raw/20260920_090509_class_error_fix/before/`, which is what stage C25 reads.

Code: `scripts/136_class_error_fix.py` (corrected tables and C1–C3, stage **N5**),
`scripts/137_class_error_figures.py` (the table above and C4, stage **C25**), `scripts/52_core_matrix.py --rows_only`,
`scripts/122_target_swap.py --reuse_gscores`. Every run used `PYTHONHASHSEED=0`.
