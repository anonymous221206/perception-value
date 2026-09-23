# Task 29 pre-registration: four analyses on cached scores

Written and committed before any of the statistics below was computed. Nothing is refit: every score is read from a
cached run (below). No file that exists in `results/final/` today is changed; all outputs are new files. Every cell and
quota is reported, including nulls and reversals. "Descriptive" means no significance claim is attached.

## Common definitions

* **Cells.** The 14 held-out cells of the submission path (Task 28): KITTI 4 and nuScenes 6 ("core", 10), nuPlan 4
  (real-perception track). Test split and units as frozen in `configs/benchmark_splits.json`.
* **Scores (cached).** Official test scores: R1 `20260922_163001_routers_r1_ego`, R2 `20260922_102405_router_r2_ego`,
  core gates `20260922_110512_budget_gate_scores_ego`, nuPlan `20260915_033311_nuplan_real_allocation`; uncertainty,
  cheap-side criticality and ego speed (`v_ego_causal`, nuPlan `nr_ego_speed`) from the cells' own columns.
  **V1 scores** (gates and R1 fitted on the train units only; validation and test scores out of sample):
  `20260922_124603_streaming_v1_scores_ego` (130 `--fit_once`, 124's V1 definition). The pixel router R2 has no
  validation scores and is excluded wherever validation scores are needed.
* **Scorer.** `rap.scoring`: exact tie expectation (scores rounded to 9 decimals), selection capacity
  `k = quota_k(n) = max(round(q n), 1)`; measured capacity from `rap.submission` (share
  `max((b - C_c - o)/C_f, 0)`, `k = floor(share n + 1e-9)`, infeasible if `o > b - C_c + 1e-9`).
* **Costs.** Cost registry version 2026-09-23.1 (Task 28): detector ms and module-convention mJ per track; allocator
  overheads of the **routers** measurement (the only one that covers every deployable signal; nuPlan inherits the
  nuScenes feature time as a proxy, as the registry states).
* **Undefined.** An objective is defined on a split at a quota when its value vector has at least 10 inputs with
  `|value| > 1e-9` and its oracle prize at that quota is above 1e-9 (the benchmark's rule, applied to every
  objective).
* **Shares.** "Share of the all-cheap loss" divides a gain by `sum J_cheap` over the same inputs (in a bootstrap draw,
  over the draw's inputs).
* **Bootstrap.** 1,000 draws; units (scenes, sequences, logs) resampled once per dataset per draw and shared by every
  cell of that dataset (datasets in the order nuScenes, KITTI, nuPlan; `default_rng(0)` per part A-D); **every draw
  kept** (no prize filter; statistics are gains and shares, not ratios to a draw's prize); 95 % percentile intervals.
  Paired differences are taken inside each draw.

## A. Does a perception objective pick a worse allocator?

Objectives: `E_dec` (decision value V), `E_perc_dE` (`dE_E1_fn_only`), `E_perc_risk` (`dE_E6_risk_weighted`). Under
an objective, a signal's score on a split and quota is its nDG against that objective's value vector (127's `eta_of`).

Pools (fixed order, which is also the tie-break order for a winner):

* **P1** target-free legal: random, uncertainty, criticality_cheap, trivial_ego_speed (where the cell has them).
* **P2** P1 + gate_ridge, gate_gbm, R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf. Sensitivity only: its members
  are trained on V and so advantaged under `E_dec`.
* **old** 127's pool (tiers 1-3: P2 + R2 + the perception diagnostics dE_exact, dE_E1_fn_only, dE_E6_risk_weighted,
  PKL, TIP), step A2 only, for continuity.

**A2, descriptive test-set disagreement.** Population: `objective_swap.csv`'s 52 cell-quota pairs (13 cells whose test
`E_dec` is defined x 4 quotas; nuPlan IDM safety is excluded there). Official test scores. Only signals with a defined
(finite) nDG under both objectives compared. The optimal set under an objective = signals within 1e-9 of the best nDG.
"No shared optimum" = the two optimal sets do not intersect. Counts per pool (P1, P2, old) and perception objective,
over the 52 pairs and separately over the 10 core cells x 4 quotas; plus the Kendall tau of the two rankings.

**A3, selection on validation, evaluation on test.** For each cell, quota, pool (P1, P2) and perception objective:

1. On the validation units (k from `quota_k(n_val)`), pick the `E_dec` winner and the perception winner (highest
   nDG; ties within 1e-9 to the earlier signal in pool order). Learned signals use their V1 validation scores; fixed
   signals need no fitting.
2. Evaluate both winners once on test by realised decision gain at `quota_k(n_test)`, V1 test scores for learned
   signals. Statistic: `D = G_test(perception winner) - G_test(decision winner)`, in loss units and as a share of the
   all-cheap loss; `D = 0` exactly when the two winners coincide. Negative D = the perception objective chose worse.
3. A (cell, quota) pair is dropped, and listed with the reason, if either objective is undefined on the validation
   units or `E_dec` is undefined on the test units. Expected from the validation counts: every nuPlan cell except
   PDM-Closed scalar_J (validation has 4, 0 and 7 affected states in PDM-Closed safety, IDM safety and IDM scalar_J).
4. Per pair: D and its 95 % interval. **Pooled**, per pool, objective and quota: the mean of the share-scale D over
   the surviving cells, with its interval from the shared draws.

**Reading rule (registered).** Primary: pool P1, each perception objective, quota 20 %. "A perception objective
selects allocators with lower decision value" is **supported** for an objective only if the pooled 95 % interval of D
at 20 % (P1) lies entirely below zero. Otherwise the A2 disagreement is reported as descriptive and the cost as
**unresolved**. The other quotas and P2 are reported beside it and do not change the reading.

## B. Benefit captured and harm incurred

Signals: random and every deployable signal with official test scores (uncertainty, criticality_cheap,
trivial_ego_speed, gate_ridge, gate_gbm, R1 x 4, R2 where it exists); all 14 cells; selection quotas
10/20/30/50 %. With `a_i` the tie-expectation inclusion probabilities of the top-k selection:
`B = sum a_i max(V_i,0)`, `H = sum a_i max(-V_i,0)`, `G = B - H`, `B_all = sum max(V_i,0)`, `O(k)` the oracle gain.
Computed as `topk_expect` of the score against the value vectors `max(V,0)` and `max(-V,0)` (linear, so exact under
ties). Report B, H, missed benefit `B_all - B`, budget-forced `B_all - O(k)` and G, all as shares of the all-cheap
loss, with intervals and paired differences to random. Assert on every row, point and draw:
`O(k) - G = (B_all - B) + H - (B_all - O(k))` to 1e-9. Plot data: core cells at 20 %, B against H per signal per cell,
and the mean over the 10 core cells. Descriptive.

## C. How much overhead can a ranking afford?

Signals: the deployable rankings the registry's routers measurement charges (uncertainty, criticality_cheap,
gate_ridge, gate_gbm, gate_gbm_batched, R1 x 4, R2 on core cells); all 14 cells; budgets `b = C_c + alpha C_f` for
alpha in {10, 20, 30, 50} %, in ms and in module-convention mJ.

* Realised gain on the whole capacity grid k = 0..n (exact tie expectation, official test scores). An overhead o
  gets the capacity `rap.submission`'s rule assigns to `b - C_c - o`; o ranges over `[0, b - C_c]` (beyond that the
  allocator is infeasible).
* References: **random** at zero overhead, `k0/n sum V` with `k0` the capacity at o = 0; **uniform full-fidelity
  inference**, i.e. running FULL instead of CHEAP on every input (gain `sum V`, no allocator), feasible only when
  `C_f <= b`.
* Report, per cell, signal, unit and alpha, the set of o for which the point-estimate gain is at least random's, and
  (where feasible) at least full-fidelity's, as a list of closed intervals in o (several if the gain is not monotone in
  capacity); no interpolation between budget levels. Mark the measured overhead and whether it lies in each set.
  Descriptive.
* At the measured overhead: the paired interval against random, computed through `rap.submission.budget_cell`, must
  equal the shipped rows exactly (`benchmark_budget_routers.csv` and `benchmark_budget_nuplan_real.csv` for ms;
  `energy_module_budget_two_level.csv` table `routers` and `energy_module_budget_nuplan_real.csv`, module convention,
  for mJ). A mismatch stops the analysis.

## D. Where the gap between the zero-cost oracle and a causal allocator goes

Cells: the 10 core cells and nuPlan PDM-Closed safety and scalar_J. Signals: gate_ridge, gate_gbm, R1_mlp_reg,
R1_mlp_clf, R1_gbm_reg, R1_gbm_clf, all with their **V1 scores** (one model fit throughout). Latency budgets at
alpha = 20 % and 50 %; overhead o = the signal's registry overhead (routers measurement, ms). With `k0` the capacity at
o = 0 and `k_o` at the measured overhead (share `f_o = k_o`'s share from the rule):

* `O_0` oracle gain at `k0`; `O_o` oracle gain at `k_o`;
* `G_rank` top-k gain of the V1 test scores at `k_o`;
* `G_stream` policy B of `scripts/130_streaming_controllers.py` (`run_B`: at most `floor(1 + f_o t)` escalations after
  t inputs of a unit, in 124's timestamp order) at rate `f_o`, threshold and tie probability calibrated on the V1
  **validation** scores at rate `f_o` (130's `Calib`), ties at the threshold broken by 130's 32 stable key sets and
  averaged.

Report `O_0 - G_stream = (O_0 - O_o) + (O_o - G_rank) + (G_rank - G_stream)` per cell, signal and budget, as shares of
the all-cheap loss, with intervals. The first two terms are non-negative by construction; the third may be negative;
the three are an accounting of this protocol, not losses and not separate causes. A row whose overhead exceeds the
headroom is **infeasible** (131's rule): only `O_0` is reported and the terms are left empty. Assertions: the terms sum
to `O_0 - G_stream` to 1e-9 on every row and draw; and `G_rank` at zero overhead equals the top-k gain the official
scorer (`rap.submission.budget_cell`, zero-overhead profile) reports for the same scores. (The V1 scores are not the
shipped ones, so there is no shipped table row to equal; the shipped rows are checked in C.) Descriptive, except that
the sign of the third term is reported per row.

## Outputs

`results/final/cached_analyses_selection.csv`, `cached_analyses_benefit_harm.csv`,
`cached_analyses_overhead_tolerance.csv`, `cached_analyses_gap_accounting.csv`, the three-panel figure data
`cached_analyses_figure_data.csv` (panel a: P1 optimum-set disagreement per cell and quota; panel b: realised gain
against overhead, core cells at 20 %, ms; panel c: the gap accounting at 20 %), and `docs/iclr_cached_analyses.md`
with the readings above applied. Script: `scripts/158_cached_analyses.py`.
