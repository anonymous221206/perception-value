# Causal streaming allocation with rate controllers

> **Pre-Task-23 run.** The numbers in this document come from the run before the ego-frame convention (Task 23, 2026-09-22) and are kept as the record of that run. `results/final/` now holds the ego-frame run; where the two differ, `results/final/` is current (Task 28 audit, `scripts/157_doc_staleness.py`).

> **Frame convention.** The figures quoted in this report were computed with the monocular lift in the camera frame, the convention before Task 23. Every table in `results/final/` is now computed in the ego frame; `docs/iclr_ego_frame_convention.md` and `results/final/ego_frame_convention.csv` give each registered quantity old beside new, and the camera-frame convention stays selectable with `--frame camera`.

**Question.** Scored as a whole-split ranking (C), the benchmark's learned allocators are hindsight top-k.
`docs/iclr_causal_threshold.md` found:
* a threshold frozen before the test stream (A) is close to C;
* adding a causal running cap (B: at most floor(1 + k·t) escalations after t inputs of a unit) loses about half the
  decision value, pooled B − C = −0.089 [−0.14, −0.02];
* and the realised rate misses the target by more than five points in most rows.

Can a causal controller recover B's loss while hitting the target rate?

**Provenance.**
* Code: `scripts/130_streaming_controllers.py`. Output: `results/final/streaming_controllers.csv`.
* CPU only. No official result file changed. Every pooled figure below is a row of the CSV.

## 1. Scores, and what was fit

**Nothing is refit in the stage.**
* Official test scores (C): R1 (`*_routers_r1`), nuPlan gates and R1 (`*_nuplan_real_allocation`), and the core gates
  (`*_budget_gate_scores`, the saved set checked against every shipped budget row).
* V1 scores (models fit on the train units only, as in script 124) existed for **none of the six learned signals**
  (gate_ridge, gate_gbm, R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf). They were fit once with 124's own code and
  saved (`*_streaming_v1_scores`); every policy uses that set.

**Tie-breaking.**
* Script 124 seeds its tie draws with Python's `hash()` of strings, which is salted per process. Its tie draws differ
  between runs, which is one source of the drift documented for its stage, besides the refits.
* 45 of the 72 pooled cell × signal rows have test scores that tie with calibration scores.
* Here, tie keys come from a CRC32 of the row identity, and those 45 rows average over 32 key draws.
* The sanity gate below mirrors 124 exactly, including its hash seeds. The stage runs with `PYTHONHASHSEED=0` so that
  it is reproducible.
* Run once without the fixed seed, the sanity row read [−0.1427, −0.0240]; with it, [−0.1426, −0.0238]. Every other
  row of the CSV was identical between the two runs.

## 2. Sanity gate: passed

From the saved scores, with 124's own calibration, masks, cap, joint bootstrap (25% prize filter) and pooled statistic:

| figure | shipped (quoted precision) | reproduced |
|---|---|---|
| pooled B − C | −0.089 | −0.0891 → −0.089 |
| its 95% interval | [−0.14, −0.02] | [−0.1426, −0.0238] → [−0.14, −0.02] |
| pooled A − C (mean over the 72 pairs) | +0.012 | +0.0117 → +0.012 |

## 3. Controllers

All controllers run on V1. Each unit (scene, sequence or log) is processed in timestamp order, with state reset per
unit.

| policy | rule |
|---|---|
| A | escalate iff the score clears the threshold calibrated on validation at share k |
| B | A, and at most floor(1 + k·t) escalations after t inputs of the unit |
| **D, adaptive threshold** | the threshold is calibrated at share k_t = clip(k + gP·(k − r_W) + gI·(k − r_cum), 0, 1), where r_W is the realised rate over the last W inputs and r_cum over all inputs so far (k_1 = k, i.e. A). B's cap still applies |
| **E, token bucket** | one token at the start; k tokens accrue per input, up to capacity c; escalate iff the score clears A's threshold and a whole token is available. c = ∞ is exactly B |

**Hyperparameters.** Chosen per rate on the **validation units**, over the 12 pooled cells × 6 signals, and frozen
before the test stream.
* Rule: among configurations with a median |realised rate − k| of at most 2 points, take the highest mean nDG − C;
  otherwise take the smallest median deviation.
* Grids: D over W ∈ {10, 30, 100}, gP ∈ {0.25, 0.5, 1}, gI ∈ {0, 0.5, 1}; E over c ∈ {1, 2, 3, 5, 10, 20, ∞}.

| rate | D chosen (W, gP, gI) | D validation median deviation | E chosen c | E validation median deviation | configurations meeting ≤ 2 points on validation |
|---|---|---|---|---|---|
| 10% | (100, 1, 1) | 2.2 pp | 20 | 4.4 pp | none |
| 20% | (100, 1, 1) | 2.8 pp | ∞ | 6.4 pp | none |
| 30% | (100, 1, 1) | 2.8 pp | ∞ | 7.4 pp | none |
| 50% | (10, 1, 1) | 1.9 pp | ∞ | 8.2 pp | 2 of 27 (D) |

* On validation, a smaller bucket missed the rate by more, and lost at least as much value, as a larger one.
* c = 20 and c = ∞ tied at 10%; 20 was taken in grid order.
* **E therefore equals B at 20, 30 and 50%, and is within 0.001 of B at 10%.**

## 4. Pre-registered reading at 20%: neither controller recovers the causal loss

Pooled over the 12 cells × 6 learned signals (72 rows). The paired cluster bootstrap resamples units once per dataset
per draw; 1,000 draws, every draw kept, each gain difference divided by the full-sample prize.

| policy | nDG − C | 95% CI | rows above C | rows missing the rate by > 5 pp | median \|rate − k\| | reading |
|---|---|---|---|---|---|---|
| A | +0.012 | [−0.054, +0.117] | 40 | 49 | 7.4 pp | — |
| B | −0.100 | [−0.179, −0.015] | 21 | 63 | 7.8 pp | — |
| **D** | **−0.083** | **[−0.153, −0.007]** | 25 | 12 | **3.6 pp** | **does not recover** (interval below 0, and deviation above 2 pp) |
| **E** | **−0.100** | **[−0.179, −0.015]** | 21 | 63 | **7.8 pp** | **does not recover** (identical to B) |

**At every rate:**

| rate | A | B | D | E | D median deviation | D rows > 5 pp |
|---|---|---|---|---|---|---|
| 10% | +0.028 [−0.039, +0.134] | −0.067 [−0.115, −0.006] | −0.060 [−0.107, +0.003] | −0.067 [−0.115, −0.006] | 2.7 pp | 2 |
| 20% | +0.012 [−0.054, +0.117] | −0.100 [−0.179, −0.015] | −0.083 [−0.153, −0.007] | −0.100 [−0.179, −0.015] | 3.6 pp | 12 |
| 30% | +0.000 [−0.057, +0.109] | −0.107 [−0.198, −0.026] | −0.096 [−0.174, −0.025] | −0.107 [−0.198, −0.026] | 3.2 pp | 20 |
| 50% | −0.007 [−0.051, +0.081] | −0.106 [−0.172, −0.029] | −0.122 [−0.200, −0.036] | −0.106 [−0.172, −0.029] | 1.9 pp | 5 |

**Random-score reference** (uniform scores, 32 draws, the same hyperparameters). At 20%: A +0.001, B −0.011, D +0.005.
The median deviation is 0.2, 3.3 and 1.1 pp. With scores that carry no information, the cap alone costs about 0.01 and
D costs nothing. The learned signals' loss under B and D is therefore about ranking, not about the controller itself.

## 5. What the controllers change

**D fixes most of the rate problem, but not the value problem.**
* At 20%, rows missing the rate by more than 5 points fall from 63 (B) to 12, and the median deviation halves.
* D recovers 0.018 of B's −0.100 on average, and its interval still excludes 0.
* D under-escalates on average: the mean realised rate is 0.164 against 0.20. The cap still refuses late frames of
  bursts, and lowering the threshold to fill the budget admits lower-scored frames, whose value is small or negative.

**Why A's rate misses.** A's threshold is calibrated on the validation units, and the test distribution of scores
differs.
* On KITTI, A escalates 28–30% of frames at a 20% target, and B's cap cuts that to 14%.
* On nuScenes and nuPlan, A escalates 13–22%.
* A's pooled advantage over C (+0.012) is partly bought by escalating more than the budget: 49 of 72 rows miss by
  more than 5 points.

**Per cell at 20%** (nDG − C, mean over the six signals):

| cell | A | B | D | D realised rate |
|---|---|---|---|---|
| KITTI mono brake | +0.070 | −0.059 | −0.053 | 0.168 |
| KITTI mono Planner B | +0.071 | +0.018 | +0.022 | 0.167 |
| KITTI oracle brake | +0.118 | −0.092 | −0.097 | 0.168 |
| KITTI oracle Planner B | +0.180 | −0.180 | −0.183 | 0.168 |
| nuPlan PDM-Closed safety | −0.062 | −0.206 | −0.170 | 0.149 |
| nuPlan PDM-Closed scalar_J | −0.088 | −0.254 | −0.231 | 0.147 |
| nuScenes mono brake | −0.054 | −0.114 | −0.145 | 0.166 |
| nuScenes mono plan_ade | −0.049 | −0.070 | −0.054 | 0.164 |
| nuScenes mono plan_fde | −0.060 | −0.070 | −0.055 | 0.166 |
| nuScenes oracle brake | +0.016 | −0.183 | −0.124 | 0.172 |
| nuScenes oracle plan_ade | +0.057 | +0.061 | +0.058 | 0.167 |
| nuScenes oracle plan_fde | −0.059 | −0.055 | +0.040 | 0.159 |

## 6. Caveats

* The controller grids are small, and chosen on few validation units (16 nuScenes scenes, 4 KITTI sequences,
  6 nuPlan logs). No D configuration met the 2-point rate criterion on validation below 50%.
* The PDM-Closed cells hold most of their prize in one test log, as in the earlier tracks.
* V1's test scores come from one fit of the train-only models, saved and reused. Script 124's regenerations refit
  them each time.

## 7. For the paper

* Report the streaming result as a property of the ranking under a causal budget, not of the controller. The best
  causal controller tried cuts the rate misses fivefold but still loses about 0.08 nDG against hindsight top-k at 20%.
* E's validation choice (c = ∞) shows that capping bursts only costs value here. A bucket smaller than B's unlimited
  carry-over does not help.
