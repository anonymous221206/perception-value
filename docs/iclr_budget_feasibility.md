# Infeasible measured budgets

> **Pre-Task-23 run.** The numbers in this document come from the run before the ego-frame convention (Task 23, 2026-09-22) and are kept as the record of that run. `results/final/` now holds the ego-frame run; where the two differ, `results/final/` is current (Task 28 audit, `scripts/157_doc_staleness.py`).

> **Frame convention.** The figures quoted in this report were computed with the monocular lift in the camera frame, the convention before Task 23. Every table in `results/final/` is now computed in the ego frame; `docs/iclr_ego_frame_convention.md` and `results/final/ego_frame_convention.csv` give each registered quantity old beside new, and the camera-frame convention stays selectable with `--frame camera`.

**What changed.** Every measured-budget table charged an allocator the escalated share
$f = \max\big((b - C_c - C_S)/C_f,\; 0\big)$ at budget $b(\alpha) = C_c + \alpha C_f$. When the allocator's own
per-input overhead $C_S$ exceeds the headroom $b - C_c = \alpha C_f$, it cannot run within the budget at all. Such a
row was still written as a 0% escalation with nDG 0, and plotted as nDG 0.

**The rule now.** `rap.budget.infeasible`, applied by every writer:
* `feasible` = $C_S \le b - C_c$, with a tolerance of 1e-9.
* An infeasible row keeps its budget, costs, overhead and `boot_dropped`.
* nDG, gain, escalated share, the paired difference to random and their intervals are NaN.
* `note` states the overhead and the headroom.
* Random and the oracle have no overhead and are always feasible.
* The skipping design also charges $C_S + C_c$ at zero escalation, so the same rule applies to its shares.

**Validation.**
* Every changed file was compared with the file before the change.
* Every row that stays feasible is unchanged in every pre-existing column, exactly and as text.
* 120 and 122 were re-run and reproduce their flagged outputs byte for byte, and 128 was re-run.
* The figure export's medians over cells are aggregates, so they change wherever a cell turned infeasible.

| file | stage | rows | changed (all flagged infeasible) |
|---|---|---|---|
| `benchmark_budget_two_level.csv` | C20 | 1,072 | 220 |
| `benchmark_budget_two_level_1thread.csv` | C20 | 1,072 | 142 |
| `benchmark_budget_multifidelity.csv` | C20 | 112 | 44 |
| `benchmark_budget_multifidelity_1thread.csv` | C20 | 112 | 40 |
| `benchmark_budget_routers.csv` | C20 | 1,712 | 568 |
| `fig_budget_curves.csv` | C20 | 1,036 | 293 (274 cells, 19 medians) |
| `benchmark_budget_nuplan_real.csv` | C12 | 416 | 120 |
| `benchmark_target_swap.csv` (summary JSON unchanged) | C19 | 1,106 | 52 |
| `causal_threshold.csv` | C14 | 3,464 | 196 |
| `skip_accounting.csv` | C18 | 126 | 62 |

## Infeasible latency rows

Headroom $\alpha C_f$ for α = 10 / 20 / 30 / 50%:
* nuScenes ($C_c, C_f$ = 12.71, 19.35 ms): 1.94 / 3.87 / 5.81 / 9.68 ms;
* KITTI (13.18, 18.47 ms): 1.85 / 3.69 / 5.54 / 9.24 ms.

**Router run** (`benchmark_budget_routers.csv`):

| allocator | $C_S$ | nuScenes | KITTI |
|---|---|---|---|
| gate_ridge | 3.93 ms | 10, 20% | 10, 20% |
| gate_gbm, batched inference | 3.54 ms | 10% (20%: escalates 1.7%) | 10% (20%: 0.8%) |
| gate_gbm, one call per input | 21.4 ms | all four | all four |
| R2 pixel router | 9.77 / 4.82 ms | all four | 10, 20% |
| R1_gbm_reg, R1_gbm_clf, one call per input | 15.9 ms | all four | all four |

* The nuPlan rows of this table (KITTI costs, nuScenes overheads) follow the same pattern.
* Primary and 1-thread runs: gate_ridge (3.99 / 4.24 ms) is infeasible at 10 and 20%, and gate_gbm (19.7–20.0 ms, or
  16.5–16.8 ms with 1 thread) at all four.
* nuPlan with real perception ($C_f$ = 23.57 ms): gate_ridge and the batched GBM gate at 10% only; the single-call GBM
  gate and both R1 GBMs at all four.
* Multi-fidelity (KITTI mono, three gate calls per input): ridge (5.03 ms) at 10 and 20%, GBM at all four.

## Infeasible energy rows

Headroom:
* nuScenes: 6.67 / 13.33 / 20.00 / 33.33 mJ;
* KITTI: 3.30 / 6.61 / 9.91 / 16.52 mJ;
* nuPlan with real perception: 10.58 / 21.16 / 31.74 / 52.89 mJ.

| table | allocator ($C_S$) | nuScenes | KITTI | nuPlan |
|---|---|---|---|---|
| routers | cheap-side criticality (9.88 / 8.76 mJ) | 10% | 10, 20% | 10, 20% |
| routers | R1 MLP, both targets (5.80 / 5.84 mJ) | — | 10% | 10% |
| routers | R2 (15.79 / 7.79 mJ) | 10, 20% | 10, 20% | — |
| routers | gate_ridge (26.4 mJ) | 10, 20, 30% | all four | all four |
| routers | gate_gbm batched (23.8 mJ) | 10, 20, 30% | all four | all four |
| routers | gate_gbm one call (143.6 mJ), R1 GBM ×2 (142.1 mJ) | all four | all four | all four |
| primary two-level | cheap-side criticality (10.6 mJ) | 10% | 10, 20, 30% | 10, 20, 30% |
| primary two-level | gate_ridge (29.5 / 31.4 mJ) | 10, 20, 30% | all four | all four |
| primary two-level | gate_gbm (146–148 mJ) | all four | all four | all four |
| 1-thread two-level | gate_ridge (6.1 / 6.5 mJ) | — | 10% | 10% |
| 1-thread two-level | gate_gbm (25.3 / 25.7 mJ) | 10, 20, 30% | all four | all four |
| nuPlan real perception | gate_ridge (26.4 mJ), batched GBM (23.8 mJ) | | | 10, 20% |
| nuPlan real perception | single-call GBM, R1 GBM ×2 | | | all four |
| multi-fidelity | ridge ×3 (37.2 mJ; 7.7 mJ with 1 thread) | | all four; 1-thread 10, 20% | |
| multi-fidelity | GBM ×3, and the 640-only GBM | | all four | |

## Figures in the reports that changed

* `iclr_causal_threshold.md` §7, 20% ms: mean feasible rate 0.076 → 0.158 and policy-B nDG +0.026 → +0.053.
  50% ms: 0.259 → 0.423 and +0.110 → +0.179. Both means now cover feasible rows only.
* No win, loss, pooled statistic or reading changes anywhere. Infeasible rows had nDG 0 and could not beat random.
* In the two-level primary table, 119 of the 164 rows worse than random were infeasible allocators; 45 feasible rows
  remain worse than random.
