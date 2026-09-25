# Four analyses on cached scores (Task 29)

> **Action history (Task 32).** The figures quoted in this report were computed with each branch of the braking and trajectory controllers charged against its own previous action. Every table in `results/final/` now charges both branches against the action of the all-CHEAP run, so that a decision value is one escalation from all-CHEAP operation; `docs/iclr_shared_history.md` gives the change and the quantities it moved, old beside new.

Pre-registered in `docs/iclr_cached_analyses_prereg.md` (committed in the development repository before anything was computed; that history is not part of this release); the readings
below are the registered ones. Nothing was refit; every score is read from a cached run. Only new files were written:
`results/final/cached_analyses_{selection,benefit_harm,overhead_tolerance,gap_accounting,figure_data}.csv`
(`scripts/158_cached_analyses.py`, runs `*_cached_analyses_{a,b,c,d,fig}`). Every cell and quota is in the CSVs,
nulls and reversals included; what follows summarises them. Costs: cost registry 2026-09-23.1, routers overhead
measurement. Intervals: 95 %, 1,000 draws, units resampled once per dataset per draw and shared across cells, every draw
kept. Shares are of the all-cheap loss.

## A. Does a perception objective pick a worse allocator?

**A2, descriptive: how often the objectives disagree on the test set.** The 52 cell-quota pairs of
`objective_swap.csv`: 13 cells whose test decision value is defined, x 4 quotas. "No shared optimum" means the optimal
sets under the two objectives (ties within 1e-9) do not intersect.

| pool | objective | no shared optimum, 52 pairs | 10 core cells x 4 quotas |
|---|---|---|---|
| P1 (random, uncertainty, cheap criticality, ego speed) | missed objects (`dE_E1_fn_only`) | 43 | 36 of 40 |
| P1 | risk-weighted (`dE_E6_risk_weighted`) | 44 | 34 of 40 |
| P2 (P1 + gates + R1, V-trained: sensitivity only) | missed objects | 49 | 38 of 40 |
| P2 | risk-weighted | 50 | 38 of 40 |
| 127's full pool, with the perception diagnostics | missed objects | 45 | 33 of 40 |
| 127's full pool | risk-weighted | 41 | 29 of 40 |

For continuity: on the full pool, every winner here equals `objective_swap.csv`'s own. The 43 and 40 that 127 reports
are its headline pool (tiers 1 and 3); the full pool gives 45 and 41. The median Kendall tau between the two P1 rankings
is -0.33 for both perception objectives. These counts are descriptive.

**A3, selection on validation, evaluation on test.** Each winner is chosen on the validation units (learned signals by
their V1 train-only scores) and then evaluated once on test by realised decision gain. The statistic is D, the gain of
the perception-selected signal minus that of the decision-selected one, as a share of the all-cheap loss; negative
means the perception objective chose worse.
- **Included:** 11 cells, the 10 core cells and nuPlan PDM-Closed scalar_J.
- **Dropped:** nuPlan PDM-Closed safety, IDM safety and IDM scalar_J at every quota. The decision value is undefined on
  their validation units, which have 4, 0 and 7 affected states (registered drop rule).
- **Excluded everywhere:** the pixel router, which has no validation scores.

Pooled mean of D over the 11 cells:

| pool | objective | 10 % | 20 % | 30 % | 50 % |
|---|---|---|---|---|---|
| P1 | missed objects | -0.039 [-0.067, -0.001] | **-0.074 [-0.108, -0.010]** | -0.074 [-0.131, -0.016] | -0.091 [-0.145, -0.017] |
| P1 | risk-weighted | -0.029 [-0.055, +0.010] | **-0.064 [-0.099, +0.004]** | -0.079 [-0.140, -0.009] | -0.089 [-0.140, -0.014] |
| P2 | missed objects | -0.048 [-0.074, -0.016] | -0.031 [-0.074, -0.004] | -0.050 [-0.110, -0.014] | -0.095 [-0.148, -0.017] |
| P2 | risk-weighted | -0.018 [-0.049, +0.005] | +0.011 [-0.044, +0.037] | -0.028 [-0.080, +0.012] | -0.084 [-0.136, -0.011] |

**Reading (registered: P1, 20 %).**
- **Missed objects: supported.** Selecting by the missed-object gain picks allocators with lower decision value; the
  pooled interval at 20 % lies below zero.
- **Risk-weighted: unresolved.** The interval at 20 % reaches +0.004. The A2 disagreement stands as descriptive for this
  objective.

The other quotas and pool P2 do not change the reading. Most P1 cells lean negative, but the size varies a lot by cell:
- The KITTI oracle cells carry most of the pooled effect: D is -0.23 for brake under both objectives, and -0.40 / -0.41
  for trajectory. There, ego speed wins on decision value, while uncertainty (missed objects) or cheap criticality
  (risk-weighted) wins on perception gain.
- KITTI mono trajectory and PDM-Closed scalar_J lean positive (up to +0.027) with intervals spanning zero.
- Four nuScenes pairs have the same winner under both objectives, so D = 0 exactly.

Across all pools and quotas, 9 (P1, missed objects), 5 (P1, risk-weighted), 11 and 7 (P2) per-cell intervals lie below
zero. One lies above zero (P2, risk-weighted).

## B. Benefit captured and harm incurred

The identity `O(k) - G = (B_all - B) + H - (B_all - O(k))` held to 1e-9 on every row and every bootstrap draw. Means
over the 10 core cells at 20 % (descriptive; per cell and quota, with intervals and paired differences to random, in
the CSV):

| signal | benefit B | harm H | missed benefit | budget-forced | gain |
|---|---|---|---|---|---|
| ego speed | 0.093 | 0.013 | 0.098 | 0.0002 | 0.080 |
| gate_gbm | 0.070 | 0.013 | 0.122 | 0.0002 | 0.057 |
| R1_gbm_clf | 0.069 | 0.013 | 0.122 | 0.0002 | 0.056 |
| R1_gbm_reg | 0.068 | 0.008 | 0.123 | 0.0002 | 0.060 |
| R1_mlp_reg | 0.062 | 0.008 | 0.129 | 0.0002 | 0.055 |
| gate_ridge | 0.062 | 0.011 | 0.129 | 0.0002 | 0.051 |
| R1_mlp_clf | 0.061 | 0.012 | 0.130 | 0.0002 | 0.049 |
| random | 0.038 | 0.013 | 0.153 | 0.0002 | 0.025 |
| cheap criticality | 0.038 | 0.017 | 0.153 | 0.0002 | 0.021 |
| R2 | 0.024 | 0.013 | 0.167 | 0.0002 | 0.011 |
| uncertainty | 0.026 | 0.021 | 0.165 | 0.0002 | 0.005 |

At 20 % almost every positive-value input fits the budget, so the budget-forced term is negligible. The oracle's gain is
close to B_all, and the gap to it is mostly missed benefit, with harm second. Against random, per cell at 20 %:
- harm is lower in 13 of the 100 signal-cell pairs and higher in 13;
- benefit is higher in 9 and lower in 11.

The two-axis plot data (B against H, core cells at 20 %) is in the CSV.

## C. How much overhead can a ranking afford?

For every deployable ranking, cell, unit (ms, module mJ) and budget level, the CSV gives the overheads at which the
point-estimate gain is at least random's. They are exact intervals in o from the scorer's own capacity rule
(`rap.submission.measured_share` / `measured_k`); 20 rows per unit at 20 % have more than one interval, because V is
signed.

The CSV also gives:
- the same set against running FULL on every input, which fits the budget only in ms at 30 % and 50 % (never in mJ);
- the measured overhead, and whether it falls inside each set.

At the measured overhead, all 1,056 rows' feasibility, nDG and paired interval against random, recomputed through
`rap.submission.budget_cell`, equal the shipped budget tables exactly.

Core cells, 20 %, ms (median over cells of the largest overhead that still beats random, against the measured
overhead):

| signal | largest affordable overhead, ms | measured, ms | measured inside the set (of 10 cells) |
|---|---|---|---|
| uncertainty | 3.69 | 0.29 | 1 |
| cheap criticality | 3.30 | 1.47 | 3 |
| gate_ridge | 2.92 | 3.93 | 0 |
| R1_gbm_reg / clf | 2.80 / 2.79 | 15.91 | 0 |
| R1_mlp_reg / clf | 2.34 / 2.26 | 0.65 | 8 / 7 |
| R2 | 1.71 | 9.77 | 0 |
| gate_gbm / batched | 1.58 | 21.38 / 3.54 | 0 / 3 |

What limits each signal differs:
- **The GBM routers and single-row gate_gbm** are limited by cost: their measured overhead is five to thirteen times
  what their ranking can afford.
- **The MLP routers** are cheap enough in most cells.
- **Uncertainty and cheap criticality** are limited by ranking quality: they cost little, but at zero overhead they beat
  random in few cells, so the set is empty in most of them (47 of 132 rows per unit at 20 % have no qualifying
  overhead).

## D. Where the gap between the zero-cost oracle and a causal allocator goes

Scope: 10 core cells and the two PDM-Closed cells; the six learned signals with V1 scores; latency budgets.

**Assertions.** Both held on every row:
- the three terms sum to `O_0 - G_stream` to 1e-9, at every draw;
- `G_rank` at zero overhead equals the gain the official scorer reports for the same scores.

**Feasibility.** The gradient-boosted models (gate_gbm, R1_gbm_reg, R1_gbm_clf) are infeasible at both levels in every
cell: their measured overhead exceeds the headroom `alpha C_f`. gate_ridge is feasible at 20 % only in the two nuPlan
cells (whose detector costs leave more headroom), and at 50 % everywhere. Infeasible rows report `O_0` only.

Means over the feasible rows (shares of the all-cheap loss):

| budget | signal (rows) | `O_0 - G_stream` | overhead term | ranking term | causal term |
|---|---|---|---|---|---|
| 20 % | R1_mlp_reg (12) | 0.184 | 0.000 | 0.179 | +0.004 |
| 20 % | R1_mlp_clf (12) | 0.197 | 0.000 | 0.179 | +0.017 |
| 20 % | gate_ridge (2, nuPlan) | 0.322 | 0.061 | 0.179 | +0.081 |
| 50 % | R1_mlp_reg (12) | 0.148 | 0.000 | 0.119 | +0.029 |
| 50 % | R1_mlp_clf (12) | 0.153 | 0.000 | 0.122 | +0.031 |
| 50 % | gate_ridge (12) | 0.140 | 0.000 | 0.095 | +0.044 |

Under this protocol, for the allocators that can run, most of the gap is the ranking term: top-k with the allocator's
own scores against the oracle at the same capacity.
- **Overhead term:** zero at the MLPs' sub-millisecond cost, 0.06 for gate_ridge on nuPlan at 20 %.
- **Causal-cap term:** positive in 47 rows, negative in 14, zero in 1. Negative means the causal policy did better than
  top-k, for example R1_mlp_reg on both PDM-Closed cells.

These three terms are an accounting of this protocol; they are not losses and not separate causes.

## Figure data

`cached_analyses_figure_data.csv`:
- panel a: the P1 optimum sets and their disagreement per cell and quota;
- panel b: the realised gain against overhead, core cells, 20 %, ms, as step intervals in o with the random reference
  and the measured overhead;
- panel c: the gap accounting at 20 %.

## Notes

- A3's pooled effect is concentrated in the KITTI oracle cells (above). The registered reading is on the pooled
  interval, and it is reported as registered.
- Parts C and D charge the routers overhead measurement for every signal, because it is the only one that covers all
  of them. The two-level tables charge the gates the primary measurement instead (gate_gbm on KITTI: 19.98 ms against
  21.38 ms). In D this changes no feasibility verdict: under either measurement, gate_ridge is infeasible at 20 % on
  KITTI and nuScenes and feasible on nuPlan and at 50 %, and gate_gbm is infeasible throughout. In C the
  measured-overhead marker would move by the difference.
