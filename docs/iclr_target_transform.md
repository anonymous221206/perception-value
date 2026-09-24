# Training objective against target transform, and training-seed variation (Task 30)

> **Post-hoc.** This analysis was decided after Task 24's test results were seen. Nothing in it was pre-registered,
> and its intervals are not corrected for the number of comparisons. Read every result as exploratory.

Script `scripts/159_target_transform.py`, CPU only, run sequentially on the board.
- **Part 1** (target transforms) took 433 s. Run `*_target_transform_part1`, outputs
  `results/final/target_transform_control.csv` and `target_transform_heatmap.csv`.
- **Part 2** (seeds) took 1,225 s. Run `*_target_transform_part2`, output `results/final/seed_variation.csv`.
- **The heatmap figure** took a few seconds: `results/final/figures/target_transform_heatmap.{pdf,png}`.

New files only.

## Part 1: rank and signed-ECDF targets for R1 regression

**What changes, and what does not.** For R1_mlp_reg and R1_gbm_reg, the fit units (train + val), the 225-dim
detection-list features, the hyperparameters, seed 0, the loss and the preprocessing are all the shipped ones (103's
`models`). Only the label changes:
- raw V;
- rank(V): ordinal rank / n on the fit units, ties by 148's Q-MORIC rule (`argsort(argsort)`);
- signed ECDF of V: F+(V) for V > 0, 0 for V = 0, F-(V) - 1 for V < 0, with F+ and F- the empirical CDFs of the
  positive and negative fit-unit values;
- Q and G in their shipped CDF forms.

**Checks.** On all ten cells, refits on raw V, Q and G reproduce the cached scores exactly (maximum difference 0). The
raw-V point nDG equals `benchmark_table_routers.csv`. Scoring uses the benchmark's scorer on routers_r1's own bootstrap
draws for each cell, so every label is paired with every other on the same 1,000 draws. The gradient-boosted models are fitted with OpenMP on one
thread (`threadpoolctl`): their multithreaded fit varies in the last bits from run to run (0 to 2e-15 on KITTI
oracle brake across six fits), which the exact-equality checks catch; one thread is stable and reproduces the shipped
scores. BLAS stays at its default for the MLPs: it is stable, and limiting it too moves the MLP scores (3.6e-8).

**A definitional note.** The task describes the signed ECDF as "the MORIC+ construction 148 uses for G". But 148's
`g_moric` is the plain right-continuous ECDF over all fit-unit values, not the signed form. The signed ECDF was
implemented exactly as the task defines it; G is used in its shipped (plain-ECDF) form.

**Pooled over the ten core cells.** Paired difference in nDG, mean over cells; the interval is from the draw-wise mean
over cells, each cell on its own draws.

| architecture | label | 10 % | 20 % | 30 % | 50 % |
|---|---|---|---|---|---|
| MLP | rank(V) - V | -0.082 [-0.154, -0.003] | -0.127 [-0.198, -0.014] | -0.139 [-0.208, +0.008] | -0.102 [-0.198, +0.038] |
| MLP | signed ECDF(V) - V | -0.056 [-0.109, +0.016] | -0.076 [-0.140, +0.020] | -0.027 [-0.098, +0.069] | +0.037 [-0.035, +0.123] |
| MLP | Q - V | -0.076 [-0.153, +0.000] | -0.135 [-0.226, -0.022] | -0.123 [-0.229, +0.008] | -0.114 [-0.223, +0.009] |
| MLP | G - V | -0.063 [-0.120, +0.029] | -0.027 [-0.118, +0.071] | -0.066 [-0.131, +0.067] | -0.047 [-0.146, +0.090] |
| GBM | rank(V) - V | -0.028 [-0.116, +0.082] | -0.029 [-0.123, +0.103] | -0.002 [-0.113, +0.117] | -0.003 [-0.118, +0.119] |
| GBM | signed ECDF(V) - V | +0.063 [-0.021, +0.114] | +0.023 [-0.033, +0.114] | +0.048 [-0.042, +0.109] | +0.036 [-0.037, +0.104] |
| GBM | Q - V | +0.043 [-0.056, +0.098] | -0.025 [-0.116, +0.106] | +0.000 [-0.117, +0.135] | +0.048 [-0.040, +0.158] |
| GBM | G - V | +0.064 [-0.011, +0.164] | +0.088 [-0.018, +0.205] | +0.072 [-0.045, +0.193] | -0.008 [-0.105, +0.155] |

Holding the transform fixed, the objective comparison:

| architecture | comparison | 10 % | 20 % | 30 % | 50 % |
|---|---|---|---|---|---|
| MLP | Q - rank(V) | +0.006 [-0.058, +0.055] | -0.008 [-0.101, +0.054] | +0.017 [-0.110, +0.088] | -0.012 [-0.138, +0.090] |
| MLP | G - rank(V) | +0.020 [-0.034, +0.097] | +0.100 [+0.002, +0.181] | +0.074 [-0.042, +0.186] | +0.055 [-0.045, +0.167] |
| MLP | Q - signed ECDF(V) | -0.020 [-0.088, +0.028] | -0.059 [-0.157, +0.020] | -0.096 [-0.193, +0.002] | -0.151 [-0.263, -0.049] |
| MLP | G - signed ECDF(V) | -0.007 [-0.062, +0.065] | +0.049 [-0.058, +0.132] | -0.039 [-0.123, +0.078] | -0.084 [-0.181, +0.034] |
| GBM | Q - rank(V) | +0.071 [-0.064, +0.127] | +0.003 [-0.137, +0.124] | +0.002 [-0.137, +0.147] | +0.051 [-0.072, +0.204] |
| GBM | G - rank(V) | +0.092 [-0.011, +0.182] | +0.116 [+0.005, +0.215] | +0.075 [-0.035, +0.185] | -0.005 [-0.116, +0.140] |
| GBM | Q - signed ECDF(V) | -0.020 [-0.111, +0.045] | -0.048 [-0.151, +0.045] | -0.047 [-0.149, +0.089] | +0.012 [-0.070, +0.132] |
| GBM | G - signed ECDF(V) | +0.001 [-0.059, +0.106] | +0.065 [-0.041, +0.148] | +0.025 [-0.067, +0.140] | -0.044 [-0.137, +0.119] |

**Per cell.** The main presentation is `results/final/figures/target_transform_heatmap.pdf`: architecture x cell, the
paired difference to raw V at 20 % with starred cells whose interval excludes zero, and 10/30/50 % as small multiples.
For context, counts over the 40 cell-quota pairs of intervals above / below zero against raw V:

| architecture | rank(V) | signed ECDF(V) | Q | G |
|---|---|---|---|---|
| MLP | 0 / 4 | 0 / 2 | 0 / 5 | 1 / 0 |
| GBM | 3 / 1 | 5 / 0 | 9 / 2 | 9 / 1 |

For the GBM router, the cell pattern is the one Task 24 reported:
- Q and G do better on the four nuScenes planner cells (+0.16 to +0.42 at 20 %).
- They do worse on the brake cells and KITTI trajectory.
- The signed ECDF of V reproduces much of the planner-cell gain (+0.09 to +0.22) without the brake-cell losses. rank(V)
  gains on three of the planner cells, but loses 0.55 on KITTI mono trajectory.

**Reading.**
- **GBM router.** Neither transform closes most of the gap between raw V and G. At 20 %, the signed ECDF of V recovers
  about a quarter of G's pooled advantage (+0.023 of +0.088), and rank(V) none of it. With the transform held fixed, G
  stays above rank(V) (+0.116 [+0.005, +0.215]) and above the signed ECDF (+0.065 [-0.041, +0.148]; interval spans
  zero). So, as far as this post-hoc comparison can say, the objective difference persists for G on this architecture.
  G's own pooled advantage over raw V has an interval that spans zero at every quota.
  - For Q there is no pooled gap to explain (-0.025 at 20 %). Its per-cell gains on the planner cells are largely
    matched by the signed ECDF of V.
- **MLP router.** The transform explains Q's deficit. Q's loss against raw V (-0.135 [-0.226, -0.022] at 20 %) is
  matched by rank(V)'s (-0.127), and Q minus rank(V) is -0.008 [-0.101, +0.054]: the rank transform, not the objective,
  costs the MLP its decision value. G on the MLP is a null against raw V at every quota. It sits above rank(V) at 20 %
  (+0.100 [+0.002, +0.181]) and not at the other quotas.

**Binary targets, a separate comparison** (V > 0, Q > 0, G > 0, from their caches; not evidence about learning
magnitudes). At 20 %, pooled against the V > 0 classifier:
- MLP: Q -0.051, G -0.086;
- GBM: Q +0.024, G -0.078.

All four intervals span zero. Per cell: 5 intervals below zero for GBM on G > 0; 1 above and 2 below for GBM on Q > 0;
2 above for MLP on Q > 0; 1 below for MLP on G > 0.

## Part 2: training-seed variation at 20 %

Seeds 1-5 are fixed beside the shipped seed 0. The seed-0 refit reproduces the shipped scores exactly for every model
and cell.

**The gradient-boosted models do not depend on the seed.** For R1_gbm_reg, R1_gbm_clf and gate_gbm, the seed-1 fit is
bit-identical to the seed-0 fit on all ten cells. In the installed scikit-learn (1.3.2), HistGradientBoosting uses its
random state only for early stopping's validation split and scoring subsample (early stopping is off) and for the
bin-threshold subsample (only above 200,000 samples; these fit sets have at most about 4,500). They were not refitted further; their spread is zero, and each
of their cells is a win for all six seeds or for none. gate_ridge is deterministic and was skipped, as the task says.

**The MLP routers vary.** nDG at 20 % over the six seeds (min / median / max), and the seeds whose paired interval
against random lies above zero:

| cell | R1_mlp_reg | wins | R1_mlp_clf | wins |
|---|---|---|---|---|
| KITTI mono brake | 0.150 / 0.169 / 0.192 | 1 | 0.151 / 0.162 / 0.185 | 0 |
| KITTI mono traj | 0.041 / 0.117 / 0.236 | 0 | -0.186 / 0.091 / 0.194 | 0 |
| KITTI oracle brake | 0.217 / 0.255 / 0.284 | 2 | 0.253 / 0.288 / 0.295 | 5 (shipped win) |
| KITTI oracle traj | 0.321 / 0.382 / 0.460 | 3 | 0.356 / 0.393 / 0.419 | 6 (shipped win) |
| nuScenes mono brake | 0.202 / 0.277 / 0.473 | 3 (shipped win) | 0.248 / 0.289 / 0.393 | 1 |
| nuScenes mono plan_ade | 0.106 / 0.251 / 0.298 | 0 | 0.034 / 0.108 / 0.150 | 0 |
| nuScenes mono plan_fde | 0.085 / 0.210 / 0.260 | 0 | -0.027 / 0.027 / 0.141 | 0 |
| nuScenes oracle brake | 0.091 / 0.217 / 0.335 | 0 | 0.001 / 0.164 / 0.245 | 0 |
| nuScenes oracle plan_ade | -0.120 / -0.058 / 0.064 | 0 | -0.137 / -0.070 / 0.043 | 0 |
| nuScenes oracle plan_fde | -0.055 / 0.004 / 0.045 | 0 | -0.047 / 0.056 / 0.164 | 0 |

**Flag.** One shipped win is not a win for at least four of the six seeds: **R1_mlp_reg on nuScenes mono brake** (3 of
6). Every other shipped win holds:
- R1_gbm_reg: 3 cells; R1_gbm_clf: 3; gate_gbm: 2. These are seed-independent, so each holds for all six seeds.
- R1_mlp_clf: KITTI oracle brake (5 of 6) and KITTI oracle trajectory (6 of 6).

Within a cell, the MLPs' seed spread reaches 0.27 nDG (R1_mlp_reg, nuScenes mono brake) and 0.38 (R1_mlp_clf, KITTI
mono trajectory). This variation is not in the benchmark's bootstrap, which holds each fitted model fixed.
