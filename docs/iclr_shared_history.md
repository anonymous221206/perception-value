# Decision values with a shared action history (Tasks 32–34)

## The problem

The benchmark defines the decision value of escalating input i as a single-step counterfactual on a fixed scene state,
`V_i = L(pi(z_i^cheap), s_i) - L(pi(z_i^full), s_i)`, additive across inputs. Until Task 32, three of the benchmark's
controllers did not compute it that way.
- **Braking controller.** `planner.decision_cost` charges `lam_jerk * |a_cmd - a_cmd(prev)|`.
- **Lateral controller.** `planner.lateral_cost` charges `lam_switch * 1[action != prev]`.
- **Trajectory controller.** `planner_b.plan` and `planner_b.executed_cost` both use the previous plan (`w_switch`).

In every per-frame table before Task 32 (`decision.build`, `65.build_b`, and through them 100, 134, 140, 144 and 118), the
CHEAP branch was charged against its own previous action and the FULL branch against its own. `V_i` therefore compared
an all-CHEAP sequence with an all-FULL sequence at frame i, not one escalation.

**Definitions used here.**
- **Own history:** the values of the earlier releases.
- **Shared history (primary):** both branches are charged against the action the all-CHEAP run took at the previous
  frame of the unit (none at the first frame). That is the CHEAP branch's own history, so the CHEAP loss is unchanged.
  The trajectory controller re-plans the FULL branch with it.
- **Memoryless (sensitivity only):** the shared plans, scored with no action-change term.

## Systems that carry no action history

- **Learned planner, q_plan and q_plan^self.** PKL's released planner (`compile_model(cin=5, cout=16)`) is run once
  per sample by `66_planner_c_pkl_planner.py`. Its five input channels (`third_party/pkl/.../data.py: render`) are the
  road layer, the two divider layers, the ego box, and the object boxes of that sample only. No previous output or
  previous frame enters.
- **nuPlan planners, PDM-Closed and IDM.** `115_nuplan_real_counterfactual.run_planner` builds and initializes a new
  planner (`cf82.make_planner`, `initialize`) for every state and branch. The history buffer holds logged ego states;
  its observations do pass through the branch's perception filter. But both planners read only
  `history.current_state`: `abstract_pdm_closed_planner.py:146` and `pdm_closed_planner.py:83` for PDM-Closed,
  `idm_planner.py:70` for IDM. Neither reads a past observation or a past action.

Their rows are identical under every variant.

## The effect: own, shared and memoryless history

`scripts/160_shared_history.py` (full-tier stage G2b) rebuilds every setting of the sign table from the cached
detections, with `decision.build` and `65.build_b` extended by a `history_variants` option (default off, which
changes no existing column). 33 builds:
- every KITTI pair (YOLOv8s 320/384/512→640, RT-DETR-l 320/480→640) under monocular and oracle geometry;
- nuScenes YOLOv8s 320→640 under both geometries;
- the calibration schemes S1–S3 of `calibration_thresholds.csv`, each run directly at its (t_cheap, t_full).

Composing a cell from single-threshold runs, as 100 does, is no longer valid under a shared history, because the FULL
loss depends on the CHEAP branch's previous action.

**Checks.**
- The own-history columns reproduce the shipped per-frame tables bit for bit. This covers the core matrix and the
  trajectory controller: braking, lateral and trajectory losses for all eight core settings, maximum difference 0.
- The own-history metrics reproduce all 48 shipped calibration cells S0–S3 (`calibration_cells.csv`) and all 20
  reference-geometry values (`reference_geometry_sweep.csv`) to 1e-9.

Full table: `results/final/shared_history_effect.csv`, 234 rows. For every setting, system and variant it gives:
affected, harmed, harmed share of affected, harmed share of all inputs, ρ, ΣV, all-full and oracle@20 reductions, and
the affected inputs whose actions coincide.

### The settings of Table 1 (own → shared → memoryless)

| setting | system | affected | same action (own) | harmed / affected | ρ | all-full / oracle@20, own → shared |
|---|---|---|---|---|---|---|
| nuScenes oracle Y8 320→640 | braking | 240 → 141 → 141 | 99 | 40.4 → 35.5 → 36.2% | 0.421 → 0.418 → 0.410 | 14.21 / 24.56 → 14.01 / 24.07% |
| nuScenes mono Y8 320→640 | braking | 454 → 273 → 273 | 181 | 42.3 → 39.6 → 38.1% | 0.412 → 0.407 → 0.401 | 16.65 / 28.32 → 16.47 / 27.77% |
| KITTI mono Y8 320→640 | braking | 2,931 → 2,075 → 2,075 | 856 | 37.5 → 34.2 → 33.9% | 0.169 → 0.168 → 0.165 | 51.96 / 62.43 → 51.68 / 62.11% |
| KITTI oracle Y8 320→640 | braking | 1,753 → 1,296 → 1,296 | 457 | 30.9 → 24.6 → 24.6% | 0.081 → 0.080 → 0.078 | 68.20 / 74.19 → 67.86 / 73.78% |
| KITTI mono Y8 320→640 | trajectory | 957 → 678 → 678 | 269 | 41.9 → 48.4 → 48.4% | 0.276 → 0.282 → 0.280 | 13.07 / 18.04 → 12.78 / 17.81% |
| KITTI mono Y8 384→640 | trajectory | 697 → 458 → 458 | 230 | 43.8 → 48.7 → 48.9% | 0.456 → 0.459 → 0.458 | 6.02 / 11.07 → 5.97 / 11.03% |
| KITTI mono Y8 512→640 | trajectory | 446 → 266 → 266 | 171 | 47.8 → 53.8 → 53.4% | 1.109 → 1.114 → 1.113 | −0.57 / 5.19 → −0.59 / 5.16% |
| KITTI mono RT 480→640 | trajectory | 492 → 295 → 295 | 182 | 45.3 → 43.4 → 44.1% | 0.578 → 0.579 → 0.578 | 2.96 / 7.01 → 2.94 / 6.98% |
| KITTI oracle Y8 320→640 | trajectory | 637 → 463 → 463 | 155 | 4.2 → 3.0 → 3.0% | 0.004 → 0.003 → 0.003 | 68.99 / 69.25 → 68.34 / 68.58% |

The learned-planner rows (nuScenes q_plan, q_plan^self) and the nuPlan rows are unchanged.

### Reading

1. **Counts and shares move; value-weighted quantities barely do.**
   - Between 24% and 44% of the affected inputs of every braking and trajectory setting (up to 50% for the lateral
     controller) chose the same action in both branches and differed only through the action-change term charged
     against two different histories. Under the shared history none
     remains: the same action in both branches now has the same loss.
   - For the braking and trajectory controllers, ρ moves by at most 0.016, the all-full reduction by at most 0.65
     points and the oracle@20 reduction by at most 0.8 points in every setting.
2. **The sign variation survives.**
   - Under the shared history escalation still harms 35–54% of affected inputs across the rows of Table 1: braking
     35.5–39.6% on nuScenes, trajectory 43.4–53.8% on the monocular KITTI pairs; the learned planner is unchanged at
     49–51%.
   - On KITTI the braking controller now harms 25–48% of affected inputs under oracle geometry (24.6% on YOLOv8s
     320→640). The trajectory controller's harm ratio stays below 0.06 on four of five pairs.
3. **Calibration (Table 6).** Under every scheme S0–S3 and the shared history:
   - every nuScenes braking cell and every moderate-gap KITTI braking and trajectory cell keeps a harm rate of at
     least 34.1%;
   - ρ stays at least 0.338 (nuScenes mono S2).
4. **The direction differs by controller.**
   - Braking loses harmed frames, mostly frames the old definition counted as harmed by jerk alone.
   - The trajectory controller's harmed share rises on the monocular pairs: the frames removed were more often helped.
   - The lateral controller's rises most (for example 56.3% → 71.9%, ρ 2.10 → 2.71 on KITTI mono). It is not in the
     benchmark, and its conclusion (harm dominates) is unchanged.
5. **Memoryless loss** gives the same counts as the shared history for the braking and trajectory controllers, harmed
   shares within 1.7 points and ρ within 0.011, so the remaining effect of the action-change term is small once both
   branches share a history.

## The regenerated tables

Every table of this release is regenerated on the shared action history. The code changes:
- `decision.build` and `65.build_b` default to `history="shared"`; `history="own"` keeps the earlier record, and
  160 passes it explicitly.
- Every per-frame table now carries its weighted loss terms (`Jterm_*`, `JBterm_*`).
- `118.rebuild` charges the FULL branch against the CHEAP branch's previous action.

OLD below is the previous release's `results/final`. The OLD-vs-NEW inventory is in
`results/raw/20260925_084513_shared_history_before/inventory/` (the snapshot itself is not shipped).

Not regenerated, because their inputs carry no action history: 100 and 101 (per-mode calibration losses; see below), 115–120
(nuPlan), 66 and 74 (learned planner), 148 (the published-objective labels Q and G, from detections only) and its R1
and R2 refits.

### Checks

- **Per-frame tables.** The new tables equal 160's shared-history columns bit for bit. This covers all eight
  core tables of 52: `J_full` = 160's `J_full_shared`, `Jlat_full` = `Jlat_full_shared`, CHEAP columns unchanged.
  It also covers all six Planner B tables of 65: `JB_full` = `JB_full_shared`. The loss terms add up to `J` and `JB`
  exactly.
- **Calibration (161).** Every composed braking cell equals 160's direct shared-history build bit for bit
  (28 pairs S0–S3). Each mode build at (t, t) equals its own composition. The trajectory S0 cells from 160 equal
  65's new tables.
- **Multi-fidelity levels (165).** The 320 branch of the 320→384 and 320→512 pairs equals the core 320→640 table's
  CHEAP column bit for bit. 93 asserts that the 320 and 640 columns are exactly the cell's own.
- The downstream stages' own reproduction gates all passed. Among them:
  - 102's registered reading;
  - 118's equality with the joined nuScenes tables;
  - 122's S1 (434/434 official nDG values);
  - 124's refit and sanity checks (0 difference over 64 rows; 444/444 rows);
  - 133's S2–S4;
  - 149's check that the V-trained rows reproduce `benchmark_table_routers.csv`;
  - 159's exact refit check, once 103 fits on one thread (Task 33, below).

### Two structural changes the shared history forced

1. **Calibration cells can no longer be composed from per-mode runs.**
   - Braking: the braking action is a threshold rule on the perceived requirement and does not depend on the
     history. So 161 builds each mode once per threshold, keeps the actions, and composes the FULL loss at
     (t_c, t_f) exactly against the CHEAP action at t_c. This covers every scheme S0–S4 and the 5 × 5 sweep.
   - Trajectory controller S0–S3: built directly at the registered thresholds (160).
   - Trajectory controller S4: kept on the own-history composition and labelled `history = own`, in
     `calibration_cells.csv` and with ‡ in `docs/calibration_tables.md`. Its sweep is not computed.
   - S4 thresholds are still chosen from the per-mode losses, as registered.
2. **The multi-fidelity allocator's levels (93).**
   - Before: the loss of escalating to 384 or 512 was the CHEAP column of the 384→640 or 512→640 table, charged
     against that level's own history. 93 asserted that the FULL (640) loss was the same in the 320→640 and L→640
     tables; under the shared history it is not.
   - Now: 165 builds 320→384 and 320→512, so each level is an escalation from all-320 operation, charged against the
     320 run's previous action. 93 reads those levels and checks, exactly, that they share the 320 branch and the
     320→640 escalation with the cell it scores.

### Results: OLD → NEW

Full inventory: `results/raw/20260925_084513_shared_history_before/inventory/` (164):
- `inventory_files.csv`: 52 files unchanged, 47 changed, 2 new;
- `inventory_columns.csv`: per column, the number of values changed and the largest change;
- `reversed_readings.csv`: every text or boolean value that changed;
- `json_changes.csv`.

**Registered readings: none reverses.**
- Calibration (102): `survives` under S1–S3, with the same pass counts (4 of 4 nuScenes, 6 of 6 moderate-gap KITTI,
  2 cells below the collapse line).
- Target swap (122): `architecture-driven`.
- Published objectives (149): `material`.
- Causal threshold (124): `inconclusive`, −0.066 [−0.129, +0.013] (was −0.064 [−0.128, +0.014]).
- Streaming controllers (130): no `reading` value changed.

**Table 1.** The values are the shared-history column above. Reference geometry (135, `reference_geometry_sweep.csv`) changes one reading: RT-DETR-l 320→640 on
the trajectory controller goes from "persists without lifting error" to "geometry-driven".

**Calibration (Table 6, `calibration_cells.csv`, all units).**
- Over S0–S3, every nuScenes braking cell and every moderate-gap KITTI cell keeps a harm rate of at least 34.1%
  (nuScenes oracle S3) and ρ of at least 0.338 (nuScenes mono S2).
- S4 on the braking controller, for example:
  - nuScenes oracle 47.9% → 41.5%;
  - nuScenes mono 50.3% → 40.5%;
  - KITTI mono Y8 384 31.0% → 32.9%.
- The trajectory controller's S4 cells are unchanged (own history, labelled).
- The sweep now has the braking and learned-planner cells only (225 rows, was 350).

**Sign disagreement and opposite signs (Table 14, Figure 11; nuScenes S0, all).**

| quantity | OLD | NEW |
|---|---|---|
| braking, oracle geometry: the four perception gains disagree in sign with V | 34.2–38.0% | 25.0–32.6% |
| braking, monocular | 45.4–47.3% | 41.9–45.1% |
| learned planner | unchanged | unchanged |
| braking vs learned planner, both non-zero / opposite sign, oracle | 66 of 128 | 40 of 79 |
| same, monocular | 150 of 286 | 88 of 171 |

**Mechanism table (Table 13; nuScenes oracle braking).**
- Harmed frames: 97 → 50. Helped frames: 143 → 91.
- Mean FP gain, harmed vs helped: 1.76 vs 1.17 → 1.98 vs 1.18.
- Misses recovered, harmed vs helped: 0.77 vs 1.20 → 0.60 vs 1.40.
- Harmed frames with no miss recovered: 50.5% → 56.0%.

The mechanism reads more sharply.

**Cells on the test split (`benchmark_cells.csv`).** Harmed share of affected:

| cell | OLD | NEW |
|---|---|---|
| nuScenes oracle braking | 32.6% | 32.0% |
| nuScenes mono braking | 41.7% | 38.6% |
| KITTI oracle braking | 42.2% | 35.4% |
| KITTI mono braking | 47.0% | 44.7% |
| KITTI mono trajectory | 48.0% | 57.8% (ρ 1.427 → 1.457) |

The mean over the 14 cells is 37.4% → 37.1% of affected inputs and 8.3% → 7.3% of all inputs. Oracle@20 moves by at
most 0.7 points.

**Benchmark tables (test split, every signal, cell and quota).**
- `benchmark_table.csv`: 832 rows.
  - The gates move most. The largest move is `gate_gbm` on KITTI mono trajectory at 20%, 0.062 → 0.184.
  - Every other signal moves by at most 0.026 nDG.
  - Signals whose filtered paired interval against random excludes zero: 182 → 185 rows.
  - Three rows now beat random that did not: `gate_gbm` on nuScenes mono braking at 30% and 50%, and on KITTI mono
    braking at 50%. None reverses the other way.
- `benchmark_table_routers.csv`: 264 rows. The learned routers move most.
  - R2 on nuScenes mono braking at 20%: −0.080 → +0.215.
  - R1 GBM (classifier) on nuScenes oracle braking at 50%: 0.701 → 0.453.
  - R2's learned-planner heads also move (up to 0.21 nDG) although their labels did not change. R2 is one network
    with a head per cell, retrained on the new labels, so this is refit variability.
  - 13 rows change whether the filtered interval against random excludes zero (36 → 39 rows beat random): 8 become
    significant and 5 stop.
- `benchmark_table_nuplan_real.csv`: unchanged.
- Deployable gate (Table 3, nuScenes, 20%):
  - GBM gate: oracle braking 0.271 [0.146, 0.431] → 0.309 [0.146, 0.477]; mono braking 0.491 → 0.510;
  - linear gate: 0.208 → 0.211 and 0.366 → 0.369;
  - uncertainty and the planner rows: unchanged.
- Budgets:
  - two-level: largest nDG move 0.052, no row changes against random;
  - routers: largest move 0.33 (R2), 8 rows change against random (123 → 125 beat it);
  - multi-fidelity: largest move 0.008.

### Significance table

`scripts/162_significance_table.py` → `results/final/significance_table.csv`, `docs/significance_table.md`.

It covers every deployable signal × cell × quota: 540 rows over the ten core cells and the nuPlan real-perception
cells. The signals are the two gates, the four R1 routers, R2, uncertainty, CHEAP criticality and ego speed. Each row
has:
- the realised gain minus random's, in loss units and as a share of the all-cheap loss;
- its 95% paired unit-bootstrap interval over every draw (125, 1,000 draws, none dropped);
- next to it, the official nDG and its interval under the 25% prize filter.

Rows whose paired interval lies above / includes / below zero, OLD → NEW:

| signal | above random | includes 0 | below random |
|---|---|---|---|
| gate_ridge | 5 → 4 | 49 → 50 | 2 → 2 |
| gate_gbm | 9 → 10 | 47 → 46 | 0 → 0 |
| R1_mlp_reg | 6 → 9 | 50 → 46 | 0 → 1 |
| R1_mlp_clf | 7 → 9 | 49 → 47 | 0 → 0 |
| R1_gbm_reg | 11 → 11 | 39 → 39 | 6 → 6 |
| R1_gbm_clf | 13 → 11 | 42 → 45 | 1 → 0 |
| R2_cnn_clf | 0 → 0 | 34 → 39 | 6 → 1 |
| uncertainty | 1 → 1 | 31 → 31 | 8 → 8 |
| criticality_cheap | 1 → 1 | 48 → 48 | 7 → 7 |
| ego speed | 10 → 10 | 46 → 46 | 0 → 0 |

R2 and uncertainty have no nuPlan rows (40 rows each instead of 56).

### Harm and benefit by loss term

`scripts/163_loss_term_shares.py` → `results/final/loss_term_shares.csv`. For each term, its share of the total harm
(over V < 0) and of the total benefit (over V > 0), all inputs, shared history. Shares add up to 1 in each set.

| controller, setting | harm: main terms | benefit: main terms |
|---|---|---|
| braking, nuScenes oracle Y8 320→640 (50 harmed / 91 helped) | excess 85.3%, shortfall 12.7%, jerk 2.0%, collision 0% | shortfall 74.7%, excess 25.2% |
| braking, nuScenes mono (108 / 165) | excess 92.8%, shortfall 5.0%, jerk 2.2% | shortfall 56.9%, excess 42.4% |
| braking, KITTI mono Y8 320→640 (710 / 1,365) | excess 79.3%, shortfall 18.5%, jerk 1.6%, collision 0.6% | shortfall 92.0%, excess 7.6% |
| braking, KITTI oracle Y8 320→640 (319 / 977) | excess 96.0%, jerk 2.9%, shortfall 1.1% | shortfall 97.3%, excess 2.4% |
| braking, KITTI mono (384, 512, RT 320, RT 480) | excess 73–77%, shortfall 22–24%, jerk ≤ 1.2%, collision ≤ 1.2% | shortfall 13–72%, excess 27–86% |
| trajectory, KITTI mono (all five pairs) | collision 58–65%, clearance 33–36%, progress ≤ 2.4%, acceleration ≤ 4.4%, lateral ≤ 0.8%, switching ≤ 0.6% | collision 64–66%, clearance 36–37% |
| trajectory, KITTI oracle Y8 320→640 (14 / 449) | clearance 51.1%, collision 50.6%, acceleration −2.6% | collision 66.3%, clearance 37.4% |

Escalation harms the braking controller mostly by braking more than the true scene needs, after a FULL detection that
the CHEAP detector did not report. It helps mostly by closing a safety shortfall. Under the shared history the
action-change term (jerk, switching) carries at most 2.9% of the harm and 0.8% of the benefit.

### Loss definitions

`docs/iclr_loss_definitions.md` gives, from the code, every term, weight and threshold of:
- the braking controller;
- the trajectory controller (Planner B, the `static_obstacles` preset);
- the lateral controller;
- the nuPlan safety and scalar losses.

## Task 33: deterministic R1

### Code change

`scripts/103_routers_r1.py`, `fit_score`: the two gradient-boosted routers (R1_gbm_reg, R1_gbm_clf) are fitted and
scored inside `threadpool_limits(limits=1, user_api="openmp")`. This is the same mechanism, in the same place (around
fit and predict), as 159's `fit_predict`. The MLPs keep BLAS at its default. Nothing else changes: features, fit
units, seeds and hyperparameters are the same.

Because 120, 122, 124 and 129 call `fit_score`, their R1 refits are now deterministic too. 120 was not rerun: its
nuPlan scores come from its own shipped run.

OLD is the state before this change. The inventory is in `results/raw/20260925_155659_task33_before/inventory/`.

### Gates: all pass

| stage | gate |
|---|---|
| 122 | S1: 434/434 official nDG values reproduced |
| 129 | 980/980 comparisons reproduced |
| 124 | refit against official scores: max difference 0 over 64 rows; sanity 444/444 rows |
| 130 | sanity passed |
| 133 | M1–M3 12/12, 2/2, 3/3; S2–S4 ok |
| 149 | the V-trained rows reproduce `benchmark_table_routers.csv` (200 rows) |
| 159 part 1 | the refits on V, Q and G equal the cached scores exactly: 60 checks, max difference 0 |
| 159 part 2 | 80 seed checks, max difference 0 |

159's exact refit check, which the 8-thread fit missed by 8.9e-16, passes; no gate was changed.

### What the deterministic fit changed

Against the Task 32 routers_r1 run, which used 8 threads, the one-thread R1 scores are identical everywhere except 55 of
8,008 R1_gbm_reg scores on KITTI oracle braking, which move by at most 8.9e-16. Every rank the benchmark takes from
them is the same.

Inventory (164, `results/raw/20260925_155659_task33_before/inventory/`):
- **96 files unchanged**, byte for byte. This includes every nDG, interval, budget, significance and energy table.
- **2 files changed without a value changing:**
  - `benchmark_table_routers.csv`: the same 264 rows in another order (103 rewrote the R1 rows, so R2's rows now come
    first);
  - `streaming_controllers.csv`: only the name of the V1-scores run it records.
- **3 files changed in value:** Task 30's, now on the shared history (below).

**No printed-precision value** (3 decimals for nDG, 1 decimal for percentages) **and no significance call changed**
outside Task 30. `significance_table.csv` is byte-identical.

### Task 30 on the shared history (before: own history; after: shared history, deterministic R1)

The figure `results/final/figures/target_transform_heatmap.{pdf,png}` is regenerated. The planner cells are unchanged,
since their V carries no action history; the brake and trajectory cells move.

Pooled paired differences at 20 % (nDG, interval from the draw-wise mean over the ten cells):

| architecture | comparison | before | after |
|---|---|---|---|
| GBM | G − V | +0.088 [−0.018, +0.205] | +0.106 [−0.012, +0.214] |
| GBM | signed ECDF(V) − V | +0.023 [−0.033, +0.114] | +0.053 [−0.008, +0.138] |
| GBM | rank(V) − V | −0.029 [−0.123, +0.103] | −0.014 [−0.120, +0.128] |
| GBM | G − rank(V) | +0.116 [+0.005, +0.215] | +0.120 [−0.005, +0.219] |
| GBM | G − signed ECDF(V) | +0.065 [−0.041, +0.148] | +0.053 [−0.058, +0.128] |
| GBM | Q − V | −0.025 [−0.116, +0.106] | −0.007 [−0.101, +0.103] |
| MLP | Q − V | −0.135 [−0.226, −0.022] | −0.139 [−0.244, −0.021] |
| MLP | rank(V) − V | −0.127 [−0.198, −0.014] | −0.170 [−0.244, −0.048] |
| MLP | Q − rank(V) | −0.008 [−0.101, +0.054] | +0.032 [−0.058, +0.095] |
| MLP | G − V | −0.027 [−0.118, +0.071] | −0.030 [−0.121, +0.070] |
| MLP | G − rank(V) | +0.100 [+0.002, +0.181] | +0.140 [+0.032, +0.213] |

Pooled intervals whose call changes (all quotas, 80 comparisons):
- **GBM G − rank(V) at 20 %:** excluded zero before, includes it now.
- **MLP rank(V) − V at 30 %:** now excludes zero, on the negative side.
- **MLP Q − signed ECDF(V) at 30 %:** now excludes zero, on the negative side.
- **MLP Q − V at 50 %:** now excludes zero, on the negative side.

Per-cell intervals against raw V over the 40 cell-quota pairs, above / below zero:

| label | GBM before → after | MLP before → after |
|---|---|---|
| rank(V) | 3/1 → 3/1 | 0/4 → 0/8 |
| signed ECDF(V) | 5/0 → 5/0 | 0/2 → 0/1 |
| Q | 9/2 → 9/3 | 0/5 → 0/4 |
| G | 9/1 → 9/0 | 1/0 → 0/0 |

Binary targets (V > 0 classifiers against Q > 0 and G > 0). At 20 %:
- MLP: Q −0.051 → −0.067; G −0.086 → −0.101.
- GBM: Q +0.024 → −0.032; G −0.078 → −0.135.

All four still include zero. Per-cell intervals over all quotas:
- GBM G > 0: 5 below → 2 below;
- GBM Q > 0: 1 above and 2 below → 1 below;
- MLP Q > 0: 2 above → 2 above and 2 below;
- MLP G > 0: 1 below → 1 below.

**Task 30's readings, one by one.**
1. **GBM, "neither transform closes most of the gap between raw V and G".** Still holds, but more weakly.
   - At 20 % the signed ECDF of V recovers about half of G's pooled advantage (+0.053 of +0.106; before, a quarter).
     rank(V) still recovers none.
   - With the transform held fixed, G − rank(V) no longer excludes zero (+0.120 [−0.005, +0.219]), and G − signed
     ECDF still does not.
   - So no pooled comparison now separates G's objective from a transform of V on the GBM router. "The objective
     difference persists for G" is no longer supported by an interval.
   - G's own pooled advantage over raw V still spans zero at every quota.
2. **GBM, Q.** There is still no pooled gap to explain (−0.007 at 20 %). Q and G on the four nuScenes planner cells
   are unchanged (+0.16 to +0.42 at 20 %), and so is the signed ECDF's gain there (+0.09 to +0.22).
   - The brake and trajectory cells move. rank(V) on KITTI mono trajectory now loses 0.685 (was 0.551).
   - On nuScenes oracle brake, G − V is now +0.252 (was +0.054) and rank(V) − V is +0.134 (was −0.190).
3. **MLP, "the transform explains Q's deficit".** Holds. Q − V is −0.139 [−0.244, −0.021] and rank(V) − V is
   −0.170; Q − rank(V) is +0.032 [−0.058, +0.095].
4. **MLP, G.** It is null against raw V at every quota. It sits above rank(V) at 20 % (+0.140 [+0.032, +0.213]) and
   not at the other quotas: at 10 % the interval's lower end is −0.000.
5. **Seeds (part 2).**
   - The gradient-boosted models are still seed-independent: min = median = max for every GBM cell. The seed-0 refits
     reproduce the cached scores exactly (80 checks).
   - Before, one shipped win was flagged as not robust (a win for fewer than 4 of 6 seeds): R1_mlp_reg on nuScenes mono
     brake. It is still flagged (2 of 6).
   - Three more shipped wins are flagged now:
     - R1_mlp_clf on KITTI oracle brake: 2 of 6 (was 5 of 6);
     - R1_mlp_clf on KITTI mono brake: a new shipped win, 3 of 6;
     - R1_mlp_reg on KITTI mono brake: a new shipped win, 2 of 6.
   - The MLPs' seed spread within a cell reaches 0.28 (R1_mlp_clf, nuScenes oracle brake; was 0.24) and 0.24 (R1_mlp_clf,
     nuScenes mono brake). Before, the largest were 0.27 and 0.38.

## Task 34: the records that read the regenerated tables in full

These records read the regenerated tables in full; they were regenerated after them, with their original procedures.

| record | stages | gates | result |
|---|---|---|---|
| C27, lift offset | 140, 141 | 14/14 settings reproduce to 3 decimals | reading still "sensitive"; KITTI mono RT-DETR-l 480→640 on the trajectory controller joins the list of settings (harmed −0.048, ρ +0.115) |
| C26, causal ego speed | 125, 127, 52, 53 (centred, drop-first, causal), 139 | sanity and C4 3/3; `52 --rows_only` equals the full table exactly | C26 run re-expressed on these tables |
| C25, class-error fix | 146, then the `prefix` run variant of 52, 62 ×2, 92, 102, 122, 125, then 137 | `results/final` restored exactly; C4 11/11 files unchanged outside nuScenes | C25 run re-expressed on these tables |
| C28, the ego-frame inventory | 147 | — | NEW is now the shared-history state; `AFTER_TASK_23` also leaves out the Task 32–35 files |
| C32, gate G1′ | 150 `g1` | 28,096 compared values, 0 differing, 16 structural (as before) | passes |
| C33, the example submissions | two examples | — | rescored |
| C34, evidence pack and claims check | 152 | — | 9 of 33 statements pass (was 11) |

In the claims check, two statements of the first manuscript that passed now fail:
- H1, "35–52% of affected inputs": the monocular rows now span 34.2–53.8%.
- S2, "29–45% under the per-mode operating points": braking now spans 25.3–46.2%.

Every report whose figures depend on the braking or trajectory controllers carries an "Action history" note; the
generated tables are current.

## Task 35

Loss-weight sensitivity and the Figure 1 frames: `docs/iclr_loss_sensitivity.md`.
