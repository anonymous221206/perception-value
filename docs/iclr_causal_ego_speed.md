# A causal ego speed on nuScenes: the defect, the fix, and every figure it moves

Pre-registration: the pre-registration record (not part of this release), 2026-09-20 11:05 (Task 22 Part B), committed before the code was written,
with one amendment at 11:10 (the length of the window) and one at 11:20 (a second stale Phase 0E artefact), both
logged before any figure was read. Every number below is a row of `results/final/causal_ego_speed.csv`.

## The defect

`NuScenesDB.ego_speeds` differentiated the ego poses with a **centred** difference,

    v[i] = ||pos[i+1] - pos[i-1]|| / (t[i+1] - t[i-1]),

so the speed attributed to frame *i* read the pose half a second **in the future**. Ego speed is used as an
allocation signal (`trivial_ego_speed`, and four of the core matrix's trivial heuristics), and a signal computed at
frame *i* may only use what is known at frame *i*.

**The other two tracks were checked, not assumed:**

| track | where the speed comes from | causal? | check |
|---|---|---|---|
| KITTI | OXTS forward velocity `vf` of that frame (`kitti.load_oxts(seq)[:, 8]`) | yes | B2: the shipped `v_ego` equals `vf` of the same frame on **8,008 / 8,008** frames |
| nuPlan | `ego.dynamic_car_state.rear_axle_velocity_2d.x` at the current iteration | yes | B3: present on all 1,440 states, registered with source `ego_state` |
| nuScenes | centred difference of ego poses | **no** | — |

## The fix

`NuScenesDB.ego_speeds_causal` is a backward difference over the previous 0.5 s, from past and current poses only:
`v[i] = ||pos[i] - pos[j]|| / (t[i] - t[j])` with `j < i` the past keyframe whose gap is closest to 0.5 s.

**The rule for a scene's first frame:** frame 0 has no earlier pose in the log, so its causal speed is **0** — the
only value that uses no future data. That is 85 of 3,376 frames, one per scene.

*Amendment, logged before any figure was read.* The pre-registration said "the largest `j` with
`t[i] - t[j] >= 0.5 s`, and keyframes are 2 Hz, so `j = i - 1`". The second half is wrong about the data: measured
keyframe gaps are 0.40–0.65 s (median 0.500) and **63% of them are just under 0.5 s**, so that rule reached back
two keyframes on 2,063 of 3,291 frames and measured a full second. The nearest-to-0.5 s rule gives `j = i - 1` on
every frame of this data, which is what the definition meant.

How far the two estimates differ, over the 3,376 frames: mean 6.089 m/s (centred) against 5.939 m/s (causal), mean
absolute difference 0.263 m/s — 0.119 m/s excluding the 85 first frames — Spearman 0.950, and 97 frames differ by
more than 0.5 m/s.

## What is deliberately not changed

The braking and lateral controllers take `v_ego` as the vehicle's **own state** when they compute J for *both*
fidelities. That is the environment, not the allocator: CHEAP and FULL see the same number, so
V = J(CHEAP) − J(FULL) does not move, and no decision value, gain, prize or nDG denominator moves with it. For the
environment the centred difference is also the more accurate estimate of the instantaneous speed at frame *i*; for
the allocator, causality is what matters. The decision tables keep the shipped `v_ego` and the signals read the new
column. This was stated in the pre-registration, before any figure was seen.

Ego speed is **not** a gate feature on this track: `rap.features` registers no `ego_state` feature for KITTI or
nuScenes (that source exists for the nuPlan real-perception gate only), so no gate, router or perception gain moves.

## Checks

| check | what it asserts | result |
|---|---|---|
| sanity gate | the **centred** column through the new plumbing reproduces the shipped files | `statistics_hardening.csv`, `objective_swap.csv`, `core_matrix.csv` all **byte-identical** |
| C4 | nothing outside nuScenes moves | 0 rows changed outside nuScenes in all three files |
| B1 | the causal value is the backward difference over the previous keyframe | 0 frames deviate; 85 coincide with the centred value (the last frame of each scene, where the centred formula degenerates to a backward difference) |
| B2, B3 | KITTI and nuPlan use no future data | as in the table above |

A fourth file, `statistical_tests.csv`, is deliberately **not** rewritten; see "A stale artefact" below. It is a
Phase 0E by-product that this release does not ship, and the evidence for section 5 is the pair of 53 runs kept in
the Part B run directory.

## OLD → NEW

### 1. Held-out nDG@20% of ego speed on the six nuScenes cells (C15)

| cell | ego speed, old | ego speed, new | random (unchanged) |
|---|---|---|---|
| oracle · q_brake | +0.082 | **+0.052** | +0.083 |
| mono · q_brake | +0.026 | **−0.003** | +0.071 |
| oracle · q_plan (ADE) | −0.023 | −0.018 | +0.017 |
| mono · q_plan (ADE) | −0.012 | −0.003 | +0.093 |
| oracle · q_plan (FDE) | +0.044 | +0.038 | +0.055 |
| mono · q_plan (FDE) | −0.014 | +0.006 | +0.069 |

Ego speed was below random in all six cells and still is. The causal signal is **worse** on both braking cells,
where the non-causal version had been drawing on the future pose, and marginally better on three planner cells.

### 2. Ego speed against random on realised gain, every draw kept (C15)

| statistic | old | new |
|---|---|---|
| rows where ego speed beats random | 3 | 3 |
| which rows | KITTI oracle q_traj @30%, @50%; PDM-Closed scalar_J @50% | unchanged |
| rows won at the 20% quota | 0 | 0 |

Unchanged, and necessarily so: all three wins are on tracks whose ego speed was already causal.

### 3. The objective-swap headline (C17), 13 defined cells × 4 budgets

| | `E_perc_dE` old → new | `E_perc_risk` old → new |
|---|---|---|
| argmax differs | 42 / 52 → **42 / 52** | 40 / 52 → **40 / 52** |
| Kendall tau between the rankings, median | +0.171 → +0.171 | 0.000 → **+0.014** |
| selection regret, median nDG | −0.104 → −0.104 | −0.122 → −0.122 |
| selection regret, median share of the all-cheap loss | −0.008 → −0.008 | −0.007 → −0.007 |
| the `E_perc` winner is worse than random on decision value | 16 / 52 → 16 / 52 | 22 / 52 → 22 / 52 |
| regret intervals excluding zero | 4 / 52 → 4 / 52 | 0 / 52 → 0 / 52 |

Every headline count is unchanged; one median moves in the third decimal.

### 4. The trivial heuristics of the core matrix

| row | best trivial heuristic | its nDG@20%, old → new |
|---|---|---|
| nuScenes oracle, longitudinal | ego speed (unchanged) | +0.242 → **+0.227** |
| nuScenes mono, longitudinal | speed + detections (unchanged) | +0.209 → +0.215 |
| nuScenes oracle / mono, lateral | n detections (unchanged) | unchanged |

The best trivial heuristic keeps its identity in all four rows.

### 5. The per-sequence ego-speed policy (53, run copies only)

Median per-sequence nDG stays 0.000 in all four nuScenes rows; one row's count of sequences above 0.5 goes from 4
to 5 (oracle, longitudinal), and the Wilcoxon p-values move in the fifth decimal. All 12 KITTI ego-speed rows are
identical.

### 6. Sensitivity: a scene's first frame dropped instead of zeroed

Re-running every stage with the 85 first frames put **out of reach of any quota** instead of given a speed of 0
reproduces every figure above **to the last digit** — all six nDG values, all six objective-swap statistics per
variant, and all four trivial-heuristic rows. The rule for the first frame cannot carry any result, because a speed
of 0 already puts those frames at the bottom of the ranking.

## A stale artefact, found by the sanity gate

The gate compared the centred column through the new plumbing against the shipped files. Three matched exactly. The
fourth, `statistical_tests.csv`, did not — but **not in the ego-speed rows**: all 16 `ego speed` rows
and all 16 `criticality` rows reproduce exactly, while 14 of 16 `perception oracle` rows differ, on KITTI as much as
on nuScenes. That policy reads `dE`, which this part does not touch. The file is a Phase 0E artefact last written by
an older `53_finalize.py` run and has drifted from the current tables — the same thing found in
`headline_table.csv` earlier the same day. Neither file is part of this release: no stage reads them.

It is therefore left untouched: regenerating it would silently move 14 KITTI rows this part does not own, and its
`holm_p` column is a Holm adjustment over every row jointly, so even a partial splice would move the rest. The two
53 runs are kept in the Part B run directory as the evidence for section 5, and the staleness of both Phase 0E
artefacts is reported as a finding rather than fixed here.

## What this changes for the paper

**Nothing in any reading.** Ego speed was below random on every nuScenes cell before and still is; it is now
further below on braking, which is the honest direction — part of the little it had was borrowed from the future.
The objective-swap conclusion, which rests on `trivial_ego_speed` winning 34 of 52 `E_dec` comparisons, is
unchanged in every count.

## Runs and code

| artefact | run |
|---|---|
| causal speed table, B1–B4, pre-change copies, sanity and sensitivity outputs | `20260920_110647_causal_ego_speed` (`nusc_v_ego_causal.csv`, `checks.csv`, `before/`, `sanity_centred/`, `sensitivity_drop_first/`) |
| the causal signal in the official files | `results/final/statistics_hardening.csv` (stage C15), `objective_swap.csv` (C17), `core_matrix.csv` |
| figures, sanity and C4 | `results/final/causal_ego_speed.csv`; stage **C26** recomputes all three |

Code: `src/rap/egospeed.py` (the helper and the `--ego_speed` switch), `NuScenesDB.ego_speeds_causal`,
`scripts/138_causal_ego_speed.py` (stage **N6**), `scripts/139_causal_ego_figures.py` (stage **C26**), and the
`--ego_speed` flag on 52, 53, 125 and 127. Every run used `PYTHONHASHSEED=0`.
