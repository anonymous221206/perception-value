# The monocular lift in the ego frame: what moved, and what the conclusions now read

> **Action history (Task 32).** The NEW column of `results/final/ego_frame_convention*` is now the current release, which also uses a shared action history for decision values; the figures quoted in this report are the ego-frame values before that change. `docs/iclr_shared_history.md` gives what the shared history moved.

Task 23 (pre-registered, with its amendment, before any code of the change was written; the pre-registration record is not part of this release). Every figure below is a row
of `results/final/ego_frame_convention.csv` (the per-row detail of groups 7 and 8 is in
`ego_frame_convention_detail.csv.gz`; the readings and the file inventory in `ego_frame_convention_reading.json`),
built by `scripts/147_ego_frame_convention.py` from the snapshot of `results/final` taken before any Task 23 run
(OLD, camera frame, as shipped) and the regenerated `results/final` (NEW, ego frame).

## Summary

* The monocular lift now reports geometry in the ego frame: the camera → ego rotation is applied inside the ground
  cue, and the fused point and the box corners are expressed in the ego frame. Until now the camera-frame lift was read
  as ego-frame geometry by every downstream system.
* Every gate passed before any figure was read (G1–G5), and the two amendment measurements (B, D) came out the way
  that keeps the comparison like for like.
* Six of the eight registered readings hold. **Two reverse**, and are restated below: **Q3** (no deployable allocator
  reliably beats random on nuScenes) and **Q7** (the KITTI Y8 320→640 cells carry little harm mass).
* The largest single move is in the KITTI Planner B cells: the benefit of running FULL collapses (all-FULL loss
  reduction on KITTI mono Y8 320→640 q_traj 0.424 → 0.131; on Y8 512→640 it turns negative, 0.047 → −0.006, with
  ρ 0.435 → 1.109). The nuScenes cells move the other way: harm mass falls (ρ −0.12 to −0.18 on q_brake).

## The convention

`rap.mono.predicted_geometry(..., cam_to_ego=(R, t))`: the corrected horizon D = n₀·(f_y/f_x)·(u_c − c_x) + n₁·(y2 − c_y)
+ n₂·f_y (n = Rᵀ(0, 0, −1)) replaces y2 − c_y in the ground cue and in the fusion weight; the height cue is unchanged;
the fused point and the two bottom corners are expressed in the ego frame (x forward, y left); ttc is unchanged.
Per-unit transforms are tabulated once (`data/cache/cam_to_ego.json`: KITTI cam2 1.08–1.14 m ahead of the IMU,
nuScenes CAM_FRONT 1.70–1.72 m ahead of the rear axle). One switch, `rap.frames` (default `ego`; `--frame camera`
reproduces what shipped); the detection caches keep their camera-frame arrays and are re-lifted on read.

## Gates, before any figure

| gate | what | result |
|---|---|---|
| G1 geometry | identity transform re-lifts every cached box of the 11 official caches | 2,008,908 detections, 0 fields differing bitwise |
| G1 tables | identity frame regenerates every per-frame table | byte-identical: 8/8 core-matrix, 6/6 Planner B, 22/22 realism, 106/106 calibration outcomes, 4 + 4 submissions |
| G2 | rotation = identity with Task 22's translations | 16/16 Task 22 tables byte-identical; `lift_offset_sensitivity.csv` recomputes byte-identically |
| G3 | ttc before and after | never differed, on every detection touched |
| G4 | fn, fp, loc, cls, crit_fn, n_det, n_gt and detection counts, camera vs ego | identical in 114/114 tables |
| G5 | oracle geometry moves only through unmatched detections | 71/71 gated tables: 0 differing frames outside touched frames and their carried state |

## B — the rotation applied to the fused point instead

Applying the full SE(3) to the fused point (F) instead of rotating the ground cue moves the geometry by metres beyond
15 m (p99 |Δx|: KITTI 1.37 / 6.98 / 16.27 m and nuScenes 1.50 / 5.48 / 14.58 m in the 0–15 / 15–30 / 30–50 m bands;
p99 |Δlateral| up to 9.2 m). Every band is material, so the registered revisit ran: against the matched reference, the
chosen lift has the smaller median |range error| in all 12 band × fidelity checks (e.g. KITTI 30–50 m CHEAP 4.18 vs
6.06 m, nuScenes 30–50 m CHEAP 6.24 vs 9.13 m). Decision 1 stands.

## D — the near-range bias, and whether it is symmetric between fidelities

The ego frame exposes the lift's own near-range bias, which the camera frame had cancelled: the camera's range was
measured from the camera and read as range from the ego origin, short by the camera's forward offset, while the bottom
edge of a loose 2D box lands on the wheel contact rather than the near face. Signed bias against the reference, 0–15
m: **+1.16 to +1.31 m** in the ego frame for every pair and both fidelities, against +0.23…+0.34 m (KITTI) and
−0.35…−0.43 m (nuScenes) in the camera frame; it grows with range (+1.8…+2.4 m at 15–30 m). Median |range error| is
worse in the ego frame at 0–15 m (KITTI ≈ 1.25–1.34 vs 0.76–0.80 m) and better beyond (KITTI 30–50 m 4.2–4.8 vs
6.2–6.8 m; nuScenes 6.3–6.8 vs 9.6–9.9 m). It is reported, not corrected (a second modelling change, tuned on the same
data, is out of scope).

The registered test is symmetry: |median bias(FULL) − median bias(CHEAP)| ≤ 0.25 m in every band ≤ 30 m with ≥ 100
paired objects. It holds on all 12 tested pair × band rows (largest 0.115 m), so CHEAP and FULL are compared like for
like. Beyond 30 m the asymmetry reaches 1.0–2.6 m in some pairs (untested by the rule; up to 1.8 m in the camera
frame too).

## G5 — how much of "oracle geometry" is the lift's

A frame is touched when a kept detection, in either fidelity, has no reference box at IoU ≥ 0.5; its geometry is then
the lift's in both frames. At the operating point **72.9% of KITTI and 70.9% of nuScenes frames are touched**
(16.7–90.7% across the calibration thresholds). In the oracle submissions, 2,654 of 8,639 CHEAP and 5,578 of 14,104
FULL boxes are lifted, and only those moved. The oracle cells are reference geometry for matched objects only — in both
frames.

## OLD → NEW

### 0. The registered reading

| | reading | OLD | NEW | holds |
|---|---|---|---|---|
| Q1 | harmed ≥ 20% and ρ ≥ 0.20 in every nuScenes and KITTI moderate-gap cell, S0–S3 | min harmed 0.383, min ρ 0.208 | min harmed 0.358, min ρ 0.342 | yes |
| Q2 | nuScenes oracle: sign disagreement ≥ 25%, Spearman < 0.2 for every gain, S0–S3 | min 0.292, max ρ 0.101 | min 0.295, max ρ 0.082 | yes |
| Q3 | no deployable allocator beats random on nuScenes at 20% (paired 95% CI) | none | **3 rows** | **no** |
| Q4 | target swap: every pooled interval contains zero, three labels | all | all | yes |
| Q5 | causal threshold B − C straddles −0.05; no streaming controller recovers B | −0.089 [−0.142, −0.024]; D, E do not recover | −0.064 [−0.128, +0.014]; D, E do not recover | yes |
| Q6 | at 50% ms, uniform FULL stronger in ≥ 8 of 14 cells | 11 of 14 | 11 of 14 | yes |
| Q7 | KITTI Y8 320→640 cells: ρ ≤ 0.2 under S0–S3 | max ρ 0.192 | **max ρ 0.341** | **no** |
| Q8 | objective swap: argmax differs in ≥ 26 of 52, per variant | 42 / 40 | 43 / 40 | yes |

**Q3 reverses**: nuScenes mono brake at 20% — `gate_gbm` nDG 0.397, paired interval against random [+0.028, +0.604];
`R1_mlp_reg` 0.309 [+0.024, +0.419]; `R1_gbm_reg` 0.380 [+0.060, +0.480]. In the camera frame no nuScenes row at
20% cleared random.

**Q7 reverses**: KITTI mono Y8 320→640 q_traj, ρ 0.062 → 0.276 (S0), 0.34 (S1, S2), 0.278 (S3); harmed 0.352 → 0.419.
The q_brake and oracle cells of the same pair stay below 0.2 (0.169, 0.081, 0.004).

Q5 still reads *inconclusive* by its registered rule, but its interval now reaches past zero: the causal cap's cost is
no longer distinguishable from zero at 95%. The three cells where an allocator beats uniform FULL at 50% ms changed
(OLD: nuPlan IDM scalar_J, nuScenes oracle brake, nuScenes oracle plan_fde; NEW: KITTI mono traj, nuPlan IDM scalar_J,
nuScenes oracle plan_ade), while the count stayed 11 of 14.

### 1. Calibration cells (the five sign-variation quantities; S0, all units)

| cell | affected | harmed | ρ | all-FULL reduction | oracle@20 reduction |
|---|---|---|---|---|---|
| nuScenes oracle q_brake | 295 → 240 | 0.485 → 0.404 | 0.602 → 0.421 | 0.100 → 0.142 | 0.250 → 0.246 |
| nuScenes oracle q_plan | 1226 → 1179 | 0.514 → 0.509 | 0.824 → 0.704 | 0.008 → 0.013 | 0.046 → 0.044 |
| nuScenes mono q_brake | 486 → 454 | 0.455 → 0.423 | 0.535 → 0.412 | 0.130 → 0.166 | 0.280 → 0.283 |
| nuScenes mono q_plan | 1531 → 1612 | 0.516 → 0.493 | 0.722 → 0.713 | 0.015 → 0.016 | 0.052 → 0.053 |
| KITTI mono Y8 320 q_traj | 1355 → 957 | 0.352 → 0.419 | 0.062 → 0.276 | 0.424 → 0.131 | 0.452 → 0.180 |
| KITTI mono Y8 320 q_brake | 3026 → 2931 | 0.407 → 0.375 | 0.192 → 0.169 | 0.502 → 0.520 | 0.620 → 0.624 |
| KITTI mono Y8 384 q_traj | 920 → 697 | 0.454 → 0.438 | 0.211 → 0.456 | 0.178 → 0.060 | 0.226 → 0.111 |
| KITTI mono Y8 384 q_brake | 2646 → 2606 | 0.446 → 0.406 | 0.551 → 0.473 | 0.163 → 0.196 | 0.363 → 0.372 |
| KITTI mono Y8 512 q_traj | 624 → 446 | 0.407 → 0.478 | 0.435 → **1.109** | 0.047 → **−0.006** | 0.084 → 0.052 |
| KITTI mono Y8 512 q_brake | 2325 → 2266 | 0.457 → 0.430 | 0.679 → 0.618 | 0.080 → 0.098 | 0.249 → 0.256 |
| KITTI mono RT 480 q_traj | 730 → 492 | 0.414 → 0.453 | 0.548 → 0.578 | 0.037 → 0.030 | 0.082 → 0.070 |
| KITTI mono RT 480 q_brake | 3091 → 3014 | 0.445 → 0.427 | 0.805 → 0.721 | 0.059 → 0.087 | 0.304 → 0.309 |
| KITTI oracle Y8 320 q_traj | 779 → 637 | 0.095 → 0.042 | 0.004 → 0.004 | 0.690 → 0.690 | 0.692 → 0.692 |
| KITTI oracle Y8 320 q_brake | 1846 → 1753 | 0.341 → 0.309 | 0.091 → 0.081 | 0.675 → 0.682 | 0.743 → 0.742 |

43 of the 70 cell × scheme rows (S0–S4) move beyond Task 22's tolerance (harmed by more than 5 points or ρ by more
than 0.05); all of them are listed in the CSV. The pattern holds across schemes: every KITTI mono q_traj row gains
harm mass (ρ +0.03 to +0.73, 20 rows) and loses benefit; the q_brake rows lose harm mass or barely move (ρ −0.18 to
+0.03, 35 rows); nuScenes oracle q_plan loses it (ρ −0.11 to −0.15), while nuScenes mono q_plan barely moves at S0
(−0.01) and gains under S1–S4 (+0.03 to +0.08).

### 2. nDG of every deployable signal (test split, paired 95% interval against random)

Rows whose interval lies above / below random, out of all deployable rows (benchmark table and routers):

| track | quota | above: OLD → NEW | below: OLD → NEW |
|---|---|---|---|
| KITTI (36 rows) | 10 / 20 / 30 / 50% | 8/7/10/11 → 7/8/11/12 | 2/3/2/3 → 4/4/3/3 |
| nuScenes (54 rows) | 10 / 20 / 30 / 50% | 1/0/2/4 → 0/3/3/1 | 0/4/3/2 → **7**/6/2/2 |
| nuPlan (32 rows) | 10 / 20 / 30 / 50% | 0/3/4/2 → 0/3/4/2 | 7/2/0/0 → 7/2/0/0 |

On the test split 52 deployable rows change side: 13 wins appear, 11 disappear, and 28 move between "includes random"
and "below random". nuScenes gains its first wins at 20% (Q3) and, at the same time, seven rows at 10% now fall below
random (R2 on both brake cells, R1_gbm_reg on three of the four plan cells, gate_ridge on mono plan_fde,
criticality_cheap on oracle plan_ade). Every
side change is a row of the CSV with the note *win appears*, *win disappears* or *interval changes side*.

### 3. Diagnostic signals (not deployable)

Rows above random on test: KITTI 53 → 72 of 160, nuScenes 3 → 13 of 288, nuPlan 23 → 23 of 48 (unchanged: no lift).

### 4. `core_matrix.csv` (every row and column in the CSV)

Longitudinal cells barely move (η perception-oracle@20 within ±0.04; harmful-FULL rate down 0.8–1.7 points in every
cell). The lateral task moves more: KITTI oracle lateral η perception-oracle@20 −0.278 → −0.650 and the best trivial
heuristic 0.236 → 0.002; nuScenes mono lateral η perception-oracle@20 −0.077 → +0.309; KITTI mono lateral
action-change rates fall by 28–40% (e.g. Y8 320 0.096 → 0.067).

### 5. nuScenes oracle sign agreement per gain (S0–S3)

| gain | sign disagreement | Spearman | P(V < 0 \| gain > 0) |
|---|---|---|---|
| exact FN | 0.352–0.521 → 0.324–0.511 | −0.017…+0.056 → −0.004…+0.060 | 0.029–0.324 → 0.022–0.310 |
| FN + FP | 0.292–0.508 → 0.295–0.523 | −0.011…+0.101 → −0.027…+0.082 | 0.027–0.319 → 0.020–0.314 |
| E5 combined | 0.335–0.499 → 0.347–0.519 | −0.004…+0.099 → −0.021…+0.081 | 0.024–0.272 → 0.019–0.273 |
| E risk | 0.370–0.525 → 0.333–0.518 | −0.017…+0.064 → −0.015…+0.074 | 0.033–0.334 → 0.023–0.326 |

### 6. Mechanism table

The braking decision: frames where FULL helps 152 → 143, harms 143 → 97, unaffected 3,081 → 3,136. On harmed frames
FULL still adds detections (mean Δn_det +2.94 → +2.44) and false positives (mean ΔFP +2.12 → +1.76; share with more
FPs 0.78 → 0.72). The Planner C self-control moves little (helped 861 → 835, harmed 627 → 578).

### 7. Measured budgets, energy, streaming and skip accounting

No latency or energy was re-measured; the allocators' overheads and the detectors' costs are the shipped ones, so
feasibility is unchanged in every table (the same rows feasible, cell by cell). Rows above random move by at most 3
per unit × level (e.g. routers table, ms at 50%: 22 → 20; at 30%: 15 → 18). Of the 39 energy-module claims, one
changes: "c | all_rails | energy evaluation rows: wins / losses / rows" no longer holds (the reading that R2 does not
beat random in any energy row still holds). Streaming at 20%, pooled over the learned signals: B − C −0.100
[−0.179, −0.015] → −0.087 [−0.170, +0.006], D − C −0.083 [−0.153, −0.007] → −0.064 [−0.141, +0.022]; both D and E
still read "does not recover" by the registered rule.

### 8. Controls

* Statistics hardening, raw-gain rows beating random (all draws kept) out of 308: 31/31/41/49 → 37/43/61/57 at
  10/20/30/50%; leave-one-unit-out median nDG range at 20% unchanged (0.166). Harm share among affected test inputs:
  nuScenes oracle brake 0.468 → 0.326, KITTI mono traj 0.402 → 0.480, KITTI oracle traj 0.172 → 0.096.
* Consumer transfer, off-diagonal entries beating random at 20%: KITTI mono 4 → 0, KITTI oracle 9 → 6, nuScenes mono
  2 → 1, nuScenes oracle 2 → 0; the diagonal on nuScenes mono 0 → 3.
* Objective swap: argmax differs 42 → 43 of 52 (E_perc_dE), 40 → 40 (E_perc_risk).
* Target swap: every pooled interval still contains zero under all three labels (architecture-driven). Primary label,
  mean V − G: gate_ridge 0.123 → 0.160 [−0.038, +0.270], R1_mlp_clf 0.142 → 0.078 [−0.092, +0.158].

### 9. Every file of `results/final`

65 files byte-identical (every nuPlan-only file, the Planner D seeds, the calibration PR curves, the shipped overheads
and `run_manifest.csv`); 38 CSVs with moved values (rows moved and the largest change per
file in `ego_frame_convention_reading.json`); 29 figures and JSON files whose bytes differ; `fig_bev_objects.csv.gz`
and `lift_offset_sensitivity.csv` with a new structure (the BEV objects in range change; C27's columns are now named
by frame); 9 of the gallery's 12 frames are new (the V ranking moved; the 9 frames they replace are kept in the
snapshot). Nothing listed as moved is a nuPlan file.

### C25, C26 and C27, re-expressed on the ego-frame tables

* **C25** (the nuScenes class-error fix): its OLD → NEW now substitutes the pre-fix class labels into the ego-frame
  tables (`146`, run variant `prefix`); C1–C3 78/78, and C4 holds — outside nuScenes 11/11 files unchanged. 119 figures
  in `class_error_fix.csv`.
* **C26** (the causal nuScenes ego speed): centred vs causal on the ego-frame tables; 104 figures in
  `causal_ego_speed.csv`. Against the shipped camera-frame values, the reported (causal) ego-speed signal now beats
  random at 20% on nuScenes oracle brake (nDG 0.052 → 0.208, random 0.147) and mono brake (−0.003 → 0.148, random
  0.130); in the camera frame it beat random on neither. Centred → causal still lowers it on oracle brake (0.240 →
  0.208) and raises it on mono brake (0.099 → 0.148).
* **C27** is restated as the cost of the old convention: the same 14 settings, ego frame as base, camera frame as the
  alternative. Reading: sensitive in 11 of 14 settings; the largest moves are the KITTI q_traj cells (ρ −0.674 on
  Y8 512→640, −0.444 on RT-DETR 320→640) and nuScenes q_brake (ρ +0.181 oracle, +0.123 mono).

## What this changes

* **Q3 is withdrawn in its camera-frame form.** "No deployable allocator reliably beats random on nuScenes" was an
  artefact of the camera-frame convention on the braking cell: in the ego frame the gradient-boosted gate and both R1
  regressors beat random on nuScenes mono brake at 20%. The claim is restated as: on nuScenes, deployable allocators
  beat random only on mono brake at 20% (3 of 54 rows), and more rows fall below random at 10% than before (7).
* **Q7 is withdrawn.** The KITTI Y8 320→640 Planner B cell carries substantial harm mass in the ego frame (ρ 0.28–0.34
  under S0–S3); "little harm mass" holds for its braking and oracle cells only.
* The KITTI Planner B cells now show that running FULL helps Planner B much less than the camera frame suggested, and on
  the Y8 512→640 pair not at all. Every statement about KITTI q_traj benefit is restated from the ego-frame tables.
* Q1, Q2, Q4, Q5, Q6 and Q8 hold, with the moves above; the harm result (Q1) is stronger on KITTI moderate-gap cells
  and weaker on nuScenes q_brake.
* A reversal is never resolved by returning to the camera frame: the ego frame is the reported convention.

## Not re-measured, and not regenerated

Detector and allocator latency and energy (the lift adds ≈ 50 flops per box and is in no shipped timing). Not
regenerated, with the reason recorded before any figure: the Planner D seeds (GT-input training, byte-identical), the
Phase 0F figures of 63, `run_manifest.csv`, and every nuPlan-only file (no `rap.mono`; all byte-identical). The
downstream stages were not re-run in the camera frame; their inputs are the per-frame tables G1 proves identical, and
the six secondary files re-run in the camera frame (75, 84, 86, 123) reproduced the shipped files byte for byte.

## Process notes

The regeneration stopped seven times, never on a gate; each stop and its fix is in the pre-registration record: a transient CUDA
out-of-memory at start-up of a Planner C chunk (resumed; two chunks retried once); 133's check S comparing unflagged
with flagged budget tables (both sides flagged now; the camera-frame check S reproduces its shipped file byte for
byte), and 128 ordered after 133; two frame-blind run globs (122, 110); a print bug in 118 after its outputs were
written; figure 13 missing its cost ratios (54 attaches them as 53 computes them); and the run variant not reaching a
run listing in 125 (`rap.runs.frame_runs`). The routers table had lost nuPlan's "n/a" geometry through a lossy
re-read; 103 and 107 now merge losslessly and 103 was re-run (identical scores, identical rows).

## Runs and code

`scripts/142`–`147`; `src/rap/frames.py`, `src/rap/cache.py`, `src/rap/mono.py`, `src/rap/runs.py`. Gate runs:
`results/raw/*_ego_frame_gate_{g1geo,g2,b,d,tables,g4,g5}`; the audit behind the pre-registration
`20260921_145440_frame_transform_audit`; snapshot `20260921_152927_ego_frame_before` (this release's `results/final`
as first shipped); every ego-frame run is tagged `_ego`. `reproduce.py --tier cached --only C28` rebuilds the
comparison; `--frame camera` on any stage selects the camera-frame convention again.
