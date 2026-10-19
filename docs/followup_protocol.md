# Follow-up experiments: protocol

Written 2026-10-04, before any decision value of any allocator, and before any setting-level quantity, was computed
by this repository on the 765 nuScenes scenes outside the benchmark's frozen 85. The commit that adds this file is
recorded in the follow-up index released with the results; analyses that the protocol does not list are secondary and labelled so wherever
they are reported. Deviations are logged, with date and reason, in that index.

Scope: E1 (the nuScenes track on all 850 trainval scenes) and E5 (escalation across detector capacity, architecture
and fine-tuning) in full; E2, E3, E6 and E7 are fixed in outline in Section 9 and in detail in their own addendum,
committed before their first decision value.

## 1. Data and split

**Scenes.** nuScenes v1.0-trainval, CAM_FRONT keyframes (2 Hz), all 850 scenes.

**The frozen 85 scenes** are the benchmark's nuScenes track (45 train / 16 validation / 24 test;
`configs/benchmark_splits.json`). They were not selected by content: they are the scenes whose keyframes are in the
first of the ten trainval keyframe archives (`v1.0-trainval01`), the archive downloaded when the track was built.

**The 765 new scenes** are split by the benchmark's own rule (`configs/benchmark_splits.json`, "rule"): sort the units
by (map location, scene name) and assign position i to train if i mod 10 ∈ {0..4}, validation if ∈ {5, 6}, test if
∈ {7, 8, 9}. Positions are counted over the whole sorted list, as for the frozen 85 (this reproduces their split
exactly; counting within each location does not).

| set | split | scenes | keyframes |
|---|---|---|---|
| new | train | 385 | 15,475 |
| new | validation | 152 | 6,117 |
| new | test | 228 | 9,181 |
| frozen | train / validation / test | 45 / 16 / 24 | 1,785 / 636 / 955 |

The scene list, split and tags below will ship with the results as `results/followup/nuscenes_xl/scene_split.csv`; they are
reproducible from the rule above and the nuScenes metadata.
As in the frozen track, the split is by scene, not by log: 52 of the 59 logs of the new scenes contribute test
scenes and also contribute training or validation scenes.

**Composition.** The frozen 85 and the 765 new scenes differ in ways that matter for decisions:

| | frozen 85 (train / val / test) | new 765 (train / val / test) |
|---|---|---|
| boston-seaport | 15 / 5 / 6 | 221 / 88 / 132 |
| singapore-onenorth | 30 / 11 / 18 | 60 / 25 / 39 |
| singapore-hollandvillage | 0 / 0 / 0 | 44 / 17 / 24 |
| singapore-queenstown | 0 / 0 / 0 | 60 / 22 / 33 |
| night (scene description) | 0 / 0 / 0 | 53 / 19 / 27 |
| rain (scene description) | 0 / 0 / 0 | 81 / 34 / 50 |
| nuScenes official train / val | 32+13 / 13+3 / 17+7 → 62 train, 23 val | 317+68 / 129+23 / 192+36 → 638 train, 127 val |
| distinct CAM_FRONT calibrations | 2 | 3 |

Read the official-split row as follows: of the 24 frozen test scenes, 17 are in nuScenes' official train split and 7
in its official validation split; of the 228 new test scenes, 192 and 36.

## 2. Prior contact with the 228 new test scenes

Before this protocol, exploratory analyses in a separate codebase evaluated on these 228 scenes. That codebase is
not part of this repository and shares no code with it. It is an independent re-implementation of the detection
inference (PyTorch on x86 instead of TensorRT FP16 on the Jetson), the monocular lift, the braking controller and its
loss. The authors saw the aggregate results. Unless stated, the consumer is the braking controller under monocular
geometry, and the detector is COCO YOLOv8s with a cheap mode at 192×320.

1. **Uniform escalation** of every frame to the whole-frame detector at long side 640, 960, 1280 and 1600, including
   the harmed share. Outcome seen: 320→640 is net harmful for braking (about −0.18 of the all-cheap loss, about half of
   the affected frames harmed), while 960 is about neutral. The braking consumer was also run under oracle geometry,
   and PKL's planner (ADE / FDE) under both geometries, for the cheap, 640 and 960 modes.
2. **Detector-upgrade and fine-tuned-detector churn.**
   - Nine upgrade pairs at 384×640, including RT-DETR-l at 640×640.
   - Outcome seen: 40–52% of changed braking decisions are harmed.
   - Fine-tuned detectors were also evaluated for braking (one fine-tuned v8m → v8l pair; fine-tuned cheap models with
     and without escalation to a COCO full model). Outcome seen: the fine-tuned v8m → v8l upgrade harmed about half of
     the changed braking decisions and slightly raised the net braking loss, and escalating a fine-tuned cheap model to
     a COCO full model was net harmful.
3. **Frame-level allocators** routing frames to the 960 model at two fixed escalation rates.
   - Random, maximum detection entropy, minimum confidence.
   - A ridge gate over cheap-side ensemble statistics: fit on the 385 new training scenes, scored as nDG at several
     budgets.
   - Re-implementations of published frame-level routing estimators.
   - An oracle frame router.
   - Outcome seen: learned frame routers gained at most about 0.08 of the all-cheap loss, well below the oracle frame
     router.

The same scenes were also used as an evaluation set in other exploratory work by the authors; that work involved none
of this repository's allocators, signals or code.

**No allocator implemented in this repository has been scored on these scenes.** That covers the benchmark's ridge
and gradient-boosted gates on its own cheap-side features, the detection-list and pixel routers, its uncertainty and
cheap-criticality signals, and ego speed. Nor has this repository's decision pipeline been run on them. The
validation scenes (152) and training scenes (385) were also used by that exploration, for fitting and for selecting
its configurations.

Consequence for interpretation: this protocol's hypotheses about the new scenes are informed by items 1–3. In
particular, a net-harmful uniform 320→640 escalation for braking is expected rather than discovered, and so is harm
with fine-tuned detectors. What remains untested is:
- whether this repository's pipeline reproduces it;
- how setting-level quantities vary across the composition differences of Section 1;
- whether the benchmark's allocators, fitted on ten times more units, beat random.

## 3. Frozen configuration

Everything below is the benchmark's configuration at release tag `v1.0` (code and canonical outputs);
nothing is tuned on the new scenes.

- **Detector modes (COCO YOLOv8s, TensorRT FP16, Jetson AGX Xavier).**
  - 192×320 (cheap) and 384×640 (full), letterboxed from 1600×900; the engines are those of the frozen track.
  - Detection decoding: traffic classes {person, bicycle, car, motorcycle, bus, truck}, NMS IoU 0.65, at most 100
    detections, candidate floor 0.10.
  - Operating threshold 0.25 for both modes (shared).
  - Engines for any further fidelity use the same letterbox convention with both sides divisible by 32: 288×480 and
    544×960.
- **Reference and matching.** Projected 3D boxes of the categories `vehicle.{car, truck, bus, trailer, construction,
  motorcycle, bicycle}` and `human.pedestrian` with at least one lidar or radar point; boxes below 10 px height are
  unmeasurable; greedy confidence-ordered class-agnostic matching at IoU 0.5.
- **Downstream systems and losses** (`docs/iclr_formulas.md`, `docs/iclr_loss_definitions.md`): braking controller
  (corridor, loss weights, monocular lift with camera height 1.51 m; oracle geometry as the control), PKL's planner
  scored by ADE and FDE against the logged future trajectory, on frames that have one.
- **Shared action history** (`docs/iclr_shared_history.md`): both branches charged against the all-cheap run's
  previous action.
- **Allocators.** The ten deployable signals of `results/final/significance_table.csv` (gates: `scripts/92`, `144`;
  detection-list routers: `103`; pixel router: `107`), with their benchmark hyperparameters, features and seeds.
  Fitted on train + validation units and scored once on test, as in the benchmark, except in the learning curve
  (Section 5).
- **Statistics.** Ties scored in exact expectation over tie-breaks. Bootstrap by scene, 1,000 draws, seed 0, paired
  against random within each draw. The significance rule is that of `results/final/significance_table.csv`: the 95%
  interval of the realized-gain difference over every draw.
- **Validation scenes (152)** serve threshold calibration and per-mode operating points (E5), and validation-selected
  comparisons. They are never used for hyperparameter search in a primary run.

Changes needed only to run on the new scene list (data root, scene list, output paths) are not deviations; anything
else is.

## 4. Gates

- **Parity.** The existing 320 and 640 engines re-run on the frozen 85 scenes reproduce the cached detections exactly.
- **(a)** The pipeline used for the new scenes, run on the frozen 85 scenes, reproduces the published benchmark values exactly:
  - setting-level rows of `benchmark_table.csv`;
  - the nuScenes rows of `significance_table.csv`.
- **(b) Informational.** The uniform 320→640 braking value on the 228 test scenes is compared with the exploratory
  value of Section 2, item 1. A difference is recorded, not corrected.

A failed gate stops the experiment until resolved.

## 5. Primary endpoints

**P1 — setting level (X1).**
- **Quantities:** affected inputs, harmed share, harm ratio ρ, all-full loss reduction and oracle@20 loss reduction,
  each with its 95% interval (1,000 scene-bootstrap draws).
- **Populations:** the 228 test scenes; all 765 new scenes; the frozen 85 recomputed with the same interval code;
  the pooled 850.
- **Cells:** YOLOv8s 320→640; braking under monocular and oracle geometry; planner ADE and FDE under both geometries.

**P2 — allocators (X2).**
- **Test:** realized-gain difference against random at a 20% selection budget, as a share of the all-cheap loss, with
  its paired interval over every draw.
- **Scope:** the ten deployable signals on the six nuScenes cells (two geometries × braking, planner ADE, planner
  FDE); 228 test scenes; allocators fitted on the 385 + 152 new train and validation scenes.
- **Claims under test:** the three wins reported for braking under monocular geometry on the frozen 85 scenes:
  - gradient-boosted gate (`gate_gbm`);
  - MLP router on V (`R1_mlp_reg`);
  - gradient-boosted router on V (`R1_gbm_reg`).

  Each claim replicates if its interval lies above zero. Holm-adjusted calls over the three are reported beside the
  unadjusted ones.

**P3 — learning curve (X4).**
- **Learners:** `gate_gbm` and `R1_gbm_reg`, braking under monocular geometry.
- **Training:** fitted on the first n of the 385 new training scenes, n ∈ {48, 96, 192, 385}; five random nested
  orderings (seeds 0–4); training scenes only, no validation scenes.
- **Scoring:** at 20% on the 228 test scenes.
- **Endpoint:** value at 385 minus value at 48 (mean over the five orderings), for realized gain (share of all-cheap
  loss) and for nDG, paired over test scenes within each draw.
- The other cells and signals are secondary.

## 6. Reading rules (every outcome is reported; none is dropped)

**P1 — setting level.**
- **Harmed share ≥ 30% with ρ > 0 in every monocular cell of the 228 test scenes:** sign variation reproduces at about
  ten times the units.
- **Harmed share below 30% in a cell:** reported as a cell where it does not.
- **All-full reduction of the opposite sign to the frozen 85 scenes:** reported with the heterogeneity tables
  (Section 7) as the explanation under test. It is not read as an error in either population.
- **Any oracle@20 gap to all-full:** reported with its interval.

**P2 — allocators.**
- **Each of the three claims:** "replicates", "not resolved" (interval contains zero) or "reverses" (interval below
  zero).
- **No learned allocator separates from random in any cell:** that null is stated as the result for this track.

**P3 — learning curve.**
- **Interval of the endpoint above zero:** "allocator gain increases with training scenes".
- **Interval contains zero:** a flat learning curve, stated as such. Learnability is then not data-limited in this
  range.
- **Interval below zero:** reported as a decrease.

## 7. Secondary analyses (labelled secondary wherever reported)

**E1.**
- **Heterogeneity** (in X1, braking, both geometries, new and frozen scenes): by map location, by split and by
  ego-speed tercile (terciles of the pooled 850 scenes), by night, by rain and by CAM_FRONT calibration (the frozen 85
  scenes contain no night and no rain scenes and one calibration fewer). Concentration: share of total benefit and of
  total harm carried by the top 5% and top 10% of scenes.
- **Resolution ladder at setting level** (in X1, secondary rows): YOLOv8s 320→480 and 320→960 on the four populations
  of P1, computed with the same code as the primary 320→640 rows.
- **Within-log dependence.** 52 of the 59 logs of the new scenes contribute test scenes and also training or
  validation scenes; the split stays as it is. Every interval of P1 and P2 is also computed with a bootstrap by log
  (1,000 draws, seed 0), reported beside the scene bootstrap. A log-grouped out-of-fold check: five folds grouped by
  log over the 765 new scenes (logs assigned to folds in sorted order of log name, position mod 5), for `gate_gbm`,
  `R1_gbm_reg` and `gate_ridge` on braking under monocular geometry at a 20% budget, each fold's allocator fitted on
  the other four folds' scenes; realized gain against random and nDG over the pooled out-of-fold scores, reported
  beside P2.
- **Planner exposure.** PKL's planner was trained on nuScenes' official train split (its released training code
  selects `create_splits_scenes()['train']` for v1.0-trainval). Every planner row of X1 and X2 is also reported on
  the official-validation scenes alone, with scene counts.
- **Budget sweep (X3):** selection budgets 5, 10, 20, 30, 40, 50%; measured latency and module-energy budgets at
  α ∈ {0.05, 0.1, 0.2, 0.3, 0.5} with each allocator's cost charged.
- **Learning curve.** Remaining cells and signals. Optional: one larger-capacity configuration per learner, chosen on
  validation from a grid of at most four, reported beside the frozen configuration, never instead of it. MLP router
  over seeds 0–4.
- **Multi-fidelity (X5):** 320 / 480 / 640 / 960, oracle and per-mode gradient-boosted value regression with greedy
  value per cost, against the 640-only allocation, in ms and mJ.
- **Transfer (X6).**
  - Frozen-85 train + validation → new test.
  - Boston ↔ Singapore within the new scenes.
  - KITTI braking → nuScenes braking on shared features.
- **Temporal-context router:** cheap detections of frames t−3..t and ego acceleration / yaw rate.
- **Sign disagreement:** between the four perception gains of the paper's sign-agreement table and decision value,
  both geometries.
- **Validation-selected comparison:** each objective picks its winner among the target-free signals on the 152
  validation scenes; the two winners compared once on test by realized decision gain.
- **Mechanism counts** on braking: added false positive the controller reacts to, recovered miss it needs, on
  harmed and on helped frames.

## 8. E5 — escalation across capacity, architecture and fine-tuning

**Populations.** Decision values on the 228 test scenes; threshold calibration on the 152 validation scenes;
allocators on this axis are secondary.

**Pairs at 384×640** (RT-DETR-l at 640×640, square, as in the frozen KITTI track):
- YOLOv8s → v8m, v8s → v8l, v8s → v8x, v8m → v8l, v8l → v8x;
- YOLO11s → YOLO11m;
- YOLOv8s → YOLO11s, YOLOv8m → YOLO11m, YOLOv8s → RT-DETR-l;
- two axes: YOLOv8s 192×320 → YOLOv8l 384×640.

**Operating points.** Every pair is reported at both:
- the shared threshold 0.25;
- equal box count: the new model's threshold set so that its total box count on the 152 validation scenes equals the
  old model's at 0.25.

Primary, declared now:
- shared threshold within a model family (YOLOv8 → YOLOv8, YOLO11 → YOLO11);
- equal box count across families (YOLOv8 ↔ YOLO11, YOLO → RT-DETR).

**Reported per pair:**
- recall and precision of both models (on the test scenes, at the operating point);
- affected inputs, harmed share and ρ with 95% scene-bootstrap intervals;
- all-full and oracle@20 reductions;
- braking under monocular and oracle geometry, planner ADE / FDE;
- measured latency and module energy of every engine (TensorRT FP16, Jetson, the frozen track's timing boundary).

**Fine-tuned detectors**, trained by code in this repository. Recipe fixed now:
- **Models:** YOLOv8s, v8m and v8l, initialised from the official COCO weights.
- **Library:** Ultralytics 8.3.40, the version used for engine export on the Jetson.
- **Training data:** CAM_FRONT keyframes of the 385 new training scenes (15,475 images).
- **Labels:** the reference 2D boxes of Section 3 (≥ 10 px high, at least one lidar or radar point), with COCO class
  ids:
  - `human.pedestrian` → person;
  - `vehicle.bicycle` → bicycle; `vehicle.motorcycle` → motorcycle;
  - `vehicle.car` → car;
  - `vehicle.bus` → bus;
  - `vehicle.truck`, `vehicle.trailer`, `vehicle.construction` → truck.

  The 80-class head is kept.
- **Optimisation:** input 640, 15 epochs, batch 16, SGD with lr0 0.01, momentum 0.937, weight decay 5e-4, 3 warm-up
  epochs; Ultralytics default augmentation (mosaic, closed for the last 10 epochs; HSV; horizontal flip 0.5;
  scale 0.5; translate 0.1); seed 0, deterministic.
- **Batch fallback:** if a model does not fit at batch 16, batch 8 with the default nominal batch 64 (gradient
  accumulation); recorded as a deviation.
- **Checkpoint:** Ultralytics' best.pt by its detection fitness (0.1·mAP50 + 0.9·mAP50-95) on the 152 validation
  scenes' keyframes. Never chosen by decision value.
- **Deployment:** every cited number from TensorRT FP16 engines built from these weights on the Jetson, with
  measured cost.
- **Rows:**
  - fine-tuned v8m → fine-tuned v8l at 384×640;
  - fine-tuned v8s 192×320 → 384×640;
  - beside them, recall and precision of the COCO and fine-tuned models on the test scenes.
- **Release:** weights and recipe ship with it.
- **Secondary:** a second seed (1) per model if resources allow.

## 9. Outline of the remaining experiments (detailed addendum before their first decision value)

- **E2 — NAVSIM navtest, PDM-Closed.**
  - Per-token decision value of 320→640 on the front camera.
  - Sub-scores and scenario strata.
  - Allocators fitted on the development split, scored once on navtest.
  - Every score computed by this repository's port with this repository's detections.
  - The reference score's discrepancy with the published PDM-Closed score is resolved and its cause recorded before
    any decision value.
- **E3 — closed loop (Bench2Drive, CARLA).**
  - Forked rollouts first: cheap versus full at one frame, the cheap policy afterwards, 8 s. The closed-loop outcome is
    the single-step loss terms accumulated over the window, plus collision and progress.
  - An A/A control (cheap in both branches) measures the simulator's run-to-run difference; sign agreement is
    reported only above that noise floor, with the count below it.
  - Then the uniform ladder with repeats; routing at matched budgets last.
- **E6:** a learned planner on the cached nuPlan observations.
- **E7:** a LiDAR detector pair on nuScenes with metric 3D boxes.

## Addenda

The addenda follow in the order of the follow-up list. Each was committed before the first value of its item.

---

## Addendum E2 — NAVSIM navtest with PDM-Closed

Addendum to `followup_protocol.md` (Section 9), committed before any decision value of E2 is computed. Reviewed afterwards
from the index. Sections 3 (statistics conventions), 4 (gates stop only what depends on them) and 6 (every outcome is
reported) of the protocol apply.

### Data

- **Scored set:** NAVSIM v1.1 navtest — 12,146 tokens in 136 logs (OpenScene test split, the navtest scene filter).
  Each token is one decision state.
- **Development set (allocator fitting only):** the OpenScene mini split with NAVSIM's default filter. Logs are sorted
  by name; a log at position i goes to validation if i mod 5 = 4, otherwise to train.
- **Metric caches:** produced by NAVSIM's own caching script (devkit v1.1) on the training machine. They are used as
  data; their file counts and the devkit commit are recorded with the run.

### Perception

- **Modes:** the benchmark's own nuScenes engines, YOLOv8s TensorRT FP16 on the Jetson, letterboxed from the
  1920×1080 CAM_F0 image: 192×320 (cheap) and 384×640 (full).
- **Detection:** operating threshold 0.25 for both modes; decoding as in Section 3 of the protocol.
- **Camera model:** intrinsics, distortion and camera-to-LiDAR extrinsics from the OpenScene metadata of each token.

### Downstream system and decision value

PDM-Closed as NAVSIM's metric-cache processor builds it, scored by NAVSIM's `pdm_score` with the default scoring
parameters.

**Objects per branch.** The rule is the benchmark's nuPlan real-perception rule:
- **reference:** the token's annotations.
- **cheap / full:** eligible annotations (vehicle, pedestrian, bicycle) are projected into CAM_F0 and matched to that
  mode's detections (Hungarian, class-compatible, IoU ≥ 0.3).
  - Unmatched in-camera eligible annotations are removed.
  - Unmatched detections are added as static objects on the ground plane. The plane height is the median bottom of
    vehicle boxes within 30 m over the navtest annotations. Sizes are the class medians of those annotations.
  - Everything outside CAM_F0 is unchanged.

**Trajectory handover.** The planner's trajectory is handed over as NAVSIM's `Trajectory`: 8 poses at 0.5 s,
ego-local, as for any agent.

**Decision value.**
- V = PDMS(full) − PDMS(cheap) per token.
- Also computed per sub-score: no-at-fault collision, drivable-area compliance, ego progress, time to collision,
  comfort.

### Gates (in order; each stops E2 until resolved)

1. **E2-0, reference fidelity.** This repository's reference branch is compared with NAVSIM's official PDM-Closed
   agent evaluation on navtest (the published value is 0.881). Any difference is traced to its cause and recorded
   before any V is computed.
   - The cause is fixed in the port, if it is the port's.
   - It is documented, if it is a devkit or setting difference that does not affect the cheap/full comparison.
2. **E2-1, agreement (informational).** On a sample of 300 tokens, this repository's cheap and full scores are
   compared with the exploratory values recorded before this protocol. A difference is recorded, not corrected.

### Quantities

**Setting level.** On navtest:
- affected tokens, harmed share and harm ratio ρ, all-full and oracle@20 reduction (relative to the all-cheap mean
  PDMS shortfall);
- intervals from a 1,000-draw bootstrap by log;
- the same for each sub-score.

PDMS is a score, not a loss: the loss is 1 − PDMS, so V is unchanged and "reduction" is of 1 − PDMS.

**Strata (secondary).**
- Ego-speed terciles.
- Terciles of the number of annotated agents within 30 m.
- Whether a vehicle annotation lies ahead within 30 m and 2 m laterally (lead vehicle).
- NAVSIM's driving command (left / straight / right).

### Allocators

All are fitted on the development train logs and scored once on navtest at selection budgets 5–50%, with an
every-draw log bootstrap paired against random.
- **Signals:** random; ego speed; detector uncertainty (summed binary entropy of the cheap detections); number of
  cheap detections.
- **Learned:** a ridge and a gradient-boosted gate over cheap-side features, and 103's detection-list routers
  (GBM / MLP on V and 1[V > 0]).
  - Gate features: counts, confidences and entropies of the cheap detections, their image-area statistics, and the
    lifted ground-plane distance of the nearest cheap detection ahead.
  - Model settings: the benchmark's models and settings.
- **Hyperparameters:** none are tuned. The development validation logs are used only for the validation-selected
  comparison of the protocol's Section 7.

### Reading

- **If PDMS barely reacts to front-camera perception** (few affected tokens), that is reported as the result: decision
  value is sparse for this consumer and this camera.
- **If affected tokens are fewer than 30**, harm ratio and allocator nDG are reported as undefined, with the count.

---

## Addendum E6 — a learned nuPlan planner on the external track

Addendum to `followup_protocol.md` (Section 9), committed before any E6 decision value. Reviewed afterwards from the
index.

### Planner

Diffusion Planner (Zheng et al., ICLR 2025), the authors' released nuPlan checkpoint and configuration, used as a nuPlan
`AbstractPlanner` with its released inference settings.

- **Fallback:** PLUTO (Cheng et al., 2024) with its released checkpoint, if the first cannot be run with its released
  code. The choice and the reason are recorded in the index before any decision value.
- **No retraining or tuning.**

### States, branches and losses

Everything except the planner is the benchmark's nuPlan real-perception track: the same 1,440 states of nuPlan mini, the
same reference, cheap and full observations (real detections on CAM_F0), a fresh planner instance per state and branch,
open-loop scoring.

**Losses.** The track's safety loss and scalar loss, scored by the track's code on the planned trajectory.

**Decision value.** V = L(cheap) − L(full) per state and loss.

### Reported

- affected states, harmed share, harm ratio ρ, all-full and oracle@20 reductions, with 1,000-draw intervals by log
  (all logs);
- the same quantities for PDM-Closed on the same states, from the released track, beside them;
- the number of states on which the planner failed in any branch; those states are dropped from every comparison.

### Reading

- **If fewer than 30 states are affected**, harm ratio and nDG are reported as undefined, with the count (as for IDM on
  this track).
- **A harmed share below 30%** is reported as such.

### Amendment 1 (2026-10-07, before any E6 decision value)

No E6 value has been computed: on the edge machine's CPU one planner call did not finish in about two hours, so no
state was scored.

- **Where it runs.** On the training machine's GPU (CUDA) with the same code, released checkpoint (md5 in
  `third_party/PROVENANCE.md`), nuPlan mini databases (same 64 files) and branch specification. The planner and its
  settings are unchanged.
- **Sampling noise.** The released planner draws Gaussian noise for its diffusion sampler at every call. Without
  control, V = L(cheap) − L(full) would mix the observation difference with sampling noise. The generator is therefore
  seeded with 0 immediately before every call (the same random numbers in every branch), and cuDNN runs deterministically.
- **A/A noise floor.** For every state, one more call on the reference observation with seed 1. Reported beside the
  cheap-vs-full row: the share of states whose loss changes between seeds and the mean absolute change
  (`aa_noise.csv`). This row is descriptive. It does not change the dropped-state rule or the reading.
- **Determinism check before the run.** Two runs of the first scenario must give identical rows. Otherwise the
  run goes ahead and the mismatch is recorded in the index.

### Amendment 2 (2026-10-07, after the run, before any E6 decision value was computed)

**What failed.** The full run (1,440 states, 5,723 planner calls on the training machine's GPU) finished. 180 states
failed in every branch, the A/A call included, with nuPlan's "dx is not monotonically increasing". They are exactly
the first three states of each of the 60 scenarios (iterations 4, 17 and 30).

**Cause: this repository's input construction, not the planner.** For these states the 2 s history reaches back
before the scenario's first iteration. 197 filled those frames by repeating the first frame, which gives duplicate
timestamps, and nuPlan's derivative computation rejects them.

**Fix.** Frames before the scenario's first iteration are now the logged ego states and tracks before it, read from the
same nuPlan database (`get_ego_past_trajectory` / `get_past_tracked_objects`), on the same 0.1 s grid. Every history
then has 21 frames spaced 0.1 s apart (checked on iterations 4, 17, 30 and 43). Branch observations are unaffected:
they lie inside the last four frames of each state, and no state has a covered frame before the scenario start.

**Rerun.** Only the 180 failed states are recomputed (`197 --redo_failed`), with the same seeds. The 1,260 states
already scored have no frame before the scenario start, so their inputs and code path are unchanged and their rows
are kept.

**Reporting.** The primary rows use every state that succeeds after the rerun; states that still fail are dropped as
the addendum says. Beside them, the same quantities on the 1,260 states of the first run, so the effect of the
rerun is visible.

---

## Addendum E7 — a LiDAR detector pair on nuScenes with metric 3D boxes

Addendum to `followup_protocol.md` (Section 9: "E7: a LiDAR detector pair on nuScenes with metric 3D boxes"; Sections 3,
4 and 6 apply). Committed before any value of E7 is computed; the sweep download is data only.
Reviewed afterwards from the index.

### Question

With metric 3D boxes from a LiDAR detector, the monocular lift plays no part. Does escalating from a cheaper to a
stronger LiDAR detector still change some braking decisions for the worse?

### Detectors

OpenPCDet's released nuScenes checkpoints and configurations, unchanged: ten LiDAR sweeps, their own voxelisation and
post-processing. They run in FP32, batch 1, on the training machine's RTX A5000.
- **Cheap:** CenterPoint-PointPillar (`cbgs_dyn_pp_centerpoint`; published validation mAP 50.0, NDS 60.7).
- **Full:** VoxelNeXt, voxel size 0.075 (`cbgs_voxel0075_voxelnext`; published mAP 60.5, NDS 66.7).
- **Secondary full:** TransFusion-L (`transfusion_lidar`; published mAP 64.6, NDS 69.4), reported beside the primary
  pair.
- **Amendment 1 (2026-10-18, before any value).** The registered full checkpoint, CenterPoint voxel 0.075, can no
  longer be downloaded: its link in OpenPCDet's model zoo returns 404. So does the next one of its family, CenterPoint
  voxel 0.1. VoxelNeXt at voxel 0.075 is the zoo's nearest: the same voxel size, centre-based heads, and a published
  mAP within 1.3 points of the registered model.
- The OpenPCDet commit (233f849) and the md5 of each checkpoint are recorded with the run.

### Scenes

- **Primary:** the 150 nuScenes official-validation scenes, all keyframes. They are outside the detectors' training
  split (nuScenes official train).
- **Secondary:** E1's 228 new test scenes. 192 of them are in the official training split, so the detectors were
  trained on them; this exposure is stated beside every row.
- Strata reported: E1's groups (frozen 85 / new 765) and split.

### Objects

- **Classes:** the detector's car, truck, construction_vehicle, bus, trailer, motorcycle, bicycle and pedestrian
  outputs. These are the reference categories of Section 3. Barrier and traffic-cone outputs are dropped.
- **Reference:** Section 3's boxes, with at least one LiDAR or radar point, in the ego frame of the keyframe
  (`rap.nusc`). The camera's 10 px height rule does not apply.
- **Field of view:** detections and references are kept under the same rule: at least four of the box's eight corners
  lie in front of CAM_FRONT, and the box projected into CAM_FRONT and clipped to the image is at least 4 px wide and
  high. This is `rap.nusc`'s rule, applied to the detected boxes too. The decision population is therefore that of the
  camera tracks.
- **Operating point:**
  - primary: score ≥ 0.25 for both detectors (Section 3's shared threshold);
  - secondary: the full detector's threshold set so that its total box count on the scored keyframes equals the cheap
    detector's at 0.25. This needs no labels.

### Geometry, consumer and decision value

- **Detected geometry:** the box footprint in the ego frame. Range is the nearest footprint x; lateral extent is the
  footprint's y range.
- **TTC:** from the box's predicted velocity. The closing speed is the ego speed minus the object's velocity along the
  ego's x axis; TTC is range / closing speed if the object is closing, otherwise infinite.
- **Reference geometry:** footprint, range rate and TTC from the shared local fit, as in the nuScenes track.
- **Ego speed:** the nuScenes track's, used in the decision cost.
- **Consumer:** the benchmark's braking controller and loss (Eq. 10), on the shared action history of the all-cheap
  run. V = J(cheap) − J(full).
- **Geometry sources:**
  - detected: primary;
  - oracle, secondary. A detection matched to a reference object takes that object's geometry. Matching is greedy by
    score, class-agnostic, at a bird's-eye-view centre distance ≤ 2 m (nuScenes' matching distance).
- **Planner rows:** secondary, and run only if the benchmark's submission path accepts these boxes unchanged (PKL's
  planner on the detected boxes, ADE / FDE against the logged trajectory).

### Gate E7-0 (stops E7 until resolved)

On the 150 validation scenes, OpenPCDet's own nuScenes evaluation of each checkpoint (all ten classes) reproduces its
published mAP within 1.0 point. This checks the sweeps, the data infos and the environment, before any decision value.

### Quantities

- **Setting level (as P1):** affected inputs, harmed share, ρ, all-full and oracle@20 reduction. 95% intervals from
  1,000 draws by scene and by log, seed 0.
- **Cells:** operating point (shared primary, equal count secondary) × geometry (detected primary, oracle secondary)
  × population.
- **Detection:** recall and precision of both detectors at the operating point (centre distance 2 m, class-agnostic),
  over the field of view, on the scored keyframes.
- **Cost (secondary):** median latency per keyframe of each detector on the A5000, over 200 keyframes, with the GPU
  shared as it is. Labelled as such.

### Reading

- **Primary cell, harmed share ≥ 30% with ρ > 0:** the sign variation reproduces with metric LiDAR boxes.
- **Harmed share below 30%:** reported as a cell where it does not.
- **All-full reduction:** reported with its interval, whatever its sign.

### Prior contact

- Exploratory work in a separate codebase used the LiDAR keyframes of the nuScenes scenes for camera pipelines: to
  re-estimate the range of camera detections, and to propose objects from LiDAR points.
- It ran no released LiDAR detector. No decision value of a LiDAR detector pair has been computed before this commit.

---

## Addendum E8 — a third camera and city set: Argoverse 1.1 (E8a) and Argoverse 2 (E8b)

Addendum to `followup_protocol.md` (Sections 3, 4 and 6 apply), for follow-ups E8a and E8b. This version fixes E8a in full and E8b in outline. E8b's dataset-specific rules (class map, hood crop,
distortion) are added in a dated section, committed before E8b's first value. No value of E8 has been computed before
this commit.

### Question

Do the benchmark's setting-level findings and allocator results hold on a third camera, vehicle and city set? The
findings are that uniform resolution escalation changes some braking decisions for the worse, and that the allocators
do not reliably beat random. Argoverse differs from KITTI and nuScenes in camera, mounting, cities (Pittsburgh, Miami;
AV2 six US cities) and labels.

### Data

- **E8a: Argoverse 1.1 tracking, validation split, all 24 logs** (14 Pittsburgh, 10 Miami), camera ring_front_center
  (1920×1200, 30 Hz). Labels are 3D boxes per LiDAR sweep (10 Hz, `per_sweep_annotations_amodal`).
- **E8b: Argoverse 2 sensor.**
  - Scored: the validation split, all 150 logs.
  - Fitting: 140 training logs, every fifth of the 700 training logs in sorted order of log id (positions 0, 5, 10, …).
- Public files only: images, poses, calibration and labels as released.

### Decision frames

- **Unit:** the log.
- **Frame:** every labelled LiDAR sweep. It is paired with the ring_front_center image nearest in time, and dropped if
  that image is more than 50 ms away. (E8a: all 5,015 sweeps pair, the largest gap is 17 ms.)
- **Frame interval:** 0.1 s, for the lift's scale-change TTC and the reference range rate.

### Images and camera

- **Distortion (E8a):** the released images carry radial lens distortion. Each image is undistorted with the published
  intrinsics and radial coefficients (OpenCV, same intrinsic matrix) before detection. The lift and the reference
  projection are then both pinhole with that matrix.
- **Camera → ego:** the published `vehicle_SE3_camera` of ring_front_center, per log.
- **Camera height above the ground:** the camera's height in the ego frame plus the height of the ego origin above the
  ground.
  - The ego origin of Argoverse is not on the ground.
  - Its height is the negative median bottom height of VEHICLE boxes 3–30 m ahead and within 6 m laterally, pooled
    over the dataset's scored logs.
  - It is measured once per dataset, from labels only (as E2 set its ground plane), and recorded with the run.

### Reference objects

- **Classes and their coarse class (E8a):**
  - vehicle: VEHICLE, LARGE_VEHICLE, BUS, SCHOOL_BUS, TRAILER, EMERGENCY_VEHICLE;
  - person: PEDESTRIAN;
  - cyclist: BICYCLE, BICYCLIST, MOTORCYCLE, MOTORCYCLIST, MOPED.

  ON_ROAD_OBSTACLE, OTHER_MOVER, ANIMAL, STROLLER and WHEELCHAIR have no counterpart among the detector's traffic
  classes and are not reference objects (nuScenes' track likewise keeps vehicles, pedestrians and two-wheelers).
- **Visibility:** Section 3's rule, at least one LiDAR point inside the box. Argoverse 1 releases no count per box,
  so `scripts/217_av1_lidar_points.py` counts the points of the released val sweeps (each sweep and its labels share
  the ego frame of the sweep time). (Changed 2026-10-18 after gate E8-0; see below. The first version said that no
  point-count rule applies in E8a.)
- **Projection:** as in the nuScenes track (`rap.nusc`).
  - Each box is moved from the ego frame at the sweep time to the ego frame at the image time, with the two released
    poses.
  - The box is kept if at least four of its eight corners lie in front of the camera. It is projected pinhole and
    clipped to the image, and kept if it is at least 4 px wide and high.
  - Boxes below 10 px high are unmeasurable (Section 3).
- **Reference geometry:** footprint in the ego frame at the image time. Range rate by the shared local fit (half-width
  2 frames at 10 Hz, as KITTI).
- **Poses:** where a log releases no pose at an image time, the two released poses around it are interpolated
  (translation linear, rotation spherical), and the count is reported. (Added 2026-10-18 after gate E8-0 stopped on
  a missing pose file; an adapter fix, before any decision value.)
- **Ego speed:** the forward component of the pose difference over about 0.1 s before the image time. It is causal;
  the first frame of a log, which has no earlier pose, uses the next frame.

### Perception

- **Engines:** the benchmark's nuScenes engines, unchanged: COCO YOLOv8s, TensorRT FP16 on the Jetson. Each image is
  letterboxed into 192×320 (cheap), 288×480, 384×640 (full) and 544×960.
- **Thresholds:** 0.25 shared; decoding as in Section 3.
- **Lift:** the benchmark's monocular lift in the ego frame (Task 23 convention), with each log's camera.

### Consumers and decision value

- **Consumer:** the benchmark's braking controller and loss under monocular and oracle geometry, on the shared action
  history; nothing changed.
- **Not run:** the planner rows. PKL's planner needs nuScenes maps.

### Gate E8-0: alignment (stops E8a, or E8b, until resolved)

Computed from detections and labels only, before any decision value.

- **Recall:** at 384×640, the recall of reference boxes at least 30 px high (IoU 0.5, class-agnostic) is at least 0.5.
- **Lift error:** the median relative range error of the monocular lift, over the matched 640 detections, lies within
  ±20%.

A failure is traced to the adapter (pose, calibration, projection or class map). The fix is recorded, and the gate is
re-run. Nothing is tuned on decision values.

**E8a, first run (2026-10-18, before any decision value).** The gate stopped twice.
- First on a missing pose file. Fix: pose interpolation (above).
- Then on recall: 0.444 on 55,194 reference boxes ≥ 30 px, below 0.5. The lift error was within bounds: median
  +1.7%, quartiles −8.9% / +12.8%, over 24,405 matched detections.

Diagnosis, from labels and detections only:
- The labels are amodal, and no visibility rule applied. One third of the boxes ≥ 30 px (18,345) are at least 75%
  covered in the image by nearer reference boxes, and their recall is 0.20.
- Boxes with less than 1% covered have recall 0.62; boxes nearer than 10 m have 0.71.
- Of the unmatched boxes, 44% have no detection with IoU ≥ 0.1. That is missed objects, not misaligned ones.

Change: Section 3's visibility rule, counted from the released LiDAR (above). The gate is re-run with it, its
threshold unchanged. If it fails again, E8a stops and is reported as not run.

### Quantities

**Setting level (as P1).**
- **Cells:** 320→640 primary; 320→480 and 320→960 secondary; braking under both geometries.
- **Quantities:** affected inputs, harmed share, ρ, all-full and oracle@20 reduction.
- **Intervals:** 1,000 bootstrap draws by log, seed 0.
- **Populations:** AV1 val (24 logs); AV2 val (150); AV2 fitting logs (140, secondary).
- **Heterogeneity (secondary):** by city.

**Allocators (as P2).**
- **Setting:** 320→640 braking under monocular and oracle geometry, 20% budget, realised gain against random with the
  paired every-draw interval by log.
- **Signals:** the benchmark's deployable signals except the pixel router R2, which would need its own training run
  per dataset. That leaves gate_ridge, gate_gbm, R1_mlp_reg, R1_mlp_clf, R1_gbm_reg, R1_gbm_clf, uncertainty,
  criticality_cheap and trivial_ego_speed, with their benchmark features, hyperparameters and seeds.
- **AV1:** five folds by log (sorted log ids, position mod 5). Each fold's allocators are fitted on the other four
  folds, and the out-of-fold scores are pooled.
- **AV2:** fitted on the 140 fitting logs, scored once on the 150 validation logs.

**Transfer (secondary).** Allocators fitted on E1's 537 new nuScenes training and validation scenes, scored on AV1
and on AV2 validation. Also AV2 fitting logs → AV1.

### Reading

- **Monocular 320→640 cell, harmed share ≥ 30% with ρ > 0:** the sign variation reproduces on this set.
- **Harmed share below 30%:** reported as a set where it does not reproduce.
- **Each allocator:** "beats random" (interval above 0), "not resolved", or "below random".
- **No learned allocator separates from random:** that null is stated as the result for the set.

### Prior contact

Before this addendum, exploratory analyses in a separate codebase used both sets. That codebase is not part of this
repository and shares no code with it. The authors saw its results.

- **AV1 val, all 24 logs** (all 15,062 images, 30 Hz):
  - YOLOv8 detection at several resolutions;
  - braking decision values from a monocular lift, for whole-frame resolution escalation and for several crop-based
    escalation methods.
- **AV2 val, all 150 logs:** the same kinds of values. Outcome seen: uniform escalation to 640 harmed about a third of
  the affected braking decisions.
- **AV2 training logs:** used there for configuration choices and model fitting. All of them count as seen.

Consequence for interpretation:
- On both sets, sign variation under uniform escalation is expected rather than discovered.
- What remains untested is whether this repository's pipeline reproduces it, and how the benchmark's allocators
  fare.

### E8b: Argoverse 2 rules (added 2026-10-18, before any E8b value)

- **Frames:** each annotated LiDAR sweep (10 Hz), paired with the nearest ring_front_center image (20 Hz; at most
  50 ms away, otherwise dropped). The released poses include every image time.
- **Image:** 1550×2048, portrait. The ego vehicle's hood is visible from row 1830 down. Rows ≥ 1830 are cropped
  before detection, leaving 1550×1830. The intrinsic matrix is unchanged, since the crop removes rows at the bottom
  only. Reference boxes are clipped to the cropped image.
- **Distortion:** the released intrinsics carry radial coefficients (k1, k2, k3). Whether the released images are
  already rectified is not stated in a form this repository can check. Gate E8-0 therefore runs on both versions: the
  raw images, and the images undistorted as in E8a. The version with the higher 640 recall on reference boxes ≥ 30 px
  is used. This uses labels and detections only, before any decision value. Both recalls are reported.
- **Classes:**
  - vehicle: REGULAR_VEHICLE, LARGE_VEHICLE, BOX_TRUCK, TRUCK, TRUCK_CAB, BUS, ARTICULATED_BUS, SCHOOL_BUS,
    VEHICULAR_TRAILER, RAILED_VEHICLE;
  - person: PEDESTRIAN, OFFICIAL_SIGNALER;
  - cyclist: BICYCLE, BICYCLIST, MOTORCYCLE, MOTORCYCLIST, WHEELED_RIDER.

  Not reference objects: BOLLARD, CONSTRUCTION_BARREL, CONSTRUCTION_CONE, SIGN, STOP_SIGN,
  MOBILE_PEDESTRIAN_CROSSING_SIGN, MESSAGE_BOARD_TRAILER, DOG, STROLLER, WHEELCHAIR and WHEELED_DEVICE (no traffic-class
  counterpart in the detector, or a device without its rider).
- **Visibility:** `num_interior_pts` ≥ 1, the counterpart of Section 3's point rule.
- **Camera height, ego speed, projection and footprint:** as in E8a. The ground offset is computed from
  REGULAR_VEHICLE boxes, pooled over the 150 validation logs.
- **Detection:** the four modes run frame by frame, so each image is decoded once. The engines, threshold and lift are
  E8a's.

---

## Addendum E9 — escalation of the range estimator: ground-plane lift or a metric depth network

Addendum to `followup_protocol.md` (Sections 3, 4 and 6 apply), for follow-up E9. Committed before any value of E9 is computed.

### Question

Oracle geometry showed how much of the decision value comes from the lift. E9 asks a deployable form of the same
question. With the detections held fixed, does replacing the ground-plane lift with a stronger, published metric depth
network change some braking decisions for the worse?

### Escalation

- **Detections:** fixed, the benchmark's cached COCO YOLOv8s TensorRT detections at 384×640 (primary) and 192×320
  (secondary); threshold 0.25.
- **Cheap range:** the benchmark's monocular ground-plane lift, ego frame.
- **Full range:** Depth Anything V2, metric outdoor (fine-tuned on Virtual KITTI 2, maximum depth 80 m), official
  weights, TensorRT FP16 on the Jetson.
  - Small: primary.
  - Large: secondary.
  - Revision hashes and engine md5 are recorded with the run.
  - ONNX opset 16, with the layer norms written as elementary operations. (Added 2026-10-18, before any value:
    the Jetson's TensorRT 8.5 cannot import opset 17's LayerNormalization, and the first build stopped there.)
  - Loaded with transformers 4.45.2, the first release that builds the checkpoints' metric head (sigmoid × 80 m);
    the loader asserts it. (Added 2026-10-18, before any decision value: gate E9-0's first run exposed that 4.44.2
    had built the relative head, so reference and engines both returned values of 0–6 with median 0. Everything
    from the export onward was redone.)
  - Large's engines hold the layer norms in FP32; all other layers may run in FP16 (`scripts/224_depth_engine.py`).
    Small keeps its plain FP16 engines.
    - Added 2026-10-19, before any decision value. In gate E9-0's second run, Small passed (median 0.07% on
      nuScenes, 0.33% on KITTI) and Large failed (1.1% and 4.1%).
    - At opset 16 each layer norm is a chain of elementary operations, and TensorRT ran them in FP16.
    - A diagnostic on the same 424 nuScenes gate boxes reproduced the failing engine's statistic exactly (1.106%).
      An engine with the norms in FP32 gave 0.033%.
    - Large's engines, inference and gate are redone. Small's are kept.
- **Depth input:**
  - The whole image is resized to 518 px wide (bilinear, aspect kept, height rounded to a multiple of 14) and
    normalised as the model's preprocessor does. Engines: nuScenes 294×518, KITTI 154×518.
  - The output is resized to the image bilinearly. The network's metric output is used as released, with no rescaling
    for the camera's focal length.
- **Range per box:**
  - The median depth over the box's central region: the middle 50% of its width, rows from 50% to 90% of its height,
    counted from the top.
  - That camera depth replaces the fused depth of the lift's bottom-centre ray. The point and the two bottom corners
    at that depth are expressed in the ego frame with the camera → ego transform, exactly as the lift's final step.
- **TTC:** the lift's scale-change TTC in both branches. Only range and lateral extent change.

### Consumer and decision value

- **Consumer:** the benchmark's braking controller and loss (Eq. 10), on the shared action history of the all-lift run.
- **Decision value:** V = J(lift) − J(depth).
- **Reference row:** oracle geometry on the same detections, V = J(lift) − J(oracle). It is the ceiling of any range
  escalation.
- **Cells:** lift → Small (primary), lift → Large, Small → Large; each for 640 and 320 detections.

### Populations

- **nuScenes:** P1's four (228 new test scenes, 765 new, frozen 85, pooled 850).
- **KITTI:** all 21 sequences, and the 6 test sequences.

### Gate E9-0 (stops E9 until resolved)

On 50 frames per dataset, the TensorRT FP16 engine is compared with the PyTorch FP32 model on the same input. The
frames are evenly spaced over the dataset's frames in sorted (unit, frame) order: positions round(linspace(0, N − 1,
50)). (Corrected with the code, before any value: the first version named "the first frame of every tenth unit",
which gives 3 frames on KITTI.) The median relative difference of the box ranges (the per-box median depth above) must
be below 1%.

### Quantities

- **P1's quantities per cell:** affected inputs, harmed share, ρ, all-full and oracle@20 reduction. 95% intervals from
  1,000 draws by unit (and by log on nuScenes), seed 0.
- **Range error:** median and quartiles of (range − reference range) / reference range for lift, Small and Large, on
  detections matched to reference objects (IoU 0.5).
- **Cost:** latency and module energy per frame of each depth engine on the Jetson, with the frozen profile's
  protocol, beside the lift's.

### Reading

- **Primary cell, harmed share ≥ 30% with ρ > 0:** sign variation appears when the range estimator is escalated.
- **Harmed share below 30%:** the cell does not show it.
- **All-full reduction against oracle's:** both are reported with their intervals. The gap is the share of the range
  ceiling the depth network recovers.

### Prior contact

- Protocol Section 2 lists no use of a depth network.
- No value of E9 has been computed before this commit.

---

## Addendum E10 — escalation in time: reuse the last detections or run the detector

Addendum to `followup_protocol.md` (Sections 3, 4 and 6 apply), for follow-up E10. Committed before any value of E10 is computed. Reviewed afterwards from the index.

### Question

Skipping frames is the most common way a deployed system saves perception compute: on a skipped frame the planner keeps
the last detections. Does the sign variation of decision value appear on this axis? That is, does running the detector
on the current frame, instead of reusing the previous frame's detections, make some braking decisions worse?

### Escalation

- **Detector:** COCO YOLOv8s, the benchmark's cached TensorRT FP16 detections; operating threshold 0.25.
  - Primary mode: 384×640 (the benchmark's full mode).
  - Secondary mode: 192×320.
- **Fresh (full):** the mode's detections of frame t, the benchmark's per-frame input.
- **Stale (cheap):** the same mode's detections of the previous cached frame t−1, carried to frame t:
  - **ego (primary): static-world propagation.**
    - Each kept detection's geometry at t−1 is taken under the cell's geometry source, evaluated at t−1.
    - It is moved into the ego frame of t with the planar ego motion between the two frames (forward, left, yaw).
    - Range: the forward coordinate of the moved near-face centre. Lateral extent: the moved bottom corners.
    - The closing speed at t−1 (range / TTC) is kept, and TTC is re-derived from the new range.
    - A carried detection whose new range is below 0.5 m (the lift's lower clip) has left the camera's view and is
      dropped. (Added with the code, before any value.)
  - **hold (secondary):** the geometry at t−1 unchanged (zero-order hold).
- **Ego motion:**
  - nuScenes: the CAM_FRONT keyframe ego poses.
  - KITTI: OXTS. Mercator position from latitude and longitude, with the first frame's scale; heading from yaw.
- **First frame of each unit:** the detector runs, so stale equals fresh. That frame is not a decision unit and is
  excluded from every count. It still gives the first action of the all-stale run.
- **Interval:** nuScenes keyframes at about 2 Hz (about 0.5 s); KITTI at 10 Hz (0.1 s). The distribution of intervals
  is reported.

### Consumer and decision value

- **Controller and loss:** the benchmark's braking controller and its loss (Eq. 10). The reference a⋆ comes from the
  ground truth at t, and the ego speed is that at t.
- **Geometry:**
  - mono: the deployed monocular lift (ego frame);
  - oracle: the benchmark's substitution of matched ground-truth geometry. For the stale branch, the detections are
    matched to the ground truth of t−1, as they were when fresh, and that geometry is then carried to t as above.
- **Shared action history:** both branches are charged against the previous action of the all-stale run.
- **Decision value:** V = J(stale) − J(fresh); V > 0 means running the detector helped.
- **Not run:**
  - the planner rows, which need rasters per branch;
  - the lateral controller, which is outside the benchmark's systems.

### Populations

- **nuScenes:** the four populations of P1 (228 new test scenes, 765 new, frozen 85, pooled 850). Detections come from
  the 850-scene cache of E1.
- **KITTI:** the benchmark's tracking track, all 21 sequences and the 6 test sequences.

### Quantities

- **P1's quantities per cell:** affected inputs (V ≠ 0), harmed share, ρ, all-fresh reduction and oracle@20 reduction.
  The reductions are shares of the all-stale loss.
- **Intervals:** 95%, from 1,000 bootstrap draws (seed 0) by unit: scene or sequence. nuScenes also by log.
- **Cells:** dataset × mode (640 primary, 320 secondary) × propagation (ego primary, hold secondary) × geometry (mono,
  oracle).

### Gate E10-0 (stops E10 until resolved)

The fresh branch reproduces the benchmark's per-frame braking requirement and discrete action exactly, under both
geometries:
- nuScenes: the E1 850-scene tables (`a_full` of 320→640 for the 640 mode, `a_cheap` for 320);
- KITTI: the benchmark's core-matrix tables (YOLOv8s 320→640).

### Secondary: which frames to refresh

Labelled secondary wherever reported.
- **Setting:** nuScenes, braking under monocular geometry, 640, ego propagation; budget 20% of frames.
- **Data:** fitted on the 385 + 152 new training and validation scenes, scored once on the 228 test scenes.
- **Signals:**
  - random;
  - uncertainty: summed binary entropy of the stale list;
  - ego speed;
  - `gate_gbm`: the benchmark's gradient-boosted gate with its settings. Its features are the benchmark's gate features
    (`scripts/92_benchmark_table.py`) of the stale list, computed at t−1 (all a skipping system knows at t), with the
    ego speed at t.
- **Measure:** realised gain against random, as a share of the all-stale loss, with the paired every-draw interval of
  Section 3.

### Reading

- **Harmed share ≥ 30% with ρ > 0 in a monocular cell:** sign variation appears on the temporal axis in that cell.
- **Harmed share below 30%:** the cell does not show it.
- **All-fresh reduction:** reported with its interval, whatever its sign. A negative value means that running the
  detector on every frame is net harmful for braking against reusing the last detections.
- **Two intervals (0.1 s and 0.5 s):** no direction is hypothesised. The two datasets also differ in camera, labels
  and scenes, so the difference between them is not attributed to the interval alone.

### Prior contact

The protocol's list of prior contact (Section 2) contains no comparison of stale and fresh detections.
- E1's temporal-context router (secondary, X6) used the cheap detections of t−3..t as features of an allocator for
  resolution. It did not compare stale and fresh decisions.
- E10's values have not been computed before this commit.
