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

## Addendum E3 — closed loop (Bench2Drive, CARLA 0.9.15)

Addendum to `followup_protocol.md` (Section 9), committed before any E3 rollout is scored.
Order: (b) forked rollouts, then (a) the uniform ladder, then (c) routing — and (c) only if (a) and (b) are done.

### System

- **Planner:** PDM-Lite, the rule-based expert of the Bench2Drive release of carla_garage. Its map, route, traffic
  lights and scenario information stay privileged.
- **Perception in the loop:** the rule of the benchmark's nuPlan and NAVSIM tracks.
  - Every vehicle or walker whose 3D box projects into the front camera (1600×900, FOV 70°, Bench2Drive's CAM_FRONT
    pose) is kept only if a detection of the active mode matches it (Hungarian, class-compatible, IoU ≥ 0.3).
  - Unmatched in-camera actors are hidden from the planner.
  - Unmatched detections become static obstacles on the ground plane (class-median sizes, heading along the viewing
    ray).
  - Actors outside the camera are untouched.
- **Modes:** COCO YOLOv8s at 192×320 (cheap) and 384×640 (full), operating threshold 0.25.
  - Inference runs in PyTorch FP16 on the simulation machine's GPU. TensorRT engines are Jetson-specific; the weights,
    letterbox and decoding are the benchmark's.
  - Detection is run on every simulator tick (20 Hz).
- **Branches:** privileged (no perception), cheap, full; the uniform ladder (a) adds 960 (544×960).

### Decision value in closed loop

**Escalation unit.** One decision frame = 0.5 s = 10 simulator ticks, the benchmark's 2 Hz frame interval. Escalating
frame t means that the full mode replaces the cheap mode for the ticks of that frame only.

**Per-tick loss.** The benchmark's braking loss (Appendix J of the paper, Eq. 10):
- a = the ego's realised longitudinal acceleration over the tick (negative = braking), from the simulator;
- a⋆ = the braking controller's requirement computed from the true actors (all actors, privileged geometry, the
  benchmark's corridor and parameters), at the ego's speed;
- terms: shortfall, excess, change of a against the previous tick, and the large-shortfall penalty.

**Window outcome**, over the 8 s window that starts at frame t:
- J = the per-tick losses summed over the window;
- collision: any collision event of the ego in the window;
- progress: the route distance driven in the window.

**Forked rollout (b).** For a route and a frame t:
- the baseline run drives cheap throughout;
- the fork re-runs the route from its start with the same seeds, cheap until t, full on frame t, cheap afterwards,
  and stops 8 s after t;
- closed-loop value: ΔJ = J(baseline window) − J(fork window), positive when escalating frame t helped;
- single-step decision value: V_t = L(a_cheap, a⋆) − L(a_full, a⋆), summed over the 10 ticks of frame t (the first
  ticks of the two runs, where the history is still shared).

**A/A control.** The baseline is run twice with identical settings. On every sampled frame, |ΔJ_AA| (the difference
between the two baseline windows) measures the simulator's run-to-run difference.
- The noise floor is the 95th percentile of |ΔJ_AA| over the sampled frames.
- Sign agreement between V_t and ΔJ is reported only on frames whose |ΔJ| exceeds the floor, together with the
  number of frames below it.
- The same is done for collision and progress, as differences in counts and metres.

**Frame sample.**
- In the baseline run both modes are evaluated on every frame (the full mode in shadow: it is computed and logged,
  but not used).
- A frame is eligible if the perceived object sets of the two modes differ on any of its ticks.
- Up to 300 eligible frames are drawn without replacement, stratified by route, with numpy seed 0 (at most 5 per
  route), from the first 40 Bench2Drive routes (in route-id order) that complete under the cheap branch.
- 50 non-eligible frames are drawn the same way as a further null control.

**Reported.**
- Counts of eligible, sampled, above-floor and below-floor frames.
- Sign agreement of V_t and ΔJ, with a 95% bootstrap interval by route.
- Spearman correlation of V_t and ΔJ.
- The same against collision and progress differences.

### Uniform ladder (a)

The 220 Bench2Drive routes with branches privileged, cheap, full (640), 960; three repetitions each. Reported:
driving score, success rate, infractions by type, and the paired difference between branches, with a route-bootstrap
interval over route × repetition means.

### Routing (c)

Defined in a further addendum after (a) and (b), before any routing run.

### Reading

- **If fewer than 30 frames exceed the noise floor**, sign agreement is reported as unresolved, with the count.
- **If the floor is large compared with typical ΔJ**, that is reported: the simulator's run-to-run variation then
  bounds what a closed-loop test of single-frame escalation can resolve.

---

## Addendum E3 (c) — routing at a matched budget in closed loop

Addendum to `followup_protocol.md` (Section 9: "routing at matched budgets last") and addendum E3 in `docs/followup_protocol.md`.
Written after (a) and (b) and committed before any development or routing run.

### Question

In closed loop, does a gate fitted the benchmark's way choose frames whose escalation helps driving more than a
random choice of the same share of frames? The gate is E2's gradient-boosted gate, trained on the decision value.

### System

As in (a) and (b):
- PDM-Lite with perception in the loop (the benchmark's rule);
- COCO YOLOv8s, cheap 192×320 and full 384×640, threshold 0.25, PyTorch FP16;
- detection on every tick (20 Hz).

Code: `scripts/202_b2d_route_pilot.py` (195's pilot plus routing; 195 unchanged), `scripts/203_b2d_routing.py`, and
`scripts/198_b2d_dispatch.sh` (cases `routedev`, `route`).

### Routing

- **Decision frame:** 10 ticks (0.5 s), counted from the route's first tick; the escalation unit of (b).
- **Decision:** at a frame's first tick the cheap mode runs, and the router decides from that output and the ego
  speed.
- **Escalation:** an escalated frame uses the full mode on all 10 of its ticks, including the first tick's planner
  input. Otherwise the frame uses cheap.
- **Budget:** 20% of frames, the paper's headline selection budget.
- **Policies:**
  - **gate:** E2's `gate_gbm` (the benchmark's gradient-boosted regressor and settings) predicts V from the gate
    features. A frame is escalated when the prediction is at least τ.
  - **random:** a frame is escalated with probability 0.2. The numpy generator is seeded with route id × 10 +
    repetition.

### Gate fitting (development)

**Routes and runs.** The first 50 Bench2Drive routes by id, the routes of (b). One run each: cheap, with the full mode
in shadow (202, `ROUTER=dev`).

**Decision value per frame.**
- Per tick, the benchmark's braking controller acts on each mode's perceived scene. Its loss (Eq. 10) is computed
  against a⋆ from the true actors; the jerk term is taken against that mode's previous action.
- V = Σ over the frame's 10 ticks of [J(cheap action) − J(full action)].
- Frames with fewer than 10 ticks (route end) are dropped.
- PDM-Lite's own response to the full mode cannot be observed without one fork per frame. The gate is therefore
  trained on the benchmark's decision value, as the paper's allocators are.

**Features**, taken at the frame's first tick from the cheap output:
- counts at the operating threshold and at the detector floor (0.10);
- confidence sum (kept) and maximum (all);
- summed binary entropy (kept, all);
- largest and summed box area (kept);
- lowest box bottom (kept);
- the ground-plane distance of the nearest kept detection;
- ego speed.

These are E2's gate columns without NAVSIM's driving command, plus the nearest-detection distance that E2's addendum
names.

**Threshold and fit.**
- τ is the lowest out-of-fold prediction (5 folds by route) at which the share of frames at or above it is still
  within 20%.
- The final model is fit on all development frames. No hyperparameter is tuned.

**Reported** (`gate_dev.json`): frames, affected frames (V ≠ 0), out-of-fold Spearman with V, and the out-of-fold gain
at τ against random's expectation and against the oracle at 20%.

### Test

- **Routes:** the other 170 Bench2Drive routes.
- **Runs:** gate and random, three repetitions each with identical settings, as in (a): 1,020 runs.
- **References:** the ladder's cheap and full runs of the same routes and repetitions (0% and 100% escalation).
- **Crashes:** as in (a). A run without a record is rerun, up to three crashes; it is reported missing after that.
- **Machines:** the runs are split between two machines with the same GPU (RTX A5000), the same driver, and the
  same software from a shared work folder.
  - The runs are shared out as the machines free up. In each pass, a machine with a free job slot claims the next
    run; the claim is an atomic folder creation in the shared workspace, so no run is handled by two machines.
  - Every machine finishes a pass before any machine starts the next one.
  - The machine of each run is recorded (its tick-log folder). Policy means are also reported per machine, as a
    check.

**Reported** (`routing.csv`, `routing_runs.csv`, `routing_escalation.csv`):
- per policy: driving score, success, route completion, vehicle collisions per run, TickRuntime stops, the realised
  share of escalated frames, and the share of full-mode ticks;
- paired differences of route means (route bootstrap, 1,000 draws, seed 0): gate − random (primary), gate − cheap,
  random − cheap, full − gate, full − random, full − cheap.

### Reading

- **Primary:** gate − random on driving score.
  - Interval above 0: the gate's choice helps beyond random.
  - Interval below 0: the gate's choice is worse than random.
  - Otherwise: unresolved.
- **Sensitivity, from the final (a) before any routing run** (`routing_sensitivity.json`, 203 `--stage
  sensitivity`; the 170 test routes, three repetitions):
  - **Run-to-run noise:** the within-route variance of the driving score, pooled over branches, is 50.0. Two
    policies with equal effect would differ in a route mean by SD √(2 × 50.0 / 3) = 5.8. The minimum detectable
    difference (80% power, two-sided 5%) is then 2.8 × 5.8 / √170 = **1.2 points**.
  - **Observed spread:** the route-mean full − cheap difference has SD 11.9 on these routes, which gives a minimum
    detectable difference of **2.6 points**.
  - **Headroom:** full − cheap on these routes is **+0.6 [−1.1, +2.4]** (169 routes; route 11755 has no cheap or
    full run). A frame-selection policy at 20% would not be expected to exceed full.
  - **Consequence:** the minimum detectable difference exceeds the headroom, so no resolved gate − random difference
    is expected. (c) reports its intervals as they come out.
- **Matching:** if the gate's realised escalation share on test lies outside 15–25%, the comparison with random is
  flagged as not matched in budget.

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

---

## Addendum E11 — an end-to-end camera planner on NAVSIM navtest

Addendum to `followup_protocol.md` (Sections 3, 4 and 6 apply), for follow-up E11 (stretch, run last).
Committed before any value of E11 is computed.

### Question

Does the sign variation of decision value reach a planner without a separate detection stage? The planner here maps
camera images to a trajectory end to end. The escalation is the resolution of its camera input.

### Planner

- **Agent:** NAVSIM's Latent TransFuser (LTF), the official checkpoints released by NAVSIM's authors
  (`autonomousvision/navsim_baselines`, `ltf/ltf_seed_0.ckpt`; Apache-2.0). It runs with NAVSIM devkit v1.1's own LTF
  agent and configuration, unchanged.
- **Seeds:** seed 0 is primary. Seeds 1 and 2 are secondary, if time allows.
- **Inputs:** the agent's own input, the stitched front cameras CAM_L0, CAM_F0 and CAM_R0, plus ego status. LTF uses
  no LiDAR.

### Escalation

- **Full:** the released camera images.
- **Cheap:** each of the three images downsampled by 2 per side (OpenCV INTER_AREA), then upsampled back to its size
  (INTER_LINEAR), before the agent's own preprocessing. This is the 2× per-side ratio of the benchmark's 320 → 640
  modes. The agent and everything after the images are unchanged.

### Data and scoring

- **Tokens:** navtest (12,146 tokens, 136 logs), as in E2.
- **Images:** CAM_L0 and CAM_R0 come from the public OpenScene test camera archives, kept for navtest's logs only;
  CAM_F0 as in E2.
- **Score:** NAVSIM v1.1 `pdm_score` with the default parameters and E2's navtest metric cache.
  - (Added 2026-10-19, before any value of E11.) E2's navtest metric cache is no longer on the training machine's
    shared volume. It is rebuilt the way E2's was: NAVSIM's own caching script, devkit v1.1 (3e8291b), navtest split,
    default parameters. The file count is recorded with the run.
- **Decision value:** V = PDMS(full) − PDMS(cheap) per token. Also per sub-score, as in E2.

### Gate E11-0 (stops E11 until resolved)

The full branch's mean PDMS on navtest reproduces LTF's published navtest PDMS (NAVSIM v1 leaderboard, seed 0, or the
three-seed mean where only that is published) within 1.0 point. This checks the cameras, the checkpoint, the devkit
and the cache, before any decision value.

### Quantities

- **Setting level (as E2):** affected tokens, harmed share, ρ, all-full and oracle@20 reduction, relative to the
  all-cheap mean PDMS shortfall. Intervals from 1,000 bootstrap draws by log, seed 0.
- **Sub-scores:** each with its affected count. ρ is undefined below 30 affected tokens (E2's rule).
- **Allocators:** not run. The agent has no cheap-side detection list for the benchmark's features.

### Reading

- **PDMS, harmed share ≥ 30% with ρ > 0:** the sign variation reaches an end-to-end camera planner.
- **Harmed share below 30%:** reported as a planner where it does not.
- **All-full reduction:** reported with its interval, whatever its sign.

### Prior runs

- No learned end-to-end NAVSIM agent had been run with this repository's code before this addendum.

---

## Addendum R1 — training-target swap on XL

Follow-up round 2, item R1 of six (order R1 → R6 → R5 → R4 → R3 → R2). Committed before any value of R1: no label, model,
score or gain of R1 has been computed. The XL box dump (inputs only) and a timing check of the official label code
(seconds only, every value discarded) were run before this commit; both are described below.

### Question (from the round's protocol)

Does training the same allocator on decision value instead of a perception-level target change the realized decision
gain it delivers?

### Learners

gate_gbm (primary) and gate_ridge (secondary), with X2's features, hyperparameters, seeds and fit units: fit on the
385 + 152 new training and validation scenes, scored once on the 228 new test scenes. Only the target changes.

### Targets

| name | per-frame target | source |
|---|---|---|
| V | J(cheap) − J(full) of the cell, as in X2 | XL tables (170) |
| fn_gain | false-negative gain, `dE_E1_fn_only` | XL tables (170), the benchmark's definition |
| risk_gain | risk-weighted gain, `dE_E6_risk_weighted` | XL tables (170), the benchmark's definition |
| Q | ORIC's offloading reward (Qiu et al.) | 148's `labels_q`, unchanged |
| G | the AP-swap objective of budget-adaptive routing (Geng et al.), dataset-wide dAP | 148's `labels_g`, unchanged |

- **Primary:** every target enters raw.
- **Secondary:** every target rank-transformed on the fit units; Q and G in their published CDF form.

### Primary endpoint

- **Cell:** braking controller, monocular geometry, YOLOv8s 192×320 → 384×640, shared threshold 0.25, shared action
  history; 20% selection budget; gate_gbm.
- **Statistic:** for each of the four other targets t, D_t = realized gain(V-trained) − realized gain(t-trained), as a
  share of the all-cheap loss, with its 95% interval (scene bootstrap, 1,000 draws, seed 0, every draw kept, paired
  within each draw).
- **Test:** two-sided bootstrap p = 2 · min(share of draws with D_t* ≤ 0, share with D_t* ≥ 0); Holm over the four
  comparisons at family-wise 0.05.
- **Reading:** each comparison is "V-trained better" (Holm rejects, D_t > 0), "t-trained better" (Holm rejects,
  D_t < 0) or "not resolved".

### Gate R1-0 (implementation)

The V-trained gate_gbm, refit by this item's code, reproduces X2's test rows exactly: gain, random gain, gain over
random with its every-draw and filtered intervals, its shares of the all-cheap loss and nDG, for every cell, budget and
bootstrap unit of `results/followup/E1/x3_budget_sweep.csv`. X2 did not keep its per-frame scores, so every number
derived from them is compared. A failure stops R1 only.

### Secondary

Oracle geometry; the planner cells (ADE, FDE); budgets 5, 10, 30, 40 and 50%; the log bootstrap; nDG beside realized
gain; gate_ridge; the rank and CDF variants.

### Implementation

- **Script:** `scripts/227_r1_target_swap.py`, stages `dump`, `labels`, `targets`, `gate`, `swap`.
- **Cells:** X2's (`rap.xl.cells` on `data/cache/xl/tables850`, the new 765 scenes and X2's split, the ego lift frame,
  X2's causal ego-speed table), with X2's features (`rap.xl.features` on `data/cache/nusc_det_xl`).
- **Models:** `rap.xl.gate_scores` (X2's call): `rap.predict.make_model(kind, "reg", 0)`, seed 0, fit on the frames of
  the training and validation scenes, scores on every frame; only the test frames are scored.
- **Gains and draws:** X2's functions (`rap.xl.prepare`, `paired_draws`, `paired_rows`): realized gain is the sum of V
  over the top-k test frames by score (k = round(budget × frames), ties by the benchmark's exact expectation); one
  stream (seed 0) per bootstrap unit over all cells, as X2, so every series of a cell shares each draw.
- **Q and G:** 148's functions on the XL frames:
  - boxes: the cached cheap (weak) and full (strong) detections of `data/cache/nusc_det_xl`, and the benchmark's
    reference objects (at least 10 px tall, coarse class) from the XL dataroot, as 148's dump (`--stage dump`);
  - Q: the official `reward.compute_orie` (edgeml-object-detection@859f702), |E| = 1,000, the global RNG seeded 42 at
    each call of `labels_q`, detections with confidence ≥ 0.10;
  - G: the official `proxy_metrics.compute_dataset_wide_oric` (bgt-ada@6669ab0), detections with confidence ≥ 0.30,
    `oric_allpoint`, context 1,000 frames (default 3); the official worker pool, capped at 6 workers through the
    open-file limit (`ulimit -n 352`) for the Jetson's 14 GB (the pool's size does not change the values);
  - each computed on the fit units and on the test units separately, each set its own context, as in 148.

### Defaults for points the protocol leaves open

1. **False-negative and risk-weighted gain:** the XL tables' `dE_E1_fn_only` and `dE_E6_risk_weighted`, the
   benchmark's definitions, as the frozen target-swap study used them (122). They do not depend on the lift's geometry
   (checked by the code).
2. **Label sets:** as in 148, each cell's labels come from its own frame set. The brake cells (30,773 frames: 21,592
   fit, 9,181 test) share one set and the planner cells (24,271: 17,037 fit, 7,234 test) another; frames in table
   order.
3. **G's context: 1,000 frames** (`context_size = 1000`, one draw, the code's own RandomState(42)), the official
   code's default and the variant the frozen study computed beside its primary (`G_dap_ctx1000`). The frozen
   primary, context 0 (every frame of the set), is not feasible on XL: on the Jetson, the official code took 194 s,
   618 s and 2,868 s for the first 1,000, 2,000 and 4,000 fit frames (worse than quadratic), which puts the four
   sets at about 60 hours. Context 0 also measures each frame against 21,592 frames here, against about 2,300 in
   the frozen study; 1,000 is closer to that scale. Q costs 0.02 s per frame (1.9 s for 100 frames of the
   21,592-frame set) and keeps 148's settings. Only timings were read; every label value was discarded.
4. **Test-unit labels:** computed (as in 148) and written to `targets.csv`; no model sees them.
5. **Rank variant:** 159's rank transform (ordinal ranks / n on the fit units, ties as 148's Q-MORIC), for all five
   targets. Its D_t compares the rank-trained V and t models.
6. **CDF variant:** 148's Q-MORIC (Qiu's ordinal rank) and G-MORIC (Geng's ECDF), fitted on the fit units. Its D_t
   compares with the raw V-trained model (X2's).
7. **Shared fits:** a non-V target does not depend on the cell's geometry or controller, so its model is the same for
   the mono and oracle cells (and for ADE and FDE); it is fit once per frame set and learner.
8. **Denominator:** D_t and its interval are divided by the all-cheap loss of the cell's test frames (point value), as
   X2's shares.
9. **p and Holm:** p over all 1,000 draws, capped at 1; Holm with `rap.xl.holm(p, alpha=0.05)` over the four primary
   comparisons only.

### Outputs

- `results/followup/R1/targets.csv`: per frame: seq, frame, split, log, V of each cell, fn_gain, risk_gain,
  Q and G of each label set, and their CDF forms on the fit units.
- `swap.csv`: one row per learner × variant × target × cell × budget × bootstrap unit: gain, random gain, gain over
  random with intervals, nDG, D_t with interval and p, and for the primary rows Holm and the reading.
- `gate_r1_0.json`.
- Doc: `results/followup/R1/`.

---

## Addendum R2 — closed-loop dynamics under replayed perception (nuPlan)

Follow-up round 2, item R2 (last). Committed before any value of R2.

### Scope (from the round's protocol)

Detector outcomes are replayed by logged time. At each simulation step the observation is built from the logged frame
at that time, not from what a camera would see at the simulated pose. R2 therefore measures closed-loop dynamics under
replayed perception, not closed-loop sensing.

### Setting

- **Scenarios:** the 60 scenarios of the nuPlan real-perception track. **Planners:** PDM-Closed (primary) and IDM
  (secondary), released configurations (82's `make_planner`, 82's route handling).
- **Simulation:** nuPlan closed loop, non-reactive agents (logged), 0.1 s steps, the default ego controller (two-stage:
  LQR tracker with the devkit's default gains, kinematic bicycle model, Pacifica parameters).
- **Observation per step:** the logged frame nearest the step time, within 50 ms, built exactly as the open-loop track
  builds a branch (114 / 115): logged tracks inside the front-camera image kept only if the step's mode detected them;
  unmatched detections as false positives on the road plane, fixed in world coordinates at the logged pose; tracks
  outside the image as logged. Steps without a frame within 50 ms reuse the previous observation; they are counted.
- **Single-step V:** the track's open-loop safety-loss V, extended from the 1,440 states to every logged frame of the
  60 scenarios (`v_all_frames.csv`).
- **Closed-loop loss:** J over a window = Σ over steps of (10 · 1[collision] + 1.5 · c²), on the executed ego state
  against the logged agents; c = the shortfall of the minimum ego–agent clearance below 1 m, clipped at 2 m. Secondary:
  + 0.1 × the ego's mean deviation from the logged trajectory.
- **Gate R2-0 (implementation only):** (a) two identical cheap-only forks give ΔJ = 0 on 20 frames; (b) on 100 sampled
  states, the first plan of a fork whose first step uses the full mode equals the open-loop full-branch plan at that
  state within 1e-6 m.
- **Forks:** at every logged frame t with V_t ≠ 0, a simulation from the logged state at t with its history. Branch
  "full": the full mode at the first step, cheap afterwards; branch "cheap": cheap throughout. Window 8 s (80 steps);
  forks with less than 8 s left are excluded from the primary analysis and reported in a truncated-window row.
  ΔJ = J(cheap) − J(full).
- **Primary endpoint:** the share of forks with ΔJ ≠ 0 whose sign equals sign(V_t), with a log-bootstrap interval
  (resampling the logs of the 60 scenarios). Reading: lower bound > 0.5 — the single-step sign predicts the closed-loop
  direction; upper bound < 0.5 — it contradicts it; otherwise unresolved. Beside it: the share of forks with ΔJ = 0,
  and Spearman ρ(V_t, ΔJ) with its interval.
- **Policies over whole scenarios (principal secondary):** all-cheap and all-full (anchors); privileged (full at every
  step whose logged V_t > 0); matched random (per scenario the same number of full-mode steps as privileged, placed
  uniformly, mean over 20 seeds). Comparison: privileged − matched random on J over the whole scenario,
  scenario-paired, log bootstrap. Also gate_gbm at 20% on the test-log scenarios (the nuPlan track's fit) and nuPlan's
  closed-loop score components for every policy.

### Implementation

- **Script:** `scripts/232_r2_closedloop.py` (nuPlan environment), stages `spec`, `vall`, `gate`, `forks`, `policies`,
  `analysis`; `--shard i/n` for the per-scenario stages; `--planner pdm_closed | idm`.
- **Machine:** the training machine's CPU (the nuPlan environment, databases and maps of the shared workspace, as E6
  used them); the Jetson is busy with R3.
- **Specification at every iteration:** 114's construction unchanged (nearest fetched CAM_F0 image, projection,
  Hungarian matching at IoU 0.3 with class compatibility, threshold 0.25, the false-positive lift), branches cheap and
  full, at every DB iteration of the 60 scenarios. Checked: equal to 114's specification on all 5,760 (state, history
  iteration) keys it shares; a difference stops R2.
- **Logged frames:** the 0.1 s grid of each scenario from iteration 4 (the track's first state iteration) in steps of
  two DB iterations, keeping the frames with an image within 50 ms.
- **Single-step V at a frame:** exactly 115's call: a fresh planner per branch, initialised at the frame, the 4-iteration
  history (20 Hz) with every observation through the branch's filter, 82's `score` against the true objects over the
  plan's horizon, 83's safety loss; V = safety(cheap) − safety(full). Both planners.
- **Rollout:** a planner initialised at the start frame with the same history as the open-loop call; at step s the plan
  is computed at DB iteration k0 + 2s, the controller propagates the ego to k0 + 2(s + 1), and the loss is evaluated
  there against the logged objects (collision = footprint overlap; clearance = footprint distance, as 82's `score`).
  The history buffer appends (executed ego, observation) at each step.
- **Gate R2-0:** (a) 20 frames drawn with `default_rng(0)` among the PDM-Closed frames with V ≠ 0, two all-cheap
  forks each; (b) 100 of the track's 1,440 states drawn with the same generator, the open-loop full plan against the
  fork's first plan, maximum absolute rear-axle difference over the sampled trajectory.
- **Statistics:** 1,000 log-bootstrap draws, seed 0, every draw kept, 95% percentile intervals.

### Defaults for points the protocol leaves open

1. **Planners for forks and policies:** PDM-Closed first (primary); IDM after it with the same code (secondary).
2. **History at a fork's start:** the open-loop state's (four DB iterations at 20 Hz through the start mode's filter),
   so the first plan equals the open-loop plan (gate b); later steps append at 0.1 s.
3. **Loss evaluation:** after each step's propagation (80 evaluations in an 8 s window); a step's collision counts
   once per step.
4. **Policies' start and end:** from iteration 4 to the last full step of the scenario; a step's mode follows the
   policy at that step's frame; privileged uses the planner's own V_t (frames without V count as V ≤ 0).
5. **gate_gbm at 20%:** 120's 18 gate features (119's definitions, cheap branch, previous iteration for its change
   feature), computed at every logged frame; the benchmark's GBM with seed 0, refit on the track's 1,440 states of the
   training and validation logs on the planner's safety V; the top 20% of the test-log scenarios' frames by score,
   pooled, ties by (scenario, iteration); one run per test-log scenario.
6. **nuPlan's closed-loop score components:** the loop stores each policy run's executed ego track (position, heading,
   speed, acceleration, contact) per step; the components are computed from those tracks after the runs, with nuPlan's
   metric definitions where they need only the ego track and the logged agents. Any component that needs nuPlan's full
   simulation history is reported as not computed.
7. **Order:** R2 starts while R3 waits for its Jetson session (a deviation from the round's order, recorded in the
   index): it runs on the training machine, so the two items share no resource and no value.

### Outputs

`results/followup/R2/`: `v_all_frames.csv`, `gate_r2_0.json`, `forks.csv` (and `forks_<planner>.csv`
per fork), `policies.csv`, `policy_runs.csv`; doc `results/followup/R2/`.

---

## Addendum R3 — what the gate's features cost

Follow-up round 2, item R3 (fifth, after R1, R6, R5 and R4). Committed before any value of R3.

### Design (from the round's protocol)

- **Feature groups:** the four existing groups of the gate's 65 features: confidence statistics (10); uncertainty (14);
  scene complexity (22: counts, areas, overlaps, image and motion statistics); criticality from the cheap detections'
  monocular geometry (19). Subsets: the 15 non-empty combinations. No new feature.
- **Cost of a subset, measured as executed, never summed from groups:** an extractor computing exactly the subset's
  groups, profiled on the Jetson at the frozen track's timing boundary, on 200 fixed nuScenes frames, 250 repetitions
  in 3 interleaved rounds (the protocol of x7_costs.csv); median ms per frame and module mJ. Model inference per fitted
  model: gate_gbm batched (primary) and one call per input; gate_ridge. C_S = extraction + the model's inference. The
  cheap and full pipelines are measured in the same session and the budgets use that session's profile.
- **Budget rule:** the benchmark's measured-budget scorer, b = C_c + α·C_f, with C_S charged on every input.
- **Selection, on validation only:** for each learner, each of the 15 subsets fit on the 385 training scenes; realized
  gain on the 152 validation scenes at the 20% latency budget with the subset's C_S charged; the maximum, ties toward
  the lower C_S. The chosen subset is refit on the 537 scenes and scored once on the 228 test scenes.
- **Primary endpoint:** the chosen gate_gbm subset against random at the 20% latency budget, primary cell, paired
  realized gain, scene bootstrap.
- **Secondary:** gate_ridge; latency budgets 10, 30, 50%; module-energy budgets at the same α; oracle geometry; the
  cost–gain table of all 15 subsets on the test scenes (descriptive, not used for selection). Frozen-85 and KITTI
  braking without retuning: the subset chosen on XL validation, fit on each track's non-test units, scored once on its
  test split.

### Implementation

- **Script:** `scripts/231_r3_gate_cost.py`, stages `profile` (exclusive Jetson session), `select`, `test`, `frozen`.
- **Groups:** `rap.features`' registry (`conf`, `unc`, `complex`, `crit`), the 65 leakage-checked columns of X2's
  features. Extractor for a subset: `frame_features`' group functions for the subset's groups only, in its order, on
  the cheap detections (threshold 0.25), with the model's input vector built inside the timed call.
- **Profile:** an idle rail baseline (10 s); then 3 rounds, each over every item in a fixed order (15 extractors; per
  subset gate_gbm per input, gate_gbm batched over the 200 frames, gate_ridge per input), each item 40 warm-up and 250
  timed calls cycling over the 200 frames; rails sampled every 50 ms during the timed calls. Then, in the same
  session, the cheap (192×320) and full (384×640) YOLOv8s TensorRT engines with `01_profile_jetson.py` exactly as
  `x7_costs.csv` was measured (real CAM_FRONT frames, 250 repetitions, 40 warm-up, 3 rounds). Round 0 discarded,
  median of the kept rounds; module mJ = Σ over GPU, SOC, CPU of (rail mean − idle mean) × median ms (133's rule).
- **Models:** X2's (`rap.predict.make_model(kind, "reg", 0)`), fit on V of the primary cell.
- **Budget:** 93's rule: escalated share = (b − C_c − C_S) / C_f, k = floor(share × N); a subset with C_S above
  α·C_f is infeasible at that budget. Random pays no overhead (k = floor(α × N)); its gain is its expectation.
- **Bootstrap:** 1,000 draws, seed 0, test scenes resampled; k recomputed per draw; paired difference chosen − random.

### Defaults for points the protocol leaves open

1. **The 200 fixed frames:** the first 200 frames of the 228 XL test scenes in (scene, frame) order, from the XL cheap
   cache, as 93's `_frames` builds them.
2. **Timing boundary:** the frozen track's for the gate (93, 151): the inputs are the cheap detections, their
   monocular geometry, and the image statistics the cheap pipeline already produces (cached beside the detections);
   the timed call computes the subset's features and builds the model's input. The image statistics themselves are
   part of the cheap pipeline in the frozen definition and are not charged to the complexity group.
3. **Inference profiles use models fit on the 385 training scenes**, the models selection uses; inference cost does not
   depend on the training targets.
4. **Energy budgets:** b = C_c + α·C_f in module mJ with C_S in module mJ, the same subsets and models.
5. **Frozen-85 and KITTI:** the frozen tracks' braking cells, monocular geometry (92's cells: the shipped features and
   splits), fit on train + validation units; engine costs from `results/final/cost_registry.json` (the frozen profile
   of each track); C_S from this session's nuScenes profile, also for KITTI (no KITTI frames are profiled).
6. **Exclusive session:** held while a pause file exists; fan at 100%.

### Outputs

`results/followup/R3/`: `feature_costs.csv`, `subsets_val.csv`, `selected.json`, `test.csv`,
`frozen_kitti.csv`; doc `results/followup/R3/`.

---

## Addendum R4 — allocators that may leave budget unused

Follow-up round 2, item R4 (fourth, after R1, R6 and R5). Committed before any value of R4.

### Design (from the round's protocol)

- **Allocators:** the four regression allocators (gate_gbm, gate_ridge, R1_gbm_reg, R1_mlp_reg). Fit on the 385
  training scenes only; the same train-only fit is used for both rules, so a threshold stays on the model's scale.
- **Rules:**
  - quota: escalate exactly k = round(α·N) inputs by score;
  - abstention: escalate the inputs whose predicted V exceeds τ, at most k, highest first.
- **Choice of τ:** per allocator and budget, on the 152 validation scenes, from the grid {−∞ (= quota), 0, and the
  50/60/70/80/90/95th percentiles of the validation predictions}. Pick the value with the largest realized gain; break
  ties toward fewer escalations. τ is fixed before the test scenes are scored once.
- **Primary endpoint:** gate_gbm, primary cell, 20%: realized gain(abstention) − realized gain(quota), same model,
  scene bootstrap.
- **Secondary:** the other three allocators, budgets 5–50%, oracle geometry; the fixed rule τ = 0 without any choice;
  each rule against random and against no escalation.
- **Report for every row:** the escalated count and the fraction of the budget used (escalated / k).

### Implementation

- **Script:** `scripts/230_r4_abstention.py`, stage `run`.
- **Cells:** X2's braking cells (monocular primary, oracle secondary) on the new 765 scenes; X2's features
  (`rap.xl.features`), models and seeds: `rap.xl.gate_scores` for gate_gbm and gate_ridge, 103's `fit_score` on the
  detection-list matrix for R1_gbm_reg and R1_mlp_reg, every one fit on the training scenes' frames only.
- **Realized gain:** the sum of V over the escalated test frames, with the benchmark's exact tie expectation at the
  k-th score (`rap.xl.gains`). Shares of the all-cheap loss of the test frames.
- **Bootstrap:** 1,000 draws, seed 0, test scenes resampled with replacement; in each draw k = round(α × frames of the
  draw), both rules evaluated on the same draw (paired), τ fixed.

### Defaults for points the protocol leaves open

1. **Abstention on a draw:** escalated count = min(k, frames of the draw with score > τ); the escalated frames are the
   highest-scoring among those.
2. **Against random:** random escalation of the same number of frames as the rule escalates (its expectation,
   count / N × ΣV); against no escalation: the rule's gain itself (no escalation gains 0). Both with the paired
   interval.
3. **τ = −∞ in the grid** is the quota rule; a chosen τ of −∞ makes abstention equal to quota (D = 0 exactly).
4. **Percentiles:** `numpy.percentile` (linear) of the allocator's predictions on the validation frames.
5. **Ties in the choice of τ:** equal validation gain → the τ escalating fewer validation frames.
6. **Planner cells:** not part of R4 (braking only, as the protocol's primary cell and its secondary list).

### Outputs

`results/followup/R4/`: `thresholds.json` (chosen τ and the validation grid per allocator, geometry and
budget), `abstention.csv` (one row per geometry × allocator × budget × rule); doc `results/followup/R4/`.

---

## Addendum R5 — when escalation helps: prevalence, event onsets, scene counts

Follow-up round 2, item R5 (third, after R1 and R6). Committed before any value of R5.

### Primary endpoint (from the round's protocol)

- Per scene s of the 850: prevalence p_s = share of frames whose reference requirement a⋆ ≥ 1.0 m/s² (the
  controller's decelerate-or-brake region), and v̄_s = mean V per frame of the scene (scenes without an affected frame
  contribute v̄_s = 0).
- Statistic: Spearman ρ(p_s, v̄_s) over the 850 scenes, with its scene-bootstrap interval; primary cell.

### Secondary

- Deciles of p_s with the pooled relative all-full reduction ΣV / ΣJ_cheap per decile.
- The relative reduction per scene (scenes with ΣJ_cheap > 0) in place of v̄_s.
- Oracle geometry.
- The same analysis on the 228 test scenes for the ten E5 replacement pairs and the two fine-tuned pairs.

### Event onsets per hour (secondary, weight-free)

- An episode is a maximal run of consecutive frames of one scene that satisfies the event condition; a gap above
  0.75 s between frames breaks a run. Onsets are counted.
- Hours = the sum over scenes of the scene's time span, each frame given its interval to the next frame and the last
  frame of a scene the median interval.
- Events: unnecessary hard braking (commanded hard braking while a⋆ < 1.0); missed braking (a⋆ ≥ 3.5 while the
  commanded action is keep speed). Secondary events: any braking or deceleration while a⋆ < 1.0; a command below the
  level a⋆ requires.
- Policies: all-cheap and all-full; oracle at 20% (top 20% of frames by V within the population); gate_gbm at 20% (the
  X2 fit, test scenes only); random at 20% (mean over 20 seeds). Commands follow each policy's per-frame mode under
  the shared action history.
- Populations: the 228 test scenes for every policy; all 850 for all-cheap, all-full and oracle.

### Scene counts (secondary)

Among scenes with at least one affected frame, the share in which ΣV < 0, with its interval; primary cell, all 850.

### Implementation

- **Script:** `scripts/229_r5_prevalence.py`, stages `pairs`, `prevalence`, `events`, `scenes`.
- **Primary cell:** the XL braking tables (`data/cache/xl/tables850`, monocular geometry, 320 → 640, shared threshold
  0.25, shared action history), all 850 scenes (34,149 frames); a⋆ = the tables' `a_gt`; V = J_cheap − J_full.
- **E5 pairs:** E5's ten replacement pairs and two fine-tuned pairs at each pair's primary operating point (shared 0.25
  within a family, equal box count across families, the thresholds of `x7_upgrades.csv` / `x7_finetuned.csv`), tables
  rebuilt on the 228 test scenes by `decision.build` exactly as 177 built them (monocular geometry), detections from
  `data/cache/xl_models`.
- **gate_gbm at 20%:** X2's gate_gbm (`rap.xl.gate_scores`, fit on the 385 + 152 new scenes; gate R1-0 showed the refit
  equal to X2), scored on the 228 test scenes' frames.
- **Statistics:** scene bootstrap, 1,000 draws, seed 0, every draw kept, 95% percentile intervals.

### Defaults for points the protocol leaves open

1. **a⋆ thresholds:** the controller's `thr_decel` (1.0 m/s²) and `thr_hard` (3.5 m/s²). "Commanded hard braking" =
   the HARD_BRAKE action, "keep speed" = KEEP, "braking or deceleration" = DECELERATE or HARD_BRAKE; "a command below
   the level a⋆ requires" = the action below `discrete_action(a⋆)`.
2. **Spearman interval:** Spearman ρ recomputed on each draw of scenes (resampled with replacement); a draw where ρ is
   undefined is counted and left out of the percentiles.
3. **Deciles:** scenes sorted by p_s, ties by scene name, cut into ten groups of equal size (85 scenes on 850); the
   interval of each decile's pooled reduction resamples that decile's scenes.
4. **Frame intervals:** each frame's interval to the next keyframe of its scene, from E10's per-frame time steps
   (`data/cache/xl/e10/tables`, the same 850 scenes); the last frame of a scene gets that scene's median interval.
5. **Episode continuation:** a flagged frame continues the previous frame's episode when that frame of the same scene
   is flagged and at most 0.75 s earlier.
6. **Policy selections:** oracle and gate take the top round(0.2 × frames) of the population by V or by score, ties
   broken by (scene, frame); random draws that many frames uniformly without replacement with `default_rng(seed)`,
   seeds 0–19, and reports the mean onsets.
7. **Onset intervals:** reported as counts and rates without intervals (secondary, descriptive).

### Outputs

`results/followup/R5/`: `prevalence_scenes.csv`, `prevalence_deciles.csv` (Spearman rows and decile rows),
`events.csv`, `scene_counts.csv`; doc `results/followup/R5/`.

---

## Addendum R6 — validity checks on XL

Follow-up round 2, item R6 (second, after R1). Committed before any value of R6.

### Primary endpoint — reference completeness (from the round's protocol)

- Recompute braking V with the reference including the boxes the visibility rule excludes: annotations with no LiDAR
  or radar point, and boxes under 10 px high.
- Report harmed share and ρ with intervals on the 228 new test scenes (primary) and on all 850.
- **Reading:** harmed share ≥ 30% with ρ > 0 means "not a labelling artefact"; otherwise report where it fails.
- Also report the share of added false positives the controller considers that overlap an excluded annotation (image
  IoU ≥ 0.3).

### Attribution of harm (secondary, deterministic, non-overlapping)

On every harmed frame (V < 0; primary cell, 228 test scenes; repeated on all 850), the following interventions are
applied to the full branch cumulatively, in this order. The first step after which V ≥ 0 is the frame's cause:

1. remove every full-mode detection that matches no box of the completed reference (IoU < 0.5): "added detection
   without a real object";
2. for objects the full mode detects and the cheap mode misses, replace their geometry (range, lateral extent, time
   to collision) by the reference box's: "recovered real object misplaced by the lift";
3. for objects both modes detect (matched to the same reference box), set the full branch's geometry to the cheap
   branch's: "geometry change of a jointly detected object";
4. otherwise: "other".

Counts and shares per cause; as a descriptive table, the share of harmed frames fixed by each intervention alone.

### Controls (secondary)

320 → 640, both geometries, 228 test and 850 scenes:
- per-mode operating points S1–S3, thresholds set on the 152 validation scenes with the paper's rules;
- a persistence filter requiring a detection in the two previous keyframes of the same mode;
- a safety-only braking loss (shortfall and collision terms only).

### Manual-audit export

100 harmed frames drawn at random (seed 0) from the primary cell on the 228 test scenes: cheap / full image pairs in
the gallery format with the controller's actions, and a CSV with frame id, V, cause from the attribution and an empty
label column (phantom, real_misplaced, real_missing_from_reference, real_correct, unclear).

### Implementation

- **Script:** `scripts/228_r6_validity.py`, stages `frames`, `completeness`, `attribution`, `thresholds`, `tables`,
  `controls`, `audit`.
- **Frame loop:** `decision.build`'s loop as 118 rebuilds it (per-object state), on all 850 XL scenes, YOLOv8s
  192×320 → 384×640 from `data/cache/nusc_det_xl`, shared threshold 0.25, shared action history, monocular and oracle
  geometry. Every frame's J of both branches under the standard reference is asserted equal to the XL tables
  (`data/cache/xl/tables850`), so the loop is the tables' own.
- **Completed reference:** `NuScenesDB.sequence_geometry(scene, min_pts=0, min_px=1)` and no 10 px height filter. The
  standard reference is the same call with its defaults (min_pts 1, min_px 4) and the 10 px filter. `min_px` is a new
  parameter whose default (4) keeps every earlier build unchanged. A box is "excluded" when its instance is not in the
  frame's standard reference.
- **Recomputed V:** a⋆ from the completed reference, each branch's action unchanged (the detections and the cheap
  branch's previous action do not depend on the reference), J of both branches recomputed. Statistics as X1:
  affected, harmed share, ρ, all-full and oracle@20 reductions, scene bootstrap 1,000 draws, seed 0.

### Defaults for points the protocol leaves open

1. **Completed reference geometry:** the standard procedure run on the completed annotation set. Range rate (hence
   TTC) comes from neighbouring keyframes of the same track, so a standard box's TTC can change where its track gains
   frames. Boxes that do not cover a pixel of the image in each direction (outside the image after clipping) stay out:
   they are outside the camera, not excluded by visibility. The category filter (the evaluated classes) is unchanged.
2. **Primary cell for completeness:** monocular geometry, the primary cell everywhere on XL. With the monocular lift
   only a⋆ depends on the reference.
3. **"Added false positives the controller considers":** full-mode detections that match no standard reference box
   (IoU < 0.5, the benchmark's greedy matching), with no cheap detection at IoU ≥ 0.5, inside the controller's
   corridor and range (monocular geometry of the full branch). Share = those overlapping an excluded box at image
   IoU ≥ 0.3, pooled, with a scene-bootstrap interval.
4. **Attribution:** matching to the completed reference (118's greedy assignment, IoU ≥ 0.5); "detected by a mode" =
   a detection of that mode is assigned to the box. V after each step uses the standard reference's a⋆ (the frames
   are harmed under it) and the cheap branch's previous action. A frame whose V stays below 0 after the three steps
   is "other". The recomputed V without intervention is asserted equal to the frame's V.
5. **S1–S3:** 100's stage-B rules on the 152 validation scenes: S1 count-matched (full threshold matching cheap's
   detections per frame at 0.25, `rap.tables.match_detection_counts`), S2 F1-optimal per mode on the 0.05 grid, S3
   the lowest full threshold on the 0.01 grid whose pooled precision reaches cheap's at 0.25. Pooled precision and
   recall against the standard reference (`rap.percep_metrics.frame_losses`). Tables rebuilt by `decision.build`
   with those thresholds.
6. **Persistence filter:** a detection is kept when the same mode has a detection of the same coarse class at image
   IoU ≥ 0.3 in each of the two previous keyframes (each at the mode's operating threshold); a scene's first two
   keyframes keep their detections (no history). Applied to both modes before the controller.
7. **Safety-only loss:** J = the shortfall and collision terms of the braking loss (`Jterm_*_shortfall +
   Jterm_*_collision` of the XL tables); the excess-braking and jerk terms dropped.
8. **Audit draw:** the harmed frames of the primary cell on the 228 test scenes, sorted by (scene, frame), 100 drawn
   without replacement with `numpy.random.default_rng(0).choice`. Images: the original image and its gallery JSON
   (118's format, with the completed reference and its exclusions), plus a side-by-side cheap | full rendering with
   the controller's actions.

### Outputs

`results/followup/R6/`: `reference_completeness.csv`, `attribution.csv`, `attribution_single.csv`,
`controls.csv`, `audit/`, `audit_frames.csv`; doc `results/followup/R6/`.

---

## Addendum R7 — influence, tail and selection audit of the XL claims

Follow-up round 3, item R7 (first of R7–R11). Committed before the first new value of R7.

### Post hoc status

R7 is post hoc. On 2026-11-06, before this addendum, an audit that read stored tables only (no refit, no bootstrap) found
the facts below, and the design of R7 was written knowing them. They are quoted here as the reason for the item; the doc
recomputes every number.

- On the 228 new test scenes, ΣV (braking, monocular) is −893.7, of which `scene-1051` alone is −1,316.5. Removing that
  one scene moves all-full from −18.2% to +10.3% on those scenes and from +1.1% to +9.1% on all 850; the harmed share
  stays at 51.0% and 46.6%. Removing the 5 best scenes of the 850 instead gives −7.9%. All-full is decided by a few
  scenes in either direction.
- The 8 scenes with the largest Σ|V| are all night scenes; 7 of them are singapore-hollandvillage, from two logs of
  2018-11-21: `n015-2018-11-21-19-11-29+0800` (1049, 1050 val; 1051 test) and `n015-2018-11-21-19-38-26+0800` (1074,
  1095 train; 1080 val; 1081 test). X2's fit includes the val scenes; R4's train-only fit does not, and R4's quota gain
  for `gate_gbm` is 3.6% against X2's 12.7%.
- `rap.planner.required_decel` computes `a_stop = v_ego**2 / (2*gap)` for every in-corridor object, whether or not it
  moves; closing speed can only raise the requirement. A moving lead vehicle therefore counts as stationary. In
  `scene-1051` frame 32 (`results/followup/R6/audit/scene-1051_32_pair.jpg`) the ego follows a car about 15 m ahead at 12.4 m/s
  at night: a⋆ = 9, the cheap mode brakes hard, the full mode misses the car and keeps speed (J = 87.12, collision flag
  set), and the logged driver keeps speed. Over the 850 scenes, of 842 frames with a⋆ ≥ 3.5 and v ≥ 3 m/s, 87% (of those
  with 2 s of future) show no logged braking and 5% a logged deceleration of at least 1 m/s²; 54 frames carry a
  collision flag in a branch and none of them shows a logged deceleration of 2 m/s² over the next 2 s. Frames with
  a⋆ ≥ 7 carry 39.5% of Σ|V|.
- X2 did not keep per-frame scores, so nobody can say how much of `gate_gbm`'s gain, of R1's D_t or of R3's +12.7% is
  carried by these scenes.
- Over-random figures flatter allocators when random is net harmful: R3's primary +12.7% is a realized gain of 9.1% plus
  random's −3.6%, and random spends 1,836 escalations against the gate's 867.
- No allocator in the repository is trained on the budgeted objective itself: every learned baseline is a regression or
  an unweighted classifier followed by top-k.
- E10's refresh allocators score each non-refreshed frame with the previous frame's detections, whatever the policy
  refreshed earlier (`scripts/206_temporal_escalation.py`, `stale = prev[0]`), so a 20% refresh policy is credited with
  one-frame-old lists rather than the staleness it would really have.

### Item, as protocolled

#### R7 — influence, tail and selection audit of the XL claims

- **Question:** which XL claims hold when a few scenes are removed, against stronger comparators, and away from the one
  test split on which `gate_gbm` was already known to be the winner?
- **Part A, no fit.** For braking (mono, oracle), planner ADE and FDE (mono, as controls), populations pooled 850,
  new 765, 228 test, frozen 85: affected, harmed, ρ, all-full, oracle@20, with intervals, for: all scenes; minus the
  1 / 3 / 5 scenes with the most negative ΣV; minus the 1 / 3 / 5 with the most positive ΣV; minus both 5s; day only;
  night only; minus the two logs named above. Rank scenes on the population at hand; list them.
  Also, for every other braking row in the follow-up block of the paper's Table 1, on its own population: the two
  capacity pairs printed there and the other E5 pairs (`data/cache/xl/r5/pair__*.pkl`), the fine-tuned 320→640 pair, the
  LiDAR primary (E7) and the reuse primary (E10): all-full, ρ and harmed with all scenes and minus the most influential
  scene of that row, with its name. Some of these print an all-full far from zero on the 228 test scenes (−9.26%,
  +36.57%), and nobody has checked what carries them.
- **Part B, stored frame-level scores.** Refit and **save per-frame scores** (`scores_<signal>.csv.gz`) for: `gate_gbm` and
  `gate_ridge` on V, fn_gain, risk_gain, Q, G (R1, raw and CDF forms); the R3 chosen subsets; and, new, `R1_gbm_reg` and
  `R1_mlp_reg` on the same five targets (R1 never ran the routers, while the frozen study found the gradient-boosted
  router leaning towards the AP-swap objective, so this is a fairness check). **Gate R7-0:** the refit reproduces X2, R1
  and R3 test rows exactly.
- **Part C, comparators.** For X2's `gate_gbm`, R1's five gates, R3's two chosen gates, all at their registered budgets:
  realized gain against all-cheap (interval, not only the point) ; against random at the same rate; against random at the
  allocator's own escalation count (matched count). Report the escalated count beside each.
- **Part D, influence.** For each row of Part C and each R1 D_t: share of the gain carried by the top 1 / 3 / 5 scenes;
  the value and interval with `scene-1051` removed, with `scene-1081` removed, with both removed, and with both test
  scenes of those logs removed; the full drop-one-scene range over the 228 scenes.
- **Part E, fit-set sensitivity.** Refit `gate_gbm` on V, test unchanged: on 537 scenes; on 537 minus scenes 1049, 1050,
  1080; on 537 minus every scene of the two logs; on 385. Report realized gain on the 228 and on the 228 minus the two
  logs. This tests whether the jump from 3.6% to 12.7% comes from same-log neighbours of the test scenes.
- **Part F, away from the one split.** 20 repeated log-grouped 5-fold out-of-fold runs over the 765 scenes (log orders
  seeded 0–19): distribution of `gate_gbm`'s realized gain, gain over random, and each of the four D_t (mean, min, max,
  share of repeats with D_t > 0), each repeat also reported without the two logs. Q and G labels stay the ones of R1
  (no recomputation). This removes the selection on the test split, since X2 had already named `gate_gbm` the winner on
  it before R1 chose it as primary.
- **Part G, robust metrics.** Evaluate every row of Part C also on capped decision value: V clipped to ±c, with c the 95th
  percentile of |V| among affected fit frames (fixed on the fit units). Secondary: `gate_gbm` trained on the capped V.
- **Part H, loss weights at scale (exact recombination from `Jterm_*`).** Over-braking weight {0, 0.03, 0.06, 0.12, 0.24,
  0.48}, collision weight {0, 3, 6, 12}, one at a time and jointly at the extremes; pooled 850 and 228 test; harmed, ρ,
  all-full, oracle@20 with intervals. Give the over-braking weight at which ρ crosses 1, if any.
- **Registered reading.** A claim is **robust** if its interval stays on the same side of zero after removing the single
  most influential scene for that claim and in at least 18 of 20 repeats of Part F; **tail-dependent** if the point
  estimate keeps its sign but the interval reaches zero; **fragile** otherwise. Apply this to: R1's four D_t, R3's primary,
  X2's primary, and the statement "all-full is not resolved on the 850 scenes".
- **Outputs:** `results/followup/R7/`, doc `results/followup/R7/`.

### Implementation

- **Script:** `scripts/234_r7_influence.py`, stages `a`, `b`, `cdg`, `e`, `f`, `h`, `reading`, run in that order on the Jetson. Python: the edge environment.
- **Cells, features, models, seeds:** X2's (`rap.xl.cells`, `rap.xl.gate_scores`), R1's targets
  (`results/followup/R1/targets.csv`), 103's models for the routers (`R1_gbm_reg` on one OpenMP thread, as
  103), R3's `fit_predict` for its chosen subsets. No library file changes.
- **Gate R7-0:** in `b`. Every X2 row of `gate_gbm`, `gate_ridge`, `R1_gbm_reg`, `R1_mlp_reg` (six cells, six budgets, scene
  and log) in `x3_budget_sweep.csv`; every raw and CDF gate row of `swap.csv`; every feasible chosen row of R3's `test.csv`
  (gain, count, gain over random, its interval): exact equality. A failure stops R7 only.
- **Bootstrap:** 1,000 draws, seed 0, every draw kept, all quantities of one row on the same draws; each row at scene and at
  log level. Shares use the point all-cheap loss of the frames kept, as X2.

### Defaults for points the protocol leaves open

1. **Removing scenes.** Scores stay as fitted; k and the all-cheap loss are recomputed on the frames kept (k = round(0.2 n);
   for R3 k = floor(share × n), share = 0.2 − C_S / C_f).
2. **Part A subsets.** Scenes ranked by ΣV of the system and population at hand. "Minus the two logs" removes every scene of
   the two logs in the population. "Day" and "night" use `scene_split.csv`'s `night`. An empty subset gives no row.
3. **Part A, other rows.** E5's ten replacement pairs and two fine-tuned pairs at their primary operating point, monocular,
   228 test scenes (R5's tables); E7's primary (VoxelNeXt over PointPillar, detected, shared 0.25, the 150 official
   validation scenes); E10's primary (640, ego carry, monocular, 228 test scenes). Each must reproduce its stored all-full
   (asserted). Most influential scene: the largest absolute change of all-full when dropped.
4. **Part C.** "Random at the same rate": α of the frames (k = round(0.2 n); R3: floor(0.2 n)). "Matched count": the row's own
   escalated count. Both are random's expectation. The escalated counts are reported.
5. **Part D.** A scene's share uses each frame's inclusion probability under the benchmark's tie rule. Shares are of the row's
   total, top scenes by contribution; the sum of negative scenes is reported beside them. The test scenes of the two logs
   are computed from the split; if they are exactly `scene-1051` and `scene-1081`, the last two sets coincide.
6. **Part F.** Log order: `default_rng(seed).permutation` of the sorted logs, fold = position mod 5, seeds 0–19; the 765 scenes
   of the primary cell. Scores of the five folds are pooled; k = round(0.2 n). Q and G: R1's labels as they are. Added so
   that R3's claim has repeats: `gate_gbm` on R3's chosen subset, at R3's escalated share, over random at floor(0.2 n).
7. **Part G.** c = 95th percentile of |V| over affected fit frames of the primary cell, 537 scenes. Capped rows use capped V
   for gains and random's expectation; shares stay of the uncapped all-cheap loss. Selection is unchanged in every row but
   the secondary one trained on capped V.
8. **Part H.** Monocular braking only. J(w) = shortfall + jerk + (b / 0.12) · excess + (c / 6) · collision, from `Jterm_*`
   (the sum at the defaults is asserted equal to J). "Jointly at the extremes": b ∈ {0, 0.48} × c ∈ {0, 12}. The crossing of
   ρ = 1 is found by bisection on b in [0, 0.48] with c = 6, if ρ − 1 changes sign at the ends.
9. **Reading.** The claim's most influential scene: the one whose removal moves its value furthest in the direction against
   the claim, among the 228 test scenes. Decision on the scene-level interval (log-level beside it).
   - **robust:** after that removal the interval stays on the claim's side of zero, and at least 18 of 20 repeats of Part F
     have the claim's sign;
   - **tail-dependent:** not robust, the point after removal keeps its sign and at least 11 of 20 repeats do;
   - **fragile:** otherwise.
   - X2's primary is `gate_gbm` over random at the same rate; R3's primary likewise, with its count.
   - "All-full is not resolved on the 850 scenes" has no repeats (no fit): **robust** if the interval still contains zero after
     removing the most negative scene and, separately, the most positive scene; **fragile** if either removal resolves it.

### Outputs

`results/followup/R7/`: `partA_settings.csv`, `partA_scenes.csv`, `partA_other_rows.csv`, `gate_r7_0.json`,
`scores_<signal>.csv.gz`, `partC_D_G_rows.csv`, `partD_drop_one.csv`, `partD_concentration.csv`, `partG_trained_capped.csv`,
`partE_fit_sets.csv`, `partF_oof.csv`, `partF_summary.csv`, `partH_weights.csv`, `partH_rho_crossing.csv`, `reading.csv`; doc
`results/followup/R7/`.

---

## Addendum R8 — does the braking reference need a moving-lead rule?

Follow-up round 3, item R8. Committed before the first new value of R8.

### Post hoc status

R8 is post hoc. An audit of stored tables on 2026-11-06, before this addendum, found that `rap.planner.required_decel` treats a
moving lead as stationary, that 87% of 842 frames with a⋆ ≥ 3.5 and v ≥ 3 m/s (of those with 2 s of future) show no logged
braking, that frames with a⋆ ≥ 7 carry 39.5% of Σ|V|, and that `scene-1051` frame 32 is such a case. R8 was designed knowing
this. The doc recomputes every number.

### Item, as protocolled

#### R8 — does the braking reference need a moving-lead rule?

- **Question:** do the braking findings hold when the requirement stops treating moving objects as stationary?
- **Part A, face validity (all 850, no fit).** Per a⋆ band ([1, 3.5), [3.5, 7), [7, 9]), for v ≥ 3 m/s: frames, scenes,
  logged mean acceleration over the next 2 s (shares ≤ −1 and ≤ −2 m/s²), and for the constraining reference object:
  speed, share moving above 2 m/s, range, time headway. Same for collision-flag frames. Frames without 2 s of future are
  reported apart.
- **Part B, variant requirements.** With gap as in the code and v_obj = max(0, v_ego + range_rate) along the ego heading:
  - `lead_b4` (primary): a = v_ego² / (2 (gap + v_obj² / (2·4.0))), the lead may brake at 4 m/s²;
  - `lead_cv` (secondary): a = max(0, v_ego² − v_obj²) / (2 gap), the lead keeps its speed.

  Both reduce to the current rule for a stationary object; keep the closing-speed term and the 9 m/s² cap. Each is applied
  two ways: `ref+ctrl` (primary; the controller uses the same rule on its perceived objects, with closing speed from the
  stored ttc, and a perceived object with no finite ttc counted as stationary, and a⋆ from the reference) and `ref-only`
  (secondary; the controller as deployed, only a⋆ changes).
- **Gate R8-0:** with the variant off, the rebuilt tables equal `data/cache/xl/tables850` and the frozen tables bit for bit.
- **Part C.** On each variant: Table-1 rows for the four populations, both geometries, with intervals; Part A's table;
  harm direction (mass where full under-brakes against over-brakes), which on the standard reference is 90% over-braking
  on the frozen 85 but 71% under-braking on the 228 test scenes; R7 Part A's drop-scene rows for pooled 850.
- **Part D.** On the primary variant, refit and report: X2's `gate_gbm` row, R1's four D_t, R3's chosen subset (no
  reselection). Report the influence rows of R7 Part D for them.
- **Registered reading.** Sign variation "persists" if harmed ≥ 30% with ρ > 0 in the primary cell on the 228 test scenes
  and the 850; each allocator claim "carries" if its interval stays above zero on the primary variant. The all-full and
  oracle@20 values are reported with no reading attached.
- **Outputs:** `results/followup/R8/`, doc `results/followup/R8/`. Frozen values are not replaced; this
  is a sensitivity of the braking system.

### Implementation

- **Script:** `scripts/235_r8_reference.py`, stages `gate`, `tables`, `parta`, `parts`, `partd`, on the Jetson.
- **Library:** `rap.planner.required_decel(..., lead=None, range_rate=None)` with `lead_speed`, and `rap.decision.build(...,
  lead=None, lead_ctrl=False)`. With `lead=None` the code path is the benchmark's; gate R8-0 and the unit tests check it.
- **Gate R8-0:** with the variant off, `decision.build` on the 850 scenes (both geometries) equals every shared column of
  `data/cache/xl/tables850` and, on the frozen 85 scenes with the frozen detection cache, of `data/cache/xl/frozen85_tables`,
  by `DataFrame.equals`. A failure stops R8 only.
- **Part D:** `gate_gbm` on V and on R3's chosen subset refit on the primary variant's V, 537 scenes; the perception-target gates
  keep R7's saved scores (their targets do not depend on the rule) and are scored by the variant's V. R3's cost, count and subset
  are unchanged (no reselection).

### Defaults for points the protocol leaves open

1. **Reference range rate:** the track's fitted range rate (`range_rate`). A reference object without an estimate (`has_rate`
   false, a track seen on one keyframe) counts as stationary.
2. **Perceived objects:** closing speed = range / ttc, v_obj = max(0, v_ego − closing). `rap.mono` caps ttc at 1,000 s for a box
   that does not expand; that cap means "no finite ttc", so such an object counts as stationary, as the protocol says.
3. **Geometry of the perceived object under oracle:** the oracle's replaced ttc enters the same way.
4. **Logged acceleration (Part A):** (v_ego at the fourth next keyframe − v_ego) / elapsed time, about 2 s, from the tables' `v_ego`
   and E10's frame times; frames without four later keyframes are the "no 2 s of future" count.
5. **Constraining object:** the reference object with the largest per-object requirement under the rule at hand; a frame with no
   requirement has none. Time headway = range / max(v_ego, 0.1). Moving = its speed above 2 m/s along the ego heading.
6. **Collision-flag block:** frames with a flag in either branch, under the standard rule, all speeds.
7. **Harm direction:** among frames with V < 0, the mass Σ|V| with `act_full` below `act_cheap` (full under-brakes) and above it
   (full over-brakes). The remainder has equal actions (a jerk-only difference).
8. **Reading's "both populations":** `new228_test` and `pooled850`, monocular. "Carries": the scene-level interval's lower bound
   is above zero (log-level beside it); the claims are X2's `gate_gbm` over random, R1's four D_t, R3's chosen `gate_gbm` subset
   over random at the same rate.
9. **Drop-scene rows:** R7's subsets on pooled 850, monocular, ranked on the variant's own ΣV.

### Outputs

`results/followup/R8/`: `gate_r8_0.json`, `partA_face_validity.csv`, `partC_settings.csv`, `partC_harm_direction.csv`,
`partC_drop_scene.csv`, `partC_drop_scene_names.csv`, `scores_primary_variant.csv.gz`, `partD_allocators.csv`, `reading.csv`; doc
`results/followup/R8/`.

---

## Addendum R9 — baselines trained directly on the budgeted objective, and an information ladder

Follow-up round 3, item R9. Committed before the first new value of R9.

### Item, as protocolled

#### R9 — baselines trained directly on the budgeted objective, and an information ladder

- **Question 1:** does training directly on the budgeted decision objective beat regression on V followed by top-k?
- **Question 2:** is the gap to the oracle mainly a limit of the architecture or of the pre-escalation information?
- **Objective.** max over z ∈ {0,1}ⁿ with Σz ≤ k of Σ z_i V_i. On the XL braking cells the budget does not bind at 20%
  (379 positive test frames against k = 1,836), and there the objective equals cost-sensitive classification: maximizing
  Σ z_i V_i is minimizing the |V_i|-weighted error of predicting sign(V_i). With a binding budget, its Lagrangian does the
  same with V_i − λ.
- **Learners, on X2's 65 gate features, same units:**
  - `cs_gbm`: `HistGradientBoostingClassifier` with the gate's hyperparameters, y = 1[V > λ], sample_weight = |V − λ| + ε;
    λ from {0 and the 50/75/90/95th percentiles of positive V}, ε from {0, 0.01, 0.1} × median |V| of affected frames.
    Score = predicted log-odds.
  - `topk_mlp`: the routers' MLP (two layers, 64 units) on standardized features, maximizing a relaxed gain
    Σ σ((s_i − θ)/T) V_i with θ the batch (1−α) quantile; whole-scene batches of at least 2,048 frames; T from {0.1, 0.3,
    1}; Adam; five seeds, scores averaged.
  - `pair_mlp`: the same MLP with a gain-weighted pairwise logistic loss on sampled pairs (i, j) with V_i ≠ V_j, weight
    |V_i − V_j|; five seeds, scores averaged.
- **Selection.** All settings, early stopping, and the choice among the three learners per cell and budget: on the 152
  validation scenes with models fit on the 385 training scenes, by realized gain at that budget. Then refit on 537 scenes
  and score once on the 228. Nothing is tuned on test. Also report the same on capped V (R7 Part G).
- **Primary endpoints (Holm over the two):** Δ = realized gain(validation-chosen direct learner) − realized gain(`gate_gbm`
  on V) as a share of the all-cheap loss, at 20%, on (a) the primary cell and (b) planner ADE, monocular, which is free of
  the braking tail. Reading: "direct better", "regression better" or "not resolved". All three are informative; none stops
  anything.
- **Secondary.** All six XL cells; budgets 5–50%; R7's influence rows; R8's primary variant; realized gain against all-cheap
  and matched-count random; with R3's criticality subset under the measured 20% latency budget (`151_profile_allocator.py`
  for the cost); the frozen 10 core cells + 2 PDM-Closed cells **through `evaluate_submission.py` in the per-budget form**,
  fit on the frozen train + validation units, to show that the public submission path takes a direct-objective allocator
  unchanged; causal policies A and B of `docs/iclr_causal_threshold.md` for the chosen direct learner on XL.
- **Information ladder (diagnostic rungs are labelled not deployable).** Same two learners, `gate_gbm` on V and the chosen
  direct learner, primary cell, 20%:
  - L0: X2's 65 cheap-side features;
  - L1: L0 + three previous frames and ego dynamics (X6's temporal features);
  - L2: L1 + the same 65-feature extractor applied to the **full** output of the same frame, and the differences to L0
    (diagnostic: needs escalation);
  - L3: L2 + reference-derived scene counts: reference objects in corridor, nearest in-corridor range, its speed and
    closing speed (diagnostic: needs the reference).

  Respect rule 8: no action, requirement or threshold-distance column at any rung. Report realized gain, nDG, and Spearman
  on affected frames; paired Δ(L1−L0), Δ(L2−L0), Δ(L3−L0) with intervals. Reading: the gap is **mainly information** if
  Δ(L2−L0) is resolved and exceeds half of the gap between L0 and the oracle; **mainly architecture or noise** if L3 itself
  stays below half of the oracle; otherwise **mixed**.
- **Outputs:** `results/followup/R9/`, doc `results/followup/R9/`.

### Implementation

`scripts/236_r9_direct.py` (stages `select`, `test`, `crit`, `frozen`, `causal`, `ladder`; run in that order by), `scripts/236b_r9_allocator.py` (the file `151_profile_allocator.py` times). Inputs: X2's cells through
`227_r1_target_swap.setup`, R8's table `lead_b4` `ref+ctrl` monocular, `results/followup/R3/`, the frozen cells through
`124_causal_threshold.core_cells` and `nuplan_cells`. Scratch: `data/cache/xl/r9/`. Seeds: 0–4 for the MLPs, `random_state=0`
for the gradient-boosted models, 0 for the bootstrap. No detector is trained or re-weighted. Every feature list is checked
against rule 8 (`decision_margin`, `boundary_distance`, `a_cheap`, `a_full`, `a_gt`, `act_`) before any fit.

### Defaults recorded before any R9 value

1. **`cs_gbm`.** The 15 settings λ × ε; percentiles of positive V and the median |V| of affected frames (|V| > the benchmark's EPS)
   on the frames being fit. A setting with one class in y is skipped and leaves no row. Hyperparameters taken from
   `predict.make_model("gbm", "reg", 0)`. Score = `decision_function`.
2. **MLPs.** Two 64-unit ReLU layers, one linear output, as the routers. Features z-scored on the fit frames and clipped at ±10
   (the gate's ±1e6 infinities would otherwise dominate); V divided by the mean |V| of affected fit frames (scale only). Adam, lr
   1e-3, weight decay 1e-4 (the routers' alpha). 200 epochs at most, validation checked every 10 epochs on the seed-averaged
   scores; epoch is one of the selected settings. Refits run the chosen number of epochs on the 537 scenes.
3. **Batches.** Scenes shuffled per epoch and seed, cut when a batch reaches 2,048 frames; a short remainder joins the last batch.
4. **`topk_mlp`.** θ is the detached batch (1 − α) quantile, trained per budget, so T × 6 budgets settings. Loss = −mean(σ((s − θ)/T) V).
5. **`pair_mlp`.** 8,192 pairs drawn uniformly inside each batch, pairs with V_i = V_j dropped, loss = mean of |V_i − V_j| ·
   softplus(−sign(V_i − V_j)(s_i − s_j)). One training run serves all budgets; the epoch is chosen per budget.
6. **Selection.** Largest validation realized gain at k = round(α n), per cell and budget; ties to the first in the order cs_gbm
   (λ then ε), topk_mlp (T), pair_mlp, earliest epoch. All 15 + 18 + 1 settings on each of the six XL cells.
7. **Primary.** Δ on the 228 test scenes at 20%, gate_gbm refit as X2 (537 scenes). Holm over the two at 0.05, on two-sided
   bootstrap p-values as R1 (`2 min(P(x ≤ 0), P(x ≥ 0))`). "Direct better" or "regression better" by the sign of a rejected Δ,
   otherwise "not resolved". Scene level decides; the log-level interval stands beside it.
8. **Capped V.** At 20% on the six cells. Selection on V clipped at c (c = R7's rule, on the cell's own fit frames: training frames
   for selection, 537 for the refit); compared with gate_gbm trained on the same capped V. Reported on capped V and on V; the
   V-trained pair is also reported on capped V. Shares stay of the uncapped all-cheap loss.
9. **R7's influence rows.** On the two primaries at 20%: without `scene-1051`, without `scene-1081`, without both, without the two logs'
   test scenes, without the scene that moves Δ most, and the drop-one range. k is recomputed after removal, as in R7.
10. **R8's primary variant.** The full selection (all settings, six budgets) repeated on the monocular braking cell with
    `lead_b4` `ref+ctrl` V; gate_gbm refit on that V.
11. **Criticality subset.** R3's gate_gbm subset. Selection on validation at R3's escalated share (from C_S = R3's gate_gbm cost, k =
    floor), all settings. Costs: `151_profile_allocator.py`, nuScenes track, 400 timed inputs, on R3's 200 fixed frames (the
    harness's export carries no cached geometry, so its input only sets the pace), R3's extractor, and the model as plain numpy
    (no per-call scikit-learn overhead), for both the chosen direct model and gate_gbm. Engines from `feature_costs.csv`. Both rows
    reported: each charged its own 151 cost, and gate_gbm at R3's batched C_S beside it. The budget is applied as in R3 (k =
    floor(frac n), random at floor(0.2 n)).
12. **Frozen cells.** All settings at the four submission budgets, fit on train, chosen on validation among the three learners per
    cell and budget; refit on train + validation; one CSV per track (their id columns differ) through `evaluate_submission.py`
    in the per-budget form, plans as its defaults; gate_gbm's scores through the same path in the same form, as the comparator. Nothing
    is written under `results/final/`; its hashes are compared before and after.
13. **Causal.** Variant V1 of `docs/iclr_causal_threshold.md`: the setting chosen at 20% refit on the 385 training scenes, threshold
    from the validation scores, applied once to the test stream in (scene, frame) order; B's cap per scene; gate_gbm beside it under the
    same policies. The six XL cells, 20%. Ties broken with `default_rng(0)` (none expected).
14. **Ladder.** "Same two learners": gate_gbm and the primary cell's validation-chosen direct setting at 20%, not re-selected at any rung.
    L1: the 65 gate features at t−3..t (zeros where the scene has no such frame, `183_xl_temporal.history_matrix`) and `ego_acc`, `yaw_rate`,
    so 262 inputs. L2: L1 + the same extractor on the 384×640 detections (with the cheap cache's image statistics, which the full
    cache does not store) + the differences, so 392. L3: L2 + four columns: in-corridor reference objects (corridor and range as in the
    requirement, objects ≥ the benchmark's height), nearest in-corridor range (max range if none), its speed (`rap.planner.lead_speed`; no
    rate = stationary) and closing speed (no rate = v_ego; 0 if none), so 396. No requirement, action or threshold-distance column.
15. **Ladder reading.** Applied to each of the two learners, gate_gbm's being the headline. "Half of the gap" = 0.5 (oracle − L0); "half of
    the oracle" = 0.5 × oracle gain at k; both from the test scenes. Information is tested first, then architecture, as worded in the item.
16. **R7's scores.** gate_gbm is refit here rather than read from R7, so this item does not depend on R7's gate; the doc checks it against
    X2's stored row.

---

## Addendum R10 — what the additive, fixed-history evaluation misses, and a decision-mask scorer

Follow-up round 3, item R10. Committed before the first new value of R10.

### Post hoc note

The audit quoted in addendum R7 in `docs/followup_protocol.md` saw that E10's refresh allocators score with one-frame-old lists; that point belongs to R11.
R10 itself was protocolled without a stored value of what it measures.

### Item, as protocolled

#### R10 — what the additive, fixed-history evaluation misses, and a decision-mask scorer

- **Part A, replay of a mask.** The braking action at t depends only on the branch at t, and the loss on a_t and a_{t−1}, so
  a mask z can be replayed exactly with each policy's own history. For oracle@20, `gate_gbm`, random (20 seeds) and all-full,
  on XL braking (mono, oracle) and KITTI braking: replayed gain against the additive Σ z_i V_i, absolute and relative
  difference, per population.
- **Part B, sequence oracle.** By dynamic programming per scene over (t, previous action), with a Lagrange multiplier on
  the number of escalations shared over the scenes, solve the budgeted replayed objective exactly. Report its value
  against the additive top-k oracle at 10, 20, 30, 50%.
- **Part C, decision-mask form.** Add a third submission form to `rap.submission` / `evaluate_submission.py`: one 0/1 column
  `z` per input (or per budget level), scored by Σ z_i V_i, valid if Σ z ≤ k; `--causal` also checks the per-unit cap
  floor(1 + αt) and reports violations. This lets a causal, abstaining or non-ranking policy be scored as itself rather than
  as a whole-split ranking. **Gate R10-0:** G1' reproduces unchanged (`150 --stage g1`); default forms are byte-identical.
  Score on it, on XL and the frozen cells: the six learned signals under policies A and B, and R9's chosen learner.
- **Note for the doc.** This bounds the error of the additive view **within the controller's action memory only**. It says
  nothing about ego-state feedback, which remains the subject of E3 and R2.
- **Outputs:** `results/followup/R10/`, doc `results/followup/R10/`.

### Implementation

`scripts/237_r10_sequence.py` (stages `gate`, `replay`, `oracle`, `masks`; run by). Library: `rap.submission` gains the forms
`mask` (`z`) and `per-budget mask` (`z_q10`..`z_q50`), `mask_stats`, `cap_violations`, `score_mask_cell`; `score(..., causal=False)`;
`evaluate_submission.py --causal`; `docs/SUBMITTING.md` one table row; `tests/test_submission_mask.py`. Inputs: XL tables, the frozen KITTI tables (`t92.CORE`),
R7's saved gate_gbm scores for XL, the official KITTI gate_gbm scores (`124.core_cells`), `results/followup/R9/{selected.json,frozen_selected.csv}`. Seeds: 0 for bootstrap,
0–19 for tie orders and random.

### Defaults recorded before any R10 value

1. **Replay.** J(b | p) per frame from `decision_cost`'s formula with the stored `a_gt`, `act_cheap`, `act_full`; asserted equal to `J_cheap` (b = p = cheap) and `J_full` (b = full, p = cheap),
   tolerance 1e-9. First frame of a unit: no jerk. All-cheap is the baseline of every gain.
2. **Policies and populations.** XL braking (mono, oracle): the 850 and the 228; KITTI braking (mono, oracle): all 21 and the test sequences. gate_gbm only on test populations (its scores outside them are fitted
   frames). Oracle at 20% = top k by V, ties (V = 0 frames) broken by 20 seeded orders, mean reported; random = 20 seeds; all-full escalates every frame. k = round(0.2 n).
3. **Interval on replayed − additive.** The mask is held fixed and per-unit differences are resampled (1,000 draws, seed 0): this is the interval of the difference for that mask.
4. **Sequence oracle.** The protocol's Lagrange form is run and reported with its multiplier; because per-unit curves F(m) (least loss with exactly m escalations) are also available from the same dynamic
   program, the exact budgeted value is computed by a max-plus knapsack over units, with at most k escalations, and is the headline; the duality gap between the two is reported.
5. **Its interval.** Per draw: the Lagrange dual on the resampled units against the additive top-k oracle of the same draw, k recomputed. Scene level and, on XL, log level. KITTI's unit is the sequence.
6. **Mask semantics.** `z` is scored at every quota; `z_q..` each at its own. Within budget if Σ z ≤ k = round(α n). A mask over budget keeps its gain and has no nDG (`undefined_reason` says so).
   The interval's filter (draws with the oracle's prize above 25% of the point prize) is the scorer's. A mask is fixed across draws; only units are resampled.
7. **Stream.** By unit, in time: (seq, frame) for KITTI and nuScenes, (t0, iteration) for nuPlan, as 124. The cap is checked per unit with α = the quota.
8. **Policies under the form.** V1 of `docs/iclr_causal_threshold.md`: models fit on the training units, threshold on validation, at each of 10, 20, 30, 50%. Policy B's cap as 124. R9's chosen learner: the setting chosen for that cell and budget, refit on the
   training units. XL: six cells, 228 test scenes, through `mask_stats` (XL is not a benchmark cell), scene level with log level beside it. Frozen: 12 cells through `S.score`, and the R9 chosen B masks also through the CLI with `--causal`.
   Policy A and B can exceed k: that is a result, and is reported, not repaired.
9. **Gate R10-0.** 150 `--stage g1` with its one write under `results/final/` redirected to a temporary folder (no file there is touched), byte-compared with the stored `submission_path_g1.csv`; the shipped examples' scored files re-scored and byte-compared.

---

## Addendum R11 — refresh scheduling with real staleness

Follow-up round 3, item R11 (conditional). Committed before the first new value of R11, and run only because R7–R10 were committed before the protocol's limit.

### Item, as protocolled

#### R11 — refresh scheduling with real staleness (conditional, see Order)

- **Question:** can DEEP evaluate a policy that decides before any current-frame perception runs, whose action changes
  future states, against an exact oracle?
- **Cell.** KITTI (10 Hz, 21 sequences, primary) and nuScenes XL (2 Hz, secondary), braking, 640 detector. At each frame the
  policy either runs the detector or reuses the last detected list, carried to the current frame with ego motion (E10's
  `ego`). Maximum age 10 frames, then a forced refresh, charged to the budget. Loss from the benchmark's controller.
- **Tables.** Per unit, the cost table L(t, s) for last refresh s ∈ [t−10, t]. **Gate R11-0:** s = t equals E10's fresh
  branch and s = t−1 its stale branch, exactly.
- **Oracle.** Dynamic programming over (t, s) with a shared Lagrange multiplier, refresh shares 10, 20, 30, 50%.
- **Policies, scored by replay with their true staleness:** periodic; random; threshold on age and ego speed; E10's
  `gate_gbm` with age added; and one sequential learner, a GBM that imitates the oracle's decisions on the state (features of
  the carried list, age, ego speed, budget left), fit on training units, chosen on validation.
- **Primary endpoint:** gain(sequential learner) − gain(periodic), KITTI, 20%. Reading as elsewhere.
- **Outputs:** `results/followup/R11/`, doc `results/followup/R11/`. Report it as a new cell type, separate from
  E10b, whose refresh scores do not carry staleness.

### Implementation

`scripts/238_r11_refresh.py` (stages `tables`, `oracle`, `policies`;). Reuses `206_temporal_escalation.py`'s `carry`, poses and
range source. Scratch: `data/cache/xl/r11/`. Reports a new cell type; E10b's rows are unchanged.

### Defaults recorded before any R11 value

1. **Cost table.** For age a = t − s in 0..10: the 640 list of frame s moved once from ego(s) to ego(t) with `carry` (closing speed kept), through the controller at t; requirement a⋆ at t. The stored form is the action per (t, a); the loss
   adds the jerk term from the policy's own previous action. Geometry: monocular is the primary; the oracle tables are built for the gate only.
2. **Gate R11-0.** Action and requirement at a = 0 equal E10's fresh branch and at a = 1 its stale branch, exact float equality, with a⋆ too, on `KITTI` and `nuScenes` (850 scenes), both geometries.
3. **Budget.** A unit's first frame always runs the detector and is charged, as is every forced refresh. k = round(share × frames) over the population. Age may reach 10; at what would be 11 a refresh is forced.
4. **Baseline of every gain.** The minimal schedule: one refresh every 11 frames, the forced ones only. Shares are of its loss. All-stale is not available under the age limit.
5. **Oracle.** Dynamic program over (t, age), a transition carrying the jerk term, with one multiplier per population and budget, found by bisection. The feasible value at count ≤ k and the dual bound are both reported, with the gap. If the forced refreshes alone exceed k, the row is infeasible
   and says so.
6. **Policies.** All causal, all forced at age 10. Periodic: every round(1/share) frames. Random: each frame with probability p, p found on refresh counts only (no outcome), 20 seeds. Age × speed: refresh when age × causal ego speed exceeds τ. `gate_gbm` with age: the gate on target
   base loss(t, a) − base loss(t, 0) for every (t, a ≥ 1) of the training units, features of the list's frame (65, as E10 used the stale list's) plus age and causal speed; refresh when the prediction exceeds τ. The jerk term is left out of this target because it depends on the path.
7. **Sequential learner.** `HistGradientBoostingClassifier` with the gate's hyperparameters, one per budget, on the states the oracle visits on the training units: the list's 65 features, age, causal speed, budget left; label = the oracle refreshed. Forced frames are not decisions. No aggregation of
   states from the learner's own runs. Fit on the training units.
8. **Budget left.** floor(share × (t + 1)) minus refreshes used in this unit, clipped to [−3, 5] (integers, so scores can be tabled).
9. **Thresholds and cap.** τ for the three scored policies: on validation, from 41 score quantiles, largest replayed gain with count ≤ k, falling back to the smallest count. A non-forced refresh also needs used < floor(1 + share × (t + 1)). The count on test is reported with `within budget`; it is not repaired.
10. **Units.** KITTI: 11 train, 4 validation, 6 test sequences, as the benchmark's split; nuScenes XL: 385, 152, 228, log level as the group for intervals. Six sequences give a coarse interval; the doc says so.
11. **Primary.** Sequential learner − periodic, KITTI, 20%, test sequences, sequence bootstrap, 1,000 draws, seed 0: "sequential better", "sequential worse" or "not resolved". One endpoint, no Holm. Other rows are secondary.
12. **Rule 8.** No action, requirement or threshold-distance column enters a feature; the feature list is checked before any fit. `a_stale`-style columns are not read by any policy.

---

## Addendum R12 — gaps left by R7, R8 and R10

Round 3, add-on item R12. Committed before the first new value of R12.

### Post hoc status

R12 is post hoc: it was written after reading R7, R8, R9 and the stored part of R10. It is a sensitivity. No frozen value is replaced, and no existing doc or output is edited.
If R12 changes how an earlier sentence should be read, `results/followup/R12/` says so.

### Why it exists

1. R7 Part H varied the loss weights on the standard reference, and R8 varied the reference at the default weights. Neither ran a loss without over-braking on the moving-lead reference.
2. R7 Part H's harmed share at over-braking weight 0 still holds the jerk term (`lam_jerk = 0.02`), so a frame can stay harmed through jerk alone.
3. R8 covered nuScenes only. KITTI uses the same stop-behind rule.
4. R7 Part B saved the routers' scores, but their rows are in neither `results/followup/R7/` nor `partC_D_G_rows.csv`. A check of the docs found them nowhere else, so Part C runs.

### Items

**Part A, reference × loss terms (no fit, exact recombination of `Jterm_*`).** Tables: standard, `lead_b4 ref+ctrl`, `lead_b4 ref-only`; monocular (primary) and oracle geometry; populations pooled 850, 228 test, frozen 85.
Losses: over-braking weight {0, 0.06, 0.12, 0.24, 0.48} at collision weight 6; collision weight {0, 12} at over-braking weight 0.12; no-jerk (default weights, jerk removed); safety-only (shortfall and collision only, as R6).
Each row: affected, harmed, harmed share, ρ, gross benefit, gross harm, all-full, oracle@20, with intervals; the count and share of harmed and of helped frames whose shortfall, excess and collision terms are equal in the two branches.

**Gate R12-0.** On the standard tables, default weights and safety-only reproduce R7's and R6's stored rows exactly.

**Registered reading of Part A.** On `lead_b4 ref+ctrl`, monocular, safety-only, separately on the 228 and the 850:
**harm in safety terms persists** if the harmed share's interval lies at or above 30% and ρ > 0; **harm in safety terms shrinks** if the point is below 30% or the interval reaches below 20%; **unclear** otherwise.

**Part B, KITTI under the moving-lead reference (no perception rerun).** All 21 sequences and the test sequences, both geometries, `ref+ctrl` and `ref-only`: affected, harmed, ρ, all-full, oracle@20, harm direction, with sequence-level intervals.

**Gate R12-1.** With the variant off, the rebuilt KITTI tables equal the frozen ones bit for bit.

**Registered reading of Part B.** R8's "persists" rule (harmed ≥ 30% with ρ > 0), monocular `ref+ctrl`, on all 21 sequences and on the test sequences. No allocator rows.

**Part C, router rows from R7's saved scores (no fit).** `R1_gbm_reg` and `R1_mlp_reg`, raw form, on V, fn_gain, risk_gain, Q, G, at 20%, on the 228: realized gain, over random at matched count, D_t against the same router on V, with intervals, on all scenes and with the row's most influential scene removed. R7's tie rule.
R7's reading is applied only to the interval's side after removal, since no Part F repeats exist for routers.

**Part D, facts (no fit, no reading).** (1) Log overlap between the XL fits and the 228. (2) Frames an oracle really uses: count and share of V > 0.

### Implementation

`scripts/239_r12_gaps.py`, stages `a`, `b`, `c`, `d`, run in that order. Inputs: `data/cache/xl/tables850`, `data/cache/xl/r8` (`lead_b4` tables), R7's `scores.pkl`, the frozen KITTI tables (`t92.CORE`), the KITTI detection cache, `scene_meta.csv`.
Outputs in `results/followup/R12/`: `gate_r12_0.json`, `partA_rows.csv`, `reading_partA.csv`, `gate_r12_1.json`, `partB_rows.csv`, `partB_no_estimate.csv`, `reading_partB.csv`, `partC_rows.csv`, `partD_log_overlap.csv`, `partD_frames_used.csv`; doc `results/followup/R12/`. No library file changes.

### Defaults recorded before any R12 value

1. **Recombination.** J = shortfall + w_e × excess × (b / 0.12) + w_c × collision × (c / 6) + jerk, on stored weighted terms, as R7 Part H. No-jerk drops the jerk term; safety-only drops excess and jerk. The sum of the four terms is asserted equal to J on every table, tolerance 1e-9.
2. **"V ≠ 0".** |V| > `EPS` of `rap.scoring`, per row, as every earlier affected count, so that the gate can compare exactly.
3. **Bootstrap.** R7's `_a_task`: scene level and log level on the same seed; 1,000 draws, seed 0. A row's all-cheap loss is under that row's loss. Part B: KITTI's unit is the sequence, one interval.
4. **Jerk-only.** On a row's own loss: a frame is jerk-only if the sum of absolute differences between branches of its weighted shortfall, excess and collision terms is at most `EPS`. So at over-braking weight 0 the excess term counts as equal. Rows without a jerk term report 0 by construction. No intervals on these counts.
5. **Gate R12-0.** Standard rows only, computed before any variant row. Every shared column exact, CSV read with `round_trip`: mono against R7 `partH_weights.csv` for the grid's seven weight rows on the 850 and the 228; both geometries against R6 `controls.csv` (safety-only and standard, 228 and 850); frozen 85 default row against R8 `partC_settings.csv`. A failed gate stops Part A only.
6. **Reading.** Scene-level interval decides, log-level beside it. "Lies at or above 30%" = lower bound ≥ 0.30.
7. **KITTI range rate.** `kitti` geometry already holds a range rate per track: the local linear fit of `long_near` over time, `rap.geometry._fill_range_rate`, the function nuScenes uses, half-width 2 frames at 10 Hz (nuScenes: 1 at 2 Hz). So no new estimate is made. No estimate = `has_rate` false = stationary. The controller side of `ref+ctrl` uses the closing speed of `ttc`, as R8. `lead_cv` is not run.
8. **Gate R12-1.** `decision.build` with `lead=None` on the frozen table's sequences, each column shared with `KITTI__YOLOv8s__cheap_320tofull_640__<geometry>.pkl`, `.equals`, both geometries, before any variant table. A failed gate stops Part B and Part D's KITTI `lead_b4` rows only.
9. **KITTI splits.** `configs/benchmark_splits.json`, six test sequences; the doc says that the interval is coarse. Harm direction: R8's `harm_direction`.
10. **Part C.** Row quantity for the influence removal: over random at matched count on V, D_t on other targets, as R7's `ask`. Most influential scene: the scene whose removal moves that quantity most toward zero, among the 228, as R7 (`idxmin` for a positive point, `idxmax` for a negative). D_t = gain(router on V) − gain(router on target). Scene level with log level beside it. Side after removal: "same side", "reaches zero", or "not resolved before removal".
11. **Part D(1).** XL = 850 scenes; fits: 537 (train + validation) and 385 (train) of the group new765; test = 228. "Adjacent" = same log and scene numbers differ by exactly 1 (`scene-NNNN`), since the release's scene numbers follow time within a log.
12. **Part D(2).** V > `EPS`. Populations: pooled 850, new 765, 228 test, frozen 85, KITTI all 21, KITTI test; both geometries; standard and `lead_b4 ref+ctrl`. The k at 20% is reported beside.
13. **Rule 8.** No allocator is fit and no feature is built in R12.

---

## Addendum R13 — causal rules at a matched fit set, and the direct learner under log-grouped folds

Round 3, add-on item R13. Committed before the first new value of R13.

### Post hoc status

R13 is post hoc: it was written after reading R7, R9, R10 and R12. It is a sensitivity. No frozen value is replaced, and no existing doc or output is edited. If R13 changes how an earlier sentence should be read, `results/followup/R13/` says so.

### Why it exists

1. X2, R1, R3 and R7 fill the budget by ranking the whole test split, which no deployed allocator can do.
2. Fit sets are mixed: X2, R1 and R3 fit on 537 scenes; Tables R9i and R10c fit on 385 with the threshold on validation. No row gives ranking and a causal rule at one fit set.
3. R3's one-group gate has no row in `masks_xl.csv`, and D_t exists for ranking only.
4. `masks_xl.csv` has no interval on realized gain, only on gain over random.
5. R9 has one split; R12 Part D found all 228 test scenes have their log in the 537-scene fit. The direct learners have no repeats.
6. R3 gives gain over random at energy budgets, and no realized gain with an interval.

### Definitions

Cell: `mono_brake` on XL, standard tables. Signals, all `gate_gbm`: `V`; `fn_gain`, `risk_gain`, `Q`, `G` (all features, raw target, as R7); `crit_ms` (subset `crit`, target V, rate 0.0944490, R3's `frac_at` at 20% latency); `crit_mJ` (subset `crit`, target V, rate 0.1600907, 20% energy). The first five use rate 0.20.
Forms: `rank` (top-k with R7's tie rule and R7's k rule for that signal), `A` and `B` (`t124.calibrate`, `mask_A`, `apply_cap`, as `policy_masks` in `237_r10_sequence.py`, with q = the row's rate, B's cap per scene). Shares are of the all-cheap loss. "Over random" = matched-count random.

### Items

**Part A, the 228 test scenes.** F385: fit on the 385 training scenes, threshold on the 152 validation scenes, as R10. F537: the scored model fit on the 537 scenes; the threshold from out-of-fold scores of those 537, five log-grouped folds by R7 Part F's rule with seed 0 over the logs of the 537, each fold scored by a model on the other four. No test frame enters; the threshold's scores come from models on four fifths of the data.
Rows: every signal × fit × form: used of k, within budget, cap violations with units, realized share, over random, each with scene and log intervals; A − rank and B − rank at the same signal and fit, paired. D_t: each of the four targets at each fit and form, gain(`V`) − gain(target), paired; these rows are at matched policy, not matched count. Influence: `V`, `crit_ms` and each D_t under B at F537, with the claim's most influential scene removed, by R7's default 9.

**Gate R13-0,** to file precision. F537 `rank`: `V` equals R7's stored realized row; `crit_ms` gain 447.477 and `crit_mJ` gain 628.466 equal R3's `test.csv`; D_t equal R7's rows. F385 `rank`: `V` equals the "385 scenes" row of `partE_fit_sets.csv`. F385 A and B: `V` gains 174.660 and 223.745 equal `masks_xl.csv`.

**Part B, out of fold by log, causal threshold.** R7 Part F's outer folds, seeds 0–19, over the 765 scenes. Within each outer fold hold out one fifth of the fit logs, drawn with `default_rng(1000 + seed)` and that fold's index; fit on the rest; threshold on the held-out logs; score the outer fold; apply A and B to its stream, by scene. `rank`: pooled top-k of the same scores, paired; these fits are smaller than R7's, so `rank` need not equal Table R7g. Per seed, on the 765 and on the 765 minus the two logs: realized and over random for `V` and `crit_ms`, D_t for the four targets, under each form; used share under A and B. Summary as Table R7g.

**Part C, the direct learner under log-grouped folds (ranking).** The same folds. Settings of `results/followup/R9/selected.json`, variant V, not chosen again: `mono_brake` 20% `pair_mlp` epoch 20; `mono_brake` 5% `cs_gbm`; `mono_plan_ade` 20% `topk_mlp`, T 1.0, alpha 0.2, epoch 40. Selection is not nested: R9 chose these on validation scenes inside these folds, which favours the direct learner. Δ = gain(direct) − gain(`gate_gbm`) on the same held-out frames, pooled per seed, as a share of the all-cheap loss, on the 765 and the 765 minus the two logs. Summary: mean (min, max), repeats above zero. Time limit: if one complete seed extrapolates to more than 4 hours for a row, that row uses seeds 0–9, decided from elapsed time alone and recorded.

### Registered reading

**Six claims, under policy B:** realized gain of `V` above zero; realized gain of `crit_ms` above zero; D_t above zero for each of the four targets. Interval and removal from Part A at F537, repeats from Part B. **robust:** after removing the claim's most influential scene the scene-level interval stays above zero, and at least 18 of 20 repeats are above zero; **tail-dependent:** not robust, the point after removal stays above zero, and at least 11 of 20 repeats do; **fragile:** otherwise. The rule tests side, not size. Policy A gets no reading, because it can exceed k.

**Cost of the causal rule:** for `V` and `crit_ms`, at each fit, for each of A and B: **costs gain** if the interval of causal − rank lies below zero, **no resolved cost** otherwise.

**Part C, each row:** **direct ahead** if at least 18 of 20 repeats have Δ > 0, **regression ahead** if at most 2 of 20 do, **no consistent difference** otherwise. With 10 seeds: 9 of 10 and 1 of 10.

### Implementation

`scripts/240_r13_causal.py`, stages `a`, `b`, `c`, `reading`. Inputs: X2's cell through `227.setup`, R1's `targets.csv`, R3's `selected.json`, `test.csv`, R9's `selected.json` and saved matrices. No library change. Outputs in `results/followup/R13/`: `gate_r13_0.json`, `partA_rows.csv`, `partA_Dt.csv`, `partA_influence.csv`, `partA_thresholds.csv`, `partA_frames.csv.gz`, `partB_repeats.csv`, `partB_summary.csv`, `partC_repeats.csv`, `partC_summary.csv`, `partC_time.json`, `reading.csv`; doc `results/followup/R13/`.

### Defaults recorded before any R13 value

1. **Rates.** The exact `escalated_frac` of R3's `test.csv` (gate_gbm, `crit`, mono, 20%, ms and mJ), asserted equal to the protocol's seven digits. k rule: `k_round` for the first five signals, `k_floor` at the row's rate for `crit_ms` and `crit_mJ`, as R7's R3 rows and `test.csv` (867 and 1,469).
2. **Fits.** Workers keep OpenMP at one thread, as R9's refit of `gate_gbm`; the gate checks this reproduces the stored rows. A fit happens only in pool workers forked before any fit.
3. **Bootstrap.** R7's draws: sorted units, `default_rng(0)`, 1,000 draws, one set per level, shared by every quantity of a level. `rank` is re-ranked on each draw with k recomputed from the draw's frames, as R7. A causal mask is held fixed and its frames resampled, as R10. Shares divide by the point all-cheap loss of the set. Over random: draw's used / n × ΣV. A − rank and D_t are differences within a draw.
4. **Used and budget.** `rank` uses exactly k. Within budget = used ≤ k. Cap violations through `rap.submission.cap_violations` with the row's rate; not defined for `rank`.
5. **Stream.** By scene, in frame order: `np.lexsort((frame, seq))`, as R10. Tie randomization in `mask_A`: `default_rng(0)`, as `policy_masks`.
6. **Influence.** Masks are unchanged by removing a scene: the threshold is fixed and the cap is per scene. Value: realized share or D_t share with the all-cheap loss of the remaining frames. Most influential scene: the one whose removal gives the smallest value (the claims are all "above zero").
7. **F537's folds.** `sorted` logs of the 537, `default_rng(0).permutation`, fold = position mod 5.
8. **Part B's inner hold-out.** `rng = default_rng([1000 + seed, fold])`, `rng.permutation` of the sorted fit logs of that fold, first round(n / 5) logs held out. Signals `V`, the four targets, `crit_ms`. `crit_ms` model is target V on the `crit` columns. `crit_mJ` is not in Part B.
9. **Part B's populations.** Masks are made per outer fold; on the 765 minus the two logs their frames are dropped, `rank` is ranked again with k recomputed, as Part F. D_t under A and B: difference of gains at each signal's own mask. Over random for `rank`: k / n × ΣV. Repeats for the reading: the row on the 765 scenes; the other population stands beside it.
10. **Part C's data.** R9's saved matrices (`data__<cell>.npz`), rows in `227.setup`'s order, asserted. All-cheap loss from the same cell. The `gate_gbm` comparator on the same fit, on each fold, one thread. k = round(b n), recomputed on each population.
11. **Part C's time limit.** Seed 0 runs first. For a row, extrapolated wall time = (sum of the seconds of its fits in seed 0, with its comparator's) × 20 / 3. Over 4 hours: that row stops at seed 9. Recorded in `partC_time.json`.
12. **Reading.** Scene-level interval decides, log level beside it. "Above zero" = lower bound > 0; repeats above zero = value > 0. Cost: scene-level upper bound < 0.
13. **Rule 8.** `check_columns` on every feature list before any fit.
