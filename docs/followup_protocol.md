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
