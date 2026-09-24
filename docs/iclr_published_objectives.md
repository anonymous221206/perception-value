# The published routing objectives, on the routers this benchmark already has

Task 24. Phase A (definitions, code, mapping) is below and was committed with the pre-registration
("Task 24 pre-registration") before any label was computed. Phases B–E follow.

## Phase A — the two objectives, from the papers and from the official code

### Sources

| | paper | official code | commit used |
|---|---|---|---|
| Q | Qiu, Wang, Hu, Guérin, Lu, *Optimizing Edge Offloading Decisions for Object Detection*, SEC 2024 (arXiv 2410.18919) | `https://github.com/qiujiaming315/edgeml-object-detection` | `859f70240aa090359ba827b374b68ca7821b7d55` (2025-03-07) |
| G | Geng, Mohan, Ott, *Budget-Adaptive Routing: Skipping the Weak When the Strong Answers Anyway*, NAIC '26 (arXiv 2606.30919) | `https://github.com/ViGeng/bgt-ada` | `6669ab0089a04fbe6257ebdc2601de13ed0e5398` (2026-06-30) |

Both repositories are cloned into `third_party/` at those commits (`third_party/PROVENANCE.md`).

### Q — ORIC (Qiu et al.), transcribed

Section II, Eq. (1): *"mAPI_i = (1/S) Σ_{j=1..S} AP_ij, where AP_ij is the AP of class j in image i and S is the
number of classes within image i."* ORI is "the difference in mAPI values between the strong and weak detectors for
the image under consideration".

Section IV-A, Eq. (4) and (5), verbatim apart from typesetting:

> *mAPC_i = mAP({h_i, H_E}) = (1/E) Σ_{j=1..E} AP_{{i}∪E, j}*, where H_E = ∪_{1≤k≤|E|} h_k denotes the consolidated
> detection results for the image set E, E is the number of classes across image i and images in E, and
> AP_{{i}∪E, j} refers to the AP of class j evaluated on {h_i} ∪ H_E.
>
> *ORIC_i = (|E| + 1) · (mAPC_{i,s} − mAPC_{i,w}) = (|E| + 1) · ( mAP({h_{i,s}, H_{E,w}}) − mAP({h_{i,w}, H_{E,w}}) )*,
> where w and s indicate the type of detector (weak or strong). […] The term |E| + 1 is a scaling factor to normalize
> ORIC across representative image sets of different sizes. […] mAPC (and therefore ORIC) defaults to mAPI (ORI)
> when E = ∅. […] Eq. (5) also assumes that the detector used for images in E is the weak detector.

The context: *"we curate E by uniformly sampling without replacement images from the complete dataset"*; *"we settled
on a value of |E| = 1,000"* (Section IV-B).

Estimator (Section V): an MLP regression on the weak detector's top-25 box proposals (the input of this benchmark's
R1), trained on *"MORIC_i = cdf(ORIC_i)"* (Eq. 6) with the weighted loss *"L = Σ_i MORIC_i · (e(h_{i,w}) − MORIC_i)²"*
(Eq. 7).

**Official implementation** (`reward.py:compute_orie`, `lib/data.py:set_data`, `lib/metrics.py:box_correct`,
`ap_per_class`, `compute_ap`; `regression.py:main`):
* E: for image i, a random permutation of all other images, first |E| (default 1000; `np.random`, unseeded).
* mAP: detections of {i} ∪ E pooled per class; a detection is correct at IoU ≥ 0.5 with the class matching, greedy by
  IoU (`box_correct`, YOLOv5's `process_batch`); AP per class by 101-point interpolation (`compute_ap`, COCO style);
  mAP is the mean over the classes present in the ground truth of {i} ∪ E (a class with no prediction has AP 0).
* ORIE = (mean strong mAP − mean weak mAP) · (|E| + 1); if no image of {i} ∪ E has a ground-truth object the result is
  NaN and is set to 0 (`reward.py:86`). An image without objects contributes no targets; its detections still count.
* Detections: the YOLOv5 `val.py --save-txt --save-conf` outputs (`yolov5_scripts.md`), i.e. confidence ≥ 0.001,
  multi-label NMS.
* MORIC on a training fold: `(argsort(argsort(r)) + 1) / n` (ordinal ranks, exact ties broken by row order); on a
  validation fold the ECDF of the training values. The weighted loss of Eq. (7) is applied to the CNN/MLP only
  (`regression.py`, `--weight`).

**Paper against code**: consistent. The paper does not state the IoU threshold, the interpolation, the detection
threshold or the no-object case; the code fixes them as above. The code's context is unseeded.

### G — ΔAP, MORIC+, OffloadBin (Geng et al.), transcribed

Section 3.1, Eq. (4)–(6), verbatim apart from typesetting:

> *ΔAP(i_t) = AP_{swap t→s} − AP_{all-weak}*, i.e., the change in dataset-wide AP@0.5 obtained by replacing the local
> detections on i_t with the cloud detections while holding all other frames at their local outputs. […] detection AP
> is computed dataset-wide via a single global Precision-Recall (PR) curve.
>
> *MORIC+(i_t) = F⁺_ΔAP(ΔAP(i_t)) if ΔAP(i_t) > 0; 0 if ΔAP(i_t) = 0; F⁻_ΔAP(ΔAP(i_t)) − 1 if ΔAP(i_t) < 0*, where F⁺
> and F⁻ are the empirical CDFs of ΔAP over the frames where offloading strictly helps and strictly hurts.
>
> *OffloadBin(i_t) = 1[ΔAP(i_t) > 0]*, trained with focal loss.

Budget-adaptive routing (Section 3.2, Eq. 8): two estimators, *f_skip* (raw image only; MobileNetV2-Lite, 128 × 128,
0.15 GFLOPs, on OffloadBin) and *f_cond* (after the weak detector; XGBoost on MORIC), and an arbiter
*α(ρ) = argmax over the two placements of offline AP@0.5 on a held-out tuning split*, piecewise constant in the
budget ρ; on VOC ρ_frontier = 0.3 and ρ_ceiling = 0.8 (f_cond for ρ ≤ 0.2 and ρ ≥ 0.8, f_skip for 0.3 ≤ ρ ≤ 0.7).

**Official implementation** (`src/proxy_metrics.py:compute_dataset_wide_oric`, `src/metrics.py:_run_greedy_matching`,
`_precompute_detection_ious`, `classes_match`; `src/phases/prepare_derive.py`, `prepare_transforms.py`,
`prepare_split.py:427-429`; `config/pipeline.py`, `config/datasets.py`):
* one PR curve over all frames and classes: detections sorted by confidence, each matched greedily to the unmatched
  ground-truth box of the same class with the highest IoU, a true positive at IoU ≥ 0.5; all-point interpolation
  (`oric_allpoint`, the variant the paper's estimators use);
* ORIC_i = (AP_modified_i − AP_base) · N, computed separately within each split; N is the context size + 1;
* context: `ORIC_CONTEXT_SIZE = 1000` random other frames (`RandomState(42)`, one draw); 0 would mean all frames;
* MORIC+ with F± fitted on the training split; OffloadBin = 1[oric_allpoint > 0]; MORIC (the conditioned target) =
  rank / N;
* detections at confidence ≥ 0.3 (`config/datasets.py`); a split without ground truth yields no rewards; a frame
  without objects stays in the curve (its detections are false positives).

**Paper against code — one discrepancy.** The paper defines ΔAP over the whole dataset; the code's default samples a
context of 1,000 frames (the code's comment calls |E| = 1000 "paper-faithful" for the EdgeML baseline). The factor N
is a positive rescaling and changes neither OffloadBin nor MORIC/MORIC+. Neither is chosen silently: the paper's
dataset-wide definition is the primary label, and the code's 1,000-frame context is computed as a registered
sensitivity whose agreement with the primary is reported.

### Variants and how they map onto this benchmark

* **Budget as a model input**: neither paper has one; there is no gradient-boosted model that takes the budget as an
  input.
* **MORIC** is not a multi-model extension: in both papers it is a CDF normalisation of the reward. Neither paper
  defines a variant with more than two fidelities, so nothing maps onto KITTI's 320/384/512/640 ladder.
* **Geng's budget-adaptive router** is a budget-indexed choice between two estimators. It maps onto this benchmark's
  one-score-per-budget form: at each quota, the arbiter picks R1 (conditioned, after CHEAP) or R2 (pixels, before CHEAP),
  each trained on G's targets. Unlike the paper's VOC thresholds, α is re-tuned here by the paper's own rule (Eq. 8) on
  this benchmark's validation units. This is a methodological difference from the benchmark's single ranking and is
  reported as a variant of the same baselines.
* **nuPlan**: the detection-profile track has no scored 2D detections to compute AP from, so neither objective is
  defined there; the nuPlan cells are out of scope.

### Computable from cached data?

Yes, with stated deviations. The caches do **not** keep the class-score vector (the task text assumed they did): per box
they store the box, the confidence, the coarse class and three uncertainty summaries. Neither objective needs the
vector — both use one class label and one confidence per box. The deviations:
1. **Detection floor.** The caches keep detections at confidence ≥ 0.10 (single label, NMS 0.65, at most 100). Q's
   official outputs go down to 0.001 with multi-label NMS; Q is computed on the ≥ 0.10 lists. G's own threshold (0.3)
   is a subset of the cache and is applied exactly.
2. **Classes.** The benchmark's coarse classes (vehicle, person, cyclist) for detections and reference objects, instead
   of COCO/VOC classes: this is the label space of both datasets here.
3. **Ground truth.** The benchmark's reference objects (2D boxes at least 10 px tall, coarse classes), the ones every
   perception primitive of the benchmark uses; KITTI DontCare regions are not special-cased (neither official code has
   ignore regions).
4. **Seeding.** Q's context is drawn with `RandomState(42)` in frame order (the official code is unseeded).
5. **Losses.** The label is swapped; each architecture keeps its own loss (the task holds architecture and
   hyperparameters fixed): Q's weighted MSE (Eq. 7) and G's focal loss are not used. The R2 head keeps its BCE with
   positive weighting.

## Phase B — the labels and their checks

Labels: `results/raw/20260922_193146_published_objective_labels` (148 `--stage labels`, 2 h 52 min on the board), from
the detection dump `20260922_192602_published_objective_dump`. Share of fit-unit frames with a positive label: KITTI
Q 0.634, G 0.820; nuScenes braking frames Q 0.498, G 0.640; nuScenes planner frames Q 0.528, G 0.655.

**Checks** (`results/raw/20260922_223505_published_objective_checks/checks.csv`):

| check | passed | largest difference |
|---|---|---|
| targets only on fit units | 10 of 10 |  |
| G base AP: official vs independent | 6 of 6 | 1.2e-14 |
| G dAP: 50 swaps recomputed independently | 1 of 6 | 0.0082 |
| G dAP: 50 swaps, official tie order, AP_swap = base + dAP/N | 6 of 6 | 2.6e-14 |
| Q ORIC: 50 images recomputed independently | 6 of 6 | 0 |

The first swap check (as registered: a full recomputation of the swapped set's AP) failed on its first run and still
fails as written: at **equal confidence** the independent code let the swapped frame's FULL detections keep their
frame-order place, whereas the official merge (`proxy_metrics._merge_swapped_precomputed`, `cloud_entries[j][0] >=
conf`) puts them before the CHEAP detections of every other frame. With the official order the recomputation equals
base + ΔAP/N to ≤ 3e-14 in AP units on every set (second row). Of the 300 sampled frames, 122 get a different ΔAP under
the frame-order rule, every one of them with a cross-frame confidence tie; none changes sign, so OffloadBin is
unaffected, and the largest change is 0.008 in ΔAP × N units. The labels are the official code's output, unchanged.
What the check found is a property of the published definition: **ΔAP depends on how equal confidences are ordered**,
which neither the paper nor the code documents. The stop rule of the chain was narrowed to exclude only that row; the
row itself is kept and reported (recorded as "Task 24 Phase B").

"The mean of per-image AP reproduces the dataset mAP" (the task's aggregate check) is not an identity for either
objective, as registered: both are context rewards. It is replaced by the base-AP and swap checks above (G) and the
per-image recomputation (Q).

**Correlation with the perception-gain labels already in use** (Spearman, fit units; `results/final/published_objective_labels.csv`):

| cell | Q vs dE_exact | Q vs dE_E5_combined | Q vs dE_E6_risk_weighted | G vs dE_exact | G vs dE_E5_combined | G vs dE_E6_risk_weighted | Q vs G | G vs G context 1000 (code default) |
|---|---|---|---|---|---|---|---|---|
| KITTI mono brake J | +0.509 | +0.617 | +0.338 | +0.768 | +0.746 | +0.532 | +0.666 | +0.999 |
| KITTI mono traj JB | +0.509 | +0.617 | +0.338 | +0.768 | +0.746 | +0.532 | +0.666 | +0.999 |
| KITTI oracle brake J | +0.509 | +0.617 | +0.338 | +0.768 | +0.746 | +0.532 | +0.666 | +0.999 |
| KITTI oracle traj JB | +0.509 | +0.617 | +0.338 | +0.768 | +0.746 | +0.532 | +0.666 | +0.999 |
| nuScenes mono brake J | +0.444 | +0.565 | +0.382 | +0.726 | +0.642 | +0.625 | +0.599 | +0.997 |
| nuScenes mono plan_ade JC_ade | +0.463 | +0.574 | +0.402 | +0.726 | +0.649 | +0.626 | +0.613 | +0.998 |
| nuScenes mono plan_fde JC_fde | +0.463 | +0.574 | +0.402 | +0.726 | +0.649 | +0.626 | +0.613 | +0.998 |
| nuScenes oracle brake J | +0.444 | +0.565 | +0.382 | +0.726 | +0.642 | +0.625 | +0.599 | +0.997 |
| nuScenes oracle plan_ade JC_ade | +0.463 | +0.574 | +0.402 | +0.726 | +0.649 | +0.626 | +0.613 | +0.998 |
| nuScenes oracle plan_fde JC_fde | +0.463 | +0.574 | +0.402 | +0.726 | +0.649 | +0.626 | +0.613 | +0.998 |

The labels depend on detections and reference objects only, so the four KITTI cells share one frame set and one set of
labels, and the nuScenes cells share one per frame set (braking frames, planner frames). G agrees with the exact
perception gain far more than Q does (ρ ≈ 0.73–0.77 against 0.44–0.51); the code's default 1,000-frame context changes G
little (ρ ≥ 0.997).

## Phase C — the refits, scored by decision value

R1 (MLP reg/clf, GBM reg/clf) and R2 refit on each objective, architecture, hyperparameters, seeds and fit units as
shipped; only the label changes (`*_reg` on the objective's MORIC, `*_clf` and R2 on its binary form). Scoring is the
benchmark's: nDG on the frozen test split at 10/20/30/50%, exact tie expectation, 1,000-draw paired unit bootstrap
against random and against the same architecture trained on V (the shipped `routers_r1_ego` and `router_r2_ego`
scores, not refit). Sanity: the V-trained rows reproduce `benchmark_table_routers.csv` exactly on all 200 rows.
Every row, with the measured-budget feasibility and nDG, is in `results/final/published_objective_routers.csv`.

The V-trained intervals in this table come from this script's own bootstrap draws, so that each published row is paired
with its counterpart inside the same draws; they differ from `benchmark_table_routers.csv`'s in the third decimal, and
the number of V-trained wins against random is the same (36 of 200 rows in both).

**What this is.** The published *objectives* (their targets, CDF normalisations and binary forms) trained on *this
benchmark's* architectures, features, fit units and scorer. The one piece of a published *method* reimplemented is
Geng et al.'s budget-adaptive arbiter (Eq. 8). Neither paper's full pipeline — their detectors, datasets, input
features, estimators or losses (Q's weighted MSE, G's focal loss) — is reproduced.

**Wins against random** (paired 95% lower bound above zero), per architecture and training target, over the ten cells
and four quotas (40 rows each):

| architecture | V (shipped) | Q | G | G, budget-adaptive |
|---|---|---|---|---|
| R1_mlp_reg | 6 beat / 0 lose | 0 beat / 0 lose | 2 beat / 0 lose |  |
| R1_mlp_clf | 7 beat / 0 lose | 3 beat / 0 lose | 4 beat / 2 lose |  |
| R1_gbm_reg | 11 beat / 6 lose | 1 beat / 0 lose | 6 beat / 0 lose | 6 beat / 0 lose |
| R1_gbm_clf | 12 beat / 1 lose | 6 beat / 0 lose | 0 beat / 0 lose |  |
| R2_cnn_clf | 0 beat / 6 lose | 2 beat / 4 lose | 4 beat / 0 lose |  |

**Published objective minus the V-trained counterpart** (paired, same architecture), over the 40 rows each:

| architecture | objective | mean difference | published significantly ahead | V significantly ahead |
|---|---|---|---|---|
| R1_mlp_reg | Q | -0.104 | 0 | 5 |
| R1_mlp_reg | G | -0.031 | 1 | 0 |
| R1_mlp_clf | Q | -0.028 | 1 | 0 |
| R1_mlp_clf | G | -0.041 | 0 | 1 |
| R1_gbm_reg | Q | +0.016 | 10 | 3 |
| R1_gbm_reg | G | +0.062 | 11 | 1 |
| R1_gbm_clf | Q | +0.016 | 1 | 2 |
| R1_gbm_clf | G | -0.044 | 0 | 4 |
| R2_cnn_clf | Q | +0.065 | 5 | 0 |
| R2_cnn_clf | G | +0.110 | 4 | 0 |
| R1_gbm_reg <-> R2_cnn_clf | G (budget-adaptive) | +0.062 | 11 | 1 |

**The budget-adaptive variant** (Geng et al., Eq. 8): at every quota of every cell the arbiter, tuned on the
validation units, picks the conditioned estimator (R1_gbm_reg on G-MORIC; its validation AP@0.5 is higher than the
skipping estimator's in all 40 cases, `results/final/published_objective_arbiter.csv`), so the variant equals
R1_gbm_reg on G throughout. The budget selects the ranking; it is not a model input.

**Measured budgets.** A refit keeps its architecture's shape, so it keeps the shipped overhead and the same feasibility
(93's two-level cascade, the shipped R1/R2 overheads, 131's rule). Both R1 GBMs, and therefore the budget-adaptive
variant, are infeasible at every ms and mJ level (single-row inference exceeds the headroom); the R1 MLPs are feasible
everywhere except the 10% mJ level on the four KITTI cells; R2 is feasible at 30% and 50% under the ms budget on KITTI
only and under the mJ budget in every cell. The nDG under each budget is in the `budget_*` columns.

## Phase D — the registered reading

**Reading: material: a published-objective router beats random where its V-trained counterpart does not, or the reverse.** Of 440 published-objective rows, 61 lie outside the paired interval of their V-trained counterpart; **23 beat random where the counterpart does not**, **72 fail to beat random where the counterpart does**, and 0 fall below random where the counterpart wins. Every row is named below, with its intervals; none is averaged away.

### Published objective beats random, V-trained counterpart does not (23)

* KITTI mono brake J | R2_cnn_clf on Q @30%: nDG +0.237; vs random [+0.004, +0.167]; V-trained vs random [-0.119, +0.249]; published - V [-0.166, +0.217]
* KITTI mono traj JB | R1_mlp_reg on G @10%: nDG +0.221; vs random [+0.026, +0.378]; V-trained vs random [-0.251, +0.459]; published - V [-0.150, +0.420]
* KITTI mono traj JB | R1_mlp_clf on G @10%: nDG +0.207; vs random [+0.011, +0.648]; V-trained vs random [-0.087, +0.440]; published - V [-0.310, +0.514]
* KITTI mono traj JB | R1_mlp_clf on G @20%: nDG +0.339; vs random [+0.138, +0.729]; V-trained vs random [-0.231, +0.532]; published - V [-0.301, +0.694]
* KITTI mono traj JB | R1_mlp_clf on G @30%: nDG +0.335; vs random [+0.216, +0.654]; V-trained vs random [-0.497, +0.592]; published - V [-0.201, +0.798]
* KITTI mono traj JB | R1_mlp_clf on G @50%: nDG +0.069; vs random [+0.133, +0.423]; V-trained vs random [-0.783, +0.525]; published - V [-0.233, +1.077]
* KITTI mono traj JB | R1_gbm_reg on G @10%: nDG +0.190; vs random [+0.020, +0.585]; V-trained vs random [-0.081, +0.427]; published - V [-0.215, +0.663]
* KITTI mono traj JB | R1_gbm_reg on G @20%: nDG +0.259; vs random [+0.031, +0.651]; V-trained vs random [-0.168, +0.556]; published - V [-0.412, +0.802]
* KITTI mono traj JB | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @10%: nDG +0.190; vs random [+0.020, +0.585]; V-trained vs random [-0.081, +0.427]; published - V [-0.215, +0.663]
* KITTI mono traj JB | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.259; vs random [+0.031, +0.651]; V-trained vs random [-0.168, +0.556]; published - V [-0.412, +0.802]
* KITTI oracle brake J | R1_mlp_clf on Q @50%: nDG +0.456; vs random [+0.035, +0.154]; V-trained vs random [-0.020, +0.315]; published - V [-0.250, +0.140]
* KITTI oracle brake J | R2_cnn_clf on G @20%: nDG +0.243; vs random [+0.016, +0.153]; V-trained vs random [-0.135, +0.170]; published - V [-0.093, +0.240]
* KITTI oracle brake J | R2_cnn_clf on G @30%: nDG +0.338; vs random [+0.003, +0.157]; V-trained vs random [-0.193, +0.187]; published - V [-0.122, +0.314]
* KITTI oracle brake J | R2_cnn_clf on G @50%: nDG +0.469; vs random [+0.033, +0.209]; V-trained vs random [-0.219, +0.196]; published - V [-0.122, +0.376]
* KITTI oracle traj JB | R1_mlp_reg on G @50%: nDG +0.571; vs random [+0.020, +0.160]; V-trained vs random [-0.020, +0.469]; published - V [-0.440, +0.137]
* KITTI oracle traj JB | R2_cnn_clf on Q @50%: nDG +0.651; vs random [+0.060, +0.252]; V-trained vs random [-0.415, +0.264]; published - V [-0.078, +0.662]
* KITTI oracle traj JB | R2_cnn_clf on G @50%: nDG +0.735; vs random [+0.030, +0.386]; V-trained vs random [-0.415, +0.264]; published - V [+0.056, +0.736]
* nuScenes oracle brake J | R1_gbm_reg on G @20%: nDG +0.449; vs random [+0.038, +0.581]; V-trained vs random [-0.143, +0.621]; published - V [-0.333, +0.572]
* nuScenes oracle brake J | R1_gbm_reg on G @30%: nDG +0.574; vs random [+0.079, +0.687]; V-trained vs random [-0.099, +0.598]; published - V [-0.304, +0.590]
* nuScenes oracle brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.449; vs random [+0.038, +0.581]; V-trained vs random [-0.143, +0.621]; published - V [-0.333, +0.572]
* nuScenes oracle brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @30%: nDG +0.574; vs random [+0.079, +0.687]; V-trained vs random [-0.099, +0.598]; published - V [-0.304, +0.590]
* nuScenes oracle plan_fde JC_fde | R1_mlp_clf on Q @50%: nDG +0.424; vs random [+0.077, +0.434]; V-trained vs random [-0.363, +0.137]; published - V [+0.053, +0.697]
* nuScenes oracle plan_fde JC_fde | R1_gbm_reg on Q @50%: nDG +0.413; vs random [+0.047, +0.445]; V-trained vs random [-0.422, +0.103]; published - V [+0.061, +0.715]

### V-trained counterpart beats random, published objective does not (72)

* KITTI mono brake J | R1_mlp_reg on Q @30%: nDG +0.130; vs random [-0.111, +0.109]; V-trained vs random [+0.011, +0.254]; published - V [-0.294, +0.032]
* KITTI mono brake J | R1_mlp_reg on Q @50%: nDG +0.231; vs random [-0.043, +0.118]; V-trained vs random [+0.054, +0.255]; published - V [-0.258, +0.011]
* KITTI mono brake J | R1_mlp_reg on G @30%: nDG +0.152; vs random [-0.041, +0.184]; V-trained vs random [+0.011, +0.254]; published - V [-0.239, +0.112]
* KITTI mono brake J | R1_mlp_reg on G @50%: nDG +0.225; vs random [-0.078, +0.147]; V-trained vs random [+0.054, +0.255]; published - V [-0.299, +0.080]
* KITTI mono brake J | R1_mlp_clf on Q @50%: nDG +0.229; vs random [-0.062, +0.100]; V-trained vs random [+0.013, +0.270]; published - V [-0.257, +0.065]
* KITTI mono brake J | R1_mlp_clf on G @50%: nDG +0.190; vs random [-0.086, +0.100]; V-trained vs random [+0.013, +0.270]; published - V [-0.262, +0.044]
* KITTI mono brake J | R1_gbm_reg on Q @20%: nDG +0.036; vs random [-0.081, +0.094]; V-trained vs random [+0.003, +0.288]; published - V [-0.350, +0.039]
* KITTI mono brake J | R1_gbm_reg on Q @30%: nDG +0.088; vs random [-0.084, +0.065]; V-trained vs random [+0.026, +0.311]; published - V [-0.381, +0.016]
* KITTI mono brake J | R1_gbm_reg on Q @50%: nDG +0.215; vs random [-0.042, +0.065]; V-trained vs random [+0.079, +0.305]; published - V [-0.333, -0.065]
* KITTI mono brake J | R1_gbm_reg on G @20%: nDG +0.145; vs random [-0.023, +0.147]; V-trained vs random [+0.003, +0.288]; published - V [-0.231, +0.105]
* KITTI mono brake J | R1_gbm_reg on G @30%: nDG +0.161; vs random [-0.037, +0.133]; V-trained vs random [+0.026, +0.311]; published - V [-0.293, +0.075]
* KITTI mono brake J | R1_gbm_reg on G @50%: nDG +0.232; vs random [-0.058, +0.126]; V-trained vs random [+0.079, +0.305]; published - V [-0.299, -0.005]
* KITTI mono brake J | R1_gbm_clf on Q @10%: nDG +0.116; vs random [-0.050, +0.184]; V-trained vs random [+0.019, +0.218]; published - V [-0.194, +0.155]
* KITTI mono brake J | R1_gbm_clf on Q @20%: nDG +0.186; vs random [-0.006, +0.208]; V-trained vs random [+0.019, +0.252]; published - V [-0.174, +0.158]
* KITTI mono brake J | R1_gbm_clf on Q @30%: nDG +0.243; vs random [-0.008, +0.250]; V-trained vs random [+0.047, +0.315]; published - V [-0.215, +0.130]
* KITTI mono brake J | R1_gbm_clf on G @10%: nDG +0.116; vs random [-0.062, +0.115]; V-trained vs random [+0.019, +0.218]; published - V [-0.168, +0.066]
* KITTI mono brake J | R1_gbm_clf on G @20%: nDG +0.117; vs random [-0.083, +0.139]; V-trained vs random [+0.019, +0.252]; published - V [-0.247, +0.064]
* KITTI mono brake J | R1_gbm_clf on G @30%: nDG +0.151; vs random [-0.207, +0.126]; V-trained vs random [+0.047, +0.315]; published - V [-0.393, +0.015]
* KITTI mono brake J | R1_gbm_clf on G @50%: nDG +0.217; vs random [-0.125, +0.102]; V-trained vs random [+0.050, +0.298]; published - V [-0.335, -0.027]
* KITTI mono brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.145; vs random [-0.023, +0.147]; V-trained vs random [+0.003, +0.288]; published - V [-0.231, +0.105]
* KITTI mono brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @30%: nDG +0.161; vs random [-0.037, +0.133]; V-trained vs random [+0.026, +0.311]; published - V [-0.293, +0.075]
* KITTI mono brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @50%: nDG +0.232; vs random [-0.058, +0.126]; V-trained vs random [+0.079, +0.305]; published - V [-0.299, -0.005]
* KITTI oracle brake J | R1_mlp_reg on Q @50%: nDG +0.384; vs random [-0.125, +0.118]; V-trained vs random [+0.029, +0.297]; published - V [-0.298, -0.034]
* KITTI oracle brake J | R1_mlp_reg on G @50%: nDG +0.336; vs random [-0.061, +0.144]; V-trained vs random [+0.029, +0.297]; published - V [-0.321, +0.060]
* KITTI oracle brake J | R1_mlp_clf on Q @20%: nDG +0.120; vs random [-0.089, +0.056]; V-trained vs random [+0.015, +0.289]; published - V [-0.338, +0.019]
* KITTI oracle brake J | R1_mlp_clf on Q @30%: nDG +0.206; vs random [-0.051, +0.096]; V-trained vs random [+0.029, +0.322]; published - V [-0.361, +0.054]
* KITTI oracle brake J | R1_mlp_clf on G @20%: nDG +0.102; vs random [-0.164, +0.194]; V-trained vs random [+0.015, +0.289]; published - V [-0.323, +0.150]
* KITTI oracle brake J | R1_mlp_clf on G @30%: nDG +0.191; vs random [-0.124, +0.237]; V-trained vs random [+0.029, +0.322]; published - V [-0.333, +0.206]
* KITTI oracle brake J | R1_gbm_reg on Q @10%: nDG +0.034; vs random [-0.091, +0.006]; V-trained vs random [+0.005, +0.307]; published - V [-0.381, -0.007]
* KITTI oracle brake J | R1_gbm_reg on G @10%: nDG +0.066; vs random [-0.063, +0.081]; V-trained vs random [+0.005, +0.307]; published - V [-0.294, +0.024]
* KITTI oracle brake J | R1_gbm_clf on Q @10%: nDG +0.085; vs random [-0.064, +0.176]; V-trained vs random [+0.069, +0.332]; published - V [-0.329, -0.042]
* KITTI oracle brake J | R1_gbm_clf on Q @20%: nDG +0.232; vs random [-0.007, +0.302]; V-trained vs random [+0.070, +0.398]; published - V [-0.371, +0.170]
* KITTI oracle brake J | R1_gbm_clf on Q @30%: nDG +0.322; vs random [-0.002, +0.270]; V-trained vs random [+0.088, +0.422]; published - V [-0.361, +0.111]
* KITTI oracle brake J | R1_gbm_clf on G @10%: nDG +0.118; vs random [-0.038, +0.111]; V-trained vs random [+0.069, +0.332]; published - V [-0.284, +0.013]
* KITTI oracle brake J | R1_gbm_clf on G @20%: nDG +0.186; vs random [-0.063, +0.130]; V-trained vs random [+0.070, +0.398]; published - V [-0.346, +0.002]
* KITTI oracle brake J | R1_gbm_clf on G @30%: nDG +0.212; vs random [-0.069, +0.102]; V-trained vs random [+0.088, +0.422]; published - V [-0.408, -0.027]
* KITTI oracle brake J | R1_gbm_clf on G @50%: nDG +0.382; vs random [-0.115, +0.187]; V-trained vs random [+0.138, +0.378]; published - V [-0.437, -0.025]
* KITTI oracle brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @10%: nDG +0.066; vs random [-0.063, +0.081]; V-trained vs random [+0.005, +0.307]; published - V [-0.294, +0.024]
* KITTI oracle traj JB | R1_mlp_reg on Q @10%: nDG +0.078; vs random [-0.099, +0.220]; V-trained vs random [+0.022, +0.378]; published - V [-0.475, +0.189]
* KITTI oracle traj JB | R1_mlp_reg on G @10%: nDG +0.084; vs random [-0.100, +0.178]; V-trained vs random [+0.022, +0.378]; published - V [-0.476, +0.138]
* KITTI oracle traj JB | R1_mlp_clf on Q @10%: nDG +0.096; vs random [-0.098, +0.246]; V-trained vs random [+0.054, +0.336]; published - V [-0.420, +0.150]
* KITTI oracle traj JB | R1_mlp_clf on Q @20%: nDG +0.307; vs random [-0.170, +0.323]; V-trained vs random [+0.088, +0.580]; published - V [-0.741, +0.234]
* KITTI oracle traj JB | R1_mlp_clf on Q @30%: nDG +0.396; vs random [-0.252, +0.351]; V-trained vs random [+0.081, +0.574]; published - V [-0.792, +0.195]
* KITTI oracle traj JB | R1_mlp_clf on G @10%: nDG +0.001; vs random [-0.100, +0.049]; V-trained vs random [+0.054, +0.336]; published - V [-0.429, -0.042]
* KITTI oracle traj JB | R1_mlp_clf on G @20%: nDG +0.096; vs random [-0.192, +0.209]; V-trained vs random [+0.088, +0.580]; published - V [-0.743, +0.103]
* KITTI oracle traj JB | R1_mlp_clf on G @30%: nDG +0.285; vs random [-0.263, +0.194]; V-trained vs random [+0.081, +0.574]; published - V [-0.801, +0.038]
* KITTI oracle traj JB | R1_mlp_clf on G @50%: nDG +0.611; vs random [-0.069, +0.267]; V-trained vs random [+0.146, +0.496]; published - V [-0.559, +0.054]
* KITTI oracle traj JB | R1_gbm_reg on Q @10%: nDG +0.094; vs random [-0.100, +0.261]; V-trained vs random [+0.036, +0.473]; published - V [-0.572, +0.224]
* KITTI oracle traj JB | R1_gbm_reg on Q @20%: nDG +0.255; vs random [-0.197, +0.370]; V-trained vs random [+0.063, +0.712]; published - V [-0.885, +0.210]
* KITTI oracle traj JB | R1_gbm_reg on Q @30%: nDG +0.360; vs random [-0.290, +0.389]; V-trained vs random [+0.106, +0.631]; published - V [-0.906, +0.136]
* KITTI oracle traj JB | R1_gbm_reg on Q @50%: nDG +0.701; vs random [-0.031, +0.397]; V-trained vs random [+0.269, +0.432]; published - V [-0.464, +0.120]
* KITTI oracle traj JB | R1_gbm_reg on G @10%: nDG +0.197; vs random [-0.063, +0.252]; V-trained vs random [+0.036, +0.473]; published - V [-0.480, +0.170]
* KITTI oracle traj JB | R1_gbm_reg on G @20%: nDG +0.358; vs random [-0.050, +0.307]; V-trained vs random [+0.063, +0.712]; published - V [-0.718, +0.129]
* KITTI oracle traj JB | R1_gbm_reg on G @30%: nDG +0.488; vs random [-0.105, +0.398]; V-trained vs random [+0.106, +0.631]; published - V [-0.706, +0.071]
* KITTI oracle traj JB | R1_gbm_clf on G @10%: nDG +0.229; vs random [-0.099, +0.324]; V-trained vs random [+0.058, +0.474]; published - V [-0.514, +0.220]
* KITTI oracle traj JB | R1_gbm_clf on G @20%: nDG +0.278; vs random [-0.166, +0.273]; V-trained vs random [+0.066, +0.662]; published - V [-0.748, +0.130]
* KITTI oracle traj JB | R1_gbm_clf on G @30%: nDG +0.327; vs random [-0.204, +0.268]; V-trained vs random [+0.116, +0.625]; published - V [-0.754, -0.026]
* KITTI oracle traj JB | R1_gbm_clf on G @50%: nDG +0.618; vs random [-0.099, +0.387]; V-trained vs random [+0.186, +0.498]; published - V [-0.534, +0.037]
* KITTI oracle traj JB | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @10%: nDG +0.197; vs random [-0.063, +0.252]; V-trained vs random [+0.036, +0.473]; published - V [-0.480, +0.170]
* KITTI oracle traj JB | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.358; vs random [-0.050, +0.307]; V-trained vs random [+0.063, +0.712]; published - V [-0.718, +0.129]
* KITTI oracle traj JB | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @30%: nDG +0.488; vs random [-0.105, +0.398]; V-trained vs random [+0.106, +0.631]; published - V [-0.706, +0.071]
* nuScenes mono brake J | R1_mlp_reg on Q @20%: nDG +0.053; vs random [-0.263, +0.201]; V-trained vs random [+0.019, +0.411]; published - V [-0.587, +0.085]
* nuScenes mono brake J | R1_mlp_reg on Q @30%: nDG +0.102; vs random [-0.270, +0.246]; V-trained vs random [+0.046, +0.476]; published - V [-0.627, +0.068]
* nuScenes mono brake J | R1_mlp_reg on G @20%: nDG +0.233; vs random [-0.115, +0.379]; V-trained vs random [+0.019, +0.411]; published - V [-0.381, +0.211]
* nuScenes mono brake J | R1_mlp_reg on G @30%: nDG +0.249; vs random [-0.111, +0.364]; V-trained vs random [+0.046, +0.476]; published - V [-0.444, +0.175]
* nuScenes mono brake J | R1_gbm_reg on Q @20%: nDG +0.109; vs random [-0.142, +0.371]; V-trained vs random [+0.063, +0.471]; published - V [-0.389, +0.059]
* nuScenes mono brake J | R1_gbm_reg on Q @30%: nDG +0.224; vs random [-0.180, +0.494]; V-trained vs random [+0.025, +0.431]; published - V [-0.408, +0.290]
* nuScenes mono brake J | R1_gbm_reg on Q @50%: nDG +0.380; vs random [-0.111, +0.433]; V-trained vs random [+0.060, +0.527]; published - V [-0.348, +0.136]
* nuScenes mono brake J | R1_gbm_reg on G @30%: nDG +0.251; vs random [-0.139, +0.343]; V-trained vs random [+0.025, +0.431]; published - V [-0.387, +0.149]
* nuScenes mono brake J | R1_gbm_reg on G @50%: nDG +0.387; vs random [-0.116, +0.415]; V-trained vs random [+0.060, +0.527]; published - V [-0.464, +0.223]
* nuScenes mono brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @30%: nDG +0.251; vs random [-0.139, +0.343]; V-trained vs random [+0.025, +0.431]; published - V [-0.387, +0.149]
* nuScenes mono brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @50%: nDG +0.387; vs random [-0.116, +0.415]; V-trained vs random [+0.060, +0.527]; published - V [-0.464, +0.223]

### Outside the V-trained counterpart's paired interval (61)

* KITTI mono brake J | R1_gbm_reg on Q @50%: nDG +0.215; vs random [-0.042, +0.065]; V-trained vs random [+0.079, +0.305]; published - V [-0.333, -0.065]
* KITTI mono brake J | R1_gbm_reg on G @50%: nDG +0.232; vs random [-0.058, +0.126]; V-trained vs random [+0.079, +0.305]; published - V [-0.299, -0.005]
* KITTI mono brake J | R1_gbm_clf on G @50%: nDG +0.217; vs random [-0.125, +0.102]; V-trained vs random [+0.050, +0.298]; published - V [-0.335, -0.027]
* KITTI mono brake J | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @50%: nDG +0.232; vs random [-0.058, +0.126]; V-trained vs random [+0.079, +0.305]; published - V [-0.299, -0.005]
* KITTI mono traj JB | R1_mlp_reg on G @30%: nDG +0.160; vs random [-0.103, +0.692]; V-trained vs random [-0.399, +0.476]; published - V [+0.001, +0.562]
* KITTI mono traj JB | R1_gbm_clf on Q @10%: nDG +0.071; vs random [-0.070, +0.406]; V-trained vs random [-0.270, -0.057]; published - V [+0.047, +0.633]
* KITTI oracle brake J | R1_mlp_reg on Q @50%: nDG +0.384; vs random [-0.125, +0.118]; V-trained vs random [+0.029, +0.297]; published - V [-0.298, -0.034]
* KITTI oracle brake J | R1_gbm_reg on Q @10%: nDG +0.034; vs random [-0.091, +0.006]; V-trained vs random [+0.005, +0.307]; published - V [-0.381, -0.007]
* KITTI oracle brake J | R1_gbm_reg on Q @20%: nDG +0.077; vs random [-0.108, +0.037]; V-trained vs random [-0.004, +0.373]; published - V [-0.462, -0.008]
* KITTI oracle brake J | R1_gbm_clf on Q @10%: nDG +0.085; vs random [-0.064, +0.176]; V-trained vs random [+0.069, +0.332]; published - V [-0.329, -0.042]
* KITTI oracle brake J | R1_gbm_clf on Q @50%: nDG +0.520; vs random [+0.056, +0.225]; V-trained vs random [+0.138, +0.378]; published - V [-0.287, -0.005]
* KITTI oracle brake J | R1_gbm_clf on G @30%: nDG +0.212; vs random [-0.069, +0.102]; V-trained vs random [+0.088, +0.422]; published - V [-0.408, -0.027]
* KITTI oracle brake J | R1_gbm_clf on G @50%: nDG +0.382; vs random [-0.115, +0.187]; V-trained vs random [+0.138, +0.378]; published - V [-0.437, -0.025]
* KITTI oracle traj JB | R1_mlp_clf on G @10%: nDG +0.001; vs random [-0.100, +0.049]; V-trained vs random [+0.054, +0.336]; published - V [-0.429, -0.042]
* KITTI oracle traj JB | R1_gbm_clf on G @30%: nDG +0.327; vs random [-0.204, +0.268]; V-trained vs random [+0.116, +0.625]; published - V [-0.754, -0.026]
* KITTI oracle traj JB | R2_cnn_clf on Q @10%: nDG +0.155; vs random [-0.010, +0.130]; V-trained vs random [-0.100, +0.038]; published - V [+0.057, +0.184]
* KITTI oracle traj JB | R2_cnn_clf on Q @20%: nDG +0.313; vs random [-0.011, +0.235]; V-trained vs random [-0.200, +0.076]; published - V [+0.083, +0.316]
* KITTI oracle traj JB | R2_cnn_clf on Q @30%: nDG +0.447; vs random [-0.025, +0.263]; V-trained vs random [-0.299, +0.111]; published - V [+0.095, +0.487]
* KITTI oracle traj JB | R2_cnn_clf on G @20%: nDG +0.261; vs random [-0.083, +0.247]; V-trained vs random [-0.200, +0.076]; published - V [+0.023, +0.397]
* KITTI oracle traj JB | R2_cnn_clf on G @30%: nDG +0.506; vs random [-0.090, +0.421]; V-trained vs random [-0.299, +0.111]; published - V [+0.045, +0.555]
* KITTI oracle traj JB | R2_cnn_clf on G @50%: nDG +0.735; vs random [+0.030, +0.386]; V-trained vs random [-0.415, +0.264]; published - V [+0.056, +0.736]
* nuScenes mono brake J | R2_cnn_clf on Q @50%: nDG +0.329; vs random [-0.181, +0.269]; V-trained vs random [-0.517, -0.082]; published - V [+0.019, +0.678]
* nuScenes mono brake J | R2_cnn_clf on G @50%: nDG +0.326; vs random [-0.183, +0.458]; V-trained vs random [-0.517, -0.082]; published - V [+0.002, +0.835]
* nuScenes mono plan_ade JC_ade | R1_mlp_reg on Q @20%: nDG +0.082; vs random [-0.248, +0.098]; V-trained vs random [-0.130, +0.290]; published - V [-0.313, -0.027]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg on Q @50%: nDG +0.491; vs random [-0.067, +0.388]; V-trained vs random [-0.320, -0.016]; published - V [+0.033, +0.565]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg on G @10%: nDG +0.116; vs random [-0.099, +0.264]; V-trained vs random [-0.240, -0.025]; published - V [+0.014, +0.416]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg on G @20%: nDG +0.272; vs random [-0.024, +0.307]; V-trained vs random [-0.281, -0.030]; published - V [+0.143, +0.523]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg on G @30%: nDG +0.286; vs random [-0.086, +0.252]; V-trained vs random [-0.300, -0.004]; published - V [+0.085, +0.464]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg on G @50%: nDG +0.370; vs random [-0.085, +0.255]; V-trained vs random [-0.320, -0.016]; published - V [+0.002, +0.486]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @10%: nDG +0.116; vs random [-0.099, +0.264]; V-trained vs random [-0.240, -0.025]; published - V [+0.014, +0.416]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.272; vs random [-0.024, +0.307]; V-trained vs random [-0.281, -0.030]; published - V [+0.143, +0.523]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @30%: nDG +0.286; vs random [-0.086, +0.252]; V-trained vs random [-0.300, -0.004]; published - V [+0.085, +0.464]
* nuScenes mono plan_ade JC_ade | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @50%: nDG +0.370; vs random [-0.085, +0.255]; V-trained vs random [-0.320, -0.016]; published - V [+0.002, +0.486]
* nuScenes mono plan_fde JC_fde | R1_mlp_reg on Q @20%: nDG -0.040; vs random [-0.373, +0.002]; V-trained vs random [-0.049, +0.294]; published - V [-0.462, -0.081]
* nuScenes mono plan_fde JC_fde | R1_mlp_reg on Q @30%: nDG +0.106; vs random [-0.332, +0.163]; V-trained vs random [-0.051, +0.268]; published - V [-0.394, -0.021]
* nuScenes mono plan_fde JC_fde | R1_mlp_reg on Q @50%: nDG +0.103; vs random [-0.364, +0.105]; V-trained vs random [-0.047, +0.318]; published - V [-0.569, -0.057]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg on Q @10%: nDG +0.181; vs random [-0.145, +0.220]; V-trained vs random [-0.340, -0.044]; published - V [+0.040, +0.378]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg on Q @20%: nDG +0.191; vs random [-0.162, +0.297]; V-trained vs random [-0.369, +0.010]; published - V [+0.057, +0.415]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg on Q @50%: nDG +0.401; vs random [-0.093, +0.336]; V-trained vs random [-0.346, +0.131]; published - V [+0.007, +0.528]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg on G @10%: nDG +0.034; vs random [-0.142, +0.119]; V-trained vs random [-0.340, -0.044]; published - V [+0.060, +0.318]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg on G @20%: nDG +0.127; vs random [-0.091, +0.180]; V-trained vs random [-0.369, +0.010]; published - V [+0.044, +0.456]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg on G @30%: nDG +0.203; vs random [-0.120, +0.191]; V-trained vs random [-0.407, +0.046]; published - V [+0.017, +0.473]
* nuScenes mono plan_fde JC_fde | R2_cnn_clf on Q @50%: nDG +0.425; vs random [-0.028, +0.368]; V-trained vs random [-0.372, +0.087]; published - V [+0.064, +0.589]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @10%: nDG +0.034; vs random [-0.142, +0.119]; V-trained vs random [-0.340, -0.044]; published - V [+0.060, +0.318]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.127; vs random [-0.091, +0.180]; V-trained vs random [-0.369, +0.010]; published - V [+0.044, +0.456]
* nuScenes mono plan_fde JC_fde | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @30%: nDG +0.203; vs random [-0.120, +0.191]; V-trained vs random [-0.407, +0.046]; published - V [+0.017, +0.473]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg on Q @10%: nDG +0.068; vs random [-0.024, +0.157]; V-trained vs random [-0.202, -0.039]; published - V [+0.083, +0.303]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg on Q @20%: nDG +0.073; vs random [-0.024, +0.201]; V-trained vs random [-0.187, +0.010]; published - V [+0.049, +0.289]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg on Q @30%: nDG +0.109; vs random [-0.030, +0.274]; V-trained vs random [-0.378, +0.059]; published - V [+0.003, +0.523]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg on Q @50%: nDG +0.167; vs random [-0.012, +0.365]; V-trained vs random [-0.354, +0.028]; published - V [+0.040, +0.560]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg on G @10%: nDG +0.109; vs random [-0.079, +0.277]; V-trained vs random [-0.202, -0.039]; published - V [+0.013, +0.415]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg on G @20%: nDG +0.193; vs random [-0.033, +0.375]; V-trained vs random [-0.187, +0.010]; published - V [+0.028, +0.472]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @10%: nDG +0.109; vs random [-0.079, +0.277]; V-trained vs random [-0.202, -0.039]; published - V [+0.013, +0.415]
* nuScenes oracle plan_ade JC_ade | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.193; vs random [-0.033, +0.375]; V-trained vs random [-0.187, +0.010]; published - V [+0.028, +0.472]
* nuScenes oracle plan_fde JC_fde | R1_mlp_clf on Q @50%: nDG +0.424; vs random [+0.077, +0.434]; V-trained vs random [-0.363, +0.137]; published - V [+0.053, +0.697]
* nuScenes oracle plan_fde JC_fde | R1_gbm_reg on Q @20%: nDG +0.168; vs random [-0.049, +0.264]; V-trained vs random [-0.414, +0.037]; published - V [+0.015, +0.572]
* nuScenes oracle plan_fde JC_fde | R1_gbm_reg on Q @50%: nDG +0.413; vs random [+0.047, +0.445]; V-trained vs random [-0.422, +0.103]; published - V [+0.061, +0.715]
* nuScenes oracle plan_fde JC_fde | R1_gbm_reg on G @20%: nDG +0.322; vs random [-0.002, +0.496]; V-trained vs random [-0.414, +0.037]; published - V [+0.065, +0.752]
* nuScenes oracle plan_fde JC_fde | R1_gbm_reg on G @30%: nDG +0.325; vs random [-0.038, +0.493]; V-trained vs random [-0.450, +0.083]; published - V [+0.014, +0.763]
* nuScenes oracle plan_fde JC_fde | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @20%: nDG +0.322; vs random [-0.002, +0.496]; V-trained vs random [-0.414, +0.037]; published - V [+0.065, +0.752]
* nuScenes oracle plan_fde JC_fde | R1_gbm_reg <-> R2_cnn_clf on G (budget-adaptive (Geng Eq. 8)) @30%: nDG +0.325; vs random [-0.038, +0.493]; V-trained vs random [-0.450, +0.083]; published - V [+0.014, +0.763]
