# Energy conventions of the measured budgets

> **Frame convention.** The figures quoted in this report were computed with the monocular lift in the camera frame, the convention before Task 23. Every table in `results/final/` is now computed in the ego frame; `docs/iclr_ego_frame_convention.md` and `results/final/ego_frame_convention.csv` give each registered quantity old beside new, and the camera-frame convention stays selectable with `--frame camera`.

**What this is.** An audit of how detector and allocator energy were charged in the energy budgets, and every energy-budget
result recomputed under one rail convention.
* Code: `scripts/132_allocator_rails.py` (board measurement) and `scripts/133_energy_module_budgets.py`.
* Outputs: `results/final/energy_module_*`, `allocator_rails.json`.
* No shipped result file changed.

Line numbers below refer to this release.

## 1. How the shipped fields are defined

**`energy_gpu_mj_per_frame` and `energy_module_mj_per_frame`** (`scripts/01_profile_jetson.py`; runs
`20260912_021305_profile` KITTI YOLOv8s, `20260912_073228_profile` nuScenes, `20260912_023524_profile` RT-DETR-l):

| aspect | definition |
|---|---|
| rails | every INA3221 rail is read (`src/rap/power.py:13-44`: CPU, GPU, SOC, CV, VDDRQ, SYS5V). `energy_gpu` uses GPU (01:127); `energy_module` uses GPU + SOC + CPU (01:128-130). SYS5V over idle is computed (01:122-125) but not used |
| idle | subtracted. One 10 s idle recording opens the run (01:171-175); over-idle power = mean during the timed loop minus the idle mean, per rail (01:122-125) |
| window | power is averaged over the whole timed loop of `reps` frames: preprocessing, forward pass and postprocessing, CUDA-synchronised (01:79-89). It is multiplied by the **median end-to-end latency** (pre + forward + post), not by the forward pass alone. Images are read before the loop (01:157), so decode is excluded |
| sampling | a thread reads the rails and sleeps 50 ms (01:30-38): nominal 20 Hz, measured 15.9–16.3 Hz; 52–140 samples per loop over 3.3–8.8 s |
| warm-up | `warmup` detections per mode per round (01:69): 50 for KITTI YOLOv8s, 40 otherwise. 3 interleaved rounds; round 0 discarded; the summary is the median of rounds 1–2 (01:195-198) |

**nuPlan detector energies, 44.7 and 105.8 mJ per frame** (`scripts/113_nuplan_detect.py:84-113`, stored in
`results/raw/nuplan_task5/detect_summary.json`, read as the mJ costs at `120_nuplan_real_allocation.py:187-188` and
`122_target_swap.py:94`):
* **Rails and idle.** CPU + GPU, over idle (113:110-111), with a 10 s idle recording per call (113:91).
* **Window.** 10 warm-up detections per mode (113:97), then 3 passes over 100 pre-decoded CAM_F0 images.
  Energy = (CPU + GPU excess) × pass wall time / 300 frames (113:101-113): 18.84 and 20.32 ms per frame, a mean
  including Python bookkeeping.
* **Latency is timed elsewhere.** The ms budget uses the main detection pass's median, 14.37 / 23.57 ms.

**Allocator overheads** (`scripts/93_budget_allocation.py`). Energy = overhead ms × power over idle / 1e3 (93:188).

| workload | what was stored | charged at |
|---|---|---|
| gate (65 features + single-row GBM) | every rail, idle and busy (`rails_idle_mw`, `rails_busy_mw`, 93:150-156), plus `all_rails_mw_over_idle` | CPU over idle |
| R1 (detection list + one-row MLP) | CPU over idle only (93:228-230) | CPU |
| R2 (resize + upload + TensorRT) | CPU + GPU over idle only (93:265-267) | CPU + GPU |
| uncertainty, cheap-side criticality, batched GBM gate | not measured on their own workloads | the gate workload's CPU power |

**Consequences.**
* The YOLOv8s detectors are charged their GPU rail only, so their CPU pre- and postprocessing costs no energy. The
  allocators' CPU work does.
* The nuPlan detectors are charged CPU + GPU.
* The shipped energy budgets therefore use three conventions at once.

## 2. One convention

**module (deciding).** The rail set {GPU, SOC, CPU} is used on both sides.
* Detectors: `energy_module_mj_per_frame`, recomputed per profile row. It reproduces the stored field and
  `energy_gpu_mj_per_frame` exactly (12/12).
* nuPlan: the same rails × the energy pass's time per frame. This reproduces the stored CPU + GPU value exactly (2/2).
* Allocators: the shipped overhead ms × the same rails over idle. Gate rails come from the stored samples (CPU matches
  the stored value, 3/3). R1 and R2 rails come from a new measurement on the reference board: 3 repetitions of 8 s idle
  + 8 s busy per workload, the same inputs as 93, median.

**Sensitivities.**
* **all_rails:** all six rails for both.
* **as_specified:** detectors on the module rails, allocators on all six rails. "Allocators: all rails over idle"
  would otherwise charge SYS5V and VDDRQ to one side only.

| mJ per frame | nuScenes 320 / 640 | KITTI 320 / 384 / 512 / 640 | nuPlan 320 / 640 |
|---|---|---|---|
| shipped | 23.7 / 66.7 | 13.9 / 19.1 / 27.4 / 33.0 | 44.7 / 105.8 |
| module | 54.6 / 132.7 | 41.1 / 47.8 / 62.0 / 84.4 | 69.5 / 143.8 |
| all rails | 61.8 / 156.5 | 47.7 / 55.1 / 73.4 / 98.1 | 88.5 / 179.3 |

| mW over idle (median of 3) | CPU | GPU | SOC | SYS5V | VDDRQ | module | all rails |
|---|---|---|---|---|---|---|---|
| gate workload, measured again | 7,393 | −130 | 1,142 | 309 | 125 | 8,407 | 8,841 |
| R1 | 8,788 | 0 | 1,194 | 382 | 131 | 9,982 | 10,514 |
| R2 | 757 | 690 | 1,248 | 608 | 142 | 2,676 | 3,423 |

**Stability.** The re-measured R1 CPU is 1.6% below the shipped 8,929 mW, and R2 CPU + GPU is 12% below the shipped
1,617 mW. As registered, the gate is charged from its stored rails, not from the re-measurement.

**Scores.**
* The core gate scores and the multi-fidelity per-level gate predictions were not saved by 93. They were regenerated
  once with the benchmark's own deterministic fits and saved (`results/raw/*_budget_gate_scores`).
* Check S (`energy_module_score_check.json`): with them and the shipped conventions, 93 reproduces every row of its
  shipped run outputs at rtol 1e-9 (two-level primary, 1-thread and routers; multi-fidelity primary and 1-thread).
* The recompute uses these saved scores, the saved router and nuPlan scores, the shipped bootstrap draws, and the
  feasibility rule of `docs/iclr_budget_feasibility.md`. Only mJ rows are recomputed.

## 3. Claims

| claim | module | all rails | as specified | verdict |
|---|---|---|---|---|
| a. The classification detection-list router (R1_mlp_clf) beats random on nuPlan IDM scalar loss at every energy budget | paired lower bounds +0.74 / +0.65 / +0.55 / +0.37 at 10 / 20 / 30 / 50% | same | same | **holds** |
| b. On the KITTI ladder 384 px costs 0.58 of 640 px in mJ (0.78 in ms), and the energy-optimal braking plan moves inputs to 384 px | ratio **0.566**; at 20% the energy-optimal oracle puts 11.5% of inputs at 384 against 9.2% for the latency-optimal oracle, and overruns the latency budget by 18.9% | 0.562; 11.6% | 0.566; 11.5% | **holds**, with the ratio 0.57 (0.58 was the GPU-only ratio) |
| c. The pixel router under skipping accounting beats random in 0 energy rows and loses in 2 | 0 wins, **3 losses** of 30 rows (nuScenes oracle braking at 30 and 50%, KITTI oracle Planner B at 50%) | 0 wins, 2 losses of 24 | 0 wins, 3 losses of 24 | **"0 wins" holds everywhere; "2 losses" holds only under all rails** |

## 4. Everything else that moves

Feasible energy rows whose paired lower bound over random is above zero (allocators only):

| table | shipped | module |
|---|---|---|
| two-level, primary | 1 | 1 (unchanged) |
| two-level, 1-thread | 2 | 6: adds ridge on both nuPlan PDM-Closed cells at 30%, cheap-side criticality on nuScenes mono braking at 30%, ridge on nuScenes mono braking at 10% |
| routers | 1 | 7: adds R1-MLP on KITTI mono braking at 50% (both targets), KITTI oracle Planner B at 20% and 30% (clf) and 50% (reg), and nuScenes mono planner FDE at 10% |
| nuPlan real perception | 12 | 16: adds ridge on both PDM-Closed cells at 30%, R1-MLP-clf on PDM-Closed scalar_J at 10%, R1-MLP-reg on PDM-Closed scalar_J at 50% |
| multi-fidelity | gate rows infeasible | still infeasible; the braking oracle mix at 20% is 11.5 / 7.9 / 7.6% at 384 / 512 / 640 |

**Reading.** Charging detectors their GPU rail only understated detector energy by 2–4× relative to allocator energy.
Under one module convention, allocators are relatively cheaper, more energy rows are feasible, and learned allocators
(mostly the MLP detection-list router and ridge) beat random in more of them at 30–50% budgets. Latency results are
unaffected.
