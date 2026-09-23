# The camera-to-ego translation in the monocular lift: a sensitivity

> **Pre-Task-23 run.** The numbers in this document come from the run before the ego-frame convention (Task 23, 2026-09-22) and are kept as the record of that run. `results/final/` now holds the ego-frame run; where the two differ, `results/final/` is current (Task 28 audit, `scripts/157_doc_staleness.py`).

> **Superseded by Task 23.** This report measured the camera-to-ego *translation* as a default-off sensitivity on the camera-frame lift. The lift now reports its geometry in the ego frame, with the rotation as well (`docs/iclr_ego_frame_convention.md`, `docs/iclr_formulas.md` §1.9), and the shipped `results/final/lift_offset_sensitivity.csv` restates the same 14 settings as the cost of the old convention: ego frame as the base, camera frame as the alternative. The figures below are the Task 22 version.

Pre-registration: the pre-registration record (not part of this release), 2026-09-20 12:05 (Task 22 Part C), committed before the code was written.
**Sensitivity only — no official output is changed.** The one new file is
`results/final/lift_offset_sensitivity.csv`.

## What is at issue

`rap.mono` lifts a 2D box to a range `z` along the **camera** optical axis and a lateral extent about the **camera**
axis, and the planners then read those as ego-frame quantities. The camera is not at the ego origin. Measured from
the calibration itself:

| track | camera → ego translation | source |
|---|---|---|
| KITTI (cam2 → IMU) | **1.078 – 1.143 m forward**, 0.310 – 0.330 m **right** | `Calib.cam_to_imu`, 21 sequences |
| nuScenes (CAM_FRONT → ego) | **1.701 – 1.722 m forward**, 0.005 – 0.016 m left | `calibrated_sensor`, 85 scenes |

So the shipped lift places every object about a metre nearer than it is in the ego frame, and on KITTI about a
third of a metre to one side. That is a modelling choice, not a defect: the shipped results use the camera frame
throughout, consistently between CHEAP and FULL. This part measures how much the choice matters.

## The flag

`mono.predicted_geometry(..., cam_offset=(t_x, t_y))`, **default off**: `z += t_x`, `lat_min += t_y`,
`lat_max += t_y`, and the two diagnostic cues `z_ground`, `z_height` likewise. Only the translation is applied,
never the rotation. TTC is untouched, because `ttc_from_scale` reads the box-height scale change, which a constant
translation does not move.

`mono.apply_cam_offset` applies the same shift to an already-lifted geometry, which is exact: the fused range is a
convex combination whose weight depends on `y2 − c_y` alone, so shifting after the fusion equals shifting each cue
before it; and the lateral extent is `(u − c_x)·z_cam/f_x`, which uses the camera-frame depth and is reached only
through `t_y`. **Check L1** asserts this against re-running the lift on 848 cached detections: max |difference| =
**0**, exactly.

## The sanity gate

With the flag off, every setting reproduces the shipped values to 3 decimals — **14 of 14**, against
`calibration_cells.csv` (scheme S0, all units) for the 13 cells that are there, and against the shipped per-frame
tables for KITTI mono RT-DETR 320→640, which has no calibration row.

## Result

Per-frame tables were rebuilt twice for every setting, flag off and flag on, through the unchanged
`decision.build` and `65.build_b`. Quantities are those of `102_calibration_cells.py`.

| cell | system | harmed off → on (Δ) | ρ off → on (Δ) |
|---|---|---|---|
| nuScenes mono Y8 320→640 | q_brake | 45.5% → 47.2% (+1.7) | 0.536 → 0.488 (**−0.048**) |
| nuScenes oracle Y8 320→640 | q_brake | 48.5% → 48.8% (+0.3) | 0.602 → 0.499 (**−0.103**) |
| KITTI mono RT-DETR 320→640 | q_brake | 45.6% → 44.4% (−1.2) | 0.796 → 0.734 (**−0.062**) |
| KITTI mono RT-DETR 320→640 | q_traj | 40.1% → 41.9% (+1.9) | 0.317 → 0.553 (**+0.237**) |
| KITTI mono RT-DETR 480→640 | q_brake | 44.6% → 43.4% (−1.1) | 0.805 → 0.788 (−0.017) |
| KITTI mono RT-DETR 480→640 | q_traj | 41.4% → 45.6% (+4.2) | 0.548 → 0.744 (**+0.196**) |
| KITTI mono Y8 320→640 | q_brake | 40.7% → 39.2% (−1.5) | 0.192 → 0.183 (−0.009) |
| KITTI mono Y8 320→640 | q_traj | 35.2% → 36.0% (+0.8) | 0.062 → 0.091 (+0.029) |
| KITTI oracle Y8 320→640 | q_brake | 34.1% → 31.8% (−2.3) | 0.091 → 0.084 (−0.008) |
| KITTI oracle Y8 320→640 | q_traj | 9.5% → 6.3% (−3.2) | 0.004 → 0.004 (−0.000) |
| KITTI mono Y8 384→640 | q_brake | 44.6% → 43.2% (−1.4) | 0.551 → 0.508 (−0.043) |
| KITTI mono Y8 384→640 | q_traj | 45.4% → 46.9% (+1.5) | 0.211 → 0.341 (**+0.131**) |
| KITTI mono Y8 512→640 | q_brake | 45.7% → 45.6% (−0.1) | 0.680 → 0.680 (+0.001) |
| KITTI mono Y8 512→640 | q_traj | 40.7% → 47.0% (**+6.2**) | 0.435 → 1.004 (**+0.569**) |

Bold marks a move outside the pre-registered tolerance (5 points of harm, 0.05 of ρ).

### The registered reading: **sensitive**, in six of fourteen settings

* `nuScenes oracle Y8 320→640 q_brake` (ρ −0.103)
* `KITTI mono RT-DETR 320→640 q_brake` (ρ −0.062) and `q_traj` (ρ +0.237)
* `KITTI mono RT-DETR 480→640 q_traj` (ρ +0.196)
* `KITTI mono Y8 384→640 q_traj` (ρ +0.131)
* `KITTI mono Y8 512→640 q_traj` (harm +6.2 points, ρ +0.569)

### What actually moves

* **The harm rate is stable; the harm-to-benefit ratio is not.** The largest move in `harmed` anywhere is 6.2
  points and the median is 1.5; every setting except one is inside the 5-point tolerance. ρ moves by up to 0.569,
  median 0.046.
* **The split is by consumer, not by dataset.** On braking the largest ρ move is 0.103 and five of seven settings
  are inside tolerance; on Planner B's trajectory cost four of five settings are outside it, all in the same
  direction — the benefit shrinks faster than the harm. `all_full` on q_traj falls in every setting (for example
  KITTI mono Y8 512→640 +0.047 → −0.000, RT-DETR 320→640 +0.128 → +0.056), so escalation buys less once objects
  sit a metre further away, while the harmed frames largely stay harmed.
* **The affected set shrinks slightly everywhere** (for example 1,355 → 1,206 frames on KITTI mono Y8 320→640
  q_traj): pushing obstacles further away moves some frames below the threshold where CHEAP and FULL choose
  different actions at all.
* **`oracle20` falls on q_traj and is flat on q_brake**, which is the same effect measured at the quota the
  benchmark reports.

### How to read this for the paper

The sign-varying result itself — that escalation harms a large minority of affected frames — **survives the change
in every setting**: the harm rate stays between 6% and 47% and moves by at most 6.2 points, and no setting crosses
from "harm is common" to "harm is rare". What is not robust is the *magnitude* of the harm-to-benefit ratio ρ on
the rollout planner, which the camera-frame convention flatters. A paper that quotes ρ on q_traj should either
state the convention or quote the range across conventions; the braking numbers and every harm rate can be quoted
as they are.

## q_plan is not re-run, and this is what it would cost

nuScenes `q_plan` is Planner C's ADE/FDE against the real trajectory, computed from rebuilt submissions and
rasters, so the flag cannot be evaluated from cached tables. The chain and its measured cost:

| step | cost |
|---|---|
| rebuild submissions with the offset applied | ~15 min |
| Planner C, oracle + mono, 12 chunks | ~2.6 h |
| `62_planning_metric_eta.py`, both variants | ~1.1 h (measured 33 + 32 min today) |
| **total** | **~4 h** |
| plus PKL/TIP gains, 24 chunks, if those are wanted too | ~5 h (→ ~9 h) |

As instructed, it was not run.

## Runs and code

| artefact | run |
|---|---|
| per-frame tables, flag off and on, 28 tables, and check L1 | `20260920_133624_lift_offset_outcomes` (stage **N7**) |
| the five quantities, the sanity gate and the reading | stage **C27**, which writes `sanity.csv` and `reading.json` into its run directory |
| the reported table | `results/final/lift_offset_sensitivity.csv` |

Code: `rap.mono.predicted_geometry(cam_offset=...)` and `rap.mono.apply_cam_offset`,
`scripts/140_lift_offset_outcomes.py` (stage N7), `scripts/141_lift_offset_sensitivity.py` (stage C27). Every run
used `PYTHONHASHSEED=0`.
