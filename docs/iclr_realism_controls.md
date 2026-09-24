# Two realism controls on the KITTI core track: detection persistence and reference geometry

> **Frame convention.** The figures quoted in this report were computed with the monocular lift in the camera frame, the convention before Task 23. Every table in `results/final/` is now computed in the ego frame; `docs/iclr_ego_frame_convention.md` and `results/final/ego_frame_convention.csv` give each registered quantity old beside new, and the camera-frame convention stays selectable with `--frame camera`.

**Objections tested.**
1. The controllers act on single-frame detections, while real stacks require a detection to persist before acting.
   A persistence requirement might remove the transient false positives that drive harm.
2. On KITTI, harm may come mostly from the monocular 2D→3D lift rather than from the detection set. For YOLOv8s
   320→640 q_traj, harmed falls from 35.2% under mono to 9.5% under reference geometry.

**Provenance.**
* Code: `scripts/134_realism_outcomes.py` writes the per-frame tables and needs the KITTI labels, calibration and
  OXTS. `scripts/135_realism_controls.py` computes every figure below from those tables.
* Outputs: `results/final/persistence_sweep.csv`, `results/final/reference_geometry_sweep.csv`.
* CPU only on cached detections: no detector re-runs, no allocator training. No official result file changed.

**Consumers.** q_brake is the braking controller (J). q_traj is the receding-horizon trajectory controller (JB,
Planner B, the benchmark's `static_obstacles` preset). The formulas are in `docs/iclr_formulas.md`.

**Quantities, over all units** (the definitions of the calibration cells):

| quantity | definition |
|---|---|
| affected | inputs with \|V\| > 1e-9 |
| harmed | share of affected inputs with V < 0 |
| rho | Σ\|V\| over V < 0 / Σ V over V > 0 |
| all-full | Σ V as a share of the all-cheap loss |
| oracle@20 | the exact top-20% expectation of V as a share of the all-cheap loss |

**Sanity gate: passed.** At n = 1 every setting reproduces the shipped values: affected exactly, and the other four
quantities at three decimals. Twelve settings were checked:
* ten from `results/final/calibration_cells.csv` (scheme S0, all units);
* the two RT-DETR-l 320→640 settings from their shipped per-frame tables, since that pair has no calibration cell.

For example, YOLOv8s 320→640 mono q_traj reads 1,355 / 35.2% / 0.062 / 42.43% / 45.24%.

## 1. Persistence

**The filter.** It is applied identically to the CHEAP and FULL lists, before both controllers and before the
reference matching.
* A detection at frame t (confidence ≥ 0.25) passes only if the same sequence and fidelity has, in each of the
  previous n − 1 frames, a detection of the same coarse class at confidence ≥ 0.25 with image IoU ≥ 0.3.
* The previous frames' lists are unfiltered.
* A frame with fewer than n − 1 predecessors checks the ones it has, so the first frame keeps everything.
* At 10 Hz, n = 3 adds 0.2 s of latency.
* **nuScenes is not run:** its keyframes are 2 Hz, so n = 2 alone would add 0.5 s.

**Pre-registered reading, per setting (n = 3 against n = 1).**
* removes: harmed and rho both at most half;
* mitigates: at least one falls by at least a third;
* otherwise, no material change.

| pair | geometry | consumer | harmed n = 1 / 2 / 3 | rho n = 1 / 2 / 3 | detections per frame, CHEAP / FULL, n = 1 → 3 | reading |
|---|---|---|---|---|---|---|
| YOLOv8s 320→640 | mono | q_brake | 40.7 / 38.7 / 36.5% | 0.192 / 0.150 / 0.120 | 3.6 / 6.1 → 2.6 / 4.6 | **mitigates** (rho ×0.62) |
| YOLOv8s 320→640 | mono | q_traj | 35.2 / 34.3 / 33.3% | 0.062 / 0.047 / 0.045 | same | no material change |
| YOLOv8s 320→640 | reference | q_brake | 34.1 / 29.9 / 26.9% | 0.091 / 0.063 / 0.047 | same | **mitigates** (rho ×0.52) |
| YOLOv8s 320→640 | reference | q_traj | 9.5 / 7.5 / 8.7% | 0.004 / 0.001 / 0.007 | same | no material change |
| YOLOv8s 384→640 | mono | q_brake | 44.6 / 44.5 / 42.9% | 0.551 / 0.461 / 0.370 | 4.5 / 6.1 → 3.3 / 4.6 | no material change (rho ×0.67) |
| YOLOv8s 384→640 | mono | q_traj | 45.4 / 46.0 / 45.6% | 0.211 / 0.170 / 0.168 | same | no material change |
| YOLOv8s 512→640 | mono | q_brake | 45.7 / 45.1 / 44.8% | 0.680 / 0.600 / 0.567 | 5.5 / 6.1 → 4.1 / 4.6 | no material change |
| YOLOv8s 512→640 | mono | q_traj | 40.7 / 40.9 / 41.0% | 0.435 / 0.310 / 0.299 | same | no material change (rho ×0.69) |
| RT-DETR-l 320→640 | mono | q_brake | 45.6 / 47.7 / 47.6% | 0.796 / 0.791 / 0.715 | 9.3 / 13.0 → 6.3 / 9.5 | no material change |
| RT-DETR-l 320→640 | mono | q_traj | 40.1 / 40.6 / 40.5% | 0.317 / 0.263 / 0.241 | same | no material change |
| RT-DETR-l 480→640 | mono | q_brake | 44.6 / 46.1 / 46.5% | 0.805 / 0.877 / 0.840 | 12.7 / 13.0 → 9.1 / 9.5 | no material change |
| RT-DETR-l 480→640 | mono | q_traj | 41.4 / 39.4 / 38.8% | 0.548 / 0.557 / 0.522 | same | no material change |

**Summary: no material change** (10 of 12 settings; 2 mitigate, both YOLOv8s 320→640 braking, none removes).

* Persistence removes 26–32% of each list's detections at n = 3, yet harmed shares move by at most 7.2 points.
* It does trim rho in most settings, by up to about half on the aggressive YOLOv8s pair. It also raises the share of
  loss that FULL recovers: all-full on YOLOv8s 320→640 mono braking goes from 50.2% to 58.1%.

The five quantities at n = 2 and the all-full and oracle@20 values are in `persistence_sweep.csv`.

## 2. Reference geometry for every KITTI pair

**The control.** The same code path and matching rule as the existing control. A kept detection whose best IoU with a
reference box is ≥ 0.5 inherits that box's range, lateral extent and TTC, while detection existence (misses and false
positives) is unchanged.

**Pre-registered reading, per pair.** Geometry-driven if harmed and rho both fall to at most half of their mono values;
otherwise the harm persists without lifting error.

| pair | consumer | mono: affected / harmed / rho / all-full / oracle@20 | reference: affected / harmed / rho / all-full / oracle@20 | harmed ratio | rho ratio | reading |
|---|---|---|---|---|---|---|
| YOLOv8s 320→640 | q_brake | 3,026 / 40.7% / 0.192 / 50.2% / 62.0% | 1,846 / 34.1% / 0.091 / 67.5% / 74.3% | 0.84 | 0.48 | persists |
| YOLOv8s 384→640 | q_brake | 2,646 / 44.6% / 0.551 / 16.3% / 36.3% | 1,362 / 44.1% / 0.318 / 30.8% / 45.2% | 0.99 | 0.58 | persists |
| YOLOv8s 512→640 | q_brake | 2,325 / 45.7% / 0.680 / 8.0% / 24.9% | 1,078 / 44.2% / 0.513 / 13.9% / 28.6% | 0.97 | 0.76 | persists |
| RT-DETR-l 320→640 | q_brake | 3,567 / 45.6% / 0.796 / 8.2% / 40.2% | 2,644 / 50.6% / 0.798 / 10.3% / 50.9% | 1.11 | 1.00 | persists |
| RT-DETR-l 480→640 | q_brake | 3,091 / 44.6% / 0.805 / 5.9% / 30.4% | 2,338 / 48.0% / 0.949 / 2.0% / 38.7% | 1.08 | 1.18 | persists |
| YOLOv8s 320→640 | q_traj | 1,355 / 35.2% / 0.062 / 42.4% / 45.2% | 779 / 9.5% / 0.004 / 69.0% / 69.2% | 0.27 | 0.06 | **geometry-driven** |
| YOLOv8s 384→640 | q_traj | 920 / 45.4% / 0.211 / 17.8% / 22.6% | 292 / 23.0% / 0.005 / 31.5% / 31.7% | 0.505 | 0.02 | persists (harmed ratio 0.505, just above 0.5) |
| YOLOv8s 512→640 | q_traj | 624 / 40.7% / 0.435 / 4.7% / 8.4% | 185 / 28.1% / 0.055 / 11.3% / 12.0% | 0.69 | 0.13 | persists |
| RT-DETR-l 320→640 | q_traj | 1,166 / 40.1% / 0.317 / 12.8% / 18.8% | 512 / 24.8% / 0.034 / 14.8% / 15.3% | 0.62 | 0.11 | persists |
| RT-DETR-l 480→640 | q_traj | 730 / 41.4% / 0.548 / 3.7% / 8.2% | 256 / 39.8% / 0.358 / 0.9% / 1.4% | 0.96 | 0.65 | persists |

**Summary: persists without lifting error, for q_brake (5 of 5 pairs) and for q_traj (4 of 5 pairs).**

**Per consumer.**
* **Braking controller.** Removing lifting error leaves the harmed share nearly unchanged (ratios 0.84–1.11) and rho at
  0.48–1.18 of its mono value. On both RT-DETR-l pairs, rho does not fall at all. Braking harm comes from the detection
  set.
* **Trajectory controller.** The reading holds by the letter of the rule, but the pattern is different from braking.
  * Rho, the size of harm relative to benefit, collapses under reference geometry on four of the five pairs, to
    0.02–0.13 of its mono value (to 0.004–0.055 in absolute terms). Only RT-DETR-l 480→640 keeps it, at 0.65.
  * The harmed share falls less, to 0.27–0.69 of mono on those pairs. YOLOv8s 384→640 misses the "geometry-driven"
    threshold by half a point.
  * For q_traj, lifting error thus drives most of the *size* of harm on YOLOv8s, but not how *often* FULL makes a plan
    worse.

## 3. For the paper

* **Persistence does not remove sign variation.** Under a 3-frame persistence requirement, harmed shares stay at
  33–48% (mono). No setting halves both harm measures; only YOLOv8s 320→640 braking loses a third of its rho.
* **The monocular lift is not the source of braking harm.** With reference geometry on every KITTI pair, braking
  harmed shares are 34–51%, and rho is 0.09–0.95.
* **For the trajectory controller, report both numbers.** Reference geometry leaves 10–40% of affected inputs harmed,
  but the harm is small relative to the benefit (rho ≤ 0.055 on four pairs). The mono q_traj harm is therefore largely
  a lifting effect in magnitude.
* **The reference control keeps false positives with their mono geometry**, so the remaining harm includes lifted
  false positives. The per-object mechanism is not separated here.
