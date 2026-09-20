# Exact formulas: monocular lifting, reference geometry, controllers, perception gains, ego speed

Task 19 Part C. Documentation only: nothing here was run or changed. Every statement cites the code that implements it.
Line numbers refer to the code release (`src/rap/*.py` there is identical to the development copy; scripts may differ
in their header lines). Constants carry their value and unit. Equations are LaTeX.

Notation: a CHEAP or FULL detection is a 2D box $(x_1, y_1, x_2, y_2)$ in original image pixels, with confidence $p$
and coarse class $\kappa \in \{\text{vehicle}, \text{person}, \text{cyclist}\}$. A frame keeps the detections with
$p \ge \theta$, where $\theta = 0.25$ for both fidelities (`RiskConfig.op_conf`, `src/rap/risk.py:26`;
`RiskConfig.thr`, `risk.py:34-44`) unless a calibration scheme sets a per-mode threshold.

---

## 1. Monocular 2D→3D lifting (the deployed geometry, "mono")

### 1.1 Where it runs, and on which detections

* The lift runs once per detection when the detection cache is built. It stores `z`, `lat_min`, `lat_max`, `ttc`
  next to every box (`src/rap/mono.py:116-134`).
  * KITTI: `scripts/02_run_detection.py:59,72`, with `calib = kitti.load_calib(seq)` and the defaults
    $h_\text{cam} = 1.65$ m and $\Delta t = 0.1$ s.
  * nuScenes: `scripts/40_nusc_detect.py:54,62`, with the calibration of the scene's first keyframe,
    $h_\text{cam} = 1.51$ m (`40_nusc_detect.py:21`) and $\Delta t = 0.5$ s (`src/rap/nusc.py:23`).
* The cache holds every detection above the engine floor $p \ge 0.10$ (`src/rap/detect.py:33`; NMS IoU 0.65 and at most
  100 boxes, `detect.py:34-35`). The operating threshold is applied afterwards by the consumer
  (`src/rap/decision.py:133-136`).
* Coarse classes come from COCO ids: person → person; bicycle and motorcycle → cyclist; car, bus and truck → vehicle
  (`detect.py:23-24`).

### 1.2 Camera model, intrinsics and extrinsics

* **Model.** Pinhole, no distortion. $f_x, f_y, c_x, c_y$ are read from a $3\times4$ projection matrix $P$
  (`src/rap/kitti.py:51-61`).
* **KITTI.** $P$ is `P2` (rectified left colour camera) from `calib/<seq>.txt` (`kitti.py:76-90`).
* **nuScenes.** $P = [K \mid 0]$, with $K$ the `camera_intrinsic` of CAM_FRONT's `calibrated_sensor` row. The
  rectification and extrinsic slots are identities (`nusc.py:80-86`). The first keyframe's $K$ is used for the whole
  scene (`40_nusc_detect.py:54`).
* **Extrinsics.** None enter the lift.
  * The ground-plane cue assumes the optical axis is parallel to a flat road: zero pitch and roll, camera at a fixed
    height $h_\text{cam}$.
  * No camera-to-ego rotation or translation is applied to the lifted quantities (§1.6).

### 1.3 Range: which pixel anchors depth

Two cues are combined (`mono.py:26-46`, applied at `mono.py:126-128`).

**Ground-plane cue.** The bottom edge $y_2$ is taken as the object's contact point with the road:

$$z_g = \frac{f_y \, h_\text{cam}}{\max(y_2 - c_y,\; 2\,\text{px})}$$

**Height-prior cue.** The box height and a class height prior $H_\kappa$ give a second estimate
(`mono.py:22,33-38`):

$$z_h = \frac{f_y \, H_\kappa}{\max(y_2 - y_1,\; 2\,\text{px})},\qquad
H_\text{vehicle} = 1.55\ \text{m},\; H_\text{person} = 1.72\ \text{m},\; H_\text{cyclist} = 1.72\ \text{m},\;
H_\text{other} = 1.6\ \text{m}$$

**Fusion.** A weight trusts the ground cue once the contact point is well below the horizon:

$$w = \operatorname{clip}\!\left(\frac{(y_2 - c_y) - 5}{25},\,0,\,1\right),\qquad
\hat z = \operatorname{clip}\big(w\,z_g + (1-w)\,z_h,\; 0.5\ \text{m},\; 200\ \text{m}\big)$$

$w = 1$ once $y_2$ is at least 30 px below the principal row, and $w = 0$ within 5 px of it.

### 1.4 Box-centre versus contact-point convention

* $\hat z$ is the range to the object's **near face**: the ground contact of its lowest visible edge.
* The controllers use it as the gap to the nearest point of the obstacle (`decision.py:136-137`). The reference
  geometry they are compared with is also a near-face quantity: $\text{long\_near}$ is the minimum forward coordinate
  of the box footprint (§2.2).
* **Only the nuScenes detection submissions** (the files PKL and TIP read) need a box centre. There the centre is
  placed half a class-prior length behind the near face, $x_\text{centre} = \hat z + l_\kappa/2$
  (`src/rap/nusc_submission.py:106-118`).
  * An earlier version wrote $\hat z$ as the centre; the comment at `nusc_submission.py:109-115` records that bug.
  * That comment calls $\hat z$ `range_ground(y2)`. Strictly, $\hat z$ is the fused range of §1.3, which equals the
    ground cue only when $w = 1$.

### 1.5 Lateral extent, 3D extent and class size priors

**Lateral extent** comes from the two vertical box edges at range $\hat z$. The camera $x$ axis points right and the
ego $y$ axis points left, hence the sign flip (`mono.py:49-59`):

$$\hat y_\text{min} = -\frac{(x_2 - c_x)\,\hat z}{f_x},\qquad \hat y_\text{max} = -\frac{(x_1 - c_x)\,\hat z}{f_x}$$

**3D extent used by the controllers.**
* A lifted object is the triple $(\hat z, [\hat y_\text{min}, \hat y_\text{max}], \widehat{\text{TTC}})$
  (`decision.py:136`; `scripts/65_planner_b_decision.py:59`).
* It has no longitudinal depth, no height and no yaw: the obstacle is a lateral interval at one forward range.

**Class size priors** enter the controllers' geometry only through $H_\kappa$ (§1.3). The nuScenes submissions add a
full box, `nusc_submission.py:33-36,106-129`:
* size $(w, l, h)$ = car $(1.95, 4.62, 1.73)$, pedestrian $(0.67, 0.73, 1.77)$, bicycle $(0.60, 1.70, 1.28)$,
  truck $(2.51, 6.93, 2.84)$, bus $(2.94, 11.19, 3.44)$, motorcycle $(0.77, 2.11, 1.47)$ m;
* centre $\big(\hat z + l/2,\; (\hat y_\text{min} + \hat y_\text{max})/2,\; h/2\big)$ in the ego frame, then mapped to
  the global frame with the ego pose;
* yaw equal to the ego heading, and velocity $(0, 0)$.

### 1.6 Transform to the ego frame

**The lift applies none.** $\hat z$ is a depth along the camera axis, and $\hat y$ is measured from the camera's
optical centre. The code states that a constant lateral mounting offset is absorbed into the principal point
(`mono.py:52-54`). No longitudinal offset is applied either.

**The reference geometry is in the vehicle frame:**
* KITTI: the IMU frame, $x$ forward, $y$ left, $z$ up. Camera points are mapped by
  $T_{\text{cam}\to\text{imu}} = T_{\text{imu}\to\text{velo}}^{-1}\,T_{\text{velo}\to\text{cam}}^{-1}\,R_\text{rect}^{-1}$
  (`kitti.py:63-66,139-142`).
* nuScenes: the ego frame at the CAM_FRONT keyframe's ego pose (`nusc.py:94-98,145`).

The mono and reference quantities therefore differ by the camera's mounting position as well as by estimation error.

Magnitudes, read from the calibration files and not applied anywhere in the code:
* **KITTI.** $T_{\text{cam}\to\text{imu}}$ places the rectified camera origin at $x = +1.08$ to $+1.14$ m, $y = -0.31$ to
  $-0.33$ m and $z = +0.73$ to $+0.75$ m in the IMU frame (range over the 21 sequences' calibrations). `P2`'s own
  baseline shifts the left colour camera by a further 0.06 m along the rectified $x$ axis.
* **nuScenes.** CAM_FRONT's `calibrated_sensor.translation` in the ego frame is $x = +1.67$ to $+1.72$ m,
  $y = -0.03$ to $+0.02$ m and $z = +1.50$ to $+1.54$ m (850 rows).
* **Consequence.** For the same obstacle point, $\hat z$ is measured from the camera, while $\text{long\_near}$ is
  measured from a vehicle origin 1.1 m (KITTI) or 1.7 m (nuScenes) further back. $\hat z$ therefore reads short by
  that offset before any estimation error. On KITTI, $\hat y$ is also offset by about 0.3 m laterally.

### 1.7 Time to collision (TTC)

**Association.** Each current box is matched greedily to a previous-frame box of the same fidelity by IoU $\ge 0.3$,
taking current boxes in order of their best IoU (`mono.py:77-99`). The previous frame's list is the full cached list
at $p \ge 0.10$ (`02_run_detection.py:72-77`; `40_nusc_detect.py:62-67`).

**Scale-change TTC.** With the box-height ratio $s = h_t / h_{t-1}$ (`mono.py:62-74`):

$$\widehat{\text{TTC}} = \min\!\left(\begin{cases} \dfrac{\Delta t}{s - 1} & s > 1.001 \text{ and both heights} > 1\ \text{px} \\ \infty & \text{otherwise}\end{cases},\; 10^3\ \text{s}\right)$$

A box with no match, and every box of a sequence's first frame, gets $10^3$ s.

### 1.8 Clipping and failure handling

* Pixel denominators are floored at 2 px (`mono.py:29,36`), and $\hat z$ is clipped to $[0.5, 200]$ m (`mono.py:128`).
* An empty detection list returns empty arrays (`mono.py:121-124`).
* A box at or above the horizon ($y_2 \le c_y + 5$) is ranged by the height cue alone. No box is rejected.
* Detector boxes are clamped to the image before lifting (`detect.py:166-167`).
* The braking controller ignores objects beyond 80 m (`src/rap/planner.py:34,62`); the trajectory controller does the
  same, and also drops non-finite ranges (`src/rap/planner_b.py:43,85-88`).

---

## 2. Reference-geometry control ("oracle")

### 2.1 Matching rule and what is inherited

For the kept detections of one fidelity and the frame's reference boxes (`decision.py:44-70`):

1. Reference boxes are the evaluable objects of the frame with 2D height $\ge 10$ px (`decision.py:127`;
   `RiskConfig.min_gt_height`, `risk.py:31`).
2. Compute IoU between every kept detection and every reference 2D box. Take $b(k) = \arg\max_g \text{IoU}(k, g)$.
   Detection $k$ is matched if $\max_g \text{IoU}(k, g) \ge 0.5$ (`decision.py:53-55`).
3. **The assignment is not one-to-one.** Two detections can inherit the same reference object. Confidence plays no role.
4. A matched detection **inherits** from $b(k)$: $\text{long\_near}$ as its range, `lat_min` and `lat_max` as its
   lateral extent, and the reference TTC (`decision.py:58,65-69`).
5. An unmatched detection (a false positive, or a box too poorly localised to reach IoU 0.5) **keeps its monocular
   geometry** (`decision.py:50-51,56-57`).

**What does not change.** Which boxes exist (misses and false positives) is identical to the mono run. The control
removes the ranging, lateral and TTC error of matched detections only.

**Where it is used.**
* Braking controller: `decision.py:136`.
* Trajectory controller: `65_planner_b_decision.py:59`, the same function.
* The `noisy_gt` variant multiplies the inherited range by $\operatorname{clip}(1 + \epsilon, 0.2, 3)$ with
  $\epsilon \sim \mathcal N(0, \sigma^2)$, and scales the lateral extent by the same factor (`decision.py:59-64`). It
  is not a benchmark cell.

**nuScenes submissions (PKL/TIP only).** A detection matched the same way (IoU $\ge$ `cfg.iou_thr` = 0.5, argmax)
inherits the reference object's exact translation, size and rotation (`nusc_submission.py:82-105`).

### 2.2 The reference geometry itself

**KITTI** (`src/rap/geometry.py:35-59`).
* Objects are the evaluable classes Car, Van, Truck, Tram, Pedestrian, Person, Person_sitting and Cyclist
  (`kitti.py:27`).
* The 3D box's four bottom-face corners are built from $(h, w, l, r_y)$ and the bottom-face centre, in the rectified
  camera frame (`kitti.py:125-136`), then mapped to the IMU frame (§1.6).
* From the corners: $\text{long\_near} = \min_i x_i$, $\text{lat\_min} = \min_i y_i$, $\text{lat\_max} = \max_i y_i$.

**nuScenes** (`nusc.py:119-189`).
* Annotations of the listed categories with at least one lidar or radar point (`nusc.py:28-30,130-133`).
* The footprint is built from the box **centre** translation, lowered by $h/2$ (`nusc.py:134-145`), and expressed in
  the ego frame.
* The 2D box is the projection of the corners in front of the camera ($z_\text{cam} > 0.5$ m, at least 4 of 8),
  clipped to the $1600 \times 900$ image. Boxes under 4 px are dropped (`nusc.py:146-165`).

**Range rate and TTC**, both datasets (`geometry.py:62-96`).
* A local linear fit of $\text{long\_near}$ against time over temporally contiguous neighbours of the same track:
  $\pm 2$ frames on KITTI, $\pm 1$ keyframe on nuScenes (`nusc.py:188`).
* With slope $\dot d$:

$$\text{TTC} = \begin{cases} \text{long\_near} / (-\dot d) & -\dot d > 10^{-3}\ \text{m/s} \\ \infty & \text{otherwise}\end{cases}$$

---

## 3. Braking controller loss $J$ ($q_\text{brake}$)

`src/rap/planner.py`. The same code runs on CHEAP geometry, FULL geometry and reference geometry
(`decision.py:137,141-142`). The reference geometry enters only through the required deceleration $a^\star$ the
action is scored against.

### 3.1 Constants

**`PlannerParams`** (`planner.py:26-36`):

| name | value | unit |
|---|---|---|
| corridor half-width $w_c$ | 1.2 | m |
| standoff $s_0$ | 2.0 | m |
| reaction time $t_r$ | 0.4 | s |
| DECELERATE command $a_d$ | 2.5 | m/s² |
| HARD_BRAKE command $a_h$ | 6.0 | m/s² |
| DECELERATE threshold $\tau_d$ | 1.0 | m/s² |
| HARD_BRAKE threshold $\tau_h$ | 3.5 | m/s² |
| maximum range | 80 | m |
| deceleration cap $a_\text{cap}$ | 9.0 | m/s² |
| use TTC | true | — |

**`CostParams`** (`planner.py:40-45`):

| name | value | unit |
|---|---|---|
| $\lambda_\text{risk}$ | 1.0 | (m/s²)⁻² |
| $\lambda_\text{brake}$ | 0.12 | (m/s²)⁻² |
| $\lambda_\text{jerk}$ | 0.02 | (m/s²)⁻¹ |
| collision shortfall $a_\text{coll}$ | 8.0 | m/s² |
| $\lambda_\text{coll}$ | 6.0 | — |

### 3.2 Corridor and required deceleration

**Corridor.** An object $i$ is in the corridor when (`planner.py:48-49,62`):

$$y_{\min,i} < w_c \;\wedge\; y_{\max,i} > -w_c \;\wedge\; d_i \le 80\ \text{m}$$

If no object qualifies, or $v \le 0.1$ m/s, then $a_\text{req} = 0$ (`planner.py:64-65`).

**Required deceleration** (`planner.py:67-80`). For each in-corridor object with range $d_i$, TTC $T_i$ and ego speed
$v$:

$$a_{\text{stop},i} = \frac{v^2}{2\max(d_i - s_0 - v\,t_r,\; 0.05)}$$

$$v_{\text{rel},i} = \operatorname{clip}\!\left(\frac{d_i}{\operatorname{clip}(T_i, 0.1, 10^3)},\,0,\,60\right),\qquad
a_{\text{close},i} = \frac{v_{\text{rel},i}^2}{2\max(d_i - s_0,\; 0.05)}$$

$$a_\text{req} = \max_i \operatorname{clip}\big(\max(a_{\text{stop},i}, a_{\text{close},i}),\, 0,\, a_\text{cap}\big)$$

### 3.3 Braking levels and thresholds

The action and its commanded deceleration are (`planner.py:83-92`):

$$\text{action} = \begin{cases} \text{HARD\_BRAKE} & a_\text{req} \ge 3.5 \\ \text{DECELERATE} & 1.0 \le a_\text{req} < 3.5 \\ \text{KEEP} & \text{otherwise}\end{cases},\qquad
a_\text{cmd} \in \{0,\; 2.5,\; 6.0\}\ \text{m/s}^2$$

### 3.4 Loss

Let $a^\star$ be $a_\text{req}$ computed on the reference geometry of the frame (`decision.py:141-142`), and let
$a_\text{cmd}^{-}$ be the previous frame's command of the **same fidelity** in the same sequence (0 terms at the first
frame; `decision.py:119,145-147`). Then (`planner.py:95-113`):

$$\text{sf} = \max(0, a^\star - a_\text{cmd}),\quad \text{ex} = \max(0, a_\text{cmd} - a^\star),\quad
c = \mathbb 1[\text{sf} \ge 8.0],\quad j = |a_\text{cmd} - a_\text{cmd}^{-}|$$

$$J = 1.0\,\text{sf}^2 + 0.12\,\text{ex}^2 + 0.02\,j + 6.0\,c,\qquad V = J_\text{CHEAP} - J_\text{FULL}\quad(\texttt{decision.py:161})$$

---

## 4. Receding-horizon trajectory controller $J_B$ ($q_\text{traj}$, "Planner B")

`src/rap/planner_b.py`, driven per frame by `scripts/65_planner_b_decision.py:32-79`.

* **Configuration.** The benchmark's Planner B run (`20260912_111225_planner_b_static_fixed`) uses
  `params = static_obstacles` and `costs = default` (its `config.json`; preset at `planner_b.py:206`). Scripts that
  recompute it name the preset as a constant.
* **What it reads.** `plan` sees only the perceived $(z, y_\text{min}, y_\text{max}, T)$ and $v$.
* **Where the reference enters.** Only in `executed_cost`, which re-simulates the chosen plan against the reference
  geometry of the frame (`65_planner_b_decision.py:50-62`).

### 4.1 Candidate set and rollout

**Candidates.** $35 = 7 \times 5$ plans (`planner_b.py:31-32`):

$$a \in \{0, -1, -2, -3, -4.5, -6, -8\}\ \text{m/s}^2 \quad\times\quad \delta \in \{-3, -1.5, 0, 1.5, 3\}\ \text{m}$$

**Rollout constants** (`PlannerBParams`, `planner_b.py:36-44`):

| name | value | unit |
|---|---|---|
| horizon $H$ | 3.0 | s |
| step $\Delta$ | 0.25 | s (12 steps, $t_k = 0.25k$) |
| lateral lag $\tau$ | 1.0 | s |
| ego half-width $e$ | 0.9 | m |
| front margin $m_f$ | 1.0 | m |
| progress speed floor | 2.0 | m/s |
| maximum range | 80 | m |
| obstacles move | **false** (`static_obstacles`) | — |

**Ego motion** per candidate (`planner_b.py:61-72`):

$$t_e = \min\big(t_k, \tfrac{v}{-a}\big)\ (a<0),\qquad s(t_k) = v\,t_e + \tfrac12 a\,t_e^2,\qquad y(t_k) = \delta\,(1 - e^{-t_k/\tau})$$

**Obstacles.** Those with $z > 80$ m or a non-finite $z$ are dropped (`planner_b.py:85-88`).
* Under `static_obstacles` they are stationary in the world: $z_o(t_k) = z$ (`planner_b.py:101-106`).
* The default preset would move them at $v_o = \max(v - z/T, 0)$ (`planner_b.py:96-100`).

### 4.2 Collision and clearance

For obstacle $o$ at step $k$ (`planner_b.py:107-121`):

$$\text{reach} = z_o(t_k) - s(t_k) \le m_f,\qquad
g_\text{lat} = \min\big(y_{\max,o} - (y(t_k) - e),\; (y(t_k) + e) - y_{\min,o}\big)$$

* **Collision at step $k$:** $\text{reach} \wedge g_\text{lat} > 0$ for any obstacle.
* **Clearance:** $\text{cl} = \min_{k,o:\,\text{reach}} (-g_\text{lat})$, or $+\infty$ if the plan never reaches an
  obstacle.

### 4.3 Cost terms and weights

**`CostBParams`** (`planner_b.py:48-58`): $w_\text{coll} = 10$, discount $\gamma = 0.9$ per step, $w_\text{cl} = 1.5$,
$c_\text{req} = 1.0$ m, $w_\text{prog} = 1.0$, $w_a = 0.01$, $w_\text{lat} = 0.50$, $w_\text{sw} = 0.05$.

Shared terms of candidate $(a, \delta)$, with $\text{cl}' = \text{cl}$ if finite and $c_\text{req}$ otherwise,
$v_\text{ref} = \max(v, 2)$, and the previous frame's plan $(a^-, \delta^-)$ of the same fidelity:

$$\text{short} = \operatorname{clip}(c_\text{req} - \text{cl}',\, 0,\, 2c_\text{req}),\qquad
P = \left(\frac{v_\text{ref}H - s(H)}{v_\text{ref}H}\right)^2$$

$$S = \frac{(a - a^-)^2}{4} + (\delta - \delta^-)^2\quad(\text{0 at the first frame})$$

**Planner objective**, minimised on perceived geometry (`planner_b.py:124-158`):

$$J^\text{plan} = 10\,\max_k \big(\mathbb 1[\text{coll}_k]\,0.9^{\,k-1}\big) + 1.5\,\text{short}^2 + 1.0\,P + 0.01\,a^2 H + 0.50\,\delta^2 + 0.05\,S$$

**Executed cost against the reference geometry** (`planner_b.py:161-192`). The same terms, except that the collision
term is undiscounted occurrence:

$$J_B = 10\,\mathbb 1\big[\exists k: \text{coll}_k\big] + 1.5\,\text{short}^2 + 1.0\,P + 0.01\,a^2 H + 0.50\,\delta^2 + 0.05\,S,
\qquad V_B = J_{B,\text{CHEAP}} - J_{B,\text{FULL}}$$

---

## 5. Perception-loss gains E1–E6 and cheap-side criticality

### 5.1 Per-frame primitives

`src/rap/percep_metrics.py:17-47`, per fidelity at its operating threshold (`scripts/50_percep_metrics.py:61-62`).
Reference objects are those of §2.2 with height $\ge 10$ px. Classes are coarse (`kitti.py:31-35`).

**Matching** (`risk.py:48-72`). Greedy in descending detection confidence: each detection takes the highest-IoU
reference object not yet taken, if that IoU is $\ge 0.5$. Matching is class-agnostic (`class_aware = False`).
An untaken reference object records its best IoU with **any** detection.

**Primitives**, with best IoU $u_g$ per reference object:

| primitive | definition | code |
|---|---|---|
| hit | $\text{hit}_g = \mathbb 1[u_g \ge 0.5]$ | `percep_metrics.py:25` |
| FN | $\sum_g (1 - \text{hit}_g)$ | `:26` |
| FP | number of kept detections not assigned by the greedy pass; no DontCare exclusion in this path | `:28-31` |
| loc | $\sum_{g:\,\text{hit}_g} (1 - u_g)$ | `:34` |
| cls | $\sum_{g:\,\text{hit}_g} \mathbb 1[\kappa(\arg\max_k \text{IoU}(k,g)) \ne \kappa_g]$ | `:36-41` |
| crit_fn | $\sum_{g:\,\neg\text{hit}_g} c_g$, with $c_g$ the composite criticality of §5.3 on reference geometry | `:43` |

**Class error on nuScenes was not what its definition says, and has been corrected.** Reference classes came from
`TYPE_TO_COARSE.get(type, "vehicle")`, a map that knows KITTI's type names only (`kitti.py:31-35`). On nuScenes the
reference `type` is the category prefix, `vehicle` or `human`, so every nuScenes reference object was labelled
`vehicle` and a correctly detected pedestrian, bicycle or motorcycle counted as a class error.
* It affected `cls`, and with it E3 and the three E5 variants on nuScenes: `dE_E5_combined`, the benchmark's combined
  gain and Task 9's primary perception-gain label on its nuScenes cells.
* It did not affect E1, E2, E4, E6, the exact FN count, matching (class-agnostic), or any KITTI or nuPlan quantity.
* Found while writing this document; **fixed in Task 22 Part A (2026-09-20)**. The geometry now carries a `coarse`
  field, filled from the full nuScenes category (`human.*` → person, `vehicle.bicycle`/`vehicle.motorcycle` →
  cyclist), and `rap.geometry.coarse_classes` reads it. Every official output that uses E3 or E5 on nuScenes was
  re-run against corrected tables; what moved, and by how much, is in `docs/iclr_class_error_fix.md`.

### 5.2 Named losses and gains

Each loss is a weighted sum of primitives (`percep_metrics.py:57-66`). The gain is
$\Delta E_k = E_k(\text{CHEAP}) - E_k(\text{FULL})$, positive when FULL is better (`50_percep_metrics.py:66-67`).

| name | $E$ |
|---|---|
| E1 `fn_only` | $\text{FN}$ |
| E2 `fn_fp` | $\text{FN} + \text{FP}$ |
| E3 `class_aware` | $\text{FN} + \text{cls}$ |
| E4 `localization` | $\text{FN} + \text{loc}$ |
| **E5 `combined`** | $\text{FN} + 0.5\,\text{FP} + \text{loc} + 0.5\,\text{cls}$ |
| E5 `fp_heavy` | $\text{FN} + 2\,\text{FP} + \text{loc} + 0.5\,\text{cls}$ |
| E5 `loc_heavy` | $\text{FN} + 0.5\,\text{FP} + 3\,\text{loc} + 0.5\,\text{cls}$ |
| **E6 `risk_weighted`** | $\text{crit\_fn} = \sum_{g \text{ missed}} c_g$ |

**The benchmark's `dE_exact`** is the column `dE`: the class-agnostic FN count difference from
`decision.add_perception_gain`. There, every reference object is labelled one class, and a reference object counts as
missed when its best IoU is below 0.5 (`decision.py:180-205`).

**nuPlan diagnostics**, computed on the logged tracks of the real-perception branches
(`scripts/119_nuplan_real_features.py:112-124`):
* $\Delta E_1$ = (in-camera tracks CHEAP removed) − (tracks FULL removed);
* $\Delta E_6 = \sum_o c_o\,(\mathbb 1[o \text{ removed by CHEAP}] - \mathbb 1[o \text{ removed by FULL}])$, with $c_o$ the
  composite model of §5.3 on the logged track's geometry, its lateral extent taken as centre ± width/2
  (`scripts/91_nuplan_cheap_features.py:77`).

### 5.3 The risk weighting: composite criticality

`CriticalityModel("composite")`, `geometry.py:101-163`, applied to $(d, y_\text{min}, y_\text{max}, T)$. Defaults:

| name | value | unit |
|---|---|---|
| corridor half-width $w$ | 0.9 | m |
| lateral softness $\sigma$ | 1.5 | m |
| free range $r_0$ | 5.0 | m |
| decay $r_\tau$ | 20.0 | m |
| hot TTC | 1.5 | s |
| cold TTC | 7.0 | s |
| TTC weight | 0.5 | — |
| maximum range | 80 | m |

$$\text{corr} = \exp\!\left(-\tfrac12\left(\frac{\max\big(0,\,\max(-y_\text{max}, y_\text{min}) - 0.9\big)}{1.5}\right)^2\right),\qquad
\text{prox} = \exp\!\left(-\frac{\max(0, d - 5)}{20}\right)$$

$$\text{ttc} = \operatorname{clip}\!\left(\frac{7 - \max(T, 0)}{7 - 1.5},\, 0,\, 1\right),\qquad
c = \operatorname{clip}\big(\text{corr}\,(0.5\,\text{prox} + 0.5\,\text{ttc}),\,0,\,1\big),\quad c = 0 \text{ if } d > 80 \text{ or } d < -2$$

### 5.4 Cheap-side criticality and the uncertainty signal (deployable)

**Cheap-side criticality** (`criticality_cheap`, column `feat_crit_sum`). This is the same composite model applied to
the **monocular** geometry of the CHEAP detections with $p \ge 0.25$ (`src/rap/features.py:167-188`, called from
`features.frame_features`, `features.py:214-221`):

$$\text{crit}_\text{cheap} = \sum_{k:\,p_k \ge 0.25} c\big(\hat z_k, \hat y_{\min,k}, \hat y_{\max,k}, \widehat{\text{TTC}}_k\big)$$

It is registered as `cheap_det` provenance. The GT counterpart `crit_sum` sums $c_g$ over reference objects
(`decision.py:115,126-128,172`), is a diagnostic, and is never a gate input.

**Uncertainty signal** (`uncertainty`, column `unc_sum`). The sum of binary entropies of the kept CHEAP confidences
(`decision.py:140,173`; `detect.py:177-178`):

$$\text{unc} = \sum_{k:\,p_k \ge 0.25} -\big[p_k \ln p_k + (1-p_k)\ln(1-p_k)\big],\qquad p_k \in [10^{-6}, 1 - 10^{-6}]$$

---

## 6. Ego-speed signal source per dataset

The ego speed is the $v$ of §3 and §4, and the `trivial_ego_speed` baseline (`scripts/125_statistics_hardening.py:58,60`).

| dataset | source | code |
|---|---|---|
| KITTI | OXTS column 8, `vf`: forward velocity in m/s at the frame's index; frames past the last OXTS row take the last value | `kitti.py:109-112`; `decision.py:39-41,124` |
| nuScenes | pose-differenced speed of the CAM_FRONT keyframes: $v_i = \lVert \mathbf p_{i+1} - \mathbf p_{i-1}\rVert / (t_{i+1} - t_{i-1})$, one-sided at a scene's ends. $\mathbf p$ is the 3D ego-pose translation of the keyframe's `ego_pose`, and $t$ the sample timestamp. It is a centred difference over ±0.5 s and unsigned. | `nusc.py:100-116`; `make_adapter`, `nusc.py:235-239` |
| nuPlan | `ego.dynamic_car_state.rear_axle_velocity_2d.x`: the longitudinal rear-axle velocity of the logged ego state at the decision iteration, in m/s | `scripts/91_nuplan_cheap_features.py:69`; `119_nuplan_real_features.py:65` |

On nuScenes the speed at keyframe $i$ uses the pose half a second later. Every benchmark decision value and baseline
uses it as above. Planner D's separate ego-velocity input (`src/rap/ego_traj.py`) was changed from this central difference to a backward
difference (`scripts/70b_fix_ego_velocity.py:1-8`); the benchmark's controllers and baselines read `nusc.py`'s speed,
not that input.
