# Loss definitions (Task 32)

Every downstream loss of the benchmark as the code computes it: the braking controller, the trajectory controller
(Planner B), the lateral controller, and the nuPlan safety and scalar losses. Each term, weight and threshold is given
with the place it is defined. The decision value of escalating input i is `V_i = L(pi(z_i^cheap), s_i) -
L(pi(z_i^full), s_i)`, with `s_i` the true scene. Ground truth enters only in L, never in pi.

**Action history (Task 32).** Where a loss has an action-change term, both branches are charged against the action the
all-CHEAP run took at the previous frame of the same sequence. There is none at a sequence's first frame, and then the
term is 0. The CHEAP loss is therefore unchanged by Task 32. The trajectory controller also plans with its previous
plan, so its FULL plan is re-planned against the CHEAP branch's previous plan.

**Reference objects.** In every KITTI and nuScenes loss the true scene is the annotated objects at least 10 px tall
(`RiskConfig.min_gt_height = 10.0`, `src/rap/risk.py`). Each has `long_near` (range to its nearest point, m), `lat_min`
and `lat_max` (lateral extent, m, ego frame) and `ttc` (time to collision, s). The ego speed `v` comes from the vehicle
(`adapter.speeds`), not the camera.

## 1. Braking controller (q_brake)

`src/rap/planner.py`: `required_decel`, `discrete_action`, `decision_cost`. Parameters are `PlannerParams()` and
`CostParams()` at their defaults; the other entries of `PLANNERS` and `COSTS` are sensitivity variants only.

**Controller.** It runs identically on CHEAP, FULL and reference geometry.

Required deceleration for each object in the corridor, with `|lateral overlap| < corridor_half_w` and range at most
`max_range`:
- `a_stop = v² / (2 · max(long_near − standoff − v · reaction_t, 0.05))`;
- with `use_ttc`, also `a_close = v_rel² / (2 · max(long_near − standoff, 0.05))`, where
  `v_rel = clip(long_near / clip(ttc, 0.1, 1000), 0, 60)`;
- the per-object value is `clip(max(a_stop, a_close), 0, a_cap)`, and `a_req` is the maximum over objects. It is 0 when
  no object is in the corridor or `v ≤ 0.1`.

The action is `HARD_BRAKE` if `a_req ≥ thr_hard`, `DECELERATE` if `a_req ≥ thr_decel`, else `KEEP`. The commanded
deceleration `a_cmd` is 0, `a_decel` or `a_hard` respectively. The action does not depend on the history.

| parameter | value | meaning |
|---|---|---|
| `corridor_half_w` | 1.2 m | ego half-width plus margin |
| `standoff` | 2.0 m | distance kept when stopped |
| `reaction_t` | 0.4 s | actuation delay |
| `a_decel`, `a_hard` | 2.5, 6.0 m/s² | commanded decelerations |
| `thr_decel`, `thr_hard` | 1.0, 3.5 m/s² | action thresholds on `a_req` |
| `max_range` | 80 m | |
| `a_cap` | 9.0 m/s² | clip on `a_req` |
| `use_ttc` | True | closing speed considered |

**Loss.** `a_gt = a_req` on the reference objects.

`J = lam_risk · shortfall² + lam_brake · excess² + lam_jerk · jerk + lam_collision · collision`

| term | definition | weight |
|---|---|---|
| safety shortfall | `max(0, a_gt − a_cmd)`, squared | `lam_risk = 1.0` |
| excess braking | `max(0, a_cmd − a_gt)`, squared | `lam_brake = 0.12` |
| jerk | `|a_cmd − a_cmd(prev)|`; 0 with no previous action | `lam_jerk = 0.02` |
| collision | `1[shortfall ≥ collision_a − 1e-9]`, with `collision_a = 8.0` m/s² | `lam_collision = 6.0` |

Per-frame weighted terms: `Jterm_<cheap|full>_<shortfall|excess|jerk|collision>` (`decision.build`). They sum to `J`
up to floating-point order.

## 2. Trajectory controller (q_traj, KITTI only)

`src/rap/planner_b.py`: `plan`, `executed_cost`. Parameters are `PARAMS_B["static_obstacles"]` (the defaults with
`obstacle_closes = False`) and `COSTS_B["default"]`; the benchmark reads the `static_obstacles` tables.

**Controller.** 35 candidates: longitudinal acceleration `A_LON ∈ {0, −1, −2, −3, −4.5, −6, −8}` m/s² × lateral target
`D_LAT ∈ {−3, −1.5, 0, 1.5, 3}` m.

Each candidate is rolled out over `horizon = 3.0` s in steps of `dt = 0.25` s:
- longitudinal motion at constant deceleration until stop;
- lateral offset `D_LAT · (1 − exp(−t / tau_lat))`, with `tau_lat = 1.0` s.

Obstacles are stationary in the world (`static_obstacles`).

A step collides when two conditions hold:
- the ego has closed the gap to within `front_margin = 1.0` m;
- the lateral extents overlap, with ego half-width `ego_half_w = 0.9` m.

Clearance is the worst lateral gap at steps where the ego reaches the obstacle longitudinally. Objects beyond
`max_range = 80` m are ignored.

`plan` returns the candidate minimising, on perceived geometry:
- `w_collision · discount^(first colliding step)`, with `discount = 0.9`;
- plus the clearance, progress, acceleration, lateral and switching terms below (same form as the executed cost).

The plan depends on the previous plan through the switching term.

**Loss.** `executed_cost` re-simulates the chosen plan against the reference objects.

| term | definition | weight |
|---|---|---|
| collision | `1[any step overlaps a reference object]` (occurrence, not discounted) | `w_collision = 10.0` |
| clearance | `clip(clear_req − min clearance, 0, 2 · clear_req)²`, with `clear_req = 1.0` m; infinite clearance counts as `clear_req` | `w_clearance = 1.5` |
| progress | `((v_ref · horizon − s_end) / (v_ref · horizon))²`, with `v_ref = max(v, v_ref_floor = 2.0 m/s)` | `w_progress = 1.0` |
| acceleration | `A_LON² · horizon` | `w_accel = 0.01` |
| lateral | `D_LAT²` | `w_lateral = 0.50` |
| switching | `(A_LON − A_LON(prev))² / 4 + (D_LAT − D_LAT(prev))²`; 0 with no previous plan | `w_switch = 0.05` |

Per-frame weighted terms: `JBterm_<cheap|full>_<collision|clearance|progress|acceleration|lateral|switching>`
(`65.build_b`). They sum to `JB` exactly, in this order.

## 3. Lateral controller (not in the benchmark)

`src/rap/planner.py`: `lateral_action`, `lateral_cost`. Parameters are `LateralParams()` and `LateralCostParams()`.

**Controller.** Below `v_floor = 1.0` m/s it keeps the lane.

The required clearance is `need = min(v · t_headway + standoff, lookahead)`, with `t_headway = 2.5` s,
`standoff = 2.0` m and `lookahead = 45` m.

Clearance of a corridor centred on an offset is the range to the nearest object overlapping it (half-width
`half_w = 1.6` m, range in (−2, `lookahead`]).

Decision:
- keep the lane if its clearance is at least `need`;
- otherwise take the better side corridor (offset ±`offset`, with `offset = 3.0` m) if its clearance is at least `need`
  or it buys at least `min_gain = 3.0` m;
- otherwise brake.

The action does not depend on the history.

**Loss.** `J = lam_collision · collision + lam_shortfall · shortfall² + lam_deviation · deviation + lam_brake · brake +
lam_switch · switch`

| term | definition | weight |
|---|---|---|
| collision | braking: `1[true clearance of the lane < standoff and v > v_floor]`; otherwise `1[true clearance of the chosen corridor < standoff]` | `lam_collision = 6.0` |
| shortfall | `max(0, need − true clearance) / need`; 0 when braking | `lam_shortfall = 2.0` |
| deviation | `|offset|`, in m | `lam_deviation = 0.30` |
| brake | `1[action = BRAKE]` | `lam_brake = 0.80` |
| switch | `1[action ≠ prev]`; 0 with no previous action | `lam_switch = 0.10` |

## 4. nuPlan losses (safety and scalar)

The trajectories come from PDM-Closed and IDM:
- IDM at its released configuration: `target_velocity = 10`, `min_gap_to_lead_agent = 1.0`, `headway_time = 1.5`,
  `accel_max = 1.0`, `decel_max = 3.0`, 16 samples at 0.5 s (`82.make_planner`).
- Each planner is created fresh for every state and branch (`115.run_planner`).

Each trajectory is scored against the logged tracked objects by `82.score`, at `HORIZON_S = 0.5, 1.0, …, 4.0` s:
- collision is `1[the ego footprint intersects any logged object box at any horizon]`;
- min clearance is the smallest footprint-to-box distance (0 on intersection; 20 m when nothing is present);
- log deviation (mean) is the mean distance between the planned and the logged rear axle over the horizons.

The losses are in `83.costs`, used unchanged by 116 and 120:

| loss | definition |
|---|---|
| clearance shortfall | `clip(1.0 − min clearance, 0, 2)²` |
| **safety** | `10 · collision + 1.5 · clearance shortfall` |
| **scalar** (`scalar_J`) | `10 · collision + 1.5 · clearance shortfall + 0.1 · log deviation (mean)` |

The nuPlan losses and the learned planner's (q_plan, PKL's planner) carry no action history: neither planner reads a
past observation or a past action (`docs/iclr_shared_history.md`).
