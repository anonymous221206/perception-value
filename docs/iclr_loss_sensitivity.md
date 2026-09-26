# Sensitivity of the sign variation to the downstream losses (Task 35)

> **Post hoc.** The weights and the corridor width were varied after the results were seen. Nothing here was
> pre-registered. The intervals are not corrected for the number of variants.

Script `scripts/167_loss_sensitivity.py`; the table is `results/final/loss_sensitivity.csv` (one row per part,
setting and variant). Every decision value uses the shared action history (`docs/iclr_shared_history.md`). No
detector is re-run. The losses and their default weights are in `docs/iclr_loss_definitions.md`.

**Quantities.** For each variant, over all units of the setting:
- affected inputs, |V| > 1e-9;
- the harmed share of affected inputs;
- the harm ratio ρ = Σ max(−V, 0) / Σ max(V, 0), with its 95% unit-bootstrap interval (1,000 draws, the calibration
  cells' `harm_stats`, a generator seeded per setting and variant);
- the all-full and oracle@20 reductions of the all-cheap loss.

**Settings.**
- Braking controller: the braking settings of the sign table, i.e. nuScenes YOLOv8s 320→640 under oracle and
  monocular geometry, every KITTI fidelity pair under monocular geometry, and KITTI YOLOv8s 320→640 under oracle
  geometry.
- Trajectory controller: the KITTI settings.
- "Headline" settings are the nuScenes ones and the moderate-gap KITTI pairs: YOLOv8s 384→640 and 512→640, and
  RT-DETR-l 480→640, all monocular.

## A. Braking controller: loss weights (exact recombination)

The braking action is a threshold rule on the perceived requirement, so it does not read the loss weights. Every
variant is therefore an exact recombination of the weighted terms the core tables store (`Jterm_<branch>_{shortfall,
excess, jerk, collision}`). One weight is varied at a time, and the safety-shortfall weight stays 1:
- over-braking (excess): 0.06, **0.12**, 0.24;
- collision penalty: 3, **6**, 12;
- change of command (jerk): 0, **0.02**, 0.04.

**Checks (241, all pass).**
- At default weights the terms add up to the tables' losses bit for bit.
- The default variant's affected count, harmed share, ρ, all-full and oracle@20 equal
  `shared_history_effect.csv` (shared, S0) and `calibration_cells.csv` (S0, all units) to 1e-12.
- The bootstrap intervals use their own seeds, so they are not the calibration cells' draws.


Every variant, with its interval: `docs/loss_sensitivity_tables.md`, section A.

## B. Trajectory controller: plan and loss weights together (rebuilt)

The receding-horizon controller plans with its cost weights, so reweighting only the evaluated loss would score
plans the controller would not have chosen. Every variant rebuilds the trajectory tables with `65.build_b`
(`static_obstacles` preset, shared history), with the planning and evaluation weights changed together, one at a
time:
- collision: 5, **10**, 20;
- clearance: 0.75, **1.5**, 3;
- progress: 0.5, **1**, 2.

**Checks (171, all pass).**
- The default rebuild equals the shipped Planner B tables bit for bit (JB_cheap and JB_full, all frames).
- Its point values equal `shared_history_effect.csv` and `calibration_cells.csv`.


Every variant: `docs/loss_sensitivity_tables.md`, section B.

At collision weights 5, 10 and 20 the affected and harmed counts are the same in every setting. Only ρ moves.

## C. Braking controller: corridor half-width (rebuilt)

The corridor half-width decides which objects the controller considers, so it changes actions. The decision tables
are rebuilt with `decision.build` at 1.0 and 1.5 m (default 1.2 m).

**Checks (83, all pass).**
- The default rebuild equals the core tables bit for bit.
- Its point values equal `shared_history_effect.csv` and `calibration_cells.csv`.


Every variant: `docs/loss_sensitivity_tables.md`, section C.

## Ranges across variants, per setting

Headline = nuScenes, or a moderate-gap KITTI pair. Generated in `docs/loss_sensitivity_tables.md`.

| part | setting | headline | harmed share | rho |
|---|---|---|---|---|
| A | RT 320->640|mono | no | 43.4%–44.3% | 0.611–0.789 |
| A | RT 480->640|mono | yes | 40.7%–43.2% | 0.684–0.786 |
| A | Y8 320->640|mono | no | 32.6%–35.3% | 0.107–0.282 |
| A | Y8 320->640|oracle | no | 23.2%–25.0% | 0.043–0.154 |
| A | Y8 384->640|mono | yes | 38.8%–40.1% | 0.351–0.648 |
| A | Y8 512->640|mono | yes | 42.9%–43.4% | 0.514–0.723 |
| A | nuSc Y8 320->640|mono | yes | 38.1%–42.5% | 0.284–0.555 |
| A | nuSc Y8 320->640|oracle | yes | 35.5%–36.2% | 0.276–0.621 |
| B | RT 320->640|mono | no | 47.1%–49.7% | 0.737–0.789 |
| B | RT 480->640|mono | yes | 42.9%–45.2% | 0.571–0.602 |
| B | Y8 320->640|mono | no | 46.0%–48.8% | 0.270–0.318 |
| B | Y8 320->640|oracle | no | 2.7%–3.7% | 0.003–0.004 |
| B | Y8 384->640|mono | yes | 47.3%–51.0% | 0.448–0.514 |
| B | Y8 512->640|mono | yes | 51.6%–54.5% | 1.099–1.148 |
| C | RT 320->640|mono | no | 44.0%–44.8% | 0.670–0.808 |
| C | RT 480->640|mono | yes | 42.2%–43.1% | 0.720–0.862 |
| C | Y8 320->640|mono | no | 32.7%–34.6% | 0.154–0.215 |
| C | Y8 320->640|oracle | no | 22.2%–26.3% | 0.070–0.097 |
| C | Y8 384->640|mono | yes | 38.9%–39.8% | 0.411–0.507 |
| C | Y8 512->640|mono | yes | 42.6%–42.9% | 0.594–0.714 |
| C | nuSc Y8 320->640|mono | yes | 39.2%–41.6% | 0.372–0.412 |
| C | nuSc Y8 320->640|oracle | yes | 35.5%–36.5% | 0.331–0.458 |

## Reading

- **The harmed share stays above 30% in every headline setting under every variant.** This covers nuScenes under
  both geometries and the three moderate-gap KITTI pairs, in all three parts.
  - Braking controller: the lowest is 35.5% (nuScenes, oracle geometry, at several variants).
  - Trajectory controller: the lowest is 42.9% (RT-DETR-l 480→640, progress weight 0.5).
- **It is below 30% only under oracle geometry on KITTI YOLOv8s 320→640, which is not a headline setting.** It is
  below 30% there at every variant, the default included:
  - braking controller: 22.2–26.3%;
  - trajectory controller: 2.7–3.7%.
  KITTI YOLOv8s 320→640 under monocular geometry stays above 30% for both controllers: braking 32.6–35.3%,
  trajectory 46.0–48.8%.
- **The harm ratio moves much more than the harmed share, and in a predictable direction.**
  - On the braking controller, harm is mostly over-braking (`docs/iclr_shared_history.md`, part b). Halving the
    over-braking weight lowers ρ, and doubling it raises ρ: nuScenes oracle 0.276 → 0.418 → 0.621, KITTI Y8 384→640
    mono 0.351 → 0.473 → 0.648.
  - The collision penalty moves ρ by at most 0.006, and the change-of-command weight by at most 0.016.
  - A wider corridor raises ρ on KITTI (Y8 384→640 mono 0.411 → 0.473 → 0.507) and lowers it on nuScenes oracle
    (0.458 → 0.418 → 0.331).
- **The trajectory controller is the least sensitive.** Across the nine variants its harmed share moves by at most
  3.7 points and ρ by at most 0.07 in every setting.

## D. The Figure 1 frames

Figure 1 shows two nuScenes frames under the braking controller and monocular geometry:
- scene-0055 frame 3 (sample `ee535106a1c546d2b99f9eeacc41066e`), harmed;
- scene-0065 frame 24 (sample `81458d7c63ee40afba6d143188cebaa2`), helped.

The gallery's selection rule (the six most negative and six most positive V, at most one frame per scene) picks
scene-0065 frame 23 on the shared action history, so frame 24 was no longer exported.

`scripts/118_figure_exports.py` now also exports these two frames by name, with the gallery's JSON and image format:
- `results/final/fig_gallery/figure1_scene-0055_03.{json,jpg}`
- `results/final/fig_gallery/figure1_scene-0065_24.{json,jpg}`
- an index, `figure1.json`

The gallery's own selection and files are unchanged (the rerun leaves every `negative_*`, `positive_*` and
`index.json` identical). On the shared history both frames keep their roles:

| frame | V | actions (CHEAP / FULL / reference) |
|---|---|---|
| scene-0055, 3 (harmed) | −4.440 | KEEP / HARD_BRAKE / KEEP |
| scene-0065, 24 (helped) | +5.346 | KEEP / DECELERATE / DECELERATE |

scene-0065 frame 23, the gallery's positive_4, has V = +5.347.
