#!/usr/bin/env python
"""Follow-up round 2, R2: closed-loop dynamics under replayed perception on the nuPlan real-perception track
(addendum R2 in docs/followup_protocol.md). Run with scripts/pynuplan (or the training machine's nuPlan Python).

Detector outcomes are replayed by logged time: at each simulation step the observation is built from the logged frame
at that time (the nearest fetched CAM_F0 image within 50 ms, its cheap or full detections matched to the logged tracks
as 114 matches them), not from what a camera would see at the simulated pose. R2 measures closed-loop dynamics under
replayed perception, not closed-loop sensing.

  --stage spec      the branch observations of 114 (cheap, full: eligible in-camera tracks the mode did not detect
                    removed, its unmatched detections added as false positives fixed in world coordinates at the logged
                    pose, tracks outside the image as logged) at every iteration of the 60 scenarios; equal to 114's
                    specification wherever both exist (checked) -> data/cache/nuplan_real/r2_spec.pkl, r2_frames.csv
  --stage vall      open-loop single-step V at every logged frame (the 0.1 s grid from iteration 4, an image within
                    50 ms), the track's way (115): fresh planner per branch, 4-iteration history through the branch
                    filter, 82's score against the true objects, 83's safety loss -> v_all_frames.csv (shardable)
  --stage gate      gate R2-0: (a) two identical cheap-only forks, dJ = 0 on 20 frames; (b) on 100 track states, the
                    first plan of a fork whose first step uses the full mode equals the open-loop full-branch plan
                    within 1e-6 m -> gate_r2_0.json
  --stage forks     at every frame with V_t != 0: branch full (full at the first step, cheap after) and branch cheap,
                    8 s (80 steps) -> rows in data/cache/nuplan_real/r2_forks/ (shardable)
  --stage policies  whole scenarios: all-cheap, all-full, privileged (full where V_t > 0), matched random (20 seeds)
                    -> rows in data/cache/nuplan_real/r2_policies/ (shardable)
  --stage analysis  forks.csv, policies.csv, policy_runs.csv
"""
from __future__ import annotations

import argparse, importlib.util, json, pickle, sqlite3, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "nuplan_devkit"))
sys.path.insert(0, str(ROOT / "third_party" / "tuplan_garage"))


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


t114 = _load("t114", "114_nuplan_project_match.py")
t115 = _load("t115", "115_nuplan_real_counterfactual.py")
t119 = _load("t119", "119_nuplan_real_features.py")
cf82 = t115.cf82

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters        # noqa: E402
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel  # noqa: E402
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker                # noqa: E402
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController  # noqa: E402
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer  # noqa: E402
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks      # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput  # noqa: E402
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration  # noqa: E402

from rap.paths import CACHE, NUPLAN_MAPS, NUPLAN_ROOT, RESULTS                            # noqa: E402

OUT = Path(RESULTS) / "followup" / "R2"
OUTC = Path(CACHE) / "nuplan_real"
SPEC = OUTC / "r2_spec.pkl"
FRAMES = OUTC / "r2_frames.csv"
MODES = ("cheap", "full")
FIRST, STEP, WINDOW = 4, 2, 80            # first frame (the track's first state), DB iterations per 0.1 s step, 8 s
FRAME_TOL_US = 50_000
PLANNERS = ("pdm_closed", "idm")
LQR = dict(q_longitudinal=np.array([10.0]), r_longitudinal=np.array([1.0]), q_lateral=np.array([1.0, 10.0, 0.0]),
           r_lateral=np.array([1.0]), discretization_time=0.1, tracking_horizon=10, jerk_penalty=1e-4,
           curvature_rate_penalty=1e-2, stopping_proportional_gain=0.5, stopping_velocity=0.2)
N_RANDOM = 20


def scenarios(args):
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
    from nuplan.planning.utils.multithreading.worker_sequential import Sequential
    builder = NuPlanScenarioBuilder(data_root=f"{args.data_root}/nuplan-v1.1/splits/mini", map_root=args.map_root,
                                    sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=60, timestamp_threshold_s=None,
                         ego_displacement_minimum_m=None, expand_scenarios=False, remove_invalid_goals=True, shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
    assert len(scen) == 60
    return scen


def shard_of(tokens, shard):
    i, n = (int(x) for x in shard.split("/"))
    return [t for j, t in enumerate(sorted(tokens)) if j % n == i]


# ------------------------------------------------------------------------------------------------ spec

def same_spec(a, b, tol=1e-9):
    """Equal removed tracks and false-positive tokens and classes exactly, false-positive coordinates and sizes within
    tol (the training machine's numpy rounds the lift's matrix product differently in the last bit)."""
    (ra, fa), (rb, fb) = a, b
    return (ra == rb and len(fa) == len(fb) and all(x[:2] == y[:2] for x, y in zip(fa, fb))
            and all(abs(u - w) <= tol for x, y in zip(fa, fb) for u, w in zip(x[2:], y[2:])))


def stage_spec(args):
    """114's per-iteration observation construction, unchanged, at every iteration of the 60 scenarios (branches
    cheap and full only)."""
    scen = scenarios(args)
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())["nuplan"]
    priors = t114.size_priors(sorted(splits["train"] + splits["val"]))
    ground_z = t114.ground_height(sorted(splits["train"] + splits["val"]))["ground_z_ego_m"]
    thresholds = {"0.25": 0.25}
    idx = pd.read_csv(t114.WORK / "image_index.csv.gz")
    old = pickle.load(open(OUTC / "branches.pkl", "rb"))["spec"]
    spec, rows, checked, differ = {}, [], 0, 0
    cams, dets_cache = {}, {}
    t0 = time.time()
    for n_sc, tok in enumerate(sorted(scen), 1):
        sc = scen[tok]
        log = sc.log_name
        if log not in cams:
            cams[log] = t114.Camera(log)
            dets_cache[log] = {m: t114.DetCache(t114.DET / t114.MODE_CACHE[m] / f"{log}.npz") for m in t114.MODE_CACHE}
        cam, li = cams[log], idx[idx.log == log].sort_values("timestamp")
        ts = li.timestamp.to_numpy()
        con = sqlite3.connect(f"file:{t114.DB_DIR / (log + '.db')}?mode=ro", uri=True)
        for k in range(sc.get_number_of_iterations()):
            t_k = sc.get_time_point(k).time_us
            j = int(np.argmin(np.abs(ts - t_k)))
            im = li.iloc[j]
            rows.append({"scenario": tok, "log": log, "iteration": k, "time_us": t_k, "image_token": im.image_token,
                         "dt_us": int(im.timestamp - t_k), "has_frame": bool(abs(int(im.timestamp) - t_k) <= FRAME_TOL_US)})
            ep = con.execute("select x, y, z, qw, qx, qy, qz from ego_pose where token = ?",
                             (bytes.fromhex(im.ego_pose_token),)).fetchone()
            Re, te, yaw_e = t114.quat_R(*ep[3:]), np.array(ep[:3]), t114.quat_yaw(*ep[3:])
            lb = {r[0].hex(): r[1:] for r in con.execute(
                "select token, x, y, z, yaw, width, length, height from lidar_box where lidar_pc_token = ?",
                (bytes.fromhex(sc._lidarpc_tokens[k]),))}
            recs = []
            for o in sc.get_tracked_objects_at_iteration(k).tracked_objects:
                cat = str(o.tracked_object_type.name).lower()
                x, y, zc, yaw, wd, ln, ht = lb[o.metadata.token]
                cg = np.array([x, y, zc]) + (t114.CORNER_SIGNS * [ln / 2, wd / 2, ht / 2]) @ np.array(
                    [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]).T
                pcam = ((cg - te) @ Re - cam.t) @ cam.R
                centre_cam = ((np.array([x, y, zc]) - te) @ Re - cam.t) @ cam.R
                box = cam.project_box(pcam) if centre_cam[2] > 0 else None
                recs.append({"token": o.metadata.token, "category": cat, "eligible": cat in t114.ELIGIBLE,
                             "box": box, "in_cam": box is not None})
            frame = int(im.frame)
            dd = {m: dets_cache[log][m].det(dets_cache[log][m].index[frame]) for m in t114.MODE_CACHE}
            elig = [i for i, r in enumerate(recs) if r["eligible"] and r["in_cam"]]
            for bname in MODES:
                cname, with_fp = t114.BRANCHES[bname]
                mode, tkey, iou_thr = t114.COMBOS[cname]
                d = dd[mode]
                keep = d["conf"] >= thresholds[tkey]
                kidx = np.flatnonzero(keep)
                boxes, cls = d["xyxy"][keep], d["coarse"][keep]
                mt = t114.hungarian([recs[i]["box"] for i in elig], [recs[i]["category"] for i in elig], boxes, cls, iou_thr)
                matched = {elig[i] for i in mt}
                used = {v[0] for v in mt.values()}
                removed = frozenset(recs[i]["token"] for i in elig if i not in matched)
                fps = []
                for jj in range(len(boxes)):
                    if jj in used:
                        continue
                    b, c, kd = boxes[jj], cls[jj], int(kidx[jj])
                    gp = cam.ground_point(0.5 * (b[0] + b[2]), b[3], ground_z)
                    if gp is None:
                        continue
                    (gx, gy), hray = gp
                    if gx < t114.FP_MIN_AHEAD or np.hypot(gx, gy) > t114.FP_MAX_RANGE:
                        continue
                    cat = t114.DET_TO_CATEGORY[str(c)]
                    L, Wd, Ht, _ = priors[cat]
                    ce = np.array([gx, gy, 0.0]) + np.array([hray[0], hray[1], 0.0]) * L / 2
                    cgl = Re @ ce + te
                    fps.append((f"fp:{mode}:{im.image_token}:{kd}", cat, float(cgl[0]), float(cgl[1]), yaw_e,
                                float(L), float(Wd), float(Ht)))
                spec[(tok, k, bname)] = (removed, fps)
                if (tok, k, bname) in old:
                    checked += 1
                    differ += int(not same_spec(old[(tok, k, bname)], (removed, fps)))
        con.close()
        print(f"  [{n_sc:2d}/60] {tok}: {sc.get_number_of_iterations()} iterations [{time.time() - t0:.0f}s]", flush=True)
    with open(SPEC, "wb") as fh:
        pickle.dump(spec, fh, protocol=pickle.HIGHEST_PROTOCOL)
    pd.DataFrame(rows).to_csv(FRAMES, index=False)
    print(f"spec: {len(spec)} entries; against 114's specification {checked} compared, {differ} differ", flush=True)
    if differ:
        raise SystemExit("spec differs from 114's")


# ------------------------------------------------------------------------------------------------ shared pieces

def load_spec():
    with open(SPEC, "rb") as fh:
        spec = pickle.load(fh)
    fr = pd.read_csv(FRAMES)
    return spec, fr


def logged_frames(fr, tok, n_iter):
    """The scenario's logged frames: the 0.1 s grid from iteration 4 with an image within 50 ms."""
    f = fr[(fr.scenario == tok)].set_index("iteration")
    return [k for k in range(FIRST, n_iter, STEP) if bool(f.loc[k, "has_frame"])]


def history(sc, tok, k, mode, filt):
    base = SimulationHistoryBuffer.initialize_from_scenario(4, sc, DetectionsTracks)
    for j in range(max(k - 4 + 1, 0), k + 1):
        base.append(sc.get_ego_state_at_iteration(j), sc.get_tracked_objects_at_iteration(j))
    return cf82.filtered_history(base, filt, tok, k, mode, sc)


def init_for(kind, sc, k):
    return PlannerInitialization(route_roadblock_ids=cf82.route_for(kind, sc, k), mission_goal=sc.get_mission_goal(),
                                 map_api=sc.map_api)


def plan(planner, sc, k, step, hist):
    return planner.compute_trajectory(PlannerInput(
        iteration=SimulationIteration(sc.get_time_point(k), step), history=hist,
        traffic_light_data=list(sc.get_traffic_light_status_at_iteration(k))))


def safety(collision, clearance):
    """83's safety loss of one evaluation: 10 * collision + 1.5 * clip(1 - clearance, 0, 2)^2."""
    return 10.0 * collision + 1.5 * float(np.clip(1.0 - clearance, 0.0, 2.0)) ** 2


def contact(ego, sc, k):
    """Executed ego footprint against the logged objects at iteration k (82's score, one instant)."""
    poly, clear, hit = ego.car_footprint.geometry, np.inf, 0.0
    for o in sc.get_tracked_objects_at_iteration(k).tracked_objects:
        g = o.box.geometry
        if poly.intersects(g):
            hit, clear = 1.0, 0.0
        else:
            clear = min(clear, float(poly.distance(g)))
    return hit, (float(clear) if np.isfinite(clear) else 20.0)


def rollout(kind, sc, tok, k0, modes, filt, has_frame, keep_first=False, keep_track=False):
    """nuPlan closed loop: non-reactive agents (logged), 0.1 s steps, the default two-stage controller (LQR tracker,
    kinematic bicycle, Pacifica). Step s plans at DB iteration k0 + 2s on the observation of that step's mode (the
    previous observation when no image lies within 50 ms), the controller moves the ego to k0 + 2(s + 1), and the
    loss is evaluated there against the logged objects."""
    n_iter = sc.get_number_of_iterations()
    planner = cf82.make_planner(kind)
    planner.initialize(init_for(kind, sc, k0))
    hist = history(sc, tok, k0, modes[0], filt)
    ego = sc.get_ego_state_at_iteration(k0)
    ctrl = TwoStageController(sc, LQRTracker(**LQR), KinematicBicycleModel(get_pacifica_parameters()))
    obs_prev = hist.observations[-1]
    J = Jdev = 0.0
    out = {"steps": 0, "steps_no_frame": 0, "collisions": 0, "min_clearance": np.inf, "dev_sum": 0.0, "first": None,
           "track": []}
    k = k0
    for s, mode in enumerate(modes):
        if k + STEP >= n_iter:
            break
        if s > 0:
            if has_frame.get((tok, k), False):
                obs_prev = filt.apply(sc.get_tracked_objects_at_iteration(k), ego, tok, k, mode)
            else:
                out["steps_no_frame"] += 1
            hist.append(ego, obs_prev)
        traj = plan(planner, sc, k, s, hist)
        if s == 0 and keep_first:
            out["first"] = traj
        k2 = k + STEP
        ctrl.update_state(SimulationIteration(sc.get_time_point(k), s), SimulationIteration(sc.get_time_point(k2), s + 1),
                          ego, traj)
        ego = ctrl.get_state()
        hit, clear = contact(ego, sc, k2)
        true = sc.get_ego_state_at_iteration(k2)
        dev = float(np.hypot(ego.rear_axle.x - true.rear_axle.x, ego.rear_axle.y - true.rear_axle.y))
        J += safety(hit, clear)
        out["steps"] += 1
        out["collisions"] += int(hit)
        out["min_clearance"] = min(out["min_clearance"], clear)
        out["dev_sum"] += dev
        if keep_track:                  # the executed ego per step, for nuPlan-style components computed afterwards
            out["track"].append([k2, round(ego.rear_axle.x, 3), round(ego.rear_axle.y, 3), round(ego.rear_axle.heading, 5),
                                 round(ego.dynamic_car_state.rear_axle_velocity_2d.x, 4),
                                 round(ego.dynamic_car_state.rear_axle_acceleration_2d.x, 4), int(hit), round(clear, 3)])
        k = k2
    out["J"] = J
    out["J_dev"] = J + 0.1 * (out["dev_sum"] / max(out["steps"], 1))
    return out


# ------------------------------------------------------------------------------------------------ vall

def stage_vall(args):
    scen = scenarios(args)
    spec, fr = load_spec()
    filt = t115.RealPerceptionFilter(spec)
    ck = OUTC / "r2_vall"
    ck.mkdir(parents=True, exist_ok=True)
    for tok in shard_of(scen, args.shard):
        f = ck / f"{tok}.json"
        if f.exists():
            continue
        sc = scen[tok]
        dt = sc.database_interval
        rows = []
        for k in logged_frames(fr, tok, sc.get_number_of_iterations()):
            row = {"scenario": tok, "log": sc.log_name, "iteration": k}
            objs, ego = t119.branch(filt, sc, k, "cheap")             # the track's 18 gate features (119), cheap branch
            row.update({f"nr_{c}": x for c, x in t119.gate_row(geom_filter(), objs, ego, sc, k,
                                                                t119.branch(filt, sc, k - 1, "cheap")).items()})
            for kind in PLANNERS:
                init = init_for(kind, sc, k)
                for m in MODES:
                    res = t115.run_planner(kind, sc, k, init, history(sc, tok, k, m, filt), dt)
                    row.update({f"{kind}_{key}_{m}": v for key, v in res.items()})
                    row[f"{kind}_safety_{m}"] = (safety(res["collision"], res["min_clearance"]) if res.get("ok") else np.nan)
                row[f"{kind}_V"] = row[f"{kind}_safety_cheap"] - row[f"{kind}_safety_full"]
            rows.append(row)
        f.write_text(json.dumps(rows))
        print(f"  {tok}: {len(rows)} frames", flush=True)


_GEOM = []


def geom_filter():
    """119's geometry helpers (rap.nuplan_perception.PerceptionFilter with the fitted miss model)."""
    if not _GEOM:
        from rap.detector_model import DetectorMissModel
        from rap.nuplan_perception import PerceptionFilter
        z = np.load(Path(CACHE) / "detector_miss_model.npz", allow_pickle=False)
        _GEOM.append(PerceptionFilter(DetectorMissModel(z["w_cheap"], z["w_rescue"], z["w_lose"])))
    return _GEOM[0]


def gate_full_frames(v, planner):
    """gate_gbm at 20% on the test-log scenarios: 120's gate (18 features, the benchmark's GBM, seed 0) refit on the
    track's 1,440 states of the training and validation logs on the planner's safety V, scored on every frame of the
    test-log scenarios; the top 20% of those frames, pooled, ties by (scenario, iteration)."""
    import glob
    from rap import predict
    gate = ["ego_speed", "ego_accel", "n_cheap", "n_cheap_fov", "n_cheap_fov_bicycle", "n_cheap_fov_pedestrian",
            "n_cheap_fov_vehicle", "n_static_fov", "min_range_fov", "min_gap_corridor", "n_corridor_20", "n_corridor_40",
            "min_ttc_corridor", "n_fov_band_30_40", "crit_cheap_sum", "crit_cheap_max", "n_red_lights", "d_n_cheap_fov"]
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    run = sorted(glob.glob(str(ROOT / "results" / "raw" / "*_nuplan_real_features")))[-1]      # 120's cells input
    sig = pd.read_csv(Path(run) / "nuplan_real_signals.csv").rename(columns={c: f"nr_{c}" for c in gate})
    r = pd.read_csv(Path(RESULTS) / "final" / f"nuplan_real_perception_{planner}_raw.csv")
    d = r.merge(sig, on=["scenario", "log", "iteration"], how="inner", validate="one_to_one")
    assert len(d) == 1440
    vv = np.array([safety(a, b) - safety(c_, e) for a, b, c_, e in
                   zip(d.collision_cheap, d.min_clearance_cheap, d.collision_full, d.min_clearance_full)])
    split_of = {u: k for k in ("train", "val", "test") for u in splits["nuplan"][k]}
    cols = [f"nr_{g}" for g in gate]
    fit = d.log.map(split_of).isin(["train", "val"]).to_numpy()
    X = np.nan_to_num(d[cols].to_numpy(float), nan=0.0, posinf=1e6, neginf=-1e6)
    m = predict.make_model("gbm", "reg", 0).fit(X[fit], vv[fit])
    test_logs = set(splits["nuplan"]["test"])
    w = v[v.log.isin(test_logs)].copy()
    w["score"] = m.predict(np.nan_to_num(w[cols].to_numpy(float), nan=0.0, posinf=1e6, neginf=-1e6))
    w = w.sort_values(["score", "scenario", "iteration"], ascending=[False, True, True])
    k = int(round(0.2 * len(w)))
    return set(map(tuple, w.head(k)[["scenario", "iteration"]].to_numpy().tolist())), set(w.scenario)


def v_table():
    rows = [r for f in sorted((OUTC / "r2_vall").glob("*.json")) for r in json.loads(f.read_text())]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------ gate

def stage_gate(args):
    scen = scenarios(args)
    spec, fr = load_spec()
    filt = t115.RealPerceptionFilter(spec)
    has = {(r.scenario, int(r.iteration)): bool(r.has_frame) for r in fr.itertuples()}
    v = v_table()
    rng = np.random.default_rng(0)
    nz = v[v.pdm_closed_V.fillna(0) != 0]
    pick_a = nz.iloc[np.sort(rng.choice(len(nz), 20, replace=False))]
    a = []
    for r in pick_a.itertuples():
        sc = scen[r.scenario]
        j1 = rollout("pdm_closed", sc, r.scenario, int(r.iteration), ["cheap"] * WINDOW, filt, has)["J"]
        j2 = rollout("pdm_closed", sc, r.scenario, int(r.iteration), ["cheap"] * WINDOW, filt, has)["J"]
        a.append({"scenario": r.scenario, "iteration": int(r.iteration), "J1": j1, "J2": j2, "dJ": j1 - j2})
    states = pd.read_csv(Path(RESULTS) / "final" / "phase0g_external_pdm_closed_raw.csv")[["scenario", "iteration"]]
    states = states.drop_duplicates().iloc[np.sort(rng.choice(1440, 100, replace=False))]
    b = []
    for r in states.itertuples():
        sc, k = scen[r.scenario], int(r.iteration)
        p = cf82.make_planner("pdm_closed")
        p.initialize(init_for("pdm_closed", sc, k))
        ol = plan(p, sc, k, 0, history(sc, r.scenario, k, "full", filt))
        cl = rollout("pdm_closed", sc, r.scenario, k, ["full"] + ["cheap"], filt, has, keep_first=True)["first"]
        pa = np.array([[s.rear_axle.x, s.rear_axle.y] for s in ol.get_sampled_trajectory()])
        pb = np.array([[s.rear_axle.x, s.rear_axle.y] for s in cl.get_sampled_trajectory()])
        b.append({"scenario": r.scenario, "iteration": k, "max_abs_m": float(np.abs(pa - pb).max()) if pa.shape == pb.shape
                  else np.inf})
    rep = {"a_frames": len(a), "a_max_abs_dJ": float(max(abs(x["dJ"]) for x in a)), "a_pass": all(x["dJ"] == 0 for x in a),
           "b_states": len(b), "b_max_abs_m": float(max(x["max_abs_m"] for x in b)),
           "b_pass": all(x["max_abs_m"] <= 1e-6 for x in b), "a_rows": a, "b_rows": b}
    rep["pass"] = rep["a_pass"] and rep["b_pass"]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_r2_0.json").write_text(json.dumps(rep, indent=1))
    print("gate R2-0:", {k: rep[k] for k in ("a_frames", "a_max_abs_dJ", "a_pass", "b_states", "b_max_abs_m", "b_pass",
                                              "pass")}, flush=True)
    if not rep["pass"]:
        raise SystemExit(1)


# ------------------------------------------------------------------------------------------------ forks and policies

def stage_forks(args):
    scen = scenarios(args)
    spec, fr = load_spec()
    filt = t115.RealPerceptionFilter(spec)
    has = {(r.scenario, int(r.iteration)): bool(r.has_frame) for r in fr.itertuples()}
    v = v_table()
    ck = OUTC / f"r2_forks_{args.planner}"
    ck.mkdir(parents=True, exist_ok=True)
    for tok in shard_of(scen, args.shard):
        f = ck / f"{tok}.json"
        if f.exists():
            continue
        sc = scen[tok]
        rows = []
        for r in v[(v.scenario == tok) & (v[f"{args.planner}_V"].fillna(0) != 0)].itertuples():
            k = int(r.iteration)
            res = {m: rollout(args.planner, sc, tok, k, ([m] if m == "full" else ["cheap"]) + ["cheap"] * (WINDOW - 1),
                              filt, has) for m in MODES}
            rows.append({"scenario": tok, "log": sc.log_name, "iteration": k, "V_t": getattr(r, f"{args.planner}_V"),
                         "steps": min(res["cheap"]["steps"], res["full"]["steps"]),
                         "truncated": min(res["cheap"]["steps"], res["full"]["steps"]) < WINDOW,
                         **{f"{key}_{m}": res[m][key] for m in MODES
                            for key in ("J", "J_dev", "collisions", "min_clearance", "steps_no_frame")}})
        f.write_text(json.dumps(rows))
        print(f"  {tok}: {len(rows)} forks", flush=True)


def policy_modes(name, frames_k, vmap, seed=None, n_full=None):
    if name == "all-cheap":
        return ["cheap"] * len(frames_k)
    if name == "all-full":
        return ["full"] * len(frames_k)
    if name == "privileged":
        return ["full" if vmap.get(k, 0.0) > 0 else "cheap" for k in frames_k]
    m = ["cheap"] * len(frames_k)
    for i in np.random.default_rng(seed).choice(len(frames_k), n_full, replace=False):
        m[i] = "full"
    return m


def stage_policies(args):
    scen = scenarios(args)
    spec, fr = load_spec()
    filt = t115.RealPerceptionFilter(spec)
    has = {(r.scenario, int(r.iteration)): bool(r.has_frame) for r in fr.itertuples()}
    v = v_table()
    gate_full, gate_scen = gate_full_frames(v, args.planner)
    ck = OUTC / f"r2_policies_{args.planner}"
    ck.mkdir(parents=True, exist_ok=True)
    for tok in shard_of(scen, args.shard):
        f = ck / f"{tok}.json"
        if f.exists():
            continue
        sc = scen[tok]
        steps_k = list(range(FIRST, sc.get_number_of_iterations() - STEP, STEP))
        vmap = {int(r.iteration): float(getattr(r, f"{args.planner}_V")) for r in v[v.scenario == tok].itertuples()
                if np.isfinite(getattr(r, f"{args.planner}_V"))}
        priv = policy_modes("privileged", steps_k, vmap)
        n_full = sum(m == "full" for m in priv)
        runs = [("all-cheap", None, policy_modes("all-cheap", steps_k, vmap)),
                ("all-full", None, policy_modes("all-full", steps_k, vmap)), ("privileged", None, priv)]
        runs += [("matched random", s, policy_modes("random", steps_k, vmap, s, n_full)) for s in range(N_RANDOM)]
        if tok in gate_scen:
            runs.append(("gate_gbm at 20% (test logs)", None, ["full" if (tok, k) in gate_full else "cheap" for k in steps_k]))
        rows = []
        for name, seed, modes in runs:
            res = rollout(args.planner, sc, tok, FIRST, modes, filt, has, keep_track=True)
            rows.append({"scenario": tok, "log": sc.log_name, "policy": name, "seed": seed, "full_steps": modes.count("full"),
                         **{k: res[k] for k in ("J", "J_dev", "steps", "collisions", "min_clearance", "steps_no_frame",
                                                "track")}})
        f.write_text(json.dumps(rows))
        print(f"  {tok}: {len(rows)} runs, {n_full} privileged full steps", flush=True)


# ------------------------------------------------------------------------------------------------ analysis

def log_boot(df, fn, nboot=1000):
    logs = sorted(df.log.unique())
    idx = {l: np.flatnonzero(df.log.to_numpy() == l) for l in logs}
    rng = np.random.default_rng(0)
    out = []
    for _ in range(nboot):
        t = np.concatenate([idx[logs[i]] for i in rng.integers(0, len(logs), len(logs))])
        out.append(fn(df.iloc[t]))
    a = np.array(out, float)
    a = a[np.isfinite(a)]
    return (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))) if len(a) else (np.nan, np.nan)


def stage_analysis(args):
    from scipy.stats import spearmanr
    v = v_table()
    OUT.mkdir(parents=True, exist_ok=True)
    v.to_csv(OUT / "v_all_frames.csv", index=False)
    rows = []
    for kind in PLANNERS:
        fs = [r for f in sorted((OUTC / f"r2_forks_{kind}").glob("*.json")) for r in json.loads(f.read_text())]
        if not fs:
            continue
        d = pd.DataFrame(fs)
        d["dJ"] = d.J_cheap - d.J_full
        d["dJ_dev"] = d.J_dev_cheap - d.J_dev_full
        d["planner"] = kind
        d.to_csv(OUT / f"forks_{kind}.csv", index=False)
        for pop, m in (("primary (8 s windows)", ~d.truncated), ("truncated-window sensitivity (all forks)", np.ones(len(d), bool))):
            x = d[m]
            for jcol, lab in (("dJ", "safety"), ("dJ_dev", "safety + 0.1 x deviation (secondary)")):
                agree = lambda y: float((np.sign(y[jcol]) == np.sign(y.V_t))[y[jcol] != 0].mean()) if (y[jcol] != 0).any() else np.nan
                rho = lambda y: float(spearmanr(y.V_t, y[jcol]).correlation)
                lo, hi = log_boot(x, agree)
                rlo, rhi = log_boot(x, rho)
                share = agree(x)
                rows.append({"planner": kind, "population": pop, "loss": lab, "forks": len(x), "logs": x.log.nunique(),
                             "forks_dJ_nonzero": int((x[jcol] != 0).sum()), "share_dJ_zero": float((x[jcol] == 0).mean()),
                             "sign_agreement": share, "lo": lo, "hi": hi, "spearman_V_dJ": rho(x), "rho_lo": rlo,
                             "rho_hi": rhi,
                             "reading": ("single-step sign predicts the closed-loop direction" if lo > 0.5 else
                                         "single-step sign contradicts it" if hi < 0.5 else "unresolved")})
    pd.DataFrame(rows).to_csv(OUT / "forks.csv", index=False)
    prow, runs = [], []
    for kind in PLANNERS:
        ps = [r for f in sorted((OUTC / f"r2_policies_{kind}").glob("*.json")) for r in json.loads(f.read_text())]
        if not ps:
            continue
        d = pd.DataFrame(ps).assign(planner=kind)
        runs.append(d)
        per = d.groupby(["scenario", "log", "policy"])[["J", "J_dev", "collisions", "full_steps"]].mean().reset_index()
        piv = per.pivot_table(index=["scenario", "log"], columns="policy", values="J").reset_index()
        for pol in ("all-cheap", "all-full", "privileged", "matched random", "gate_gbm at 20% (test logs)"):
            if pol not in piv:
                continue
            x = piv[piv[pol].notna()]
            prow.append({"planner": kind, "row": pol, "scenarios": len(x), "J_mean": float(x[pol].mean()),
                         "full_steps_mean": float(per[per.policy == pol].full_steps.mean())})
        diff = lambda y: float((y["privileged"] - y["matched random"]).mean())
        lo, hi = log_boot(piv, diff)
        prow.append({"planner": kind, "row": "privileged - matched random (principal secondary)", "scenarios": len(piv),
                     "J_mean": diff(piv), "lo": lo, "hi": hi, "resolved": bool(lo > 0 or hi < 0)})
    pd.DataFrame(prow).to_csv(OUT / "policies.csv", index=False)
    if runs:
        pd.concat(runs).drop(columns=["track"], errors="ignore").to_csv(OUT / "policy_runs.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(prow).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["spec", "vall", "gate", "forks", "policies", "analysis"])
    ap.add_argument("--planner", default="pdm_closed", choices=PLANNERS)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--data_root", default=str(NUPLAN_ROOT))
    ap.add_argument("--map_root", default=str(NUPLAN_MAPS))
    args = ap.parse_args()
    {"spec": stage_spec, "vall": stage_vall, "gate": stage_gate, "forks": stage_forks, "policies": stage_policies,
     "analysis": stage_analysis}[args.stage](args)


if __name__ == "__main__":
    main()
