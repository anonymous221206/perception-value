#!/usr/bin/env python
"""Follow-up E6: Diffusion Planner (released nuPlan checkpoint) on the benchmark's nuPlan real-perception states.

States, branch observations (114's specification through 115's RealPerceptionFilter), scenario construction and scoring
(82's `score`, then 83's safety and scalar losses) are the external track's. The planner is the released
Diffusion Planner with its released configuration (args.json, model.pth with EMA weights, 2 s of history at 0.1 s,
8 s / 80-pose output), on DEEP_DEVICE (cpu by default; the run used cuda on the training machine, see the addendum's
amendment 1). Sampling noise: the generator is seeded with SEED immediately before every planner call (common random
numbers across branches), cuDNN deterministic; one extra reference call with seed SEED + 1 gives the A/A noise floor
(columns *_reference_s1).

History: the planner's released input is 21 observations at 0.1 s. The track defines branch observations on the last
four database frames of each state (0.15 s at 20 Hz); on the history frames inside that span (the current frame and
the one 0.1 s earlier) the branch replaces the logged tracks, older frames keep the logged tracks in every branch
(frames before the scenario start are the logged frames before it, from the scenario's database; amendment 2). Branches: reference, cheap, full; branches whose buffered
observations are identical share one planner call. Checkpointed per scenario under data/cache/xl/e6_rows/.
Writes results/followup/E6/raw.csv. Run with scripts/pynuplan.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, sys, time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DP = ROOT / "third_party" / "diffusion_planner"
sys.path.insert(0, str(DP))
sys.path.insert(0, str(ROOT / "scripts" / "shims"))      # mmengine.fileio for local paths (see shims/)
_s = importlib.util.spec_from_file_location("t115", ROOT / "scripts" / "115_nuplan_real_counterfactual.py")
t115 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(t115)
cf82 = t115.cf82

from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer   # noqa: E402
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput  # noqa: E402
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling          # noqa: E402
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder  # noqa: E402
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter                      # noqa: E402
from nuplan.planning.utils.multithreading.worker_sequential import Sequential                     # noqa: E402
from rap.paths import CACHE, NUPLAN_MAPS, NUPLAN_ROOT, RESULTS                                    # noqa: E402

OUT = Path(RESULTS) / "followup" / "E6"
CK = Path(CACHE) / "xl" / "e6_rows"
BRANCHES = ("cheap", "full")
HIST, STEP = 21, 2                    # 21 observations, every second 20 Hz frame (0.1 s)
DEVICE = os.environ.get("DEEP_DEVICE", "cpu")
SEED = 0


def make_planner():
    from diffusion_planner.planner.planner import DiffusionPlanner
    from diffusion_planner.utils.config import Config
    cfg = Config(str(DP / "checkpoints" / "args.json"), guidance_fn=None)
    return DiffusionPlanner(config=cfg, ckpt_path=str(DP / "checkpoints" / "model.pth"),
                            past_trajectory_sampling=TrajectorySampling(num_poses=20, time_horizon=2),
                            future_trajectory_sampling=TrajectorySampling(num_poses=80, time_horizon=8),
                            enable_ema=True, device=DEVICE)


def history(sc, it, filt, tok, mode, covered):
    its = list(range(it - STEP * (HIST - 1), it + 1, STEP))
    egos, obs = [], []
    n_pre = sum(k < 0 for k in its)
    if n_pre:                         # logged frames before the scenario's first iteration, from the same database,
        h0 = n_pre * STEP * sc.database_interval      # on the same 0.1 s grid, counted back from the first k >= 0
        egos += list(sc.get_ego_past_trajectory(its[n_pre], time_horizon=h0, num_samples=n_pre))
        obs += list(sc.get_past_tracked_objects(its[n_pre], time_horizon=h0, num_samples=n_pre))
        assert len(egos) == len(obs) == n_pre
        its = its[n_pre:]
    for k in its:
        e, o = sc.get_ego_state_at_iteration(k), sc.get_tracked_objects_at_iteration(k)
        if mode != "reference" and (tok, k, mode) in covered:
            o = filt.apply(o, e, tok, k, mode)
        egos.append(e); obs.append(o)
    return SimulationHistoryBuffer(deque(egos), deque(obs), 0.1)


def run(planner_factory, sc, it, init, hist, dt, seed=SEED):
    import torch
    p = planner_factory()
    p.initialize(init)
    try:
        torch.manual_seed(seed)
        traj = p.compute_trajectory(PlannerInput(
            iteration=SimulationIteration(sc.get_ego_state_at_iteration(it).time_point, it),
            history=hist, traffic_light_data=list(sc.get_traffic_light_status_at_iteration(it))))
        out = cf82.score(traj, sc, it, dt)
        out["ok"] = 1
    except Exception as e:                                                       # noqa: BLE001
        out = {"ok": 0, "err": f"{type(e).__name__}: {e}"[:200]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="scenarios (smoke test)")
    ap.add_argument("--redo_failed", action="store_true",
                    help="recompute only the states whose reference call failed, in place (amendment 2)")
    args = ap.parse_args()
    import torch
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    t0 = time.time()
    import pickle
    with open(t115.OUTC / "branches.pkl", "rb") as f:
        spec = pickle.load(f)["spec"]
    filt = t115.RealPerceptionFilter(spec)
    covered = set(spec)
    trackb = pd.read_csv(Path(RESULTS) / "final" / "phase0g_external_pdm_closed_raw.csv")
    states = trackb[["scenario", "log", "iteration"]]
    builder = NuPlanScenarioBuilder(data_root=f"{NUPLAN_ROOT}/nuplan-v1.1/splits/mini", map_root=str(NUPLAN_MAPS),
                                    sensor_root=None, db_files=None, map_version="nuplan-maps-v1.0")
    flt = ScenarioFilter(scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
                         num_scenarios_per_type=None, limit_total_scenarios=60, timestamp_threshold_s=None,
                         ego_displacement_minimum_m=None, expand_scenarios=False, remove_invalid_goals=True,
                         shuffle=False)
    scen = {s.token: s for s in builder.get_scenarios(flt, Sequential())}
    assert set(scen) == set(states.scenario)
    CK.mkdir(parents=True, exist_ok=True)
    calls = 0
    groups = list(states.groupby("scenario", sort=True))
    for n, (tok, g) in enumerate(groups[: args.limit or None], 1):
        f = CK / f"{tok}.json"
        prev = {r["iteration"]: r for r in json.loads(f.read_text())} if f.exists() else {}
        if f.exists() and (not args.redo_failed or all(r.get("ok_reference") == 1 for r in prev.values())):
            continue
        sc = scen[tok]
        rows = []
        for it in sorted(int(i) for i in g.iteration):
            if prev.get(it, {}).get("ok_reference") == 1:
                rows.append(prev[it]); continue
            init = PlannerInitialization(route_roadblock_ids=sc.get_route_roadblock_ids(),
                                         mission_goal=sc.get_mission_goal(), map_api=sc.map_api)
            row = {"scenario": tok, "log": sc.log_name, "iteration": it}
            by_sig = {}
            for mode in ("reference",) + BRANCHES:
                h = history(sc, it, filt, tok, mode, covered)
                s = t115.obs_signature(h)
                if s in by_sig:
                    res = by_sig[s]; row[f"shared_{mode}"] = 1
                else:
                    res = run(make_planner, sc, it, init, h, sc.database_interval); calls += 1
                    by_sig[s] = res; row[f"shared_{mode}"] = 0
                row.update({f"{k}_{mode}": v for k, v in res.items()})
            res = run(make_planner, sc, it, init, history(sc, it, filt, tok, "reference", covered),
                      sc.database_interval, SEED + 1); calls += 1
            row.update({f"{k}_reference_s1": v for k, v in res.items()})
            rows.append(row)
        f.write_text(json.dumps(rows, default=float))
        print(f"  [{n}/{len(groups)}] {tok}: {len(rows)} states, {calls} calls, {time.time() - t0:.0f}s", flush=True)
    d = pd.DataFrame([r for f in sorted(CK.glob("*.json")) for r in json.loads(f.read_text())])
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "raw.csv", index=False)
    print("wrote", OUT / "raw.csv", len(d))


if __name__ == "__main__":
    main()
