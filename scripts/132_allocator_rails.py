#!/usr/bin/env python
"""Task 19 Part B: every power rail over idle while the allocators run, on this board.

Pre-registered in the pre-registration record (not part of this release), Task 19 Part B.  `93_budget_allocation.py`
charged allocator energy at the power over idle of one or two rails and kept the per-rail samples only for the gate
workload:

  gate workload (65 features + single-row GBM)  all rails stored (rails_idle_mw, rails_busy_mw), CPU charged  93:139-151
  R1 (detection list + MLP, one row)            CPU over idle only                                             93:215-221
  R2 (resize + upload + TensorRT CNN)           CPU + GPU over idle only                                       93:251-258

This script runs the same three workloads, with the same inputs and the same sampler (93's `_rails_during`: 8 s
idle, then 8 s busy, rails read every 50 ms), and stores every rail.  Three repetitions, interleaved.  Latency is
not re-measured; the budgets keep the shipped overhead times.

Output: results/final/allocator_rails.json
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time, urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    from rap.device import warn_if_not_reference                          # noqa: E402
    warn_if_not_reference("132_allocator_rails.py", strict=True)
from rap.paths import DATASETS as _DS                                     # noqa: E402
from rap import features as F, geometry as G, power, predict, runmeta          # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

MODULE = ("GPU", "SOC", "CPU")          # the rails energy_module_mj_per_frame sums (01_profile_jetson.py:124-126)
NUSC_IMAGES = str(_DS / "nuscenes/trainval/samples/CAM_FRONT/*.jpg")


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def board_state():
    try:
        s = json.loads(urllib.request.urlopen("http://127.0.0.1:8765/api/jetson/status", timeout=5).read())
        return {k: s.get(k) for k in ("throttle", "temperatures", "fan", "nvpmodel", "power")}
    except Exception as e:                                                      # noqa: BLE001
        return {"error": str(e)}


def over_idle(idle, busy, rails):
    return float(sum(busy.get(r, 0.0) - idle.get(r, 0.0) for r in rails))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    if not power.AVAILABLE:
        raise SystemExit("no power rails on this machine")
    run = runmeta.new_run("allocator_rails", vars(args))
    t93 = _load("t93", "93_budget_allocation.py")
    r1 = _load("r103", "103_routers_r1.py")
    import cv2, torch
    from rap.trt import TRTModule

    cfg = RiskConfig()
    # ---- gate workload, as 93:114-143: nuScenes frames, timing-only GBM on the 65 features
    fr400 = t93._frames(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", 400)
    X = np.nan_to_num(np.array([list(F.frame_features(*it, G.PRIMARY, cfg.op_conf).values()) for it in fr400], float))
    y = X[:, :5].sum(1) + np.random.default_rng(0).normal(size=len(X))
    gbm = predict.make_model("gbm", "reg", 0).fit(X, y)
    fr200 = t93._frames(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", 200)
    ig = {"i": 0}

    def gate_work():
        it = fr200[ig["i"] % len(fr200)]
        ig["i"] += 1
        x = np.nan_to_num(np.array([list(F.frame_features(*it, G.PRIMARY, cfg.op_conf).values())], float))
        gbm.predict(x)

    # ---- R1 workload, as 93:190-218
    XR = np.array([r1.det_list_features(it[0], *r1.IMG_WH["nuScenes"]) for it in fr400])
    yr = XR[:, 0] + np.random.default_rng(0).normal(size=len(XR))
    mlp = r1.models()["R1_mlp_reg"][1]().fit(XR, yr)
    i1 = {"i": 0}

    def r1_work():
        it = fr400[i1["i"] % len(fr400)]
        i1["i"] += 1
        mlp.predict(r1.det_list_features(it[0], *r1.IMG_WH["nuScenes"])[None])

    # ---- R2 workload, as 93:223-258
    dev = torch.device("cuda:0")
    decoded = [cv2.imread(q, cv2.IMREAD_COLOR) for q in sorted(glob.glob(NUSC_IMAGES))[:200]]
    assert len(decoded) == 200 and all(im is not None for im in decoded)

    def pre(im):
        x = torch.from_numpy(np.ascontiguousarray(
            cv2.resize(im, (128, 128), interpolation=cv2.INTER_AREA)[:, :, ::-1].transpose(2, 0, 1)))
        x = x.to(dev).float()[None]
        torch.cuda.synchronize()
        return x
    r2run = sorted(glob.glob(str(ROOT / "results" / "raw" / "*_router_r2")))[-1]
    eng = TRTModule(Path(r2run) / "r2_nuScenes_fp16.engine")
    x0 = torch.zeros(1, 3, 128, 128, device=dev)
    for _ in range(50):
        eng(x0)
    i2 = {"i": 0}

    def r2_work():
        im = decoded[i2["i"] % len(decoded)]
        i2["i"] += 1
        eng(pre(im))
        torch.cuda.synchronize()

    work = {"gate": gate_work, "r1": r1_work, "r2": r2_work}
    for fn in work.values():                                                    # warm every path once
        for _ in range(20):
            fn()
    state_before = board_state()
    reps = []
    for rep in range(args.reps):
        for name, fn in work.items():
            idle = t93._rails_during(lambda: time.sleep(0.02), 8.0)
            busy = t93._rails_during(fn, 8.0)
            reps.append({"rep": rep, "workload": name, "idle_mw": idle, "busy_mw": busy,
                         "module_mw_over_idle": over_idle(idle, busy, MODULE),
                         "all_rails_mw_over_idle": over_idle(idle, busy, sorted(busy)),
                         "cpu_mw_over_idle": over_idle(idle, busy, ("CPU",)),
                         "cpu_gpu_mw_over_idle": over_idle(idle, busy, ("CPU", "GPU"))})
            r = reps[-1]
            print(f"  rep {rep} {name:4s}: module {r['module_mw_over_idle']:7.0f} mW  all rails "
                  f"{r['all_rails_mw_over_idle']:7.0f}  CPU {r['cpu_mw_over_idle']:7.0f}  CPU+GPU {r['cpu_gpu_mw_over_idle']:7.0f}",
                  flush=True)
    summary = {}
    for name in work:
        rr = [r for r in reps if r["workload"] == name]
        summary[name] = {k: float(np.median([r[k] for r in rr])) for k in
                         ("module_mw_over_idle", "all_rails_mw_over_idle", "cpu_mw_over_idle", "cpu_gpu_mw_over_idle")}
        summary[name]["median_rails_over_idle_mw"] = {
            rail: float(np.median([r["busy_mw"].get(rail, 0.0) - r["idle_mw"].get(rail, 0.0) for r in rr]))
            for rail in sorted(rr[0]["busy_mw"])}
    shipped = json.loads((Path(RESULTS) / "final" / "benchmark_budget_overheads_routers.json").read_text())["overheads"]
    out = {"rails_module": list(MODULE), "reps": reps, "median": summary, "board_before": state_before,
           "board_after": board_state(), "r2_engine_run": Path(r2run).name,
           "shipped_for_comparison": {"gate_cpu_mw_over_idle": shipped["cpu_mw_over_idle"],
                                      "gate_all_rails_mw_over_idle": shipped["all_rails_mw_over_idle"],
                                      "r1_cpu_mw_over_idle": shipped["r1_cpu_mw_over_idle"],
                                      "r2_cpu_gpu_mw_over_idle": shipped["r2_cpu_gpu_mw_over_idle"]}}
    for dst in (Path(RESULTS) / "final" / "allocator_rails.json", run / "allocator_rails.json"):
        dst.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: round(vv) for kk, vv in v.items() if not isinstance(vv, dict)} for k, v in summary.items()}))
    print("  shipped:", {k: round(v) for k, v in out["shipped_for_comparison"].items()})


if __name__ == "__main__":
    main()
