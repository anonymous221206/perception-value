#!/usr/bin/env python
"""Task 28: the versioned cost registry every measured-budget score is charged against.

One file, results/final/cost_registry.json, derived here from the same runs and the same functions the paper's budget
tables were built with -- nothing is re-measured:

  detectors    per track and mode: ms and mJ, the energy convention, the timing boundary, the run each value is from
               KITTI and nuScenes: 93's `profile_costs` (01's end-to-end median, image decoding included); mJ on the
               module rails (133's `profile_energy`, GPU + SOC + CPU over idle)
               nuPlan: Task 5's own detector pass (`detect_summary.json`; pre-processing + inference + post-processing,
               image decoding excluded); mJ on the module rails (133's `nuplan_real_cost`)
  allocators   per overhead measurement ("primary", "1thread", "routers" -- the three the budget tables use), track
               and signal: ms from 93's `signal_overhead`, mJ = ms x the same module-rail power over idle 133 charges
               (the gate workload's stored rails; R1 and R2 from 132's medians), and where the time was measured
               (nuPlan has no timed allocator workload: it inherits the nuScenes feature time, as the tables say)
  tables       which canonical table is charged with which detector entries and which overhead measurement

The version string goes into every scored output of the submission path.
"""
from __future__ import annotations

import argparse, hashlib, importlib.util, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta                                                         # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
VERSION = "2026-09-23.1"
CONVENTION = "module"
SIGNALS = ("random", "uncertainty", "criticality_cheap", "gate_ridge", "gate_gbm", "gate_gbm_batched",
           "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "R2_cnn_clf")
OVERHEADS = {"primary": "benchmark_budget_overheads.json", "1thread": "benchmark_budget_overheads_1thread.json",
             "routers": "benchmark_budget_overheads_routers.json"}


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def build() -> dict:
    t133 = _load("t133", "133_energy_module_budgets.py")
    t93 = t133.t93
    rails = t133.detector_rails(CONVENTION)
    det = {}
    costs = t133.profile_costs_fn(rails)
    for track in ("KITTI", "nuScenes"):
        det[track] = {}
        for lvl, x in costs(track).items():
            det[track][lvl] = {
                "ms": x["ms"], "mJ": x["mJ"], "mode": x["mode"],
                "timing_boundary": "end-to-end per frame, image decoding included (01_profile_jetson.py "
                                   "lat_e2e_ms_median)",
                "ms_source": f"results/raw/{x['run']}/profile_summary.json [{x['mode']}].lat_e2e_ms_median",
                "mJ_source": f"results/raw/{x['run']}/profile_rows.json + idle_baseline.json: median over kept rounds "
                             f"of sum over {list(rails)} of (rail mean - idle mean) x lat_e2e_ms_median"}
    nr = t133.nuplan_real_cost(rails)
    det["nuPlan"] = {}
    for lvl, mode in (("cheap", "ns_cheap_320"), ("640", "ns_full_640")):
        det["nuPlan"][lvl] = {
            "ms": nr[lvl]["ms"], "mJ": nr[lvl]["mJ"], "mode": mode,
            "timing_boundary": "pre-processing + inference + post-processing per frame, image decoding excluded "
                               "(Task 5 ms_total_median)",
            "ms_source": f"results/raw/nuplan_task5/detect_summary.json [{mode}].ms_total_median",
            "mJ_source": f"results/raw/nuplan_task5/detect_summary.json energy[{mode}]: sum over {list(rails)} of "
                         "excess_mw x seconds / frames"}
    measured = json.loads((FINAL / "allocator_rails.json").read_text())["median"]
    alloc = {}
    for name, f in OVERHEADS.items():
        ov = json.loads((FINAL / f).read_text())["overheads"]
        power = t133.allocator_power(ov, t133.allocator_rails(CONVENTION), measured, CONVENTION)
        alloc[name] = {"file": f"results/final/{f}", "module_mw_over_idle": power,
                       "power_source": {"gate": f"results/final/{f}: rails_busy_mw - rails_idle_mw, summed over "
                                                f"{list(rails)}",
                                        "r1": "results/final/allocator_rails.json median.r1.module_mw_over_idle (132)",
                                        "r2": "results/final/allocator_rails.json median.r2.module_mw_over_idle (132)"},
                       "tracks": {}}
        for track in ("KITTI", "nuScenes", "nuPlan"):
            alloc[name]["tracks"][track] = {}
            for s in SIGNALS:
                if s in ("gate_gbm_batched", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "R2_cnn_clf") \
                        and name != "routers":
                    continue                                   # timed only in the routers measurement
                if s == "R2_cnn_clf" and track == "nuPlan":
                    continue                                   # no pixel router on nuPlan
                try:
                    ms, mj, src = t93.signal_overhead(ov, track, s, power=power)
                except KeyError:
                    continue
                alloc[name]["tracks"][track][s] = {
                    "ms": ms, "mJ": mj,
                    "provenance": ("no overhead" if s == "random" else
                                   "measured on nuScenes, inherited by nuPlan as a proxy (track features not timed)"
                                   if track == "nuPlan" else f"measured on {track}"),
                    "signal_overhead_source": src}
    tables = {
        "benchmark_table.csv": {"kind": "selection", "tracks": ["KITTI", "nuScenes"], "plan": "benchmark_table"},
        "benchmark_table_routers.csv": {"kind": "selection", "tracks": ["KITTI", "nuScenes"],
                                        "plan": "routers_r1 / router_r2:<dataset>"},
        "benchmark_table_nuplan_real.csv": {"kind": "selection", "tracks": ["nuPlan"], "plan": "nuplan_real"},
        "benchmark_budget_two_level.csv": {"kind": "latency", "unit": "ms", "tracks": ["KITTI", "nuScenes"],
                                           "overheads": "primary", "plan": "benchmark_budget"},
        "benchmark_budget_routers.csv": {"kind": "latency", "unit": "ms", "tracks": ["KITTI", "nuScenes"],
                                         "overheads": "routers", "plan": "benchmark_budget"},
        "benchmark_budget_nuplan_real.csv": {"kind": "latency", "unit": "ms", "tracks": ["nuPlan"],
                                             "overheads": "routers", "plan": "nuplan_real_budget"},
        "benchmark_budget_multifidelity.csv": {"kind": "latency", "unit": "ms", "tracks": ["KITTI"],
                                               "overheads": "primary", "note": "multi-level rows: see the report"},
        "energy_module_budget_two_level.csv": {"kind": "energy", "unit": "mJ", "convention": CONVENTION,
                                               "tracks": ["KITTI", "nuScenes"],
                                               "overheads": "per row: the table column (primary, 1thread, routers)",
                                               "plan": "benchmark_budget"},
        "energy_module_budget_nuplan_real.csv": {"kind": "energy", "unit": "mJ", "convention": CONVENTION,
                                                 "tracks": ["nuPlan"], "overheads": "routers",
                                                 "plan": "nuplan_real_budget"},
        "energy_module_budget_multifidelity.csv": {"kind": "energy", "unit": "mJ", "convention": CONVENTION,
                                                   "tracks": ["KITTI"], "overheads": "per row: the table column",
                                                   "note": "multi-level rows: see the report"}}
    superseded = {
        "benchmark_budget_overheads.json costs (mJ)": "GPU rail only for KITTI and nuScenes; nuPlan charged the KITTI "
                                                     "detector costs. Superseded by the module convention.",
        "benchmark_budget_two_level.csv / benchmark_budget_routers.csv mJ rows": "the old convention; the paper's "
                                                                                 "energy results are the energy_module "
                                                                                 "tables",
        "benchmark_budget_nuplan_real.csv mJ rows": "CPU + GPU rails; superseded by energy_module_budget_nuplan_real.csv"}
    reg = {"version": VERSION, "energy_convention": {"name": CONVENTION, "rails": list(rails),
                                                     "definition": "per-frame energy = sum over the rails of (busy mean - "
                                                                   "idle mean) x per-frame time, for detectors and "
                                                                   "allocators alike"},
           "detectors": det, "allocators": alloc, "tables": tables, "superseded": superseded}
    reg["content_sha256"] = hashlib.sha256(json.dumps({k: v for k, v in reg.items()}, sort_keys=True,
                                                      default=float).encode()).hexdigest()
    return reg


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    run = runmeta.new_run("cost_registry", vars(args))
    reg = build()
    txt = json.dumps(reg, indent=1, default=float)
    for f in (FINAL / "cost_registry.json", run / "cost_registry.json"):
        f.write_text(txt)
    d = reg["detectors"]
    print(f"cost registry {reg['version']} ({reg['energy_convention']['name']}): " + "; ".join(
        f"{t} cheap {d[t]['cheap']['ms']:.3f} ms {d[t]['cheap']['mJ']:.3f} mJ, full {d[t]['640']['ms']:.3f} ms "
        f"{d[t]['640']['mJ']:.3f} mJ" for t in ("KITTI", "nuScenes", "nuPlan")))
    print("wrote", FINAL / "cost_registry.json")


if __name__ == "__main__":
    main()
