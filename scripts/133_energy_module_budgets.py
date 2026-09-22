#!/usr/bin/env python
"""Task 19 Part B: every energy-budget result under one module energy convention, into new files.

Pre-registered in the pre-registration record (not part of this release), Task 19 Part B.  The shipped energy budgets
charge the YOLOv8s detectors their GPU rail over idle (01_profile_jetson.py `energy_gpu_mj_per_frame`), nuPlan's
detectors CPU + GPU (113), and the allocators CPU or CPU + GPU (93).  Here one rail set R is used on both sides:

  module        R = {GPU, SOC, CPU} for detectors and allocators (deciding; the rails `energy_module_mj_per_frame` sums)
  all_rails     R = every rail for both (sensitivity)
  as_specified  detectors on the module rails, allocators on every rail (sensitivity: the task's literal wording)

Detector energy = sum over R of (rail mean - idle mean) x the per-frame time of the pass the rails were sampled over
(median end-to-end ms for the profiles, as 01 does; pass time per frame for nuPlan's energy pass, as 113 does).
Allocator energy = the shipped overhead ms x sum over R of (busy - idle): the gate workload's rails are stored in each
overheads file, R1's and R2's come from `132_allocator_rails.py` (results/final/allocator_rails.json).  Only mJ rows are
recomputed; the bootstrap draws are the shipped ones.  The Part A feasibility rule applies.

  --save_gate_scores   one-off: the core gate scores and the multi-fidelity per-level predictions exist nowhere on
                       disk; regenerate them once with the benchmark's own deterministic fits and save them
  --check_scores       check S: with the saved scores and the shipped conventions, 93 reproduces every row of its
                       shipped run outputs (results/final/energy_module_score_check.json)
  (default)            checks M1-M3, S2-S3 and the recompute:
                       results/final/energy_module_budget_{two_level,multifidelity,nuplan_real}.csv,
                       energy_module_skip_accounting.csv, energy_module_costs.json, energy_module_claims.csv
"""
from __future__ import annotations

import argparse, glob, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap.budget import flag_multifidelity, flag_two_level                       # noqa: E402
from rap.paths import RESULTS                                                   # noqa: E402

FINAL = Path(RESULTS) / "final"
RAW = ROOT / "results" / "raw"
MODULE = ("GPU", "SOC", "CPU")
CONVENTIONS = ("module", "all_rails", "as_specified")
TABLES = {"primary": ("benchmark_budget", "benchmark_budget_overheads.json", False),
          "1thread": ("benchmark_budget_1thread", "benchmark_budget_overheads_1thread.json", False),
          "routers": ("benchmark_budget_routers", "benchmark_budget_overheads_routers.json", True)}
NBOOT = 1000


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t93 = _load("t93", "93_budget_allocation.py")
t92 = t93.t92
t120 = _load("t120", "120_nuplan_real_allocation.py")
t128 = _load("t128", "128_skip_accounting.py")


def _latest(tag):
    """The newest run of `tag` in the current lift frame (rap.runs)."""
    return rap_runs.latest(tag)


def read(path):
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False, na_values=[""], low_memory=False)


# ------------------------------------------------------------------------------------------------ constants

def detector_rails(convention):
    return MODULE if convention in ("module", "as_specified") else None           # None = every rail


def allocator_rails(convention):
    return MODULE if convention == "module" else None


def _sum_rails(over, rails):
    return float(sum(v for k, v in over.items() if rails is None or k in rails))


def profile_energy(run, mode, rails):
    """Median over the kept rounds of sum_R (rail mean - idle mean) x median end-to-end ms / 1e3 (01:118-126)."""
    rows = json.loads((RAW / run / "profile_rows.json").read_text())
    idle = json.loads((RAW / run / "idle_baseline.json").read_text())
    kept = [r for r in rows if r["mode"] == mode and r["round"] > 0]
    names = [k[:-len("_mw_mean")] for k in idle if k.endswith("_mw_mean")]
    use = [n for n in names if rails is None or n in rails]
    return float(np.median([sum(r[f"{n}_mw_mean"] - idle[f"{n}_mw_mean"] for n in use) * r["lat_e2e_ms_median"] / 1e3
                            for r in kept]))


def profile_costs_fn(rails):
    """93's `profile_costs` with the mJ entry recomputed on the rail set; ms entries are the shipped ones."""
    def costs(track):
        out = t93.profile_costs(track)
        for lvl, x in out.items():
            x = dict(x)
            x["mJ"] = profile_energy(x["run"], x["mode"], rails)
            out[lvl] = x
        return out
    return costs


def nuplan_real_cost(rails):
    det = json.loads((RAW / "nuplan_task5" / "detect_summary.json").read_text())
    out = {}
    for lvl, mode in (("cheap", "ns_cheap_320"), ("640", "ns_full_640")):
        e = det["energy"][mode]
        out[lvl] = {"ms": det[mode]["ms_total_median"], "mJ": _sum_rails(e["excess_mw"], rails) * e["seconds"] / e["frames"]}
    return out


def allocator_power(ov, rails, measured, convention):
    """mW over idle per workload: the gate from the overheads file's stored rails, R1 and R2 from 132's medians."""
    gate = _sum_rails({k: ov["rails_busy_mw"][k] - ov["rails_idle_mw"].get(k, 0.0) for k in ov["rails_busy_mw"]}, rails)
    key = "module_mw_over_idle" if rails is not None else "all_rails_mw_over_idle"
    return {"gate": gate, "r1": measured["r1"][key], "r2": measured["r2"][key]}


def method_checks(ovs, measured):
    """M1: per-row profile energies reproduce the stored fields; M2: nuPlan CPU+GPU; M3: stored gate CPU power."""
    chk = {"M1": [], "M2": [], "M3": []}
    for run, modes in ({v[0]: v[1] for v in t93.PROFILE.values()}).items():
        s = json.loads((RAW / run / "profile_summary.json").read_text())
        for mode in modes.values():
            for field, rails in (("energy_gpu_mj_per_frame", ("GPU",)), ("energy_module_mj_per_frame", MODULE)):
                got = profile_energy(run, mode, rails)
                chk["M1"].append({"run": run, "mode": mode, "field": field, "stored": s[mode][field], "recomputed": got,
                                  "ok": bool(np.isclose(got, s[mode][field], rtol=1e-9, atol=1e-12))})
    det = json.loads((RAW / "nuplan_task5" / "detect_summary.json").read_text())
    for mode in ("ns_cheap_320", "ns_full_640"):
        e = det["energy"][mode]
        got = _sum_rails(e["excess_mw"], ("CPU", "GPU")) * e["seconds"] / e["frames"]
        chk["M2"].append({"mode": mode, "stored": e["cpu_gpu_mj_per_frame"], "recomputed": got,
                          "ok": bool(np.isclose(got, e["cpu_gpu_mj_per_frame"], rtol=1e-9, atol=1e-12))})
    for table, ov in ovs.items():
        got = ov["rails_busy_mw"]["CPU"] - ov["rails_idle_mw"]["CPU"]
        chk["M3"].append({"overheads": table, "stored": ov["cpu_mw_over_idle"], "recomputed": got,
                          "ok": bool(np.isclose(got, ov["cpu_mw_over_idle"], rtol=1e-9, atol=1e-12))})
    return chk


# ------------------------------------------------------------------------------------------------ saved scores

KEYCOLS = ("seq", "frame", "scenario", "iteration")


def _keys(d):
    return {k: d[k].astype(str).to_numpy().astype(str) for k in KEYCOLS if k in d.columns}   # unicode, not object


def core_cells():
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    return [c for gen in (t92.nuscenes_cells, t92.kitti_cells, t92.nuplan_cells) for c in gen(splits)]


def save_gate_scores(cells):
    run = runmeta.new_run("budget_gate_scores", {"note": "regenerated once for Task 19 Part B"})
    for c in cells:
        d = c["d"]
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        g = t92.gate_predictions(d, c["fcols"], v_all)
        geom = "na" if c["geometry"] == "n/a" else c["geometry"]
        np.savez_compressed(run / f"gates__{c['track']}__{geom}__{c['system']}__{c['target']}.npz",
                            V=v_all, **{f"key_{k}": v for k, v in _keys(d).items()}, **g)
        print(f"  saved gates {c['track']} {c['geometry']} {c['system']} {c['target']}", flush=True)
    d, X, m, fit = t93.multi_fidelity_frames()
    out = {f"key_{k}": v for k, v in _keys(d).items()}
    for system, base, levels in t93.MF_SYSTEMS:
        gains = np.stack([(d[base] - d[c]).to_numpy(float) for c in levels], 1)
        for name, P in t93.level_predictions(X, fit, gains).items():
            out[f"{system}__{name}"] = P
    np.savez_compressed(run / "levels__KITTI__mono.npz", **out)
    print(f"  saved multi-fidelity level predictions to {run}", flush=True)


def load_gate_scores():
    run = _latest("budget_gate_scores")
    prov = {}
    for f in sorted(run.glob("gates__*.npz")):
        _, track, geom, system, target = f.stem.split("__")
        geom = "n/a" if geom == "na" else geom
        z = np.load(f, allow_pickle=False)

        def provider(d, z=z):
            for k, v in _keys(d).items():
                assert np.array_equal(z[f"key_{k}"], v), f"saved gate scores are not aligned on {k}"
            return {"gate_ridge": z["gate_ridge"], "gate_gbm": z["gate_gbm"]}
        prov[(track, geom, system, target)] = provider
    z = np.load(run / "levels__KITTI__mono.npz", allow_pickle=False)
    d, _, _, _ = t93.multi_fidelity_frames()
    for k, v in _keys(d).items():
        assert np.array_equal(z[f"key_{k}"], v), "saved level predictions are not aligned"
    levels = {s: {n: z[f"{s}__{n}"] for n in ("gate_ridge", "gate_gbm")} for s, _, _ in t93.MF_SYSTEMS}
    return prov, levels, run.name


def advance_past_two_level(cells, nboot, rng):
    """Consume exactly the draws 93's `two_level` takes, so `multi_fidelity` sees the shipped continuation."""
    for c in cells:
        uniq = np.unique(c["d"].unit.to_numpy()[(c["d"].split == "test").to_numpy()])
        for _ in range(nboot):
            rng.integers(0, len(uniq), len(uniq))


# ------------------------------------------------------------------------------------------------ comparisons

def compare_frames(new, old, keys):
    """Rows matched on `keys`; every shared numeric column at rtol 1e-9 (NaN = NaN), text exactly."""
    new, old = new.reset_index(drop=True), old.reset_index(drop=True)
    assert len(new) == len(old), (len(new), len(old))
    j = new.merge(old, on=keys, how="outer", suffixes=("", "__old"), indicator=True, validate="one_to_one")
    unmatched = int((j._merge != "both").sum())
    bad = {}
    for col in [c for c in old.columns if c not in keys and c in new.columns]:
        a, b = j[col], j[f"{col}__old"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            ok = np.isclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True)
        else:
            ok = (a.astype(str) == b.astype(str)).to_numpy()
        if not ok.all():
            bad[col] = int((~ok).sum())
    return {"rows": int(len(new)), "unmatched": unmatched, "columns_differing": bad, "ok": unmatched == 0 and not bad}


TWO_KEYS = ["track", "geometry", "system", "target", "unit", "budget_level", "signal"]
MF_KEYS = ["system", "unit", "budget_level", "signal"]


def as_93_writes(df, kind):
    """A 93 table as 93 writes it since Task 19 Part A: with the feasibility rule applied.

    93's runs from before that rule (the shipped camera-frame runs) were written unflagged and later flagged by 131;
    runs written since (Task 23's ego-frame runs) are flagged by 93 itself.  The rule is idempotent, so applying it to
    both sides of a comparison compares every value the benchmark reports, whichever code wrote the run.
    """
    if kind == "two_level":
        return flag_two_level(df)
    kc = t93.profile_costs("KITTI")["cheap"]
    return flag_multifidelity(df, {"ms": kc["ms"], "mJ": kc["mJ"]})


def check_scores(cells):
    prov, levels, run_name = load_gate_scores()
    res = {"gate_scores_run": run_name}
    routers = t93.load_router_scores()
    for table, (tag, ovfile, with_routers) in TABLES.items():
        raw = _latest(tag)
        ov = json.loads((raw / ovfile).read_text())["overheads"]
        rng = np.random.default_rng(0)
        t0 = time.time()
        two = pd.DataFrame(t93.two_level(cells, ov, NBOOT, rng, routers if with_routers else None, gate_scores=prov))
        sfx = "" if table != "1thread" else "_1thread"
        name = "benchmark_budget_routers.csv" if with_routers else f"benchmark_budget_two_level{sfx}.csv"
        old = read_back(as_93_writes(read(raw / name), "two_level"))
        two = read_back(as_93_writes(two, "two_level"))
        res[f"{table}__two_level"] = compare_frames(two, old, TWO_KEYS)
        print(f"  S {table} two-level: {res[f'{table}__two_level']} [{time.time() - t0:.0f}s]", flush=True)
        if not with_routers:
            multi = read_back(as_93_writes(pd.DataFrame(t93.multi_fidelity(ov, NBOOT, rng, level_preds=levels)), "mf"))
            old_mf = read_back(as_93_writes(read(raw / f"benchmark_budget_multifidelity{sfx}.csv"), "mf"))
            res[f"{table}__multifidelity"] = compare_frames(multi, old_mf, MF_KEYS)
            print(f"  S {table} multi-fidelity: {res[f'{table}__multifidelity']}", flush=True)
    res["ok"] = all(v["ok"] for k, v in res.items() if isinstance(v, dict))
    (FINAL / "energy_module_score_check.json").write_text(json.dumps(res, indent=1))
    print(f"  check S: {'passed' if res['ok'] else 'FAILED'}", flush=True)
    return res


def read_back(df):
    """Parse a frame the way the shipped CSVs are parsed, so text and NaN compare like for like."""
    import io
    return pd.read_csv(io.StringIO(df.to_csv(index=False)), float_precision="round_trip", keep_default_na=False,
                       na_values=[""], low_memory=False)


# ------------------------------------------------------------------------------------------------ recompute

def nuplan_real_stash():
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    nr = _latest("nuplan_real_allocation")
    stash = []
    for c in t120.cells(splits):
        d = c["d"]
        v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
        z = np.load(nr / f"scores__{c['system']}__{c['target']}.npz", allow_pickle=False)
        assert np.array_equal(z["V"], v_all) and (z["scenario"].astype(str) == d.scenario.astype(str).to_numpy()).all()
        m = (d.split == "test").to_numpy()
        v, units = v_all[m], d.unit.to_numpy()[m]
        scores = {"random": None, "criticality_cheap": d["nr_crit_cheap_sum"].to_numpy(float)[m],
                  **{s: z[s][m] for s in ("gate_ridge", "gate_gbm", *t120.R1)}, "oracle": v}
        stash.append((c, v, units, int((np.abs(v) > t92.EPS).sum()), scores))
    return stash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_gate_scores", action="store_true")
    ap.add_argument("--check_scores", action="store_true")
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    t_all = time.time()
    if args.save_gate_scores or args.check_scores:
        cells = core_cells()
        if args.save_gate_scores:
            save_gate_scores(cells)
        if args.check_scores:
            check_scores(cells)
        return

    run = runmeta.new_run("energy_module_budgets", vars(args))
    measured = json.loads((FINAL / "allocator_rails.json").read_text())["median"]
    ovs = {t: json.loads((FINAL / f).read_text())["overheads"] for t, (_, f, _) in TABLES.items()}
    checks = method_checks(ovs, measured)
    for k, rows in checks.items():
        print(f"  {k}: {sum(r['ok'] for r in rows)}/{len(rows)} reproduced", flush=True)
        assert all(r["ok"] for r in rows), f"check {k} failed: the energy recomputation does not reproduce stored values"
    prov, levels, gate_run = load_gate_scores()

    costs_doc = {"allocator_rails_median_mw": measured, "gate_scores_run": gate_run, "checks": checks, "conventions": {}}
    for conv in CONVENTIONS:
        cf = profile_costs_fn(detector_rails(conv))
        costs_doc["conventions"][conv] = {
            "detector_rails": list(detector_rails(conv)) if detector_rails(conv) else "all",
            "allocator_rails": list(allocator_rails(conv)) if allocator_rails(conv) else "all",
            "detector_mJ": {t: {lvl: x["mJ"] for lvl, x in cf(t).items()} for t in ("nuScenes", "KITTI")},
            "nuplan_real_detector_mJ": {lvl: x["mJ"] for lvl, x in nuplan_real_cost(detector_rails(conv)).items()},
            "allocator_mw_over_idle": {t: allocator_power(ov, allocator_rails(conv), measured, conv) for t, ov in ovs.items()}}

    # ---- S2, S3: the refactored nuPlan and skipping paths reproduce their shipped rows under the shipped convention
    stash = nuplan_real_stash()
    det = json.loads((RAW / "nuplan_task5" / "detect_summary.json").read_text())
    ship_cost = {"cheap": {"ms": det["ns_cheap_320"]["ms_total_median"], "mJ": det["energy"]["ns_cheap_320"]["cpu_gpu_mj_per_frame"]},
                 "640": {"ms": det["ns_full_640"]["ms_total_median"], "mJ": det["energy"]["ns_full_640"]["cpu_gpu_mj_per_frame"]}}
    s2 = compare_frames(read_back(flag_two_level(pd.DataFrame(t120.budget_track(stash, ovs["routers"], ship_cost, NBOOT)))),
                        read(FINAL / "benchmark_budget_nuplan_real.csv"), TWO_KEYS)
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    skip_cells = t128.build_cells(splits)
    ship_skip = read(FINAL / "skip_accounting.csv")
    ship_ev = ship_skip[ship_skip.section == "evaluation"]
    s3 = compare_frames(read_back(pd.DataFrame(t128.evaluation_rows(skip_cells, ovs["routers"], None, NBOOT)))
                        .drop(columns=["cascade_eta_shipped", "cascade_minus_random_lo_shipped", "cascade_minus_random_hi_shipped"]),
                        ship_ev.dropna(axis=1, how="all").drop(columns=["section", "cascade_eta_shipped",
                                                                          "cascade_minus_random_lo_shipped",
                                                                          "cascade_minus_random_hi_shipped"]),
                        ["track", "geometry", "system", "target", "unit", "budget_level"])
    costs_doc["checks"]["S2_nuplan_real"], costs_doc["checks"]["S3_skip_evaluation"] = s2, s3
    print(f"  S2 nuPlan real: {s2}\n  S3 skipping evaluation: {s3}", flush=True)
    assert s2["ok"] and s3["ok"], "a refactored path does not reproduce its shipped rows"

    # ---- recompute, mJ rows only
    cells = core_cells()
    routers = t93.load_router_scores()
    # S4: skipping two_level's draws leaves multi_fidelity on the shipped continuation (primary run, shipped convention)
    rng = np.random.default_rng(0)
    advance_past_two_level(cells, NBOOT, rng)
    s4 = compare_frames(read_back(as_93_writes(pd.DataFrame(t93.multi_fidelity(ovs["primary"], NBOOT, rng,
                                                                                 level_preds=levels)), "mf")),
                        read_back(as_93_writes(read(_latest("benchmark_budget") / "benchmark_budget_multifidelity.csv"),
                                               "mf")), MF_KEYS)
    costs_doc["checks"]["S4_multifidelity_draws"] = s4
    print(f"  S4 multi-fidelity draws: {s4}", flush=True)
    assert s4["ok"], "advancing past two_level's draws does not reproduce the shipped multi-fidelity rows"
    two_rows, mf_rows, nr_rows, skip_rows = [], [], [], []
    for table, (tag, _, with_routers) in TABLES.items():
        ov = ovs[table]
        for conv in ("module",):                                         # latency-independent tables: deciding convention only
            power = allocator_power(ov, allocator_rails(conv), measured, conv)
            rng = np.random.default_rng(0)
            t0 = time.time()
            two = flag_two_level(pd.DataFrame(t93.two_level(cells, ov, NBOOT, rng, routers if with_routers else None,
                                                            gate_scores=prov, costs=profile_costs_fn(detector_rails(conv)),
                                                            power=power, budget_units=("mJ",))))
            two_rows.append(two.assign(table=table, convention=conv))
            print(f"  two-level {table} {conv}: {len(two)} rows [{time.time() - t0:.0f}s]", flush=True)
        if with_routers:
            continue
        for conv in CONVENTIONS:
            power = allocator_power(ov, allocator_rails(conv), measured, conv)
            rng = np.random.default_rng(0)
            advance_past_two_level(cells, NBOOT, rng)
            cf = profile_costs_fn(detector_rails(conv))
            multi = pd.DataFrame(t93.multi_fidelity(ov, NBOOT, rng, level_preds=levels, costs=cf, power=power,
                                                    budget_units=("mJ",)))
            kc = cf("KITTI")["cheap"]
            mf_rows.append(flag_multifidelity(multi, {"ms": kc["ms"], "mJ": kc["mJ"]}).assign(table=table, convention=conv))
            print(f"  multi-fidelity {table} {conv}: {len(multi)} rows", flush=True)
    for conv in CONVENTIONS:
        power = allocator_power(ovs["routers"], allocator_rails(conv), measured, conv)
        nr = flag_two_level(pd.DataFrame(t120.budget_track(
            stash, ovs["routers"], nuplan_real_cost(detector_rails(conv)), NBOOT, power=power, budget_units=("mJ",),
            cost_source=f"Task 5 energy pass: rails over idle ({conv} convention) x pass time per frame")))
        nr_rows.append(nr.assign(convention=conv))
        cf = profile_costs_fn(detector_rails(conv))
        sh = t128.shares_rows(ovs["routers"], cf, power, ("mJ",))
        ev = t128.evaluation_rows(skip_cells, ovs["routers"], None, NBOOT, cf, power, ("mJ",))
        skip_rows.append(pd.DataFrame(sh + ev).assign(convention=conv))
        print(f"  nuPlan real and skipping {conv}", flush=True)

    outs = {"energy_module_budget_two_level.csv": pd.concat(two_rows, ignore_index=True),
            "energy_module_budget_multifidelity.csv": pd.concat(mf_rows, ignore_index=True),
            "energy_module_budget_nuplan_real.csv": pd.concat(nr_rows, ignore_index=True),
            "energy_module_skip_accounting.csv": pd.concat(skip_rows, ignore_index=True)}
    claims = evaluate_claims(outs, costs_doc)
    outs["energy_module_claims.csv"] = claims
    for name, df in outs.items():
        df.to_csv(FINAL / name, index=False)
        df.to_csv(run / name, index=False)
    for dst in (FINAL / "energy_module_costs.json", run / "energy_module_costs.json"):
        dst.write_text(json.dumps(costs_doc, indent=1, default=float))
    print(claims.to_string(index=False))
    print(f"  wrote {len(outs) + 1} files in {time.time() - t_all:.0f}s", flush=True)


def evaluate_claims(outs, costs_doc):
    rows = []
    shipped_mf = read(FINAL / "benchmark_budget_multifidelity.csv")
    ms_or = shipped_mf[(shipped_mf.signal == "oracle (multi-fidelity)") & (shipped_mf.system == "brake")
                       & (shipped_mf.unit == "ms")].set_index("budget_level")
    for conv in CONVENTIONS:
        # a
        nr = outs["energy_module_budget_nuplan_real.csv"]
        x = nr[(nr.convention == conv) & (nr.system == "idm") & (nr.target == "scalar_J") & (nr.signal == "R1_mlp_clf")]
        per = []
        for _, r in x.sort_values("budget_level").iterrows():
            ok = str(r.feasible) == "True" and bool(r.ndg_defined) and np.isfinite(r.minus_random_lo) and r.minus_random_lo > 0
            per.append(ok)
            rows.append({"claim": "a", "convention": conv, "budget_level": r.budget_level, "quantity": "R1_mlp_clf minus random, lower bound",
                         "value": r.minus_random_lo, "detail": f"feasible={r.feasible}; eta={r.eta:.3f}; escalated={r.escalated_frac:.3f}",
                         "holds": ok})
        rows.append({"claim": "a", "convention": conv, "quantity": "beats random at every energy budget", "holds": bool(len(per) == 4 and all(per))})
        # b
        det = costs_doc["conventions"][conv]["detector_mJ"]["KITTI"]
        c = t93.profile_costs("KITTI")
        ratio_mj, ratio_ms = det["384"] / det["640"], c["384"]["ms"] / c["640"]["ms"]
        rows.append({"claim": "b", "convention": conv, "quantity": "C384/C640 in mJ", "value": ratio_mj,
                     "detail": f"ms ratio {ratio_ms:.3f}", "holds": bool(ratio_mj < ratio_ms)})
        mf = outs["energy_module_budget_multifidelity.csv"]
        o = mf[(mf.convention == conv) & (mf.table == "primary") & (mf.signal == "oracle (multi-fidelity)") & (mf.system == "brake")]
        holds_ii = None
        for _, r in o.sort_values("budget_level").iterrows():
            ms_share = float(ms_or.loc[r.budget_level, "share_384"])
            more = bool(r.share_384 > ms_share)
            overrun = r.extra_ms_per_frame / r.extra_ms_budget_at_same_level - 1
            if np.isclose(r.budget_level, 0.2):
                holds_ii = more
            rows.append({"claim": "b", "convention": conv, "budget_level": r.budget_level,
                         "quantity": "braking oracle share at 384: energy budget vs latency budget", "value": r.share_384,
                         "detail": f"latency-budget oracle {ms_share:.4f}; latency overrun of the energy plan {overrun:+.1%}",
                         "holds": more})
        rows.append({"claim": "b", "convention": conv, "quantity": "holds (ratio below the ms ratio and more 384 at 20%)",
                     "holds": bool(ratio_mj < ratio_ms and holds_ii)})
        # c
        sk = outs["energy_module_skip_accounting.csv"]
        ev = sk[(sk.convention == conv) & (sk.section == "evaluation")]
        wins, losses = int(ev.beats_random.astype(bool).sum()), int(ev.loses_to_random.astype(bool).sum())
        rows.append({"claim": "c", "convention": conv, "quantity": "energy evaluation rows: wins / losses / rows", "value": wins,
                     "detail": f"losses {losses}; rows {len(ev)}; losing rows: " + "; ".join(
                         f"{r.track} {r.geometry} {r.system} {r.budget_level:.0%}" for _, r in ev[ev.loses_to_random.astype(bool)].iterrows()),
                     "holds": bool(wins == 0 and losses == 2)})
        rows.append({"claim": "c", "convention": conv, "quantity": "reading: R2 does not beat random in any energy row",
                     "holds": bool(wins == 0)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
