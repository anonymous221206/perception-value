#!/usr/bin/env python
"""Task 35: sensitivity of the sign variation to the downstream losses' weights (post hoc), on the shared action history.

  --part A   braking controller, loss weights one at a time: over-braking (excess) {0.06, 0.12, 0.24}, collision
             penalty {3, 6, 12}, change of command (jerk) {0, 0.02, 0.04}; the safety-shortfall weight stays 1. The
             braking action is a threshold rule on the perceived requirement and does not read the loss, so every
             variant is an exact recombination of the stored weighted terms (`Jterm_*` of the core tables, 52).
  --part B   trajectory controller (Planner B, `static_obstacles`): collision {5, 10, 20}, clearance {0.75, 1.5, 3},
             progress {0.5, 1, 2}, one at a time. The controller plans with its cost weights, so planning and
             evaluation change together: every variant is a rebuild with 65.build_b (shared history).
  --part C   braking controller with corridor half-width 1.0 and 1.5 m (default 1.2): the actions change, so the
             decision tables are rebuilt with decision.build (shared history).
  --part report   results/final/loss_sensitivity.csv from the three runs, and the markdown tables.

Settings: the braking settings of the sign table (nuScenes YOLOv8s 320->640 under oracle and monocular geometry;
every KITTI fidelity pair under monocular geometry and YOLOv8s 320->640 under oracle geometry); for B, the KITTI ones.
All units. Per variant: affected inputs (|V| > 1e-9), harmed share of affected, harm ratio rho = sum(-V-)/sum(V+)
with its unit-bootstrap interval (1,000 draws, calibration cells' `harm_stats`, a generator seeded 0 per setting and
variant), all-full and oracle@20 reductions of the all-cheap loss.

Checks (each stops the run): the default variant reproduces the core tables' losses bit for bit (A: recombined terms;
B, C: rebuilt tables), and its point values reproduce shared_history_effect.csv (shared rows, S0) and
calibration_cells.csv (S0, all units) where the setting is there.
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from dataclasses import replace
from importlib import import_module
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import frames as rap_frames                                              # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap import runs as rap_runs                                                # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL, RESULTS                        # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

FINAL = Path(RESULTS) / "final"
EPS = 1e-9
# label -> (dataset, detector (core / Planner B file name), det_dir, cheap, full)
PAIRS = {"Y8 320->640": ("KITTI", "YOLOv8s", "YOLOv8s", "det", "cheap_320", "full_640"),
         "Y8 384->640": ("KITTI", "YOLOv8s", "YOLOv8s", "det", "cheap_384", "full_640"),
         "Y8 512->640": ("KITTI", "YOLOv8s", "YOLOv8s", "det512", "cheap_512", "full_640"),
         "RT 320->640": ("KITTI", "RT-DETR-l", "RTDETRl", "rtdetr_kitti", "rt_cheap_320", "rt_full_640"),
         "RT 480->640": ("KITTI", "RT-DETR-l", "RTDETRl", "rtdetr_kitti_mid", "rt_mid_480", "rt_full_640"),
         "nuSc Y8 320->640": ("nuScenes", "YOLOv8s", None, "nusc_det_tv", "ns_cheap_320", "ns_full_640")}
BRAKE_SETTINGS = [("nuSc Y8 320->640", "oracle"), ("nuSc Y8 320->640", "mono"), ("Y8 320->640", "mono"),
                  ("Y8 320->640", "oracle"), ("Y8 384->640", "mono"), ("Y8 512->640", "mono"),
                  ("RT 320->640", "mono"), ("RT 480->640", "mono")]
TRAJ_SETTINGS = [s for s in BRAKE_SETTINGS if not s[0].startswith("nuSc")]
# calibration_cells.csv names of the settings it has
CALIB = {("nuSc Y8 320->640", "oracle"): "nuScenes oracle Y8 320->640", ("nuSc Y8 320->640", "mono"): "nuScenes mono Y8 320->640",
         ("Y8 320->640", "mono"): "KITTI mono Y8 320->640", ("Y8 320->640", "oracle"): "KITTI oracle Y8 320->640",
         ("Y8 384->640", "mono"): "KITTI mono Y8 384->640", ("Y8 512->640", "mono"): "KITTI mono Y8 512->640",
         ("RT 480->640", "mono"): "KITTI mono RT-DETR 480->640"}
# the nuScenes and moderate-gap KITTI settings (calibration's `moderate_gap_kitti`)
HEADLINE = {("nuSc Y8 320->640", "oracle"), ("nuSc Y8 320->640", "mono"), ("Y8 384->640", "mono"),
            ("Y8 512->640", "mono"), ("RT 480->640", "mono")}
A_VARIANTS = [("excess", "lam_brake", [0.06, 0.12, 0.24]), ("collision", "lam_collision", [3.0, 6.0, 12.0]),
              ("jerk", "lam_jerk", [0.0, 0.02, 0.04])]
B_VARIANTS = [("collision", "w_collision", [5.0, 10.0, 20.0]), ("clearance", "w_clearance", [0.75, 1.5, 3.0]),
              ("progress", "w_progress", [0.5, 1.0, 2.0])]
C_WIDTHS = [1.0, 1.2, 1.5]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def core_name(pair, geo):
    ds, det, _, _, cm, fm = PAIRS[pair]
    return f"{ds}__{det}__{cm}to{fm}__{geo}.pkl"


def planb_name(pair, geo):
    ds, _, det, _, cm, fm = PAIRS[pair]
    return f"planB__{ds}__{det}__{cm}to{fm}__{geo}.pkl"


def metrics(jc, jf, units, seed_key, t102, t92):
    v = np.asarray(jc, float) - np.asarray(jf, float)
    rng = np.random.default_rng(seed_key)
    h = t102.harm_stats(v, np.asarray(units), rng)
    tot = float(np.sum(jc))
    k20 = max(int(round(0.2 * len(v))), 1)
    prize = float(t92.topk_expect(v, [v], [k20])[0][0][0])
    return {"n_inputs": int(len(v)), "affected": h["affected"], "harmed": h["harmed"], "harmed_share": h["harm_rate"],
            "harmed_share_lo": h["harm_rate_lo"], "harmed_share_hi": h["harm_rate_hi"], "rho": h["rho"],
            "rho_lo": h["rho_lo"], "rho_hi": h["rho_hi"], "all_full": float(v.sum() / tot),
            "oracle20": float(prize / tot)}


def seed_of(*parts):
    """A stable per-(setting, variant) seed (Python's hash is salted per process)."""
    import zlib
    return zlib.crc32("|".join(map(str, parts)).encode())


def reference_checks(rows):
    """Default rows against shared_history_effect.csv (shared, S0) and calibration_cells.csv (S0, all)."""
    eff = pd.read_csv(FINAL / "shared_history_effect.csv")
    eff = eff[(eff.variant == "shared") & (eff.scheme == "S0")]
    cal = pd.read_csv(FINAL / "calibration_cells.csv")
    cal = cal[(cal.scheme == "S0") & (cal.split == "all")]
    out = []
    for r in rows:
        if not r["is_default"]:
            continue
        sysname = "brake" if r["system"] == "brake" else "traj"
        e = eff[(eff.pair == r["pair"]) & (eff.geometry == r["geometry"]) & (eff.system == sysname)]
        if len(e) == 1:
            e = e.iloc[0]
            for mine, ref in (("affected", "affected"), ("harmed_share", "harmed_share_affected"), ("rho", "rho"),
                              ("all_full", "all_full_reduction"), ("oracle20", "oracle20_reduction")):
                d = abs(float(r[mine]) - float(e[ref]))
                out.append({"part": r["part"], "setting": r["setting"], "reference": "shared_history_effect.csv",
                            "quantity": mine, "mine": r[mine], "shipped": e[ref], "abs_diff": d, "pass": d <= 1e-12})
        key = (r["pair"], r["geometry"])
        if key in CALIB:
            c = cal[cal.cell == f"{CALIB[key]} q_{sysname}"]
            if len(c) == 1:
                c = c.iloc[0]
                for mine, ref in (("affected", "affected"), ("harmed_share", "harm_rate"), ("rho", "rho"),
                                  ("all_full", "all_full_reduction"), ("oracle20", "oracle20_reduction")):
                    d = abs(float(r[mine]) - float(c[ref]))
                    out.append({"part": r["part"], "setting": r["setting"], "reference": "calibration_cells.csv",
                                "quantity": mine, "mine": r[mine], "shipped": c[ref], "abs_diff": d,
                                "pass": d <= 1e-12})
    return out


# ------------------------------------------------------------------------------------------------------------ A

def part_a(run, t102, t92):
    core = rap_runs.core_matrix("postreview")
    from rap import planner as P
    cp = P.CostParams()
    rows, checks = [], []
    for pair, geo in BRAKE_SETTINGS:
        d = pd.read_pickle(core / core_name(pair, geo))
        base = {k: getattr(cp, a) for k, a, _ in A_VARIANTS}
        for tag in ("cheap", "full"):
            # the recombination at default weights must be the table's loss, bit for bit
            j = (d[f"Jterm_{tag}_shortfall"] + d[f"Jterm_{tag}_excess"] + d[f"Jterm_{tag}_jerk"]
                 + d[f"Jterm_{tag}_collision"]).to_numpy(float)
            diff = float(np.max(np.abs(j - d[f"J_{tag}"].to_numpy(float))))
            checks.append({"setting": f"{pair}|{geo}", "check": f"terms sum to J_{tag}", "abs_diff": diff,
                           "pass": diff == 0.0})
        for term, attr, values in A_VARIANTS:
            for val in values:
                w = dict(base)
                w[term] = val
                jj = {}
                for tag in ("cheap", "full"):
                    jj[tag] = (d[f"Jterm_{tag}_shortfall"].to_numpy(float)
                               + d[f"Jterm_{tag}_excess"].to_numpy(float) * (w["excess"] / base["excess"])
                               + d[f"Jterm_{tag}_jerk"].to_numpy(float) * (w["jerk"] / base["jerk"])
                               + d[f"Jterm_{tag}_collision"].to_numpy(float) * (w["collision"] / base["collision"]))
                is_def = val == base[term]
                if is_def:
                    jj = {tag: d[f"J_{tag}"].to_numpy(float) for tag in ("cheap", "full")}
                m = metrics(jj["cheap"], jj["full"], d.seq.astype(str).to_numpy(), seed_of("A", pair, geo, term, val),
                            t102, t92)
                rows.append({"part": "A", "system": "brake", "setting": f"{pair}|{geo}", "pair": pair, "geometry": geo,
                             "parameter": term, "value": val, "default_value": base[term], "is_default": is_def, **m})
        print(f"  A {pair:18s} {geo:6s} done", flush=True)
    return rows, checks


# ------------------------------------------------------------------------------------------------------------ B, C

_W = {}


def _init(dataset):
    m52 = import_module("52_core_matrix")
    _W.update(m52=m52, m65=import_module("65_planner_b_decision"))
    _W["adapter"] = (m52.make_adapter(m52.NuScenesDB(str(NUSCENES_TRAINVAL), "v1.0-trainval")) if dataset == "nuScenes"
                     else m52.decision.KittiAdapter)
    seqdir = (CACHE / "nusc_det_tv" / "ns_cheap_320") if dataset == "nuScenes" else (CACHE / "det" / "cheap_320")
    _W["seqs"] = [p.stem for p in sorted(seqdir.glob("*.npz"))]


def build_task(part, pair, geo, param, val, outdir):
    t0 = time.time()
    m52, m65 = _W["m52"], _W["m65"]
    _, _, _, det_dir, cm, fm = PAIRS[pair]
    cfg = RiskConfig(op_conf=0.25)
    if part == "B":
        B = m65.B
        cb = B.COSTS_B["default"] if param == "default" else replace(B.COSTS_B["default"], **{param: val})
        d = m65.build_b(CACHE / det_dir, cm, fm, _W["seqs"], cfg, _W["adapter"], geo, B.PARAMS_B["static_obstacles"], cb)
        d = d[["seq", "frame", "JB_cheap", "JB_full"]].rename(columns={"JB_cheap": "Jc", "JB_full": "Jf"})
    else:
        P = m52.P
        pp = P.PlannerParams() if param == "default" else replace(P.PlannerParams(), corridor_half_w=val)
        d = m52.decision.build(CACHE / det_dir, cm, fm, _W["seqs"], cfg, pp, P.CostParams(), m52.G.PRIMARY,
                               range_source=geo, adapter=_W["adapter"])
        d = d[["seq", "frame", "J_cheap", "J_full"]].rename(columns={"J_cheap": "Jc", "J_full": "Jf"})
    d["seq"] = d.seq.astype(str)
    name = f"{part}__{pair.replace(' ', '_').replace('->', 'to')}__{geo}__{param}_{val}.pkl"
    d.to_pickle(outdir / name)
    return part, pair, geo, param, val, name, time.time() - t0


def part_bc(run, part, workers, t102, t92):
    outdir = run / "tables"
    outdir.mkdir(exist_ok=True)
    defaults = {"w_collision": 10.0, "w_clearance": 1.5, "w_progress": 1.0, "corridor_half_w": 1.2}
    grid = ([(attr, vals) for _, attr, vals in B_VARIANTS] if part == "B" else [("corridor_half_w", C_WIDTHS)])
    settings = TRAJ_SETTINGS if part == "B" else BRAKE_SETTINGS
    # one default build per setting, shared by every parameter's default row
    jobs = [(pair, geo, "default", 0.0) for pair, geo in settings]
    jobs += [(pair, geo, attr, v) for pair, geo in settings for attr, vals in grid for v in vals if v != defaults[attr]]
    kitti = [j for j in jobs if PAIRS[j[0]][0] == "KITTI"]
    nusc = [j for j in jobs if PAIRS[j[0]][0] == "nuScenes"]
    done = []
    with get_context("spawn").Pool(workers, initializer=_init, initargs=("KITTI",)) as pool:
        futs = [pool.apply_async(build_task, (part, *j, outdir)) for j in kitti]
        if nusc:
            _init("nuScenes")
            for j in nusc:
                done.append(build_task(part, *j, outdir))
                print(f"  {part} {done[-1][5]} [{done[-1][6]:.0f}s]", flush=True)
        for f in futs:
            done.append(f.get())
            print(f"  {part} {done[-1][5]} [{done[-1][6]:.0f}s]", flush=True)
    idx = pd.DataFrame(done, columns=["part", "pair", "geometry", "parameter", "value", "file", "seconds"])
    idx.to_csv(run / "build_index.csv", index=False)

    core, planb = rap_runs.core_matrix("postreview"), rap_runs.planner_b()
    rows, checks = [], []
    names = {"w_collision": "collision", "w_clearance": "clearance", "w_progress": "progress",
             "corridor_half_w": "corridor_half_width"}
    expanded = []
    for _, r in idx.iterrows():
        if r.parameter == "default":
            expanded += [{**r.to_dict(), "parameter": attr, "value": defaults[attr], "_default": True} for attr, _ in grid]
        else:
            expanded.append({**r.to_dict(), "_default": False})
    for r in [pd.Series(x) for x in expanded]:
        d = pd.read_pickle(outdir / r.file)
        is_def = bool(r._default)
        if is_def and r.parameter == grid[0][0]:     # the default rebuild must be the shipped table, bit for bit
            if part == "B":
                s = pd.read_pickle(planb / planb_name(r.pair, r.geometry))[["seq", "frame", "JB_cheap", "JB_full"]]
                s.columns = ["seq", "frame", "Jc", "Jf"]
            else:
                s = pd.read_pickle(core / core_name(r.pair, r.geometry))[["seq", "frame", "J_cheap", "J_full"]]
                s.columns = ["seq", "frame", "Jc", "Jf"]
            s["seq"] = s.seq.astype(str)
            j = s.merge(d, on=["seq", "frame"], suffixes=("_s", "_r"), validate="one_to_one")
            diff = float(max(np.max(np.abs(j.Jc_s - j.Jc_r)), np.max(np.abs(j.Jf_s - j.Jf_r))))
            checks.append({"setting": f"{r.pair}|{r.geometry}", "check": f"default rebuild = shipped table",
                           "n": len(j), "n_shipped": len(s), "n_rebuilt": len(d), "abs_diff": diff,
                           "pass": diff == 0.0 and len(j) == len(s) == len(d)})
        m = metrics(d.Jc.to_numpy(float), d.Jf.to_numpy(float), d.seq.to_numpy(),
                    seed_of(part, r.pair, r.geometry, names[r.parameter], float(r.value)), t102, t92)
        rows.append({"part": part, "system": "traj" if part == "B" else "brake", "setting": f"{r.pair}|{r.geometry}",
                     "pair": r.pair, "geometry": r.geometry, "parameter": names[r.parameter], "value": float(r.value),
                     "default_value": defaults[r.parameter], "is_default": is_def, **m})
    return rows, checks


# ------------------------------------------------------------------------------------------------------------ report

def report():
    parts = []
    for part in ("A", "B", "C"):
        runs = rap_runs.frame_runs(f"loss_sensitivity_{part}")
        if runs:
            parts.append(pd.read_csv(runs[-1] / "sensitivity.csv"))
            ch = json.loads((runs[-1] / "checks.json").read_text())
            assert ch["all_pass"], f"part {part}: checks failed in {runs[-1].name}"
    df = pd.concat(parts, ignore_index=True)
    df["headline_setting"] = [(p, g) in HEADLINE for p, g in zip(df.pair, df.geometry)]
    df = df.sort_values(["part", "setting", "parameter", "value"]).reset_index(drop=True)
    df.to_csv(FINAL / "loss_sensitivity.csv", index=False)
    rng = df.groupby(["part", "system", "setting"]).agg(
        harmed_share_min=("harmed_share", "min"), harmed_share_max=("harmed_share", "max"),
        rho_min=("rho", "min"), rho_max=("rho", "max"), headline=("headline_setting", "first")).reset_index()
    print(rng.to_string(index=False))
    write_markdown(df, rng)
    low = df[df.headline_setting & (df.harmed_share < 0.30)]
    print("\n  headline rows with harmed share < 30%:", len(low))
    if len(low):
        print(low[["part", "setting", "parameter", "value", "harmed_share", "rho"]].to_string(index=False))
    print("  wrote", FINAL / "loss_sensitivity.csv")


def write_markdown(df, rng):
    """docs/loss_sensitivity_tables.md: every variant of every part, and the ranges per setting."""
    pct = lambda x: f"{100 * x:.1f}%"                                          # noqa: E731
    head = {"A": "A. Braking controller, loss weights (exact recombination)",
            "B": "B. Trajectory controller, plan and loss weights (rebuilt)",
            "C": "C. Braking controller, corridor half-width in m (rebuilt)"}
    md = ["# Loss-sensitivity tables (generated by scripts/167_loss_sensitivity.py --part report; do not edit by hand)", "",
          "Harmed = share of affected inputs with V < 0; rho = sum of harm / sum of benefit, with its 95% unit-bootstrap "
          "interval; all-full and oracle@20 = reductions of the all-cheap loss. Bold = default. All units, shared "
          "action history.", ""]
    for part in ("A", "B", "C"):
        x = df[df.part == part]
        if x.empty:
            continue
        md += [f"## {head[part]}", "", "| setting | parameter | value | affected | harmed | rho [95%] | all-full | oracle@20 |",
               "|---|---|---|---|---|---|---|---|"]
        for _, r in x.iterrows():
            v = f"**{r.value:g}**" if r.is_default else f"{r.value:g}"
            md.append(f"| {r.setting} | {r.parameter} | {v} | {int(r.affected)} | {pct(r.harmed_share)} | "
                      f"{r.rho:.3f} [{r.rho_lo:.3f}, {r.rho_hi:.3f}] | {pct(r.all_full)} | {pct(r.oracle20)} |")
        md.append("")
    md += ["## Ranges across variants, per setting", "",
           "| part | setting | headline | harmed share | rho |", "|---|---|---|---|---|"]
    for _, r in rng.iterrows():
        md.append(f"| {r.part} | {r.setting} | {'yes' if r.headline else 'no'} | {pct(r.harmed_share_min)}–"
                  f"{pct(r.harmed_share_max)} | {r.rho_min:.3f}–{r.rho_max:.3f} |")
    (ROOT / "docs" / "loss_sensitivity_tables.md").write_text("\n".join(md) + "\n")
    print("  wrote", ROOT / "docs" / "loss_sensitivity_tables.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["A", "B", "C", "report"])
    ap.add_argument("--workers", type=int, default=3)
    rap_frames.add_argument(ap)
    args = ap.parse_args()
    rap_frames.configure(args)
    if args.part == "report":
        return report()
    t102, t92 = _load("t102", "102_calibration_cells.py"), _load("t92", "92_benchmark_table.py")
    run = runmeta.new_run(f"loss_sensitivity_{args.part}", vars(args))
    t0 = time.time()
    if args.part == "A":
        rows, checks = part_a(run, t102, t92)
    else:
        rows, checks = part_bc(run, args.part, args.workers, t102, t92)
    checks += reference_checks(rows)
    pd.DataFrame(rows).to_csv(run / "sensitivity.csv", index=False)
    ok = all(c["pass"] for c in checks)
    pd.DataFrame(checks).to_csv(run / "checks.csv", index=False)
    (run / "checks.json").write_text(json.dumps({"all_pass": ok, "n_checks": len(checks),
                                                 "seconds": round(time.time() - t0)}, indent=1))
    for c in checks:
        if not c["pass"]:
            print("  FAIL", c, flush=True)
    print(f"  part {args.part}: {len(rows)} rows, {len(checks)} checks, all_pass={ok}, {time.time() - t0:.0f}s, {run}",
          flush=True)
    if not ok:
        raise SystemExit("checks failed")


if __name__ == "__main__":
    main()
