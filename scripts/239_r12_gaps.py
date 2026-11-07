#!/usr/bin/env python
"""Follow-up round 3, R12 (post hoc, a sensitivity): gaps left by R7, R8 and R10 (addendum R12 in docs/followup_protocol.md).

  --stage a  reference x loss terms, by recombination of Jterm_*; gate R12-0 first -> partA_rows.csv, reading_partA.csv
  --stage b  KITTI under lead_b4; gate R12-1 first -> partB_rows.csv, partB_no_estimate.csv, reading_partB.csv
  --stage b_rows  Part B's rows and reading alone, from the saved tables
  --stage c  routers from R7's saved scores -> partC_rows.csv
  --stage d  log overlap and frames an oracle uses -> partD_log_overlap.csv, partD_frames_used.csv
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, sys, time

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):                                   # before numpy, as 236
    os.environ.setdefault(_k, "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, geometry as G, planner as P, runmeta, xl                  # noqa: E402
from rap.paths import CACHE, RESULTS                                                # noqa: E402
from rap.risk import RiskConfig                                                     # noqa: E402
from rap.scoring import EPS                                                         # noqa: E402

OUT = Path(RESULTS) / "followup" / "R12"
WORK = Path(CACHE) / "xl" / "r12"
REB = Path(RESULTS) / "followup"
NPROC = int(os.environ.get("R12_PROC", 3))                                          # beside R11
B0, C0 = 0.12, 6.0
TERMS = ("shortfall", "collision", "excess", "jerk")
LOSSES = [(f"over-braking {b}", dict(b=b, c=6.0, jerk=1.0, excess=1.0)) for b in (0.0, 0.06, 0.12, 0.24, 0.48)]
LOSSES += [(f"collision {c:g}", dict(b=0.12, c=float(c), jerk=1.0, excess=1.0)) for c in (0, 12)]
LOSSES += [("no-jerk", dict(b=0.12, c=6.0, jerk=0.0, excess=1.0)), ("safety-only", dict(b=0.12, c=6.0, jerk=0.0, excess=0.0))]
RULES = [("standard", "-"), ("b4", "ref+ctrl"), ("b4", "ref-only")]


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


t235 = _load("t235", "235_r8_reference.py")
t234 = t235.t234


def table(rule, a, geometry):
    return pd.read_pickle(t235.TABLES / f"{geometry}__{xl.PAIR}.pkl") if rule == "standard" else t235.vtable(rule, a, geometry)


def weighted(x, t, w):
    return {"shortfall": x[f"Jterm_{t}_shortfall"].to_numpy(float), "collision": w["c"] / C0 * x[f"Jterm_{t}_collision"].to_numpy(float),
            "excess": w["excess"] * w["b"] / B0 * x[f"Jterm_{t}_excess"].to_numpy(float), "jerk": w["jerk"] * x[f"Jterm_{t}_jerk"].to_numpy(float)}


def loss_row(x, lab, w, log_of):
    c, f = weighted(x, "cheap", w), weighted(x, "full", w)
    jc, jf = sum(c.values()), sum(f.values())
    v = jc - jf
    seq = x.seq.to_numpy()
    other = sum(np.abs(c[k] - f[k]) for k in ("shortfall", "collision", "excess"))
    jo = other <= EPS
    harm, helped = v < -EPS, v > EPS
    extra = {"harmed_jerk_only": int((harm & jo).sum()), "harmed_jerk_only_share": float((harm & jo).sum() / max(harm.sum(), 1)),
             "helped_jerk_only": int((helped & jo).sum()), "helped_jerk_only_share": float((helped & jo).sum() / max(helped.sum(), 1))}
    return ({**lab, **extra}, v, jc, seq, np.array([log_of[s] for s in seq]))


def stage_a(args):
    runmeta.new_run("r12_a", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    sp = t234.meta()
    log_of = dict(zip(sp.scene, sp.log))
    pops = t234.populations(sp)
    use = ("pooled850", "new228_test", "frozen85")

    def tasks_of(rule, a):
        out = []
        for geometry in xl.GEOMETRIES:
            d = table(rule, a, geometry)
            for t in ("cheap", "full"):
                assert (d[[f"Jterm_{t}_{k}" for k in TERMS]].sum(axis=1) - d[f"J_{t}"]).abs().max() < 1e-9
            for pop in use:
                x = d[d.seq.isin(pops[pop])]
                for name, w in LOSSES:
                    out.append(loss_row(x, {"rule": rule, "apply": a, "geometry": geometry, "population": pop, "loss": name}, w, log_of))
        return out
    with Pool(NPROC) as p:
        std = pd.DataFrame(p.map(t234._a_task, tasks_of("standard", "-"), chunksize=4))
    # gate R12-0, before any variant row
    rd = lambda f: pd.read_csv(f, float_precision="round_trip")
    bad, n = [], 0

    def cmp(ours, ref, tag):
        nonlocal n
        cols = [c for c in ours.index if c in ref.index and isinstance(ours[c], (int, float, np.integer, np.floating))]
        assert len(cols) >= 10, tag
        for c in cols:
            n += 1
            a_, b_ = float(ours[c]), float(ref[c])
            if not (a_ == b_ or (np.isnan(a_) and np.isnan(b_))):
                bad.append([tag, c, a_, b_])
    h = rd(REB / "R7" / "partH_weights.csv")
    for _, r in h.iterrows():
        name = {("over-braking", True): f"over-braking {r.over_braking_weight}", ("collision", True): f"collision {r.collision_weight:g}"}.get((r.family, True))
        if name is None or name not in dict(LOSSES) or dict(LOSSES)[name]["b"] != r.over_braking_weight or dict(LOSSES)[name]["c"] != r.collision_weight:
            continue
        q = std[(std.geometry == "mono") & (std.population == r.population) & (std.loss == name)]
        assert len(q) == 1, name
        cmp(q.iloc[0], r, f"R7 {r.population} {name}")
    c6 = rd(REB / "R6" / "controls.csv")
    for _, r in c6[c6.control.isin(["standard (shared 0.25)", "safety-only loss (shortfall + collision)"])].iterrows():
        q = std[(std.geometry == r.geometry) & (std.population == {"all850": "pooled850"}.get(r.population, r.population))
                & (std.loss == ("safety-only" if r.control.startswith("safety") else "over-braking 0.12"))]
        assert len(q) == 1
        cmp(q.iloc[0], r, f"R6 {r.geometry} {r.population} {r.control}")
    s8 = rd(REB / "R8" / "partC_settings.csv")
    for _, r in s8[(s8.rule == "standard") & (s8.population == "frozen85")].iterrows():
        q = std[(std.geometry == r.geometry) & (std.population == "frozen85") & (std.loss == "over-braking 0.12")]
        assert len(q) == 1
        cmp(q.iloc[0], r, f"R8 {r.geometry} frozen85")
    rep = {"compared": n, "differ": len(bad), "first": bad[:10], "pass": not bad}
    (OUT / "gate_r12_0.json").write_text(json.dumps(rep, indent=1))
    print("gate R12-0:", rep, flush=True)
    if bad:
        raise SystemExit(1)
    with Pool(NPROC) as p:
        var = pd.DataFrame(p.map(t234._a_task, tasks_of("b4", "ref+ctrl") + tasks_of("b4", "ref-only"), chunksize=4))
    t = pd.concat([std, var], ignore_index=True)
    t.to_csv(OUT / "partA_rows.csv", index=False)
    rows = []
    for pop in ("new228_test", "pooled850"):
        r = t[(t["rule"] == "b4") & (t["apply"] == "ref+ctrl") & (t.geometry == "mono") & (t.population == pop) & (t.loss == "safety-only")].iloc[0]
        if r.harmed_share < 0.30 or r.harmed_share_lo < 0.20:
            read = "harm in safety terms shrinks"
        elif r.harmed_share_lo >= 0.30 and r.rho > 0:
            read = "harm in safety terms persists"
        else:
            read = "unclear"
        rows.append({"population": pop, "harmed": r.harmed, "affected": r.affected, "harmed_share": r.harmed_share, "lo": r.harmed_share_lo, "hi": r.harmed_share_hi,
                     "lo_log": r.harmed_share_lo_log, "hi_log": r.harmed_share_hi_log, "rho": r.rho, "reading": read})
    pd.DataFrame(rows).to_csv(OUT / "reading_partA.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def kitti_build(seqs, geometry, lead=None, ctrl=False):
    cfg, pp, cp = RiskConfig(), P.PlannerParams(), P.CostParams()
    return decision.build(Path(CACHE) / "det", "cheap_320", "full_640", seqs, cfg, pp, cp, G.PRIMARY, range_source=geometry, adapter=decision.KittiAdapter, lead=lead, lead_ctrl=ctrl)


def kitti_frozen(geometry):
    t = pd.read_pickle(Path(xl.t92().CORE) / f"KITTI__YOLOv8s__cheap_320tofull_640__{geometry}.pkl")
    return t.sort_values(["seq", "frame"]).reset_index(drop=True)


def _key(d):
    return d.sort_values(["seq", "frame"]).reset_index(drop=True)


def _kt(a):
    g, lead, ctrl = a
    seqs = sorted(kitti_frozen(g).seq.unique())
    t0 = time.time()
    d = _key(kitti_build(seqs, g, lead, ctrl))
    print(f"  KITTI {g} {lead} ctrl={ctrl}: {len(d)} frames [{time.time() - t0:.0f}s]", flush=True)
    return d


def stage_b(args):
    runmeta.new_run("r12_b", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    rep, fails = {}, []
    with Pool(2) as p:
        off = p.map(_kt, [(g, None, False) for g in xl.GEOMETRIES])
    for g, d in zip(xl.GEOMETRIES, off):
        ref = kitti_frozen(g)
        same = [c for c in d.columns if c in ref.columns]
        bad = [c for c in same if not d[c].equals(ref[c])]
        rep[g] = {"frames": len(d), "columns": len(same), "differ": bad}
        fails += bad
    rep["pass"] = not fails
    (OUT / "gate_r12_1.json").write_text(json.dumps(rep, indent=1))
    print("gate R12-1:", rep, flush=True)
    if fails:
        raise SystemExit(1)
    jobs = [(g, "b4", c) for g in xl.GEOMETRIES for c in (True, False)]
    with Pool(NPROC) as p:
        tabs = p.map(_kt, jobs)
    for (g, _, c), d in zip(jobs, tabs):
        d.to_pickle(WORK / f"kitti_b4_{'ref_ctrl' if c else 'ref_only'}_{g}.pkl")
    cfg = RiskConfig()
    no = []
    for s in sorted(kitti_frozen("mono").seq.unique()):
        g = decision.KittiAdapter.geometry(s)
        g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
        no.append({"seq": s, "objects": len(g), "no_estimate": int((~g["has_rate"]).sum()), "tracks": len(set(g["track_id"])),
                   "tracks_no_estimate": len(set(g["track_id"][~g["has_rate"]]) - set(g["track_id"][g["has_rate"]]))})
    n = pd.DataFrame(no)
    n = pd.concat([n, pd.DataFrame([{"seq": "all 21", **n.drop(columns="seq").sum().to_dict()}])], ignore_index=True)
    n["share_no_estimate"] = n.no_estimate / n.objects
    n.to_csv(OUT / "partB_no_estimate.csv", index=False)
    print(n.tail(1).to_string(index=False), flush=True)
    stage_b_rows(args)


def stage_b_rows(args):
    """Part B's rows from the frozen and the saved lead_b4 tables (the cached tier, which has no KITTI labels)."""
    OUT.mkdir(parents=True, exist_ok=True)
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    test = set(splits["kitti"]["test"])
    tasks, dirs = [], []
    items = [("standard", "-", g, kitti_frozen(g)) for g in xl.GEOMETRIES]
    items += [("b4", "ref+ctrl" if c else "ref-only", g, pd.read_pickle(WORK / f"kitti_b4_{'ref_ctrl' if c else 'ref_only'}_{g}.pkl")) for g in xl.GEOMETRIES for c in (True, False)]
    for rule, a, g, d in items:
        for pop, keep in (("all 21", np.ones(len(d), bool)), ("test", d.seq.isin(test).to_numpy())):
            x = d[keep]
            v, c, seq = (x.J_cheap - x.J_full).to_numpy(float), x.J_cheap.to_numpy(float), x.seq.astype(str).to_numpy()
            lab = {"rule": rule, "apply": a, "geometry": g, "population": pop}
            tasks.append((lab, v, c, seq))
            dirs.append({**lab, **t235.harm_direction(x)})
    with Pool(NPROC) as p:
        rows = p.map(_b_task, tasks, chunksize=1)
    t = pd.DataFrame(rows).merge(pd.DataFrame(dirs), on=["rule", "apply", "geometry", "population"], validate="one_to_one")
    t.to_csv(OUT / "partB_rows.csv", index=False)
    rd = []
    for pop in ("all 21", "test"):
        r = t[(t["rule"] == "b4") & (t["apply"] == "ref+ctrl") & (t.geometry == "mono") & (t.population == pop)].iloc[0]
        rd.append({"population": pop, "harmed": r.harmed, "affected": r.affected, "harmed_share": r.harmed_share, "lo": r.harmed_share_lo, "hi": r.harmed_share_hi, "rho": r.rho,
                   "reading": "persists" if r.harmed_share >= 0.30 and r.rho > 0 else "does not persist"})
    pd.DataFrame(rd).to_csv(OUT / "reading_partB.csv", index=False)
    print(t[["rule", "apply", "geometry", "population", "affected", "harmed", "harmed_share", "rho", "all_full_reduction", "oracle20_reduction", "under_share", "over_share"]].round(3).to_string(index=False), flush=True)
    print(pd.DataFrame(rd).to_string(index=False), flush=True)


def _b_task(t):
    lab, v, c, seq = t
    return {**lab, "n_units": len(set(seq)), **t234.nboot_describe(v, c, seq)}


def stage_c(args):
    runmeta.new_run("r12_c", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    cell = t234.Cell("mono")
    tt = cell.test
    v, j, seq, log = cell.v_all[tt], cell.j[tt], cell.seq[tt], cell.log[tt]
    full = {k: x[tt] for k, x in cell.s.items()}
    res = []
    for rt in ("R1_gbm_reg", "R1_mlp_reg"):
        for tg in t234.TARGETS:
            row = {"row": f"{rt} on {tg}", "signal": (rt, "raw", tg), "kfun": t234.k_round, "krand": t234.k_round, "budget": "20%"}
            if tg != "V":
                row["ref"] = (rt, "raw", "V")
            ask = "D_share" if tg != "V" else "over_random_count_share"
            s_map = {k: full[k] for k in (row["signal"], row.get("ref")) if k is not None}
            base = t234.quantities(t234.value(row, s_map, v, j, seq))
            dd = {}
            for u in sorted(set(seq)):
                m = seq != u
                dd[u] = t234.quantities(t234.value(row, {k: x[m] for k, x in s_map.items()}, v[m], j[m], seq[m]))
            dq = pd.DataFrame(dd).T
            worst = dq[ask].idxmin() if base[ask] >= 0 else dq[ask].idxmax()
            sides = {}
            for name, rem in (("all", set()), (f"minus the most influential scene ({worst})", {worst})):
                m = ~np.isin(seq, sorted(rem))
                sm = {k: x[m] for k, x in s_map.items()}
                q = t234.value(row, sm, v[m], j[m], seq[m])
                r = {"row": row["row"], "router": rt, "target": tg, "claim_quantity": ask, "set": name, "n_frames": int(m.sum()), "escalated": q["k"], "gain": q["gain"],
                     "all_cheap_loss": q["tot"], **t234.quantities(q)}
                r.update(t234.intervals(row, sm, v[m], j[m], seq[m], seq[m], q["tot"]))
                r.update({k + "_log": x for k, x in t234.intervals(row, sm, v[m], j[m], seq[m], log[m], q["tot"]).items()})
                sides[name == "all"] = (r[ask + "_lo"], r[ask + "_hi"])
                res.append(r)
            lo, hi = sides[True]
            lo2, hi2 = sides[False]
            side = ("not resolved before removal" if lo <= 0 <= hi else "same side" if (lo > 0 and lo2 > 0) or (hi < 0 and hi2 < 0) else "reaches zero")
            res[-1]["side_after_removal"] = side
            res[-2]["side_after_removal"] = side
            print(f"  {row['row']}: {base[ask]:+.4f} [{lo:+.4f}, {hi:+.4f}] -> [{lo2:+.4f}, {hi2:+.4f}] {side}", flush=True)
    pd.DataFrame(res).to_csv(OUT / "partC_rows.csv", index=False)


def stage_d(args):
    runmeta.new_run("r12_d", vars(args))
    OUT.mkdir(parents=True, exist_ok=True)
    sp = t234.meta()
    new = sp[sp.group == "new765"]
    test = new[new.deep_split == "test"]
    num = lambda s: int(re.search(r"(\d+)$", s).group(1))
    rows = []
    for name, fit in (("537 (train + validation)", new[new.deep_split.isin(["train", "val"])]), ("385 (train)", new[new.deep_split == "train"])):
        keys = set(zip(fit.log, fit.scene.map(num)))
        fl = set(fit.log)
        adj = [(l, n) for l, n in zip(test.log, test.scene.map(num)) if (l, n - 1) in keys or (l, n + 1) in keys]
        rows.append({"fit": name, "logs_in_xl": sp.log.nunique(), "logs_in_765": new.log.nunique(), "fit_scenes": len(fit), "fit_logs": len(fl), "test_scenes": len(test), "test_logs": test.log.nunique(),
                     "logs_in_both": len(fl & set(test.log)), "test_scenes_with_log_in_fit": int(test.log.isin(fl).sum()), "test_scenes_with_adjacent_fit_scene": len(adj)})
    pd.DataFrame(rows).to_csv(OUT / "partD_log_overlap.csv", index=False)
    print(pd.DataFrame(rows).T.to_string(), flush=True)
    pops = t234.populations(sp)
    used = []

    def add(track, rule, a, g, pop, x):
        v = (x.J_cheap - x.J_full).to_numpy(float)
        used.append({"track": track, "rule": rule, "apply": a, "geometry": g, "population": pop, "frames": len(v), "k_20": t234.k_round(len(v)), "v_positive": int((v > EPS).sum()), "share": float((v > EPS).mean())})
    for rule, a in RULES[:2]:
        for g in xl.GEOMETRIES:
            d = table(rule, a, g)
            for pop, u in pops.items():
                add("nuScenes XL", rule, a, g, pop, d[d.seq.isin(u)])
    splits = json.loads((ROOT / "configs" / "benchmark_splits.json").read_text())
    tst = set(splits["kitti"]["test"])
    for rule, a in RULES[:2]:
        for g in xl.GEOMETRIES:
            d = kitti_frozen(g) if rule == "standard" else pd.read_pickle(WORK / f"kitti_b4_ref_ctrl_{g}.pkl")
            add("KITTI", rule, a, g, "all 21", d)
            add("KITTI", rule, a, g, "test", d[d.seq.isin(tst)])
    pd.DataFrame(used).to_csv(OUT / "partD_frames_used.csv", index=False)
    print(pd.DataFrame(used).round(4).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=("a", "b", "b_rows", "c", "d"))
    args = ap.parse_args()
    globals()[f"stage_{args.stage}"](args)


if __name__ == "__main__":
    main()
