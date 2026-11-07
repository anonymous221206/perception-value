#!/usr/bin/env python
"""Follow-up round 3, R8: does the braking reference need a moving-lead rule? (addendum R8 in docs/followup_protocol.md)

  --stage gate    gate R8-0: the rebuild with the variant off equals the XL tables and the frozen 85 tables bit for bit
  --stage tables  decision.build on the 850 scenes: lead_b4 and lead_cv, each ref+ctrl and ref-only, both geometries
  --stage parta   face validity per a* band under the standard rule and both variants
  --stage parts   Table-1 rows, harm direction, drop-scene rows on each variant
  --stage partd   primary variant: X2's gate_gbm row, R1's four D_t, R3's chosen subset, with influence rows; reading
"""
from __future__ import annotations

import argparse, importlib.util, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, geometry as G, planner as P, runmeta, xl                # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                     # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL, NUSCENES_XL, RESULTS              # noqa: E402
from rap.risk import RiskConfig                                                   # noqa: E402

OUT = Path(RESULTS) / "followup" / "R8"
WORK = Path(CACHE) / "xl" / "r8"
DET = Path(CACHE) / "nusc_det_xl"
CHEAP, FULL = "ns_cheap_320", "ns_full_640"
TABLES = Path(CACHE) / "xl" / "tables850"
FROZEN = Path(CACHE) / "xl" / "frozen85_tables"
PRIMARY = ("b4", "ref+ctrl")
VARIANTS = [(l, a) for l in ("b4", "cv") for a in ("ref+ctrl", "ref-only")]
BANDS = (("[1, 3.5)", 1.0, 3.5), ("[3.5, 7)", 3.5, 7.0), ("[7, 9]", 7.0, 9.01))
NPROC = 6


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


t234 = _load("t234", "234_r7_influence.py")


def seqs_850():
    return sorted(xl.scene_split().scene)


def vtable(lead, apply_, geometry):
    return pd.read_pickle(WORK / f"table_{lead}_{apply_.replace('+', '_')}_{geometry}.pkl")


def stage_gate(args):
    run = {}
    fails = []
    nad = make_adapter(NuScenesDB(args.dataroot, "v1.0-trainval"))
    pp, cp, cfg = P.PlannerParams(), P.CostParams(), RiskConfig()
    for geometry in xl.GEOMETRIES:
        t0 = time.time()
        d = decision.build(DET, CHEAP, FULL, seqs_850(), cfg, pp, cp, G.PRIMARY, range_source=geometry, adapter=nad)
        ref = pd.read_pickle(TABLES / f"{geometry}__{xl.PAIR}.pkl")
        d, ref = d.sort_values(["seq", "frame"]).reset_index(drop=True), ref.sort_values(["seq", "frame"]).reset_index(drop=True)
        same = [c for c in d.columns if c in ref.columns]
        bad = [c for c in same if not d[c].equals(ref[c])]
        run[f"xl_{geometry}"] = {"frames": len(d), "columns": len(same), "differ": bad}
        fails += bad
        print(f"  xl {geometry}: {len(d)} frames, {len(same)} columns, differ {bad} [{time.time() - t0:.0f}s]", flush=True)
    nf = make_adapter(NuScenesDB(str(NUSCENES_TRAINVAL), "v1.0-trainval"))
    for geometry in xl.GEOMETRIES:
        ref = pd.read_pickle(FROZEN / f"{geometry}__{xl.PAIR}.pkl").sort_values(["seq", "frame"]).reset_index(drop=True)
        d = decision.build(Path(CACHE) / "nusc_det_tv", CHEAP, FULL, sorted(ref.seq.unique()), cfg, pp, cp, G.PRIMARY,
                           range_source=geometry, adapter=nf).sort_values(["seq", "frame"]).reset_index(drop=True)
        same = [c for c in d.columns if c in ref.columns]
        bad = [c for c in same if not d[c].equals(ref[c])]
        run[f"frozen85_{geometry}"] = {"frames": len(d), "columns": len(same), "differ": bad}
        fails += bad
        print(f"  frozen85 {geometry}: {len(d)} frames, differ {bad}", flush=True)
    run["pass"] = not fails
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_r8_0.json").write_text(json.dumps(run, indent=1))
    print("gate R8-0:", run["pass"], flush=True)
    if fails:
        raise SystemExit(1)


def stage_tables(args):
    nad = make_adapter(NuScenesDB(args.dataroot, "v1.0-trainval"))
    pp, cp, cfg = P.PlannerParams(), P.CostParams(), RiskConfig()
    WORK.mkdir(parents=True, exist_ok=True)
    for lead, a in VARIANTS:
        for geometry in xl.GEOMETRIES:
            f = WORK / f"table_{lead}_{a.replace('+', '_')}_{geometry}.pkl"
            if f.exists():
                continue
            t0 = time.time()
            d = decision.build(DET, CHEAP, FULL, seqs_850(), cfg, pp, cp, G.PRIMARY, range_source=geometry, adapter=nad,
                               lead=lead, lead_ctrl=a == "ref+ctrl")
            d.to_pickle(f)
            print(f"  {f.name}: {len(d)} frames [{time.time() - t0:.0f}s]", flush=True)


def stage_parta(args):
    """Per frame, the constraining reference object under each rule, then the bands (v >= 3 m/s)."""
    nad = make_adapter(NuScenesDB(args.dataroot, "v1.0-trainval"))
    pp, cfg = P.PlannerParams(), RiskConfig()
    std = pd.read_pickle(TABLES / f"mono__{xl.PAIR}.pkl").sort_values(["seq", "frame"]).reset_index(drop=True)
    e10 = pd.read_pickle(Path(CACHE) / "xl" / "e10" / "tables" / "nuScenes__ns_full_640__ego__mono.pkl")[["seq", "frame", "dt"]]
    std = std.merge(e10, on=["seq", "frame"], how="left", validate="one_to_one")
    std["t"] = std.groupby("seq").dt.transform(lambda x: x.fillna(0).cumsum())
    std["v_later"] = std.groupby("seq").v_ego.shift(-4)
    std["t_later"] = std.groupby("seq").t.shift(-4)
    std["acc_2s"] = (std.v_later - std.v_ego) / (std.t_later - std.t)
    rows = []
    for rule in (None, "b4", "cv"):
        vals = []
        for s, g_ in std.groupby("seq", sort=True):
            geom = nad.geometry(s)
            for r in g_[["frame", "v_ego"]].itertuples():
                g = geom[geom["frame"] == r.frame]
                g = g[(g["y2"] - g["y1"]) >= cfg.min_gt_height]
                rr = np.where(g["has_rate"], g["range_rate"], np.nan)
                a, ao = P.required_decel(g["long_near"], g["lat_min"], g["lat_max"], g["ttc"], r.v_ego, pp, lead=rule,
                                         range_rate=None if rule is None else rr)
                if len(g) and a > 0:
                    i = int(np.argmax(ao))
                    spd = float(P.lead_speed(g["long_near"][i:i + 1], g["ttc"][i:i + 1], r.v_ego, rr[i:i + 1])[0])
                    vals.append((a, spd, float(g["long_near"][i])))
                else:
                    vals.append((a, np.nan, np.nan))
        x = std.assign(a_rule=[q[0] for q in vals], obj_speed=[q[1] for q in vals], obj_range=[q[2] for q in vals])
        x["headway"] = x.obj_range / x.v_ego.clip(lower=0.1)
        lab = "standard" if rule is None else f"lead_{rule}"
        if rule is None:
            assert np.abs(x.a_rule - x.a_gt).max() < 1e-9
        fast = x[x.v_ego >= 3.0]

        def block(name, y):
            fut = y[y.acc_2s.notna()]
            rows.append({"rule": lab, "band": name, "frames": len(y), "scenes": y.seq.nunique(), "no_2s_future": int(y.acc_2s.isna().sum()),
                         "mean_logged_acc_2s": fut.acc_2s.mean(), "share_le_-1": float((fut.acc_2s <= -1).mean()),
                         "share_le_-2": float((fut.acc_2s <= -2).mean()), "obj_speed_mean": y.obj_speed.mean(),
                         "share_obj_moving_gt_2": float((y.obj_speed > 2).mean()), "obj_range_median": y.obj_range.median(),
                         "headway_median_s": y.headway.median()})
        for name, lo, hi in BANDS:
            block(name, fast[(fast.a_rule >= lo) & (fast.a_rule < hi)])
        if rule is None:
            block("collision flag, either branch", x[(x.collision_cheap > 0) | (x.collision_full > 0)])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "partA_face_validity.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False), flush=True)


def harm_direction(d):
    v = (d.J_cheap - d.J_full).to_numpy(float)
    h = v < 0
    m = -v[h]
    under = (d.act_full.to_numpy() < d.act_cheap.to_numpy())[h]
    over = (d.act_full.to_numpy() > d.act_cheap.to_numpy())[h]
    return {"harm_mass": float(m.sum()), "under_share": float(m[under].sum() / m.sum()), "over_share": float(m[over].sum() / m.sum()),
            "other_share": float(m[~under & ~over].sum() / m.sum()), "under_frames": int(under.sum()), "over_frames": int(over.sum())}


def stage_parts(args):
    sp = t234.meta()
    pops = t234.populations(sp)
    log_of = dict(zip(sp.scene, sp.log))
    tasks, dirs, names, drops = [], [], [], []
    for lead, a in [("standard", "-")] + VARIANTS:
        for geometry in xl.GEOMETRIES:
            d = (pd.read_pickle(TABLES / f"{geometry}__{xl.PAIR}.pkl") if lead == "standard" else vtable(lead, a, geometry))
            for pop, u in pops.items():
                x = d[d.seq.isin(u)]
                v, c, seq = (x.J_cheap - x.J_full).to_numpy(float), x.J_cheap.to_numpy(float), x.seq.to_numpy()
                lab = {"rule": lead, "apply": a, "geometry": geometry, "population": pop}
                tasks.append((dict(lab), v, c, seq, np.array([log_of[s] for s in seq])))
                dirs.append({**lab, **harm_direction(x)})
            if geometry == "mono" and lead != "standard":
                t234.drop_rows(d, {"rule": lead, "apply": a, "geometry": geometry, "population": "pooled850"}, sp, "J_cheap", "J_full", drops, names)
    with Pool(NPROC) as p:
        rows = p.map(t234._a_task, tasks, chunksize=2)
        rows2 = p.map(t234._a_task, drops, chunksize=2)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "partC_settings.csv", index=False)
    pd.DataFrame(dirs).to_csv(OUT / "partC_harm_direction.csv", index=False)
    pd.DataFrame(rows2).to_csv(OUT / "partC_drop_scene.csv", index=False)
    pd.DataFrame(names).to_csv(OUT / "partC_drop_scene_names.csv", index=False)
    r = pd.DataFrame(rows)
    print(r[(r.geometry == "mono") & r.population.isin(["new228_test", "pooled850"])][["rule", "apply", "population", "affected", "harmed_share", "rho",
          "all_full_reduction"]].round(3).to_string(index=False), flush=True)


def stage_partd(args):
    t227 = _load("t227", "227_r1_target_swap.py")
    t231 = _load("t231", "231_r3_gate_cost.py")
    a = argparse.Namespace(det=str(t227.DET), dataroot=args.dataroot)
    cells_, log_of = t227.setup(a)
    c = next(x for x in cells_ if x["geometry"] == "mono" and x["system"] == "brake")
    d = c["d"]
    var = vtable(*PRIMARY, "mono")[["seq", "frame", "J_cheap", "J_full"]].astype({"seq": str, "frame": int})
    j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(var, on=["seq", "frame"], how="left", validate="one_to_one")
    v, jc = (j.J_cheap - j.J_full).to_numpy(float), j.J_cheap.to_numpy(float)
    fit, test = d.split.isin(["train", "val"]).to_numpy(), (d.split == "test").to_numpy()
    seq = d.seq.astype(str).to_numpy()
    log = np.array([log_of[s] for s in seq])
    cell = t234.Cell("mono")
    assert (cell.seq == seq).all()
    s_v = xl.gate_scores(d, c["fcols"], v, fit, which=("gate_gbm",))["gate_gbm"]
    sel = json.loads((Path(RESULTS) / "followup" / "R3" / "selected.json").read_text())["gate_gbm"]["subset"]
    s_c = t231.fit_predict(d, t231.subset_columns(sel, t231.group_columns(c["fcols"])), v, fit, "gate_gbm")[1]
    pd.DataFrame({"seq": seq, "frame": d.frame.to_numpy(), "split": d.split.to_numpy(), "gate_gbm_V_b4": s_v, f"gate_gbm_{sel}_b4": s_c}).to_csv(
        OUT / "scores_primary_variant.csv.gz", index=False, float_format="%.17g")
    rows_ = t234.rows_for(cell)
    res, drop = [], []
    for row in rows_:
        row = dict(row)
        if row["row"].startswith("X2"):
            mp = {row["signal"]: s_v[test]}
        elif row.get("ref"):
            mp = {row["signal"]: cell.s[row["signal"]][test], row["ref"]: s_v[test]}
        elif "gate_gbm" in row["row"]:
            mp = {row["signal"]: s_c[test]}
        else:
            continue
        vt, jt, st, lt = v[test], jc[test], seq[test], log[test]
        ask = "D_share" if row.get("ref") else "over_random_rate_share"
        dd = {}
        for u in sorted(set(st)):
            m = st != u
            dd[u] = t234.quantities(t234.value(row, {k: x[m] for k, x in mp.items()}, vt[m], jt[m], st[m]))
        dq = pd.DataFrame(dd).T
        base = t234.quantities(t234.value(row, mp, vt, jt, st))
        worst = dq[ask].idxmin() if base[ask] >= 0 else dq[ask].idxmax()
        two = sorted(set(st) & set(t234.meta().query("log in @t234.LOGS").scene))
        for name, rem in (("all", set()), ("minus scene-1051", {"scene-1051"}), ("minus scene-1081", {"scene-1081"}),
                          ("minus both", {"scene-1051", "scene-1081"}), ("minus test scenes of the two logs", set(two)),
                          (f"minus the most influential scene ({worst})", {worst})):
            m = ~np.isin(st, sorted(rem))
            sm = {k: x[m] for k, x in mp.items()}
            q = t234.value(row, sm, vt[m], jt[m], st[m])
            r = {"row": row["row"], "set": name, "escalated": q["k"], "gain": q["gain"], "all_cheap_loss": q["tot"], **t234.quantities(q)}
            r.update(t234.intervals(row, sm, vt[m], jt[m], st[m], st[m], q["tot"]))
            r.update({k + "_log": x for k, x in t234.intervals(row, sm, vt[m], jt[m], st[m], lt[m], q["tot"]).items()})
            res.append(r)
        print(f"  {row['row']}: {base}", flush=True)
    t = pd.DataFrame(res)
    t.to_csv(OUT / "partD_allocators.csv", index=False)
    st = pd.read_csv(OUT / "partC_settings.csv")
    p_ = st[(st.rule == "b4") & (st["apply"] == "ref+ctrl") & (st.geometry == "mono")]
    rd = []
    for pop in ("new228_test", "pooled850"):
        r = p_[p_.population == pop].iloc[0]
        rd.append({"claim": f"sign variation persists, {pop}", "harmed": r.harmed_share, "harmed_lo": r.harmed_share_lo, "harmed_hi": r.harmed_share_hi,
                   "rho": r.rho, "rho_lo": r.rho_lo, "rho_hi": r.rho_hi, "reading": "persists" if r.harmed_share >= 0.30 and r.rho > 0 else "does not persist"})
    al = t[t.set == "all"]
    for r in al.itertuples():
        q = "D_share" if pd.notna(getattr(r, "D_share", np.nan)) else "over_random_rate_share"
        lo, hi = getattr(r, f"{q}_lo"), getattr(r, f"{q}_hi")
        rd.append({"claim": r.row, "value": getattr(r, q), "lo": lo, "hi": hi, "lo_log": getattr(r, f"{q}_lo_log"), "hi_log": getattr(r, f"{q}_hi_log"),
                   "reading": "carries" if lo > 0 else "does not carry"})
    pd.DataFrame(rd).to_csv(OUT / "reading.csv", index=False)
    print(pd.DataFrame(rd).round(3).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gate", "tables", "parta", "parts", "partd"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    runmeta.new_run(f"r8_{args.stage}", vars(args))
    {"gate": stage_gate, "tables": stage_tables, "parta": stage_parta, "parts": stage_parts, "partd": stage_partd}[args.stage](args)


if __name__ == "__main__":
    main()
