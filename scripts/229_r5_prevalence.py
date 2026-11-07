#!/usr/bin/env python
"""Follow-up round 2, R5: when escalation helps -- prevalence, event onsets, scene counts (addendum R5 in docs/followup_protocol.md).

  --stage pairs       the E5 replacement pairs (ten) and the two fine-tuned pairs at their primary operating points,
                      braking tables on the 228 new test scenes (decision.build, monocular geometry, shared action
                      history) -> data/cache/xl/r5/pair__<old>__<new>.pkl
  --stage prevalence  per scene p_s (share of frames with a* >= 1.0 m/s^2) and mean V per frame; Spearman over the 850
                      scenes (primary: braking, monocular) with its scene-bootstrap interval; deciles of p_s; the
                      relative reduction per scene; oracle geometry; the E5 pairs on the 228 test scenes
                      -> prevalence_scenes.csv, prevalence_deciles.csv
  --stage events      event onsets per hour under all-cheap, all-full, oracle at 20%, gate_gbm at 20% (X2's fit, test
                      scenes) and random at 20% (20 seeds) -> events.csv
  --stage scenes      among scenes with an affected frame, the share with sum V < 0 (all 850) -> scene_counts.csv
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import decision, geometry as G, planner as P, runmeta, xl              # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                               # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

OUT = Path(RESULTS) / "followup" / "R5"
WORK = Path(CACHE) / "xl" / "r5"
TABLES = Path(CACHE) / "xl" / "tables850"
E10 = Path(CACHE) / "xl" / "e10" / "tables" / "nuScenes__ns_cheap_320__ego__mono.pkl"
MODELS = Path(CACHE) / "xl_models"
PP = P.PlannerParams()
A_DECEL, A_HARD = PP.thr_decel, PP.thr_hard            # 1.0 and 3.5 m/s^2: the controller's decelerate / brake regions
GAP = 0.75
NBOOT = 1000


def populations():
    sp = xl.scene_split()
    return sorted(sp.scene), set(sp[(sp.group == "new765") & (sp.deep_split == "test")].scene)


def table(geometry):
    d = pd.read_pickle(TABLES / f"{geometry}__{xl.PAIR}.pkl")
    return d.assign(V=d.J_cheap - d.J_full)


# ------------------------------------------------------------------------------------------------ pairs

def e5_pairs():
    """The ten E5 replacement pairs and the two fine-tuned pairs, each at its primary operating point (x7 tables)."""
    up = pd.read_csv(Path(RESULTS) / "followup" / "E5" / "x7_upgrades.csv")
    ft = pd.read_csv(Path(RESULTS) / "followup" / "E5" / "x7_finetuned.csv")
    out = []
    for t, tag in ((up, "replacement"), (ft, "fine-tuned")):
        p = t[t.primary & (t.geometry == "mono")].drop_duplicates("pair")
        out += [(r.old, r.new, float(r.full_threshold), tag) for r in p.itertuples()]
    return out


def stage_pairs(args):
    from rap.nusc import NuScenesDB, make_adapter
    _, test = populations()
    nad = make_adapter(NuScenesDB(args.dataroot, "v1.0-trainval"))
    pp, cp = P.PlannerParams(), P.CostParams()
    WORK.mkdir(parents=True, exist_ok=True)
    for old, new, t_full, tag in e5_pairs():
        f = WORK / f"pair__{old}__{new}.pkl"
        if f.exists():
            continue
        t0 = time.time()
        cfg = RiskConfig(op_conf=0.25, op_conf_full=t_full)
        d = decision.build(MODELS, old, new, sorted(test), cfg, pp, cp, G.PRIMARY, range_source="mono", adapter=nad)
        d.assign(pair_kind=tag).to_pickle(f)
        print(f"  {old} -> {new} ({tag}, full threshold {t_full:.4f}): {len(d)} frames [{time.time() - t0:.0f}s]", flush=True)


# ------------------------------------------------------------------------------------------------ prevalence

def per_scene(d):
    g = d.assign(dec=(d.a_gt >= A_DECEL).astype(float)).groupby("seq")
    s = pd.DataFrame({"frames": g.size(), "p_s": g.dec.mean(), "v_bar": g.V.mean(), "sum_V": g.V.sum(),
                      "sum_J_cheap": g.J_cheap.sum(), "affected": g.V.apply(lambda x: int((x != 0).sum()))})
    s["rel_reduction"] = np.where(s.sum_J_cheap > 0, s.sum_V / s.sum_J_cheap.where(s.sum_J_cheap > 0), np.nan)
    return s.reset_index()


def spearman_ci(x, y, nboot=NBOOT):
    """Spearman's rho over scenes and its scene-bootstrap interval (1,000 draws, seed 0, every draw kept)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(0)
    draws = np.empty(nboot)
    for b in range(nboot):
        i = rng.integers(0, len(x), len(x))
        draws[b] = spearmanr(x[i], y[i]).correlation
    lo, hi = xl.ci(draws)
    return rho, lo, hi, int(np.isnan(draws).sum())


def deciles(s):
    """Ten groups of scenes by p_s (ascending; ties by scene name), pooled sum V / sum J_cheap per group with a
    scene-bootstrap interval inside the group."""
    s = s.sort_values(["p_s", "seq"]).reset_index(drop=True)
    s["decile"] = np.arange(len(s)) * 10 // len(s) + 1
    rows = []
    for k, g in s.groupby("decile"):
        rng = np.random.default_rng(0)
        v, c = g.sum_V.to_numpy(), g.sum_J_cheap.to_numpy()
        dr = []
        for _ in range(NBOOT):
            i = rng.integers(0, len(g), len(g))
            dr.append(v[i].sum() / c[i].sum() if c[i].sum() > 0 else np.nan)
        lo, hi = xl.ci(np.array(dr))
        rows.append({"decile": int(k), "scenes": len(g), "p_s_min": g.p_s.min(), "p_s_max": g.p_s.max(),
                     "p_s_mean": g.p_s.mean(), "sum_V": v.sum(), "sum_J_cheap": c.sum(),
                     "all_full_reduction": v.sum() / c.sum() if c.sum() > 0 else np.nan, "lo": lo, "hi": hi})
    return s, pd.DataFrame(rows)


def stage_prevalence(args):
    _, test = populations()
    scenes, summ, dec = [], [], []
    cases = [("mono", "XL 320->640", "all850", table("mono")), ("oracle", "XL 320->640", "all850", table("oracle"))]
    for f in sorted(WORK.glob("pair__*.pkl")):
        d = pd.read_pickle(f)
        _, old, new = f.stem.split("__", 2)
        cases.append(("mono", f"{d.pair_kind.iloc[0]}: {old} -> {new}", "new228_test", d.assign(V=d.J_cheap - d.J_full)))
    for geometry, cell, pop, d in cases:
        s = per_scene(d)
        s, dd = deciles(s)
        lab = {"geometry": geometry, "cell": cell, "population": pop}
        scenes.append(s.assign(**lab))
        dec.append(dd.assign(**lab, table="decile"))
        primary = geometry == "mono" and cell == "XL 320->640"
        for stat, sub in (("v_bar", s), ("rel_reduction", s[s.sum_J_cheap > 0])):
            rho, lo, hi, nan = spearman_ci(sub.p_s, sub[stat])
            summ.append({**lab, "table": "spearman", "statistic": f"rho(p_s, {stat})", "scenes": len(sub), "rho": rho,
                         "lo": lo, "hi": hi, "draws_undefined": nan,
                         "primary": bool(primary and stat == "v_bar"),
                         "resolved": bool(lo > 0 or hi < 0)})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(scenes).to_csv(OUT / "prevalence_scenes.csv", index=False)
    out = pd.concat([pd.DataFrame(summ), pd.concat(dec)], ignore_index=True)
    out.to_csv(OUT / "prevalence_deciles.csv", index=False)
    print(pd.DataFrame(summ)[["geometry", "cell", "statistic", "scenes", "rho", "lo", "hi", "primary"]].to_string(index=False))


# ------------------------------------------------------------------------------------------------ events

def frame_intervals(d):
    """Each frame's interval to the next keyframe of its scene (E10's per-frame dt, shifted); a scene's last frame gets
    the scene's median interval."""
    e = pd.read_pickle(E10)[["seq", "frame", "dt"]]
    d = d.merge(e, on=["seq", "frame"], how="left", validate="one_to_one").sort_values(["seq", "frame"])
    nxt = d.groupby("seq").dt.shift(-1)
    med = d.groupby("seq").dt.transform("median")
    d["interval"] = nxt.fillna(med)
    d["gap_before"] = d.dt
    return d


EVENTS = {"unnecessary hard braking": lambda a, cmd: (cmd == P.HARD_BRAKE) & (a < A_DECEL),
          "missed braking": lambda a, cmd: (a >= A_HARD) & (cmd == P.KEEP),
          "any braking or deceleration while a* < 1.0 (secondary)": lambda a, cmd: (cmd >= P.DECELERATE) & (a < A_DECEL),
          "command below the level a* requires (secondary)":
              lambda a, cmd: cmd < np.array([P.discrete_action(x, PP) for x in a])}


def onsets(d, flag):
    """Episode onsets: a flagged frame starts an episode unless the previous frame of the same scene is flagged and at
    most 0.75 s before it."""
    prev_flag = flag.groupby(d.seq).shift(1).fillna(False).astype(bool)
    cont = prev_flag & (d.gap_before <= GAP)
    return flag & ~cont


def select_top(score, d, frac, tiebreak):
    """Top round(frac * n) frames by score, ties broken by (scene, frame)."""
    k = int(round(frac * len(d)))
    order = np.lexsort((tiebreak[1], tiebreak[0], -score))
    m = np.zeros(len(d), bool)
    m[order[:k]] = True
    return m


def event_rows(d, policies, pop):
    rows = []
    a = d.a_gt.to_numpy(float)
    hours = d.interval.sum() / 3600.0
    for pol, esc_list in policies.items():
        for ev, fn in EVENTS.items():
            counts = []
            for esc in esc_list:
                cmd = np.where(esc, d.act_full.to_numpy(int), d.act_cheap.to_numpy(int))
                flag = pd.Series(fn(a, cmd), index=d.index)
                counts.append(int(onsets(d, flag).sum()))
            rows.append({"population": pop, "policy": pol, "event": ev, "onsets": float(np.mean(counts)),
                         "hours": hours, "onsets_per_hour": float(np.mean(counts)) / hours,
                         "draws": len(esc_list), "frames": len(d), "scenes": d.seq.nunique()})
    return rows


def stage_events(args):
    _, test = populations()
    d = frame_intervals(table("mono"))
    rows = []
    for pop, m in (("new228_test", d.seq.isin(test).to_numpy()), ("all850", np.ones(len(d), bool))):
        x = d[m].reset_index(drop=True)
        tb = (x.seq.to_numpy(), x.frame.to_numpy())
        V = x.V.to_numpy(float)
        pol = {"all-cheap": [np.zeros(len(x), bool)], "all-full": [np.ones(len(x), bool)],
               "oracle at 20%": [select_top(V, x, 0.2, tb)]}
        if pop == "new228_test":
            pol["gate_gbm at 20% (X2's fit)"] = [select_top(gate_scores_test(x), x, 0.2, tb)]
            k = int(round(0.2 * len(x)))
            rnd = []
            for seed in range(20):
                e = np.zeros(len(x), bool)
                e[np.random.default_rng(seed).choice(len(x), k, replace=False)] = True
                rnd.append(e)
            pol["random at 20% (mean of 20 seeds)"] = rnd
        rows += event_rows(x, pol, pop)
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "events.csv", index=False)
    print(t.pivot_table(index=["population", "policy"], columns="event", values="onsets_per_hour").round(2).to_string())


def gate_scores_test(x):
    """X2's gate_gbm (fit on the 385 + 152 new scenes, braking, monocular), scored on the frames of x."""
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    xl.use_ego_table(TABLES / "v_ego_causal.csv")
    feats, fcols = xl.features(str(Path(CACHE) / "nusc_det_xl"))
    c = next(c for c in xl.cells(TABLES, feats, fcols, split_of, geometries=("mono",)) if c["system"] == "brake")
    dd = c["d"]
    fit = dd.split.isin(["train", "val"]).to_numpy()
    s = xl.gate_scores(dd, fcols, (dd[c["cheap"]] - dd[c["full"]]).to_numpy(float), fit, which=("gate_gbm",))["gate_gbm"]
    sc = pd.DataFrame({"seq": dd.seq.astype(str), "frame": dd.frame.astype(int), "s": s})
    j = x[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(sc, on=["seq", "frame"], how="left", validate="one_to_one")
    assert j.s.notna().all()
    return j.s.to_numpy(float)


# ------------------------------------------------------------------------------------------------ scene counts

def stage_scenes(args):
    rows = []
    for geometry in ("mono", "oracle"):
        s = per_scene(table(geometry))
        a = s[s.affected > 0]
        neg = (a.sum_V < 0).to_numpy(float)
        rng = np.random.default_rng(0)
        dr = [neg[rng.integers(0, len(a), len(a))].mean() for _ in range(NBOOT)]
        lo, hi = xl.ci(np.array(dr))
        rows.append({"geometry": geometry, "cell": "XL 320->640, braking", "population": "all850",
                     "scenes": len(s), "scenes_with_affected_frame": len(a), "scenes_sum_V_negative": int(neg.sum()),
                     "share_sum_V_negative": float(neg.mean()), "lo": lo, "hi": hi,
                     "scenes_sum_V_positive": int((a.sum_V > 0).sum()), "scenes_sum_V_zero": int((a.sum_V == 0).sum()),
                     "primary_cell": geometry == "mono"})
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "scene_counts.csv", index=False)
    print(t.to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pairs", "prevalence", "events", "scenes"])
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    runmeta.new_run(f"r5_{args.stage}", vars(args))
    {"pairs": stage_pairs, "prevalence": stage_prevalence, "events": stage_events, "scenes": stage_scenes}[args.stage](args)


if __name__ == "__main__":
    main()
