#!/usr/bin/env python
"""Follow-up E3 (c): routing at a matched budget in closed loop (addendum E3c in docs/followup_protocol.md).

  --stage fit       (simulation machine, the CARLA environment's scikit-learn) from the development runs
                    (ticks_rdev.jsonl of each machine, 202 with ROUTER=dev): one row per complete 10-tick frame -> gate features at the
                    frame's first tick, decision value V = sum over the frame's ticks of J(cheap action) - J(full
                    action), the benchmark's braking controller and loss (Eq. 10) with a* from the true actors.
                    Gate = the benchmark's gradient-boosted regressor (E2's gate_gbm) on V; tau = the lowest
                    out-of-fold prediction (5 folds by route) at which the share of frames at or above it is within
                    the budget. The final model is fit on all frames.
                    -> <router_dir>/gate.pkl, gate_dev.json, gate_dev_frames.csv
  --stage outcomes  from the routing checkpoints route_<id>_rt_<gate|random>_r<rep>.json and the ladder's cheap and
                    full runs on the same routes: per policy driving score, success, route completion, collisions;
                    paired differences of route means (route bootstrap, 1,000 draws, seed 0); realised escalation
                    share per run from ticks_rt_<policy>_r<rep>.jsonl, and the machine that ran it (the tick log's
                    folder); per policy and machine, the same means (kind policy_machine), as a check
                    -> results/followup/E3/{routing.csv, routing_runs.csv, routing_escalation.csv}
  --stage sensitivity  before any routing run, from the final ladder on the test routes (all but the first --n_dev
                    by id): the run-to-run noise of a route mean over three repetitions (pooled within-route variance
                    of driving score over branches), the noise SD of a difference of two policies' route means,
                    sqrt(2 s2 / 3), and the minimum detectable difference 2.8 x SD / sqrt(routes) (80% power,
                    two-sided 5%); also the headroom full - cheap on these routes, with its route bootstrap
                    -> results/followup/E3/routing_sensitivity.json
"""
from __future__ import annotations

import argparse, importlib.util, json, lzma, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import planner as Pl                                                    # noqa: E402

FRAME = 10
PP, CP = Pl.PlannerParams(), Pl.CostParams()
GATE_COLS = ["n_det", "n_cand", "conf_sum", "conf_max", "ent_sum", "unc_sum", "area_max", "area_sum", "bottom_max",
             "near_dist", "ego_speed"]
POLICIES = ["gate", "random"]
PAIRS = [("gate", "random"), ("gate", "cheap"), ("random", "cheap"), ("full", "gate"), ("full", "random"),
         ("full", "cheap")]
METRICS = ["driving_score", "success", "route_completion"]


def chain_J(a_req, a_star):
    """Per-tick loss of the braking controller acting on a_req, against a_star; jerk against its own last action."""
    out, prev = np.zeros(len(a_req)), None
    for i, (r, s) in enumerate(zip(a_req, a_star)):
        act = Pl.discrete_action(float(r), PP)
        out[i] = Pl.decision_cost(act, float(s), prev, PP, CP)["J"]
        prev = act
    return out


def read_runs(path):
    """One JSON line per run; a line that does not parse (an interrupted write) is skipped and counted. A tick log
    may also be its xz-compressed copy (.jsonl.xz)."""
    p = Path(path)
    text = lzma.open(p, "rt").read() if p.suffix == ".xz" else p.read_text()
    out, bad = [], 0
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"{path}: {bad} unreadable lines skipped")
    return out


def dev_frames(ticks_paths):
    runs = {}
    for path in ticks_paths:
        for r in read_runs(path):
            # a route run more than once (a crash, then a rerun, possibly on the other machine) keeps the run that
            # drove longest: the crashed run stopped early, a run that failed at set-up has no tick
            if len(r["ticks"]) >= len(runs.get(r["routes"], {"ticks": []})["ticks"]):
                runs[r["routes"]] = r
    rows = []
    for route, r in runs.items():
        t = pd.DataFrame(r["ticks"])
        if not {"a_req_cheap", "a_req_full"} <= set(t.columns):
            continue
        ok = t.a_req_cheap.notna() & t.a_req_full.notna()
        t = t[ok].copy()
        t["Jc"] = chain_J(t.a_req_cheap.to_numpy(float), t.a_star.to_numpy(float))
        t["Jf"] = chain_J(t.a_req_full.to_numpy(float), t.a_star.to_numpy(float))
        t["frame"] = t.tick // FRAME
        for k, g in t.groupby("frame"):
            first = g[g.tick == k * FRAME]
            if len(g) < FRAME or first.empty or not isinstance(first.gate.iloc[0], dict):
                continue
            rows.append({"route": route, "frame": int(k), **first.gate.iloc[0],
                         "V": float((g.Jc - g.Jf).sum()), "Jc": float(g.Jc.sum()), "Jf": float(g.Jf.sum())})
    return pd.DataFrame(rows)


def stage_fit(args):
    from scipy import stats
    from sklearn.model_selection import GroupKFold
    from rap import predict
    d = dev_frames(args.ticks)
    out = Path(args.router_dir)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "gate_dev_frames.csv", index=False)
    X, v, g = d[GATE_COLS].to_numpy(float), d.V.to_numpy(float), d.route.to_numpy()
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits=5).split(X, v, g):
        oof[te] = predict.make_model("gbm", "reg", 0).fit(X[tr], v[tr]).predict(X[te])
    # tau: the lowest out-of-fold prediction at which the share of frames at or above it stays within the budget
    # (ties never push the share over it; no such value -> the gate never escalates)
    srt = np.sort(oof)
    share = 1 - np.searchsorted(srt, srt, side="left") / len(srt)
    ok = share <= args.budget
    tau = float(srt[ok].min()) if ok.any() else float("inf")
    model = predict.make_model("gbm", "reg", 0).fit(X, v)
    with open(out / "gate.pkl.tmp", "wb") as fh:     # renamed into place: other machines wait for gate.pkl
        pickle.dump({"model": model, "cols": GATE_COLS, "tau": tau, "budget": args.budget}, fh)
    (out / "gate.pkl.tmp").replace(out / "gate.pkl")
    sel = oof >= tau
    k = int(round(args.budget * len(v)))
    oracle = np.sort(v)[::-1][:k].sum()
    rep = {"routes": int(d.route.nunique()), "frames": len(d), "affected_frames": int((np.abs(v) > 1e-9).sum()),
           "V_sum": float(v.sum()), "V_positive_sum": float(v[v > 0].sum()), "budget": args.budget, "tau": tau,
           "oof_escalation_share": float(sel.mean()), "oof_spearman": float(stats.spearmanr(oof, v)[0]),
           "oof_gain_at_tau": float(v[sel].sum()), "random_expected_gain": float(args.budget * v.sum()),
           "oracle_gain_at_budget": float(oracle)}
    (out / "gate_dev.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def _ladder_module():
    s = importlib.util.spec_from_file_location("b2d_ladder", ROOT / "scripts" / "200_b2d_ladder.py")
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def boot(diff, rng):
    n = len(diff)
    return np.percentile([diff[rng.integers(0, n, n)].mean() for _ in range(1000)], [2.5, 97.5]) if n else [np.nan] * 2


def stage_outcomes(args):
    lad = _ladder_module()
    rt = lad.runs(args.out_dir, "rt")
    test = set(rt.route)
    ld = lad.runs(args.out_dir, "lad")
    ld = ld[ld.route.isin(test) & ld.branch.isin(["cheap", "full"])]
    d = pd.concat([rt, ld], ignore_index=True)
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    d.sort_values(["route", "branch", "rep"]).to_csv(out / "routing_runs.csv", index=False)
    esc = []
    for pol in POLICIES:
        root = Path(args.ticks[0])
        for f in sorted(root.rglob(f"ticks_rt_{pol}_r*.jsonl")) or sorted(root.rglob(f"ticks_rt_{pol}_r*.jsonl.xz")):
            rep = int(f.name.split(".")[0].rsplit("_r", 1)[1])
            for r in read_runs(f):
                dec = [t["escalate"] for t in r["ticks"] if "escalate" in t]
                full_ticks = sum(t.get("mode") == "full" for t in r["ticks"])
                esc.append({"route": int(r["routes"].split("_")[1]), "policy": pol, "rep": rep,
                            "machine": f.parent.name if f.parent != Path(args.ticks[0]) else "", "frames": len(dec),
                            "escalated": int(sum(dec)), "share": float(np.mean(dec)) if dec else np.nan,
                            "full_tick_share": full_ticks / max(len(r["ticks"]), 1), "ticks": len(r["ticks"])})
    e = pd.DataFrame(esc)
    if not e.empty:   # a crashed and rerun run logged twice: keep the run that drove longest
        e = e.sort_values("ticks").groupby(["route", "policy", "rep"]).tail(1)
    e.to_csv(out / "routing_escalation.csv", index=False)
    rows = []
    for b in POLICIES + ["cheap", "full"]:
        g = d[d.branch == b]
        if g.empty:
            continue
        r = {"kind": "policy", "branch": b, "runs": len(g), "routes": g.route.nunique(),
             **{m: g[m].mean() for m in METRICS},
             "vehicle_collisions_per_run": g.get("collisions_vehicle", pd.Series(dtype=float)).mean(),
             "tickruntime_stops": int((g.status == "Failed - TickRuntime").sum())}
        if b in POLICIES and not e.empty:
            ee = e[e.policy == b]
            r.update({"escalation_share": ee.escalated.sum() / max(ee.frames.sum(), 1),
                      "full_tick_share": ee.full_tick_share.mean()})
        rows.append(r)
    if not e.empty:
        dm = d.merge(e[["route", "policy", "rep", "machine"]].rename(columns={"policy": "branch"}),
                     on=["route", "branch", "rep"], how="inner")
        for (b, mach), g in dm.groupby(["branch", "machine"]):
            rows.append({"kind": "policy_machine", "branch": f"{b} @ {mach}", "runs": len(g),
                         "routes": g.route.nunique(), **{m: g[m].mean() for m in METRICS}})
    _pairs(d, "pair", rows)
    s = pd.DataFrame(rows)
    s.to_csv(out / "routing.csv", index=False)
    print(s.T.to_string())


def _pairs(d, kind, rows):
    rm = d.groupby(["branch", "route"])[METRICS].mean()
    for a, b in PAIRS:
        if a not in rm.index.get_level_values(0) or b not in rm.index.get_level_values(0):
            continue
        both = rm.loc[a].join(rm.loc[b], lsuffix="_a", rsuffix="_b", how="inner")
        row = {"kind": kind, "branch": f"{a} - {b}", "routes": len(both)}
        for m in METRICS:
            diff = (both[f"{m}_a"] - both[f"{m}_b"]).to_numpy(float)
            lo, hi = boot(diff, np.random.default_rng(0))
            row.update({m: diff.mean() if len(diff) else np.nan, f"{m}_lo": lo, f"{m}_hi": hi,
                        f"{m}_sd": diff.std(ddof=1) if len(diff) > 1 else np.nan})
        rows.append(row)


def stage_sensitivity(args):
    lad = _ladder_module()
    d = lad.runs(args.out_dir, "lad")
    ids = sorted(d.route.unique())
    test = set(ids[args.n_dev:])
    t = d[d.route.isin(test)]
    g = t.groupby(["branch", "route"]).driving_score
    s2 = float((g.var(ddof=1) * (g.count() - 1)).sum() / (g.count() - 1).sum())
    n_rep = 3
    sd_noise = float(np.sqrt(2 * s2 / n_rep))
    rm = t.groupby(["branch", "route"]).driving_score.mean()
    both = rm.loc["full"].to_frame("f").join(rm.loc["cheap"].to_frame("c"), how="inner")
    diff = (both.f - both.c).to_numpy(float)
    lo, hi = boot(diff, np.random.default_rng(0))
    n = len(test)
    rep = {"test_routes": n, "first_test_route": int(min(test)), "runs": len(t),
           "within_route_var_driving_score": s2, "sd_noise_policy_difference": sd_noise,
           "mdd_noise": 2.8 * sd_noise / np.sqrt(n), "sd_full_minus_cheap": float(diff.std(ddof=1)),
           "mdd_full_minus_cheap_sd": float(2.8 * diff.std(ddof=1) / np.sqrt(len(diff))),
           "headroom_full_minus_cheap": float(diff.mean()), "headroom_lo": float(lo), "headroom_hi": float(hi),
           "headroom_routes": len(diff)}
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "routing_sensitivity.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "outcomes", "sensitivity"], required=True)
    ap.add_argument("--ticks", nargs="+", help="fit: the ticks_rdev.jsonl files (one per machine); outcomes: the folder holding ticks_rt_*.jsonl (searched recursively)")
    ap.add_argument("--router_dir", help="fit: output folder (gate.pkl, gate_dev.json)")
    ap.add_argument("--budget", type=float, default=0.2)
    ap.add_argument("--n_dev", type=int, default=50, help="sensitivity: development routes (first by id)")
    ap.add_argument("--out_dir", help="outcomes: folder holding the routing and ladder checkpoints")
    ap.add_argument("--results", default=str(ROOT / "results" / "followup" / "E3"))
    args = ap.parse_args()
    {"fit": stage_fit, "outcomes": stage_outcomes, "sensitivity": stage_sensitivity}[args.stage](args)


if __name__ == "__main__":
    main()
