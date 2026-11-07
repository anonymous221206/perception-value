#!/usr/bin/env python
"""Follow-up E3 (b): frame sample, fork jobs and outcomes of the forked rollouts (addendum E3 in docs/followup_protocol.md).

  --stage sample    from the two baseline runs (cheap, SHADOW=1; tags base_A and base_B) of the first 40 routes (by
                    id) that complete under cheap (status of the base_A checkpoint in --out_dir not "Failed...", and a
                    base_B run present): eligible frames (10-tick blocks whose perceived-set signature differs between
                    cheap and the shadow full mode on any tick), up to 300 drawn (<= 5 per route, seed 0) plus 50
                    non-eligible -> fork_jobs.csv
  --stage outcomes  per sampled frame: V_t (single step, the 10 ticks of the frame), dJ = J(baseline window) -
                    J(fork window), dJ_AA (base_A - base_B on the same window), collision and progress differences;
                    noise floor = 95th percentile of |dJ_AA|; sign agreement above the floor with a route bootstrap
                    -> results/followup/E3/{aa_control,forked_rollout}.csv
                    Primary rows: frames whose 8 s window lies inside all three runs (A, B, fork); a run that ends
                    inside the window is reported by count, and as a sensitivity row over the span all three cover;
                    a fork whose route ended before the fork tick is excluded (docs/followup_index.md, open points).
                    Per population: eligible (primary), non-eligible (null control), both; plus diagnostics (V_t_AA,
                    the A/A analogue of V_t; the ego's distance to base_A at the fork tick, fork and base_B)
Loss per tick: the benchmark's braking loss (Eq. 10) with a = the ego's realised deceleration over the tick and a* the
braking controller's requirement from the true actors; the change term against the previous tick.
"""
from __future__ import annotations

import argparse, json, lzma, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import planner as Pl                                                    # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402

OUT = Path(RESULTS) / "followup" / "E3"
DT, FRAME, WINDOW = 0.05, 10, 160
CP = Pl.CostParams()


def read(path):
    """A tick log as written, or its xz-compressed copy (<path>.xz) when the plain file is absent."""
    p = Path(path)
    text = p.read_text() if p.exists() else lzma.open(f"{p}.xz", "rt").read()
    rows = []
    for line in text.splitlines():
        r = json.loads(line)
        rows.append(r)
    return rows


def tick_frame(rec):
    t = pd.DataFrame(rec["ticks"])
    v = t.v.to_numpy(float)
    a = np.r_[0.0, -(v[1:] - v[:-1]) / DT]                      # realised deceleration, positive when braking
    a = np.clip(a, 0.0, None)
    short = np.clip(t.a_star.to_numpy(float) - a, 0, None)
    excess = np.clip(a - t.a_star.to_numpy(float), 0, None)
    jerk = np.r_[0.0, np.abs(a[1:] - a[:-1])]
    t["L"] = (CP.lam_risk * short ** 2 + CP.lam_brake * excess ** 2 + CP.lam_jerk * jerk
              + CP.lam_collision * (short >= CP.collision_a))
    step = np.r_[0.0, np.hypot(np.diff(t.x.to_numpy(float)), np.diff(t.y.to_numpy(float)))]
    t["dist"] = step
    return t


def window(t, start, span=WINDOW):
    w = t[(t.tick >= start) & (t.tick < start + span)]
    return {"J": float(w.L.sum()), "collision": bool(w.overlap.any()), "progress": float(w.dist.sum())}


def stage_sample(args):
    OUT.mkdir(parents=True, exist_ok=True)
    base = {}
    for tag in ("base_A", "base_B"):
        for r in read(Path(args.logs) / f"ticks_{tag}.jsonl"):
            base[(tag, r["routes"])] = r
    status = {}
    for f in Path(args.out_dir).glob("route_*_base_A.json"):
        rec = json.loads(f.read_text())["_checkpoint"]["records"]
        status[f.name[: -len("_base_A.json")]] = rec[0]["status"] if rec else "missing"
    key = lambda r: int(Path(str(r)).stem.split("_")[-1])                                   # noqa: E731
    both = {k[1] for k in base if k[0] == "base_A"} & {k[1] for k in base if k[0] == "base_B"}
    done = [r for r in both if not status.get(Path(str(r)).stem, "missing").startswith(("Failed", "missing"))]
    routes = sorted(done, key=key)[:40]
    print(f"  {len(both)} routes in both baselines, {len(done)} complete under cheap; using {len(routes)}:",
          [Path(str(r)).stem for r in routes][-3:], flush=True)
    pd.Series({Path(str(r)).stem: status.get(Path(str(r)).stem, "missing") for r in sorted(both, key=key)}).to_csv(
        OUT / "baseline_status.csv", header=["status_base_A"])
    rng = np.random.default_rng(0)
    elig, non = [], []
    for rt in routes:
        t = pd.DataFrame(base[("base_A", rt)]["ticks"])
        if "shadow_sig" not in t:
            continue
        t["frame"] = t.tick // FRAME
        diff = t.groupby("frame").apply(lambda g: bool((g.sig != g.shadow_sig).any()))
        last = t.tick.max() - WINDOW
        fr = diff[diff.index * FRAME <= last]
        e, n = list(fr[fr].index), list(fr[~fr].index)
        elig += [(rt, int(f)) for f in rng.permutation(e)[:5]]
        non += [(rt, int(f)) for f in rng.permutation(n)[:2]]
    elig = elig[:300]
    non = [non[i] for i in rng.permutation(len(non))[:50]]
    jobs = pd.DataFrame([{"route": r, "frame": f, "fork_tick": f * FRAME, "eligible": True} for r, f in elig]
                        + [{"route": r, "frame": f, "fork_tick": f * FRAME, "eligible": False} for r, f in non])
    OUT.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(OUT / "fork_jobs.csv", index=False)
    print(jobs.groupby("eligible").size(), "routes:", jobs.route.nunique())


def stage_outcomes(args):
    jobs = pd.read_csv(OUT / "fork_jobs.csv")
    base = {}
    for tag in ("base_A", "base_B"):
        for r in read(Path(args.logs) / f"ticks_{tag}.jsonl"):
            if r["ticks"]:                                   # last non-empty line wins (reruns after a set-up OOM)
                base[(tag, r["routes"])] = tick_frame(r)
    forks = {}
    for r in read(Path(args.logs) / "ticks_fork.jsonl"):
        if r["ticks"]:
            forks[(r["routes"], r["fork_tick"], r["fork_branch"])] = tick_frame(r)
    rows = []
    for j in jobs.itertuples():
        A, B = base.get(("base_A", j.route)), base.get(("base_B", j.route))
        F = forks.get((j.route, j.fork_tick, "full"))
        if A is None or B is None or F is None:
            rows.append({"route": j.route, "frame": j.frame, "fork_tick": j.fork_tick, "eligible": j.eligible,
                         "status": "missing_run"})
            continue
        ends = {k: int(t.tick.max()) for k, t in (("A", A), ("B", B), ("F", F))}
        complete = min(ends.values()) >= j.fork_tick + WINDOW - 1
        reached = ends["F"] >= j.fork_tick + FRAME - 1
        # truncated windows (a run ended inside the 8 s): the sensitivity rows use the span all three runs cover
        span = WINDOW if complete else max(0, min(ends.values()) - j.fork_tick + 1)
        wa, wb, wf = window(A, j.fork_tick, span), window(B, j.fork_tick, span), window(F, j.fork_tick, span)
        first = lambda t: float(t[(t.tick >= j.fork_tick) & (t.tick < j.fork_tick + FRAME)].L.sum())   # noqa: E731
        at = lambda t: t.set_index("tick").loc[j.fork_tick, ["x", "y"]].to_numpy(float) \
            if j.fork_tick in set(t.tick) else np.full(2, np.nan)                                      # noqa: E731
        rows.append({"route": j.route, "frame": j.frame, "fork_tick": j.fork_tick, "eligible": j.eligible,
                     "status": "complete" if complete else ("truncated" if reached else "fork_not_reached"),
                     "span": span, "end_A": ends["A"], "end_B": ends["B"], "end_F": ends["F"],
                     "V_t": first(A) - first(F), "V_t_AA": first(A) - first(B),
                     "dJ": wa["J"] - wf["J"], "dJ_AA": wa["J"] - wb["J"],
                     "d_collision": int(wa["collision"]) - int(wf["collision"]),
                     "d_collision_AA": int(wa["collision"]) - int(wb["collision"]),
                     "d_progress": wf["progress"] - wa["progress"], "d_progress_AA": wb["progress"] - wa["progress"],
                     "pre_fork_dist_FA": float(np.hypot(*(at(F) - at(A)))),
                     "pre_fork_dist_BA": float(np.hypot(*(at(B) - at(A))))})
    d = pd.DataFrame(rows)
    prim = d[d.status == "complete"]
    floor = float(np.percentile(np.abs(prim.dJ_AA), 95)) if len(prim) else np.nan
    pfloor = float(np.percentile(np.abs(prim.d_progress_AA), 95)) if len(prim) else np.nan
    d["above_floor"] = np.abs(d.dJ) > floor
    prim = d[d.status == "complete"]
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / "forked_rollout.csv", index=False)
    pops = [("eligible", "primary", prim[prim.eligible]), ("null_control", "primary", prim[~prim.eligible]),
            ("all_sampled", "primary", prim),
            ("eligible", "sensitivity_truncated_kept", d[d.eligible & d.status.isin(["complete", "truncated"])])]
    out = []
    for pop, analysis, g in pops:
        out.append({"population": pop, "analysis": analysis, **summarise(g, floor, pfloor)})
    s = pd.DataFrame(out)
    s.insert(2, "excluded_truncated", int((d.status == "truncated").sum()))
    s.insert(3, "excluded_fork_not_reached", int((d.status == "fork_not_reached").sum()))
    s.to_csv(OUT / "aa_control.csv", index=False)
    print(s.T.to_string())


def summarise(g, floor, pfloor):
    above = g[g.above_floor & (g.V_t.abs() > 1e-9)]
    agree = (np.sign(above.V_t) == np.sign(above.dJ)).to_numpy()
    routes = above.route.to_numpy()
    uq = np.unique(routes)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(1000):
        pick = rng.integers(0, len(uq), len(uq)) if len(uq) else []
        m = np.concatenate([np.flatnonzero(routes == uq[i]) for i in pick]) if len(uq) else np.zeros(0, int)
        bs.append(agree[m].mean() if len(m) else np.nan)
    bs = np.asarray(bs, float)
    sp = lambda a, b: float(pd.Series(g[a]).corr(pd.Series(g[b]), method="spearman")) if len(g) > 2 else np.nan  # noqa: E731
    return {"frames": len(g), "routes": g.route.nunique(), "noise_floor_abs_dJ_AA_p95": floor,
            "median_abs_dJ": float(g.dJ.abs().median()), "median_abs_dJ_AA": float(g.dJ_AA.abs().median()),
            "above_floor": int(g.above_floor.sum()), "below_floor": int((~g.above_floor).sum()),
            "nonzero_V": int((g.V_t.abs() > 1e-9).sum()), "above_floor_with_nonzero_V": len(above),
            "sign_agreement": float(agree.mean()) if len(agree) else np.nan,
            "sign_agreement_lo": float(np.nanpercentile(bs, 2.5)) if len(agree) else np.nan,
            "sign_agreement_hi": float(np.nanpercentile(bs, 97.5)) if len(agree) else np.nan,
            "resolved": bool(len(above) >= 30),
            "spearman_V_dJ": sp("V_t", "dJ"), "spearman_V_dJ_AA": sp("V_t", "dJ_AA"),
            "spearman_V_AA_dJ_AA": sp("V_t_AA", "dJ_AA"),
            "collision_removed": int((g.d_collision == 1).sum()), "collision_added": int((g.d_collision == -1).sum()),
            "collision_changed_AA": int((g.d_collision_AA != 0).sum()),
            "progress_floor_p95_m": pfloor, "progress_above_floor": int((g.d_progress.abs() > pfloor).sum()),
            "spearman_V_dprogress": sp("V_t", "d_progress"),
            "median_pre_fork_dist_FA_m": float(g.pre_fork_dist_FA.median()),
            "median_pre_fork_dist_BA_m": float(g.pre_fork_dist_BA.median()),
            "pre_fork_diverged_FA_gt_1cm": int((g.pre_fork_dist_FA > 0.01).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sample", "outcomes"], required=True)
    ap.add_argument("--logs", required=True, help="folder holding ticks_<tag>.jsonl (copied from the simulation machine)")
    ap.add_argument("--out_dir", default=None, help="folder holding the evaluator checkpoints route_<id>_<tag>.json")
    args = ap.parse_args()
    {"sample": stage_sample, "outcomes": stage_outcomes}[args.stage](args)


if __name__ == "__main__":
    main()
