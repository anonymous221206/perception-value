#!/usr/bin/env python
"""Follow-up round 2, R4: allocators that may leave budget unused (addendum R4 in docs/followup_protocol.md).

The four regression allocators (gate_gbm, gate_ridge, R1_gbm_reg, R1_mlp_reg), fit on the 385 new training scenes only
(X2's features, models and seeds), score every frame. Two rules spend a budget alpha on N frames:
  quota       escalate exactly k = round(alpha * N) frames, highest score first;
  abstention  escalate the frames whose predicted V exceeds tau, at most k, highest first.
tau per allocator and budget is chosen on the 152 validation scenes from {-inf (= quota), 0, the 50/60/70/80/90/95th
percentiles of the validation predictions}: the largest realized gain, ties toward fewer escalations. It is fixed
before the 228 test scenes are scored once.

  --stage run  thresholds.json, abstention.csv
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta, xl                                                       # noqa: E402
from rap.paths import CACHE, NUSCENES_XL, RESULTS                                  # noqa: E402

OUT = Path(RESULTS) / "followup" / "R4"
TABLES = Path(CACHE) / "xl" / "tables850"
DET = Path(CACHE) / "nusc_det_xl"
ALLOCATORS = ("gate_gbm", "gate_ridge", "R1_gbm_reg", "R1_mlp_reg")
PCTS = (50, 60, 70, 80, 90, 95)
NBOOT = 1000


def scores(c, mat, fit):
    d = c["d"]
    v = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    out = xl.gate_scores(d, c["fcols"], v, fit, which=("gate_gbm", "gate_ridge"))
    r1 = xl.t103().fit_score(xl.r1_rows(d, mat), v, fit)
    out.update({k: r1[k] for k in ("R1_gbm_reg", "R1_mlp_reg")})
    return v, out


def rule_gain(score, v, k, tau):
    """Realized gain of escalating the frames with score > tau, at most k, highest first (the benchmark's exact tie
    expectation at the k-th score); returns (gain, escalated count)."""
    s = np.where(score > tau, score, -np.inf) if np.isfinite(tau) else score
    n = int(min(k, np.sum(score > tau))) if np.isfinite(tau) else int(k)
    if n == 0:
        return 0.0, 0
    return float(xl.gains(s, v, np.array([n]))[0]), n


def choose_tau(score, v, k):
    grid = [-np.inf, 0.0] + [float(np.percentile(score, p)) for p in PCTS]
    res = []
    for tau in grid:
        g, n = rule_gain(score, v, k, tau)
        res.append((g, -n, tau))
    best = max(res, key=lambda r: (r[0], r[1]))                 # largest gain, then fewer escalations
    return best[2], [{"tau": t, "gain": g, "escalated": -n} for g, n, t in res]


def draws(score, v, units, k_frac, tau, nboot, rng):
    """Per draw (units resampled with replacement): gain of quota, of abstention at tau, and random's expectation at
    each rule's escalated count; k = round(alpha * frames in the draw)."""
    uniq = sorted(set(units))
    idx = {u: np.flatnonzero(units == u) for u in uniq}
    out = np.full((nboot, 5), np.nan)
    for b in range(nboot):
        t = np.concatenate([idx[uniq[i]] for i in rng.integers(0, len(uniq), len(uniq))])
        st, vt = score[t], v[t]
        k = max(int(round(k_frac * len(t))), 1)
        gq, _ = rule_gain(st, vt, k, -np.inf)
        ga, na = rule_gain(st, vt, k, tau)
        out[b] = (gq, ga, k / len(t) * vt.sum(), na / len(t) * vt.sum(), ga - gq)
    return out


def stage_run(args):
    sp = xl.scene_split()
    new = sp[sp.group == "new765"]
    split_of = dict(zip(new.scene, new.deep_split))
    xl.use_ego_table(TABLES / "v_ego_causal.csv")
    feats, fcols = xl.features(args.det)
    mat = xl.r1_matrix(args.det)
    rows, thr = [], {}
    for c in xl.cells(TABLES, feats, fcols, split_of):
        if c["system"] != "brake":
            continue
        d = c["d"]
        fit = (d.split == "train").to_numpy()
        val, test = (d.split == "val").to_numpy(), (d.split == "test").to_numpy()
        v, sc = scores(c, mat, fit)
        units = d.unit.astype(str).to_numpy()
        tot = float(d[c["cheap"]].to_numpy(float)[test].sum())
        for a in ALLOCATORS:
            for q in xl.SWEEP:
                kv = max(int(round(q * val.sum())), 1)
                tau, grid = choose_tau(sc[a][val], v[val], kv)
                thr[f"{c['geometry']}|{a}|{q}"] = {"tau": tau if np.isfinite(tau) else "-inf", "val_grid": [
                    {**g, "tau": g["tau"] if np.isfinite(g["tau"]) else "-inf"} for g in grid]}
                kt = max(int(round(q * test.sum())), 1)
                for rule, t in (("quota", -np.inf), ("abstention (tau chosen on validation)", tau), ("fixed tau = 0", 0.0)):
                    g, n = rule_gain(sc[a][test], v[test], kt, t)
                    dr = draws(sc[a][test], v[test], units[test], q, t, NBOOT, np.random.default_rng(0))
                    rnd = n / test.sum() * v[test].sum()
                    lo_r, hi_r = xl.ci(dr[:, 1] - dr[:, 3])
                    lo_n, hi_n = xl.ci(dr[:, 1])
                    row = {"geometry": c["geometry"], "system": "brake", "allocator": a, "budget": q, "rule": rule,
                           "tau": t if np.isfinite(t) else -np.inf, "k": kt, "escalated": n, "budget_used": n / kt,
                           "gain": g, "gain_share_all_cheap": g / tot, "gain_random_same_count": rnd,
                           "vs_random": g - rnd, "vs_random_lo": lo_r, "vs_random_hi": hi_r,
                           "vs_no_escalation_lo": lo_n, "vs_no_escalation_hi": hi_n, "all_cheap_loss": tot,
                           "n_test_frames": int(test.sum()), "n_test_scenes": len(set(units[test]))}
                    if rule != "quota":
                        gq, _ = rule_gain(sc[a][test], v[test], kt, -np.inf)
                        x = dr[:, 4] if rule.startswith("abstention") else None
                        if x is None:
                            dq = draws(sc[a][test], v[test], units[test], q, 0.0, NBOOT, np.random.default_rng(0))
                            x = dq[:, 4]
                        lo, hi = xl.ci(x)
                        row.update(D_vs_quota=g - gq, D_lo=lo, D_hi=hi, D_share_all_cheap=(g - gq) / tot,
                                   D_share_lo=lo / tot, D_share_hi=hi / tot, resolved=bool(lo > 0 or hi < 0))
                    rows.append(row)
        print(f"  {c['geometry']} brake done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t["primary"] = ((t.geometry == "mono") & (t.allocator == "gate_gbm") & (t.budget == 0.2)
                    & (t.rule == "abstention (tau chosen on validation)"))
    t.to_csv(OUT / "abstention.csv", index=False)
    (OUT / "thresholds.json").write_text(json.dumps(thr, indent=1))
    print(t[t.primary | ((t.geometry == "mono") & (t.allocator == "gate_gbm") & (t.budget == 0.2))]
          [["rule", "tau", "escalated", "budget_used", "gain", "D_vs_quota", "D_lo", "D_hi"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["run"])
    ap.add_argument("--det", default=str(DET))
    ap.add_argument("--dataroot", default=str(NUSCENES_XL))
    args = ap.parse_args()
    runmeta.new_run(f"r4_{args.stage}", vars(args))
    stage_run(args)


if __name__ == "__main__":
    main()
