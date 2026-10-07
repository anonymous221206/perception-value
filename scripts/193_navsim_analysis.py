#!/usr/bin/env python
"""Follow-up E2: decision value on NAVSIM navtest, its strata, and allocators (addendum E2 in docs/followup_protocol.md).

Reads 192's per-token rows (test and dev), 190's token table and 191's detections.
  --gates     E2-0: this repository's reference branch against NAVSIM's official PDM-Closed evaluation (a CSV of
              per-token scores from NAVSIM's run_pdm_score); E2-1: cheap / full against the exploratory values on a
              300-token sample (informational). -> results/followup/E2/agreement_check.json
  (default)   settings.csv (V = PDMS(full) - PDMS(cheap) and per sub-score; loss 1 - PDMS), strata.csv, allocators.csv,
              validation_selected.csv (protocol Section 7 with the decision objective only: per quota the target-free
              signal with the largest realized gain on the development validation logs, then its navtest row of
              allocators.csv; NAVSIM has no per-token perception objective in the addendum)
"""
from __future__ import annotations

import argparse, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import runmeta, xl                                                      # noqa: E402
from rap.paths import RESULTS                                                    # noqa: E402
from rap.scoring import EPS                                                      # noqa: E402

D = ROOT / "data" / "cache" / "navsim"
OUT = Path(RESULTS) / "followup" / "E2"
SUBS = ("score", "no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
        "time_to_collision_within_bound", "comfort")
MIN_AFFECTED = 30


def load_rows(tag):
    d = pd.read_csv(D / f"pdms_rows_{tag}.csv")
    return d


def table(tags=("test", "mini")):
    """Tokens with their cheap / full / reference scores; navtest tokens from the test run, development tokens from
    the mini run; development tokens of logs shared with navtest are excluded (190)."""
    tok = pd.read_pickle(D / "tokens.pkl")
    tok = tok[tok.split != "dev_excluded_shared_log"]
    parts = []
    for t in tags:
        if not (D / f"pdms_rows_{t}.csv").exists():
            continue
        r = load_rows(t).drop(columns=["log"])
        want = tok[(tok.split == "test") if t == "test" else (tok.split != "test")]
        parts.append(want.merge(r, on="token", how="inner", validate="one_to_one"))
    d = pd.concat(parts, ignore_index=True)
    ok = d[[f"score_{b}" for b in ("cheap", "full")]].notna().all(axis=1)
    return d[ok].reset_index(drop=True), int((~ok).sum())


def features(d):
    """Cheap-side features per token (detections of 191 at the floor 0.10; counts at 0.25), plus ego state."""
    rows, R1 = [], []
    t103 = xl.t103()
    cache = {}
    for r in d.itertuples():
        if r.log not in cache:
            cache.clear()
            cache[r.log] = pickle.load(open(D / "dets" / f"{r.log}.pkl", "rb"))
        e = cache[r.log][r.token]["cheap"]
        conf, xyxy, ent = np.asarray(e["conf"], float), np.asarray(e["xyxy"], float).reshape(-1, 4), np.asarray(e["binent"], float)
        k = conf >= 0.25
        area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]) / (1920 * 1080) if len(xyxy) else np.zeros(0)
        bottom = xyxy[:, 3] / 1080 if len(xyxy) else np.zeros(0)
        rows.append({"n_det": int(k.sum()), "n_cand": len(conf), "conf_sum": float(conf[k].sum()),
                     "conf_max": float(conf.max()) if len(conf) else 0.0, "ent_sum": float(ent[k].sum()),
                     "unc_sum": float(ent.sum()), "area_max": float(area[k].max()) if k.any() else 0.0,
                     "area_sum": float(area[k].sum()), "bottom_max": float(bottom[k].max()) if k.any() else 0.0,
                     "ego_speed": r.ego_speed, "cmd_left": int(r.driving_command == 0),
                     "cmd_straight": int(r.driving_command == 1), "cmd_right": int(r.driving_command == 2)})
        R1.append(t103.det_list_features({"conf": conf, "xyxy": xyxy, "coarse": np.asarray(e["coarse"]).astype(str)},
                                         1920.0, 1080.0))
    return pd.DataFrame(rows), np.asarray(R1)


def settings(d):
    out = []
    te = d[d.split == "test"]
    for sub in SUBS:
        c, f = 1 - te[f"{sub}_cheap"].to_numpy(float), 1 - te[f"{sub}_full"].to_numpy(float)
        v = c - f
        r = xl.describe(v, c, (0.2,))
        r.update(xl.bootstrap_describe(v, c, te.log.to_numpy(), 1000, np.random.default_rng(0), (0.2,)))
        if r["affected"] < MIN_AFFECTED:
            for k in ("rho", "rho_lo", "rho_hi"):
                r[k] = np.nan
            r["undefined_reason"] = f"{r['affected']} affected tokens (< {MIN_AFFECTED})"
        r.update({"sub_score": sub, "population": "navtest", "n_tokens": len(te), "n_logs": te.log.nunique(),
                  "mean_cheap": float(te[f"{sub}_cheap"].mean()), "mean_full": float(te[f"{sub}_full"].mean()),
                  "mean_reference": float(te[f"{sub}_reference"].mean()) if f"{sub}_reference" in te else np.nan})
        out.append(r)
    return pd.DataFrame(out)


def strata(d):
    te = d[d.split == "test"].copy()
    te["speed_tercile"] = np.digitize(te.ego_speed, np.quantile(te.ego_speed, [1 / 3, 2 / 3]))
    te["agents_tercile"] = np.digitize(te.n_agents_30m, np.quantile(te.n_agents_30m, [1 / 3, 2 / 3]))
    out = []
    for by in ("speed_tercile", "agents_tercile", "lead_vehicle", "driving_command"):
        for val, g in te.groupby(by):
            c, f = 1 - g.score_cheap.to_numpy(float), 1 - g.score_full.to_numpy(float)
            r = xl.describe(c - f, c, (0.2,))
            r.update({"by": by, "value": val, "n_tokens": len(g), "n_logs": g.log.nunique()})
            out.append(r)
    return pd.DataFrame(out)


def allocators(d):
    F, R1 = features(d)
    X = F.to_numpy(float)
    c, f = 1 - d.score_cheap.to_numpy(float), 1 - d.score_full.to_numpy(float)
    v = c - f
    fit, te = d.split.eq("dev_train").to_numpy(), d.split.eq("test").to_numpy()
    sc = {"ego_speed": F.ego_speed.to_numpy(float), "uncertainty": F.unc_sum.to_numpy(float),
          "n_detections": F.n_det.to_numpy(float)}
    for name, mdl in xl.GATES.items():
        sc[name] = xl.predict.make_model(mdl, "reg", 0).fit(X[fit], v[fit]).predict(X)
    sc.update(xl.t103().fit_score(R1, v, fit))
    cell = {"track": "NAVSIM", "geometry": "n/a", "system": "pdm_closed", "target": "1-PDMS",
            "d": d.assign(unit=d.log, PDMS_loss_cheap=c, PDMS_loss_full=f), "cheap": "PDMS_loss_cheap",
            "full": "PDMS_loss_full"}
    n_aff = int((np.abs(v[te]) > EPS).sum())
    p = xl.prepare(cell, te, sc, xl.SWEEP)
    dr, kp = xl.paired_draws([p], 1000, np.random.default_rng(0), xl.SWEEP)
    t = pd.DataFrame(xl.paired_rows([p], dr, kp, xl.SWEEP, {"bootstrap_unit": "log", "n_affected_test": n_aff,
                                                            "fit_units": "dev_train"}))
    if n_aff < MIN_AFFECTED:
        t["ndg"] = np.nan
        t["undefined_reason"] = f"{n_aff} affected test tokens (< {MIN_AFFECTED})"
    va = d.split.eq("dev_val").to_numpy()
    free = ("random", "ego_speed", "uncertainty", "n_detections")
    ks = xl.k_of(int(va.sum()), xl.SWEEP)
    val = {name: (np.asarray(xl.SWEEP) * v[va].sum() if name == "random" else xl.gains(sc[name][va], v[va], ks))
           for name in free}
    vs = []
    for i, q in enumerate(xl.SWEEP):
        g = {name: float(val[name][i]) for name in free}
        w = max(free, key=lambda n: g[n])
        r = {"quota": q, "n_val_tokens": int(va.sum()), "n_val_logs": int(d.log[va].nunique()),
             "n_affected_val": int((np.abs(v[va]) > EPS).sum()), "winner": w,
             **{f"val_gain_{n}": g[n] for n in free}}
        if w != "random":
            row = t[(t.signal == w) & np.isclose(t.quota, q)].iloc[0]
            r.update({k: row[k] for k in ("gain", "gain_random", "delta_vs_random", "delta_lo_all_draws",
                                          "delta_hi_all_draws", "beats_random_all_draws", "ndg")})
        vs.append(r)
    pd.DataFrame(vs).to_csv(OUT / "validation_selected.csv", index=False)
    return t


def gates(args):
    d, n_err = table(("test",))
    rep = {"tokens_scored": len(d), "tokens_with_errors": n_err,
           "reference_mean_pdms": float(d.score_reference.mean())}
    if args.official and Path(args.official).exists():
        o = pd.read_csv(args.official)
        o = o[o.token.isin(d.token)] if "token" in o else o
        j = d.merge(o[["token", "score"]].rename(columns={"score": "official"}), on="token", how="inner")
        diff = (j.score_reference - j.official).abs()
        rep["E2_0"] = {"tokens_compared": len(j), "official_mean": float(j.official.mean()),
                       "reference_mean_same_tokens": float(j.score_reference.mean()),
                       "max_abs_diff": float(diff.max()), "tokens_differing_1e-6": int((diff > 1e-6).sum())}
    if args.exploratory and Path(args.exploratory).exists():
        x = pd.read_csv(args.exploratory)
        s = d.sample(n=min(300, len(d)), random_state=0)
        j = s.merge(x[["token", "score_cheap", "score_full"]].rename(columns={"score_cheap": "x_cheap", "score_full": "x_full"}),
                    on="token", how="inner")
        rep["E2_1"] = {"tokens_compared": len(j),
                       "cheap_equal_share": float(np.mean(np.isclose(j.score_cheap, j.x_cheap, atol=1e-6))),
                       "full_equal_share": float(np.mean(np.isclose(j.score_full, j.x_full, atol=1e-6))),
                       "cheap_mean_diff": float((j.score_cheap - j.x_cheap).mean()),
                       "full_mean_diff": float((j.score_full - j.x_full).mean())}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "agreement_check.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--official", default=str(D / "official_pdm_closed_navtest.csv"))
    ap.add_argument("--exploratory", default=str(D / "exploratory_pdms_states.csv"))
    args = ap.parse_args()
    run = runmeta.new_run("navsim_analysis" + ("_gates" if args.gates else ""), vars(args))
    if args.gates:
        return gates(args)
    OUT.mkdir(parents=True, exist_ok=True)
    d, n_err = table()
    s = settings(d); s["tokens_with_errors"] = n_err
    s.to_csv(OUT / "settings.csv", index=False)
    strata(d).to_csv(OUT / "strata.csv", index=False)
    allocators(d).to_csv(OUT / "allocators.csv", index=False)
    for f in ("settings.csv", "strata.csv", "allocators.csv", "validation_selected.csv"):
        (run / f).write_text((OUT / f).read_text())
    print(s[["sub_score", "affected", "harmed_share", "rho", "all_full_reduction", "oracle20_reduction"]].to_string(index=False))


if __name__ == "__main__":
    main()
