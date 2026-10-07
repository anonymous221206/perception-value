"""Follow-up E1 (the nuScenes track on all 850 trainval scenes): cells, signals and statistics.

Everything here reuses the frozen track's code: cells are `92_benchmark_table.nuscenes_cells` on other tables and
another split, the gates are 92's `gate_predictions` with an explicit fit mask, the detection-list routers are 103's
`fit_score` on 103's frame matrix, and the statistics are `rap.scoring` (exact tie expectation, paired unit
bootstrap) and 125's every-draw paired difference, generalised to any list of quotas. With the frozen split, the frozen
tables and the frozen random-number streams (`compat_*` helpers in 172) they reproduce the released values; that is
gate (a) of docs/followup_protocol.md.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from . import egospeed, predict
from .paths import CACHE, RESULTS
from .scoring import EPS, QUOTAS, topk_expect

ROOT = Path(__file__).resolve().parents[2]
PAIR = "ns_cheap_320tons_full_640"
SYSTEMS = (("brake", "J_cheap", "J_full"), ("plan_ade", "JC_ade_cheap", "JC_ade_full"),
           ("plan_fde", "JC_fde_cheap", "JC_fde_full"))
GEOMETRIES = ("oracle", "mono")
# the ten deployable signals of results/final/significance_table.csv, in its order
SIGNALS = ["gate_ridge", "gate_gbm", "R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf", "R2_cnn_clf",
           "uncertainty", "criticality_cheap", "trivial_ego_speed"]
COLS = {"uncertainty": "unc_sum", "criticality_cheap": "feat_crit_sum", "trivial_ego_speed": egospeed.COL}
GATES = {"gate_ridge": "linear", "gate_gbm": "gbm"}
SWEEP = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
OUT = Path(RESULTS) / "followup" / "E1"


def load_script(name: str, file: str):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_T = {}


def t92():
    if "92" not in _T:
        _T["92"] = load_script("t92", "92_benchmark_table.py")
    return _T["92"]


def t103():
    if "103" not in _T:
        _T["103"] = load_script("t103", "103_routers_r1.py")
    return _T["103"]


# ------------------------------------------------------------------------------------------------ inputs

def scene_split(path=None) -> pd.DataFrame:
    return pd.read_csv(path or OUT / "scene_split.csv")


def use_ego_table(csv) -> None:
    """Point egospeed at the causal ego-speed table of these scenes (170 writes it beside the tables)."""
    egospeed._TABLE = pd.read_csv(csv).astype({"seq": str, "frame": int})


def features(det_dir):
    """92's gate features (84's builder on the CHEAP cache) and their leakage-checked columns."""
    from .risk import RiskConfig
    from .tables import feature_columns
    f = t92().g84.build_features(Path(det_dir), "ns_cheap_320", RiskConfig())
    return f, feature_columns(f)


def cells(tables_dir, feats, fcols, split_of, unit_of=None, geometries=GEOMETRIES, pair=PAIR):
    """92's nuScenes cells on another table set: only the frames of scenes in `split_of` are kept."""
    for geom in geometries:
        j = pd.read_pickle(Path(tables_dir) / f"{geom}__{pair}.pkl")
        j = j[j.seq.isin(set(split_of))].merge(feats, on=["seq", "frame"], how="inner", validate="one_to_one")
        j["unit"] = j.seq if unit_of is None else j.seq.map(unit_of)
        j["split"] = j.seq.map(split_of)
        for system, cc, cf in SYSTEMS:
            if cc not in j:
                continue
            d = egospeed.attach(j[j[cc].notna()].reset_index(drop=True), "nuScenes")
            yield dict(track="nuScenes", geometry=geom, system=system, target=cc.replace("_cheap", ""),
                       d=d, cheap=cc, full=cf, fcols=fcols)


def r1_matrix(det_dir):
    return t103().frame_matrix(Path(det_dir), "ns_cheap_320", "nuScenes")


def r1_rows(d, mat):
    keys, M = mat
    kk = keys.assign(_row=np.arange(len(keys)))
    j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(kk, on=["seq", "frame"], how="left",
                                                                     validate="one_to_one")
    assert j._row.notna().all()
    return M[j._row.to_numpy(int)]


# ------------------------------------------------------------------------------------------------ signals

def gate_scores(d, fcols, v, fit, which=tuple(GATES)):
    """92's `gate_predictions` with the fit mask passed in (92 fixes it to train + val)."""
    X = np.nan_to_num(d[fcols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64),
                      nan=0.0, posinf=1e6, neginf=-1e6)
    out = {}
    for name in which:
        m = predict.make_model(GATES[name], "reg", 0)
        m.fit(X[fit], v[fit])
        out[name] = m.predict(X)
    return out


def column_scores(d):
    return {s: pd.to_numeric(d[c], errors="coerce").fillna(-np.inf).to_numpy(float)
            for s, c in COLS.items() if c in d.columns}


def r2_scores(d, npz):
    """R2's per-frame scores of one cell from a 107 run (missing frames rank last, as in 125)."""
    if npz is None or not Path(npz).exists():
        return {}
    y = np.load(npz, allow_pickle=False)
    r2 = pd.DataFrame({"seq": y["seq"].astype(str), "frame": y["frame"].astype(int),
                       "s": np.asarray(y["R2_cnn_clf"], float)})
    j = d[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(r2, on=["seq", "frame"], how="left",
                                                                     validate="one_to_one")
    return {"R2_cnn_clf": np.nan_to_num(j.s.to_numpy(float), nan=-np.inf)}


# ------------------------------------------------------------------------------------------------ statistics

def k_of(n, quotas):
    return np.array([max(int(round(q * n)), 1) for q in quotas])


def gains(score, v, ks):
    return np.asarray(topk_expect(score, [v], ks)[0][0], float)


def describe(v, cheap, quotas=(0.2,)):
    """92's cell descriptives (affected, harmed share, harm ratio, all-full and oracle reductions)."""
    v = np.asarray(v, float)
    pos, neg = v[v > EPS].sum(), -v[v < -EPS].sum()
    aff = np.abs(v) > EPS
    tot = float(np.sum(cheap))
    prize = gains(v, v, k_of(len(v), quotas))
    r = {"n_frames": len(v), "affected": int(aff.sum()), "affected_share": float(aff.mean()),
         "harmed": int((v < -EPS).sum()),
         "harmed_share": float((v < -EPS).sum() / aff.sum()) if aff.sum() else np.nan,
         "rho": float(neg / pos) if pos > EPS else np.nan,
         "gross_benefit": float(pos), "gross_harm": float(neg), "all_cheap_loss": tot,
         "all_full_reduction": float(v.sum() / tot) if tot > EPS else np.nan}
    for q, p in zip(quotas, prize):
        r[f"oracle{int(round(q * 100))}_reduction"] = float(p / tot) if tot > EPS else np.nan
    return r


def unit_index(units):
    uniq = np.array(sorted(set(units)))
    return uniq, {u: np.flatnonzero(units == u) for u in uniq}


def bootstrap_describe(v, cheap, units, nboot, rng, quotas=(0.2,)):
    """95% unit-bootstrap intervals of `describe`'s ratios (every draw kept)."""
    uniq, idx = unit_index(np.asarray(units))
    keys = ["harmed_share", "rho", "all_full_reduction"] + [f"oracle{int(round(q * 100))}_reduction" for q in quotas]
    draws = {k: np.full(nboot, np.nan) for k in keys + ["affected"]}
    for b in range(nboot):
        t = np.concatenate([idx[u] for u in uniq[rng.integers(0, len(uniq), len(uniq))]])
        r = describe(v[t], cheap[t], quotas)
        for k in draws:
            draws[k][b] = r[k]
    out = {}
    for k, x in draws.items():
        x = x[np.isfinite(x)]
        out[f"{k}_lo"], out[f"{k}_hi"] = ((float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)))
                                          if len(x) >= 50 else (np.nan, np.nan))
    return out


def evaluate_q(v, units, scores, nboot, rng, quotas=QUOTAS):
    """`rap.scoring.evaluate` for any list of quotas (identical draws and filter; equal to it at QUOTAS)."""
    v = np.asarray(v, float)
    nz = (np.abs(v) > EPS).astype(float)
    n = len(v)
    ks = k_of(n, quotas)
    prize0 = topk_expect(v, [v], ks)[0][0]
    point = {}
    for name, s in scores.items():
        if s is None:
            gain = ks / n * v.sum()
            resp, tie = np.full(len(ks), nz.mean()), np.full(len(ks), np.nan)
        else:
            (gain, rsum), tie = topk_expect(s, [v, nz], ks)
            resp = rsum / ks
        point[name] = dict(gain=gain, tie=tie, resp=resp,
                           eta=np.where(prize0 > EPS, gain / np.where(prize0 > EPS, prize0, 1), np.nan))
    uniq = np.unique(units)
    idx = [np.flatnonzero(units == u) for u in uniq]
    draws = {name: np.full((nboot, len(ks)), np.nan) for name in scores}
    dropped = np.zeros(len(ks), int)
    for b in range(nboot):
        t = np.concatenate([idx[i] for i in rng.integers(0, len(uniq), len(uniq))])
        vt = v[t]
        kt = k_of(len(t), quotas)
        pz = topk_expect(vt, [vt], kt)[0][0]
        ok = pz > np.maximum(EPS, 0.25 * prize0)
        dropped += ~ok
        den = np.where(ok, pz, 1.0)
        for name, s in scores.items():
            g = kt / len(t) * vt.sum() if s is None else topk_expect(s[t], [vt], kt)[0][0]
            draws[name][b] = np.where(ok, g / den, np.nan)
    return ks, prize0, point, draws, dropped


def ci(x):
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) >= 50 else (np.nan, np.nan)


def paired_draws(cells_, nboot, rng, quotas, extra=()):
    """125's every-draw paired difference to random for a list of prepared cells sharing one unit pool.

    Each cell carries v (test values), units (test units), series {signal: test scores}. Per draw the pool is
    resampled once (`rng.integers(0, n, n)`), then `extra` further integer draws are consumed -- that is how 125's
    other datasets advance the stream, needed only to reproduce its nuScenes intervals.
    """
    pool = sorted(set().union(*[set(c["units"]) for c in cells_]))
    for c in cells_:
        c["upos"] = {u: i for i, u in enumerate(sorted(set(c["units"])))}
        c["idx"] = [np.flatnonzero(c["units"] == u) for u in sorted(set(c["units"]))]
    out = {(ci_, s): np.full((nboot, len(quotas)), np.nan) for ci_, c in enumerate(cells_) for s in c["series"]}
    keep = {ci_: np.zeros((nboot, len(quotas)), bool) for ci_ in range(len(cells_))}
    for b in range(nboot):
        pick = rng.integers(0, len(pool), len(pool))
        for n in extra:
            rng.integers(0, n, n)
        for ci_, c in enumerate(cells_):
            us = [u for u in (pool[i] for i in pick) if u in c["upos"]]
            if not us:
                continue
            t = np.concatenate([c["idx"][c["upos"][u]] for u in us])
            vt = c["v"][t]
            ks = k_of(len(t), quotas)
            pz = gains(vt, vt, ks)
            keep[ci_][b] = pz > np.maximum(EPS, 0.25 * c["prize"])
            rnd = ks / len(t) * vt.sum()
            for s, sc in c["series"].items():
                out[(ci_, s)][b] = gains(sc[t], vt, ks) - rnd
    return out, keep


def prepare(c, test, scores, quotas):
    """Test-split arrays of one cell for `paired_draws` and the point estimates."""
    d = c["d"]
    v_all = (d[c["cheap"]] - d[c["full"]]).to_numpy(float)
    v = v_all[test]
    p = dict(key=(c["track"], c["geometry"], c["system"], c["target"]), v=v,
             units=d.unit.astype(str).to_numpy()[test], tot=float(d[c["cheap"]].to_numpy(float)[test].sum()),
             n=int(test.sum()))
    p["ks"] = k_of(p["n"], quotas)
    p["prize"] = gains(v, v, p["ks"])
    p["series"] = {s: np.nan_to_num(np.asarray(x, float)[test], nan=-np.inf) for s, x in scores.items()}
    return p


def paired_rows(prepared, draws, keep, quotas, extra_cols=None):
    """Rows in the schema of results/final/significance_table.csv (every-draw and filtered intervals)."""
    rows = []
    for ci_, c in enumerate(prepared):
        rnd = c["ks"] / c["n"] * c["v"].sum()
        for s, sc in c["series"].items():
            g = gains(sc, c["v"], c["ks"])
            dr = draws[(ci_, s)]
            for qi, q in enumerate(quotas):
                x = dr[:, qi]
                lo, hi = ci(x)
                xf = np.where(keep[ci_][:, qi], x, np.nan)
                flo, fhi = ci(xf)
                delta = float(g[qi] - rnd[qi])
                rows.append({"track": c["key"][0], "geometry": c["key"][1], "system": c["key"][2],
                             "target": c["key"][3], "signal": s, "quota": q, "n_test_units": len(set(c["units"])),
                             "n_test_frames": c["n"], "all_cheap_loss": c["tot"], "gain": float(g[qi]),
                             "gain_random": float(rnd[qi]), "delta_vs_random": delta,
                             "delta_lo_all_draws": lo, "delta_hi_all_draws": hi,
                             "delta_share_all_cheap": delta / c["tot"] if c["tot"] > EPS else np.nan,
                             "delta_share_lo_all_draws": lo / c["tot"] if c["tot"] > EPS else np.nan,
                             "delta_share_hi_all_draws": hi / c["tot"] if c["tot"] > EPS else np.nan,
                             "beats_random_all_draws": bool(lo > 0) if np.isfinite(lo) else False,
                             "ndg": float(g[qi] / c["prize"][qi]) if c["prize"][qi] > EPS else np.nan,
                             "delta_lo_filtered": flo, "delta_hi_filtered": fhi,
                             **(extra_cols or {})})
    return rows


# ------------------------------------------------------------------------------------------------ scene metadata

SCENE_META = Path(CACHE) / "xl" / "scene_meta.csv"


def scene_meta(dataroot, version="v1.0-trainval") -> pd.DataFrame:
    """scene -> log file name and CAM_FRONT calibration id (A, B, ... in order of first appearance by scene name).

    Without the dataset's tables (the cached tier) the shipped `data/cache/xl/scene_meta.csv`, written from them, is read.
    """
    import json
    t = Path(dataroot) / version
    if not (t / "scene.json").exists():
        return pd.read_csv(SCENE_META)
    rd = lambda n: json.loads((t / f"{n}.json").read_text())
    logs = {r["token"]: r["logfile"] for r in rd("log")}
    scenes = sorted(rd("scene"), key=lambda s: s["name"])
    samples = {r["token"]: r for r in rd("sample")}
    cs = {r["token"]: r for r in rd("calibrated_sensor")}
    cam = {}
    for r in rd("sample_data"):
        if r["is_key_frame"] and r["filename"].startswith("samples/CAM_FRONT/"):
            cam[r["sample_token"]] = r["calibrated_sensor_token"]
    ids, rows = {}, []
    for s in scenes:
        c = cs[cam[s["first_sample_token"]]]
        k = (tuple(np.round(c["translation"], 4)), tuple(np.round(c["rotation"], 4)))
        ids.setdefault(k, chr(ord("A") + len(ids)))
        rows.append({"scene": s["name"], "log": logs[s["log_token"]], "calibration": ids[k]})
    return pd.DataFrame(rows)


def holm(pvals, alpha=0.025):
    """Holm step-down calls for one-sided p-values at level alpha (the 95% two-sided interval's one tail)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    calls = np.zeros(len(p), bool)
    for rank, i in enumerate(order):
        if p[i] <= alpha / (len(p) - rank):
            calls[i] = True
        else:
            break
    return calls
