#!/usr/bin/env python
"""Task 24, Phase C: R1 and R2 refit on the published objectives, scored by decision value.

Pre-registered in the pre-registration record (not part of this release) (Task 24); labels from `148_published_objective_labels.py`, R2 refits from
`107_router_r2.py --objective {Q,G}` (and `--fit_units train` for the arbiter).

  --stage r1     R1 (MLP reg/clf, GBM reg/clf; architectures, hyperparameters, seeds and fit units of 103) refit on
                 Q and G: `*_reg` on Q-MORIC / G-MORIC, `*_clf` on 1[Q > 0] / OffloadBin; plus R1_gbm_reg on the
                 train-only G-MORIC for the budget arbiter.  Asserts the input and model shapes of the shipped R1.
  --stage score  per cell x architecture x target (V as shipped, Q, G) x quota: nDG on test with the benchmark's exact
                 tie expectation, the paired unit bootstrap against random and against the V-trained counterpart, and
                 feasibility and nDG under the measured latency and energy budgets (shipped overheads, not re-measured);
                 Geng's budget-adaptive variant (arbiter tuned on validation units, Eq. 8).  Writes
                 results/final/published_objective_routers.csv, published_objective_arbiter.csv,
                 published_objective_reading.json.
"""
from __future__ import annotations

import argparse, importlib.util, json, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import frames                                                           # noqa: E402
from rap import runs as rap_runs                                                 # noqa: E402
from rap import runmeta                                                         # noqa: E402
from rap.budget import infeasible                                               # noqa: E402
from rap.paths import CACHE, RESULTS                                            # noqa: E402

FINAL = Path(RESULTS) / "final"
R1 = ("R1_mlp_reg", "R1_mlp_clf", "R1_gbm_reg", "R1_gbm_clf")
ARCHS = R1 + ("R2_cnn_clf",)
OBJ = ("Q", "G")
G_CONF = 0.30
NBOOT = 1000


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    sys.modules[name] = m
    s.loader.exec_module(m)
    return m


def label_files(label_run):
    out = {}
    for f in sorted(Path(label_run).glob("labels__*.npz")):
        _, track, geometry, system, target = f.stem.split("__")
        out[(track, geometry, system, target)] = f
    return out


# ------------------------------------------------------------------------------------------------ R1

def stage_r1(run, label_run):
    t103 = _load("t103", "103_routers_r1.py")
    from rap import predict
    mats = {"nuScenes": t103.frame_matrix(Path(CACHE) / "nusc_det_tv", "ns_cheap_320", "nuScenes"),
            "KITTI": t103.frame_matrix(Path(CACHE) / "det", "cheap_320", "KITTI")}
    shapes = []
    for (track, geometry, system, target), f in label_files(label_run).items():
        z = np.load(f, allow_pickle=False)
        lab = pd.DataFrame({k: z[k] for k in z.files})
        keys, M = mats[track]
        kk = keys.assign(_row=np.arange(len(keys)))
        j = lab[["seq", "frame"]].astype({"seq": str, "frame": int}).merge(kk, on=["seq", "frame"], how="left",
                                                                            validate="one_to_one")
        assert j._row.notna().all()
        X = M[j._row.to_numpy(int)]
        assert X.shape[1] == 225, X.shape                  # the shipped R1 input: 25 detections x 9
        fit = lab.split.isin(["train", "val"]).to_numpy()
        out = {"seq": lab.seq.to_numpy().astype("U32"), "frame": lab.frame.to_numpy(), "split": lab.split.to_numpy().astype("U8"),
               "unit": lab.unit.to_numpy().astype("U40"), "V": lab.V.to_numpy(float)}
        for o in OBJ:
            for name, (task, make) in t103.models().items():
                y = lab[f"{o}_moric" if task == "reg" else f"{o}_bin"].to_numpy(float)
                if task == "clf" and len(np.unique(y[fit])) < 2:
                    out[f"{name}__{o}"] = np.zeros(len(lab)); continue
                m = make().fit(X[fit], y[fit])
                out[f"{name}__{o}"] = m.predict(X) if task == "reg" else m.predict_proba(X)[:, 1]
                if name.startswith("R1_mlp"):
                    shapes.append({"cell": f"{track} {geometry} {system} {target}", "model": f"{name}__{o}",
                                   "coef_shapes": str([c.shape for c in m.named_steps["m"].coefs_]),
                                   "ok": [c.shape for c in m.named_steps["m"].coefs_] == [(225, 64), (64, 64), (64, 1)]})
                else:
                    ref = predict.make_model("gbm", task, 0).get_params()
                    shapes.append({"cell": f"{track} {geometry} {system} {target}", "model": f"{name}__{o}",
                                   "coef_shapes": "gbm parameters", "ok": m.get_params() == ref})
        # the arbiter's conditioned estimator: R1_gbm_reg on G-MORIC of the train units alone, fit on them alone
        tr = (lab.split == "train").to_numpy()
        m = predict.make_model("gbm", "reg", 0).fit(X[tr], lab.G_moric_train.to_numpy(float)[tr])
        out["R1_gbm_reg__G_train"] = m.predict(X)
        np.savez_compressed(run / f"r1__{track}__{geometry}__{system}__{target}.npz", **out)
        print(f"  R1 {track:8s} {geometry:6s} {system:9s} {target:7s} refit on Q and G", flush=True)
    s = pd.DataFrame(shapes)
    s.to_csv(run / "r1_shapes.csv", index=False)
    if not bool(s.ok.all()):
        raise SystemExit("an R1 refit does not have the shipped shape: see r1_shapes.csv")


# ------------------------------------------------------------------------------------------------ arbiter

def pooled_ap_escalated(boxes, keys, top):
    """Geng et al.'s utility: set-wide AP@0.5 (official greedy matching, all-point) when `top` frames are escalated."""
    t148 = _load("t148", "148_published_objective_labels.py")
    _, gm = t148._geng()                                  # bgt-ada's metrics, loaded as 148 loads them
    raw = t148.geng_raw(boxes, keys)
    dets = []
    for r, esc in zip(raw, top):
        for dd in (r["cloud_dets"] if esc else r["edge_dets"]):
            dets.append((dd, r["video_name"], r["frame_id"]))
    dets.sort(key=lambda x: x[0][4], reverse=True)
    gt = {(r["video_name"], r["frame_id"]): r["gt_objs"] for r in raw}
    total = sum(len(v) for v in gt.values())
    return gm._run_greedy_matching(gm._precompute_detection_ious(dets, gt), gt, total, 0.5)["ap_allpoint"]


def top_fraction(score, q):
    k = max(int(round(q * len(score))), 1)
    o = np.argsort(-np.asarray(score, float), kind="stable")
    sel = np.zeros(len(score), bool)
    sel[o[:k]] = True
    return sel


# ------------------------------------------------------------------------------------------------ score

def stage_score(run, label_run, r1_run, r2_runs, r2_arb_run, dump_run):
    t92 = _load("t92", "92_benchmark_table.py")
    t93 = _load("t93", "93_budget_allocation.py")
    ov = json.loads((FINAL / "benchmark_budget_overheads_routers.json").read_text())["overheads"]
    off_r1, off_r2 = rap_runs.latest("routers_r1"), rap_runs.latest("router_r2")
    boxes = {}
    for t in ("KITTI", "nuScenes"):
        with open(Path(dump_run) / f"boxes__{t}.pkl", "rb") as fh:
            boxes[t] = pickle.load(fh)
    # R2 shapes: the refits keep the shipped width and FLOPs
    shape_rows = []
    for t in ("KITTI", "nuScenes"):
        ship = json.loads((off_r2 / f"r2_{t}_meta.json").read_text())
        for o, r in list(r2_runs.items()) + [("G_train", r2_arb_run)]:
            meta = json.loads((Path(r) / f"r2_{t}_meta.json").read_text())
            shape_rows.append({"dataset": t, "run": Path(r).name, "objective": o,
                               "ok": meta["width_mult"] == ship["width_mult"] and meta["gflops"] == ship["gflops"]
                               and meta["heads"] == ship["heads"]})
    if not all(r["ok"] for r in shape_rows):
        raise SystemExit(f"an R2 refit does not have the shipped shape: {shape_rows}")
    rng = np.random.default_rng(0)
    rows, arb_rows = [], []
    for (track, geometry, system, target), f in label_files(label_run).items():
        z = np.load(f, allow_pickle=False)
        lab = pd.DataFrame({k: z[k] for k in z.files}).astype({"seq": str, "frame": int})
        r1p = np.load(Path(r1_run) / f"r1__{track}__{geometry}__{system}__{target}.npz", allow_pickle=False)
        r1v = np.load(off_r1 / f"scores__{track}__{geometry}__{system}__{target}.npz", allow_pickle=False)
        assert (r1p["seq"] == lab.seq.to_numpy()).all() and (r1v["seq"].astype(str) == lab.seq.to_numpy()).all()
        test = (lab.split == "test").to_numpy()
        v, units = lab.V.to_numpy(float)[test], lab.unit.to_numpy()[test]
        keys_test = lab[["seq", "frame"]][test].reset_index(drop=True)

        def r2_scores(r, col="R2_cnn_clf"):
            s = np.load(Path(r) / f"scores__{track}__{geometry}__{system}__{target}.npz", allow_pickle=False)
            t = pd.DataFrame({"seq": s["seq"].astype(str), "frame": s["frame"].astype(int), "s": s[col]})
            jj = keys_test.merge(t, on=["seq", "frame"], how="left", validate="one_to_one")
            assert jj.s.notna().all()
            return jj.s.to_numpy(float)
        scores = {"random": None}
        for a in R1:
            scores[f"{a}|V"] = r1v[a].astype(float)[test]
            for o in OBJ:
                scores[f"{a}|{o}"] = r1p[f"{a}__{o}"].astype(float)[test]
        scores["R2_cnn_clf|V"] = r2_scores(off_r2)
        for o in OBJ:
            scores[f"R2_cnn_clf|{o}"] = r2_scores(r2_runs[o])
        ks, prize0, point, draws, dropped = t92.evaluate(v, units, scores, NBOOT, rng)

        # the budget arbiter on validation units: train-only estimators, the paper's utility (Eq. 8)
        val = (lab.split == "val").to_numpy()
        vkeys = [(s, int(fr)) for s, fr in zip(lab.seq[val], lab.frame[val])]
        cond = r1p["R1_gbm_reg__G_train"].astype(float)[val]
        a_lab = pd.read_pickle(Path(r2_arb_run) / f"r2_{track}_labels.pkl").astype({"seq": str, "frame": int})
        a_meta = json.loads((Path(r2_arb_run) / f"r2_{track}_meta.json").read_text())
        a_sc = np.load(Path(r2_arb_run) / f"r2_{track}_torch_scores.npz")["scores"][:, a_meta["heads"].index(f"{geometry}__{system}__{target}")]
        jj = pd.DataFrame(vkeys, columns=["seq", "frame"]).merge(a_lab[["seq", "frame"]].assign(s=a_sc), on=["seq", "frame"],
                                                                  how="left", validate="one_to_one")
        skip = jj.s.to_numpy(float)
        choice = {}
        for q in t92.QUOTAS:
            u_c = pooled_ap_escalated(boxes[track], vkeys, top_fraction(cond, q))
            u_s = pooled_ap_escalated(boxes[track], vkeys, top_fraction(skip, q))
            choice[q] = "R1_gbm_reg" if u_c >= u_s else "R2_cnn_clf"
            arb_rows.append({"track": track, "geometry": geometry, "system": system, "target": target, "quota": q,
                             "val_frames": int(val.sum()), "ap50_conditioned": u_c, "ap50_skipping": u_s,
                             "chosen": choice[q]})

        costs = t93.profile_costs(track)
        key = dict(track=track, geometry=geometry, system=system, target=target)

        def budget_cols(sc, name, q):
            out = {}
            for unit in ("ms", "mJ"):
                c0, c1 = costs["cheap"][unit], costs["640"][unit]
                budget = c0 + q * c1
                o_ms, o_mj, _ = t93.signal_overhead(ov, track, name)
                o = o_ms if unit == "ms" else o_mj
                bad = bool(infeasible(budget, c0, o))
                frac = max((budget - c0 - o) / c1, 0.0)
                k = int(np.floor(frac * len(v) + 1e-9))
                kp = int(np.floor(q * len(v) + 1e-9))
                prize = float(t92.topk_expect(v, [v], [kp])[0][0][0]) if kp > 0 else 0.0
                g = float(t92.topk_expect(sc, [v], [k])[0][0][0]) if k > 0 else 0.0
                out[f"budget_{unit}_feasible"] = not bad
                out[f"budget_{unit}_escalated_frac"] = np.nan if bad else frac
                out[f"budget_{unit}_ndg"] = np.nan if bad or prize <= t92.EPS else g / prize
            return out

        def emit(arch, obj, name_for_cost, sc_key_by_q, counterpart_by_q, variant=""):
            for qi, q in enumerate(t92.QUOTAS):
                sk, ck = sc_key_by_q(q), counterpart_by_q(q)
                d = draws[sk][:, qi]
                dr = d - draws["random"][:, qi]
                lo, hi = t92.ci(d); rlo, rhi = t92.ci(dr)
                row = {**key, "architecture": arch, "objective": obj, "variant": variant, "quota": q, "k": int(ks[qi]),
                       "ndg": float(point[sk]["eta"][qi]), "ndg_lo": lo, "ndg_hi": hi,
                       "minus_random": float(np.nanmean(dr)), "minus_random_lo": rlo, "minus_random_hi": rhi,
                       "boot_dropped": int(dropped[qi]), "n_frames": int(len(v)), "n_units": int(len(np.unique(units)))}
                if ck is not None:
                    dv = d - draws[ck][:, qi]
                    vlo, vhi = t92.ci(dv)
                    row.update(counterpart=ck, minus_V=float(np.nanmean(dv)), minus_V_lo=vlo, minus_V_hi=vhi)
                row.update(budget_cols(scores[sk], name_for_cost(q), q))
                rows.append(row)
        for a in ARCHS:
            emit(a, "V", lambda q, a=a: a, lambda q, a=a: f"{a}|V", lambda q: None)
            for o in OBJ:
                emit(a, o, lambda q, a=a: a, lambda q, a=a, o=o: f"{a}|{o}", lambda q, a=a: f"{a}|V")
        emit("R1_gbm_reg <-> R2_cnn_clf", "G", lambda q: choice[q], lambda q: f"{choice[q]}|G",
             lambda q: f"{choice[q]}|V", variant="budget-adaptive (Geng Eq. 8)")
        print(f"  {track:8s} {geometry:6s} {system:9s} {target:7s} scored; arbiter "
              + " ".join(f"{int(q * 100)}%:{'cond' if c == 'R1_gbm_reg' else 'skip'}" for q, c in choice.items()), flush=True)

    df, arb = pd.DataFrame(rows), pd.DataFrame(arb_rows)
    # the V-trained rows use the shipped scores, so their nDG is the shipped routers table's, exactly
    off = pd.read_csv(FINAL / "benchmark_table_routers.csv", keep_default_na=False, na_values=[""])
    chk = df[df.objective == "V"].merge(off[off.split == "test"].rename(columns={"signal": "architecture"})[
        ["track", "geometry", "system", "target", "architecture", "quota", "eta"]],
        on=["track", "geometry", "system", "target", "architecture", "quota"], how="left", validate="one_to_one")
    bad = chk[~np.isclose(chk.ndg, chk.eta, rtol=0, atol=1e-12)]
    if len(bad) or chk.eta.isna().any():
        raise SystemExit(f"the V-trained rows do not reproduce benchmark_table_routers.csv:\n{bad.head()}")
    print(f"  V-trained counterparts reproduce benchmark_table_routers.csv: {len(chk)} rows", flush=True)
    for out in (FINAL, run):
        df.to_csv(out / "published_objective_routers.csv", index=False)
        arb.to_csv(out / "published_objective_arbiter.csv", index=False)
    # the registered reading: every published row against its V-trained counterpart, and against random
    pub = df[df.objective.isin(OBJ)].copy()
    vrows = df[df.objective == "V"][["track", "geometry", "system", "target", "architecture", "quota",
                                    "minus_random_lo", "minus_random_hi"]].rename(
        columns={"architecture": "cp_arch", "minus_random_lo": "V_lo", "minus_random_hi": "V_hi"})
    pub["cp_arch"] = [c.split("|")[0] for c in pub.counterpart]
    m = pub.merge(vrows, on=["track", "geometry", "system", "target", "cp_arch", "quota"], how="left",
                  validate="many_to_one")
    outside = m[~((m.minus_V_lo <= 0) & (m.minus_V_hi >= 0))]
    win_new = m[(m.minus_random_lo > 0) & ~(m.V_lo > 0)]
    win_lost = m[~(m.minus_random_lo > 0) & (m.V_lo > 0)]
    loss_vs_win = m[(m.minus_random_hi < 0) & (m.V_lo > 0)]

    def fmt(x):
        return [f"{r.track} {r.geometry} {r.system} {r.target} | {r.architecture} on {r.objective}"
                f"{' (' + r.variant + ')' if r.variant else ''} @{int(round(r.quota * 100))}%: nDG {r.ndg:+.3f}; "
                f"vs random [{r.minus_random_lo:+.3f}, {r.minus_random_hi:+.3f}]; V-trained vs random "
                f"[{r.V_lo:+.3f}, {r.V_hi:+.3f}]; published - V [{r.minus_V_lo:+.3f}, {r.minus_V_hi:+.3f}]"
                for r in x.itertuples()]
    material = len(win_new) > 0 or len(win_lost) > 0
    reading = {"registered_reading": ("material: a published-objective router beats random where its V-trained "
                                      "counterpart does not, or the reverse" if material else
                                      ("the training objective does not separate these architectures"
                                       if len(outside) == 0 else "no side change against random, but some published "
                                       "routers fall outside the paired interval of their V-trained counterpart")),
               "published_rows": int(len(m)), "rows_outside_V_interval": fmt(outside),
               "beats_random_where_V_does_not": fmt(win_new), "V_beats_random_published_does_not": fmt(win_lost),
               "published_below_random_where_V_beats_random": fmt(loss_vs_win), "arbiter": arb.to_dict("records")}
    for out in (FINAL, run):
        (out / "published_objective_reading.json").write_text(json.dumps(reading, indent=1, default=float))
    print(f"  reading: {reading['registered_reading']}; outside the V interval {len(outside)}, wins appearing "
          f"{len(win_new)}, wins lost {len(win_lost)}, below random where V wins {len(loss_vs_win)} "
          f"(of {len(m)} published rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["r1", "score"])
    ap.add_argument("--label_run", default=None)
    ap.add_argument("--r1_run", default=None)
    ap.add_argument("--r2_q", default=None)
    ap.add_argument("--r2_g", default=None)
    ap.add_argument("--r2_arbiter", default=None)
    ap.add_argument("--dump_run", default=None)
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(f"published_objective_{args.stage}", vars(args))
    label_run = Path(args.label_run) if args.label_run else rap_runs.latest("published_objective_labels")
    if args.stage == "r1":
        stage_r1(run, label_run)
    else:
        stage_score(run, label_run,
                    Path(args.r1_run) if args.r1_run else rap_runs.latest("published_objective_r1"),
                    {"Q": Path(args.r2_q), "G": Path(args.r2_g)}, Path(args.r2_arbiter),
                    Path(args.dump_run) if args.dump_run else rap_runs.latest("published_objective_dump"))
    print("wrote", run)


if __name__ == "__main__":
    main()
