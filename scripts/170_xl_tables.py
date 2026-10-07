#!/usr/bin/env python
"""Follow-up E1: per-frame decision tables for any set of nuScenes scenes, as the frozen track builds them.

For each geometry and detector pair this is `52_core_matrix`'s nuScenes table (`decision.build` on the shared action
history, `add_perception_gain`, the perception primitives of 50), joined as `62_planning_metric_eta` joins it: sample
tokens, and PKL's planner scored against the logged trajectory (`planC_vs_truth[_mono].csv` written by 74) where those
costs exist. PKL and TIP gains are not joined (diagnostics outside the follow-up protocol).

Nothing under data/cache or results/final is written. Tables go to --out as <geometry>__<cheap>to<full>.pkl, with the
causal ego-speed table of the scenes (138's `build`) beside them.

Protocol: docs/followup_protocol.md (Sections 3 and 4).
"""
from __future__ import annotations

import argparse, importlib.util, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from rap import frames                                                           # noqa: E402
from rap import decision, geometry as G, planner as P, runmeta                  # noqa: E402
from rap.nusc import NuScenesDB, make_adapter                                   # noqa: E402
from rap.paths import CACHE, NUSCENES_TRAINVAL                                  # noqa: E402
from rap.risk import RiskConfig                                                 # noqa: E402

from importlib import import_module                                             # noqa: E402
_pm = import_module("50_percep_metrics")


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def token_map(seqs, dataroot, version) -> pd.DataFrame:
    """62's token map, computed without its cache file (which it would overwrite with this scene set)."""
    t = Path(dataroot) / version
    scenes = json.loads((t / "scene.json").read_text())
    samples = {r["token"]: r for r in json.loads((t / "sample.json").read_text())}
    want, rows = set(seqs), []
    for sc in scenes:
        if sc["name"] not in want:
            continue
        tok, fr = sc["first_sample_token"], 0
        while tok:
            rows.append((sc["name"], fr, tok))
            tok, fr = samples[tok]["next"], fr + 1
    return pd.DataFrame(rows, columns=["seq", "frame", "sample_token"])


def join_planner(d, planc_dir, geometry):
    """62's join of the truth-referenced planner costs (same file per geometry, same de-duplication)."""
    if planc_dir is None:
        return d
    f = Path(planc_dir) / ("planC_vs_truth.csv" if geometry == "oracle" else f"planC_vs_truth_{geometry}.csv")
    if not f.exists():
        print(f"  no planner costs at {f}")
        return d
    g = pd.read_csv(f).drop_duplicates("sample_token", keep="first")
    keep = ["sample_token", "JC_ade_cheap", "JC_ade_full", "JC_fde_cheap", "JC_fde_full"]
    d = d.merge(g[keep], on="sample_token", how="left")
    print(f"  planner vs truth ({f.name}): {len(g)} samples, coverage {float(d.JC_ade_cheap.notna().mean()):.3f}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", required=True, help="detection cache with one folder per mode")
    ap.add_argument("--dataroot", default=str(NUSCENES_TRAINVAL))
    ap.add_argument("--version", default="v1.0-trainval")
    ap.add_argument("--scenes", default=None, help="CSV with a 'scene' column (default: every scene in the cache)")
    ap.add_argument("--pairs", nargs="+", default=["ns_cheap_320:ns_full_640"])
    ap.add_argument("--geometries", nargs="+", default=["oracle", "mono"])
    ap.add_argument("--planc", default=None, help="folder holding planC_vs_truth[_mono].csv (74's output)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="xl_tables")
    frames.add_argument(ap)
    args = ap.parse_args()
    frames.configure(args)
    run = runmeta.new_run(args.tag, vars(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    det = Path(args.det)
    cached = sorted(p.stem for p in (det / args.pairs[0].split(":")[0]).glob("*.npz"))
    seqs = cached if args.scenes is None else sorted(set(pd.read_csv(args.scenes).scene) & set(cached))
    missing = [] if args.scenes is None else sorted(set(pd.read_csv(args.scenes).scene) - set(cached))
    print(f"  {len(seqs)} scenes ({len(missing)} requested scenes have no detections)", flush=True)
    if missing:
        raise SystemExit(f"scenes without detections: {missing[:5]} ...")

    db = NuScenesDB(args.dataroot, args.version)
    nad = make_adapter(db)

    t138 = _load("t138", "138_causal_ego_speed.py")
    t138.build(db, seqs, []).to_csv(out / "v_ego_causal.csv", index=False)     # 138's rows; its gates run

    cfg = RiskConfig()
    pp, cp = P.PlannerParams(), P.CostParams()
    toks = token_map(seqs, args.dataroot, args.version)
    for pair in args.pairs:
        cm, fm = pair.split(":")
        for geo in args.geometries:
            t0 = time.time()
            d = decision.build(det, cm, fm, seqs, cfg, pp, cp, G.PRIMARY, range_source=geo, adapter=nad)
            d = decision.add_perception_gain(d, det, cm, fm, seqs, cfg, adapter=nad)
            prim = _pm.primitives(det, cm, fm, seqs, cfg, nad, cfg.min_gt_height)
            d = d.merge(prim, on=["seq", "frame"], how="left", validate="one_to_one")
            d = d.merge(toks, on=["seq", "frame"], how="left", validate="one_to_one")
            assert d.sample_token.notna().all(), "unmapped frames"
            if (cm, fm) == ("ns_cheap_320", "ns_full_640"):
                d = join_planner(d, args.planc, geo)
            f = out / f"{geo}__{cm}to{fm}.pkl"
            d.to_pickle(f)
            print(f"  {f.name}: {len(d)} frames, {d.seq.nunique()} scenes [{time.time() - t0:.0f}s]", flush=True)
    (run / "summary.json").write_text(json.dumps({"scenes": len(seqs), "out": str(out), "pairs": args.pairs}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
