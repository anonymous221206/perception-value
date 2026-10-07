#!/usr/bin/env python
"""Follow-up E2: the NAVSIM token table (addendum E2 in docs/followup_protocol.md).

From the OpenScene log metadata (navsim_logs/{test,mini}/<log>.pkl, read as plain pickles):
  navtest  the tokens of NAVSIM v1.1's navtest scene filter (split "test")
  dev      the OpenScene mini tokens that have a NAVSIM metric cache (split "dev_train" / "dev_val": logs sorted by name,
           position mod 5 = 4 -> validation)
Per token: log, CAM_F0 image path and camera model (K, distortion, camera->LiDAR R, t), ego speed, driving command,
and the strata of the addendum (agents within 30 m, lead vehicle ahead within 30 m and 2 m laterally).
Writes data/cache/navsim/tokens.pkl and the image lists for the copy (images_<split>.txt).
"""
from __future__ import annotations

import argparse, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap.paths import NAVSIM_ROOT                                                # noqa: E402

OUT = ROOT / "data" / "cache" / "navsim"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(NAVSIM_ROOT / "navsim_logs"))
    args = ap.parse_args()
    navtest = set(yaml.safe_load((OUT / "navtest.yaml").read_text())["tokens"])
    dev = {Path(l.strip()).name for l in (OUT / "deep_metric_cache_dev.txt").read_text().split()}
    rows = []
    for split, folder, want in (("test", "test", navtest), ("dev", "mini", dev)):
        logs = sorted(p.stem for p in (Path(args.logs) / folder).glob("*.pkl"))
        for li, log in enumerate(logs):
            frames = pickle.load(open(Path(args.logs) / folder / f"{log}.pkl", "rb"))
            for f in frames:
                if f["token"] not in want:
                    continue
                c = f["cams"]["CAM_F0"]
                b, n = np.asarray(f["anns"]["gt_boxes"], float).reshape(-1, 7), np.asarray(f["anns"]["gt_names"])
                dist = np.hypot(b[:, 0], b[:, 1]) if len(b) else np.zeros(0)
                agents = np.isin(n, ["vehicle", "pedestrian", "bicycle"]) & (dist <= 30)
                lead = (n == "vehicle") & (b[:, 0] > 0) & (b[:, 0] <= 30) & (np.abs(b[:, 1]) <= 2) if len(b) else np.zeros(0, bool)
                vx, vy = f["ego_dynamic_state"][:2]
                rows.append({"split": split if split == "test" else ("dev_val" if li % 5 == 4 else "dev_train"),
                             "log": log, "token": f["token"], "image": c["data_path"], "folder": folder,
                             "K": np.asarray(c["cam_intrinsic"], float), "D": np.asarray(c["distortion"], float),
                             "R": np.asarray(c["sensor2lidar_rotation"], float),
                             "t": np.asarray(c["sensor2lidar_translation"], float),
                             "ego_speed": float(np.hypot(vx, vy)), "driving_command": int(np.argmax(f["driving_command"])),
                             "n_agents_30m": int(agents.sum()), "lead_vehicle": bool(lead.any())})
    d = pd.DataFrame(rows)
    # OpenScene mini shares logs with navtest: a development token from such a log is excluded, so no navtest log
    # contributes to allocator fitting (found 2026-10-05: 10 logs, 336 tokens on each side)
    shared = set(d[d.split == "test"].log) & set(d[d.split != "test"].log)
    d.loc[d.split.ne("test") & d.log.isin(shared), "split"] = "dev_excluded_shared_log"
    print("logs shared by navtest and the development set:", len(shared))
    d.to_pickle(OUT / "tokens.pkl")
    for folder in ("test", "mini"):
        (OUT / f"images_{folder}.txt").write_text("\n".join(d[d.folder == folder].image) + "\n")
    print(d.groupby("split").agg(tokens=("token", "size"), logs=("log", "nunique")))
    miss = {"test": len(navtest - set(d.token)), "dev": len(dev - set(d.token))}
    print("tokens without metadata:", miss)


if __name__ == "__main__":
    main()
