#!/usr/bin/env python
"""Follow-up E8b: label-only breakdown of gate E8-0 (Argoverse 2, raw images, 640 mode). Recall of the reference boxes
>= 30 px high by box height, by the share covered by nearer reference boxes, by range and by class. Descriptive only;
run after the gate failed, it changes nothing. Output on stdout (results/followup/E8/av2/diag_gate.txt).
"""
import importlib.util, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from rap.cache import DetCache
from rap.paths import CACHE
from rap.risk import RiskConfig, match
s = importlib.util.spec_from_file_location("t216", "scripts/216_av2_tables.py"); t = importlib.util.module_from_spec(s); s.loader.exec_module(t)
sl, split, ad = t.setup()
cfg = RiskConfig()
scale = min(384 / 1830, 640 / 1550)
rows = []
for u in [u for u, sp in split.items() if sp == "val"]:
    geom = ad.geometry(u)
    c = DetCache(Path(CACHE) / "argo2_det_raw" / "ns_full_640" / f"{u}.npz")
    for i, fr in enumerate(c.frames):
        ga = geom[(geom["frame"] == int(fr)) & ((geom["y2"] - geom["y1"]) >= cfg.min_gt_height)]
        g = ga[(ga["y2"] - ga["y1"]) >= 30]
        if not len(g):
            continue
        d = c.det(i); k = d["conf"] >= cfg.thr("full")
        gt = np.stack([ga["x1"], ga["y1"], ga["x2"], ga["y2"]], 1).astype(float)
        best, _ = match(gt, np.array(["v"] * len(gt)), d["xyxy"][k].astype(float), d["conf"][k], d["coarse"][k], cfg)
        for j in np.flatnonzero((ga["y2"] - ga["y1"]) >= 30):
            x1, y1, x2, y2 = gt[j]
            m = np.zeros((int(y2 - y1) + 1, int(x2 - x1) + 1), bool)
            for o in ga[ga["long_near"] < ga["long_near"][j] - 0.5]:
                a1, b1 = max(int(o["x1"] - x1), 0), max(int(o["y1"] - y1), 0)
                a2, b2 = min(int(o["x2"] - x1), m.shape[1]), min(int(o["y2"] - y1), m.shape[0])
                if a2 > a1 and b2 > b1:
                    m[b1:b2, a1:a2] = True
            rows.append((str(ga["coarse"][j]), float(best[j]), float(m.mean()), float(ga["long_near"][j]), y2 - y1))
r = np.array([x[1:] for x in rows]); cls = np.array([x[0] for x in rows]); hit = r[:, 0] >= cfg.iou_thr
print("n", len(r), "recall %.3f" % hit.mean(), "network scale %.3f" % scale)
un = r[~hit]
print("unmatched best-IoU: <0.1 %.3f, 0.1-0.3 %.3f, 0.3-0.5 %.3f" % ((un[:, 0] < 0.1).mean(), ((un[:, 0] >= 0.1) & (un[:, 0] < 0.3)).mean(), (un[:, 0] >= 0.3).mean()))
for lo, hi in ((30, 48), (48, 72), (72, 120), (120, 240), (240, 4000)):
    s_ = (r[:, 3] >= lo) & (r[:, 3] < hi)
    print(f"height {lo}-{hi} px (network {lo*scale:.0f}-{hi*scale:.0f}): n {s_.sum():6d} recall {hit[s_].mean():.3f}")
for lo, hi in ((0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
    s_ = (r[:, 1] >= lo) & (r[:, 1] < hi)
    print(f"covered {lo:.2f}-{hi:.2f}: n {s_.sum():6d} recall {hit[s_].mean():.3f}")
s_ = r[:, 1] < 0.01
for lo, hi in ((30, 48), (48, 72), (72, 4000)):
    q = s_ & (r[:, 3] >= lo) & (r[:, 3] < hi)
    print(f"uncovered, height {lo}-{hi}: n {q.sum()} recall {hit[q].mean():.3f}")
for c_ in sorted(set(cls)):
    q = cls == c_
    print(c_, q.sum(), "recall %.3f" % hit[q].mean())
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 300)):
    q = (r[:, 2] >= lo) & (r[:, 2] < hi)
    print(f"range {lo}-{hi} m: n {q.sum()} recall {hit[q].mean():.3f}")
