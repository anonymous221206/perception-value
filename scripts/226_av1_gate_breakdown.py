#!/usr/bin/env python
"""Follow-up E8a: label-only breakdown of gate E8-0's second run (Section 3's LiDAR-point rule), descriptive only.

addendum E8 in docs/followup_protocol.md. Recall at 640 on the reference boxes >= 30 px, by the share of the box covered
by nearer reference boxes, by class, by contact with the image border and by range; and how far the missed boxes are
from any detection. Run after the gate stopped E8a, it changes nothing. Output on stdout
(results/followup/E8/diag_points.txt).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rap import argo
from rap.cache import DetCache
from rap.mono import box_iou
from rap.paths import CACHE, DATASETS
from rap.risk import RiskConfig, match

root = DATASETS / "argoverse1" / "argoverse-tracking" / "val"
logs = sorted(p.name for p in root.iterdir() if p.is_dir())
ad = argo.make_adapter(root, logs)
cfg = RiskConfig()
rows = []
for l in logs:
    s = argo.unit_name(l)
    geom = ad.geometry(s)
    c = DetCache(Path(CACHE) / "argo1_det" / "ns_full_640" / f"{s}.npz")
    for i, fr in enumerate(c.frames):
        g = geom[(geom["frame"] == int(fr)) & ((geom["y2"] - geom["y1"]) >= 30)]
        if not len(g):
            continue
        d = c.det(i); k = d["conf"] >= 0.25
        gt = np.stack([g["x1"], g["y1"], g["x2"], g["y2"]], 1).astype(float)
        best, _ = match(gt, np.array(["v"] * len(gt)), d["xyxy"][k].astype(float), d["conf"][k], d["coarse"][k], cfg)
        # label-only occlusion proxy: share of the box area covered by nearer reference boxes (all heights >= 10 px)
        ga = geom[(geom["frame"] == int(fr)) & ((geom["y2"] - geom["y1"]) >= 10)]
        for j in range(len(g)):
            x1, y1, x2, y2 = gt[j]
            m = np.zeros((int(y2 - y1) + 1, int(x2 - x1) + 1), bool)
            for o in ga[ga["long_near"] < g["long_near"][j] - 0.5]:
                a1, b1 = max(int(o["x1"] - x1), 0), max(int(o["y1"] - y1), 0)
                a2, b2 = min(int(o["x2"] - x1), m.shape[1]), min(int(o["y2"] - y1), m.shape[0])
                if a2 > a1 and b2 > b1:
                    m[b1:b2, a1:a2] = True
            rows.append((str(g["coarse"][j]), float(best[j]), float(m.mean()), float(g["long_near"][j]), float(x1 <= 1 or x2 >= 1918 or y2 >= 1198)))
r = np.array([x[1:] for x in rows]); cls = np.array([x[0] for x in rows])
print("n", len(r), "recall", (r[:, 0] >= 0.5).mean())
un = r[r[:, 0] < 0.5]
print("unmatched best-IoU: <0.1 %.3f, 0.1-0.3 %.3f, 0.3-0.5 %.3f" % ((un[:, 0] < 0.1).mean(), ((un[:, 0] >= 0.1) & (un[:, 0] < 0.3)).mean(), (un[:, 0] >= 0.3).mean()))
for lo, hi in ((0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)):
    s = (r[:, 1] >= lo) & (r[:, 1] < hi)
    print(f"covered {lo:.2f}-{hi:.2f}: n {s.sum():6d} recall {(r[s, 0] >= 0.5).mean():.3f}")
for c_ in sorted(set(cls)):
    s = cls == c_
    print(c_, s.sum(), "recall %.3f" % (r[s, 0] >= 0.5).mean())
s = r[:, 3] == 1
print("touching image border", s.sum(), "recall %.3f" % (r[s, 0] >= 0.5).mean(), "not touching %.3f" % (r[~s, 0] >= 0.5).mean())
for lo, hi in ((0, 10), (10, 20), (20, 40), (40, 200)):
    s = (r[:, 2] >= lo) & (r[:, 2] < hi)
    print(f"range {lo}-{hi} m: n {s.sum()} recall {(r[s, 0] >= 0.5).mean():.3f}")
