#!/usr/bin/env python3
"""Follow-up E8a, on the training machine: LiDAR points inside every labelled Argoverse 1.1 val box.

Section 3's reference objects need at least one LiDAR or radar point; Argoverse 1 releases no count per box, so it is
counted here from the released sweeps (addendum E8 in docs/followup_protocol.md, the change after gate E8-0). The val
archive (61 GB, public bucket) is streamed once at a capped rate and only its LiDAR sweeps are kept; each sweep and its
labels share the ego frame of the sweep time.

Writes <out>/counts/<log>.csv (timestamp, track_label_uuid, num_pts) and <out>/summary.json.
"""
from __future__ import annotations

import argparse, io, json, os, subprocess, tarfile
from pathlib import Path

import numpy as np

URL = "https://argoverse.s3.amazonaws.com/datasets/av1.1/tars/tracking_val_v1.1.tar.gz"
PLY_TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "uchar": "u1", "uint8": "u1", "int": "<i4",
             "short": "<i2", "ushort": "<u2"}


def read_ply(b: bytes) -> np.ndarray:
    head, _, body = b.partition(b"end_header\n")
    n, props = 0, []
    for line in head.decode().splitlines():
        t = line.split()
        if t[:2] == ["element", "vertex"]:
            n = int(t[2])
        elif t[:1] == ["property"]:
            props.append((t[2], PLY_TYPES[t[1]]))
    a = np.frombuffer(body, dtype=np.dtype(props), count=n)
    return np.stack([a["x"], a["y"], a["z"]], 1).astype(np.float64)


def count(points, labels):
    out = []
    for a in labels:
        c = np.array([a["center"][k] for k in "xyz"])
        q = a["rotation"]
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        p = (points - c) @ R                                   # box frame
        inside = (np.abs(p[:, 0]) <= a["length"] / 2) & (np.abs(p[:, 1]) <= a["width"] / 2) & \
                 (np.abs(p[:, 2]) <= a["height"] / 2)
        out.append((a["track_label_uuid"], int(inside.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="the val logs' folder with per_sweep_annotations_amodal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rate", default="25M")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "counts").mkdir(parents=True, exist_ok=True)
    curl = subprocess.Popen(["curl", "-sfL", "--limit-rate", args.rate, "--retry", "5", URL], stdout=subprocess.PIPE)
    rows, n_sweeps = {}, 0
    with tarfile.open(fileobj=curl.stdout, mode="r|gz") as t:
        for m in t:
            parts = m.name.split("/")
            if not (m.isfile() and "lidar" in parts and m.name.endswith(".ply")):
                continue
            log = parts[parts.index("lidar") - 1]
            ts = int(parts[-1].split("_")[-1][:-4])
            lab = Path(args.labels) / log / "per_sweep_annotations_amodal" / f"tracked_object_labels_{ts}.json"
            if not lab.exists():
                continue
            pts = read_ply(t.extractfile(m).read())
            for uuid, k in count(pts, json.loads(lab.read_text())):
                rows.setdefault(log, []).append((ts, uuid, k))
            n_sweeps += 1
            if n_sweeps % 500 == 0:
                print(n_sweeps, "sweeps", flush=True)
    curl.wait()
    for log, r in rows.items():
        with open(out / "counts" / f"{log}.csv", "w") as f:
            f.write("timestamp,track_label_uuid,num_pts\n")
            f.writelines(f"{a},{b},{c}\n" for a, b, c in r)
    (out / "summary.json").write_text(json.dumps({"logs": len(rows), "sweeps": n_sweeps,
                                                  "boxes": sum(len(r) for r in rows.values()),
                                                  "curl_exit": curl.returncode}, indent=1))
    print("done", len(rows), "logs", n_sweeps, "sweeps")


if __name__ == "__main__":
    main()
