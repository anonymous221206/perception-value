#!/usr/bin/env bash
# Follow-up E8b: what E8b reads of Argoverse 2, from a local copy of the Argoverse 2 sensor dataset.
# Logs: all 150 validation logs and every fifth of the 700 training logs in sorted order (140). Per log: the tables
# (annotations, poses, calibration; written as CSV, so the detection side needs no feather reader) and only the
# ring_front_center images paired with an annotated sweep (nearest image, at most 50 ms away).
# The images are fetched from the public Argoverse 2 bucket, as E8b's were.
# A: the sensor dataset root (train/ and val/, as released). Output: $DATASETS/av2_deep/{av2_tables,sensor}.
set -eu
A=${A:?set A to the Argoverse 2 sensor dataset root}
DATASETS=${RAP_DATASETS:-$(cd "$(dirname "$0")/.." && pwd)/datasets}
T=$DATASETS/av2_deep/av2_tables
DST=$DATASETS/av2_deep/sensor
mkdir -p $T $DST
[ -s $T/files.txt ] || python3 - $A $T <<'PY'
import os, sys
import numpy as np, pandas as pd
A, T = sys.argv[1:]
train = sorted(os.listdir(f"{A}/train"))[::5]
logs = [("val", l) for l in sorted(os.listdir(f"{A}/val"))] + [("train", l) for l in train]
files, n_img, dropped = [], 0, 0
for sp, l in logs:
    d = f"{A}/{sp}/{l}"
    a = pd.read_feather(f"{d}/annotations.feather")
    cams = np.array(sorted(int(f[:-4]) for f in os.listdir(f"{d}/sensors/cameras/ring_front_center") if f.endswith(".jpg")))
    pairs = []
    for s in sorted(a.timestamp_ns.unique()):
        j = int(np.argmin(np.abs(cams - s)))
        if abs(int(cams[j]) - int(s)) <= 50_000_000:
            pairs.append((int(s), int(cams[j])))
        else:
            dropped += 1
    o = f"{T}/{sp}/{l}"
    os.makedirs(o, exist_ok=True)
    a.to_csv(f"{o}/annotations.csv", index=False)
    pd.read_feather(f"{d}/city_SE3_egovehicle.feather").to_csv(f"{o}/poses.csv", index=False)
    pd.read_feather(f"{d}/calibration/intrinsics.feather").to_csv(f"{o}/intrinsics.csv", index=False)
    pd.read_feather(f"{d}/calibration/egovehicle_SE3_sensor.feather").to_csv(f"{o}/extrinsics.csv", index=False)
    pd.DataFrame(pairs, columns=["sweep_ns", "image_ns"]).to_csv(f"{o}/frames.csv", index=False)
    files += [f"{sp}/{l}/sensors/cameras/ring_front_center/{c}.jpg" for _, c in pairs]
    n_img += len(pairs)
open(f"{T}/files.txt", "w").write("\n".join(files) + "\n")
print(len(logs), "logs", n_img, "images", dropped, "sweeps without an image within 50 ms")
PY
U=https://argoverse.s3.amazonaws.com/datasets/av2/sensor
fetch() {   # fetch <path relative to the sensor folder>
  local f=$1
  [ -s $DST/$f ] && return 0
  mkdir -p $(dirname $DST/$f)
  curl -sf --retry 5 -o $DST/$f.part $U/$f && mv $DST/$f.part $DST/$f || echo "FAILED $f"
}
export -f fetch
export DST U
xargs -P 24 -I{} bash -c 'fetch "$@"' _ {} < $T/files.txt
echo "copied $(find $DST -name '*.jpg' | wc -l) images"
