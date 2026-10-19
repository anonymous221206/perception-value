#!/usr/bin/env bash
# Follow-up E7, on the training machine: the LIDAR_TOP sweeps (non-keyframe LiDAR) of the scenes E7 reads, fetched
# from nuScenes' public mirror. The machine holds the keyframes only. The sweeps sit in all ten trainval blobs, so each
# blob is fetched once, at a capped rate (shared volume and network), and deleted after the sweeps of those scenes are
# extracted from it.
# Scenes: the 150 nuScenes official-validation scenes and the 380 new validation and test scenes of
# results/followup/E1/scene_split.csv (471 scenes).
# Output: $OUT/sweeps/LIDAR_TOP/*.pcd.bin, $OUT/sweeps_needed.txt, $OUT/fetch.log; $OUT/sweeps.done at the end.
set -u
B=${B:?set B to the work root on the GPU machine}
NS=$B/datasets/nuscenes
OUT=${OUT:-$B/deep_ext/nusc_lidar}
TMP=$B/deep_ext/nusc_fetch
U=https://d36yt3mvayqw5m.cloudfront.net/public/v1.0
RATE=${RATE:-25M}
SPLIT=${SPLIT:?set SPLIT to results/followup/E1/scene_split.csv}
mkdir -p $OUT $TMP
log() { echo "$(date '+%F %T') $*" | tee -a $OUT/fetch.log; }

[ -s $OUT/sweeps_needed.txt ] || python3 - "$NS/v1.0-trainval" "$SPLIT" "$OUT/sweeps_needed.txt" <<'PY'
import csv, json, sys
meta, split, out = sys.argv[1:]
keep = {r["scene"] for r in csv.DictReader(open(split))
        if r["official"] == "val" or (r["group"] != "frozen85" and r["deep_split"] in ("val", "test"))}
scene = {s["token"]: s["name"] for s in json.load(open(f"{meta}/scene.json"))}
sample = {s["token"]: scene[s["scene_token"]] for s in json.load(open(f"{meta}/sample.json"))}
calib = {c["token"]: c["sensor_token"] for c in json.load(open(f"{meta}/calibrated_sensor.json"))}
lidar = {s["token"] for s in json.load(open(f"{meta}/sensor.json")) if s["channel"] == "LIDAR_TOP"}
names = sorted(d["filename"] for d in json.load(open(f"{meta}/sample_data.json"))
               if not d["is_key_frame"] and calib[d["calibrated_sensor_token"]] in lidar
               and sample[d["sample_token"]] in keep)
open(out, "w").write("\n".join(names) + "\n")
print(len(keep), "scenes", len(names), "sweeps")
PY
N=$(wc -l < $OUT/sweeps_needed.txt)
log "need $N sweeps"

get() {   # resumable fetch of one blob at the capped rate
  local f=$1 want have
  want=$(curl -sI $U/$f | awk 'tolower($1)=="content-length:" {print $2}' | tr -d '\r')
  for i in $(seq 1 200); do
    have=$(stat -c %s $TMP/$f 2>/dev/null || echo 0)
    if [ -n "$want" ] && [ "$have" -ge "$want" ]; then log "complete $f $have"; return 0; fi
    log "$f attempt $i have=$have want=$want"
    curl -sL --limit-rate $RATE --retry 5 --retry-delay 5 --speed-time 120 --speed-limit 20000 -C - -o $TMP/$f $U/$f
    sleep 3
  done
  return 1
}
for n in 01 02 03 04 05 06 07 08 09 10; do
  [ -f $OUT/blob$n.done ] && continue
  f=v1.0-trainval${n}_blobs.tgz
  get $f || { log "fetch failed $f"; continue; }
  # one streaming pass over the blob; members are matched against a set (tar -T matches a long list linearly)
  python3 - "$TMP/$f" "$OUT" <<'PY' >> $OUT/fetch.log 2>&1 || { log "extraction failed $f, blob kept"; continue; }
import os, shutil, sys, tarfile
blob, out = sys.argv[1:]
need = {l.strip() for l in open(f"{out}/sweeps_needed.txt") if l.strip()}
k = 0
with tarfile.open(blob, "r|gz") as t:
    for m in t:
        if m.isfile() and m.name in need:
            p = os.path.join(out, m.name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with t.extractfile(m) as src, open(p + ".part", "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.replace(p + ".part", p)
            k += 1
print(os.path.basename(blob), "extracted", k)
PY
  rm -f $TMP/$f && touch $OUT/blob$n.done
  log "have $(find $OUT/sweeps/LIDAR_TOP -name '*.pcd.bin' | wc -l) of $N"
done
H=$(cd $OUT && python3 -c "import os,sys; print(sum(not os.path.isfile(l.strip()) for l in open('sweeps_needed.txt')))")
if [ "$H" = 0 ]; then touch $OUT/sweeps.done; log "all $N sweeps present"; else log "$H sweeps missing"; fi
