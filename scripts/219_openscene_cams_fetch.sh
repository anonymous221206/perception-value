#!/usr/bin/env bash
# Follow-up E11, on the training machine: the left and right front cameras of OpenScene's test split (navtest's logs),
# from the public OpenScene archives on HuggingFace, at a capped rate; each archive is deleted after its
# CAM_L0 / CAM_R0 images are extracted.
# Output: $OUT/sensor_blobs/test/<log>/CAM_{L0,R0}/*.jpg, $OUT/fetch.log; $OUT/cams.done at the end.
set -u
B=${B:?set B to the work root on the GPU machine}
D=$B/deep_ext
OUT=$D/navsim_cams
TMP=$D/navsim_fetch
U=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1/openscene_sensor_test_camera
RATE=${RATE:-25M}
mkdir -p $OUT $TMP
log() { echo "$(date '+%F %T') $*" >> $OUT/fetch.log; }
for i in $(seq 0 31); do
  f=openscene_sensor_test_camera_$i.tgz
  [ -f $OUT/part$i.done ] && continue
  log "fetch $f"
  curl -sfL --limit-rate $RATE --retry 10 --retry-delay 10 -C - -o $TMP/$f $U/$f || { log "fetch failed $f"; continue; }
  tar -xzf $TMP/$f -C $OUT --wildcards '*/CAM_L0/*' '*/CAM_R0/*' 2> $OUT/tar$i.err || { log "extract failed $f"; continue; }
  rm -f $TMP/$f && touch $OUT/part$i.done
  log "extracted $f: $(find $OUT -name '*.jpg' | wc -l) images so far"
done
[ $(ls $OUT/part*.done 2>/dev/null | wc -l) = 32 ] && touch $OUT/cams.done && log "all 32 parts"
