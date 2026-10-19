#!/usr/bin/env bash
# Follow-up E7, on the training machine: after 205's sweeps, 210's data root and infos, detection with each checkpoint on
# a GPU that has at least MIN_FREE_MB free (the shared-GPU guard; E3 (c) keeps its own slots), then gate E7-0.
# Marker: $OUT/done. Log: $D/logs/e7_chain.log
set -u
B=${B:?set B to the work root on the GPU machine}
D=$B/deep_ext
OUT=$D/nusc_lidar/e7_det
P=$D/envs/pcdet/bin/python
S=$D/code/scripts/210_lidar_detect.py
MIN_FREE_MB=${MIN_FREE_MB:-9000}
export B
log() { echo "$(date '+%F %T') $(hostname -s) $*" >> $D/logs/e7_chain.log; }
free_gpu() {   # the GPU with the most free memory, if it has at least MIN_FREE_MB
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |
    awk -F', ' -v m=$MIN_FREE_MB '$2 >= m {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'
}
until [ -f $D/nusc_lidar/sweeps.done ]; do sleep 300; done
log "sweeps present"
$P $S --stage root > $D/logs/e7_root.log 2>&1 || { log "root FAILED"; exit 1; }
log "root: $(tail -1 $D/logs/e7_root.log)"
for m in pp voxel transfusion; do
  if [ ! -f $OUT/$m/summary.json ]; then
    until g=$(free_gpu) && [ -n "$g" ]; do sleep 120; done
    log "detect $m on GPU $g"
    CUDA_VISIBLE_DEVICES=$g $P $S --stage detect --model $m > $D/logs/e7_detect_$m.log 2>&1 || { log "detect $m FAILED"; continue; }
  fi
  until g=$(free_gpu) && [ -n "$g" ]; do sleep 120; done
  CUDA_VISIBLE_DEVICES=$g $P $S --stage eval --model $m > $D/logs/e7_eval_$m.log 2>&1 || log "eval $m FAILED"
  log "gate $m: $(cat $OUT/$m/gate_e7_0.json 2>/dev/null | tr -d '\n ')"
done
touch $OUT/done
log "e7 chain done"
