#!/usr/bin/env bash
# Follow-up E11, on the training machine: LTF (NAVSIM's released seed-0 checkpoint) scored on navtest by NAVSIM v1.1's
# run_pdm_score, once per branch (addendum E11 in docs/followup_protocol.md). CPU only: 8 worker processes, 2 threads each,
# so the machine's GPU jobs keep their share. Starts after 219's cameras. Output: $W/<branch>/<timestamp>.csv
# The navtest metric cache is built here with NAVSIM's own caching script (devkit v1.1), as E2's was.
# Run in NAVSIM v1.1's environment, as 192 (OPENSCENE_DATA_ROOT, NAVSIM_DEVKIT_ROOT, NUPLAN_MAPS_ROOT,
# NUPLAN_MAP_VERSION, NAVSIM_EXP_ROOT). Marker $W/done only when both branches wrote their table.
# REPEAT=1: afterwards, branch full once more into $W/full_repeat (diagnostic, not registered: the run-to-run floor of V).
# SEED=1 or 2: the secondary seeds' checkpoints; tables in $W/full_s<seed>, $W/cheap_s<seed>, marker $W/done_s<seed>.
set -u
: ${OPENSCENE_DATA_ROOT:?run in the NAVSIM v1.1 environment} ${NAVSIM_DEVKIT_ROOT:?}
B=${B:?set B to the work root on the GPU machine}
D=$B/deep_ext
W=$D/navsim/e11
CAMS=$D/navsim_cams
BLOBS=$D/navsim_cams/merged/test                      # CAM_F0 from the dataset folder, CAM_L0 / CAM_R0 from 219
SEED=${SEED:-0}
S=$([ $SEED = 0 ] || echo _s$SEED)
CK=$D/weights/navsim/ltf_seed_$SEED.ckpt
MC=$D/navsim/metric_cache_test
log() { echo "$(date '+%F %T') $*" >> $D/logs/e11_run.log; }
until [ -f $CAMS/cams.done ]; do sleep 600; done
mkdir -p $W $(dirname $CK) $BLOBS
HF=https://huggingface.co/autonomousvision/navsim_baselines/resolve/a068068540abdc32dd6985830ae8463a7ac2f9d2
[ -s $CK ] || curl -sfL -o $CK $HF/ltf/ltf_seed_$SEED.ckpt     # the revision E11 ran with (sha256 checked)
log "checkpoint md5 $(md5sum $CK | cut -d' ' -f1)"
SRC=$(dirname $(find $CAMS -type d -name CAM_L0 | head -1))/..       # the archives' test folder
for l in $(ls $OPENSCENE_DATA_ROOT/sensor_blobs/test); do
  mkdir -p $BLOBS/$l
  ln -sfn $OPENSCENE_DATA_ROOT/sensor_blobs/test/$l/CAM_F0 $BLOBS/$l/CAM_F0
  for c in CAM_L0 CAM_R0; do [ -d $SRC/$l/$c ] && ln -sfn $(readlink -f $SRC/$l/$c) $BLOBS/$l/$c; done
done
log "merged camera tree: $(ls $BLOBS | wc -l) logs, $(ls -d $BLOBS/*/CAM_L0 | wc -l) with CAM_L0"
export PYTHONPATH=$(cd "$(dirname "$0")" && pwd)/navsim_e11:$PYTHONPATH
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
if ! ls $MC/metadata/*.csv > /dev/null 2>&1; then
  log "metric cache: navtest -> $MC"
  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py train_test_split=navtest cache.cache_path=$MC \
    worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=8 \
    > $D/logs/e11_metric_cache.log 2>&1 || { log "metric cache FAILED"; exit 1; }
fi
log "metric cache: $(find $MC -name metric_cache.pkl | wc -l) tokens, devkit $(git -C $NAVSIM_DEVKIT_ROOT rev-parse --short HEAD)"
score() {   # score <branch> <output name>
  ls $W/$2/*.csv > /dev/null 2>&1 && return 0
  log "branch $2"
  python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py agent=transfuser_agent \
    agent._target_=deep_e11_agent.E11LTFAgent +agent.branch=$1 agent.config.latent=true agent.checkpoint_path=$CK \
    sensor_blobs_path=$BLOBS metric_cache_path=$MC experiment_name=e11_ltf_$2 \
    output_dir=$W/$2 worker=single_machine_thread_pool worker.use_process_pool=true worker.max_workers=8 \
    > $D/logs/e11_$2.log 2>&1 || log "branch $2 FAILED"
  log "branch $2: $(grep -o 'Final average score of valid results: [0-9.]*' $D/logs/e11_$2.log)"
}
score full full$S
score cheap cheap$S
ls $W/full$S/*.csv $W/cheap$S/*.csv > /dev/null 2>&1 && touch $W/done$S && log done$S || { log "a branch wrote no table"; exit 1; }
if [ "${REPEAT:-0}" = 1 ]; then
  score full full_repeat
  ls $W/full_repeat/*.csv > /dev/null 2>&1 && touch $W/repeat.done && log "repeat done"
fi
