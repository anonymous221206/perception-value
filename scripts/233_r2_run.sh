#!/usr/bin/env bash
# Follow-up round 2, R2 on the training machine's CPU (addendum R2 in docs/followup_protocol.md), from $B/deep_ext/code:
# spec -> vall (sharded) -> gate R2-0 -> forks and policies (sharded; PDM-Closed, then IDM) -> analysis.
# Every per-scenario stage must leave 60 scenario files, or the chain stops. Marker done at the end.
set -u
B=${B:?set B to the work root on the GPU machine}
C=$B/deep_ext/code
cd $C
export RAP_NUPLAN_ROOT=$B/datasets/nuplan PYTHONPATH=$C/src PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       CUDA_VISIBLE_DEVICES=
PY=$B/envs/nuplan/bin/python
S=scripts/232_r2_closedloop.py
N=${N:-12}
L=$C/log_r2
mkdir -p $L
log() { echo "$(date '+%F %T') $(hostname -s) $*" >> $L/r2.log; }
par() {   # par <stage> <checkpoint folder> [args]: N shards, then 60 scenario files or stop
  local st=$1 ck=$2; shift 2
  local tag=$(echo "$*" | tr -c 'A-Za-z0-9_\n' '_')       # a log name without spaces (a space broke the redirect)
  for i in $(seq 0 $((N - 1))); do $PY $S --stage $st --shard $i/$N "$@" > "$L/${st}${tag}_$i.log" 2>&1 & done
  wait
  local n=$(ls $ck/*.json 2>/dev/null | wc -l)
  log "$st $*: $n scenario files"
  [ "$n" = 60 ] || { log "$st $* FAILED (see $L/${st}_*)"; exit 1; }
}
R=data/cache/nuplan_real
if [ ! -s $R/r2_spec.pkl ]; then
  log "spec"; $PY $S --stage spec > $L/spec.log 2>&1 || { log "spec FAILED"; exit 1; }
fi
log "vall"; par vall $R/r2_vall
G=results/followup/R2/gate_r2_0.json
if grep -q '"pass": true' $G 2>/dev/null; then
  log "gate R2-0 already passed ($G)"
else
  log "gate"; $PY $S --stage gate > $L/gate.log 2>&1 || { log "gate R2-0 FAILED (stops R2)"; exit 1; }
fi
for p in pdm_closed idm; do
  log "forks $p"; par forks $R/r2_forks_$p --planner $p
  log "policies $p"; par policies $R/r2_policies_$p --planner $p
done
log "analysis"; $PY $S --stage analysis > $L/analysis.log 2>&1 || { log "analysis FAILED"; exit 1; }
log "done"; touch $L/done
