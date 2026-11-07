#!/usr/bin/env bash
# Follow-up E3 (c), on each simulation machine: the routing stages in order, so that they do not depend on a
# watcher elsewhere (addendum E3c in docs/followup_protocol.md). Machines share the workspace folder and split the runs as they go:
# in each pass, a machine with a free job slot claims the next run (198 CLAIM), so no run is handled by two machines.
# A pass ends on every machine before any machine starts the next one: a run still going on one machine is then never
# taken for a crash by another. Each pass reruns once every run without a record, and a run is given up after three
# crashes, as in the ladder; three passes of each case let that rule apply.
#   1. development runs: 198 routedev, passes 1-3
#   2. FIT=1 (one machine only): 203 --stage fit on the tick logs of all machines
#      -> $R/{gate.pkl, gate_dev.json, gate_dev_frames.csv}; the other machines wait for gate.pkl
#   3. routing runs: 198 route, passes 1-3; then marker $R/routedone.<host>
# Environment: CARLA_ENV (as for 198), HOSTS (all machines taking part, e.g. "host1 host2"), FIT (0/1), MAXJ (3); DEV_PASSES (default
# "1 2 3"; empty skips step 1) and DEV_HOSTS (the machines taking part in those passes, default HOSTS), for a restart.
set -u
: ${CARLA_ENV:?set CARLA_ENV to the CARLA environment script, see scripts/b2d_env.example.sh}
B=${B:?set B to the work root on the GPU machine}
C=$(cd "$(dirname "$0")" && pwd)
D=$B/deep_ext/b2d
R=$D/router
H=$(hostname -s)
export MAXJ=${MAXJ:-3}
log() { echo "$(date '+%F %T') $H $*"; }
pass() {   # pass <case> <n> [hosts]: this machine's share of the pass, then wait until every machine has finished it
  export CLAIM=$R/claims/$1.pass$2
  mkdir -p $CLAIM
  log "$1 pass $2"
  bash $C/198_b2d_dispatch.sh $1
  touch $R/$1.pass$2.done.$H
  for h in ${3:-${HOSTS:-$H}}; do
    until [ -f $R/$1.pass$2.done.$h ]; do sleep 120; done
  done
}
mkdir -p $R
for n in ${DEV_PASSES-1 2 3}; do pass routedev $n "${DEV_HOSTS:-${HOSTS:-$H}}"; done
if [ "${FIT:-0}" = 1 ]; then
  log "fit"
  ( . $CARLA_ENV && python $C/203_b2d_routing.py --stage fit \
      --ticks $D/ticks/*/ticks_rdev.jsonl --router_dir $R ) || { log "fit failed"; exit 1; }
fi
until [ -f $R/gate.pkl ]; do sleep 120; done
for n in 1 2 3; do pass route $n; done
touch $R/routedone.$H
log "routing done"
