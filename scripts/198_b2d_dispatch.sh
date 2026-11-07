#!/usr/bin/env bash
# Follow-up E3 (b), on the simulation machine: forked rollouts of PDM-Lite with perception in the loop
# (addendum E3 in docs/followup_protocol.md). One Bench2Drive route per evaluator run, the pilot of scripts/195_b2d_pilot.py.
#   198_b2d_dispatch.sh routes [N]        split Bench2Drive's 220-route file into one file per route (first N by id,
#                                         default 40; more when some of the first 40 do not complete, addendum E3)
#   198_b2d_dispatch.sh baseline A|B      cheap with the full mode in shadow, every route, tag base_A / base_B
#   198_b2d_dispatch.sh forks <jobs.csv>  one run per (route, fork_tick): full on that frame, cheap elsewhere
#   198_b2d_dispatch.sh ladder            uniform ladder (a): every route x privileged / cheap / full / u960 x 3 reps
#   198_b2d_dispatch.sh routedev          routing (c), development: first 50 routes by id, 202 with ROUTER=dev
#   198_b2d_dispatch.sh route             routing (c): the other 170 routes x gate / random x 3 reps (202; needs
#                                         $D/router/gate.pkl from 203 --stage fit)
#   CLAIM=<folder> (several machines sharing the workspace, one pass each, 204): a machine with a free job slot
#   claims a run by creating <folder>/<run> (mkdir is atomic on the shared folder) and only then moves a crashed
#   checkpoint aside and launches; a run claimed by another machine is skipped. Tick logs of routedev / route go to
#   $D/ticks/<host>/ (appends from two machines to one file are not safe on NFS)
# Environment: B (work root on the simulation machine), D=$B/deep_ext/b2d. CARLA_ENV: a script that sets up CARLA
# and Bench2Drive (scripts/b2d_env.example.sh); it is sourced again inside each job. This repository's src/ and
# scripts/; YOLOv8s weights in $RAP_MODELS (default models/). MAXJ parallel evaluators, each started through
# scripts/b2d_gpu_run.sh on a GPU with MIN_FREE_MB free.
set -u
: ${CARLA_ENV:?set CARLA_ENV to the CARLA environment script, see scripts/b2d_env.example.sh}
B=${B:?set B to the work root on the GPU machine}
D=$B/deep_ext/b2d
MAXJ=${MAXJ:-2}
. $CARLA_ENV
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export DEEP_SRC=$ROOT/src DEEP_WEIGHTS=${RAP_MODELS:-$ROOT/models}/yolov8s.pt
AGENT=$ROOT/scripts/195_b2d_pilot.py
AGENT_RT=$ROOT/scripts/202_b2d_route_pilot.py
mkdir -p $D/routes $D/logs $D/out $D/ticks

# jobs alive, running or still waiting for a GPU: one b2d_gpu_run.sh wrapper per job stays alive until the job
# exits (it and its subshell share a command line), so count distinct job logs (the wrapper's first argument)
n_eval() { for p in /proc/[0-9]*; do { tr '\0' '\n' < $p/cmdline; } 2>/dev/null | grep -A1 -s "b2d_gpu_run.sh" | sed -n 2p
  done | grep "$D" | sort -u | wc -l; }

FREE_PORT='import random, socket
def free(p):
    for q in (p, p + 1, p + 2, p + 2000):
        s = socket.socket()
        try:
            s.bind((str(), q))
        except OSError:
            return False
        finally:
            s.close()
    return True
p = random.randint(20000, 38000)
while not free(p):
    p = random.randint(20000, 38000)
print(p)'

TH=$D/ticks/$(hostname -s)

one() {   # one() <route file> <tag> <env assignments...>; AG=<agent file> overrides the pilot, SAVE_DIR the tick logs
  local R=$1 TAG=$2; shift 2
  local OUT=$D/out/$(basename $R .xml)_$TAG.json NAME=$(basename $R .xml)_$TAG
  [ -n "${CLAIM:-}" ] && [ -d $CLAIM/$NAME ] && return 0
  local L=$D/logs/$(basename $R .xml)_$TAG
  [ -f $OUT ] && python - "$OUT" "$L.log" <<'PY' && return 0
import json, sys
c = json.load(open(sys.argv[1]))["_checkpoint"]
p = c["progress"]
# done only with a record: a simulator crash ("please restart") leaves progress [1, 1] and no record
ok = bool(p and p[0] >= p[1] and c["records"])
# a record that is a machine failure, not driving, is not done either: the agent could not be set up, or it
# crashed because the GPU ran out of memory (another user's job grew after the free-memory check)
if ok:
    st = c["records"][0]["status"]
    try:
        log = open(sys.argv[2], errors="replace").read() if st.startswith("Failed - Agent") else ""
    except OSError:
        log = ""
    if st == "Failed - Agent couldn't be set up" or (st == "Failed - Agent crashed" and "out of memory" in log):
        ok = False
sys.exit(0 if ok else 1)
PY
  # a checkpoint without a record (simulator crash, or an evaluator stopped by 201's watchdog) or with a machine-
  # failure record is a crashed run: it is moved aside (checkpoint and log as .crash<k>) and run again; after three
  # crashes the run is given up
  if [ -n "${CLAIM:-}" ]; then
    # a free slot first, then the claim: a machine claims a run only when it can start it
    while [ "$(n_eval)" -ge $MAXJ ]; do sleep 60; done
    mkdir $CLAIM/$NAME 2>/dev/null || return 0
    hostname -s > $CLAIM/$NAME/host
  fi
  if [ -f $OUT ]; then
    local k=1
    while [ -f ${OUT%.json}.crash$k.json ]; do k=$((k + 1)); done
    mv $OUT ${OUT%.json}.crash$k.json; [ -f $L.log ] && mv $L.log $L.crash$k.log
  fi
  [ -f ${OUT%.json}.crash3.json ] && { echo "give up $(basename $OUT): three crashes"; return 0; }
  while [ "$(n_eval)" -ge $MAXJ ]; do sleep 60; done
  # the CARLA ports are picked when the job starts (after the GPU wait), free at that moment: rpc PORT, streaming
  # PORT + 1, secondary PORT + 2 (CARLA 0.9.15's default), traffic manager PORT + 2000
  MIN_FREE_MB=${MIN_FREE_MB:-9000} bash $ROOT/scripts/b2d_gpu_run.sh $D/logs/$(basename $R .xml)_$TAG.log env "$@" \
    SAVE_PATH=${SAVE_DIR:-$D/ticks} ROUTES=$R DATAGEN=0 bash -c ". $CARLA_ENV; \
      export DEEP_SRC=$DEEP_SRC DEEP_WEIGHTS=$DEEP_WEIGHTS; PORT=\$(python -c '$FREE_PORT'); \
      cd \$WORK_DIR && python \$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py \
      --routes=$R --repetitions=1 --track=SENSORS --checkpoint=$OUT --agent=${AG:-$AGENT} --agent-config=none --debug=0 \
      --resume=True --port=\$PORT --traffic-manager-port=\$((PORT + 2000)) --gpu-rank=\${CUDA_VISIBLE_DEVICES:-0}" &
  sleep 30
}

case $1 in
  routes)
    python - "$WORK_DIR/leaderboard/data/bench2drive220.xml" "$D/routes" "${2:-40}" <<'PY'
import sys, copy
import xml.etree.ElementTree as ET
src, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
t = ET.parse(src); root = t.getroot()
routes = sorted(root.findall("route"), key=lambda r: int(r.get("id")))[:n]
for r in routes:
    nr = ET.Element(root.tag, root.attrib); nr.append(copy.deepcopy(r))
    ET.ElementTree(nr).write(f"{out}/route_{r.get('id')}.xml")
print(len(routes), "route files")
PY
    ;;
  baseline)
    for R in $D/routes/route_*.xml; do
      one $R base_$2 PERC_BRANCH=cheap SHADOW=1 DEEP_TAG=base_$2
    done; wait ;;
  forks)
    # process substitution, not a pipeline: the loop runs in this shell, so `wait` waits for its jobs
    while IFS=, read route frame tick eligible; do
      one $D/routes/$route.xml fork_$tick PERC_BRANCH=cheap FORK_TICK=$tick FORK_BRANCH=full DEEP_TAG=fork
    done < <(tail -n +2 $2); wait ;;
  ladder)
    # uniform ladder (a): every route file in $D/routes (run `routes 220` first), branches privileged, cheap, full,
    # u960, repetitions 0..2; route-major within a repetition, so a stopped run leaves paired branches
    for REP in 0 1 2; do
      for F in $(cd $D/routes && ls route_*.xml | sort -t_ -k2 -n); do
        R=$D/routes/$F
        for BR in privileged cheap full u960; do
          one $R lad_${BR}_r$REP PERC_BRANCH=$BR DEEP_TAG=lad_${BR}_r$REP
        done
      done
    done; wait ;;
  routedev)
    # routing (c), development data: cheap with the full mode in shadow, braking requirement per mode, gate features
    for F in $(cd $D/routes && ls route_*.xml | sort -t_ -k2 -n | head -50); do
      SAVE_DIR=$TH AG=$AGENT_RT one $D/routes/$F rdev PERC_BRANCH=cheap SHADOW=1 ROUTER=dev DEEP_TAG=rdev
    done; wait ;;
  route)
    # routing (c): gate and random at budget 0.2 on the routes after the first 50, repetitions 0..2, route-major;
    # the random policy's generator is seeded with route id x 10 + repetition
    [ -f $D/router/gate.pkl ] || { echo "no gate: run 203 --stage fit first"; exit 1; }
    for REP in 0 1 2; do
      for F in $(cd $D/routes && ls route_*.xml | sort -t_ -k2 -n | tail -n +51); do
        ID=${F//[^0-9]/}
        for POL in gate random; do
          SAVE_DIR=$TH AG=$AGENT_RT one $D/routes/$F rt_${POL}_r$REP PERC_BRANCH=cheap ROUTER=$POL ROUTER_MODEL=$D/router/gate.pkl \
            ROUTER_BUDGET=0.2 ROUTER_SEED=$((ID * 10 + REP)) DEEP_TAG=rt_${POL}_r$REP
        done
      done
    done; wait ;;
esac
