#!/usr/bin/env bash
# Follow-up E3, on the simulation machine: stop evaluators of this repository's runs (checkpoint under $D/out) whose
# CARLA server has died. A CARLA segfault can leave the leaderboard evaluator waiting forever, holding a job slot.
# Every 5 min: an evaluator is stopped when no live CarlaUE4-Linux-Shipping process serves its --port and its job
# log has not changed for 15 min, or when its server has been found gone on checks at least 10 min apart: after
# a segfault the evaluator can keep writing one time-out line per actor every 10 min (route 11755, 2026-10-16), so
# the log alone never goes idle. Its checkpoint then has no record; 198 handles it as a crash.
# A CARLA server can also stay up but stop ticking, leaving the evaluator blocked in an RPC call (route 28087,
# 2026-10-15): when the job log has not changed for 30 min while the server is alive, that server is stopped first
# (TERM, then KILL after 60 s), then the evaluator. The agent logs every tick, so 30 min of silence is a hang.
# Log: $D/logs/watchdog.log. DRY=1 ONCE=1: report only, one pass.
set -u
B=${B:?set B to the work root on the GPU machine}
D=$B/deep_ext/b2d
# one file per evaluator PID, created when its server is first found gone; one folder per machine (PIDs of two
# machines sharing the folder can coincide). Every machine that runs evaluators runs its own watchdog.
ST=$D/logs/.watchdog_dead_$(hostname -s)
mkdir -p $ST
while true; do
  for f in $ST/*; do [ -e "$f" ] && ! kill -0 "$(basename $f)" 2>/dev/null && rm -f "$f"; done
  for p in $(pgrep -f "leaderboard_evaluator.py"); do
    cmd=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null) || continue
    case "$cmd" in python*) ;; *) continue ;; esac
    ck=$(echo "$cmd" | grep -o "checkpoint=$D/out/[^ ]*" | cut -d= -f2)
    port=$(echo "$cmd" | grep -o -- "--port=[0-9]*" | cut -d= -f2)
    [ -n "$ck" ] && [ -n "$port" ] || continue
    alive=0; servers=""
    for c in $(pgrep -f "CarlaUE4-Linux-Shipping.*-carla-rpc-port=$port( |$)"); do
      [ "$(awk '{print $3}' /proc/$c/stat 2>/dev/null)" != "Z" ] && { alive=1; servers="$servers $c"; }
    done
    log=$D/logs/$(basename $ck .json).log
    [ "${DRY:-0}" = 1 ] && echo "evaluator $p $(basename $ck) port $port carla_alive $alive log_idle_15min $([ -n "$(find $log -mmin +15 2>/dev/null)" ] && echo yes || echo no) log_idle_30min $([ -n "$(find $log -mmin +30 2>/dev/null)" ] && echo yes || echo no)"
    [ $alive = 1 ] && rm -f $ST/$p
    [ $alive = 0 ] && [ "${DRY:-0}" != 1 ] && [ ! -e $ST/$p ] && touch $ST/$p
    if [ $alive = 0 ] && { { [ -f $log ] && [ -n "$(find $log -mmin +15)" ]; } || [ -n "$(find $ST/$p -mmin +10 2>/dev/null)" ]; }; then
      [ "${DRY:-0}" = 1 ] && { echo "would stop $p $(basename $ck) port $port"; continue; }
      echo "$(date '+%F %T') $(hostname -s) stop evaluator $p ($(basename $ck)): CARLA on port $port gone (log idle > 15 min or gone > 10 min)" >> $D/logs/watchdog.log
      kill $p; sleep 20; kill -0 $p 2>/dev/null && kill -9 $p; rm -f $ST/$p
    elif [ $alive = 1 ] && [ -f $log ] && [ -n "$(find $log -mmin +30)" ]; then
      [ "${DRY:-0}" = 1 ] && { echo "would stop CARLA$servers and evaluator $p $(basename $ck) port $port"; continue; }
      echo "$(date '+%F %T') $(hostname -s) stop evaluator $p ($(basename $ck)): CARLA on port $port alive but log idle > 30 min; server$servers stopped first" >> $D/logs/watchdog.log
      kill $servers 2>/dev/null; sleep 60
      for c in $servers; do kill -0 $c 2>/dev/null && kill -9 $c; done
      sleep 30; kill $p 2>/dev/null; sleep 30; kill -0 $p 2>/dev/null && kill -9 $p
    fi
  done
  [ "${ONCE:-0}" = 1 ] && break
  sleep 300
done
