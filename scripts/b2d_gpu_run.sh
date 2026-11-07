#!/usr/bin/env bash
# Follow-up E3: run one evaluator job on a GPU that has at least MIN_FREE_MB free (198's jobs share the GPUs with
# other work). Usage: [MIN_FREE_MB=n] b2d_gpu_run.sh <log> <command...>
# The free memory is checked twice, 60 s apart. The pick and the launch are serialised across this machine's jobs by a
# lock held until 120 s after the launch, so two jobs never count on the same free memory before the first one has
# allocated its own. The command runs with CUDA_VISIBLE_DEVICES set to the GPU; its output is appended to <log>.
log=$1; shift
MIN=${MIN_FREE_MB:-9000}
pick() {   # the GPU with the most free memory, if it has at least MIN
  nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits |
    awk -F', ' -v m=$MIN '$2 - $3 >= m {print $2 - $3, $1}' | sort -rn | head -1 | awk '{print $2}'
}
still_free() {
  nvidia-smi -i $1 --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' -v m=$MIN '$1 - $2 >= m'
}
exec 9>"${TMPDIR:-/tmp}/b2d_gpu_pick.lock"
while true; do
  g=$(pick)
  if [ -n "$g" ]; then
    flock 9
    sleep 60
    [ -n "$(still_free $g)" ] && break
    flock -u 9
  fi
  sleep 120
done
echo "$(date +%T) running on GPU $g: $*" >> "$log"
( CUDA_VISIBLE_DEVICES=$g "$@" >> "$log" 2>&1; echo "$(date +%T) exit $?" >> "$log" ) 9>&- &
pid=$!
sleep 120
flock -u 9; exec 9>&-
wait $pid
