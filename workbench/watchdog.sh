#!/usr/bin/env bash
# GPU queue watchdog: every 10 minutes write ~/logs/health_status.txt (latest) and append to ~/logs/health.txt.
# Flags: ERROR (a traceback or fatal OOM in an active log), STALL (job running but its log unchanged for 75 min),
# IDLE (no job running while the queue script is still alive, or GPU idle for 60 min with a job running),
# DONE (queue finished: nothing running). It never kills or restarts anything.
LOGS="$HOME/logs/log_audit_full.txt $HOME/log_synth_hard.txt $HOME/logs/log_semisyn.txt $HOME/logs/log_tierB_chain_s1.txt $HOME/logs/log_tierB_chain_s2.txt $HOME/log_tierB_full_s1.txt $HOME/log_tierB_full_s2.txt $HOME/log_tierB_rescore_s1.txt $HOME/log_tierB_rescore_s2.txt"
idle=0
while true; do
  now=$(date +%s); flags=""
  jobs=$(ps -eo args | grep -E "python -u .*(run_audit|synthetic_detection|semisynthetic_rho|51_train_visit_model|rescore_tierA)" | grep -v grep | cut -c1-80)
  queue=$(ps -eo args | grep -E "bash .*queue3.sh" | grep -v grep | wc -l)
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
  for f in $LOGS; do
    [ -f "$f" ] || continue
    age=$(( now - $(stat -c %Y "$f") ))
    if tail -n 30 "$f" | grep -qE "Traceback|CUDA out of memory|RuntimeError|Killed"; then flags="$flags ERROR:$(basename $f)"; fi
    if [ -n "$jobs" ] && [ $age -gt 4500 ] && [ $age -lt 14400 ] && ! tail -n 3 "$f" | grep -qE "COMPLETE|DONE|EXIT"; then flags="$flags STALL:$(basename $f):${age}s"; fi
  done
  if [ -n "$jobs" ] && [ "${util:-0}" -lt 5 ]; then idle=$((idle + 1)); else idle=0; fi
  [ $idle -ge 6 ] && flags="$flags IDLE:gpu_util_${util}"
  [ -z "$jobs" ] && [ "$queue" -gt 0 ] && flags="$flags IDLE:queue_waiting_no_job"
  [ -z "$jobs" ] && [ "$queue" -eq 0 ] && flags="$flags DONE"
  [ -z "$flags" ] && flags="OK"
  { echo "$(TZ=America/Los_Angeles date '+%Y-%m-%d %H:%M %Z') gpu=${util}% flags:$flags"; echo "jobs:"; echo "${jobs:-none}"; } > ~/logs/health_status.txt
  echo "$(TZ=America/Los_Angeles date '+%m-%d %H:%M') gpu=${util}% $flags" >> ~/logs/health.txt
  sleep 600
done
