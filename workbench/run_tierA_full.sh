#!/usr/bin/env bash
# Tier A on full data, after the replication controls free the GPU. One seed first; seeds 1 and 2 follow
# only after seed 0 is checked. Usage: nohup bash workbench/run_tierA_full.sh > ~/logs/log_tierA_chain.txt 2>&1 &
set -u
mkdir -p ~/logs ~/results ~/runs
cd ~/delphi-aou
while ! grep -q "CTA_DONE full" ~/logs/log_ctA_full.txt 2>/dev/null; do sleep 60; done
echo "== visits full $(date)"
RUN_MODE=full python -u scripts/50_build_visits.py > ~/logs/log_visits_full.txt 2>&1; echo VISITS_EXIT $?
echo "== tier A full seed ${SEED:-0} $(date)"
RUN_MODE=full ITERS=${ITERS:-30000} BATCH=${BATCH:-64} SEED=${SEED:-0} SAMPLE_CHUNK=512 \
  python -u scripts/51_train_visit_model.py > ~/logs/log_tierA_full_s${SEED:-0}.txt 2>&1; echo TIERA_EXIT $?
echo "TIERA_DONE $(date)"
