#!/usr/bin/env bash
# Tier B (detection layer) on full data: smoke test, then seed 0, then calibration by visit quintile on the same 6,000
# anchors used for Tier A. The chain stops if the smoke test fails or shows a non-finite loss.
# Usage: nohup bash workbench/run_tierB_full.sh > ~/logs/log_tierB_chain.txt 2>&1 &
set -u
mkdir -p ~/logs ~/results ~/runs
cd ~/delphi-aou
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RUN_MODE=full DETECTION=1 SAMPLE_CHUNK=512 BATCH=${BATCH:-64}
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
echo "== smoke $(date)"
ITERS=300 SEED=99 MAX_EVAL_ANCHORS=1000 python -u scripts/51_train_visit_model.py > ~/logs/log_tierB_smoke.txt 2>&1
echo SMOKE_EXIT $?
# AUC is NaN by design with 1,000 anchors (too few cases per unit); only a non-finite loss fails the smoke test
if ! grep -q "STAGE 5 COMPLETE" ~/logs/log_tierB_smoke.txt || grep -qiE "val (nan|inf)" ~/logs/log_tierB_smoke.txt; then
  echo "SMOKE_FAILED $(date)"; exit 1
fi
fi
echo "== tier B full seed ${SEED:-0} $(date)"
ITERS=${ITERS:-30000} SEED=${SEED:-0} python -u scripts/51_train_visit_model.py > ~/logs/log_tierB_full_s${SEED:-0}.txt 2>&1
echo TIERB_EXIT $?
echo "== quintile calibration $(date)"
TIER=B SEED=${SEED:-0} N_ANCHORS=6000 ESTIMATORS=rb TAG=_q python -u experiments/rescore_visit_model.py > ~/logs/log_tierB_rescore_s${SEED:-0}.txt 2>&1
echo RESCORE_EXIT $?
echo "TIERB_DONE $(date)"
