#!/usr/bin/env bash
# Plan steps 1 and 2 after the Delphi-2M replication (inference only, no retraining).
#   0. back up the converted dataset, checkpoints and evaluation outputs to the workspace bucket
#   1. out-of-vocabulary report (CPU)
#   2. protocol sweep (offset x horizon), real and timing-only inputs: MODE=test on the 5% checkpoint first,
#      MODE=full on the 150k-iteration checkpoint
# Usage: MODE=test|full nohup bash workbench/run_replication_checks.sh > ~/logs/log_checks_$MODE.txt 2>&1 &
set -u
mkdir -p ~/logs ~/results ~/runs
MODE=${MODE:-test}
BUCKET=${WORKSPACE_BUCKET}/notebooks/replication
cd ~/delphi-aou
if [ "$MODE" = full ]; then
  echo "== backup $(date)"
  gsutil -m -q rsync -r ~/delphi_rep_data $BUCKET/delphi_rep_data
  gsutil -m -q rsync -r ~/delphi_rep_runs $BUCKET/delphi_rep_runs
  gsutil -m -q rsync -r ~/delphi_rep_data_test $BUCKET/delphi_rep_data_test
  echo BACKUP_EXIT $?
  echo "== oov report $(date)"
  RUN_MODE=full python -u -m delphi_aou.replication.oov_report > ~/logs/log_oov.txt 2>&1; echo OOV_EXIT $?
  DATA=~/delphi_rep_data; CKPT=~/delphi_rep_runs/full/ckpt/ckpt.pt; OUT=~/delphi_rep_runs/full/sweep; EXTRA=""
else
  DATA=~/delphi_rep_data_test; CKPT=~/delphi_rep_runs/test/ckpt/ckpt.pt; OUT=~/delphi_rep_runs/test/sweep; EXTRA="--grid offsets"
fi
echo "== sweep real $(date)"
python -u -m delphi_aou.replication.sweep --input_path $DATA --model_ckpt_path $CKPT --out $OUT $EXTRA > $OUT.real.log 2>&1; echo SWEEP_REAL_EXIT $?
echo "== sweep erased $(date)"
python -u -m delphi_aou.replication.sweep --input_path $DATA --model_ckpt_path $CKPT --out $OUT --erase_identity $EXTRA > $OUT.erased.log 2>&1; echo SWEEP_ERASED_EXIT $?
[ "$MODE" = full ] && gsutil -m -q rsync -r ~/delphi_rep_runs $BUCKET/delphi_rep_runs
echo "CHECKS_DONE $MODE $(date)"
