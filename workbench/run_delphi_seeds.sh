#!/usr/bin/env bash
# Train the three Delphi objectives for 3 seeds on one GPU (3 processes in parallel, separate TMPDIR to avoid
# temp file races), then run the calibration analysis. Run inside the All of Us Workbench app:
#   nohup bash workbench/run_delphi_seeds.sh > /dev/null 2>&1 &
set -u
mkdir -p ~/logs ~/results ~/runs
ITERS=${ITERS:-30000}
for s in 0 1 2; do
  for cfg in "delphi delphi" "observed delphi" "observed pp"; do
    set -- $cfg
    mkdir -p ~/runs/tmp_$1_$2_$s
    TMPDIR=~/runs/tmp_$1_$2_$s SEED=$s RUN_MODE=full MODE=$1 OBJECTIVE=$2 ITERS=$ITERS N_LAYER=12 N_EMBD=120 N_HEAD=12 \
      TORCH_THREADS=1 python -u scripts/40_train_delphi.py > ~/logs/log_main_$1_$2_s$s.txt 2>&1 &
    sleep 5
  done
  wait
done
TAGS=$(for m in delphi observed observed_pp; do for s in 0 1 2; do printf "%s_s%s_H5," $m $s; done; done)
TMPDIR=~/runs/tmp_calib TAGS=${TAGS%,} RUN_MODE=full N_BOOT=200 python -u scripts/60_calibration.py > ~/logs/log_calib_main.txt 2>&1
echo ALLDONE > ~/logs/log_chain_main.txt
