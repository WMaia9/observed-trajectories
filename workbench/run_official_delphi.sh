#!/usr/bin/env bash
mkdir -p ~/logs ~/results ~/runs
# Exact Delphi-2M replication with the authors' unchanged code (third_party/delphi_official).
# MODE=test: 5% sample, 2000 iterations (pipeline check). MODE=full: config/train_delphi.py as in the paper.
set -eu
MODE=${MODE:-test}
REPO=~/delphi-aou
OFF=$REPO/third_party/delphi_official
if [ "$MODE" = test ]; then DATA=~/delphi_rep_data_test; EXTRA="--max_iters=2000 --lr_decay_iters=2000 --warmup_iters=200"; else DATA=~/delphi_rep_data; EXTRA=""; fi
OUT=~/delphi_rep_runs/$MODE
mkdir -p $OUT
cd $OFF
python -u train.py config/train_delphi.py --dataset=$DATA --out_dir=$OUT/ckpt --device=cuda $EXTRA > $OUT/train.log 2>&1
cd $REPO
python -u -m delphi_aou.replication.evaluate --input_path $DATA --output_path $OUT/eval --model_ckpt_path $OUT/ckpt/ckpt.pt > $OUT/eval.log 2>&1
echo REPLICATION_DONE > $OUT/DONE
