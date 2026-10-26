#!/usr/bin/env bash
# Oct 14 queue. MODE=test: input controls on the 5% replication checkpoint, then visit table and Tier A
# on the 5% sample (500 iterations). MODE=full: input controls (shuffled identity, age-sex rates) on the
# full replication checkpoint. Tier A full is launched separately after the test is checked.
# Usage: MODE=test|full nohup bash workbench/run_controls_tierA.sh > ~/logs/log_ctA_$MODE.txt 2>&1 &
set -u
mkdir -p ~/logs ~/results ~/runs
MODE=${MODE:-test}
cd ~/delphi-aou
if [ "$MODE" = test ]; then
  D=~/delphi_rep_data_test; C=~/delphi_rep_runs/test/ckpt/ckpt.pt; O=~/delphi_rep_runs/test/sweep; S="0.1:open,365.25:open"
else
  D=~/delphi_rep_data; C=~/delphi_rep_runs/full/ckpt/ckpt.pt; O=~/delphi_rep_runs/full/sweep; S="0.1:open,30:open,365.25:open,0.1:365.25"
fi
for ctl in shuffled agesex; do
  echo "== control $ctl $(date)"
  python -u -m delphi_aou.replication.sweep --input_path $D --model_ckpt_path $C --out $O --control $ctl \
    --settings $S --workers 4 --batch_size 512 > $O.$ctl.log 2>&1; echo CONTROL_${ctl}_EXIT $?
done
if [ "$MODE" = test ]; then
  echo "== visits test $(date)"
  RUN_MODE=test python -u scripts/50_build_visits.py > ~/logs/log_visits_test.txt 2>&1; echo VISITS_EXIT $?
  echo "== tier A test $(date)"
  RUN_MODE=test ITERS=500 SEED=0 python -u scripts/51_train_visit_model.py > ~/logs/log_tierA_test.txt 2>&1; echo TIERA_EXIT $?
fi
gsutil -m -q rsync -r ~/delphi_rep_runs ${WORKSPACE_BUCKET}/notebooks/replication/delphi_rep_runs
echo "CTA_DONE $MODE $(date)"
