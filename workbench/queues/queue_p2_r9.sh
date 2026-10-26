#!/bin/bash
# P2 round 9 experiments (Oct 23, 2026). Logs in ~/logs/log_r9_*.txt, aggregates in ~/results.
#  GPU lane: record-token identity-free model (smoke, full 150k training, paired audit in three settings, corrected conditional)
#  CPU/GPU lane after the running seed-43 corrected job: seed-43 audits for the 1-year horizon and offset settings,
#  their corrected conditional, conditional analyses of the returned people, seed-43 anchored calibration.
set -u
cd ~/delphi-aou
export RUN_MODE=full PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ~/logs ~/results
D=/home/jupyter/delphi_rep_data
R=/home/jupyter/delphi_rep_runs

gpu_lane () {
  python -u -m delphi_aou.replication.train_identity_free config/train_delphi.py --dataset=$D \
    --out_dir=$R/idrec_smoke/ckpt --device=cuda --max_iters=300 --lr_decay_iters=300 --eval_interval=150 \
    --eval_iters=5 --always_save_checkpoint=True --record_token=1270 > ~/logs/log_r9_idrec_smoke.txt 2>&1
  echo "SMOKE_EXIT $? $(date)"
  grep -q "to token 1270" ~/logs/log_r9_idrec_smoke.txt || { echo "SMOKE FAILED"; return 1; }
  python -u -m delphi_aou.replication.train_identity_free config/train_delphi.py --dataset=$D \
    --out_dir=$R/idrec/ckpt --device=cuda --record_token=1270 > ~/logs/log_r9_train_idrec.txt 2>&1
  echo "TRAIN_EXIT $? $(date)"
  python -u experiments/run_audit.py --ckpt $R/idrec/ckpt/ckpt.pt --suffix _idrec --erase_token 1270 --chunk 30 \
    > ~/logs/log_r9_audit_idrec.txt 2>&1
  echo "AUDIT_EXIT $? $(date)"
  python -u experiments/audit_corrected_conditional.py \
    --tags off0.1_horopen_idrec,off0.1_hor365.25_idrec,off365.25_horopen_idrec \
    --out ~/results/audit_corrected_conditional_idrec.json > ~/logs/log_r9_corrected_idrec.txt 2>&1
  echo "CORRECTED_IDREC_EXIT $? $(date)"
}

second_lane () {
  while pgrep -f "audit_corrected_conditional.py --tags off0.1_horopen_s1" > /dev/null; do sleep 120; done
  echo "S1 CORRECTED DONE $(date)"
  python -u experiments/audit_returned_conditional.py > ~/logs/log_r9_returned.txt 2>&1
  echo "RETURNED_EXIT $? $(date)"
  rm -rf ~/runs/audit/off0.1_hor365.25_s1 ~/runs/audit/off0.1_hor365.25_idfree_s1
  python -u experiments/run_audit.py --ckpt $R/full_s1/ckpt/ckpt.pt --suffix _s1 --settings 0.1:365.25,365.25:open \
    --chunk 25 > ~/logs/log_r9_audit_full_s1.txt 2>&1
  echo "AUDIT_FULL_S1_EXIT $? $(date)"
  python -u experiments/run_audit.py --ckpt $R/idfree_s1/ckpt/ckpt.pt --suffix _idfree_s1 --settings 0.1:365.25,365.25:open \
    --chunk 25 > ~/logs/log_r9_audit_idfree_s1.txt 2>&1
  echo "AUDIT_IDFREE_S1_EXIT $? $(date)"
  python -u experiments/audit_corrected_conditional.py \
    --tags off0.1_hor365.25_s1,off0.1_hor365.25_idfree_s1,off365.25_horopen_s1,off365.25_horopen_idfree_s1 \
    --out ~/results/audit_corrected_conditional_s1b.json > ~/logs/log_r9_corrected_s1b.txt 2>&1
  echo "CORRECTED_S1B_EXIT $? $(date)"
  DELPHI_CKPT=$R/full_s1/ckpt/ckpt.pt OUT_SUFFIX=_s1 TAGS= python -u experiments/official_anchored_calibration.py \
    > ~/logs/log_r9_anchored_s1.txt 2>&1
  echo "ANCHORED_S1_EXIT $? $(date)"
}

gpu_lane > ~/logs/log_r9_gpu_lane.txt 2>&1 &
second_lane > ~/logs/log_r9_second_lane.txt 2>&1 &
wait
echo "R9 QUEUE COMPLETE $(date)"
