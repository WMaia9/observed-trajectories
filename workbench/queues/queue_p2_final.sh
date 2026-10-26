#!/bin/bash
# P2 final experiments (Oct 18, 2026). Three jobs, logs in ~/logs/log_p2final_*.txt, aggregates in ~/results.
#  E1 conditional analyses without the no-event-target controls (CPU, saved points)
#  E2 AUC with the people the official window cut drops (keep-sex batch, GPU inference)
#  E3 second training seed (seed 43) of the identity-free model and of the main model, then their paired audits
set -u
cd ~/delphi-aou
export RUN_MODE=full PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p ~/logs ~/results
D=/home/jupyter/delphi_rep_data
R=/home/jupyter/delphi_rep_runs

seed_job () {   # $1 = idfree | full
  if [ "$1" = idfree ]; then
    python -u -m delphi_aou.replication.train_identity_free config/train_delphi.py --dataset=$D \
      --out_dir=$R/idfree_s1/ckpt --device=cuda --seed=43 > ~/logs/log_p2final_train_idfree_s1.txt 2>&1
    echo "TRAIN_EXIT $? idfree $(date)"
    python -u experiments/run_audit.py --ckpt $R/idfree_s1/ckpt/ckpt.pt --suffix _idfree_s1 \
      > ~/logs/log_p2final_audit_idfree_s1.txt 2>&1
    echo "AUDIT_EXIT $? idfree $(date)"
  else
    (cd third_party/delphi_official && python -u train.py config/train_delphi.py --dataset=$D \
      --out_dir=$R/full_s1/ckpt --device=cuda --seed=43) > ~/logs/log_p2final_train_full_s1.txt 2>&1
    echo "TRAIN_EXIT $? full $(date)"
    python -u experiments/run_audit.py --ckpt $R/full_s1/ckpt/ckpt.pt --suffix _s1 \
      > ~/logs/log_p2final_audit_full_s1.txt 2>&1
    echo "AUDIT_EXIT $? full $(date)"
  fi
}

seed_job idfree > ~/logs/log_p2final_queue_idfree.txt 2>&1 &
seed_job full > ~/logs/log_p2final_queue_full.txt 2>&1 &

python -u experiments/audit_keep_sex.py > ~/logs/log_p2final_keepsex.txt 2>&1
echo "KEEPSEX_EXIT $? $(date)"
python -u experiments/audit_corrected_conditional.py > ~/logs/log_p2final_corrected.txt 2>&1
echo "CORRECTED_EXIT $? $(date)"
wait
python -u experiments/audit_corrected_conditional.py \
  --tags off0.1_horopen_s1,off0.1_horopen_idfree_s1,off0.1_hor365.25_s1,off0.1_hor365.25_idfree_s1,off365.25_horopen_s1,off365.25_horopen_idfree_s1 \
  --out ~/results/audit_corrected_conditional_s1.json > ~/logs/log_p2final_corrected_s1.txt 2>&1
echo "CORRECTED_S1_EXIT $? $(date)"
echo "P2 FINAL QUEUE COMPLETE $(date)"
