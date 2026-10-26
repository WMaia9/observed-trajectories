#!/bin/bash
# P2 round 9d: the final (iteration 150,000) checkpoints of Delphi-2M (AoU) and the identity-free model, seed 42, so no
# checkpoint is selected on the people the AUC is computed on (the official script keeps the lowest validation loss:
# iteration 145,500 for the full model and 89,750 for the identity-free model).
set -u
cd ~/delphi-aou
R=/home/jupyter/delphi_rep_runs
while ! grep -q "SELECTION_EXIT" ~/logs/log_r9b_lane_p.txt 2>/dev/null; do sleep 300; done
python -u experiments/run_audit.py --ckpt $R/full/ckpt/ckpt_150000.pt --suffix _final --settings 0.1:open --chunk 25 \
  > ~/logs/log_r9d_audit_full_final.txt 2>&1
echo "AUDIT_FULL_FINAL_EXIT $? $(date)"
python -u experiments/run_audit.py --ckpt $R/idfree/ckpt/ckpt_150000.pt --suffix _idfree_final --settings 0.1:open --chunk 25 \
  > ~/logs/log_r9d_audit_idfree_final.txt 2>&1
echo "AUDIT_IDFREE_FINAL_EXIT $? $(date)"
python -u experiments/audit_paired_gap.py --main_suffix _final --free _idfree_final --Bc 50 > ~/logs/log_r9d_paired_final.txt 2>&1
echo "PAIRED_FINAL_EXIT $? $(date)"
