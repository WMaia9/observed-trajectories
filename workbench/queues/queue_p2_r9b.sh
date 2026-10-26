#!/bin/bash
# P2 round 9b (Oct 23, 2026), alongside queue_p2_r9.sh. Logs ~/logs/log_r9b_*.txt.
#  lane S: semi-synthetic identity-free and record-token models (synthetic data only; small GPU models)
#  lane P: after queue_p2_r9 second lane: paired gap and time bins (seed 42, seed 43), selection-seed variance
#  lane R: after the record-token audit: paired gap for the record-token model
set -u
cd ~/delphi-aou
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S="python -u experiments/semisynthetic_idfree.py --n 20000 --iters 3000 --device cuda --workers 2 --layers 6 --embd 128 --out $HOME/results/semisyn_idfree.json"
lane_s () {
  python -u experiments/semisynthetic_idfree.py --n 1500 --iters 30 --device cuda --workers 1 --layers 2 --embd 32 \
    --grid 1:0 --out $HOME/results/semisyn_idfree_smoke.json > ~/logs/log_r9b_semi_smoke.txt 2>&1 || { echo SEMI_SMOKE_FAIL; return 1; }
  $S --mode frailty --grid 1:0,0:2,1:2 --seeds 0,1,2,3,4 > ~/logs/log_r9b_semi.txt 2>&1
  echo "SEMI_EXIT $? $(date)"
}
lane_p () {
  while ! grep -q "ANCHORED_S1_EXIT" ~/logs/log_r9_second_lane.txt 2>/dev/null; do sleep 300; done
  python -u experiments/audit_paired_gap.py > ~/logs/log_r9b_paired.txt 2>&1
  echo "PAIRED_EXIT $? $(date)"
  python -u experiments/audit_paired_gap.py --main_suffix _s1 --free _idfree_s1 > ~/logs/log_r9b_paired_s1.txt 2>&1
  echo "PAIRED_S1_EXIT $? $(date)"
  python -u experiments/audit_selection_seeds.py > ~/logs/log_r9b_selection.txt 2>&1
  echo "SELECTION_EXIT $? $(date)"
}
lane_r () {
  while ! grep -q "CORRECTED_IDREC_EXIT" ~/logs/log_r9_gpu_lane.txt 2>/dev/null; do sleep 300; done
  while ! grep -q "PAIRED_S1_EXIT" ~/logs/log_r9b_lane_p.txt 2>/dev/null; do sleep 300; done
  python -u experiments/audit_paired_gap.py --free _idrec > ~/logs/log_r9b_paired_idrec.txt 2>&1
  echo "PAIRED_IDREC_EXIT $? $(date)"
}
lane_s > ~/logs/log_r9b_lane_s.txt 2>&1 &
lane_p > ~/logs/log_r9b_lane_p.txt 2>&1 &
lane_r > ~/logs/log_r9b_lane_r.txt 2>&1 &
wait
echo "R9B QUEUE COMPLETE $(date)"
