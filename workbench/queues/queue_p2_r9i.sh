#!/bin/bash
# P2 r9i (Oct 24 2026, 19:00 PDT): SERIAL rerun of the CPU-heavy jobs that were OOM killed when lanes overlapped.
# Waits for the reference models (r9h) to finish; one heavy job at a time. semisyn (r9g, GPU) keeps running.
set -u
cd ~/delphi-aou
export RUN_MODE=full PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=/home/jupyter/delphi_rep_runs
while pgrep -f audit_reference_models.py > /dev/null; do sleep 120; done
echo "REFERENCE DONE $(date)"
run () { name=$1; shift; "$@" > ~/logs/log_r9i_$name.txt 2>&1; echo "${name^^}_EXIT $? $(date)"; }
run paired python -u experiments/audit_paired_gap.py
run paired_s1 python -u experiments/audit_paired_gap.py --main_suffix _s1 --free _idfree_s1
run paired_idrec python -u experiments/audit_paired_gap.py --free _idrec
run corrected_idrec python -u experiments/audit_corrected_conditional.py --tags off0.1_horopen_idrec,off0.1_hor365.25_idrec,off365.25_horopen_idrec --out ~/results/audit_corrected_conditional_idrec.json
run corrected_s1b python -u experiments/audit_corrected_conditional.py --tags off0.1_hor365.25_s1,off0.1_hor365.25_idfree_s1,off365.25_horopen_s1,off365.25_horopen_idfree_s1 --out ~/results/audit_corrected_conditional_s1b.json
run hist96 env MAX_HIST_TOKENS=96 OUT_SUFFIX=_hist96 python -u experiments/official_anchored_calibration.py
run selection python -u experiments/audit_selection_seeds.py
while pgrep -f semisynthetic_idfree.py > /dev/null; do sleep 120; done
run audit_final python -u experiments/run_audit.py --ckpt $R/full/ckpt/ckpt_150000.pt --suffix _final --settings 0.1:open --chunk 25
run audit_idfree_final python -u experiments/run_audit.py --ckpt $R/idfree/ckpt/ckpt_150000.pt --suffix _idfree_final --settings 0.1:open --chunk 25
run paired_final python -u experiments/audit_paired_gap.py --main_suffix _final --free _idfree_final --Bc 50
echo "R9I QUEUE COMPLETE $(date)"
