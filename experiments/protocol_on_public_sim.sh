#!/usr/bin/env bash
# P2 (review weakness 1 and 5): run the two official-protocol checks on the authors' own public simulated UK Biobank
# data (gerstung-lab/Delphi, data/ukb_simulated_data, commit fb72166), outside All of Us and without controlled access.
# The protocol properties (no-event targets among controls, window cut of the sex token) come from the evaluation
# code, not from the model, so a short CPU training run is enough; AUC values here are illustrative only.
# Outputs: ~/runs/audit/off0.1_horopen_ukbsim, ~/results/audit_off0.1_horopen_ukbsim.json and
# ~/results/audit_protocol_checks_ukbsim.json (the All of Us files are left untouched).
# Usage: SIM=/path/to/ukb_simulated_data OUT=/tmp/ukbsim ITERS=500 bash experiments/protocol_on_public_sim.sh
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
SIM=${SIM:?path to data/ukb_simulated_data of the official repository}
OUT=${OUT:-/tmp/ukbsim}; ITERS=${ITERS:-500}
mkdir -p "$OUT"
cd "$REPO/third_party/delphi_official"
python -u train.py config/train_delphi.py --dataset="$SIM" --out_dir="$OUT/ckpt" --device=cpu --max_iters=$ITERS \
  --lr_decay_iters=$ITERS --warmup_iters=50 --eval_interval=$ITERS --eval_iters=5 --always_save_checkpoint=True \
  > "$OUT/train.log" 2>&1
cd "$REPO"
python -u experiments/run_audit.py --data "$SIM" --ckpt "$OUT/ckpt/ckpt.pt" --settings 0.1:open \
  --suffix _ukbsim --B 20 --device cpu > "$OUT/audit.log" 2>&1
# audit_protocol_checks writes ~/results/audit_protocol_checks.json: keep the All of Us file, save this one beside it
R=~/results/audit_protocol_checks.json; [ -f "$R" ] && cp "$R" "$R.aou_backup"
python -u experiments/audit_protocol_checks.py --data "$SIM" --tags off0.1_horopen_ukbsim > "$OUT/checks.log" 2>&1
mv "$R" ~/results/audit_protocol_checks_ukbsim.json; [ -f "$R.aou_backup" ] && mv "$R.aou_backup" "$R"
grep -E "^(WINDOW|RESULT)" "$OUT/checks.log"
