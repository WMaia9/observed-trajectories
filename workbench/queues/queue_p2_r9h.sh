#!/bin/bash
# P2 round 9h: cross-fitted bag-of-codes and calendar reference scores on the paired-audit points (CPU).
cd ~/delphi-aou
while pgrep -f run_audit.py > /dev/null; do sleep 300; done
python -u experiments/audit_reference_models.py > ~/logs/log_r9h_reference.txt 2>&1
echo "REFERENCE_EXIT $? $(date)"
