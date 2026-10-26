#!/bin/bash
# P2 round 9c: anchored calibration of Delphi-2M (AoU) on anchors whose history fits in the official training window.
set -u
cd ~/delphi-aou
while ! grep -q "ANCHORED_S1_EXIT" ~/logs/log_r9_second_lane.txt 2>/dev/null; do sleep 300; done
MAX_HIST_TOKENS=96 OUT_SUFFIX=_hist96 python -u experiments/official_anchored_calibration.py > ~/logs/log_r9c_hist96.txt 2>&1
echo "HIST96_EXIT $? $(date)"
