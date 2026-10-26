#!/usr/bin/env bash
# One command that reproduces every result of the audit paper that does not need All of Us data:
#   1. unit tests of the converter, controls, matching, calibration and scoring code;
#   2. the seven-step protocol on the authors' own simulated UK Biobank data (official repository, commit fb72166);
#   3. semi-synthetic worlds with known structure (QUICK=1: two worlds, one seed; QUICK=0: the full multi-seed grid);
#   4. a comparison with reproduce/expected_public.json (PASS or FAIL per item).
# CPU only is fine (QUICK=1 takes about 1 to 2 hours on 4 cores; a GPU makes step 3 much faster: DEVICE=cuda).
# Usage: bash reproduce/run_public.sh            [env: WORK=reproduce/out QUICK=1 DEVICE=cpu SIM=<existing data dir>]
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=${WORK:-$REPO/reproduce/out}; QUICK=${QUICK:-1}; DEVICE=${DEVICE:-cpu}
mkdir -p "$WORK/home/results"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

echo ":: 1/4 unit tests $(date)"
python -m pytest -q -m "not slow" tests --deselect tests/test_sampling_risk.py::test_main

echo ":: 2/4 protocol on the authors' simulated data $(date)"
if [ -z "${SIM:-}" ]; then
  [ -d "$WORK/delphi_public" ] || git clone -q https://github.com/gerstung-lab/Delphi "$WORK/delphi_public"
  git -C "$WORK/delphi_public" checkout -q fb72166
  SIM="$WORK/delphi_public/data/ukb_simulated_data"
fi
# run_audit.py and audit_protocol_checks.py write under $HOME: point HOME at the work folder so nothing else is touched
HOME="$WORK/home" SIM="$SIM" OUT="$WORK/ukbsim" ITERS=500 bash experiments/protocol_on_public_sim.sh

echo ":: 3/4 semi-synthetic worlds $(date)"
if [ "$QUICK" = "1" ]; then
  SEMI="$WORK/semisyn_quick.json"
  python -u experiments/semisynthetic_matched.py --mode frailty --grid 1:0 --n 10000 --iters 1500 --device "$DEVICE" --out "$SEMI"
  python -u experiments/semisynthetic_matched.py --mode detect --grid 2:0 --n 10000 --iters 1500 --device "$DEVICE" --out "$SEMI"
else
  SEMI="$WORK/semisyn_multiseed.json"
  sed -e "s#^cd .*#cd $REPO#" -e "s#^O=.*#O=$SEMI#" -e "s#--device cuda#--device $DEVICE#g" \
      -e "s#\$HOME/results/semisyn_smoke.json#$WORK/semisyn_smoke.json#" workbench/queues/queue_p2_semisyn.sh > "$WORK/queue.sh"
  bash "$WORK/queue.sh"
fi
python experiments/semisynthetic_summary.py "$SEMI" > "$WORK/semisyn_summary.txt"

echo ":: 4/4 comparison with the published values $(date)"
python reproduce/check_expected.py --results "$WORK/home/results" --semisyn "$SEMI" | tee "$WORK/check.txt"
