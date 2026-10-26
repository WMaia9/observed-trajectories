cd $HOME/delphi-aou
O=$HOME/results/semisyn_multiseed.json
S="python -u experiments/semisynthetic_matched.py --n 20000 --iters 3000 --device cuda --workers 2 --out $O"
echo ":: smoke $(date)"
python -u experiments/semisynthetic_matched.py --n 1500 --iters 30 --device cuda --workers 1 --mode detect --grid 2:0 --out $HOME/results/semisyn_smoke.json || { echo SMOKE_FAIL; exit 1; }
echo ":: A frailty 4x64 $(date)"; $S --mode frailty --grid 1:0,2:0,1:1,1:2,0:2 --seeds 0,1,2,3,4
echo ":: A detect 4x64 $(date)"; $S --mode detect --grid 2:0,0:2,2:2 --seeds 0,1,2,3,4
for vp in 0.35:0.2 1.4:0.2 0.7:0.1 0.7:0.4; do
  echo ":: C detect sweep $vp $(date)"; $S --mode detect --grid 2:0,2:2 --seeds 0,1,2 --visit_rate ${vp%:*} --p_detect ${vp#*:}
done
echo ":: B frailty 6x128 $(date)"; $S --mode frailty --grid 1:0,2:0,1:1,1:2,0:2 --seeds 0,1,2,3,4 --layers 6 --embd 128
echo ":: B detect 6x128 $(date)"; $S --mode detect --grid 2:0,0:2,2:2 --seeds 0,1,2,3,4 --layers 6 --embd 128
echo "P2_SEMISYN_DONE $(date)"
