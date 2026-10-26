"""Does the matched analysis separate the observation process from disease content? (P2, R10 follow-up)

Same semi-synthetic worlds as semisynthetic_rho.py (activity effect tau, parent-disease effect gamma), the official code
trained on each, and the paired audit of the paper (audit.collect, same selection as the official evaluation) with the
real and timing-only inputs and the model-free record count. Reports, per world, plain, matched (record-count quintile
x recency band) and 1-year wash-out AUCs. Expected if matching works: with tau = 0 (no observation-process effect) the
matched timing-only and count AUCs are near 0.5; with gamma = 0 (no content) real and timing-only agree in matched cells.

Usage: python experiments/semisynthetic_matched.py --grid "1:0,2:0,1:1,1:2,0.5:2,0:2" --n 20000 --iters 3000
Writes results/semisynthetic_matched.json (synthetic data only, nothing from All of Us).
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semisynthetic_rho import train_world
from delphi_aou.replication import audit
from delphi_aou.replication.evaluate import OFFICIAL, EraseIdentity


def evaluate(m, val, device, out_dir, workers):
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    p2i = get_p2i(val)
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device=device, padding="random",
                  no_event_token_rate=5)
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    models = {"real": m, "erased": EraseIdentity(m).eval()}
    audit.collect(models, d, labels, out_dir, offset=0.1, horizon=None, device=device, workers=workers,
                  disease_chunk_size=50, batch_size=256)
    meta, rows = audit.load(out_dir)
    res = {}
    for tag, kw in [("plain", {}), ("matched", {"matched": True}), ("washout", {"filt": audit.washout()})]:
        df = audit.point_aucs(rows, meta["variants"], **kw)
        res[tag] = {c: round(float(df[c].mean()), 4) for c in ["real", "erased", "count_only"]}
        r = res[tag]
        res[tag]["rho"] = round((r["erased"] - 0.5) / (r["real"] - 0.5), 3) if r["real"] - 0.5 > 0.02 else None
        res[tag]["n_diseases"] = int(df["real"].notna().sum())
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="1:0,2:0,1:1,1:2,0.5:2,0:2")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="results/semisynthetic_matched.json")
    ap.add_argument("--mode", default="frailty", choices=["frailty", "detect"],
                    help="frailty: activity scales every recording hazard; detect: onset independent of activity, "
                         "recording only at visits whose rate scales with activity")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--embd", type=int, default=64)
    ap.add_argument("--visit_rate", type=float, default=0.7, help="detect mode: routine visits per year at A = 1")
    ap.add_argument("--p_detect", type=float, default=0.2, help="detect mode: detection probability per visit")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="semimatch_"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(out.read_text()) if out.exists() else []
    def key(r):
        return (r.get("mode", "frailty"), r["tau"], r["gamma"], r["iters"], r.get("seed", 0), r.get("layers", 4),
                r.get("embd", 64), r.get("visit_rate", 0.7), r.get("p_detect", 0.2))
    done = {key(r) for r in allr}
    for seed in (int(x) for x in a.seeds.split(",")):
        for g in a.grid.split(","):
            tau, gamma = (float(x) for x in g.split(":"))
            cfg = dict(mode=a.mode, seed=seed, tau=tau, gamma=gamma, n=a.n, iters=a.iters, layers=a.layers, embd=a.embd,
                       visit_rate=a.visit_rate, p_detect=a.p_detect)
            if key(cfg) in done:
                continue
            m, val, nrec = train_world(tau, gamma, a.n, a.iters, a.device, work, seed=seed, mode=a.mode,
                                       layers=a.layers, embd=a.embd, visit_rate=a.visit_rate, p_detect=a.p_detect)
            res = evaluate(m, val, a.device, work / f"audit_{a.mode}_{tau}_{gamma}_{seed}_{a.layers}x{a.embd}", a.workers)
            res.update(cfg, records=int(nrec))
            allr.append(res); out.write_text(json.dumps(allr, indent=1))
            print("RESULT", json.dumps(res), flush=True)
    print("SEMISYNTHETIC MATCHED COMPLETE", flush=True)
