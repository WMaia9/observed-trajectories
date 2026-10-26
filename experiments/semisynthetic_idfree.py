"""Does the real minus identity-free gap track known disease content? (P2, reviewer round 9, item 1)

Semi-synthetic worlds of semisynthetic_rho.py (activity effect tau, parent-disease content gamma). On each world and
seed we train three models with the official code: the full model, an identity-free model whose erased records share
the no-event token (as on All of Us), and an identity-free model whose erased records use a dedicated record token.
All are scored on one batch and one selection (audit.collect), with the erased input of the full model and the
diagnosis count. Expected if the gap measures content: matched real minus identity-free near 0 without content
(gamma = 0) and positive with learned content; the shared-token artifact shows as a gap at gamma = 0 that the record
token removes. Synthetic data only. Usage:
  python experiments/semisynthetic_idfree.py --grid "1:0,0:2,1:2" --seeds 0,1,2,3,4 --layers 6 --embd 128 --out ...
"""
import argparse, json, sys, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semisynthetic_rho import train_world
from delphi_aou.replication import audit
from delphi_aou.replication.evaluate import OFFICIAL, EraseIdentity

NAMES = ["real", "erased", "idfree", "idrec"]


def evaluate(models, val, device, out_dir, workers):
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    p2i = get_p2i(val)
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device=device, padding="random",
                  no_event_token_rate=5)
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    audit.collect(models, d, labels, out_dir, offset=0.1, horizon=None, device=device, workers=workers,
                  disease_chunk_size=50, batch_size=256)
    meta, rows = audit.load(out_dir)
    res = {}
    for tag, kw in [("plain", {}), ("matched", {"matched": True}), ("washout", {"filt": audit.washout()})]:
        df = audit.point_aucs(rows, meta["variants"], **kw)
        r = {c: round(float(df[c].mean()), 4) for c in NAMES + ["count_only"]}
        for c in ("idfree", "idrec"):
            r[f"gap_real_minus_{c}"] = round(r["real"] - r[c], 4)
        r["n_diseases"] = int(df["real"].notna().sum())
        res[tag] = r
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="1:0,0:2,1:2")
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--mode", default="frailty", choices=["frailty", "detect"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--layers", type=int, default=6); ap.add_argument("--embd", type=int, default=128)
    ap.add_argument("--out", default="results/semisynthetic_idfree.json")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="semiidf_"))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(out.read_text()) if out.exists() else []
    done = {(r["mode"], r["tau"], r["gamma"], r["seed"], r["layers"], r["embd"], r["iters"]) for r in allr}
    for seed in (int(x) for x in a.seeds.split(",")):
        for g in a.grid.split(","):
            tau, gamma = (float(x) for x in g.split(":"))
            cfg = dict(mode=a.mode, tau=tau, gamma=gamma, seed=seed, layers=a.layers, embd=a.embd, iters=a.iters, n=a.n)
            if (a.mode, tau, gamma, seed, a.layers, a.embd, a.iters) in done:
                continue
            ms = {}
            for tr in ("official", "idfree", "idrec"):
                ms[tr], val, nrec = train_world(tau, gamma, a.n, a.iters, a.device, work, seed=seed, mode=a.mode,
                                                layers=a.layers, embd=a.embd, trainer=tr)
            models = {"real": ms["official"], "erased": EraseIdentity(ms["official"]).eval(),
                      "idfree": EraseIdentity(ms["idfree"]).eval(), "idrec": EraseIdentity(ms["idrec"], 1270).eval()}
            res = evaluate(models, val, a.device, work / f"audit_{a.mode}_{tau}_{gamma}_{seed}", a.workers)
            res.update(cfg, records=int(nrec))
            allr.append(res); out.write_text(json.dumps(allr, indent=1))
            print("RESULT", json.dumps(res), flush=True)
    print("SEMISYNTHETIC IDFREE COMPLETE", flush=True)
