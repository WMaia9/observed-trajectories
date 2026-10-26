"""Variance of the official mean AUC due to the random choice of points (P2 round 9, reviewer item 2b).

The official evaluation draws one random point per person, age bracket and disease from the unseeded NumPy generator;
our paired audit fixes the draw (key 0). Here the scores of the main checkpoint (real and erased input) and of the
identity-free model, and three further shuffled inputs (permutation seeds 1 to 3; the paper's uses seed 0), are computed
once on the run_audit batch, the selection is drawn with keys 1..K, and the mean AUC,
matched AUC and rho are reported per key, with and without padding controls. Aggregates only, to
~/results/audit_selection_seeds.json. Usage: python -u experiments/audit_selection_seeds.py [--keys 1,2,3,4,5]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from delphi_aou.replication.evaluate import OFFICIAL, EraseIdentity
from delphi_aou.replication import audit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_matching_checks import matched, summary
from audit_corrected_conditional import target_noevent, subset


def load_model(path, device):
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    ck = torch.load(path, map_location=device)
    m = Delphi(DelphiConfig(**ck["model_args"])); m.load_state_dict(ck["model"]); return m.eval().to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--ckpt", default=str(Path.home() / "delphi_rep_runs/full/ckpt/ckpt.pt"))
    ap.add_argument("--ckpt_idfree", default=str(Path.home() / "delphi_rep_runs/idfree/ckpt/ckpt.pt"))
    ap.add_argument("--keys", default="1,2,3,4,5")
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    keys = [int(k) for k in a.keys.split(",")]
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device=a.device, padding="random",
                  no_event_token_rate=5)
    out_dir = Path.home() / "runs" / "audit" / "off0.1_horopen_sel"
    if not all((out_dir / f"key{k}" / "meta.json").exists() for k in keys):
        m, mf = load_model(a.ckpt, a.device), load_model(a.ckpt_idfree, a.device)
        from delphi_aou.replication.controls import ShuffleIdentity
        models = {"real": m, "erased": EraseIdentity(m).eval(), "idfree": EraseIdentity(mf).eval(),
                  **{f"shuffled_p{q}": ShuffleIdentity(m, seed=q).eval().to(a.device) for q in (1, 2, 3)}}
        labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
        audit.collect(models, d, labels, out_dir, offset=0.1, horizon=None, device=a.device,
                      disease_chunk_size=a.chunk, keys=keys)
        del m, mf, models
    y, b = d[2].cpu().numpy(), d[3].cpu().numpy()
    res = {}
    for k in keys:
        meta, rows = audit.load(out_dir / f"key{k}")
        names = meta["variants"]
        target_noevent(rows, y, b)
        r = {}
        for label, rs in (("all", rows), ("padding_removed", subset(rows, lambda q: q["case"] | ~q["target_noevent"]))):
            df = audit.point_aucs(rs, names)
            r[label] = {"plain": summary(df[[c for c in df if c != "recency_only"]]),
                        "matched": summary(matched(rs, names, False)[0])}
        res[f"key{k}"] = r
        print("RESULT", k, json.dumps(r), flush=True)
        (Path.home() / "results" / "audit_selection_seeds.json").write_text(json.dumps(res, indent=1))
    sm = {}
    for label in ("all", "padding_removed"):
        for an in ("plain", "matched"):
            for c in ("real", "erased", "idfree", "rho_erased", "rho_idfree", "shuffled_p1", "shuffled_p2", "shuffled_p3"):
                v = np.array([res[f"key{k}"][label][an][c] for k in keys])
                sm[f"{label}_{an}_{c}"] = {"mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4),
                                           "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
    res["summary_over_keys"] = sm
    (Path.home() / "results" / "audit_selection_seeds.json").write_text(json.dumps(res, indent=1))
    print("SELECTION SEEDS COMPLETE", flush=True)
