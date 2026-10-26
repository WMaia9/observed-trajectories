"""Fidelity of the anchored scoring path for the official checkpoint (P2, defends the calibration appendix).

The anchored analysis reads the official Delphi-2M checkpoint through our own input construction
(official_anchored_calibration.query_sequence / score_official). Here the same construction scores the points of the
official protocol itself (the saved paired-audit points, offset 0.1 day, open horizon), so the result can be compared
with the official scores of the same points: if the per-disease AUCs agree, our input construction reads the model
as its own evaluation does, and the low anchored AUC is a property of the anchored task, not of our reader.

For each point (person, prediction age z, disease k) we build the input from the person's val.bin rows with age <= z,
regular no-event padding, and read (a) the last real token and (b) a query no-event token at z. AUC is aggregated
exactly as in the paper (audit.point_aucs). A random subset of diseases keeps the run short.
Only aggregates are printed. Usage: python -u experiments/official_fidelity_check.py [--n_dis 100] [--device cuda]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from official_anchored_calibration import score_official
from delphi_aou.replication import audit
from delphi_aou.replication.evaluate import OFFICIAL


def batch_person_ids(val):
    """Person id of each batch row: the official batch takes patients in get_p2i order (by first row of val.bin)."""
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i
    p2i = get_p2i(val)
    return val[p2i[:, 0], 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--ckpt", default=str(Path.home() / "delphi_rep_runs/full/ckpt/ckpt.pt"))
    ap.add_argument("--points", default=str(Path.home() / "runs/audit/off0.1_horopen"))
    ap.add_argument("--n_dis", type=int, default=100)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    pid_of_row = batch_person_ids(val)
    meta, rows = audit.load(a.points)
    ks = sorted({r["k"] for r in rows})
    rng = np.random.default_rng(0)
    pick = set(rng.choice(ks, min(a.n_dis, len(ks)), replace=False).tolist())
    rows = [r for r in rows if r["k"] in pick]
    ireal = meta["variants"].index("real")
    # unique (person, z) pairs across the selected diseases, scored once for all selected diseases
    P = np.concatenate([pid_of_row[r["person"]] for r in rows]); Z = np.concatenate([r["z"].astype(np.float64) for r in rows])
    pairs, inv = np.unique(np.stack([P, Z], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    cols = np.array(sorted(pick), np.int64)
    print(f"diseases {len(cols)}; points {len(P):,}; unique (person, age) inputs {len(pairs):,}", flush=True)
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    ck = torch.load(a.ckpt, map_location=a.device)
    m = Delphi(DelphiConfig(**ck["model_args"])); m.load_state_dict(ck["model"]); m.eval().to(a.device)
    persons = pairs[:, 0].astype(np.int64); z = pairs[:, 1]
    sq, sl = score_official(m, val, persons, z, z, cols, 5, a.device, batch=256, raw=True)   # logits: no saturation
    col_of = {k: i for i, k in enumerate(cols)}
    off = 0
    for r in rows:
        n = len(r["case"]); idx = inv[off:off + n]; off += n
        j = col_of[r["k"]]
        r["scores"] = np.stack([r["scores"][:, ireal].astype(np.float32), sl[idx, j], sq[idx, j]], 1)
    names = ["official_scores", "ours_last_token", "ours_query"]
    df = audit.point_aucs(rows, names, extra_scores=False)
    res = {"n_diseases": int(df["official_scores"].notna().sum()),
           "mean_auc": {c: float(df[c].mean()) for c in names},
           "per_disease_corr_with_official": {c: float(df[["official_scores", c]].corr().iloc[0, 1]) for c in names[1:]},
           "max_abs_diff_per_disease": {c: float((df[c] - df["official_scores"]).abs().max()) for c in names[1:]}}
    print("FIDELITY_CHECK", json.dumps(res), flush=True)
    out = Path.home() / "results" / "official_fidelity_check.json"
    out.parent.mkdir(exist_ok=True); out.write_text(json.dumps(res, indent=1))
    print("FIDELITY CHECK COMPLETE", flush=True)


if __name__ == "__main__":
    main()
