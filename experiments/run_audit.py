"""Paired audit of the official Delphi-2M checkpoint on All of Us (P2 review items R1 to R4).

One batch, four input variants (real, erased, erased without lifestyle, shuffled), several protocol settings.
Writes participant-level points to ~/runs/audit/<setting>/ (never exported) and aggregate JSON to
~/results/audit_<setting>.json (means, intervals, per-disease rho quantiles; no counts).
Usage: RUN_MODE=full python -u experiments/run_audit.py [--settings 0.1:open,0.1:365.25,365.25:open] [--B 200]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from delphi_aou.replication.evaluate import OFFICIAL, EraseIdentity
from delphi_aou.replication.controls import ShuffleIdentity
from delphi_aou.replication import audit

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
ap.add_argument("--ckpt", default=str(Path.home() / "delphi_rep_runs/full/ckpt/ckpt.pt"))
ap.add_argument("--settings", default="0.1:open,0.1:365.25,365.25:open")
ap.add_argument("--B", type=int, default=200)
ap.add_argument("--subset", type=int, default=-1)
ap.add_argument("--suffix", default="", help="appended to output names, e.g. _smoke")
ap.add_argument("--chunk", type=int, default=50, help="diseases per inference chunk (RAM: about 0.2 GB per variant per 10 diseases)")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--erase_token", type=int, default=1, help="token that erased diseases become (1270 for the record-token model)")
a = ap.parse_args()
sys.path.insert(0, str(OFFICIAL))
from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch
ck = torch.load(a.ckpt, map_location=a.device)
m = Delphi(DelphiConfig(**ck["model_args"])); m.load_state_dict(ck["model"]); m.eval().to(a.device)
models = {"real": m, "erased": EraseIdentity(m, a.erase_token).eval(),
          "erased_nolife": audit.EraseIdentityAndLifestyle(m, a.erase_token).eval(),
          "shuffled": ShuffleIdentity(m).eval().to(a.device)}
val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
p2i = get_p2i(val)
n = len(p2i) if a.subset == -1 else a.subset
np.random.seed(1337); torch.manual_seed(1337)
d = get_batch(range(n), val, p2i, select="left", block_size=80, device=a.device, padding="random", no_event_token_rate=5)
labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
res_dir = Path.home() / "results"; res_dir.mkdir(exist_ok=True)
for st in a.settings.split(","):
    off, hor = st.split(":"); off = float(off); hor = None if hor == "open" else float(hor)
    tag = f"off{off:g}_hor{'open' if hor is None else f'{hor:g}'}" + a.suffix
    out = Path.home() / "runs" / "audit" / tag
    if not (out / "meta.json").exists():
        audit.collect(models, d, labels, out, offset=off, horizon=hor, device=a.device, disease_chunk_size=a.chunk)
    meta, rows = audit.load(out)
    names = meta["variants"]
    plain = audit.point_aucs(rows, names)
    matched = audit.point_aucs(rows, names, matched=True)
    wash = audit.point_aucs(rows, names, filt=audit.washout())
    s = {"setting": tag, "n_diseases": int(plain["real"].notna().sum()),
         "mean_auc": {c: float(plain[c].mean()) for c in plain}, "case_matched_mean_auc": {c: float(matched[c].mean()) for c in matched},
         "washout_1y_mean_auc": {c: float(wash[c].mean()) for c in wash},
         "corr_real_vs_erased_per_disease": float(plain[["real", "erased"]].corr().iloc[0, 1])}
    print("POINT", json.dumps(s), flush=True)
    s["bootstrap"] = audit.bootstrap(rows, names, B=a.B)
    s["bootstrap_matched_note"] = "intervals computed for the plain AUC only"
    (res_dir / f"audit_{tag}.json").write_text(json.dumps(s, indent=1))
    print("RESULT", json.dumps(s), flush=True)
print("AUDIT COMPLETE", flush=True)
