"""Two properties of the official evaluation batch that favour calendar signals (P2 reviewer items W2, round 2 item 2).

1. Window and sex-token cut. The official evaluation calls get_batch(select='left', block_size=80, padding='random'):
   it takes each person's first 81 tokens, adds 20 random no-event tokens, drops those after the last real token,
   sorts, and cuts from the LEFT to 81. For a person whose window is full, every surviving no-event token removes the
   earliest token, which is the sex token at age 0; evaluate_auc then selects people by the presence of a sex token,
   so these people leave the evaluation. We count how many people and diagnoses this affects.
2. Controls with a no-event target. Controls are scored before whatever their next target is, including the random
   no-event tokens. We rebuild the same batch (same seeds as run_audit.py), recover each saved point's target token,
   and recompute the paired AUCs without control points whose target is a no-event token.

Aggregates only are written to ~/results/audit_protocol_checks.json.
Usage: RUN_MODE=full python experiments/audit_protocol_checks.py [--data ~/delphi_rep_data]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
ap.add_argument("--tags", default="off0.1_horopen,off0.1_hor365.25,off365.25_horopen")
a = ap.parse_args()
sys.path.insert(0, str(OFFICIAL))
from utils import get_p2i, get_batch

val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
p2i = get_p2i(val)
np.random.seed(1337); torch.manual_seed(1337)
d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device="cpu", padding="random",
              no_event_token_rate=5)
x, ages, y, b = [t.numpy() for t in d]
res = {}

# ---- 1. who is in the evaluation
has_sex = ((x == 2) | (x == 3)).any(1)
seq = np.concatenate([x[:, :1], y], 1)                                 # full window: inputs plus the last target
length = p2i[:, 1]
pid_of_row = np.repeat(np.arange(len(p2i)), length)
dis_raw = (val[:, 2] >= 12) & (val[:, 2] < 1268)                       # raw ids: shifted by +1 inside get_batch
n_dis = np.bincount(pid_of_row, weights=dis_raw, minlength=len(p2i))
full = length >= 81
res["window"] = {
    "persons": int(len(p2i)),
    "share_full_window": round(float(full.mean()), 4),
    "share_without_sex_token": round(float((~has_sex).mean()), 4),
    "share_of_full_window_without_sex_token": round(float((~has_sex[full]).mean()), 4) if full.any() else None,
    "median_diagnoses_kept": float(np.median(n_dis[has_sex])),
    "median_diagnoses_dropped": float(np.median(n_dis[~has_sex])) if (~has_sex).any() else None,
    "share_of_all_first_diagnoses_in_dropped_people": round(float(n_dis[~has_sex].sum() / n_dis.sum()), 4),
    "share_of_first_diagnoses_beyond_window_in_kept_people":
        round(float(np.clip(n_dis[has_sex] - ((seq[has_sex] >= 13) & (seq[has_sex] < 1269)).sum(1), 0, None).sum()
                    / n_dis[has_sex].sum()), 4),
}
print("WINDOW", json.dumps(res["window"]), flush=True)

# ---- 2. controls whose target is a no-event token
for tag in a.tags.split(","):
    meta, rows = audit.load(Path.home() / "runs" / "audit" / tag)
    names = meta["variants"]
    matched_rate, ne_share = [], []
    for r in rows:
        pr, zk = r["person"].astype(np.int64), r["zk"]
        hit = np.isclose(b[pr], zk[:, None], atol=0.5)                    # target position(s) at that age
        found = hit.any(1)
        ne = (hit & (y[pr] == 1)).any(1) & ~(hit & (y[pr] != 1)).any(1)
        r["target_noevent"] = ne
        matched_rate.append(found.mean())
        ne_share.append(ne[~r["case"]].mean() if (~r["case"]).any() else np.nan)
    keep = lambda r: r["case"] | ~r["target_noevent"]
    plain = audit.point_aucs(rows, names)
    excl = audit.point_aucs(rows, names, filt=keep)
    res[tag] = {
        "share_points_target_recovered": round(float(np.mean(matched_rate)), 4),
        "share_controls_with_noevent_target": round(float(np.nanmean(ne_share)), 4),
        "plain": {c: round(float(plain[c].mean()), 4) for c in ["real", "erased", "shuffled", "count_only"] if c in plain},
        "without_noevent_controls": {c: round(float(excl[c].mean()), 4) for c in ["real", "erased", "shuffled", "count_only"] if c in plain},
        "n_diseases_without": int(excl["real"].notna().sum()),
    }
    for k in ("plain", "without_noevent_controls"):
        v = res[tag][k]; v["rho_erased"] = round((v["erased"] - 0.5) / (v["real"] - 0.5), 4)
    print("RESULT", tag, json.dumps(res[tag]), flush=True)
(Path.home() / "results" / "audit_protocol_checks.json").write_text(json.dumps(res, indent=1))
print("PROTOCOL CHECKS COMPLETE", flush=True)
