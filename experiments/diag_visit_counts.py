"""Does Tier A simulate the right number of future visits for each access level?
Compares, by quintile of visits before the anchor, the observed number of visit days in (anchor, anchor + H]
(persons followed through H) with the mean number of simulated visits. Aggregates only.
Usage: SEED=1 N_ANCHORS=6000 python experiments/diag_visit_counts.py"""
import json, os, subprocess, tempfile
from pathlib import Path
import numpy as np, pandas as pd, torch
from delphi_aou import config as common
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3
from delphi_aou.train import visit as s5
from delphi_aou.models.visit_pp import VisitPP, VisitConfig

CFG = common.load_config(); ev = CFG["evaluation"]; DAY = 365.25
dev = "cuda" if torch.cuda.is_available() else "cpu"
seed = int(os.environ.get("SEED", "1")); n = int(os.environ.get("N_ANCHORS", "6000"))
tmp = Path(tempfile.gettempdir())
coh = s2._read_local_from_cache("s2_cohort"); fo = s2._read_local_from_cache("s2_fo_events"); vis = s2._read_local_from_cache("s5_visits")
subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/s5_model_visitA_s{seed}.pt", str(tmp)], check=True)
data = s5.VisitData(coh, fo, vis)
model = VisitPP(VisitConfig(data.K, n_layer=4, n_head=8, n_embd=128)).to(dev)
model.load_state_dict(torch.load(tmp / f"s5_model_visitA_s{seed}.pt", map_location=dev)); model.eval()
an = s3.build_anchors(coh, CFG); te = an[an["split"] == "test"].reset_index(drop=True)
sub = te.iloc[np.sort(np.random.default_rng(1).choice(len(te), min(n, len(te)), replace=False))].reset_index(drop=True)
names = ["DEATH"]
out = s5.sample_risk(model, data, sub, names, (1, 5), ev["gap_days"], 10, dev, chunk=512, count_visits=True)
v = vis[vis.person_id.isin(set(sub.person_id))][["person_id", "age_days"]].merge(sub[["person_id", "anchor_age_days", "follow_end_age"]], on="person_id")
before = v[v.age_days <= v.anchor_age_days - ev["gap_days"]].groupby(["person_id", "anchor_age_days"]).size()
key = pd.MultiIndex.from_frame(sub[["person_id", "anchor_age_days"]])
nb = before.reindex(key, fill_value=0).to_numpy()
q = pd.qcut(pd.Series(nb).rank(method="first"), 5, labels=False).to_numpy()
res = {"n_anchors": int(len(sub)), "seed": seed}
for H in (1, 5):
    after = v[(v.age_days > v.anchor_age_days) & (v.age_days <= v.anchor_age_days + H * DAY)].groupby(["person_id", "anchor_age_days"]).size()
    obs = after.reindex(key, fill_value=0).to_numpy().astype(float)
    full = sub["follow_end_age"].to_numpy() >= sub["anchor_age_days"].to_numpy() + H * DAY
    sim = out[f"nvis{H}"]
    res[f"H{H}"] = {str(k): {"observed_mean_visits": round(float(obs[(q == k) & full].mean()), 2),
                             "simulated_mean_visits": round(float(sim[(q == k) & full].mean()), 2),
                             "n_anchors_followed": int(((q == k) & full).sum())} for k in range(5)}
print("RESULT", json.dumps(res), flush=True)
Path.home().joinpath("results").mkdir(exist_ok=True)
Path.home().joinpath("results", f"diag_visit_counts_s{seed}.json").write_text(json.dumps(res, indent=1))
print("DIAG COMPLETE")
