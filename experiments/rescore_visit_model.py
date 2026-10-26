"""Re-score a saved visit-level model (Tier A or Tier B, TIER=A|B) with both risk estimators on a random subset of the test anchors
(no retraining). Usage: SEED=0 N_ANCHORS=6000 python experiments/rescore_visit_model.py"""
import json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import torch
from delphi_aou import config as common
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.eval.calibration import log_oe
from delphi_aou.train import visit as s5
from delphi_aou.models.visit_pp import VisitPP, VisitConfig

CFG = common.load_config(); ev = CFG["evaluation"]; bands = [tuple(b) for b in ev["age_bands"]]
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("data", s2.OUT_DIR.split("/")[-1], "(set RUN_MODE=full for the full cohort)", flush=True)
seed = int(os.environ.get("SEED", "0")); n = int(os.environ.get("N_ANCHORS", "6000"))
tmp = Path(tempfile.gettempdir())
coh = s2._read_local_from_cache("s2_cohort"); fo = s2._read_local_from_cache("s2_fo_events")
vis = s2._read_local_from_cache("s5_visits")
for f in ["s3_scores_H5.npz", f"s5_model_visit{os.environ.get('TIER', 'A')}_s{seed}.pt"]:
    subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/{f}", str(tmp / f)], check=True)
npz = np.load(tmp / "s3_scores_H5.npz"); names = [str(x) for x in npz["names"]]
data = s5.VisitData(coh, fo, vis)
tier = os.environ.get("TIER", "A")
model = VisitPP(VisitConfig(data.K, n_layer=int(os.environ.get("N_LAYER", "4")), n_head=int(os.environ.get("N_HEAD", "8")),
                            n_embd=int(os.environ.get("N_EMBD", "128")), detection=tier == "B",
                            rate_anchor=os.environ.get("RATE_ANCHOR", "none"),
                            horizons=tuple(float(x) for x in os.environ.get("HORIZON_HEAD", "").split(",") if x))).to(dev)
model.load_state_dict(torch.load(tmp / f"s5_model_visit{tier}_s{seed}.pt", map_location=dev)); model.eval()
an = s3.build_anchors(coh, CFG); te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
keep = np.sort(np.random.default_rng(1).choice(len(te), min(n, len(te)), replace=False))
sub = te.iloc[keep].reset_index(drop=True)
out = {"n_anchors": int(len(keep))}
import pandas as pd
from delphi_aou.eval.calibration import log_oe as log_oe_rows
vv = vis[vis.person_id.isin(set(sub.person_id))][["person_id", "age_days"]].merge(sub[["person_id", "anchor_age_days"]], on="person_id")
nb = vv[vv.age_days <= vv.anchor_age_days - ev["gap_days"]].groupby(["person_id", "anchor_age_days"]).size()
nb = nb.reindex(pd.MultiIndex.from_frame(sub[["person_id", "anchor_age_days"]]), fill_value=0).to_numpy()
vq = pd.qcut(pd.Series(nb).rank(method="first"), 5, labels=False).to_numpy()
for est in os.environ.get("ESTIMATORS", "hits,rb").split(","):
    risks = s5.sample_risk(model, data, sub, names, (1, 5), ev["gap_days"], 10, dev, chunk=512, estimator=est,
                           close_cap=os.environ.get("CLOSE_CAP", "1") == "1")
    r = {}
    for H in (1, 5):
        Y, P, E, _ = s3.attach(te, fo, names[:-1], names[:-1], H, ev["gap_days"])
        Yk, Ek = Y[keep], E[keep]
        tab = eval_auc.evaluate_auc(risks[H], Yk, Ek, sub["age_years"].to_numpy(), sub["sex"].to_numpy(), names,
                                    bands, ev["min_cases"], ev["min_controls"])
        r[f"H{H}"] = {"auc": eval_auc.summarize(tab), "mean_log_OE": float(np.nanmean(log_oe(risks[H][:, :-1], Yk[:, :-1], Ek[:, :-1])))}
        qv = [float(np.nanmean(log_oe_rows(risks[H][:, :-1], Yk[:, :-1].astype(bool), Ek[:, :-1].astype(bool), rows=(vq == k)))) for k in range(5)]
        r[f"H{H}"]["log_OE_by_visit_quintile"] = [round(x, 3) for x in qv]
        r[f"H{H}"]["access_gap_Q5_minus_Q1"] = round(qv[4] - qv[0], 3)
        if H == 5:
            for ref in ["gbm_visits", "gbm"]:
                r[f"paired_vs_{ref}"] = eval_auc.compare_models(risks[5], npz[f"score_{ref}"][keep], Yk, Ek,
                    sub["age_years"].to_numpy(), sub["sex"].to_numpy(), names, bands, ev["min_cases"], ev["min_controls"])
    out[est] = r
    f = tmp / f"s5_scores_visit{tier}_s{seed}_{est}_n{len(keep)}{os.environ.get('TAG', '')}.npz"
    np.savez_compressed(f, keep=keep, r1=risks[1], r5=risks[5])
    subprocess.run(["gsutil", "-q", "cp", str(f), f"{s2.OUT_DIR}/"], check=False)
    print("RESULT", est, json.dumps(r, default=float), flush=True)
Path.home().joinpath("results").mkdir(exist_ok=True)
Path.home().joinpath("results", f"rescore_tier{tier}_s{seed}_n{len(keep)}{os.environ.get('TAG', '')}.json").write_text(json.dumps(out, default=float, indent=1))
print("RESCORE COMPLETE")
