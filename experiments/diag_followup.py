#!/usr/bin/env python
"""Diagnostic: how much does the DEFINITION OF FOLLOW UP change observed incidence and calibration?
  last_record : eligible if the person has an EHR record >= anchor + H (or died)  [used so far]
  admin       : eligible if anchor + H <= data cutoff (latest obs_end in the cohort minus 1 year for
                the Registered Tier date shift), whatever the person's later records
Prints mean log(O/E) for each generative model under both definitions (aggregates only)."""
import os, sys, tempfile, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from delphi_aou import config as common
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3
from delphi_aou.eval.calibration import log_oe

H = 5
DAY = 365.25
tags = os.environ.get("TAGS", "delphi_s0_H5,observed_s0_H5,observed_pp_s0_H5").split(",")
tmp = Path(tempfile.gettempdir())
def fetch(name):
    subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/{name}", str(tmp / name)], check=True)
    return tmp / name
z = np.load(fetch(f"s3_scores_H{H}.npz"))
names = [str(x) for x in z["names"]]
coh = s2._read_local_from_cache("s2_cohort")
fo = s2._read_local_from_cache("s2_fo_events")
an = s3.build_anchors(coh)
te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
assert np.array_equal(te["person_id"].to_numpy(), z["person_id"])
Y, P, E_last, _ = s3.attach(te, fo, names[:-1], names[:-1], H, 1)
cut = pd.to_datetime(coh["obs_end"]).max() - pd.Timedelta(days=365)
bd = pd.to_datetime(te["birth_date"])
cut_age = (cut - bd).dt.days.to_numpy(float)
a = te["anchor_age_days"].to_numpy(float)
admin_ok = a + H * DAY <= cut_age
E_admin = ~P & admin_ok[:, None]
print("data cutoff used:", str(cut.date()), " anchors eligible last_record:", int(E_last[:, 0].sum()),
      " admin:", int(E_admin[:, 0].sum()), flush=True)
inc_last = np.nanmean([Y[E_last[:, j], j].mean() for j in range(len(names) - 1)])
inc_admin = np.nanmean([Y[E_admin[:, j], j].mean() for j in range(len(names) - 1)])
print(f"mean recorded 5y incidence over diseases: last_record {inc_last:.4f}  admin {inc_admin:.4f}  ratio {inc_last/inc_admin:.2f}")
for t in tags:
    p = np.load(fetch(f"s4_scores_{t}.npy"))
    a1 = np.nanmean(log_oe(p[:, :-1], Y[:, :-1], E_last[:, :-1]))
    a2 = np.nanmean(log_oe(p[:, :-1], Y[:, :-1], E_admin[:, :-1]))
    print(f"{t}: mean log O/E  last_record {a1:+.3f}   admin {a2:+.3f}", flush=True)
for k in ["gbm", "gbm_visits", "agesex"]:
    p = z[f"score_{k}"]
    print(f"{k} (trained on last_record eligibility): last_record {np.nanmean(log_oe(p[:, :-1], Y[:, :-1], E_last[:, :-1])):+.3f}  admin {np.nanmean(log_oe(p[:, :-1], Y[:, :-1], E_admin[:, :-1])):+.3f}")
print("DIAG COMPLETE")
