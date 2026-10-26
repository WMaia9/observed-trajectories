"""Smoke test of the lab-onset validation on the synthetic world (labels and small-cell rules, no BigQuery)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import lab_onset as lo
from delphi_aou.data import visits as s5v
from delphi_aou.models.visit_pp import VisitPP, VisitConfig
from test_stage3b import make_world


def test_describe_and_posterior_run_on_synthetic_world(monkeypatch):
    cohort, fo, visits = make_world(n=1500, seed=3)
    vis, _ = s5v.build(cohort, fo, [visits])
    codes = sorted({t[2:] for t in fo["token"] if t.startswith("D=")})[:2]
    monkeypatch.setattr(lo, "GROUPS", {"g1": [codes[0]], "g2": [codes[1]]})
    # fake labs: lab onset one year before the recorded diagnosis for everyone recorded, tested from EHR entry
    bd = pd.to_datetime(cohort["birth_date"]); start = (pd.to_datetime(cohort["obs_start"]) - bd).dt.days
    rows = []
    for g, c in [("g1", codes[0]), ("g2", codes[1])]:
        f = fo[fo["token"] == f"D={c}"].groupby("person_id")["age_days"].min()
        t = pd.DataFrame({"person_id": cohort["person_id"], "criterion": g, "n_tests": 3,
                          "first_test_age_days": start.to_numpy()})
        t["first_abn_age_days"] = t["person_id"].map(f - 365).astype(float)
        rows.append(t)
    labs = pd.concat(rows, ignore_index=True)
    d = lo.describe(cohort, fo, labs, vis)
    assert set(d) == {"g1", "g2"}
    for g in d.values():
        if "all" in g:
            assert g["all"]["dx_before_lab"] == 0.0                 # lab is always one year earlier here
    def load(data, seed):
        torch.manual_seed(seed)
        return VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, detection=True)).eval()
    r = lo.posterior_validation(cohort, fo, labs, vis, [0], 300, "cpu", load_model=load)
    assert r["n_anchors"] == 300 and "g1" in r
