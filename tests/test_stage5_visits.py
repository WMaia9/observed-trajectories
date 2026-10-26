"""Visit table checks on the Stage 3b synthetic world (visits drive recording). Run: python tests/test_stage5_visits.py"""
import sys
from pathlib import Path
import numpy as np
from delphi_aou.data import visits as s5
from test_stage3b import make_world  # noqa: E402


def test_visits():
    cohort, fo, visits = make_world(n=3000)
    vis, st = s5.build(cohort, fo, [visits.iloc[:5000], visits.iloc[5000:]])
    assert not vis.duplicated(["person_id", "age_days"]).any()                    # one row per day
    dx = fo[fo["token"] != "DEATH"].groupby(["person_id", "age_days"]).size()
    m = vis.set_index(["person_id", "age_days"])["n_new"]
    common = dx.index.intersection(m.index)
    assert (m.loc[common] == dx.loc[common]).all()                                  # new dx counts match
    assert st["share_visit_days_without_new_dx"] > 0.5                              # empty visits are kept
    w = cohort.set_index("person_id")
    bd = w["birth_date"]
    ent = (w["obs_start"] - bd).dt.days
    ext = (w["obs_end"] - bd).dt.days
    j = vis.join(ent.rename("e"), on="person_id").join(ext.rename("x"), on="person_id")
    assert ((j["age_days"] >= j["e"]) & (j["age_days"] <= j["x"])).all()          # inside the window
    print("PASS visits", {k: st[k] for k in ["share_visit_days_without_new_dx", "mean_new_dx_per_nonempty_day"]})


if __name__ == "__main__":
    test_visits()
