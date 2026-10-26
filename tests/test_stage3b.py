"""Stage 3b validation: paired DeLong and visit (access to care) features.
World: true onset hazard does NOT depend on care access, but a disease is only recorded at the
first visit after onset. People who visit more get diagnosed sooner, so visit counts predict the
recorded label. This is the observation process effect the paper is about.
Run: python tests/test_stage3b.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from delphi_aou import config as common
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.baselines import classical as s3

DAY = 365.25


def test_delong_paired():
    r = np.random.default_rng(0)
    y = r.random(20000) < 0.1
    z = y + r.normal(size=y.size)
    good, bad = z, z + r.normal(scale=2, size=y.size)
    d, v = eval_auc.delong_paired(good, bad, y)
    assert abs(d - (roc_auc_score(y, good) - roc_auc_score(y, bad))) < 1e-9
    assert d / np.sqrt(v) > 5
    d0, v0 = eval_auc.delong_paired(good, good, y)
    assert d0 == 0 and v0 < 1e-12
    # calibration of the test: two equally good, independent noisy scores, repeated
    zs = []
    for k in range(200):
        yk = r.random(3000) < 0.2
        a = yk + r.normal(size=yk.size); b = yk + r.normal(size=yk.size)
        dk, vk = eval_auc.delong_paired(a, b, yk)
        zs.append(dk / np.sqrt(vk))
    rej = np.mean(np.abs(zs) > 1.96)
    assert rej < 0.10, rej                                     # nominal 5%


def make_world(n=30000, seed=11):
    r = np.random.default_rng(seed)
    pid = np.arange(50_000, 50_000 + n)
    birth = pd.Timestamp("1940-01-01") + pd.to_timedelta(r.integers(0, 365 * 20, n), unit="D")
    s0 = r.uniform(38, 50, n) * DAY
    end = s0 + r.uniform(12, 25, n) * DAY
    rate = np.exp(r.normal(np.log(2.0), 0.9, n))              # visits per year, varies a lot
    vrows = []
    for i in range(n):
        k = r.poisson(rate[i] * (end[i] - s0[i]) / DAY)
        t = np.sort(r.uniform(s0[i], end[i], k))
        vrows.append(pd.DataFrame({"person_id": pid[i], "age": t}))
    vis = pd.concat(vrows, ignore_index=True)
    onset = s0 + r.exponential(1 / 0.03, n) * DAY             # biology: independent of visits
    # recorded at the first visit after onset
    vis_sorted = vis.sort_values(["person_id", "age"])
    rec = np.full(n, np.nan)
    g = vis_sorted.groupby("person_id")["age"].apply(np.asarray)
    for i, p in enumerate(pid):
        a = g.get(p, np.array([]))
        after = a[a >= onset[i]]
        if len(after):
            rec[i] = after[0]
    ok = ~np.isnan(rec)
    fo = pd.DataFrame({"person_id": pid[ok], "token": "D=V01", "age_days": np.round(rec[ok]).astype(int)})
    filler = pd.DataFrame({"person_id": pid, "token": "D=Z99", "age_days": np.round(s0 + 10).astype(int)})
    fo = pd.concat([fo, filler], ignore_index=True)
    cohort = pd.DataFrame({"person_id": pid, "sex_token": "SEX=F", "race_token": "RACE=X", "birth_date": birth,
                           "obs_start": birth + pd.to_timedelta(s0, unit="D"),
                           "obs_end": birth + pd.to_timedelta(end, unit="D"), "n_periods": 1,
                           "death_age_days": np.nan})
    cohort["split"] = common.assign_split(cohort["person_id"].to_numpy())
    bmap = dict(zip(pid, birth))
    visits = pd.DataFrame({"person_id": vis["person_id"],
                           "date": vis["person_id"].map(bmap) + pd.to_timedelta(vis["age"], unit="D"),
                           "visit_concept_id": np.where(r.random(len(vis)) < 0.05, 9201, 9202)})
    return cohort, fo, visits


def test_visit_features_no_leak_and_access_effect():
    cohort, fo, visits = make_world()
    an = s3.build_anchors(cohort)
    sub = an[an["split"] == "test"].head(300).reset_index(drop=True)
    sub["anchor_id"] = np.arange(len(sub))
    F = s3.visit_features(sub, [visits.iloc[: len(visits) // 2], visits.iloc[len(visits) // 2:]], 1)
    vb = visits.assign(age=(visits["date"] - visits["person_id"].map(dict(zip(cohort.person_id, cohort.birth_date)))).dt.days)
    for i in range(0, 300, 7):                                 # recount by hand: only visits before anchor - gap
        p, a = sub.loc[i, "person_id"], sub.loc[i, "anchor_age_days"]
        n = int(((vb["person_id"] == p) & (vb["age"] <= a - 1)).sum())
        assert abs(F[i, 0] - np.log1p(n)) < 1e-5
    res = s3.run(cohort, fo, horizon_years=5, n_diseases=2, models=("agesex", "density", "visits", "gbm_visits"),
                 max_train_anchors=60000, n_hist_tokens=2, verbose=False, visits=visits)
    t = res["tables"]["visits"]; t = t[t["disease"] == "D=V01"]
    auc_vis = float(np.average(t["auc"], weights=t["n_cases"]))
    t = res["tables"]["agesex"]; t = t[t["disease"] == "D=V01"]
    auc_as = float(np.average(t["auc"], weights=t["n_cases"]))
    assert auc_vis > auc_as + 0.03, (auc_vis, auc_as)          # access predicts the recorded label
    assert "visits_minus_gbm_visits" in res["paired"]
    print("access effect: visits AUC", round(auc_vis, 3), "age-sex AUC", round(auc_as, 3))


if __name__ == "__main__":
    test_delong_paired(); print("PASS delong_paired")
    test_visit_features_no_leak_and_access_effect(); print("PASS visits")
