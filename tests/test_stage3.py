"""Stage 3 validation on a synthetic world with KNOWN structure. Run: python tests/test_stage3.py
World: disease D0 raises the hazard of D1 and D2 after it occurs (comorbidity). D3 depends on sex only.
Record density (obs years) does not affect any hazard. So we expect:
  logreg/gbm with history > age-sex on D1, D2;  all models near equal on D3;
  shuffled labels near 0.5; anchors never use the future."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from delphi_aou import config as common
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.baselines import classical as s3
from delphi_aou.data import cohort as s2

DAY = 365.25


def make_world(n=60000, seed=5):
    r = np.random.default_rng(seed)
    pid = np.arange(20_000, 20_000 + n)
    sex = r.integers(0, 2, n)
    birth = pd.Timestamp("1940-01-01") + pd.to_timedelta(r.integers(0, 365 * 20, n), unit="D")
    obs_start_age = r.uniform(38, 52, n) * DAY
    span = r.uniform(8, 25, n) * DAY
    end_age = obs_start_age + span
    # D0 : constant hazard 0.05/yr from obs start. D1, D2: base 0.02/yr, x6 after D0. D3: 0.03/yr * (1 + sex)
    def draw(rate_per_year):
        return r.exponential(1.0 / rate_per_year, n) * DAY
    t0 = obs_start_age + draw(0.05)
    ev = []
    for d, base in [(1, 0.02), (2, 0.02)]:
        pre = obs_start_age + draw(base)
        post = t0 + draw(base * 6)
        td = np.where(pre < t0, pre, post)
        ev.append((d, td))
    ev.append((3, obs_start_age + draw(0.03) / (1 + sex)))
    ev.insert(0, (0, t0))
    rows = []
    for d, td in ev:
        ok = td <= end_age
        rows.append(pd.DataFrame({"person_id": pid[ok], "token": f"D=X0{d}", "age_days": np.round(td[ok]).astype(int)}))
    fo = pd.concat(rows, ignore_index=True)
    death_age = np.full(n, np.nan)
    cohort = pd.DataFrame({
        "person_id": pid, "sex_token": np.where(sex == 1, "SEX=F", "SEX=M"), "race_token": "RACE=X",
        "birth_date": birth, "obs_start": birth + pd.to_timedelta(obs_start_age, unit="D"),
        "obs_end": birth + pd.to_timedelta(end_age, unit="D"), "n_periods": 1, "death_age_days": death_age})
    cohort["split"] = common.assign_split(cohort["person_id"].to_numpy())
    return cohort, fo


def test_anchors_do_not_leak():
    cohort, fo = make_world(n=8000)
    an = s3.build_anchors(cohort)
    assert (an["obs_years_before"] >= 2 - 1e-9).all()                       # 2 years of EHR before anchor
    assert (an["follow_end_age"] >= an["anchor_age_days"]).all()
    assert an["age_years"].between(40, 80).all()
    an2 = s3.build_anchors(cohort)
    assert an.equals(an2)                                                    # deterministic
    names = ["D=X00", "D=X01", "D=X02", "D=X03"]
    sub = an[an["split"] == "test"].reset_index(drop=True)
    sub["anchor_id"] = np.arange(len(sub))
    Y, P, E, H = s3.attach(sub, fo, names, names, 5, 1)                      # asserts leakage internally
    assert not (Y & P).any()
    # prevalent diseases are never eligible, and labels only exist when not prevalent
    assert not (E & P).any()
    # history never contains a disease first diagnosed after anchor - gap: rebuild by hand for 200 rows
    f = fo.set_index("person_id")
    for i in range(200):
        pid, a = sub.loc[i, "person_id"], sub.loc[i, "anchor_age_days"]
        ev = f.loc[[pid]] if pid in f.index else f.iloc[:0]
        for t in names:
            age_t = ev.loc[ev["token"] == t, "age_days"]
            if len(age_t):
                col = names.index(t)
                assert H[i, col] == float(age_t.iloc[0] <= a - 1)
                assert Y[i, col] == bool(a < age_t.iloc[0] <= a + 5 * DAY)


def test_baselines_recover_known_structure():
    cohort, fo = make_world()
    res = s3.run(cohort, fo, horizon_years=5, n_diseases=4, models=("agesex", "density", "logreg", "gbm"),
                 max_train_anchors=60000, n_hist_tokens=4, verbose=False)
    def auc(model, d):
        t = res["tables"][model]
        t = t[t["disease"] == d]
        return float(np.average(t["auc"], weights=t["n_cases"])) if len(t) else np.nan
    for d in ["D=X01", "D=X02"]:
        assert auc("logreg", d) > auc("agesex", d) + 0.05, (d, auc("logreg", d), auc("agesex", d))
        assert auc("gbm", d) > auc("agesex", d) + 0.05
        # n_hist carries part of the comorbidity signal here, so density sits between age-sex and full history
        assert auc("agesex", d) - 0.02 < auc("density", d) < auc("logreg", d) + 0.01, (d,)
    # D3 depends on sex only. Evaluation is stratified by sex, so every model must sit near 0.5.
    # This also checks the eligibility rule: a rule that keeps short follow up only for cases
    # would give features correlated with remaining follow up a spurious signal.
    for m in ["agesex", "density", "logreg", "gbm"]:
        assert abs(auc(m, "D=X03") - 0.5) < 0.03, (m, auc(m, "D=X03"))
    # chance test: shuffled labels
    rng = np.random.default_rng(0)
    Y = res["Y"].copy()
    for d in range(Y.shape[1]):
        rng.shuffle(Y[:, d])
    t = eval_auc.evaluate_auc(res["scores"]["logreg"], Y, res["E"], res["test_anchors"]["age_years"].to_numpy(),
                              res["test_anchors"]["sex"].to_numpy(), res["names"],
                              [tuple(b) for b in common.load_config()["evaluation"]["age_bands"]], 25, 25)
    assert abs(t["auc"].mean() - 0.5) < 0.015
    print({m: s for m, s in res["summary"].items()})


if __name__ == "__main__":
    test_anchors_do_not_leak(); print("PASS anchors")
    test_baselines_recover_known_structure(); print("PASS baselines")
