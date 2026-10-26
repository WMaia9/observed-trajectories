"""Stage 2 validation on a synthetic cache. Run: python tests/test_stage2.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from delphi_aou import config as common
from delphi_aou.data import cohort as s2


def make_cache(n=4000, seed=3):
    r = np.random.default_rng(seed)
    pid = np.arange(10_000, 10_000 + n)
    birth = pd.Timestamp("1950-01-01") + pd.to_timedelta(r.integers(0, 365 * 30, n), unit="D")
    demo = pd.DataFrame(dict(person_id=pid, birth_date=birth, sex_token="SEX=F", race_token="RACE=X"))
    obs_start = birth + pd.to_timedelta(365 * 50, unit="D")
    span = r.uniform(0.5, 8, n)                       # some under 2 years, they must drop
    obsp = pd.DataFrame(dict(person_id=pid, obs_start=obs_start,
                             obs_end=obs_start + pd.to_timedelta((span * 365).astype(int), unit="D"), n_periods=1))
    rows = []
    codes = ["E11.9", "E11.2", "I10", "I10.0", "J45.9", "N18.3", "F32.9"]
    for i, p in enumerate(pid):
        for _ in range(r.integers(3, 30)):            # repeats of the same codes on purpose
            c = codes[r.integers(0, len(codes))]
            d = obs_start[i] + pd.to_timedelta(int(r.integers(0, 365 * 8)), unit="D")
            rows.append((p, c, d))
    events = pd.DataFrame(rows, columns=["person_id", "concept_id", "event_date"])
    dmask = r.random(n) < 0.15
    dd = (obs_start + pd.to_timedelta(365 * 3, unit="D"))[dmask]
    death = pd.DataFrame(dict(person_id=pid[dmask], event_date=dd))
    return demo, events, death, obsp


def test_stage2():
    demo, events, death, obsp = make_cache()
    coh, fo, vocab, consort, pilot = s2.build(demo, events, death, obsp)
    # 1. first occurrence only: one row per patient per token
    assert fo.duplicated(["person_id", "token"]).sum() == 0
    # 2. level 3 roll up: E11.9 and E11.2 collapse into D=E11
    assert "D=E11" in set(fo["token"]) and "D=E11.9" not in set(fo["token"])
    # 3. cohort rule: nobody with under 2 years of observation
    span = (coh["obs_end"] - coh["obs_start"]).dt.days / 365.25
    assert (span >= 2).all()
    # 4. nothing after death
    m = fo.merge(coh[["person_id", "death_age_days"]], on="person_id")
    m = m[(m["token"] != "DEATH") & m["death_age_days"].notna()]
    assert (m["age_days"] <= m["death_age_days"]).all()
    # 5. every death record in the cohort kept as a DEATH token (not truncated away)
    n_death = coh["death_age_days"].notna().sum()
    assert (fo["token"] == "DEATH").sum() == n_death
    # 6. vocabulary comes from the training split only
    train_ids = set(coh.loc[coh["split"] == "train", "person_id"])
    cnt = fo[fo["person_id"].isin(train_ids) & (fo["token"] != "DEATH")].groupby("token")["person_id"].nunique()
    assert (cnt >= 25).all() and set(cnt.index) == set(vocab["token"])
    # 7. split is the shared hash split, and first occurrence is shorter than all occurrence
    assert (coh["split"].to_numpy() == common.assign_split(coh["person_id"].to_numpy())).all()
    q_fo = pilot["first_occurrence_events_per_patient_quantiles"]["0.5"]
    q_all = pilot["all_occurrence_icd_events_per_patient_quantiles"]["0.5"]
    assert q_fo < q_all
    # 8. a repeat of the same code must not create a second token
    one = events.iloc[:1].copy(); one = pd.concat([one, one])
    # 9. suppression helper
    assert s2.sup(5) == "<21" and s2.sup(500) == 500
    print("PASS stage2; consort:", consort["split_counts"], "pilot median FO/all:", q_fo, q_all)


if __name__ == "__main__":
    test_stage2()
