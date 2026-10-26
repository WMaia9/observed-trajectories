"""Simulator with KNOWN disease dynamics observed through an EHR window (paper experiment).

Truth: onsets of K diseases follow age dependent hazards; D0 multiplies the hazard of D1 and D2.
Observation (the knobs the paper varies):
  entry age E ~ U(entry_lo, entry_hi), end = min(death, E + U(5, 20) years)
  prevalent burst: a disease with onset before E is recorded AT E with probability p_burst
  detection delay: an onset inside the window is recorded at the next visit, visits ~ Poisson
      with a person specific rate (lognormal); if that visit falls after the end it is never seen
Outputs the same schema as Stage 2 (cohort, fo) plus the true onset ages, so models trained on
recorded data can be scored against the TRUTH (true incident onsets) as well as the record.
"""
import numpy as np
import pandas as pd
from delphi_aou import config as common

DAY = 365.25


def simulate(n=30000, K=6, seed=0, entry_lo=35, entry_hi=70, p_burst=0.8, visit_mu=np.log(2.0), visit_sd=0.9,
             comorb=4.0, start_age=30.0, end_age=90.0, step=1 / 12, frailty_sd=0.8, shared_visits=False):
    r = np.random.default_rng(seed)
    ages = np.arange(start_age, end_age, step)
    b0 = np.array([0.006, 0.003, 0.003, 0.004, 0.002, 0.005])[:K]
    slope = np.array([0.05, 0.06, 0.04, 0.07, 0.09, 0.03])[:K]
    frail = np.exp(r.normal(0, frailty_sd, n) - frailty_sd ** 2 / 2)   # shared across diseases, mean 1
    onset = np.full((n, K), np.inf)
    death = np.full(n, np.inf)
    for a in ages:
        alive = ~np.isfinite(death)
        h = b0 * np.exp(slope * (a - 60))
        h = np.broadcast_to(h, (n, K)) * frail[:, None]
        has0 = np.isfinite(onset[:, 0])
        h[has0, 1:3] *= comorb
        new = (r.random((n, K)) < h * step) & ~np.isfinite(onset) & alive[:, None]
        onset[new] = a + r.random(new.sum()) * step
        hd = 0.0005 * np.exp(0.09 * (a - 60)) * np.where(has0, 2.0, 1.0)
        dies = alive & (r.random(n) < hd * step)
        death[dies] = a + r.random(dies.sum()) * step
    E = r.uniform(entry_lo, entry_hi, n)
    end = np.minimum(death, E + r.uniform(5, 20, n))
    keep = E < end - 2                                          # at least two observed years
    vr = np.exp(r.normal(visit_mu, visit_sd, n))                # visits per year
    rec = np.full((n, K), np.nan)
    if shared_visits:
        visit_times = [np.sort(r.uniform(E[i], max(end[i], E[i] + 1e-6), r.poisson(vr[i] * max(end[i] - E[i], 0))))
                       for i in range(n)]
    for k in range(K):
        o = onset[:, k]
        pre = o < E
        burst = pre & (r.random(n) < p_burst)
        rec[burst, k] = E[burst]
        inside = (o >= E) & (o <= end)
        if shared_visits:
            # recorded at the first VISIT after onset; visits are shared by all diseases of a person,
            # so diagnoses arrive in bursts on the same day (as in real EHR)
            t = np.full(n, np.inf)
            for i in np.flatnonzero(inside):
                vt = visit_times[i]
                after = vt[vt >= o[i]]
                if len(after):
                    t[i] = after[0]
        else:
            delay = r.exponential(1.0 / vr)
            t = o + delay
        ok = inside & (t <= end)
        rec[ok, k] = t[ok]
    pid = np.arange(1_000_000, 1_000_000 + n)
    birth = pd.Timestamp("1930-01-01") + pd.to_timedelta(r.integers(0, 365 * 30, n), unit="D")
    rows = []
    for k in range(K):
        m = keep & np.isfinite(rec[:, k])
        rows.append(pd.DataFrame({"person_id": pid[m], "token": f"D=S{k:02d}",
                                  "age_days": np.round(rec[m, k] * DAY).astype(int)}))
    fo = pd.concat(rows, ignore_index=True)
    dobs = np.where(np.isfinite(death) & (death <= end), death * DAY, np.nan)
    cohort = pd.DataFrame({"person_id": pid, "sex_token": np.where(r.random(n) < 0.5, "SEX=F", "SEX=M"),
                           "race_token": "RACE=X", "birth_date": birth,
                           "obs_start": birth + pd.to_timedelta(E * DAY, unit="D"),
                           "obs_end": birth + pd.to_timedelta(end * DAY, unit="D"), "n_periods": 1,
                           "death_age_days": dobs})[keep].reset_index(drop=True)
    dth = cohort[cohort["death_age_days"].notna()][["person_id", "death_age_days"]]
    fo = pd.concat([fo, dth.rename(columns={"death_age_days": "age_days"}).assign(token="DEATH")[["person_id", "token", "age_days"]]],
                   ignore_index=True)
    fo["age_days"] = fo["age_days"].astype(int)
    cohort["split"] = common.assign_split(cohort["person_id"].to_numpy())
    truth = pd.DataFrame(onset[keep], columns=[f"D=S{k:02d}" for k in range(K)])
    truth["person_id"] = pid[keep]
    return cohort, fo, truth


def true_labels(anchors, truth, names, horizon_years):
    """True incident onset in (anchor, anchor + H], eligible if no TRUE onset before the anchor."""
    t = truth.set_index("person_id").loc[anchors["person_id"].to_numpy()]
    a = anchors["anchor_age_days"].to_numpy() / DAY
    Y = np.zeros((len(a), len(names)), bool); E = np.zeros_like(Y)
    for j, nm in enumerate(names):
        if nm not in t.columns:
            continue
        o = t[nm].to_numpy()
        E[:, j] = ~(o <= a)
        Y[:, j] = (o > a) & (o <= a + horizon_years)
    return Y, E
