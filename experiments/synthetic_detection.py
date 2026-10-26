"""Synthetic world with known truth: onset rate 0.03 per year, independent of visits; visit rates vary 10-fold;
each visit after onset detects the disease with probability d (default 0.3). Tier A (perfect detection) versus
Tier B (detection layer): calibration of 5-year recorded risk by quintile of past visits, and recovered rate and
detection probability. Usage: python experiments/synthetic_detection.py N ITERS N_ANCHORS [DEVICE] [D] [VARIANTS] [WINDOW] [MEDIAN_VISITS_PER_YEAR]"""
import os, sys, time, numpy as np, pandas as pd, torch
from delphi_aou import config as common
from delphi_aou.data import visits as s5v
from delphi_aou.train import visit as s5
from delphi_aou.baselines import classical as s3
DAY = 365.25
def world(n, d, seed=11, med_rate=2.0):
    r = np.random.default_rng(seed)
    pid = np.arange(50_000, 50_000 + n)
    birth = pd.Timestamp("1940-01-01") + pd.to_timedelta(r.integers(0, 365 * 20, n), unit="D")
    s0 = r.uniform(38, 50, n) * DAY; end = s0 + r.uniform(12, 25, n) * DAY
    rate = np.exp(r.normal(np.log(med_rate), 0.9, n))
    vrows, rec = [], np.full(n, np.nan)
    onset = s0 + r.exponential(1 / 0.03, n) * DAY
    for i in range(n):
        k = r.poisson(rate[i] * (end[i] - s0[i]) / DAY); t = np.unique(np.floor(r.uniform(s0[i], end[i], k)))   # whole days, as in the visit table
        vrows.append(pd.DataFrame({"person_id": pid[i], "age": t}))
        after = t[t >= onset[i]]
        hit = after[r.random(len(after)) < d]                 # each visit detects with prob d
        if len(hit): rec[i] = hit[0]
    vis = pd.concat(vrows, ignore_index=True); ok = ~np.isnan(rec)
    fo = pd.concat([pd.DataFrame({"person_id": pid[ok], "token": "D=V01", "age_days": rec[ok].astype(int)}),
                    pd.DataFrame({"person_id": pid, "token": "D=Z99", "age_days": np.round(s0 + 10).astype(int)})], ignore_index=True)
    cohort = pd.DataFrame({"person_id": pid, "sex_token": "SEX=F", "race_token": "RACE=X", "birth_date": birth,
                           "obs_start": birth + pd.to_timedelta(s0, unit="D"), "obs_end": birth + pd.to_timedelta(end, unit="D"),
                           "n_periods": 1, "death_age_days": np.nan})
    cohort["split"] = common.assign_split(cohort["person_id"].to_numpy())
    bmap = dict(zip(pid, birth))
    visits = pd.DataFrame({"person_id": vis["person_id"], "date": vis["person_id"].map(bmap) + pd.to_timedelta(vis["age"], unit="D"),
                           "visit_concept_id": 9202})
    return cohort, fo, visits
n, iters, na = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
dev = sys.argv[4] if len(sys.argv) > 4 else "cpu"
D_TRUE = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
MED = float(sys.argv[8]) if len(sys.argv) > 8 else 2.0          # median visits per year
cohort, fo, visits = world(n, D_TRUE, med_rate=MED)
vis, _ = s5v.build(cohort, fo, [visits])
data = s5.VisitData(cohort, fo, vis)
an = s3.build_anchors(cohort); te = an[an.split == "test"].reset_index(drop=True); te = te.sample(min(na, len(te)), random_state=0).reset_index(drop=True)
te["anchor_id"] = np.arange(len(te))
Y, P, E, _ = s3.attach(te, fo, ["D=V01"], ["D=V01"], 5, 1)
nb = np.array([np.sum(data.v_age[data.row[p]] * DAY <= a - 1) for p, a in zip(te.person_id, te.anchor_age_days)])
q = pd.qcut(pd.Series(nb).rank(method="first"), 5, labels=False).to_numpy()
def oracle(lam=0.03, d=D_TRUE):
    """Recorded 5-year risk with the true parameters and the real future visits (removes label noise from comparisons)."""
    out = np.zeros(len(te))
    for j, (p, a) in enumerate(zip(te.person_id, te.anchor_age_days)):
        i = data.row[p]; v = data.v_age[i]; A = a / DAY
        s0, s1, last, notrec = 1.0, 0.0, data.entry[i], 1.0
        for t in v[(v * DAY <= a - 1) | ((v > A) & (v <= A + 5))]:
            x = np.exp(-lam * (t - last)); s1 += s0 * (1 - x); s0 *= x; last = t
            if t > A: notrec *= 1 - d * s1
            s1n = s1 * (1 - d); z = s0 + s1n; s0, s1 = s0 / z, s1n / z
        out[j] = 1 - notrec
    return out
r_or = oracle(); e_ = E[:, 0] == 1
by_or = np.array([np.log(Y[e_ & (q == k), 0].mean() / r_or[e_ & (q == k)].mean()) for k in range(5)])
print("oracle logOE by quintile", np.round(by_or, 2), "gap", round(by_or[4] - by_or[0], 2), "median visits per year", MED,
      "cases per quintile", [int(Y[e_ & (q == k), 0].sum()) for k in range(5)], flush=True)
variants = (sys.argv[6] if len(sys.argv) > 6 else "A,B,B0").split(",")
win = int(sys.argv[7]) if len(sys.argv) > 7 else 128
for var in variants:
    det = var != "A"
    t = time.time()
    m, _ = s5.train(data, s5.VisitConfig(data.K, n_layer=2, n_head=4, n_embd=64, detection=det, det_history=var != "B0"), iters, 64, 1e-3, 0, dev, window=win, log=lambda *a: None)
    rs = {est: s5.sample_risk(m, data, te, ["D=V01", "DEATH"], (5,), 1, n_samples=10, device=dev, window=win, chunk=512,
                              estimator=est, seed=0)[5][:, 0] for est in os.environ.get("ESTIMATORS", "rb").split(",")}
    r = rs[list(rs)[0]]
    for est, rr in rs.items():                                   # model minus oracle for every estimator
        byk = [np.log(Y[(E[:, 0] == 1) & (q == k), 0].mean() / max(rr[(E[:, 0] == 1) & (q == k)].mean(), 1e-9)) for k in range(5)]
        print("tier" + var, est, "minus oracle by quintile", np.round(np.array(byk) - by_or, 3),
              "mean", round(float(np.mean(np.array(byk) - by_or)), 3), flush=True)
    with torch.no_grad():
        b = data.batch(np.flatnonzero(data.split == "val")[:512], np.random.default_rng(0), win, dev)
        o = m.heads(m.encode(b["vt"], b["age"], b["gap"], b["sex"], b["code_b"], b["code_t"], b["code_k"], b["valid"]))
        k = data.cidx["D=V01"]; v = b["at_risk"][..., k] & b["has_next"]      # positions where V01 is still at risk
        lam = float(torch.exp(o["log_rate"][..., k])[v].mean())
        dd = float(m.detect_prob(o, b["vt_next"].clamp(max=4))[..., k][v].mean()) if det else 1.0
    print("recovered onset rate", round(lam, 4), "(truth 0.03)", "detection", round(dd, 3), f"(truth {D_TRUE})", flush=True)
    e = E[:, 0] == 1
    by = [np.log(Y[e & (q == k), 0].mean() / max(r[e & (q == k)].mean(), 1e-9)) for k in range(5)]
    print("tier" + var, "logOE by quintile", np.round(by, 2), "gap Q5-Q1", round(by[4] - by[0], 2),
          "overall", round(float(np.log(Y[e, 0].mean() / r[e].mean())), 2), "time", round(time.time() - t), flush=True)
    vs = np.array(by) - by_or                                   # model minus oracle: label noise cancels
    print("tier" + var, "minus oracle by quintile", np.round(vs, 2), "slope Q5-Q1", round(vs[4] - vs[0], 2), flush=True)
