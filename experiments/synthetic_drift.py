"""Does rollout drift come from illness-driven visits? (P1, method step after the Tier B seeds.)

On All of Us, Tier A and Tier B over-predict 5-year recorded risk for the lowest-utilization fifth of people, and the
over-prediction disappears when the rates are frozen at the anchor along the same simulated visits (rb vs rb_frozen).
Hypothesis: low-utilization people's real visits are mostly triggered by illness, so the model learns that rates rise
after a visit; simulated visits carry no illness but still move the rates.

World (truth known): one disease with onset rate 0.03 per year, independent of activity. Routine visits are Poisson
with a person-level rate (median MED per year, 10-fold spread). With probability SYMP an onset triggers a symptomatic
visit after an exponential delay (mean 0.05 years). Every visit after onset records the disease with probability D.
SYMP = 0 is the world of synthetic_detection.py, where rollouts were calibrated.

For each variant the script reports 5-year log(O/E) by quintile of past visits under the rollout estimator (rb) and,
for Tier A, under rb_frozen (same simulated visits, rates frozen at the anchor). Reproducing the All of Us pattern means:
Q1 over-predicted under rb but not under rb_frozen.

Usage: python experiments/synthetic_drift.py --n 20000 --iters 1500 --anchors 3000 --symp 0.8 --variants A,B
Writes results/synthetic_drift.json (synthetic only), one entry per run, appended as each variant finishes.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from delphi_aou import config as common
from delphi_aou.data import visits as s5v
from delphi_aou.train import visit as s5
from delphi_aou.baselines import classical as s3

DAY = 365.25


def world(n, d, symp, med_rate, seed=11, lam=0.03, delay=0.05):
    r = np.random.default_rng(seed)
    pid = np.arange(50_000, 50_000 + n)
    birth = pd.Timestamp("1940-01-01") + pd.to_timedelta(r.integers(0, 365 * 20, n), unit="D")
    s0 = r.uniform(38, 50, n) * DAY; end = s0 + r.uniform(12, 25, n) * DAY
    rate = np.exp(r.normal(np.log(med_rate), 0.9, n))
    onset = s0 + r.exponential(1 / lam, n) * DAY
    vrows, rec = [], np.full(n, np.nan)
    for i in range(n):
        k = r.poisson(rate[i] * (end[i] - s0[i]) / DAY)
        t = r.uniform(s0[i], end[i], k)
        if onset[i] < end[i] and r.random() < symp:                       # symptomatic visit triggered by onset
            ts = onset[i] + r.exponential(delay) * DAY
            if ts < end[i]:
                t = np.append(t, ts)
        t = np.unique(np.floor(t))                                         # whole days, as in the visit table
        vrows.append(pd.DataFrame({"person_id": pid[i], "age": t}))
        after = t[t >= np.floor(onset[i])]
        hit = after[r.random(len(after)) < d]
        if len(hit):
            rec[i] = hit[0]
    vis = pd.concat(vrows, ignore_index=True); ok = ~np.isnan(rec)
    fo = pd.concat([pd.DataFrame({"person_id": pid[ok], "token": "D=V01", "age_days": rec[ok].astype(int)}),
                    pd.DataFrame({"person_id": pid, "token": "D=Z99", "age_days": np.round(s0 + 10).astype(int)})],
                   ignore_index=True)
    cohort = pd.DataFrame({"person_id": pid, "sex_token": "SEX=F", "race_token": "RACE=X", "birth_date": birth,
                           "obs_start": birth + pd.to_timedelta(s0, unit="D"), "obs_end": birth + pd.to_timedelta(end, unit="D"),
                           "n_periods": 1, "death_age_days": np.nan})
    cohort["split"] = common.assign_split(cohort["person_id"].to_numpy())
    bmap = dict(zip(pid, birth))
    visits = pd.DataFrame({"person_id": vis["person_id"], "date": vis["person_id"].map(bmap) + pd.to_timedelta(vis["age"], unit="D"),
                           "visit_concept_id": 9202})
    return cohort, fo, visits


def by_quintile(Y, r, e, q):
    by = [float(np.log(Y[e & (q == k)].mean() / max(r[e & (q == k)].mean(), 1e-9))) for k in range(5)]
    return [round(x, 3) for x in by], round(by[4] - by[0], 3), round(float(np.log(Y[e].mean() / r[e].mean())), 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000); ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--anchors", type=int, default=3000); ap.add_argument("--d", type=float, default=0.3)
    ap.add_argument("--symp", type=float, default=0.8); ap.add_argument("--med", type=float, default=2.0)
    ap.add_argument("--variants", default="A,B"); ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--samples", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rate_anchor", default="none", help="none | last_dx (anchored onset rates)")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--out", default="results/synthetic_drift.json")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    cohort, fo, visits = world(a.n, a.d, a.symp, a.med)
    vis, _ = s5v.build(cohort, fo, [visits])
    data = s5.VisitData(cohort, fo, vis)
    an = s3.build_anchors(cohort); te = an[an.split == "test"].reset_index(drop=True)
    te = te.sample(min(a.anchors, len(te)), random_state=0).reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    Y, P, E, _ = s3.attach(te, fo, ["D=V01"], ["D=V01"], 5, 1)
    nb = np.array([np.sum(data.v_age[data.row[p]] * DAY <= x - 1) for p, x in zip(te.person_id, te.anchor_age_days)])
    q = pd.qcut(pd.Series(nb).rank(method="first"), 5, labels=False).to_numpy()
    e = E[:, 0] == 1; Yk = Y[:, 0]
    print("world", vars(a), "eligible anchors", int(e.sum()), "cases by quintile", [int(Yk[e & (q == k)].sum()) for k in range(5)], flush=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    allr = json.loads(out.read_text()) if out.exists() else []
    for var in a.variants.split(","):
        t0 = time.time()
        cfg = s5.VisitConfig(data.K, n_layer=2, n_head=4, n_embd=64, detection=var != "A", det_history=True,
                             rate_anchor=a.rate_anchor)
        m, _ = s5.train(data, cfg, a.iters, 64, 1e-3, a.seed, a.device, window=a.window, log=lambda *x: None)
        res = {"variant": var, "rate_anchor": a.rate_anchor, "n": a.n, "iters": a.iters, "anchors": a.anchors,
               "d": a.d, "symp": a.symp, "med": a.med, "seed": a.seed}
        ests = ["rb", "rb_frozen"] if var == "A" else ["rb"]
        for est in ests:
            r = s5.sample_risk(m, data, te, ["D=V01", "DEATH"], (5,), 1, n_samples=a.samples, device=a.device,
                               window=a.window, chunk=512, estimator=est)[5][:, 0]
            byq, gap, overall = by_quintile(Yk, r, e, q)
            res[est] = {"logOE_by_quintile": byq, "gap_Q5_minus_Q1": gap, "overall": overall}
            print(var, a.rate_anchor, est, "logOE", byq, "gap", gap, "overall", overall, flush=True)
        res["seconds"] = round(time.time() - t0)
        allr.append(res); out.write_text(json.dumps(allr, indent=1))
    print("DRIFT DONE", flush=True)
