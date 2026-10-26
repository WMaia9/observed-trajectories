#!/usr/bin/env python
"""Stage 3: landmark anchors, incident labels and baselines (age-sex, record density,
logistic with history, gradient boosting), scored with the Stage 1 evaluation.

Design (matches the protocol in the report):
  * anchor = a random age inside each 5 year band, per person, seeded. The person must be
    alive, inside the observation period, and have at least min_ehr_years of EHR before it.
  * history = first occurrence events with age <= anchor - gap. Events in (anchor - gap, anchor]
    count as prevalent (never as labels, never as history), so nothing leaks.
  * label = disease first occurs in (anchor, anchor + H years].
  * eligible = not prevalent AND (followed up through the horizon OR died in the window). The rule
    never looks at the label, because keeping short follow up people only when they have the event
    leaks the outcome. Everyone else is dropped, never counted as a control.
Usage: RUN_MODE=test|full HORIZON=5 N_DISEASES=60 MODELS=agesex,density,logreg,gbm python stage3_baselines.py
"""
import json, os, sys, time, tempfile, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from delphi_aou import config as common
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.data import cohort as s2

CFG = common.load_config()
DAY = 365.25


# --------------------------------------------------------------------------- anchors
def build_anchors(cohort, cfg=CFG, seed=None):
    seed = cfg["seed"] if seed is None else seed
    ev = cfg["evaluation"]
    c = cohort.sort_values("person_id").reset_index(drop=True).copy()
    for col in ["birth_date", "obs_start", "obs_end"]:
        c[col] = pd.to_datetime(c[col])
    c["obs_start_age"] = (c["obs_start"] - c["birth_date"]).dt.days.astype(float)
    c["obs_end_age"] = (c["obs_end"] - c["birth_date"]).dt.days.astype(float)
    death = c["death_age_days"].astype(float).to_numpy()
    sex = c["sex_token"].astype("category").cat.codes.to_numpy()      # stable: sorted categories
    rng = np.random.default_rng(seed)
    parts = []
    for lo, hi in ev["age_bands"]:
        a = rng.uniform(lo * DAY, hi * DAY, len(c))
        ok = ((a - c["obs_start_age"].to_numpy() >= cfg["cohort"]["min_ehr_years_before_index"] * DAY)
              & (a <= c["obs_end_age"].to_numpy())
              & (np.isnan(death) | (death > a)))
        p = pd.DataFrame({
            "person_id": c["person_id"].to_numpy()[ok], "split": c["split"].to_numpy()[ok],
            "sex": sex[ok], "birth_date": c["birth_date"].to_numpy()[ok],
            "anchor_age_days": a[ok], "age_years": a[ok] / DAY,
            "obs_years_before": (a[ok] - c["obs_start_age"].to_numpy()[ok]) / DAY,
            "follow_end_age": np.fmin(c["obs_end_age"].to_numpy()[ok], np.where(np.isnan(death[ok]), np.inf, death[ok])),
            "death_age": death[ok]})
        parts.append(p)
    an = pd.concat(parts, ignore_index=True)
    an["anchor_id"] = np.arange(len(an))
    return an


def attach(anchors, fo, disease_tokens, hist_tokens, horizon_years, gap_days):
    """Returns Y, P, E (N x D bool, last column DEATH) and sparse history H (N x V)."""
    N = len(anchors)
    D = len(disease_tokens) + 1
    d_index = {t: i for i, t in enumerate(disease_tokens)}
    h_index = {t: i for i, t in enumerate(hist_tokens)}
    keep_tokens = set(disease_tokens) | set(hist_tokens)
    f = fo[fo["token"].isin(keep_tokens) & fo["person_id"].isin(set(anchors["person_id"]))]
    m = anchors[["anchor_id", "person_id", "anchor_age_days"]].merge(f, on="person_id", how="inner")
    age, a = m["age_days"].to_numpy(float), m["anchor_age_days"].to_numpy(float)
    is_hist = age <= a - gap_days
    is_prev = age <= a
    is_lab = (age > a) & (age <= a + horizon_years * DAY)
    row = m["anchor_id"].to_numpy()
    di = m["token"].map(d_index).to_numpy(dtype=float)
    hi = m["token"].map(h_index).to_numpy(dtype=float)
    Y = np.zeros((N, D), bool); P = np.zeros((N, D), bool)
    ok = ~np.isnan(di)
    Y[row[ok & is_lab], di[ok & is_lab].astype(int)] = True
    P[row[ok & is_prev], di[ok & is_prev].astype(int)] = True
    okh = ~np.isnan(hi) & is_hist
    H = sparse.csr_matrix((np.ones(okh.sum(), np.float32), (row[okh], hi[okh].astype(int))), shape=(N, len(hist_tokens)))
    H.data[:] = 1.0
    # DEATH column
    da = anchors["death_age"].to_numpy(float); aa = anchors["anchor_age_days"].to_numpy(float)
    died = ~np.isnan(da) & (da > aa) & (da <= aa + horizon_years * DAY)
    Y[:, -1] = died
    followed = anchors["follow_end_age"].to_numpy(float) >= aa + horizon_years * DAY
    # Eligibility must not depend on the label: keeping short follow up people only when they
    # have the event would leak (cases are over represented among short follow up). So an anchor
    # is eligible if the person was followed through the horizon or died inside it (competing
    # event, scored as no event for non fatal diseases). This is the censoring independent rule.
    E = ~P & (followed[:, None] | died[:, None])
    # leakage self checks
    assert not (Y & P).any(), "label overlaps prevalent"
    assert (age[is_hist] <= a[is_hist] - gap_days).all()
    return Y, P, E, H


def dense_features(anchors, H):
    n_hist = np.asarray(H.sum(axis=1)).ravel()
    x = np.column_stack([anchors["age_years"], anchors["age_years"] ** 2 / 100.0, anchors["sex"],
                         np.log1p(n_hist), anchors["obs_years_before"]]).astype(np.float32)
    return x


VISIT_INPATIENT = {9201, 262}      # OMOP: Inpatient Visit, ER and Inpatient Visit
VISIT_ER = {9203, 262}             # OMOP: Emergency Room Visit, ER and Inpatient Visit


def visit_features(anchors, visit_chunks, gap_days):
    """Access to care features, all strictly before anchor - gap:
    log counts of visits (ever, last 1 year), inpatient and ER visits, log days since last visit."""
    a = anchors[["anchor_id", "person_id", "birth_date", "anchor_age_days"]]
    persons = set(a["person_id"])
    parts = []
    for v in visit_chunks:
        v = v[v["person_id"].isin(persons)]
        if len(v) == 0:
            continue
        v = v[["person_id", "date", "visit_concept_id"]].merge(a, on="person_id")
        age = (pd.to_datetime(v["date"]) - pd.to_datetime(v["birth_date"])).dt.days.to_numpy(float)
        cut = v["anchor_age_days"].to_numpy(float) - gap_days
        before = age <= cut
        v = v.assign(age=age)[before]
        cut = cut[before]
        vc = v["visit_concept_id"].fillna(0).astype(int)
        g = pd.DataFrame({"anchor_id": v["anchor_id"].to_numpy(), "n": 1,
                          "n1y": (v["age"].to_numpy() > cut - DAY).astype(int),
                          "inp": vc.isin(VISIT_INPATIENT).astype(int).to_numpy(),
                          "er": vc.isin(VISIT_ER).astype(int).to_numpy(),
                          "gap": cut - v["age"].to_numpy()})
        parts.append(g.groupby("anchor_id").agg(n=("n", "sum"), n1y=("n1y", "sum"), inp=("inp", "sum"),
                                                 er=("er", "sum"), gap=("gap", "min")))
    if parts:
        f = pd.concat(parts).groupby(level=0).agg(n=("n", "sum"), n1y=("n1y", "sum"), inp=("inp", "sum"),
                                                 er=("er", "sum"), gap=("gap", "min"))
    else:
        f = pd.DataFrame(columns=["n", "n1y", "inp", "er", "gap"])
    f = f.reindex(anchors["anchor_id"]).fillna({"n": 0, "n1y": 0, "inp": 0, "er": 0, "gap": 36500})
    return np.column_stack([np.log1p(f["n"]), np.log1p(f["n1y"]), np.log1p(f["inp"]), np.log1p(f["er"]),
                            np.log1p(f["gap"])]).astype(np.float32)


# --------------------------------------------------------------------------- models
def agesex_scores(tr, Ytr, Etr, te, mincount=20):
    """Incidence by sex and single year of age from the training anchors."""
    D = Ytr.shape[1]
    sx_tr, sx_te = tr["sex"].to_numpy(), te["sex"].to_numpy()
    ag_tr = np.clip(tr["age_years"].to_numpy().astype(int), 0, 119)
    ag_te = np.clip(te["age_years"].to_numpy().astype(int), 0, 119)
    ns = int(max(sx_tr.max(), sx_te.max())) + 1
    num = np.zeros((ns, 120, D)); den = np.zeros((ns, 120, D))
    np.add.at(num, (sx_tr, ag_tr), (Ytr & Etr).astype(float))
    np.add.at(den, (sx_tr, ag_tr), Etr.astype(float))
    # pool +-2 years so that single ages with few anchors stay stable
    k = np.ones(5)
    num_s = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, num)
    den_s = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 1, den)
    rate = (num_s + 0.01) / (den_s + mincount)
    return rate[sx_te, ag_te].astype(np.float32)


def _fit_each(make, Xtr, Ytr, Etr, Xte, fallback, min_pos=20, cap=None, seed=0):
    out = fallback.copy()
    rng = np.random.default_rng(seed)
    for d in range(Ytr.shape[1]):
        use = Etr[:, d]
        y = Ytr[use, d]
        if y.sum() < min_pos or (~y).sum() < min_pos:
            continue
        idx = np.flatnonzero(use)
        if cap and len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        X = Xtr[idx]
        model = make()
        model.fit(X, Ytr[idx, d])
        out[:, d] = model.predict_proba(Xte)[:, 1]
    return out


def density_scores(Xtr, Ytr, Etr, Xte, fb, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=300))
    return _fit_each(mk, Xtr, Ytr, Etr, Xte, fb, seed=seed)


def logreg_scores(Xtr, Htr, Ytr, Etr, Xte, Hte, fb, seed=0):
    from sklearn.linear_model import LogisticRegression
    sc = StandardScaler_fit(Xtr)
    A = sparse.hstack([sparse.csr_matrix(sc[0]), Htr]).tocsr()
    B = sparse.hstack([sparse.csr_matrix(sc[1](Xte)), Hte]).tocsr()
    mk = lambda: LogisticRegression(C=0.1, max_iter=300)
    return _fit_each(mk, A, Ytr, Etr, B, fb, seed=seed)


def StandardScaler_fit(X):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, (lambda Z: (Z - mu) / sd)


def gbm_scores(Xtr, Htr, Ytr, Etr, Xte, Hte, fb, cap=60000, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    A = np.hstack([Xtr, Htr.toarray()]); B = np.hstack([Xte, Hte.toarray()])
    mk = lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=4, learning_rate=0.1,
                                                early_stopping=True, random_state=seed)
    return _fit_each(mk, A, Ytr, Etr, B, fb, cap=cap, seed=seed)


# --------------------------------------------------------------------------- driver
def run(cohort, fo, horizon_years=5, n_diseases=60, models=("agesex", "density", "logreg", "gbm"),
        max_train_anchors=150_000, n_hist_tokens=300, cfg=CFG, verbose=True, visits=None):
    """visits: None, a DataFrame, or a zero argument callable returning an iterable of
    chunks (person_id, date, visit_concept_id). Needed for the *_visits models."""
    t0 = time.time()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    anchors = build_anchors(cohort, cfg)
    log(f"anchors: {len(anchors):,}  by split: {anchors['split'].value_counts().to_dict()}")
    tr_all = anchors[anchors["split"] == "train"]
    tr_ids = set(tr_all["person_id"])
    prev = fo[fo["person_id"].isin(tr_ids) & (fo["token"] != "DEATH")].groupby("token")["person_id"].nunique()
    prev = prev.sort_values(ascending=False)
    disease_tokens = list(prev.index[:n_diseases])
    hist_tokens = list(prev.index[:n_hist_tokens])
    names = disease_tokens + ["DEATH"]
    tr = tr_all.sample(n=min(max_train_anchors, len(tr_all)), random_state=cfg["seed"]).sort_values("anchor_id")
    te = anchors[anchors["split"] == "test"]
    gap = cfg["evaluation"]["gap_days"]
    def mats(a):
        sub = a.reset_index(drop=True).copy(); sub["anchor_id"] = np.arange(len(sub))
        Y, P, E, H = attach(sub, fo, disease_tokens, hist_tokens, horizon_years, gap)
        return sub, Y, P, E, H, dense_features(sub, H)
    trs, Ytr, Ptr, Etr, Htr, Xtr = mats(tr)
    tes, Yte, Pte, Ete, Hte, Xte = mats(te)
    log(f"train anchors {len(trs):,} test anchors {len(tes):,} diseases {len(names)} "
        f"mean prevalence in window {Ytr[:, :-1].mean():.4f}")
    base_te = agesex_scores(trs, Ytr, Etr, tes)
    scores = {}
    if visits is not None:
        get = (lambda: [visits]) if isinstance(visits, pd.DataFrame) else visits
        Vtr, Vte = visit_features(trs, get(), gap), visit_features(tes, get(), gap)
        A_tr, A_te = np.column_stack([Xtr[:, [0, 1, 2, 4]], Vtr]), np.column_stack([Xte[:, [0, 1, 2, 4]], Vte])
        XVtr, XVte = np.column_stack([Xtr, Vtr]), np.column_stack([Xte, Vte])
        if "visits" in models: scores["visits"] = density_scores(A_tr, Ytr, Etr, A_te, base_te)
        if "density_visits" in models: scores["density_visits"] = density_scores(XVtr, Ytr, Etr, XVte, base_te)
        if "logreg_visits" in models: scores["logreg_visits"] = logreg_scores(XVtr, Htr, Ytr, Etr, XVte, Hte, base_te)
        if "gbm_visits" in models: scores["gbm_visits"] = gbm_scores(XVtr, Htr, Ytr, Etr, XVte, Hte, base_te)
    if "agesex" in models: scores["agesex"] = base_te
    if "density" in models: scores["density"] = density_scores(Xtr, Ytr, Etr, Xte, base_te)
    if "logreg" in models: scores["logreg"] = logreg_scores(Xtr, Htr, Ytr, Etr, Xte, Hte, base_te)
    if "gbm" in models: scores["gbm"] = gbm_scores(Xtr, Htr, Ytr, Etr, Xte, Hte, base_te)
    ev = cfg["evaluation"]
    tables, summ = {}, {}
    for k, s in scores.items():
        t = eval_auc.evaluate_auc(s, Yte, Ete, tes["age_years"].to_numpy(), tes["sex"].to_numpy(), names,
                                  [tuple(b) for b in ev["age_bands"]], ev["min_cases"], ev["min_controls"])
        tables[k] = t
        summ[k] = eval_auc.summarize(t)
        log(f"  {k:8s} {summ[k]}  (t+{time.time()-t0:,.0f}s)")
    # paired comparison of every model against the strongest classical baseline available
    ref = "gbm_visits" if "gbm_visits" in scores else ("gbm" if "gbm" in scores else None)
    paired = {}
    if ref:
        for k, s_ in scores.items():
            if k != ref:
                paired[f"{k}_minus_{ref}"] = eval_auc.compare_models(
                    s_, scores[ref], Yte, Ete, tes["age_years"].to_numpy(), tes["sex"].to_numpy(), names,
                    [tuple(b) for b in ev["age_bands"]], ev["min_cases"], ev["min_controls"])
        for k, v in paired.items():
            log(f"  paired {k}: {v}")
    return dict(anchors=anchors, tables=tables, summary=summ, names=names, test_anchors=tes,
                Y=Yte, E=Ete, scores=scores, paired=paired, disease_tokens=disease_tokens,
                hist_tokens=hist_tokens)


def main():
    mode = os.environ.get("RUN_MODE", "test")
    H = float(os.environ.get("HORIZON", "5"))
    K = int(os.environ.get("N_DISEASES", "60"))
    models = tuple(os.environ.get("MODELS", "agesex,density,logreg,gbm,visits,density_visits,logreg_visits,gbm_visits").split(","))
    coh = s2._read_local_from_cache("s2_cohort")
    fo = s2._read_local_from_cache("s2_fo_events")
    visits = None
    if any(m.endswith("visits") for m in models):
        visits = lambda: s2.read_parquet_chunks("x_visits", ["person_id", "date", "visit_concept_id"])
    res = run(coh, fo, H, K, models, visits=visits)
    out = {"horizon_years": H, "n_diseases": K, "summary": res["summary"], "paired": res["paired"]}
    te = res["test_anchors"]
    np.savez_compressed(Path(tempfile.gettempdir()) / f"s3_scores_H{int(H)}.npz",
                        names=np.array(res["names"]), Y=res["Y"], E=res["E"],
                        person_id=te["person_id"].to_numpy(), age_years=te["age_years"].to_numpy(),
                        anchor_age_days=te["anchor_age_days"].to_numpy(), sex=te["sex"].to_numpy(),
                        **{f"score_{k}": v for k, v in res["scores"].items()})
    sh = common.split_hash(coh["person_id"].to_numpy(), coh["split"].to_numpy())
    local = Path(tempfile.gettempdir()) / f"s3_summary_H{int(H)}.json"
    payload = common.write_results(local, out, CFG, sh)
    for k, t in res["tables"].items():
        t.to_csv(Path(tempfile.gettempdir()) / f"s3_auc_{k}_H{int(H)}.csv", index=False)
    if s2.OUT_DIR.startswith("gs://"):
        subprocess.run(f"gsutil -q cp {tempfile.gettempdir()}/s3_*_H{int(H)}.* {s2.OUT_DIR}/", shell=True, check=True)
    print(json.dumps(payload, indent=2, default=float)); print("STAGE 3 COMPLETE")


if __name__ == "__main__":
    main()
