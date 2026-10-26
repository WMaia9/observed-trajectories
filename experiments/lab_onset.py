"""Validate the Tier B latent onset against lab-defined (silver standard) onset, and describe the delay from lab-defined
onset to a recorded diagnosis by utilization. Needs x_lab_onset (scripts/14_extract_labs.py) in the cache.
Usage: RUN_MODE=full SEEDS=0,1,2 N_ANCHORS=20000 python experiments/lab_onset.py
Outputs only aggregates (cells of 20 or fewer people are suppressed) to ~/results/lab_onset{TAG}.json."""
import json, os, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from delphi_aou import config as common
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3
from delphi_aou.train import visit as s5
from delphi_aou.models.visit_pp import VisitPP, VisitConfig

DAY = 365.25
GROUPS = {"t2d": ["E11"], "ckd": ["N18"], "anemia": [f"D{x}" for x in range(50, 65)],
          "htn": ["I10", "I11", "I12", "I13", "I15"], "ldl": ["E78"]}
MIN_CELL = 21


def sup(n):
    """Small-cell rule: report counts of 20 or fewer as None."""
    return int(n) if n >= MIN_CELL else None


def first_dx(fo, codes):
    f = fo[fo["token"].isin([f"D={c}" for c in codes])]
    return f.groupby("person_id")["age_days"].min()


def describe(coh, fo, labs, vis):
    """Lab-defined onset vs first recorded diagnosis, inside the EHR observation window."""
    bd = pd.to_datetime(coh["birth_date"])
    c = coh.assign(start=(pd.to_datetime(coh["obs_start"]) - bd).dt.days, end=(pd.to_datetime(coh["obs_end"]) - bd).dt.days)
    nvis = vis.groupby("person_id").size()
    c["vpy"] = c["person_id"].map(nvis).fillna(0) / ((c["end"] - c["start"]).clip(lower=365) / DAY)
    c["uq"] = pd.qcut(c["vpy"].rank(method="first"), 5, labels=False)
    out = {}
    for g, codes in GROUPS.items():
        L = labs[(labs["criterion"] == g) & labs["first_abn_age_days"].notna()][["person_id", "first_abn_age_days"]]
        d = c.merge(L, on="person_id").assign(dx=lambda x: x["person_id"].map(first_dx(fo, codes)))
        d = d[(d["first_abn_age_days"] >= d["start"]) & (d["first_abn_age_days"] <= d["end"])]
        d = d[~(d["dx"] < d["start"])]                                   # recorded before EHR entry: not informative
        lag = (d["dx"] - d["first_abn_age_days"]) / DAY
        fu = (d["end"] - d["first_abn_age_days"]) / DAY
        res = {"n_lab_onset_in_window": sup(len(d))}
        for q in [None] + list(range(5)):
            m = np.ones(len(d), bool) if q is None else (d["uq"] == q).to_numpy()
            n = int(m.sum())
            if n < MIN_CELL:
                continue
            lg, f = lag[m], fu[m]
            key = "all" if q is None else f"Q{q + 1}"
            res[key] = {"n": sup(n),
                        "dx_before_lab": round(float((lg < 0).mean()), 3),
                        "dx_within_1y_after": round(float(((lg >= 0) & (lg <= 1)).mean()), 3),
                        "never_dx_with_3y_followup": (round(float((lg.isna() & (f >= 3)).sum() / max((f >= 3).sum(), 1)), 3)
                                                      if (f >= 3).sum() >= MIN_CELL else None),
                        "median_lag_years_if_dx_after": (round(float(lg[lg >= 0].median()), 2)
                                                         if (lg >= 0).sum() >= MIN_CELL else None)}
        out[g] = res
    return out


def posterior_validation(coh, fo, labs, vis, seeds, n_anchors, dev, load_model=None):
    CFG = common.load_config(); ev = CFG["evaluation"]
    tmp = Path(tempfile.gettempdir())
    data = s5.VisitData(coh, fo, vis)
    an = s3.build_anchors(coh, CFG); te = an[an["split"] == "test"].reset_index(drop=True)
    keep = np.sort(np.random.default_rng(1).choice(len(te), min(n_anchors, len(te)), replace=False))
    sub = te.iloc[keep].reset_index(drop=True)
    vv = vis[vis.person_id.isin(set(sub.person_id))][["person_id", "age_days"]].merge(sub[["person_id", "anchor_age_days"]], on="person_id")
    nb = vv[vv.age_days <= vv.anchor_age_days - ev["gap_days"]].groupby(["person_id", "anchor_age_days"]).size()
    nb = nb.reindex(pd.MultiIndex.from_frame(sub[["person_id", "anchor_age_days"]]), fill_value=0).to_numpy()
    vq = pd.qcut(pd.Series(nb).rank(method="first"), 5, labels=False).to_numpy()
    cols = [t for g in GROUPS for t in [f"D={c}" for c in GROUPS[g]] if t in data.cidx]
    post = []
    for seed in seeds:
        if load_model is not None:                                       # tests
            m = load_model(data, seed)
        else:
            f = f"s5_model_visitB_s{seed}.pt"
            subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/{f}", str(tmp / f)], check=True)
            m = VisitPP(VisitConfig(data.K, n_layer=4, n_head=8, n_embd=128, detection=True)).to(dev)
            m.load_state_dict(torch.load(tmp / f, map_location=dev)); m.eval()
        post.append(s5.sample_risk(m, data, sub, cols, (1,), ev["gap_days"], 1, dev, chunk=512, estimator="posterior")[1])
    anchor = sub["anchor_age_days"].to_numpy(float)
    res = {"n_anchors": int(len(sub))}
    for g, codes in GROUPS.items():
        ci = [cols.index(f"D={c}") for c in codes if f"D={c}" in cols]
        if not ci:
            continue
        L = labs[labs["criterion"] == g].set_index("person_id")
        abn = sub["person_id"].map(L["first_abn_age_days"]).to_numpy(float)
        tested = sub["person_id"].map(L["first_test_age_days"]).to_numpy(float) < anchor
        dx = sub["person_id"].map(first_dx(fo, codes)).to_numpy(float)
        undx = ~(dx < anchor)                                            # nothing in the group recorded before the anchor
        y = (abn < anchor) & undx                                        # lab-positive, not diagnosed, at the anchor
        r = {"n_eligible": sup(undx.sum()), "n_positive": sup(y.sum()), "n_tested_eligible": sup((undx & tested).sum())}
        for s, P in zip(seeds, post):
            p = 1 - np.prod(1 - np.nan_to_num(P[:, ci], nan=0.0), axis=1)     # any onset in the group
            for name, mk in [("all", undx), ("tested", undx & tested)]:
                yy, pp = y[mk], p[mk]
                if yy.sum() >= MIN_CELL and (~yy).sum() >= MIN_CELL:
                    r.setdefault(f"auc_{name}", []).append(round(float(roc_auc_score(yy, pp)), 4))
                    age = sub["age_years"].to_numpy()[mk]
                    r.setdefault(f"auc_age_only_{name}", [round(float(roc_auc_score(yy, age)), 4)])
            by_q = []
            for q in range(5):
                mk = undx & tested & (vq == q)
                by_q.append([round(float(p[mk].mean()), 4), round(float(y[mk].mean()), 4), sup(mk.sum())] if mk.sum() >= MIN_CELL else None)
            r.setdefault("mean_posterior_vs_lab_rate_by_quintile_tested", []).append(by_q)
        res[g] = r
    return res


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    coh = s2._read_local_from_cache("s2_cohort"); fo = s2._read_local_from_cache("s2_fo_events")
    vis = s2._read_local_from_cache("s5_visits")
    labs = s2.read_parquet("x_lab_onset")
    out = {"describe": describe(coh, fo, labs, vis)}
    print(json.dumps(out, default=float), flush=True)
    if os.environ.get("SKIP_MODEL", "0") != "1":
        seeds = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
        out["posterior"] = posterior_validation(coh, fo, labs, vis, seeds, int(os.environ.get("N_ANCHORS", "20000")), dev)
    p = Path.home() / "results" / f"lab_onset{os.environ.get('TAG', '')}.json"
    p.parent.mkdir(exist_ok=True); p.write_text(json.dumps(out, indent=1, default=float))
    print("RESULT", json.dumps(out, default=float), flush=True)


if __name__ == "__main__":
    main()
