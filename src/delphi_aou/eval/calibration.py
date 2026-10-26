#!/usr/bin/env python
"""Paper analysis: calibration and discrimination by ACCESS group (visit quintile before the
anchor) and by self reported race/ethnicity, for every model scored on the Stage 3 test anchors.

For a model with predicted horizon risk p and a disease d, inside a group g:
  O = observed incidence among eligible anchors, E = mean predicted risk among the same anchors,
  log(O/E) < 0 means the model OVER predicts.
We report, per model: mean log(O/E) over diseases (calibration in the large), mean |log(O/E)|, and
the access gap = mean log(O/E) in the top visit quintile minus the bottom one, with a patient level
bootstrap CI. Only aggregates are exported, and any disease x group cell with fewer than the policy
minimum of cases is dropped before aggregation.
Usage: RUN_MODE=full TAGS=delphi_s0_H5,observed_pp_s0_H5 [NPZ_TAGS=visitA_s0_rb_n61331] [OUT_SUFFIX=_tierA]
       python scripts/60_calibration.py
"""
import json, os, sys, tempfile, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

from delphi_aou import config as common
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3

CFG = common.load_config()
MIN_CELL = CFG["policy"]["min_cell_count_for_export"]


def log_oe(pred, Y, E, rows=None, min_cases=MIN_CELL):
    """Per disease log(O/E) on the given rows (default all). Returns array with NaN for small cells."""
    if rows is None:
        rows = np.ones(len(Y), bool)
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        e = E[:, j] & rows
        n_cases = int(Y[e, j].sum())
        if n_cases < min_cases or e.sum() < 5 * min_cases:
            continue
        out[j] = np.log(Y[e, j].mean() / max(float(pred[e, j].mean()), 1e-12))
    return out


def group_summary(pred, Y, E, groups, labels, common_cols=None):
    """Per group: mean log(O/E) and mean |log(O/E)| over diseases that are valid in EVERY group
    (so groups are compared on the same diseases)."""
    per = {g: log_oe(pred, Y, E, groups == g) for g in labels}
    if not per:                      # no group labels (e.g. race unavailable for these rows)
        return {}, None
    valid = np.all(np.vstack([np.isfinite(v) for v in per.values()]), axis=0)
    if common_cols is not None:
        valid &= common_cols
    res = {str(g): {"mean_log_oe": float(np.mean(per[g][valid])) if valid.any() else None,
                    "mean_abs_log_oe": float(np.mean(np.abs(per[g][valid]))) if valid.any() else None}
           for g in labels}
    return res, valid


def bootstrap_gap(pred, Y, E, groups, lo, hi, person, valid, n_boot=200, seed=0):
    """Patient level cluster bootstrap of [mean log O/E in group hi] minus [in group lo]."""
    rng = np.random.default_rng(seed)
    upid, inv = np.unique(person, return_inverse=True)
    rows_by_p = pd.Series(np.arange(len(person))).groupby(inv).apply(np.asarray).to_list()
    def gap(idx):
        g = groups[idx]; p, y, e = pred[idx], Y[idx], E[idx]
        a = log_oe(p, y, e, g == hi, min_cases=1)[valid]
        b = log_oe(p, y, e, g == lo, min_cases=1)[valid]
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(a[m] - b[m])) if m.any() else np.nan
    point = gap(np.arange(len(person)))
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(upid), len(upid))
        idx = np.concatenate([rows_by_p[k] for k in pick])
        bs.append(gap(idx))
    bs = np.array(bs)
    return {"gap": point, "ci_lo": float(np.nanpercentile(bs, 2.5)), "ci_hi": float(np.nanpercentile(bs, 97.5))}


def analyse(models, Y, E, person, visit_q, race, n_boot=200):
    """models: dict name -> risk matrix aligned with Y. visit_q: 0..4 quintile codes. race: labels."""
    out = {}
    for name, pred in models.items():
        r = {"overall": {"mean_log_oe": float(np.nanmean(log_oe(pred, Y, E))),
                         "mean_abs_log_oe": float(np.nanmean(np.abs(log_oe(pred, Y, E))))}}
        r["by_visit_quintile"], valid = group_summary(pred, Y, E, visit_q, list(range(5)))
        r["access_gap_top_minus_bottom"] = bootstrap_gap(pred, Y, E, visit_q, 0, 4, person, valid, n_boot)
        r["n_diseases_common_visit"] = int(valid.sum())
        labs = [l for l in pd.Series(race).value_counts().index if (race == l).sum() >= 2000]
        r["by_race"], _ = group_summary(pred, Y, E, race, labs)
        out[name] = r
    return out


def main():
    H = int(float(os.environ.get("HORIZON", "5")))
    tags = [t for t in os.environ.get("TAGS", "").split(",") if t]
    tmp = Path(tempfile.gettempdir())
    def fetch(name):
        local = tmp / name
        if s2.OUT_DIR.startswith("gs://"):
            subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/{name}", str(local)], check=True)
        else:
            local = Path(s2.OUT_DIR) / name
        return local
    z = np.load(fetch(f"s3_scores_H{H}.npz"), allow_pickle=False)
    Y, E, person = z["Y"], z["E"], z["person_id"]
    models = {k[6:]: z[k] for k in z.files if k.startswith("score_")}
    for t in tags:
        models[t] = np.load(fetch(f"s4_scores_{t}.npy"))
    for t in [t for t in os.environ.get("NPZ_TAGS", "").split(",") if t]:   # Tier A: s5_scores_<tag>.npz (keep, r1, r5)
        zt = np.load(fetch(f"s5_scores_{t}.npz"))
        assert len(zt["keep"]) == len(Y), f"{t}: scores cover {len(zt['keep'])} of {len(Y)} anchors"
        models[t] = zt[f"r{H}"]
    coh = s2._read_local_from_cache("s2_cohort")
    an = s3.build_anchors(coh)
    te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    assert np.array_equal(te["person_id"].to_numpy(), person)
    V = s3.visit_features(te, s2.read_parquet_chunks("x_visits", ["person_id", "date", "visit_concept_id"]),
                          CFG["evaluation"]["gap_days"])
    nvis = V[:, 0]
    visit_q = pd.qcut(pd.Series(nvis).rank(method="first"), 5, labels=False).to_numpy()
    race = coh.set_index("person_id").loc[person, "race_token"].astype(str).to_numpy()
    res = analyse(models, Y, E, person, visit_q, race, n_boot=int(os.environ.get("N_BOOT", "200")))
    sh = common.split_hash(coh["person_id"].to_numpy(), coh["split"].to_numpy())
    out_name = f"calibration_H{H}{os.environ.get('OUT_SUFFIX', '')}.json"
    payload = common.write_results(tmp / out_name, res, CFG, sh)
    if s2.OUT_DIR.startswith("gs://"):
        subprocess.run(["gsutil", "-q", "cp", str(tmp / out_name), f"{s2.OUT_DIR}/"], check=True)
    for k, v in res.items():
        print(k, json.dumps({"overall": v["overall"], "access_gap": v["access_gap_top_minus_bottom"],
                             "by_visit_quintile": v["by_visit_quintile"]}, default=float), flush=True)
    print("CALIBRATION COMPLETE")


if __name__ == "__main__":
    main()
