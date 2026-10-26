#!/usr/bin/env python
"""Stage 5, step 1: visit table for the visit-level model (Tier A).
A visit = a day inside the person's EHR window [entry, min(end, death)] with any OMOP visit record or
any newly recorded first-occurrence diagnosis. Each visit carries a type (inpatient > ER > outpatient
> other > dx_only) and, in the model, the SET of first-occurrence codes recorded that day (possibly
empty). Output: s5_visits.parquet (person_id, age_days, vtype) and s5_visit_stats.json (aggregates).
Usage: RUN_MODE=test|full python stage5_visits.py
"""
import json, os, sys, tempfile, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

from delphi_aou import config as common
from delphi_aou.data import cohort as s2

CFG = common.load_config()
VTYPES = {"inpatient": 4, "er": 3, "outpatient": 2, "other": 1, "dx_only": 0}
INPATIENT, ER, OUTPATIENT = {9201, 262, 8717}, {9203}, {9202, 581477, 5083, 38004207}


def vtype_code(cid):
    cid = np.asarray(cid, dtype=float)
    out = np.full(len(cid), VTYPES["other"], dtype=np.int8)
    out[np.isin(cid, list(OUTPATIENT))] = VTYPES["outpatient"]
    out[np.isin(cid, list(ER))] = VTYPES["er"]
    out[np.isin(cid, list(INPATIENT))] = VTYPES["inpatient"]
    return out


def build(cohort, fo, visit_chunks):
    c = cohort[["person_id", "birth_date", "obs_start", "obs_end", "death_age_days"]].copy()
    bd = pd.to_datetime(c["birth_date"])
    c["entry"] = (pd.to_datetime(c["obs_start"]) - bd).dt.days
    end = (pd.to_datetime(c["obs_end"]) - bd).dt.days.astype(float)
    c["exit"] = np.where(c["death_age_days"].notna(), np.minimum(end, c["death_age_days"].astype(float)), end)
    win = c.set_index("person_id")[["entry", "exit"]]
    bmap = bd.set_axis(c["person_id"])
    parts = []
    for v in visit_chunks:
        v = v[v["person_id"].isin(win.index)]
        if len(v) == 0:
            continue
        age = (pd.to_datetime(v["date"]).to_numpy() - bmap.loc[v["person_id"]].to_numpy()).astype("timedelta64[D]").astype(int)
        df = pd.DataFrame({"person_id": v["person_id"].to_numpy(), "age_days": age, "vtype": vtype_code(v["visit_concept_id"])})
        w = win.loc[df["person_id"]].to_numpy()
        df = df[(df["age_days"].to_numpy() >= w[:, 0]) & (df["age_days"].to_numpy() <= w[:, 1])]
        parts.append(df.groupby(["person_id", "age_days"], as_index=False)["vtype"].max())
    vis = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["person_id", "age_days", "vtype"])
    vis = vis.groupby(["person_id", "age_days"], as_index=False)["vtype"].max()
    dx = fo[(fo["token"] != "DEATH") & fo["person_id"].isin(win.index)]
    dxd = dx.groupby(["person_id", "age_days"]).size().rename("n_new").reset_index()
    w = win.loc[dxd["person_id"]].to_numpy()
    dxd = dxd[(dxd["age_days"].to_numpy() >= w[:, 0]) & (dxd["age_days"].to_numpy() <= w[:, 1])]
    vis = vis.merge(dxd, on=["person_id", "age_days"], how="outer")
    vis["vtype"] = vis["vtype"].fillna(VTYPES["dx_only"]).astype(np.int8)
    vis["n_new"] = vis["n_new"].fillna(0).astype(np.int32)
    vis = vis.sort_values(["person_id", "age_days"]).reset_index(drop=True)
    per = vis.groupby("person_id").size()
    nonempty = vis[vis["n_new"] > 0]
    span = (win["exit"] - win["entry"]).clip(lower=1) / 365.25
    rate = (per / span.reindex(per.index)).replace([np.inf], np.nan)
    stats = {
        "n_persons_with_visits": int(per.size), "n_visit_days": int(len(vis)),
        "visits_per_person_quantiles": {q: float(per.quantile(float(q))) for q in ("0.1", "0.5", "0.9", "0.99")},
        "visits_per_year_quantiles": {q: float(rate.quantile(float(q))) for q in ("0.1", "0.5", "0.9")},
        "share_visit_days_without_new_dx": float((vis["n_new"] == 0).mean()),
        "mean_new_dx_per_nonempty_day": float(nonempty["n_new"].mean()) if len(nonempty) else float("nan"),
        "share_new_dx_on_dx_only_days": float(nonempty.loc[nonempty["vtype"] == 0, "n_new"].sum() / max(nonempty["n_new"].sum(), 1)),
        "vtype_share": {k: float((vis["vtype"] == v).mean()) for k, v in VTYPES.items()},
    }
    return vis[["person_id", "age_days", "vtype", "n_new"]], stats


def main():
    coh = s2._read_local_from_cache("s2_cohort")
    fo = s2._read_local_from_cache("s2_fo_events")
    vis, stats = build(coh, fo, s2.read_parquet_chunks("x_visits", ["person_id", "date", "visit_concept_id"]))
    s2.write_parquet(vis, "s5_visits")
    s2.write_json(stats, "s5_visit_stats.json")
    print(json.dumps(stats, indent=1)); print("STAGE 5 VISITS COMPLETE")


if __name__ == "__main__":
    main()
