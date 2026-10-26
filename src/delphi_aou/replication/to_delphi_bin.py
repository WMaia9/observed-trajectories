"""Convert All of Us tables to the official Delphi data format (train.bin / val.bin, uint32 rows of
[patient_id, age_in_days, token_id]), following data/ukb_simulated_data/example_ukb_to_bin.ipynb.

Token ids come from the official labels.csv exactly as the authors build them: the first word of
row r (0 based) gets id r - 1, so Padding = -1 (unused), No event = 0, Female = 1, Male = 2,
BMI low/mid/high = 3/4/5, Smoking low/mid/high = 6/7/8, Alcohol low/mid/high = 9/10/11, then ICD-10
level 3 codes (A00 = 12, ...) and Death. Then, as in the notebook: sort by patient, age, death last;
keep ages >= 0; keep the first occurrence of each (patient, token); split 80/20 by patient id.

All of Us differences (documented, unavoidable):
  * diagnoses come from EHR condition records (ICD-10-CM source codes plus SNOMED mapped to ICD-10-CM),
    not UK Biobank first-occurrence fields; there is no cancer registry
  * lifestyle: BMI from the enrollment physical measurement (same cut points 22 and 28); smoking from
    "Smoke Frequency" (every day = high, some days = mid, not at all or never smoked 100 cigarettes = low);
    alcohol from "Drink Frequency Past Year" (4 or more times a week = high, 2 to 3 times a week = mid,
    anything less or never drinks = low), dated at the survey or measurement
  * ages are from Registered Tier date-shifted dates (a constant shift per person, so ages and gaps hold)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

OFFICIAL = Path(__file__).resolve().parents[3] / "third_party" / "delphi_official"


def label_dict(labels_csv=None):
    path = Path(labels_csv) if labels_csv else OFFICIAL / "data" / "ukb_real_data" / "labels.csv"
    d, i = {}, -1
    for line in path.read_text().splitlines():
        d[line.strip().split(" ")[0]] = i
        i += 1
    return d


def bmi_status(v):
    v = np.asarray(v, float)
    return np.where(v > 28, 5, np.where(v > 22, 4, 3))


def smoking_status(answer):
    a = pd.Series(answer).astype(str).str.lower()
    out = np.full(len(a), 6)
    out[a.str.contains("every day").to_numpy()] = 8
    out[a.str.contains("some days").to_numpy()] = 7
    return out


def alcohol_status(answer):
    a = pd.Series(answer).astype(str).str.lower()
    out = np.full(len(a), 9)
    out[a.str.contains(r"2 to 3 (?:times )?(?:per|a) week|2-3 times a week", regex=True).to_numpy()] = 10
    out[a.str.contains(r"4 or more", regex=True).to_numpy()] = 11
    return out


def build_rows(persons, icd, death, bmi=None, smoking=None, alcohol=None, labels=None):
    """persons: person_id, birth_date, sex ('Female'/'Male'); icd: person_id, code3, date;
    death: person_id, date; bmi: person_id, date, value; smoking/alcohol: person_id, date, answer.
    Returns int64 array [n, 3] before the official ordering/dedup step, plus a small report."""
    L = labels or label_dict()
    p = persons[persons["sex"].isin(["Female", "Male"])].copy()
    p["birth_date"] = pd.to_datetime(p["birth_date"])
    bd = p.set_index("person_id")["birth_date"]
    parts, report = [], {}
    sexid = np.where(p["sex"].to_numpy() == "Female", L["Female"], L["Male"])
    parts.append(np.c_[p["person_id"].to_numpy(), np.zeros(len(p), np.int64), sexid])

    def age_days(df):
        d = df[df["person_id"].isin(bd.index)]
        return d, (pd.to_datetime(d["date"]).to_numpy() - bd.loc[d["person_id"]].to_numpy()).astype("timedelta64[D]").astype(np.int64)

    d, a = age_days(icd)
    lab = d["code3"].astype(str).str.upper().str[:3].map(L)
    report["icd_rows"] = int(len(d)); report["icd_rows_not_in_delphi_vocab"] = int(lab.isna().sum())
    ok = lab.notna().to_numpy()
    parts.append(np.c_[d["person_id"].to_numpy()[ok], a[ok], lab.to_numpy()[ok].astype(np.int64)])
    d, a = age_days(death)
    parts.append(np.c_[d["person_id"].to_numpy(), a, np.full(len(d), L["Death"])])
    for df, fn, col in [(bmi, bmi_status, "value"), (smoking, smoking_status, "answer"), (alcohol, alcohol_status, "answer")]:
        if df is None or len(df) == 0:
            continue
        d, a = age_days(df)
        parts.append(np.c_[d["person_id"].to_numpy(), a, fn(d[col].to_numpy())])
    return np.vstack(parts).astype(np.int64), report


def official_order_and_split(data, train_proportion=0.8, death_id=None):
    """Exactly the notebook's ordering and dedup, then an 80/20 split by sorted patient id."""
    death_id = data[:, 2].max() if death_id is None else death_id
    data = data[np.lexsort((data[:, 1], data[:, 2] == death_id, data[:, 0]))]
    data = data[data[:, 1] >= 0]
    data = pd.DataFrame(data).drop_duplicates([0, 2]).values
    ids = np.sort(np.unique(data[:, 0]))
    cut = ids[int(len(ids) * train_proportion)]      # the notebook intends this (its ids.sort() returns None)
    train = data[data[:, 0] <= cut].astype(np.uint32)
    val = data[data[:, 0] > cut].astype(np.uint32)
    return train, val


def write_dataset(train, val, out_dir, labels_csv=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    train.tofile(out / "train.bin"); val.tofile(out / "val.bin")
    src = Path(labels_csv) if labels_csv else OFFICIAL / "data" / "ukb_real_data" / "labels.csv"
    (out / "labels.csv").write_text(src.read_text())
    return out
