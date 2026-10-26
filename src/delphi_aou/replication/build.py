"""Build the official-format Delphi dataset (train.bin, val.bin, labels.csv) from the rep_ extraction.
Smoking: Smoke Frequency answer; if missing and '100 Cigs Lifetime' = No, low. Alcohol: Drink Frequency
Past Year; if missing and 'Alcohol Participant' = No, low. Population: persons with EHR records and
sex at birth Female or Male (UK Biobank participants all have linked hospital records).
Usage: RUN_MODE=test|full python scripts/70_build_delphi_dataset.py  (writes ~/delphi_rep_data[_test])"""
import json, os, subprocess, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from delphi_aou.data import cohort as s2
from delphi_aou.replication import to_delphi_bin as tdb


def lifestyle(df, freq_q, ever_q, never_value):
    if df is None or len(df) == 0:
        return df
    df = df[~df["answer"].astype(str).str.startswith("PMI")]  # skip / don't know / prefer not: missing, not "never"
    f = df[df["question"].str.startswith(freq_q)][["person_id", "date", "answer"]]
    e = df[df["question"].str.startswith(ever_q) & df["answer"].astype(str).str.lower().str.endswith("no")]
    e = e[~e["person_id"].isin(f["person_id"])][["person_id", "date"]].assign(answer=never_value)
    return pd.concat([f, e], ignore_index=True)


def main():
    mode = os.environ.get("RUN_MODE", "test")
    out = Path(os.environ.get("OUT", str(Path.home() / ("delphi_rep_data" + ("_test" if mode == "test" else "")))))
    rd = lambda n: s2._read_local_from_cache(n)
    persons = rd("rep_persons"); persons = persons[persons["has_ehr"].astype(bool)]
    smoking = lifestyle(rd("rep_smoking"), "Smoking: Smoke Frequency", "Smoking: 100 Cigs Lifetime", "Not At All")
    alcohol = lifestyle(rd("rep_alcohol"), "Alcohol: Drink Frequency Past Year", "Alcohol: Alcohol Participant", "Never")
    rows, rep = tdb.build_rows(persons, rd("rep_icd3"), rd("rep_death"), rd("rep_bmi"), smoking, alcohol)
    tr, va = tdb.official_order_and_split(rows)
    tdb.write_dataset(tr, va, out)
    per = pd.Series(np.r_[tr[:, 0], va[:, 0]]).value_counts()
    tok = np.r_[tr[:, 2], va[:, 2]].astype(np.int64)
    rep.update({"persons_train": int(len(np.unique(tr[:, 0]))), "persons_val": int(len(np.unique(va[:, 0]))),
                "tokens_per_person_quantiles": {q: float(per.quantile(float(q))) for q in ("0.5", "0.9", "0.99")},
                "share_persons_with_bmi": float(np.isin(np.unique(tr[:, 0]), tr[np.isin(tr[:, 2], [3, 4, 5]), 0]).mean()),
                "share_persons_with_smoking": float(np.isin(np.unique(tr[:, 0]), tr[np.isin(tr[:, 2], [6, 7, 8]), 0]).mean()),
                "share_persons_with_alcohol": float(np.isin(np.unique(tr[:, 0]), tr[np.isin(tr[:, 2], [9, 10, 11]), 0]).mean()),
                "distinct_disease_tokens": int(len(np.unique(tok[tok >= 12]))),
                "death_tokens": int((tok == 1268).sum())})
    (out / "build_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)); print("DELPHI DATASET COMPLETE", out)


if __name__ == "__main__":
    main()
