#!/usr/bin/env python
"""Lab-defined (silver standard) disease onset, for validating the Tier B latent onset against an outside signal.
Usage:  RUN_MODE=test python scripts/14_extract_labs.py   (5% of patients)
        RUN_MODE=full python scripts/14_extract_labs.py
Output (GCS cache, parquet) x_lab_onset: person_id, criterion, n_tests, first_test_age_days, first_abn_age_days
(first_abn_age_days is null if the criterion was never met). Ages are days since birth (person.birth_datetime).
Criteria (LOINC codes through the concept table, plausible value ranges only):
  t2d       HbA1c >= 6.5 % on two different days              (4548-4, 17856-6, 4549-2)
  ckd       eGFR (CKD-EPI 2021, no race term) < 60, confirmed by a second value < 60 at least 90 days later
            (creatinine 2160-0, 38483-4, mg/dL)
  anemia    hemoglobin < 12 g/dL (female) or < 13 g/dL (male), confirmed at least 90 days later
            (718-7; g/L values converted)
  htn       systolic BP >= 140 mmHg on two different days     (8480-6)
  ldl       LDL cholesterol >= 160 mg/dL                      (13457-7, 18262-6, 2089-1)
Only aggregates of this table leave the Workbench (counts above 20)."""
import time
from delphi_aou.data.extract.extras import run_bq, save, section, sample, CDR, T0  # noqa: F401

LABS = {
    "t2d": (["4548-4", "17856-6", "4549-2"], 3.0, 20.0),
    "ckd": (["2160-0", "38483-4"], 0.1, 20.0),
    "anemia": (["718-7"], 3.0, 250.0),                # g/dL, or g/L (values above 30, divided by 10)
    "htn": (["8480-6"], 60.0, 260.0),
    "ldl": (["13457-7", "18262-6", "2089-1"], 10.0, 500.0),
}


def sql_for(name):
    codes, lo, hi = LABS[name]
    code_list = ",".join(f"'{c}'" for c in codes)
    base = f"""
    WITH c AS (SELECT concept_id FROM `{CDR}.concept` WHERE vocabulary_id = 'LOINC' AND concept_code IN ({code_list})),
    p AS (SELECT person_id, DATE(birth_datetime) AS bd, gender_concept_id AS g FROM `{CDR}.person` WHERE {sample()}),
    m AS (
      SELECT m.person_id, DATE(m.measurement_date) AS d,
             AVG({"IF(m.value_as_number > 30, m.value_as_number / 10, m.value_as_number)" if name == "anemia" else "m.value_as_number"}) AS v
      FROM `{CDR}.measurement` m JOIN c ON m.measurement_concept_id = c.concept_id
      WHERE m.value_as_number BETWEEN {lo} AND {hi} AND m.measurement_date IS NOT NULL AND {sample('m.person_id')}
      GROUP BY 1, 2),
    x AS (SELECT m.person_id, m.d, m.v, DATE_DIFF(m.d, p.bd, DAY) AS age_days, p.g FROM m JOIN p USING (person_id))"""
    confirm = {"t2d": 1, "anemia": 90, "ldl": 0}       # days between two abnormal values (0: one value is enough)
    if name in confirm:
        abn = {"t2d": "v >= 6.5", "ldl": "v >= 160",
               "anemia": "((g = 8532 AND v < 12) OR (g != 8532 AND v < 13))"}[name]
        return base + f""",
    a AS (SELECT person_id, d, age_days, MAX(d) OVER (PARTITION BY person_id) AS last_abn FROM x WHERE {abn}),
    onset AS (SELECT person_id, MIN(age_days) AS first_abn_age_days FROM a
              WHERE DATE_DIFF(last_abn, d, DAY) >= {confirm[name]} GROUP BY person_id),
    tested AS (SELECT person_id, COUNT(*) AS n_tests, MIN(age_days) AS first_test_age_days FROM x GROUP BY person_id)
    SELECT t.person_id, '{name}' AS criterion, t.n_tests, t.first_test_age_days, o.first_abn_age_days
    FROM tested t LEFT JOIN onset o USING (person_id)"""
    if name == "ckd":
        # CKD-EPI 2021: 142 * min(S/k,1)^a * max(S/k,1)^-1.2 * 0.9938^age * 1.012 [female]
        egfr = ("142 * POW(LEAST(v / IF(g = 8532, 0.7, 0.9), 1), IF(g = 8532, -0.241, -0.302))"
                " * POW(GREATEST(v / IF(g = 8532, 0.7, 0.9), 1), -1.2) * POW(0.9938, age_days / 365.25)"
                " * IF(g = 8532, 1.012, 1.0)")
        return base + f""",
    e AS (SELECT person_id, d, age_days, {egfr} < 60 AS low FROM x),
    lowd AS (SELECT person_id, d, age_days, MAX(d) OVER (PARTITION BY person_id) AS last_low FROM e WHERE low),
    onset AS (SELECT person_id, MIN(age_days) AS first_abn_age_days FROM lowd
              WHERE DATE_DIFF(last_low, d, DAY) >= 90 GROUP BY person_id),
    tested AS (SELECT person_id, COUNT(*) AS n_tests, MIN(age_days) AS first_test_age_days FROM e GROUP BY person_id)
    SELECT t.person_id, 'ckd' AS criterion, t.n_tests, t.first_test_age_days, o.first_abn_age_days
    FROM tested t LEFT JOIN onset o USING (person_id)"""
    elif name == "htn":
        return base + f""",
    hi AS (SELECT person_id, d, age_days FROM x WHERE v >= 140),
    hi2 AS (SELECT person_id, age_days, LEAD(d) OVER (PARTITION BY person_id ORDER BY d) AS nd FROM hi),
    onset AS (SELECT person_id, MIN(age_days) AS first_abn_age_days FROM hi2 WHERE nd IS NOT NULL GROUP BY person_id),
    tested AS (SELECT person_id, COUNT(*) AS n_tests, MIN(age_days) AS first_test_age_days FROM x GROUP BY person_id)
    SELECT t.person_id, 'htn' AS criterion, t.n_tests, t.first_test_age_days, o.first_abn_age_days
    FROM tested t LEFT JOIN onset o USING (person_id)"""
    raise ValueError(name)


def main():
    import pandas as pd
    parts = []
    for name in LABS:
        def go(name=name):
            df = run_bq(sql_for(name))
            print(f"  {name}: persons tested {len(df):,}, meeting criterion {int(df['first_abn_age_days'].notna().sum()):,}")
            parts.append(df)
        section(f"lab onset: {name}", go)
    if parts:
        save(pd.concat(parts, ignore_index=True), "x_lab_onset")
    print(f"done in {time.time() - T0:,.0f}s")


if __name__ == "__main__":
    main()
