#!/usr/bin/env python
"""Stage 1d: extraction for the exact Delphi-2M replication (executable module, BigQuery).

Writes to the GCS cache (rep_ prefix):
  rep_persons   person_id, birth_date, sex, has_ehr       (sex from sex_at_birth: Female/Male/other)
  rep_icd3      person_id, code3, date                    first record of EVERY ICD-10 level 3 code:
                ICD-10-CM source codes directly, other sources (ICD-9-CM, SNOMED) through the standard
                concept's 'Mapped from' ICD-10-CM codes (shortest code)
  rep_death     person_id, date
  rep_bmi       person_id, date, value                    earliest enrollment physical measurement BMI
  rep_smoking   person_id, date, question, answer         Smoke Frequency and 100 Cigs Lifetime
  rep_alcohol   person_id, date, question, answer         Drink Frequency Past Year and Alcohol Participant
Usage: RUN_MODE=test|full python scripts/13_extract_replication.py
"""
import os, subprocess, time, warnings
warnings.filterwarnings("ignore")
import pandas as pd, pandas_gbq
from google.cloud import bigquery as bq

RUN_MODE = os.environ.get("RUN_MODE", "test")
MOD = 20 if RUN_MODE == "test" else 1
CACHE = (os.environ.get("WORKSPACE_BUCKET") or "gs://YOUR-WORKSPACE-BUCKET") + "/notebooks/" + ("cache_v8_test" if RUN_MODE == "test" else "cache_v8")
CDR = os.environ.get("WORKSPACE_CDR") or "YOUR-CDR-PROJECT.R2025Q4R6"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
MAX_GB = float(os.environ.get("MAX_GB", "300"))
C = bq.Client(project=PROJECT)
S = lambda col: f"MOD(ABS(FARM_FINGERPRINT(CAST({col} AS STRING))), {MOD}) = 0"


def run(sql):
    j = C.query(sql, job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False))
    gb = j.total_bytes_processed / 1e9
    print(f"    [dry-run OK] would scan {gb:,.1f} GB", flush=True)
    if gb > MAX_GB:
        raise RuntimeError(f"query would scan {gb:,.0f} GB > MAX_GB={MAX_GB}")
    t = time.time()
    df = pandas_gbq.read_gbq(sql, dialect="standard", project_id=PROJECT, use_bqstorage_api=True, progress_bar_type=None)
    print(f"    [query done] {len(df):,} rows in {time.time()-t:,.0f}s", flush=True)
    return df


def save(df, name):
    local = f"/tmp/{name}.parquet"; df.to_parquet(local, index=False)
    r = subprocess.run(["gsutil", "-q", "cp", local, f"{CACHE}/{name}.parquet"])
    print(f"  {'OK' if r.returncode == 0 else 'FAILED'} {name}: {len(df):,} rows", flush=True)


print(f"RUN_MODE={RUN_MODE} CDR={CDR} cache={CACHE}", flush=True)

print("\n== persons ==", flush=True)
persons = run(f"""
WITH ehr AS (SELECT DISTINCT person_id FROM `{CDR}.condition_occurrence` WHERE {S('person_id')}
             UNION DISTINCT SELECT DISTINCT person_id FROM `{CDR}.visit_occurrence` WHERE {S('person_id')})
SELECT p.person_id, DATE(p.birth_datetime) AS birth_date, c.concept_name AS sex,
       e.person_id IS NOT NULL AS has_ehr
FROM `{CDR}.person` p
LEFT JOIN `{CDR}.concept` c ON c.concept_id = p.sex_at_birth_concept_id
LEFT JOIN ehr e ON e.person_id = p.person_id
WHERE {S('p.person_id')}""")
print(persons["sex"].value_counts().head(5).to_string(), "\nhas_ehr share", round(float(persons["has_ehr"].mean()), 3))
save(persons, "rep_persons")

print("\n== ICD-10 level 3 first records (all codes) ==", flush=True)
icd = run(f"""
WITH src AS (
  SELECT co.person_id, DATE(co.condition_start_date) AS d, UPPER(LEFT(REPLACE(cs.concept_code, '.', ''), 3)) AS code3
  FROM `{CDR}.condition_occurrence` co
  JOIN `{CDR}.concept` cs ON cs.concept_id = co.condition_source_concept_id
  WHERE cs.vocabulary_id = 'ICD10CM' AND co.condition_start_date IS NOT NULL AND {S('co.person_id')}
),
map AS (
  SELECT cr.concept_id_1 AS std_id, UPPER(LEFT(REPLACE(ci.concept_code, '.', ''), 3)) AS code3
  FROM `{CDR}.concept_relationship` cr
  JOIN `{CDR}.concept` ci ON ci.concept_id = cr.concept_id_2
  WHERE cr.relationship_id = 'Mapped from' AND ci.vocabulary_id = 'ICD10CM'
  QUALIFY ROW_NUMBER() OVER (PARTITION BY cr.concept_id_1 ORDER BY LENGTH(ci.concept_code), ci.concept_code) = 1
),
other AS (
  SELECT co.person_id, DATE(co.condition_start_date) AS d, m.code3
  FROM `{CDR}.condition_occurrence` co
  LEFT JOIN `{CDR}.concept` cs ON cs.concept_id = co.condition_source_concept_id
  JOIN map m ON m.std_id = co.condition_concept_id
  WHERE (cs.vocabulary_id IS NULL OR cs.vocabulary_id != 'ICD10CM') AND co.condition_start_date IS NOT NULL
    AND {S('co.person_id')}
)
SELECT person_id, code3, MIN(d) AS date FROM (SELECT * FROM src UNION ALL SELECT * FROM other)
WHERE REGEXP_CONTAINS(code3, r'^[A-Z][0-9][0-9A-Z]$')
GROUP BY person_id, code3""")
print("distinct level 3 codes:", icd["code3"].nunique())
save(icd, "rep_icd3")

print("\n== death ==", flush=True)
save(run(f"SELECT person_id, MIN(DATE(death_date)) AS date FROM `{CDR}.death` WHERE death_date IS NOT NULL AND {S('person_id')} GROUP BY person_id"), "rep_death")

print("\n== BMI (earliest of PM source concept 903124 or LOINC 39156-5 concept 3038553) ==", flush=True)
save(run(f"""
SELECT person_id, date, value FROM (
  SELECT person_id, DATE(measurement_date) AS date, value_as_number AS value,
         ROW_NUMBER() OVER (PARTITION BY person_id ORDER BY measurement_date) AS rn
  FROM `{CDR}.measurement`
  WHERE (measurement_source_concept_id = 903124 OR measurement_concept_id = 3038553) AND value_as_number BETWEEN 10 AND 100 AND {S('person_id')})
WHERE rn = 1"""), "rep_bmi")

for name, pats in [("rep_smoking", ["Smoking: Smoke Frequency%", "Smoking: 100 Cigs Lifetime%"]),
                   ("rep_alcohol", ["Alcohol: Drink Frequency Past Year%", "Alcohol: Alcohol Participant%"])]:
    print(f"\n== {name} ==", flush=True)
    cond = " OR ".join(f"q.concept_name LIKE '{p}'" for p in pats)
    df = run(f"""
    SELECT person_id, date, question, answer FROM (
      SELECT o.person_id, DATE(o.observation_date) AS date, q.concept_name AS question, a.concept_name AS answer,
             ROW_NUMBER() OVER (PARTITION BY o.person_id, q.concept_name ORDER BY o.observation_date) AS rn
      FROM `{CDR}.observation` o
      JOIN `{CDR}.concept` q ON q.concept_id = o.observation_source_concept_id
      LEFT JOIN `{CDR}.concept` a ON a.concept_id = o.value_source_concept_id
      WHERE ({cond}) AND {S('o.person_id')})
    WHERE rn = 1""")
    summ = df.groupby(["question", "answer"])["person_id"].nunique()
    print(summ[summ >= 21].to_string())
    save(df, name)

print("\nREPLICATION EXTRACT COMPLETE", flush=True)
