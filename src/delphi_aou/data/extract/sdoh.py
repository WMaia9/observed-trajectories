#!/usr/bin/env python
"""Delphi-AoU 01c: SDoH survey answers (housing, food insecurity, discrimination).
Usage: RUN_MODE=test python delphi_01c_sdoh.py   |   RUN_MODE=full python delphi_01c_sdoh.py
Output: x_sdoh_survey.parquet (person_id, date, question, answer) in the GCS cache.
Tokenization (HOUSING=, FOOD=, DISCRIM=) is done downstream from these raw answers.
"""
import os, time, subprocess, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd, pandas_gbq
from google.cloud import bigquery as bq

RUN_MODE = os.environ.get("RUN_MODE", "test")
MOD = 20 if RUN_MODE == "test" else 1
CACHE_DIR = (os.environ.get("WORKSPACE_BUCKET") or "gs://YOUR-WORKSPACE-BUCKET") + "/notebooks/" + ("cache_v8_test" if RUN_MODE == "test" else "cache_v8")
CDR = os.environ.get("WORKSPACE_CDR") or "YOUR-CDR-PROJECT.R2025Q4R6"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
pd.set_option("display.max_colwidth", 90, "display.width", 200)

SQL = f"""
SELECT o.person_id, o.observation_date AS date,
       q.concept_name AS question,
       COALESCE(a.concept_name, o.value_as_string) AS answer
FROM `{CDR}.observation` o
JOIN `{CDR}.concept` q ON q.concept_id = o.observation_source_concept_id
LEFT JOIN `{CDR}.concept` a ON a.concept_id = o.value_source_concept_id
WHERE MOD(ABS(FARM_FINGERPRINT(CAST(o.person_id AS STRING))), {MOD}) = 0
  AND (q.concept_name LIKE 'Home Own%'
    OR q.concept_name LIKE 'What is the main type of housing%'
    OR q.concept_name LIKE 'Within the past 12 months, were you worried whether%'
    OR q.concept_name LIKE 'Discrimination:%')
"""
c = bq.Client(project=PROJECT)
job = c.query(SQL, job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False))
print(f"[dry-run OK] would scan {job.total_bytes_processed/1e9:,.1f} GB", flush=True)
assert job.total_bytes_processed < 300e9, "scan too large"
t = time.time()
df = pandas_gbq.read_gbq(SQL, dialect="standard", project_id=PROJECT, use_bqstorage_api=True, progress_bar_type=None)
print(f"[query done] {len(df):,} rows in {time.time()-t:,.0f}s", flush=True)
df["question"] = df["question"].str.strip().str.slice(0, 160)
summ = df.groupby(["question", "answer"], dropna=False)["person_id"].nunique().reset_index(name="n_persons")
summ = summ[summ["n_persons"] >= 20].sort_values(["question", "n_persons"], ascending=[True, False])
print(summ.to_string(index=False))
local = "/tmp/x_sdoh_survey.parquet"
df.to_parquet(local, index=False)
r = subprocess.run(["gsutil", "-q", "cp", local, f"{CACHE_DIR}/x_sdoh_survey.parquet"])
print("saved" if r.returncode == 0 else "SAVE FAILED", f"{len(df):,} rows", f"{Path(local).stat().st_size/1e6:.1f} MB")
