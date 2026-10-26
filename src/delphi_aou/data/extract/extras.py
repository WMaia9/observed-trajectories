#!/usr/bin/env python
"""Delphi-AoU 01b: extra data for the paper (runs after delphi_01_prepare.py).
Usage:  RUN_MODE=test python delphi_01b_extras.py     (5% of patients, ~minutes)
        RUN_MODE=full python delphi_01b_extras.py
Each section is independent: a failure in one is reported and the rest continue.
Outputs go to the same GCS cache folder as script 01 (parquet files).
"""
import os, sys, time, subprocess, traceback, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import pandas as pd
import pandas_gbq
from google.cloud import bigquery as bq

RUN_MODE = os.environ.get("RUN_MODE", "test")
MOD = 20 if RUN_MODE == "test" else 1
CACHE_VERSION = "cache_v8_test" if RUN_MODE == "test" else "cache_v8"
BUCKET = os.environ.get("WORKSPACE_BUCKET") or "gs://YOUR-WORKSPACE-BUCKET"
CACHE_DIR = f"{BUCKET}/notebooks/{CACHE_VERSION}"
CDR = os.environ.get("WORKSPACE_CDR") or "YOUR-CDR-PROJECT.R2025Q4R6"
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
CLIENT = bq.Client(project=PROJECT)
T0 = time.time()
MAX_GB = float(os.environ.get("MAX_GB", "300"))  # safety cap per query
print(f"RUN_MODE={RUN_MODE} CDR={CDR} cache={CACHE_DIR}", flush=True)

def sample(col="person_id"):
    return f"MOD(ABS(FARM_FINGERPRINT(CAST({col} AS STRING))), {MOD}) = 0"

def run_bq(sql):
    t = time.time()
    job = CLIENT.query(sql, job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False))
    gb = job.total_bytes_processed / 1e9
    print(f"    [dry-run OK] would scan {gb:,.1f} GB", flush=True)
    if gb > MAX_GB:
        raise RuntimeError(f"SKIPPED: query would scan {gb:,.0f} GB (> MAX_GB={MAX_GB})")
    df = pandas_gbq.read_gbq(sql, dialect="standard", project_id=PROJECT,
                             use_bqstorage_api=True, progress_bar_type=None)
    print(f"    [query done] {len(df):,} rows in {time.time()-t:,.0f}s", flush=True)
    return df

def save(df, name):
    local = f"/tmp/{name}.parquet"
    df.to_parquet(local, index=False)
    r = subprocess.run(["gsutil", "-q", "cp", local, f"{CACHE_DIR}/{name}.parquet"])
    mb = Path(local).stat().st_size / 1024**2
    print(f"  {'OK' if r.returncode == 0 else 'FAILED'} {name}: {len(df):,} rows, {mb:.1f} MB", flush=True)

def section(title, fn):
    print(f"\n===== {title} (t+{time.time()-T0:,.0f}s) =====", flush=True)
    try:
        fn()
    except Exception as e:
        print(f"  !! SECTION FAILED: {type(e).__name__}: {str(e)[:400]}", flush=True)
        traceback.print_exc(limit=2)

def s_inventory():
    df = run_bq(f"SELECT table_id, row_count FROM `{CDR}.__TABLES__` ORDER BY row_count DESC")
    print(df.to_string(index=False, max_rows=80))
    save(df, "x_table_inventory")

def s_observation_period():
    df = run_bq(f"""
        SELECT person_id,
               MIN(observation_period_start_date) AS obs_start,
               MAX(observation_period_end_date)   AS obs_end,
               COUNT(*) AS n_periods
        FROM `{CDR}.observation_period` WHERE {sample()} GROUP BY person_id""")
    save(df, "x_observation_period")

def s_visits():
    df = run_bq(f"""
        SELECT v.person_id, v.visit_start_date AS date, v.visit_concept_id,
               c.concept_name AS visit_type
        FROM `{CDR}.visit_occurrence` v
        LEFT JOIN `{CDR}.concept` c ON c.concept_id = v.visit_concept_id
        WHERE {sample('v.person_id')}""")
    print(df["visit_type"].value_counts().head(8).to_string())
    save(df, "x_visits")

def s_top_measurements():
    df = run_bq(f"""
        WITH m AS (
          SELECT person_id, measurement_date, measurement_concept_id, value_as_number
          FROM `{CDR}.measurement`
          WHERE value_as_number IS NOT NULL AND measurement_concept_id != 0 AND {sample()}),
        top AS (SELECT measurement_concept_id FROM m
                GROUP BY 1 ORDER BY COUNT(*) DESC LIMIT 25)
        SELECT m.person_id, m.measurement_date AS date,
               m.measurement_concept_id AS concept_id, AVG(m.value_as_number) AS value
        FROM m JOIN top USING (measurement_concept_id) GROUP BY 1, 2, 3""")
    names = run_bq(f"""
        SELECT concept_id, concept_name FROM `{CDR}.concept`
        WHERE concept_id IN ({",".join(str(int(x)) for x in df['concept_id'].unique())})""")
    print(names.to_string(index=False))
    save(df, "x_measurements_top25")
    save(names, "x_measurements_top25_names")

def s_wearables():
    tabs = run_bq(f"SELECT table_name FROM `{CDR}.INFORMATION_SCHEMA.TABLES`")["table_name"].tolist()
    cand = [t for t in tabs if any(k in t.lower() for k in
            ("activity_summary", "heart_rate_summary", "sleep_daily_summary", "wear"))
            and not t.lower().startswith("ds_") and "intraday" not in t.lower()
            and "level" not in t.lower()]
    print("wearable-like tables:", cand if cand else "none")
    for t in cand:
        cols = run_bq(f"SELECT column_name, data_type FROM `{CDR}.INFORMATION_SCHEMA.COLUMNS` "
                      f"WHERE table_name = '{t}'")
        print(f"  {t}: {cols['column_name'].tolist()}")
        if "person_id" not in cols["column_name"].tolist():
            continue
        datecol = next((c for c in cols["column_name"] if c in ("date", "sleep_date", "activity_date")), None)
        sel = f", MIN({datecol}) AS first_date, MAX({datecol}) AS last_date" if datecol else ""
        df = run_bq(f"SELECT person_id, COUNT(*) AS n_rows{sel} FROM `{CDR}.{t}` "
                    f"WHERE {sample()} GROUP BY person_id")
        save(df, f"x_wear_{t}")

def s_sdoh_discovery():
    df = run_bq(f"""
        SELECT c.concept_name, COUNT(DISTINCT o.person_id) AS n_persons
        FROM `{CDR}.observation` o
        JOIN `{CDR}.concept` c ON c.concept_id = o.observation_source_concept_id
        WHERE REGEXP_CONTAINS(LOWER(c.concept_name),
              r'food|housing|rent|evict|homeless|discriminat|utilit|insecur')
        GROUP BY 1 HAVING n_persons >= 20 ORDER BY 2 DESC LIMIT 60""")
    pd.set_option("display.max_colwidth", 110)
    print(df.to_string(index=False))
    save(df, "x_sdoh_candidates")

if __name__ == "__main__":
    section("Table inventory", s_inventory)
    section("Observation period (follow-up window)", s_observation_period)
    section("Visits (inpatient / outpatient / ER)", s_visits)
    section("Top 25 numeric measurements (labs and vitals)", s_top_measurements)
    section("Wearables availability", s_wearables)
    section("SDoH discovery (housing, food, discrimination)", s_sdoh_discovery)
    print(f"\nEXTRAS COMPLETE in {time.time()-T0:,.0f}s")
