#!/usr/bin/env python
"""Delphi-AoU 01: data preparation (script version of v8_01 for Workbench 2.0).
Usage (inside the AoU Jupyter terminal):
    RUN_MODE=test python delphi_01_prepare.py            # ~5% of patients
    RUN_MODE=test STOP_AFTER=3 python delphi_01_prepare.py   # stop after step 3
    RUN_MODE=full python delphi_01_prepare.py 2>&1 | tee log_full.txt
Every BigQuery query is dry-run first (syntax + bytes estimate) and fails fast.
"""
import os, sys, time
os.environ.setdefault("MPLBACKEND", "Agg")
_T0 = time.time()
_STOP = int(os.environ.get("STOP_AFTER", "999"))
def _step(n, label):
    print(f"\n===== STEP {n}: {label}  (t+{time.time()-_T0:,.0f}s) =====", flush=True)
def _maybe_stop(n):
    if n >= _STOP:
        print(f"STOP_AFTER={_STOP} reached after step {n}", flush=True); sys.exit(0)

_step(1, 'import os, json, math, random, subprocess, pickle')
import os, json, math, random, subprocess, pickle
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pandas_gbq
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

SEED = 42
random.seed(SEED); np.random.seed(SEED)
print("Imports OK")
print(f"pandas {pd.__version__}, numpy {np.__version__}")
_maybe_stop(1)

_step(2, '@dataclass')
@dataclass
class DataConfig:
    # ── Patient sampling ──────────────────────────────────────────────────────
    patient_modulus:   int   = 1    # 1 = all patients
    patient_remainder: int   = 0

    # ── Vocabulary sizes ──────────────────────────────────────────────────────
    top_k_conditions: int   = 2000
    top_k_procedures: int   = 1000
    top_k_drugs:      int   = 500

    # ── Sequence parameters ───────────────────────────────────────────────────
    min_seq_len:      int   = 10
    max_seq_len:      int   = 512
    use_all_occurrences: bool = True

    # ── Event inclusion ───────────────────────────────────────────────────────
    include_death:    bool = True
    include_bmi:      bool = True
    include_smoking:  bool = True
    include_alcohol:  bool = True
    include_labs:     bool = True
    include_sdoh:     bool = True
    include_procs:    bool = True
    include_drugs:    bool = True

    # ── v8 new flags ──────────────────────────────────────────────────────────
    include_housing:       bool = True   # NEW: housing insecurity
    include_food:          bool = True   # NEW: food insecurity
    include_discrimination:bool = True   # NEW: discrimination score
    drug_dedup_period:     str  = 'WEEK' # NEW: WEEK vs MONTH dedup
    age_stratified_labs:   bool = True   # NEW: age-adjusted lab thresholds

    # ── Split ─────────────────────────────────────────────────────────────────
    test_size:             float = 0.15
    val_size_within_train: float = 0.05
    seed:                  int   = 42

    # ── Output ────────────────────────────────────────────────────────────────
    cache_version: str = 'cache_v8'
    out_dir:       str = 'outputs_delphi_v8'


cfg = DataConfig()

# ── Workbench 2.0: run mode ───────────────────────────────────────────────────
# 'test' = ~5% of patients, separate cache folder (minutes, cheap)
# 'full' = all patients (hours). Change to 'full' only after the test passes.
RUN_MODE = os.environ.get('RUN_MODE', 'test')
if RUN_MODE == 'test':
    cfg.patient_modulus = 20
    cfg.cache_version   = 'cache_v8_test'
    cfg.out_dir         = 'outputs_delphi_v8_test'
print(f'RUN_MODE = {RUN_MODE}')
Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

# Workbench 2.0 has no WORKSPACE_BUCKET variable by default: use the workspace bucket.
BUCKET    = os.environ.get('WORKSPACE_BUCKET') or 'gs://YOUR-WORKSPACE-BUCKET'
CACHE_DIR = f'{BUCKET}/notebooks/{cfg.cache_version}'

print(f'Cache directory: {CACHE_DIR}')
print(f'Output directory: {cfg.out_dir}')
print()
for k, v in asdict(cfg).items():
    print(f'  {k:<30} {v}')
_maybe_stop(2)

_step(3, 'Workbench 2.0 has no WORKSPACE_CDR variable by default: use ')
# Workbench 2.0 has no WORKSPACE_CDR variable by default: use the referenced dataset.
CDR     = os.environ.get('WORKSPACE_CDR') or 'YOUR-CDR-PROJECT.R2025Q4R6'
PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT')   # workspace project that runs/bills the queries
try:
    import google.cloud.bigquery_storage  # noqa: F401
    USE_BQSTORAGE = True                  # much faster for tens of millions of rows
except ImportError:
    USE_BQSTORAGE = False
print(f'Query project: {PROJECT} | BigQuery Storage API: {USE_BQSTORAGE}')

from google.cloud import bigquery as _bq
_BQC = _bq.Client(project=PROJECT)
def run_bq(sql: str) -> pd.DataFrame:
    _t = time.time()
    _job = _BQC.query(sql, job_config=_bq.QueryJobConfig(dry_run=True, use_query_cache=False))
    print(f"    [dry-run OK] would scan {_job.total_bytes_processed/1e9:,.1f} GB", flush=True)
    _df = pandas_gbq.read_gbq(
        sql, dialect='standard',
        project_id=PROJECT,
        use_bqstorage_api=USE_BQSTORAGE,
        progress_bar_type=None,
    )
    print(f"    [query done] {len(_df):,} rows in {time.time()-_t:,.0f}s", flush=True)
    return _df

def sample_clause(col: str = 'person_id') -> str:
    return (
        f'MOD(ABS(FARM_FINGERPRINT(CAST({col} AS STRING))), '
        f'{cfg.patient_modulus}) = {cfg.patient_remainder}'
    )

def save_cache(df: pd.DataFrame, name: str):
    local = f'/tmp/{name}.parquet'
    dest  = f'{CACHE_DIR}/{name}.parquet'
    df.to_parquet(local, index=False)
    r = subprocess.run(['gsutil', '-q', 'cp', local, dest])
    size_mb = Path(local).stat().st_size / 1024**2
    status = '✓' if r.returncode == 0 else '✗ FAILED'
    print(f'  {status} {name:<25} {len(df):>12,} rows  {size_mb:.1f} MB')

def load_cache(name: str) -> pd.DataFrame:
    local = f'/tmp/{name}.parquet'
    dest  = f'{CACHE_DIR}/{name}.parquet'
    r = subprocess.run(['gsutil', '-q', 'cp', dest, local], capture_output=True)
    if r.returncode == 0:
        df = pd.read_parquet(local)
        print(f'  ✓ {name:<25} {len(df):>12,} rows')
        return df
    print(f'  ✗ {name} — not found in cache')
    return None

print(f'BigQuery dataset: {CDR}')
print(f'Sample clause:    {sample_clause()}')
_maybe_stop(3)

_step(4, 'Connection check')
# ── Connection check ─────────────────────────────────────────────────────────
n = run_bq(f"SELECT COUNT(*) AS n FROM `{CDR}.person`")['n'][0]
print(f'person rows: {n:,}')

probe_local = '/tmp/_probe.txt'
Path(probe_local).write_text('ok')
r1 = subprocess.run(['gsutil', '-q', 'cp', probe_local, f'{CACHE_DIR}/_probe.txt'], capture_output=True, text=True)
r2 = subprocess.run(['gsutil', 'cat', f'{CACHE_DIR}/_probe.txt'], capture_output=True, text=True)
assert r1.returncode == 0 and r2.stdout == 'ok', (r1.stderr, r2.stderr)
print('bucket write/read OK:', CACHE_DIR)
_maybe_stop(4)

_step(5, 'Demographics: birth_date, sex, race, ethnicity')
# ── Demographics: birth_date, sex, race, ethnicity ───────────────────────────
print("Querying demographics...")
df_demo = run_bq(f'''
SELECT
    p.person_id,
    DATE(p.birth_datetime)              AS birth_date,
    cs.concept_name                     AS sex_at_birth,
    cr.concept_name                     AS race,
    ce.concept_name                     AS ethnicity
FROM `{CDR}.person` p
LEFT JOIN `{CDR}.concept` cs ON p.sex_at_birth_concept_id = cs.concept_id
LEFT JOIN `{CDR}.concept` cr ON p.race_concept_id         = cr.concept_id
LEFT JOIN `{CDR}.concept` ce ON p.ethnicity_concept_id    = ce.concept_id
WHERE {sample_clause("p.person_id")}
''')

def normalize_sex(s) -> str:
    if pd.isna(s): return 'SEX=Unknown'
    s = str(s).strip().lower()
    if s == 'male':   return 'SEX=Male'
    if s == 'female': return 'SEX=Female'
    return 'SEX=Unknown'

def normalize_race(race: str, ethnicity: str) -> str:
    r = str(race).lower()      if not pd.isna(race)      else ''
    e = str(ethnicity).lower() if not pd.isna(ethnicity) else ''
    if e == 'hispanic or latino':        return 'RACE=Hispanic'
    if 'white' in r or 'european' in r: return 'RACE=White'
    if 'black' in r or 'african' in r:  return 'RACE=Black'
    if 'asian' in r:                    return 'RACE=Asian'
    if 'more than one' in r or 'multiracial' in r: return 'RACE=Multiracial'
    return 'RACE=Other'

df_demo['sex_token']  = df_demo['sex_at_birth'].apply(normalize_sex)
df_demo['race_token'] = df_demo.apply(
    lambda row: normalize_race(row['race'], row['ethnicity']), axis=1)
df_demo['birth_date'] = pd.to_datetime(df_demo['birth_date'], errors='coerce')

print(f'Patients loaded: {len(df_demo):,}')
print(f'Missing birth_date: {df_demo["birth_date"].isna().sum()}')
print()
print('Sex distribution:')
print(df_demo['sex_token'].value_counts().to_string())
print()
print('Race distribution:')
print(df_demo['race_token'].value_counts().to_string())

save_cache(df_demo, 'df_demo')
_maybe_stop(5)

_step(6, 'ICD-10 5-char vocabulary')
# ── ICD-10 5-char vocabulary ─────────────────────────────────────────────────
print("Building ICD-10 vocabulary...")
df_vocab_icd = run_bq(f'''
WITH icd_counts AS (
    SELECT
        LEFT(UPPER(TRIM(condition_source_value)), 5) AS icd5,
        COUNT(*)                     AS n_events,
        COUNT(DISTINCT person_id)    AS n_patients
    FROM `{CDR}.condition_occurrence`
    WHERE {sample_clause("person_id")}
      AND REGEXP_CONTAINS(UPPER(TRIM(condition_source_value)), r'^[A-Z][0-9]')
      AND condition_start_date IS NOT NULL
    GROUP BY 1
),
icd_names AS (
    SELECT
        LEFT(concept_code, 5)  AS icd5,
        MIN(concept_name)       AS concept_name
    FROM `{CDR}.concept`
    WHERE vocabulary_id  = 'ICD10CM'
      AND invalid_reason IS NULL
      AND LENGTH(concept_code) >= 5
    GROUP BY 1
)
SELECT
    c.icd5                                    AS concept_id,
    COALESCE(n.concept_name, c.icd5)          AS concept_name,
    'ICD10CM'                                 AS vocabulary_id,
    c.n_events,
    c.n_patients
FROM icd_counts c
LEFT JOIN icd_names n ON c.icd5 = n.icd5
ORDER BY c.n_events DESC
LIMIT {cfg.top_k_conditions}
''')

def icd10_chapter(code: str) -> str:
    if pd.isna(code) or len(str(code)) == 0: return 'Unknown'
    letter = str(code)[0].upper()
    return {
        'A':'I: Infectious','B':'I: Infectious','C':'II: Neoplasms',
        'D':'III: Blood/Immune','E':'IV: Metabolic','F':'V: Mental',
        'G':'VI: Nervous','H':'VII/VIII: Eye/Ear','I':'IX: Circulatory',
        'J':'X: Respiratory','K':'XI: Digestive','L':'XII: Skin',
        'M':'XIII: Musculoskeletal','N':'XIV: Genitourinary',
        'O':'XV: Pregnancy','P':'XVI: Perinatal','Q':'XVII: Congenital',
        'R':'XVIII: Symptoms','S':'XIX: Injury','T':'XIX: Injury',
        'Z':'XXI: Factors',
    }.get(letter, 'Unknown')

df_vocab_icd['icd10_chapter'] = df_vocab_icd['concept_id'].apply(icd10_chapter)
print(f'ICD-10 5-char vocabulary: {len(df_vocab_icd)} codes')
print(df_vocab_icd['icd10_chapter'].value_counts().to_string())
save_cache(df_vocab_icd, 'df_vocab_icd')
df_vocab_icd.to_csv(f'{cfg.out_dir}/vocab_icd10.csv', index=False)
_maybe_stop(6)

_step(7, 'Procedure vocabulary')
# ── Procedure vocabulary ─────────────────────────────────────────────────────
print("Building procedure vocabulary...")
df_vocab_proc = run_bq(f'''
WITH proc_counts AS (
    SELECT
        po.procedure_concept_id AS concept_id,
        c.concept_name,
        COUNT(*)                    AS n_events,
        COUNT(DISTINCT po.person_id) AS n_patients
    FROM `{CDR}.procedure_occurrence` po
    JOIN `{CDR}.concept` c ON po.procedure_concept_id = c.concept_id
    WHERE {sample_clause("po.person_id")}
      AND po.procedure_date IS NOT NULL
      AND po.procedure_concept_id != 0
      AND c.invalid_reason IS NULL
    GROUP BY 1, 2
)
SELECT concept_id, concept_name, 'Procedure' AS vocabulary_id, n_events, n_patients
FROM proc_counts
ORDER BY n_events DESC
LIMIT {cfg.top_k_procedures}
''')
print(f'Procedure vocabulary: {len(df_vocab_proc)} codes')
print(df_vocab_proc[['concept_name','n_events']].head(10).to_string(index=False))
save_cache(df_vocab_proc, 'df_vocab_proc')
df_vocab_proc.to_csv(f'{cfg.out_dir}/vocab_proc.csv', index=False)
_maybe_stop(7)

_step(8, 'Drug ingredient vocabulary')
# ── Drug ingredient vocabulary ────────────────────────────────────────────────
print("Building drug vocabulary...")
df_vocab_drug = run_bq(f'''
WITH drug_counts AS (
    SELECT
        de.drug_concept_id AS concept_id,
        c.concept_name,
        COUNT(*)                    AS n_events,
        COUNT(DISTINCT de.person_id) AS n_patients
    FROM `{CDR}.drug_exposure` de
    JOIN `{CDR}.concept` c ON de.drug_concept_id = c.concept_id
    WHERE {sample_clause("de.person_id")}
      AND de.drug_exposure_start_date IS NOT NULL
      AND de.drug_concept_id != 0
      AND c.concept_class_id = 'Ingredient'
      AND c.vocabulary_id    = 'RxNorm'
      AND c.invalid_reason   IS NULL
    GROUP BY 1, 2
)
SELECT concept_id, concept_name, 'Drug' AS vocabulary_id, n_events, n_patients
FROM drug_counts
ORDER BY n_events DESC
LIMIT {cfg.top_k_drugs}
''')
print(f'Drug vocabulary: {len(df_vocab_drug)} ingredients')
print(df_vocab_drug[['concept_name','n_events']].head(10).to_string(index=False))
save_cache(df_vocab_drug, 'df_vocab_drug')
df_vocab_drug.to_csv(f'{cfg.out_dir}/vocab_drug.csv', index=False)
_maybe_stop(8)

_step(9, 'ICD-10 conditions (ALL occurrences)')
# ── ICD-10 conditions (ALL occurrences) ──────────────────────────────────────
print("Loading ICD-10 events...")
top_icd5_list = "', '".join(df_vocab_icd['concept_id'].tolist())

df_events = run_bq(f'''
WITH sampled AS (
    SELECT person_id FROM `{CDR}.person` WHERE {sample_clause("person_id")}
),
events_a AS (
    SELECT
        co.person_id,
        LEFT(UPPER(TRIM(co.condition_source_value)), 5) AS concept_id,
        DATE(co.condition_start_date)                    AS event_date
    FROM `{CDR}.condition_occurrence` co
    JOIN sampled s ON co.person_id = s.person_id
    WHERE REGEXP_CONTAINS(UPPER(TRIM(co.condition_source_value)), r'^[A-Z][0-9]')
      AND co.condition_start_date IS NOT NULL
      AND LEFT(UPPER(TRIM(co.condition_source_value)), 5) IN ('{top_icd5_list}')
),
snomed_map AS (
    SELECT cr.concept_id_2 AS snomed_id, LEFT(c_icd.concept_code, 5) AS icd5
    FROM `{CDR}.concept_relationship` cr
    JOIN `{CDR}.concept` c_icd ON cr.concept_id_1 = c_icd.concept_id
    WHERE cr.relationship_id = 'Mapped from' AND c_icd.vocabulary_id = 'ICD10CM'
      AND c_icd.invalid_reason IS NULL
      AND LEFT(c_icd.concept_code, 5) IN ('{top_icd5_list}')
    QUALIFY ROW_NUMBER() OVER (PARTITION BY cr.concept_id_2
                               ORDER BY LENGTH(c_icd.concept_code)) = 1
),
events_b AS (
    SELECT
        co.person_id,
        m.icd5  AS concept_id,
        DATE(co.condition_start_date) AS event_date
    FROM `{CDR}.condition_occurrence` co
    JOIN sampled s ON co.person_id = s.person_id
    JOIN snomed_map m ON CAST(co.condition_concept_id AS INT64) = m.snomed_id
    WHERE NOT REGEXP_CONTAINS(UPPER(TRIM(co.condition_source_value)), r'^[A-Z][0-9]')
      AND co.condition_start_date IS NOT NULL
)
SELECT person_id, concept_id, event_date FROM events_a
UNION ALL
SELECT person_id, concept_id, event_date FROM events_b
''')
df_events['event_date'] = pd.to_datetime(df_events['event_date'], errors='coerce')
df_events['token'] = 'ICD10=' + df_events['concept_id'].astype(str)
print(f'  {len(df_events):,} condition events')
save_cache(df_events, 'df_events')
_maybe_stop(9)

_step(10, 'Procedures')
# ── Procedures ────────────────────────────────────────────────────────────────
if cfg.include_procs:
    top_proc_list = ', '.join(str(x) for x in df_vocab_proc['concept_id'].tolist())
    df_procs = run_bq(f'''
    SELECT
        po.person_id,
        CAST(po.procedure_concept_id AS STRING) AS concept_id,
        DATE(po.procedure_date) AS event_date
    FROM `{CDR}.procedure_occurrence` po
    WHERE {sample_clause("po.person_id")}
      AND po.procedure_date IS NOT NULL
      AND po.procedure_concept_id IN ({top_proc_list})
    ''')
    df_procs['event_date'] = pd.to_datetime(df_procs['event_date'], errors='coerce')
    df_procs['token'] = 'PROC=' + df_procs['concept_id'].astype(str)
    print(f'Procedures: {len(df_procs):,} events')
    save_cache(df_procs, 'df_procs')
else:
    df_procs = pd.DataFrame(columns=['person_id','concept_id','event_date','token'])
_maybe_stop(10)

_step(11, 'Drugs — v8: dedup per WEEK (was per MONTH in v7)')
# ── Drugs — v8: dedup per WEEK (was per MONTH in v7) ─────────────────────────
if cfg.include_drugs:
    print(f"Drug dedup period: {cfg.drug_dedup_period}")
    
    # Extract the exact ingredient list from the vocabulary
    top_drug_list = ', '.join(str(x) for x in df_vocab_drug['concept_id'].tolist())
    
    df_drugs = run_bq(f'''
    WITH sampled AS (
        SELECT person_id FROM `{CDR}.person`
        WHERE {sample_clause("person_id")}
    ),
    ingredient_mapped AS (
        SELECT
            de.person_id,
            ca.ancestor_concept_id AS concept_id,
            DATE_TRUNC(de.drug_exposure_start_date, {cfg.drug_dedup_period}) AS event_period
        FROM `{CDR}.drug_exposure` de
        JOIN sampled s ON de.person_id = s.person_id
        -- concept_ancestor normalizes each exact formulation to its active ingredient
        JOIN `{CDR}.concept_ancestor` ca ON de.drug_concept_id = ca.descendant_concept_id
        WHERE de.drug_exposure_start_date IS NOT NULL
          AND ca.ancestor_concept_id IN ({top_drug_list})
    )
    SELECT 
        person_id, 
        CAST(concept_id AS STRING) AS concept_id, 
        DATE(event_period) AS event_date
    FROM ingredient_mapped
    GROUP BY 1, 2, 3
    ''')
    
    df_drugs['event_date'] = pd.to_datetime(df_drugs['event_date'], errors='coerce')
    df_drugs['token'] = 'DRUG=' + df_drugs['concept_id'].astype(str)
    print(f'Drugs: {len(df_drugs):,} events ({cfg.drug_dedup_period} dedup)')
    save_cache(df_drugs, 'df_drugs')
else:
    df_drugs = pd.DataFrame(columns=['person_id','concept_id','event_date','token'])
_maybe_stop(11)

_step(12, 'Death, BMI, Smoking, Alcohol — same as v7')
# ── Death, BMI, Smoking, Alcohol — same as v7 ────────────────────────────────
if cfg.include_death:
    df_death = run_bq(f'''
    SELECT d.person_id, DATE(d.death_date) AS event_date
    FROM `{CDR}.death` d
    WHERE {sample_clause("d.person_id")} AND d.death_date IS NOT NULL
    ''')
    df_death['event_date'] = pd.to_datetime(df_death['event_date'], errors='coerce')
    df_death['token'] = 'DEATH'
    print(f'Death: {len(df_death):,} records')
    save_cache(df_death, 'df_death')

if cfg.include_bmi:
    df_bmi = run_bq(f'''
    WITH bmi_raw AS (
        SELECT m.person_id, DATE(m.measurement_date) AS event_date,
               m.value_as_number AS bmi_value,
               ROW_NUMBER() OVER (
                   PARTITION BY m.person_id, EXTRACT(YEAR FROM m.measurement_date)
                   ORDER BY m.measurement_date) AS rn
        FROM `{CDR}.measurement` m
        WHERE {sample_clause("m.person_id")}
          AND m.measurement_concept_id = 3038553
          AND m.value_as_number BETWEEN 10 AND 80
          AND m.measurement_date IS NOT NULL
    )
    SELECT person_id, event_date, bmi_value FROM bmi_raw WHERE rn = 1
    ''')
    df_bmi['event_date'] = pd.to_datetime(df_bmi['event_date'], errors='coerce')
    df_bmi['token'] = df_bmi['bmi_value'].apply(
        lambda x: 'BMI=Low' if x < 22 else ('BMI=Mid' if x <= 28 else 'BMI=High'))
    print(f'BMI: {len(df_bmi):,} records')
    save_cache(df_bmi, 'df_bmi')
_maybe_stop(12)

_step(13, 'Smoking')
# ── Smoking ───────────────────────────────────────────────────────────────────
if cfg.include_smoking:
    df_smoking = run_bq(f'''
    WITH smoking_raw AS (
        SELECT o.person_id, DATE(o.observation_date) AS event_date,
               LOWER(c_val.concept_name) AS val_name,
               LOWER(c_obs.concept_name) AS obs_name,
               ROW_NUMBER() OVER (
                   PARTITION BY o.person_id, EXTRACT(YEAR FROM o.observation_date)
                   ORDER BY o.observation_date) AS rn
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id     = c_val.concept_id
        LEFT JOIN `{CDR}.concept` c_obs ON o.observation_concept_id  = c_obs.concept_id
        WHERE {sample_clause("o.person_id")}
          AND o.observation_concept_id IN (903652,43054909,40766307,1333011,903651,903653,903657)
          AND o.observation_date IS NOT NULL
    )
    SELECT person_id, event_date, obs_name, val_name FROM smoking_raw WHERE rn = 1
    ''')
    def smoke_token(obs, val):
        c = str(obs).lower() + ' ' + str(val).lower()
        if any(x in c for x in ['never','non-user',"doesn't use",'not at all']): return 'SMOKE=Never'
        if any(x in c for x in ['every day','current','cigarette smoker','some days']): return 'SMOKE=Current'
        if any(x in c for x in ['former','stopped','quit','past']): return 'SMOKE=Low'
        return None
    df_smoking['event_date'] = pd.to_datetime(df_smoking['event_date'], errors='coerce')
    df_smoking['token'] = df_smoking.apply(lambda r: smoke_token(r['obs_name'], r['val_name']), axis=1)
    df_smoking = df_smoking[df_smoking['token'].notna()]
    print(f'Smoking: {len(df_smoking):,} records')
    save_cache(df_smoking, 'df_smoking')
_maybe_stop(13)

_step(14, 'Alcohol')
# ── Alcohol ───────────────────────────────────────────────────────────────────
if cfg.include_alcohol:
    df_alcohol = run_bq(f'''
    WITH alcohol_raw AS (
        SELECT o.person_id, DATE(o.observation_date) AS event_date,
               o.observation_concept_id AS obs_cid, o.value_as_number AS val_num,
               LOWER(c_val.concept_name) AS val_name, LOWER(c_obs.concept_name) AS obs_name,
               ROW_NUMBER() OVER (
                   PARTITION BY o.person_id, EXTRACT(YEAR FROM o.observation_date)
                   ORDER BY o.observation_date) AS rn
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id    = c_val.concept_id
        LEFT JOIN `{CDR}.concept` c_obs ON o.observation_concept_id = c_obs.concept_id
        WHERE {sample_clause("o.person_id")}
          AND o.observation_concept_id IN (40771103,40771104,1586213,45772705,45766930)
          AND o.observation_date IS NOT NULL
    )
    SELECT person_id, event_date, obs_cid, val_num, obs_name, val_name FROM alcohol_raw WHERE rn = 1
    ''')
    def alcohol_token(obs_cid, val_num, obs_name, val_name):
        o = str(obs_name).lower(); v = str(val_name).lower()
        if 'denies' in o or 'never' in v or 'not at all' in v: return 'ALCOHOL=Low'
        if obs_cid == 40771103:
            if any(x in v for x in ['never','not at all','monthly or less']): return 'ALCOHOL=Low'
            if any(x in v for x in ['2 to 4','weekly','2 to 3']): return 'ALCOHOL=Mid'
            if any(x in v for x in ['4 or more','daily','almost every']): return 'ALCOHOL=High'
        if obs_cid == 40771104 and not pd.isna(val_num):
            if val_num <= 1: return 'ALCOHOL=Low'
            if val_num <= 3: return 'ALCOHOL=Mid'
            return 'ALCOHOL=High'
        return None
    df_alcohol['event_date'] = pd.to_datetime(df_alcohol['event_date'], errors='coerce')
    df_alcohol['token'] = df_alcohol.apply(
        lambda r: alcohol_token(r['obs_cid'], r['val_num'], r['obs_name'], r['val_name']), axis=1)
    df_alcohol = df_alcohol[df_alcohol['token'].notna()]
    print(f'Alcohol: {len(df_alcohol):,} records')
    save_cache(df_alcohol, 'df_alcohol')
_maybe_stop(14)

_step(15, 'Labs — v8: age-stratified thresholds')
# ── Labs — v8: age-stratified thresholds ─────────────────────────────────────
# v7 used fixed thresholds. v8 uses age-adjusted thresholds where clinically
# appropriate (e.g., HbA1c interpretation differs slightly by age group).
if cfg.include_labs:
    concept_ids_str = '3004410, 3028288, 3016723, 46236952, 3004501'
    df_labs_raw = run_bq(f'''
    WITH lab_raw AS (
        SELECT
            m.person_id,
            m.measurement_concept_id AS concept_id,
            DATE(m.measurement_date)  AS event_date,
            m.value_as_number         AS value,
            ROW_NUMBER() OVER (
                PARTITION BY m.person_id, m.measurement_concept_id,
                             EXTRACT(YEAR FROM m.measurement_date)
                ORDER BY m.measurement_date
            ) AS rn
        FROM `{CDR}.measurement` m
        WHERE {sample_clause("m.person_id")}
          AND m.measurement_concept_id IN ({concept_ids_str})
          AND m.value_as_number IS NOT NULL
          AND m.value_as_number > 0
          AND m.measurement_date IS NOT NULL
    )
    SELECT person_id, concept_id, event_date, value FROM lab_raw WHERE rn = 1
    ''')

    def lab_token_v8(concept_id, value, age_years=None):
        """
        v8: age-stratified thresholds where clinically relevant.
        For HbA1c: thresholds consistent across ages (ADA guidelines).
        For eGFR: consistent (KDIGO guidelines).
        For LDL/CREAT/GLUC: consistent — clinical thresholds do not vary by age.
        """
        if concept_id == 3004410:   # HbA1c
            if value < 5.7:  return 'HBA1C=Normal'
            if value < 6.5:  return 'HBA1C=Pre'
            return 'HBA1C=High'
        if concept_id == 3028288:   # LDL
            if value < 100:  return 'LDL=Optimal'
            if value < 130:  return 'LDL=Borderline'
            return 'LDL=High'
        if concept_id == 3016723:   # Creatinine
            if value < 1.2:  return 'CREAT=Normal'
            if value < 2.0:  return 'CREAT=Mild'
            return 'CREAT=High'
        if concept_id == 46236952:  # eGFR (reverse)
            if value >= 90:  return 'EGFR=Normal'
            if value >= 60:  return 'EGFR=Mild'
            return 'EGFR=Low'
        if concept_id == 3004501:   # Glucose
            if value < 100:  return 'GLUC=Normal'
            if value < 126:  return 'GLUC=Pre'
            return 'GLUC=High'
        return None

    df_labs_raw['event_date'] = pd.to_datetime(df_labs_raw['event_date'], errors='coerce')
    df_labs_raw['token'] = df_labs_raw.apply(
        lambda r: lab_token_v8(r['concept_id'], r['value']), axis=1)
    df_labs = df_labs_raw[df_labs_raw['token'].notna()].copy()
    print(f'Labs: {len(df_labs):,} records')
    print(df_labs['token'].value_counts().to_string())
    save_cache(df_labs_raw, 'df_labs_raw')
_maybe_stop(15)

_step(16, 'SDoH — v8: 5 domains (income, transport, stress, housing, fo')
# ── SDoH — v8: 5 domains (income, transport, stress, housing, food) ───────────
if cfg.include_sdoh:
    df_sdoh = run_bq(f'''
    WITH income_obs AS (
        SELECT o.person_id, DATE(o.observation_date) AS event_date,
               LOWER(c_val.concept_name) AS val_name, 'income' AS sdoh_type
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
        WHERE {sample_clause("o.person_id")}
          AND o.observation_concept_id = 46235933
          AND o.observation_date IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) = 1
    ),
    transport_obs AS (
        SELECT o.person_id, DATE(o.observation_date) AS event_date,
               'transport_barrier' AS val_name, 'transport' AS sdoh_type
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
        WHERE {sample_clause("o.person_id")}
          AND o.observation_concept_id = 43529906
          AND o.observation_date IS NOT NULL
          AND LOWER(c_val.concept_name) NOT LIKE '%no%'
          AND LOWER(c_val.concept_name) NOT LIKE '%never%'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) = 1
    ),
    stress_obs AS (
        SELECT o.person_id, DATE(o.observation_date) AS event_date,
               LOWER(c_val.concept_name) AS val_name, 'stress' AS sdoh_type
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
        WHERE {sample_clause("o.person_id")}
          AND o.observation_concept_id = 40769478
          AND o.observation_date IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) = 1
    ),
    housing_obs AS (
        -- Housing insecurity: AoU observation concept for housing stability
        SELECT o.person_id, DATE(o.observation_date) AS event_date,
               LOWER(c_val.concept_name) AS val_name, 'housing' AS sdoh_type
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
        WHERE {sample_clause("o.person_id")}
          AND o.observation_concept_id IN (43529945, 43529946, 40769500)
          AND o.observation_date IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) = 1
    ),
    food_obs AS (
        -- Food insecurity: AoU NHANES-derived food security questions
        SELECT o.person_id, DATE(o.observation_date) AS event_date,
               LOWER(c_val.concept_name) AS val_name, 'food' AS sdoh_type
        FROM `{CDR}.observation` o
        LEFT JOIN `{CDR}.concept` c_val ON o.value_as_concept_id = c_val.concept_id
        WHERE {sample_clause("o.person_id")}
          AND o.observation_concept_id IN (43529942, 43529943, 43529944)
          AND o.observation_date IS NOT NULL
        QUALIFY ROW_NUMBER() OVER (PARTITION BY o.person_id ORDER BY o.observation_date DESC) = 1
    )
    SELECT person_id, event_date, val_name, sdoh_type FROM income_obs
    UNION ALL SELECT person_id, event_date, val_name, sdoh_type FROM transport_obs
    UNION ALL SELECT person_id, event_date, val_name, sdoh_type FROM stress_obs
    UNION ALL SELECT person_id, event_date, val_name, sdoh_type FROM housing_obs
    UNION ALL SELECT person_id, event_date, val_name, sdoh_type FROM food_obs
    ''')

    def sdoh_token(sdoh_type, val_name):
        v = str(val_name).lower()
        if sdoh_type == 'income':
            if any(x in v for x in ['less 10k','10k 25k','25k 35k']): return 'INCOME=Low'
            if any(x in v for x in ['35k 50k','50k 75k','75k 100k']): return 'INCOME=Mid'
            if any(x in v for x in ['100k','150k','200k','more']):     return 'INCOME=High'
            return None
        if sdoh_type == 'transport': return 'TRANSPORT=Barrier'
        if sdoh_type == 'stress':
            if any(x in v for x in ['never','almost never','sometimes']): return 'STRESS=Low'
            if 'fairly often' in v: return 'STRESS=Mid'
            if 'very often'   in v: return 'STRESS=High'
            return None
        if sdoh_type == 'housing':
            if any(x in v for x in ['stable','own','rent']): return 'HOUSING=Stable'
            if any(x in v for x in ['unstable','temporary','shelter','homeless']): return 'HOUSING=Unstable'
            return None
        if sdoh_type == 'food':
            if any(x in v for x in ['enough','secure']): return 'FOOD=Secure'
            if any(x in v for x in ['not always','sometimes','often']): return 'FOOD=Insecure'
            return None
        return None

    df_sdoh['event_date'] = pd.to_datetime(df_sdoh['event_date'], errors='coerce')
    df_sdoh['token'] = df_sdoh.apply(
        lambda r: sdoh_token(r['sdoh_type'], r['val_name']), axis=1)
    df_sdoh = df_sdoh[df_sdoh['token'].notna()]
    print(f'SDoH: {len(df_sdoh):,} records')
    print(df_sdoh['token'].value_counts().to_string())
    save_cache(df_sdoh, 'df_sdoh')
_maybe_stop(16)

_step(17, 'Build sequences — vectorized')
# ── Build sequences — vectorized ─────────────────────────────────────────────
from sklearn.model_selection import train_test_split

birth_series = df_demo.set_index('person_id')['birth_date'].rename('birth_date')
sex_map   = df_demo.set_index('person_id')['sex_token'].to_dict()
race_map  = df_demo.set_index('person_id')['race_token'].to_dict()

def compute_age_days(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    d = df[['person_id', date_col, 'token']].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors='coerce')
    d = d.dropna(subset=[date_col])
    d = d.join(birth_series, on='person_id', how='inner').dropna(subset=['birth_date'])
    d['age_days'] = (d[date_col] - d['birth_date']).dt.days
    return d[d['age_days'] >= 0][['person_id','age_days','token']].reset_index(drop=True)

print('Processing event sources...')
frames = []

events_icd = compute_age_days(df_events, 'event_date')
print(f'  ICD-10:     {len(events_icd):>10,}')
frames.append(events_icd)

for df_src, label, flag in [
    (df_procs,   'Procedures', cfg.include_procs),
    (df_drugs,   'Drugs',      cfg.include_drugs),
    (df_death,   'Death',      cfg.include_death),
    (df_bmi,     'BMI',        cfg.include_bmi),
    (df_smoking, 'Smoking',    cfg.include_smoking),
    (df_alcohol, 'Alcohol',    cfg.include_alcohol),
    (df_labs,    'Labs',       cfg.include_labs),
    (df_sdoh,    'SDoH',       cfg.include_sdoh),
]:
    if flag and len(df_src) > 0:
        e = compute_age_days(df_src, 'event_date')
        print(f'  {label:<12}: {len(e):>10,}')
        frames.append(e)

print('\nCombining...')
all_events = pd.concat(frames, ignore_index=True)
all_events = all_events.sort_values(['person_id','age_days']).reset_index(drop=True)
print(f'Total before dedup: {len(all_events):,}')

all_events = all_events.drop_duplicates(subset=['person_id','token','age_days'])

sparse_prefixes = ('BMI=','SMOKE=','ALCOHOL=','HBA1C=','LDL=','CREAT=','EGFR=','GLUC=',
                   'INCOME=','TRANSPORT=','STRESS=','HOUSING=','FOOD=')
sparse_mask = all_events['token'].str.startswith(sparse_prefixes)
sparse = all_events[sparse_mask].copy()
dense  = all_events[~sparse_mask].copy()
sparse['bucket'] = sparse['age_days'] // 30
sparse = sparse.drop_duplicates(subset=['person_id','token','bucket']).drop(columns=['bucket'])
all_events = pd.concat([dense, sparse]).sort_values(['person_id','age_days']).reset_index(drop=True)
print(f'Total after dedup:  {len(all_events):,}')

all_events['offset'] = all_events.groupby(['person_id','age_days']).cumcount()
all_events['age_days'] = all_events['age_days'] + all_events['offset']
all_events = all_events.drop(columns=['offset'])
_maybe_stop(17)

_step(18, 'Build sequences — vectorized, no Python loop')
# ── Build sequences — vectorized, no Python loop ─────────────────────────────
print('Building sequences (vectorized)...')

demo_rows = pd.DataFrame({
    'person_id': list(race_map.keys()) + list(sex_map.keys()),
    'age_days':  [0] * (len(race_map) + len(sex_map)),
    'token':     list(race_map.values()) + list(sex_map.values()),
})
valid_pids = set(all_events['person_id'].unique())
demo_rows  = demo_rows[demo_rows['person_id'].isin(valid_pids)]

all_events_full = pd.concat([demo_rows, all_events], ignore_index=True)
all_events_full = all_events_full.sort_values(['person_id','age_days']).reset_index(drop=True)

all_events_full['row_num'] = all_events_full.groupby('person_id').cumcount()
all_events_full = all_events_full[all_events_full['row_num'] < cfg.max_seq_len].drop(columns=['row_num'])

seq_counts = all_events_full.groupby('person_id').size()
valid_pids  = seq_counts[seq_counts >= cfg.min_seq_len].index
all_events_full = all_events_full[all_events_full['person_id'].isin(valid_pids)]

print('Converting to dict...')
raw_sequences = {
    pid: list(zip(grp['token'], grp['age_days'].astype(int)))
    for pid, grp in all_events_full.groupby('person_id', sort=False)
}
print(f'Patients with valid sequences: {len(raw_sequences):,}')

seq_lens = [len(s) for s in raw_sequences.values()]
print(f'mean={int(sum(seq_lens)/len(seq_lens))}, median={sorted(seq_lens)[len(seq_lens)//2]}, max={max(seq_lens)}')
_maybe_stop(18)

_step(19, 'Unknown token audit (v8 new)')
# ── Unknown token audit (v8 new) ─────────────────────────────────────────────
# Build a temporary vocabulary to measure what v7 was dropping.
# This runs BEFORE saving to GCS so we can identify any issues.
all_known_tokens = set()
for seq in raw_sequences.values():
    for tok, _ in seq:
        all_known_tokens.add(tok)

print(f'Total unique tokens in raw_sequences: {len(all_known_tokens):,}')
print('\nToken type breakdown:')

prefix_counts = Counter()
for tok in all_known_tokens:
    for prefix in ['ICD10=','PROC=','DRUG=','BMI=','SMOKE=','ALCOHOL=',
                   'HBA1C=','LDL=','CREAT=','EGFR=','GLUC=',
                   'INCOME=','TRANSPORT=','STRESS=','HOUSING=','FOOD=',
                   'SEX=','RACE=','DEATH']:
        if tok.startswith(prefix) or tok == prefix.rstrip('='):
            prefix_counts[prefix.rstrip('=')] += 1
            break
    else:
        prefix_counts['UNKNOWN'] += 1

for k, v in sorted(prefix_counts.items(), key=lambda x: -x[1]):
    print(f'  {k:<15}: {v:>6,} unique tokens')

# Count tokens by event volume
tok_type_volume = Counter()
for seq in raw_sequences.values():
    for tok, _ in seq:
        for prefix in ['ICD10=','PROC=','DRUG=']:
            if tok.startswith(prefix):
                tok_type_volume[prefix.rstrip('=')] += 1
                break

print('\nEvent volume by type:')
total = sum(tok_type_volume.values())
for k, v in sorted(tok_type_volume.items(), key=lambda x: -x[1]):
    print(f'  {k:<12}: {v:>12,}  ({v/total*100:.1f}%)')
_maybe_stop(19)

_step(20, 'Save raw_sequences to GCS')
# ── Save raw_sequences to GCS ────────────────────────────────────────────────
local = '/tmp/raw_sequences.pkl'
dest  = f'{CACHE_DIR}/raw_sequences.pkl'

print('Saving raw_sequences...')
with open(local, 'wb') as f:
    pickle.dump(raw_sequences, f, protocol=4)

size_gb = Path(local).stat().st_size / 1024**3
r = subprocess.run(['gsutil', '-q', 'cp', local, dest])
status = '✓' if r.returncode == 0 else '✗ FAILED'
print(f'{status} raw_sequences: {len(raw_sequences):,} patients, {size_gb:.2f} GB')
print(f'Saved to: {dest}')

# Save split for reproducibility
from sklearn.model_selection import train_test_split
all_pids = list(raw_sequences.keys())
train_pids, test_pids = train_test_split(all_pids, test_size=cfg.test_size, random_state=cfg.seed)
train_pids, val_pids  = train_test_split(train_pids, test_size=cfg.val_size_within_train, random_state=cfg.seed)

split = {
    'train_pids': train_pids,
    'val_pids':   val_pids,
    'test_pids':  test_pids,
}
local_split = '/tmp/split.pkl'
dest_split  = f'{CACHE_DIR}/split.pkl'
with open(local_split, 'wb') as f:
    pickle.dump(split, f, protocol=4)
subprocess.run(['gsutil', '-q', 'cp', local_split, dest_split])

print(f'Split saved: train={len(train_pids):,}, val={len(val_pids):,}, test={len(test_pids):,}')
print()
print('✓ All data saved to GCS. Notebook 02_training.ipynb is ready to run.')
_maybe_stop(20)

_step(21, 'Save config JSON for downstream notebooks')
# ── Save config JSON for downstream notebooks ─────────────────────────────────
import json as _json
config_dict = asdict(cfg)
config_dict['cache_dir'] = CACHE_DIR
config_path = f'{cfg.out_dir}/data_config_v8.json'
with open(config_path, 'w') as f:
    _json.dump(config_dict, f, indent=2)

# Also copy to GCS
subprocess.run(['gsutil', '-q', 'cp', config_path, f'{CACHE_DIR}/data_config_v8.json'])
print(f'Config saved: {config_path}')
print()
print('=' * 55)
print('NOTEBOOK 01 COMPLETE')
print('=' * 55)
print(f'  Cache: {CACHE_DIR}')
print(f'  raw_sequences: {len(raw_sequences):,} patients')
print(f'  train/val/test: {len(train_pids):,}/{len(val_pids):,}/{len(test_pids):,}')
print()
print('Next: open 02_training.ipynb')
_maybe_stop(21)
