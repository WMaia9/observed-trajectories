"""Stage 1: data extraction from the All of Us CDR with BigQuery (runs only inside the Workbench).

  core.py    demographics, first and all occurrence conditions, procedures, drugs, labs, BMI, smoking,
             alcohol, SDoH survey items, death; vocabularies; raw sequences (21 numbered steps)
  extras.py  observation periods, visits, top 25 numeric measurements, wearables availability,
             SDoH question discovery
  sdoh.py    survey answers for housing, food insecurity and discrimination

These are executable modules: importing them runs BigQuery queries, so run them through
scripts/10_extract_core.py, 11_extract_extras.py and 12_extract_sdoh.py (or python -m). Every query
is dry run first and capped in GB scanned. Outputs go to the GCS cache, never to this repository.
"""
