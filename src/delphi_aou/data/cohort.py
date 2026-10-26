#!/usr/bin/env python
"""Stage 2: observation period cohort, first occurrence ICD-10 level 3 tokens, pilot stats.

Reads the cache written by delphi_01_prepare.py and delphi_01b_extras.py (no BigQuery,
so it costs only VM time). Writes, next to the cache:
  s2_fo_events.parquet   person_id, token, age_days   (first occurrence, ties kept, DEATH kept)
  s2_cohort.parquet      person_id, split, sex_token, race_token, birth_date, obs_start, obs_end, death_age_days
  s2_vocab.csv           tokens kept (train only, >= min_train_patients)
  s2_consort.json        cohort flow counts
  s2_pilot.json          aggregate pilot stats (cells under 20 are suppressed)
Usage:  RUN_MODE=test|full python stage2_cohort_tokens.py
Local test:  CACHE_DIR=/path/to/dir OUT_DIR=/path/to/dir python stage2_cohort_tokens.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from delphi_aou import config as common

CFG = common.load_config()
RUN_MODE = os.environ.get("RUN_MODE", "test")
BUCKET = (os.environ.get("WORKSPACE_BUCKET") or "gs://YOUR-WORKSPACE-BUCKET") + "/notebooks/"
CACHE_DIR = os.environ.get("CACHE_DIR") or BUCKET + ("cache_v8_test" if RUN_MODE == "test" else "cache_v8")
OUT_DIR = os.environ.get("OUT_DIR") or CACHE_DIR
MIN_CELL = CFG["policy"]["min_cell_count_for_export"]


def sup(n):
    """Small cell suppression for anything that may leave the Workbench."""
    n = int(n)
    return n if n >= MIN_CELL else f"<{MIN_CELL}"


def read_parquet(name):
    if CACHE_DIR.startswith("gs://"):
        local = Path(tempfile.gettempdir()) / f"{name}.parquet"
        subprocess.run(["gsutil", "-q", "cp", f"{CACHE_DIR}/{name}.parquet", str(local)], check=True)
        return _read_local(local)
    return _read_local(Path(CACHE_DIR) / f"{name}.parquet")


def _read_local_from_cache(name):
    """Used by stage 3 to read the stage 2 outputs from OUT_DIR."""
    global CACHE_DIR
    old, CACHE_DIR = CACHE_DIR, OUT_DIR
    try:
        return read_parquet(name)
    finally:
        CACHE_DIR = old


def _read_local(path):
    """Read parquet ignoring pandas metadata: BigQuery date columns are stored with
    the 'dbdate' extension dtype, which plain pandas cannot parse."""
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pandas(ignore_metadata=True)


def read_parquet_chunks(name, columns, batch_rows=4_000_000):
    """Yield DataFrames of at most batch_rows rows (keeps the 80M row event table out of RAM)."""
    import pyarrow.parquet as pq
    if CACHE_DIR.startswith("gs://"):
        path = Path(tempfile.gettempdir()) / f"{name}.parquet"
        subprocess.run(["gsutil", "-q", "cp", f"{CACHE_DIR}/{name}.parquet", str(path)], check=True)
    else:
        path = Path(CACHE_DIR) / f"{name}.parquet"
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows, columns=columns):
        yield batch.to_pandas(ignore_metadata=True)


def write_parquet(df, name):
    local = Path(tempfile.gettempdir()) / f"{name}.parquet"
    df.to_parquet(local, index=False)
    if OUT_DIR.startswith("gs://"):
        subprocess.run(["gsutil", "-q", "cp", str(local), f"{OUT_DIR}/{name}.parquet"], check=True)
    else:
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        local.replace(Path(OUT_DIR) / f"{name}.parquet")


def write_json(obj, name):
    local = Path(tempfile.gettempdir()) / name
    local.write_text(json.dumps(obj, indent=2, default=float))
    if OUT_DIR.startswith("gs://"):
        subprocess.run(["gsutil", "-q", "cp", str(local), f"{OUT_DIR}/{name}"], check=True)
    else:
        Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
        local.replace(Path(OUT_DIR) / name)


def build(demo, events, death, obsp, cfg=CFG):
    """Pure function, unit tested on synthetic data. Returns cohort, fo_events,
    vocab, consort, pilot."""
    consort = {}
    demo = demo.drop_duplicates("person_id").copy()
    demo["birth_date"] = pd.to_datetime(demo["birth_date"], errors="coerce")
    consort["persons_in_cache"] = sup(len(demo))
    demo = demo.dropna(subset=["birth_date"])
    consort["with_birth_date"] = sup(len(demo))

    obsp = obsp.copy()
    obsp["obs_start"] = pd.to_datetime(obsp["obs_start"])
    obsp["obs_end"] = pd.to_datetime(obsp["obs_end"])
    coh = demo.merge(obsp[["person_id", "obs_start", "obs_end", "n_periods"]], on="person_id", how="inner")
    consort["with_observation_period"] = sup(len(coh))
    span_years = (coh["obs_end"] - coh["obs_start"]).dt.days / 365.25
    coh = coh[span_years >= cfg["cohort"]["min_ehr_years_before_index"]]
    consort["ehr_span_at_least_min_years"] = sup(len(coh))

    # ---- death first, so each event chunk can be filtered without a second pass
    death = death.copy()
    death["event_date"] = pd.to_datetime(death["event_date"], errors="coerce")
    death = death.dropna(subset=["event_date"]).sort_values("event_date").drop_duplicates("person_id")
    death = death.merge(coh[["person_id", "birth_date"]], on="person_id", how="inner")
    death["death_age_days"] = (death["event_date"] - death["birth_date"]).dt.days
    consort["death_records_in_cohort"] = sup(len(death))
    coh = coh.merge(death[["person_id", "death_age_days"]], on="person_id", how="left")

    # ---- events in chunks (the full table does not fit in memory as one frame):
    # level 3 token, all occurrence counts, post death drop, first occurrence by min age.
    # Chunks may split a person, so counts are summed and first occurrences re-reduced.
    chunks = [events] if isinstance(events, pd.DataFrame) else events
    cols = ["person_id", "birth_date", "obs_start", "obs_end", "death_age_days"]
    cvals = coh[cols]
    count_parts, first_parts = [], []
    n_dropped = n_outside = n_total = 0
    for ev in chunks:
        ev = ev[["person_id", "concept_id", "event_date"]].copy()
        ev["event_date"] = pd.to_datetime(ev["event_date"], errors="coerce")
        ev = ev.dropna(subset=["event_date"])
        ev["token"] = ("D=" + ev["concept_id"].astype(str).str.upper().str[:3]).astype("category")
        ev = ev.drop(columns="concept_id").merge(cvals, on="person_id", how="inner")
        ev["age_days"] = (ev["event_date"] - ev["birth_date"]).dt.days
        ev = ev[ev["age_days"] >= 0]
        count_parts.append(ev.groupby("person_id").size())
        if cfg["tokens"]["drop_post_death_events"]:
            keep_ev = ev["death_age_days"].isna() | (ev["age_days"] <= ev["death_age_days"])
            n_dropped += int((~keep_ev).sum())
            ev = ev[keep_ev]
        n_total += len(ev)
        n_outside += int(((ev["event_date"] < ev["obs_start"]) | (ev["event_date"] > ev["obs_end"])).sum())
        first_parts.append(ev.groupby(["person_id", "token"], observed=True)["age_days"].min().reset_index())
    n_all_occurrence = pd.concat(count_parts).groupby(level=0).sum()
    if cfg["tokens"]["drop_post_death_events"]:
        consort["post_death_events_dropped"] = sup(n_dropped)
    fo = pd.concat(first_parts, ignore_index=True)
    fo["token"] = fo["token"].astype(str)
    fo = fo.groupby(["person_id", "token"])["age_days"].min().reset_index()
    fo = fo[["person_id", "token", "age_days"]]

    # ---- keep patients with at least one event, then split and train only vocab
    has_event = set(fo["person_id"])
    coh = coh[coh["person_id"].isin(has_event)].copy()
    consort["with_at_least_one_first_occurrence_event"] = sup(len(coh))
    coh["split"] = common.assign_split(coh["person_id"].to_numpy(), cfg)
    consort["split_counts"] = {k: sup((coh["split"] == k).sum()) for k in ["train", "val", "test"]}

    fo = fo[fo["person_id"].isin(set(coh["person_id"]))]
    train_ids = set(coh.loc[coh["split"] == "train", "person_id"])
    cnt = fo[fo["person_id"].isin(train_ids)].groupby("token")["person_id"].nunique()
    keep = cnt[cnt >= cfg["tokens"]["min_train_patients_per_token"]]
    vocab = keep.rename("n_train_patients").reset_index().sort_values("n_train_patients", ascending=False)
    fo = fo[fo["token"].isin(set(vocab["token"]))]

    # death is kept for every patient with a death record, after the vocab filter
    dth = coh.loc[coh["death_age_days"].notna(), ["person_id", "death_age_days"]]
    dth_rows = dth.rename(columns={"death_age_days": "age_days"}).assign(token="DEATH")[["person_id", "token", "age_days"]]
    fo = pd.concat([fo, dth_rows], ignore_index=True).sort_values(["person_id", "age_days"]).reset_index(drop=True)

    # ---- pilot stats (aggregates only)
    per = fo.groupby("person_id").size()
    q = [0.5, 0.9, 0.99]
    ties = fo.groupby(["person_id", "age_days"]).size()
    tie_share = float((ties[ties > 1]).sum() / max(len(fo), 1))
    ev_all = n_all_occurrence.reindex(coh["person_id"]).fillna(0)
    outside = n_outside / max(n_total, 1)
    span = (coh["obs_end"] - coh["obs_start"]).dt.days / 365.25
    pilot = {
        "first_occurrence_events_per_patient_quantiles": {str(k): float(per.quantile(k)) for k in q},
        "all_occurrence_icd_events_per_patient_quantiles": {str(k): float(ev_all.quantile(k)) for k in q},
        "share_first_occurrence_patients_over_512_tokens": float((per > 512).mean()),
        "share_all_occurrence_patients_over_512_icd_events": float((ev_all > 512).mean()),
        "share_first_occurrence_events_in_same_day_ties": tie_share,
        "share_events_outside_observation_period": float(outside),
        "ehr_span_years_quantiles": {str(k): float(span.quantile(k)) for k in [0.1, 0.5, 0.9]},
        "death_tokens_retained": sup((fo["token"] == "DEATH").sum()),
        "vocab_size_disease_tokens": int(len(vocab)),
    }
    return coh, fo, vocab, consort, pilot


def main():
    print(f"RUN_MODE={RUN_MODE} cache={CACHE_DIR}", flush=True)
    demo = read_parquet("df_demo")
    events = read_parquet_chunks("df_events", ["person_id", "concept_id", "event_date"])
    try:
        death = read_parquet("df_death")
    except Exception:
        death = pd.DataFrame(columns=["person_id", "event_date"])
    obsp = read_parquet("x_observation_period")
    print("schemas:", {n: list(d.columns) for n, d in dict(demo=demo, death=death, obsp=obsp).items()}, flush=True)
    coh, fo, vocab, consort, pilot = build(demo, events, death, obsp)
    write_parquet(coh, "s2_cohort")
    write_parquet(fo, "s2_fo_events")
    write_json(consort, "s2_consort.json")
    write_json(pilot, "s2_pilot.json")
    vtmp = Path(tempfile.gettempdir()) / "s2_vocab.csv"
    vocab.to_csv(vtmp, index=False)
    if OUT_DIR.startswith("gs://"):
        subprocess.run(["gsutil", "-q", "cp", str(vtmp), f"{OUT_DIR}/s2_vocab.csv"], check=True)
    else:
        vtmp.replace(Path(OUT_DIR) / "s2_vocab.csv")
    print("\nCONSORT:", json.dumps(consort, indent=2, default=float))
    print("\nPILOT:", json.dumps(pilot, indent=2, default=float))
    sh = common.split_hash(coh["person_id"].to_numpy(), coh["split"].to_numpy())
    print("\nsplit_hash:", sh, "config_hash:", common.config_hash(CFG))
    print("STAGE 2 COMPLETE")


if __name__ == "__main__":
    main()
