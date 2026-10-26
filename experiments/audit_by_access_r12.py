"""Reviewer round 11 follow-ups for the access strata (P2): race and ethnicity, enrolling EHR site, and insurance
restricted to ages 65 to 80 (Medicare against private at the same ages).

Same points, matched cells and bootstrap as audit_by_access.py (whose functions are reused). Race and ethnicity come
from the person table (ethnicity Hispanic or Latino first, then race); the EHR site is each person's most frequent
src_id among their visits (visit_occurrence_ext). Sites are anonymized by size rank (site A, B, ...); only sites with
at least 3% of the validation people are reported, the others are pooled. Only aggregates are written.
Output: ~/results/audit_by_access_r12.json
Usage: python -u experiments/audit_by_access_r12.py [--B 100] [--max_people N --B 5 for a smoke test]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit
from audit_matching_checks import matched
from audit_corrected_conditional import subset
from audit_paired_gap import join
from audit_by_access import analyse, strata, SQL as SQL_SURVEY

SQL_RACE = """
SELECT p.person_id, r.concept_name AS race, e.concept_name AS ethnicity
FROM `{cdr}.person` p
LEFT JOIN `{cdr}.concept` r ON r.concept_id = p.race_concept_id
LEFT JOIN `{cdr}.concept` e ON e.concept_id = p.ethnicity_concept_id
"""
SQL_SITE = """
SELECT person_id, src_id FROM (
  SELECT v.person_id, x.src_id, COUNT(*) AS n,
         ROW_NUMBER() OVER (PARTITION BY v.person_id ORDER BY COUNT(*) DESC, x.src_id) AS rk
  FROM `{cdr}.visit_occurrence` v
  JOIN `{cdr}.visit_occurrence_ext` x ON x.visit_occurrence_id = v.visit_occurrence_id
  WHERE LOWER(x.src_id) LIKE 'ehr site%'
  GROUP BY v.person_id, x.src_id)
WHERE rk = 1
"""


def race_group(race, eth):
    race, eth = str(race or "").lower(), str(eth or "").lower()
    if "hispanic or latino" in eth and "not hispanic" not in eth:
        return "hispanic_or_latino"
    if race == "white":
        return "white"
    if "black" in race:
        return "black"
    if race == "asian":
        return "asian"
    if "more than one" in race or "another single" in race or "native" in race or "pacific" in race:
        return "other_or_multiple"
    return None                                            # none indicated, prefer not to answer, skip -> no_answer


def bq(sql, cdr, max_gb=150):
    from google.cloud import bigquery
    import pandas_gbq
    c = bigquery.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    q = sql.format(cdr=cdr)
    gb = c.query(q, job_config=bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)).total_bytes_processed / 1e9
    print(f"[dry-run] {gb:.1f} GB", flush=True)
    assert gb < max_gb, "scan too large"
    return pandas_gbq.read_gbq(q, dialect="standard", project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                               use_bqstorage_api=True, progress_bar_type=None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--tag", default="off0.1_horopen")
    ap.add_argument("--free", default="_idfree")
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--max_people", type=int, default=-1)
    ap.add_argument("--out", default=str(Path.home() / "results" / "audit_by_access_r12.json"))
    a = ap.parse_args()
    t0 = time.time()
    cdr = os.environ.get("WORKSPACE_CDR") or "YOUR-CDR-PROJECT.R2025Q4R6"

    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    pid_of_row = val[np.asarray(p2i)[:, 0], 0]
    del val
    S = {}
    rc = bq(SQL_RACE, cdr)
    S["race_ethnicity"] = {int(p) % 2 ** 32: g for p, g in zip(rc["person_id"], map(race_group, rc["race"], rc["ethnicity"]))
                           if g is not None}
    print(rc.assign(g=[race_group(r, e) for r, e in zip(rc["race"], rc["ethnicity"])]).groupby("g", dropna=False).size(), flush=True)
    del rc
    st = bq(SQL_SITE, cdr)
    sid = {int(p) % 2 ** 32: s for p, s in zip(st["person_id"], st["src_id"])}
    labs = pd.Series([sid.get(int(p)) for p in pid_of_row])
    share = labs.value_counts(normalize=True)
    big = [s for s in share.index if share[s] >= 0.03]
    name = {s: f"site_{chr(65 + i)}" for i, s in enumerate(big)}            # anonymized by size rank
    S["ehr_site"] = {k: name.get(v, "smaller_sites") for k, v in sid.items()}
    print("SITES", len(share), "reported", len(big), {name[s]: round(float(share[s]), 3) for s in big}, flush=True)
    del st, sid
    sv = bq(SQL_SURVEY, cdr)
    S["insurance_age_65_80"] = {int(p) % 2 ** 32: g for p, g in strata(sv)["insurance"].items()}; del sv
    print("SURVEY/PERSON", {k: len(v) for k, v in S.items()}, f"{time.time() - t0:.0f}s", flush=True)

    base = Path.home() / "runs" / "audit"
    mm, rm = audit.load(base / a.tag)
    mf, rf = audit.load(base / (a.tag + a.free))
    rows = join(rm, rf, mm["variants"], mf["variants"]); del rm, rf
    if a.max_people > 0:
        rows = subset(rows, lambda r: r["person"] < a.max_people)
    names = ["real", "erased", "idfree"]
    print("LOADED", sum(len(r["person"]) for r in rows), "points", f"{time.time() - t0:.0f}s", flush=True)

    out = {"tag": a.tag, "free": a.tag + a.free, "B": a.B, "max_people": a.max_people}
    for proxy, mp in S.items():
        lab = np.array([mp.get(int(p), "no_answer") for p in pid_of_row])
        keep = (lambda r: r["bracket"] >= 5) if proxy == "insurance_age_65_80" else (lambda r: np.ones(len(r["case"]), bool))
        if proxy == "insurance_age_65_80":
            lab = np.where(np.isin(lab, ["medicare", "private", "medicaid"]), lab, "other_or_no_answer")
        groups = sorted(set(lab))
        n_people = {g: int((lab == g).sum()) for g in groups}
        assert min(n_people.values()) > 20, n_people
        dfm = {}
        for g in groups:
            rs = subset(rows, lambda r, g=g: (lab[r["person"]] == g) & keep(r))
            dfm[g] = matched(rs, names, False)[0]; del rs
        big = [g for g in groups if n_people[g] >= 0.03 * len(lab) and g not in ("no_answer", "other_or_no_answer")]
        assert big, n_people
        common = set.intersection(*[set(dfm[g].dropna().index) for g in big])
        out[proxy] = {"n_people_val": n_people, "n_common_diseases": len(common), "common_from": big, "strata": {}}
        for g in groups:
            rs = subset(rows, lambda r, g=g: (lab[r["person"]] == g) & keep(r))
            out[proxy]["strata"][g] = analyse(rs, names, a.B, common, dfm[g]); del rs
            r = out[proxy]["strata"][g]
            print("RESULT", proxy, g, n_people[g], json.dumps(r.get("matched_common", {})),
                  json.dumps(r.get("matched_boot_common", {})), f"{time.time() - t0:.0f}s", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("R12 ACCESS COMPLETE", f"{time.time() - t0:.0f}s", flush=True)
