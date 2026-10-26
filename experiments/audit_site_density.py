"""Record density of the EHR-site groups (P2, reviewer round 12): is the group whose visits carry no site identifier a
record-density / provenance stratum?

Groups (as in audit_by_access_r12.py): people whose visits carry no site identifier, the ten largest sites pooled, the
smaller sites pooled. For each group and for cases and controls separately: median and mean number of prior diagnosis
tokens in the input at the scored point (nreal), share of points with no prior diagnosis, share whose last diagnosis is
over one year earlier or absent, and each group's median first-record age and number of first diagnoses per person in
the validation data. Then the plain and matched AUCs and the matched gap again, restricted to points with at least
one prior diagnosis (post-record points), with a person bootstrap of the gap. Only aggregates are written:
~/results/audit_site_density.json
Usage: python -u experiments/audit_site_density.py [--B 100] [--max_people N --B 5]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit
from audit_corrected_conditional import subset
from audit_paired_gap import join
from audit_by_access import analyse
from audit_by_access_r12 import SQL_SITE, bq

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--tag", default="off0.1_horopen")
    ap.add_argument("--free", default="_idfree")
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--max_people", type=int, default=-1)
    ap.add_argument("--out", default=str(Path.home() / "results" / "audit_site_density.json"))
    a = ap.parse_args()
    t0 = time.time()
    cdr = os.environ.get("WORKSPACE_CDR") or "YOUR-CDR-PROJECT.R2025Q4R6"
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = np.asarray(get_p2i(val))
    pid_of_row = val[p2i[:, 0], 0]
    st = bq(SQL_SITE, cdr)
    sid = {int(p) % 2 ** 32: s for p, s in zip(st["person_id"], st["src_id"])}
    labs = pd.Series([sid.get(int(p)) for p in pid_of_row])
    share = labs.value_counts(normalize=True)                 # over people with a site id
    big = set(s for s in share.index if share[s] >= 0.03)
    grp = np.array(["no_site_id" if s is None else ("ten_largest_sites" if s in big else "smaller_sites") for s in labs])
    print("GROUPS", {g: int((grp == g).sum()) for g in np.unique(grp)}, "big", len(big), f"{time.time() - t0:.0f}s", flush=True)

    # person-level record summaries from the validation data (first-occurrence diagnosis tokens >= 12 in .bin ids)
    person_row = np.repeat(np.arange(len(p2i)), p2i[:, 1])
    dis = val[:, 2] >= 12
    n_first = np.bincount(person_row[dis], minlength=len(p2i))
    first_age = np.full(len(p2i), np.nan)
    order = np.flatnonzero(dis)
    first_idx = np.unique(person_row[order], return_index=True)
    first_age[first_idx[0]] = val[order[first_idx[1]], 1] / 365.25
    del val
    out = {"groups": {}, "people": {}}
    for g in np.unique(grp):
        m = grp == g
        out["people"][g] = {"n": int(m.sum()), "median_first_diagnoses": float(np.median(n_first[m])),
                            "share_without_diagnosis": round(float((n_first[m] == 0).mean()), 4),
                            "median_first_record_age": round(float(np.nanmedian(first_age[m])), 1)}
    print("PEOPLE", json.dumps(out["people"]), flush=True)

    base = Path.home() / "runs" / "audit"
    mm, rm = audit.load(base / a.tag)
    mf, rf = audit.load(base / (a.tag + a.free))
    rows = join(rm, rf, mm["variants"], mf["variants"]); del rm, rf
    if a.max_people > 0:
        rows = subset(rows, lambda r: r["person"] < a.max_people)
    names = ["real", "erased", "idfree"]
    print("LOADED", sum(len(r["person"]) for r in rows), "points", f"{time.time() - t0:.0f}s", flush=True)
    for g in np.unique(grp):
        res = {}
        for who, sel in (("cases", True), ("controls", False)):
            nr = np.concatenate([r["nreal"][(grp[r["person"]] == g) & (r["case"] == sel)] for r in rows]).astype(np.float64)
            rc = np.concatenate([r["recency"][(grp[r["person"]] == g) & (r["case"] == sel)] for r in rows])
            res[who] = {"median_prior_diagnoses": float(np.median(nr)), "mean_prior_diagnoses": round(float(nr.mean()), 2),
                        "share_no_prior_diagnosis": round(float((nr == 0).mean()), 4),
                        "share_last_over_1y_or_none": round(float((~(rc <= 365.25)).mean()), 4)}
        rs = subset(rows, lambda r, g=g: (grp[r["person"]] == g) & (r["nreal"] >= 1))
        res["post_record"] = analyse(rs, names, a.B); del rs
        out["groups"][g] = res
        pr = res["post_record"]
        print("RESULT", g, json.dumps({k: res[k] for k in ("cases", "controls")}), json.dumps(pr["plain"]),
              json.dumps(pr["matched"]), json.dumps(pr.get("matched_boot_all", {})), f"{time.time() - t0:.0f}s", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("DENSITY COMPLETE", f"{time.time() - t0:.0f}s", flush=True)
