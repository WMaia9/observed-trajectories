"""Identity-free share and matched gap by access strata (P2, reviewer round 10: "access is asserted, never measured").

The record calendar is said to mix morbidity with access to care. Here the paired audit points of the main Delphi-2M
(AoU) checkpoint and of the identity-free model (as in audit_paired_gap.py) are split by three survey proxies of
access, taken from The Basics survey: annual household income, health insurance type and education. For every
stratum we report, over the points of the people in it:
  plain: mean AUC of real, erased and identity-free, rho_f (identity-free share) and the count-only score;
  matched: the same inside cells of count quintile x recency band (cells built within the stratum);
  a person Poisson bootstrap (B resamples) of the matched gap real minus identity-free and of rho_f,
both over all diseases evaluable in the stratum and over the diseases evaluable in every stratum of that proxy
(common set), so strata are compared on the same diseases.
People with no answer or a skip form their own stratum. Only aggregates are written (strata have far more than 20
people; the person count per stratum is reported and checked). Output: ~/results/audit_by_access.json
Usage: python -u experiments/audit_by_access.py [--B 100] [--max_people N --B 5 for a smoke test]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit
from audit_matching_checks import matched, summary
from audit_corrected_conditional import subset
from audit_paired_gap import join

SQL = """
SELECT o.person_id, q.concept_name AS question, a.concept_name AS answer
FROM `{cdr}.observation` o
JOIN `{cdr}.concept` q ON q.concept_id = o.observation_source_concept_id
LEFT JOIN `{cdr}.concept` a ON a.concept_id = o.value_source_concept_id
WHERE q.concept_name LIKE 'Income: Annual Income%'
   OR q.concept_name LIKE 'Health Insurance: Insurance Type Update%'
   OR q.concept_name LIKE 'Education Level: Highest Grade%'
"""


def income(a):
    a = a.lower()
    if any(s in a for s in ("less 10k", "10k 25k", "25k 35k")):
        return "under_35k"
    if any(s in a for s in ("35k 50k", "50k 75k", "75k 100k")):
        return "35k_to_100k"
    if any(s in a for s in ("100k 150k", "150k 200k", "more 200k")):
        return "over_100k"
    return None


INS_ORDER = ["uninsured", "medicaid", "medicare", "private", "other"]


def insurance(a):
    a = a.lower()
    if a.endswith(": none") or "none of these" in a:
        return "uninsured"
    if "medicaid" in a:
        return "medicaid"
    if "medicare" in a:
        return "medicare"
    if "employer" in a or "purchased" in a:
        return "private"
    if a.endswith(": va") or any(s in a for s in ("military", "indian", "other health plan")):
        return "other"
    return None


def education(a):
    a = a.lower()
    if any(s in a for s in ("never attended", "one through four", "five through eight", "nine through eleven",
                             "twelve or ged", "less than a high school")):
        return "high_school_or_less"
    if "college one to three" in a:
        return "some_college"
    if "college graduate" in a or "advanced degree" in a:
        return "college_degree"
    return None


def strata(df):
    """person_id -> stratum for each proxy (missing / skip -> 'no_answer')."""
    out = {}
    q = df["question"].str.lower()
    for name, key, fn in (("income", "income", income), ("insurance", "insurance type", insurance),
                          ("education", "education", education)):
        d = df[q.str.contains(key)].copy()
        d["s"] = d["answer"].fillna("").map(fn)
        d = d.dropna(subset=["s"])
        if name == "insurance":                     # several answers allowed: keep the first in INS_ORDER
            d["o"] = d["s"].map(INS_ORDER.index)
            d = d.sort_values("o").drop_duplicates("person_id")
        else:
            d = d.drop_duplicates("person_id")
        out[name] = dict(zip(d["person_id"].astype(np.int64), d["s"]))
    return out


def ci(v):
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]


def unit_keys(rows):
    """Sorted diseases with at least one unit, as audit._units yields them (without copying the arrays)."""
    ks = set()
    for r in rows:
        for br in np.unique(r["bracket"]):
            m = r["bracket"] == br
            if r["case"][m].sum() >= 2 and (~r["case"][m]).sum() >= 2:
                ks.add(r["k"]); break
    return sorted(ks)


def analyse(rs, names, B, common=None, dfm=None):
    dfp = audit.point_aucs(rs, names); dfp = dfp[[c for c in dfp if c != "recency_only"]]
    dfm = matched(rs, names, False)[0] if dfm is None else dfm
    res = {"plain": summary(dfp), "matched": summary(dfm)}
    if common is not None:
        res["plain_common"] = summary(dfp.loc[dfp.index.intersection(common)])
        res["matched_common"] = summary(dfm.loc[dfm.index.intersection(common)])
    if B and len(unit_keys(rs)) > 0:
        persons = np.unique(np.concatenate([r["person"] for r in rs]))
        W = np.random.default_rng(0).poisson(1.0, (B, len(persons))).astype(np.float32)
        A = matched(rs, names, False, W, persons)
        dis = unit_keys(rs)                              # disease order of matched(..., W) (all units, any cells)
        for tag, sel in (("all", np.ones(len(dis), bool)),
                         ("common", np.isin(dis, list(common)) if common is not None else None)):
            if sel is None or not sel.any():
                continue
            m = {n: np.nanmean(A[n][:, sel], 1) for n in names}
            res[f"matched_boot_{tag}"] = {"B": B, "gap_real_minus_idfree": ci(m["real"] - m["idfree"]),
                                         "rho_idfree": ci((m["idfree"] - 0.5) / (m["real"] - 0.5)),
                                         "auc_real": ci(m["real"]), "auc_idfree": ci(m["idfree"])}
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--tag", default="off0.1_horopen")
    ap.add_argument("--free", default="_idfree")
    ap.add_argument("--B", type=int, default=100)
    ap.add_argument("--max_people", type=int, default=-1, help="smoke test: only the first N batch rows")
    ap.add_argument("--out", default=str(Path.home() / "results" / "audit_by_access.json"))
    a = ap.parse_args()
    t0 = time.time()
    import pandas_gbq
    cdr = os.environ.get("WORKSPACE_CDR") or "YOUR-CDR-PROJECT.R2025Q4R6"
    sv = pandas_gbq.read_gbq(SQL.format(cdr=cdr), dialect="standard", project_id=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                             use_bqstorage_api=True, progress_bar_type=None)
    summ = sv.groupby(["question", "answer"], dropna=False)["person_id"].nunique().reset_index(name="n")
    print(summ[summ["n"] > 20].to_string(index=False), flush=True)        # mapping check (cells of 20 or fewer hidden)
    S = strata(sv); del sv
    print("SURVEY", {k: len(v) for k, v in S.items()}, f"{time.time() - t0:.0f}s", flush=True)

    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i
    val = np.fromfile(f"{a.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    pid_of_row = val[np.asarray(p2i)[:, 0], 0]          # uint32-wrapped person id of each batch row
    del val
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
        wrapped = {int(p) % 2 ** 32: s for p, s in mp.items()}
        lab = np.array([wrapped.get(int(p), "no_answer") for p in pid_of_row])
        groups = sorted(set(lab))
        n_people = {g: int((lab == g).sum()) for g in groups}
        assert min(n_people.values()) > 20, n_people
        dfm = {}                                       # one stratum in memory at a time (RAM)
        for g in groups:
            rs = subset(rows, lambda r, g=g: lab[r["person"]] == g)
            dfm[g] = matched(rs, names, False)[0]; del rs
        big = [g for g in groups if n_people[g] >= 0.03 * len(lab)]      # small strata (e.g. uninsured) would empty
        common = set.intersection(*[set(dfm[g].dropna().index) for g in big])  # the common set; they keep their own
        out[proxy] = {"n_people_val": n_people, "n_common_diseases": len(common), "common_from": big, "strata": {}}
        for g in groups:
            rs = subset(rows, lambda r, g=g: lab[r["person"]] == g)
            out[proxy]["strata"][g] = analyse(rs, names, a.B, common, dfm[g]); del rs
            r = out[proxy]["strata"][g]
            print("RESULT", proxy, g, n_people[g], json.dumps({k: r[k] for k in ("matched_common",) if k in r}),
                  json.dumps(r.get("matched_boot_common", {})), f"{time.time() - t0:.0f}s", flush=True)
            Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("ACCESS COMPLETE", f"{time.time() - t0:.0f}s", flush=True)
