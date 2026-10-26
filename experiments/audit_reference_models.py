"""Two cross-fitted reference scores on the paired-audit points (P2 reviewer round 12, items W1 and W4).

  bag:      identities WITHOUT timing: a logistic regression per disease on indicators of which of the most frequent
            disease tokens occur before the point (plus age at the point); no ages of records, no recency, no order.
  calendar: timing WITHOUT identities, richer than count and recency: a logistic regression per disease on
            calendar features of the record before the point (count, log recency, years since the first record,
            number of distinct record days, mean and minimum gap between record days, number of codes on the last record
            day, records in the last year, age at the point, pre-record flag).

Both are fitted with 2-fold cross-fitting by person inside each disease (fit on one half of the people, score the
other), so no point is scored by a model that saw it. They are appended to the saved scores of the main tag and
evaluated exactly like the other variants: all points, padding controls removed, matched cells, wash-out. The matched
AUC of 'calendar' asks whether a stronger calendar-only reference closes the matched gap; the share of 'bag' asks
whether identities alone also reach most of the AUC. Only aggregates are written, to ~/results/audit_reference.json.
Usage: python -u experiments/audit_reference_models.py [--tag off0.1_horopen] [--vocab 300]
"""
import argparse, json, sys, zlib
from pathlib import Path
import numpy as np, torch
from delphi_aou.replication.evaluate import OFFICIAL
from delphi_aou.replication import audit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_matching_checks import matched, summary
from audit_corrected_conditional import target_noevent, subset

FIRST_DISEASE, DEATH, DAY = 13, 1269, 365.25
CAL_NAMES = ["count", "log_recency", "years_since_first", "record_days", "mean_gap", "min_gap", "codes_last_day",
             "last_year", "age", "pre_record"]


def features(r, x, a, vocab_lut, chunk=20000):
    """Vectorized bag-of-codes and calendar features for every point of one unit file. The input the model saw is the
    batch row up to the prediction token, i.e. tokens with age <= z (ages are sorted within a row)."""
    pr = r["person_row"].astype(np.int64); z = r["z"].astype(np.float64)
    n = len(pr); V = int(vocab_lut.max()) + 1
    bag = np.zeros((n, V + 1), np.float32); cal = np.zeros((n, len(CAL_NAMES)), np.float32)
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(s0 + chunk, n))
        X, A, Z = x[pr[sl]], a[pr[sl]].astype(np.float64), z[sl][:, None]
        m = (X >= FIRST_DISEASE) & (X <= DEATH) & (A <= Z)
        vi = np.where(m, vocab_lut[X], -1)
        rr, cc = np.nonzero(vi >= 0)
        bag[sl][rr, vi[rr, cc]] = 1.0
        bag[sl][:, -1] = (Z[:, 0] / DAY)
        cnt = m.sum(1)
        As = np.sort(np.where(m, A, np.inf), 1); ok = np.isfinite(As)
        first = np.where(cnt > 0, As[:, 0], np.nan)
        last = np.where(cnt > 0, np.max(np.where(ok, As, -np.inf), 1), np.nan)
        new = ok & np.concatenate([np.ones((len(As), 1), bool), As[:, 1:] != As[:, :-1]], 1)
        days = new.sum(1)
        dif = np.diff(As, axis=1); dv = ok[:, 1:] & ok[:, :-1] & (dif > 0)
        mingap = np.where(dv.any(1), np.where(dv, dif, np.inf).min(1), 0.0)
        meangap = np.where(days > 1, (last - first) / np.maximum(days - 1, 1), 0.0)
        lastday = np.where(cnt > 0, (ok & (As == last[:, None])).sum(1), 0)
        lastyear = (m & ((Z - A) <= DAY)).sum(1)
        rec = np.where(cnt > 0, np.log1p(np.maximum(Z[:, 0] - last, 0)), np.log1p(1e5))
        since = np.where(cnt > 0, (Z[:, 0] - first) / DAY, 0.0)
        cal[sl] = np.stack([cnt, rec, since, days, np.nan_to_num(meangap) / DAY, mingap / DAY, lastday, lastyear,
                            Z[:, 0] / DAY, (cnt == 0).astype(float)], 1)
    return bag, cal


def crossfit(X, y, groups, seed=0, max_fit=50000):
    """2-fold cross-fitted probabilities; folds by person. Returns scores (float32) or None if a fold has one class."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    fold = np.array([zlib.crc32(f"{seed}-{g}".encode()) % 2 for g in groups])
    out = np.zeros(len(y), np.float32)
    for f in (0, 1):
        tr, te = fold != f, fold == f
        if te.sum() == 0:
            continue
        if len(np.unique(y[tr])) < 2:
            out[te] = y[tr].mean() if tr.any() else 0.0
            continue
        idx = np.flatnonzero(tr)
        if len(idx) > max_fit:
            idx = np.random.default_rng(seed + f).choice(idx, max_fit, replace=False)
        if len(np.unique(y[idx])) < 2:
            idx = np.flatnonzero(tr)
        sc = StandardScaler().fit(X[idx])
        m = LogisticRegression(C=1.0, max_iter=300).fit(sc.transform(X[idx]), y[idx])
        out[te] = m.decision_function(sc.transform(X[te]))
    return out


def add_reference_scores(rows, x, a, row_of_person, vocab_lut):
    """Append bag and calendar scores to each row's score matrix (per disease, pooled over sexes, cross-fitted)."""
    by_k = {}
    for r in rows:
        r["person_row"] = row_of_person[r["person"].astype(np.int64)]
        by_k.setdefault(r["k"], []).append(r)
    for k, rs in by_k.items():
        F = [features(r, x, a, vocab_lut) for r in rs]
        B = np.concatenate([f[0] for f in F]); C = np.concatenate([f[1] for f in F])
        y = np.concatenate([r["case"] for r in rs]).astype(int)
        g = np.concatenate([r["person"] for r in rs])
        sb, sc = crossfit(B, y, g), crossfit(C, y, g)
        i = 0
        for r in rs:
            n = len(r["case"])
            r["scores"] = np.concatenate([r["scores"].astype(np.float32), sb[i:i + n, None], sc[i:i + n, None]], 1)
            i += n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(Path.home() / "delphi_rep_data"))
    ap.add_argument("--tag", default="off0.1_horopen")
    ap.add_argument("--vocab", type=int, default=300)
    ap.add_argument("--out", default=str(Path.home() / "results" / "audit_reference.json"))
    a_ = ap.parse_args()
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    val = np.fromfile(f"{a_.data}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    np.random.seed(1337); torch.manual_seed(1337)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device="cpu", padding="random",
                  no_event_token_rate=5)
    x, a, y, b = [t.numpy() for t in d]
    meta, rows = audit.load(Path.home() / "runs" / "audit" / a_.tag)
    names = meta["variants"]
    # saved person ids index the batch rows directly (audit.collect stores gid = batch row index)
    row_of_person = np.arange(len(x))
    tok = x[(x >= FIRST_DISEASE) & (x <= DEATH)]
    top = np.argsort(-np.bincount(tok, minlength=DEATH + 1))[:a_.vocab]
    vocab_lut = np.full(DEATH + 2, -1, np.int64); vocab_lut[top] = np.arange(len(top))
    add_reference_scores(rows, x, a, row_of_person, vocab_lut)
    names2 = names + ["bag", "calendar"]
    target_noevent(rows, y, b)
    out = {"tag": a_.tag, "vocab": a_.vocab, "variants": names2}
    for label, rs in (("all", rows), ("padding_removed", subset(rows, lambda r: r["case"] | ~r["target_noevent"]))):
        df = audit.point_aucs(rs, names2)
        res = {"plain": summary(df[[c for c in df if c != "recency_only"]]),
               "matched_default": summary(matched(rs, names2, False)[0]),
               "matched_fine": summary(matched(rs, names2, True)[0])}
        dw = audit.point_aucs(rs, names2, filt=audit.washout(365.25))
        res["washout_1y"] = summary(dw[[c for c in dw if c != "recency_only"]])
        out[label] = res
        print("RESULT", label, json.dumps(res), flush=True)
        Path(a_.out).write_text(json.dumps(out, indent=1))
    print("REFERENCE MODELS COMPLETE", flush=True)
