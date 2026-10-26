"""Robustness checks of the matched and wash-out analyses of the P2 paired audit (reviewer items W5, W7).

Reads the per-point files written by run_audit.py (~/runs/audit/<tag>/points_*.npz, Workbench only) and writes
aggregates only to ~/results/audit_checks_<tag>.json:
  * share of cases kept by the default matching (count quintile x 3 recency bands) and by a fine matching
    (exact number of prior records up to 30, then one cell; 6 recency bands);
  * matched AUC of every variant and of the record count under both matchings;
  * person-level Poisson bootstrap intervals (B=100) for the default matched AUCs and rho;
  * wash-out lengths of 1, 2, 3 and 5 years after the person's first record;
  * share of case and control points without any prior record.
Usage: python experiments/audit_matching_checks.py --tags off0.1_horopen,off0.1_hor365.25,off365.25_horopen
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from delphi_aou.replication import audit

RECENCY_DEFAULT = [30, 365.25]
RECENCY_FINE = [7, 30, 90, 365.25, 1095.75]


def cells(u, fine):
    rc = np.nan_to_num(u["recency"], nan=1e9)
    if fine:
        q = np.minimum(u["nreal"], 30)
        t = np.digitize(rc, RECENCY_FINE)
        return q * 10 + t
    q = np.searchsorted(np.quantile(u["nreal"], [0.2, 0.4, 0.6, 0.8]), u["nreal"], "right")
    return q * 3 + np.digitize(rc, RECENCY_DEFAULT)


def matched(rows, names, fine=False, W=None, persons=None):
    """Mean over diseases of the case-weighted within-cell AUC (official aggregation over units). With W ([B, n_persons]
    Poisson weights) returns [B, n_dis] arrays for the model variants. Also returns the share of cases kept."""
    cols = list(names) + ["count_only"]
    per = {c: {} for c in cols}
    kept = total = 0
    for k, u in audit._units(rows):
        S = {n: u["scores"][:, i].astype(np.float32) for i, n in enumerate(names)}
        S["count_only"] = u["nreal"].astype(np.float32)
        cell = cells(u, fine)
        total += int(u["case"].sum())
        wi = None if W is None else W[:, np.searchsorted(persons, u["person"])]
        num = {c: 0.0 for c in cols}; den = 0.0 if W is None else np.zeros(W.shape[0])
        for cc in np.unique(cell):
            m = cell == cc
            n1 = u["case"][m].sum()
            if n1 >= 1 and (~u["case"][m]).sum() >= 1:
                kept += int(n1)
                if W is None:
                    for c in cols:
                        num[c] += n1 * audit._auc(S[c][m], u["case"][m])
                    den += n1
                else:
                    w = wi[:, m]; n1w = (w * u["case"][m]).sum(1)
                    for c in names:
                        num[c] = num[c] + n1w * np.nan_to_num(audit._auc(S[c][m], u["case"][m], w))
                    den = den + n1w
        if W is None:
            if den > 0:
                for c in cols:
                    per[c].setdefault(k, []).append(num[c] / den)
        else:
            for c in names:
                per[c].setdefault(k, []).append(np.where(den > 0, num[c] / np.where(den > 0, den, 1), np.nan))
    if W is None:
        df = pd.DataFrame({c: {k: float(np.nanmean(v)) for k, v in per[c].items()} for c in cols})
        return df, kept / max(total, 1)
    dis = sorted(per[names[0]])
    return {n: np.stack([np.nanmean(np.stack(per[n][k]), 0) for k in dis], 1) for n in names}


def summary(df, ref="real"):
    out = {c: round(float(df[c].mean()), 4) for c in df}
    for c in df:
        if c != ref:
            out[f"rho_{c}"] = round((out[c] - 0.5) / (out[ref] - 0.5), 4)
    out["n_diseases"] = int(df[ref].notna().sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="off0.1_horopen,off0.1_hor365.25,off365.25_horopen")
    ap.add_argument("--B", type=int, default=100)
    a = ap.parse_args()
    for tag in a.tags.split(","):
        meta, rows = audit.load(Path.home() / "runs" / "audit" / tag)
        names = meta["variants"]
        res = {"setting": tag}
        nz_case = np.concatenate([r["nreal"][r["case"]] == 0 for r in rows]).mean()
        nz_ctrl = np.concatenate([r["nreal"][~r["case"]] == 0 for r in rows]).mean()
        res["share_no_prior_record"] = {"cases": round(float(nz_case), 4), "controls": round(float(nz_ctrl), 4)}
        for fine in (False, True):
            df, kept = matched(rows, names, fine)
            res["matched_fine" if fine else "matched_default"] = {**summary(df), "share_cases_kept": round(kept, 4)}
            print(tag, "fine" if fine else "default", json.dumps(res["matched_fine" if fine else "matched_default"]), flush=True)
        for yrs in (1, 2, 3, 5):
            df = audit.point_aucs(rows, names, filt=audit.washout(365.25 * yrs))
            res[f"washout_{yrs}y"] = summary(df[[c for c in df if c != "recency_only"]])
            print(tag, f"washout {yrs}y", json.dumps(res[f"washout_{yrs}y"]), flush=True)
        persons = np.unique(np.concatenate([r["person"] for r in rows]))
        W = np.random.default_rng(0).poisson(1.0, (a.B, len(persons))).astype(np.float32)
        A = matched(rows, names, False, W, persons)
        ci = lambda v: [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]
        mean = {n: np.nanmean(A[n], 1) for n in names}
        res["matched_default_bootstrap"] = {"B": a.B, **{f"auc_{n}": ci(mean[n]) for n in names},
                                            **{f"rho_{n}": ci((mean[n] - 0.5) / (mean["real"] - 0.5)) for n in names if n != "real"}}
        print(tag, "bootstrap", json.dumps(res["matched_default_bootstrap"]), flush=True)
        out = Path.home() / "results" / f"audit_checks_{tag}.json"
        out.write_text(json.dumps(res, indent=1))
        print("RESULT", json.dumps(res), flush=True)
    print("CHECKS COMPLETE", flush=True)
