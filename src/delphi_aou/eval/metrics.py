"""Delphi equivalent evaluation: per disease, per age band, per sex ROC AUC with
DeLong variance, incident only labels, minimum case counts, and padding safe
read out helpers. Scores must be rates or risks, not a softmax over all tokens
that mixes drugs and procedures with diseases.
"""
import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------- read out
def last_real_index(token_ids, pad_id=0):
    """Index of the last non pad token per row (right padded batches)."""
    token_ids = np.asarray(token_ids)
    real = token_ids != pad_id
    lengths = real.sum(axis=1)
    if np.any(lengths == 0):
        raise ValueError("row with no real tokens")
    return lengths - 1


def gather_last(logits, token_ids, pad_id=0):
    """logits [B, T, V] -> [B, V] taken at the last REAL position of each row.
    The old code used logits[:, -1, :], which is a PAD position for every row
    shorter than the longest row in the batch."""
    idx = last_real_index(token_ids, pad_id)
    return logits[np.arange(logits.shape[0]), idx]


def padding_mask(token_ids, pad_id=0):
    """Mask on the PAD token id, never on age == 0. Sex and race tokens sit at
    age 0 and must stay visible to attention."""
    return np.asarray(token_ids) == pad_id


# ------------------------------------------------------------------ DeLong
def _midrank(x):
    order = np.argsort(x)
    z = x[order]
    n = len(x)
    t = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = t
    return out


def delong_auc(scores, labels):
    """AUC and DeLong variance (Sun and Xu 2014 fast algorithm)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return np.nan, np.nan
    tx, ty = _midrank(pos), _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    var = np.var(v01, ddof=1) / m + np.var(v10, ddof=1) / n
    return float(auc), float(var)


# ------------------------------------------------------------- main routine
def evaluate_auc(scores, labels, eligible, age, sex, disease_names,
                 bands, min_cases=25, min_controls=25):
    """scores, labels, eligible: arrays [N, D].
    labels = 1 if the disease is first diagnosed inside the horizon.
    eligible = False for prevalent cases (already had the disease at the anchor),
    sex impossible diseases, and people whose follow up ended before the horizon
    without an event. Ineligible rows are dropped, never counted as controls.
    age in years at the anchor, sex coded as any hashable value.
    Returns one row per (disease, age band, sex) with AUC, CI and counts."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    eligible = np.asarray(eligible).astype(bool)
    age, sex = np.asarray(age), np.asarray(sex)
    rows = []
    for lo, hi in bands:
        in_band = (age >= lo) & (age < hi)
        for sx in np.unique(sex):
            base = in_band & (sex == sx)
            if base.sum() == 0:
                continue
            for d, name in enumerate(disease_names):
                use = base & eligible[:, d]
                y = labels[use, d].astype(bool)
                n_pos, n_neg = int(y.sum()), int((~y).sum())
                if n_pos < min_cases or n_neg < min_controls:
                    continue
                auc, var = delong_auc(scores[use, d], y)
                se = np.sqrt(var) if var == var else np.nan
                rows.append(dict(disease=name, band=f"{lo}-{hi}", sex=sx,
                                 n_cases=n_pos, n_controls=n_neg, auc=auc,
                                 ci_lo=auc - 1.96 * se, ci_hi=auc + 1.96 * se,
                                 se=se))
    return pd.DataFrame(rows)


def summarize(table):
    """Macro mean over disease token units and a case weighted mean. Never mixes
    in drugs or procedures: pass disease tokens only."""
    if len(table) == 0:
        return dict(n_units=0, macro_auc=np.nan, case_weighted_auc=np.nan)
    w = table["n_cases"].to_numpy(float)
    return dict(n_units=int(len(table)),
                n_diseases=int(table["disease"].nunique()),
                macro_auc=float(table["auc"].mean()),
                case_weighted_auc=float(np.average(table["auc"], weights=w)))


def paired_auc_diff_z(auc_a, var_a, auc_b, var_b):
    """Rough z test treating the two AUCs as independent (conservative when the
    models are positively correlated). For paired tests use the patient level
    bootstrap in the baselines stage."""
    z = (auc_a - auc_b) / np.sqrt(var_a + var_b)
    return z, 2 * (1 - stats.norm.cdf(abs(z)))


def delong_paired(s1, s2, labels):
    """AUC1 - AUC2 and its DeLong variance for two scores on the same people."""
    labels = np.asarray(labels).astype(bool)
    m, n = labels.sum(), (~labels).sum()
    if m < 2 or n < 2:
        return np.nan, np.nan
    comps = []
    for s in (np.asarray(s1, float), np.asarray(s2, float)):
        pos, neg = s[labels], s[~labels]
        tx, ty, tz = _midrank(pos), _midrank(neg), _midrank(np.concatenate([pos, neg]))
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        comps.append((auc, (tz[:m] - tx) / n, 1.0 - (tz[m:] - ty) / m))
    (a1, v01a, v10a), (a2, v01b, v10b) = comps
    s01 = np.cov(np.vstack([v01a, v01b]))
    s10 = np.cov(np.vstack([v10a, v10b]))
    var = (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / m + (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n
    return float(a1 - a2), float(var)


def compare_models(sa, sb, labels, eligible, age, sex, disease_names, bands, min_cases=25, min_controls=25):
    """Per unit paired DeLong difference (A minus B). Returns the mean difference, how many
    units are significantly better or worse at the 5% level, and a Bonferroni count."""
    sa, sb, labels = np.asarray(sa), np.asarray(sb), np.asarray(labels)
    eligible = np.asarray(eligible).astype(bool)
    diffs, better, worse, better_bonf, worse_bonf = [], 0, 0, 0, 0
    units = 0
    rows = []
    for lo, hi in bands:
        for sx in np.unique(sex):
            base = (age >= lo) & (age < hi) & (sex == sx)
            for d, name in enumerate(disease_names):
                use = base & eligible[:, d]
                y = labels[use, d].astype(bool)
                if y.sum() < min_cases or (~y).sum() < min_controls:
                    continue
                dlt, var = delong_paired(sa[use, d], sb[use, d], y)
                if not np.isfinite(var) or var <= 0:
                    continue
                rows.append((dlt, dlt / np.sqrt(var)))
    if not rows:
        return dict(n_units=0)
    d = np.array(rows)
    zc = stats.norm.ppf(1 - 0.025 / len(d))
    return dict(n_units=int(len(d)), mean_diff=float(d[:, 0].mean()),
                median_diff=float(np.median(d[:, 0])),
                units_better_p05=int((d[:, 1] > 1.96).sum()), units_worse_p05=int((d[:, 1] < -1.96).sum()),
                units_better_bonferroni=int((d[:, 1] > zc).sum()), units_worse_bonferroni=int((d[:, 1] < -zc).sum()))
