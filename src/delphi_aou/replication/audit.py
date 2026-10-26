"""Paired audit of the official Delphi-2M evaluation: several input variants on ONE batch, saved per case/control.

The sweep (sweep.py) runs each input control in its own process, so the random no-event padding of the official
get_batch, and therefore the selected (person, time) points, differ slightly between controls. Here every variant is
scored on the same batch and the same selected points, which makes the comparisons paired and allows:

  R1  patient-level (Poisson) bootstrap intervals for each mean AUC, for real minus control, and for the share
      rho = (AUC_control - 0.5) / (AUC_real - 0.5), overall and per disease;
  R2  a timing-only variant that also erases the lifestyle tokens (BMI, smoking, alcohol);
  R3  AUC inside strata matched on the number of prior real records and the time since the last one, and two
      model-free timing scores (record count, recency) that need no out-of-distribution input;
  R4  a wash-out that drops targets in the first year after a person's first record (left truncation).

Selection mirrors evaluate_auc.get_calibration_auc of the official code (commit fb72166) line by line, with the seed
key of fast_eval, so the per-disease AUC of the 'real' variant equals fast_eval for the same setting (tested).
Per-point arrays are participant level and stay inside the Workbench; only aggregates are exported.

Batch token ids: 0 padding, 1 no event, 2/3 sex, 4..12 lifestyle, 13..1268 diseases, 1269 death.
"""
import os, json, zlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import torch

FIRST_DISEASE, FIRST_LIFESTYLE = 13, 4


class EraseIdentityAndLifestyle(torch.nn.Module):
    """Timing-only control without lifestyle: lifestyle, disease and death tokens all become no event (id 1)."""

    def __init__(self, model, token=1):
        super().__init__(); self.m = model; self.token = token

    def forward(self, x, ages, *rest):
        x = torch.where((x >= FIRST_DISEASE) & (x < 1270), torch.full_like(x, self.token), x)
        return self.m(torch.where((x >= FIRST_LIFESTYLE) & (x < FIRST_DISEASE), torch.ones_like(x), x), ages, *rest)


# ------------------------------------------------------------------------------------------- collection
_G = {}


def _select(args):
    """Official selection for one (sex, disease). Returns per-point arrays (scores for every variant)."""
    sex, j, k = args
    x, a, y, b, P, pidx, first = _G["sex"][sex]
    off_key, hor = _G["key"], _G["horizon"]
    ages = _G["age_groups"]; step = ages[1] - ages[0]
    np.random.seed(zlib.crc32(f"{off_key}-{sex}-{k}".encode()))
    wk = np.where(y == k)
    if len(wk[0]) < 2:
        return None
    wc = np.where((y != k) * (~(y == k).any(-1))[..., None])
    w0 = np.concatenate([wk[0], wc[0]]); w1 = np.concatenate([wk[1], wc[1]])
    pred = pidx[w0, w1]
    keep = pred != -1
    z = a[w0, pred][keep]; zk = b[w0, w1][keep]
    p_idx = w0[keep]; pos = pred[keep]
    ncase = int((pred[: len(wk[0])] != -1).sum())
    out = []
    for i, aa in enumerate(ages):
        m = np.logical_and(z / 365.25 >= aa, z / 365.25 < aa + step)
        if hor is not None:
            m = m * (zk - z < hor)
        sel = p_idx[m]
        perm = np.random.permutation(len(sel))
        _, ind = np.unique(sel[perm], return_index=True)
        ind = perm[ind]
        s = np.zeros(int(np.sum(m)), dtype=bool); s[ind] = True
        m[m] = s
        if m[ncase:].sum() < 2 or m[:ncase].sum() < 2:
            continue
        idx = np.flatnonzero(m)
        pp, ps = p_idx[idx], pos[idx]
        xr = x[pp]                                                      # input tokens of these rows
        upto = np.arange(x.shape[1])[None, :] <= ps[:, None]
        real = (xr >= FIRST_DISEASE) & upto
        nreal = real.sum(1).astype(np.int16)
        last = np.where(real, a[pp], -np.inf).max(1)
        rec = np.where(np.isfinite(last), z[idx] - last, np.nan).astype(np.float32)
        out.append({"person": _G["gid"][sex][pp].astype(np.int32), "case": idx < ncase,
                    "bracket": np.full(len(idx), i, np.int8), "z": z[idx].astype(np.float32),
                    "zk": zk[idx].astype(np.float32), "nreal": nreal, "recency": rec,
                    "first": first[pp].astype(np.float32),
                    "scores": np.stack([P[v][pp, ps, j] for v in range(len(P))], 1)})
    if not out:
        return None
    r = {key: np.concatenate([o[key] for o in out]) for key in out[0]}
    r["k"] = k; r["sex"] = sex
    return r


def collect(models, d, labels, out_dir, offset=0.1, horizon=None, age_groups=np.arange(40, 80, 5),
            filter_min_total=100, disease_chunk_size=100, batch_size=512, workers=None, device="cuda", key=0, keys=None):
    """models: dict name -> model (official interface). Writes out_dir/points_<sex>_<k>.npz and meta.json.
    keys: optional list of selection seeds; the scores are computed once and the selection is drawn once per key,
    written to out_dir/key<key>/ (used to measure the variance due to point selection)."""
    import sys
    from delphi_aou.replication.evaluate import OFFICIAL
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_auc as ev
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    dirs = {key: out_dir} if keys is None else {kk: out_dir / f"key{kk}" for kk in keys}
    for dd_ in dirs.values():
        dd_.mkdir(parents=True, exist_ok=True)
    names = list(models)
    workers = workers or max(1, os.cpu_count() or 2)
    diseases = ev.get_common_diseases(labels, filter_min_total)
    chunks = np.array_split(diseases, (len(diseases) + disease_chunk_size - 1) // disease_chunk_size)
    dn = [t.cpu().numpy() for t in d]
    pidx = ((d[1][:, :, None] < d[3][:, None, :] - offset).sum(1) - 1).cpu().numpy()
    x, a, y, b = dn
    rec_any = np.where(x >= FIRST_DISEASE, a, np.inf).min(1)
    rec_tgt = np.where((y >= FIRST_DISEASE) & (y < 1269), b, np.inf).min(1)
    first = np.minimum(rec_any, rec_tgt)
    masks = {s: ((x == idx).sum(1) > 0) for s, idx in [("female", 2), ("male", 3)]}
    gid = {s: np.flatnonzero(m) for s, m in masks.items()}
    n_written = {kk: 0 for kk in dirs}
    for ck, chunk in enumerate(chunks):
        P = []
        for name in names:
            parts = []
            with torch.no_grad():
                for dd in zip(*[torch.split(t, batch_size) for t in d]):
                    dd = [t.to(device) for t in dd]
                    parts.append(models[name](*dd)[0][:, :, torch.as_tensor(chunk, device=dd[0].device)]
                                 .float().cpu().numpy().astype("float16"))
            P.append(np.vstack(parts))
        for kk, odir in dirs.items():
            _G.update(key=kk, horizon=horizon, age_groups=age_groups, gid=gid,
                      sex={s: (x[m], a[m], y[m], b[m], [p[m] for p in P], pidx[m], first[m]) for s, m in masks.items()})
            tasks = [(s, j, int(k)) for s in masks for j, k in enumerate(chunk)]
            if workers == 1:
                res = map(_select, tasks)
            else:
                pool = mp.get_context("fork").Pool(workers); res = pool.imap_unordered(_select, tasks, chunksize=4)
            for r in res:
                if r is None:
                    continue
                np.savez_compressed(odir / f"points_{r['sex']}_{r['k']}.npz",
                                    **{q: v for q, v in r.items() if q not in ("k", "sex")})
                n_written[kk] += 1
            if workers != 1:
                pool.close(); pool.join()
            _G.clear()
        print(f"audit chunk {ck + 1}/{len(chunks)}", flush=True)
    for kk, odir in dirs.items():
        (odir / "meta.json").write_text(json.dumps({"variants": names, "offset": offset, "horizon": horizon,
                                                    "files": n_written[kk], "key": kk}, indent=1))
    return out_dir


# ------------------------------------------------------------------------------------------- statistics
def _auc(score, case, w=None):
    """Mann-Whitney AUC (ties count one half). w: [B, n] nonnegative weights (bootstrap) or None. Returns scalar or [B]."""
    o = np.argsort(score, kind="mergesort"); s = score[o]; c = case[o]
    cs, ct = s[c], s[~c]
    if w is None:
        if len(cs) == 0 or len(ct) == 0:
            return np.nan
        lo = np.searchsorted(ct, cs, "left"); hi = np.searchsorted(ct, cs, "right")
        return float((lo + 0.5 * (hi - lo)).sum() / (len(cs) * len(ct)))
    w = w[:, o]; wc, wt = w[:, c], w[:, ~c]
    cw = np.concatenate([np.zeros((w.shape[0], 1)), np.cumsum(wt, 1)], 1)
    lo = np.searchsorted(ct, cs, "left"); hi = np.searchsorted(ct, cs, "right")
    below, upto = cw[:, lo], cw[:, hi]
    num = (wc * (below + 0.5 * (upto - below))).sum(1)
    den = wc.sum(1) * wt.sum(1)
    return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def load(out_dir):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "meta.json").read_text())
    rows = []
    for f in sorted(out_dir.glob("points_*.npz")):
        _, sex, k = f.stem.split("_")
        z = np.load(f)
        rows.append({"k": int(k), "sex": sex, **{kk: z[kk] for kk in z.files}})
    return meta, rows


def _units(rows, filt=None):
    """Yield (k, unit arrays) for each (disease, sex, bracket) with at least 2 cases and 2 controls after filt."""
    for r in rows:
        keep = np.ones(len(r["case"]), bool) if filt is None else filt(r)
        for br in np.unique(r["bracket"]):
            m = keep & (r["bracket"] == br)
            if r["case"][m].sum() >= 2 and (~r["case"][m]).sum() >= 2:
                yield r["k"], {kk: r[kk][m] for kk in ("person", "case", "scores", "nreal", "recency")}


def point_aucs(rows, names, filt=None, matched=False, extra_scores=True):
    """Per-disease AUC (official aggregation: mean over sex x bracket units) for each variant, plus the model-free
    timing scores. matched=True computes AUC inside nreal-quintile x recency-tertile cells of each unit."""
    cols = list(names) + (["count_only", "recency_only"] if extra_scores else [])
    per = {c: {} for c in cols}
    for k, u in _units(rows, filt):
        S = {n: u["scores"][:, i].astype(np.float32) for i, n in enumerate(names)}
        if extra_scores:
            S["count_only"] = u["nreal"].astype(np.float32)
            S["recency_only"] = -np.nan_to_num(u["recency"], nan=1e9)
        if matched:
            q = np.searchsorted(np.quantile(u["nreal"], [0.2, 0.4, 0.6, 0.8]), u["nreal"], "right")
            rc = np.nan_to_num(u["recency"], nan=1e9)
            t = np.digitize(rc, [30, 365.25])
            cell = q * 3 + t
            for c in cols:
                num = den = 0.0
                for cc in np.unique(cell):
                    m = cell == cc
                    n1 = u["case"][m].sum()
                    if n1 >= 1 and (~u["case"][m]).sum() >= 1:
                        num += n1 * _auc(S[c][m], u["case"][m]); den += n1
                if den > 0:
                    per[c].setdefault(k, []).append(num / den)
        else:
            for c in cols:
                per[c].setdefault(k, []).append(_auc(S[c], u["case"]))
    return pd.DataFrame({c: {k: float(np.nanmean(v)) for k, v in per[c].items()} for c in cols})


def bootstrap(rows, names, ref="real", B=200, seed=0, filt=None):
    """Paired Poisson bootstrap over persons. Returns dict of summaries (mean AUC, differences and rho with 95% CIs)
    and per-disease rho percentiles."""
    persons = np.unique(np.concatenate([r["person"] for r in rows]))
    rng = np.random.default_rng(seed)
    W = rng.poisson(1.0, (B, len(persons))).astype(np.float32)
    acc = {n: {} for n in names}
    for k, u in _units(rows, filt):
        wi = W[:, np.searchsorted(persons, u["person"])]
        for i, n in enumerate(names):
            acc[n].setdefault(k, []).append(_auc(u["scores"][:, i].astype(np.float32), u["case"], wi))
    dis = sorted(set(acc[ref]))
    A = {n: np.stack([np.nanmean(np.stack(acc[n][k]), 0) for k in dis], 1) for n in names}   # [B, n_dis]
    point = point_aucs(rows, names, filt, extra_scores=False).loc[dis]
    ci = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    out = {"n_diseases": len(dis), "B": B}
    for n in names:
        out[f"mean_auc_{n}"] = {"point": float(point[n].mean()), "ci": ci(np.nanmean(A[n], 1))}
        if n != ref:
            d = np.nanmean(A[ref], 1) - np.nanmean(A[n], 1)
            rho = (np.nanmean(A[n], 1) - 0.5) / (np.nanmean(A[ref], 1) - 0.5)
            out[f"diff_{ref}_minus_{n}"] = {"point": float(point[ref].mean() - point[n].mean()), "ci": ci(d)}
            out[f"rho_{n}"] = {"point": float((point[n].mean() - 0.5) / (point[ref].mean() - 0.5)), "ci": ci(rho)}
            rd = (point[n] - 0.5) / (point[ref] - 0.5)
            ok = (point[ref] - 0.5) > 0.05                                 # rho is unstable when the real AUC is near 0.5
            out[f"rho_per_disease_{n}"] = {"median": float(rd[ok].median()), "q25": float(rd[ok].quantile(0.25)),
                                           "q75": float(rd[ok].quantile(0.75)), "n_used": int(ok.sum()),
                                           "share_above_0.5": float((rd[ok] > 0.5).mean())}
    return out


def washout(days=365.25):
    return lambda r: r["zk"] >= r["first"] + days
