"""Faster equivalent of the official evaluate_auc_pipeline for many protocol settings at once.

Profiling the official pipeline on the full validation set (T4, 4 vCPU): about 8 of every 28 minutes is
model inference with batch 128, about 20 is a single-core numpy loop over diseases and sexes, and the
inference is recomputed for every setting. This module keeps the official per-disease function
(get_calibration_auc, optionally with the horizon line enabled) and the official aggregation, and changes
only the orchestration:
  1. logits for a chunk of diseases are computed once (larger batch) and reused for every setting;
  2. the per-disease loop runs in parallel worker processes (fork, arrays shared copy on write);
  3. each (setting, sex, disease) task seeds numpy from its own key, so results do not depend on the
     number of workers (the official loop draws from one unseeded global numpy stream instead).
"""
import os, sys, zlib
import multiprocessing as mp

import numpy as np
import pandas as pd
import torch

from delphi_aou.replication.evaluate import OFFICIAL

_G = {}


def _task(args):
    ci, sex, j, k = args
    fn, ns = _G["fn"], _G["ns"]
    off, hor = _G["configs"][ci]
    ns["HORIZON_DAYS"] = hor
    np.random.seed(zlib.crc32(f"{ci}-{sex}-{k}".encode()))
    d, p, pidx = _G["sex"][sex]
    out = fn(j, k, d, p, offset=off, age_groups=_G["age_groups"], precomputed_idx=pidx[off],
             n_bootstrap=1, use_delong=True)
    if out is None:
        return []
    for o in out:
        o["sex"] = sex; o["cfg"] = ci
    return out


def _aggregate(rows, labels):
    """Same as the official aggregation: mean DeLong AUC over age brackets, variance sum / n^2."""
    u = pd.DataFrame(rows)
    if u.empty:
        return u, u
    g = u.groupby("token")
    df = pd.DataFrame({"auc": g["auc_delong"].mean(),
                       "auc_variance_delong": g["auc_variance_delong"].sum() / g.size() ** 2,
                       "n_samples": g.size(), "n_diseased": g["n_diseased"].sum(),
                       "n_healthy": g["n_healthy"].sum()}).reset_index()
    sub = labels[["index", "ICD-10 Chapter (short)", "name", "color", "count"]]
    return (u.merge(sub, left_on="token", right_on="index", how="inner"),
            df.merge(labels, left_on="token", right_on="index", how="inner"))


def run(model, d100k, labels, configs, filter_min_total=100, disease_chunk_size=200, batch_size=512,
        age_groups=np.arange(40, 80, 5), workers=None, device="cuda", horizon_fn=None):
    """configs: list of (offset_days, horizon_days or None). Returns {config_index: (df_unpooled, df_both)}."""
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_auc as ev
    if horizon_fn is None:
        from delphi_aou.replication.sweep import patched_calibration_auc
        horizon_fn = patched_calibration_auc(ev)
    fn, ns = horizon_fn
    workers = workers or max(1, (os.cpu_count() or 2))
    diseases = ev.get_common_diseases(labels, filter_min_total)
    chunks = np.array_split(diseases, (len(diseases) + disease_chunk_size - 1) // disease_chunk_size)
    offsets = sorted({c[0] for c in configs})
    pidx_all = {o: ((d100k[1][:, :, None] < d100k[3][:, None, :] - o).sum(1) - 1).cpu().numpy() for o in offsets}
    dn = [x.cpu().numpy() for x in d100k]
    masks = {s: ((dn[0] == idx).sum(1) > 0) for s, idx in [("female", 2), ("male", 3)]}
    rows = []
    torch.manual_seed(1337)
    for ck, chunk in enumerate(chunks):
        parts = []
        with torch.no_grad():
            for dd in zip(*[torch.split(x, batch_size) for x in d100k]):
                dd = [x.to(device) for x in dd]
                parts.append(model(*dd)[0][:, :, torch.as_tensor(chunk, device=dd[0].device)].float().cpu().numpy().astype("float16"))
        p = np.vstack(parts)
        _G.update(fn=fn, ns=ns, configs=configs, age_groups=age_groups,
                  sex={s: ([x[m] for x in dn], p[m], {o: pidx_all[o][m] for o in offsets}) for s, m in masks.items()})
        tasks = [(ci, s, j, int(k)) for ci in range(len(configs)) for s in masks for j, k in enumerate(chunk)]
        if workers == 1:
            res = map(_task, tasks)
        else:
            pool = mp.get_context("fork").Pool(workers)
            res = pool.imap_unordered(_task, tasks, chunksize=8)
        for r in res:
            rows.extend(r)
        if workers != 1:
            pool.close(); pool.join()
        print(f"chunk {ck + 1}/{len(chunks)} done ({len(tasks)} tasks)", flush=True)
        _G.clear()
    out = {}
    for ci in range(len(configs)):
        out[ci] = _aggregate([r for r in rows if r["cfg"] == ci], labels)
    return out
