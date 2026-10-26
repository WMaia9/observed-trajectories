"""Calibration by utilization of the OFFICIAL Delphi-2M checkpoint in the anchored protocol (P2 review weakness 3).

The calibration results in the paper so far come from our Delphi-style reimplementations. Here the checkpoint trained
with the unchanged official code on All of Us (~/delphi_rep_runs/full/ckpt/ckpt.pt) is scored on the anchored test
anchors of the paper (same anchors, outcomes, labels Y and eligibility E as every other model), restricted to people in
the official validation split, so the checkpoint never trained on them. The same rows give a fidelity table: the
official model next to the reimplementations and the baselines, with macro AUC, overall log(O/E), log(O/E) by visit
quintile and the access gap with a patient bootstrap interval.

Input for each anchor, in the official format (utils.get_batch): the person's rows of val.bin with age <= anchor - gap,
regular "no event" tokens every 5 years up to the anchor (the official padding="regular"), at most 80 tokens with the
sex token kept, tokens shifted by +1 (0 is padding), ages in days. Two read-outs, as in the diagnostics of the paper:
  query: a "no event" token is appended at the anchor age and its rates are used (what the paper reports for the
         reimplementations);
  last:  rates at the last real token before the anchor.
Risk over H years: 1 - exp(-exp(logit) * H * 365.25) (the official rates are per day).

Only aggregates are written (~/results/official_anchored_calibration_H5.json); log_oe drops cells under the policy
minimum. Usage: RUN_MODE=full [TAGS=delphi_s0_H5,...] [N_BOOT=200] python -u experiments/official_anchored_calibration.py
"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

DAY = 365.25
NO_EVENT_BIN = 0      # ids as stored in the .bin files; the model sees id + 1


def query_sequence(ages, toks, cutoff_days, query_days, block=80, pad_every_years=5.0):
    """One anchored input in the official format. ages, toks: the person's .bin rows (sorted by age).
    Returns (model token ids, ages in days, index of the last real token). The query token is last."""
    keep = ages <= cutoff_days
    a, t = ages[keep].astype(np.float32), toks[keep].astype(np.int64)
    pad = np.arange(1.0, query_days, DAY * pad_every_years, dtype=np.float32)
    pad = pad[pad <= (a.max() if len(a) else 0.0)]                 # official: no padding after the last real token
    a = np.concatenate([a, pad]); t = np.concatenate([t, np.full(len(pad), NO_EVENT_BIN)])
    o = np.argsort(a, kind="stable"); a, t = a[o], t[o]
    a = np.append(a, np.float32(query_days)); t = np.append(t, NO_EVENT_BIN)
    if len(t) > block:                                             # keep the sex token (first row, age 0)
        t = np.concatenate([t[:1], t[-(block - 1):]]); a = np.concatenate([a[:1], a[-(block - 1):]])
    real = np.nonzero(t[:-1] != NO_EVENT_BIN)[0]
    last = int(real[-1]) if len(real) else 0
    return t + 1, a, last


def history_tokens(val, persons, cutoff_days):
    """Number of a person's .bin rows (sex, lifestyle, first diagnoses) with age <= cutoff, per anchor."""
    order = np.lexsort((val[:, 1], val[:, 0])); v = val[order]
    starts = np.searchsorted(v[:, 0], persons, "left"); ends = np.searchsorted(v[:, 0], persons, "right")
    return np.array([np.searchsorted(v[s:e, 1], c, "right") for s, e, c in zip(starts, ends, cutoff_days)])


@torch.no_grad()
def score_official(model, val, persons, cutoff, query, cols, H, device, batch=512, block=80, pad_years=5.0, raw=False):
    """Risks (N x len(cols)) for the query and last-real-token read-outs. cols: model ids (NaN columns allowed: -1).
    Risk is computed as -expm1(-rate H) in float64, so very small rates do not round to exactly 0 (ties).
    raw=True returns the logits (log rates per day) instead."""
    order = np.lexsort((val[:, 1], val[:, 0]))
    v = val[order]
    starts = np.searchsorted(v[:, 0], persons, "left"); ends = np.searchsorted(v[:, 0], persons, "right")
    ok_cols = cols >= 0
    cidx = torch.tensor(cols[ok_cols], device=device)
    outq = np.full((len(persons), len(cols)), np.nan, np.float64); outl = outq.copy()
    for s in range(0, len(persons), batch):
        idx = range(s, min(s + batch, len(persons)))
        seqs = [query_sequence(v[starts[k]:ends[k], 1], v[starts[k]:ends[k], 2], cutoff[k], query[k], block, pad_years)
                for k in idx]
        T = max(len(q[0]) for q in seqs)
        tok = np.zeros((len(seqs), T), np.int64); age = np.full((len(seqs), T), -10000.0, np.float32)
        lastpos = np.zeros(len(seqs), np.int64)
        for j, (t, a, l) in enumerate(seqs):                       # left padding, as in get_batch
            tok[j, T - len(t):] = t; age[j, T - len(a):] = a; lastpos[j] = T - len(t) + l
        logits = model(torch.tensor(tok, device=device), torch.tensor(age, device=device))[0]
        lq = logits[:, -1, :][:, cidx].double()
        ll = logits[torch.arange(len(seqs), device=device), torch.tensor(lastpos, device=device)][:, cidx].double()
        sl = slice(s, s + len(seqs))
        f = (lambda v: v) if raw else (lambda v: -torch.expm1(-torch.exp(v) * H * DAY))
        outq[sl][:, ok_cols] = f(lq).cpu().numpy()
        outl[sl][:, ok_cols] = f(ll).cpu().numpy()
    return outq, outl


def outcome_columns(names, label_dict):
    """Model ids (bin id + 1) for the paper's outcome tokens ('D=E11', ..., 'DEATH'); -1 if outside the vocabulary."""
    out = []
    for n in names:
        key = "Death" if n == "DEATH" else n.split("=", 1)[-1][:3].upper()
        out.append(label_dict[key] + 1 if key in label_dict else -1)
    return np.array(out, np.int64)


def main():
    from delphi_aou import config as common
    from delphi_aou.data import cohort as s2
    from delphi_aou.baselines import classical as s3
    from delphi_aou.eval import metrics as eval_auc
    from delphi_aou.eval.calibration import analyse
    from delphi_aou.replication.evaluate import OFFICIAL
    from delphi_aou.replication.to_delphi_bin import label_dict
    CFG = common.load_config()
    H = int(float(os.environ.get("HORIZON", "5")))
    device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    home = Path.home()
    data = Path(os.environ.get("DELPHI_DATA", home / "delphi_rep_data"))
    ckpt = Path(os.environ.get("DELPHI_CKPT", home / "delphi_rep_runs/full/ckpt/ckpt.pt"))
    tmp = Path(tempfile.gettempdir())

    def fetch(name):
        local = tmp / name
        if s2.OUT_DIR.startswith("gs://"):
            r = subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/{name}", str(local)])
            return local if r.returncode == 0 else None
        local = Path(s2.OUT_DIR) / name
        return local if local.exists() else None

    z = np.load(fetch(f"s3_scores_H{H}.npz"), allow_pickle=False)
    names = [str(x) for x in z["names"]]
    Y, E, person = z["Y"], z["E"], z["person_id"]
    models = {k[6:]: z[k] for k in z.files if k.startswith("score_")}
    for t in [t for t in os.environ.get("TAGS", "delphi_s0_H5,observed_s0_H5,observed_pp_s0_H5").split(",") if t]:
        f = fetch(f"s4_scores_{t}.npy")
        if f is not None:
            models[t] = np.load(f)

    # people the official checkpoint never trained on
    val = np.fromfile(data / "val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    tr_ids = np.unique(np.fromfile(data / "train.bin", dtype=np.uint32).reshape(-1, 3)[:, 0])
    val_ids = np.unique(val[:, 0])
    assert len(np.intersect1d(tr_ids, val_ids)) == 0
    pbin = person % 2 ** 32           # the .bin files store person ids as uint32 (12-digit ids wrap, order kept)
    rows = np.isin(pbin, val_ids)
    assert len(np.unique(pbin)) == len(np.unique(person))
    print(f"anchors {len(person):,}; in the official validation split {rows.sum():,} "
          f"({rows.mean():.1%}); in its training split {np.isin(pbin, tr_ids).sum():,}", flush=True)

    # optional (reviewer round 9): keep only anchors whose history before the cutoff fits in the training window of the
    # official script (each person's first 97 tokens, select='left'); MAX_HIST_TOKENS=96 counts the person's .bin rows
    # (sex, lifestyle and first diagnoses) up to the cutoff, before padding.
    K = int(os.environ.get("MAX_HIST_TOKENS", "0"))
    gap0 = CFG["evaluation"]["gap_days"]
    if K:
        nb = history_tokens(val, pbin, z["anchor_age_days"].astype(float) - gap0)
        inside = nb <= K
        print(f"HISTORY anchors in the validation split {rows.sum():,}; history within {K} tokens "
              f"{(rows & inside).sum():,} ({(rows & inside).sum() / max(rows.sum(), 1):.1%}); "
              f"median tokens before the anchor {float(np.median(nb[rows])):.0f}", flush=True)
        rows = rows & inside

    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    ck = torch.load(ckpt, map_location=device)
    m = Delphi(DelphiConfig(**ck["model_args"])); m.load_state_dict(ck["model"]); m.eval().to(device)
    cols = outcome_columns(names, label_dict())
    print(f"outcomes {len(names)}; outside the official vocabulary {(cols < 0).sum()}: "
          f"{[n for n, c in zip(names, cols) if c < 0]}", flush=True)
    gap = CFG["evaluation"]["gap_days"]
    ad = z["anchor_age_days"].astype(float)
    sq, sl = score_official(m, val, pbin[rows], ad[rows] - gap, ad[rows], cols, H, device,
                            block=int(os.environ.get("BLOCK", "80")), pad_years=float(os.environ.get("PAD_YEARS", "5")))
    # sanity: anchors with an empty history are not silently dropped
    print("query risk: median", float(np.nanmedian(sq)), "last-token risk: median", float(np.nanmedian(sl)), flush=True)

    # restrict everything to the same rows and the outcomes the official vocabulary has
    keep_c = cols >= 0
    Yr, Er = Y[rows][:, keep_c], E[rows][:, keep_c]
    mod = {k: v[rows][:, keep_c] for k, v in models.items()}
    mod["official_query"] = sq[:, keep_c]; mod["official_last"] = sl[:, keep_c]
    rnames = [n for n, k in zip(names, keep_c) if k]

    coh = s2._read_local_from_cache("s2_cohort")
    an = s3.build_anchors(coh)
    te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    assert np.array_equal(te["person_id"].to_numpy(), person)
    V = s3.visit_features(te, s2.read_parquet_chunks("x_visits", ["person_id", "date", "visit_concept_id"]), gap)
    visit_q = pd.qcut(pd.Series(V[:, 0]).rank(method="first"), 5, labels=False).to_numpy()   # thresholds of the full test set
    race = coh.set_index("person_id").loc[person, "race_token"].astype(str).to_numpy()

    # optional: the reimplementations read at their last real token (no query token), on the same rows
    specs = [s for s in os.environ.get("LAST_TAGS", "").split(",") if s]
    if specs:
        from delphi_aou.train import delphi as s4
        from delphi_aou.models.delphi import Delphi as RDelphi, DelphiConfig as RConfig, PAD, NO_EVENT
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from diag_readout import readout_rates, risks
        rdata = s4.Data(coh, s2._read_local_from_cache("s2_fo_events"))
        sub = te[rows].reset_index(drop=True)
        for spec in specs:
            tag, mode, objective = spec.split(":")
            c = RConfig(vocab_size=len(rdata.itos), n_layer=12, n_head=12, n_embd=120, objective=objective, mask_ties=True)
            rm = RDelphi(c, ignore_in_rate=[PAD] + rdata.sex_ids + ([NO_EVENT] if objective == "pp" else []))
            rm.load_state_dict(torch.load(fetch(f"s4_model_{tag}.pt"), map_location=device)); rm.to(device).eval()
            for q, name in [(True, "query_check"), (False, "last")]:
                lam, Lam = readout_rates(rm, rdata, sub, names, mode, gap, q, device, batch=128)
                mod[f"{tag}_{name}"] = risks(lam, Lam, H, "independent")[:, keep_c].astype(np.float32)
    only = [s for s in os.environ.get("ONLY", "").split(",") if s]
    if only:
        mod = {k: v for k, v in mod.items() if any(k.startswith(o) for o in only)}

    ev = CFG["evaluation"]; bands = [tuple(b) for b in ev["age_bands"]]
    disc = {}
    for k, p in mod.items():
        tab = eval_auc.evaluate_auc(p, Yr, Er, z["age_years"][rows], z["sex"][rows], rnames, bands,
                                    ev["min_cases"], ev["min_controls"])
        disc[k] = eval_auc.summarize(tab)
    cal = analyse(mod, Yr, Er, person[rows], visit_q[rows], race[rows], n_boot=int(os.environ.get("N_BOOT", "200")))
    res = {"n_outcomes": int(keep_c.sum()), "share_of_test_anchors": float(rows.mean()),
           "discrimination": disc, "calibration": cal}
    sh = common.split_hash(coh["person_id"].to_numpy(), coh["split"].to_numpy())
    out = home / "results" / f"official_anchored_calibration_H{H}{os.environ.get('OUT_SUFFIX', '')}.json"
    out.parent.mkdir(exist_ok=True)
    common.write_results(out, res, CFG, sh)
    for k in mod:
        c = cal[k]; q = c["by_visit_quintile"]
        print("FIDELITY", json.dumps({"model": k, "macro_auc": disc[k].get("macro_auc"), "overall": c["overall"]["mean_log_oe"],
              "q1": q["0"]["mean_log_oe"], "q5": q["4"]["mean_log_oe"], "gap": c["access_gap_top_minus_bottom"]},
              default=float), flush=True)
    print("OFFICIAL CALIBRATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
