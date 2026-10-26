#!/usr/bin/env python
"""Diagnostics D1, D2, D4, D5, D6 from the observation process report, on trained Stage 4 models.
D1  read-out triangulation on the same test anchors:
      A  NO_EVENT query token at the anchor, independent clocks 1 - exp(-lambda_k H)   [current]
      B  NO_EVENT query, first-event formula (lambda_k / Lambda)(1 - exp(-Lambda H))
      C  NO query: rates at the last real token before the anchor, independent clocks
      D  trajectory rollouts from the query (subsample)
D2  mean log(O/E) at H = 0.5, 1, 2, 3, 5 years for read-outs A and C
D4  share of training NO_EVENT tokens that fall before the first real record (by mode)
D5  exposure ledger without any model: implied rates under Delphi exposure (birth to last event)
    versus the observed EHR window (entry to end), per outcome
D6  post-hoc per-outcome multiplier fitted on VALIDATION anchors, then test O/E by visit quintile
Only aggregates are printed or saved.
Usage: RUN_MODE=full TAGS=delphi_s0_H5:delphi:delphi,observed_s0_H5:observed:delphi python diag_readout.py
"""
import json, os, sys, tempfile, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from delphi_aou import config as common
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3
from delphi_aou.train import delphi as s4
from delphi_aou.models.delphi import Delphi, DelphiConfig, PAD, NO_EVENT
from delphi_aou.eval.calibration import log_oe

CFG = common.load_config()
DAY = 365.25


def history_batch(data, rows, cutoffs, mode, query_ages, device):
    """Sequences with events <= cutoff (fixed tie order). query_ages None: no query token."""
    seqs = []
    for k, i in enumerate(rows):
        t, a = data.tok[i], data.age[i]
        keep = a <= cutoffs[k]
        t, a = t[keep], a[keep]
        start = 0.0 if mode == "delphi" else data.entry[i]
        tt, aa = [data.sex[i]] + list(t), [start] + list(a)
        if query_ages is not None:
            tt.append(NO_EVENT); aa.append(query_ages[k])
        tt, aa = tt[-255:], aa[-255:]
        seqs.append((tt, aa))
    T = max(len(q[0]) for q in seqs)
    tok = np.zeros((len(rows), T), np.int64); age = np.zeros((len(rows), T), np.float32)
    for j, (t, a) in enumerate(seqs):
        tok[j, :len(t)] = t; age[j, :len(a)] = a; age[j, len(a):] = a[-1]
    return torch.tensor(tok, device=device), torch.tensor(age, device=device)


@torch.no_grad()
def readout_rates(model, data, anchors, names, mode, gap_days, with_query, device, batch=1024):
    cols = torch.tensor([data.stoi[t] for t in names], device=device)
    rows = anchors["person_id"].map(data.row).to_numpy()
    a = anchors["anchor_age_days"].to_numpy(float) / DAY
    lam = np.zeros((len(anchors), len(names)), np.float64); Lam = np.zeros(len(anchors))
    for s in range(0, len(anchors), batch):
        idx = np.arange(s, min(s + batch, len(anchors)))
        tok, age = history_batch(data, rows[idx], a[idx] - gap_days / DAY, mode, a[idx] if with_query else None, device)
        r = model.rates_at_last(tok, age)
        lam[idx] = r[:, cols].double().cpu().numpy()
        Lam[idx] = r.sum(-1).double().cpu().numpy()
    return lam, Lam


def risks(lam, Lam, H, formula):
    if formula == "independent":
        return 1 - np.exp(-lam * H)
    return lam / np.maximum(Lam, 1e-12)[:, None] * (1 - np.exp(-Lam[:, None] * H))


def noevent_audit(data, mode, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    tr = np.flatnonzero(data.split == "train")
    pick = rng.choice(tr, min(n, len(tr)), replace=False)
    before = total = 0
    for i in pick:
        t, a, _ = data.sequence(i, rng, mode, 256)
        first = data.age[i].min() if len(data.age[i]) else np.inf
        m = t == NO_EVENT
        total += int(m.sum()); before += int((a[m] < first).sum())
    return before / max(total, 1)


def exposure_ledger(data, names):
    """Per outcome: implied rate under Delphi exposure (0 to last event) / rate in the EHR window."""
    tr = np.flatnonzero(data.split == "train")
    ex_d = ex_w = 0.0
    ev_all = {k: 0 for k in names}; ev_win = {k: 0 for k in names}
    sid = {data.stoi[k]: k for k in names if k in data.stoi}
    for i in tr:
        a = data.age[i]
        if len(a) == 0:
            continue
        ex_d += a.max()
        ex_w += max(data.end[i] - data.entry[i], 0)
        for tkn, ag in zip(data.tok[i], a):
            k = sid.get(int(tkn))
            if k is not None:
                ev_all[k] += 1
                if data.entry[i] <= ag <= data.end[i]:
                    ev_win[k] += 1
    ratios = [np.log((ev_all[k] / ex_d) / (ev_win[k] / ex_w)) for k in names if ev_win[k] >= 25 and ev_all[k] > 0]
    return {"exposure_delphi_over_window": ex_d / max(ex_w, 1e-9), "mean_log_rate_ratio_delphi_vs_window": float(np.mean(ratios)),
            "n_outcomes": len(ratios)}


def run(coh, fo, models, names, visit_q=None, val_visit=None, device="cpu", rollout_n=6000, log=print):
    data = s4.Data(coh, fo)
    an = s3.build_anchors(coh)
    te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    va = an[an["split"] == "val"].reset_index(drop=True); va["anchor_id"] = np.arange(len(va))
    gap = CFG["evaluation"]["gap_days"]
    lab = {H: s3.attach(te, fo, names[:-1], names[:-1], H, gap) for H in (0.5, 1, 2, 3, 5)}
    Yv, _, Ev, _ = s3.attach(va, fo, names[:-1], names[:-1], 5, gap)
    out = {"D4_noevent_before_first_record": {m: noevent_audit(data, m) for m in ("delphi", "observed")},
           "D5_exposure": exposure_ledger(data, names[:-1])}
    log(json.dumps(out, default=float))
    for tag, (model, mode) in models.items():
        r = {}
        lamA, LamA = readout_rates(model, data, te, names, mode, gap, True, device)
        lamC, LamC = readout_rates(model, data, te, names, mode, gap, False, device)
        prof = {}
        for H in (0.5, 1, 2, 3, 5):
            Y, P, E, _ = lab[H]
            prof[str(H)] = {
                "A_query_indep": float(np.nanmean(log_oe(risks(lamA, LamA, H, "independent")[:, :-1], Y[:, :-1], E[:, :-1]))),
                "B_query_firstevent": float(np.nanmean(log_oe(risks(lamA, LamA, H, "first")[:, :-1], Y[:, :-1], E[:, :-1]))),
                "C_lasttoken_indep": float(np.nanmean(log_oe(risks(lamC, LamC, H, "independent")[:, :-1], Y[:, :-1], E[:, :-1])))}
        r["D1_D2_mean_log_OE_by_horizon"] = prof
        sub = te.sample(min(rollout_n, len(te)), random_state=0)
        Y5, P5, E5, _ = lab[5]
        rs = s4.sample_risk(model, data, sub, names, mode, 5, gap, n_samples=10, device=device, chunk=512)
        ii = sub.index.to_numpy()
        r["D1_rollout_H5_mean_log_OE"] = float(np.nanmean(log_oe(rs[:, :-1], Y5[ii, :-1], E5[ii, :-1])))
        r["total_rate_query_vs_lasttoken_median"] = [float(np.median(LamA)), float(np.median(LamC))]
        # D6: per outcome multiplier from validation anchors, read-out A, H = 5
        lamV, LamV = readout_rates(model, data, va, names, mode, gap, True, device)
        pV = risks(lamV, LamV, 5, "independent")
        mult = np.exp(np.nan_to_num(log_oe(pV, Yv, Ev, min_cases=25)[:-1] if False else log_oe(pV[:, :-1], Yv[:, :-1], Ev[:, :-1], min_cases=25)))
        pT = risks(lamA, LamA, 5, "independent")[:, :-1] * mult[None, :]
        pT = np.clip(pT, 0, 1)
        r["D6_recalibrated_test_mean_log_OE"] = float(np.nanmean(log_oe(pT, Y5[:, :-1], E5[:, :-1])))
        if visit_q is not None:
            r["D6_recalibrated_by_visit_quintile"] = {str(q): float(np.nanmean(log_oe(pT, Y5[:, :-1], E5[:, :-1], visit_q == q)))
                                                     for q in range(5)}
            r["A_by_visit_quintile"] = {str(q): float(np.nanmean(log_oe(risks(lamA, LamA, 5, "independent")[:, :-1], Y5[:, :-1], E5[:, :-1], visit_q == q)))
                                        for q in range(5)}
        out[tag] = r
        log(tag + " " + json.dumps(r, default=float))
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tmp = Path(tempfile.gettempdir())
    def fetch(n):
        subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/{n}", str(tmp / n)], check=True); return tmp / n
    z = np.load(fetch("s3_scores_H5.npz")); names = [str(x) for x in z["names"]]
    coh = s2._read_local_from_cache("s2_cohort"); fo = s2._read_local_from_cache("s2_fo_events")
    data = s4.Data(coh, fo)
    models = {}
    for spec in os.environ["TAGS"].split(","):
        tag, mode, objective = spec.split(":")
        c = DelphiConfig(vocab_size=len(data.itos), n_layer=12, n_head=12, n_embd=120, objective=objective, mask_ties=True)
        m = Delphi(c, ignore_in_rate=[PAD] + data.sex_ids + ([NO_EVENT] if objective == "pp" else []))
        m.load_state_dict(torch.load(fetch(f"s4_model_{tag}.pt"), map_location=dev)); m.to(dev).eval()
        models[tag] = (m, mode)
    an = s3.build_anchors(coh); te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    V = s3.visit_features(te, s2.read_parquet_chunks("x_visits", ["person_id", "date", "visit_concept_id"]), CFG["evaluation"]["gap_days"])
    vq = pd.qcut(pd.Series(V[:, 0]).rank(method="first"), 5, labels=False).to_numpy()
    out = run(coh, fo, models, names, vq, device=dev, log=lambda s: print(s, flush=True))
    common.write_results(tmp / "diag_readout.json", out)
    subprocess.run(["gsutil", "-q", "cp", str(tmp / "diag_readout.json"), f"{s2.OUT_DIR}/"], check=True)
    print("DIAG READOUT COMPLETE")


if __name__ == "__main__":
    main()
