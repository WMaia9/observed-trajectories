#!/usr/bin/env python
"""Stage 4: train a Delphi-2M style model on the Stage 2 tokens and score it on EXACTLY the
Stage 3 test anchors and labels, so it is compared with the baselines on the same people.

Two likelihood modes (the paper's ablation axis):
  MODE=delphi    faithful: sex token at birth, 20 no event tokens uniform over 0 to 100 years kept
                 only before the last real event, no censoring term.
  MODE=observed  observation aware: entry at the start of EHR observation (left truncation), no
                 event tokens only inside the observed window up to the end of follow up, and
                 CENSOR=1 adds the survival term from the last token to the end of follow up.
Usage (sample first!):
  RUN_MODE=test MODE=delphi ITERS=2000 N_LAYER=4 N_EMBD=64 N_HEAD=4 python stage4_delphi.py
"""
import json, os, sys, time, tempfile, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from delphi_aou import config as common
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3
from delphi_aou.models.delphi import Delphi, DelphiConfig, PAD, NO_EVENT

CFG = common.load_config()
DAY = 365.25


def _log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------- data
class Data:
    """Per patient arrays: token ids and ages (years), sex token id, observation entry and end."""
    def __init__(self, cohort, fo):
        c = cohort.sort_values("person_id").reset_index(drop=True)
        sex_tokens = sorted(c["sex_token"].astype(str).unique())
        dis = sorted(set(fo["token"]) - {"DEATH"})
        self.itos = ["PAD", "NO_EVENT"] + sex_tokens + dis + ["DEATH"]
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.sex_ids = [self.stoi[t] for t in sex_tokens]
        bd = pd.to_datetime(c["birth_date"])
        self.pids = c["person_id"].to_numpy()
        self.row = {p: i for i, p in enumerate(self.pids)}
        self.split = c["split"].to_numpy()
        self.sex = c["sex_token"].astype(str).map(self.stoi).to_numpy()
        self.entry = ((pd.to_datetime(c["obs_start"]) - bd).dt.days / DAY).to_numpy()
        end = ((pd.to_datetime(c["obs_end"]) - bd).dt.days / DAY).to_numpy()
        death = c["death_age_days"].astype(float).to_numpy() / DAY
        self.end = np.where(np.isnan(death), end, np.minimum(end, death))
        f = fo[fo["person_id"].isin(self.row)].copy()
        f["r"] = f["person_id"].map(self.row)
        f["t"] = f["token"].map(self.stoi)
        f = f.sort_values(["r", "age_days", "t"])
        r = f["r"].to_numpy()
        cuts = np.searchsorted(r, np.arange(len(self.pids) + 1))
        tok, age = f["t"].to_numpy(np.int64), f["age_days"].to_numpy(float) / DAY
        self.tok = [tok[cuts[i]:cuts[i + 1]] for i in range(len(self.pids))]
        self.age = [age[cuts[i]:cuts[i + 1]] for i in range(len(self.pids))]

    def sequence(self, i, rng, mode, max_len, cutoff=None, query_age=None, train_window=None):
        """One input sequence. For training: random tie order and random no event tokens.
        For scoring: cutoff (years) keeps events <= cutoff, ties in fixed order, and a single
        NO_EVENT query token is appended at query_age."""
        t, a = self.tok[i], self.age[i]
        if cutoff is not None:
            keep = a <= cutoff
            t, a = t[keep], a[keep]
        if rng is not None and len(a):
            o = np.lexsort((rng.random(len(a)), a))            # shuffle inside ties only
            t, a = t[o], a[o]
        if mode == "delphi":
            start = 0.0
        else:
            start = self.entry[i]
        if query_age is None:
            if mode == "delphi":
                ne = rng.uniform(0, 100, 20)
                ne = ne[ne < (a.max() if len(a) else 0)]
            else:
                span = max(self.end[i] - self.entry[i], 0)
                ne = rng.uniform(self.entry[i], self.end[i], max(1, int(round(span / 5))))
        else:
            ne = np.array([query_age])
        tt = np.concatenate([t, np.full(len(ne), NO_EVENT)])
        aa = np.concatenate([a, ne])
        o = np.argsort(aa, kind="stable")
        tt, aa = tt[o], aa[o]
        if query_age is not None:                               # query token must be last
            assert tt[-1] == NO_EVENT and aa[-1] == query_age
        self._tail = True
        if query_age is None and train_window and len(tt) > train_window - 1:
            st = rng.integers(0, len(tt) - (train_window - 1) + 1)      # random window, as in Delphi
            self._tail = st + train_window - 1 >= len(tt)               # censor term only if the window keeps the end
            tt, aa = tt[st:st + train_window - 1], aa[st:st + train_window - 1]
        if len(tt) > max_len - 1:
            tt, aa = tt[-(max_len - 1):], aa[-(max_len - 1):]
        return np.concatenate([[self.sex[i]], tt]), np.concatenate([[start], aa]), self._tail


    def batch(self, idx, rng, mode, max_len, device, **kw):
        seqs = [self.sequence(i, rng, mode, max_len, **kw) for i in idx]
        T = max(len(s[0]) for s in seqs)
        tok = np.zeros((len(idx), T), np.int64); age = np.zeros((len(idx), T), np.float32)
        for k, (t, a, _) in enumerate(seqs):
            tok[k, :len(t)] = t; age[k, :len(a)] = a
            age[k, len(a):] = a[-1]                             # padding keeps the last age
        end = np.where([s[2] for s in seqs], self.end[idx], np.nan)  # NaN: window does not reach the end
        end = torch.tensor(end, dtype=torch.float32, device=device)
        return torch.tensor(tok, device=device), torch.tensor(age, device=device), end


# --------------------------------------------------------------------------- training
def train(data, mode, mcfg, iters=2000, batch=128, lr=6e-4, seed=0, device="cpu", eval_every=250, log=_log):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ignore = [PAD] + data.sex_ids + ([NO_EVENT] if (mcfg.objective == "pp" or mcfg.censor) else [])
    model = Delphi(mcfg, ignore_in_rate=ignore).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.99))
    warm = max(1, iters // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(s, iters) / iters))))
    tr = np.flatnonzero(data.split == "train")
    va = np.flatnonzero(data.split == "val")
    vrng = np.random.default_rng(12345)
    val_batches = [vrng.choice(va, min(batch, len(va)), replace=False) for _ in range(8)]
    def val_loss():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for k, b in enumerate(val_batches):
                tk, ag, en = data.batch(b, np.random.default_rng(k), mode, mcfg.max_len, device,
                                        train_window=mcfg.train_window)
                tot += float(model.loss(tk, ag, en)["loss"])
        model.train(); return tot / len(val_batches)
    best, best_state, hist = float("inf"), None, []
    t0 = time.time()
    v0 = val_loss(); hist.append((0, v0)); log(f"iter 0 val {v0:.4f}")
    for it in range(1, iters + 1):
        b = rng.choice(tr, batch, replace=False)
        tk, ag, en = data.batch(b, rng, mode, mcfg.max_len, device, train_window=mcfg.train_window)
        out = model.loss(tk, ag, en)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % eval_every == 0 or it == iters:
            v = val_loss(); hist.append((it, v))
            if v < best:
                best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
            log(f"iter {it} train {float(out['loss'].detach()):.4f} (ce {float(out['ce']):.3f} time {float(out['time']):.3f}"
                + (f" cens {float(out['censor']):.3f}" if "censor" in out else "") + f") val {v:.4f} "
                f"t+{time.time()-t0:,.0f}s")
    model.load_state_dict(best_state)                           # selected on VALIDATION loss only
    return model, hist


@torch.no_grad()
def score(model, data, anchors, token_cols, mode, horizon_years, gap_days, device="cpu", batch=512):
    """Risk over the horizon for each anchor and each column token, using events <= anchor - gap."""
    model.eval()
    cols = torch.tensor([data.stoi[t] for t in token_cols], device=device)
    rows = anchors["person_id"].map(data.row).to_numpy()
    a_years = anchors["anchor_age_days"].to_numpy(float) / DAY
    out = np.zeros((len(anchors), len(cols)), np.float32)
    for s in range(0, len(anchors), batch):
        idx = np.arange(s, min(s + batch, len(anchors)))
        seqs = [data.sequence(rows[k], None, mode, model.c.max_len, cutoff=a_years[k] - gap_days / DAY,
                              query_age=a_years[k]) for k in idx]
        T = max(len(q[0]) for q in seqs)
        tok = np.zeros((len(idx), T), np.int64); age = np.zeros((len(idx), T), np.float32)
        for j, (t, a, _) in enumerate(seqs):
            tok[j, :len(t)] = t; age[j, :len(a)] = a; age[j, len(a):] = a[-1]
        r = model.rates_at_last(torch.tensor(tok, device=device), torch.tensor(age, device=device))[:, cols]
        out[idx] = (1 - torch.exp(-r * horizon_years)).cpu().numpy()
    return out


def run(cohort, fo, mode="delphi", censor=False, horizon_years=5, n_diseases=60, iters=2000, batch=128,
        n_layer=4, n_embd=64, n_head=4, lr=6e-4, seed=0, device="cpu", baseline_npz=None, log=_log, cfg=CFG):
    data = Data(cohort, fo)
    mcfg = DelphiConfig(vocab_size=len(data.itos), n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                        objective="pp" if censor else "delphi", mask_ties=True)
    t_iter = None
    log(f"vocab {len(data.itos)} patients {len(data.pids):,} params "
        f"{sum(p.numel() for p in Delphi(mcfg).parameters())/1e6:.2f}M mode={mode} objective={mcfg.objective}")
    model, hist = train(data, mode, mcfg, iters, batch, lr, seed, device, log=log)
    # the SAME anchors, diseases and labels as stage 3
    anchors = s3.build_anchors(cohort, cfg)
    te = anchors[anchors["split"] == "test"].reset_index(drop=True)
    te["anchor_id"] = np.arange(len(te))
    if baseline_npz is not None:
        names = [str(x) for x in baseline_npz["names"]]
        assert np.array_equal(baseline_npz["person_id"], te["person_id"].to_numpy()), "anchors differ from stage 3"
        assert np.allclose(baseline_npz["anchor_age_days"], te["anchor_age_days"].to_numpy())
    else:
        tr_ids = set(anchors.loc[anchors["split"] == "train", "person_id"])
        prev = fo[fo["person_id"].isin(tr_ids) & (fo["token"] != "DEATH")].groupby("token")["person_id"].nunique()
        names = list(prev.sort_values(ascending=False).index[:n_diseases]) + ["DEATH"]
    gap = cfg["evaluation"]["gap_days"]
    Y, P, E, H = s3.attach(te, fo, names[:-1], names[:-1], horizon_years, gap)
    sc = score(model, data, te, names, mode, horizon_years, gap, device)
    ev = cfg["evaluation"]
    bands = [tuple(b) for b in ev["age_bands"]]
    tab = eval_auc.evaluate_auc(sc, Y, E, te["age_years"].to_numpy(), te["sex"].to_numpy(), names, bands,
                                ev["min_cases"], ev["min_controls"])
    res = {"summary": eval_auc.summarize(tab), "val_loss_curve": hist}
    if baseline_npz is not None:
        for ref in ["gbm_visits", "gbm", "logreg"]:
            key = f"score_{ref}"
            if key in baseline_npz.files:
                res[f"paired_vs_{ref}"] = eval_auc.compare_models(
                    sc, baseline_npz[key], Y, E, te["age_years"].to_numpy(), te["sex"].to_numpy(), names, bands,
                    ev["min_cases"], ev["min_controls"])
    log(json.dumps({k: v for k, v in res.items() if k != "val_loss_curve"}, default=float))
    return model, res, sc, tab


def main():
    mode = os.environ.get("MODE", "delphi")
    censor = os.environ.get("CENSOR", "0") == "1" or os.environ.get("OBJECTIVE", "delphi") == "pp"
    H = float(os.environ.get("HORIZON", "5"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count() or 4)))
    coh = s2._read_local_from_cache("s2_cohort")
    fo = s2._read_local_from_cache("s2_fo_events")
    npz = None
    try:
        local = Path(tempfile.gettempdir()) / f"s3_scores_H{int(H)}.npz"
        if s2.OUT_DIR.startswith("gs://"):
            subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/s3_scores_H{int(H)}.npz", str(local)], check=True)
        else:
            local = Path(s2.OUT_DIR) / f"s3_scores_H{int(H)}.npz"
        npz = np.load(local, allow_pickle=False)
    except Exception as e:
        print("no stage 3 scores found, paired comparison skipped:", e)
    model, res, sc, tab = run(coh, fo, mode, censor, H, int(os.environ.get("N_DISEASES", "100")),
                              int(os.environ.get("ITERS", "2000")), int(os.environ.get("BATCH", "128")),
                              int(os.environ.get("N_LAYER", "4")), int(os.environ.get("N_EMBD", "64")),
                              int(os.environ.get("N_HEAD", "4")), float(os.environ.get("LR", "6e-4")),
                              int(os.environ.get("SEED", "0")), dev, npz)
    tag = f"{mode}{'_pp' if censor else ''}_s{os.environ.get('SEED', '0')}_H{int(H)}"
    sh = common.split_hash(coh["person_id"].to_numpy(), coh["split"].to_numpy())
    tmp = Path(tempfile.gettempdir())
    payload = common.write_results(tmp / f"s4_summary_{tag}.json", {"tag": tag, "device": dev, **res}, CFG, sh)
    np.save(tmp / f"s4_scores_{tag}.npy", sc)
    torch.save(model.state_dict(), tmp / f"s4_model_{tag}.pt")
    tab.to_csv(tmp / f"s4_auc_{tag}.csv", index=False)
    if s2.OUT_DIR.startswith("gs://"):
        subprocess.run(f"gsutil -q cp {tmp}/s4_*_{tag}.* {s2.OUT_DIR}/", shell=True, check=True)
    print(json.dumps({k: v for k, v in payload["results"].items() if k != "val_loss_curve"}, indent=1, default=float))
    print("STAGE 4 COMPLETE")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- absolute risk by simulation
@torch.no_grad()
def sample_risk(model, data, anchors, token_cols, mode, horizon_years, gap_days, n_samples=20, device="cpu",
                chunk=4096, max_steps=48, update_years=1.0, seed=0):
    """Absolute risk of each column token within the horizon by SIMULATING future trajectories from the
    anchor (as Delphi does for long term predictions), instead of treating the instantaneous rate at the
    anchor as constant. Each step: rate = sum exp(logits) over allowed tokens, waiting time ~ Exp(rate),
    token ~ softmax. Tokens already present cannot occur again (first occurrences). If no event happens
    within update_years, a NO_EVENT token is inserted (exact under the memoryless property) so the
    hazard is refreshed as the person ages. Stops at the horizon or at DEATH."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    V = model.c.vocab_size
    cols = torch.tensor([data.stoi[t] for t in token_cols], device=device)
    death = data.stoi.get("DEATH", -1)
    rows = anchors["person_id"].map(data.row).to_numpy()
    a_years = anchors["anchor_age_days"].to_numpy(float) / DAY
    base = [data.sequence(rows[k], None, mode, model.c.max_len - max_steps, cutoff=a_years[k] - gap_days / DAY,
                          query_age=a_years[k]) for k in range(len(anchors))]
    out = np.zeros((len(anchors), len(cols)), np.float32)
    per_chunk = max(1, chunk // n_samples)
    for s in range(0, len(anchors), per_chunk):
        idx = np.arange(s, min(s + per_chunk, len(anchors)))
        seqs = [base[k] for k in idx for _ in range(n_samples)]
        B = len(seqs)
        L0 = np.array([len(q[0]) for q in seqs])
        T = int(L0.max()) + max_steps
        tok = torch.zeros((B, T), dtype=torch.long, device=device)
        age = torch.zeros((B, T), dtype=torch.float32, device=device)
        for j, (t, a, _) in enumerate(seqs):
            tok[j, :len(t)] = torch.as_tensor(t, device=device); age[j, :len(a)] = torch.as_tensor(a, dtype=torch.float32, device=device)
            age[j, len(a):] = float(a[-1])
        L = torch.as_tensor(L0, device=device)
        start = age[torch.arange(B, device=device), L - 1].clone()
        end = start + horizon_years
        occurred = torch.zeros((B, V), dtype=torch.bool, device=device)
        occurred.scatter_(1, tok, True)
        occurred[:, PAD] = False; occurred[:, NO_EVENT] = False
        hit = torch.zeros((B, len(cols)), dtype=torch.bool, device=device)
        done = torch.zeros(B, dtype=torch.bool, device=device)
        ar = torch.arange(B, device=device)
        for step in range(max_steps):
            if bool(done.all()):
                break
            Tm = int(L.max())
            logits = model(tok[:, :Tm], age[:, :Tm])[ar, L - 1]
            logits = logits.masked_fill(occurred, -1e4)
            rate = torch.exp(torch.logsumexp(logits, -1))
            dt = -torch.log(torch.rand(B, generator=g, device=device).clamp_min(1e-12)) / rate.clamp_min(1e-12)
            now = age[ar, L - 1]
            refresh = dt > update_years
            t_new = torch.where(refresh, now + update_years, now + dt)
            nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).squeeze(1)
            nxt = torch.where(refresh, torch.full_like(nxt, NO_EVENT), nxt)
            finished = done | (t_new > end)
            write = ~finished
            pos = L.clamp(max=T - 1)
            tok[ar[write], pos[write]] = nxt[write]
            age[ar[write], pos[write]] = t_new[write]
            age[ar[write], (pos[write] + 1).clamp(max=T - 1)] = t_new[write]
            L = torch.where(write, (L + 1).clamp(max=T - 1), L)
            ev = write & (nxt != NO_EVENT)
            occurred[ar[ev], nxt[ev]] = True
            hit |= ev.unsqueeze(1) & (nxt.unsqueeze(1) == cols.unsqueeze(0))
            done = finished | (write & (nxt == death)) | (L >= T - 1)
        out[idx] = hit.float().view(len(idx), n_samples, -1).mean(1).cpu().numpy()
    return out
