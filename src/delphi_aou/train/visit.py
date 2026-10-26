#!/usr/bin/env python
"""Stage 5: train the Tier A visit-level model and score it on the SAME Stage 3 anchors and labels.
Usage: RUN_MODE=test ITERS=500 python stage5_train.py   (sample first, then full on the GPU)"""
import json, os, sys, time, tempfile, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from delphi_aou import config as common
from delphi_aou.eval import metrics as eval_auc
from delphi_aou.data import cohort as s2
from delphi_aou.baselines import classical as s3
from delphi_aou.models.visit_pp import VisitPP, VisitConfig, N_VTYPE

CFG = common.load_config()
DAY = 365.25
ENTRY = N_VTYPE - 1
NEVER = 10 ** 9


def _log(*a):
    print(*a, flush=True)


class VisitData:
    def __init__(self, cohort, fo, visits):
        c = cohort.sort_values("person_id").reset_index(drop=True)
        self.codes = sorted(set(fo["token"]) - {"DEATH"})
        self.cidx = {t: i for i, t in enumerate(self.codes)}
        self.K = len(self.codes)
        self.pids = c["person_id"].to_numpy(); self.row = {p: i for i, p in enumerate(self.pids)}
        self.split = c["split"].to_numpy()
        self.sex = c["sex_token"].astype("category").cat.codes.clip(0, 3).to_numpy()
        bd = pd.to_datetime(c["birth_date"])
        self.entry = ((pd.to_datetime(c["obs_start"]) - bd).dt.days / DAY).to_numpy()
        end = ((pd.to_datetime(c["obs_end"]) - bd).dt.days / DAY).to_numpy()
        death = c["death_age_days"].astype(float).to_numpy() / DAY
        self.died = ~np.isnan(death) & (death <= end + 1 / DAY)
        self.exit = np.where(self.died, death, end)
        v = visits[visits["person_id"].isin(self.row)].copy()
        v["r"] = v["person_id"].map(self.row)
        v = v.sort_values(["r", "age_days"])
        f = fo[(fo["token"] != "DEATH") & fo["person_id"].isin(self.row)].copy()
        f["r"] = f["person_id"].map(self.row); f["k"] = f["token"].map(self.cidx)
        f = f.merge(v[["r", "age_days"]].assign(vi=v.groupby("r").cumcount().to_numpy() + 1), on=["r", "age_days"], how="inner")
        r = v["r"].to_numpy(); cuts = np.searchsorted(r, np.arange(len(self.pids) + 1))
        ages = v["age_days"].to_numpy(float) / DAY; vts = v["vtype"].to_numpy(np.int64)
        self.v_age = [ages[cuts[i]:cuts[i + 1]] for i in range(len(self.pids))]
        self.v_type = [vts[cuts[i]:cuts[i + 1]] for i in range(len(self.pids))]
        f = f.sort_values(["r", "vi"])
        fr = f["r"].to_numpy(); fc = np.searchsorted(fr, np.arange(len(self.pids) + 1))
        fk, fv = f["k"].to_numpy(np.int64), f["vi"].to_numpy(np.int64)
        self.rec_k = [fk[fc[i]:fc[i + 1]] for i in range(len(self.pids))]   # code index
        self.rec_v = [fv[fc[i]:fc[i + 1]] for i in range(len(self.pids))]   # token index (1 = first visit)

    def tokens(self, i, cutoff=None):
        """Token arrays for patient i: index 0 is ENTRY, 1..J visits; optionally only visits <= cutoff."""
        a, t = self.v_age[i], self.v_type[i]
        if cutoff is not None:
            keep = a <= cutoff; a, t = a[keep], t[keep]
        return np.concatenate([[self.entry[i]], a]), np.concatenate([[ENTRY], t])

    def batch(self, idx, rng, window, device):
        B = len(idx)
        W = window
        vt = np.zeros((B, W), np.int64); age = np.zeros((B, W), np.float32); gap = np.zeros((B, W), np.float32)
        valid = np.zeros((B, W), bool); has_next = np.zeros((B, W), bool); cens = np.zeros((B, W), bool)
        dt = np.zeros((B, W), np.float32); vt_next = np.zeros((B, W), np.int64); died_here = np.zeros((B, W), bool)
        tok_index = np.full((B, W), -1, np.int64)
        rec_age = np.full((B, self.K), np.inf, np.float32); hz_u = np.zeros((B, W), np.float32)
        rec_idx = np.full((B, self.K), NEVER, np.int64)
        cb, ct, ck = [], [], []
        pre = []                                         # cropped prefix (tokens 0..s) for the Tier B filter start
        for b, i in enumerate(idx):
            a, t = self.tokens(i)
            n = len(a)
            s = 0 if n <= W or rng is None else int(rng.integers(0, n - W + 1))
            pre.append((a[:s + 1], t[:s + 1]) if s > 0 else None)
            e = min(n, s + W)
            L = e - s
            vt[b, :L] = t[s:e]; age[b, :L] = a[s:e]; valid[b, :L] = True
            g = np.diff(a, prepend=a[0]); gap[b, :L] = g[s:e]
            tok_index[b, :L] = np.arange(s, e)
            nxt = np.arange(s, e) + 1
            hn = nxt < n
            has_next[b, :L] = hn
            dt[b, :L][hn] = a[nxt[hn]] - a[s:e][hn]
            vt_next[b, :L][hn] = t[nxt[hn]]
            if e == n:                                   # last token overall is in the window
                cens[b, L - 1] = True
                dt[b, L - 1] = max(self.exit[i] - a[-1], 0.0)
                died_here[b, L - 1] = self.died[i]
            rec_idx[b, self.rec_k[i]] = self.rec_v[i]
            okv = self.rec_v[i] < n
            rec_age[b, self.rec_k[i][okv]] = a[self.rec_v[i][okv]]
            span_b = dt[b, :L] * (has_next[b, :L] | cens[b, :L])
            # separate generator, so the main rng (and with it the batches of older runs) is unchanged
            if not hasattr(self, "_hzr"):
                self._hzr = np.random.default_rng(12345)
            hz_u[b, :L] = span_b * (self._hzr.random(L) if rng is not None else 0.5)
            inwin = (self.rec_v[i] >= s) & (self.rec_v[i] < e)
            cb += [b] * int(inwin.sum()); ct += list(self.rec_v[i][inwin] - s); ck += list(self.rec_k[i][inwin])
        T = lambda x, dt_=None: torch.as_tensor(x, device=device)
        ti = T(tok_index); ri = T(rec_idx)
        at_risk = (ri.unsqueeze(1) > ti.unsqueeze(-1)) & T(valid).unsqueeze(-1)
        y_next = ri.unsqueeze(1) == (ti + 1).unsqueeze(-1)
        extra = {}
        if any(q is not None for q in pre):
            P = max(len(q[0]) for q in pre if q is not None)
            pa = np.zeros((B, P), np.float32); pv = np.zeros((B, P), np.int64); pm = np.zeros((B, P), bool)
            for b, q in enumerate(pre):
                if q is not None:
                    pa[b, :len(q[0])] = q[0]; pv[b, :len(q[0])] = q[1]; pm[b, :len(q[0])] = True
            extra = {"pre_age": T(pa), "pre_vt": T(pv), "pre_mask": T(pm)}
        return {**extra, "vt": T(vt), "age": T(age), "gap": T(gap), "sex": T(self.sex[idx]).long(), "valid": T(valid),
                "has_next": T(has_next), "censored": T(cens), "dt_next": T(dt), "vt_next": T(vt_next),
                "died_here": T(died_here), "at_risk": at_risk, "y_next": y_next,
                "rec_age": T(rec_age), "exit_age": T(self.exit[idx].astype(np.float32)), "died_b": T(self.died[idx]),
                "hz_u": T(hz_u),
                "code_b": T(np.array(cb, np.int64)), "code_t": T(np.array(ct, np.int64)), "code_k": T(np.array(ck, np.int64))}


def train(data, cfg, iters, batch, lr, seed, device, window=128, eval_every=250, log=_log):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = VisitPP(cfg).to(device)
    dec = [p for n, p in model.named_parameters() if p.dim() >= 2]; nod = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": 0.1}, {"params": nod, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.99))
    warm = max(1, iters // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(s, iters) / iters))))
    tr = np.flatnonzero(data.split == "train")      # people with NO visits are kept: they are the low utilisers
    va = np.flatnonzero(data.split == "val")
    vr = np.random.default_rng(1); vb = [vr.choice(va, min(batch, len(va)), replace=False) for _ in range(8)]
    def vloss():
        model.eval(); tot = 0.0
        with torch.no_grad():
            for k, b in enumerate(vb):
                tot += float(model.loss(data.batch(b, np.random.default_rng(k), window, device))["loss"])
        model.train(); return tot / len(vb)
    best, best_state, hist, t0, skipped = float("inf"), None, [], time.time(), 0
    v0 = vloss(); hist.append((0, v0)); log(f"iter 0 val {v0:.4f}")
    for it in range(1, iters + 1):
        out = model.loss(data.batch(rng.choice(tr, batch, replace=False), rng, window, device))
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(out["loss"]):            # skip a non-finite batch instead of poisoning the weights
            skipped += 1; sched.step(); continue
        out["loss"].backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            skipped += 1; opt.zero_grad(set_to_none=True); sched.step(); continue
        opt.step(); sched.step()
        if it % eval_every == 0 or it == iters:
            v = vloss(); hist.append((it, v))
            if v < best:
                best, best_state = v, {k: x.detach().clone() for k, x in model.state_dict().items()}
            log(f"iter {it} train {float(out['loss'].detach()):.4f} (visit {float(out['visit']):.3f} mark {float(out['mark']):.3f} "
                f"death {float(out['death']):.3f} hz {float(out['hz']):.3f}) val {v:.4f} skipped {skipped} t+{time.time()-t0:,.0f}s")
    model.load_state_dict(best_state)
    return model, hist


@torch.no_grad()
def sample_risk(model, data, anchors, names, horizons, gap_days, n_samples=10, device="cpu", chunk=1024,
                max_steps=80, window=128, seed=0, estimator="rb", count_visits=False, close_cap=True):
    """Recorded risk within each horizon by simulating future VISITS from the anchor. The first visit is
    sampled conditional on no visit between the last observed visit and the anchor (truncated mixture).
    estimator: "hits" (share of paths that record k), "rb" (Rao-Blackwellized 1 - prod(1 - p) along each path) or
    "frozen" (diagnostic: no rollout, the rate read after the last history visit is held fixed to the horizon) or
    "rb_frozen" (diagnostic: visits simulated as in rb, but every mark uses the rate read at the anchor, so the
    difference from rb isolates drift of the rates along the simulated path) or "rb_nodx" (diagnostic: as rb, but the
    diagnoses sampled along the path are not written into the history, so rates move with simulated visits and age but
    not with simulated comorbidity; rb minus rb_nodx isolates the cascade of simulated diagnoses) or "comp" (unbiased:
    the compensator sum_j p_jk over the visits at which k is still at risk on the path; "rb" stops the product
    1 - prod(1 - p_jk) at the visit where k is sampled as recorded, which biases it downward by about F/2 in relative
    terms, F the recorded risk). Estimates are averaged over paths and clipped to [0, 1].
    "posterior" (Tier B only, no rollout): the filter's probability that onset of k has happened by the anchor but k is
    not yet recorded (NaN where k is already recorded); the same array is returned for every horizon. "head": the
    direct horizon head (models trained with horizons)."""
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    cols = torch.tensor([data.cidx[t] for t in names if t in data.cidx], dtype=torch.long, device=device)
    has_death = "DEATH" in names
    K, D = data.K, model.c.n_embd
    rows = anchors["person_id"].map(data.row).to_numpy()
    a_years = anchors["anchor_age_days"].to_numpy(float) / DAY
    Hmax = max(horizons)
    out = {H: np.zeros((len(anchors), len(names)), np.float32) for H in horizons}
    per = max(1, chunk // n_samples)
    W = model.c.max_visits
    t_start = time.time()
    for s in range(0, len(anchors), per):
        if s and (s // per) % 50 == 0:
            done = s / len(anchors)
            print(f"rollout {s:,}/{len(anchors):,} anchors ({done:.1%}), eta {(time.time() - t_start) / done * (1 - done) / 60:.0f} min", flush=True)
        idx = np.arange(s, min(s + per, len(anchors)))
        B = len(idx) * n_samples
        seqs = []
        for k in idx:
            i = rows[k]
            a, t = data.tokens(i, cutoff=a_years[k] - gap_days / DAY)
            a, t = a[-(window):], t[-(window):]
            rec = np.zeros(K, bool)
            rv = data.rec_v[i]; rk = data.rec_k[i]
            rec[rk[rv <= len(data.tokens(i, cutoff=a_years[k] - gap_days / DAY)[0]) - 1]] = True
            for _ in range(n_samples):
                seqs.append((a, t, rec, i))
        T0 = max(len(q[0]) for q in seqs); T = T0 + max_steps
        vt = torch.zeros(B, T, dtype=torch.long, device=device); age = torch.zeros(B, T, device=device)
        gap = torch.zeros(B, T, device=device); valid = torch.zeros(B, T, dtype=torch.bool, device=device)
        bag = torch.zeros(B, T, D, device=device); cnt = torch.zeros(B, T, device=device)
        recorded = torch.zeros(B, K, dtype=torch.bool, device=device)
        L = torch.zeros(B, dtype=torch.long, device=device)
        sex = torch.zeros(B, dtype=torch.long, device=device)
        for j, (a, t, rec, i) in enumerate(seqs):
            n = len(a)
            vt[j, :n] = torch.as_tensor(t); age[j, :n] = torch.as_tensor(a, dtype=torch.float32)
            gap[j, :n] = torch.as_tensor(np.diff(a, prepend=a[0]), dtype=torch.float32)
            valid[j, :n] = True; L[j] = n; recorded[j] = torch.as_tensor(rec); sex[j] = int(data.sex[i])
            age[j, n:] = float(a[-1])
        # history code bags (visits inside the window)
        for j, (a, t, rec, i) in enumerate(seqs):
            n_all = len(data.tokens(i, cutoff=a_years[idx[j // n_samples]] - gap_days / DAY)[0])
            off = n_all - len(a)
            m = (data.rec_v[i] >= off) & (data.rec_v[i] < n_all)
            if m.any():
                pos = torch.as_tensor(data.rec_v[i][m] - off, device=device)
                bag[j].index_add_(0, pos, model.code.weight[torch.as_tensor(data.rec_k[i][m], device=device)])
                cnt[j].index_add_(0, pos, torch.ones(int(m.sum()), device=device))
        ar = torch.arange(B, device=device)
        anchor = torch.as_tensor(np.repeat(a_years[idx], n_samples), dtype=torch.float32, device=device)
        hit_time = torch.full((B, len(names)), float("inf"), device=device)
        nc = len(cols)
        hz = torch.tensor([float(H) for H in horizons], device=device)
        notrec = torch.ones(B, nc, len(horizons), device=device)        # P(k not recorded by H) along the path
        compens = torch.zeros(B, nc, len(horizons), device=device)         # compensator: sum of p while k is at risk
        nvis = torch.zeros(B, len(horizons), device=device)              # simulated visits inside each horizon
        alive = torch.ones(B, dtype=torch.bool, device=device)
        now_min = anchor.clone()                       # first visit must fall after the anchor
        det = getattr(model.c, "detection", False)
        anchored = getattr(model.c, "rate_anchor", "none") == "last_dx"
        if estimator == "head":                        # direct horizon head: no rollout
            Th = int(L.max())
            hh = model.encode_bag(vt[:, :Th], age[:, :Th], gap[:, :Th], sex, bag[:, :Th], cnt[:, :Th], valid[:, :Th])[ar, L - 1]
            t_last = age[ar, L - 1]
            pk = torch.sigmoid(model.horizon_logits(hh, anchor - t_last))[:, cols]           # [B, nc, nH]
            gam = torch.exp(model.heads(hh)["log_death"])
            for H in horizons:
                r = torch.zeros(B, len(names), device=device)
                r[:, :nc] = pk[:, :, list(model.c.horizons).index(float(H))]
                if has_death:
                    r[:, -1] = -torch.expm1(-gam * H)
                out[H][idx] = r.view(len(idx), n_samples, -1).mean(1).clamp(0, 1).cpu().numpy()
            continue
        if det:                                        # Tier B: filter the latent onset state through the history
            Th = int(L.max())
            hh = model.encode_bag(vt[:, :Th], age[:, :Th], gap[:, :Th], sex, bag[:, :Th], cnt[:, :Th], valid[:, :Th])
            oh = model.heads(hh)
            oh["log_rate"] = model.rates(hh, oh, age[:, :Th], cnt[:, :Th]); del hh
            pres = []                                  # history cut by the window: start from the prefix state
            for j, (a, t, rec, i) in enumerate(seqs):
                fa, ft = data.tokens(i, cutoff=a_years[idx[j // n_samples]] - gap_days / DAY)
                off = len(fa) - len(a)
                pres.append((fa[:off + 1], ft[:off + 1]) if off > 0 else None)
            if any(q is not None for q in pres):
                # same prefix for the n_samples paths of an anchor: compute once per anchor, in memory-bounded mini batches
                s0 = torch.ones(B, K, device=device); s1 = torch.zeros(B, K, device=device)
                rows_ = [j for j in range(0, B, n_samples) if pres[j] is not None]
                P = max(len(pres[j][0]) for j in rows_)
                mb = max(1, int(32e6 // (P * K)))                      # about 128 MB per [mb, P, K] float tensor
                for c0 in range(0, len(rows_), mb):
                    rr = rows_[c0:c0 + mb]; Pc = max(len(pres[j][0]) for j in rr)
                    pa = torch.zeros(len(rr), Pc, device=device); pv = torch.zeros(len(rr), Pc, dtype=torch.long, device=device)
                    pm = torch.zeros(len(rr), Pc, dtype=torch.bool, device=device)
                    for u, j in enumerate(rr):
                        q = pres[j]; pa[u, :len(q[0])] = torch.as_tensor(q[0], dtype=torch.float32)
                        pv[u, :len(q[0])] = torch.as_tensor(q[1]); pm[u, :len(q[0])] = True
                    rt = torch.as_tensor(rr, device=device)
                    a0, a1 = model.prefix_state(oh["log_rate"][rt, 0], oh["det_h"][rt, 0], pa, pv, pm)
                    for k_ in range(n_samples):
                        s0[rt + k_] = a0; s1[rt + k_] = a1
                    del pa, pv, pm, a0, a1
            else:
                s0 = torch.ones(B, K, device=device); s1 = torch.zeros(B, K, device=device)
            for j in range(Th - 1):
                m = (j + 1 < L).unsqueeze(-1)
                lam_dt = torch.exp(oh["log_rate"][:, j]) * (age[:, j + 1] - age[:, j]).unsqueeze(-1)
                d = model.detect_prob({"det_h": oh["det_h"][:, j]}, vt[:, j + 1].clamp(max=N_VTYPE - 2))
                _, n0, n1 = model.filter_step(s0, s1, lam_dt, d)
                s0 = torch.where(m, n0, s0); s1 = torch.where(m, n1, s1)
            lr_last = oh["log_rate"][ar, L - 1]
            del oh
            if estimator == "posterior":               # Tier B: P(onset happened, not yet recorded) at the anchor
                tau = (anchor - age[ar, L - 1]).clamp(min=0).unsqueeze(-1)
                s1a = s1 + s0 * (-torch.expm1(-torch.exp(lr_last) * tau))
                r = torch.full((B, len(names)), float("nan"), device=device)
                r[:, :nc] = torch.where(recorded[:, cols], torch.full_like(s1a[:, cols], float("nan")), s1a[:, cols])
                for H in horizons:
                    out[H][idx] = r.view(len(idx), n_samples, -1).mean(1).cpu().numpy()
                continue
        for step in range(0 if estimator == "frozen" else max_steps):   # frozen: no rollout, rate at the anchor held fixed
            if not bool(alive.any()):
                break
            Tm = int(L.max())
            hall = model.encode_bag(vt[:, :Tm], age[:, :Tm], gap[:, :Tm], sex, bag[:, :Tm], cnt[:, :Tm], valid[:, :Tm])
            h = hall[ar, L - 1]
            o = model.heads(h)
            t_last = age[ar, L - 1]
            if anchored:
                jl = model.anchor_index(cnt[:, :Tm])[ar, L - 1]
                o["log_rate"] = model.anchored(hall[ar, jl], jl, t_last, age[ar, jl])
            del hall
            tau = (now_min - t_last).clamp(min=0)
            # truncated log-normal mixture sample, conditional on gap > tau
            s_ = torch.exp(o["logs"]); lt = torch.log(tau.clamp(min=1e-6)).unsqueeze(-1)
            z_tau = torch.where(tau.unsqueeze(-1) > 0, (lt - o["mu"]) / s_, torch.full_like(s_, -1e4))
            surv = torch.special.ndtr(-z_tau).clamp(min=1e-12)
            comp = torch.multinomial(torch.softmax(o["logw"] + torch.log(surv), -1), 1, generator=g).squeeze(1)
            mu_c, s_c, zt = o["mu"][ar, comp], s_[ar, comp], z_tau[ar, comp]
            u = torch.rand(B, generator=g, device=device)
            pz = torch.special.ndtr(zt) + u * (1 - torch.special.ndtr(zt))
            z = torch.special.ndtri(pz.clamp(1e-9, 1 - 1e-9))
            dt = torch.exp(mu_c + s_c * z)
            t_new = t_last + dt
            gamma = torch.exp(o["log_death"])
            t_death = torch.maximum(t_last, now_min) + (-torch.log(torch.rand(B, generator=g, device=device).clamp_min(1e-12)) / gamma)
            end = anchor + Hmax
            if has_death:
                dh = alive & (t_death <= torch.minimum(t_new, end))
                hit_time[dh, -1] = torch.minimum(hit_time[dh, -1], t_death[dh])
            stop = (t_death <= t_new) | (t_new > end)
            go = alive & ~stop
            # marks at the new visit: each not yet recorded disease with p = 1 - exp(-lambda * dt) (Tier A)
            # or p = d * s1 after the interval (Tier B, d depends on the type of the new visit)
            if det:
                vtn = torch.multinomial(torch.softmax(o["vtype"], -1), 1, generator=g).squeeze(1)
                d = model.detect_prob(o, vtn)
                p, n0, n1 = model.filter_step(s0, s1, torch.exp(o["log_rate"]) * dt.unsqueeze(-1), d)
            else:
                if estimator == "rb_frozen":               # diagnostic: simulated visits, marks at the anchor-time rate
                    if step == 0:
                        lr0 = o["log_rate"].clone()
                    p = -torch.expm1(-torch.exp(lr0) * dt.unsqueeze(-1))
                else:
                    p = -torch.expm1(-torch.exp(o["log_rate"]) * dt.unsqueeze(-1))
            pc = p[:, cols]
            act = (go.unsqueeze(-1) & ~recorded[:, cols]).unsqueeze(-1) & (t_new.unsqueeze(-1) <= anchor.unsqueeze(-1) + hz).unsqueeze(1)
            notrec = torch.where(act, notrec * (1 - pc).unsqueeze(-1), notrec)
            compens = torch.where(act, compens + pc.unsqueeze(-1), compens)
            nvis += (go.unsqueeze(-1) & (t_new.unsqueeze(-1) <= anchor.unsqueeze(-1) + hz)).float()
            newk = (torch.rand(B, K, generator=g, device=device) < p) & ~recorded & go.unsqueeze(-1)
            recorded |= newk
            hk = newk[:, cols]
            ht = hit_time[:, :nc]
            hit_time[:, :nc] = torch.where(hk & torch.isinf(ht), t_new.unsqueeze(-1).expand_as(ht), ht)
            if det:
                s0 = torch.where(go.unsqueeze(-1), n0, s0); s1 = torch.where(go.unsqueeze(-1), n1, s1)
            else:
                vtn = torch.multinomial(torch.softmax(o["vtype"], -1), 1, generator=g).squeeze(1)
            pos = L.clamp(max=T - 1)
            gi = ar[go]
            vt[gi, pos[go]] = vtn[go]; age[gi, pos[go]] = t_new[go]; gap[gi, pos[go]] = dt[go]; valid[gi, pos[go]] = True
            if estimator != "rb_nodx":                 # rb_nodx: simulated diagnoses are not written into the history
                bag[gi, pos[go]] = newk[go].float() @ model.code.weight
                cnt[gi, pos[go]] = newk[go].float().sum(-1)
            L = torch.where(go, (L + 1).clamp(max=T - 1), L)
            alive = go & (L < T - 1)
            now_min = torch.where(go, t_new, now_min)
        if (close_cap or estimator == "frozen") and bool(alive.any()):            # paths cut by the step cap before the horizon: close the
            Tm = int(L.max())                          # remaining time analytically with one final interval
            hall = model.encode_bag(vt[:, :Tm], age[:, :Tm], gap[:, :Tm], sex, bag[:, :Tm], cnt[:, :Tm], valid[:, :Tm])
            o = model.heads(hall[ar, L - 1])
            t_last = age[ar, L - 1]
            if anchored:
                jl = model.anchor_index(cnt[:, :Tm])[ar, L - 1]
                o["log_rate"] = model.anchored(hall[ar, jl], jl, t_last, age[ar, jl])
            del hall
            lam = torch.exp((lr0 if estimator == "rb_frozen" else o["log_rate"])[:, cols])
            for hi, H in enumerate(horizons):
                rest = ((anchor + H) - t_last).clamp(min=0)
                use = (alive & (rest > 0)).unsqueeze(-1) & ~recorded[:, cols]
                if det:
                    pv = torch.softmax(o["vtype"], -1)                       # expected detection over visit types
                    dk = sum(pv[:, v:v + 1] * model.detect_prob(o, torch.full_like(L, v))[:, cols] for v in range(pv.shape[1]))
                    pf, _, _ = model.filter_step(s0[:, cols], s1[:, cols], lam * rest.unsqueeze(-1), dk)
                else:
                    pf = -torch.expm1(-lam * rest.unsqueeze(-1))
                notrec[:, :, hi] = torch.where(use, notrec[:, :, hi] * (1 - pf), notrec[:, :, hi])
                compens[:, :, hi] = torch.where(use, compens[:, :, hi] + pf, compens[:, :, hi])
        for hi, H in enumerate(horizons):
            r = (hit_time <= (anchor + H).unsqueeze(-1)).float()
            if estimator in ("rb", "frozen", "rb_frozen", "rb_nodx"):
                r[:, :nc] = 1 - notrec[:, :, hi]
            elif estimator == "comp":
                r[:, :nc] = compens[:, :, hi]
            out[H][idx] = r.view(len(idx), n_samples, -1).mean(1).clamp(0, 1).cpu().numpy()
            if count_visits:
                out.setdefault(f"nvis{H}", np.zeros(len(anchors), np.float32))[idx] = nvis[:, hi].view(len(idx), n_samples).mean(1).cpu().numpy()
    return out


def run(cohort, fo, visits, iters=1000, batch=32, lr=5e-4, seed=0, device="cpu", n_layer=4, n_embd=128, n_head=8,
        baseline_npz=None, names=None, n_samples=10, max_eval_anchors=None, log=_log, cfg=CFG, detection=False):
    data = VisitData(cohort, fo, visits)
    vc = VisitConfig(data.K, n_layer=n_layer, n_head=n_head, n_embd=n_embd, detection=detection,
                     rate_anchor=os.environ.get("RATE_ANCHOR", "none"),
                     horizons=tuple(float(x) for x in os.environ.get("HORIZON_HEAD", "").split(",") if x),
                     hz_weight=float(os.environ.get("HZ_WEIGHT", "1.0")),
                     hz_person=os.environ.get("HZ_PERSON", "0") == "1")
    log(f"codes {data.K} patients {len(data.pids):,} params {sum(p.numel() for p in VisitPP(vc).parameters())/1e6:.2f}M")
    model, hist = train(data, vc, iters, batch, lr, seed, device, log=log)
    an = s3.build_anchors(cohort, cfg)
    te = an[an["split"] == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    if baseline_npz is not None:
        names = [str(x) for x in baseline_npz["names"]]
        assert np.array_equal(baseline_npz["person_id"], te["person_id"].to_numpy())
    if max_eval_anchors and len(te) > max_eval_anchors:
        keep = np.sort(np.random.default_rng(0).choice(len(te), max_eval_anchors, replace=False))
    else:
        keep = np.arange(len(te))
    gap = cfg["evaluation"]["gap_days"]
    ev = cfg["evaluation"]; bands = [tuple(b) for b in ev["age_bands"]]
    risks = sample_risk(model, data, te.iloc[keep].reset_index(drop=True), names, (1, 5), gap, n_samples, device,
                        chunk=int(os.environ.get("SAMPLE_CHUNK", "512")), estimator=os.environ.get("EVAL_ESTIMATOR", "rb"))  # 1024 hit the T4 memory limit in the Oct 14 sample run
    from delphi_aou.eval.calibration import log_oe
    res = {"val_loss_curve": hist}
    for H in (1, 5):
        Y, P, E, _ = s3.attach(te, fo, names[:-1], names[:-1], H, gap)
        Yk, Ek = Y[keep], E[keep]
        tab = eval_auc.evaluate_auc(risks[H], Yk, Ek, te["age_years"].to_numpy()[keep], te["sex"].to_numpy()[keep], names,
                                    bands, ev["min_cases"], ev["min_controls"])
        res[f"H{H}"] = {"auc": eval_auc.summarize(tab),
                        "mean_log_OE": float(np.nanmean(log_oe(risks[H][:, :-1], Yk[:, :-1], Ek[:, :-1])))}
        if H == 5 and baseline_npz is not None and len(keep) == len(te):
            for ref in ["gbm_visits", "gbm"]:
                res[f"paired_vs_{ref}"] = eval_auc.compare_models(risks[5], baseline_npz[f"score_{ref}"], Yk, Ek,
                    te["age_years"].to_numpy(), te["sex"].to_numpy(), names, bands, ev["min_cases"], ev["min_controls"])
    log(json.dumps({k: v for k, v in res.items() if k != "val_loss_curve"}, default=float))
    return model, res, risks, keep


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", os.cpu_count() or 4)))
    tmp = Path(tempfile.gettempdir())
    coh = s2._read_local_from_cache("s2_cohort"); fo = s2._read_local_from_cache("s2_fo_events")
    vis = s2._read_local_from_cache("s5_visits")
    subprocess.run(["gsutil", "-q", "cp", f"{s2.OUT_DIR}/s3_scores_H5.npz", str(tmp / "s3_scores_H5.npz")], check=True)
    npz = np.load(tmp / "s3_scores_H5.npz")
    seed = int(os.environ.get("SEED", "0"))
    mea = os.environ.get("MAX_EVAL_ANCHORS")
    model, res, risks, keep = run(coh, fo, vis, int(os.environ.get("ITERS", "1000")), int(os.environ.get("BATCH", "32")),
                                  float(os.environ.get("LR", "5e-4")), seed, dev, int(os.environ.get("N_LAYER", "4")),
                                  int(os.environ.get("N_EMBD", "128")), int(os.environ.get("N_HEAD", "8")), npz,
                                  n_samples=int(os.environ.get("N_SAMPLES", "10")), max_eval_anchors=int(mea) if mea else None,
                                  detection=os.environ.get("DETECTION", "0") == "1")
    tag = f"visit{'B' if os.environ.get('DETECTION', '0') == '1' else 'A'}_s{seed}"
    sh = common.split_hash(coh["person_id"].to_numpy(), coh["split"].to_numpy())
    common.write_results(tmp / f"s5_summary_{tag}.json", res, CFG, sh)
    np.savez_compressed(tmp / f"s5_scores_{tag}.npz", keep=keep, r1=risks[1], r5=risks[5])
    torch.save(model.state_dict(), tmp / f"s5_model_{tag}.pt")
    subprocess.run(f"gsutil -q cp {tmp}/s5_*_{tag}.* {s2.OUT_DIR}/", shell=True, check=True)
    print("STAGE 5 COMPLETE")


if __name__ == "__main__":
    main()
