"""Tier A visit-level marked point process (the paper's model, first tier).

Data object per patient: EHR entry E, exit C (end of observation or death, death flag), and visits
v_1 < ... < v_J inside [E, C], each with a type and the SET S_j of first-occurrence diagnoses newly
recorded that day (possibly empty). A causal transformer reads (entry, v_1, ..., v_J). The state after
token i gives:
  visit head   time to the next visit: log-normal mixture (intensity-free, Shchur et al. 2020),
               right censored at C; plus the type of the next visit (categorical)
  death head   constant death rate gamma_i over the interval (exact death time at C when died)
  mark head    for every disease k not yet recorded, an onset rate lambda_ik; under Tier A (perfect
               detection, recorded at the first visit after onset)
               P(k in S_{i+1}) = 1 - exp(-lambda_ik * Delta_i)
Likelihood = visit timing + visit type + death + one Bernoulli per not-yet-recorded disease per visit.

Tier B (detection=True): onset is a latent state. Each visit detects an existing undiagnosed onset with probability
d = sigmoid(a(h_i) + b[type of the next visit] + c_k). Per disease, a two-state forward filter over the visits:
  between visits   s1 <- s1 + s0 (1 - exp(-lambda Delta)),  s0 <- s0 exp(-lambda Delta)
  at a visit       P(recorded) = d s1; if not recorded: s1 <- s1 (1 - d) / (1 - d s1), s0 <- s0 / (1 - d s1)
With d = 1 this is exactly Tier A. With d < 1, people who visit more often get their diseases recorded sooner,
which is the access effect Tier A cannot express.
Exposure is exactly [E, C]: no pre-entry time, no NO_EVENT pseudo tokens, no within-day order.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from delphi_aou.models.delphi import AgeEncoding, Block

N_VTYPE = 6          # 0 dx_only, 1 other, 2 outpatient, 3 er, 4 inpatient, 5 = ENTRY token


class VisitConfig:
    def __init__(self, n_codes, n_layer=6, n_head=8, n_embd=128, dropout=0.1, n_mix=8, max_visits=256, detection=False, det_history=True,
                 rate_anchor="none", horizons=(), hz_weight=1.0, hz_person=False):
        self.detection, self.det_history = detection, det_history
        self.horizons, self.hz_weight = tuple(float(x) for x in horizons), hz_weight   # direct horizon-risk head
        self.hz_person = hz_person    # True: each person's pseudo anchors are averaged, so every person weighs the same
        self.rate_anchor = rate_anchor        # "none" | "last_dx": onset rates read at the last visit with a new diagnosis
        self.n_codes, self.n_layer, self.n_head, self.n_embd = n_codes, n_layer, n_head, n_embd
        self.dropout, self.n_mix, self.max_visits = dropout, n_mix, max_visits


SLOPE_MAX = 0.3          # largest change of an anchored log onset rate per year of age (35% per year)


class VisitPP(nn.Module):
    def __init__(self, c: VisitConfig):
        super().__init__()
        self.c = c
        D = c.n_embd
        self.code = nn.Embedding(c.n_codes, D)
        self.vtype = nn.Embedding(N_VTYPE, D)
        self.sex = nn.Embedding(4, D)
        self.age = AgeEncoding(D)
        self.gap = nn.Linear(1, D)
        self.ncode = nn.Linear(1, D)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(_Cfg(D, c.n_head, c.dropout)) for _ in range(c.n_layer)])
        self.ln = nn.LayerNorm(D)
        self.mark = nn.Linear(D, c.n_codes)
        self.visit = nn.Linear(D, 3 * c.n_mix)
        self.vtype_head = nn.Linear(D, N_VTYPE - 1)
        self.death = nn.Linear(D, 1)
        if getattr(c, "rate_anchor", "none") != "none":              # only then: older checkpoints load unchanged
            self.rate_slope = nn.Parameter(torch.zeros(c.n_codes))  # per-disease change of log rate per year
        if getattr(c, "horizons", ()):                              # only then: older checkpoints load unchanged
            self.hz = nn.Linear(D + 1, c.n_codes * len(c.horizons))
        if getattr(c, "detection", False):
            self.det_h = nn.Linear(D, 1)
            self.det_vt = nn.Embedding(N_VTYPE, 1)
            self.det_k = nn.Parameter(torch.zeros(c.n_codes))
        self.apply(self._init)
        nn.init.constant_(self.mark.bias, math.log(0.01))      # 1% per year per disease at init
        nn.init.constant_(self.death.bias, math.log(0.005))
        if getattr(c, "horizons", ()):
            nn.init.constant_(self.hz.bias, -5.0)              # about 0.7% risk per disease and horizon at init
        if getattr(c, "detection", False):
            nn.init.constant_(self.det_h.bias, 1.0)            # d about 0.73 at init

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(self, vt, age, gap, sex, code_b, code_t, code_k, valid):
        """vt, age(years), gap(years) [B,T]; sex [B]; codes given as flat (batch, position, code) indices."""
        B, T = vt.shape
        bag = torch.zeros(B, T, self.c.n_embd, device=vt.device)
        cnt = torch.zeros(B, T, device=vt.device)
        if code_k.numel():
            bag.index_put_((code_b, code_t), self.code(code_k), accumulate=True)
            cnt.index_put_((code_b, code_t), torch.ones_like(code_k, dtype=torch.float), accumulate=True)
        return self.encode_bag(vt, age, gap, sex, bag, cnt, valid)

    def encode_bag(self, vt, age, gap, sex, bag, cnt, valid):
        """Same as encode, with the per visit code embeddings already summed (used when sampling)."""
        T = vt.shape[1]
        x = self.vtype(vt) + self.age(age) + self.gap(torch.log1p(gap * 365.25).unsqueeze(-1) / 5.0)
        x = x + bag + self.ncode(torch.log1p(cnt).unsqueeze(-1)) + self.sex(sex).unsqueeze(1)
        x = self.drop(x)
        m = torch.ones(T, T, dtype=torch.bool, device=vt.device).tril()
        allowed = (m.unsqueeze(0) & valid.unsqueeze(1)) | torch.eye(T, dtype=torch.bool, device=vt.device)
        allowed = allowed.unsqueeze(1)
        for b in self.blocks:
            x = b(x, allowed)
        return self.ln(x)

    def heads(self, h):
        mix = self.visit(h).view(*h.shape[:-1], 3, self.c.n_mix)
        logw, mu, logs = mix[..., 0, :], mix[..., 1, :] + math.log(0.3), mix[..., 2, :].clamp(-4, 3)
        o = {"log_rate": self.mark(h), "logw": F.log_softmax(logw, -1), "mu": mu, "logs": logs,
             "vtype": self.vtype_head(h), "log_death": self.death(h).squeeze(-1)}
        if getattr(self.c, "detection", False):
            o["det_h"] = self.det_h(h).squeeze(-1)
        return o

    def horizon_logits(self, h, since):
        """[..., K, n_horizons] logit of recording disease k within H years of an anchor that lies `since` years
        after the visit whose state is h (the anchor falls before the next visit). Trained directly on observed
        outcomes, so it carries no rollout and no cascade of simulated diagnoses."""
        x = torch.cat([h, (torch.log1p(since.clamp(min=0) * 365.25) / 5.0).unsqueeze(-1)], -1)
        return self.hz(x).view(*h.shape[:-1], self.c.n_codes, len(self.c.horizons))

    @staticmethod
    def anchor_index(cnt):
        """[B,T] index of the last position <= i whose visit recorded at least one new diagnosis (or 0, the entry)."""
        pos = torch.arange(cnt.shape[1], device=cnt.device).expand_as(cnt)
        return torch.cummax(torch.where((cnt > 0) | (pos == 0), pos, torch.zeros_like(pos)), 1).values

    def anchored(self, h, j, age_now, age_j):
        """log onset rates read at the anchor positions j (h: [B,T,D] or [B,D] gathered), plus the age slope.
        The slope is bounded to +-SLOPE_MAX per year and the log rate to [-20, 3]: an unbounded slope times decades of
        elapsed age overflowed exp(log_rate) and turned the first anchored run into NaN after 10k iterations."""
        slope = SLOPE_MAX * torch.tanh(self.rate_slope / SLOPE_MAX)
        return (self.mark(h) + slope * (age_now - age_j).unsqueeze(-1)).clamp(-20.0, 3.0)

    def rates(self, h, o, age, cnt):
        """Log onset rates for every position. rate_anchor="none": the mark head at each position. "last_dx": the mark
        head at the last position with a new diagnosis plus a learned age slope, so visits without new diagnoses (most
        simulated visits) move onset rates only through age; detection and the visit process still use every visit."""
        if getattr(self.c, "rate_anchor", "none") != "last_dx":
            return o["log_rate"]
        j = self.anchor_index(cnt)
        hj = torch.gather(h, 1, j.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        return self.anchored(hj, j, age, torch.gather(age, 1, j))

    @staticmethod
    def code_count(B, T, code_b, code_t, device):
        cnt = torch.zeros(B, T, device=device)
        if code_b.numel():
            cnt.index_put_((code_b, code_t), torch.ones_like(code_b, dtype=torch.float), accumulate=True)
        return cnt

    def detect_prob(self, o, vt_next):
        """[...,K] detection probability at the next visit, given its type."""
        logit = self.det_vt(vt_next).squeeze(-1)
        if getattr(self.c, "det_history", True):
            logit = logit + o["det_h"]
        else:
            logit = logit + self.det_h.bias            # same constant, no dependence on the history
        return torch.sigmoid(logit.unsqueeze(-1) + self.det_k)

    @staticmethod
    def filter_step(s0, s1, lam_dt, d):
        """One interval plus one visit of the per-disease forward filter. Returns (p_record, s0_post, s1_post),
        the posterior being conditional on NOT recorded at this visit."""
        e = torch.exp(-lam_dt)
        s1 = s1 + s0 * (1 - e); s0 = s0 * e
        p = (d * s1).clamp(1e-8, 1 - 1e-6)
        return p, s0 / (1 - p), s1 * (1 - d) / (1 - p)

    def prefix_state(self, log_rate0, det_h0, pre_age, pre_vt, pre_mask):
        """Filter state after the last visit BEFORE the window, for persons whose history was cropped.
        Rates and the history part of detection are held at their values at the first window position (closed form):
        s0 ~ exp(-lam (a_s - a_0)), s1 ~ sum_j (exp(-lam (a_{j-1} - a_0)) - exp(-lam (a_j - a_0))) prod_{i>=j} (1 - d_i).
        pre_age [B,P] ages of tokens 0..s (0 = entry), pre_vt [B,P] their visit types, pre_mask [B,P] valid.
        Returns (s0, s1) [B,K]; s0 = 1, s1 = 0 for persons without a prefix."""
        B, K = log_rate0.shape
        dev = log_rate0.device
        if pre_age.shape[1] < 2:
            return torch.ones(B, K, device=dev), torch.zeros(B, K, device=dev)
        lam = torch.exp(log_rate0).unsqueeze(1)                                  # [B,1,K]
        t = (pre_age - pre_age[:, :1]).unsqueeze(-1)                             # time since entry [B,P,1]
        m = pre_mask[:, 1:].unsqueeze(-1)                                        # visits 1..s
        hist = det_h0 if getattr(self.c, "det_history", True) else self.det_h.bias.expand_as(det_h0)
        logit = self.det_vt(pre_vt[:, 1:].clamp(max=N_VTYPE - 2)).squeeze(-1) + hist.unsqueeze(1)
        log1md = torch.where(m, F.logsigmoid(-(logit.unsqueeze(-1) + self.det_k)), torch.zeros((), device=dev))
        tail = torch.flip(torch.cumsum(torch.flip(log1md, [1]), 1), [1])        # sum over i >= j of log(1 - d_i)
        # log space throughout: long prefixes underflow float32 otherwise
        width = (t[:, 1:] - t[:, :-1]).clamp(min=0)
        log_on = -lam * t[:, :-1] + torch.log((-torch.expm1(-lam * width)).clamp(min=1e-30))   # onset in interval j
        neg = torch.full((), -1e30, device=dev)
        log_s1u = torch.logsumexp(torch.where(m, log_on + tail, neg), 1)
        last = (pre_mask.long().sum(1) - 1).clamp(min=0)
        log_s0u = -lam.squeeze(1) * t.squeeze(-1)[torch.arange(B, device=dev), last].unsqueeze(-1)
        has = m.any(1)                                                              # [B,1]: a prefix exists
        log_s1u = torch.where(has, log_s1u, neg)
        lz = torch.logaddexp(log_s0u, log_s1u)
        return torch.exp(log_s0u - lz), torch.exp(log_s1u - lz)

    @staticmethod
    def gap_logpdf_logsurv(o, dt):
        """log density and log survival of the log-normal mixture at dt (years)."""
        ldt = torch.log(dt.clamp(min=1e-4)).unsqueeze(-1)
        s = torch.exp(o["logs"])
        z = (ldt - o["mu"]) / s
        logpdf = torch.logsumexp(o["logw"] - 0.5 * z ** 2 - o["logs"] - 0.5 * math.log(2 * math.pi) - ldt, -1)
        surv = torch.special.ndtr(-z)
        logsurv = torch.logsumexp(o["logw"] + torch.log(surv.clamp(min=1e-12)), -1)
        return logpdf, logsurv

    def loss(self, batch):
        """batch from VisitData.batch. Positions i with has_next: next visit observed; with is_last &
        reaches_end: censored at exit (and death if died)."""
        h = self.encode(batch["vt"], batch["age"], batch["gap"], batch["sex"], batch["code_b"], batch["code_t"],
                        batch["code_k"], batch["valid"])
        o = self.heads(h)
        if getattr(self.c, "rate_anchor", "none") != "none":
            B_, T_ = batch["vt"].shape
            cnt = self.code_count(B_, T_, batch["code_b"], batch["code_t"], h.device)
            o["log_rate"] = self.rates(h, o, batch["age"], cnt)
        dt_next = batch["dt_next"]                    # [B,T] years to the next visit (or to exit)
        has_next, cens = batch["has_next"], batch["censored"]
        logpdf, logsurv = self.gap_logpdf_logsurv(o, dt_next)
        visit_nll = -(logpdf * has_next).sum() - (logsurv * cens).sum()
        vt_nll = (F.cross_entropy(o["vtype"].transpose(1, 2), batch["vt_next"].clamp(max=N_VTYPE - 2), reduction="none") * has_next).sum()
        gamma = torch.exp(o["log_death"])
        span = has_next | cens
        death_nll = (gamma * dt_next * span).sum() - (o["log_death"] * batch["died_here"]).sum()
        # marks: Bernoulli per not-yet-recorded disease at the next visit
        lr = o["log_rate"]                                                  # [B,T,K]
        x = torch.exp(lr) * dt_next.unsqueeze(-1)                           # lambda * Delta
        at_risk = batch["at_risk"] & has_next.unsqueeze(-1)                 # [B,T,K]
        y = batch["y_next"]                                                 # [B,T,K] recorded at next visit
        if getattr(self.c, "detection", False):
            d = self.detect_prob(o, batch["vt_next"].clamp(max=N_VTYPE - 2))
            B, T, K = lr.shape
            if "pre_age" in batch:                                          # cropped history: start from its state
                # no gradient through the start state: [B, P, K] with P in the thousands does not fit a T4 with autograd;
                # the forward value is exact, lambda and d are still learned from the window itself
                with torch.no_grad():
                    s0, s1 = self.prefix_state(lr[:, 0], o["det_h"][:, 0], batch["pre_age"], batch["pre_vt"], batch["pre_mask"])
            else:
                s0 = torch.ones(B, K, device=lr.device); s1 = torch.zeros(B, K, device=lr.device)
            mark_nll = lr.new_zeros(())
            for t in range(T):
                ar = at_risk[:, t]
                p, n0, n1 = self.filter_step(s0, s1, x[:, t], d[:, t])
                mark_nll = mark_nll - (torch.where(y[:, t], torch.log(p), torch.log1p(-p)) * ar).sum()
                s0 = torch.where(ar, n0, s0); s1 = torch.where(ar, n1, s1)
        else:
            log_p = torch.log(-torch.expm1(-x.clamp(min=1e-8)))
            mark_nll = -((y * log_p - (~y) * x) * at_risk).sum()
        n = span.sum().clamp(min=1)
        hz_nll = lr.new_zeros(())
        if getattr(self.c, "horizons", ()) and "hz_u" in batch:
            # pseudo anchors: a random time u after each visit, before the next one (or before exit); label = disease k
            # recorded within H of the anchor; observed if recorded, followed for H, or died (death competes)
            Hs = torch.tensor(self.c.horizons, device=lr.device)
            u = batch["hz_u"]
            tau = (batch["age"] + u).unsqueeze(-1).unsqueeze(-1)                       # [B,T,1,1]
            ev = batch["rec_age"][:, None, :, None] <= tau + Hs                          # [B,T,K,nH]
            obs = ev | (batch["exit_age"][:, None, None, None] >= tau + Hs) | batch["died_b"][:, None, None, None]
            m = obs & (batch["at_risk"] & (span & (u > 0)).unsqueeze(-1)).unsqueeze(-1)
            lg = self.horizon_logits(h, u)
            bce = F.binary_cross_entropy_with_logits(lg, ev.float(), reduction="none") * m
            if getattr(self.c, "hz_person", False):
                # one weight per person: mean over that person's pseudo anchors, scaled to the batch's span count so
                # the loss stays on the same scale as the unweighted version
                pos = (span & (u > 0)).float()                                         # [B,T]
                per_person = bce.sum((1, 2, 3)) / pos.sum(1).clamp(min=1)               # [B]
                hz_nll = per_person.sum() * (pos.sum() / pos.any(1).sum().clamp(min=1))
            else:
                hz_nll = bce.sum()
        total = (visit_nll + vt_nll + death_nll + mark_nll + getattr(self.c, "hz_weight", 1.0) * hz_nll) / n
        return {"loss": total, "visit": (visit_nll / n).detach(), "mark": (mark_nll / n).detach(), "hz": (hz_nll / n).detach(),
                "death": (death_nll / n).detach(), "vtype": (vt_nll / n).detach()}


class _Cfg:
    def __init__(self, n_embd, n_head, dropout):
        self.n_embd, self.n_head, self.dropout = n_embd, n_head, dropout
