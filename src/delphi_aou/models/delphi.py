"""Delphi-2M style model (reimplementation, not the authors' code).

GPT with a sinusoidal AGE encoding instead of positions, and one logit vector per position used
twice: softmax for the next token, and an exponential waiting time with rate sum(exp(logits)).
Objectives:
  "delphi"  faithful: cross entropy over the next token (NO_EVENT included as a target) plus the
            exponential waiting time NLL with rate = sum(exp(logits)), so NO_EVENT pseudo tokens
            count as events in the time loss, and the time after the last token is ignored.
  "pp"      exact marked point process likelihood on the observed window (the paper's objective):
            NO_EVENT tokens are inputs only (they update the rate, they are never targets and they
            are excluded from the rate); every interval pays its compensator rate * dt; every real
            diagnosis pays -log rate_mark; and the end of follow up pays rate_last * (end - t_last).
Ties: with mask_ties=True a token cannot attend to OTHER tokens of the same age, because the order
inside a day is arbitrary (60% of first diagnoses in All of Us share a day with another).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, NO_EVENT = 0, 1


@dataclass
class DelphiConfig:
    vocab_size: int
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 120
    dropout: float = 0.1
    max_len: int = 256          # longest context used when scoring
    train_window: int = 64      # random contiguous window during training (Delphi used 48)
    mask_ties: bool = True
    t_min_years: float = 1.0 / 365.25
    objective: str = "delphi"   # "delphi" (faithful) or "pp" (exact point process likelihood)
    censor: bool = False        # kept for backward compatibility: censor=True means objective="pp"


class AgeEncoding(nn.Module):
    def __init__(self, n_embd, n_freq=32):
        super().__init__()
        # periods from about 1 week to about 300 years, in years
        freqs = torch.exp(torch.linspace(math.log(1 / 300.0), math.log(52.0), n_freq))
        self.register_buffer("freqs", freqs * 2 * math.pi)
        self.proj = nn.Linear(2 * n_freq, n_embd)

    def forward(self, age_years):
        x = age_years.unsqueeze(-1) * self.freqs
        return self.proj(torch.cat([torch.sin(x), torch.cos(x)], dim=-1))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n_head, self.p = c.n_head, c.dropout
        self.ln1, self.ln2 = nn.LayerNorm(c.n_embd), nn.LayerNorm(c.n_embd)
        self.qkv = nn.Linear(c.n_embd, 3 * c.n_embd)
        self.proj = nn.Linear(c.n_embd, c.n_embd)
        self.mlp = nn.Sequential(nn.Linear(c.n_embd, 4 * c.n_embd), nn.GELU(), nn.Linear(4 * c.n_embd, c.n_embd),
                                 nn.Dropout(c.dropout))

    def forward(self, x, allowed):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        q, k, v = (z.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=self.p if self.training else 0.0)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, C))
        return x + self.mlp(self.ln2(x))


def build_attn_mask(tokens, age_years, n_head=None, mask_ties=True):
    """Boolean mask of shape (B, 1, T, T), True = BLOCKED. Causal, no attention to PAD keys, and
    (optionally) no attention to other tokens with the same age. Each position always sees itself."""
    B, T = tokens.shape
    dev = tokens.device
    m = torch.ones(T, T, dtype=torch.bool, device=dev).triu(1).unsqueeze(0).expand(B, T, T).clone()
    m |= (tokens == PAD).unsqueeze(1)
    if mask_ties:
        m |= (age_years.unsqueeze(2) == age_years.unsqueeze(1))
    m &= ~torch.eye(T, dtype=torch.bool, device=dev).unsqueeze(0)
    return m.unsqueeze(1)


class Delphi(nn.Module):
    def __init__(self, c: DelphiConfig, ignore_in_rate=()):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab_size, c.n_embd)
        self.age = AgeEncoding(c.n_embd)
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layer)])
        self.ln = nn.LayerNorm(c.n_embd)
        self.head = nn.Linear(c.n_embd, c.vocab_size, bias=False)
        self.head.weight = self.tok.weight                     # tied, as in Delphi
        self.apply(self._init_weights)                         # GPT-2 / nanoGPT init, as in Delphi
        # A shared scalar added to every logit. It cancels in the softmax (cross entropy is unchanged)
        # and only sets the overall event rate, initialised at 0.5 events per year instead of
        # vocab_size per year, which otherwise makes the first steps of the time loss explode.
        self.log_rate_offset = nn.Parameter(torch.tensor(math.log(0.5) - math.log(c.vocab_size)))
        ign = torch.zeros(c.vocab_size, dtype=torch.bool)
        ign[list(set(ignore_in_rate) | {PAD})] = True
        self.register_buffer("ignore_in_rate", ign)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, tokens, age_years):
        x = self.drop(self.tok(tokens) + self.age(age_years))
        allowed = ~build_attn_mask(tokens, age_years, None, self.c.mask_ties)
        for b in self.blocks:
            x = b(x, allowed)
        logits = self.head(self.ln(x)) + self.log_rate_offset
        return logits.masked_fill(self.ignore_in_rate, -1e4)   # these tokens are never predicted

    def loss(self, tokens, age_years, end_years=None):
        """Returns a dict with 'loss' and diagnostics. end_years: per row end of follow up (NaN if
        the training window does not reach the end of the sequence); used by objective 'pp'."""
        full = self(tokens, age_years)              # causal, so position i only sees <= i
        logits = full[:, :-1]
        tgt = tokens[:, 1:]
        valid = (tgt != PAD) & (tokens[:, :-1] != PAD)
        log_rate = torch.logsumexp(logits, dim=-1)
        if self.c.objective == "pp" or self.c.censor:
            real = valid & (tgt != NO_EVENT)
            dt = (age_years[:, 1:] - age_years[:, :-1]).clamp(min=0)
            comp = (torch.exp(log_rate) * dt * valid).sum()
            mark = -(logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * real).sum()
            end = torch.zeros((), device=tokens.device)
            if end_years is not None:
                last = (tokens != PAD).sum(1) - 1
                idx = torch.arange(tokens.size(0), device=tokens.device)
                ok = torch.isfinite(end_years)
                gap = (torch.nan_to_num(end_years) - age_years[idx, last]).clamp(min=0) * ok
                end = (torch.exp(torch.logsumexp(full[idx, last], dim=-1)) * gap).sum()
            n = real.sum().clamp(min=1)
            total = (mark + comp + end) / n
            return {"loss": total, "ce": (mark / n).detach(), "time": (comp / n).detach(),
                    "censor": (end / n).detach()}
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
        ce = (ce.view_as(tgt) * valid).sum() / valid.sum().clamp(min=1)
        dt = (age_years[:, 1:] - age_years[:, :-1]).clamp(min=self.c.t_min_years)
        nll_t = -log_rate + torch.exp(log_rate) * dt
        nll_t = (nll_t * valid).sum() / valid.sum().clamp(min=1)
        return {"loss": ce + nll_t, "ce": ce.detach(), "time": nll_t.detach()}

    @torch.no_grad()
    def rates_at_last(self, tokens, age_years):
        """Per token rates (per year) at the last real position of each row."""
        logits = self(tokens, age_years)
        last = (tokens != PAD).sum(1) - 1
        return torch.exp(logits[torch.arange(tokens.size(0), device=tokens.device), last])
