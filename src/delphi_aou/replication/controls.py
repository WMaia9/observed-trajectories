"""Input controls for the official Delphi evaluation (same checkpoint, same cases and controls).

Token ids inside get_batch output are labels.csv ids + 1: padding 0, no event 1, female 2, male 3,
lifestyle 4 to 12, diseases 13 to 1268, death 1269.

  EraseIdentity   (evaluate.py) disease and death tokens become NO_EVENT: timing and count kept, identity lost
  ShuffleIdentity every disease and death token is replaced through one fixed random permutation of the disease
                  ids: timing, count and 'something was recorded' kept, clinical meaning destroyed
  AgeSexRates     no model at all: the score for disease k at a position is the training-set log rate of k at
                  that age (year) and sex. Since the official AUC is computed within 5-year age brackets and sex,
                  this is the floor that age and sex alone can reach.
"""
import numpy as np
import torch

FIRST_DISEASE, DEATH, VOCAB = 13, 1269, 1270


class ShuffleIdentity(torch.nn.Module):
    def __init__(self, model, seed=0):
        super().__init__(); self.m = model
        ids = np.arange(FIRST_DISEASE, DEATH + 1)
        perm = np.random.default_rng(seed).permutation(ids)
        lut = np.arange(VOCAB + 1); lut[ids] = perm
        self.register_buffer("lut", torch.as_tensor(lut, dtype=torch.long))

    def forward(self, x, ages, *rest):
        return self.m(self.lut.to(x.device)[x], ages, *rest)


class AgeSexRates(torch.nn.Module):
    def __init__(self, train_rows, max_year=110):
        """train_rows: uint32 array [n, 3] of (patient, age_days, label id) from train.bin."""
        super().__init__()
        r = train_rows.astype(np.int64)
        pid, age, tok = r[:, 0], r[:, 1], r[:, 2] + 1                # to batch ids
        import pandas as pd
        df = pd.DataFrame({"p": pid, "yr": np.clip(age // 365, 0, max_year), "tok": tok})
        male_ids = set(df.loc[df.tok == 3, "p"])
        df["s"] = df["p"].isin(male_ids).astype(int)
        counts = np.zeros((2, max_year + 1, VOCAB), np.float64)
        dis = df[df.tok >= FIRST_DISEASE]
        np.add.at(counts, (dis.s.to_numpy(), dis.yr.to_numpy(), dis.tok.to_numpy()), 1)
        last = df.groupby("p").agg(yr=("yr", "max"), s=("s", "first"))   # persons observed up to each age
        risk = np.zeros((2, max_year + 1))
        for sx in (0, 1):
            ends = np.bincount(last.loc[last.s == sx, "yr"], minlength=max_year + 1)
            risk[sx] = ends[::-1].cumsum()[::-1]                          # observed at age a if last age >= a
        rate = np.log((counts + 0.5) / (risk[:, :, None] + 1.0))
        self.register_buffer("rate", torch.as_tensor(rate, dtype=torch.float32))
        self.max_year = max_year

    def forward(self, x, ages, *rest):
        male = (x == 3).any(1).long()                                # [B]
        yr = torch.clamp((ages // 365).long(), 0, self.max_year)     # [B, T]
        rate = self.rate.to(x.device)
        return (rate[male[:, None].expand_as(yr), yr],)
