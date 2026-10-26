"""Anchored scoring of the official Delphi-2M checkpoint (P2 review weakness 3)."""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from official_anchored_calibration import query_sequence, score_official, outcome_columns, DAY
from delphi_aou.replication.evaluate import OFFICIAL


def test_query_sequence_cutoff_padding_and_order():
    ages = np.array([0, 3000, 9000, 12000, 20000]); toks = np.array([1, 4, 50, 60, 70])  # sex, BMI, three diseases
    t, a, last = query_sequence(ages, toks, cutoff_days=12000, query_days=12030)
    assert t[0] == 2 and a[0] == 0                     # sex token first, shifted by +1
    assert t[-1] == 1 and a[-1] == 12030               # query "no event" (bin 0 -> model 1) at the anchor
    assert 71 not in t                                 # event after the cutoff is dropped
    assert np.all(np.diff(a) >= 0)
    pads = a[(t == 1)][:-1]
    assert len(pads) == int(12000 // (5 * DAY)) + 1 and pads.max() <= 12000   # every 5 years, not after the last record
    assert t[last] == 61 and a[last] == 12000         # last real token


def test_query_sequence_truncation_keeps_sex():
    ages = np.arange(0, 200 * 30, 30); toks = np.r_[2, np.arange(20, 219)]
    t, a, _ = query_sequence(ages, toks, cutoff_days=10 ** 6, query_days=10 ** 6 + 1, pad_every_years=1000)
    assert len(t) == 80 and t[0] == 3 and t[-1] == 1


def test_outcome_columns():
    L = {"E11": 100, "Death": 1269}
    assert outcome_columns(["D=E11", "D=Z99", "DEATH"], L).tolist() == [101, -1, 1270]


def test_score_official_readouts_match_direct_forward():
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    torch.manual_seed(0)
    m = Delphi(DelphiConfig(vocab_size=200, n_layer=2, n_head=2, n_embd=32, block_size=96, dropout=0.0)).eval()
    rng = np.random.default_rng(0)
    rows = []
    for pid in [7, 3, 9]:
        ages = np.sort(rng.integers(1, 25000, 30)); rows += [(pid, 0, 1)] + [(pid, int(x), int(rng.integers(12, 190))) for x in ages]
    val = np.array(rows, np.int64)
    persons = np.array([3, 9, 7]); q = np.array([15000.0, 20000.0, 9000.0]); cut = q - 30
    cols = np.array([20, 50, -1, 150])
    sq, sl = score_official(m, val, persons, cut, q, cols, 5, "cpu", batch=2)
    assert np.isnan(sq[:, 2]).all() and np.isfinite(sq[:, [0, 1, 3]]).all()
    for k, pid in enumerate(persons):                  # compare with a single unpadded forward pass
        v = val[val[:, 0] == pid]; v = v[np.argsort(v[:, 1], kind="stable")]
        t, a, last = query_sequence(v[:, 1], v[:, 2], cut[k], q[k])
        with torch.no_grad():
            lg = m(torch.tensor(t[None]), torch.tensor(a[None]))[0][0]
        want_q = -np.expm1(-np.exp(lg[-1, [20, 50, 150]].numpy().astype(np.float64)) * 5 * DAY)
        want_l = -np.expm1(-np.exp(lg[last, [20, 50, 150]].numpy().astype(np.float64)) * 5 * DAY)
        assert np.allclose(sq[k, [0, 1, 3]], want_q, rtol=1e-4, atol=1e-6)
        assert np.allclose(sl[k, [0, 1, 3]], want_l, rtol=1e-4, atol=1e-6)


def test_history_tokens_counts_rows_up_to_cutoff():
    import numpy as np, sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from official_anchored_calibration import history_tokens
    val = np.array([[1, 0, 1], [1, 100, 5], [1, 200, 6], [2, 0, 2], [2, 50, 7]])
    assert history_tokens(val, np.array([1, 1, 2, 3]), np.array([150.0, 200.0, 10.0, 99.0])).tolist() == [2, 3, 1, 0]
