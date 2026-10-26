"""experiments/audit_keep_sex.py: the patched batch keeps the sex token and leaves kept people unchanged."""
import sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from audit_keep_sex import make_get_batch_keep_sex
from delphi_aou.replication.evaluate import OFFICIAL


def test_keep_sex_batch():
    sys.path.insert(0, str(OFFICIAL))
    from utils import get_p2i, get_batch
    rng = np.random.default_rng(0); rows = []
    for pid, n in [(5, 30), (6, 95), (7, 81), (8, 10)]:                    # two long histories fill the window
        ages = np.sort(rng.choice(np.arange(100, 30000), n, replace=False))
        rows += [(pid, 0, 1)] + [(pid, int(t), int(rng.integers(12, 1200))) for t in ages]
    val = np.array(rows, np.int64); p2i = get_p2i(val)
    kw = dict(select="left", block_size=80, device="cpu", padding="random", no_event_token_rate=5)
    x0 = get_batch(range(len(p2i)), val, p2i, **kw)[0].numpy()
    d = make_get_batch_keep_sex()(range(len(p2i)), val, p2i, **kw)
    x1, a1 = d[0].numpy(), d[1].numpy()
    k0 = ((x0 == 2) | (x0 == 3)).any(1)
    assert not k0.all()                                                    # the official batch loses someone
    assert ((x1 == 2) | (x1 == 3)).any(1).all() and x1.shape == x0.shape
    assert (x1[k0] == x0[k0]).all()
    r = np.flatnonzero(~k0)[0]
    assert x1[r, 0] == 2 and a1[r, 0] == 0 and np.all(np.diff(a1[r]) >= 0)
