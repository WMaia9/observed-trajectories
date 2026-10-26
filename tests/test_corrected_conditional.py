"""Helpers of experiments/audit_corrected_conditional.py (P2 corrected conditional analyses)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from audit_corrected_conditional import target_noevent, subset, analyse


def _rows(rng, n=400):
    case = rng.random(n) < 0.3
    real = rng.normal(size=n) + case
    return [{"person": np.arange(n, dtype=np.int32), "case": case, "bracket": np.zeros(n, np.int8),
             "z": np.full(n, 100.0, np.float32), "zk": np.full(n, 200.0, np.float32),
             "nreal": rng.integers(0, 6, n).astype(np.int16), "recency": rng.uniform(1, 900, n).astype(np.float32),
             "first": np.full(n, 50.0, np.float32), "scores": np.stack([real, real * 0.5], 1).astype(np.float16),
             "k": 50, "sex": "female"}]


def test_target_noevent_flags_only_noevent_targets():
    rows = _rows(np.random.default_rng(0), 4)
    y = np.array([[1, 5], [7, 1], [1, 1], [9, 9]]); b = np.array([[200.0, 300.0], [200.0, 300.0], [10.0, 200.0], [1.0, 2.0]])
    rec = target_noevent(rows, y, b)
    assert rows[0]["target_noevent"].tolist() == [True, False, True, False]
    assert rec == 0.75


def test_subset_slices_every_point_array():
    rows = _rows(np.random.default_rng(1))
    s = subset(rows, lambda r: r["nreal"] >= 1)
    m = rows[0]["nreal"] >= 1
    assert len(s[0]["case"]) == m.sum() and s[0]["scores"].shape == (m.sum(), 2) and s[0]["k"] == 50
    assert subset(rows, lambda r: np.zeros(len(r["case"]), bool)) == []


def test_analyse_runs_and_real_beats_noise():
    rows = _rows(np.random.default_rng(2), 2000)
    res = analyse(rows, ["real", "erased"], B=5)
    assert res["plain"]["real"] > 0.65 and abs(res["plain"]["real"] - res["plain"]["erased"]) < 1e-3
    assert 0 <= res["share_controls_no_prior_diagnosis"] <= 1 and "auc_real" in res["matched_default_bootstrap"]


def test_paired_gap_join_checks_points():
    import pytest
    from audit_paired_gap import join, by_bins
    rng = np.random.default_rng(5)
    a = _rows(rng, 300); f = [dict(a[0])]
    f[0]["scores"] = a[0]["scores"][:, ::-1].copy()
    j = join(a, f, ["real", "erased"], ["real", "erased"])
    assert j[0]["scores"].shape[1] == 3 and np.array_equal(j[0]["scores"][:, 2], a[0]["scores"][:, 0])
    g = [dict(a[0])]; g[0]["zk"] = a[0]["zk"] + 1
    with pytest.raises(ValueError):
        join(a, g, ["real", "erased"], ["real", "erased"])
    res = by_bins(j, ["real", "erased", "idfree"])
    assert set(res) == {"0-1y", "1-2y", "2-5y", "5-infy"}


def test_collect_multiple_keys_differ(tmp_path):
    """Selection with several keys from one scoring pass: same scores, different draws, meta per key."""
    import torch, pandas as pd, json as _json
    from delphi_aou.replication import audit
    from delphi_aou.replication.evaluate import OFFICIAL
    rng = np.random.default_rng(0)
    n, T = 600, 40
    x = np.zeros((n, T), np.int64); x[:, 0] = rng.integers(2, 4, n)
    x[:, 1:] = rng.choice([1, 50, 60], (n, T - 1), p=[0.94, 0.03, 0.03])
    ages = np.sort(rng.uniform(30 * 365.25, 85 * 365.25, (n, T)), 1); ages[:, 0] = 0
    y = np.roll(x, -1, 1); y[:, -1] = 1; b = np.roll(ages, -1, 1); b[:, -1] = ages[:, -1] + 400
    d = [torch.tensor(x), torch.tensor(ages, dtype=torch.float32), torch.tensor(y), torch.tensor(b, dtype=torch.float32)]
    class M(torch.nn.Module):
        def forward(self, x, a, *r):
            return (torch.randn(x.shape[0], x.shape[1], 1270, generator=torch.Generator().manual_seed(0)),)
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    import delphi_aou.replication.audit as A
    import sys as _s; _s.path.insert(0, str(OFFICIAL)); import evaluate_auc as ev
    orig = ev.get_common_diseases; ev.get_common_diseases = lambda *_: np.array([50, 60])
    try:
        A.collect({"real": M()}, d, labels, tmp_path, device="cpu", workers=1, keys=[1, 2])
    finally:
        ev.get_common_diseases = orig
    m1, r1 = A.load(tmp_path / "key1"); m2, r2 = A.load(tmp_path / "key2")
    assert m1["key"] == 1 and m2["key"] == 2 and len(r1) > 0
    assert any(not np.array_equal(p["z"], q["z"]) for p, q in zip(r1, r2))
