"""Stage 0 and 1 validation tests. Run:  python -m pytest -q   (or python tests/test_stage01.py)
All data here is synthetic. No participant level data is used."""
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from delphi_aou import config as common
from delphi_aou.eval import metrics as eval_auc

RNG = np.random.default_rng(0)
BANDS = [(40, 60), (60, 80)]


def make_world(n=120_000, d=6, seed=1):
    r = np.random.default_rng(seed)
    age = r.uniform(40, 80, n)
    sex = r.integers(0, 2, n)
    z = r.normal(size=(n, d))                       # hidden patient risk factor
    a = np.linspace(-3.2, -2.2, d)                  # base rates of a few percent
    b = np.linspace(0.3, 0.9, d)                    # age slope
    g = np.linspace(0.6, 1.4, d)                    # strength of hidden factor
    logit = a + b * (age[:, None] - 60) / 10 + g * z
    p = 1 / (1 + np.exp(-logit))
    y = (r.random((n, d)) < p).astype(int)
    p_age = 1 / (1 + np.exp(-(a + b * (age[:, None] - 60) / 10)))
    return age, sex, p, p_age, y


def test_delong_matches_sklearn():
    age, sex, p, _, y = make_world()
    for d in range(3):
        auc, var = eval_auc.delong_auc(p[:, d], y[:, d])
        assert abs(auc - roc_auc_score(y[:, d], p[:, d])) < 1e-9
        assert var > 0


def test_oracle_auc_and_ci_coverage():
    """Oracle score = true probability. Its AUC must match a very large sample
    estimate, and the DeLong CI must cover it for most units."""
    age, sex, p, p_age, y = make_world()
    names = [f"D{i}" for i in range(p.shape[1])]
    elig = np.ones_like(y, dtype=bool)
    tab = eval_auc.evaluate_auc(p, y, elig, age, sex, names, BANDS)
    assert len(tab) > 20
    big_age, big_sex, big_p, _, big_y = make_world(n=1_500_000, seed=7)
    covered, total = 0, 0
    for _, row in tab.iterrows():
        lo, hi = map(float, row["band"].split("-"))
        d = int(row["disease"][1:])
        m = (big_age >= lo) & (big_age < hi) & (big_sex == row["sex"])
        truth = roc_auc_score(big_y[m, d], big_p[m, d])
        total += 1
        covered += int(row["ci_lo"] <= truth <= row["ci_hi"])
    assert covered / total >= 0.85, covered / total


def test_noisier_score_has_lower_auc():
    age, sex, p, p_age, y = make_world()
    names = [f"D{i}" for i in range(p.shape[1])]
    elig = np.ones_like(y, dtype=bool)
    full = eval_auc.summarize(eval_auc.evaluate_auc(p, y, elig, age, sex, names, BANDS))
    agesex = eval_auc.summarize(eval_auc.evaluate_auc(p_age, y, elig, age, sex, names, BANDS))
    assert full["macro_auc"] > agesex["macro_auc"] + 0.05


def test_shuffled_labels_near_chance():
    """Leakage test: shuffled labels must give AUC around 0.5."""
    age, sex, p, _, y = make_world()
    names = [f"D{i}" for i in range(p.shape[1])]
    elig = np.ones_like(y, dtype=bool)
    ys = y.copy()
    for d in range(ys.shape[1]):
        RNG.shuffle(ys[:, d])
    tab = eval_auc.evaluate_auc(p, ys, elig, age, sex, names, BANDS)
    assert abs(tab["auc"].mean() - 0.5) < 0.01
    frac_sig = ((tab["ci_lo"] > 0.5) | (tab["ci_hi"] < 0.5)).mean()
    assert frac_sig < 0.2


def test_prevalent_cases_inflate_auc_if_not_excluded():
    """Reproduces the old failure mode: recurrences of chronic codes make the
    task easy. With prevalent cases marked ineligible the AUC must be lower."""
    age, sex, p, _, y = make_world()
    n, d = y.shape
    names = [f"D{i}" for i in range(d)]
    prevalent = RNG.random((n, d)) < 0.05
    y_all = np.where(prevalent, 1, y)                      # recurrence counted as case
    score = np.where(prevalent, 0.99, p)                   # history makes it trivial
    inflated = eval_auc.summarize(eval_auc.evaluate_auc(
        score, y_all, np.ones_like(y, dtype=bool), age, sex, names, BANDS))["macro_auc"]
    honest = eval_auc.summarize(eval_auc.evaluate_auc(
        score, y_all, ~prevalent, age, sex, names, BANDS))["macro_auc"]
    assert inflated > honest + 0.05


def test_min_cases_filter():
    age, sex, p, _, y = make_world(n=3000)
    names = [f"D{i}" for i in range(p.shape[1])]
    elig = np.ones_like(y, dtype=bool)
    tab = eval_auc.evaluate_auc(p, y, elig, age, sex, names, BANDS, min_cases=25)
    assert (tab["n_cases"] >= 25).all() and (tab["n_controls"] >= 25).all()


def test_readout_uses_last_real_position():
    B, T, V = 4, 6, 5
    lengths = np.array([6, 3, 2, 5])
    tok = np.zeros((B, T), dtype=int)
    for i, L in enumerate(lengths):
        tok[i, :L] = np.arange(1, L + 1)
    logits = np.zeros((B, T, V))
    for t in range(T):
        logits[:, t, :] = t                     # position id is stored in the logits
    right = eval_auc.gather_last(logits, tok, pad_id=0)
    assert (right[:, 0] == lengths - 1).all()
    old_bug = logits[:, -1, :]
    assert (old_bug[:, 0] == T - 1).all()
    assert (old_bug[:, 0] != right[:, 0]).sum() == 3   # all rows shorter than the longest


def test_mask_keeps_age_zero_demographic_tokens():
    SEX, DIS, PAD = 3, 7, 0
    tok = np.array([[SEX, DIS, PAD, PAD]])
    age = np.array([[0.0, 12000.0, 0.0, 0.0]])          # sex token sits at age 0
    good = eval_auc.padding_mask(tok, pad_id=PAD)
    old_bug = age == 0
    assert good.tolist() == [[False, False, True, True]]
    assert old_bug[0, 0] is np.True_ or old_bug[0, 0]    # the old mask would hide SEX


def test_split_deterministic_order_free_and_balanced():
    ids = np.arange(1_000_000, 1_060_000)
    s1 = common.assign_split(ids)
    perm = RNG.permutation(len(ids))
    s2 = common.assign_split(ids[perm])
    assert (s1[perm] == s2).all()                        # order free
    frac = {k: float((s1 == k).mean()) for k in ["train", "val", "test"]}
    assert abs(frac["train"] - 0.8) < 0.01 and abs(frac["val"] - 0.1) < 0.01
    h1 = common.split_hash(ids, s1)
    assert h1 == common.split_hash(ids[perm], s2)        # same fingerprint everywhere
    s3 = s1.copy(); s3[0] = "test" if s3[0] != "test" else "train"
    assert common.split_hash(ids, s3) != h1              # detects any change


def test_no_typed_metric_constants():
    """Grep style check: no hard coded v7 numbers or labels in the code."""
    bad = re.compile(r"v7[_ ]?(top1|auc|ece|gap)|0\.766|0\.2494|0\.2211|0\.0188|0\.7700")
    root = Path(__file__).resolve().parents[1] / "src" / "delphi_aou"
    for f in root.rglob("*.py"):
        assert not bad.search(f.read_text()), f"typed constant in {f.name}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"{len(fns)} passed")
