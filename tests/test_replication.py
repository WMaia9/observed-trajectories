"""Replication path: our converter must produce files the UNCHANGED official Delphi code accepts.
Fast test: token ids, ordering, dedup and split. Slow test: official train.py runs on the converted
data and the official evaluation pipeline produces an AUC table."""
import os, subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from delphi_aou.replication import to_delphi_bin as tdb

OFFICIAL = tdb.OFFICIAL


def synthetic(n=3000, seed=0):
    r = np.random.default_rng(seed)
    L = tdb.label_dict()
    codes = [k for k, v in L.items() if v >= 12 and k != "Death"][:300]
    pid = np.arange(100, 100 + n)
    birth = pd.Timestamp("1940-01-01") + pd.to_timedelta(r.integers(0, 365 * 30, n), unit="D")
    persons = pd.DataFrame({"person_id": pid, "birth_date": birth,
                            "sex": r.choice(["Female", "Male", "Unknown"], n, p=[0.5, 0.48, 0.02])})
    rows = []
    for i in range(n):
        k = r.integers(1, 25)
        cs = r.choice(codes[:60], k)                       # common codes repeat, so first occurrence matters
        ages = np.sort(r.uniform(40, 75, k)) * 365.25
        for c, a in zip(cs, ages):
            rows.append((pid[i], c + ".1", birth[i] + pd.Timedelta(days=float(a))))
    icd = pd.DataFrame(rows, columns=["person_id", "code3", "date"])
    dd = r.random(n) < 0.05
    death = pd.DataFrame({"person_id": pid[dd], "date": birth[dd] + pd.to_timedelta(80 * 365, unit="D")})
    bmi = pd.DataFrame({"person_id": pid, "date": birth + pd.to_timedelta(55 * 365, unit="D"), "value": r.normal(28, 5, n)})
    smoking = pd.DataFrame({"person_id": pid, "date": birth + pd.to_timedelta(55 * 365, unit="D"),
                            "answer": r.choice(["Every Day", "Some Days", "Not At All"], n)})
    alcohol = pd.DataFrame({"person_id": pid, "date": birth + pd.to_timedelta(55 * 365, unit="D"),
                            "answer": r.choice(["Never", "Monthly Or Less", "2 to 3 Times a Week", "4 or More Times a Week"], n)})
    return persons, icd, death, bmi, smoking, alcohol


def test_token_ids_match_official_notebook():
    L = tdb.label_dict()
    assert (L["No"], L["Female"], L["Male"], L["BMI_low"], L["Alcohol_high"], L["A00"]) == (0, 1, 2, 3, 11, 12)
    assert L["Death"] == 1268                                  # vocab_size 1270 in config/train_delphi.py
    assert list(tdb.bmi_status([20, 25, 30])) == [3, 4, 5]
    assert list(tdb.smoking_status(["Every Day", "Some Days", "Not At All"])) == [8, 7, 6]
    assert list(tdb.alcohol_status(["4 or More Times a Week", "2 to 3 Times a Week", "Never"])) == [11, 10, 9]


def test_converter_order_dedup_split(tmp_path):
    rows, rep = tdb.build_rows(*synthetic())
    tr, va = tdb.official_order_and_split(rows)
    for d in (tr, va):
        df = pd.DataFrame(d.astype(np.int64), columns=["p", "a", "t"])
        assert not df.duplicated(["p", "t"]).any()                          # first occurrence only
        assert (df.groupby("p")["a"].apply(lambda s: s.is_monotonic_increasing)).all()
        assert (df["a"] >= 0).all()
        sex = df[df["t"].isin([1, 2])]
        assert (sex["a"] == 0).all()                                       # sex token at birth
    assert tr[:, 0].max() < va[:, 0].min()                                 # split by sorted id
    assert abs(len(np.unique(tr[:, 0])) / (len(np.unique(tr[:, 0])) + len(np.unique(va[:, 0]))) - 0.8) < 0.01
    assert rep["icd_rows_not_in_delphi_vocab"] == 0


@pytest.mark.slow
def test_official_code_trains_and_evaluates(tmp_path):
    rows, _ = tdb.build_rows(*synthetic())
    tr, va = tdb.official_order_and_split(rows)
    data = tdb.write_dataset(tr, va, tmp_path / "aou_test")
    # train.py uses os.path.join("data", dataset): an absolute dataset path is used as is
    try:
        out = tmp_path / "ckpt"
        cmd = [sys.executable, "train.py", "config/train_delphi.py", f"--dataset={data.resolve()}", f"--out_dir={out}",
               "--device=cpu", "--max_iters=60", "--lr_decay_iters=60", "--eval_interval=30", "--eval_iters=2",
               "--warmup_iters=10", "--always_save_checkpoint=True", "--batch_size=32"]
        r = subprocess.run(cmd, cwd=OFFICIAL, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        ck = out / "ckpt.pt"
        assert ck.exists()
        r = subprocess.run([sys.executable, "-m", "delphi_aou.replication.evaluate", "--input_path", str(data),
                            "--output_path", str(tmp_path / "eval"), "--model_ckpt_path", str(ck), "--device", "cpu",
                            "--dataset_subset_size", "400", "--filter_min_total", "100"], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        produced = list((tmp_path / "eval").glob("*"))
        assert produced, "no evaluation output"
    finally:
        pass


def test_oov_report_counts_and_suppression():
    import pandas as pd
    from delphi_aou.replication.oov_report import report
    icd = pd.DataFrame({"person_id": list(range(30)) * 2 + [99],
                        "code3": ["I10"] * 30 + ["R07"] * 30 + ["E08"]})
    r = report(icd, {"I10"})
    assert r["oov_rows"] == 31 and r["oov_share_by_first_letter"]["R"] > 0.9
    assert r["top_oov_codes_persons"] == {"R07": 30}            # E08 has 1 person: suppressed
    assert r["a_to_q_missing_codes_total"] == 1 and r["a_to_q_missing_from_vocab_persons"] == {}


def test_sweep_horizon_patch_matches_official_without_horizon():
    import sys
    import numpy as np
    from delphi_aou.replication.evaluate import OFFICIAL
    from delphi_aou.replication.sweep import patched_calibration_auc
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_auc as ev
    rng = np.random.default_rng(0)
    n, t = 400, 12
    ages = np.sort(rng.uniform(40, 80, (n, t)) * 365.25, 1)
    x = rng.integers(13, 20, (n, t)); y = rng.integers(13, 20, (n, t))
    tgt = ages + rng.uniform(1, 3000, (n, t))
    d = [x, ages, y, tgt]
    p = rng.normal(size=(n, t, 7)).astype("float16")
    fn, ns = patched_calibration_auc(ev)
    np.random.seed(1); a = ev.get_calibration_auc(2, 15, d, p, offset=0.1, age_groups=range(40, 80, 5), use_delong=True)
    np.random.seed(1); b = fn(2, 15, d, p, offset=0.1, age_groups=range(40, 80, 5), use_delong=True)
    assert [r["auc_delong"] for r in a] == [r["auc_delong"] for r in b]
    ns["HORIZON_DAYS"] = 365.25
    np.random.seed(1); c = fn(2, 15, d, p, offset=0.1, age_groups=range(40, 80, 5), use_delong=True)
    assert sum(r["n_diseased"] for r in c) < sum(r["n_diseased"] for r in a)


def _small_eval_setup(n=6000):
    import sys, torch
    sys.path.insert(0, str(OFFICIAL))
    from model import Delphi, DelphiConfig
    from utils import get_p2i, get_batch
    rows, _ = tdb.build_rows(*synthetic(n=n, seed=3))
    _, va = tdb.official_order_and_split(rows)
    val = va.astype(np.int64)
    torch.manual_seed(0)
    m = Delphi(DelphiConfig(block_size=96, vocab_size=1270, n_layer=1, n_head=2, n_embd=32, t_min=0.1,
                            ignore_tokens=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])).eval()
    p2i = get_p2i(val)
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=80, device="cpu", padding="random",
                  no_event_token_rate=5)
    labels = pd.read_csv(OFFICIAL / "delphi_labels_chapters_colours_icd.csv")
    return m, d, labels


def test_fast_eval_matches_official_pipeline():
    """Same diseases and age-bracket units as the official pipeline; per disease AUC equal up to the random
    choice of one event per person and bracket; result independent of the number of workers."""
    import os, sys
    from delphi_aou.replication import fast_eval
    sys.path.insert(0, str(OFFICIAL))
    import evaluate_auc as ev
    m, d, labels = _small_eval_setup()
    cwd = os.getcwd(); os.chdir(OFFICIAL)
    try:
        _, off = ev.evaluate_auc_pipeline(m, d, None, labels, filter_min_total=100, device="cpu", offset=0.1)
    finally:
        os.chdir(cwd)
    r2 = fast_eval.run(m, d, labels, [(0.1, None), (365.25, 1826.25)], device="cpu", workers=2, batch_size=64)
    r1 = fast_eval.run(m, d, labels, [(0.1, None), (365.25, 1826.25)], device="cpu", workers=1, batch_size=256)
    fast = r2[0][1]
    assert set(fast.token) == set(off.token)
    j = off.set_index("token")[["auc", "n_samples"]].join(fast.set_index("token")[["auc", "n_samples"]], rsuffix="_f")
    assert (j.n_samples == j.n_samples_f).all()
    assert abs(j.auc.mean() - j.auc_f.mean()) < 0.01 and np.corrcoef(j.auc, j.auc_f)[0, 1] > 0.9
    for ci in (0, 1):                                   # determinism across workers and batch sizes
        a, b = r1[ci][1].sort_values("token"), r2[ci][1].sort_values("token")
        assert np.allclose(a.auc.to_numpy(), b.auc.to_numpy(), atol=1e-3)


def test_input_controls_run_through_fast_eval():
    import torch
    from delphi_aou.replication import fast_eval
    from delphi_aou.replication.controls import ShuffleIdentity, AgeSexRates
    m, d, labels = _small_eval_setup(n=4000)
    rows, _ = tdb.build_rows(*synthetic(n=4000, seed=3))
    tr, _ = tdb.official_order_and_split(rows)
    sh = ShuffleIdentity(m).eval()
    x = d[0][:4]
    y = sh.lut[x]
    assert torch.equal(y[x < 13], x[x < 13]) and not torch.equal(y[x >= 13], x[x >= 13])
    ag = AgeSexRates(tr).eval()
    assert ag(d[0][:8], d[1][:8])[0].shape == (8, d[0].shape[1], 1270)
    for model in (sh, ag):
        r = fast_eval.run(model, d, labels, [(0.1, None)], device="cpu", workers=2, batch_size=256)
        assert 0.3 < r[0][1].auc.mean() < 0.8


def test_audit_collect_matches_fast_eval_and_stats(tmp_path):
    """The paired audit must reproduce the per-disease AUC of fast_eval (same official selection and seed key) and its
    statistics must behave: unit weights give the point AUC, identical variants give zero difference."""
    import torch
    from delphi_aou.replication import fast_eval, audit
    from delphi_aou.replication.evaluate import EraseIdentity
    m, d, labels = _small_eval_setup(n=4000)
    ref = fast_eval.run(m, d, labels, [(0.1, None)], device="cpu", workers=2, batch_size=256)[0][1].set_index("token")
    models = {"real": m, "real_copy": m, "erased": EraseIdentity(m).eval(),
              "erased_nolife": audit.EraseIdentityAndLifestyle(m).eval()}
    audit.collect(models, d, labels, tmp_path / "a", offset=0.1, device="cpu", workers=2, batch_size=256)
    meta, rows = audit.load(tmp_path / "a")
    pa = audit.point_aucs(rows, meta["variants"])
    j = ref[["auc"]].join(pa, how="inner")
    assert len(j) == len(ref) and np.allclose(j["auc"], j["real"], atol=2e-3)       # float16 rank ties only
    assert np.allclose(pa["real"], pa["real_copy"])
    # bootstrap: unit weights reproduce the point AUC; identical variants give an exactly zero difference
    u = next(audit._units(rows))[1]
    s = u["scores"][:, 0].astype(np.float32)
    assert abs(audit._auc(s, u["case"], np.ones((3, len(s))))[0] - audit._auc(s, u["case"])) < 1e-9
    bs = audit.bootstrap(rows, meta["variants"], B=20)
    assert bs["diff_real_minus_real_copy"]["ci"] == [0.0, 0.0]
    assert bs["mean_auc_real"]["ci"][0] <= bs["mean_auc_real"]["point"] <= bs["mean_auc_real"]["ci"][1] + 0.02
    # lifestyle erasure maps 4..1269 to no event and keeps sex
    x = torch.tensor([[2, 4, 12, 13, 1269, 1]])
    seen = {}
    class Spy(torch.nn.Module):
        def forward(self, x, a, *r):
            seen["x"] = x; return (torch.zeros(1),)
    audit.EraseIdentityAndLifestyle(Spy())(x, None)
    assert seen["x"].tolist() == [[2, 1, 1, 1, 1, 1]]
    # matched strata and wash-out run and give AUCs in range
    pm = audit.point_aucs(rows, meta["variants"], matched=True)
    pw = audit.point_aucs(rows, meta["variants"], filt=audit.washout())
    for t in (pm, pw):
        v = t["real"].dropna()
        assert len(v) > 0 and ((v >= 0) & (v <= 1)).all()
    assert {"count_only", "recency_only"} <= set(pa.columns)

