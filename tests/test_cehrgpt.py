"""CEHR-GPT conversion and read-out (second model family). Needs the cehrgpt package (skipped otherwise)."""
import numpy as np
import pytest

from delphi_aou.cehrgpt.sequences import to_record, att_token, records, last_visit_day


def _arr(rng, n=120):
    rows = []
    for p in range(n):
        rows.append((p, 0.0, 2 + p % 2))
        for _ in range(rng.integers(1, 25)):
            rows.append((p, float(rng.integers(30 * 365, 75 * 365)), int(rng.integers(4, 60))))
        if p % 7 == 0:
            rows.append((p, 50 * 365.0, 1))                                   # no-event token: dropped
    return np.array(sorted(rows, key=lambda r: (r[0], r[1])))


def test_record_layout_and_prefix():
    a = np.array([0.0, 100.0, 100.0, 100.0, 500.0, 2000.0]); t = np.array([3, 20, 20, 5, 30, 1])
    r = to_record(7, a, t)
    assert r["concept_ids"] == ["year:unknown", "age:0", "Gender/male", "Race/0", "[VS]", "9202", "t20", "t5", "[VE]",
                                "D400", "[VS]", "9202", "t30", "[VE]"]
    assert len(r["epoch_times"]) == len(r["concept_ids"]) == r["num_of_concepts"]
    p = to_record(7, a, t, upto=200.0)
    assert p["concept_ids"][-1] == "[VE]" and "t30" not in p["concept_ids"]
    assert to_record(7, a, t, upto=50.0) is None
    q = to_record(7, a, t, upto=50.0, prompt_age_day=60 * 365.25)
    assert q["concept_ids"] == ["year:unknown", "age:60", "Gender/male", "Race/0"]
    assert att_token(1080) == "D1080" and att_token(1081) == "LT"
    assert last_visit_day(a, t, 499.0) == 100.0 and last_visit_day(a, t, 50.0) is None


cg = pytest.importorskip("cehrgpt")


def _setup():
    import datasets, torch
    from delphi_aou.cehrgpt import model as M
    rng = np.random.default_rng(0)
    arr = _arr(rng)
    ds = datasets.Dataset.from_list(records(arr)).train_test_split(0.2, seed=0)
    ds = datasets.DatasetDict({"train": ds["train"], "validation": ds["test"]})
    tok = M.train_tokenizer(ds)
    torch.manual_seed(0)
    m = M.make_model(tok, n_embd=32, n_layer=2, n_head=2, n_positions=256, dropout=0.0).eval()
    return arr, ds, tok, m, M


def test_cached_branches_equal_explicit_prefix():
    import torch
    torch.set_grad_enabled(False)
    arr, ds, tok, m, M = _setup()
    v = M.Vocab(tok)
    rec = records(arr)[3]
    enc = v.encode(rec)
    ve = [i for i, c in enumerate(rec["concept_ids"]) if c == "[VE]"]
    br = [[v.rep_ids[2], v.vs, v.vt], [v.rep_ids[7], v.vs, v.vt]]
    for erase in (False, True):
        p, q = M.code_dists_after(m, v, [enc, enc[:5]], [ve, [4]], br, "cpu", erase=erase)
        for n, e in enumerate(ve):
            for b, bb in enumerate(br):
                ids = torch.as_tensor([enc[:e + 1] + bb])
                ids = v.erase(ids) if erase else ids
                lg = m(input_ids=ids, ages=torch.zeros_like(ids)).logits[0, -1][torch.as_tensor(v.code_ids)]
                assert np.allclose(q[n, b], torch.softmax(lg, -1).numpy(), atol=1e-5)
            ids = torch.as_tensor([enc[:e + 1]]); ids = v.erase(ids) if erase else ids
            pn = torch.softmax(m(input_ids=ids, ages=torch.zeros_like(ids)).logits[0, -1], -1).numpy()
            assert np.allclose(p[n], pn, atol=1e-5)


def test_erase_and_weights_and_scores():
    import torch
    arr, ds, tok, m, M = _setup()
    v = M.Vocab(tok)
    ids = torch.as_tensor([[v.code_of_k[20], v.code_of_k[5], v.vs, v.code_of_k[1269] if v.code_of_k[1269] >= 0 else v.vs]])
    e = v.erase(ids)
    assert e[0, 0] == v.oov and e[0, 1] == v.code_of_k[5] and e[0, 2] == v.vs
    W = M.bin_weights(np.ones((2, len(v.att_days))) / len(v.att_days), v.att_days, np.array([1.0, 2000.0]))
    assert np.allclose(W.sum(1), 1) and W[1, -1] == 1.0 and W[0, 0] > 0
    from delphi_aou.cehrgpt.sequences import by_person
    people = []
    for pid, a, t in list(by_person(arr))[:6]:
        z = np.array([a.max(), 20 * 365.0, a.min() + 1]); k = np.array([20, 21, 22])
        people.append((pid, a, t, z, k))
    s = M.score_people(m, v, people, "cpu")
    s2 = M.score_people(m, v, people, "cpu", erase=True)
    assert all(np.isfinite(x).all() and (x >= 0).all() and (x <= 1).all() for x in s + s2)
    assert all(x[1] > 0 for x in s)                                          # prompt-only prefix scored
    loss = M.lm_loss(m.train(), _batch(ds, tok), v, erase=True)
    assert torch.isfinite(loss)
    torch.set_grad_enabled(True)


def _batch(ds, tok):
    from cehrgpt.data.hf_cehrgpt_dataset_collator import CehrGptDataCollator
    from delphi_aou.cehrgpt.model import tokenize
    pds = tokenize(ds, tok)
    return CehrGptDataCollator(tok, max_length=128, pretraining=True)([pds["train"][i] for i in range(8)])


def test_fast_weights_and_scores_match_reference():
    import torch
    torch.set_grad_enabled(False)
    arr, ds, tok, m, M = _setup()
    v = M.Vocab(tok)
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(len(v.att_days)), size=5)
    rows = rng.integers(0, 5, 40); g = rng.integers(1, 1500, 40).astype(float)
    assert np.allclose(M.bin_weights_fast(p, v.att_days, rows, g), M.bin_weights(p[rows], v.att_days, g))
    from delphi_aou.cehrgpt.sequences import by_person, to_record
    pid, a, t = list(by_person(arr))[5]
    z = np.array([a.max() + 400, np.sort(a)[2] + 3, 25 * 365.0]); k = np.array([20, 30, 40])
    s = M.score_people(m, v, [(pid, a, t, z, k)], "cpu")[0]
    for j in range(2):                                    # explicit prefix + each branch, mixed by the weights
        rec = to_record(pid, a, t, upto=z[j]); enc = v.encode(rec)
        ids = torch.as_tensor([enc])
        pn = torch.softmax(m(input_ids=ids, ages=torch.zeros_like(ids)).logits[0, -1], -1).numpy()[v.att_ids]
        d = rec["epoch_times"][-1] / 86400.0
        w = M.bin_weights(pn[None] / pn.sum(), v.att_days, np.array([max(1.0, np.ceil(z[j] - d))]))[0]
        qk = []
        for r in v.rep_ids:
            ids = torch.as_tensor([enc + [r, v.vs, v.vt]])
            lg = m(input_ids=ids, ages=torch.zeros_like(ids)).logits[0, -1][torch.as_tensor(v.code_ids)]
            qk.append(torch.softmax(lg, -1).numpy()[list(v.code_ids).index(v.code_of_k[k[j]])])
        assert np.isclose(s[j], float(w @ np.array(qk)), atol=1e-5)
    torch.set_grad_enabled(True)
