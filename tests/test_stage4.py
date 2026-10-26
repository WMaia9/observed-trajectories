"""Stage 4 validation on the Stage 3 synthetic world (D0 raises the hazard of D1 and D2).
Run: python tests/test_stage4.py   (about a minute on CPU)"""
import sys
from pathlib import Path
import numpy as np
import torch

from delphi_aou.train import delphi as s4
from delphi_aou.baselines import classical as s3
from delphi_aou.models.delphi import Delphi, DelphiConfig, build_attn_mask, PAD  # noqa: E402
from test_stage3 import make_world  # noqa: E402


def test_tie_mask():
    torch.manual_seed(0)
    c = DelphiConfig(vocab_size=10, n_layer=2, n_head=2, n_embd=16, dropout=0.0, mask_ties=True)
    m = Delphi(c).eval()
    age = torch.tensor([[0.0, 50.0, 50.0, 50.0, 60.0]])
    a = torch.tensor([[2, 3, 4, 5, 6]]); b = torch.tensor([[2, 3, 7, 5, 6]])     # differ at a same day sibling
    with torch.no_grad():
        la, lb = m(a, age), m(b, age)
    assert torch.allclose(la[0, 3], lb[0, 3], atol=1e-6)    # position 3 cannot see its same day sibling
    assert not torch.allclose(la[0, 4], lb[0, 4])           # a later day sees it
    c2 = DelphiConfig(vocab_size=10, n_layer=2, n_head=2, n_embd=16, dropout=0.0, mask_ties=False)
    m2 = Delphi(c2).eval(); m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        assert not torch.allclose(m2(a, age)[0, 3], m2(b, age)[0, 3])
    mk = build_attn_mask(torch.tensor([[2, 3, PAD]]), torch.tensor([[0.0, 1.0, 1.0]]), 1, True)
    assert mk[0, 0, 1, 2] and mk[0, 0, 0, 1] and not mk[0, 0, 1, 0]


def test_initial_loss_is_sane():
    """Realistic scale: 730 tokens, ages around 50 to 80 years. At init the time NLL and the
    cross entropy must be moderate (the default N(0,1) embedding init gave ~1e26)."""
    torch.manual_seed(0)
    m = Delphi(DelphiConfig(vocab_size=730, n_layer=4, n_head=4, n_embd=64))
    tok = torch.randint(2, 730, (64, 40)); tok[:, 0] = 2
    age = torch.sort(torch.rand(64, 40) * 30 + 50, dim=1).values; age[:, 0] = 0.0
    out = m.loss(tok, age)
    assert out["ce"] < 8 and out["time"] < 20, out


def test_train_learns_and_scoring_does_not_peek():
    cohort, fo = make_world(n=20000)
    logs = []
    model, res, sc, tab = s4.run(cohort, fo, mode="delphi", censor=False, horizon_years=5, n_diseases=4,
                                 iters=600, batch=128, n_layer=2, n_embd=32, n_head=2, lr=2e-3,
                                 log=logs.append)
    curve = res["val_loss_curve"]
    assert curve[-1][1] < curve[0][1] - 0.2, curve
    base = s3.run(cohort, fo, horizon_years=5, n_diseases=4, models=("agesex",), max_train_anchors=60000,
                  n_hist_tokens=4, verbose=False)
    def auc(t, d):
        t = t[t["disease"] == d]; return float(np.average(t["auc"], weights=t["n_cases"]))
    for d in ["D=X01", "D=X02"]:
        assert auc(tab, d) > auc(base["tables"]["agesex"], d) + 0.05, (d, auc(tab, d))
    # no peeking: moving an event that happens AFTER the anchor does not change the score
    data = s4.Data(cohort, fo)
    an = s3.build_anchors(cohort); te = an[an["split"] == "test"].reset_index(drop=True).head(50)
    names = ["D=X00", "D=X01", "D=X02", "D=X03", "DEATH"]
    s_before = s4.score(model, data, te, names, "delphi", 5, 1)
    i = data.row[te.loc[0, "person_id"]]
    a = te.loc[0, "anchor_age_days"] / 365.25
    data.tok[i] = np.concatenate([data.tok[i], [data.stoi["D=X03"]]])
    data.age[i] = np.concatenate([data.age[i], [a + 0.5]])
    s_after = s4.score(model, data, te, names, "delphi", 5, 1)
    assert np.allclose(s_before, s_after)
    print("D1/D2 AUC delphi:", [round(auc(tab, d), 3) for d in ["D=X01", "D=X02"]],
          "age-sex:", [round(auc(base["tables"]["agesex"], d), 3) for d in ["D=X01", "D=X02"]])


def test_observed_mode_and_censor_run():
    cohort, fo = make_world(n=6000)
    model, res, sc, tab = s4.run(cohort, fo, mode="observed", censor=True, horizon_years=5, n_diseases=4,
                                 iters=150, batch=64, n_layer=1, n_embd=16, n_head=2, lr=2e-3, log=lambda *a: None)
    assert np.isfinite(sc).all() and (sc >= 0).all() and (sc <= 1).all()


if __name__ == "__main__":
    test_tie_mask(); print("PASS tie mask")
    test_initial_loss_is_sane(); print("PASS initial loss")
    test_train_learns_and_scoring_does_not_peek(); print("PASS delphi learns, no peeking")
    test_observed_mode_and_censor_run(); print("PASS observed + censor")
