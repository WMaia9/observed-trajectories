"""Direct horizon head (HORIZON_HEAD=1,5): a per-disease risk-within-H head on the state at the last visit plus the
time from that visit to the anchor, trained on observed outcomes at pseudo anchors. No rollout, so no cascade."""
import numpy as np
import torch
from delphi_aou.data import visits as s5v
from delphi_aou.train import visit as s5
from delphi_aou.models.visit_pp import VisitPP, VisitConfig
from test_stage3b import make_world


def _setup(n=600):
    cohort, fo, visits = make_world(n=n, seed=3)
    vis, _ = s5v.build(cohort, fo, [visits])
    data = s5.VisitData(cohort, fo, vis)
    return cohort, data


def test_labels_match_manual_computation():
    cohort, data = _setup()
    rng = np.random.default_rng(0)
    idx = np.arange(40)
    b = data.batch(idx, rng, 64, "cpu")
    for bb, i in enumerate(idx):
        a, _ = data.tokens(i)
        for k, v in zip(data.rec_k[i], data.rec_v[i]):
            if v < len(a):
                assert abs(float(b["rec_age"][bb, k]) - a[v]) < 1e-4
    u, dt = b["hz_u"], b["dt_next"]
    span = b["has_next"] | b["censored"]
    assert (u >= 0).all() and (u[span] <= dt[span] + 1e-6).all() and (u[~span] == 0).all()


def test_trains_and_head_estimator_runs():
    from delphi_aou.baselines import classical as s3
    cohort, data = _setup()
    an = s3.build_anchors(cohort).head(30).reset_index(drop=True)
    names = [t for t in ["D=V01", "D=Z99"] if t in data.cidx] + ["DEATH"]
    for det, anch in ((False, "none"), (True, "last_dx")):
        cfg = VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, detection=det, rate_anchor=anch, horizons=(1, 5))
        m = VisitPP(cfg)
        b = data.batch(np.arange(64), np.random.default_rng(0), 64, "cpu")
        out = m.loss(b); out["loss"].backward()
        assert torch.isfinite(out["loss"]) and float(out["hz"]) > 0 and m.hz.weight.grad.abs().sum() > 0
        r = s5.sample_risk(m, data, an, names, (1, 5), 1, n_samples=2, max_steps=10, estimator="head")
        assert np.isfinite(r[5]).all() and (r[5] >= 0).all() and (r[5] <= 1).all()


def test_head_learns_the_observed_risk_level():
    """After a short training run the mean predicted 5-year risk is within a factor 2 of the share recorded."""
    cohort, data = _setup(n=800)
    torch.manual_seed(0)
    cfg = VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, horizons=(5,), hz_weight=1.0)
    m, _ = s5.train(data, cfg, 300, 64, 3e-3, 0, "cpu", window=64, eval_every=300, log=lambda *a: None)
    b = data.batch(np.flatnonzero(data.split == "val")[:200], np.random.default_rng(1), 64, "cpu")
    with torch.no_grad():
        h = m.encode(b["vt"], b["age"], b["gap"], b["sex"], b["code_b"], b["code_t"], b["code_k"], b["valid"])
        p = torch.sigmoid(m.horizon_logits(h, b["hz_u"]))[..., 0]
    tau = b["age"] + b["hz_u"]
    ev = b["rec_age"][:, None, :] <= tau.unsqueeze(-1) + 5
    obs = ev | (b["exit_age"][:, None, None] >= tau.unsqueeze(-1) + 5) | b["died_b"][:, None, None]
    span = (b["has_next"] | b["censored"]) & (b["hz_u"] > 0)
    msk = obs & b["at_risk"] & span.unsqueeze(-1)
    pred, rate = float(p[msk].mean()), float(ev[msk].float().mean())
    assert rate > 0 and 0.5 < pred / rate < 2.0, (pred, rate)


def test_old_checkpoints_load_without_head():
    a = VisitPP(VisitConfig(7, n_layer=1, n_head=2, n_embd=16))
    assert not any(k.startswith("hz.") for k in a.state_dict())
    VisitPP(VisitConfig(7, n_layer=1, n_head=2, n_embd=16)).load_state_dict(a.state_dict())


def test_person_weighting_equals_plain_when_every_person_has_the_same_number_of_anchors():
    """hz_person rescales by the mean number of pseudo anchors per person: identical to the plain loss when all
    persons have the same count, and it shifts weight toward people with few anchors otherwise."""
    cohort, data = _setup()
    b = data.batch(np.arange(64), np.random.default_rng(0), 64, "cpu")
    torch.manual_seed(0)
    m0 = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, horizons=(1, 5)))
    m1 = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, horizons=(1, 5), hz_person=True))
    m1.load_state_dict(m0.state_dict())
    m0.eval(); m1.eval()
    a, c = float(m0.loss(b)["hz"]), float(m1.loss(b)["hz"])
    assert np.isfinite(c) and c > 0 and abs(c - a) / a < 0.5          # same scale
    one = data.batch(np.arange(1), np.random.default_rng(0), 64, "cpu")  # a single person: identical by construction
    assert abs(float(m0.loss(one)["hz"]) - float(m1.loss(one)["hz"])) < 1e-4 * max(1.0, float(m0.loss(one)["hz"]))
