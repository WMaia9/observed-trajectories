"""Anchored onset rates (rate_anchor="last_dx"): onset rates are read at the last visit with a new diagnosis plus a
learned age slope, so visits without new diagnoses change them only through age."""
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
    b = data.batch(np.arange(64), np.random.default_rng(0), 64, "cpu")
    return cohort, data, b


def test_anchor_index():
    cnt = torch.tensor([[0., 2., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0.]])
    j = VisitPP.anchor_index(cnt)
    assert j.tolist() == [[0, 1, 1, 1, 4, 4], [0, 0, 0, 0, 0, 0]]


def test_equals_unanchored_when_every_visit_has_a_diagnosis():
    """If every position is an anchor and the slope is 0, anchored rates are the plain mark head."""
    torch.manual_seed(0)
    m = VisitPP(VisitConfig(5, n_layer=1, n_head=2, n_embd=16, dropout=0.0, rate_anchor="last_dx")).eval()
    h = torch.randn(2, 4, 16); age = torch.tensor([[50., 51, 52, 53], [60, 60.5, 61, 62]])
    o = m.heads(h); cnt = torch.ones(2, 4)
    assert torch.allclose(m.rates(h, o, age, cnt), o["log_rate"])
    # with no new diagnoses after entry, rates are the entry rate plus slope times elapsed years
    with torch.no_grad():
        m.rate_slope.fill_(float(np.arctanh(0.1 / 0.3) * 0.3))      # effective slope 0.1 after the tanh bound
    r = m.rates(h, o, age, torch.zeros(2, 4))
    assert torch.allclose(r, o["log_rate"][:, :1] + 0.1 * (age - age[:, :1]).unsqueeze(-1), atol=1e-6)


def test_trains_and_samples_tier_a_and_b():
    from delphi_aou.baselines import classical as s3
    cohort, data, b = _setup()
    an = s3.build_anchors(cohort).head(30).reset_index(drop=True)
    names = [t for t in ["D=V01", "D=Z99"] if t in data.cidx] + ["DEATH"]
    for det in (False, True):
        cfg = VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, detection=det, rate_anchor="last_dx")
        m = VisitPP(cfg)
        out = m.loss(b); out["loss"].backward()
        assert torch.isfinite(out["loss"]) and m.rate_slope.grad is not None
        r = s5.sample_risk(m, data, an, names, (1, 5), 1, n_samples=2, max_steps=10)
        assert np.isfinite(r[5]).all() and (r[5] >= 0).all() and (r[5] <= 1).all()
        assert (r[5][:, 0] >= r[1][:, 0] - 1e-6).all()


def test_old_checkpoints_load():
    """Models without rate_anchor have no rate_slope parameter, so saved Tier A/B checkpoints still load strictly."""
    a = VisitPP(VisitConfig(7, n_layer=1, n_head=2, n_embd=16))
    assert "rate_slope" not in a.state_dict()
    VisitPP(VisitConfig(7, n_layer=1, n_head=2, n_embd=16)).load_state_dict(a.state_dict())


def test_rb_nodx_estimator():
    """rb_nodx: a probability, monotone in the horizon, and equal to rb when no diagnosis can be sampled."""
    from delphi_aou.baselines import classical as s3
    cohort, data, b = _setup()
    an = s3.build_anchors(cohort).head(30).reset_index(drop=True)
    names = [t for t in ["D=V01", "D=Z99"] if t in data.cidx] + ["DEATH"]
    torch.manual_seed(0)
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32))
    r = s5.sample_risk(m, data, an, names, (1, 5), 1, n_samples=2, max_steps=10, estimator="rb_nodx")
    assert np.isfinite(r[5]).all() and (r[5] >= 0).all() and (r[5] <= 1).all()
    assert (r[5][:, 0] >= r[1][:, 0] - 1e-6).all()
    with torch.no_grad():
        m.mark.bias.fill_(-50.0)                     # no disease can be recorded: both estimators see the same history
    a = s5.sample_risk(m, data, an, names, (5,), 1, n_samples=2, max_steps=10, estimator="rb", seed=1)[5]
    c = s5.sample_risk(m, data, an, names, (5,), 1, n_samples=2, max_steps=10, estimator="rb_nodx", seed=1)[5]
    assert np.allclose(a, c, atol=1e-6)


def test_compensator_is_unbiased_where_truncated_product_is_not():
    """Toy check of the estimator algebra: n steps with constant p, stop at the first sampled recording.
    The truncated product 1 - prod_{j<=tau}(1 - p) is biased downward; the compensator sum_{j<=tau} p is unbiased."""
    rng = np.random.default_rng(0); p, n, R = 0.3, 6, 400_000
    hit = rng.random((R, n)) < p
    tau = np.where(hit.any(1), hit.argmax(1) + 1, n)
    truth = 1 - (1 - p) ** n
    prod = (1 - (1 - p) ** tau).mean(); comp = (p * tau).mean()
    assert abs(comp - truth) < 3e-3 and prod < truth - 0.05


def test_comp_estimator_runs_and_dominates_rb():
    from delphi_aou.baselines import classical as s3
    cohort, data, b = _setup()
    an = s3.build_anchors(cohort).head(30).reset_index(drop=True)
    names = [t for t in ["D=V01", "D=Z99"] if t in data.cidx] + ["DEATH"]
    torch.manual_seed(0)
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32))
    a = s5.sample_risk(m, data, an, names, (5,), 1, n_samples=4, max_steps=10, estimator="rb", seed=3)[5]
    c = s5.sample_risk(m, data, an, names, (5,), 1, n_samples=4, max_steps=10, estimator="comp", seed=3)[5]
    assert (c[:, :-1] >= a[:, :-1] - 1e-6).all() and (c <= 1).all()


def test_bounded_slope_keeps_rates_finite():
    """A huge raw slope and decades of elapsed age give bounded, finite log rates and a finite loss."""
    from delphi_aou.models.visit_pp import SLOPE_MAX
    torch.manual_seed(0)
    m = VisitPP(VisitConfig(5, n_layer=1, n_head=2, n_embd=16, dropout=0.0, rate_anchor="last_dx")).eval()
    with torch.no_grad():
        m.rate_slope.fill_(50.0)
    h = torch.randn(1, 3, 16); age = torch.tensor([[20., 60., 95.]])
    r = m.rates(h, m.heads(h), age, torch.zeros(1, 3))
    assert torch.isfinite(r).all() and r.max() <= 3.0 and r.min() >= -20.0
    assert torch.allclose(r[0, 1] - r[0, 0], torch.full((5,), 40 * SLOPE_MAX), atol=1e-4) or r[0, 1].max() == 3.0
    cohort, data, b = _setup()
    mm = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, detection=True, rate_anchor="last_dx"))
    with torch.no_grad():
        mm.rate_slope.fill_(50.0)
    assert torch.isfinite(mm.loss(b)["loss"])
