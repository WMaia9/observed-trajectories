"""Tier B (detection layer): reduces exactly to Tier A when detection is certain, and its likelihood is a valid
probability model (filter probabilities sum correctly)."""
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
    return data, b


def test_detection_one_equals_tier_a():
    data, b = _setup()
    torch.manual_seed(0)
    a = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, dropout=0.0)).eval()
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, dropout=0.0, detection=True)).eval()
    m.load_state_dict(a.state_dict(), strict=False)
    with torch.no_grad():
        m.det_h.bias.fill_(30.0)                     # d = 1
        la, lm = a.loss(b), m.loss(b)
    assert torch.allclose(la["mark"], lm["mark"], rtol=1e-4), (la["mark"], lm["mark"])
    assert torch.allclose(la["visit"], lm["visit"])


def test_filter_step_is_a_distribution():
    s0, s1 = torch.tensor([1.0]), torch.tensor([0.0])
    lam_dt, d = torch.tensor([0.3]), torch.tensor([0.4])
    total_rec, surv = 0.0, 1.0
    for _ in range(200):                            # P(recorded eventually) -> 1 for a disease with positive rate
        p, n0, n1 = VisitPP.filter_step(s0, s1, lam_dt, d)
        total_rec += surv * float(p); surv *= 1 - float(p)
        s0, s1 = n0, n1
    assert abs(total_rec - 1.0) < 1e-3
    # first visit: P(record) = d (1 - exp(-lam_dt))
    p, _, _ = VisitPP.filter_step(torch.tensor([1.0]), torch.tensor([0.0]), lam_dt, d)
    assert abs(float(p) - 0.4 * (1 - np.exp(-0.3))) < 1e-6


def test_tier_b_trains():
    data, b = _setup()
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, detection=True))
    out = m.loss(b); out["loss"].backward()
    assert torch.isfinite(out["loss"]) and m.det_k.grad is not None and torch.isfinite(m.det_k.grad).all()


def test_frozen_estimator():
    """Frozen-rate diagnostic: deterministic, a probability, and monotone in the horizon."""
    from delphi_aou.baselines import classical as s3
    cohort, fo, visits = make_world(n=600, seed=3)
    vis, _ = s5v.build(cohort, fo, [visits])
    data = s5.VisitData(cohort, fo, vis)
    an = s3.build_anchors(cohort).head(40).reset_index(drop=True)
    names = [t for t in ["D=V01", "D=Z99"] if t in data.cidx] + ["DEATH"]
    torch.manual_seed(0)
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, dropout=0.0)).eval()
    with torch.no_grad():
        r = s5.sample_risk(m, data, an, names, (1, 5), 1, n_samples=3, estimator="frozen")
        r1 = s5.sample_risk(m, data, an, names, (1, 5), 1, n_samples=1, estimator="frozen", seed=7)
    k = len(names) - 1
    assert np.allclose(r[5][:, :k], r1[5][:, :k], atol=1e-6)
    assert (r[5][:, :k] >= r[1][:, :k] - 1e-7).all() and (r[5] <= 1).all() and (r[1] >= 0).all()
    assert (r[5][:, :k] > 0).any()


def test_prefix_state_matches_full_filter():
    """With constant rate and detection, a cropped window started from prefix_state gives the same mark NLL as the
    full-history filter summed over the same positions."""
    data, _ = _setup()
    lens = np.array([len(data.v_age[i]) for i in range(len(data.pids))])
    idx = np.flatnonzero(lens > 30)[:16]
    torch.manual_seed(0)
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, dropout=0.0, detection=True)).eval()
    lam, d = 0.2, 0.35
    with torch.no_grad():
        m.mark.weight.zero_(); m.mark.bias.fill_(np.log(lam))
        m.det_h.weight.zero_(); m.det_h.bias.fill_(np.log(d / (1 - d))); m.det_vt.weight.zero_(); m.det_k.zero_()
        W = 12
        cb = data.batch(idx, np.random.default_rng(5), W, "cpu")
        assert "pre_age" in cb
        fb = data.batch(idx, None, 4000, "cpu")
        s = (cb["pre_mask"].sum(1) - 1).clamp(min=0)                    # crop start per person
        def mark_nll(b, start, width):
            B, T = b["dt_next"].shape; K = data.K
            lr = torch.full((B, T, K), np.log(lam)); x = torch.exp(lr) * b["dt_next"].unsqueeze(-1)
            ar = b["at_risk"] & b["has_next"].unsqueeze(-1); y = b["y_next"]
            if "pre_age" in b:
                s0, s1 = m.prefix_state(lr[:, 0], torch.full((B,), float(m.det_h.bias)), b["pre_age"], b["pre_vt"], b["pre_mask"])
            else:
                s0, s1 = torch.ones(B, K), torch.zeros(B, K)
            tot = torch.zeros(B)
            for t in range(T):
                p, n0, n1 = VisitPP.filter_step(s0, s1, x[:, t], torch.full((B, K), d))
                inw = ((t >= start) & (t < start + width)).unsqueeze(-1) & ar[:, t]
                tot -= (torch.where(y[:, t], torch.log(p), torch.log1p(-p)) * inw).sum(-1)
                s0 = torch.where(ar[:, t], n0, s0); s1 = torch.where(ar[:, t], n1, s1)
            return tot
        crop = mark_nll(cb, torch.zeros_like(s), W)
        full = mark_nll(fb, s, W)
    assert torch.allclose(crop, full, rtol=1e-4, atol=1e-4), (crop, full)
    assert (s > 0).sum() >= 8


def test_rb_frozen_runs():
    from delphi_aou.baselines import classical as s3
    cohort, fo, visits = make_world(n=600, seed=3)
    vis, _ = s5v.build(cohort, fo, [visits])
    data = s5.VisitData(cohort, fo, vis)
    an = s3.build_anchors(cohort).head(30).reset_index(drop=True)
    names = [t for t in ["D=V01", "D=Z99"] if t in data.cidx] + ["DEATH"]
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, dropout=0.0)).eval()
    r = s5.sample_risk(m, data, an, names, (1, 5), 1, n_samples=4, estimator="rb_frozen")
    assert np.isfinite(r[5]).all() and (r[5][:, :-1] >= r[1][:, :-1] - 1e-6).all()
