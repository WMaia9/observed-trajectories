"""Tier B onset posterior at the anchor (estimator="posterior"): P(onset of k happened, k not yet recorded)."""
import numpy as np
import torch
from delphi_aou.data import visits as s5v
from delphi_aou.train import visit as s5
from delphi_aou.models.visit_pp import VisitPP, VisitConfig
from test_stage3b import make_world


def _setup(n=600):
    from delphi_aou.baselines import classical as s3
    cohort, fo, visits = make_world(n=n, seed=3)
    vis, _ = s5v.build(cohort, fo, [visits])
    data = s5.VisitData(cohort, fo, vis)
    an = s3.build_anchors(cohort).head(40).reset_index(drop=True)
    names = [t for t in data.codes[:5]]
    return data, an, names


def test_posterior_is_probability_and_nan_where_recorded():
    data, an, names = _setup()
    torch.manual_seed(0)
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, detection=True)).eval()
    r = s5.sample_risk(m, data, an, names, (1,), 1, n_samples=1, estimator="posterior")[1]
    ok = ~np.isnan(r)
    assert ok.any() and (r[ok] >= 0).all() and (r[ok] <= 1).all()
    rows = an["person_id"].map(data.row).to_numpy()
    for q, i in enumerate(rows):                      # NaN exactly for diseases recorded before the anchor
        a, _ = data.tokens(i, cutoff=an["anchor_age_days"].iloc[q] / 365.25 - 1 / 365.25)
        rec = {data.codes[k] for k, v in zip(data.rec_k[i], data.rec_v[i]) if v <= len(a) - 1}
        for c, nm in enumerate(names):
            assert np.isnan(r[q, c]) == (nm in rec)


def test_no_detection_gives_onset_cdf_since_entry():
    """With detection close to 0 nothing is ever found, so the posterior is the onset CDF since EHR entry."""
    data, an, names = _setup()
    m = VisitPP(VisitConfig(data.K, n_layer=1, n_head=2, n_embd=32, detection=True)).eval()
    with torch.no_grad():
        m.mark.weight.zero_(); m.mark.bias.fill_(np.log(0.02))
        m.det_h.weight.zero_(); m.det_h.bias.fill_(-30.0)
    r = s5.sample_risk(m, data, an, names, (1,), 1, n_samples=1, estimator="posterior")[1]
    rows = an["person_id"].map(data.row).to_numpy()
    t = an["anchor_age_days"].to_numpy() / 365.25 - data.entry[rows]
    want = 1 - np.exp(-0.02 * t)
    ok = ~np.isnan(r)
    assert np.allclose(r[ok], np.broadcast_to(want[:, None], r.shape)[ok], atol=2e-3)
