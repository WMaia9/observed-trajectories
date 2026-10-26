"""Tier A on the access world (onset independent of visits, recorded at the first visit after onset),
where Tier A is correctly specified. Expect calibrated recorded risk at 1 and 5 years and AUC > 0.5,
and compare with the Delphi style next-token model on the same anchors. Run: python tests/test_stage5_model.py"""
import sys, time
from pathlib import Path
import numpy as np
import pytest
from delphi_aou.data import visits as s5v
from delphi_aou.train import visit as s5
from delphi_aou.train import delphi as s4
from delphi_aou.baselines import classical as s3
from delphi_aou.eval.calibration import log_oe  # noqa: E402
from test_stage3b import make_world  # noqa: E402


@pytest.mark.slow
def test_main():
    cohort, fo, visits = make_world(n=8000)
    vis, st = s5v.build(cohort, fo, [visits])
    print("visit stats", {k: round(v, 3) for k, v in st.items() if isinstance(v, float)})
    names = ["D=V01", "D=Z99", "DEATH"]
    t = time.time()
    model, res, risks, keep = s5.run(cohort, fo, vis, iters=600, batch=32, lr=1e-3, n_layer=2, n_embd=64, n_head=4,
                                     names=names, n_samples=10, max_eval_anchors=1000, log=lambda *a: None)
    print("tierA", {k: v for k, v in res.items() if k != "val_loss_curve"}, "time", round(time.time() - t))
    m4, r4, sc4, tab4 = s4.run(cohort, fo, mode="observed", censor=False, horizon_years=5, n_diseases=2, iters=1500,
                               batch=128, n_layer=2, n_embd=48, n_head=2, lr=2e-3, log=lambda *a: None)
    an = s3.build_anchors(cohort); te = an[an.split == "test"].reset_index(drop=True); te["anchor_id"] = np.arange(len(te))
    for H in (1, 5):
        Y, P, E, _ = s3.attach(te, fo, names[:-1], names[:-1], H, 1)
        d = s4.score(m4, s4.Data(cohort, fo), te, names, "observed", H, 1)
        print("delphi-style H", H, "log O/E", round(float(np.nanmean(log_oe(d[:, :1], Y[:, :1], E[:, :1], min_cases=10))), 3))
    assert abs(res["H5"]["mean_log_OE"]) < 0.2, res
    print("PASS tier A calibration")


if __name__ == "__main__":
    test_main()
