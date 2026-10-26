"""Does simulated-trajectory risk fix the calibration of the instantaneous-rate shortcut when
diagnoses arrive in bursts (shared visits)? Run: python tests/test_sampling_risk.py"""
import sys, time
from pathlib import Path
import numpy as np
import pytest
from delphi_aou.sim import observation as so
from delphi_aou.baselines import classical as s3
from delphi_aou.train import delphi as s4
from delphi_aou.eval.calibration import log_oe  # noqa: E402


@pytest.mark.slow
def test_main():
    cohort, fo, truth = so.simulate(n=60000, seed=3, p_burst=0.5, shared_visits=True)
    names = sorted(t for t in fo["token"].unique() if t != "DEATH") + ["DEATH"]
    model, res, sc, tab = s4.run(cohort, fo, mode="observed", censor=False, horizon_years=5, n_diseases=len(names) - 1,
                                 iters=1500, batch=128, n_layer=2, n_embd=48, n_head=2, lr=2e-3, log=lambda *a: None)
    data = s4.Data(cohort, fo)
    an = s3.build_anchors(cohort)
    te = an[an["split"] == "test"].reset_index(drop=True).sample(3000, random_state=0).reset_index(drop=True)
    te["anchor_id"] = np.arange(len(te))
    Y, P, E, _ = s3.attach(te, fo, names[:-1], names[:-1], 5, 1)
    inst = s4.score(model, data, te, names, "observed", 5, 1)
    t = time.time()
    samp = s4.sample_risk(model, data, te, names, "observed", 5, 1, n_samples=30)
    print("sampling s", round(time.time() - t, 1))
    for nm, p in [("instantaneous", inst), ("sampled", samp)]:
        v = log_oe(p[:, :-1], Y[:, :-1], E[:, :-1], min_cases=10)
        print(nm, "mean log O/E", round(float(np.nanmean(v)), 3), [round(x, 2) for x in v])
    vi = np.nanmean(log_oe(inst[:, :-1], Y[:, :-1], E[:, :-1], min_cases=10))
    vs = np.nanmean(log_oe(samp[:, :-1], Y[:, :-1], E[:, :-1], min_cases=10))
    assert abs(vs) < abs(vi), (vs, vi)
    assert abs(vs) < 0.2, vs
    print("PASS sampling calibration")


if __name__ == "__main__":
    test_main()
