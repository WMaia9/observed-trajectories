"""Validation of the calibration analysis on arrays with KNOWN miscalibration."""
import sys
from pathlib import Path
import numpy as np
from delphi_aou.eval import calibration as ac


def test_known_miscalibration():
    r = np.random.default_rng(0)
    n, D = 60000, 5
    p = r.uniform(0.02, 0.2, (n, D))
    Y = r.random((n, D)) < p
    E = np.ones((n, D), bool)
    person = np.arange(n)
    groups = r.integers(0, 5, n)
    race = np.where(r.random(n) < 0.5, "A", "B")
    over = np.where((groups == 0)[:, None], p * 2.0, p)     # 2x over prediction ONLY in group 0
    res = ac.analyse({"true": p, "over_g0": over}, Y, E, person, groups, race, n_boot=50)
    assert abs(res["true"]["overall"]["mean_log_oe"]) < 0.03
    g0 = res["over_g0"]["by_visit_quintile"]["0"]["mean_log_oe"]
    assert abs(g0 - np.log(0.5)) < 0.05, g0                   # log(O/E) = log(1/2)
    gap = res["over_g0"]["access_gap_top_minus_bottom"]
    assert gap["ci_lo"] > 0.5 and abs(gap["gap"] - np.log(2)) < 0.06, gap
    gt = res["true"]["access_gap_top_minus_bottom"]
    assert gt["ci_lo"] < 0 < gt["ci_hi"], gt                  # no gap when calibrated
    print("PASS calibration", round(g0, 3), gap)


if __name__ == "__main__":
    test_known_miscalibration()
