"""Feature builder and cross-fitting of experiments/audit_reference_models.py."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from audit_reference_models import features, crossfit, CAL_NAMES


def test_features_use_only_history_up_to_the_point():
    x = np.array([[2, 4, 20, 30, 1, 40], [3, 20, 1, 1, 1, 1]])
    a = np.array([[0, 100, 1000, 1000, 1500, 3000], [0, 2000, 2100, 2200, 2300, 2400]], float)
    lut = np.full(1271, -1); lut[20] = 0; lut[30] = 1; lut[40] = 2
    r = {"person_row": np.array([0, 0, 1]), "z": np.array([1500.0, 3000.0, 1900.0]), "case": np.array([True, False, False])}
    bag, cal = features(r, x, a, lut)
    assert bag[0, :3].tolist() == [1, 1, 0] and bag[1, :3].tolist() == [1, 1, 1] and bag[2, :3].tolist() == [0, 0, 0]
    c = dict(zip(CAL_NAMES, cal[0]))
    assert c["count"] == 2 and c["record_days"] == 1 and c["codes_last_day"] == 2 and c["pre_record"] == 0
    assert dict(zip(CAL_NAMES, cal[1]))["record_days"] == 2 and dict(zip(CAL_NAMES, cal[2]))["pre_record"] == 1


def test_crossfit_scores_every_point_out_of_fold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3)); y = (X[:, 0] + rng.normal(size=400) > 0).astype(int)
    s = crossfit(X, y, np.arange(400))
    assert s.shape == (400,) and np.corrcoef(s, X[:, 0])[0, 1] > 0.8
