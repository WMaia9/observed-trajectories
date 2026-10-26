"""Helper of experiments/audit_returned_conditional.py (conditional analyses for the people the window cut drops)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_returned_conditional import evaluable
from test_corrected_conditional import _rows


def test_evaluable_lists_only_diseases_with_a_unit():
    rng = np.random.default_rng(3)
    r1 = _rows(rng)[0]
    r2 = dict(_rows(rng)[0]); r2["k"] = 60
    r2["case"] = np.zeros_like(r2["case"]); r2["case"][0] = True          # one case only: no evaluable unit
    assert evaluable([r1, r2], ["real", "erased"]) == {50}
