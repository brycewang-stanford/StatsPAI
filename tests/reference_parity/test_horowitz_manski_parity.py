"""Analytical parity: sp.horowitz_manski worst-case bounds bracket the truth.

Horowitz-Manski conditional worst-case bounds on the ATE. On a bounded outcome
in [0, 1] with a modest planted effect the identified interval must (a) contain
the true ATE and (b) have width no larger than the outcome range. Analytical
evidence tier (known-truth containment on a deterministic DGP).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

TRUE_ATE = 0.3


def _fit(seed, n=3000):
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    x = rng.normal(0, 1, n)
    y = (0.3 + TRUE_ATE * d + 0.2 * x + rng.normal(0, 0.1, n)).clip(0, 1)
    df = pd.DataFrame({"y": y, "d": d, "x": x})
    return sp.horowitz_manski(df, y="y", treatment="d", covariates=["x"])


def test_bounds_contain_true_effect():
    for seed in range(3):
        r = _fit(seed)
        assert float(r.lower) <= TRUE_ATE <= float(r.upper)


def test_bounds_ordered_and_within_outcome_range():
    r = _fit(0)
    assert float(r.lower) < float(r.upper)
    # bounded outcome in [0, 1] => worst-case width cannot exceed the range
    assert (float(r.upper) - float(r.lower)) <= 1.0 + 1e-9
