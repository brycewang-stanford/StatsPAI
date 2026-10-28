"""``sp.cox`` stops at the maximiser of the partial likelihood.

Until 2026-10-28 the Newton loop required each step to lower the negative
log partial likelihood by 1e-10. Near the optimum a Newton step of 1e-9 to
~1e-6 changes it by less than that, so the step-halving search ran out,
the 2^-20-damped step passed the convergence test and the fit stopped one
Newton step short. On the simulated designs below 436 of 800 fits stopped
short, by up to 5.8e-7 on a coefficient; on Linux / numpy 2.2 the same
mechanism put ``sp.cox(tvc=)`` 4e-6 away from Stata ``stcox`` and R
``coxph`` (``tests/reference_parity/test_cox_tvc_parity.py``).

The assertion is the first-order condition: at the reported coefficients
the Newton step ``H^-1 score`` is below the convergence threshold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.survival._cox_core import CoxKernel


def _design(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(80, 400))
    X = rng.normal(size=(n, 3))
    T = rng.exponential(np.exp(-X @ np.array([0.5, -0.3, 0.2])))
    E = (rng.uniform(size=n) < 0.8).astype(float)
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(t=T, e=E)
    return df, X, T, E


# seeds 8 and 9 stopped 5.8e-7 and 1.6e-7 short before the fix
@pytest.mark.parametrize("seed", [6, 8, 9, 10, 11])
@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_reported_coefficients_satisfy_the_score_equation(seed, ties):
    df, X, T, E = _design(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.cox(data=df, duration="t", event="e", x=["a", "b", "c"], ties=ties)
    kern = CoxKernel(
        X, T, E, None, breslow=(ties == "breslow"), weights=None, start=None
    )
    score, hessian = kern.score_hessian(r.params.to_numpy())
    newton_step = np.linalg.solve(-hessian, score)
    assert np.max(np.abs(newton_step)) < 1e-9


def test_a_further_newton_step_does_not_move_the_fit():
    # the fit is a fixed point of its own iteration: starting a second fit
    # from the data with rows reversed (same likelihood) lands on the same
    # coefficients to 1e-10
    df, *_ = _design(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.cox(data=df, duration="t", event="e", x=["a", "b", "c"])
        b = sp.cox(data=df.iloc[::-1], duration="t", event="e", x=["a", "b", "c"])
    np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=1e-10)
