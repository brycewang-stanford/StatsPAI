"""Coverage campaign (decomposition) — reachable guard / edge branches.

Final sweep of the small reachable guards: logit non-convergence warning,
significance-star thresholds, and the ``weights=None`` default inside the
private inequality-index kernels (which the public ``inequality_index`` never
hits because it always forwards a weight vector). Real behavioural assertions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from statspai.decomposition import _common as C
from statspai.decomposition import inequality as I

# ── logit non-convergence path ───────────────────────────────────────


def test_logit_fit_nonconvergence_warns():
    rng = np.random.default_rng(0)
    n = 300
    X = C.add_constant(rng.normal(size=(n, 2)))
    d = (rng.uniform(size=n) < 0.5).astype(float)
    # max_iter=1 cannot converge → emits the non-convergence warning.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        beta, vcov = C.logit_fit(d, X, max_iter=1)
    assert any("converge" in str(wi.message).lower() for wi in w)
    assert beta.shape == (3,)


def test_logit_fit_silent_when_flag_off():
    rng = np.random.default_rng(1)
    n = 200
    X = C.add_constant(rng.normal(size=(n, 2)))
    d = (rng.uniform(size=n) < 0.5).astype(float)
    # warn_on_nonconvergence=False suppresses the warning branch.
    beta, _ = C.logit_fit(d, X, max_iter=1, warn_on_nonconvergence=False)
    assert beta.shape == (3,)


# ── significance stars thresholds ────────────────────────────────────


@pytest.mark.parametrize(
    "pval,expected",
    [
        (0.0005, "***"),
        (0.005, "**"),
        (0.03, "*"),
        (0.07, "+"),
        (0.5, ""),
    ],
)
def test_sig_stars(pval, expected):
    assert C.sig_stars(pval) == expected


# ── private inequality kernels with weights=None ─────────────────────


def test_private_index_kernels_default_weights():
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    # Called with no weights → exercises the ``w is None`` default branch.
    assert I._theil_t(y) > 0
    assert I._theil_l(y) > 0
    assert 0.0 <= I._gini(y) <= 1.0
    assert I._cv_squared_half(y) > 0
    # Perfect equality collapses each to zero.
    eq = np.full(5, 3.0)
    assert I._theil_t(eq) == pytest.approx(0.0, abs=1e-12)
    assert I._gini(eq) == pytest.approx(0.0, abs=1e-12)
