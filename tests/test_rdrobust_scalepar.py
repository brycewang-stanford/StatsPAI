"""``sp.rdrobust(scalepar=)``: the rescaling of a regression kink estimate.

In a sharp kink design the policy variable is a known function of the
running variable whose slope changes at the cutoff. ``deriv=1`` estimates
the change in the slope of the outcome; dividing by the change in the slope
of the policy rule gives the effect of the policy variable. Stata and R
``rdrobust`` take that constant as ``scalepar``.

On data simulated in Stata (``d = min(0.5 x, 5)``, ``y = 2 d - 0.5 x + e``,
cutoff 10) ``rdrobust y x, c(10) kernel(uni) deriv(1) scalepar(-2)`` printed
1.8716841116 (conventional), 2.0257429310 (bias-corrected) and standard
errors 0.4750085794 / 0.6305038968; ``sp.rdrobust`` on the same file gave
those to 1e-11. That file is not shipped, so the tests here pin the property
that makes the numbers agree: the option multiplies the unscaled result.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def kink() -> pd.DataFrame:
    rng = np.random.default_rng(2345)
    n = 6000
    x = rng.normal(10, 2, n)
    d = np.minimum(0.5 * x, 5.0)
    y = 2 * d - 0.5 * x + rng.normal(size=n)
    return pd.DataFrame({"x": x, "d": d, "y": y})


def _fit(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdrobust(df, "y", "x", c=10, deriv=1, **kwargs)


@pytest.mark.parametrize("scale", [-2.0, 0.5, 3.0])
def test_scalepar_multiplies_the_unscaled_result(kink, scale):
    base = _fit(kink).model_info
    scaled = _fit(kink, scalepar=scale).model_info
    assert scaled["scalepar"] == scale
    for row in ("conventional", "robust"):
        assert scaled[row]["estimate"] == pytest.approx(
            scale * base[row]["estimate"], rel=1e-13
        )
        assert scaled[row]["se"] == pytest.approx(
            abs(scale) * base[row]["se"], rel=1e-13
        )
        assert scaled[row]["pvalue"] == pytest.approx(base[row]["pvalue"], rel=1e-12)
        lo, hi = sorted(scale * np.asarray(base[row]["ci"]))
        assert scaled[row]["ci"] == pytest.approx((lo, hi), rel=1e-12)
    # the bandwidth is chosen for the unscaled problem
    assert scaled["bandwidth_h"] == base["bandwidth_h"]


def test_scaled_kink_recovers_the_policy_effect(kink):
    # The rule's slope falls from 0.5 to 0 at the cutoff, a kink of -0.5,
    # and the effect of d on y is 2.
    res = _fit(kink, scalepar=1 / -0.5)
    assert res.ci[0] < 2.0 < res.ci[1]


def test_default_leaves_results_untouched(kink):
    a, b = _fit(kink), _fit(kink, scalepar=1.0)
    assert a.estimate == b.estimate and a.se == b.se


@pytest.mark.parametrize("bad", [0, float("nan"), float("inf"), "two"])
def test_invalid_scalepar_raises(kink, bad):
    with pytest.raises(MethodIncompatibility, match="scalepar"):
        _fit(kink, scalepar=bad)


@pytest.mark.parametrize("kwargs", [{"bwselect": "cct"}, {"engine": "bayes"}])
def test_scalepar_is_refused_on_paths_that_would_ignore_it(kink, kwargs):
    with pytest.raises(MethodIncompatibility, match="native estimator"):
        _fit(kink, scalepar=2.0, **kwargs)


def test_stata_option_is_translated(kink):
    out = sp.from_stata("rdrobust y x, c(10) deriv(1) scalepar(-2)")
    assert out["arguments"]["scalepar"] == -2.0
    assert out["untranslated_options"] == []
