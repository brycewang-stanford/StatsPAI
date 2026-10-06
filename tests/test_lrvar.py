"""``sp.lrvar``: definition, invariances and argument checks.

Agreement with R ``sandwich`` is in
``tests/reference_parity/test_lrvar_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.timeseries.lrvar import LongRunVariance, lrvar


def _ar1(n: int, phi: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 200)
    x = np.zeros(n + 200)
    for t in range(1, n + 200):
        x[t] = phi * x[t - 1] + e[t]
    return x[200:]


def _gamma(x: np.ndarray, j: int) -> float:
    d = x - x.mean()
    return float(d[j:] @ d[: d.size - j]) / d.size


def test_bartlett_is_the_weighted_sum_of_autocovariances():
    x = _ar1(120, 0.5, 0)
    fit = lrvar(x, kernel="bartlett", bandwidth=4)
    by_hand = _gamma(x, 0) + 2 * sum((1 - j / 4) * _gamma(x, j) for j in (1, 2, 3))
    assert float(fit) == pytest.approx(by_hand, rel=1e-12)
    assert fit.var_mean == pytest.approx(by_hand / 120, rel=1e-12)
    assert fit.se_mean == pytest.approx(np.sqrt(by_hand / 120), rel=1e-12)
    assert fit.bandwidth == 4.0 and fit.bandwidth_rule == "fixed"


def test_truncated_and_other_kernels_by_hand():
    x = _ar1(80, 0.3, 1)
    trunc = lrvar(x, kernel="truncated", bandwidth=2)
    by_hand = _gamma(x, 0) + 2 * (_gamma(x, 1) + _gamma(x, 2))
    assert float(trunc) == pytest.approx(by_hand, rel=1e-12)
    th = lrvar(x, kernel="tukey-hanning", bandwidth=3)
    w = [(1 + np.cos(np.pi * j / 3)) / 2 for j in (1, 2)]
    by_hand = _gamma(x, 0) + 2 * (w[0] * _gamma(x, 1) + w[1] * _gamma(x, 2))
    assert float(th) == pytest.approx(by_hand, rel=1e-12)
    pz = lrvar(x, kernel="parzen", bandwidth=4)
    w = [1 - 6 * 0.25**2 + 6 * 0.25**3, 1 - 6 * 0.5**2 + 6 * 0.5**3, 2 * 0.25**3]
    by_hand = _gamma(x, 0) + 2 * sum(w[j - 1] * _gamma(x, j) for j in (1, 2, 3))
    assert float(pz) == pytest.approx(by_hand, rel=1e-12)


def test_bandwidth_one_is_the_variance():
    x = _ar1(100, 0.6, 2)
    fit = lrvar(x, bandwidth=1)
    assert float(fit) == pytest.approx(fit.variance, rel=1e-12)
    assert fit.variance == pytest.approx(np.var(x), rel=1e-12)


def test_recovers_the_long_run_variance_of_an_ar1():
    # truth 1 / (1 - 0.5)^2 = 4; a long sample, loose statistical tolerance
    x = _ar1(20000, 0.5, 3)
    for kwargs in (
        dict(kernel="bartlett", bandwidth="andrews"),
        dict(kernel="qs", bandwidth="andrews", prewhite=1),
        dict(kernel="parzen", bandwidth="newey-west"),
    ):
        assert float(lrvar(x, **kwargs)) == pytest.approx(4.0, rel=0.12)


def test_prewhitened_ar1_with_bandwidth_one_is_the_ar_formula():
    x = _ar1(300, 0.6, 4)
    d = x - x.mean()
    rho = float(d[1:] @ d[:-1]) / float(d[:-1] @ d[:-1])
    resid = d[1:] - rho * d[:-1]
    by_hand = float(resid @ resid) / 300 / (1 - rho) ** 2
    assert float(lrvar(x, bandwidth=1, prewhite=1)) == pytest.approx(by_hand)


def test_scale_and_shift():
    x = _ar1(200, 0.4, 5)
    base = lrvar(x, kernel="qs")
    moved = lrvar(3.0 * x + 7.0, kernel="qs")
    assert moved.bandwidth == pytest.approx(base.bandwidth, rel=1e-10)
    assert float(moved) == pytest.approx(9.0 * float(base), rel=1e-10)


def test_adjust_and_demean():
    x = _ar1(50, 0.2, 6)
    plain = lrvar(x, bandwidth=3)
    assert float(lrvar(x, bandwidth=3, adjust=True)) == pytest.approx(
        float(plain) * 50 / 49, rel=1e-12
    )
    centred = x - x.mean()
    assert float(lrvar(centred, bandwidth=3, demean=False)) == pytest.approx(
        float(plain), rel=1e-12
    )
    with pytest.raises(MethodIncompatibility, match="demean"):
        lrvar(x, adjust=True, demean=False)


def test_rules_of_thumb():
    x = _ar1(300, 0.4, 7)
    assert lrvar(x, bandwidth="rule").bandwidth == np.floor(4 * 3 ** (2 / 9)) + 1
    assert lrvar(x, bandwidth="sw").bandwidth == np.floor(0.75 * 300 ** (1 / 3)) + 1
    auto = lrvar(x, bandwidth="andrews")
    lagged = lrvar(x, bandwidth="andrews", integer_lag=True)
    assert lagged.bandwidth == np.floor(auto.bandwidth) + 1


def test_multivariate_matrix():
    rng = np.random.default_rng(8)
    z = np.column_stack([_ar1(250, 0.5, 9), _ar1(250, -0.2, 10)])
    z[:, 1] += 0.5 * z[:, 0] + rng.normal(size=250) * 0.1
    frame = pd.DataFrame(z, columns=["a", "b"])
    fit = lrvar(frame, kernel="parzen", bandwidth=6)
    assert isinstance(fit, LongRunVariance)
    assert fit.lrvar.shape == (2, 2) and fit.names == ["a", "b"]
    assert np.allclose(fit.lrvar, fit.lrvar.T)
    assert np.linalg.eigvalsh(fit.lrvar).min() > 0
    one = lrvar(frame, "a", kernel="parzen", bandwidth=6)
    assert float(one) == pytest.approx(fit.lrvar[0, 0], rel=1e-12)
    assert fit.to_frame().loc["a", "b"] == fit.lrvar[0, 1]
    assert "covariance matrix" in fit.summary()
    with pytest.raises(TypeError):
        float(fit)
    # recolouring with a VAR(1) keeps the matrix symmetric
    pw = lrvar(frame, kernel="qs", prewhite=1)
    assert np.allclose(pw.lrvar, pw.lrvar.T)


def test_summary_and_series_input():
    x = pd.Series(_ar1(150, 0.5, 11), name="growth")
    fit = lrvar(x, kernel="QS", prewhite=1)
    text = fit.summary()
    assert "qs" in text and "VAR(1)" in text and "se(mean)" in text
    assert fit.names == ["growth"] and fit.n_obs == 150
    assert fit.to_dict()["kernel"] == "qs"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(kernel="triangle"), "kernel"),
        (dict(bandwidth="silverman"), "not a known rule"),
        (dict(bandwidth=0), "positive"),
        (dict(bandwidth=-2.0), "positive"),
        (dict(prewhite=-1), "negative"),
        (dict(kernel="truncated", bandwidth="newey-west"), "not defined"),
    ],
)
def test_bad_arguments(kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match):
        lrvar(_ar1(100, 0.3, 12), **kwargs)


def test_bad_data():
    x = _ar1(100, 0.3, 13)
    gap = x.copy()
    gap[40] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing"):
        lrvar(gap)
    with pytest.raises(MethodIncompatibility, match="not in the columns"):
        lrvar(pd.DataFrame({"a": x}), "b")
    with pytest.raises(DataInsufficient):
        lrvar(x[:3])
    with pytest.raises(DataInsufficient, match="constant"):
        lrvar(np.ones(50))
    with pytest.raises(MethodIncompatibility, match="1-D or 2-D"):
        lrvar(np.zeros((4, 3, 2)))


def test_truncated_kernel_can_go_negative_and_says_so():
    # alternating series: gamma(1) is close to -gamma(0)
    x = np.tile([1.0, -1.0], 30) + np.random.default_rng(14).normal(size=60) * 0.01
    with pytest.raises(MethodIncompatibility, match="negative"):
        lrvar(x, kernel="truncated", bandwidth=1.5)
    assert float(lrvar(x, kernel="bartlett", bandwidth=1.5)) > 0
