"""Coverage tests for ``sp.dml_sensitivity`` on fits without score elements.

A sample-weighted PLR fit stores the residuals but not the per-repetition
score elements, so the omitted-variable bound is computed in closed form
and its sampling error is not available. These tests recompute that
closed form from the stored residuals. The IRM bound needs the Riesz
representer, which weighted fits do not store, and must refuse.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import MethodIncompatibility  # noqa: E402

XS = ["x0", "x1"]


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 2))
    d = 0.5 * X[:, 0] + rng.normal(size=n)
    df = pd.DataFrame({"x0": X[:, 0], "x1": X[:, 1], "d": d})
    df["y"] = 1.5 * d + X[:, 1] + rng.normal(size=n)
    df["y_null"] = X[:, 1] + rng.normal(size=n)
    df["db"] = (X[:, 0] + rng.normal(size=n) > 0).astype(int)
    df["yb"] = 1.0 * df["db"] + X[:, 1] + rng.normal(size=n)
    df["w"] = rng.uniform(0.5, 1.5, n)
    return df


def _plr(df, y="y", **kw):
    return sp.dml(
        df, y=y, treat="d", covariates=XS, ml_g="linear", ml_m="linear", n_folds=3, **kw
    )


def test_weighted_plr_bound_is_the_closed_form_of_the_stored_residuals(frame):
    fit = _plr(frame, sample_weight="w")
    assert not fit.model_info.get("_sens")  # no score elements stored
    sens = sp.dml_sensitivity(fit, cf_y=0.05, cf_d=0.04)

    theta, se = float(fit.estimate), float(fit.se)
    y_res = np.asarray(fit.model_info["_y_resid"], dtype=float).ravel()
    d_res = np.asarray(fit.model_info["_d_resid"], dtype=float).ravel()
    eps = y_res - theta * d_res
    s = np.sqrt(np.mean(eps**2)) / np.sqrt(np.mean(d_res**2))
    assert sens.s == pytest.approx(s, rel=1e-12)

    strength = np.sqrt(0.05 * 0.04 / (1 - 0.04))
    assert sens.bias_bound == pytest.approx(strength * s, rel=1e-12)
    assert sens.adjusted_estimate_low == pytest.approx(theta - strength * s, rel=1e-12)
    assert sens.adjusted_estimate_high == pytest.approx(theta + strength * s, rel=1e-12)
    # No score elements, so no sampling error for the bound.
    assert np.isnan(sens.se_low) and np.isnan(sens.se_high)
    assert np.isnan(sens.ci_low) and np.isnan(sens.ci_high)

    # The robustness value solves cf^2 / (1 - cf) = (target / s)^2.
    def residual(cf, target):
        return cf**2 / (1 - cf) - (target / s) ** 2

    crit = stats.norm.ppf(0.975)
    assert residual(sens.rv_q, abs(theta)) == pytest.approx(0.0, abs=1e-9)
    assert residual(sens.rv_qa, abs(theta) - crit * se) == pytest.approx(0.0, abs=1e-9)
    assert 0 < sens.rv_qa < sens.rv_q < 1


def test_weighted_irm_sensitivity_is_refused(frame):
    fit = sp.dml(
        frame,
        y="yb",
        treat="db",
        covariates=XS,
        model="irm",
        ml_g="linear",
        ml_m="logistic",
        n_folds=3,
        sample_weight="w",
    )
    with pytest.raises(MethodIncompatibility, match="needs the Riesz representer"):
        sp.dml_sensitivity(fit)


def test_interval_that_already_covers_the_null_has_zero_robustness(frame):
    fit = _plr(frame, y="y_null")
    assert abs(fit.estimate) < 1.96 * fit.se  # not significant by construction
    sens = sp.dml_sensitivity(fit)
    assert sens.rv_qa == 0.0
    assert sens.rv_q > 0.0


def test_plot_draws_on_a_supplied_axis(frame):
    sens = sp.dml_sensitivity(_plr(frame))
    fig, ax = plt.subplots()
    try:
        returned_fig, returned_ax = sens.plot(ax=ax)
        assert returned_fig is fig and returned_ax is ax
        assert ax.has_data()
        assert "DML-OVB sensitivity" in ax.get_title()
    finally:
        plt.close(fig)
