"""Option scan (Oct 2026) of sp.twoway_cluster.

The returned result used to copy the input fit's data_info, add ``vcov``
and overwrite ``df_resid``. The stale ``var_cov`` (read by
predict(what='confidence')), the stale ``df_inference`` (read by the
p-values) and ``df_resid = min(G) - 1`` (the divisor of the error variance
in a prediction interval) described a different fit, and a logit fit got the
least-squares sandwich.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _panel(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "firm": rng.integers(0, 15, size=n),
            "year": rng.integers(2000, 2010, size=n),
        }
    )
    d["y"] = 1.0 + 0.5 * d["x1"] + rng.normal(size=n)
    return d


def test_confidence_se_uses_twoway_covariance():
    d = _panel()
    tw = sp.twoway_cluster(sp.regress("y ~ x1", data=d), d, "firm", "year")
    # at x1 = 0 the prediction is the intercept: same standard error
    pc = tw.predict(pd.DataFrame({"x1": [0.0]}), what="confidence")
    assert pc["se"].iloc[0] == pytest.approx(tw.std_errors.iloc[0], rel=1e-10)


def test_prediction_interval_error_variance_is_rss_over_n_minus_k():
    d = _panel()
    fit = sp.regress("y ~ x1", data=d)
    tw = sp.twoway_cluster(fit, d, "firm", "year")
    new = pd.DataFrame({"x1": [0.3]})
    pc = tw.predict(new, what="confidence")
    pp = tw.predict(new, what="prediction")
    resid = np.asarray(fit.data_info["residuals"], dtype=float)
    s2 = float(resid @ resid) / (len(d) - 2)
    assert pp["se"].iloc[0] ** 2 - pc["se"].iloc[0] ** 2 == pytest.approx(s2, rel=1e-10)


def test_pvalues_use_the_same_df_as_the_intervals():
    d = _panel()
    fit = sp.regress("y ~ x1", data=d, cluster="firm")
    tw = sp.twoway_cluster(fit, d, "firm", "year")
    df = min(d["firm"].nunique(), d["year"].nunique()) - 1
    t = tw.params / tw.std_errors
    np.testing.assert_allclose(
        tw.pvalues.to_numpy(), 2 * stats.t.sf(np.abs(t.to_numpy()), df), rtol=1e-10
    )
    crit = stats.t.ppf(0.975, df)
    np.testing.assert_allclose(
        tw.conf_int_upper.to_numpy(),
        (tw.params + crit * tw.std_errors).to_numpy(),
        rtol=1e-12,
    )


def test_twoway_cluster_refuses_a_likelihood_fit():
    d = _panel()
    d["yb"] = (d["y"] > d["y"].median()).astype(int)
    fit = sp.logit("yb ~ x1", data=d)
    with pytest.raises(MethodIncompatibility):
        sp.twoway_cluster(fit, d, "firm", "year")


def test_margins_after_twoway_keeps_formula_and_weights():
    # Found by the before/after snapshot: the new result lost the formula
    # (recorded on sp.regress's provenance), so margins of I(x**2) refused,
    # and the weights, so averages were unweighted.
    d = _panel().assign(w=lambda f: 1.0 + (f["x1"] > 0))
    fit = sp.regress("y ~ x1 + I(x1 ** 2)", data=d, weights="w")
    tw = sp.twoway_cluster(fit, d, "firm", "year")
    a = sp.margins(fit, data=d)
    b = sp.margins(tw, data=d)
    np.testing.assert_allclose(b["dy/dx"], a["dy/dx"], rtol=1e-12)
    assert b.attrs["weights"] == "w"
