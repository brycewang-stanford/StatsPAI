"""Fail-loud input validation for the proximal / negative-control family.

These estimators previously sliced ``data[[cols]].dropna()`` with no checks,
so a typo'd column name surfaced as a cryptic ``KeyError: "['x'] not in
index"`` and an empty / all-NaN / single-row frame produced a *silent NaN*
ATE from the downstream linear algebra — exactly the "吞异常返回 NaN"
anti-pattern design principle 7 forbids.

This suite pins the hardened contract: every malformed input now raises
``sp.StatsPAIError`` (the agent-native umbrella; data-size failures use
``DataInsufficient``, also a ``ValueError``), while the happy path is
numerically untouched.  It also pins the ``proximal_regression``
graceful-degradation path, which previously raised ``NameError`` when the
treatment-bridge logistic fit failed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ConvergenceWarning, DataInsufficient, StatsPAIError


@pytest.fixture(scope="module")
def good():
    rng = np.random.default_rng(0)
    n = 300
    U = rng.normal(size=n)
    X = rng.normal(size=n)
    D = (0.5 * U + 0.3 * X + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(
        dict(
            Y=1.5 * D + 1.0 * U + 0.5 * X + rng.normal(size=n),
            D=D,
            X=X,
            NCE=0.8 * U + rng.normal(size=n) * 0.5,
            NCO=0.8 * U + rng.normal(size=n) * 0.5,
            Z=0.7 * U + rng.normal(size=n) * 0.5,
            W=0.7 * U + rng.normal(size=n) * 0.5,
        )
    )


# A callable per estimator that raises on the supplied frame, parametrised so
# each malformed-input case runs against all four entry points.
def _call(name, data):
    if name == "double_negative_control":
        return sp.double_negative_control(data, y="Y", treat="D", nce="NCE", nco="NCO")
    if name == "negative_control_outcome":
        return sp.negative_control_outcome(data, nco="NCO", treat="D")
    if name == "negative_control_exposure":
        return sp.negative_control_exposure(data, y="Y", nce="NCE")
    if name == "proximal_regression":
        return sp.proximal_regression(data, y="Y", treat="D", z_proxy="Z", w_proxy="W")
    raise AssertionError(name)


ALL = [
    "double_negative_control",
    "negative_control_outcome",
    "negative_control_exposure",
    "proximal_regression",
]


def _call_bad_column(name, data):
    """Call each estimator with one column argument pointing at a name that is
    guaranteed absent, so the missing-column guard must fire."""
    if name == "double_negative_control":
        return sp.double_negative_control(data, y="Y", treat="D", nce="NCE", nco="NOPE")
    if name == "negative_control_outcome":
        return sp.negative_control_outcome(data, nco="NOPE", treat="D")
    if name == "negative_control_exposure":
        return sp.negative_control_exposure(data, y="Y", nce="NOPE")
    if name == "proximal_regression":
        return sp.proximal_regression(
            data, y="Y", treat="D", z_proxy="Z", w_proxy="NOPE"
        )
    raise AssertionError(name)


class TestMissingColumn:
    @pytest.mark.parametrize("name", ALL)
    def test_typo_raises_statspai_error_not_keyerror(self, name, good):
        with pytest.raises(StatsPAIError) as exc:
            _call_bad_column(name, good)
        # The message must name the offending column (actionable for an agent).
        assert "not found" in str(exc.value)
        assert "NOPE" in str(exc.value)

    @pytest.mark.parametrize("name", ALL)
    def test_non_dataframe_raises(self, name):
        with pytest.raises(StatsPAIError):
            _call(name, [1, 2, 3])


class TestInsufficientData:
    @pytest.mark.parametrize("name", ALL)
    def test_empty_frame_raises(self, name, good):
        with pytest.raises(DataInsufficient):
            _call(name, good.iloc[:0])

    @pytest.mark.parametrize("name", ALL)
    def test_single_row_raises(self, name, good):
        with pytest.raises(DataInsufficient):
            _call(name, good.iloc[:1])

    @pytest.mark.parametrize("name", ALL)
    def test_all_nan_outcome_raises(self, name, good):
        # Blank out a column each estimator depends on so dropna empties it.
        col = "NCO" if "outcome" in name or name.startswith("double") else "Y"
        with pytest.raises(DataInsufficient):
            _call(name, good.assign(**{col: np.nan}))

    def test_datainsufficient_is_valueerror(self, good):
        # Back-compat: callers doing ``except ValueError`` still catch it.
        with pytest.raises(ValueError):
            sp.proximal_regression(
                good.iloc[:1], y="Y", treat="D", z_proxy="Z", w_proxy="W"
            )


class TestHappyPathUnchanged:
    """Validation must not perturb a valid fit."""

    def test_estimators_still_run(self, good):
        dnc = sp.double_negative_control(
            good, y="Y", treat="D", nce="NCE", nco="NCO", covariates=["X"]
        )
        pci = sp.proximal_regression(
            good, y="Y", treat="D", z_proxy="Z", w_proxy="W", covariates=["X"]
        )
        assert np.isfinite(dnc.estimate)
        assert np.isfinite(pci.ate)


class TestProximalGracefulFallback:
    """When the treatment-bridge logistic fit fails, the estimator must
    degrade gracefully (constant-propensity fallback) — not raise NameError,
    the latent bug fixed alongside the validation hardening."""

    def test_logistic_failure_falls_back(self, good, monkeypatch):
        import sklearn.linear_model as lm

        class _BoomLR:
            def __init__(self, *a, **k):
                pass

            def fit(self, *a, **k):
                raise RuntimeError("forced logistic failure")

        monkeypatch.setattr(lm, "LogisticRegression", _BoomLR)
        with pytest.warns(ConvergenceWarning):
            r = sp.proximal_regression(good, y="Y", treat="D", z_proxy="Z", w_proxy="W")
        # Fallback path reached, result still finite, no NameError, and the
        # propensity-coefficient dict is empty because ``lr`` stayed None.
        assert r.detail["propensity_fallback"] is True
        assert r.propensity_coefs == {}
        assert np.isfinite(r.ate)


def test_proximal_regression_refuses_a_continuous_treatment():
    """A non-0/1 treatment used to be truncated to integers and fitted.

    The estimator contrasts D=1 with D=0, so there is nothing to report;
    sp.proximal handles the continuous case (and recovers the effect).
    """
    import numpy as np
    import pandas as pd
    import pytest

    import statspai as sp

    rng = np.random.default_rng(0)
    n = 2000
    a = rng.normal(size=n)
    q = 2 * a + rng.normal(size=n)
    s = 2 * a + rng.normal(size=n)
    d = q - a + rng.normal(size=n)
    y = 2.0 * d + 3 * a + s + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "q": q, "s": s})
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="0/1"):
        sp.proximal_regression(df, y="y", treat="d", z_proxy="q", w_proxy="s")
    fit = sp.proximal(df, y="y", treat="d", proxy_z=["q"], proxy_w=["s"])
    assert abs(fit.estimate - 2.0) < 4 * fit.se
