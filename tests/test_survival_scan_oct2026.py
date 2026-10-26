"""Option scan of ``survival/`` (2026-10-27).

Each test states the correct behaviour; the defects it found are fixed.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ConvergenceWarning, MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


def _weibull_data(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    g = rng.integers(0, 2, n).astype(float)
    t = rng.weibull(1.5, n) * np.exp(1.0 + 0.4 * x - 0.3 * g)
    c = rng.exponential(np.exp(1.6), n)
    return pd.DataFrame(
        {"time": np.minimum(t, c), "event": (t <= c).astype(int), "x": x, "g": g}
    )


# --------------------------------------------------------------------------- #
#  sp.aft is the same model as sp.survreg: same MLE, same SEs. Each reports
#  the log-likelihood of its reference: sp.aft Stata streg's (log time),
#  sp.survreg R survreg's (time).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "family", ["weibull", "exponential", "lognormal", "loglogistic"]
)
def test_aft_equals_survreg_estimates_and_se(family):
    # before the fix: coefficients 1e-5 and SEs 7e-6 (relative) off the MLE
    df = _weibull_data()
    a = sp.aft("time + event ~ x + g", df, family=family)
    s = sp.survreg(data=df, duration="time", event="event", x=["x", "g"], dist=family)
    np.testing.assert_allclose(a.params.to_numpy(), s.params.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(
        a.std_errors.to_numpy(), s.std_errors.to_numpy(), rtol=1e-12
    )
    assert a.log_likelihood == pytest.approx(s.model_info["ll_log_time"], rel=1e-12)


@pytest.mark.parametrize(
    "family", ["weibull", "exponential", "lognormal", "loglogistic"]
)
def test_aft_matches_stata_streg_including_loglik(family):
    data = pd.read_csv(_FIX / "streg_data.csv")
    ref = json.loads((_FIX / "streg_stata.json").read_text(encoding="utf-8"))[family]
    a = sp.aft("time + event ~ x1 + x2", data, family=family)
    p, se = a.params, a.std_errors
    names = ["x1", "x2", "Intercept"]
    est = [p[n] for n in names]
    err = [se[n] for n in names]
    if family != "exponential":
        sign = -1.0 if family == "weibull" else 1.0  # Stata's ln_p = -log(sigma)
        est.append(sign * p["log(sigma)"])
        err.append(se["log(sigma)"])
    np.testing.assert_allclose(est, ref["b"], rtol=1e-6)
    np.testing.assert_allclose(err, ref["se"], rtol=1e-6)
    assert a.log_likelihood == pytest.approx(ref["ll"], abs=1e-8)
    assert a.log_likelihood_scale.startswith("log time")


def test_survreg_loglik_is_r_survreg():
    data = pd.read_csv(_FIX / "aft_data.csv")
    ref = json.loads((_FIX / "aft_R.json").read_text(encoding="utf-8"))
    s = sp.survreg(data=data, duration="time", event="event", x=["x", "group"])
    assert s.diagnostics["Log-likelihood"] == pytest.approx(ref["loglik"], abs=1e-6)
    assert s.model_info["ll_scale"].startswith("duration t")


def test_aft_refuses_zero_duration():
    df = _weibull_data()
    df.loc[0, "time"] = 0.0
    with pytest.raises(MethodIncompatibility):
        sp.aft("time + event ~ x + g", df, family="weibull")


def test_aft_refuses_unknown_family():
    with pytest.raises(MethodIncompatibility):
        sp.aft("time + event ~ x + g", _weibull_data(), family="gompertz")


# --------------------------------------------------------------------------- #
#  Event coding: every right-censored estimator refuses codes other than 0/1
# --------------------------------------------------------------------------- #


def _event_012() -> pd.DataFrame:
    df = _weibull_data(200)
    df["event"] = df["event"] * np.where(np.arange(len(df)) % 3 == 0, 2, 1)
    df["cl"] = np.arange(len(df)) % 20
    return df


def test_aft_refuses_event_code_2():
    with pytest.raises(MethodIncompatibility):
        sp.aft("time + event ~ x + g", _event_012(), family="weibull")


def test_survreg_refuses_event_code_2():
    with pytest.raises(MethodIncompatibility):
        sp.survreg(data=_event_012(), duration="time", event="event", x=["x"])


def test_cox_frailty_refuses_event_code_2():
    with pytest.raises(MethodIncompatibility):
        sp.cox_frailty("time + event ~ x", _event_012(), cluster="cl", theta=0.5)


def test_cox_refuses_event_code_2_baseline():
    with pytest.raises(ValueError):
        sp.cox(data=_event_012(), duration="time", event="event", x=["x"])


# --------------------------------------------------------------------------- #
#  sp.cox robust=: no silent fallback to the model-based variance
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", ["hc1", "HC3", "cluster", "sandwich"])
def test_cox_unknown_robust_value_is_refused(value):
    df = _weibull_data()
    with pytest.raises(MethodIncompatibility):
        sp.cox(data=df, duration="time", event="event", x=["x", "g"], robust=value)


@pytest.mark.parametrize("value", [True, "Robust", "HC0", "robust"])
def test_cox_robust_spellings_give_the_sandwich(value):
    df = _weibull_data()
    ref = sp.cox(data=df, duration="time", event="event", x=["x", "g"], robust="hc0")
    r = sp.cox(data=df, duration="time", event="event", x=["x", "g"], robust=value)
    np.testing.assert_array_equal(r.std_errors.to_numpy(), ref.std_errors.to_numpy())


def test_cox_robust_false_is_model_variance():
    df = _weibull_data()
    m = sp.cox(data=df, duration="time", event="event", x=["x", "g"])
    r = sp.cox(data=df, duration="time", event="event", x=["x", "g"], robust=False)
    np.testing.assert_array_equal(r.std_errors.to_numpy(), m.std_errors.to_numpy())
    h = sp.cox(data=df, duration="time", event="event", x=["x", "g"], robust="hc0")
    assert abs(h.std_errors["x"] / m.std_errors["x"] - 1) > 1e-3


def test_cox_cluster_spelling_with_cluster_column():
    df = _weibull_data().assign(cl=lambda d: np.arange(len(d)) % 30)
    a = sp.cox(data=df, duration="time", event="event", x=["x"], cluster="cl")
    b = sp.cox(
        data=df, duration="time", event="event", x=["x"], cluster="cl", robust="cluster"
    )
    np.testing.assert_array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())


# --------------------------------------------------------------------------- #
#  sp.cox: monotone likelihood (infinite coefficient) is flagged
# --------------------------------------------------------------------------- #


def test_cox_monotone_likelihood_warns_and_records():
    # every failure while x = 1 subjects are at risk is an x = 1 subject:
    # the partial likelihood increases without bound in beta (R coxph:
    # "Loglik converged before variable 1; coefficient may be infinite")
    n = 20
    df = pd.DataFrame(
        {
            "time": np.arange(1.0, n + 1),
            "event": np.ones(n, dtype=int),
            "x": np.r_[np.ones(n // 2), np.zeros(n // 2)],
        }
    )
    with pytest.warns(ConvergenceWarning, match="coefficient may be infinite"):
        r = sp.cox(data=df, duration="time", event="event", x=["x"])
    assert r.diagnostics["Converged"] is False
    assert "['x']" in r.model_info["convergence_warning"]


def test_cox_regular_fit_does_not_warn():
    df = _weibull_data()
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        r = sp.cox(data=df, duration="time", event="event", x=["x", "g"])
    assert "Converged" not in r.diagnostics


def test_km_median_follows_stata_stci_at_exact_half():
    # Stata 18 `stci` on deaths at 1, 2, 3, 4 reports 2 (and 3 for 1..6);
    # R survfit reports the midpoint 2.5. Documented in median_survival.
    for n, stata in ((4, 2.0), (6, 3.0)):
        df = pd.DataFrame({"t": np.arange(1.0, n + 1), "d": np.ones(n, dtype=int)})
        km = sp.kaplan_meier(data=df, duration="t", event="d")
        assert km.median_survival == stata
