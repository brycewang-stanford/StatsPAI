"""``weights_as=``: saying what the weights are.

Thirteen regression entry points report a model-based variance when
``weights=`` is passed alone. ``weights_as`` lets the caller state the
reading. The contract checked here, for every entry point:

* ``'sampling'`` is the same fit as asking for the robust variance by
  hand, bit for bit (and so inherits that option's Stata ``[pw=]``
  evidence), and it never overrides a variance the caller did ask for;
* the entry point's own model-based reading (``'analytic'`` for linear
  estimators, ``'frequency'`` for likelihood ones) changes no number and
  silences the warning about dispersed weights, because the caller has
  now said the classical variance is the one they mean;
* the other model-based reading is refused, with the reason;
* the declaration is recorded on the result.

Which reading is native is itself checked: an analytic variance does not
move when the weights are multiplied by a constant, a frequency variance
shrinks by the square root of it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning, MethodIncompatibility


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(20261005)
    G, T = 60, 10
    n = G * T
    unit = np.repeat(np.arange(G), T)
    x = rng.normal(size=n)
    q = rng.normal(size=n)
    u = rng.normal(size=n)
    d = q + 0.5 * u + rng.normal(size=n)
    w = np.repeat(np.exp(rng.normal(scale=1.5, size=G)), T)
    y = 0.5 * x + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "yiv": 0.5 * d + u,
            "yb": (y > 0).astype(int),
            "c": rng.poisson(np.exp(0.2 + 0.3 * x)),
            "x": x,
            "d": d,
            "q": q,
            "w": w,
            "w4": 4.0 * w,
            "id": unit,
            "t": np.tile(np.arange(T), G),
        }
    )


def _panel(d, **kw):
    return sp.panel(d, "y ~ x", entity="id", time="t", method="fe", **kw)


#: name -> (call, native reading, keyword that asks for the robust
#: variance by hand, keyword that clusters, coefficient to read)
ENTRIES = {
    "regress": (
        lambda d, **k: sp.regress("y ~ x", d, **k),
        "analytic",
        {"robust": "hc1"},
        {"cluster": "id"},
        "x",
    ),
    "panel": (_panel, "analytic", {"robust": "robust"}, {"cluster": "id"}, "x"),
    "feols": (
        lambda d, **k: sp.feols("y ~ x | id", d, **k),
        "analytic",
        {"vcov": "hetero"},
        {"cluster": "id"},
        "x",
    ),
    "fast.feols": (
        lambda d, **k: sp.fast.feols("y ~ x | id", d, **k),
        "analytic",
        {"vcov": "hc1"},
        {"vcov": "cr1", "cluster": "id"},
        "x",
    ),
    "iv": (
        lambda d, **k: sp.iv("yiv ~ (d ~ q)", data=d, **k),
        "analytic",
        {"robust": "hc1"},
        {"cluster": "id"},
        "d",
    ),
    "ivreg": (
        lambda d, **k: sp.ivreg("yiv ~ (d ~ q)", data=d, **k),
        "analytic",
        {"robust": "hc1"},
        {"cluster": "id"},
        "d",
    ),
    "glm_gaussian": (
        lambda d, **k: sp.glm("y ~ x", d, family="gaussian", **k),
        "analytic",
        {"robust": "robust"},
        {"cluster": "id"},
        "x",
    ),
    "logit": (
        lambda d, **k: sp.logit("yb ~ x", d, **k),
        "frequency",
        {"robust": "robust"},
        {"cluster": "id"},
        "x",
    ),
    "probit": (
        lambda d, **k: sp.probit("yb ~ x", d, **k),
        "frequency",
        {"robust": "robust"},
        {"cluster": "id"},
        "x",
    ),
    "cloglog": (
        lambda d, **k: sp.cloglog("yb ~ x", d, **k),
        "frequency",
        {"robust": "robust"},
        {"cluster": "id"},
        "x",
    ),
    "glm_binomial": (
        lambda d, **k: sp.glm("yb ~ x", d, family="binomial", **k),
        "frequency",
        {"robust": "robust"},
        {"cluster": "id"},
        "x",
    ),
    "poisson": (
        lambda d, **k: sp.poisson("c ~ x", d, **k),
        "frequency",
        {"robust": "robust"},
        {"cluster": "id"},
        "x",
    ),
    "nbreg": (
        lambda d, **k: sp.nbreg("c ~ x", d, **k),
        "frequency",
        {"robust": "robust"},
        {"cluster": "id"},
        "x",
    ),
    "fast.fepois": (
        lambda d, **k: sp.fast.fepois("c ~ x | id", d, **k),
        "frequency",
        {"vcov": "hc1"},
        {"vcov": "cr1", "cluster": "id"},
        "x",
    ),
}
NAMES = sorted(ENTRIES)


def _numbers(res, name):
    if hasattr(res, "std_errors"):
        return float(res.params[name]), float(res.std_errors[name])
    row = res.tidy().loc[name]
    return float(row["Estimate"]), float(row["Std. Error"])


def _info(res):
    if hasattr(res, "weight_info"):
        return res.weight_info or {}
    return res.model_info


def _fit(call, d, **kw):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = call(d, **kw)
    msgs = [str(m.message) for m in rec if issubclass(m.category, AssumptionWarning)]
    return res, msgs


@pytest.mark.parametrize("name", NAMES)
def test_sampling_is_the_robust_fit_bit_for_bit(df, name):
    if name == "feols":
        pytest.importorskip("pyfixest")
    call, _, robust_kw, _, coef = ENTRIES[name]
    declared, _ = _fit(call, df, weights="w", weights_as="sampling")
    by_hand, _ = _fit(call, df, weights="w", **robust_kw)
    default, _ = _fit(call, df, weights="w")
    assert _numbers(declared, coef) == _numbers(by_hand, coef)
    # same coefficient as the default fit, a different standard error
    assert _numbers(declared, coef)[0] == _numbers(default, coef)[0]
    assert _numbers(declared, coef)[1] > 1.3 * _numbers(default, coef)[1]
    assert _info(declared)["weights_as"] == "sampling"


@pytest.mark.parametrize("name", NAMES)
def test_sampling_does_not_override_a_variance_the_caller_chose(df, name):
    if name == "feols":
        pytest.importorskip("pyfixest")
    call, _, _, cluster_kw, coef = ENTRIES[name]
    declared, _ = _fit(call, df, weights="w", weights_as="sampling", **cluster_kw)
    by_hand, _ = _fit(call, df, weights="w", **cluster_kw)
    assert _numbers(declared, coef) == _numbers(by_hand, coef)


@pytest.mark.parametrize("name", NAMES)
def test_native_reading_changes_no_number_and_silences_the_warning(df, name):
    if name == "feols":
        pytest.importorskip("pyfixest")
    call, native, _, _, coef = ENTRIES[name]
    default, msgs = _fit(call, df, weights="w")
    assert any("weights are dispersed" in m for m in msgs), msgs
    declared, msgs = _fit(call, df, weights="w", weights_as=native)
    assert not [m for m in msgs if "weights are dispersed" in m], msgs
    assert _numbers(declared, coef) == _numbers(default, coef)
    assert _info(declared)["weights_as"] == native
    assert "weights_as" not in _info(default)


@pytest.mark.parametrize("name", NAMES)
def test_the_native_reading_is_the_one_the_variance_implements(df, name):
    """Analytic: invariant to the scale of the weights. Frequency: not."""
    if name == "feols":
        pytest.importorskip("pyfixest")
    call, native, _, _, coef = ENTRIES[name]
    one, _ = _fit(call, df, weights="w", weights_as=native)
    four, _ = _fit(call, df, weights="w4", weights_as=native)
    ratio = _numbers(one, coef)[1] / _numbers(four, coef)[1]
    assert ratio == pytest.approx(1.0 if native == "analytic" else 2.0, rel=5e-3)


@pytest.mark.parametrize("name", NAMES)
def test_the_other_model_based_reading_is_refused(df, name):
    if name == "feols":
        pytest.importorskip("pyfixest")
    call, native, _, _, _ = ENTRIES[name]
    other = "frequency" if native == "analytic" else "analytic"
    with pytest.raises(MethodIncompatibility, match="not implemented") as err:
        call(df, weights="w", weights_as=other)
    assert native in str(err.value) and "sampling" in err.value.recovery_hint


@pytest.mark.parametrize("name", NAMES)
def test_weights_as_needs_weights_and_a_known_value(df, name):
    if name == "feols":
        pytest.importorskip("pyfixest")
    call = ENTRIES[name][0]
    with pytest.raises(MethodIncompatibility, match="no weights were passed"):
        call(df, weights_as="sampling")
    with pytest.raises(MethodIncompatibility, match="must be one of"):
        call(df, weights="w", weights_as="pweight")


def test_hdfe_ols_needs_a_cluster_for_sampling_weights(df):
    """Its weighted fit has no heteroskedasticity-robust variance."""
    with pytest.raises(MethodIncompatibility, match="needs cluster=") as err:
        sp.hdfe_ols("y ~ x | id", df, weights="w", weights_as="sampling")
    assert "sp.feols" in err.value.recovery_hint
    declared = sp.hdfe_ols(
        "y ~ x | id", df, weights="w", weights_as="sampling", cluster="id"
    )
    by_hand = sp.hdfe_ols("y ~ x | id", df, weights="w", cluster="id")
    assert float(declared.std_errors["x"]) == float(by_hand.std_errors["x"])
    assert declared.weight_info["weights_as"] == "sampling"
    native, msgs = _fit(
        lambda d, **k: sp.hdfe_ols("y ~ x | id", d, **k),
        df,
        weights="w",
        weights_as="analytic",
    )
    assert not [m for m in msgs if "weights are dispersed" in m]
    with pytest.raises(MethodIncompatibility, match="not implemented"):
        sp.hdfe_ols("y ~ x | id", df, weights="w", weights_as="frequency")


def test_small_sample_warning_survives_the_declaration(df):
    """Declaring 'sampling' fixes the reading, not a small effective sample."""
    few = df.groupby("id").head(1).reset_index(drop=True)  # 60 rows, Kish ~ 15
    _, msgs = _fit(
        lambda d, **k: sp.regress("y ~ x", d, **k),
        few,
        weights="w",
        weights_as="sampling",
    )
    assert any("carry most of the weight" in m for m in msgs), msgs


@pytest.mark.parametrize(
    "name,expected",
    [
        ("regress", ["analytic", "sampling"]),
        ("panel", ["analytic", "sampling"]),
        ("iv", ["analytic", "sampling"]),
        ("ivreg", ["analytic", "sampling"]),
        ("hdfe_ols", ["analytic", "sampling"]),
        ("feols", ["analytic", "sampling"]),
        ("logit", ["frequency", "sampling"]),
        ("probit", ["frequency", "sampling"]),
        ("poisson", ["frequency", "sampling"]),
        ("nbreg", ["frequency", "sampling"]),
        ("glm", ["analytic", "frequency", "sampling"]),
    ],
)
def test_schema_offers_the_readings_the_function_accepts(name, expected):
    prop = sp.function_schema(name)["parameters"]["properties"]["weights_as"]
    assert prop["enum"] == expected
    assert "sampling" in prop["description"]
