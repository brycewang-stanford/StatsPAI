"""Weighted ``sp.poisson`` / ``sp.ppmlhdfe`` / ``sp.logit`` / ``sp.probit`` vs Stata 18.

Fixture: ``_fixtures/_generate_count_weights_stata.do`` on the translation
holdout cross-section (400 rows, a continuous weight ``w``, an integer
weight ``fw``, 40 clusters ``g``).

``weights=`` in these functions is one column, read three ways by Stata
depending on the variance asked for, and StatsPAI follows the same map:

==========================  ==============================================
StatsPAI call               Stata
==========================  ==============================================
``weights=w``               ``[iw=w]`` (``[fw=w]`` for integer weights):
                            model-based variance, scales with the weights
``weights=w, robust=``      ``[pw=w]``: sandwich, ``N/(N-1)``
``weights=w, cluster=g``    ``[pw=w], vce(cluster g)``
==========================  ==============================================

The defect this caught (2026-10-03): ``sp.poisson`` and ``sp.ppmlhdfe``
fitted the weighted coefficients and then computed the covariance without
the weights. Against Stata the Poisson model-based SE was 10% high (42%
high with the integer weights), the robust SE 7% low; ``ppmlhdfe``'s
robust and clustered SEs were 8% and 12% off. The reported log-likelihoods
of ``sp.poisson``, ``sp.ppmlhdfe`` and ``sp.nbreg`` were unweighted sums.
``sp.logit`` and ``sp.probit`` were correct and are pinned here so the
family stays consistent.

``sp.nbreg`` has no Stata row: on this outcome Stata's ``alpha`` goes to
the boundary. Its weights are checked by the frequency-expansion identity
(an integer weight equals that many copies of the row), which also holds
the other three to a second, reference-free standard.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_HERE = pathlib.Path(__file__).parent
_FIX = _HERE / "_fixtures" / "count_weights_stata.json"
_DATA = _HERE.parent / "stata_translation_holdout" / "holdout_cross.csv"
#: IRLS / Newton on both sides. Everything is within 6e-10 except probit
#: with the integer weights, where Stata's default stopping rule leaves
#: 1.3e-7 on the coefficient nearest zero and 1.8e-8 on a standard error.
RTOL = 1e-6

MODELS = {
    "poisson": lambda d, **kw: sp.poisson("cnt ~ x1 + x2", d, **kw),
    "logit": lambda d, **kw: sp.logit("yb ~ x1 + x2", d, **kw),
    "probit": lambda d, **kw: sp.probit("yb ~ x1 + x2", d, **kw),
}
KINDS = {
    "iw": dict(weights="w"),
    "fw": dict(weights="fw"),
    "pw": dict(weights="w", robust="robust"),
    "pw_cluster": dict(weights="w", cluster="g"),
}


@pytest.fixture(scope="module")
def ref():
    if not _FIX.exists():  # pragma: no cover
        pytest.skip("run _generate_count_weights_stata.do first")
    return json.loads(_FIX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_DATA)


def _fit(fn, d, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(d, **kw)


def _assert_matches(res, cell):
    for name in ("x1", "x2"):
        assert float(res.params[name]) == pytest.approx(cell[f"b_{name}"], rel=RTOL)
        assert float(res.std_errors[name]) == pytest.approx(
            cell[f"se_{name}"], rel=RTOL
        )


@pytest.mark.parametrize("model", sorted(MODELS))
@pytest.mark.parametrize("kind", sorted(KINDS))
def test_matches_stata(ref, data, model, kind):
    _assert_matches(_fit(MODELS[model], data, **KINDS[kind]), ref[f"{model}_{kind}"])


@pytest.mark.parametrize("kind", ["iw", "fw"])
def test_poisson_log_likelihood_is_the_weighted_sum(ref, data, kind):
    res = _fit(MODELS["poisson"], data, **KINDS[kind])
    cell = ref[f"poisson_{kind}"]
    assert res.model_info["ll"] == pytest.approx(cell["ll"], rel=1e-10)
    assert res.model_info["ll_null"] == pytest.approx(cell["ll_0"], rel=1e-10)


@pytest.mark.parametrize(
    "key,kw",
    [
        ("ppmlhdfe_pw", dict(weights="w")),
        ("ppmlhdfe_pw_cluster", dict(weights="w", cluster="g")),
        ("ppmlhdfe_cluster", dict(cluster="g")),
    ],
)
def test_ppmlhdfe_matches_stata(ref, data, key, kw):
    res = _fit(lambda d, **k: sp.ppmlhdfe("cnt ~ x1 + x2 | k", d, **k), data, **kw)
    _assert_matches(res, ref[key])
    assert res.model_info["ll"] == pytest.approx(ref[key]["ll"], rel=1e-10)


# ── reference-free: an integer weight is that many copies of the row ───── #


@pytest.fixture(scope="module")
def overdispersed():
    rng = np.random.default_rng(20261006)
    n = 500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    mu = np.exp(0.4 + 0.3 * x1 - 0.2 * x2)
    df = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "cnt": rng.negative_binomial(n=2, p=2 / (2 + mu)).astype(float),
            "yb": (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x1))).astype(float),
            "fw": rng.integers(1, 4, size=n),
        }
    )
    expanded = df.loc[df.index.repeat(df["fw"])].reset_index(drop=True)
    return df, expanded


EXPANSION = dict(MODELS)
EXPANSION["nbreg"] = lambda d, **kw: sp.nbreg("cnt ~ x1 + x2", d, **kw)


@pytest.mark.parametrize("model", sorted(EXPANSION))
def test_frequency_weights_equal_the_expanded_data(overdispersed, model):
    df, expanded = overdispersed
    weighted = _fit(EXPANSION[model], df, weights="fw")
    copies = _fit(EXPANSION[model], expanded)
    np.testing.assert_allclose(weighted.params, copies.params, rtol=2e-6)
    np.testing.assert_allclose(weighted.std_errors, copies.std_errors, rtol=2e-6)
    if "ll" in weighted.model_info and "ll" in copies.model_info:
        assert weighted.model_info["ll"] == pytest.approx(
            copies.model_info["ll"], rel=1e-7
        )


def test_weights_move_the_poisson_variance(data):
    """The defect in one line: the same weights, scaled, changed nothing."""
    base = _fit(MODELS["poisson"], data, weights="w")
    scaled = _fit(MODELS["poisson"], data.assign(w=4.0 * data["w"]), weights="w")
    np.testing.assert_allclose(scaled.params, base.params, rtol=1e-9)
    # model-based variance: four times the information, half the SE
    np.testing.assert_allclose(scaled.std_errors, base.std_errors / 2.0, rtol=1e-9)
    robust = _fit(MODELS["poisson"], data, weights="w", robust="robust")
    robust4 = _fit(
        MODELS["poisson"], data.assign(w=4.0 * data["w"]), weights="w", robust="robust"
    )
    # sandwich: invariant to the scale of the weights
    np.testing.assert_allclose(robust4.std_errors, robust.std_errors, rtol=1e-9)
