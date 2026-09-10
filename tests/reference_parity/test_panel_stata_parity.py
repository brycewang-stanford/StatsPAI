"""Cross-language parity: panel FGLS and random-effects binary models vs Stata.

Fixture: ``_fixtures/_generate_panel_stata.do`` (Stata 18, all commands
official). Balanced N=60, T=12, with ``x1`` correlated with the unit effect.

Two defects this file was written to close, and they are different in kind:

* **A default that did not match the command it claimed to be.**
  ``sp.panel_fgls`` iterated its variance estimates to convergence, which
  is Stata's ``igls`` option, while its docstring said "Equivalent to
  Stata's ``xtgls y x, panels(het)``" — the two-step estimator. 2.8% apart
  on the slope. Both are now reachable and the default is Stata's.
  The old behaviour is pinned too, against ``xtgls, igls``, so the change
  is a relabelling and not a loss.

* **A missing intercept.** ``sp.panel_logit(method='re')`` and
  ``sp.panel_probit(method='re')`` built their design from the regressor
  list alone. That is right for conditional FE logit, where the constant is
  differenced out, and wrong for a random-effects model, where it is a
  parameter. Every slope was biased.

  What identified it was not the size of the gap but its *stubbornness*:
  0.39%, unchanged from 12 to 30 quadrature points. A non-adaptive
  Gauss-Hermite rule that is merely coarse converges as points are added.
  One that is integrating the wrong likelihood does not.

The RE assertions below are therefore written as **convergence**
assertions, not fixed tolerances. Stata integrates adaptively and StatsPAI
does not, so the two cannot agree at machine level at any fixed number of
points; what a correct likelihood must do is get closer as the rule gets
finer, and that is what is checked.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def sjson():
    path = _FIX / "panel_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_panel_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    path = _FIX / "panel_stata_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_panel_stata.do first")
    return pd.read_csv(path)


def _fgls(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.panel_fgls(data, y="y", x=["x1", "x2"], id="id", time="t", **kw)


def _binary(fn, data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(data, y="ybin", x=["x1", "x2"], id="id", time="t", **kw)


# ── panel FGLS ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name,key", [("x1", "x1"), ("x2", "x2"), ("_cons", "cons")])
def test_panel_fgls_two_step_matches_xtgls(sjson, data, name, key):
    res = _fgls(data)
    got = res.params.get(name, res.params.get("const"))
    assert float(got) == pytest.approx(sjson["xtgls"][key], rel=1e-10)


@pytest.mark.parametrize("name,key", [("x1", "se_x1"), ("x2", "se_x2")])
def test_panel_fgls_standard_errors_match_xtgls(sjson, data, name, key):
    res = _fgls(data)
    assert float(res.std_errors[name]) == pytest.approx(sjson["xtgls"][key], rel=1e-10)


def test_panel_fgls_igls_matches_xtgls_igls(sjson, data):
    """The old default, under the name Stata gives it."""
    res = _fgls(data, igls=True)
    assert float(res.params["x1"]) == pytest.approx(sjson["xtgls_igls"]["x1"], rel=1e-6)


def test_two_step_and_igls_are_different_estimators(data):
    """Regression guard, no reference needed.

    If ``igls`` ever becomes the default again the two collapse onto each
    other and this fails.
    """
    two_step = float(_fgls(data).params["x1"])
    iterated = float(_fgls(data, igls=True).params["x1"])
    assert two_step != pytest.approx(iterated, rel=1e-4)


# ── random-effects binary models ─────────────────────────────────────────


@pytest.mark.parametrize(
    "fn,key",
    [(sp.panel_logit, "xtlogit_re"), (sp.panel_probit, "xtprobit_re")],
)
def test_re_binary_includes_an_intercept(fn, key, sjson, data):
    """The constant must be estimated, and estimated correctly."""
    res = _binary(fn, data, method="re", n_quadrature=60)
    assert "_cons" in res.params.index
    assert float(res.params["_cons"]) == pytest.approx(sjson[key]["cons"], rel=1e-3)


@pytest.mark.parametrize(
    "fn,key",
    [(sp.panel_logit, "xtlogit_re"), (sp.panel_probit, "xtprobit_re")],
)
def test_re_binary_converges_to_stata_with_quadrature(fn, key, sjson, data):
    """Convergence, not a fixed tolerance.

    Stata integrates adaptively; StatsPAI does not. A correct likelihood
    must therefore approach Stata's answer as the rule is refined, and a
    wrong one will sit at a constant distance — which is exactly how the
    missing intercept was found.
    """
    ref = sjson[key]

    def err(n_quad):
        res = _binary(fn, data, method="re", n_quadrature=n_quad)
        return max(
            abs(float(res.params["x1"]) - ref["x1"]) / abs(ref["x1"]),
            abs(float(res.params["x2"]) - ref["x2"]) / abs(ref["x2"]),
        )

    coarse = err(8)
    fine = err(60)
    assert fine < coarse
    assert fine < 1e-5


@pytest.mark.parametrize(
    "fn,key",
    [(sp.panel_logit, "xtlogit_re"), (sp.panel_probit, "xtprobit_re")],
)
def test_re_binary_log_likelihood_matches_stata(fn, key, sjson, data):
    """The likelihood is the sharpest single check: same objective, same optimum."""
    res = _binary(fn, data, method="re", n_quadrature=60)
    assert res.model_info["log_likelihood"] == pytest.approx(sjson[key]["ll"], rel=1e-7)


@pytest.mark.parametrize(
    "fn,key",
    [(sp.panel_logit, "xtlogit_re"), (sp.panel_probit, "xtprobit_re")],
)
def test_re_binary_variance_components_match_stata(fn, key, sjson, data):
    res = _binary(fn, data, method="re", n_quadrature=60)
    assert res.model_info["sigma_u"] == pytest.approx(sjson[key]["sigma_u"], rel=1e-4)
    assert res.model_info["rho"] == pytest.approx(sjson[key]["rho"], rel=1e-4)


def test_conditional_fe_logit_has_no_intercept(data):
    """Deliberately unchanged: the constant is differenced out there.

    The RE fix must not leak into the FE path, which shares the grouping
    helper.
    """
    res = _binary(sp.panel_logit, data, method="fe")
    assert "_cons" not in res.params.index
