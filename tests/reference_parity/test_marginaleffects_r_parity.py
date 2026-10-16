"""sp.predictions / comparisons / slopes / hypotheses against R marginaleffects.

R ``marginaleffects`` 0.32.0 (GPL-3) is a black box: only its printed
estimates and standard errors are compared. Fixtures:
``_fixtures/_generate_marginaleffects_data.py`` then
``Rscript _fixtures/_generate_marginaleffects_R.R``.

The probit model is fitted with ``sp.glm(..., information='expected')``:
R ``glm`` takes the covariance from the expected information, while
``sp.probit`` (like Stata ``probit``) uses the observed information, which
differs for a non-canonical link (standard errors here differ by 1.7%).
The GLMs are fitted in R with ``epsilon = 1e-14``; at glm's default the
covariance is about 1e-5 away from the inverse information at the
reported estimates.

Conventions checked here: averaged ratios are ratios of average
predictions; ``"sd"`` / ``"2sd"`` contrast mean +/- sd/2 (+/- sd) for
every row, not the row's own x +/- sd/2 as the label reads; a 0/1 variable is
a 1 - 0 contrast in ``slopes`` as well as in ``comparisons``.

Tolerances. Predictions, contrasts and ratios are the same closed-form
quantity on both sides (estimates ``rtol=1e-8``); marginaleffects takes
the Jacobian for the standard errors by finite differences (largest gap
observed 4e-6, so ``rtol=2e-5``). Slopes are a
finite difference in marginaleffects and analytic here; the largest
gap observed is 4e-5 (a standard error of an average slope), so
``rtol=2e-4``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
R = json.loads((FIX / "marginaleffects_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(FIX / "marginaleffects_data.csv")
DATA["g"] = DATA["g"].astype(int)

EXACT = {"est": dict(rtol=1e-8, atol=1e-12), "se": dict(rtol=2e-5, atol=1e-10)}
SLOPE = {"est": dict(rtol=1e-6, atol=1e-10), "se": dict(rtol=2e-4, atol=1e-8)}


@pytest.fixture(scope="module")
def fits():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "lm": sp.regress("yc ~ x + I(x**2) + b + C(g)", data=DATA),
            "logit": sp.logit("y ~ x*b + z + C(g)", data=DATA),
            "probit": sp.glm(
                "y ~ x + z + b",
                data=DATA,
                family="binomial",
                link="probit",
                information="expected",
            ),
            "poisson": sp.poisson("cnt ~ x + z + C(g)", data=DATA),
        }


def _check(py: pd.DataFrame, ref, keys=("term", "contrast"), tol=EXACT):
    ref = pd.DataFrame(ref if isinstance(ref, list) else [ref])
    keys = [k for k in keys if k in ref.columns]
    if keys:
        py = py.assign(**{k: py[k].astype(str) for k in keys})
        want = [tuple(r) for r in ref[keys].astype(str).to_numpy()]
        py = py.set_index(keys).loc[want if len(keys) > 1 else [w[0] for w in want]]
    assert len(py) == len(ref)
    np.testing.assert_allclose(py["estimate"].to_numpy(), ref["estimate"], **tol["est"])
    np.testing.assert_allclose(
        py["std_error"].to_numpy(), ref["std.error"], **tol["se"]
    )


MODELS = ["lm", "logit", "probit", "poisson"]


@pytest.mark.parametrize("m", MODELS)
def test_average_predictions(fits, m):
    _check(sp.avg_predictions(fits[m]), R[m]["avg_predictions"], ())
    by = sp.avg_predictions(fits[m], by="g")
    _check(by, R[m]["avg_predictions_by_g"], ("g",))


@pytest.mark.parametrize("m", MODELS)
def test_unit_predictions(fits, m):
    _check(sp.predictions(fits[m]).head(5), R[m]["predictions_head"], ())
    if "predictions_link_head" in R[m]:
        link = sp.predictions(fits[m], type="link").head(5)
        _check(link, R[m]["predictions_link_head"], ())


@pytest.mark.parametrize("m", MODELS)
def test_average_comparisons_of_every_variable(fits, m):
    _check(sp.avg_comparisons(fits[m]), R[m]["avg_comparisons"])


@pytest.mark.parametrize("m", MODELS)
def test_comparisons_with_a_chosen_change(fits, m):
    _check(sp.avg_comparisons(fits[m], {"x": "sd"}), R[m]["comparisons_x_sd"])
    _check(sp.avg_comparisons(fits[m], {"x": 5}), R[m]["comparisons_x_5"])
    for spec, key in (
        ("2sd", "2sd"),
        ("iqr", "iqr"),
        ("minmax", "minmax"),
        ((-1, 2), "pair"),
    ):
        _check(sp.avg_comparisons(fits[m], {"x": spec}), R[m][f"comparisons_x_{key}"])


@pytest.mark.parametrize("m", MODELS)
def test_average_slopes(fits, m):
    _check(sp.avg_slopes(fits[m]), R[m]["avg_slopes"], tol=SLOPE)
    for s in ("eyex", "eydx", "dyex"):
        py = sp.avg_slopes(fits[m], "x", slope=s).drop(columns="contrast")
        _check(py, R[m][f"avg_slopes_{s}"], ("term",), tol=SLOPE)


@pytest.mark.parametrize("m", MODELS)
def test_slopes_by_group_weighted_and_unit_level(fits, m):
    _check(
        sp.avg_slopes(fits[m], "x", by="g"),
        R[m]["avg_slopes_by_g"],
        ("term", "g"),
        tol=SLOPE,
    )
    _check(sp.avg_slopes(fits[m], "x", wts="w"), R[m]["avg_slopes_wts"], tol=SLOPE)
    _check(sp.slopes(fits[m], "x").head(5), R[m]["slopes_head"], (), tol=SLOPE)


@pytest.mark.parametrize("m", ["logit", "poisson"])
def test_ratio_comparisons(fits, m):
    _check(sp.avg_comparisons(fits[m], "g", comparison="ratioavg"), R[m]["ratioavg_g"])
    _check(
        sp.avg_comparisons(fits[m], "g", comparison="lnratioavg"),
        R[m]["lnratioavg_g"],
    )
    _check(
        sp.comparisons(fits[m], "x", comparison="ratio").head(5),
        R[m]["ratio_head"],
        (),
    )
    # Averaged, 'lnratio' / 'ratio' mean the ratio of average predictions.
    _check(sp.avg_comparisons(fits[m], "x", comparison="lnratio"), R[m]["lnratio_avg"])


# R's coefficient order b2, b3 -> the same coefficients by name here (patsy
# puts factor terms before numeric ones, so positions differ).
_B2_B3 = {
    "lm": ("x", "I(x ** 2)"),
    "logit": ("x", "b"),
    "probit": ("x", "z"),
    "poisson": ("x", "z"),
}


@pytest.mark.parametrize("m", MODELS)
def test_hypothesis_on_coefficients(fits, m):
    names = list(fits[m].params.index)
    i, j = (names.index(n) + 1 for n in _B2_B3[m])
    _check(sp.hypotheses(fits[m], f"b{i} - b{j} = 0"), R[m]["hypothesis_b2_b3"], ())


def test_hypothesis_on_a_comparisons_table(fits):
    tab = sp.avg_comparisons(fits["logit"], "g")
    _check(sp.hypotheses(tab, "b1 - b2 = 0"), R["logit"]["hypothesis_on_table"], ())
    _check(sp.avg_comparisons(fits["logit"], "b"), R["logit"]["comparisons_b"])


def test_ratio_by_group_is_a_ratio_of_group_averages(fits):
    _check(
        sp.comparisons(fits["logit"], "b", comparison="ratio", by="g"),
        R["logit"]["ratio_by_g"],
        ("term", "contrast", "g"),
    )
