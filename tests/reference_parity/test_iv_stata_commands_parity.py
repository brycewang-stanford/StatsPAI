"""Stata IV commands, run through ``sp.stata``, against Stata 18.

Each case is a line of Stata as it appears in a do-file -- factor-variable
instruments, an equation written without blanks, ``ivregress``'s
large-sample default -- executed by ``sp.stata`` on
``sp.datasets.card_1995()`` (plus ``expcat``, four experience bands). The
references are ``_b[educ]``, ``_se[educ]`` and ``e(kappa)`` from Stata 18.0
on the same data written to a .dta, printed with 14 to 16 decimals.

What these pin, each a defect found while replicating a textbook's
do-files:

* ``sp.iv`` could not read ``C(g)`` or an interaction in an IV formula, so
  ``(educ = i.q)`` had no translation that ran;
* ``(educ=z1 - z3)`` -- no blanks around ``=``, blanks around the range
  hyphen -- was translated to the formula ``educ ~ z1 + - + z3``;
* ``ivregress`` without ``small`` reports large-sample standard errors,
  which the translation noted but did not request;
* the LIML ``kappa`` lost as many digits as it is close to 1.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.regression.iv import _liml_kappa

X = "exper expersq black south smsa"

# label: (command, b, se, kappa or None, rtol)
# rtol 1e-8: 2SLS and LIML are closed form on both sides. The two weakly
# identified LIML rows (first-stage F below 2) are 1e-7: the estimate
# divides by a first stage that is nearly zero, and they agree to 7e-9.
CASES = {
    "liml_classical": (
        f"ivregress liml lwage (educ = nearc4 nearc2) {X}",
        0.17463797477376,
        0.05376300838951,
        1.0008582983448175,
        1e-8,
    ),
    "liml_robust": (
        f"ivregress liml lwage (educ = nearc4 nearc2) {X}, vce(robust)",
        0.17463797477376,
        0.05785176093554,
        1.0008582983448175,
        1e-8,
    ),
    "liml_robust_small": (
        f"ivregress liml lwage (educ = nearc4 nearc2) {X}, vce(robust) small",
        0.17463797477376,
        0.05791914798335,
        None,
        1e-8,
    ),
    "liml_interacted_instruments": (
        "ivregress liml lwage (educ = nearc2 i.expcat#c.nearc2) black south "
        "smsa i.expcat, vce(robust)",
        1.08199996594340,
        6.39613490203404,
        1.0013954461996699,
        1e-7,
    ),
    "tsls_interacted_instruments": (
        "ivregress 2sls lwage (educ = nearc2 i.expcat#c.nearc2) black south "
        "smsa i.expcat, vce(robust)",
        0.14212717960761,
        0.09172932286826,
        None,
        1e-8,
    ),
    "liml_factor_instrument_no_blanks": (
        "ivregress liml lwage (educ=i.expcat)black south smsa exper",
        -0.00202111406929,
        0.02358526649578,
        1.0072242940003060,
        1e-7,
    ),
    "tsls_factor_and_square_exogenous": (
        "ivregress 2sls lwage (educ = nearc4) black i.expcat c.exper#c.exper, "
        "vce(robust)",
        0.26513627960726,
        0.04148215446923,
        None,
        1e-8,
    ),
    "tsls_spaced_range_cluster": (
        "ivregress 2sls lwage (educ=nearc4 - nearc2) exper - smsa, "
        "vce(cluster expcat)",
        0.16084872836697,
        0.07968232736432,
        None,
        1e-8,
    ),
}


@pytest.fixture(scope="module")
def card() -> pd.DataFrame:
    d = sp.datasets.card_1995().copy()
    d["expcat"] = np.select([d.exper <= 6, d.exper <= 9, d.exper <= 12], [1, 2, 3], 4)
    return d


@pytest.mark.parametrize("label", sorted(CASES))
def test_command_reproduces_stata(card, label):
    command, b, se, kappa, rtol = CASES[label]
    res = sp.stata(command, card)
    assert float(res.params["educ"]) == pytest.approx(b, rel=rtol)
    assert float(res.std_errors["educ"]) == pytest.approx(se, rel=rtol)
    if kappa is not None:
        assert res.model_info["kappa"] == pytest.approx(kappa, rel=1e-12)


def test_spaced_range_needs_the_dataset_columns():
    out = sp.from_stata("ivregress 2sls y (d=z1 - z3) x")
    assert out["ok"] is False
    assert "z1-z3" in out["error"]
    out = sp.from_stata(
        "ivregress 2sls y (d=z1 - z3) x", columns=["y", "d", "z1", "z2", "z3", "x"]
    )
    assert out["arguments"]["formula"] == "y ~ x + (d ~ z1 + z2 + z3)"


def test_factor_terms_equal_hand_made_dummies(card):
    dummies = pd.get_dummies(card["expcat"], prefix="e", drop_first=True).astype(float)
    wide = pd.concat([card, dummies], axis=1)
    by_formula = sp.iv(
        "lwage ~ black + C(expcat) + (educ ~ nearc4 + nearc2)", card, robust="hc1"
    )
    by_hand = sp.iv(
        "lwage ~ black + e_2 + e_3 + e_4 + (educ ~ nearc4 + nearc2)",
        wide,
        robust="hc1",
    )
    assert float(by_formula.params["educ"]) == pytest.approx(
        float(by_hand.params["educ"]), rel=1e-12
    )
    assert float(by_formula.std_errors["educ"]) == pytest.approx(
        float(by_hand.std_errors["educ"]), rel=1e-10
    )
    assert "expcat[2]" in by_formula.params.index


def test_plain_formula_and_data_pass_through_untouched(card):
    from statspai.regression.iv import _materialise_formula_terms

    formula = "lwage ~ exper + (educ ~ nearc4) - 1"
    out_formula, out_data = _materialise_formula_terms(formula, card)
    assert out_formula == formula and out_data is card


def test_transformed_endogenous_regressor_is_refused(card):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="endogenous"):
        sp.iv("lwage ~ exper + (I(educ**2) ~ nearc4)", card)


def test_kappa_does_not_lose_digits_to_the_level_of_the_outcome():
    # kappa - 1 is about 8e-5 here. The old computation formed
    # W'W - W'P W for the two projections and took the generalized
    # eigenvalue of the pair, so its error grew with the size of the
    # uncentred cross products; adding a constant to y (absorbed by the
    # intercept, so kappa cannot change) by 1e6 turned kappa - 1 negative.
    rng = np.random.default_rng(3)
    n = 20_000
    z = rng.normal(size=(n, 3))
    x = np.column_stack([np.ones(n), rng.normal(size=n)])
    v = rng.normal(size=n)
    d = 0.01 * z[:, 0] + v
    y = 0.5 * d + x[:, 1] + 0.8 * v + rng.normal(size=n)
    base = _liml_kappa(y, x, d[:, None], z)
    assert 0 < base - 1 < 1e-3
    for shift in (1e3, 1e6):
        moved = _liml_kappa(y + shift, x, (d + shift)[:, None], z)
        # (kappa - 1) itself is preserved to 1e-6 relative.
        assert moved - 1 == pytest.approx(base - 1, rel=1e-6)


def test_kappa_ignores_a_collinear_instrument():
    rng = np.random.default_rng(5)
    n = 2_000
    z = rng.normal(size=(n, 2))
    x = np.ones((n, 1))
    v = rng.normal(size=n)
    d = z @ np.array([0.5, 0.3]) + v
    y = d + 0.5 * v + rng.normal(size=n)
    clean = _liml_kappa(y, x, d[:, None], z)
    redundant = _liml_kappa(y, x, d[:, None], np.column_stack([z, z[:, 0] + z[:, 1]]))
    assert redundant == pytest.approx(clean, rel=1e-10)


# ---------------------------------------------------------------------------
# estat endogenous after a robust or clustered fit
# ---------------------------------------------------------------------------
#
# Stata 18, same data:
#   ivregress 2sls lwage (educ = nearc4 nearc2) exper expersq black south smsa, <vce>
#   estat endogenous
# vce(robust):       robust score chi2(1) 3.96186750203015 (p .04654204192604)
#                    robust regression F(1,3002) 3.97786108490807 (p .04619235644120)
# vce(cluster band): robust regression F(1,11) 1.935232994298283 (p .1916830891411493)
#                    with band = floor(exper / 2); no score test is reported
# unadjusted:        Wu-Hausman F(1,3002) 3.86849860538490
# two endogenous regressors (educ exper = nearc4 nearc2 black#c.nearc4), robust:
#                    score chi2(2) 4.20308868366919, F(2,3001) 2.10758448653565

_ENDOG_FORMULA = (
    "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4 + nearc2)"
)


def test_robust_endogeneity_tests_match_estat_endogenous(card):
    res = sp.iv(_ENDOG_FORMULA, card, robust="hc1", small=False)
    out = sp.estat(res, "endogenous", print_results=False)
    assert out["statistic"] == pytest.approx(3.97786108490807, rel=1e-9)
    assert out["pvalue"] == pytest.approx(0.04619235644120, rel=1e-8)
    assert out["statistic_label"] == "F(1, 3002)"
    assert out["robust_score_chi2"] == pytest.approx(3.96186750203015, rel=1e-9)
    assert out["robust_score_pvalue"] == pytest.approx(0.04654204192604, rel=1e-8)
    # the homoskedastic statistic is still there for comparison
    assert out["wu_hausman_F"] == pytest.approx(3.86849860538490, rel=1e-9)


def test_clustered_endogeneity_test_matches_estat_endogenous(card):
    banded = card.assign(band=(card["exper"] // 2).astype(int))
    res = sp.iv(_ENDOG_FORMULA, banded, cluster="band")
    out = sp.estat(res, "endogenous", print_results=False)
    assert out["statistic"] == pytest.approx(1.935232994298283, rel=1e-9)
    assert out["pvalue"] == pytest.approx(0.1916830891411493, rel=1e-8)
    assert out["statistic_label"] == "F(1, 11)"
    assert "robust_score_chi2" not in out


def test_two_endogenous_regressors(card):
    wide = card.assign(bn4=card["black"] * card["nearc4"])
    res = sp.iv(
        "lwage ~ expersq + black + south + smsa + (educ + exper ~ nearc4 + nearc2 + bn4)",
        wide,
        robust="hc1",
    )
    out = sp.estat(res, "endogenous", print_results=False)
    assert out["statistic"] == pytest.approx(2.10758448653565, rel=1e-9)
    assert out["statistic_label"] == "F(2, 3001)"
    assert out["robust_score_chi2"] == pytest.approx(4.20308868366919, rel=1e-9)


def test_classical_fit_keeps_the_wu_hausman_test(card):
    res = sp.iv(_ENDOG_FORMULA, card)
    out = sp.estat(res, "endogenous", print_results=False)
    assert out["test"] == "Durbin-Wu-Hausman endogeneity test"
    assert out["statistic"] == pytest.approx(3.86849860538490, rel=1e-9)
    assert not any("Robust" in key for key in res.diagnostics)
