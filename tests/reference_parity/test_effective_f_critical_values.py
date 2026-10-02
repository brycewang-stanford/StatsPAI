"""Montiel Olea-Pflueger critical values for the effective F, against Stata
``weakivtest``.

``sp.effective_f_test`` reported the effective F and compared it with 23.1,
a single rule-of-thumb number. The test in Montiel Olea and Pflueger
(``olea2013robust``) has critical values that depend on the data: on the
effective degrees of freedom, and on the worst-case Nagar bias of the
estimator in use, which differs between TSLS and LIML. With several
instruments they can be far from 23.1 in either direction.

Data: ``sp.datasets.card_1995()`` (with ``z3 = nearc4 * exper`` and
``z4 = nearc2 * black`` for the four-instrument design), written to a .dta
and run in Stata 18.0 as::

    ivreg2 lwage (educ = nearc4 nearc2 [z3 z4]) exper expersq black south smsa, robust
    weakivtest

``weakivtest`` returns its critical values with seven significant digits;
all agree. At ``tau = 30%`` it evaluates the non-centrality at ``x = 3.33``
where the definition is ``1 / 0.3``, so its 30% values differ from the exact
ones by 6e-4 and are not used as references.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

X = ["exper", "expersq", "black", "south", "smsa"]

# Stata prints seven significant digits.
RTOL = 1e-6

# design: (instruments, F_eff, {estimator: (tau 5%, 10%, 20%)}, x at 5%)
STATA = {
    "two_instruments": (
        ["nearc4", "nearc2"],
        9.642772138295,
        {
            "tsls": (4.958264, 4.058641, 3.554023),
            "liml": (19.92372, 12.54592, 8.41869),
            "simplified": (32.52147, 19.44288, 12.2795),
        },
        {"tsls": 0.76452, "liml": 10.35163},
    ),
    "four_instruments": (
        ["nearc4", "nearc2", "z3", "z4"],
        7.209330280788,
        {
            "tsls": (19.87776, 12.00725, 7.723269),
            "liml": (11.42467, 7.402526, 5.168333),
            "simplified": (29.35681, 17.0883, 10.49779),
        },
        {"tsls": 12.21472, "liml": 5.676648},
    ),
}


@pytest.fixture(scope="module")
def card() -> pd.DataFrame:
    d = sp.datasets.card_1995().copy()
    d["z3"] = d["nearc4"] * d["exper"]
    d["z4"] = d["nearc2"] * d["black"]
    return d


@pytest.fixture(scope="module")
def fits(card):
    return {
        name: sp.effective_f_test(card, "educ", z, X, y="lwage")
        for name, (z, *_rest) in STATA.items()
    }


@pytest.mark.parametrize("design", sorted(STATA))
def test_effective_f_matches_stata(fits, design):
    assert fits[design]["F_eff"] == pytest.approx(STATA[design][1], rel=1e-11)


@pytest.mark.parametrize("design", sorted(STATA))
@pytest.mark.parametrize("estimator", ["tsls", "liml", "simplified"])
def test_critical_values_match_weakivtest(fits, design, estimator):
    reference = STATA[design][2][estimator]
    got = fits[design]["critical_values"][estimator]
    for tau, ref in zip((0.05, 0.10, 0.20), reference):
        assert got[tau] == pytest.approx(ref, rel=RTOL)


@pytest.mark.parametrize("design", sorted(STATA))
def test_worst_case_bias_matches_weakivtest(fits, design):
    # weakivtest returns x = B / tau at tau = 5%.
    for estimator, x5 in STATA[design][3].items():
        assert fits[design]["worst_case_bias"][estimator] / 0.05 == pytest.approx(
            x5, rel=RTOL
        )


def test_the_estimator_matters(fits):
    # TSLS is the less biased of the two with two instruments here and the
    # more biased with four: neither critical value is uniformly smaller.
    two, four = fits["two_instruments"], fits["four_instruments"]
    assert two["critical_values"]["tsls"][0.10] < two["critical_values"]["liml"][0.10]
    assert four["critical_values"]["tsls"][0.10] > four["critical_values"]["liml"][0.10]
    for fit in (two, four):
        for est in ("tsls", "liml"):
            # the simplified bound is conservative
            assert (
                fit["critical_values"][est][0.10]
                <= fit["critical_values"]["simplified"][0.10] + 1e-9
            )
            assert 0 < fit["worst_case_bias"][est] <= 1 + 1e-9


def test_one_instrument_gives_the_rule_of_thumb(card):
    # With one instrument K_eff = 1 and the worst-case bias is 1, so every
    # set of critical values is the simplified one: 23.1 at tau = 10%.
    fit = sp.effective_f_test(card, "educ", ["nearc4"], X, y="lwage")
    for est in ("tsls", "liml", "simplified"):
        assert fit["critical_values"][est][0.10] == pytest.approx(23.1085, abs=1e-3)
        assert fit["effective_df"][est][0.10] == pytest.approx(1.0, rel=1e-9)
    exact = stats.ncx2.ppf(0.95, 1, 10.0)
    assert fit["critical_values"]["simplified"][0.10] == pytest.approx(exact, rel=1e-12)


def test_without_the_outcome_only_the_simplified_values(card):
    fit = sp.effective_f_test(card, "educ", ["nearc4", "nearc2"], X)
    assert list(fit["critical_values"]) == ["simplified"]
    assert fit["worst_case_bias"] == {}
    with_y = sp.effective_f_test(card, "educ", ["nearc4", "nearc2"], X, y="lwage")
    assert fit["F_eff"] == with_y["F_eff"]
    assert fit["critical_values"]["simplified"] == pytest.approx(
        with_y["critical_values"]["simplified"]
    )


def test_thirty_percent_is_the_exact_value_not_weakivtest_rounding(fits):
    # weakivtest: c_simp_30 uses x = 3.33; the definition is x = 1 / 0.3.
    fit = fits["two_instruments"]
    k_eff = fit["effective_df"]["simplified"][0.30]
    exact = stats.ncx2.ppf(0.95, k_eff, k_eff / 0.3) / k_eff
    assert fit["critical_values"]["simplified"][0.30] == pytest.approx(exact, rel=1e-12)


def test_alpha_changes_the_critical_value(card):
    z = ["nearc4", "nearc2"]
    five = sp.effective_f_test(card, "educ", z, X, y="lwage")
    one = sp.effective_f_test(card, "educ", z, X, y="lwage", alpha=0.01)
    assert one["alpha"] == 0.01
    assert one["critical_values"]["tsls"][0.10] > five["critical_values"]["tsls"][0.10]
    with pytest.raises(ValueError, match="alpha"):
        sp.effective_f_test(card, "educ", z, X, alpha=1.5)


def test_clustered_and_classical_variances_run(card):
    banded = card.assign(band=(card["exper"] // 2).astype(int))
    z = ["nearc4", "nearc2"]
    clustered = sp.effective_f_test(banded, "educ", z, X, y="lwage", cluster="band")
    classic = sp.effective_f_test(card, "educ", z, X, y="lwage", vcov="classic")
    for fit in (clustered, classic):
        assert set(fit["critical_values"]) == {"simplified", "tsls", "liml"}
        assert all(np.isfinite(v) for v in fit["critical_values"]["tsls"].values())
    # under homoskedasticity the effective F is the first-stage F
    assert classic["F_eff"] == pytest.approx(classic["first_stage_F"], rel=1e-10)
