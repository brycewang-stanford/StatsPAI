"""Defects found while writing the teaching notebooks in ``examples/notebooks``.

Each test pins a call that used to return a wrong answer, or an answer that
looked better than it was, without saying so.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.matching.ps_diagnostics import _smd

COVS = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]


# --------------------------------------------------------------------- #
#  sp.liml: formula parsing
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def iv_data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 800
    z1, z2, w1, w2 = rng.normal(size=(4, n))
    u = rng.normal(size=n)
    d = 0.5 * z1 + 0.3 * z2 + 0.4 * w1 + u + rng.normal(size=n)
    y = 1.0 + 0.7 * d + 0.5 * w1 - 0.3 * w2 + 1.2 * u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "w1": w1, "w2": w2})


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ w1 + w2 + (d ~ z1 + z2)",
        "y ~ (d ~ z1 + z2) + w1 + w2",
        "y ~ w1 + (d ~ z1 + z2) + w2",
        "y ~ w1 + w2 | d | z1 + z2",
    ],
)
def test_liml_formula_reads_the_bracket_wherever_it_sits(iv_data, formula):
    # The endogenous-first spelling used to turn every control after the
    # bracket into an instrument and drop it from the structural equation.
    ref = sp.liml(
        data=iv_data, y="y", x_endog=["d"], x_exog=["w1", "w2"], z=["z1", "z2"]
    )
    got = sp.liml(formula, data=iv_data)
    assert list(got.params.index) == list(ref.params.index)
    np.testing.assert_allclose(got.params.values, ref.params.values, rtol=1e-12)
    np.testing.assert_allclose(got.std_errors.values, ref.std_errors.values, rtol=1e-12)


@pytest.mark.parametrize(
    "formula",
    ["y ~ d + w1", "y ~ (d ~ z1) + (w1 ~ z2)", "y ~ w1 | d", "y w1 d"],
)
def test_liml_rejects_a_formula_it_cannot_read(iv_data, formula):
    with pytest.raises(MethodIncompatibility):
        sp.liml(formula, data=iv_data)


# --------------------------------------------------------------------- #
#  Balance diagnostics: missing weights
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def matched():
    df = sp.datasets.nsw_lalonde()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.match(df, y="re78", treat="treat", covariates=COVS)
    return res


def test_missing_matching_weights_are_not_reported_as_perfect_balance(matched):
    # Unmatched comparison rows carry a missing ``_weight``. The weighted
    # control mean was NaN, and every weighted SMD came back as 0.0000 with
    # ``balanced=True``.
    frame = matched.matched_data
    assert frame["_weight"].isna().any()
    bal = sp.balance_diagnostics(
        frame, treatment="treat", covariates=COVS, weights="_weight"
    )
    filled = sp.balance_diagnostics(
        frame.assign(w=frame["_weight"].fillna(0.0)),
        treatment="treat",
        covariates=COVS,
        weights="w",
    )
    np.testing.assert_allclose(
        bal.table["smd_weighted"].values, filled.table["smd_weighted"].values
    )
    assert bal.table["weighted_mean_control"].notna().all()
    # Independent arithmetic for one covariate.
    ctrl = frame[frame["treat"] == 0]
    w = ctrl["_weight"].fillna(0.0)
    assert bal.table.loc["age", "weighted_mean_control"] == pytest.approx(
        float((ctrl["age"] * w).sum() / w.sum())
    )
    assert bal.table["smd_weighted"].abs().max() > 0.1  # age is not balanced

    ps_bal = sp.ps_balance(frame, "treat", COVS, weights=frame["_weight"])
    np.testing.assert_allclose(
        ps_bal.table["smd_weighted"].values, bal.table["smd_weighted"].values
    )


def test_balance_weights_that_define_nothing_are_an_error(matched):
    frame = matched.matched_data
    no_controls = np.where(frame["treat"] == 0, np.nan, 1.0)
    with pytest.raises(ValueError, match="no positive weight"):
        sp.balance_diagnostics(
            frame, treatment="treat", covariates=COVS, weights=no_controls
        )
    with pytest.raises(ValueError, match="non-negative"):
        sp.ps_balance(frame, "treat", COVS, weights=-np.ones(len(frame)))


def test_smd_is_missing_not_zero_when_it_is_undefined():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isnan(_smd(x, x, np.ones(3), np.full(3, np.nan)))
    # A constant covariate is balanced by construction.
    assert _smd(np.ones(3), np.ones(3)) == 0.0


def test_love_plot_reads_an_sp_match_result(matched):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    assert matched.model_info["treat"] == "treat"
    assert matched.model_info["covariates"] == COVS
    fig, ax = sp.love_plot(matched)
    # Raw and weighted markers for each covariate.
    weighted = [c for c in ax.collections if len(c.get_offsets()) == len(COVS)]
    assert weighted
    offsets = np.concatenate([np.asarray(c.get_offsets())[:, 0] for c in weighted])
    assert np.isfinite(offsets).all()
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------- #
#  Synthetic control
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def small_panel() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    units, periods = 12, 16
    factor = np.cumsum(rng.normal(size=periods))
    load = rng.uniform(0.5, 1.5, units)
    rows = []
    for i in range(units):
        for t in range(periods):
            y = 10 + load[i] * factor[t] + 0.3 * i + rng.normal(scale=0.3)
            if i == 0 and t >= 11:
                y -= 2.0
            # The treated unit's covariate sits in the middle of the donors'.
            x = 5 + 0.1 * ((i + 5) % units) + rng.normal(scale=0.01)
            rows.append((f"u{i}", t, y, x))
    return pd.DataFrame(rows, columns=["unit", "t", "y", "x"])


SYNTH = dict(outcome="y", unit="unit", time="t", treated_unit="u0", treatment_time=11)


def test_synth_compare_seed_makes_the_resampled_rows_reproducible(small_panel):
    first = sp.synth_compare(small_panel, methods=["classic", "sdid"], seed=7, **SYNTH)
    again = sp.synth_compare(small_panel, methods=["classic", "sdid"], seed=7, **SYNTH)
    cols = ["method", "att", "se", "ci_lower", "ci_upper"]
    pd.testing.assert_frame_equal(
        first.comparison_table[cols], again.comparison_table[cols]
    )
    direct = sp.sdid(small_panel, seed=7, **SYNTH)
    row = first.comparison_table.set_index("method").loc["sdid"]
    assert row["se"] == pytest.approx(direct.se, rel=1e-12)


def test_synth_compare_still_rejects_a_misspelled_option(small_panel):
    with pytest.raises(TypeError, match="unexpected keyword"):
        sp.synth_compare(small_panel, methods=["classic"], sead=1, **SYNTH)


def test_synth_compare_reports_missing_fit_as_missing(small_panel):
    comp = sp.synth_compare(small_panel, methods=["classic", "sdid"], seed=1, **SYNTH)
    table = comp.comparison_table.set_index("method")
    # Synthetic DID reports no level fit: that used to be ``inf``, and its
    # donor count used to be 0.
    assert np.isnan(table.loc["sdid", "pre_rmspe"])
    sdid_w = comp.results["sdid"].model_info["unit_weights"]["weight"]
    assert table.loc["sdid", "n_effective_donors"] == int((sdid_w.abs() > 0.01).sum())
    # Classic SCM used to report the size of the donor pool.
    classic_w = comp.results["classic"].model_info["weights"]["weight"]
    assert table.loc["classic", "n_effective_donors"] == int(
        (classic_w.abs() > 0.01).sum()
    )
    assert table.loc["classic", "n_effective_donors"] < 11
    assert comp.recommended == "classic"


def test_augmented_scm_cites_its_own_paper(small_panel):
    res = sp.synth(small_panel, method="augmented", placebo=False, **SYNTH)
    assert "benmichael2021augmented" in res.cite()


def test_classic_scm_says_when_the_predictors_do_not_determine_the_weights(
    small_panel,
):
    # One covariate, eleven donors: infinitely many weight vectors match it.
    with pytest.warns(UserWarning, match="do not pin down the donor weights"):
        res = sp.synth(small_panel, covariates=["x"], placebo=False, **SYNTH)
    assert res.model_info["in_predictor_hull"] is True
    # The outcome-only fit is determined and stays quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.synth(small_panel, placebo=False, **SYNTH)
