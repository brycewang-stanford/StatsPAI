"""``sp.rdms`` at several boundary points, with the ``xnorm`` pooled row.

Reference: R ``rdmulti`` 2.0 ``rdms(Y, X, C, X2, zvar, C2, xnorm=)`` on a
two-score design in which treatment requires both scores to clear zero
(``_fixtures/_generate_rdms_points_R.R``), the layout of Section 5 of
Cattaneo, Idrobo & Titiunik (2024). Same bytes on both sides, deterministic
estimator: tolerance 1e-9.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-9
Z = 1.959963984540054


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rdms_points_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdms_points_R.R to build the fixture")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def design():
    return pd.read_csv(_FIX / "rdms_points_design.csv")


@pytest.fixture(scope="module")
def fit(design, rjson):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdms(
            design,
            y="y",
            x1="x1",
            x2="x2",
            cutoff1=rjson["cutoff1"],
            cutoff2=rjson["cutoff2"],
            treat="tr",
            xnorm="xnorm",
        )


def test_returns_one_row_per_boundary_point(fit, rjson):
    assert isinstance(fit, sp.rd.RDMultiResult)
    assert [cr["cutoff"] for cr in fit.cutoff_results] == [
        (float(a), float(b)) for a, b in zip(rjson["cutoff1"], rjson["cutoff2"])
    ]


@pytest.mark.parametrize("i", [0, 1, 2])
def test_point_estimates_and_robust_inference_match_r(fit, rjson, i):
    cr = fit.cutoff_results[i]
    assert cr["estimate"] == pytest.approx(rjson["coefs"][i], rel=RTOL)
    assert cr["estimate_robust"] == pytest.approx(rjson["coefs_rb"][i], rel=RTOL)
    assert cr["se"] ** 2 == pytest.approx(rjson["var_rb"][i], rel=RTOL)
    assert cr["ci_lower"] == pytest.approx(rjson["ci_lower"][i], rel=RTOL)
    assert cr["ci_upper"] == pytest.approx(rjson["ci_upper"][i], rel=RTOL)
    assert cr["p_value"] == pytest.approx(rjson["pvalues"][i], rel=1e-6, abs=1e-300)
    assert cr["bandwidth"] == pytest.approx(rjson["h"][i], rel=RTOL)
    assert cr["n"] == int(rjson["Nh"][i])


def test_xnorm_pooled_row_matches_r(fit, rjson):
    assert fit.pooled_estimate == pytest.approx(rjson["pooled_coef"], rel=RTOL)
    assert fit.pooled_estimate_robust == pytest.approx(
        rjson["pooled_coef_rb"], rel=RTOL
    )
    assert fit.pooled_se**2 == pytest.approx(rjson["pooled_var_rb"], rel=RTOL)
    assert fit.pooled_ci[0] == pytest.approx(rjson["pooled_ci"][0], rel=RTOL)
    assert fit.pooled_ci[1] == pytest.approx(rjson["pooled_ci"][1], rel=RTOL)


def test_each_row_is_the_single_point_call(design, fit, rjson):
    """The multi-point form adds nothing to the estimator."""
    for cr, a, b in zip(fit.cutoff_results, rjson["cutoff1"], rjson["cutoff2"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            one = sp.rdms(
                design, y="y", x1="x1", x2="x2", cutoff1=a, cutoff2=b, treat="tr"
            )
        assert cr["estimate_robust"] == one.estimate
        assert (cr["ci_lower"], cr["ci_upper"]) == pytest.approx(one.ci, rel=1e-14)


def test_pooled_row_is_rdrobust_on_xnorm(design, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = sp.rdrobust(design, y="y", x="xnorm", c=0)
    assert fit.pooled_ci == pytest.approx(ref.ci, rel=1e-14)


def test_without_xnorm_there_is_no_pooled_row(design, rjson):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdms(
            design,
            y="y",
            x1="x1",
            x2="x2",
            cutoff1=rjson["cutoff1"],
            cutoff2=rjson["cutoff2"],
            treat="tr",
        )
    assert np.isnan(res.pooled_estimate)
    text = res.summary()
    assert "Pooled" not in text and "(30,0)" in text


def test_scalar_cutoffs_still_return_a_causal_result(design):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdms(design, y="y", x1="x1", x2="x2", treat="tr")
    assert type(res).__name__ == "CausalResult"


def test_point_lists_must_pair_up(design):
    with pytest.raises(ValueError, match="same number of boundary points"):
        sp.rdms(
            design,
            y="y",
            x1="x1",
            x2="x2",
            cutoff1=[0, 30],
            cutoff2=[0, 0, 40],
            treat="tr",
        )


def test_several_points_need_the_treatment_indicator(design):
    with pytest.raises(ValueError, match="treat= is required"):
        sp.rdms(design, y="y", x1="x1", x2="x2", cutoff1=[0, 30], cutoff2=[0, 0])


# ── one score, cumulative cutoffs ────────────────────────────────────────


@pytest.fixture(scope="module")
def cjson():
    path = _FIX / "rdms_cumulative_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdms_cumulative_R.R to build the fixture")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cumulative():
    return pd.read_csv(_FIX / "rdms_cumulative_design.csv")


def _check(res, ref):
    for i, cr in enumerate(res.cutoff_results):
        assert cr["estimate"] == pytest.approx(ref["coefs"][i], rel=RTOL)
        assert cr["estimate_robust"] == pytest.approx(ref["coefs_rb"][i], rel=RTOL)
        assert cr["se"] ** 2 == pytest.approx(ref["var_rb"][i], rel=RTOL)
        assert cr["ci_lower"] == pytest.approx(ref["ci_lower"][i], rel=RTOL)
        assert cr["ci_upper"] == pytest.approx(ref["ci_upper"][i], rel=RTOL)
        assert cr["bandwidth"] == pytest.approx(ref["h"][i], rel=RTOL)
        assert cr["n"] == int(ref["Nh"][i])


def test_cumulative_cutoffs_match_r(cumulative, cjson):
    """``rdms(Y, X, C)``: a sharp RD at each cutoff on the whole sample."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdms(cumulative, y="y", x1="x", cutoff1=cjson["cutoffs"])
    assert [cr["cutoff"] for cr in res.cutoff_results] == cjson["cutoffs"]
    _check(res, cjson["full"])
    assert np.isnan(res.pooled_estimate)
    assert "Pooled" not in res.summary()


def test_cumulative_cutoffs_with_ranges_match_r(cumulative, cjson):
    """``rangemat``: each cutoff is estimated on the units in its interval."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdms(
            cumulative,
            y="y",
            x1="x",
            cutoff1=cjson["cutoffs"],
            ranges=[tuple(r) for r in cjson["ranges"]],
        )
    _check(res, cjson["restricted"])
    assert res.cutoff_results[0]["range"] == (10.0, 50.0)


def test_cumulative_cutoff_is_rdrobust_at_that_cutoff(cumulative):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdms(cumulative, y="y", x1="x", cutoff1=[33])
        ref = sp.rdrobust(cumulative, y="y", x="x", c=33)
    cr = res.cutoff_results[0]
    assert (cr["ci_lower"], cr["ci_upper"]) == pytest.approx(ref.ci, rel=1e-14)


def test_cumulative_ranges_are_checked(cumulative):
    with pytest.raises(ValueError, match="does not contain cutoff"):
        sp.rdms(
            cumulative, y="y", x1="x", cutoff1=[33, 66], ranges=[(40, 50), (40, 90)]
        )
    with pytest.raises(ValueError, match="intervals for 2 cutoffs"):
        sp.rdms(cumulative, y="y", x1="x", cutoff1=[33, 66], ranges=[(10, 50)])


def test_ranges_need_the_cumulative_form(design):
    with pytest.raises(ValueError, match="cumulative cutoffs"):
        sp.rdms(
            design,
            y="y",
            x1="x1",
            x2="x2",
            treat="tr",
            cutoff1=[0],
            cutoff2=[0],
            ranges=[(-1, 1)],
        )
