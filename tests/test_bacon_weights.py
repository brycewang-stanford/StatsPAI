"""``sp.bacon_decomposition(weights=)`` and the comparison labels.

The weighted decomposition is checked against identities that pin it down
completely:

* the weighted TWFE coefficient from ``sp.feols(..., weights=)`` equals
  ``sum(weight * estimate)`` (Theorem 1 with weight shares), and with
  covariates ``omega * beta_within + (1 - omega) * beta_between``;
* an integer weight is the same as repeating the unit that many times, so
  the weighted table equals the unweighted table of the replicated panel;
* equal weights give the unweighted table.

Tolerance 1e-10: every number is a ratio of sums over at most a few
thousand rows, so the identities hold to rounding (observed <= 5e-16).
Stata ``bacondecomp [aw=]`` is compared in
``tests/reference_parity/test_bacon_weights_Stata_parity.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

TOL = 1e-10


def _panel(seed: int = 0) -> pd.DataFrame:
    """Staggered panel with never-, always- and three timing groups."""
    rng = np.random.default_rng(seed)
    cohorts = [1, 1, 4, 4, 4, 6, 6, 8, 8, 8, 0, 0, 0, 0]  # 1 = always, 0 = never
    rows = []
    for unit, g in enumerate(cohorts):
        a = rng.normal()
        w = float(np.exp(rng.normal(0, 0.7)))
        for t in range(1, 11):
            d = int(g > 0 and t >= g)
            effect = (1.0 + 0.3 * (t - g)) * d
            rows.append(
                {
                    "unit": unit,
                    "t": t,
                    "d": d,
                    "y": a + 0.2 * t + effect + rng.normal(0, 0.3),
                    "x": rng.normal() + 0.1 * t,
                    "w": w,
                    "wi": float(1 + unit % 3),
                }
            )
    return pd.DataFrame(rows)


def _twfe(df: pd.DataFrame, rhs: str, weights=None) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols(f"y ~ {rhs} | unit + t", df, weights=weights)
    return float(fit.params["d"])


def _bacon(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.bacon_decomposition(df, y="y", treat="d", time="t", id="unit", **kw)


def _sorted(decomp: pd.DataFrame) -> pd.DataFrame:
    key = decomp["control"].astype(str)
    return (
        decomp.assign(_c=key)
        .sort_values(["type", "treated", "_c"])
        .reset_index(drop=True)
    )


def test_weighted_identity_equals_weighted_twfe():
    df = _panel()
    r = _bacon(df, weights="w")
    beta = _twfe(df, "d", weights="w")
    assert r["beta_twfe"] == pytest.approx(beta, abs=TOL)
    assert r["weighted_sum"] == pytest.approx(beta, abs=TOL)
    dec = r["decomposition"]
    assert float((dec["weight"] * dec["estimate"]).sum()) == pytest.approx(
        beta, abs=TOL
    )
    # weighting changes the estimand on this panel
    assert abs(beta - _twfe(df, "d")) > 1e-3


def test_weighted_with_covariates_identity():
    df = _panel(1)
    r = _bacon(df, weights="w", covariates=["x"])
    beta = _twfe(df, "d + x", weights="w")
    assert r["beta_twfe"] == pytest.approx(beta, abs=TOL)
    rebuilt = r["omega"] * r["beta_within"] + (1 - r["omega"]) * r["beta_between"]
    assert rebuilt == pytest.approx(beta, abs=TOL)
    assert r["weighted_sum"] == pytest.approx(beta, abs=TOL)


@pytest.mark.parametrize("covariates", [None, ["x"]])
def test_integer_weights_equal_replicated_units(covariates):
    df = _panel(2)
    weighted = _bacon(df, weights="wi", covariates=covariates)
    copies = [
        df[df["wi"] > k].assign(unit=lambda f, k=k: f["unit"] * 10 + k)
        for k in range(int(df["wi"].max()))
    ]
    replicated = _bacon(pd.concat(copies, ignore_index=True), covariates=covariates)
    assert weighted["beta_twfe"] == pytest.approx(replicated["beta_twfe"], abs=TOL)
    a = _sorted(weighted["decomposition"])
    b = _sorted(replicated["decomposition"])
    assert list(a["type"]) == list(b["type"])
    np.testing.assert_allclose(a["estimate"], b["estimate"], atol=TOL, rtol=0)
    np.testing.assert_allclose(a["weight"], b["weight"], atol=TOL, rtol=0)
    if covariates:
        assert weighted["omega"] == pytest.approx(replicated["omega"], abs=TOL)
        assert weighted["beta_within"] == pytest.approx(
            replicated["beta_within"], abs=1e-8
        )


@pytest.mark.parametrize("covariates", [None, ["x"]])
def test_equal_weights_reproduce_unweighted_table(covariates):
    df = _panel(3).assign(c=3.7)
    weighted = _bacon(df, weights="c", covariates=covariates)
    plain = _bacon(df, covariates=covariates)
    assert weighted["beta_twfe"] == pytest.approx(plain["beta_twfe"], abs=1e-12)
    a, b = weighted["decomposition"], plain["decomposition"]
    # rtol: the covariate path has a pair with a tiny weight whose estimate
    # (-30) is the ratio of two near-zero variances
    np.testing.assert_allclose(a["estimate"], b["estimate"], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a["weight"], b["weight"], atol=1e-12, rtol=0)
    assert weighted["weights"] == "c" and plain["weights"] is None


def test_weights_varying_within_unit_raise_with_hint():
    df = _panel()
    df.loc[df["unit"] == 3, "w"] = np.linspace(1, 2, 10)
    with pytest.raises(MethodIncompatibility) as info:
        _bacon(df, weights="w")
    assert "varies within" in str(info.value)
    assert "first" in info.value.recovery_hint


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan])
def test_nonpositive_or_missing_weights_raise(bad):
    df = _panel()
    df.loc[df["unit"] == 2, "w"] = bad
    with pytest.raises(MethodIncompatibility) as info:
        _bacon(df, weights="w")
    assert info.value.recovery_hint


def test_comparison_class_labels_the_forbidden_types():
    r = _bacon(_panel())
    dec = r["decomposition"]
    forbidden = dec["type"].isin(
        ["Later vs Earlier Treated", "Later vs Always Treated"]
    )
    assert set(dec["type"]) == {
        "Treated vs Untreated",
        "Earlier vs Later Treated",
        "Later vs Earlier Treated",
        "Later vs Always Treated",
    }
    assert (dec["is_forbidden"] == forbidden).all()
    assert (dec.loc[forbidden, "comparison_class"] == "forbidden").all()
    assert (dec.loc[~forbidden, "comparison_class"] == "clean").all()
    # the share already reported equals the weight of the labelled rows
    assert float(dec.loc[dec["is_forbidden"], "weight"].sum()) == pytest.approx(
        r["already_treated_control_weight_share"], abs=1e-15
    )
    summary = r["summary"].set_index("type")["comparison_class"]
    assert summary["Later vs Earlier Treated"] == "forbidden"
    assert summary["Treated vs Untreated"] == "clean"
    assert "comparison_class" in r.summary() and "is_forbidden" in r.summary()


def test_comparison_class_with_covariates():
    dec = _bacon(_panel(), covariates=["x"])["decomposition"]
    cls = dict(zip(dec["type"], dec["comparison_class"]))
    assert cls["Both Treated"] == "mixed"
    assert cls["Later vs Always Treated"] == "forbidden"
    # the always-treated vs never-treated pair has no 2x2 variation
    no_2x2 = dec["beta_2x2"].isna()
    assert no_2x2.any()
    assert (dec.loc[no_2x2, "comparison_class"] == "covariates_only").all()
    assert not dec.loc[~dec["type"].eq("Later vs Always Treated"), "is_forbidden"].any()


@pytest.mark.parametrize("weights", [None, "w"])
def test_float32_columns_are_decomposed_in_double_precision(weights):
    # A .dta float arrives as float32; the means and group sums used to stay
    # in single precision and the identity failed by 2e-7 (divorce data).
    df = _panel(4)
    df32 = df.astype({"y": np.float32, "d": np.float32, "w": np.float32})
    r = _bacon(df32, weights=weights)
    assert r["weighted_sum"] == pytest.approx(r["beta_twfe"], abs=1e-12)
    beta = _twfe(df32.astype({"y": float, "d": float, "w": float}), "d", weights)
    assert r["beta_twfe"] == pytest.approx(beta, abs=TOL)
