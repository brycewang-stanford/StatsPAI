"""Known-truth and boundary tests for ``sp.panel_match``.

Parity with R PanelMatch is in
``tests/reference_parity/test_panel_match_R_parity.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _panel(n_units=60, n_periods=12, tau=2.0, noise=0.0, seed=0):
    """y = a_i + g_t + tau * d_it (+ noise); d switches on and off."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n_units)
    g = rng.normal(size=n_periods).cumsum()
    rows = []
    for i in range(n_units):
        d = 0
        x = rng.normal()
        for t in range(n_periods):
            x = 0.5 * x + rng.normal()
            if t > 0:
                d = int(rng.random() < (0.75 if d else 0.25))
            y = a[i] + g[t] + tau * d + noise * rng.normal()
            rows.append((i, t, d, y, x))
    return pd.DataFrame(rows, columns=["id", "t", "d", "y", "x"])


@pytest.mark.parametrize(
    "refinement,covariates",
    [
        ("none", None),
        ("mahalanobis", ["x", "lag(y, 1:2)"]),
        ("ps_weight", ["x"]),
        ("ps_match", ["x", "lag(x, 1)"]),
    ],
)
def test_contemporaneous_effect_is_recovered_exactly(refinement, covariates):
    """Without noise the lead-0 DiD removes unit and period effects and the
    matched controls share the treated unit's untreated t-1, so every set
    contributes exactly tau whatever the refinement weights are."""
    df = _panel(tau=2.0)
    res = sp.panel_match(
        df,
        "y",
        "d",
        id="id",
        time="t",
        lag=2,
        refinement=refinement,
        covariates=covariates,
        se_method="conditional",
    )
    assert abs(res.estimate - 2.0) < 1e-10
    weights = res.model_info["matched_sets"].dropna(subset=["control_unit"])
    sums = weights.groupby(["treated_unit", "time"])["weight"].sum()
    np.testing.assert_allclose(sums, 1.0, atol=1e-12)


def test_atc_and_art_signs():
    """ATC (effect of treatment on those staying out) and ART (effect of
    switching out) are tau and -tau under a constant additive effect."""
    df = _panel(tau=1.5)
    atc = sp.panel_match(
        df,
        "y",
        "d",
        id="id",
        time="t",
        lag=2,
        refinement="none",
        qoi="atc",
        se_method="conditional",
    )
    art = sp.panel_match(
        df,
        "y",
        "d",
        id="id",
        time="t",
        lag=2,
        refinement="none",
        qoi="art",
        se_method="conditional",
    )
    assert abs(atc.estimate - 1.5) < 1e-10
    assert abs(art.estimate + 1.5) < 1e-10


def test_ate_is_set_weighted_average_of_att_and_atc():
    df = _panel(noise=1.0, seed=3)
    kw = dict(id="id", time="t", lag=2, refinement="none", lead=[0, 1])
    att = sp.panel_match(df, "y", "d", se_method="conditional", **kw)
    atc = sp.panel_match(df, "y", "d", qoi="atc", se_method="conditional", **kw)
    ate = sp.panel_match(df, "y", "d", qoi="ate", n_boot=50, seed=1, **kw)
    n1, n0 = att.model_info["n_matched_sets"], atc.model_info["n_matched_sets"]
    expected = (att.detail["estimate"] * n1 + atc.detail["estimate"] * n0) / (n1 + n0)
    np.testing.assert_allclose(ate.detail["estimate"], expected, atol=1e-12)


def test_pooled_estimate_is_mean_of_leads_and_bootstrap_is_seeded():
    df = _panel(noise=1.0, seed=4)
    kw = dict(id="id", time="t", lag=2, refinement="none", lead=[0, 1, 2], n_boot=200)
    r1 = sp.panel_match(df, "y", "d", seed=5, **kw)
    r2 = sp.panel_match(df, "y", "d", seed=5, **kw)
    assert abs(r1.estimate - r1.detail["estimate"].mean()) < 1e-12
    assert r1.se == r2.se
    assert r1.model_info["bootstrap_draws"].shape == (200, 3)
    lo, hi = r1.ci
    assert lo < r1.estimate < hi


def test_placebo_is_zero_under_parallel_trends_without_noise():
    df = _panel(tau=2.0)
    res = sp.panel_match(
        df,
        "y",
        "d",
        id="id",
        time="t",
        lag=4,
        refinement="none",
        se_method="conditional",
        placebo_test=True,
    )
    pl = res.model_info["placebo_test"]
    assert list(pl["term"]) == ["t-4", "t-3", "t-2"]
    np.testing.assert_allclose(pl["estimate"], 0.0, atol=1e-10)


def test_non_consecutive_time_is_ranked_with_a_warning():
    df = _panel(noise=1.0, seed=2)
    df2 = df.assign(t=df["t"] * 2 + 1990)
    a = sp.panel_match(
        df,
        "y",
        "d",
        id="id",
        time="t",
        lag=2,
        refinement="none",
        se_method="conditional",
    )
    with pytest.warns(UserWarning, match="consecutive"):
        b = sp.panel_match(
            df2,
            "y",
            "d",
            id="id",
            time="t",
            lag=2,
            refinement="none",
            se_method="conditional",
        )
    assert abs(a.estimate - b.estimate) < 1e-12


def test_missing_rows_are_missing_periods():
    """Dropping a unit-period row equals blanking its values."""
    df = _panel(noise=1.0, seed=6)
    blank = df.copy()
    hit = (blank["id"] == 3) & (blank["t"] == 5)
    blank.loc[hit, ["d", "y", "x"]] = np.nan
    kw = dict(id="id", time="t", lag=2, covariates=["x"], se_method="conditional")
    a = sp.panel_match(df[~hit], "y", "d", **kw)
    b = sp.panel_match(blank, "y", "d", **kw)
    assert abs(a.estimate - b.estimate) < 1e-12


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (dict(qoi="xyz"), MethodIncompatibility),
        (dict(refinement="ps.msm.weight", covariates=["x"]), MethodIncompatibility),
        (dict(refinement="mahalanobis"), MethodIncompatibility),  # no covariates
        (dict(qoi="ate", se_method="conditional"), MethodIncompatibility),
        (dict(lead=-1), MethodIncompatibility),
        (dict(lag=0), MethodIncompatibility),
        (dict(qoi="atc", forbid_treatment_reversal=True), MethodIncompatibility),
        (dict(covariates=["x + y"]), MethodIncompatibility),
        (dict(lag=50), DataInsufficient),
    ],
)
def test_refusals(kwargs, exc):
    df = _panel(noise=1.0, seed=7)
    base = dict(id="id", time="t", refinement="none", se_method="conditional")
    base.update(kwargs)
    with pytest.raises(exc):
        sp.panel_match(df, "y", "d", **base)


def test_non_binary_treatment_is_refused():
    df = _panel(noise=1.0).assign(d=lambda f: f["d"] * 2)
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.panel_match(df, "y", "d", id="id", time="t", lag=2, refinement="none")


def test_duplicate_rows_are_refused():
    df = _panel(noise=1.0)
    with pytest.raises(MethodIncompatibility, match="repeated"):
        sp.panel_match(
            pd.concat([df, df.head(1)]), "y", "d", id="id", time="t", refinement="none"
        )


def test_registered_and_schema():
    assert "panel_match" in sp.list_functions()
    schema = sp.function_schema("panel_match")
    props = schema.get("parameters", schema).get("properties", {})
    assert "refinement" in props and "qoi" in props
