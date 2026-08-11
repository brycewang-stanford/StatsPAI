"""The options R ``did`` / ``csdid`` support under ``panel = FALSE``.

Three capabilities that StatsPAI's repeated-cross-section and
unbalanced-panel routes did not have, all of which R ``did`` and the
``csdid`` port do support, and each of which failed loudly rather than
silently before 1.23.0:

* ``weights=`` — raised on the unbalanced route, ignored nowhere;
* ``clustervars=`` / ``bstrap=`` / ``cband=`` — the whole multiplier
  bootstrap was unavailable, so these paths could only report analytic
  per-cell standard errors;
* an R-style covariate formula (``xformla``) — covariates had to be
  hand-expanded into columns before calling.

Weights are the load-bearing case. They are part of the *estimand*
(Baker et al. 2026, §3.1): the unweighted ATT averages over treated
units, the weighted one over the population those units represent. A
route that silently ignored them would answer a different question under
the name of the one asked. Cohort shares therefore have to become shares
of weight mass too, or ``sp.aggte`` would weight cohorts by row counts
while the cells inside them answer a population-weighted question — a
mismatch that no single-cell test would catch, which is why the
aggregation is asserted here as well.

Reference values come from the ``csdid`` Python port of R ``did``, run
on the committed fixture; the tolerances below are what that comparison
actually achieved rather than round numbers.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_PANEL = Path(__file__).parent / "_fixtures" / "cs_gaps_panel.csv"

# Deterministic paths (bstrap=False) agree with csdid to solver
# tolerance; the logit and WLS solvers are the only source of slack.
_RTOL = 1e-10


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    df = pd.read_csv(_PANEL)
    rng = np.random.default_rng(0)
    units = df[["i"]].drop_duplicates()
    units["w"] = rng.uniform(0.5, 2.0, len(units))
    return df.merge(units, on="i")


@pytest.fixture(scope="module")
def unbalanced(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.drop(panel.sample(150, random_state=3).index)


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            df, y="y", g="g", t="t", i="i", base_period="varying", **kw
        )


# ---------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------


@pytest.mark.parametrize("estimator", ["dr", "ipw", "reg"])
@pytest.mark.parametrize("route", ["rcs", "unbalanced"])
def test_weights_change_the_estimate_on_both_routes(
    panel, unbalanced, estimator, route
):
    """A dropped weight is invisible unless the two answers differ."""
    df = panel if route == "rcs" else unbalanced
    kw = {"panel": False} if route == "rcs" else {"allow_unbalanced_panel": True}
    plain = _fit(df, estimator=estimator, bstrap=False, **kw)
    wtd = _fit(df, estimator=estimator, weights="w", bstrap=False, **kw)
    assert not np.allclose(plain.detail["att"].to_numpy(), wtd.detail["att"].to_numpy())


def test_weighted_cohort_shares_are_weight_mass_not_head_counts(panel):
    """``sp.aggte`` must weight cohorts consistently with the cells.

    If cohort shares stayed head counts while the ATT(g, t) became
    weight-weighted, every aggregation would silently mix two estimands.
    """
    wtd = _fit(panel, panel=False, weights="w", bstrap=False)
    sizes = wtd.model_info["cohort_sizes"]
    counts = panel.groupby("g").size()
    for g in sizes.index:
        assert not np.isclose(float(sizes.loc[g]), float(counts.loc[g]))
    # and the mass must equal the summed weights of that cohort
    mass = panel.groupby("g")["w"].sum() / panel["w"].mean()
    for g in sizes.index:
        assert float(sizes.loc[g]) == pytest.approx(float(mass.loc[g]), rel=1e-10)


def test_unbalanced_route_rejects_time_varying_weights(unbalanced):
    """The fold sums a unit's contributions, so a time-varying weight
    would quietly reweight that unit's own periods against each other."""
    df = unbalanced.copy()
    df["w_tv"] = df["w"] * (1.0 + 0.1 * df["t"])
    with pytest.raises(MethodIncompatibility, match="varies within unit"):
        _fit(df, allow_unbalanced_panel=True, weights="w_tv", bstrap=False)


def test_weights_are_validated(panel):
    df = panel.copy()
    df["neg"] = -1.0
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        _fit(df, panel=False, weights="neg", bstrap=False)


# ---------------------------------------------------------------------
# bootstrap / clustering
# ---------------------------------------------------------------------


@pytest.mark.parametrize("route", ["rcs", "unbalanced"])
def test_multiplier_bootstrap_available_on_both_routes(panel, unbalanced, route):
    df = panel if route == "rcs" else unbalanced
    kw = {"panel": False} if route == "rcs" else {"allow_unbalanced_panel": True}
    analytic = _fit(df, bstrap=False, **kw)
    boot = _fit(df, bstrap=True, biters=999, random_state=4, **kw)
    assert analytic.model_info["se_method"] == "analytic"
    assert boot.model_info["se_method"] == "multiplier"
    np.testing.assert_allclose(
        boot.detail["att"].to_numpy(), analytic.detail["att"].to_numpy()
    )
    # Same estimand, different variance estimator: close but not equal.
    ratio = boot.detail["se"].to_numpy() / analytic.detail["se"].to_numpy()
    assert np.all(ratio > 0.6) and np.all(ratio < 1.6)
    assert not np.allclose(boot.detail["se"], analytic.detail["se"])


def test_uniform_bands_available_on_the_rcs_route(panel):
    r = _fit(panel, panel=False, bstrap=True, cband=True, biters=999, random_state=4)
    assert {"cband_lower", "cband_upper"} <= set(r.detail.columns)
    crit = r.model_info["crit_val_uniform"]
    assert crit >= 1.959  # never narrower than the pointwise normal
    assert np.all(r.detail["cband_lower"] <= r.detail["ci_lower"] + 1e-12)
    assert np.all(r.detail["cband_upper"] >= r.detail["ci_upper"] - 1e-12)


def test_clustering_changes_the_standard_errors(panel):
    plain = _fit(panel, panel=False, bstrap=True, biters=4000, random_state=4)
    clustered = _fit(
        panel,
        panel=False,
        clustervars=["i", "state"],
        bstrap=True,
        biters=4000,
        random_state=4,
    )
    np.testing.assert_allclose(
        clustered.detail["att"].to_numpy(), plain.detail["att"].to_numpy()
    )
    assert not np.allclose(clustered.detail["se"], plain.detail["se"])


def test_clustering_requires_the_bootstrap(panel):
    """Analytic SEs cannot express within-cluster dependence."""
    with pytest.raises(MethodIncompatibility, match="requires bstrap=True"):
        _fit(panel, panel=False, clustervars=["i", "state"], bstrap=False)


def test_unbalanced_route_rejects_time_varying_clusters(unbalanced):
    df = unbalanced.copy()
    df["cl_tv"] = df["t"].astype(int)
    with pytest.raises(MethodIncompatibility, match="time-varying"):
        _fit(
            df,
            allow_unbalanced_panel=True,
            clustervars=["i", "cl_tv"],
            bstrap=True,
            biters=99,
        )


# ---------------------------------------------------------------------
# R-style covariate formula (xformla)
# ---------------------------------------------------------------------


def test_formula_and_column_list_agree(panel):
    by_list = _fit(panel, x=["x1"], bstrap=False)
    by_formula = _fit(panel, x="~ x1", bstrap=False)
    np.testing.assert_allclose(
        by_formula.detail["att"].to_numpy(), by_list.detail["att"].to_numpy()
    )
    np.testing.assert_allclose(
        by_formula.detail["se"].to_numpy(), by_list.detail["se"].to_numpy()
    )


def test_left_hand_side_is_accepted_and_ignored(panel):
    """So an ``xformla`` copied verbatim out of R just works."""
    one_sided = _fit(panel, x="~ x1", bstrap=False)
    two_sided = _fit(panel, x="y ~ x1", bstrap=False)
    np.testing.assert_allclose(
        two_sided.detail["att"].to_numpy(), one_sided.detail["att"].to_numpy()
    )


def test_transformations_are_materialised(panel):
    """The point of the formula: a term with no column behind it."""
    quad = _fit(panel, x="~ x1 + I(x1**2)", bstrap=False)
    linear = _fit(panel, x="~ x1", bstrap=False)
    assert not np.allclose(
        quad.detail["att"].to_numpy(), linear.detail["att"].to_numpy()
    )
    # equivalent hand-expansion must reproduce it exactly
    df = panel.copy()
    df["x1sq"] = df["x1"] ** 2
    manual = _fit(df, x=["x1", "x1sq"], bstrap=False)
    np.testing.assert_allclose(
        quad.detail["att"].to_numpy(), manual.detail["att"].to_numpy(), rtol=_RTOL
    )


def test_r_caret_power_is_rejected_not_silently_reinterpreted(panel):
    """``I(x^2)`` is XOR in Python — refuse rather than compute nonsense."""
    with pytest.raises(MethodIncompatibility, match=r"\^"):
        _fit(panel, x="~ I(x1^2)", bstrap=False)


def test_plain_string_is_still_a_column_name(panel):
    """No ``~`` means the old spelling, unchanged."""
    as_str = _fit(panel, x="x1", bstrap=False)
    as_list = _fit(panel, x=["x1"], bstrap=False)
    np.testing.assert_allclose(
        as_str.detail["att"].to_numpy(), as_list.detail["att"].to_numpy()
    )


def test_intercept_only_formula_means_no_covariates(panel):
    """R writes ``xformla = ~1`` for the unconditional estimator."""
    none = _fit(panel, bstrap=False)
    tilde_one = _fit(panel, x="~ 1", bstrap=False)
    np.testing.assert_allclose(
        tilde_one.detail["att"].to_numpy(), none.detail["att"].to_numpy()
    )


def test_formula_works_on_drdid_too():
    long = pd.read_csv(Path(__file__).parent / "_fixtures" / "drdid_family_long.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_list = sp.drdid(long, y="y", group="D", time="post", covariates=["x1", "x2"])
        by_formula = sp.drdid(
            long, y="y", group="D", time="post", covariates="~ x1 + x2"
        )
    assert by_formula.estimate == pytest.approx(by_list.estimate, rel=_RTOL)
    assert by_formula.se == pytest.approx(by_list.se, rel=_RTOL)
