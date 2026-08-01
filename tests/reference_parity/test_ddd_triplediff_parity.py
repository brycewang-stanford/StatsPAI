"""Reference parity: ``sp.ddd_heterogeneous`` vs ``triplediff::ddd``.

Triple differences had no cross-language reference. CRAN gained one:
``triplediff`` (Ortiz-Villavicencio & Sant'Anna 2025), written by the paper's
own authors. With ``xformla = ~1`` its doubly-robust DDD reduces to the
unconditional cell means StatsPAI computes, so the per-``(g, t)`` estimates
are directly comparable -- and agree to 1e-14.

Reference generation (R 4.5.2, triplediff 0.2.4)::

    res <- ddd(yname = "y", tname = "time", idname = "id", gname = "state",
               pname = "partition", xformla = ~1, data = d,
               control_group = "nevertreated", base_period = "varying",
               est_method = "dr", panel = TRUE, boot = FALSE)
    agg_ddd(res, type = "simple")$aggte_ddd$overall.att

Fixture: ``_fixtures/ddd_staggered_panel.csv`` -- 600 units x 4 periods,
cohorts {2, 3, 4} plus never-treated (coded 0), affected share deliberately
varying across cohorts (0.70 / 0.50 / 0.35), plus two covariates that shift
both selection and the outcome path so the conditional and unconditional
estimands genuinely differ.

Two conventions differ and both are pinned:

* **Aggregation weights.** ``triplediff``'s ``agg_ddd(type="simple")``
  weights cohort ``g`` by ``mean(first_treat == g)`` over *all* units.
  ``sp.ddd_heterogeneous`` defaults to weighting by treated units in the
  affected subgroup. On this fixture that is 2.2251 versus 2.2075 -- which
  is why the affected share varies across cohorts here, so the gap cannot
  pass as rounding. ``weight_by="cohort"`` reproduces the package exactly.
* **Standard errors.** On the unconditional path this function reports a
  cluster bootstrap and the reference an analytic influence-function
  variance, so those are not compared. On the CONDITIONAL path they are:
  ``se="analytic"`` is the same estimator, and the SEs agree at 1e-13.

Pre-treatment placebo cells are also not compared: ``base_period="varying"``
makes ``triplediff`` report ``(g, t)`` for ``t < g``, which StatsPAI does not
build.

References
----------
Ortiz-Villavicencio, M. and Sant'Anna, P. H. C. (2025). "Better Understanding
Triple Differences Estimators." *arXiv preprint* arXiv:2505.09942.
[@ortiz2025better]
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "ddd_staggered_panel.csv"

# triplediff::ddd, post-treatment cells only.
R_CELLS = {
    (2, 2): 1.43571621318487,
    (2, 3): 2.1097359608128,
    (2, 4): 2.63198407956612,
    (3, 3): 2.51109681208115,
    (3, 4): 2.64727811934505,
    (4, 4): 2.01468518284991,
}
R_SIMPLE_ATT = 2.2250827280
SP_ELIGIBLE_ATT = 2.2074819726

COVARS = ["cov1", "cov2"]

# triplediff::ddd with xformla = ~cov1 + cov2, per est_method:
# {(g, t): (att, se)} plus the cohort-weighted aggregate.
R_CONDITIONAL = {
    "dr": {
        (2, 2): (1.18736502590493, 0.285210080760415),
        (2, 3): (1.91239709371401, 0.322327354758482),
        (2, 4): (2.35541557748624, 0.288419094416386),
        (3, 3): (2.40120301338968, 0.258352754021396),
        (3, 4): (2.50478640634718, 0.2375996651804),
        (4, 4): (1.77472330887725, 0.278890439765695),
    },
    "ipw": {
        (2, 2): (1.21437737682563, 0.296701632871344),
        (2, 3): (2.01756270752827, 0.384379634413682),
        (2, 4): (2.4780577162959, 0.347654151236991),
        (3, 3): (2.45522661684315, 0.288870306716963),
        (3, 4): (2.5496626839735, 0.258441919551313),
        (4, 4): (1.80933951797088, 0.270165819398959),
    },
    "reg": {
        (2, 2): (1.22086657786615, 0.292367834968852),
        (2, 3): (1.9999260231156, 0.315255877949331),
        (2, 4): (2.39985568973544, 0.284908327996232),
        (3, 3): (2.44964076077002, 0.26341077695216),
        (3, 4): (2.57038178542631, 0.236204841627053),
        (4, 4): (1.81166060137642, 0.270586418984563),
    },
}
R_CONDITIONAL_AGG = {
    "dr": (2.02264840428655, 0.150045324551069),
    "ipw": (2.08737110323955, 0.177956444726184),
    "reg": (2.07538857304832, 0.148845681297161),
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    return pd.read_csv(_FIXTURE)


def _fit(df: pd.DataFrame, weight_by: str = "eligible"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ddd_heterogeneous(
            df,
            y="y",
            unit="id",
            time="time",
            cohort="state",
            subgroup="partition",
            n_boot=0,
            seed=0,
            weight_by=weight_by,
        )


@pytest.fixture(scope="module")
def fit(panel):
    return _fit(panel)


def test_all_post_cells_are_present(fit):
    got = {(int(r.cohort), int(r.time)) for r in fit.detail.itertuples()}
    assert got == set(R_CELLS)


@pytest.mark.parametrize("cell", sorted(R_CELLS))
def test_cell_matches_triplediff(fit, cell):
    g, t = cell
    detail = fit.detail
    row = detail[(detail["cohort"] == g) & (detail["time"] == t)]
    got = float(row["ddd"].iloc[0])
    assert got == pytest.approx(
        R_CELLS[cell], abs=1e-9
    ), f"(g={g}, t={t}): StatsPAI {got:.10f} vs triplediff {R_CELLS[cell]:.10f}"


def test_cohort_weighted_aggregate_matches_triplediff(panel):
    assert _fit(panel, weight_by="cohort").estimate == pytest.approx(
        R_SIMPLE_ATT, abs=1e-9
    )


def test_default_weighting_is_unchanged(fit):
    """Guards the default: switching it would move published numbers."""
    assert fit.model_info["weight_by"] == "eligible"
    assert fit.estimate == pytest.approx(SP_ELIGIBLE_ATT, abs=1e-9)


def test_the_two_weightings_actually_differ_here(panel):
    """The fixture varies the affected share across cohorts on purpose, so
    the convention gap cannot pass as rounding."""
    gap = abs(_fit(panel, "cohort").estimate - _fit(panel, "eligible").estimate)
    assert gap > 0.01, gap


def test_unknown_weight_by_fails_loudly(panel):
    with pytest.raises(ValueError, match="weight_by"):
        _fit(panel, weight_by="population")


def test_placebo_arm_is_near_zero(fit):
    """The DGP gives the affected subgroup its own linear trend, which the
    DDD nets out; the placebo DIDs carry that trend, not the effect."""
    assert fit.detail["did_placebo"].abs().max() < 5.0
    assert (fit.detail["ddd"] > 1.0).all()


# --------------------------------------------------------------------------
# Conditional DDD: covariates, all three nuisance combinations, analytic SEs.
# --------------------------------------------------------------------------


def _fit_conditional(df: pd.DataFrame, est_method: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ddd_heterogeneous(
            df,
            y="y",
            unit="id",
            time="time",
            cohort="state",
            subgroup="partition",
            n_boot=0,
            seed=0,
            weight_by="cohort",
            x=COVARS,
            est_method=est_method,
            se="analytic",
        )


@pytest.fixture(scope="module", params=sorted(R_CONDITIONAL))
def conditional(request, panel):
    return request.param, _fit_conditional(panel, request.param)


def test_conditional_cells_match_triplediff(conditional):
    method, fit = conditional
    detail = fit.detail.set_index(["cohort", "time"])
    for cell, (att, _se) in R_CONDITIONAL[method].items():
        got = float(detail.loc[cell, "ddd"])
        assert got == pytest.approx(
            att, abs=1e-9
        ), f"{method} {cell}: StatsPAI {got:.12f} vs triplediff {att:.12f}"


def test_conditional_cell_standard_errors_match_triplediff(conditional):
    """The analytic path is the reference's variance estimator, not a
    bootstrap standing in for it, so the SEs are held to the same bar as
    the point estimates."""
    method, fit = conditional
    detail = fit.detail.set_index(["cohort", "time"])
    for cell, (_att, se) in R_CONDITIONAL[method].items():
        got = float(detail.loc[cell, "se"])
        assert got == pytest.approx(
            se, abs=1e-9
        ), f"{method} {cell}: StatsPAI SE {got:.12f} vs triplediff {se:.12f}"


def test_conditional_aggregate_matches_triplediff(conditional):
    method, fit = conditional
    att, se = R_CONDITIONAL_AGG[method]
    assert fit.estimate == pytest.approx(att, abs=1e-9)
    assert fit.se == pytest.approx(se, abs=1e-9)


def test_conditioning_moves_the_estimate(panel, fit):
    """The fixture's covariates shift selection and the outcome path, so
    the conditional and unconditional answers differ by more than noise.
    If they did not, the covariate path would be untested in substance."""
    cond = _fit_conditional(panel, "dr")
    assert abs(cond.estimate - _fit(panel, weight_by="cohort").estimate) > 0.1


def test_no_covariates_collapses_the_three_methods(panel):
    """Without covariates the propensity score and the outcome regression
    are both constants, so dr / ipw / reg are the same estimator."""
    ests = [
        _fit(panel, weight_by="cohort").estimate,
    ]
    for method in ("dr", "ipw", "reg"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ests.append(
                sp.ddd_heterogeneous(
                    panel,
                    y="y",
                    unit="id",
                    time="time",
                    cohort="state",
                    subgroup="partition",
                    n_boot=0,
                    weight_by="cohort",
                    est_method=method,
                    se="analytic",
                ).estimate
            )
    assert max(ests) - min(ests) < 1e-9


def test_analytic_se_is_reported_as_such(panel):
    fit = _fit_conditional(panel, "dr")
    assert fit.model_info["se_method"] == "analytic"
    assert fit.model_info["est_method"] == "dr"
    assert fit.model_info["covariates"] == COVARS
    # The placebo joint test needs the placebo arms' joint covariance, which
    # the analytic path does not build. It says None rather than pretending.
    assert fit.model_info["placebo_joint_test"] is None


def test_bootstrap_stays_the_default_without_covariates(fit):
    assert fit.model_info["se_method"] == "bootstrap"


def test_unknown_est_method_fails_loudly(panel):
    with pytest.raises(ValueError, match="est_method"):
        _fit_conditional(panel, "aipw")


def test_missing_covariate_fails_loudly(panel):
    with pytest.raises(ValueError, match="Covariate"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.ddd_heterogeneous(
                panel,
                y="y",
                unit="id",
                time="time",
                cohort="state",
                subgroup="partition",
                n_boot=0,
                x=["not_a_column"],
            )


# --------------------------------------------------------------------------
# Not-yet-treated controls: partial parity, and the reason for the rest.
# --------------------------------------------------------------------------

# triplediff::ddd(control_group = "notyettreated"). Split by whether the
# reference's own influence-function indexing is well defined for the cell.
#
# The reference runs the DDD against each control cohort separately and
# combines them by minimum distance. It writes each cohort's influence
# function into the panel-length vector with a boolean index of the wrong
# length; R prints "number of items to replace is not a multiple of
# replacement length" on every call. Where the comparison happens to span
# the whole panel the boolean is full length and nothing goes wrong -- those
# cells agree with us exactly. Where it does not, the combined influence
# function picks up units that are in no comparison for that cell (on this
# fixture, all 150 units of a cohort that is neither treated nor a control),
# and that feeds the weights, the estimate and the SE.
R_NYT_AGREEING = {
    # Three control cohorts here, so the comparison covers all 600 units.
    (2, 2): (1.742374788449, 0.273305025090),
    # One control cohort left (not-yet-treated has collapsed to
    # never-treated), so there is nothing to combine.
    (2, 4): (2.631984079566, 0.493468900211),
    (3, 4): (2.647278119345, 0.351755941284),
    (4, 4): (2.014685182850, 0.296390462781),
}
# Cells where the reference's combination rests on misindexed influence
# functions. Our values, recorded so a change in ours is still visible.
SP_NYT_DIVERGENT = {
    (2, 3): 2.1213476381,
    (3, 3): 2.4231033165,
}


def _fit_nyt(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ddd_heterogeneous(
            df,
            y="y",
            unit="id",
            time="time",
            cohort="state",
            subgroup="partition",
            n_boot=0,
            weight_by="cohort",
            se="analytic",
            control_group="notyettreated",
        )


@pytest.fixture(scope="module")
def nyt_fit(panel):
    return _fit_nyt(panel)


@pytest.mark.parametrize("cell", sorted(R_NYT_AGREEING))
def test_not_yet_treated_agrees_where_the_reference_is_well_defined(nyt_fit, cell):
    att, se = R_NYT_AGREEING[cell]
    detail = nyt_fit.detail.set_index(["cohort", "time"])
    assert float(detail.loc[cell, "ddd"]) == pytest.approx(att, abs=1e-9)
    assert float(detail.loc[cell, "se"]) == pytest.approx(se, abs=1e-9)


@pytest.mark.parametrize("cell", sorted(SP_NYT_DIVERGENT))
def test_not_yet_treated_divergent_cells_are_pinned(nyt_fit, cell):
    """Pinned to OUR value, not the reference's, with the reason above."""
    detail = nyt_fit.detail.set_index(["cohort", "time"])
    assert float(detail.loc[cell, "ddd"]) == pytest.approx(
        SP_NYT_DIVERGENT[cell], abs=1e-9
    )


def test_per_control_cohort_estimates_match_the_reference(panel):
    """The divergence is in the combination, not the estimator.

    Restricting the panel to one control cohort at a time reproduces
    triplediff exactly -- which is what makes the combination the only
    place the two can differ.
    """
    r_per_cohort = {0: 2.109735960813, 4: 2.133731347533}
    for ctrl, expected in r_per_cohort.items():
        sub = panel[panel["state"].isin([2, ctrl])].copy()
        sub["state2"] = np.where(sub["state"] == 2, 2, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.ddd_heterogeneous(
                sub,
                y="y",
                unit="id",
                time="time",
                cohort="state2",
                subgroup="partition",
                n_boot=0,
                weight_by="cohort",
                se="analytic",
            )
        detail = fit.detail.set_index(["cohort", "time"])
        assert float(detail.loc[(2, 3), "ddd"]) == pytest.approx(expected, abs=1e-9)


def test_our_influence_functions_are_correctly_supported(nyt_fit):
    """The property the reference violates.

    Every cell's influence function must be supported on the units that
    actually enter that comparison. The aggregate mixes all cells, so what
    is asserted here is the consequence that survives aggregation: a
    mean-zero influence function. A vector polluted by units from an
    unrelated cohort does not integrate to zero.
    """
    psi = nyt_fit.model_info["influence_function"]
    assert np.isfinite(psi).all()
    assert abs(float(psi.mean())) < 1e-8


def test_never_and_not_yet_treated_differ(panel, nyt_fit):
    assert abs(nyt_fit.estimate - _fit(panel, weight_by="cohort").estimate) > 1e-3
    assert nyt_fit.model_info["control_group"] == "notyettreated"


def test_unknown_control_group_fails_loudly(panel):
    with pytest.raises(ValueError, match="control_group"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.ddd_heterogeneous(
                panel,
                y="y",
                unit="id",
                time="time",
                cohort="state",
                subgroup="partition",
                n_boot=0,
                control_group="notyet",
            )
