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
varying across cohorts (0.70 / 0.50 / 0.35).

Two conventions differ and both are pinned:

* **Aggregation weights.** ``triplediff``'s ``agg_ddd(type="simple")``
  weights cohort ``g`` by ``mean(first_treat == g)`` over *all* units.
  ``sp.ddd_heterogeneous`` defaults to weighting by treated units in the
  affected subgroup. On this fixture that is 2.369 versus 2.455 -- which is
  why the affected share varies across cohorts here, so the gap cannot pass
  as rounding. ``weight_by="cohort"`` reproduces the package exactly.
* **Standard errors.** ``triplediff`` reports analytical influence-function
  SEs; this function has only a cluster bootstrap. Not compared.

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

import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "ddd_staggered_panel.csv"

# triplediff::ddd, post-treatment cells only.
R_CELLS = {
    (2, 2): 1.8775691954,
    (2, 3): 2.8069089417,
    (2, 4): 3.1709051076,
    (3, 3): 1.9670077796,
    (3, 4): 2.7291233854,
    (4, 4): 1.6626219554,
}
R_SIMPLE_ATT = 2.3690227275
SP_ELIGIBLE_ATT = 2.4548929790


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
    assert abs(_fit(panel, "cohort").estimate - _fit(panel, "eligible").estimate) > 0.05


def test_unknown_weight_by_fails_loudly(panel):
    with pytest.raises(ValueError, match="weight_by"):
        _fit(panel, weight_by="population")


def test_placebo_arm_is_near_zero(fit):
    """The DGP gives the affected subgroup its own linear trend, which the
    DDD nets out; the placebo DIDs carry that trend, not the effect."""
    assert fit.detail["did_placebo"].abs().max() < 5.0
    assert (fit.detail["ddd"] > 1.0).all()
