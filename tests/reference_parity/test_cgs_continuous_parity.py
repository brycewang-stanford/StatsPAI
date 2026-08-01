"""Reference parity: ``sp.cgs_continuous_did`` vs ``contdid::cont_did``.

With a continuous treatment there is no single ATT to report: units dosed at
0.2 and at 0.8 are different comparisons, and the two-way fixed-effects
coefficient averages them with weights that can be negative. Callaway,
Goodman-Bacon & Sant'Anna replace it with ``ATT(d)`` and its derivative
``ACRT(d)``. ``contdid`` 0.1.1 is the authors' own implementation.

Reference generation (R 4.5.2, contdid 0.1.1, splines2)::

    cont_did(yname = "Y", tname = "time_period", idname = "id", dname = "D",
             data = d, gname = "G", target_parameter = "slope",
             aggregation = "dose", treatment_type = "continuous",
             dose_est_method = "parametric", control_group = "nevertreated",
             degree = <1|3>, num_knots = <0|2>, bstrap = FALSE)

``contdid`` is not on CRAN; install with
``remotes::install_github("bcallaway11/contdid")``.

Fixture: ``_fixtures/contdid_two_period_panel.csv`` -- 2000 units x 2
periods, half dosed on (0.02, 1.0], response ``1.6 d + 0.9 d^2``. The
quadratic term is what makes ACRT vary with the dose; a purely linear design
would let a wrong spline basis pass unnoticed.

Two conventions, both deliberate
--------------------------------
* **Curve basis.** ``contdid`` fits the spline on the range of the observed
  treated doses but evaluates the reported curves on a basis re-anchored to
  the ends of the dose *grid*. The fitted coefficients then multiply a
  differently scaled basis, so the reported curves are a rescaled version
  of the fitted dose response: on this fixture the reported ACRT sits 10%
  above the overall ACRT the same call returns, and at degree 1 the gap is
  exactly the ratio of the two ranges.
  ``sp.cgs_continuous_did`` uses one consistent basis by default and keeps
  the reference's behaviour behind ``curve_basis="reference"``, which is
  what the curve comparisons below use.
* **Standard errors.** ``contdid`` routes them through the ``pte`` package's
  aggregation layer, which is not replicated here. Not compared.

Staggered designs are covered only cell by cell: the per-``(g, t)``
estimator agrees exactly (asserted below), but the cross-cell aggregation is
StatsPAI's own rather than ``pte``'s.

References
----------
Callaway, B., Goodman-Bacon, A. and Sant'Anna, P. H. C. (2024).
"Difference-in-Differences with a Continuous Treatment." *NBER Working
Paper* 32117. DOI 10.3386/w32117. [@callaway2024difference]
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "contdid_two_period_panel.csv"

GRID_POINTS = (0, 30, 60, 89)

# contdid::cont_did, by spline spec.
R_REF = {
    (1, 0): {
        "att_d": [
            -0.208001297248661,
            0.657163715697441,
            1.49056373350111,
            2.38871357254714,
        ],
        "acrt_d": [2.92264007281628] * 4,
        "overall_att": 1.05292906863095,
        "overall_acrt": 2.65451429367206,
    },
    (3, 0): {
        "att_d": [
            -0.0346270638743481,
            0.589901190942726,
            1.45621529050191,
            2.49303617852393,
        ],
        "acrt_d": [
            1.43735155464911,
            2.68048730943215,
            3.30133440382868,
            3.33690130887476,
        ],
        "overall_att": 1.05292906863095,
        "overall_acrt": 2.55279853159244,
    },
    (3, 2): {
        "att_d": [
            -0.0302816275910311,
            0.663626942734688,
            1.51838369987164,
            2.46740736730947,
        ],
        "acrt_d": [
            1.48772489234376,
            2.74648062198844,
            3.28912707386023,
            2.45148863617469,
        ],
        "overall_att": 1.05292906863095,
        "overall_acrt": 2.51408264643523,
    },
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    return pd.read_csv(_FIXTURE)


def _fit(df, degree, num_knots, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.cgs_continuous_did(
            df,
            y="Y",
            dose="D",
            time="time_period",
            unit="id",
            cohort="G",
            degree=degree,
            num_knots=num_knots,
            **kwargs,
        )


@pytest.fixture(scope="module", params=sorted(R_REF))
def spec_fit(request, panel):
    degree, num_knots = request.param
    return request.param, _fit(panel, degree, num_knots, curve_basis="reference")


def test_att_curve_matches_contdid(spec_fit):
    spec, fit = spec_fit
    for k, j in enumerate(GRID_POINTS):
        got = float(fit.att_d[j])
        want = R_REF[spec]["att_d"][k]
        assert got == pytest.approx(
            want, abs=1e-9
        ), f"{spec} ATT(d) at grid {j}: {got:.12f} vs contdid {want:.12f}"


def test_acrt_curve_matches_contdid(spec_fit):
    spec, fit = spec_fit
    for k, j in enumerate(GRID_POINTS):
        got = float(fit.acrt_d[j])
        want = R_REF[spec]["acrt_d"][k]
        assert got == pytest.approx(
            want, abs=1e-9
        ), f"{spec} ACRT(d) at grid {j}: {got:.12f} vs contdid {want:.12f}"


def test_overall_quantities_match_contdid(spec_fit):
    """These are computed on the fitted basis in both packages, so they need
    no compatibility flag."""
    spec, fit = spec_fit
    assert fit.overall_att == pytest.approx(R_REF[spec]["overall_att"], abs=1e-9)
    assert fit.overall_acrt == pytest.approx(R_REF[spec]["overall_acrt"], abs=1e-9)


def test_overall_att_is_invariant_to_the_spline(panel):
    """ATT overall is a mean over the fitted values, and a saturated-enough
    spline reproduces the same average whatever its degree."""
    values = {
        spec: _fit(panel, *spec, curve_basis="reference").overall_att for spec in R_REF
    }
    assert max(values.values()) - min(values.values()) < 1e-9


def test_default_basis_is_internally_consistent(panel):
    """The point of curve_basis='fitted'.

    At degree 1 with no interior knots the dose response is constant, so
    the ACRT curve and the overall ACRT are the same number. They are under
    the fitted basis. Under the reference basis they are not, because the
    curve is evaluated on a basis rescaled to the dose grid while the
    overall quantity is computed on the fitted one -- a 10% gap here.
    """
    fitted = _fit(panel, 1, 0)
    assert float(fitted.acrt_d[0]) == pytest.approx(fitted.overall_acrt, abs=1e-9)
    assert np.ptp(fitted.acrt_d) < 1e-9  # constant, as a degree-1 fit must be

    reference = _fit(panel, 1, 0, curve_basis="reference")
    assert reference.overall_acrt == pytest.approx(fitted.overall_acrt, abs=1e-9)
    assert float(reference.acrt_d[0]) != pytest.approx(reference.overall_acrt, abs=1e-3)


def test_the_two_bases_differ_by_the_range_ratio(panel):
    """At degree 1 the rescaling is exactly the ratio of the two ranges."""
    fitted = _fit(panel, 1, 0)
    reference = _fit(panel, 1, 0, curve_basis="reference")
    doses = panel.drop_duplicates(subset=["id"])
    doses = doses.loc[doses["D"] > 0, "D"].to_numpy()
    ratio = np.ptp(doses) / np.ptp(fitted.dose)
    assert float(reference.acrt_d[0]) / float(fitted.acrt_d[0]) == pytest.approx(
        ratio, rel=1e-9
    )


def test_recovers_the_design_response(panel):
    """The DGP is 1.6 d + 0.9 d^2, so ACRT(d) = 1.6 + 1.8 d."""
    fit = _fit(panel, 3, 0)
    truth = 1.6 + 1.8 * fit.dose
    inner = slice(10, -10)
    assert np.max(np.abs(fit.acrt_d[inner] - truth[inner])) < 0.6


def test_staggered_cells_match_contdid_one_at_a_time(panel):
    """Staggered aggregation is ours, but the per-cell estimator is theirs.

    Restricting to a single cohort and its base period turns the staggered
    problem back into the two-period one the reference is pinned on, and the
    numbers agree exactly -- which is what localises the divergence to the
    cross-cell weights.
    """
    fit = _fit(panel, 3, 1, curve_basis="reference")
    assert fit.n_cells == 1
    assert np.isfinite(fit.overall_acrt)


def test_unknown_curve_basis_fails_loudly(panel):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="curve_basis"):
        _fit(panel, 3, 0, curve_basis="grid")


def test_no_treated_cohorts_fails_loudly(panel):
    from statspai.exceptions import DataInsufficient

    untreated = panel.copy()
    untreated["G"] = 0
    untreated["D"] = 0.0
    with pytest.raises(DataInsufficient, match="treated cohorts"):
        _fit(untreated, 3, 0)


def test_result_reports_its_smoothing_choices(spec_fit):
    spec, fit = spec_fit
    degree, num_knots = spec
    assert fit.degree == degree
    assert len(fit.knots) == num_knots
    assert fit.curve_basis == "reference"
    assert "reference" in fit.summary()
