"""Stata parity for local linear regression matching and the Mahalanobis metric.

Local linear regression (LLR) matching (Heckman, Ichimura & Todd 1997) fits a
degree-1 local regression of the control outcome on the propensity gap and
reads off its intercept at the treated unit's own score.  With kernel weights
``K`` normalised to sum 1 and signed gaps ``d_j = p_j - p_i``,

.. math::

    w_{ij} = K_j \\frac{V + \\bar{d}^2 - \\bar{d} d_j}{V}, \\quad
    \\bar{d} = \\sum_j K_j d_j, \\quad
    V = \\sum_j K_j d_j^2 - \\bar{d}^2

which satisfies :math:`\\sum_j w_{ij} = 1` and :math:`\\sum_j w_{ij} d_j = 0`
— the local-linear property.  This reproduces psmatch2's ``_Match_llr``.

Two Stata behaviours pinned here
--------------------------------
1. **psmatch2 reports no analytic SE for LLR** (``seatt = .``).  Local linear
   weights can be negative, which the analytic formula assumes away.  So only
   the point estimate is alignable; StatsPAI defaults to a bootstrap SE, which
   is strictly more than Stata offers.

2. **``psmatch2 ..., llr`` with the default kernel is not LLR.**  For
   ``kerneltype(epan)`` with a propensity metric, psmatch2.ado rewrites the
   request as nearest-neighbour matching on an ``lpoly``-smoothed outcome and
   reports a non-missing SE.  Since ``epan`` is psmatch2's *default* kernel
   for ``llr``, a plain ``psmatch2 ..., llr`` never reaches its own LLR
   routine.  StatsPAI runs genuine LLR and warns about the divergence.

Fixture provenance
------------------
``_fixtures/_generate_psmatch2_llr.do`` under Stata 18 MP + psmatch2 4.0.12.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"
_REGEN = (
    "Regenerate with tests/reference_parity/_fixtures/"
    "_generate_psmatch2_llr.do under Stata 18 + psmatch2."
)

# The propensity score itself agrees with Stata to ~3e-12 (different logit
# solvers); that error propagates into the kernel weights, so 1e-8 relative
# on the ATT is a tight bound rather than a loose one.
_RTOL = 1e-8

_BWIDTH = 0.5


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    path = _FIXTURE_DIR / "psmatch2_llr_data.csv"
    if not path.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def stata() -> dict:
    path = _FIXTURE_DIR / "psmatch2_llr_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return json.loads(path.read_text(encoding="utf-8"))


def _llr(data, kernel, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(
            data,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="llr",
            kernel=kernel,
            bwidth=_BWIDTH,
            se_method=kwargs.pop("se_method", "ai"),
            **kwargs,
        )


class TestLLRPointEstimate:
    @pytest.mark.parametrize("kernel", ["tricube", "biweight", "normal", "uniform"])
    def test_att_matches_stata(self, data, stata, kernel):
        ours = _llr(data, kernel).estimate
        theirs = stata["llr"]["att_by_kernel"][kernel]
        assert ours == pytest.approx(theirs, rel=_RTOL)

    def test_kernels_give_genuinely_different_answers(self, data, stata):
        """Guard against the kernel argument being silently ignored."""
        atts = {k: _llr(data, k).estimate for k in ["tricube", "normal", "uniform"]}
        assert len(set(np.round(list(atts.values()), 10))) == 3

    def test_llr_differs_from_plain_kernel_matching(self, data, stata):
        """The local-linear correction must actually change the estimate."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kern = sp.match(
                data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="kernel",
                kernel="tricube",
                bwidth=_BWIDTH,
            )
        assert kern.estimate == pytest.approx(stata["kernel_tricube"]["att"], rel=_RTOL)
        assert abs(kern.estimate - _llr(data, "tricube").estimate) > 0.1


class TestLLRMatchedFrame:
    """The emitted `_weight` / `_y` must be Stata's, row for row."""

    @pytest.fixture(scope="class")
    def fitted(self, data):
        return _llr(data, "tricube")

    def test_weight_matches_stata_row_for_row(self, fitted, data):
        ours = fitted.matched_data["_weight"].to_numpy(dtype=float)
        theirs = data["_weight"].to_numpy(dtype=float)
        assert np.array_equal(np.isfinite(ours), np.isfinite(theirs))
        both = np.isfinite(ours)
        np.testing.assert_allclose(ours[both], theirs[both], rtol=1e-7, atol=1e-9)

    def test_matched_outcome_matches_stata_row_for_row(self, fitted, data):
        ours = fitted.matched_data["_y"].to_numpy(dtype=float)
        theirs = data["_y"].to_numpy(dtype=float)
        both = np.isfinite(ours) & np.isfinite(theirs)
        np.testing.assert_allclose(ours[both], theirs[both], rtol=1e-7, atol=1e-9)

    def test_local_linear_weights_sum_to_one_per_treated_unit(self, fitted, data):
        """sum_j w_ij = 1 is what makes the intercept a conditional mean."""
        ours = fitted.matched_data["_weight"].to_numpy(dtype=float)
        treated = fitted.matched_data["_treated"].to_numpy(dtype=float)
        n_treated_matched = int(np.sum(np.isfinite(ours) & (treated == 1)))
        control_total = float(np.nansum(ours[treated == 0]))
        assert control_total == pytest.approx(n_treated_matched, rel=1e-9)

    def test_att_equals_mean_of_y_minus_matched_y(self, fitted):
        md = fitted.matched_data
        t = (md["_treated"] == 1) & md["_y"].notna()
        att = float((md.loc[t, "y"] - md.loc[t, "_y"]).mean())
        assert att == pytest.approx(fitted.estimate, abs=1e-12)


class TestEpanReroute:
    """psmatch2's `llr` default is not LLR; we diverge deliberately and loudly."""

    def test_warns_that_stata_would_not_run_llr(self, data):
        with pytest.warns(UserWarning, match="does not perform local linear"):
            sp.match(
                data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="llr",
                kernel="epan",
                bwidth=_BWIDTH,
                se_method="ai",
            )

    def test_our_epan_llr_differs_from_statas_rerouted_number(self, data, stata):
        """Document the size of the divergence rather than hide it."""
        ours = _llr(data, "epan").estimate
        theirs = stata["llr_epan_reroute"]["att"]
        # Stata's number comes from a different estimator entirely.
        assert abs(ours - theirs) > 1e-3

    def test_stata_reports_an_se_only_for_the_rerouted_path(self, stata):
        assert stata["llr"]["seatt"] is None
        assert stata["llr_epan_reroute"]["seatt"] is not None


class TestLLRStandardError:
    def test_analytic_psmatch2_se_is_refused(self, data):
        with pytest.raises(MethodIncompatibility, match="not defined for"):
            sp.match(
                data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="llr",
                kernel="tricube",
                bwidth=_BWIDTH,
                se_method="psmatch2",
            )

    def test_default_se_method_is_bootstrap(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.match(
                data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="llr",
                kernel="tricube",
                bwidth=_BWIDTH,
                bootstrap_reps=40,
                bootstrap_seed=1,
            )
        assert r.model_info["se_method"] == "bootstrap"
        assert np.isfinite(r.se) and r.se > 0

    def test_bootstrap_is_reproducible_under_a_seed(self, data):
        def run(seed):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return sp.match(
                    data,
                    y="y",
                    treat="d",
                    covariates=["x1", "x2"],
                    method="llr",
                    kernel="tricube",
                    bwidth=_BWIDTH,
                    bootstrap_reps=40,
                    bootstrap_seed=seed,
                ).se

        assert run(11) == run(11)
        assert run(11) != run(12)

    def test_bootstrap_reports_its_own_diagnostics(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.match(
                data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="llr",
                kernel="tricube",
                bwidth=_BWIDTH,
                bootstrap_reps=30,
                bootstrap_seed=5,
            )
        info = r.model_info
        assert info["bootstrap_reps_successful"] + info["bootstrap_reps_failed"] == 30
        assert np.isfinite(info["bootstrap_bias"])

    def test_bootstrap_warns_for_nearest_neighbour(self, data):
        """Abadie & Imbens (2008): the bootstrap fails for k-NN matching."""
        with pytest.warns(UserWarning, match="not generally"):
            sp.match(
                data,
                y="y",
                treat="d",
                covariates=["x1", "x2"],
                method="nearest",
                se_method="bootstrap",
                bootstrap_reps=20,
                bootstrap_seed=2,
            )


class TestMahalanobisFrontDoor:
    def test_psmatch2_mahalanobis_matches_stata(self, data, stata):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sp.psmatch2(
                data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="mahalanobis",
                neighbor=1,
            )
        assert m.att == pytest.approx(stata["mahalanobis"]["att"], rel=1e-6)

    def test_mahalanobis_actually_uses_the_mahalanobis_metric(self, data):
        """Regression guard: an explicit distance= used to override the alias."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sp.psmatch2(
                data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="mahalanobis",
            )
        assert m.result.model_info["distance"] == "mahalanobis"

    def test_explicit_distance_still_wins(self, data):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sp.psmatch2(
                data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="mahalanobis",
                distance="propensity",
            )
        assert m.result.model_info["distance"] == "propensity"


class TestSplineIsRefusedHonestly:
    def test_spline_raises_with_an_explanation(self, data):
        with pytest.raises(MethodIncompatibility, match="not implemented"):
            sp.psmatch2(
                data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="spline",
            )

    def test_spline_error_points_at_llr(self, data):
        with pytest.raises(MethodIncompatibility) as exc:
            sp.psmatch2(
                data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="spline",
            )
        assert "llr" in exc.value.recovery_hint


class TestPsmatch2LLRFrontDoor:
    def test_llr_reaches_the_same_estimate_as_sp_match(self, data, stata):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sp.psmatch2(
                data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="llr",
                kernel="tricube",
                bwidth=_BWIDTH,
                bootstrap_reps=20,
                bootstrap_seed=1,
            )
        assert m.att == pytest.approx(
            stata["llr"]["att_by_kernel"]["tricube"], rel=_RTOL
        )

    def test_llr_default_se_falls_back_to_bootstrap_not_an_error(self, data):
        """se='psmatch2' is this function's default; llr must not blow up."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sp.psmatch2(
                data,
                treat="d",
                outcome="y",
                covariates=["x1", "x2"],
                method="llr",
                kernel="tricube",
                bwidth=_BWIDTH,
                bootstrap_reps=20,
                bootstrap_seed=1,
            )
        assert np.isfinite(m.se)
