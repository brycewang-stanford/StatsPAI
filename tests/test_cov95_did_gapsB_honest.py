"""Coverage gaps (batch B) for statspai.did.honest_did.

Branches reached here: window / method-alias handling, the documented
worst-case-bias fallbacks used when a result carries an event-study table
but no joint covariance, the relative-magnitude breakdown value at its two
ends (0 and infinity), validation in ``event_study_from_moments``, and the
argument preparation of the R backend up to a failing ``Rscript``.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
import statspai.did.honest_did  # noqa: F401  (registers the submodule)
from statspai.core.results import CausalResult
from statspai.exceptions import ConvergenceFailure, MethodIncompatibility

H = sys.modules["statspai.did.honest_did"]

TIMES = [-3, -2, 0, 1, 2]
BETA = np.array([0.1, -0.05, 1.0, 1.2, 1.5])
SIGMA = np.diag([0.04, 0.03, 0.05, 0.06, 0.07]) + 0.01
Z = stats.norm.ppf(0.975)


@pytest.fixture(scope="module")
def with_cov():
    return H.event_study_from_moments(BETA, SIGMA, event_times=TIMES)


def _bare(es):
    """An event-study table with standard errors but no joint covariance."""
    return CausalResult(
        method="event study",
        estimand="ATT",
        estimate=1.0,
        se=0.2,
        pvalue=0.0,
        ci=(0.0, 1.0),
        alpha=0.05,
        n_obs=10,
        detail=None,
        model_info={"event_study": es},
    )


@pytest.fixture(scope="module")
def no_cov(with_cov):
    return _bare(with_cov.detail.copy())


# ----------------------------------------------------------------------
# window handling
# ----------------------------------------------------------------------
def test_window_filter_outside_every_event_time_gives_none():
    moments = (BETA, SIGMA, np.array(TIMES))
    assert H._window_filter(moments, (10, 20)) is None
    b, s, t = H._window_filter(moments, (-2, 1))
    assert t.tolist() == [-2, 0, 1] and s.shape == (3, 3)
    np.testing.assert_array_equal(b, BETA[1:4])


@pytest.mark.parametrize("window", ["ab", 5, (1, 2, 3)])
def test_window_must_be_a_pair(with_cov, window):
    with pytest.raises(MethodIncompatibility, match=r"window must be a \(lo, hi\)"):
        sp.honest_did(with_cov, e=0, window=window)


# ----------------------------------------------------------------------
# method spelling / honestdid_method
# ----------------------------------------------------------------------
def test_relative_magnitudes_plural_is_an_alias(with_cov):
    a = sp.honest_did(with_cov, e=0, method="relative_magnitudes", m_grid=[0, 0.5])
    b = sp.honest_did(with_cov, e=0, method="relative_magnitude", m_grid=[0, 0.5])
    pd.testing.assert_frame_equal(a, b)
    assert sp.breakdown_m(
        with_cov, e=0, method="relative_magnitudes"
    ) == sp.breakdown_m(with_cov, e=0, method="relative_magnitude")


def test_flci_is_not_a_relative_magnitude_method(with_cov):
    with pytest.raises(MethodIncompatibility, match="honestdid_method='FLCI' is not"):
        sp.honest_did(
            with_cov, e=0, method="relative_magnitude", honestdid_method="FLCI"
        )


# ----------------------------------------------------------------------
# fallbacks without a joint covariance
# ----------------------------------------------------------------------
def test_l_vec_without_covariance_uses_independent_se(no_cov):
    lv = np.array([0.5, 0.5, 0.0])
    theta = float(lv @ BETA[2:])
    se = float(np.sqrt(lv**2 @ np.diag(SIGMA)[2:]))
    with pytest.warns(UserWarning, match="event-study covariance"):
        out = sp.honest_did(no_cov, e=0, l_vec=lv, m_grid=[0.0])
    assert out.loc[0, "ci_lower"] == pytest.approx(theta - Z * se)
    assert out.loc[0, "ci_upper"] == pytest.approx(theta + Z * se)


def test_relative_magnitude_fallback_without_pre_periods(with_cov):
    post_only = _bare(with_cov.detail.query("relative_time >= 0"))
    with pytest.warns(UserWarning, match="worst-case-bias"):
        out = sp.honest_did(
            post_only, e=0, method="relative_magnitude", m_grid=[0.0, 0.5]
        )
    se0 = float(np.sqrt(SIGMA[2, 2]))
    assert out.loc[0, "ci_lower"] == pytest.approx(1.0 - Z * se0)
    assert out.loc[0, "ci_upper"] == pytest.approx(1.0 + Z * se0)
    # the bound is nested in Mbar even with no lead to calibrate on
    assert out.loc[1, "ci_lower"] <= out.loc[0, "ci_lower"]
    assert out.loc[1, "ci_upper"] >= out.loc[0, "ci_upper"]


def test_breakdown_relative_magnitude_fallback_formula(no_cov, with_cov):
    se0 = float(np.sqrt(SIGMA[2, 2]))
    with pytest.warns(UserWarning, match="worst-case-bias fallback"):
        got = sp.breakdown_m(no_cov, e=0, method="relative_magnitude")
    assert got == pytest.approx((1.0 - Z * se0) / 0.1)

    # with no lead at all the scale falls back to the SE itself
    post_only = _bare(with_cov.detail.query("relative_time >= 0"))
    with pytest.warns(UserWarning, match="worst-case-bias fallback"):
        got = sp.breakdown_m(post_only, e=0, method="relative_magnitude")
    assert got == pytest.approx((1.0 - Z * se0) / se0)


# ----------------------------------------------------------------------
# _rm_inputs / _breakdown_rm
# ----------------------------------------------------------------------
def test_rm_inputs_returns_none_when_target_is_unusable():
    moments = (BETA, SIGMA, np.array(TIMES))
    assert H._rm_inputs(moments, 7) is None  # e is not a post period
    assert H._rm_inputs((BETA[2:], SIGMA[2:, 2:], np.array([0, 1, 2])), 0) is None
    assert H._rm_inputs(moments, 0, np.array([1.0, 0.0])) is None  # wrong length
    b, s, n_pre, n_post, l_post = H._rm_inputs(moments, 1)
    assert (n_pre, n_post) == (2, 3) and l_post.tolist() == [0.0, 1.0, 0.0]


def test_breakdown_rm_is_zero_when_already_insignificant():
    weak = H.event_study_from_moments(
        np.array([0.1, -0.05, 0.01, 1.2, 1.5]), SIGMA, event_times=TIMES
    )
    assert sp.breakdown_m(weak, e=0, method="relative_magnitude") == 0.0


def test_breakdown_rm_is_infinite_when_leads_are_exactly_flat():
    # Leads of 1e-6 known to 1e-6: even Mbar = 1e4 allows a bias of ~0.02,
    # far short of a post coefficient of 3 with SE 0.22.
    flat = H.event_study_from_moments(
        np.array([1e-6, -1e-6, 3.0, 1.2, 1.5]),
        np.diag([1e-12, 1e-12, 0.05, 0.06, 0.07]),
        event_times=TIMES,
    )
    assert sp.breakdown_m(flat, e=0, method="relative_magnitude") == float("inf")


# ----------------------------------------------------------------------
# event_study_from_moments validation
# ----------------------------------------------------------------------
def test_moments_must_be_finite():
    with pytest.raises(MethodIncompatibility, match="must be finite"):
        H.event_study_from_moments(
            np.array([np.nan, 1.0]), np.eye(2), event_times=[-2, 0]
        )


@pytest.mark.parametrize("k", [0, 2])
def test_num_pre_periods_must_leave_a_post_period(k):
    with pytest.raises(MethodIncompatibility, match="at least one post period") as ei:
        H.event_study_from_moments(np.array([0.1, 1.0]), np.eye(2), num_pre_periods=k)
    assert ei.value.diagnostics == {"num_pre_periods": k, "n_beta": 2}


@pytest.mark.parametrize("times", [[0, 0], [-2, 0, 1]])
def test_event_times_must_be_distinct_and_aligned(times):
    with pytest.raises(MethodIncompatibility, match="one distinct time per coef"):
        H.event_study_from_moments(np.array([0.1, 1.0]), np.eye(2), event_times=times)


def test_fit_without_consistent_covariance_is_rejected():
    fit = types.SimpleNamespace(
        params=pd.Series({"lead2": 0.1, "lag0": 1.0}),
        std_errors=pd.Series({"lead2": 0.2, "lag0": 0.3}),
    )
    with pytest.raises(MethodIncompatibility, match="carries no covariance matrix"):
        H.event_study_from_moments(fit, None, event_times={"lead2": -2, "lag0": 0})


# ----------------------------------------------------------------------
# R backend
# ----------------------------------------------------------------------
def test_find_rscript_falls_back_to_standard_install_paths(monkeypatch):
    monkeypatch.setattr(H.shutil, "which", lambda name: None)
    monkeypatch.setattr(H.Path, "exists", lambda self: True)
    assert H._find_rscript() == "/Library/Frameworks/R.framework/Resources/bin/Rscript"
    monkeypatch.setattr(H.Path, "exists", lambda self: False)
    assert H._find_rscript() is None


@pytest.mark.parametrize("fixture", ["with_cov", "no_cov"])
def test_r_backend_reports_a_failing_rscript(monkeypatch, request, fixture):
    # Point the backend at an executable that cannot run R code: the windowed
    # l_vec target is prepared in Python first, then the non-zero exit status
    # must surface as ConvergenceFailure rather than as empty output.
    monkeypatch.setattr(H, "_find_rscript", lambda: sys.executable)
    res = request.getfixturevalue(fixture)
    with pytest.raises(ConvergenceFailure, match="failed while running the R") as ei:
        sp.honest_did(
            res, e=0, backend="honestdid", l_vec=[0.5, 0.5, 0.0], window=(-3, 2)
        )
    assert ei.value.diagnostics["returncode"] != 0


def test_r_backend_validates_l_vec_before_calling_r(monkeypatch, with_cov):
    monkeypatch.setattr(H, "_find_rscript", lambda: sys.executable)
    with pytest.raises(MethodIncompatibility, match="l_vec"):
        sp.honest_did(with_cov, e=0, backend="honestdid", l_vec=[1.0, 0.0])
