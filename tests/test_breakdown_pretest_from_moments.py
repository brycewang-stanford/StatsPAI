"""``sp.breakdown_m`` and ``sp.pretrends_test`` from coefficients and covariance.

The published-numbers workflow: a paper prints event-study coefficients and
standard errors, and the sensitivity analysis runs on them (``sigma=``), as
``sp.pretrends_power`` / ``sp.honest_did_from_moments`` already allow.
"""

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# Event-study coefficients (k = -7..3, reference -1) as printed in a published
# staggered-DiD paper's Stata log: standard errors only, so the covariance is
# taken as diagonal.
ET = [-7, -6, -5, -4, -3, -2, 0, 1, 2, 3]
B = np.array(
    [0.0008649, 0.0004272, -0.003648, -0.0013166, -0.000964, -0.0001196]
    + [0.0046343, 0.0124709, 0.0161297, 0.0322956]
)
SE = np.array(
    [0.0043195, 0.0040154, 0.0039911, 0.0037085, 0.004026, 0.003477]
    + [0.0022429, 0.0041571, 0.0079063, 0.016128]
)
V = np.diag(SE**2)
L = np.full(4, 0.25)


@pytest.fixture(scope="module")
def mpdta_es():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(
            sp.datasets.mpdta(), y="lemp", g="first_treat", t="year", i="countyreal"
        )
        es = sp.aggte(cs, type="dynamic")
    vc = sp.event_study_vcov(es)
    return es, np.asarray(vc.beta), np.asarray(vc.vcov), np.asarray(vc.times)


def test_pretrends_test_from_moments_is_the_wald_statistic():
    out = sp.pretrends_test(B, sigma=V, event_times=ET)
    pre = B[:6]
    assert out["type"] == "wald" and out["df"] == 6
    assert out["statistic"] == pytest.approx(
        float(pre @ np.linalg.solve(V[:6, :6], pre)), rel=1e-14
    )
    assert out["statistic"] == pytest.approx(1.0714264506307303, rel=1e-12)
    assert out["pre_periods"] == [-7, -6, -5, -4, -3, -2]


def test_pretrends_test_from_moments_refuses_f():
    with pytest.raises(MethodIncompatibility, match="degrees of freedom"):
        sp.pretrends_test(B, type="f", sigma=V, event_times=ET)


def test_pretrends_test_moments_equal_fitted(mpdta_es):
    es, b, S, t = mpdta_es
    fitted = sp.pretrends_test(es)
    moments = sp.pretrends_test(b, sigma=S, event_times=t)
    assert moments["statistic"] == pytest.approx(fitted["statistic"], rel=1e-12)
    assert moments["pvalue"] == pytest.approx(fitted["pvalue"], rel=1e-12)


@pytest.mark.parametrize("method", ["smoothness", "relative_magnitude"])
@pytest.mark.parametrize("l_vec", [None, "average"])
def test_breakdown_moments_equal_fitted(mpdta_es, method, l_vec):
    es, b, S, t = mpdta_es
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = sp.breakdown_m(es, e=0, method=method, l_vec=l_vec)
        moments = sp.breakdown_m(
            b, sigma=S, event_times=t, e=0, method=method, l_vec=l_vec
        )
    assert moments == pytest.approx(fitted, rel=1e-12)


@pytest.mark.parametrize(
    "method, stata_excludes, stata_includes",
    [
        # Stata honestdid (Rambachan-Roth package), same b / V / l_vec: lb
        # 3.9e-5 at Mbar 0.3679, -1.6e-4 at 0.3716 (C-LF); lb 2.1e-7 at
        # M 0.000957, -7.1e-5 at 0.000967 (FLCI).
        ("relative_magnitude", 0.3679, 0.3716),
        ("smoothness", 0.000957, 0.000967),
    ],
)
def test_breakdown_of_published_coefficients(method, stata_excludes, stata_includes):
    m_star = sp.breakdown_m(B, method=method, l_vec=L, sigma=V, event_times=ET)
    assert stata_excludes <= m_star <= stata_includes
    # it is the crossing of the interval honest_did_from_moments reports
    grid = sp.honest_did_from_moments(
        B, V, event_times=ET, method=method, l_vec=L, m_grid=[0.99 * m_star]
    )
    assert bool(grid["rejects_zero"].iloc[0])
    grid = sp.honest_did_from_moments(
        B, V, event_times=ET, method=method, l_vec=L, m_grid=[1.01 * m_star]
    )
    assert not bool(grid["rejects_zero"].iloc[0])
    stored = sp.honest_did_from_moments(
        B, V, event_times=ET, method=method, l_vec=L, m_grid=[0.0], breakdown=True
    ).attrs["breakdown_m"]
    assert m_star == pytest.approx(stored, rel=1e-12)


def test_breakdown_from_moments_needs_sigma():
    with pytest.raises(MethodIncompatibility, match="sigma"):
        sp.breakdown_m(B, event_times=ET)
