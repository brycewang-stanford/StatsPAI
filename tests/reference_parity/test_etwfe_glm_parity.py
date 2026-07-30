"""Reference parity: ``sp.etwfe(family=...)`` vs R ``etwfe`` 0.6.2.

Wooldridge (2023) nonlinear ETWFE. The estimator fits the saturated
Wooldridge/Mundlak design

    h(E[Y]) = a + sum_g gamma_g 1{G=g} + sum_t delta_t 1{T=t}
              + sum_{g, t>=g} beta_gt 1{G=g, T=t}

by maximum likelihood, then reports the **average marginal effect on the
response scale** over treated post-treatment observations — the same estimand
R ``etwfe::emfx(type='simple')`` returns. A Poisson fit therefore reports an
effect in counts, not log points; a logit fit reports a probability
difference, not a log-odds coefficient.

Fixtures
--------
Both panels were simulated in R and are committed alongside this module so the
reference numbers below stay reproducible without re-running R:

* ``_fixtures/etwfe_poisson_panel.csv`` — 300 units x 6 periods, cohorts
  {0, 3, 4, 5}, log-link DGP ``log(mu) = 1 + 0.05*t + u_i + 0.30*post``.
* ``_fixtures/etwfe_logit_panel.csv`` — 400 units x 6 periods, same cohort
  structure, ``logit(p) = -0.4 + 0.08*t + u_i + 0.70*post``.

Reference generation (R 4.5.2, etwfe 0.6.2, fixest 0.14.0)::

    m <- etwfe(fml = y ~ 1, tvar = year, gvar = g, data = d,
               vcov = ~id, family = "poisson")   # or "binomial"
    emfx(m, type = "simple")
    emfx(m, type = "event")

Point estimates are asserted to 1e-9. Standard errors carry a small
(~1e-5 relative) difference because ``fixest`` and ``statsmodels`` apply
different finite-sample corrections to the clustered sandwich, so those use a
relative tolerance.

References
----------
- Wooldridge, J.M. (2021). "Two-Way Fixed Effects, the Two-Way Mundlak
  Regression, and Difference-in-Differences Estimators." [@wooldridge2021two]
"""

from __future__ import annotations

import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIXTURES = pathlib.Path(__file__).parent / "_fixtures"

# --- R etwfe 0.6.2, family="poisson" ---------------------------------------
R_POISSON_SIMPLE = (1.2720480537, 0.2050045776)
R_POISSON_EVENT = {
    0: (1.28810400, 0.21143010),
    1: (1.17832900, 0.24445880),
    2: (1.38071700, 0.31961570),
    3: (1.30416200, 0.40881460),
}
# Same panel, family=NULL (linear ETWFE) — pins that family='gaussian' and the
# default are the historical path and did not move.
R_GAUSSIAN_SIMPLE = 1.2834720619

# --- R etwfe 0.6.2, family="binomial" --------------------------------------
R_LOGIT_SIMPLE = (0.2438009516, 0.0381933373)
R_LOGIT_EVENT = {
    0: (0.19780252, 0.03923641),
    1: (0.26436977, 0.04471341),
    2: (0.25802051, 0.05528479),
    3: (0.29385293, 0.07914499),
}


def _load(name: str) -> pd.DataFrame:
    path = _FIXTURES / name
    if not path.exists():  # pragma: no cover - fixture shipped with the repo
        pytest.skip(f"missing fixture: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def poisson_panel() -> pd.DataFrame:
    return _load("etwfe_poisson_panel.csv")


@pytest.fixture(scope="module")
def logit_panel() -> pd.DataFrame:
    return _load("etwfe_logit_panel.csv")


def _fit(df: pd.DataFrame, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.etwfe(df, y="y", group="id", time="year", first_treat="g", **kw)


# ===========================================================================
# Poisson
# ===========================================================================


def test_poisson_simple_ame_matches_r(poisson_panel):
    res = _fit(poisson_panel, family="poisson")
    att_r, se_r = R_POISSON_SIMPLE
    assert res.estimate == pytest.approx(att_r, abs=1e-9)
    assert res.se == pytest.approx(se_r, rel=1e-4)


@pytest.mark.parametrize("event_time", sorted(R_POISSON_EVENT))
def test_poisson_event_ame_matches_r(poisson_panel, event_time):
    res = _fit(poisson_panel, family="poisson")
    es = res.model_info["event_study"].set_index("relative_time")
    att_r, se_r = R_POISSON_EVENT[event_time]
    assert es.loc[event_time, "att"] == pytest.approx(att_r, abs=1e-6)
    assert es.loc[event_time, "se"] == pytest.approx(se_r, rel=1e-3)


def test_poisson_estimand_is_response_scale(poisson_panel):
    """The AME must be in counts, not log points.

    The DGP has a 0.30 log-point effect on a mean of roughly 4-5 counts, so a
    response-scale AME near 1.27 is right and a value near 0.30 would mean the
    link-scale coefficient leaked out as the headline.
    """
    res = _fit(poisson_panel, family="poisson")
    assert "response scale" in res.estimand
    assert res.estimate > 1.0


# ===========================================================================
# Logit
# ===========================================================================


def test_logit_simple_ame_matches_r(logit_panel):
    res = _fit(logit_panel, family="logit")
    att_r, se_r = R_LOGIT_SIMPLE
    assert res.estimate == pytest.approx(att_r, abs=1e-9)
    assert res.se == pytest.approx(se_r, rel=1e-4)


@pytest.mark.parametrize("event_time", sorted(R_LOGIT_EVENT))
def test_logit_event_ame_matches_r(logit_panel, event_time):
    res = _fit(logit_panel, family="logit")
    es = res.model_info["event_study"].set_index("relative_time")
    att_r, se_r = R_LOGIT_EVENT[event_time]
    assert es.loc[event_time, "att"] == pytest.approx(att_r, abs=1e-6)
    assert es.loc[event_time, "se"] == pytest.approx(se_r, rel=1e-3)


def test_logit_ame_is_a_probability_difference(logit_panel):
    res = _fit(logit_panel, family="logit")
    assert -1.0 < res.estimate < 1.0


# ===========================================================================
# The linear path must be untouched, and unsupported options must be loud
# ===========================================================================


def test_gaussian_and_default_are_the_historical_linear_path(poisson_panel):
    default = _fit(poisson_panel)
    gaussian = _fit(poisson_panel, family="gaussian")
    assert default.estimate == pytest.approx(R_GAUSSIAN_SIMPLE, abs=1e-8)
    assert gaussian.estimate == pytest.approx(default.estimate, rel=1e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"family": "poisson", "xvar": "lpop"},
        {"family": "poisson", "panel": False},
        {"family": "poisson", "cgroup": "nevertreated"},
        {"family": "probit"},
    ],
)
def test_unsupported_nonlinear_options_raise(poisson_panel, kwargs):
    """Options the nonlinear branch cannot honour must raise, never be dropped.

    Silently ignoring ``cgroup`` or ``xvar`` would change the estimand without
    telling the caller.
    """
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        _fit(poisson_panel, **kwargs)


def test_poisson_rejects_negative_outcomes(poisson_panel):
    from statspai.exceptions import MethodIncompatibility

    df = poisson_panel.copy()
    df.loc[df.index[0], "y"] = -1.0
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        _fit(df, family="poisson")


def test_logit_rejects_non_binary_outcomes(poisson_panel):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="0/1"):
        _fit(poisson_panel, family="logit")


# ===========================================================================
# etwfe_emfx must serve the nonlinear fit rather than reject it
# ===========================================================================


def test_emfx_simple_returns_the_fitted_ame(poisson_panel):
    res = _fit(poisson_panel, family="poisson")
    out = sp.etwfe_emfx(res, type="simple")
    assert out.estimate == pytest.approx(R_POISSON_SIMPLE[0], abs=1e-9)


@pytest.mark.parametrize(
    "agg_type,label", [("event", "relative_time"), ("group", "cohort")]
)
def test_emfx_event_and_group_expose_the_cell_tables(poisson_panel, agg_type, label):
    res = _fit(poisson_panel, family="poisson")
    out = sp.etwfe_emfx(res, type=agg_type)
    assert label in out.detail.columns
    assert len(out.detail) > 0
    assert out.se > 0


def test_emfx_calendar_raises_for_nonlinear_fits(poisson_panel):
    """The GLM branch does not retain the period index calendar needs."""
    from statspai.exceptions import MethodIncompatibility

    res = _fit(poisson_panel, family="poisson")
    with pytest.raises(MethodIncompatibility, match="calendar"):
        sp.etwfe_emfx(res, type="calendar")


def test_emfx_still_works_for_the_linear_path(poisson_panel):
    res = _fit(poisson_panel)
    out = sp.etwfe_emfx(res, type="event")
    assert out.detail is not None and len(out.detail) > 0
