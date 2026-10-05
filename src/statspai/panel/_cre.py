"""Correlated random effects (Mundlak / Chamberlain) with Stata's variance components.

linearmodels' ``RandomEffects`` divides its Swamy-Arora variance components
by degrees of freedom built from the *column count* of the design. The CRE
designs add unit means of the regressors; those columns are swept out by the
within transformation and duplicate the between regressors, so the column
count overstates the rank (``N - G - 4`` instead of ``N - G - 2`` for two
regressors). sigma_e, sigma_u and theta come out slightly wrong, and with
them the coefficients on the means and the constant.

Because the added means contribute nothing to either the within or the
between residuals, the correct theta is exactly the one from the random-
effects fit *without* them. This module takes that theta, applies
linearmodels' own quasi-demeaning to the full design, and solves it with
``PooledOLS`` (which reproduces ``RandomEffects``' coefficients and every
covariance type on the transformed data). The result is wrapped so it reads
like a ``RandomEffectsResults``; applied to a design without mean columns it
reproduces linearmodels' fit attribute for attribute
(``tests/test_panel_cre_theta.py``). Stata 18 ``xtreg ..., re`` with the
means added is the reference (``test_panel_ssc_stata_parity.py``).

The same miscount happens in a plain random-effects fit with a regressor
that does not vary within unit (an industry dummy, a listing indicator):
the within transformation sweeps it out, so the within regression has one
parameter fewer than the design has columns. ``variance_components`` counts
the rank instead, as Stata's ``xtreg, re`` and R's ``plm`` do, and
``fit_re`` applies the resulting theta.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class CREFit:
    """``RandomEffectsResults``-shaped view of the corrected CRE fit."""

    def __init__(self, pooled: Any, template: Any, extras: Dict[str, Any]):
        self._pooled = pooled
        self._template = template
        self.__dict__.update(extras)

    # Quantities of the corrected GLS fit.
    @property
    def params(self) -> pd.Series:
        return self._pooled.params

    @property
    def std_errors(self) -> pd.Series:
        return self._pooled.std_errors

    @property
    def cov(self) -> pd.DataFrame:
        return self._pooled.cov

    @property
    def pvalues(self) -> pd.Series:
        return self._pooled.pvalues

    @property
    def tstats(self) -> pd.Series:
        return self._pooled.tstats

    @property
    def df_resid(self) -> int:
        return int(self._pooled.df_resid)

    def __getattr__(self, name: str) -> Any:
        # Sample descriptors (nobs, entity_info, ...) are shared with the
        # uncorrected fit on the same design.
        return getattr(self._template, name)


def within_rank_deficient(exog: pd.DataFrame) -> bool:
    """True when some regressor (other than the constant) adds no rank to
    the within-unit variation: it is constant within unit, or collinear
    with the others once unit means are removed."""
    from .xt_tools import _scaled_rank

    x = exog.drop(columns=[c for c in exog.columns if c == "const"]).dropna()
    if x.shape[1] == 0 or len(x) == 0:
        return False
    xw = (x - x.groupby(level=0).transform("mean")).to_numpy(dtype=float)
    return _scaled_rank(xw, x.to_numpy(dtype=float)) < x.shape[1]


def variance_components(dep: pd.Series, exog: pd.DataFrame) -> tuple:
    """Swamy-Arora ``theta`` and variance decomposition of a random-effects
    design, in the layout of ``RandomEffectsResults``.

    When every regressor varies within unit these are linearmodels' own.
    Otherwise they are ``xt_tools._swamy_arora``'s: the within residual
    variance is divided by ``N - G - r`` with ``r`` the rank of the
    within-transformed regressors (the number of slopes a fixed-effects fit
    estimates), and the between residual variance by ``G`` minus the rank
    of the unit means. The unbalanced-panel form is the same in both: the
    harmonic mean of the panel lengths. ``dep`` and ``exog`` must be free
    of missing values.
    """
    from linearmodels.panel import RandomEffects

    if not within_rank_deficient(exog):
        plain = RandomEffects(dep, exog).fit()
        return plain.theta, plain.variance_decomposition

    from .xt_tools import _swamy_arora

    x = exog.drop(columns=[c for c in exog.columns if c == "const"])
    ids = dep.index.get_level_values(0)
    codes, units = pd.factorize(ids, sort=True)
    parts = _swamy_arora(
        dep.to_numpy(dtype=float), x.to_numpy(dtype=float), codes, len(units)
    )
    sigma2_e, sigma2_u, t = parts["s2e"], parts["s2u"], parts["Ti"]
    ybar = dep.groupby(level=0).mean().reindex(units)
    theta = pd.DataFrame(
        1.0 - np.sqrt(sigma2_e / (t * sigma2_u + sigma2_e)),
        index=ybar.index,
        columns=["theta"],
    )
    decomposition = pd.Series(
        {
            "Effects": sigma2_u,
            "Residual": sigma2_e,
            "Percent due to Effects": sigma2_u / (sigma2_u + sigma2_e),
        }
    )
    return theta, decomposition


def fit_re(dep: pd.Series, exog: pd.DataFrame, cov_kwargs: Dict[str, Any]) -> CREFit:
    """Random-effects GLS with the rank-based variance components."""
    return fit_cre(dep, exog, [], cov_kwargs)


def fit_cre(
    dep: pd.Series,
    exog: pd.DataFrame,
    mean_cols: list,
    cov_kwargs: Dict[str, Any],
) -> CREFit:
    """Random-effects GLS of ``dep`` on ``exog`` with theta from the fit
    that omits ``mean_cols`` (the regressors' unit means)."""
    from linearmodels.panel import PooledOLS, RandomEffects

    # unit means below must be taken over the estimation sample
    complete = dep.notna().to_numpy() & exog.notna().all(axis=1).to_numpy()
    if not complete.all():
        dep, exog = dep[complete], exog[complete]
    theta_frame, decomposition = variance_components(dep, exog.drop(columns=mean_cols))
    model = RandomEffects(dep, exog)
    template = model.fit(**cov_kwargs)

    theta = theta_frame["theta"]
    th = theta.reindex(dep.index.get_level_values(0)).to_numpy()
    ybar = dep.groupby(level=0).transform("mean")
    xbar = exog.groupby(level=0).transform("mean")
    wy = dep - th * ybar
    wx = exog - xbar.mul(th, axis=0)
    pooled = PooledOLS(wy, wx).fit(**cov_kwargs)

    params = pooled.params.to_numpy()
    weps = (wy - wx @ params).to_numpy()
    wy_a = wy.to_numpy()[:, None]
    wx_a = wx.to_numpy()
    root_w = np.ones_like(wy_a)
    r2o, r2w, r2b = model._rsquared(params[:, None])
    fitted = pd.DataFrame(
        exog.to_numpy() @ params, index=dep.index, columns=["fitted_values"]
    )
    resids = pd.Series(weps, index=dep.index, name="residual")
    wmu = wy_a.mean() if "const" in exog.columns else 0.0
    total_ss = float(((wy_a - wmu) ** 2).sum())
    extras = {
        "theta": theta_frame,
        "variance_decomposition": decomposition,
        "fitted_values": fitted,
        "resids": resids,
        "rsquared": 1.0 - float(weps @ weps) / total_ss,
        "rsquared_within": r2w,
        "rsquared_between": r2b,
        "rsquared_overall": r2o,
        "f_statistic": model._f_statistic(
            weps[:, None], wy_a, wx_a, root_w, pooled.df_resid
        ),
    }
    return CREFit(pooled, template, extras)
