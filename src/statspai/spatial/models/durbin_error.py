"""Spatial Durbin error (SDEM) and general nesting (GNS) models by ML.

SDEM:  ``y = X b + W X t + u,          u = lambda W u + e``
GNS:   ``y = rho W y + X b + W X t + u, u = lambda W u + e``

with ``e ~ N(0, sigma^2 I)``. ``(b, t, sigma^2)`` are profiled out of the
log-likelihood

    l = -n/2 log(2 pi sigma^2) + log|I - rho W| + log|I - lambda W|
        - e'e / (2 sigma^2),   e = (I - lambda W)((I - rho W) y - Z g)

and the spatial parameters maximise the concentrated function. Standard
errors invert the observed information of the full log-likelihood in
``(b, t, rho, lambda, sigma^2)`` -- Stata ``spregress, ml`` (its default
``vce(oim)``), which the estimates and standard errors equal to 1e-6 on
the fixture (``tests/reference_parity/test_spatial_durbin_error_Stata_parity.py``).
The Hessian is taken by central differences with Richardson
extrapolation; the log-determinants use the eigenvalues of ``W``, so
``n`` up to a few thousand.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from ...core.results import EconometricResults
from ...exceptions import MethodIncompatibility, NumericalInstability
from ._fiml import _fit
from .ml import ArrayOrW, _coerce_W, _make_results, _parse_formula

__all__ = ["sdem", "gns"]


def _durbin_model(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool,
    alpha: float,
    lag: bool,
) -> EconometricResults:
    y, X, dep_var, indep = _parse_formula(formula, data)
    n = len(y)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise MethodIncompatibility(
            "spatial ML: missing or infinite values; every unit enters W.",
            recovery_hint="Impute or drop units (and their rows / columns of W).",
        )
    M = _coerce_W(W, n_expected=n, row_normalize=row_normalize)
    Wd = M.toarray()
    Z = np.column_stack([X, Wd @ X[:, 1:]])
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise NumericalInstability(
            "spatial ML: X and WX are collinear.",
            recovery_hint="Drop regressors whose spatial lag is (nearly) constant.",
        )
    fit = _fit(y, Z, Wd, lag)
    names: List[str] = ["const", *indep, *(f"W_{v}" for v in indep)]
    if lag:
        names.append("rho")
    names.append("lambda")
    k = len(names)
    params = np.concatenate([fit["g"], [fit["rho"]] if lag else [], [fit["lam"]]])
    se = np.sqrt(np.clip(np.diag(fit["V"])[:k], 0.0, None))
    res = _make_results(
        alpha=alpha,
        model_type="gns" if lag else "sdem",
        spatial_param_name="rho,lambda" if lag else "lambda",
        spatial_param_value=fit["rho"] if lag else fit["lam"],
        var_names=names,
        params_vec=params,
        se_vec=se,
        sigma2=fit["sigma2"],
        resid=fit["resid_e"],
        fitted=y - fit["resid_u"],
        n=n,
        dep_var=dep_var,
        log_lik=fit["loglik"],
        extra_diag={
            "se_sigma2": float(np.sqrt(fit["V"][-1, -1])),
            "vce": "oim",
        },
        _w_sparse=M,
    )
    res.model_info["model_type"] = (
        "GNS (Spatial Lag + Durbin + Error)" if lag else "SDEM (Spatial Durbin Error)"
    )
    res.model_info["sigma2"] = float(fit["sigma2"])
    res.model_info["se_sigma2"] = float(np.sqrt(fit["V"][-1, -1]))
    res.model_info["loglik"] = float(fit["loglik"])
    res.model_info["vcov"] = pd.DataFrame(fit["V"][:k, :k], index=names, columns=names)
    return res


def sdem(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """Spatial Durbin error model: ``y = Xb + WXt + u, u = lambda W u + e``.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
        n x n spatial weights, rows in the order of ``data``.
    data : pd.DataFrame
    formula : str
        ``"y ~ x1 + x2"``; a constant is added, and the spatial lag of each
        regressor (not of the constant) enters as ``W_<name>``.
    row_normalize : bool, default True
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` ``[const, x..., W_x..., lambda]`` with OIM standard
        errors (Stata ``spregress, ml errorlag(W) ivarlag(W: x...)``);
        ``diagnostics['sigma2']`` and its SE ``se_sigma2``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "x": x})
    >>> list(sp.sdem(W, df, "y ~ x").params.index)
    ['const', 'x', 'W_x', 'lambda']
    """
    return _durbin_model(W, data, formula, row_normalize, alpha, lag=False)


def gns(
    W: ArrayOrW,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """General nesting spatial model (Manski-type): spatial lag of ``y``,
    of the regressors, and of the error.

    ``y = rho W y + Xb + WXt + u, u = lambda W u + e`` by maximum
    likelihood, OIM standard errors as Stata ``spregress, ml dvarlag(W)
    errorlag(W) ivarlag(W: x...)``. The model nests SAR, SEM, SDM, SDEM and
    SAC; it is weakly identified in small samples (``rho`` and ``lambda``
    trade off), so read the standard errors of the two before the point
    estimates.

    Parameters
    ----------
    W : ndarray, scipy.sparse matrix, or statspai.spatial.weights.W
    data : pd.DataFrame
    formula : str
    row_normalize : bool, default True
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` ``[const, x..., W_x..., rho, lambda]``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 40
    >>> W = np.zeros((n, n))
    >>> for i in range(n):
    ...     W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1.0
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "x": x})
    >>> list(sp.gns(W, df, "y ~ x").params.index)
    ['const', 'x', 'W_x', 'rho', 'lambda']
    """
    return _durbin_model(W, data, formula, row_normalize, alpha, lag=True)
