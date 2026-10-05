"""
Conformal prediction intervals for a linear regression.

A prediction interval from ``fit.predict(new, what='prediction')`` is exact
when the errors are normal and homoskedastic and can be badly off when they
are not. The intervals here need only that the observations are
exchangeable: whatever the error distribution, and whether or not the
linear model is right, they cover a new outcome with probability at least
``1 - alpha`` (``1 - 2 alpha`` for the jackknife+).

Three constructions, all for least squares, all computed without refitting
in a loop:

* ``"split"`` -- fit on one part of the sample, take a quantile of the
  absolute residuals on the other [@lei2018distribution].
* ``"jackknife+"`` -- every observation's leave-one-out fit and
  leave-one-out residual, from the hat matrix [@barber2021predictive].
* ``"full"`` -- the set of outcome values for the new point that would not
  make it look unusual next to the rest once the model is refitted with
  it. For least squares the residuals are linear in the candidate value,
  so the set is found exactly [@lei2018distribution].
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

_METHODS = {
    "split": "split",
    "jackknife+": "jackknife+",
    "jackknife_plus": "jackknife+",
    "jackknifeplus": "jackknife+",
    "full": "full",
}


def _full_interval(
    X: np.ndarray, y: np.ndarray, x0: np.ndarray, alpha: float
) -> Tuple[float, float]:
    """Exact full-conformal interval for OLS at one new point.

    With the new point appended, the residuals of the refit are ``a + b y``
    in the candidate outcome ``y``. A candidate is kept when the share of
    observations whose absolute residual is at least the new point's
    exceeds ``alpha``. That count only changes where ``|a_i + b_i y| =
    |a_new + b_new y|``, two values of ``y`` per observation, so it is
    evaluated once between each pair of neighbouring crossings.
    """
    n = X.shape[0]
    Xa = np.vstack([X, x0])
    G = np.linalg.pinv(Xa.T @ Xa)
    Hc = Xa @ (G @ x0)  # last column of the hat matrix
    ya = np.append(y, 0.0)
    a = ya - Xa @ (G @ (Xa.T @ ya))
    b = -Hc
    b[-1] += 1.0
    a_i, b_i, a_n, b_n = a[:-1], b[:-1], a[-1], b[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = np.concatenate([(a_n - a_i) / (b_i - b_n), -(a_n + a_i) / (b_i + b_n)])
    cross = np.unique(cross[np.isfinite(cross)])
    if cross.size == 0:
        return -np.inf, np.inf
    pad = max(1.0, float(np.ptp(cross)))
    mids = np.concatenate(
        [[cross[0] - pad], (cross[:-1] + cross[1:]) / 2.0, [cross[-1] + pad]]
    )

    def kept(points: np.ndarray) -> np.ndarray:
        own = np.abs(a_n + b_n * points)
        others = np.abs(a_i[:, None] + b_i[:, None] * points[None, :])
        # the tolerance makes a crossing count as "at least", on both sides
        count = np.sum(others >= own[None, :] * (1.0 - 1e-12) - 1e-300, axis=0)
        return np.asarray((count + 1.0) > alpha * (n + 1.0), dtype=bool)

    inside_mid = kept(mids)
    inside_cross = kept(cross)
    # the kept set is a union of closed pieces; report its hull
    ends = list(cross[inside_cross])
    for j in np.flatnonzero(inside_mid[1:-1]):
        ends.extend([cross[j], cross[j + 1]])
    lower = -np.inf if inside_mid[0] else (min(ends) if ends else np.nan)
    upper = np.inf if inside_mid[-1] else (max(ends) if ends else np.nan)
    return float(lower), float(upper)


def _jackknife_plus(
    X: np.ndarray, y: np.ndarray, X0: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    G = np.linalg.pinv(X.T @ X)
    beta = G @ (X.T @ y)
    h = np.einsum("ij,jk,ik->i", X, G, X)
    if np.any(h >= 1.0 - 1e-10):
        raise DataInsufficient(
            "conformal_regression: an observation has leverage one; its "
            "leave-one-out fit does not exist.",
            recovery_hint="Drop the regressor that singles it out, or use "
            "method='split'.",
        )
    loo = (y - X @ beta) / (1.0 - h)
    # prediction at each new point from the fit without observation i:
    # x0'b - x0'(X'X)^-1 x_i e_i / (1 - h_i)
    pred = (X0 @ beta)[:, None] - (X0 @ G @ X.T) * loo[None, :]
    lo_all = np.sort(pred - np.abs(loo)[None, :], axis=1)
    hi_all = np.sort(pred + np.abs(loo)[None, :], axis=1)
    k_lo = int(np.floor(alpha * (n + 1)))
    k_hi = int(np.ceil((1.0 - alpha) * (n + 1)))
    lower = lo_all[:, k_lo - 1] if k_lo >= 1 else np.full(len(X0), -np.inf)
    upper = hi_all[:, k_hi - 1] if k_hi <= n else np.full(len(X0), np.inf)
    return lower, upper


def conformal_regression(
    formula: str,
    data: pd.DataFrame,
    newdata: Optional[pd.DataFrame] = None,
    method: str = "jackknife+",
    alpha: float = 0.1,
    train_frac: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Distribution-free prediction intervals for a least-squares regression.

    Returns, for each row of ``newdata``, the least-squares prediction and
    an interval that contains the outcome of a new, exchangeable
    observation with probability at least ``1 - alpha``, without assuming
    normal or homoskedastic errors or a correctly specified mean.

    Parameters
    ----------
    formula : str
        Regression formula, e.g. ``"medv ~ rm + lstat"``.
    data : pd.DataFrame
        The sample the model is fitted on.
    newdata : pd.DataFrame, optional
        Rows to predict. When omitted, each row of ``data`` is predicted
        from all the others (its own outcome plays no part in its
        interval), which is the way to check coverage on a sample.
    method : {"jackknife+", "split", "full"}, default "jackknife+"
        ``"split"`` fits on a random ``train_frac`` of the sample and
        calibrates on the rest: the cheapest, exact ``1 - alpha``
        coverage, wider intervals because each step sees part of the
        data. ``"jackknife+"`` uses every observation for both: coverage
        at least ``1 - 2 alpha`` in the worst case and close to ``1 -
        alpha`` in practice. ``"full"`` is the exact conformal set for
        least squares: coverage at least ``1 - alpha``, the most work per
        new point (it grows with the square of the sample size).
    alpha : float, default 0.1
        One minus the target coverage.
    train_frac : float, default 0.5
        Share of the sample used for fitting under ``method="split"``.
    seed : int, optional
        Seed of the random split (``method="split"`` only).

    Returns
    -------
    pd.DataFrame
        Columns ``yhat``, ``lower``, ``upper``, indexed like ``newdata``
        (or ``data``). ``attrs`` records ``method``, ``alpha`` and ``n``.
        An interval end is infinite when the sample is too small for the
        requested level.

    Notes
    -----
    The guarantee is marginal: it holds on average over new points, not at
    each value of the regressors. With heteroskedastic errors the
    intervals have the right coverage overall and are too wide where the
    noise is small and too narrow where it is large. They cover an
    outcome, not a mean or a coefficient, and say nothing about causal
    effects; for those see ``sp.conformal("cate", ...)``.

    Exchangeability fails for time series and for clustered samples whose
    new point comes from a new cluster.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(-2, 2, size=400)
    >>> df = pd.DataFrame({"x": x, "y": 1 + x + rng.standard_t(3, size=400)})
    >>> new = pd.DataFrame({"x": [0.0, 1.0]})
    >>> out = sp.conformal_regression("y ~ x", df, new, alpha=0.1)
    >>> list(out.columns)
    ['yhat', 'lower', 'upper']
    >>> bool((out["lower"] < out["yhat"]).all() and (out["yhat"] < out["upper"]).all())
    True

    References
    ----------
    [@lei2018distribution]
    [@barber2021predictive]
    """
    key = _METHODS.get(str(method).lower())
    if key is None:
        raise MethodIncompatibility(
            f"conformal_regression: method={method!r} is not 'jackknife+', "
            "'split' or 'full'.",
            diagnostics={"method": method},
        )
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            f"conformal_regression: alpha must be in (0, 1), got {alpha}."
        )
    from patsy import build_design_matrices

    from ..core.utils import create_design_matrices

    y_df, X_df = create_design_matrices(formula, data)
    design_info = getattr(X_df, "design_info", None)
    y = np.asarray(y_df, dtype=float).reshape(len(X_df), -1)[:, -1]
    X = np.asarray(X_df, dtype=float)
    n, p = X.shape
    if n <= p + 1:
        raise DataInsufficient(
            f"conformal_regression: {n} complete rows for {p} coefficients."
        )

    def design_of(frame: pd.DataFrame) -> np.ndarray:
        if design_info is not None:
            built = build_design_matrices([design_info], frame)[0]
            return np.asarray(built, dtype=float)
        cols = [c for c in X_df.columns if c != "Intercept"]
        Z = frame[cols].to_numpy(dtype=float)
        if "Intercept" in X_df.columns:
            Z = np.column_stack([np.ones(len(Z)), Z])
        return np.asarray(Z, dtype=float)

    in_sample = newdata is None
    if newdata is None:
        X0 = X
        index = X_df.index
    else:
        X0 = design_of(newdata)
        index = newdata.index
        if X0.shape[0] != len(newdata):
            raise MethodIncompatibility(
                "conformal_regression: newdata has rows with missing " "regressors.",
                recovery_hint="Drop or fill them before predicting.",
            )
    m = X0.shape[0]
    lower = np.empty(m)
    upper = np.empty(m)

    def fit_predict(Xt: np.ndarray, yt: np.ndarray, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z @ np.linalg.lstsq(Xt, yt, rcond=None)[0], dtype=float)

    if key == "split":
        if not 0.0 < train_frac < 1.0:
            raise MethodIncompatibility(
                "conformal_regression: train_frac must be in (0, 1)."
            )
        rng = np.random.default_rng(seed)

        def split_interval(
            Xs: np.ndarray, ys: np.ndarray, Z: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            ns = len(ys)
            perm = rng.permutation(ns)
            n_fit = int(round(train_frac * ns))
            if n_fit <= p or ns - n_fit < 1:
                raise DataInsufficient(
                    "conformal_regression: the split leaves too few rows to "
                    "fit or to calibrate."
                )
            fit_rows, cal_rows = perm[:n_fit], perm[n_fit:]
            beta = np.linalg.lstsq(Xs[fit_rows], ys[fit_rows], rcond=None)[0]
            scores = np.sort(np.abs(ys[cal_rows] - Xs[cal_rows] @ beta))
            k = int(np.ceil((1.0 - alpha) * (len(scores) + 1)))
            q = scores[k - 1] if k <= len(scores) else np.inf
            centre = Z @ beta
            return centre, centre - q, centre + q

        if in_sample:
            # one split serves every row: a row is predicted by a fit and a
            # calibration set that both leave it out
            keep = np.ones(n, dtype=bool)
            yhat = np.empty(m)
            for i in range(n):
                keep[i] = False
                c, lo, hi = split_interval(X[keep], y[keep], X[i : i + 1])
                yhat[i], lower[i], upper[i] = c[0], lo[0], hi[0]
                keep[i] = True
        else:
            yhat, lower, upper = split_interval(X, y, X0)
    elif key == "jackknife+":
        if in_sample:
            yhat = np.empty(m)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                keep[i] = False
                lo, hi = _jackknife_plus(X[keep], y[keep], X[i : i + 1], alpha)
                yhat[i] = fit_predict(X[keep], y[keep], X[i : i + 1])[0]
                lower[i], upper[i] = lo[0], hi[0]
                keep[i] = True
        else:
            yhat = fit_predict(X, y, X0)
            lower, upper = _jackknife_plus(X, y, X0, alpha)
    else:
        if in_sample:
            yhat = np.empty(m)
            keep = np.ones(n, dtype=bool)
            for i in range(n):
                keep[i] = False
                yhat[i] = fit_predict(X[keep], y[keep], X[i : i + 1])[0]
                lower[i], upper[i] = _full_interval(X[keep], y[keep], X[i], alpha)
                keep[i] = True
        else:
            yhat = fit_predict(X, y, X0)
            for j in range(m):
                lower[j], upper[j] = _full_interval(X, y, X0[j], alpha)

    out = pd.DataFrame({"yhat": yhat, "lower": lower, "upper": upper}, index=index)
    out.attrs.update({"method": key, "alpha": float(alpha), "n": int(n)})
    return out
