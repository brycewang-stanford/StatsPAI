"""Binscatter the way Cattaneo, Crump, Farrell and Feng (2024) define it.

The canonical binscatter of ``sp.binscatter(method='binsreg')``:

* bins at the empirical quantiles of ``x`` (the averaged-inverse-CDF
  quantile, R ``type = 2``), each bin ``(k_{j-1}, k_j]`` with the first one
  closed;
* covariates enter the same regression as the bin indicators,
  ``y = sum_j 1{x in bin j} b_j + w'g + e``, and the dot of bin ``j`` is
  ``b_j + mean(w)'g`` placed at the mean of ``x`` in the bin -- not the
  bin means of residualized ``y`` and ``x``, which estimate a different
  function when ``x`` and ``w`` are related;
* the number of bins minimises the integrated mean squared error of that
  estimator: a rule-of-thumb choice (global quadratic fits for the
  conditional mean and variance, a Gaussian reference density bounded
  below at its value 1.96 standard deviations out), floored at
  ``ceil((2n)^(1/3))``, starts a direct plug-in choice that estimates the
  variance constant from the binscatter's HC1 variance and the bias
  constant from a continuous piecewise-linear fit on the same bins;
* confidence intervals and the uniform band use the continuous
  piecewise-linear fit on the same bins (robust bias correction,
  ``ci = cb = (1, 1)``), HC1 standard errors and, for the band, a
  simulated critical value of the supremum of the t-process.

R ``binsreg`` 2.2 (GPL-3, used as a black box: only its outputs on
test data were compared) gives the same bins, dots, intervals and the
same IMSE choice of the number of bins on the 40 designs of
``tests/test_binsreg_parity.py``; the band's critical value is
simulated by both and agrees only in distribution.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..exceptions import AssumptionWarning, DataInsufficient

__all__ = ["binsreg_knots", "binsreg_series", "select_nbins"]

_Z975 = norm.ppf(0.975)


def binsreg_knots(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-spaced knots: min, the ``j / J`` quantiles, max."""
    knots = np.quantile(
        x, np.linspace(0.0, 1.0, n_bins + 1), method="averaged_inverted_cdf"
    )
    knots[0], knots[-1] = float(np.min(x)), float(np.max(x))
    return knots


def _bin_index(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    return np.searchsorted(knots[1:-1], x, side="left")


def _design(x: np.ndarray, degree: int, w: Optional[np.ndarray]) -> np.ndarray:
    cols = [x**k for k in range(degree + 1)]
    if w is not None:
        cols.append(w)
    return np.column_stack(cols)


def _spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Continuous piecewise-linear basis on the knots (p = 1, s = 1)."""
    cols = [np.ones_like(x), x] + [np.maximum(x - k, 0.0) for k in knots[1:-1]]
    return np.column_stack(cols)


def _hc1(X: np.ndarray, e: np.ndarray) -> np.ndarray:
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    meat = (X * (e**2)[:, None]).T @ X
    return bread @ meat @ bread * n / (n - k)


def _lstsq(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def _w_matrix(w: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if w is None:
        return None
    w = np.asarray(w, dtype=float)
    return w.reshape(len(w), -1)


def _rot(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray]) -> int:
    n = len(y)
    X = _design(x, 2, w)
    b = _lstsq(X, y)
    slope = b[1] + 2.0 * b[2] * x
    m1 = X @ b
    m2 = X @ _lstsq(X, y**2)
    sd = float(np.std(x, ddof=1))
    z = (x - np.mean(x)) / sd
    dens = np.maximum(norm.pdf(z), norm.pdf(_Z975)) / sd
    bias = np.mean(slope**2 / dens**2) / 12.0
    var = float(np.mean(m2 - m1**2))
    if not (var > 0 and bias > 0):
        return 1
    return int(np.ceil((2.0 * bias / var) ** (1.0 / 3.0) * n ** (1.0 / 3.0)))


def _dpi(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], start: int) -> int:
    n = len(y)
    knots = binsreg_knots(x, start)
    idx = _bin_index(x, knots)
    J = start
    B0 = np.eye(J)[idx]
    X = B0 if w is None else np.column_stack([B0, w])
    beta = _lstsq(X, y)
    V = n * _hc1(X, y - X @ beta)
    nj = np.bincount(idx, minlength=J)
    var = float(np.trace(V[:J, :J] @ np.diag(nj / n)) / J)
    S = _spline_basis(x, knots)
    Xs = S if w is None else np.column_stack([S, w])
    bs = _lstsq(Xs, y)
    slope = np.cumsum(np.r_[bs[1], bs[2 : 2 + J - 1]])[idx]
    lead = slope * x
    resid = lead - B0 @ _lstsq(B0, lead)
    bias = J**2 * float(np.mean(resid**2))
    if not (var > 0 and bias > 0):
        return start
    return int(np.ceil((2.0 * bias / var) ** (1.0 / 3.0) * n ** (1.0 / 3.0)))


def select_nbins(
    x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
) -> Dict[str, int]:
    """IMSE-optimal number of bins of the canonical binscatter.

    Returns ``rot_poly`` (the rule of thumb), ``rot`` (floored at
    ``ceil((2n)^(1/3))``) and ``dpi`` (the direct plug-in started from
    ``rot``), as R ``binsregselect`` reports ``nbinsrot.poly``,
    ``nbinsrot.regul`` and ``nbinsdpi``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = _w_matrix(w)
    n = len(y)
    if n <= 20 + 1:
        raise DataInsufficient(
            f"binsreg: {n} observations are too few to select the number of bins",
            recovery_hint="Pass n_bins= explicitly.",
        )
    n_unique = int(np.unique(x).size)
    if n_unique < n:
        warnings.warn(
            AssumptionWarning(
                f"binsreg: x takes {n_unique} distinct values in {n} rows; the "
                "number of bins is selected as if they were all distinct (R / "
                "Stata binsreg adjust the effective sample size for mass "
                "points, which is not implemented here)",
                recovery_hint="Pass n_bins= to fix the number of bins.",
                diagnostics={"n": n, "n_unique": n_unique},
            ),
            stacklevel=3,
        )
    rot_poly = _rot(x, y, w)
    rot = max(rot_poly, int(np.ceil((2.0 * n) ** (1.0 / 3.0))))
    # the degrees-of-freedom checks of the binsreg article (dfcheck 20, 30):
    # the plug-in needs n > 30 + J_rot, else the rule of thumb stands
    dpi = _dpi(x, y, w, rot) if n > 30 + rot else rot
    return {"rot_poly": rot_poly, "rot": rot, "dpi": dpi, "n_unique": n_unique}


def binsreg_series(
    x: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray],
    n_bins: Optional[int],
    ci: bool,
    cb: bool,
    level: float,
    nsims: int,
    simsgrid: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Dots, intervals and band of one series (see the module docstring)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = _w_matrix(w)
    n = len(y)
    selection: Dict[str, Any] = {}
    if n_bins is None:
        selection = select_nbins(x, y, w)
        J = selection["dpi"]
    else:
        J = int(n_bins)
    knots = np.unique(binsreg_knots(x, J))
    if len(knots) - 1 < J:
        selection["unique_knots"] = True
    J = len(knots) - 1
    if J < 1:
        raise DataInsufficient(
            "binsreg: x takes a single value", recovery_hint="Use a variable x."
        )
    idx = _bin_index(x, knots)
    nj = np.bincount(idx, minlength=J)
    B0 = np.eye(J)[idx]
    X = B0 if w is None else np.column_stack([B0, w])
    beta = _lstsq(X, y)
    wbar = None if w is None else w.mean(axis=0)
    dots = beta[:J] + (0.0 if w is None else float(beta[J:] @ wbar))
    with np.errstate(invalid="ignore"):
        xdot = np.bincount(idx, x, minlength=J) / nj
    out = pd.DataFrame({"x_mean": xdot, "y_mean": dots, "n": nj})
    band = None
    cval = None
    if ci or cb:
        S = _spline_basis(x, knots)
        Xs = S if w is None else np.column_stack([S, w])
        if n <= Xs.shape[1] + 1:
            raise DataInsufficient(
                f"binsreg: {n} observations cannot support intervals with {J} bins",
                recovery_hint="Use fewer bins (n_bins=) or ci=False.",
            )
        bs = _lstsq(Xs, y)
        Vs = _hc1(Xs, y - Xs @ bs)

        def evaluate(points: np.ndarray):
            L = _spline_basis(points, knots)
            if wbar is not None:
                L = np.column_stack([L, np.tile(wbar, (len(points), 1))])
            fit = L @ bs
            se = np.sqrt(np.einsum("ij,jk,ik->i", L, Vs, L))
            return L, fit, se

        z = norm.ppf(0.5 + level / 2.0)
        if ci:
            _, fit, se = evaluate(xdot)
            out["ci_lower"] = fit - z * se
            out["ci_upper"] = fit + z * se
        if cb:
            grid = []
            for j in range(J):
                inner = np.linspace(knots[j], knots[j + 1], simsgrid + 2)[1:-1]
                grid.extend(inner)
                if j < J - 1:
                    grid.append(knots[j + 1])
            grid = np.asarray(grid)
            L, fit, se = evaluate(grid)
            rng = np.random.default_rng(seed)
            # sup_x |L(x)'Z| / se(x), Z ~ N(0, V): the Gaussian process the
            # band's critical value is the (level) quantile of
            vals, vecs = np.linalg.eigh(Vs)
            root = vecs * np.sqrt(np.clip(vals, 0.0, None))
            draws = rng.standard_normal((nsims, Vs.shape[0])) @ root.T
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.abs(draws @ L.T) / se
            cval = float(np.quantile(np.nanmax(t, axis=1), level))
            band = pd.DataFrame(
                {
                    "x": grid,
                    "fit": fit,
                    "cb_lower": fit - cval * se,
                    "cb_upper": fit + cval * se,
                }
            )
    return {
        "dots": out,
        "n_bins": J,
        "knots": knots,
        "selection": selection,
        "band": band,
        "cb_critical_value": cval,
    }
