"""Bandwidth selection for GWR — golden-section search as in ``GWmodel``.

Criteria (each evaluated exactly as ``GWmodel`` does, see
``tests/reference_parity/test_spatial_survey_R_parity.py``):

- ``"AICc"`` (default) — ``GWmodel::gwr.aic`` (which, despite its name and
  ``bw.gwr(approach = "AIC")``, returns the *corrected* AIC); also mgwr's
  default criterion.
- ``"CV"`` — leave-one-out score ``sum_i (e_i / (1 - S_ii))^2``, equal to
  ``GWmodel::gwr.cv``'s refit-with-``W_ii = 0`` sum of squared residuals.
- ``"AIC"``, ``"BIC"`` — the uncorrected AIC and the BIC of
  ``gwr.basic``'s diagnostics.

The search is ``GWmodel``'s ``gold()`` routine, reproduced step for step:
default bounds ``[20, n]`` (adaptive) or ``[D / 5000, D]`` with ``D`` the
largest inter-point distance (fixed); integer probes use ``floor`` for the
upper and ``round`` for the lower golden point; iteration stops when the
bracket step or the criterion difference falls to ``tol`` (GWmodel's
hard-coded ``1e-4`` by default). The criterion
is generally *not* unimodal on the neighbour-count lattice, so any golden
search returns a search-path-dependent local minimum; reproducing the
reference's path is what makes the selected bandwidth comparable.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

import numpy as np

from .gwr import GWRResult, KernelName, gwr

Criterion = Literal["AICc", "AIC", "BIC", "CV"]


def _criterion(result: GWRResult, criterion: Criterion) -> float:
    if criterion == "AICc":
        return float(result.aicc)
    if criterion == "AIC":
        return float(result.aic)
    if criterion == "BIC":
        return float(result.bic)
    if criterion == "CV":
        return float(result.cv)
    raise ValueError(f"unknown criterion {criterion!r}")


def _gold(
    fun: Callable[[float], float],
    xL: float,
    xU: float,
    adapt_bw: bool,
    eps: float = 1e-4,
) -> float:
    """Golden-section minimiser of ``GWmodel:::gold`` (GWmodel 2.4).

    Transcribed from the R source, including its stopping rule
    (``|d| <= 1e-4`` or ``|f2 - f1| <= 1e-4``) and its integer probes
    (``floor`` / ``round``) for adaptive bandwidths. R's ``round`` and
    NumPy's both round half to even.
    """
    R = (np.sqrt(5.0) - 1.0) / 2.0
    d = R * (xU - xL)
    if adapt_bw:
        x1 = float(np.floor(xL + d))
        x2 = float(np.round(xU - d))
    else:
        x1 = xL + d
        x2 = xU - d
    f1 = fun(x1)
    f2 = fun(x2)
    d1 = f2 - f1
    xopt = x1 if f1 < f2 else x2
    while abs(d) > eps and abs(d1) > eps:
        d = R * d
        if f1 < f2:
            xL = x2
            x2 = x1
            x1 = float(np.round(xL + d)) if adapt_bw else xL + d
            f2 = f1
            f1 = fun(x1)
        else:
            xU = x1
            x1 = x2
            x2 = float(np.floor(xU - d)) if adapt_bw else xU - d
            f1 = f2
            f2 = fun(x2)
        xopt = x1 if f1 < f2 else x2
        d1 = f2 - f1
    return float(xopt)


def _max_pairwise_distance(coords: np.ndarray) -> float:
    n = coords.shape[0]
    if n <= 5000:
        diff = coords[:, None, :] - coords[None, :, :]
        return float(np.sqrt((diff**2).sum(axis=2)).max())
    # GWmodel switches to the bounding-box diagonal when it does not build
    # the distance matrix (dp.n + dp.n > 10000).
    return float(np.linalg.norm(coords.max(axis=0) - coords.min(axis=0)))


def gwr_bandwidth(
    coords: Any,
    y: Any,
    X: Any,
    kernel: KernelName = "bisquare",
    fixed: bool = False,
    criterion: Criterion = "AICc",
    bw_min: Optional[float] = None,
    bw_max: Optional[float] = None,
    add_constant: bool = True,
    tol: float = 1e-4,
) -> float:
    """Select a GWR bandwidth by golden-section search (``GWmodel::bw.gwr``).

    Parameters
    ----------
    coords : (n, 2) array-like
        Projected point coordinates.
    y : (n,) array-like
    X : (n, p) array-like
        Regressors, without a constant unless ``add_constant=False``.
    kernel : {"bisquare", "gaussian", "exponential"}
    fixed : bool, default False
        False: the bandwidth is a nearest-neighbour count (integer probes).
        True: a distance.
    criterion : {"AICc", "AIC", "BIC", "CV"}, default "AICc"
        ``"AICc"`` is ``bw.gwr(approach = "AICc")`` (and mgwr's default);
        ``"CV"`` is ``bw.gwr(approach = "CV")``, GWmodel's default.
    bw_min, bw_max : float, optional
        Search bounds. Defaults are GWmodel's: ``[20, n]`` for an adaptive
        bandwidth (``[k + 2, n]`` when ``n <= 20``, where GWmodel's bounds
        would be empty) and ``[D / 5000, D]`` for a fixed one, ``D`` being
        the largest inter-point distance.
    add_constant : bool, default True
    tol : float, default 1e-4
        Stopping threshold of the golden-section search, applied (as in
        GWmodel, where it is hard-coded to ``1e-4``) both to the bracket
        step and to the difference of the two probed criterion values.

    Returns
    -------
    float
        The selected bandwidth.

    Notes
    -----
    The criterion is generally not unimodal in the bandwidth, so the
    result is the local minimum reached by the golden-section path, not
    necessarily the global minimum. To scan globally, evaluate
    ``sp.gwr(...).aicc`` / ``.cv`` over a grid.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> coords = rng.uniform(0, 10, size=(n, 2))
    >>> x = rng.normal(size=n)
    >>> beta = 0.5 + 0.1 * coords[:, 1]  # spatially varying slope
    >>> y = 1.0 + beta * x + rng.normal(0, 0.3, n)
    >>> bw = sp.gwr_bandwidth(coords, y, x.reshape(-1, 1), criterion="AICc")
    >>> bool(bw > 0)
    True
    >>> res = sp.gwr(coords, y, x.reshape(-1, 1), bw=bw, fixed=False)
    >>> type(res).__name__
    'GWRResult'
    """
    coords = np.asarray(coords, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    X = np.asarray(X, dtype=float)
    n = coords.shape[0]
    k = X.shape[1] + (1 if add_constant else 0)
    if criterion not in ("AICc", "AIC", "BIC", "CV"):
        raise ValueError(f"unknown criterion {criterion!r}")
    if fixed:
        if bw_max is None:
            bw_max = _max_pairwise_distance(coords)
        if bw_min is None:
            bw_min = bw_max / 5000.0
    else:
        if bw_min is None:
            bw_min = 20.0 if n > 20 else float(k + 2)
        if bw_max is None:
            bw_max = float(n)

    def objective(bw: float) -> float:
        try:
            res = gwr(
                coords,
                y,
                X,
                bw,
                kernel=kernel,
                fixed=fixed,
                add_constant=add_constant,
            )
        except (np.linalg.LinAlgError, ValueError):
            # GWmodel's gwr.aic / gwr.cv return Inf when a local fit fails.
            return np.inf
        val = _criterion(res, criterion)
        return np.inf if np.isnan(val) else val

    return _gold(
        objective, float(bw_min), float(bw_max), adapt_bw=not fixed, eps=float(tol)
    )
