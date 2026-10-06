"""Series (sieve) regression with a cross-validated number of terms.

The regression function of ``y`` on one variable ``x`` is approximated by
a polynomial or a spline in ``x`` and fitted by least squares; further
regressors enter linearly. The number of terms is a tuning parameter. It
is chosen here by leave-one-out cross-validation, which for least squares
needs no refitting: the prediction error of observation ``i`` from a fit
without it is ``e_i / (1 - h_ii)`` with ``h_ii`` its leverage.

References
----------
[@hansen2022econometrics]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core._vcov import sandwich_vcov
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["series"]

_VCE = ("ols", "hc0", "hc1", "hc2", "hc3", "cluster")


def _basis(
    x: np.ndarray, order: int, kind: str, degree: int, knots: Optional[np.ndarray]
) -> np.ndarray:
    """Columns for ``x`` without the constant. ``order`` is the degree of a
    polynomial, or the number of knots of a spline of degree ``degree``."""
    if kind == "polynomial":
        return np.column_stack([x**j for j in range(1, order + 1)])
    powers = [x**j for j in range(1, degree + 1)]
    assert knots is not None
    hinges = [np.where(x > k, (x - k) ** degree, 0.0) for k in knots]
    return np.column_stack(powers + hinges)


def _knots(x: np.ndarray, count: int, placement: str) -> np.ndarray:
    if count == 0:
        return np.zeros(0)
    inner = np.arange(1, count + 1) / (count + 1.0)
    if placement == "uniform":
        return np.asarray(x.min() + inner * (x.max() - x.min()))
    return np.asarray(np.quantile(x, inner))


def series(
    formula: str,
    data: pd.DataFrame,
    x: str,
    *,
    basis: str = "polynomial",
    order: Optional[int] = None,
    orders: Optional[Sequence[int]] = None,
    degree: int = 3,
    knots: str = "quantile",
    grid: Optional[Any] = None,
    vce: str = "robust",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """Series regression of ``y`` on a flexible function of ``x``.

    Parameters
    ----------
    formula : str
        ``"y ~ 1"``, or ``"y ~ z1 + z2"`` for regressors that enter
        linearly next to the function of ``x``. ``x`` itself is left out.
    data : pandas.DataFrame
    x : str
        The variable whose effect is left unrestricted.
    basis : {'polynomial', 'spline'}, default 'polynomial'
        Powers of ``x``, or a spline of degree ``degree`` with knots.
    order : int, optional
        Degree of the polynomial, or number of knots of the spline. When
        omitted it is chosen from ``orders`` by cross-validation.
    orders : sequence of int, optional
        Candidates for cross-validation. Default: polynomial degrees 1 to
        8, or 0 to 8 knots.
    degree : int, default 3
        Degree of the spline pieces (3 is a cubic spline).
    knots : {'quantile', 'uniform'}, default 'quantile'
        Knots at equally spaced quantiles of ``x``, or equally spaced over
        its range.
    grid : int or array-like, optional
        Points at which the fitted function is tabulated. An integer is
        that many equally spaced points over the range of ``x`` (default
        101).
    vce : {'robust', 'ols', 'hc0', 'hc2', 'hc3'}, default 'robust'
        ``'robust'`` is HC1.
    cluster : str, optional
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` are the coefficients of the linear regressors and of the
        basis columns: ``s(x):p1``, ``s(x):p2``, ... for the powers and
        ``s(x):k1``, ... for the spline pieces that start at each knot, all
        in the standardised variable ``(x - mean) / sd``. They are not of
        interest one by one. ``model_info`` has ``order``, ``cv`` (one
        row per candidate: number of terms, cross-validation criterion,
        residual sum of squares), ``knots`` and ``function``: the fitted
        function on the grid with standard errors and pointwise intervals,
        and its first derivative. The function is evaluated with the linear
        regressors at their sample means.

    Notes
    -----
    The intervals are pointwise and take the number of terms as given:
    they measure sampling error and not the approximation error of a
    function that is not exactly a polynomial or a spline, nor the fact
    that the order was chosen on the same data. They are too short where
    the function bends more than the basis can follow.

    ``x`` is standardised and the fit goes through an orthogonal
    decomposition, so a high-order polynomial is computed stably. A
    polynomial is erratic near the ends of the data at high orders; a
    spline is the usual remedy.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.uniform(0, 3, 500)})
    >>> df["y"] = np.sin(2 * df.x) + rng.normal(scale=0.3, size=500)
    >>> fit = sp.series("y ~ 1", df, "x")
    >>> bool(fit.model_info["order"] >= 4)
    True
    >>> curve = fit.model_info["function"]
    >>> bool(np.abs(curve["fit"] - np.sin(2 * curve["x"])).max() < 0.35)
    True

    References
    ----------
    [@hansen2022econometrics]
    """
    from ..regression.ols import regress

    basis = str(basis).lower()
    if basis not in ("polynomial", "spline"):
        raise MethodIncompatibility(
            f"sp.series: basis={basis!r} is not available.",
            recovery_hint="Use 'polynomial' or 'spline'.",
        )
    if knots not in ("quantile", "uniform"):
        raise MethodIncompatibility(
            f"sp.series: knots={knots!r} is not available.",
            recovery_hint="Use 'quantile' or 'uniform'.",
        )
    vce = {"robust": "hc1", "unadjusted": "ols", "nonrobust": "ols"}.get(
        str(vce).lower(), str(vce).lower()
    )
    if cluster is not None:
        vce = "cluster"
    if vce not in _VCE or (vce == "cluster" and cluster is None):
        raise MethodIncompatibility(
            f"sp.series: vce={vce!r} is not available.",
            recovery_hint="Use 'robust', 'ols', 'hc0', 'hc2', 'hc3', or cluster=.",
        )
    for name, column in (("x", x), ("cluster", cluster)):
        if column is not None and column not in data.columns:
            raise MethodIncompatibility(
                f"sp.series: {name}={column!r} is not a column of the data.",
                recovery_hint="Check the spelling.",
            )
    if degree < 1:
        raise MethodIncompatibility(
            f"sp.series: degree={degree!r} has to be at least 1.",
            recovery_hint="degree=3 is a cubic spline.",
        )
    extra = [c for c in (x, cluster) if c is not None]
    frame = data.dropna(subset=extra)
    base = regress(formula, data=frame)
    info = base.data_info
    frame = frame.loc[info["sample_index"]]
    Z = np.asarray(info["X"], dtype=float)
    z_names = [str(v) for v in info["var_names"]]
    if x in z_names:
        raise MethodIncompatibility(
            f"sp.series: {x!r} must not be in the formula.",
            recovery_hint="Its basis columns are added by the function.",
        )
    y = np.asarray(info["y"], dtype=float)
    xv = np.asarray(frame[x], dtype=float)
    n = len(y)
    if np.ptp(xv) == 0:
        raise DataInsufficient(
            f"sp.series: {x!r} does not vary.",
            recovery_hint="Nothing can be learned about its effect.",
        )
    # centred and scaled: the powers of years or dollars overflow otherwise
    centre, spread = float(xv.mean()), float(xv.std())
    u = (xv - centre) / spread

    def design(k: int) -> np.ndarray:
        kn = _knots(u, k, knots) if basis == "spline" else None
        return np.column_stack([Z, _basis(u, k, basis, degree, kn)])

    def loo(k: int) -> Dict[str, float]:
        W = design(k)
        Q, R = np.linalg.qr(W)
        if np.abs(np.diag(R)).min() < 1e-10 * np.abs(np.diag(R)).max():
            return {"order": k, "terms": W.shape[1], "cv": np.inf, "rss": np.nan}
        e = y - Q @ (Q.T @ y)
        h = (Q**2).sum(axis=1)
        return {
            "order": k,
            "terms": W.shape[1],
            "cv": float(((e / (1.0 - h)) ** 2).sum()),
            "rss": float(e @ e),
        }

    if orders is None:
        candidates: List[int] = (
            list(range(1, 9)) if basis == "polynomial" else list(range(0, 9))
        )
    else:
        candidates = sorted({int(k) for k in orders})
    lowest = 1 if basis == "polynomial" else 0
    if order is not None:
        candidates = [int(order)]
    if not candidates or min(candidates) < lowest:
        raise MethodIncompatibility(
            f"sp.series: the order has to be at least {lowest}.",
            recovery_hint="It is the polynomial degree or the number of knots.",
        )
    table = pd.DataFrame([loo(k) for k in candidates])
    usable = table[np.isfinite(table["cv"]) & (table["terms"] < n)]
    if usable.empty:
        raise DataInsufficient(
            "sp.series: no candidate order gives a design of full rank.",
            recovery_hint="Lower the orders, or check that x takes enough "
            "distinct values.",
        )
    chosen = int(usable.loc[usable["cv"].idxmin(), "order"])

    kn = _knots(u, chosen, knots) if basis == "spline" else None
    B = _basis(u, chosen, basis, degree, kn)
    W = np.column_stack([Z, B])
    k_total = W.shape[1]
    # the fit through an orthonormal basis; coefficients are reported for
    # the columns as written
    Q, R = np.linalg.qr(W)
    beta = np.linalg.solve(R, Q.T @ y)
    resid = y - W @ beta
    rss = float(resid @ resid)
    df_resid = n - k_total
    R_inv = np.linalg.inv(R)
    bread = R_inv @ R_inv.T
    clusters = None if cluster is None else pd.factorize(frame[cluster])[0]
    if vce == "ols":
        cov = bread * (rss / df_resid)
    elif vce == "cluster":
        assert clusters is not None
        cov = sandwich_vcov(
            bread, W * resid[:, None], clusters=clusters, correction="stata"
        )
    else:
        if vce == "hc0":
            weight = np.ones(n)
        elif vce == "hc1":
            weight = np.full(n, n / df_resid)
        else:
            h = (Q**2).sum(axis=1)
            weight = 1.0 / (1.0 - h) if vce == "hc2" else 1.0 / (1.0 - h) ** 2
        cov = bread @ ((W * (resid**2 * weight)[:, None]).T @ W) @ bread
    se = np.sqrt(np.diag(cov))

    # the function on a grid, the linear regressors at their means
    if grid is None or np.isscalar(grid):
        count = int(grid) if grid else 101  # type: ignore[arg-type]
        points = np.linspace(xv.min(), xv.max(), count)
    else:
        points = np.asarray(grid, dtype=float)
    ug = (points - centre) / spread
    Bg = _basis(ug, chosen, basis, degree, kn)
    Wg = np.column_stack([np.tile(Z.mean(axis=0), (len(points), 1)), Bg])
    fit = Wg @ beta
    fit_se = np.sqrt(np.einsum("ij,jk,ik->i", Wg, cov, Wg))
    step = 1e-6
    Dg = (
        _basis(ug + step, chosen, basis, degree, kn)
        - _basis(ug - step, chosen, basis, degree, kn)
    ) / (2.0 * step * spread)
    Dw = np.column_stack([np.zeros((len(points), Z.shape[1])), Dg])
    slope = Dw @ beta
    slope_se = np.sqrt(np.einsum("ij,jk,ik->i", Dw, cov, Dw))
    crit = (
        float(stats.norm.ppf(1.0 - alpha / 2.0))
        if clusters is None
        else float(stats.t.ppf(1.0 - alpha / 2.0, int(clusters.max())))
    )
    function = pd.DataFrame(
        {
            "x": points,
            "fit": fit,
            "se": fit_se,
            "ci_lower": fit - crit * fit_se,
            "ci_upper": fit + crit * fit_se,
            "derivative": slope,
            "derivative_se": slope_se,
        }
    )

    if basis == "polynomial":
        b_names = [f"s({x}):p{j}" for j in range(1, chosen + 1)]
        knots_x: Optional[np.ndarray] = None
    else:
        assert kn is not None
        knots_x = centre + spread * kn
        b_names = [f"s({x}):p{j}" for j in range(1, degree + 1)] + [
            f"s({x}):k{j}" for j in range(1, len(knots_x) + 1)
        ]
    names = z_names + b_names
    label = {"ols": "nonrobust", "hc1": "robust"}.get(vce, vce)
    tss = float(((y - y.mean()) ** 2).sum())
    model_info: Dict[str, Any] = {
        "model_type": "Series regression",
        "method": f"{basis} series, least squares",
        "formula": formula,
        "series_var": x,
        "basis": basis,
        "order": chosen,
        "order_selected_by": "cross-validation" if order is None else "user",
        "cv": table,
        "degree": degree if basis == "spline" else chosen,
        "knots": knots_x,
        "function": function,
        "scaling": {"centre": centre, "scale": spread},
        "robust": label,
        "cluster": cluster,
        "alpha": alpha,
        "has_constant": True,
    }
    if clusters is not None:
        model_info["n_clusters"] = int(clusters.max()) + 1
    data_info: Dict[str, Any] = {
        "nobs": n,
        "df_model": k_total - 1,
        "df_resid": df_resid if clusters is None else int(clusters.max()),
        "dependent_var": info.get("dependent_var"),
        "var_names": names,
        "var_cov": cov,
        "X": W,
        "y": y,
        "residuals": resid,
        "fitted_values": y - resid,
        "rss": rss,
        "tss": tss,
        "sample_index": frame.index,
    }
    diagnostics = {
        "R-squared": 1.0 - rss / tss if tss > 0 else float("nan"),
        "Root MSE": float(np.sqrt(rss / df_resid)),
        "Residual SS": rss,
        "Cross-validation": float(table.loc[table["order"] == chosen, "cv"].iloc[0]),
    }
    index = pd.Index(names)
    return EconometricResults(
        params=pd.Series(beta, index=index),
        std_errors=pd.Series(se, index=index),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
