"""
Quantile Regression.

Estimates conditional quantiles of the outcome distribution, allowing
analysis of heterogeneous effects across the distribution (not just
the mean). More robust to outliers than OLS.

References
----------
Koenker, R. and Bassett, G. (1978).
"Regression Quantiles."
*Econometrica*, 46(1), 33-50. [@koenker1978regression]

Koenker, R. (2005).
*Quantile Regression*. Cambridge University Press.

Chernozhukov, V. and Hansen, C. (2005).
"An IV Model of Quantile Treatment Effects."
*Econometrica*, 73(1), 245-261. [@chernozhukov2005model]
"""

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linprog

from .._aliases import accepts_formula_first
from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility


def _as_float_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


@accepts_formula_first()
def qreg(
    data: pd.DataFrame,
    formula: Optional[str] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
    quantile: float = 0.5,
    alpha: float = 0.05,
    vce: Optional[str] = None,
) -> CausalResult:
    """
    Quantile regression at a single quantile.

    Equivalent to Stata's ``qreg y x, quantile(0.5)``.

    Parameters
    ----------
    data : pd.DataFrame
    formula : str, optional
        Formula like ``"y ~ x1 + x2"`` (patsy-style).
    y : str, optional
        Outcome variable (alternative to formula).
    x : list of str, optional
        Regressors (alternative to formula).
    quantile : float, default 0.5
        Quantile to estimate (0 < q < 1). 0.5 = median.
    alpha : float, default 0.05
        Also sets the Hall-Sheather bandwidth's ``z_{1-alpha/2}``, as
        Stata's ``level()`` does.
    vce : {None, 'iid', 'robust', 'nid', 'powell'}, optional
        Standard errors (all use the Hall-Sheather bandwidth ``h`` and the
        quantile fits at ``tau +/- h``):

        * ``None`` / ``'iid'`` -- Stata's default ``qreg`` (``vce(iid)``,
          fitted sparsity): ``tau (1-tau) s^2 (X'X)^{-1}`` with ``s`` the
          difference quotient of the mean fitted quantiles.
        * ``'robust'`` -- Stata ``vce(robust)``: the Hendricks-Koenker
          sandwich ``tau (1-tau) H X'X H``, ``H = (sum f_i x_i x_i')^{-1}``
          with ``f_i = 2h / x_i'(b(tau+h) - b(tau-h))``.
        * ``'nid'`` -- R ``quantreg::summary.rq(se="nid")``: the same
          sandwich with quantreg's conventions (``alpha = 0.05`` in the
          bandwidth, ``h`` halved until ``tau +/- h`` is inside (0, 1),
          ``sqrt(eps)`` subtracted from the fitted differences).
        * ``'powell'`` -- the Silverman-bandwidth Gaussian-kernel iid
          sandwich this function used before 1.32 (matches neither
          reference; kept to reproduce old numbers).

        Cluster-robust quantile-regression SEs are not offered: the Stata
        18 ``qreg`` used as the reference refuses ``vce(cluster)``, and a
        variance with no reference check is not shipped.

    Returns
    -------
    CausalResult
        Coefficients at the specified quantile.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> # Median (0.5) regression of log wage on education and experience
    >>> result = sp.qreg(df, y='log_wage', x=['education', 'experience'],
    ...                  quantile=0.5)
    >>> # 90th percentile
    >>> result = sp.qreg(df, y='log_wage', x=['education', 'experience'],
    ...                  quantile=0.9)
    >>> bool(0 < result.estimate < 1)
    True

    Notes
    -----
    Quantile regression minimizes:

    .. math::
        \\min_\\beta \\sum_i \\rho_\\tau(Y_i - X_i'\\beta)

    where ρ_τ(u) = u(τ - 1(u < 0)) is the check function.

    Standard errors default to Stata's ``qreg`` (``vce(iid)``, fitted
    sparsity, Hall-Sheather bandwidth); see ``vce``.

    See Koenker & Bassett (1978, *Econometrica*).
    """
    if not (0 < quantile < 1):
        raise MethodIncompatibility(f"quantile must be in (0, 1), got {quantile}")

    # Parse inputs
    if formula is not None:
        y_name, x_names = _parse_formula(formula)
    elif y is not None and x is not None:
        y_name, x_names = y, x
    else:
        raise MethodIncompatibility("Provide either formula or (y, x)")

    kind = str(vce or "iid").lower()
    if kind not in ("iid", "robust", "nid", "powell"):
        raise MethodIncompatibility(
            f"qreg: vce must be one of 'iid', 'robust', 'nid', 'powell'; "
            f"got {vce!r}.",
            recovery_hint=(
                "Cluster-robust quantile-regression SEs are not available; "
                "use sp.bootstrap with a cluster resampling scheme."
            ),
            diagnostics={"vce": vce},
        )
    cols = [y_name] + x_names
    missing_cols = [c for c in cols if c not in data]
    if missing_cols:
        raise MethodIncompatibility(
            f"qreg: columns not found in data: {missing_cols}",
            diagnostics={"missing": missing_cols},
        )
    df = data[list(dict.fromkeys(cols))].dropna()
    Y = df[y_name].values.astype(float)
    X = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in x_names]
    )
    n, k = X.shape
    var_names = ["const"] + x_names

    # Solve quantile regression via linear programming
    beta = _qreg_fit(Y, X, quantile)
    resid = Y - X @ beta

    if kind == "powell":
        se = _qreg_se(Y, X, beta, resid, quantile)
        bandwidth = None
    else:
        vcov, bandwidth = _qreg_vcov(
            Y,
            X,
            resid,
            quantile,
            kind,
            alpha=alpha,
        )
        se = _as_float_array(np.sqrt(np.maximum(np.diag(vcov), 0.0)))

    z_stats = beta / se
    pvals = 2 * stats.norm.sf(np.abs(z_stats))
    z_crit = stats.norm.ppf(1 - alpha / 2)

    detail = pd.DataFrame(
        {
            "variable": var_names,
            "coefficient": beta,
            "se": se,
            "z": z_stats,
            "pvalue": pvals,
        }
    )

    # Main estimate: first regressor (after constant)
    main_coef = float(beta[1])
    main_se = float(se[1])
    main_p = float(pvals[1])
    ci = (main_coef - z_crit * main_se, main_coef + z_crit * main_se)

    model_info = {
        "quantile": quantile,
        "pseudo_r2": _pseudo_r2(Y, resid, quantile),
        "n_obs": n,
        "vce": kind,
        "bandwidth": bandwidth,
    }

    return CausalResult(
        method=f"Quantile Regression (tau={quantile})",
        estimand=f"Q({quantile}) {x_names[0]}",
        estimate=main_coef,
        se=main_se,
        pvalue=main_p,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="qreg",
    )


def sqreg(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    quantiles: Optional[List[float]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Simultaneous quantile regression at multiple quantiles.

    Equivalent to Stata's ``sqreg y x, quantiles(10 25 50 75 90)``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    x : list of str
    quantiles : list of float, optional
        Default: [0.1, 0.25, 0.5, 0.75, 0.9].
    alpha : float, default 0.05

    Returns
    -------
    pd.DataFrame
        Rows: variables. Columns: quantiles with coefficients and SEs.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> table = sp.sqreg(df, y='log_wage', x=['education', 'experience'])
    >>> bool('Q(0.5)' in table.columns)
    True
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    results = {}
    for q in quantiles:
        r = qreg(data, y=y, x=x, quantile=q, alpha=alpha)
        detail = r.detail
        if not isinstance(detail, pd.DataFrame):
            raise MethodIncompatibility("qreg detail table is unavailable")
        for _, row in detail.iterrows():
            var = row["variable"]
            if var not in results:
                results[var] = {"variable": var}
            # Reported at full precision. These used to be rounded to four
            # decimals, which is not a display choice -- it is the value the
            # caller gets back. On this fixture that capped agreement with
            # quantreg::rq at 1.2e-04 on x2 and 7.8e-03 on x3, because a
            # coefficient near zero loses every significant digit to a fixed
            # number of decimal places. Rounding for display belongs in
            # `.summary()` / `.to_latex()`, which already do it.
            results[var][f"Q({q})"] = float(row["coefficient"])
            results[var][f"SE({q})"] = float(row["se"])

    return pd.DataFrame(list(results.values()))


# ======================================================================
# Internal
# ======================================================================


def _qreg_fit(Y: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    """Solve quantile regression via linear programming (interior point)."""
    n, k = X.shape

    # Reformulate as LP:
    # min tau * 1'u + (1-tau) * 1'v
    # s.t. X β + u - v = Y, u >= 0, v >= 0
    # where u = max(residual, 0) and v = max(-residual, 0)

    c = np.concatenate([np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)])

    # Equality: X β + I u - I v = Y. Stored sparse: the two identity blocks
    # are 2n nonzeros, not 2n^2 dense entries (n = 5,000 would otherwise
    # allocate 400 MB per solve). The LP is unchanged.
    from scipy import sparse

    eye = sparse.identity(n, format="csc")
    A_eq = sparse.hstack([sparse.csc_matrix(X), eye, -eye], format="csc")
    b_eq = Y

    # Bounds: β unbounded, u >= 0, v >= 0
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)

    try:
        # HiGHS interior point with its default crossover returns the same
        # basic (vertex) solution as dual simplex, ~45x faster at
        # n = 100,000 (3 s vs 143 s). The old maxiter=5000 cap made the
        # simplex fail from n ~ 20,000 and dropped every fit to IRLS.
        result = linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs-ipm",
        )
        if result.success:
            return _as_float_array(result.x[:k])
        _lp_note = f"linprog did not converge (status={result.status})"
    except Exception as exc:  # pragma: no cover - solver-dependent
        _lp_note = f"linprog raised {type(exc).__name__}: {exc}"

    # Fallback: iteratively reweighted least squares. The IRLS solution is
    # an approximation to the exact LP quantile fit, so switching solvers
    # must be loud (§7), not silent.
    warnings.warn(
        f"quantile regression LP solver failed ({_lp_note}); falling back "
        "to iteratively reweighted least squares. Coefficients may differ "
        "slightly from the exact linear-programming solution.",
        RuntimeWarning,
        stacklevel=2,
    )
    return _qreg_irls(Y, X, tau)


def _qreg_irls(
    Y: np.ndarray,
    X: np.ndarray,
    tau: float,
    max_iter: int = 50,
) -> np.ndarray:
    """IRLS fallback for quantile regression."""
    n, k = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]

    for _ in range(max_iter):
        resid = Y - X @ beta
        w = np.where(resid >= 0, tau, 1 - tau)
        w = w / (np.abs(resid) + 1e-6)
        Xw = X * w[:, None]
        try:
            beta_new = np.linalg.solve(Xw.T @ X, Xw.T @ Y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    return _as_float_array(beta)


def _hall_sheather(n: int, tau: float, alpha: float) -> float:
    """Hall-Sheather (1988) bandwidth, as Stata qreg and quantreg use it."""
    x0 = stats.norm.ppf(tau)
    f0 = stats.norm.pdf(x0)
    z = stats.norm.ppf(1 - alpha / 2)
    return float(
        n ** (-1 / 3) * z ** (2 / 3) * (1.5 * f0**2 / (2 * x0**2 + 1)) ** (1 / 3)
    )


def _qreg_vcov(
    Y: np.ndarray,
    X: np.ndarray,
    resid: np.ndarray,
    tau: float,
    kind: str,
    *,
    alpha: float,
) -> Tuple[np.ndarray, float]:
    """Koenker (2005, sec. 3.4) sparsity-based covariances of ``b(tau)``.

    Returns ``(V, h)``. All kinds refit the quantile regression at
    ``tau +/- h`` (Hall-Sheather ``h``); see :func:`qreg` for which
    reference convention each ``kind`` reproduces.
    """
    n = X.shape[0]
    h = _hall_sheather(n, tau, 0.05 if kind == "nid" else alpha)
    if kind == "nid":
        while tau - h < 0 or tau + h > 1:
            h /= 2
    elif tau - h <= 0 or tau + h >= 1:
        raise MethodIncompatibility(
            f"qreg: the bandwidth h={h:.4g} puts tau +/- h outside (0, 1); "
            "the sparsity cannot be estimated at this quantile and n.",
            recovery_hint="Use vce='nid' (halves h) or a bootstrap.",
            diagnostics={"tau": tau, "bandwidth": h},
        )
    b_lo = _qreg_fit(Y, X, tau - h)
    b_hi = _qreg_fit(Y, X, tau + h)
    XtX = X.T @ X
    if kind == "iid":
        s = (float(np.mean(X @ b_hi)) - float(np.mean(X @ b_lo))) / (2 * h)
        return tau * (1 - tau) * s**2 * np.linalg.inv(XtX), h
    dyhat = X @ (b_hi - b_lo)
    if kind == "nid":
        f = np.maximum(0.0, 2 * h / (dyhat - np.sqrt(np.finfo(float).eps)))
    else:  # Stata: a non-positive fitted difference gives zero density
        f = np.where(dyhat > np.sqrt(np.finfo(float).eps), 2 * h / dyhat, 0.0)
    H = np.linalg.inv((X * f[:, None]).T @ X)
    return tau * (1 - tau) * H @ XtX @ H, h


def _qreg_se(
    Y: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
    resid: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Powell (1991) kernel sandwich SE for quantile regression."""
    n, k = X.shape

    # Bandwidth (Silverman rule)
    h = 1.06 * np.std(resid) * n ** (-1 / 5)
    h = max(h, 1e-6)

    # Kernel density of residuals at 0
    f0 = np.mean(stats.norm.pdf(resid / h)) / h
    f0 = max(f0, 1e-6)

    # Powell (1991) iid kernel sandwich for QR:
    #   V = tau(1-tau) / f0² * (X'X)^{-1}
    # Reference: Koenker (2005, eq. 3.7). This is ``vce='powell'``: its
    # Silverman-bandwidth Gaussian kernel matches neither Stata qreg's
    # default (fitted sparsity) nor quantreg's "iid" / "nid" -- 3-7% off
    # both on the Track A fixture -- which is why it is no longer the
    # default (1.32). Earlier versions of this
    # file divided by an extra factor of n, producing SE that were
    # smaller by sqrt(n) (~20x at n=500) and meaningless inference.
    XtX_inv = np.linalg.pinv(X.T @ X)
    vcov = tau * (1 - tau) / (f0**2) * XtX_inv

    return _as_float_array(np.sqrt(np.maximum(np.diag(vcov), 1e-20)))


def _pseudo_r2(Y: np.ndarray, resid: np.ndarray, tau: float) -> float:
    """Koenker-Machado (1999) pseudo R² for quantile regression."""

    def rho(u: np.ndarray) -> np.ndarray:
        return _as_float_array(u * (tau - (u < 0)))

    obj_full = np.sum(rho(resid))
    obj_null = np.sum(rho(Y - np.quantile(Y, tau)))
    return float(1 - obj_full / obj_null) if obj_null > 0 else 0.0


def _parse_formula(formula: str) -> Tuple[str, List[str]]:
    """Parse 'y ~ x1 + x2' into (y, [x1, x2])."""
    parts = formula.split("~")
    if len(parts) != 2:
        raise ValueError(f"Invalid formula: {formula}")
    y = parts[0].strip()
    x = [v.strip() for v in parts[1].split("+") if v.strip()]
    return y, x


# Citation
CausalResult._CITATIONS["qreg"] = (
    "@article{koenker1978regression,\n"
    "  title={Regression Quantiles},\n"
    "  author={Koenker, Roger and Bassett, Gilbert},\n"
    "  journal={Econometrica},\n"
    "  volume={46},\n"
    "  number={1},\n"
    "  pages={33--50},\n"
    "  year={1978},\n"
    "  publisher={Wiley}\n"
    "}"
)
