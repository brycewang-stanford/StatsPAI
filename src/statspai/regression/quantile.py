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
    cluster: Optional[str] = None,
    kernel_scale: str = "mad",
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
        * ``'kernel'`` -- the heteroskedasticity-robust Powell (1984)
          kernel sandwich of Stata ``qreg2`` (its default): uniform kernel
          on the residuals, bandwidth ``kappa [Phi^-1(tau + h) -
          Phi^-1(tau - h)]`` with Hall-Sheather ``h``.
        * ``'cluster'`` (or ``'cluster <var>'``, ``'cluster(<var>)'``) --
          the cluster-robust covariance of Parente and Santos Silva
          (2016) [@parente2016quantile], as Stata ``qreg2, cluster()``
          computes it; the same kernel ``A`` with the scores
          ``tau - 1(u <= 0)`` summed within clusters. Official ``qreg``
          has no cluster option; ``qreg2`` is the authors' own
          implementation and the reference (full covariance matrix within
          1e-13 at four quantiles).

        ``'kernel'`` and ``'cluster'`` report t(N - k) inference, as
        ``qreg2`` does.
    cluster : str, optional
        Cluster column; implies ``vce='cluster'``. Rows with a missing
        cluster id are dropped, as ``qreg2`` does.
    kernel_scale : {'mad', 'silverman'}, default 'mad'
        The ``kappa`` of the ``'kernel'`` / ``'cluster'`` bandwidth: the
        median absolute deviation of the residuals (``qreg2`` default) or
        Silverman's ``min(sd, IQR/1.34)`` (``qreg2, silverman``).

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

    kind, cluster = _parse_qreg_vce(vce, cluster)
    if kernel_scale not in ("mad", "silverman"):
        raise MethodIncompatibility(
            f"qreg: kernel_scale must be 'mad' or 'silverman'; got "
            f"{kernel_scale!r}.",
            diagnostics={"kernel_scale": kernel_scale},
        )
    cols = [y_name] + x_names + ([cluster] if cluster is not None else [])
    missing_cols = [c for c in cols if c not in data]
    if missing_cols:
        raise MethodIncompatibility(
            f"qreg: columns not found in data: {missing_cols}",
            diagnostics={"missing": missing_cols},
        )
    df = data[list(dict.fromkeys(cols))].dropna()
    if len(df) == 0:
        raise MethodIncompatibility(
            "qreg: no complete rows after dropping missing values.",
            diagnostics={"columns": cols},
        )
    Y = df[y_name].values.astype(float)
    X = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in x_names]
    )
    n, k = X.shape
    var_names = ["const"] + x_names

    # Solve quantile regression via linear programming
    beta = _qreg_fit(Y, X, quantile)
    resid = Y - X @ beta

    n_clusters = None
    if kind == "powell":
        se = _qreg_se(Y, X, beta, resid, quantile)
        bandwidth = None
    elif kind in ("kernel", "cluster"):
        groups = pd.factorize(df[cluster])[0] if kind == "cluster" else np.arange(n)
        n_clusters = int(groups.max()) + 1
        if kind == "cluster" and n_clusters < 2:
            raise MethodIncompatibility(
                "qreg: cluster-robust SEs need at least two clusters.",
                diagnostics={"cluster": cluster, "n_clusters": n_clusters},
            )
        vcov, bandwidth = _pss_vcov(X, resid, quantile, groups, kernel_scale)
        se = _as_float_array(np.sqrt(np.maximum(np.diag(vcov), 0.0)))
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
    if kind in ("kernel", "cluster"):
        # qreg2 keeps qreg's residual degrees of freedom: t(N - k).
        pvals = 2 * stats.t.sf(np.abs(z_stats), n - k)
        z_crit = stats.t.ppf(1 - alpha / 2, n - k)
    else:
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
    if kind != "powell":
        model_info["vcov"] = pd.DataFrame(vcov, index=var_names, columns=var_names)
    if kind in ("kernel", "cluster"):
        model_info.update(
            {
                "kernel_scale": kernel_scale,
                "df_inference": n - k,
                "reference": (
                    "Stata qreg2" + (f", cluster({cluster})" if cluster else "")
                ),
            }
        )
        if kind == "cluster":
            model_info.update({"cluster": cluster, "n_clusters": n_clusters})

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


_QREG_KINDS = ("iid", "robust", "nid", "powell", "kernel", "cluster")


def _parse_qreg_vce(
    vce: Optional[str], cluster: Optional[str]
) -> Tuple[str, Optional[str]]:
    """Resolve ``vce`` / ``cluster`` into ``(kind, cluster column)``."""
    import re

    raw = "" if vce is None else str(vce).strip()
    m = re.fullmatch(r"(?i)cluster\s*(?:\(\s*([^()\s]+)\s*\)|\s+(\S+))?", raw)
    if m:
        named = m.group(1) or m.group(2)
        if named is not None and cluster is not None and named != cluster:
            raise MethodIncompatibility(
                f"qreg: vce={vce!r} and cluster={cluster!r} name different "
                "cluster variables.",
                diagnostics={"vce": vce, "cluster": cluster},
            )
        cluster = named or cluster
        kind = "cluster"
    else:
        kind = raw.lower() or ("cluster" if cluster is not None else "iid")
    if kind not in _QREG_KINDS:
        raise MethodIncompatibility(
            f"qreg: vce must be one of {', '.join(map(repr, _QREG_KINDS))} "
            f"(or 'cluster <var>'); got {vce!r}.",
            diagnostics={"vce": vce},
        )
    if kind == "cluster" and cluster is None:
        raise MethodIncompatibility(
            "qreg: vce='cluster' needs a cluster column.",
            recovery_hint="Pass cluster='<column>' or vce='cluster <column>'.",
            diagnostics={"vce": vce},
        )
    if kind != "cluster" and cluster is not None:
        raise MethodIncompatibility(
            f"qreg: cluster={cluster!r} conflicts with vce={vce!r}.",
            recovery_hint="Drop vce= (cluster= implies vce='cluster').",
            diagnostics={"vce": vce, "cluster": cluster},
        )
    return kind, cluster


def _stata_percentile(x: np.ndarray, p: float) -> float:
    """``summarize, detail`` percentile: mean of the two straddling order
    statistics when ``n p / 100`` is an integer, else the next one up."""
    s = np.sort(x)
    pos = len(s) * p / 100.0
    whole = round(pos)
    if abs(pos - whole) < 1e-9:
        return float((s[int(whole) - 1] + s[int(whole)]) / 2)
    return float(s[int(np.floor(pos))])


def _pss_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    tau: float,
    groups: np.ndarray,
    scale: str,
    epsilon: float = 1e-7,
) -> Tuple[np.ndarray, float]:
    """Powell kernel sandwich, cluster-robust when ``groups`` repeat.

    ``V = D^{-1} A D^{-1}`` with ``D = sum 1(|u_i| < c) x_i x_i' / (2c)``
    and ``A = sum_g s_g s_g'``, ``s_g = sum_{i in g} (tau - 1(u_i <= 0)) x_i``
    (Parente and Santos Silva 2016). Conventions follow their Stata
    ``qreg2``: residuals below ``epsilon`` times the median absolute
    residual are set to zero (a basic solution has k exact zeros that the
    solver returns as ~1e-13); ``c = kappa [Phi^-1(tau + h) -
    Phi^-1(tau - h)]``, ``h`` Hall-Sheather at the 95% level, ``kappa`` the
    MAD of the residuals about their median or Silverman's
    ``min(sd, IQR / 1.34)``. Returns ``(V, c)``.
    """
    n, k = X.shape
    u = np.asarray(resid, dtype=float)
    au = np.abs(u)
    u = np.where(au < _stata_percentile(au, 50) * epsilon, 0.0, u)
    h = _hall_sheather(n, tau, 0.05)
    if tau + h > 1 or tau - h < 0:
        raise MethodIncompatibility(
            f"qreg: the bandwidth h={h:.4g} puts tau +/- h outside (0, 1); "
            "too few observations for this quantile.",
            diagnostics={"tau": tau, "bandwidth": h, "n": n},
        )
    if scale == "silverman":
        kappa = min(
            float(np.std(u, ddof=1)),
            (_stata_percentile(u, 75) - _stata_percentile(u, 25)) / 1.34,
        )
    else:
        kappa = _stata_percentile(np.abs(u - _stata_percentile(u, 50)), 50)
    c = kappa * float(stats.norm.ppf(tau + h) - stats.norm.ppf(tau - h))
    inside = np.abs(u) < c
    if c <= 0 or inside.sum() < k:
        raise MethodIncompatibility(
            f"qreg: the kernel bandwidth c={c:.4g} keeps {int(inside.sum())} "
            f"residuals, fewer than the {k} parameters; the density cannot "
            "be estimated.",
            diagnostics={"bandwidth": c, "inside": int(inside.sum())},
        )
    D = (X * (inside / (2 * c))[:, None]).T @ X
    scores = X * (tau - (u <= 0))[:, None]
    sg = np.zeros((int(groups.max()) + 1, k))
    np.add.at(sg, groups, scores)
    D_inv = np.linalg.inv(D)
    V = D_inv @ (sg.T @ sg) @ D_inv
    return (V + V.T) / 2, c


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
