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
from ..core.results import CausalResult, EconometricResults
from ..exceptions import MethodIncompatibility


def _as_float_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=float)


class QuantileRegressionResult(CausalResult):
    """``sp.qreg`` result: a ``CausalResult`` whose coefficient accessors
    cover every regressor.

    ``estimate`` / ``se`` / ``ci`` stay the first regressor's (the headline
    of the causal-result interface), but ``params``, ``std_errors``,
    ``tvalues``, ``pvalues``, ``vcov()`` and ``conf_int()`` report the whole
    coefficient table, as Stata's ``qreg`` and ``sp.regress`` do. Through
    1.32 they held the first regressor alone (top-5 replication list), so a
    regression table of a ``qreg`` fit showed one row.
    """

    def _coef_table(self) -> pd.DataFrame:
        return self.detail.set_index("variable")

    @property
    def params(self) -> pd.Series:
        return self._coef_table()["coefficient"].rename(None)

    @property
    def std_errors(self) -> pd.Series:
        return self._coef_table()["se"].rename(None)

    @property
    def tvalues(self) -> pd.Series:
        return self._coef_table()["t"].rename(None)

    @property
    def pvalues(self) -> pd.Series:
        return self._coef_table()["pvalue"].rename(None)

    def vcov(self) -> pd.DataFrame:
        names = list(self._coef_table().index)
        V = self.model_info.get("vcov")
        if isinstance(V, pd.DataFrame):
            return V.loc[names, names]
        # vce='powell' stores SEs only; its sandwich is not kept.
        se = self.std_errors.to_numpy()
        return pd.DataFrame(np.diag(se**2), index=names, columns=names)

    def conf_int(self, alpha: Optional[float] = None) -> pd.DataFrame:
        a = self.alpha if alpha is None else alpha
        crit = stats.t.ppf(1 - a / 2, self.model_info["df_inference"])
        b, se = self.params, self.std_errors
        return pd.DataFrame({0: b - crit * se, 1: b + crit * se})


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
    weights: Optional[object] = None,
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
    vce : {None, 'iid', 'robust', 'nid', 'kernel', 'ker', 'cluster', 'powell'}, optional
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

        * ``'ker'`` -- R ``quantreg::summary.rq(se="ker")``: Powell's
          sandwich with a Gaussian kernel, ``H = (sum f_i x_i x_i')^{-1}``,
          ``f_i = phi(u_i / c) / c`` and ``c = [Phi^-1(tau + h) -
          Phi^-1(tau - h)] min(sd(u), IQR(u) / 1.34)``.

        Every choice reports t(N - k) p-values and intervals, as Stata
        ``qreg`` / ``qreg2`` and R ``quantreg::summary.rq`` do (normal
        before 1.32).
    cluster : str, optional
        Cluster column; implies ``vce='cluster'``. Rows with a missing
        cluster id are dropped, as ``qreg2`` does.
    kernel_scale : {'mad', 'silverman'}, default 'mad'
        The ``kappa`` of the ``'kernel'`` / ``'cluster'`` bandwidth: the
        median absolute deviation of the residuals (``qreg2`` default) or
        Silverman's ``min(sd, IQR/1.34)`` (``qreg2, silverman``).

    weights : str or array-like, optional
        Sampling weights: the fit minimises ``sum w_i rho_tau(y_i -
        x_i'b)``. The covariance follows ``quantreg``: every formula above
        is applied to the rescaled data ``(w_i y_i, w_i x_i)``, which for
        the sandwich kinds is the weighted bread ``sum w_i f_i x_i x_i'``
        around the meat ``sum w_i^2 x_i x_i'``. With weights and no
        ``vce``, the default is ``'robust'`` rather than ``'iid'``: an iid
        sparsity has no meaning for a weighted sample (Stata ``qreg``
        switches to ``vce(robust)`` under ``pweight`` for the same reason).
        Weights must be positive; rows with a missing weight are dropped.

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
        if any(name not in data for name in [y_name] + x_names):
            # transformed, categorical or interaction terms: built as columns
            from ..core.utils import formula_to_columns

            data, y_name, x_names = formula_to_columns(formula, data)
    elif y is not None and x is not None:
        y_name, x_names = y, x
    else:
        raise MethodIncompatibility("Provide either formula or (y, x)")

    kind, cluster = _parse_qreg_vce(vce, cluster)
    if weights is not None and vce is None and cluster is None:
        kind = "robust"
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
    df = data[list(dict.fromkeys(cols))]
    w_obs: Optional[np.ndarray] = None
    if weights is not None:
        if isinstance(weights, str):
            if weights not in data:
                raise MethodIncompatibility(
                    f"qreg: weights column {weights!r} not found in data.",
                    diagnostics={"weights": weights},
                )
            w_all = data[weights].to_numpy(dtype=float)
        else:
            w_all = np.asarray(weights, dtype=float).ravel()
            if w_all.shape[0] != len(data):
                raise MethodIncompatibility(
                    f"qreg: weights has {w_all.shape[0]} entries for "
                    f"{len(data)} rows.",
                )
        df = df.assign(**{"__qreg_w__": w_all})
    df = df.dropna()
    if len(df) == 0:
        raise MethodIncompatibility(
            "qreg: no complete rows after dropping missing values.",
            diagnostics={"columns": cols},
        )
    Y = df[y_name].values.astype(float)
    X = np.column_stack(
        [np.ones(len(df))] + [df[v].values.astype(float) for v in x_names]
    )
    if weights is not None:
        w_obs = df["__qreg_w__"].to_numpy(dtype=float)
        if not np.all(np.isfinite(w_obs)) or np.any(w_obs <= 0):
            raise MethodIncompatibility(
                "qreg: weights must be finite and strictly positive.",
                recovery_hint="Drop the rows with a zero or negative weight.",
            )
        # quantreg's rq.wfit: the weighted check loss is the unweighted
        # one on (w y, w x), and summary.rq works on the same rescaling
        Y = Y * w_obs
        X = X * w_obs[:, None]
    n, k = X.shape
    var_names = ["const"] + x_names

    # Solve quantile regression via linear programming
    beta = _qreg_fit(Y, X, quantile)
    resid = Y - X @ beta

    n_clusters = None
    if kind == "powell":
        se = _qreg_se(Y, X, beta, resid, quantile)
        bandwidth = None
    elif kind == "ker":
        vcov, bandwidth = _quantreg_ker_vcov(X, resid, quantile)
        se = _as_float_array(np.sqrt(np.maximum(np.diag(vcov), 0.0)))
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

    # Stata qreg / qreg2 and R quantreg::summary.rq all refer the statistic
    # to t(N - k); before 1.32 this used the normal distribution.
    t_stats = beta / se
    pvals = 2 * stats.t.sf(np.abs(t_stats), n - k)
    t_crit = stats.t.ppf(1 - alpha / 2, n - k)

    detail = pd.DataFrame(
        {
            "variable": var_names,
            "coefficient": beta,
            "se": se,
            "t": t_stats,
            "z": t_stats,  # pre-1.32 column name, kept for compatibility
            "pvalue": pvals,
        }
    )

    # Main estimate: first regressor (after constant)
    main_coef = float(beta[1])
    main_se = float(se[1])
    main_p = float(pvals[1])
    ci = (main_coef - t_crit * main_se, main_coef + t_crit * main_se)

    model_info = {
        "quantile": quantile,
        "pseudo_r2": _pseudo_r2(Y, resid, quantile, w_obs),
        "weighted": w_obs is not None,
        "n_obs": n,
        "vce": kind,
        "bandwidth": bandwidth,
        "df_inference": n - k,
    }
    if kind != "powell":
        model_info["vcov"] = pd.DataFrame(vcov, index=var_names, columns=var_names)
    if kind in ("kernel", "cluster"):
        model_info.update(
            {
                "kernel_scale": kernel_scale,
                "reference": (
                    "Stata qreg2" + (f", cluster({cluster})" if cluster else "")
                ),
            }
        )
        if kind == "cluster":
            model_info.update({"cluster": cluster, "n_clusters": n_clusters})

    return QuantileRegressionResult(
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
    reps: Optional[int] = None,
    seed: Optional[int] = None,
    cluster: Optional[str] = None,
) -> "pd.DataFrame | EconometricResults":
    """
    Simultaneous quantile regression at multiple quantiles.

    Equivalent to Stata's ``sqreg y x, quantiles(10 25 50 75 90)``. With
    ``reps`` the covariance is Stata's: one bootstrap across all quantiles
    jointly, so coefficients can be compared across quantiles
    (``sp.test(r, "q25:x = q75:x")``, Stata ``test [q25]x = [q75]x``).

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    x : list of str
    quantiles : list of float, optional
        Default: [0.1, 0.25, 0.5, 0.75, 0.9].
    alpha : float, default 0.05
    reps : int, optional
        Bootstrap replications for the joint covariance (Stata ``sqreg``
        always bootstraps, ``reps(20)`` by default). Without ``reps`` each
        quantile keeps its own analytic ``qreg`` SE and the return value is
        the table below.
    seed : int, optional
        Bootstrap seed.
    cluster : str, optional
        Resample clusters instead of observations.

    Returns
    -------
    pd.DataFrame or EconometricResults
        Without ``reps``: rows are variables, columns the coefficient and SE
        at each quantile. With ``reps``: an ``EconometricResults`` whose
        coefficients are named ``q<100 tau>:<variable>`` (``q25:x``), with
        the bootstrap covariance across quantiles in ``vcov()``.

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

    if reps is not None:
        return _sqreg_bootstrap(
            data, y, list(x), list(quantiles), alpha, int(reps), seed, cluster
        )

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
    """Solve quantile regression by linear programming.

    The dual LP [@koenker2005quantile] -- ``max y'a`` subject to
    ``X'a = (1 - tau) X'1``, ``0 <= a <= 1`` -- has ``k`` equality rows
    instead of ``n``; HiGHS' interior point with crossover returns its
    vertex, and ``b(tau)`` is the vector of equality multipliers. It is the
    same solution as the primal LP (to 1e-15, ties included) and 5-8x
    faster (n = 20,000, k = 11: 0.5 s vs 3.8 s), which matters because every
    ``qreg`` SE refits at ``tau +/- h``. The primal LP is the fallback.
    """
    n, k = X.shape
    try:
        dual = linprog(
            -Y,
            A_eq=X.T,
            b_eq=(1.0 - tau) * X.sum(axis=0),
            bounds=(0.0, 1.0),
            method="highs-ipm",
        )
        if dual.success and dual.eqlin is not None:
            beta = -np.asarray(dual.eqlin.marginals, dtype=float)
            if np.all(np.isfinite(beta)):
                return _as_float_array(beta)
    except Exception:  # pragma: no cover - solver-dependent; primal below
        pass

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


_QREG_KINDS = ("iid", "robust", "nid", "powell", "kernel", "ker", "cluster")


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


def _quantreg_ker_vcov(
    X: np.ndarray, resid: np.ndarray, tau: float
) -> Tuple[np.ndarray, float]:
    """Powell's kernel sandwich as ``quantreg::summary.rq(se="ker")``.

    Gaussian kernel; bandwidth ``c = [Phi^-1(tau + h) - Phi^-1(tau - h)]
    min(sd(u), IQR(u) / 1.34)`` with Hall-Sheather ``h`` at the 95% level
    and R's default (type 7) quartiles. Returns ``(V, c)``.
    """
    n = X.shape[0]
    h = _hall_sheather(n, tau, 0.05)
    if tau + h > 1 or tau - h < 0:
        raise MethodIncompatibility(
            f"qreg: the bandwidth h={h:.4g} puts tau +/- h outside (0, 1); "
            "too few observations for this quantile.",
            diagnostics={"tau": tau, "bandwidth": h, "n": n},
        )
    q75, q25 = np.quantile(resid, [0.75, 0.25])
    c = float(stats.norm.ppf(tau + h) - stats.norm.ppf(tau - h)) * min(
        float(np.std(resid, ddof=1)), float(q75 - q25) / 1.34
    )
    f = stats.norm.pdf(resid / c) / c
    H = np.linalg.inv((X * f[:, None]).T @ X)
    V = tau * (1 - tau) * H @ (X.T @ X) @ H
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


def _pseudo_r2(
    Y: np.ndarray,
    resid: np.ndarray,
    tau: float,
    w: Optional[np.ndarray] = None,
) -> float:
    """Koenker-Machado (1999) pseudo R² for quantile regression.

    With weights, ``Y`` and ``resid`` arrive already multiplied by ``w``;
    the null model is the weighted ``tau`` quantile of the outcome.
    """

    def rho(u: np.ndarray) -> np.ndarray:
        return _as_float_array(u * (tau - (u < 0)))

    obj_full = np.sum(rho(resid))
    if w is not None:
        y = Y / w
        order = np.argsort(y)
        cum = np.cumsum(w[order]) / np.sum(w)
        q = y[order][min(int(np.searchsorted(cum, tau)), len(y) - 1)]
        obj_null = np.sum(w * rho(y - q))
        return float(1 - obj_full / obj_null) if obj_null > 0 else 0.0
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


def _quantile_label(q: float) -> str:
    """Stata's equation name for a quantile: 0.25 -> 'q25', 0.125 -> 'q12.5'."""
    v = round(100 * q, 6)
    return f"q{int(v)}" if v == int(v) else f"q{v:g}"


def _sqreg_bootstrap(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    quantiles: List[float],
    alpha: float,
    reps: int,
    seed: Optional[int],
    cluster: Optional[str],
) -> "EconometricResults":
    """Stata ``sqreg ..., reps()``: all quantiles bootstrapped jointly."""
    from ..inference.bootstrap import bootstrap

    cols = [y] + list(x) + ([cluster] if cluster else [])
    df = data[cols].dropna().reset_index(drop=True)
    names = ["_cons"] + list(x)

    def stat(d: pd.DataFrame) -> pd.Series:
        Y = d[y].to_numpy(dtype=float)
        X = np.column_stack([np.ones(len(d))] + [d[v].to_numpy(float) for v in x])
        out = {}
        for q in quantiles:
            b = _qreg_fit(Y, X, q)
            for name, val in zip(names, b):
                out[f"{_quantile_label(q)}:{name}"] = val
        return pd.Series(out)

    res = bootstrap(df, stat, n_boot=reps, cluster=cluster, alpha=alpha, seed=seed)
    res.model_info.update(
        {
            "model_type": "Simultaneous quantile regression",
            "quantiles": list(quantiles),
            "method": f"sqreg, bootstrap ({res.model_info['n_boot']} reps)"
            + (f", cluster({cluster})" if cluster else ""),
        }
    )
    return res
