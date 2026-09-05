"""
Multiplicative interaction models with diagnostics (interflex).

Native port of the estimation core of Hainmueller, Mummolo and Xu's
**interflex** (R and Stata): the conditional marginal effect of a
treatment ``D`` on ``Y`` across a moderator ``X``, estimated three ways.

- ``estimator="linear"``: the conventional ``Y ~ X + D + D*X (+ Z)``
  model; ``ME(x) = b_D + b_DX * x`` with delta-method standard errors
  from a heteroskedasticity-robust (HC1) covariance, plus the average
  treatment / marginal effect over the sample.
- ``estimator="binning"``: the moderator is cut into ``nbins`` bins at
  its sample quantiles and a fully interacted model is fitted with one
  intercept, slope, treatment effect and treatment-by-moderator slope
  per bin (each centred at the bin median), so the treatment effect at
  each bin median is a single coefficient. The Wald and likelihood-ratio
  tests of the linear-interaction restriction against this model and the
  L-kurtosis of ``X`` are the diagnostics of the paper.
- ``estimator="kernel"``: a local linear regression of ``Y`` on
  ``D`` and ``X - x`` at every evaluation point ``x`` with Gaussian
  kernel weights ``phi((X - x)/h(x))``, where ``h(x)`` adapts the
  bandwidth to the moderator's density exactly as ``interflex`` does
  (``h(x) = bw * sqrt(geometric mean density / density(x))``, with the
  density from R's ``stats::density`` algorithm, ported here).

Every convention -- R's type-7 quantiles and right-closed ``cut`` for
the bins, the bin medians as centring points, HC1 covariances, the
linear-binning-plus-FFT density estimate, the adaptive bandwidth, the
zero-weight guard -- follows the R package so that Track A module 87
can compare the two implementations on identical bytes.

References
----------
[@hainmueller2019much] Hainmueller, Mummolo and Xu (2019), *Political
Analysis* 27(2), 163--192.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult

__all__ = ["interflex", "interflex_plot"]


# ----------------------------------------------------------------------
# Ports of the R primitives interflex relies on
# ----------------------------------------------------------------------


def r_density(
    x: np.ndarray, n_user: int = 512, cut: float = 3.0, bw: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Port of ``stats::density.default`` (Gaussian kernel, ``bw.nrd0``).

    Linear binning of the sample onto ``n`` grid points over
    ``[min - 3bw - 4bw, max + 3bw + 4bw]``, circular convolution with the
    Gaussian kernel by FFT, and linear interpolation back to the
    ``n_user`` output points. Returns ``(x_grid, y, bw)``.
    """
    x = np.asarray(x, dtype=float)
    nx = x.size
    if bw is None:
        sd = float(np.std(x, ddof=1))
        iqr = float(np.subtract(*np.percentile(x, [75, 25])))
        lo = min(sd, iqr / 1.34)
        if not lo:
            lo = sd or abs(x[0]) or 1.0
        bw = 0.9 * lo * nx ** (-0.2)
    n = 512 if n_user <= 512 else int(2 ** np.ceil(np.log2(n_user)))
    frm = x.min() - cut * bw
    to = x.max() + cut * bw
    lo = frm - 4.0 * bw
    up = to + 4.0 * bw
    # BinDist: linear binning with unit total mass onto n points, padded to 2n.
    weights = np.full(nx, 1.0 / nx)
    y = np.zeros(2 * n)
    xdelta = (up - lo) / (n - 1)
    xpos = (x - lo) / xdelta
    ix = np.floor(xpos).astype(int)
    fx = xpos - ix
    for xi, fi, wi in zip(ix, fx, weights):
        if 0 <= xi <= n - 2:
            y[xi] += wi * (1 - fi)
            y[xi + 1] += wi * fi
        elif xi == -1:
            y[0] += wi * fi
        elif xi == n - 1:
            y[xi] += wi * (1 - fi)
    # R >= 4.4 (old.coords = FALSE): the kernel grid uses the binning
    # spacing (up - lo)/(n - 1), i.e. 2n points from 0 to (2n-1)/(n-1)*(up-lo).
    kords = np.linspace(0.0, (2 * n - 1) / (n - 1) * (up - lo), 2 * n)
    kords[n + 1 : 2 * n] = -kords[n - 1 : 0 : -1]
    kords = stats.norm.pdf(kords, scale=bw)
    # numpy's ifft carries the 1/(2n) that R applies as "/ length(y)".
    conv = np.fft.ifft(np.fft.fft(y) * np.conj(np.fft.fft(kords)))
    dens = np.maximum(0.0, np.real(conv)[:n])
    xords = np.linspace(lo, up, n)
    xout = np.linspace(frm, to, n_user)
    yout = np.interp(xout, xords, dens)
    return xout, yout, float(bw)


def l_kurtosis(x: np.ndarray) -> float:
    """Sample L-kurtosis ``tau_4 = l_4 / l_2`` (Hosking's unbiased estimators)."""
    xs = np.sort(np.asarray(x, dtype=float))
    n = xs.size
    i = np.arange(1, n + 1, dtype=float)
    b0 = xs.mean()
    b1 = np.sum((i - 1) / (n - 1) * xs) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * xs) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * xs) / n
    l2 = 2 * b1 - b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return float(l4 / l2)


def _r_cut_groups(x: np.ndarray, cuts: np.ndarray) -> np.ndarray:
    """R ``cut(x, breaks, labels = FALSE)`` (right-closed) with the minimum
    assigned to the first bin, as interflex does."""
    g = np.searchsorted(cuts, x, side="left")  # cuts[g-1] < x <= cuts[g]
    g = np.clip(g, 1, len(cuts) - 1)
    g[x == x.min()] = 1
    return g


def _ols(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray], cov_type: str):
    """(W)LS with HC1 or homoscedastic covariance, R ``lm`` + ``sandwich`` conventions."""
    n, k = X.shape
    if w is None:
        XtX = X.T @ X
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        bread = np.linalg.pinv(XtX)
        if cov_type == "robust":
            meat = (X * (resid**2)[:, None]).T @ X
            V = bread @ meat @ bread * (n / (n - k))
        else:
            sigma2 = float(resid @ resid) / (n - k)
            V = bread * sigma2
    else:
        sw = np.sqrt(w)
        Xw = X * sw[:, None]
        yw = y * sw
        beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        resid = y - X @ beta
        bread = np.linalg.pinv(Xw.T @ Xw)
        if cov_type == "robust":
            # sandwich on a weighted lm: scores w_i e_i x_i.
            meat = (X * ((w * resid) ** 2)[:, None]).T @ X
            V = bread @ meat @ bread * (n / (n - k))
        else:
            sigma2 = float(np.sum(w * resid**2)) / (n - k)
            V = bread * sigma2
    return beta, V, resid


def _gaussian_loglik(resid: np.ndarray, w: Optional[np.ndarray]) -> float:
    """``logLik.lm`` for an unweighted / weighted least-squares fit."""
    n = resid.size
    if w is None:
        rss = float(resid @ resid)
        return float(-0.5 * n * (np.log(2 * np.pi) + 1 - np.log(n) + np.log(rss)))
    rss = float(np.sum(w * resid**2))
    return float(
        0.5 * np.sum(np.log(w))
        - 0.5 * n * (np.log(2 * np.pi) + 1 - np.log(n) + np.log(rss))
    )


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def interflex(
    data: pd.DataFrame,
    y: str,
    d: str,
    x: str,
    z: Optional[Sequence[str]] = None,
    estimator: str = "binning",
    nbins: int = 3,
    cutoffs: Optional[Sequence[float]] = None,
    bw: Optional[float] = None,
    neval: int = 50,
    x_eval: Optional[Sequence[float]] = None,
    weights: Optional[str] = None,
    treat_type: Optional[str] = None,
    wald: bool = True,
    wald_full_moderate: bool = True,
    wald_test: str = "chisq",
    adaptive: bool = True,
    vce: str = "robust",
    n_boot: int = 200,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Conditional marginal effects of ``d`` across a moderator ``x``
    (Hainmueller, Mummolo and Xu 2019).

    Parameters
    ----------
    data : pd.DataFrame
    y, d, x : str
        Outcome, treatment (binary 0/1 or continuous) and moderator.
    z : sequence of str, optional
        Additional covariates entering linearly.
    estimator : {'binning', 'linear', 'kernel'}, default 'binning'
    nbins : int, default 3
        Number of moderator bins (cut at sample quantiles) for the
        binning estimator; ``cutoffs`` overrides the quantiles.
    bw : float, optional
        Kernel bandwidth on the moderator's scale (before the adaptive
        density scaling). Required for ``estimator='kernel'``; interflex's
        cross-validated choice is not ported.
    neval : int, default 50
        Evaluation points, equally spaced over the moderator's range.
    x_eval : sequence, optional
        Explicit evaluation points.
    vce : {'robust', 'homoscedastic', 'bootstrap'}, default 'robust'
        Variance construction. ``'robust'`` is the HC1 sandwich
        (interflex's default) and ``'homoscedastic'`` the classical
        covariance for the linear and binning models and the Wald test;
        ``'bootstrap'`` (kernel estimator only) draws nonparametric
        bootstrap standard errors, the linear and binning estimators
        always reporting delta-method SEs.
    weights : str, optional
        Observation weights column.
    treat_type : {'discrete', 'continuous'}, optional
        Inferred from ``d`` when omitted (two distinct values = discrete).
    wald : bool, default True
        Report the Wald and LR tests of the linear interaction against
        the binning model (``estimator='binning'`` or ``'linear'``).
    wald_full_moderate : bool, default True
        Whether the fully interacted model behind the Wald / LR tests also
        interacts the covariates ``z`` with the bins (R interflex). The
        Stata command leaves the covariates uninteracted; set ``False`` to
        reproduce its test.
    wald_test : {'chisq', 'F'}, default 'chisq'
        Reference distribution of the Wald statistic: chi-square (R
        ``lmtest::waldtest(test = "Chisq")``) or the F distribution with
        the full model's residual degrees of freedom (Stata interflex).
    adaptive : bool, default True
        Kernel estimator only. ``True`` scales the bandwidth at each
        evaluation point by the moderator's density as the R package
        does (``h(x) = bw * sqrt(geometric-mean density / density(x))``);
        ``False`` uses the fixed Gaussian kernel ``phi((X - x)/bw)`` of
        the Stata ``interflex`` command.
    n_boot, seed : int
        Bootstrap replications and seed.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``.estimate`` is the average treatment (discrete ``d``) or
        marginal (continuous ``d``) effect over the sample under the
        chosen estimator; ``.detail`` is the marginal-effect table on
        the evaluation grid (``linear`` / ``kernel``) or at the bin
        medians (``binning``); ``.model_info["tests"]`` carries the
        L-kurtosis of ``x`` and the Wald / LR p-values.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.interflex(df, y="Y", d="D", x="X", estimator="binning")  # doctest: +SKIP
    >>> res.model_info["tests"]["p_wald"]  # doctest: +SKIP

    References
    ----------
    [@hainmueller2019much]
    """
    estimator = str(estimator).lower()
    if estimator not in {"linear", "binning", "kernel"}:
        raise ValueError("estimator must be 'linear', 'binning', or 'kernel'")
    if vce not in {"robust", "homoscedastic", "bootstrap"}:
        raise ValueError("vce must be 'robust', 'homoscedastic', or 'bootstrap'")
    if estimator == "kernel" and (bw is None or float(bw) <= 0):
        raise ValueError("estimator='kernel' needs a positive bandwidth bw=")
    if vce == "bootstrap" and estimator != "kernel":
        raise ValueError(
            "vce='bootstrap' is available for estimator='kernel'; the linear "
            "and binning estimators report delta-method SEs"
        )
    cov_type = "homoscedastic" if vce == "homoscedastic" else "robust"
    zc = list(z) if z else []
    cols = [y, d, x, *zc] + ([weights] if weights else [])
    for c in cols:
        if c not in data.columns:
            raise ValueError(f"column {c!r} not in data")
    df = data[cols].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(dtype=float)
    D = df[d].to_numpy(dtype=float)
    X = df[x].to_numpy(dtype=float)
    Z = df[zc].to_numpy(dtype=float) if zc else np.zeros((len(df), 0))
    W = df[weights].to_numpy(dtype=float) if weights else None
    n = len(df)
    if treat_type is None:
        treat_type = "discrete" if len(np.unique(D)) == 2 else "continuous"
    if treat_type == "discrete" and not set(np.unique(D)).issubset({0.0, 1.0}):
        raise ValueError("discrete d must be coded 0/1 (0 = base group)")
    if x_eval is None:
        grid = np.linspace(X.min(), X.max(), int(neval))
    else:
        grid = np.asarray(list(x_eval), dtype=float)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    tests: Dict[str, Any] = {"treat_type": treat_type, "x_lkurtosis": l_kurtosis(X)}
    model_info: Dict[str, Any] = {
        "estimator": estimator,
        "treat_type": treat_type,
        "vcov_type": cov_type,
        "vce": vce,
        "n_obs": int(n),
        "x_eval": grid,
        "covariates": zc,
    }

    # ---- shared: linear interaction model -------------------------------
    Xlin = np.column_stack([np.ones(n), X, D, D * X, Z])
    names_lin = ["(Intercept)", x, d, "DX", *zc]
    b_lin, V_lin, e_lin = _ols(Xlin, Y, W, cov_type)

    # ---- binning design (also needed for the Wald / LR diagnostics) -------
    if cutoffs is None:
        cuts = np.quantile(X, np.linspace(0, 1, nbins + 1))
        cuts = np.unique(cuts)
    else:
        inner = [c for c in cutoffs if X.min() < c < X.max()]
        cuts = np.unique(np.concatenate([[X.min()], inner, [X.max()]]))
    groups = _r_cut_groups(X, cuts)
    bins = sorted(np.unique(groups))
    nb = len(bins)
    x0 = np.array([np.median(X[groups == g]) for g in bins])
    model_info["bins"] = {
        "cutoffs": cuts,
        "medians": x0,
        "counts": np.array([int(np.sum(groups == g)) for g in bins]),
    }

    if wald and estimator in {"linear", "binning"} and nb > 1:
        # Restricted model: Y ~ X + D + DX (+ Z); full model adds, for
        # bins 2..nb, G_i, G_i*X, D*G_i, D*G_i*X (+ Z*G_i, Z*G_i*X).
        extra = []
        for g in bins[1:]:
            Gi = (groups == g).astype(float)
            extra += [Gi, Gi * X, Gi * D, Gi * D * X]
            if wald_full_moderate:
                for j in range(Z.shape[1]):
                    extra += [Gi * Z[:, j], Gi * Z[:, j] * X]
        Xfull = np.column_stack([Xlin, *extra])
        b_full, V_full, e_full = _ols(Xfull, Y, W, cov_type)
        k0 = Xlin.shape[1]
        b_x = b_full[k0:]
        V_x = V_full[k0:, k0:]
        try:
            wald_stat = float(b_x @ np.linalg.solve(V_x, b_x))
        except np.linalg.LinAlgError:
            wald_stat = float(b_x @ np.linalg.pinv(V_x) @ b_x)
        df_w = len(b_x)
        if str(wald_test).lower() == "f":
            p_wald = float(stats.f.sf(wald_stat / df_w, df_w, n - Xfull.shape[1]))
        elif str(wald_test).lower() == "chisq":
            p_wald = float(stats.chi2.sf(wald_stat, df_w))
        else:
            raise ValueError("wald_test must be 'chisq' or 'F'")
        ll0 = _gaussian_loglik(e_lin, W)
        ll1 = _gaussian_loglik(e_full, W)
        lr_stat = 2.0 * (ll1 - ll0)
        p_lr = float(stats.chi2.sf(lr_stat, df_w))
        tests.update(
            {
                "wald_stat": wald_stat,
                "p_wald": p_wald,
                "lr_stat": lr_stat,
                "p_lr": p_lr,
                "df": int(df_w),
                "wald_full_moderate": bool(wald_full_moderate),
                "wald_test": str(wald_test),
            }
        )

    # ---- estimator-specific marginal effects ------------------------------
    if estimator == "linear":
        iD, iDX = 2, 3
        me = b_lin[iD] + b_lin[iDX] * grid
        Vsub = V_lin[np.ix_([iD, iDX], [iD, iDX])]
        se_me = np.sqrt(
            np.array([np.array([1.0, g]) @ Vsub @ np.array([1.0, g]) for g in grid])
        )
        detail = pd.DataFrame({"x": grid, "me": me, "se": se_me})
        # Average effect: over treated observations for a discrete D
        # (interflex's ATE), over all observations for a continuous D (AME).
        mask = D == 1.0 if treat_type == "discrete" else np.ones(n, bool)
        wsub = np.ones(mask.sum()) if W is None else W[mask]
        te = b_lin[iD] + b_lin[iDX] * X[mask]
        avg = float(np.average(te, weights=wsub))
        vec = np.array([1.0, float(np.average(X[mask], weights=wsub))])
        avg_se = float(np.sqrt(vec @ Vsub @ vec))
        model_info.update(
            {
                "coefficients": pd.Series(b_lin, index=names_lin),
                "vcov": pd.DataFrame(V_lin, index=names_lin, columns=names_lin),
            }
        )
    elif estimator == "binning":
        colsb: List[np.ndarray] = []
        namesb: List[str] = []
        for g, m in zip(bins, x0):
            Gi = (groups == g).astype(float)
            colsb += [Gi, Gi * (X - m), Gi * D, Gi * D * (X - m)]
            namesb += [f"G.{g}", f"GX.{g}", f"D.G.{g}", f"DX.G.{g}"]
        if zc:
            colsb += [Z[:, j] for j in range(Z.shape[1])]
            namesb += zc
        Xb = np.column_stack(colsb)
        b_b, V_b, e_b = _ols(Xb, Y, W, cov_type)
        idx = [namesb.index(f"D.G.{g}") for g in bins]
        me = b_b[idx]
        se_me = np.sqrt(np.diag(V_b)[idx])
        detail = pd.DataFrame(
            {
                "bin": bins,
                "x": x0,
                "me": me,
                "se": se_me,
                "n": model_info["bins"]["counts"],
            }
        )
        mask = D == 1.0 if treat_type == "discrete" else np.ones(n, bool)
        wsub = np.ones(mask.sum()) if W is None else W[mask]
        te = np.zeros(n)
        for g, m in zip(bins, x0):
            sel = groups == g
            te[sel] = b_b[namesb.index(f"D.G.{g}")] + b_b[namesb.index(f"DX.G.{g}")] * (
                X[sel] - m
            )
        avg = float(np.average(te[mask], weights=wsub))
        # delta-method SE of the average over the bin coefficients
        grad = np.zeros(len(namesb))
        for g, m in zip(bins, x0):
            sel = (groups == g) & mask
            if sel.any():
                wg = np.ones(sel.sum()) if W is None else W[sel]
                share = wg.sum() / wsub.sum()
                grad[namesb.index(f"D.G.{g}")] += share
                grad[namesb.index(f"DX.G.{g}")] += share * float(
                    np.average(X[sel] - m, weights=wg)
                )
        avg_se = float(np.sqrt(grad @ V_b @ grad))
        model_info.update(
            {
                "coefficients": pd.Series(b_b, index=namesb),
                "vcov": pd.DataFrame(V_b, index=namesb, columns=namesb),
            }
        )
    else:
        xd, yd, bw_dens = r_density(X)
        dens_mean = float(np.exp(np.mean(np.log(yd[yd > 0]))))
        base_w = np.ones(n) if W is None else W

        def _local_fit(Yv, Dv, Xv, Zv, wv, x_pt):
            delta = Xv - x_pt
            if adaptive:
                temp = yd[int(np.argmin(np.abs(xd - x_pt)))]
                bw_use = float(bw) * np.sqrt(dens_mean / temp)
            else:
                bw_use = float(bw)
            w = stats.norm.pdf(delta / bw_use) * wv
            if np.any(w == 0):
                w = w + w[w != 0].min()
            Xl = np.column_stack([np.ones(len(Yv)), delta, Dv, Dv * delta, Zv])
            sw = np.sqrt(w)
            beta, *_ = np.linalg.lstsq(Xl * sw[:, None], Yv * sw, rcond=None)
            return float(beta[2])

        me = np.array([_local_fit(Y, D, X, Z, base_w, g) for g in grid])
        detail = pd.DataFrame({"x": grid, "me": me})
        mask = D == 1.0 if treat_type == "discrete" else np.ones(n, bool)
        wsub = np.ones(mask.sum()) if W is None else W[mask]
        avg = float(np.average(np.interp(X[mask], grid, me), weights=wsub))
        avg_se = None
        se_me = None
        if vce == "bootstrap":
            rng = np.random.default_rng(seed)
            draws = []
            for _ in range(int(n_boot)):
                idx = rng.integers(0, n, n)
                try:
                    draws.append(
                        [
                            _local_fit(Y[idx], D[idx], X[idx], Z[idx], base_w[idx], g)
                            for g in grid
                        ]
                    )
                except np.linalg.LinAlgError:
                    continue
            if len(draws) >= 2:
                arr = np.asarray(draws)
                se_me = arr.std(axis=0, ddof=1)
                detail["se"] = se_me
                avg_draws = [
                    float(np.average(np.interp(X[mask], grid, row), weights=wsub))
                    for row in arr
                ]
                avg_se = float(np.std(avg_draws, ddof=1))
                model_info["n_boot_success"] = len(draws)
        model_info.update(
            {
                "bw": float(bw),
                "adaptive": bool(adaptive),
                "density_bw": bw_dens,
                "density_geometric_mean": dens_mean,
            }
        )

    if se_me is not None:
        detail["ci_lower"] = detail["me"] - z_crit * detail["se"]
        detail["ci_upper"] = detail["me"] + z_crit * detail["se"]
    model_info["tests"] = tests
    model_info["average_effect_label"] = "ATE" if treat_type == "discrete" else "AME"
    ci = (avg - z_crit * avg_se, avg + z_crit * avg_se) if avg_se is not None else None
    pval = (
        float(2 * (1 - stats.norm.cdf(abs(avg / avg_se))))
        if (avg_se is not None and avg_se > 0)
        else None
    )

    return CausalResult(
        method=f"interflex {estimator} (Hainmueller, Mummolo and Xu 2019)",
        estimand="ATE" if treat_type == "discrete" else "AME",
        estimate=avg,
        se=avg_se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(n),
        detail=detail,
        model_info=model_info,
        _citation_key="interflex",
    )


def interflex_plot(
    result: CausalResult, ax: Any = None, show_hist: bool = True, **kwargs: Any
) -> Any:
    """The interflex figure: marginal effect of ``d`` across ``x`` with its
    confidence band and, below it, the distribution of the moderator.

    Parameters
    ----------
    result : CausalResult
        Output of :func:`interflex`.
    ax : matplotlib Axes, optional
        Draw into an existing axes; a new figure is created otherwise.
    show_hist : bool, default True
        Add the moderator histogram (treated / control) under the curve.

    Returns
    -------
    matplotlib Figure

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.interflex(df, y="Y", d="D", x="X", z=["Z1"], estimator="kernel", bw=1.0)
    >>> fig = sp.interflex_plot(res)
    """
    import matplotlib.pyplot as plt

    mi = result.model_info
    det = result.detail
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (7, 4.5)))
    else:
        fig = ax.figure
    if mi["estimator"] == "binning":
        ax.errorbar(
            det["x"],
            det["me"],
            yerr=1.96 * det["se"],
            fmt="o",
            color="black",
            capsize=3,
        )
    else:
        ax.plot(det["x"], det["me"], color="black")
        if "ci_lower" in det.columns:
            ax.fill_between(
                det["x"], det["ci_lower"], det["ci_upper"], color="grey", alpha=0.3
            )
    ax.axhline(0.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel(kwargs.get("xlabel", "moderator"))
    ax.set_ylabel(kwargs.get("ylabel", "marginal effect of treatment"))
    ax.set_title(kwargs.get("title", f"interflex ({mi['estimator']})"))
    if show_hist and "bins" in mi:
        ax2 = ax.twinx()
        ax2.hist(mi["x_eval"], bins=20, alpha=0.0)  # keep axis aligned
        ax2.set_yticks([])
    return fig


CausalResult._CITATIONS["interflex"] = (
    "@article{hainmueller2019much,\n"
    "  title={How Much Should We Trust Estimates from Multiplicative "
    "Interaction Models? Simple Tools to Improve Empirical Practice},\n"
    "  author={Hainmueller, Jens and Mummolo, Jonathan and Xu, Yiqing},\n"
    "  journal={Political Analysis},\n"
    "  volume={27},\n"
    "  number={2},\n"
    "  pages={163--192},\n"
    "  year={2019},\n"
    "  doi={10.1017/pan.2018.46}\n"
    "}"
)
