"""Threshold regression and regression kink with an unknown threshold.

Two models share one search. In the threshold model the coefficients of
some regressors change when a threshold variable ``q`` crosses ``gamma``::

    y = x'b + 1(q > gamma) * z'd + e

In the kink model the regression function is continuous in ``q`` and its
slope changes at ``gamma``::

    y = b1 * min(q - gamma, 0) + b2 * max(q - gamma, 0) + x'b + e

``gamma`` is estimated by least squares: for each candidate the other
coefficients are fitted by OLS, and the candidate with the smallest sum of
squared errors is kept.

What can be said about ``gamma`` differs between the two. In the threshold
model the estimate converges faster than root-n and is not normal; its
confidence set is the set of candidates the likelihood ratio does not
reject (Hansen 2000), and the standard errors of the other coefficients
take ``gamma`` as known. In the kink model the estimate is root-n normal
jointly with the slopes (Hansen 2017), so it gets a standard error like any
other nonlinear least squares parameter.

References
----------
[@hansen2000sample; @hansen2017regression; @hansen1996inference;
@hansen2022econometrics]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from ..core._vcov import sandwich_vcov
from ..core.results import EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility, StatsPAIWarning

__all__ = ["threshold"]

_VCE = ("ols", "hc0", "hc1", "hc2", "hc3", "cluster")
#: bootstrap replications are computed against at most this many candidates
_BOOT_GRID = 200


def _within(M: np.ndarray, codes: Optional[np.ndarray], n_groups: int) -> np.ndarray:
    """Deviations from group means, column by column."""
    if codes is None:
        return M
    M = np.asarray(M, dtype=float)
    flat = M.reshape(len(M), -1)
    counts = np.bincount(codes, minlength=n_groups).astype(float)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        means = np.bincount(codes, weights=flat[:, j], minlength=n_groups) / counts
        out[:, j] = flat[:, j] - means[codes]
    return out.reshape(M.shape)


def _candidates(q: np.ndarray, trim: float, grid: Optional[Any]) -> np.ndarray:
    if not 0.0 <= trim < 0.5:
        raise MethodIncompatibility(
            f"sp.threshold: trim={trim!r} has to be in [0, 0.5).",
            recovery_hint="trim=0.1 leaves 10% of the sample out at each end.",
        )
    if grid is not None and not np.isscalar(grid):
        values = np.unique(np.asarray(grid, dtype=float))
        if values.size == 0:
            raise MethodIncompatibility(
                "sp.threshold: grid is empty.",
                recovery_hint="Pass candidate values, a number of points, or None.",
            )
        return np.asarray(values, dtype=float)
    if grid is None:
        # every sample value that leaves floor(n * trim) observations on
        # each side, which is the candidate set of Stata's threshold
        ordered = np.sort(q)
        skip = int(np.floor(q.size * trim))
        values = np.unique(ordered[max(skip - 1, 0) : q.size - skip])
        # the largest value puts no observation above it
        values = values[values < ordered[-1]]
    else:
        lo, hi = np.quantile(q, [trim, 1.0 - trim])
        points = int(grid)  # type: ignore[arg-type]
        if points < 2:
            raise MethodIncompatibility(
                f"sp.threshold: grid={grid!r} has to be at least 2.",
                recovery_hint="grid=100 is the number of equally spaced candidates.",
            )
        values = np.linspace(lo, hi, points)
    if values.size == 0:
        raise DataInsufficient(
            "sp.threshold: no candidate threshold is left after trimming.",
            recovery_hint="Lower trim, or check that the threshold variable varies.",
        )
    return np.asarray(values, dtype=float)


def _switch(
    q: np.ndarray, gamma: float, Z: np.ndarray, kink: bool, shift: bool
) -> np.ndarray:
    """The columns that depend on the threshold."""
    if kink:
        d = q - gamma
        return np.column_stack([np.minimum(d, 0.0), np.maximum(d, 0.0)])
    above = (q > gamma).astype(float)
    parts = [above[:, None]] if shift else []
    if Z.shape[1]:
        parts.append(Z * above[:, None])
    return np.column_stack(parts)


def threshold(
    formula: str,
    data: pd.DataFrame,
    threshold: str,
    *,
    regime: Optional[Sequence[str]] = None,
    kink: bool = False,
    trim: float = 0.10,
    grid: Optional[Any] = None,
    absorb: Optional[str] = None,
    vce: str = "robust",
    cluster: Optional[str] = None,
    n_boot: int = 0,
    seed: Optional[int] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """Threshold regression, or regression kink, with an unknown threshold.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2"``: the regressors whose coefficients are the same
        on both sides, unless ``regime`` names them. With ``kink=True`` the
        threshold variable is left out of the formula; its two slopes are
        added.
    data : pandas.DataFrame
    threshold : str
        The threshold variable ``q``.
    regime : sequence of str, optional
        Regressors whose coefficients change above the threshold. The
        intercept always changes. Default: every regressor of the formula.
        A name that is not in the formula is added as a regressor that
        appears only above the threshold. Not used with ``kink=True``.
    kink : bool, default False
        Fit the continuous model with a change of slope in ``q``.
    trim : float, default 0.10
        Share of the sample kept out of the search at each end of ``q``.
    grid : int or array-like, optional
        ``None`` searches every sample value of ``q`` that leaves
        ``floor(n * trim)`` observations on each side (the candidates of
        Stata's ``threshold``). An integer searches that many equally
        spaced points between the ``trim`` and ``1 - trim`` quantiles (the
        grid of Hansen's programs), and an array is taken as the
        candidates.
    absorb : str, optional
        A categorical variable whose fixed effects are removed. Every
        column, the threshold indicator included, is taken in deviations
        from its group mean at each candidate.
    vce : {'robust', 'ols', 'hc0', 'hc2', 'hc3'}, default 'robust'
        Covariance of the coefficients. ``'robust'`` is HC1, as in
        ``sp.regress``; Stata's ``threshold, vce(robust)`` is ``'hc0'`` and
        its default is ``'ols'``.
    cluster : str, optional
        Cluster the covariance on this variable.
    n_boot : int, default 0
        Replications of the multiplier bootstrap test of the linear model
        against the threshold model. ``0`` skips the test.
    seed : int, optional
        Seed of the bootstrap.
    alpha : float, default 0.05
        One minus the confidence level, for the coefficients and for the
        threshold.

    Returns
    -------
    EconometricResults
        ``params`` holds the common coefficients, then the changes above
        the threshold (``above``, ``above:x``) or the two slopes
        (``q:below``, ``q:above``); with ``kink=True`` the last one is the
        threshold itself. ``model_info`` has ``threshold``,
        ``threshold_ci``, ``criterion`` (candidates, sum of squared errors,
        likelihood ratio), ``regimes`` (coefficients below and above with
        standard errors),
        ``n_below`` / ``n_above``, and ``linearity_test`` when
        ``n_boot > 0``.

    Notes
    -----
    The confidence set of the threshold in the threshold model inverts the
    likelihood ratio ``n (S(gamma) - S(gamma_hat)) / S(gamma_hat)`` against
    the critical value ``-2 log(1 - sqrt(1 - alpha))`` of Hansen (2000),
    which assumes homoskedastic errors; the interval reported is the convex
    hull of that set. Standard errors of the other coefficients treat the
    threshold as known. Both are first-order valid; in small samples the
    coefficient intervals are too short because they ignore that the
    threshold was estimated.

    The test of linearity is not a standard F test, because the threshold
    is not identified under the null. Its p-value comes from a bootstrap
    in which the residuals are multiplied by random signs (drawn by
    cluster when ``cluster`` is given).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"q": rng.uniform(0, 1, 400), "x": rng.normal(size=400)})
    >>> noise = rng.normal(size=400) / 2
    >>> df["y"] = 1 + 0.5 * df.x + (df.q > 0.4) * (1 + df.x) + noise
    >>> fit = sp.threshold("y ~ x", df, "q")
    >>> bool(abs(fit.model_info["threshold"] - 0.4) < 0.05)
    True

    References
    ----------
    [@hansen2000sample; @hansen2017regression; @hansen1996inference]
    """
    from .ols import regress

    vce = {"robust": "hc1", "unadjusted": "ols", "nonrobust": "ols"}.get(
        str(vce).lower(), str(vce).lower()
    )
    if cluster is not None:
        vce = "cluster"
    if vce not in _VCE or (vce == "cluster" and cluster is None):
        raise MethodIncompatibility(
            f"sp.threshold: vce={vce!r} is not available.",
            recovery_hint="Use 'robust', 'ols', 'hc0', 'hc2', 'hc3', or cluster=.",
        )
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            f"sp.threshold: alpha={alpha!r} has to be between 0 and 1.",
            recovery_hint="alpha=0.05 gives 95% intervals.",
        )
    extra = [c for c in (threshold, absorb, cluster) if c is not None]
    regime = None if regime is None else [str(r) for r in regime]
    for column in extra + [r for r in (regime or []) if r not in ("Intercept",)]:
        if column not in data.columns and column in extra:
            raise MethodIncompatibility(
                f"sp.threshold: {column!r} is not a column of the data.",
                recovery_hint="Check the spelling of threshold=, absorb=, cluster=.",
            )
    if kink and regime is not None:
        raise MethodIncompatibility(
            "sp.threshold: regime= does not apply to the kink model.",
            recovery_hint="With kink=True only the slope of the threshold "
            "variable changes; drop regime=, or use kink=False.",
        )
    outside = [r for r in (regime or []) if r in data.columns]
    frame = data.dropna(subset=list(dict.fromkeys(extra + outside)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = regress(formula, data=frame)
    info = base.data_info
    frame = frame.loc[info["sample_index"]]
    X = np.asarray(info["X"], dtype=float)
    y = np.asarray(info["y"], dtype=float)
    names: List[str] = [str(v) for v in info["var_names"]]
    q = np.asarray(frame[threshold], dtype=float)
    n = len(y)
    constant = [j for j in range(X.shape[1]) if np.ptp(X[:, j]) == 0 and X[0, j] != 0]
    if kink and threshold in names:
        raise MethodIncompatibility(
            f"sp.threshold: with kink=True the threshold variable "
            f"{threshold!r} must not be in the formula.",
            recovery_hint="Its two slopes are added by the function; remove "
            "it from the right-hand side.",
        )

    codes: Optional[np.ndarray] = None
    n_groups = 0
    if absorb is not None:
        codes = pd.factorize(frame[absorb])[0]
        n_groups = int(codes.max()) + 1
        keep = [j for j in range(X.shape[1]) if j not in constant]
        X, names = X[:, keep], [names[j] for j in keep]
        constant = []
    clusters = None if cluster is None else pd.factorize(frame[cluster])[0]

    # the regressors that switch
    switch_names: List[str]
    Z: np.ndarray
    if kink:
        Z = np.empty((n, 0))
        switch_names = [f"{threshold}:below", f"{threshold}:above"]
        shift = False
    else:
        shift = True
        if regime is None:
            which = [j for j in range(X.shape[1]) if j not in constant]
            Z = np.asarray(X[:, which], dtype=float)
            z_names = [names[j] for j in which]
        else:
            columns, z_names = [], []
            for r in regime:
                if r in names:
                    columns.append(X[:, names.index(r)])
                elif r in frame.columns:
                    columns.append(np.asarray(frame[r], dtype=float))
                else:
                    raise MethodIncompatibility(
                        f"sp.threshold: regime variable {r!r} is neither a "
                        "regressor of the formula nor a column.",
                        recovery_hint=f"Regressors: {names}.",
                    )
                z_names.append(r)
            Z = np.column_stack(columns) if columns else np.empty((n, 0))
        switch_names = ["above"] + [f"above:{z}" for z in z_names]

    Xw = _within(X, codes, n_groups)
    yw = _within(y, codes, n_groups)
    k_x = Xw.shape[1]
    Qx = np.linalg.qr(Xw)[0] if k_x else np.empty((n, 0))
    y_r = yw - Qx @ (Qx.T @ yw)

    def partial(gamma: float) -> np.ndarray:
        D = _within(_switch(q, gamma, Z, kink, shift), codes, n_groups)
        return np.asarray(D - Qx @ (Qx.T @ D), dtype=float)

    def sse_at(gamma: float) -> float:
        D = partial(gamma)
        coef = np.linalg.lstsq(D, y_r, rcond=None)[0]
        resid = y_r - D @ coef
        return float(resid @ resid)

    gammas = _candidates(q, trim, grid)
    sse = np.array([sse_at(g) for g in gammas])
    best = int(np.argmin(sse))
    gamma_hat = float(gammas[best])
    if kink and gammas.size > 1:
        # the criterion is continuous in the threshold: finish between the
        # neighbouring candidates
        lo = gammas[max(best - 1, 0)]
        hi = gammas[min(best + 1, gammas.size - 1)]
        if hi > lo:
            polished = optimize.minimize_scalar(
                sse_at, bounds=(lo, hi), method="bounded", options={"xatol": 1e-12}
            )
            if polished.fun <= sse[best]:
                gamma_hat = float(polished.x)
    s_min = min(float(sse[best]), sse_at(gamma_hat))
    if best in (0, gammas.size - 1) and gammas.size > 2:
        warnings.warn(
            "sp.threshold: the sum of squared errors is smallest at the edge "
            "of the search range, so the threshold may lie outside it or not "
            "exist. Look at model_info['criterion'], and lower trim.",
            StatsPAIWarning,
            stacklevel=2,
        )

    if kink:
        # A search on function values locates the minimum of a smooth
        # criterion to about the square root of machine precision. A few
        # Gauss-Newton steps on all parameters jointly finish the job.
        for _ in range(25):
            D = _within(_switch(q, gamma_hat, Z, True, False), codes, n_groups)
            Wg = np.column_stack([Xw, D])
            bg = np.linalg.lstsq(Wg, yw, rcond=None)[0]
            eg = yw - Wg @ bg
            slope = np.where(q < gamma_hat, -bg[k_x], 0.0) + np.where(
                q > gamma_hat, -bg[k_x + 1], 0.0
            )
            Jg = np.column_stack([Wg, _within(slope, codes, n_groups)])
            step = float(np.linalg.lstsq(Jg, eg, rcond=None)[0][-1])
            trial = gamma_hat + step
            if not (gammas[0] <= trial <= gammas[-1]) or sse_at(trial) > sse_at(
                gamma_hat
            ):
                break
            gamma_hat = trial
            if abs(step) <= 1e-14 * max(1.0, abs(gamma_hat)):
                break
        s_min = min(s_min, sse_at(gamma_hat))

    # the fit at the estimate
    D_hat = _switch(q, gamma_hat, Z, kink, shift)
    W = np.column_stack([Xw, _within(D_hat, codes, n_groups)])
    coef_names = names + switch_names
    rank = int(np.linalg.matrix_rank(W))
    if rank < W.shape[1]:
        raise DataInsufficient(
            "sp.threshold: the regressors are collinear at the estimated "
            f"threshold {gamma_hat:g} (rank {rank} of {W.shape[1]}).",
            recovery_hint="A regime variable may not vary on one side of the "
            "threshold; shorten regime=, or raise trim.",
        )
    beta = np.linalg.lstsq(W, yw, rcond=None)[0]
    resid = yw - W @ beta
    rss = float(resid @ resid)

    J = W
    theta = beta
    out_names = list(coef_names)
    if kink:
        b_below, b_above = beta[k_x], beta[k_x + 1]
        slope = np.where(q < gamma_hat, -b_below, 0.0) + np.where(
            q > gamma_hat, -b_above, 0.0
        )
        J = np.column_stack([W, _within(slope, codes, n_groups)])
        theta = np.append(beta, gamma_hat)
        out_names.append("threshold")
        if np.linalg.matrix_rank(J) < J.shape[1]:
            raise DataInsufficient(
                "sp.threshold: the two slopes are equal at the estimate, so "
                "the kink point is not identified.",
                recovery_hint="There may be no kink; compare with the linear "
                "model (n_boot=) before reading the threshold.",
            )
    k = J.shape[1]
    absorbed = n_groups if absorb is not None else 0
    df_resid = n - k - absorbed
    if df_resid <= 0:
        raise DataInsufficient(
            f"sp.threshold: {n} observations for {k + absorbed} parameters.",
            recovery_hint="Use fewer regime variables or more data.",
        )
    bread = np.linalg.inv(J.T @ J)
    if vce == "ols":
        cov = bread * (rss / df_resid)
    elif vce == "cluster":
        assert clusters is not None
        # fixed effects nested in the clusters cost one parameter, as in
        # Stata's xtreg, fe
        nested = absorb is not None and bool(
            pd.Series(clusters).groupby(codes).nunique().max() == 1
        )
        cov = sandwich_vcov(
            bread,
            J * resid[:, None],
            clusters=clusters,
            correction="stata",
            n_params=k + (1 if nested else absorbed),
        )
    else:
        if vce == "hc0":
            weight = np.ones(n)
        elif vce == "hc1":
            weight = np.full(n, n / df_resid)
        else:
            h = np.einsum("ij,jk,ik->i", J, bread, J)
            if absorb is not None:
                h = h + 1.0 / np.bincount(np.asarray(codes))[codes]
            weight = 1.0 / (1.0 - h) if vce == "hc2" else 1.0 / (1.0 - h) ** 2
        cov = bread @ ((J * (resid**2 * weight)[:, None]).T @ J) @ bread
    se = np.sqrt(np.diag(cov))

    # what can be said about the threshold
    lr = n * (sse - s_min) / s_min
    critical = float(-2.0 * np.log(1.0 - np.sqrt(1.0 - alpha)))
    if kink:
        from scipy import stats

        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        interval: Tuple[float, float] = (
            gamma_hat - z * float(se[-1]),
            gamma_hat + z * float(se[-1]),
        )
        interval_method = "normal"
    else:
        inside = gammas[lr <= critical]
        interval = (float(inside.min()), float(inside.max()))
        interval_method = "likelihood ratio"

    regimes = None
    if not kink:
        rows = {}
        for j, z_name in enumerate(["Intercept"] + z_names):
            c = k_x + j
            change = float(beta[c])
            if z_name == "Intercept":
                b = constant[0] if constant else None
            else:
                b = names.index(z_name) if z_name in names else None
            if b is None:
                # no coefficient below: absorbed intercept (not identified)
                # or a regressor that enters only above the threshold
                below = float("nan") if z_name == "Intercept" else 0.0
                var_below, var_above = 0.0, float(cov[c, c])
            else:
                below = float(beta[b])
                var_below = float(cov[b, b])
                var_above = float(cov[b, b] + cov[c, c] + 2.0 * cov[b, c])
            rows[z_name] = {
                "below": below,
                "above": below + change,
                "change": change,
                "se_below": (
                    float(np.sqrt(var_below)) if b is not None else float("nan")
                ),
                "se_above": (
                    float(np.sqrt(var_above)) if np.isfinite(below) else float("nan")
                ),
                "se_change": float(se[c]),
            }
        regimes = pd.DataFrame(rows).T

    test = None
    if n_boot and n_boot > 0:
        test = _linearity_test(
            y_r=y_r,
            resid=resid,
            partial=partial,
            gammas=gammas,
            null_extra=(
                _within(q, codes, n_groups) - Qx @ (Qx.T @ _within(q, codes, n_groups))
                if kink
                else None
            ),
            s1=s_min,
            clusters=clusters,
            n_boot=int(n_boot),
            seed=seed,
        )

    label = {"ols": "nonrobust", "hc1": "robust"}.get(vce, vce)
    model_info: Dict[str, Any] = {
        "model_type": "Regression kink" if kink else "Threshold regression",
        "method": "least squares over candidate thresholds",
        "formula": formula,
        "threshold_var": threshold,
        "threshold": gamma_hat,
        "threshold_ci": interval,
        "threshold_ci_method": interval_method,
        "lr_critical": critical,
        "criterion": pd.DataFrame({"threshold": gammas, "sse": sse, "lr": lr}),
        "n_candidates": int(gammas.size),
        "trim": trim,
        "kink": bool(kink),
        "regime": None if kink else z_names,
        "regimes": regimes,
        "n_below": int((q <= gamma_hat).sum()),
        "n_above": int((q > gamma_hat).sum()),
        "absorb": absorb,
        "robust": label,
        "cluster": cluster,
        "alpha": alpha,
        "has_constant": bool(constant) or absorb is not None,
    }
    if clusters is not None:
        model_info["n_clusters"] = int(clusters.max()) + 1
    if test is not None:
        model_info["linearity_test"] = test
    tss = float(((yw - yw.mean()) ** 2).sum())
    data_info: Dict[str, Any] = {
        "nobs": n,
        "df_model": k - (1 if constant else 0),
        "df_resid": df_resid if clusters is None else int(clusters.max()),
        "dependent_var": info.get("dependent_var"),
        "var_names": out_names,
        "var_cov": cov,
        "X": J,
        "y": yw,
        "residuals": resid,
        "fitted_values": yw - resid,
        "rss": rss,
        "tss": tss,
        "sample_index": frame.index,
    }
    diagnostics = {
        "R-squared": 1.0 - rss / tss if tss > 0 else float("nan"),
        "Root MSE": float(np.sqrt(rss / df_resid)),
        "Residual SS": rss,
        "Threshold": gamma_hat,
    }
    index = pd.Index(out_names)
    return EconometricResults(
        params=pd.Series(theta, index=index),
        std_errors=pd.Series(se, index=index),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )


def _linearity_test(
    *,
    y_r: np.ndarray,
    resid: np.ndarray,
    partial: Any,
    gammas: np.ndarray,
    null_extra: Optional[np.ndarray],
    s1: float,
    clusters: Optional[np.ndarray],
    n_boot: int,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Sup-F test of the linear model, with a multiplier bootstrap.

    ``F = n (S0 - S1) / S1`` with ``S1`` the smallest sum of squared errors
    over the candidates. Under the null the threshold is not identified and
    ``F`` has no pivotal distribution (Hansen 1996); the reference
    distribution is that of the same statistic computed on the residuals
    with random signs.
    """
    n = len(y_r)
    if gammas.size > _BOOT_GRID:
        take = np.unique(
            np.linspace(0, gammas.size - 1, _BOOT_GRID).round().astype(int)
        )
        gammas = gammas[take]

    def null_residual(v: np.ndarray) -> np.ndarray:
        if null_extra is None:
            return v
        e = null_extra[:, None] if null_extra.ndim == 1 else null_extra
        return np.asarray(v - e @ np.linalg.lstsq(e, v, rcond=None)[0])

    # orthonormal bases of what each candidate adds to the null model
    bases = []
    for g in gammas:
        D = partial(float(g))
        if null_extra is not None:
            D = null_residual(D)
        u, s, _ = np.linalg.svd(D, full_matrices=False)
        bases.append(u[:, s > 1e-10 * max(s.max(), 1e-300)])

    def sup_f(v0: np.ndarray) -> np.ndarray:
        """v0: (n, B) outcomes already residualised on the null model."""
        s0 = (v0**2).sum(axis=0)
        gain = np.zeros_like(s0)
        for basis in bases:
            gain = np.maximum(gain, ((basis.T @ v0) ** 2).sum(axis=0))
        return np.asarray(n * gain / (s0 - gain))

    y0 = null_residual(y_r)
    s0 = float(y0 @ y0)
    stat = float(n * (s0 - s1) / s1)
    rng = np.random.default_rng(seed)
    if clusters is None:
        signs = rng.integers(0, 2, size=(n, n_boot)) * 2.0 - 1.0
    else:
        by_cluster = rng.integers(0, 2, size=(int(clusters.max()) + 1, n_boot))
        signs = by_cluster[clusters] * 2.0 - 1.0
    draws = resid[:, None] * signs
    # the draws are residuals of the full model: project them as y was
    draws = np.column_stack([null_residual(draws[:, b]) for b in range(n_boot)])
    boot = sup_f(draws)
    return {
        "statistic": stat,
        "pvalue": float((boot >= stat).mean()),
        "n_boot": n_boot,
        "sse_linear": s0,
        "sse_threshold": s1,
        "critical_values": {
            level: float(np.quantile(boot, level)) for level in (0.90, 0.95, 0.99)
        },
    }
