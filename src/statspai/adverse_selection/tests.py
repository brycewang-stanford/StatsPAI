"""Tests for asymmetric information and welfare in insurance markets.

``positive_correlation_test`` is the conditional correlation test of
Chiappori and Salanie: once everything the insurer prices on is held
fixed, asymmetric information predicts that people with more coverage
have more claims. Two versions are offered. ``'biprobit'`` fits a
bivariate probit of coverage and claims and tests that the error
correlation is zero (Stata ``biprobit``). ``'residuals'`` fits two
separate probits and tests that their generalized residuals are
uncorrelated, which needs no bivariate likelihood.

``efc_welfare`` is the Einav-Finkelstein-Cullen welfare analysis with
linear curves. Exogenous variation in the price of coverage traces out
demand. The cost of the people who buy at each price traces out the
average cost curve. Marginal cost follows from the two, and the gap
between the competitive equilibrium (price equals average cost) and the
efficient allocation (price equals marginal cost) gives the deadweight
loss of selection.

References
----------
[@chiappori2000testing]
[@einav2010estimating]
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
)
from ._ordered import ordered_generalized_residual, outcome_kind
from ._results import EFCWelfareResult, PositiveCorrelationResult

__all__ = ["positive_correlation_test", "efc_welfare"]


# ---------------------------------------------------------------------
# Shared input handling
# ---------------------------------------------------------------------


def _as_list(cols: Optional[Sequence[str]]) -> List[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _require_binary(values: np.ndarray, name: str) -> None:
    if not np.isin(values, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"{name!r} must be coded 0/1",
            recovery_hint=f"Recode {name!r} as a 0/1 indicator.",
        )
    if values.min() == values.max():
        raise DataInsufficient(
            f"{name!r} takes a single value in the estimation sample",
            recovery_hint=f"{name!r} must vary for the model to be estimable.",
        )


# ---------------------------------------------------------------------
# Positive correlation test
# ---------------------------------------------------------------------


def _generalized_residual(y: np.ndarray, index: np.ndarray) -> np.ndarray:
    """``E[e | y, x]`` for a probit, ``q phi(a) / Phi(q a)`` with ``q = 2y - 1``."""
    q = 2.0 * y - 1.0
    return np.asarray(
        q * np.exp(stats.norm.logpdf(index) - stats.norm.logcdf(q * index))
    )


def _probit_index(
    df: pd.DataFrame, y: str, x: List[str], weights: Optional[str]
) -> np.ndarray:
    """Fitted probit index ``x'b`` of ``y`` on ``x`` (with an intercept)."""
    yv = df[y].to_numpy(dtype=float)
    if not x:
        w = np.ones(len(df)) if weights is None else df[weights].to_numpy(dtype=float)
        share = float(np.sum(w * yv) / np.sum(w))
        return np.full(len(df), stats.norm.ppf(share))
    from ..regression.logit_probit import probit as _probit

    fit = _probit(data=df, y=y, x=x, weights=weights)
    dropped = fit.model_info.get("n_perfect_prediction_dropped") or 0
    if dropped:
        raise DataInsufficient(
            f"the probit of {y!r} predicts some outcomes perfectly",
            recovery_hint="Drop or merge the covariate cells that predict the "
            "outcome perfectly, then rerun.",
        )
    params = fit.params.reindex(["Intercept"] + x).fillna(0.0).to_numpy()
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(dtype=float)])
    return np.asarray(X @ params)


def positive_correlation_test(
    data: pd.DataFrame,
    coverage: str,
    claim: str,
    covariates: Optional[Sequence[str]] = None,
    method: Literal["biprobit", "residuals"] = "biprobit",
    claim_covariates: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> PositiveCorrelationResult:
    """Conditional positive-correlation test for asymmetric information.

    Under asymmetric information (adverse selection or moral hazard), and
    conditional on the variables the insurer prices on, people who choose
    more coverage file more claims. The test asks whether coverage and
    claims are positively correlated once ``covariates`` are held fixed.

    Parameters
    ----------
    data : pandas.DataFrame
    coverage : str
        0/1 indicator of the more comprehensive contract, or (with
        ``method='residuals'``) 3 to 20 ordered coverage levels.
    claim : str
        0/1 indicator of a claim (the ex post risk), or (with
        ``method='residuals'``) 3 to 20 ordered levels such as a claim
        count top-coded at 2.
    covariates : list of str, optional
        Variables the insurer prices on; they enter both equations.
    method : {'biprobit', 'residuals'}, default 'biprobit'
        ``'biprobit'``: bivariate probit of coverage and claims, testing
        that the error correlation ``rho`` is zero. Without ``cluster``
        the test is the likelihood ratio test against two separate probits,
        with ``cluster`` the Wald test on ``athrho``; both as Stata's
        ``biprobit``. ``'residuals'``: two separate probits and the score
        statistic ``W = (sum w e1 e2)^2 / sum (w e1 e2)^2`` built from
        their generalized residuals ``e1`` and ``e2``.
    claim_covariates : list of str, optional
        Covariates of the claim equation when they differ from
        ``covariates``.
    weights : str, optional
        Sampling weights (``method='residuals'`` only; the bivariate probit
        has no weighted likelihood here).
    cluster : str, optional
        Cluster variable for the variance of the test.
    alpha : float, default 0.05

    Returns
    -------
    PositiveCorrelationResult
        ``estimate`` is ``rho`` or the correlation of the generalized
        residuals, ``statistic`` the chi-squared(1) statistic, ``pvalue``
        its two-sided p-value and ``pvalue_greater`` the one-sided p-value
        against positive correlation.

    Notes
    -----
    In the ``'residuals'`` version the probit coefficients are estimated,
    but at ``rho = 0`` the information matrix of the bivariate probit is
    block diagonal between ``rho`` and the coefficients, so the first
    step does not change the limit distribution of ``W``. ``sum e1 e2`` is
    the score of the bivariate probit likelihood with respect to ``rho``
    at ``rho = 0``; ``W`` uses the outer product of that score for its
    variance, which is what makes weights and clusters easy to add.

    An outcome with three or more ordered levels gets an ordered probit
    and its generalized residual ``(phi(l) - phi(u)) / (Phi(u) - Phi(l))``,
    where ``(l, u]`` is the interval of the latent error that produces the
    observed level. ``sum e1 e2`` is then the score with respect to the
    error correlation of the bivariate model with ordered probit margins
    at zero correlation; the binary case is the special case of two
    levels.

    A rejection is evidence of asymmetric information of some kind; the
    test does not separate adverse selection from moral hazard.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x = rng.normal(size=n)
    >>> u = rng.normal(size=n)
    >>> e1 = 0.6 * u + 0.8 * rng.normal(size=n)
    >>> e2 = 0.6 * u + 0.8 * rng.normal(size=n)
    >>> df = pd.DataFrame({"x": x,
    ...                    "cover": (0.3 * x + e1 > 0).astype(int),
    ...                    "claim": (-0.5 + 0.4 * x + e2 > 0).astype(int)})
    >>> res = sp.positive_correlation_test(df, "cover", "claim", ["x"],
    ...                                    method="residuals")
    >>> bool(res.pvalue_greater < 0.01)
    True

    References
    ----------
    [@chiappori2000testing]
    """
    if method not in ("biprobit", "residuals"):
        raise MethodIncompatibility(
            f"method must be 'biprobit' or 'residuals', got {method!r}",
            recovery_hint="Use method='biprobit' or method='residuals'.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be in (0, 1)", recovery_hint="Pass e.g. alpha=0.05."
        )
    x1 = _as_list(covariates)
    x2 = x1 if claim_covariates is None else _as_list(claim_covariates)
    if weights is not None and method == "biprobit":
        raise MethodIncompatibility(
            "weights are supported only with method='residuals'",
            recovery_hint="Use method='residuals' for weighted data.",
        )
    cols = {"coverage": coverage, "claim": claim, "covariates": x1}
    if claim_covariates is not None:
        cols["claim_covariates"] = x2
    if weights is not None:
        cols["weights"] = weights
    if cluster is not None:
        cols["cluster"] = cluster
    check_columns(data, cols, function="positive_correlation_test")

    used = list(dict.fromkeys([coverage, claim] + x1 + x2))
    used += [c for c in (weights, cluster) if c is not None and c not in used]
    df = data.loc[:, used].dropna().reset_index(drop=True)
    n = len(df)
    if n < len(set(x1) | set(x2)) + 4:
        raise DataInsufficient(
            f"only {n} complete observations",
            recovery_hint="Use fewer covariates or more data.",
        )
    ordered = [
        c
        for c in (coverage, claim)
        if outcome_kind(df[c].to_numpy(dtype=float), c) == "ordered"
    ]
    if ordered and (method == "biprobit" or weights is not None):
        raise MethodIncompatibility(
            f"ordered outcomes {ordered} need method='residuals' without weights",
            recovery_hint="Use method='residuals' (ordered probit residuals), "
            "or recode the outcomes as 0/1.",
        )
    for c in (coverage, claim):
        if c not in ordered:
            _require_binary(df[c].to_numpy(dtype=float), c)
    if weights is not None and (df[weights].to_numpy(dtype=float) <= 0).any():
        raise MethodIncompatibility(
            "weights must be positive",
            recovery_hint="Drop observations with non-positive weights.",
        )

    if method == "biprobit":
        return _pcp_biprobit(df, coverage, claim, x1, x2, cluster, alpha)
    return _pcp_residuals(df, coverage, claim, x1, x2, weights, cluster, alpha, ordered)


def _pcp_biprobit(
    df: pd.DataFrame,
    coverage: str,
    claim: str,
    x1: List[str],
    x2: List[str],
    cluster: Optional[str],
    alpha: float,
) -> PositiveCorrelationResult:
    from ..regression.selection import biprobit

    fit = biprobit(
        df,
        y1=coverage,
        y2=claim,
        x1=x1,
        x2=x2,
        robust="cluster" if cluster is not None else "nonrobust",
        cluster=cluster,
        alpha=alpha,
    )
    rho = float(fit.model_info["rho"])
    rho_se = float(fit.model_info["rho_se"])
    if not fit.model_info.get("converged", True):
        warnings.warn(
            AssumptionWarning(
                "the bivariate probit did not converge cleanly",
                recovery_hint="Check for separation, or use method='residuals'.",
            ),
            stacklevel=3,
        )
    if abs(rho) >= 1 - 1e-12:
        raise NumericalInstability(
            "the bivariate probit correlation is at the boundary |rho| = 1",
            recovery_hint="Coverage and claims are (near) deterministic given "
            "the covariates; use method='residuals'.",
        )
    athrho = float(np.arctanh(rho))
    athrho_se = rho_se / (1.0 - rho**2)
    ll_bi = float(fit.diagnostics["log_likelihood"])
    ll1 = _probit_loglik(df, coverage, x1)
    ll2 = _probit_loglik(df, claim, x2)
    lr = max(2.0 * (ll_bi - ll1 - ll2), 0.0)
    if cluster is None:
        statistic, test = lr, "Likelihood-ratio test of rho = 0"
    else:
        statistic, test = (athrho / athrho_se) ** 2, "Wald test of rho = 0"
    z = float(np.sign(rho) * np.sqrt(statistic))
    return PositiveCorrelationResult(
        method="biprobit",
        estimate=rho,
        se=rho_se,
        statistic=float(statistic),
        z=z,
        pvalue=float(stats.chi2.sf(statistic, 1)),
        pvalue_greater=float(stats.norm.sf(z)),
        test=test,
        alpha=alpha,
        n_obs=len(df),
        detail={
            "rho": rho,
            "rho_se": rho_se,
            "athrho": athrho,
            "athrho_se": athrho_se,
            "athrho_z": athrho / athrho_se,
            "lr_chi2": lr,
            "ll_biprobit": ll_bi,
            "ll_probit_coverage": ll1,
            "ll_probit_claim": ll2,
            "params": fit.params.to_dict(),
        },
        diagnostics={"converged": bool(fit.model_info.get("converged", True))},
    )


def _probit_loglik(df: pd.DataFrame, y: str, x: List[str]) -> float:
    yv = df[y].to_numpy(dtype=float)
    index = _probit_index(df, y, x, None)
    q = 2.0 * yv - 1.0
    return float(np.sum(stats.norm.logcdf(q * index)))


def _pcp_residuals(
    df: pd.DataFrame,
    coverage: str,
    claim: str,
    x1: List[str],
    x2: List[str],
    weights: Optional[str],
    cluster: Optional[str],
    alpha: float,
    ordered: Sequence[str] = (),
) -> PositiveCorrelationResult:
    w = np.ones(len(df)) if weights is None else df[weights].to_numpy(dtype=float)
    e1, e2 = (
        (
            ordered_generalized_residual(df, y, x)
            if y in ordered
            else _generalized_residual(
                df[y].to_numpy(dtype=float), _probit_index(df, y, x, weights)
            )
        )
        for y, x in ((coverage, x1), (claim, x2))
    )
    s = w * e1 * e2
    if cluster is None:
        var = float(np.sum(s**2))
        n_clusters = None
    else:
        sums = pd.Series(s).groupby(df[cluster].to_numpy()).sum().to_numpy()
        var = float(np.sum(sums**2))
        n_clusters = len(sums)
    if var <= 1e-300:
        raise NumericalInstability(
            "the variance of the residual products is zero",
            recovery_hint="Check that both outcomes vary given the covariates.",
        )
    score = float(np.sum(s))
    z = score / np.sqrt(var)
    statistic = z**2
    sw = float(np.sum(w))
    m1 = np.sum(w * e1) / sw
    m2 = np.sum(w * e2) / sw
    cov = np.sum(w * (e1 - m1) * (e2 - m2)) / sw
    corr = float(
        cov / np.sqrt(np.sum(w * (e1 - m1) ** 2) / sw * np.sum(w * (e2 - m2) ** 2) / sw)
    )
    return PositiveCorrelationResult(
        method="residuals",
        estimate=corr,
        se=None,
        statistic=float(statistic),
        z=float(z),
        pvalue=float(stats.chi2.sf(statistic, 1)),
        pvalue_greater=float(stats.norm.sf(z)),
        test="Score test on generalized residuals (W)",
        alpha=alpha,
        n_obs=len(df),
        detail={
            "score": score,
            "score_variance": var,
            "n_clusters": n_clusters,
            **({"ordered": list(ordered)} if ordered else {}),
        },
        diagnostics={},
    )


# ---------------------------------------------------------------------
# Einav-Finkelstein-Cullen welfare
# ---------------------------------------------------------------------

_EFC_ROWS = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "price_eq",
    "coverage_eq",
    "price_eff",
    "coverage_eff",
    "dwl",
    "surplus_eff",
    "dwl_share",
)


def _efc_quantities(
    alpha: float, beta: float, gamma: float, delta: float
) -> np.ndarray:
    """Equilibrium, efficient allocation and deadweight loss from four numbers.

    ``D(p) = alpha + beta p`` and ``AC(p) = gamma + delta p``. In coverage
    space the inverse demand is ``P(D) = (D - alpha) / beta`` and the gap
    between demand and marginal cost is linear, ``g(D) = c0 + c1 D``.
    """

    def inv(q: float) -> float:
        return (q - alpha) / beta

    c0 = -(1.0 - delta) * alpha / beta - gamma
    c1 = (1.0 - 2.0 * delta) / beta

    def surplus(q: float) -> float:
        return c0 * q + 0.5 * c1 * q * q

    # Competitive equilibrium: price equals average cost.
    q_eq = float(np.clip(alpha + beta * gamma / (1.0 - delta), 0.0, 1.0))
    p_eq = gamma + delta * inv(q_eq)
    # Efficient coverage maximises the surplus on [0, 1].
    cands = [0.0, 1.0]
    if c1 != 0.0:
        root = -c0 / c1
        if 0.0 < root < 1.0:
            cands.append(root)
    q_eff = max(cands, key=surplus)
    p_eff = inv(q_eff)
    dwl = surplus(q_eff) - surplus(q_eq)
    s_eff = surplus(q_eff)
    share = dwl / s_eff if s_eff > 0 else np.nan
    return np.array(
        [alpha, beta, gamma, delta, p_eq, q_eq, p_eff, q_eff, dwl, s_eff, share]
    )


def _ols_influence(
    X: np.ndarray, y: np.ndarray, n_total: int, label: str
) -> Tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and their influence rows scaled to ``n_total``."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise NumericalInstability(
            f"the {label} regression is rank deficient",
            recovery_hint="Drop collinear covariates; price must vary within "
            "the sample used for this regression.",
        )
    xtx_inv = np.linalg.inv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    e = y - X @ b
    return b, n_total * (X * e[:, None]) @ xtx_inv


def efc_welfare(
    data: pd.DataFrame,
    price: str,
    coverage: str,
    cost: str,
    covariates: Optional[Sequence[str]] = None,
    cluster: Optional[str] = None,
    se_method: Literal["delta", "bootstrap"] = "delta",
    n_boot: int = 500,
    seed: Optional[int] = 0,
    alpha: float = 0.05,
) -> EFCWelfareResult:
    """Welfare cost of selection from price variation (Einav-Finkelstein-Cullen).

    With price variation that is exogenous given ``covariates``, a linear
    probability regression of ``coverage`` on ``price`` is the demand
    curve ``D(p) = alpha + beta p``, and a regression of ``cost`` on
    ``price`` among the covered is the average cost curve
    ``AC(p) = gamma + delta p``. Total cost is ``D(p) AC(p)``, so marginal
    cost is ``MC(p) = AC(p) + delta D(p) / beta``. Average cost rising with
    price (``delta > 0``) is adverse selection: the people who drop out
    when price rises are the cheap ones.

    The competitive equilibrium sets price equal to average cost; the
    efficient allocation covers everyone whose willingness to pay exceeds
    the marginal cost of covering them. The deadweight loss is the area
    between demand and marginal cost from one to the other.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per potential buyer.
    price : str
        Price of coverage (relative to no coverage), in the units of
        ``cost``.
    coverage : str
        0/1 indicator of buying coverage.
    cost : str
        Cost of covering the person (claims of the covered, or their
        incremental cost). Used only where ``coverage == 1``; may be
        missing elsewhere.
    covariates : list of str, optional
        Controls that make price variation exogenous. Both curves are
        evaluated at the sample means of the covariates.
    cluster : str, optional
        Cluster variable for the standard errors.
    se_method : {'delta', 'bootstrap'}, default 'delta'
        ``'delta'``: delta method from the stacked influence functions of
        the two regressions (and of the covariate means), heteroskedasticity
        robust. ``'bootstrap'``: resample rows (or clusters) ``n_boot``
        times and report the bootstrap standard deviation and percentile
        interval.
    n_boot : int, default 500
    seed : int, optional
    alpha : float, default 0.05

    Returns
    -------
    EFCWelfareResult
        ``table`` holds the curve coefficients, the equilibrium and
        efficient price and coverage, the deadweight loss ``dwl``, the
        surplus at the efficient allocation ``surplus_eff`` and their ratio
        ``dwl_share``, each with a standard error and interval.

    Notes
    -----
    All welfare numbers are per potential buyer. Coverage is restricted to
    ``[0, 1]``: when the curves cross outside it, the equilibrium or the
    efficient allocation sits at a corner (full coverage or unravelling).
    ``surplus_eff`` integrates demand minus marginal cost from zero
    coverage, so it extrapolates the linear curves outside the observed
    prices; the deadweight loss itself only needs them between the
    equilibrium and the efficient allocation. Corners make the welfare
    numbers non-differentiable in the coefficients, where the delta-method
    standard errors are not valid; use the bootstrap there.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 4000
    >>> p = rng.choice([2.0, 3.0, 4.0, 5.0], size=n)
    >>> risk = rng.uniform(0, 10, size=n)          # willingness to pay
    >>> buy = (risk > p + rng.normal(0, 1, n)).astype(int)
    >>> cost = np.where(buy == 1, 0.6 * risk + rng.normal(0, 1, n), np.nan)
    >>> df = pd.DataFrame({"p": p, "buy": buy, "cost": cost})
    >>> res = sp.efc_welfare(df, "p", "buy", "cost")
    >>> bool(res.curves["beta"] < 0 and res.curves["delta"] > 0)
    True

    References
    ----------
    [@einav2010estimating]
    """
    if se_method not in ("delta", "bootstrap"):
        raise MethodIncompatibility(
            f"se_method must be 'delta' or 'bootstrap', got {se_method!r}",
            recovery_hint="Use se_method='delta' or 'bootstrap'.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be in (0, 1)", recovery_hint="Pass e.g. alpha=0.05."
        )
    xs = _as_list(covariates)
    cols: Dict[str, Any] = {
        "price": price,
        "coverage": coverage,
        "cost": cost,
        "covariates": xs,
    }
    if cluster is not None:
        cols["cluster"] = cluster
    check_columns(data, cols, function="efc_welfare")

    keep = [price, coverage] + xs + ([cluster] if cluster is not None else [])
    df = data.loc[:, list(dict.fromkeys(keep + [cost]))]
    df = df.dropna(subset=list(dict.fromkeys(keep))).reset_index(drop=True)
    d = df[coverage].to_numpy(dtype=float)
    _require_binary(d, coverage)
    covered = d == 1
    if df.loc[covered, cost].isna().any():
        raise DataInsufficient(
            f"{cost!r} is missing for some covered observations",
            recovery_hint="Cost must be observed for everyone with coverage == 1.",
        )
    k = 2 + len(xs)
    if covered.sum() <= k or (~covered).sum() == 0 or len(df) <= k:
        raise DataInsufficient(
            "too few covered or uncovered observations",
            recovery_hint="Both curves need more observations than coefficients.",
        )
    if df[price].nunique() < 2 or df.loc[covered, price].nunique() < 2:
        raise IdentificationFailure(
            "price does not vary, so neither curve is identified",
            recovery_hint="The method needs exogenous variation in the price "
            "of coverage, also among the covered.",
        )

    clusters = df[cluster].to_numpy() if cluster is not None else None
    theta, infl = _efc_fit(df, price, coverage, cost, xs)
    est = _efc_from_theta(theta, len(xs))
    curves = dict(zip(_EFC_ROWS[:4], est[:4].tolist()))
    if curves["beta"] >= 0:
        raise IdentificationFailure(
            f"estimated demand does not slope down (beta = {curves['beta']:.4g})",
            recovery_hint="Check that the price variation is exogenous given the "
            "covariates, and that price is coded as the price of coverage.",
        )
    if abs(1.0 - curves["delta"]) < 1e-10:
        raise NumericalInstability(
            "average cost moves one for one with price, so the equilibrium "
            "is not unique",
            recovery_hint="Inspect the cost regression; the linear model "
            "cannot place the equilibrium.",
        )

    z = stats.norm.ppf(1 - alpha / 2)
    n_failed = 0
    if se_method == "delta":
        vcov = _stacked_vcov(infl, clusters)
        grad = _numeric_jacobian(lambda t: _efc_from_theta(t, len(xs)), theta)
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", grad, vcov, grad), 0.0))
        lo, hi = est - z * se, est + z * se
    else:
        draws, n_failed = _efc_bootstrap(
            df, price, coverage, cost, xs, clusters, n_boot, seed
        )
        se = np.nanstd(draws, axis=0, ddof=1)
        lo = np.nanquantile(draws, alpha / 2, axis=0)
        hi = np.nanquantile(draws, 1 - alpha / 2, axis=0)
    table = pd.DataFrame(
        {"estimate": est, "se": se, "ci_lower": lo, "ci_upper": hi},
        index=list(_EFC_ROWS),
    )

    p_obs = df[price].to_numpy(dtype=float)
    p_range = (float(p_obs.min()), float(p_obs.max()))
    diagnostics: Dict[str, Any] = {
        "selection": "adverse" if curves["delta"] > 0 else "advantageous",
        "equilibrium_corner": bool(est[5] in (0.0, 1.0)),
        "efficient_corner": bool(est[7] in (0.0, 1.0)),
        "n_boot_failed": n_failed,
    }
    outside = [
        name
        for name, val in (("price_eq", est[4]), ("price_eff", est[6]))
        if not p_range[0] <= val <= p_range[1]
    ]
    diagnostics["extrapolated"] = outside
    if outside:
        warnings.warn(
            AssumptionWarning(
                f"{', '.join(outside)} lies outside the observed prices "
                f"[{p_range[0]:g}, {p_range[1]:g}]; the linear curves are "
                "extrapolated",
                recovery_hint="Read the welfare numbers as an extrapolation, "
                "or collect wider price variation.",
                diagnostics={"extrapolated": outside},
            ),
            stacklevel=2,
        )
    if se_method == "delta" and (
        diagnostics["equilibrium_corner"] or diagnostics["efficient_corner"]
    ):
        warnings.warn(
            AssumptionWarning(
                "an allocation is at a corner of [0, 1]; the delta-method "
                "standard errors of the welfare numbers are not valid there",
                recovery_hint="Use se_method='bootstrap'.",
            ),
            stacklevel=2,
        )
    return EFCWelfareResult(
        table=table,
        curves=curves,
        se_method=se_method,
        alpha=alpha,
        n_obs=len(df),
        n_covered=int(covered.sum()),
        price_range=p_range,
        diagnostics=diagnostics,
    )


def _efc_fit(
    df: pd.DataFrame, price: str, coverage: str, cost: str, xs: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Stacked parameters (demand, cost, covariate means) and influence rows."""
    n = len(df)
    X = np.column_stack(
        [np.ones(n), df[price].to_numpy(dtype=float)]
        + [df[c].to_numpy(dtype=float) for c in xs]
    )
    d = df[coverage].to_numpy(dtype=float)
    covered = d == 1
    b_d, if_d = _ols_influence(X, d, n, "demand")
    b_c, if_c_sub = _ols_influence(
        X[covered], df.loc[covered, cost].to_numpy(dtype=float), n, "average cost"
    )
    if_c = np.zeros((n, X.shape[1]))
    if_c[covered] = if_c_sub
    xbar = X[:, 2:].mean(axis=0)
    if_x = X[:, 2:] - xbar
    theta = np.concatenate([b_d, b_c, xbar])
    return theta, np.hstack([if_d, if_c, if_x])


def _efc_from_theta(theta: np.ndarray, m: int) -> np.ndarray:
    k = 2 + m
    b_d, b_c, xbar = theta[:k], theta[k : 2 * k], theta[2 * k :]
    alpha = b_d[0] + float(xbar @ b_d[2:])
    gamma = b_c[0] + float(xbar @ b_c[2:])
    return _efc_quantities(alpha, b_d[1], gamma, b_c[1])


def _stacked_vcov(infl: np.ndarray, clusters: Optional[np.ndarray]) -> np.ndarray:
    n = infl.shape[0]
    if clusters is not None:
        infl = pd.DataFrame(infl).groupby(clusters).sum().to_numpy()
    out: np.ndarray = np.asarray(infl.T @ infl) / n**2
    return out


def _numeric_jacobian(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray
) -> np.ndarray:
    f0 = f(x)
    jac = np.empty((f0.size, x.size))
    for j in range(x.size):
        h = 1e-6 * max(1.0, abs(x[j]))
        up, dn = x.copy(), x.copy()
        up[j] += h
        dn[j] -= h
        jac[:, j] = (f(up) - f(dn)) / (2 * h)
    return jac


def _efc_bootstrap(
    df: pd.DataFrame,
    price: str,
    coverage: str,
    cost: str,
    xs: List[str],
    clusters: Optional[np.ndarray],
    n_boot: int,
    seed: Optional[int],
) -> Tuple[np.ndarray, int]:
    if n_boot < 2:
        raise MethodIncompatibility(
            "n_boot must be at least 2", recovery_hint="Pass e.g. n_boot=500."
        )
    rng = np.random.default_rng(seed)
    if clusters is None:
        groups = [np.array([i]) for i in range(len(df))]
    else:
        codes, uniq = pd.factorize(clusters)
        order = np.argsort(codes, kind="stable")
        bounds = np.searchsorted(codes[order], np.arange(len(uniq) + 1))
        groups = [order[bounds[g] : bounds[g + 1]] for g in range(len(uniq))]
    out = np.full((n_boot, len(_EFC_ROWS)), np.nan)
    failed = 0
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[g] for g in pick])
        sub = df.iloc[rows].reset_index(drop=True)
        try:
            theta, _ = _efc_fit(sub, price, coverage, cost, xs)
            out[b] = _efc_from_theta(theta, len(xs))
        except (NumericalInstability, np.linalg.LinAlgError):
            failed += 1
    if failed > n_boot // 10:
        warnings.warn(
            AssumptionWarning(
                f"{failed} of {n_boot} bootstrap draws could not be fitted",
                recovery_hint="The curves are weakly identified in resamples; "
                "treat the bootstrap interval with caution.",
            ),
            stacklevel=3,
        )
    return out, failed
