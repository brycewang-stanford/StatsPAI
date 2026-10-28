"""Simulation-extrapolation (SIMEX) for regressors measured with error.

A regressor ``x`` is observed as ``w = x + u`` with ``u`` normal, mean zero
and known standard deviation ``sigma``. SIMEX adds *more* error,
``w + sqrt(lambda) sigma z`` with ``z`` standard normal, so that the total
error variance is ``(1 + lambda) sigma^2``, refits the naive model ``B``
times for each ``lambda`` in a grid, averages the estimates, fits a curve
to the estimates as a function of ``lambda`` (including the naive fit at
``lambda = 0``) and reads the curve off at ``lambda = -1``, where the
error variance is zero.

The default variance is the simulation-extrapolation jackknife of
Stefanski and Cook: at each ``lambda`` the average model-based covariance
of the simulated fits minus the sample covariance of the simulated
estimates, extrapolated to ``lambda = -1`` the same way. The asymptotic
(estimating-equation) variance of Carroll et al. is in
``_simex_asymptotic.py``; misclassification SIMEX for a binary or
categorical variable in ``_simex_misclass.py``.

References
----------
[@cook1994simulation]
[@stefanski1995simulation]
[@carroll1996asymptotics]
[@kuchenhoff2006general]
"""

from __future__ import annotations

import re
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import EconometricResults
from ..exceptions import (
    ConvergenceFailure,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from . import _simex_asymptotic, _simex_misclass
from ._simex_extrapolate import _curves, _extrapolate
from .eivreg import _design

__all__ = ["simex"]

_FAMILIES = ("gaussian", "logit", "probit", "poisson")
_EXTRAPOLANTS = ("quadratic", "linear", "nonlinear", "loglinear")


# --------------------------------------------------------------------
# Naive fits
# --------------------------------------------------------------------


def _link(family: str) -> Tuple[Callable, Callable, Callable]:
    """Inverse link, its derivative and the variance function."""
    if family == "logit":

        def mu(eta: np.ndarray) -> np.ndarray:
            out: np.ndarray = 1.0 / (1.0 + np.exp(-eta))
            return out

        def dmu(eta: np.ndarray) -> np.ndarray:
            p = mu(eta)
            return p * (1.0 - p)

        def var(m: np.ndarray) -> np.ndarray:
            return m * (1.0 - m)

    elif family == "probit":

        def mu(eta: np.ndarray) -> np.ndarray:
            out: np.ndarray = stats.norm.cdf(eta)
            return out

        def dmu(eta: np.ndarray) -> np.ndarray:
            out: np.ndarray = stats.norm.pdf(eta)
            return out

        def var(m: np.ndarray) -> np.ndarray:
            return m * (1.0 - m)

    else:  # poisson

        def mu(eta: np.ndarray) -> np.ndarray:
            out: np.ndarray = np.exp(eta)
            return out

        dmu = mu

        def var(m: np.ndarray) -> np.ndarray:
            return m

    return mu, dmu, var


def _fit(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    family: str,
    vce: str,
    start: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Naive fit: coefficients and their covariance.

    Weighted least squares for ``'gaussian'``; Fisher scoring with prior
    weights for the GLMs, whose model-based covariance is the inverse
    expected information (R ``vcov.glm``). ``vce='robust'`` is the HC0
    sandwich.
    """
    n, k = X.shape
    if family == "gaussian":
        XtW = X.T * w
        bread = np.linalg.inv(XtW @ X)
        beta = bread @ (XtW @ y)
        e = y - X @ beta
        if vce == "robust":
            meat = (X * (w * e)[:, None]).T @ (X * (w * e)[:, None])
            return beta, bread @ meat @ bread
        s2 = float(w @ e**2) / (n - k)
        return beta, bread * s2

    mu_f, dmu_f, var_f = _link(family)
    if start is None:
        beta = np.zeros(k)
    else:
        beta = start.copy()
    eps = 1e-10
    for _ in range(100):
        eta = X @ beta
        m = mu_f(eta)
        if family != "poisson":
            m = np.clip(m, eps, 1.0 - eps)
        d = np.maximum(dmu_f(eta), eps)
        v = np.maximum(var_f(m), eps)
        wi = w * d**2 / v
        z = eta + (y - m) / d
        XtW = X.T * wi
        new = np.linalg.solve(XtW @ X, XtW @ z)
        if not np.all(np.isfinite(new)):
            break
        moved = float(np.max(np.abs(new - beta)))
        beta = new
        if moved <= 1e-10 * (1.0 + float(np.max(np.abs(beta)))):
            break
    else:
        raise ConvergenceFailure(
            f"sp.simex: the {family} fit did not converge.",
            recovery_hint="Check for separation in the outcome, or rescale the "
            "regressors.",
        )
    if not np.all(np.isfinite(beta)) or (
        family in ("logit", "probit") and float(np.max(np.abs(X @ beta))) > 30
    ):
        raise ConvergenceFailure(
            f"sp.simex: the {family} fit diverges (complete or quasi-complete "
            "separation).",
            recovery_hint="Drop or coarsen the separating regressor.",
        )
    eta = X @ beta
    m = mu_f(eta)
    if family != "poisson":
        m = np.clip(m, eps, 1.0 - eps)
    d = np.maximum(dmu_f(eta), eps)
    v = np.maximum(var_f(m), eps)
    bread = np.linalg.inv((X.T * (w * d**2 / v)) @ X)
    if vce == "robust":
        score = X * (w * (y - m) * d / v)[:, None]
        return beta, bread @ (score.T @ score) @ bread
    return beta, bread


# --------------------------------------------------------------------
# Public entry
# --------------------------------------------------------------------


def _plain_column(
    name: str, names: List[str], df: pd.DataFrame, X: np.ndarray, rows: np.ndarray
) -> int:
    """Index of ``name`` in the design; it must enter as the plain column."""
    if name not in names:
        raise MethodIncompatibility(
            f"sp.simex: {name!r} is not a term of the model {names}.",
            recovery_hint="Key error_sd / misclassification by a regressor of "
            "the formula.",
            diagnostics={"terms": names},
        )
    others = [t for t in names if t != name and _mentions(t, name)]
    j = names.index(name)
    raw = (
        pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)[rows]
        if name in df.columns
        else None
    )
    if others or raw is None or not np.allclose(raw, X[:, j], rtol=0, atol=0):
        raise MethodIncompatibility(
            f"sp.simex: {name!r} must enter the model as a plain column; it "
            f"also appears in {others or 'a transformed term'}.",
            recovery_hint="SIMEX perturbs the column itself, so every term "
            "built from it would have to be rebuilt. Fit the model with "
            "the plain column, or construct the interaction by hand and "
            "treat it as its own error-prone variable only if its error "
            "is additive and normal.",
        )
    return j


def _pseudo_errors(rng: Any, n: int, p: int) -> np.ndarray:
    """Standard normal pseudo-errors for one simulated data set (``n x p``)."""
    out: np.ndarray = rng.standard_normal((n, p))
    return out


def _mentions(term: str, name: str) -> bool:
    return re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", term) is not None


@records_call("simex")
def simex(
    formula: str,
    data: pd.DataFrame,
    error_sd: Optional[Mapping[str, Union[float, str]]] = None,
    family: Literal["gaussian", "logit", "probit", "poisson"] = "gaussian",
    weights: Optional[str] = None,
    *,
    misclassification: Optional[Mapping[str, Any]] = None,
    lambdas: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    B: int = 100,
    extrapolant: Literal["quadratic", "linear", "nonlinear", "loglinear"] = "quadratic",
    jackknife_extrapolant: Literal[
        "quadratic", "linear", "nonlinear", "loglinear"
    ] = "quadratic",
    vce: Literal["ols", "robust"] = "ols",
    variance: Literal["jackknife", "asymptotic"] = "jackknife",
    seed: Optional[int] = 0,
    alpha: float = 0.05,
) -> EconometricResults:
    """Simulation-extrapolation estimator for regressors measured with error.

    Each regressor named in ``error_sd`` is taken to be observed with
    additive normal error of known standard deviation. SIMEX adds further
    simulated error at several multiples ``lambda`` of the error variance,
    refits the naive model, and extrapolates the coefficients back to the
    error-free case ``lambda = -1``. It works for linear regression and for
    logit, probit and Poisson models, where no closed-form correction
    exists.

    Parameters
    ----------
    formula : str
        ``"y ~ x + z"``. Each error-prone regressor must enter as a plain
        term (not inside an interaction or a transformation).
    data : pandas.DataFrame
        The data. Rows with a missing value in a model variable are dropped.
    error_sd : dict, optional
        ``{"x": 0.5}``: the standard deviation of the measurement error of
        each error-prone regressor, either a number or the name of a column
        holding a per-row standard deviation (heteroskedastic error).
        Required unless ``misclassification`` is given.
    family : {'gaussian', 'logit', 'probit', 'poisson'}, default 'gaussian'
        The naive model: least squares, or a GLM fitted by maximum
        likelihood.
    weights : str, optional
        Column of prior weights for the naive fits (``lm`` / ``glm``
        weights in R).
    misclassification : dict, optional
        MC-SIMEX (R ``mcsimex``): ``{"x": P}`` for a misclassified
        variable, with ``P[j][i] = Pr(observed j | true i)`` (columns sum
        to one). A 0/1 variable has a 2 x 2 matrix (a DataFrame is read by
        its 0/1 labels) and is a plain regressor or the 0/1 outcome. A
        variable with ``k > 2`` levels has a ``k x k`` matrix and enters
        the formula as a factor (``C(x)``, ``i.x`` or a categorical
        column); an array is read by the levels ``0 .. k-1``, a DataFrame
        by its labels. The variable is re-misclassified with
        ``P ** lambda`` instead of receiving normal error, and a factor's
        indicators are rebuilt from the new levels. Not combined with
        ``error_sd``.
    lambdas : sequence of float, default (0.5, 1, 1.5, 2)
        Positive multiples of the error variance (or exponents of the
        misclassification matrix) to simulate. The naive fit is added at
        ``lambda = 0``.
    B : int, default 100
        Simulated data sets per ``lambda``.
    extrapolant : {'quadratic', 'linear', 'nonlinear', 'loglinear'}, default 'quadratic'
        Curve fitted to the coefficient paths. ``'nonlinear'`` is the
        rational function ``a + b / (c + lambda)``, which is exact for a
        linear model with homoskedastic error; the quadratic is more stable
        and usually corrects only part of the bias. ``'loglinear'`` fits
        ``log(path + shift)`` linearly in ``lambda`` (R ``mcsimex``; the
        shift is ``|min| + 1`` for a path with a non-positive value).
    jackknife_extrapolant : str, default 'quadratic'
        Curve fitted to the jackknife variance paths, one of the
        ``extrapolant`` values.
    vce : {'ols', 'robust'}, default 'ols'
        Covariance of each naive fit used by the jackknife variance:
        model-based (classical least squares, inverse information) or the
        HC0 sandwich.
    variance : {'jackknife', 'asymptotic'}, default 'jackknife'
        Standard errors from the simulation-extrapolation jackknife, or the
        asymptotic estimating-equation variance of
        [@carroll1996asymptotics] (R ``asymptotic = TRUE``), which stacks
        the scores of the naive and simulated fits and applies the delta
        method through the extrapolant. The asymptotic variance needs a
        homoskedastic error standard deviation.
    seed : int, optional
        Seed for the simulated errors.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    EconometricResults
        SIMEX coefficients with jackknife or asymptotic standard errors.
        ``model_info`` holds ``naive_params``, the averaged coefficient path
        ``path`` (one row per ``lambda``, starting at 0) and the settings;
        with ``variance='asymptotic'`` also ``jackknife_cov``.

    Notes
    -----
    The error standard deviation must come from outside the data
    (replicate measurements, a validation sample, the known noise of a
    privacy mechanism). SIMEX assumes the error is normal, additive and
    independent of everything else.

    A jackknife variance path whose extrapolated diagonal is negative
    leaves that standard error undefined and is warned about.

    The asymptotic variance uses the likelihood score ``w x (y - mu) d /
    v``. R ``simex`` uses ``x (y - mu)`` and ignores prior weights; the two
    agree for unweighted least squares, logit and Poisson and differ for
    probit, where R's is not the score of the fitted model.

    MC-SIMEX [@kuchenhoff2006general] needs every power of ``P`` to be a
    misclassification matrix: ``P[0][0] + P[1][1] > 1`` for two levels,
    and for ``k > 2`` R's ``check.mc.matrix`` condition that the matrix
    logarithm has no negative off-diagonal entry. A misclassified outcome
    must be 0/1.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x = rng.normal(size=n)
    >>> df = pd.DataFrame({"w": x + rng.normal(scale=0.5, size=n)})
    >>> df["y"] = 1 + 2 * x + rng.normal(size=n)
    >>> fit = sp.simex("y ~ w", df, error_sd={"w": 0.5}, B=50)
    >>> naive = fit.model_info["naive_params"]["w"]
    >>> bool(abs(fit.params["w"] - 2) < abs(naive - 2))
    True
    >>> asy = sp.simex("y ~ w", df, error_sd={"w": 0.5}, B=50,
    ...                variance="asymptotic")
    >>> bool(asy.std_errors["w"] > 0)
    True

    A misclassified binary regressor (sensitivity 0.9, specificity 0.8):

    >>> xb = rng.binomial(1, 0.5, size=n)
    >>> flip = np.where(xb == 1, rng.random(n) > 0.9, rng.random(n) > 0.8)
    >>> df["xb"] = np.where(flip, 1 - xb, xb)
    >>> df["yb"] = 1 + 2 * xb + rng.normal(size=n)
    >>> P = [[0.8, 0.1], [0.2, 0.9]]
    >>> mc = sp.simex("yb ~ xb", df, misclassification={"xb": P}, B=50)
    >>> naive = mc.model_info["naive_params"]["xb"]
    >>> bool(abs(mc.params["xb"] - 2) < abs(naive - 2))
    True

    References
    ----------
    [@cook1994simulation]
    [@stefanski1995simulation]
    [@carroll1996asymptotics]
    [@kuchenhoff2006general]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "sp.simex: data must be a pandas DataFrame.",
            recovery_hint="Pass the data as a DataFrame.",
        )
    family = str(family).lower()  # type: ignore[assignment]
    if family not in _FAMILIES:
        raise MethodIncompatibility(
            f"sp.simex: family={family!r} is not one of {list(_FAMILIES)}.",
            recovery_hint="Use family='gaussian', 'logit', 'probit' or 'poisson'.",
        )
    for label, choice in (
        ("extrapolant", extrapolant),
        ("jackknife_extrapolant", jackknife_extrapolant),
    ):
        if choice not in _EXTRAPOLANTS:
            raise MethodIncompatibility(
                f"sp.simex: {label}={choice!r} is not one of {list(_EXTRAPOLANTS)}.",
                recovery_hint="Use 'quadratic', 'linear', 'nonlinear' or "
                "'loglinear'.",
            )
    if vce not in ("ols", "robust"):
        raise MethodIncompatibility(
            f"sp.simex: vce={vce!r} is not 'ols' or 'robust'.",
            recovery_hint="Use vce='ols' (model-based) or 'robust' (sandwich).",
        )
    if variance not in ("jackknife", "asymptotic"):
        raise MethodIncompatibility(
            f"sp.simex: variance={variance!r} is not 'jackknife' or 'asymptotic'.",
            recovery_hint="Use variance='jackknife' (default) or 'asymptotic'.",
        )
    grid_pos = np.asarray(sorted(float(v) for v in lambdas), dtype=float)
    if grid_pos.size < 1 or np.any(grid_pos <= 0) or len(set(grid_pos)) < grid_pos.size:
        raise MethodIncompatibility(
            f"sp.simex: lambdas={list(lambdas)!r} must be distinct positive numbers.",
            recovery_hint="Use the default (0.5, 1, 1.5, 2).",
        )
    need = {"linear": 2, "quadratic": 3, "nonlinear": 3, "loglinear": 2}
    worst = max(need[extrapolant], need[jackknife_extrapolant])
    if grid_pos.size + 1 < worst:
        raise MethodIncompatibility(
            f"sp.simex: {grid_pos.size + 1} grid points (with lambda = 0) cannot "
            f"identify the {extrapolant} extrapolant.",
            recovery_hint="Pass more lambdas, e.g. (0.5, 1, 1.5, 2).",
        )
    if int(B) < 2:
        raise MethodIncompatibility(
            f"sp.simex: B={B!r}; the jackknife variance needs at least 2 "
            "simulations per lambda.",
            recovery_hint="Use B=100 or more.",
        )
    B = int(B)
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"sp.simex: alpha={alpha!r} is not in (0, 1).",
            recovery_hint="Use alpha=0.05 for 95% intervals.",
        )
    if error_sd and misclassification:
        raise MethodIncompatibility(
            "sp.simex: pass error_sd (additive normal error) or "
            "misclassification (a 0/1 variable), not both.",
            recovery_hint="Fit the two error models separately.",
        )
    if not error_sd and not misclassification:
        raise MethodIncompatibility(
            "sp.simex: error_sd names no error-prone regressor.",
            recovery_hint="Pass error_sd={'x': <error standard deviation>}, or "
            "misclassification={'x': P} for a misclassified 0/1 variable.",
        )
    error_sd = dict(error_sd or {})

    df = data.reset_index(drop=True)
    sd_cols = {k: v for k, v in error_sd.items() if isinstance(v, str)}
    if sd_cols:
        check_columns(
            df, {f"error_sd[{k}]": v for k, v in sd_cols.items()}, function="sp.simex"
        )
    if weights is not None:
        check_columns(df, {"weights": weights}, function="sp.simex")
    extra = [c for c in list(sd_cols.values()) + ([weights] if weights else [])]
    if extra:
        keep = df[extra].notna().all(axis=1).to_numpy()
        df = df.loc[keep].reset_index(drop=True)

    d = _design(formula, df, function="sp.simex")
    X, y, names, rows = d["X"], d["y"], d["names"], d["rows"]
    n, k = X.shape
    if n <= k:
        raise DataInsufficient(
            f"sp.simex: {n} observations for {k} coefficients.",
            recovery_hint="Use fewer regressors or more data.",
        )
    if np.linalg.matrix_rank(X) < k:
        raise MethodIncompatibility(
            "sp.simex: the regressors are collinear.",
            recovery_hint="Drop the redundant regressors.",
        )
    if family in ("logit", "probit") and not np.all(np.isin(y, (0.0, 1.0))):
        raise MethodIncompatibility(
            f"sp.simex: family='{family}' needs a 0/1 outcome.",
            recovery_hint="Recode the outcome, or use family='gaussian'.",
        )
    if family == "poisson" and np.any(y < 0):
        raise MethodIncompatibility(
            "sp.simex: family='poisson' needs a non-negative outcome.",
            recovery_hint="Use family='gaussian' for a signed outcome.",
        )

    cols: List[int] = []
    sds: List[np.ndarray] = []
    for name, value in error_sd.items():
        j = _plain_column(name, names, df, X, rows)
        if isinstance(value, str):
            s = pd.to_numeric(df[value], errors="coerce").to_numpy(dtype=float)[rows]
        else:
            try:
                s = np.full(n, float(value))
            except (TypeError, ValueError):
                s = np.full(n, np.nan)
        if not np.all(np.isfinite(s)) or np.any(s < 0):
            raise MethodIncompatibility(
                f"sp.simex: the error standard deviation of {name!r} must be a "
                "finite non-negative number.",
                recovery_hint="Pass error_sd={'x': 0.5} or the name of a column "
                "of standard deviations.",
            )
        if variance == "asymptotic" and float(np.ptp(s)) > 0:
            raise MethodIncompatibility(
                f"sp.simex: variance='asymptotic' needs one error standard "
                f"deviation for all rows; {name!r} has a per-row one.",
                recovery_hint="Use variance='jackknife' for heteroskedastic "
                "error (R simex has the same restriction).",
            )
        cols.append(j)
        sds.append(s)

    # MC-SIMEX targets: (column index, or None for the outcome, or the
    # factor's dummy columns; matrix; level codes of the observed values)
    mc_specs: List[Tuple[Any, np.ndarray, Optional[np.ndarray]]] = []
    for name, value in (misclassification or {}).items():
        P, levels = _simex_misclass.mc_matrix_levels(name, value)
        if P.shape == (2, 2) and (name == d["depvar"] or name in names):
            if name == d["depvar"]:
                target, values = None, y
            else:
                target = _plain_column(name, names, df, X, rows)
                values = X[:, target]
            if not np.all(np.isin(values, (0.0, 1.0))):
                raise MethodIncompatibility(
                    f"sp.simex: the misclassified variable {name!r} must be "
                    "coded 0/1.",
                    recovery_hint="Recode it to 0/1 (the matrix rows and "
                    "columns are in the order 0, 1).",
                )
            mc_specs.append((target, P, None))
            continue
        if name == d["depvar"]:
            raise MethodIncompatibility(
                f"sp.simex: a misclassified outcome must be 0/1 with a 2 x 2 "
                f"matrix; {name!r} has a {P.shape[0]} x {P.shape[0]} one.",
                recovery_hint="MC-SIMEX for a categorical outcome would need a "
                "multinomial model; R mcsimex has the same restriction.",
            )
        dummies, level_codes = _simex_misclass.factor_columns(
            name, levels, names, df, X, rows
        )
        mc_specs.append((dummies, P, level_codes))

    if weights is not None:
        w = pd.to_numeric(df[weights], errors="coerce").to_numpy(dtype=float)[rows]
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise MethodIncompatibility(
                "sp.simex: weights must be finite and non-negative.",
                recovery_hint="Drop or fix the offending rows.",
            )
    else:
        w = np.ones(n)

    beta0, V0 = _fit(X, y, w, family, vce)
    grid = np.concatenate([[0.0], grid_pos])
    path = np.empty((grid.size, k))
    var_path = np.empty((grid.size, k, k))
    path[0], var_path[0] = beta0, V0
    asymptotic = variance == "asymptotic"
    psi_blocks: List[np.ndarray] = []
    jac_blocks: List[np.ndarray] = []
    if asymptotic:
        psi0, jac0 = _simex_asymptotic.score_parts(X, y, w, family, beta0)
        psi_blocks.append(psi0)
        jac_blocks.append(jac0 / n)
    sd_mat = np.column_stack(sds) if sds else np.zeros((n, 0))
    if mc_specs:
        no_error = all(
            np.allclose(P, np.eye(P.shape[0]), rtol=0, atol=0) for _, P, _ in mc_specs
        )
    else:
        no_error = float(sd_mat.max()) == 0.0
    rng = np.random.default_rng(seed)
    jack_cov: Optional[np.ndarray] = None
    fallback: List[str] = []
    fallback_var: List[str] = []
    if no_error:
        path[1:] = beta0
        var_path[1:] = V0
        estimate = beta0.copy()
        cov = V0.copy()
    else:
        for g, lam in enumerate(grid_pos, start=1):
            draws = np.empty((B, k))
            covs = np.zeros((k, k))
            powers = [_simex_misclass.mc_power(P, lam) for _, P, _ in mc_specs]
            psi_sum = np.zeros((n, k))
            jac_sum = np.zeros((k, k))
            for b in range(B):
                Xb = X.copy()
                yb = y
                if cols:
                    Xb[:, cols] += (
                        np.sqrt(lam) * sd_mat * _pseudo_errors(rng, n, len(cols))
                    )
                for (target, _, mc_codes), P_lam in zip(mc_specs, powers):
                    if target is None:
                        yb = _simex_misclass.misclassify(y, P_lam, rng)
                    elif mc_codes is None:
                        Xb[:, target] = _simex_misclass.misclassify(
                            X[:, target], P_lam, rng
                        )
                    else:
                        new = _simex_misclass.misclassify(mc_codes, P_lam, rng)
                        for j, level in target:
                            Xb[:, j] = (new == level).astype(float)
                draws[b], Vb = _fit(Xb, yb, w, family, vce, start=beta0)
                covs += Vb
                if asymptotic:
                    psi_b, jac_b = _simex_asymptotic.score_parts(
                        Xb, yb, w, family, draws[b]
                    )
                    psi_sum += psi_b
                    jac_sum += jac_b
            path[g] = draws.mean(axis=0)
            var_path[g] = covs / B - np.cov(draws, rowvar=False, ddof=1).reshape(k, k)
            if asymptotic:
                psi_blocks.append(psi_sum / B)
                jac_blocks.append(jac_sum / (B * n))
        estimate, failed = _extrapolate(grid, path, extrapolant)
        flat, failed_v = _extrapolate(
            grid, var_path.reshape(grid.size, k * k), jackknife_extrapolant
        )
        cov = flat.reshape(k, k)
        cov = (cov + cov.T) / 2.0
        if asymptotic:
            jack_cov = cov
            cov = _simex_asymptotic.asymptotic_covariance(
                psi_blocks,
                jac_blocks,
                grid,
                _curves(grid, path, extrapolant, failed),
            )
        fallback = [names[j] for j in failed]
        fallback_var = sorted(
            {names[j // k] for j in failed_v} | {names[j % k] for j in failed_v}
        )
        if fallback or fallback_var:
            warnings.warn(
                ConvergenceWarning(
                    "sp.simex: the rational extrapolant a + b / (c + lambda) has "
                    "no finite value at lambda = -1 (its pole lies at or beyond "
                    f"-1) for the estimates of {fallback} and the variances of "
                    f"{fallback_var}; the quadratic was used for them. This is "
                    "typical of a coefficient whose path is flat up to "
                    "simulation noise.",
                    recovery_hint="Use extrapolant='quadratic' throughout, or "
                    "increase B so that flat paths are less noisy.",
                    diagnostics={"terms": fallback, "variance_terms": fallback_var},
                ),
                stacklevel=2,
            )
    diag = np.diag(cov)
    se = np.where(diag > 0, np.sqrt(np.clip(diag, 0.0, None)), np.nan)
    neg_var_terms = [names[j] for j in range(k) if not diag[j] > 0]
    if neg_var_terms:
        warnings.warn(
            ConvergenceWarning(
                f"sp.simex: the extrapolated {variance} variance is not "
                f"positive for {neg_var_terms}; their standard errors are "
                "undefined.",
                recovery_hint="Increase B, use jackknife_extrapolant='linear', or "
                "bootstrap the whole SIMEX procedure.",
                diagnostics={"terms": neg_var_terms},
            ),
            stacklevel=2,
        )

    index = pd.Index(names)
    labels = {"ols": "model-based", "robust": "sandwich (HC0)"}
    if asymptotic:
        robust_label = "asymptotic SIMEX (estimating equations)"
    else:
        robust_label = f"jackknife SIMEX; naive {labels[vce]}"
    model_info: Dict[str, Any] = {
        "model_type": f"{'MC-' if mc_specs else ''}SIMEX ({family})",
        "family": family,
        "error_sd": dict(error_sd),
        "lambdas": grid.tolist(),
        "B": B,
        "extrapolant": extrapolant,
        "jackknife_extrapolant": jackknife_extrapolant,
        "naive_params": pd.Series(beta0, index=index),
        "naive_std_errors": pd.Series(np.sqrt(np.diag(V0)), index=index),
        "path": pd.DataFrame(path, index=pd.Index(grid, name="lambda"), columns=index),
        "seed": seed,
        "robust": robust_label,
        "alpha": alpha,
        "variance": variance,
    }
    if mc_specs:
        model_info["misclassification"] = {
            name: P.tolist()
            for name, (_, P, _) in zip(misclassification or {}, mc_specs)
        }
    if jack_cov is not None:
        model_info["jackknife_cov"] = jack_cov
    data_info: Dict[str, Any] = {
        "nobs": n,
        "df_model": k - 1,
        "df_resid": n - k,
        "dependent_var": d["depvar"],
        "var_names": names,
        "var_cov": cov,
        "X": X,
        "y": y,
    }
    if family != "gaussian":
        data_info["inference"] = "z"
    diagnostics: Dict[str, Any] = {
        "error_prone_terms": list(error_sd) + list(misclassification or {}),
        "nonpositive_variance_terms": neg_var_terms,
        "no_measurement_error": no_error,
        "quadratic_fallback_terms": fallback,
        "quadratic_fallback_variance_terms": fallback_var,
    }
    return EconometricResults(
        params=pd.Series(estimate, index=index),
        std_errors=pd.Series(se, index=index),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
