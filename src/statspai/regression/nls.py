"""Nonlinear least squares.

The model is ``y = m(x, theta) + e`` with ``m`` known up to the parameter
vector. The estimate minimises the sum of squared residuals; its covariance
is that of a linear regression on the *linearised* regressors, the columns
of the Jacobian ``d m / d theta'`` at the estimate.

The regression function is given either as a formula with the parameters in
braces, ``"y ~ {a} + {b} * x^{c}"``, or as a Python function of the
parameters and the data.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import optimize

from ..core._vcov import sandwich_vcov
from ..core.results import EconometricResults
from ..exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

__all__ = ["nls"]

_PARAM = re.compile(r"\{\s*([A-Za-z_]\w*)\s*(?:=\s*([^{}]+?)\s*)?\}")
_VCE = ("ols", "hc1", "hc2", "hc3", "cluster")


def _formula_function(
    formula: str, data: pd.DataFrame, start: Dict[str, float]
) -> tuple:
    """(outcome column, parameter names, f(theta) -> fitted values) for a
    formula with ``{parameters}``; ``{name=value}`` adds a starting value."""
    from ..agent._translation._stata_expr import StataExprError, evaluate

    if "~" not in formula:
        raise MethodIncompatibility(
            "sp.nls: the formula needs an outcome, 'y ~ {a} + {b} * x'.",
            recovery_hint="Write the outcome, a tilde, then the regression "
            "function with its parameters in braces.",
        )
    left, right = (part.strip() for part in formula.split("~", 1))
    if left not in data.columns:
        raise MethodIncompatibility(
            f"sp.nls: the outcome {left!r} is not a column.",
            recovery_hint="Transform the outcome into a column first.",
        )
    names: List[str] = []
    for name, value in _PARAM.findall(right):
        if name not in names:
            names.append(name)
        if value and name not in start:
            try:
                start[name] = float(value)
            except ValueError:
                raise MethodIncompatibility(
                    f"sp.nls: the starting value {value!r} of {{{name}}} is "
                    "not a number.",
                    recovery_hint="Write {name=0.5}, or pass start={...}.",
                ) from None
    if not names:
        raise MethodIncompatibility(
            "sp.nls: the formula has no parameter. Parameters are written "
            "in braces: 'y ~ {a} + {b} * x'.",
            recovery_hint="For a model that is linear in its coefficients "
            "use sp.regress.",
        )
    # A parameter lives in its own namespace: {b} may share its name with a
    # column b. Inside the expression each parameter gets a private name.
    private = {n: f"_nls_parameter_{n}" for n in names}
    expression = _PARAM.sub(lambda m: private[m.group(1)], right)

    def fitted(theta: np.ndarray) -> np.ndarray:
        scalars = {private[name]: float(v) for name, v in zip(names, theta)}
        try:
            value = evaluate(expression, data, {"scalars": scalars})
        except StataExprError as exc:
            raise MethodIncompatibility(
                f"sp.nls: cannot evaluate the regression function: {exc}",
                recovery_hint="Operators + - * / ^, comparisons and the "
                "functions ln, exp, sqrt, abs, normal, cond ... are read; "
                "for anything else pass a Python function.",
            ) from exc
        return np.asarray(value, dtype=float)

    return left, names, fitted


def _jacobian(f: Callable[[np.ndarray], np.ndarray], theta: np.ndarray) -> np.ndarray:
    """Central differences, one parameter at a time."""
    columns = []
    for j in range(theta.size):
        h = np.finfo(float).eps ** (1.0 / 3.0) * max(abs(theta[j]), 1.0)
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        columns.append((f(up) - f(down)) / (2.0 * h))
    return np.column_stack(columns)


def nls(
    formula: Union[str, Callable[..., Any]],
    data: pd.DataFrame,
    start: Optional[Union[Mapping[str, float], Sequence[float]]] = None,
    *,
    y: Optional[str] = None,
    vce: str = "ols",
    cluster: Optional[str] = None,
    tol: float = 1e-10,
    maxiter: int = 1000,
    alpha: float = 0.05,
) -> EconometricResults:
    """Nonlinear least squares.

    Parameters
    ----------
    formula : str or callable
        Either ``"y ~ expression"`` with each parameter in braces, such as
        ``"q ~ {a} + ({nu}/{rho}) * ln({d}*k^{rho} + (1-{d})*l^{rho})"``.
        The expression uses column names, numbers, ``+ - * / ^``,
        comparisons (which give 0 / 1) and the functions ``ln`` / ``log``,
        ``exp``, ``sqrt``, ``abs``, ``normal``, ``cond`` and the others of
        :func:`statspai.stata`. A starting value may sit in the braces:
        ``{rho=0.4}``.

        Or a function ``f(params, data)`` that returns the fitted values,
        where ``params`` is a dict of parameter name to value; then ``y``
        names the outcome and ``start`` is required.
    data : pandas.DataFrame
        The data. Rows on which the outcome or the regression function at
        the starting values is missing are dropped.
    start : dict or sequence, optional
        Starting values by parameter name. Parameters without one start at
        1 for a formula. The answer can depend on them: a sum of squares
        that is not convex has local minima.
    y : str, optional
        Outcome column when ``formula`` is a function.
    vce : {'ols', 'robust', 'hc2', 'hc3'}, default 'ols'
        Covariance estimator; ``'robust'`` is HC1.
    cluster : str, optional
        Column to cluster on. Overrides ``vce``.
    tol : float, default 1e-10
        Relative tolerance on the sum of squares, the step and the gradient.
    maxiter : int, default 1000
        Limit on function evaluations.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    EconometricResults
        Parameter estimates and standard errors. ``data_info['X']`` is the
        Jacobian at the estimate (the linearised regressors), so
        :func:`statspai.test`, :func:`statspai.lincom` and
        :func:`statspai.nlcom` apply. ``model_info`` holds ``converged``,
        ``iterations`` and ``start``.

    Notes
    -----
    Standard errors use ``n - k`` degrees of freedom, and ``n / (n - k)``
    in the HC1 factor, as Stata's ``nl`` does. R-squared is taken about the
    mean of the outcome when one parameter enters as an additive constant
    (one Jacobian column is constant) and about zero otherwise.

    Inference treats the regression function as smooth in the parameters.
    At a kink (a threshold or change-point parameter) the usual normal
    approximation is poor in small samples, and when a parameter is not
    identified under a null of interest (``{b} * x^{c}`` with ``b = 0``)
    the t and Wald statistics do not have their usual distributions
    [@hansen2022econometrics, chapter 23].

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(1, 5, size=300)
    >>> df = pd.DataFrame({"x": x, "y": 2 + 3 * x**0.5 + rng.normal(size=300)})
    >>> fit = sp.nls("y ~ {a} + {b} * x^{c}", df, start={"c": 1}, vce="robust")
    >>> list(fit.params.index)
    ['a', 'b', 'c']
    >>> same = sp.nls(
    ...     lambda p, d: p["a"] + p["b"] * d["x"] ** p["c"],
    ...     df, start={"a": 1, "b": 1, "c": 1}, y="y", vce="robust",
    ... )
    >>> bool(np.allclose(fit.params, same.params, rtol=1e-6))
    True

    References
    ----------
    hansen2022econometrics
    """
    vce = {"robust": "hc1"}.get(str(vce).lower(), str(vce).lower())
    if cluster is not None:
        vce = "cluster"
    if vce not in _VCE or (vce == "cluster" and cluster is None):
        raise MethodIncompatibility(
            f"sp.nls: vce={vce!r} is not 'ols', 'robust', 'hc2' or 'hc3'.",
            recovery_hint="Use vce='robust', or cluster='<column>'.",
        )
    if cluster is not None and cluster not in data.columns:
        raise MethodIncompatibility(
            f"sp.nls: cluster={cluster!r} is not a column.",
            recovery_hint="Pass the name of the cluster variable.",
        )

    start_values: Dict[str, float] = {}
    if isinstance(start, Mapping):
        start_values = {str(k): float(v) for k, v in start.items()}
    if callable(formula):
        if y is None or y not in data.columns:
            raise MethodIncompatibility(
                "sp.nls: with a function, y= must name the outcome column.",
                recovery_hint="Pass y='<column>'.",
            )
        if not start_values:
            raise MethodIncompatibility(
                "sp.nls: with a function, start= must be a dict of "
                "parameter name to starting value.",
                recovery_hint="Pass start={'a': 1.0, 'b': 0.5}.",
            )
        outcome, names = y, list(start_values)
        user = formula

        def build(frame: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
            def fitted(theta: np.ndarray) -> np.ndarray:
                params = {n: float(v) for n, v in zip(names, theta)}
                return np.asarray(user(params, frame), dtype=float)

            return fitted

    else:
        text = str(formula)
        outcome, names, _ = _formula_function(text, data, start_values)
        if start is not None and not isinstance(start, Mapping):
            values = [float(v) for v in start]
            if len(values) != len(names):
                raise MethodIncompatibility(
                    f"sp.nls: {len(values)} starting values for the "
                    f"{len(names)} parameters {names}.",
                    recovery_hint="Pass one value per parameter, or a dict.",
                )
            start_values = dict(zip(names, values))

        def build(frame: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
            return _formula_function(text, frame, {})[2]

    unknown = [n for n in start_values if n not in names]
    if unknown:
        raise MethodIncompatibility(
            f"sp.nls: start= names {unknown}, which are not parameters " f"({names}).",
            recovery_hint="Check the spelling of the parameter names.",
        )
    theta0 = np.array([start_values.get(n, 1.0) for n in names], dtype=float)
    k = theta0.size

    # the estimation sample: outcome and regression function both defined
    y_all = pd.to_numeric(data[outcome], errors="coerce").to_numpy(dtype=float)
    with np.errstate(all="ignore"):
        first = build(data)(theta0)
    keep = np.isfinite(y_all) & np.isfinite(first)
    if cluster is not None:
        keep &= data[cluster].notna().to_numpy()
    frame = data.loc[keep]
    yv = y_all[keep]
    n = int(keep.sum())
    if n <= k:
        raise DataInsufficient(
            f"sp.nls: {n} usable observations for {k} parameters.",
            recovery_hint="Check the starting values: the regression "
            "function must be defined at them.",
        )
    f = build(frame)

    def residuals(theta: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            r = yv - f(theta)
        # outside the domain of the function: a large finite penalty keeps
        # the trust region inside it
        return np.where(np.isfinite(r), r, 1e150)

    fit = optimize.least_squares(
        residuals,
        theta0,
        jac=lambda t: -_jacobian(f, t),
        method="lm",
        xtol=tol,
        ftol=tol,
        gtol=tol,
        max_nfev=maxiter,
    )
    theta = np.asarray(fit.x, dtype=float)
    resid = yv - f(theta)
    if not np.all(np.isfinite(resid)):
        raise NumericalInstability(
            "sp.nls: the regression function is not defined at the estimate.",
            recovery_hint="Try other starting values.",
        )
    J = _jacobian(f, theta)
    rss = float(resid @ resid)
    converged = bool(fit.status > 0)
    if not converged:
        warnings.warn(
            "sp.nls did not converge; the estimates are where the search "
            "stopped. Try other starting values or a larger maxiter.",
            ConvergenceWarning,
            stacklevel=2,
        )
    try:
        bread = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            "sp.nls: the Jacobian is rank deficient at the estimate: some "
            "parameter is not identified.",
            recovery_hint="Fix one of the parameters, or change the model.",
        ) from exc
    df_resid = n - k
    clusters = None
    if vce == "ols":
        cov = bread * rss / df_resid
    elif vce == "cluster":
        clusters = pd.factorize(frame[cluster])[0]
        cov = sandwich_vcov(
            bread, J * resid[:, None], clusters=clusters, correction="stata"
        )
    else:
        if vce == "hc1":
            weight = np.full(n, n / df_resid)
        else:
            h = np.einsum("ij,jk,ik->i", J, bread, J)
            weight = 1.0 / (1.0 - h) if vce == "hc2" else 1.0 / (1.0 - h) ** 2
        cov = bread @ ((J * (resid**2 * weight)[:, None]).T @ J) @ bread
    se = np.sqrt(np.diag(cov))

    # a parameter that enters as an additive constant: centred R-squared
    scale = np.abs(J).max(axis=0)
    has_const = bool(
        np.any((np.ptp(J, axis=0) <= 1e-8 * np.maximum(scale, 1e-300)) & (scale > 0))
    )
    tss = float(((yv - yv.mean()) ** 2).sum()) if has_const else float(yv @ yv)
    df_total = n - 1 if has_const else n
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")
    r2_adj = 1.0 - (rss / df_resid) / (tss / df_total) if tss > 0 else float("nan")

    label = {"ols": "nonrobust", "hc1": "robust"}.get(vce, vce)
    model_info: Dict[str, Any] = {
        "model_type": "Nonlinear least squares",
        "method": "Levenberg-Marquardt",
        "formula": formula if isinstance(formula, str) else None,
        "start": dict(zip(names, theta0.tolist())),
        "converged": converged,
        "iterations": int(fit.nfev),
        "has_constant": has_const,
        "robust": label,
        "cluster": cluster,
        "alpha": alpha,
    }
    if clusters is not None:
        model_info["n_clusters"] = int(clusters.max()) + 1
    data_info: Dict[str, Any] = {
        "nobs": n,
        "df_model": k - has_const,
        "df_resid": df_resid if clusters is None else int(clusters.max()),
        "dependent_var": outcome,
        "var_names": names,
        "var_cov": cov,
        "X": J,
        "y": yv,
        "residuals": resid,
        "fitted_values": yv - resid,
        "rss": rss,
        "tss": tss,
        "sample_index": frame.index,
    }
    diagnostics = {
        "R-squared": r2,
        "Adj. R-squared": r2_adj,
        "Root MSE": float(np.sqrt(rss / df_resid)),
        "Residual SS": rss,
        "Residual deviance": float(n * (np.log(2 * np.pi * rss / n) + 1.0)),
    }
    index = pd.Index(names)
    return EconometricResults(
        params=pd.Series(theta, index=index),
        std_errors=pd.Series(se, index=index),
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
