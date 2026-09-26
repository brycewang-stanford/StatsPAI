"""
Marginal effects and predictive margins.

Average marginal effects (AME) / marginal effects at the means (MEM),
predictive margins at covariate values, contrasts and pairwise comparisons
of margins, all on the model's default prediction scale with delta-method
standard errors. Equivalent to Stata's margins family.

Supports:
- Continuous variables: dy/dx, through interactions and formula transforms
  (I(x**2), np.log(x)) -- the design is rebuilt from the formula
- Factor variables (C(g)): discrete change of each level vs the base
- Conditional margins at specific covariate values, atmeans
- Estimation-sample averaging with the fit's weights, offsets / exposure
"""

import re
import warnings
from itertools import product as itertools_product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility, StatsPAIWarning
from ._covariance import coefficient_covariance, inference_df, require_covariance
from ._design import design_for, model_setting


def margins(
    result: Any,
    data: Optional[pd.DataFrame] = None,
    variables: Optional[List[str]] = None,
    at: Optional[Dict[str, Any]] = None,
    method: str = "ame",
    eps: float = 1e-5,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute marginal effects from a fitted model.

    Equivalent to Stata's ``margins, dydx(varlist)`` (``atmeans`` with
    ``method='mem'``), on the model's default prediction scale: the linear
    prediction after ``regress``, ``Pr(y=1)`` after ``logit`` / ``probit`` /
    ``cloglog``, the expected count (including any ``offset`` / ``exposure``)
    after ``poisson`` / ``nbreg`` / log-link ``glm``.

    * **Continuous variables** — ``dy/dx`` of the prediction, differentiating
      *through* every term the variable enters: interactions, and formula
      transformations such as ``I(x**2)`` or ``np.log(x)`` (the design is
      rebuilt from the fitted formula, so an ``x + I(x**2)`` model reports the
      total effect ``b1 + 2 b2 x``, like Stata's ``c.x##c.x``).
    * **Factor variables** (entered as ``C(g)``) — Stata's discrete change
      of each level against the base level, ``E[mu | g=l] - E[mu | g=base]``,
      one row per non-base level labelled ``"l.g"``.  A numeric 0/1 column
      *not* wrapped in ``C()`` is continuous, as in Stata without ``i.``.
    * Averages are over the estimation sample (Stata ``e(sample)``: rows of
      ``data`` complete on the outcome and every model variable) and use the
      fit's ``weights`` when it had any. Standard errors are delta-method on
      the full coefficient covariance, with t(df) after ``regress`` and z
      after likelihood estimators.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result (must have ``.params`` and associated data).
    data : pd.DataFrame, optional
        Data to compute margins on (the estimation data). Required for
        factor variables and formula transformations. Defaults to the
        stored design when the model has neither.
    variables : list of str, optional
        Variables to compute dy/dx for. Default: every model variable,
        factor variables included (Stata ``dydx(*)``).
    at : dict, optional
        Fix covariates at specific values for conditional margins.
        E.g., ``{'age': 30, 'female': 1}``. Every key must be a model
        variable (Stata error 322 otherwise).
    method : str, default 'ame'
        - 'ame': Average Marginal Effect (average dy/dx across all obs)
        - 'mem': Marginal Effect at the Mean (Stata ``atmeans``: every
          design component at its sample mean, factor indicators at their
          shares, interactions as products of those means)
    eps : float, default 1e-5
        Relative step for the central difference used on transformed terms.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Table with columns: variable, dy/dx, se, z, pvalue, ci_lower, ci_upper.
        ``.attrs`` records ``n`` (rows averaged over), ``weights``,
        ``design_backend`` and, for factor rows, ``base_levels``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=200),
    ...     "x2": rng.normal(size=200),
    ...     "female": rng.integers(0, 2, size=200).astype(float),
    ... })
    >>> df["y"] = (1.0 + 0.5 * df["x1"] - 0.3 * df["x2"]
    ...            + 0.2 * df["x1"] * df["x2"] + rng.normal(size=200))
    >>> result = sp.regress("y ~ x1 + x2 + x1:x2", data=df)
    >>> me = sp.margins(result, data=df)
    >>> me.columns.tolist()
    ['variable', 'dy/dx', 'se', 'z', 'pvalue', 'ci_lower', 'ci_upper']

    Conditional margins: marginal effect of x1 with x2 fixed at 1.

    >>> me_at = sp.margins(result, data=df, variables=['x1'], at={'x2': 1})
    >>> me_at['variable'].tolist()
    ['x1']

    Quadratic term and a factor: the effect of ``x`` includes ``2 b2 x``,
    ``g`` gets one discrete-change row per non-base level.

    >>> df["g"] = rng.integers(1, 4, size=200)
    >>> r2 = sp.regress("y ~ x1 + I(x1**2) + C(g)", data=df)
    >>> sp.margins(r2, data=df)["variable"].tolist()
    ['x1', '2.g', '3.g']
    """
    if method not in ("ame", "mem"):
        raise MethodIncompatibility(
            f"margins: method must be 'ame' or 'mem', got {method!r}."
        )
    ctx = _MarginsContext(result, data, at=at)
    design = ctx.design
    if variables is None:
        variables = [
            v
            for v in design.variables
            if v in ctx.frame.columns
            and (not design.is_factor(v) or v in design.factors)
        ]
    else:
        unknown = [v for v in variables if v not in design.variables]
        if unknown:
            raise MethodIncompatibility(
                f"margins: {unknown} do not enter the model.",
                recovery_hint=f"Variables in the model: {design.variables}.",
            )

    rows: List[Dict[str, Any]] = []
    base_levels: Dict[str, Any] = {}
    for var in variables:
        if design.is_factor(var):
            if var not in design.factors:
                raise MethodIncompatibility(
                    f"margins: {var!r} enters as a factor but its levels cannot "
                    "be read from the data (pass data= with the raw column).",
                )
            levels, base = design.factors[var]
            base_levels[var] = base
            b_eta, b_X = ctx.index_at({var: base}, method)
            mu_b, f_b = _mu_and_slope(ctx.link, b_eta)
            for lev in levels:
                if _same_level(lev, base):
                    continue
                l_eta, l_X = ctx.index_at({var: lev}, method)
                mu_l, f_l = _mu_and_slope(ctx.link, l_eta)
                est = ctx.mean(mu_l - mu_b)
                grad = ctx.mean(f_l[:, None] * l_X - f_b[:, None] * b_X)
                rows.append(ctx.inference_row(f"{_level_label(lev)}.{var}", est, grad))
            continue

        X, D = ctx.design_and_derivative(var, method, eps)
        eta = X @ ctx.beta + ctx.offset(method)
        f, f_prime = _link_derivatives(ctx.link, eta)
        d_eta = D @ ctx.beta
        est = ctx.mean(f * d_eta)
        # Delta method: d AME / d beta = mean(f'(eta) X d_eta + f(eta) D).
        grad = ctx.mean(f_prime[:, None] * X * d_eta[:, None] + f[:, None] * D)
        rows.append(ctx.inference_row(var, est, grad, label=f"margins({var!r})"))

    out = pd.DataFrame(
        rows, columns=["variable", "dy/dx", "se", "z", "pvalue", "ci_lower", "ci_upper"]
    )
    out.attrs.update(ctx.attrs())
    if base_levels:
        out.attrs["base_levels"] = base_levels
    return out


class _MarginsContext:
    """Estimation sample, design backend, weights and inference for margins.

    ``margins`` / ``margins_at`` / ``contrast`` / ``pwcompare`` share it so
    the four functions agree on the averaging sample (Stata ``e(sample)``),
    the averaging weights, the prediction scale and the reference
    distribution.
    """

    def __init__(
        self,
        result: Any,
        data: Optional[pd.DataFrame],
        at: Optional[Dict[str, Any]] = None,
        alpha: float = 0.05,
    ) -> None:
        self.result = result
        self.link = _response_link(result)
        self.beta = result.params.to_numpy(dtype=float)
        self.cov_source = result
        if (getattr(result, "model_info", None) or {}).get("irr"):
            # irr=True reports exp(b) with delta-method SEs; the index and the
            # stored covariance are on the b scale.
            self.beta = np.log(self.beta)
            self.cov_source = _IndexScaleView(result)
        frame = _margins_frame(result, data)
        self.design = design_for(result, frame)
        frame, self.n_dropped = _estimation_sample(
            result, frame, self.design, restrict=data is not None
        )
        if at:
            unknown = [k for k in at if k not in self.design.variables]
            if unknown:
                raise MethodIncompatibility(
                    f"margins: at() variable(s) {unknown} are not in the model "
                    f"(Stata: 'not found in list of covariates', r(322)).",
                    recovery_hint=f"Model variables: {self.design.variables}.",
                    diagnostics={"not_in_model": unknown},
                )
            frame = frame.copy()
            for name, value in at.items():
                frame[name] = value
        self.frame = frame
        self.at = dict(at or {})
        self.weights, self.weights_name = _averaging_weights(result, frame, data)
        self.df = inference_df(result)
        self.alpha = alpha

    # -- averaging -------------------------------------------------------
    def mean(self, values: np.ndarray) -> Any:
        values = np.asarray(values, dtype=float)
        if self.weights is None:
            return values.mean(axis=0)
        if isinstance(self.weights, str):  # weights exist but are unavailable
            spread = np.ptp(values, axis=0) if values.shape[0] else 0.0
            if np.any(np.abs(spread) > 1e-12 * (1.0 + np.abs(values).max())):
                raise MethodIncompatibility(
                    f"margins: the fit used weights={self.weights!r}, which "
                    "Stata margins averages with, but that column is not in the "
                    "data margins was given.",
                    recovery_hint="Pass data= containing the weight column.",
                )
            return values.mean(axis=0)
        w = self.weights / self.weights.sum()
        return np.tensordot(w, values, axes=(0, 0))

    # -- designs ---------------------------------------------------------
    def offset(self, method: str = "ame", frame: Optional[pd.DataFrame] = None) -> Any:
        fr = self.frame if frame is None else frame
        off = _offset(self.result, fr)
        if method == "mem":
            return np.array([self.mean(off)])
        return off

    def index_at(
        self, setting: Dict[str, Any], method: str = "ame"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Linear index and design with ``setting`` imposed on every row."""
        fr = self.frame.copy()
        for k, v in setting.items():
            fr[k] = v
        if method == "mem":
            X = self._design_at_means(fr)
            return X @ self.beta + self.offset("mem", fr), X
        X = self.design.build(fr)
        return X @ self.beta + _offset(self.result, fr), X

    def design_and_derivative(
        self, var: str, method: str, eps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        if method == "mem":
            at_means: Tuple[np.ndarray, np.ndarray] = self._design_at_means(
                self.frame, deriv_var=var, eps=eps
            )
            return at_means
        return (
            self.design.build(self.frame),
            self.design.derivative(self.frame, var, eps),
        )

    def _design_at_means(
        self, frame: pd.DataFrame, deriv_var: Optional[str] = None, eps: float = 1e-5
    ) -> Any:
        """Stata ``atmeans``: every component at its (weighted) sample mean.

        Numeric raw variables are set to their means; factor indicators to
        their shares, with interactions formed as products of those means
        (a weighted sum over the factor-level grid with product weights).
        Returns the 1 x k design (and its derivative when asked).
        """
        design = self.design
        fixed = set(self.at)
        numeric = [
            v
            for v in design.variables
            if v in frame.columns and not design.is_factor(v)
        ]
        factor_vars = [v for v in design.variables if design.is_factor(v)]
        base = {}
        for v in numeric:
            base[v] = float(self.mean(frame[v].to_numpy(dtype=float)))
        grid: List[Tuple[Dict[str, Any], float]] = [({}, 1.0)]
        for v in factor_vars:
            col = frame[v]
            if v in fixed or col.nunique(dropna=True) == 1:
                val = col.iloc[0]
                grid = [({**g, v: val}, w) for g, w in grid]
                continue
            levels = design.factors.get(v, (list(pd.unique(col.dropna())), None))[0]
            shares = {}
            for lev in levels:
                hit = col.map(lambda o, _l=lev: _same_level(o, _l))
                shares[lev] = float(self.mean(hit.to_numpy(dtype=float)))
            grid = [
                ({**g, v: lev}, w * s) for g, w in grid for lev, s in shares.items()
            ]
            if len(grid) > 20000:
                raise MethodIncompatibility(
                    "margins(method='mem'): too many factor-level combinations "
                    "to evaluate atmeans.",
                )
        cells = pd.DataFrame([{**base, **g} for g, _ in grid])
        for c in frame.columns:
            if c not in cells.columns:
                cells[c] = frame[c].iloc[0]
        wts = np.array([w for _, w in grid])
        X = wts @ design.build(cells)
        if deriv_var is None:
            return X[None, :]
        D = wts @ design.derivative(cells, deriv_var, eps)
        return X[None, :], D[None, :]

    # -- inference -------------------------------------------------------
    def inference_row(
        self, name: str, est: float, grad: np.ndarray, label: Optional[str] = None
    ) -> Dict[str, Any]:
        grad = np.asarray(grad, dtype=float).reshape(-1)
        V = require_covariance(self.cov_source, grad.reshape(1, -1), label or "margins")
        se = float(np.sqrt(max(float(grad @ V @ grad), 0.0)))
        return _inference_dict(float(est), se, self.df, self.alpha, name)

    def vcov(self) -> np.ndarray:
        return _get_vcov(self.cov_source)

    def attrs(self) -> Dict[str, Any]:
        return {
            "n": int(len(self.frame)),
            "n_dropped": int(self.n_dropped),
            "weights": self.weights_name,
            "design_backend": self.design.backend,
            "prediction_scale": _SCALE_LABEL.get(self.link, self.link),
        }


class _IndexScaleView:
    """A result reported as exp(b) (``irr=True``), viewed on the b scale."""

    def __init__(self, result: Any) -> None:
        self.params = np.log(result.params)
        self.std_errors = result.std_errors / result.params
        self.data_info = getattr(result, "data_info", None)
        self.model_info = getattr(result, "model_info", None)


_SCALE_LABEL = {
    "identity": "linear prediction",
    "logit": "Pr(y=1)",
    "probit": "Pr(y=1)",
    "cloglog": "Pr(y=1)",
    "log": "expected count (exp(xb + offset))",
}


def _inference_dict(
    est: float, se: float, df: float, alpha: float, name: str
) -> Dict[str, Any]:
    finite = np.isfinite(df)
    crit = stats.t.ppf(1 - alpha / 2, df) if finite else stats.norm.ppf(1 - alpha / 2)
    stat = est / se if se > 0 else float("nan")
    pv = float(2 * (stats.t.sf(abs(stat), df) if finite else stats.norm.sf(abs(stat))))
    return {
        "variable": name,
        "dy/dx": est,
        "se": se,
        "z": stat,
        "pvalue": pv,
        "ci_lower": est - crit * se,
        "ci_upper": est + crit * se,
    }


def _mu_and_slope(link: str, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prediction ``mu = g^{-1}(eta)`` and ``d mu / d eta``."""
    if link == "identity":
        return eta, np.ones_like(eta)
    if link == "logit":
        p = 1.0 / (1.0 + np.exp(-eta))
        return p, p * (1.0 - p)
    if link == "probit":
        return stats.norm.cdf(eta), stats.norm.pdf(eta)
    if link == "cloglog":
        e = np.exp(eta)
        return 1.0 - np.exp(-e), e * np.exp(-e)
    mu = np.exp(eta)
    return mu, mu


def _same_level(a: Any, b: Any) -> bool:
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _level_label(level: Any) -> str:
    if isinstance(level, (float, np.floating)) and float(level).is_integer():
        return str(int(level))
    return str(level)


def _estimation_sample(
    result: Any, frame: pd.DataFrame, design: Any, restrict: bool
) -> Tuple[pd.DataFrame, int]:
    """Rows of ``frame`` in Stata's ``e(sample)``.

    Complete on the dependent variable (when present), every model variable,
    and the weight / offset / exposure columns. The stored design
    (``data=None``) is already the estimation sample.
    """
    if not restrict:
        return frame, 0
    cols = [v for v in design.variables if v in frame.columns]
    dep = (getattr(result, "data_info", None) or {}).get("dependent_var")
    if isinstance(dep, str) and dep in frame.columns:
        cols.append(dep)
    for key in ("weights", "offset", "exposure", "cluster"):
        spec = model_setting(result, key)
        specs = spec if isinstance(spec, (list, tuple)) else [spec]
        cols.extend(c for c in specs if isinstance(c, str) and c in frame.columns)
    if not cols:
        return frame, 0
    keep = frame[cols].notna().all(axis=1)
    n_drop = int((~keep).sum())
    kept = frame.loc[keep] if n_drop else frame
    nobs = (getattr(result, "data_info", None) or {}).get("nobs")
    if n_drop and isinstance(nobs, (int, np.integer)) and len(kept) != int(nobs):
        warnings.warn(
            StatsPAIWarning(
                f"margins: averaging over {len(kept)} complete row(s) of data "
                f"({n_drop} dropped for missing model variables), but the fit's "
                f"estimation sample had {int(nobs)} rows — data may not be the "
                "estimation data.",
                recovery_hint="Pass the data the model was fitted on.",
                diagnostics={"n_used": len(kept), "n_dropped": n_drop, "nobs": nobs},
            ),
            stacklevel=4,
        )
    return kept, n_drop


def _averaging_weights(
    result: Any, frame: pd.DataFrame, data: Optional[pd.DataFrame]
) -> Tuple[Any, Optional[str]]:
    """Weights Stata ``margins`` averages with (the fit's own weights)."""
    spec = model_setting(result, "weights")
    if spec is None:
        return None, None
    if isinstance(spec, str):
        if spec in frame.columns:
            return frame[spec].to_numpy(dtype=float), spec
        return spec, spec  # sentinel: needed only when the average varies
    w = np.asarray(spec, dtype=float).ravel()
    nobs = (getattr(result, "data_info", None) or {}).get("nobs")
    if len(w) == len(frame) and (data is None or len(frame) == nobs):
        return w, "array"
    raise MethodIncompatibility(
        "margins: the fit used array weights that cannot be aligned with the "
        "data passed to margins.",
        recovery_hint="Fit with weights='<column>' so margins can align them.",
    )


# ---------------------------------------------------------------------------
# Index-model margins: prediction scale, design and its derivative
# ---------------------------------------------------------------------------
#
# ``margins`` used to return the index coefficients for every model without an
# interaction term -- so after ``sp.logit`` it reported beta (0.568) where
# Stata's ``margins, dydx(*)`` reports the average marginal effect on Pr(y)
# (0.126) -- and, with interactions, a "standard error" std(dydx)/sqrt(n) that
# is not a delta-method SE at all. Both are replaced by the delta method on
# the model's own prediction scale, using the full coefficient covariance.

_LINKS_SUPPORTED = ("identity", "logit", "probit", "cloglog", "log")


def _response_link(result: Any) -> str:
    """The inverse link of the default prediction, as a name."""
    model_info = getattr(result, "model_info", None) or {}
    data_info = getattr(result, "data_info", None) or {}
    link = model_info.get("link")
    link_obj = data_info.get("link_obj")
    if link_obj is not None:
        link = getattr(link_obj, "name", link)
    if isinstance(link, str) and link.lower() in _LINKS_SUPPORTED:
        return link.lower()
    likelihood_fit = (
        data_info.get("inference") == "z" and model_info.get("vcov_type") is None
    )
    if link is not None or likelihood_fit:
        raise MethodIncompatibility(
            f"margins: the prediction scale of this model (link={link!r}) is "
            "not supported; reporting index coefficients would not be "
            "marginal effects.",
            recovery_hint=(
                "margins supports linear models and logit / probit / cloglog / "
                "log-link (poisson, nbreg, glm) index models."
            ),
        )
    return "identity"


def _link_derivatives(link: str, eta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """``(d mu / d eta, d2 mu / d eta2)`` for the inverse link."""
    if link == "identity":
        return np.ones_like(eta), np.zeros_like(eta)
    if link == "logit":
        p = 1.0 / (1.0 + np.exp(-eta))
        f = p * (1.0 - p)
        return f, f * (1.0 - 2.0 * p)
    if link == "probit":
        f = stats.norm.pdf(eta)
        return f, -eta * f
    if link == "cloglog":
        e = np.exp(eta)
        f = e * np.exp(-e)
        return f, f * (1.0 - e)
    mu = np.exp(eta)  # log
    return mu, mu


def _margins_frame(result: Any, data: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Covariate values to average over: ``data`` or the stored design."""
    if data is not None:
        return data
    data_info = getattr(result, "data_info", None) or {}
    X = data_info.get("X")
    names = data_info.get("var_names")
    if X is not None and names is not None and np.shape(X)[1] == len(names):
        return pd.DataFrame(np.asarray(X, dtype=float), columns=list(names))
    raise MethodIncompatibility(
        "margins needs the covariate values: pass data=<estimation sample>.",
    )


def _offset(result: Any, frame: pd.DataFrame) -> np.ndarray:
    """Offset / log-exposure in the linear predictor, when the fit had one."""
    total = np.zeros(len(frame))
    for key, transform in (("offset", lambda v: v), ("exposure", np.log)):
        spec = model_setting(result, key)
        if spec is None:
            continue
        if isinstance(spec, str) and spec in frame.columns:
            total = total + transform(frame[spec].to_numpy(dtype=float))
        else:
            raise MethodIncompatibility(
                f"margins: the fit used {key}={spec!r}, which is not a column of "
                "the data passed to margins.",
                recovery_hint="Pass data= containing the offset/exposure column.",
            )
    return total


def _get_vcov(result: Any) -> np.ndarray:
    """Full coefficient covariance for the gradient-based margins functions.

    ``margins_at`` / ``contrast`` / ``pwcompare`` combine several coefficients
    through a gradient, so the diagonal-of-standard-errors fallback used before
    silently dropped every covariance term. Without a stored matrix that
    matches the reported standard errors they now refuse.
    """
    V, _ = coefficient_covariance(result)
    if V is None:
        raise MethodIncompatibility(
            "this margins function combines several coefficients and needs "
            "their full covariance matrix, which the result does not carry.",
            recovery_hint="Refit with an estimator that stores data_info['var_cov'].",
        )
    return V


def marginsplot(
    margins_df: pd.DataFrame,
    ax: Any = None,
    figsize: Tuple[float, float] = (8, 5),
    color: str = "#2C3E50",
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Plot marginal effects with confidence intervals.

    Parameters
    ----------
    margins_df : pd.DataFrame
        Output from ``margins()``.
    ax : matplotlib Axes, optional
    figsize : tuple
    color : str
    title : str, optional

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> result = sp.regress("log_wage ~ education + experience + female",
    ...                     data=df)
    >>> me = sp.margins(result, data=df)
    >>> fig, ax = sp.marginsplot(me, title='AME of wage determinants')
    >>> fig.savefig('margins.png')  # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    vars = margins_df["variable"].values
    dydx = margins_df["dy/dx"].values
    lo = margins_df["ci_lower"].values
    hi = margins_df["ci_upper"].values

    y_pos = np.arange(len(vars))

    ax.scatter(dydx, y_pos, color=color, s=50, zorder=5)
    ax.errorbar(
        dydx,
        y_pos,
        xerr=[dydx - lo, hi - dydx],
        fmt="none",
        color=color,
        capsize=4,
        linewidth=1.5,
        zorder=3,
    )
    ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(vars)
    ax.invert_yaxis()
    ax.set_xlabel("Marginal Effect (dy/dx)")
    ax.set_title(title or "Average Marginal Effects")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# margins_at: Predictive margins at specific covariate values
# ---------------------------------------------------------------------------


def _level_margins(
    ctx: "_MarginsContext", variable: str, levels: List[Any]
) -> Tuple[Dict[Any, float], Dict[Any, np.ndarray]]:
    """Predictive margin and its gradient with ``variable`` set to each level."""
    margins_: Dict[Any, float] = {}
    grads: Dict[Any, np.ndarray] = {}
    for lev in levels:
        eta, X = ctx.index_at({variable: lev})
        mu, slope = _mu_and_slope(ctx.link, eta)
        margins_[lev] = float(ctx.mean(mu))
        grads[lev] = np.asarray(ctx.mean(slope[:, None] * X), dtype=float)
    return margins_, grads


def _require_model_variable(ctx: "_MarginsContext", variable: str, fn: str) -> None:
    if variable not in ctx.design.variables:
        raise MethodIncompatibility(
            f"{fn}: {variable!r} is not in the model "
            "(Stata: 'not found in list of covariates', r(322)).",
            recovery_hint=f"Model variables: {ctx.design.variables}.",
        )


def _t_or_z(df: float) -> Tuple[Any, Any]:
    """(ppf, sf) of the reference law: t(df) after regress, N(0,1) after ML."""
    if np.isfinite(df):
        return (lambda q: stats.t.ppf(q, df)), (lambda x: stats.t.sf(x, df))
    return stats.norm.ppf, stats.norm.sf


def margins_at(
    result: Any,
    data: pd.DataFrame,
    at: Dict[str, Any],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute predictive margins at specific covariate values.

    Equivalent to Stata's ``margins, at(experience=(1 5 10 15 20))``.

    For each combination of *at* values, every observation has the *at*
    variables set to those values while all other covariates stay at their
    observed levels.  The prediction -- on the model's default scale: the
    linear prediction after ``regress``, ``Pr(y=1)`` after ``logit`` /
    ``probit``, the expected count after ``poisson`` -- is averaged over the
    estimation sample (weighted when the fit was) to give the *predictive
    margin* at that point, with delta-method SEs and t(df) / z inference as
    the fit's own.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result (must have ``.params`` and associated data).
    data : pd.DataFrame
        Data to compute margins on.
    at : dict
        Mapping of variable names to lists/arrays of values.
        If multiple variables are given, the Cartesian product of all
        value lists is used.  Example::

            at={"experience": [1, 5, 10], "female": [0, 1]}

        produces 6 grid points. Every key must be a model variable.
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    pd.DataFrame
        One row per grid point with columns for each *at* variable,
        plus ``margin``, ``se``, ``ci_lower``, ``ci_upper``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "experience": rng.integers(0, 20, size=200).astype(float),
    ...     "female": rng.integers(0, 2, size=200).astype(float),
    ... })
    >>> df["wage"] = (10.0 + 0.5 * df["experience"]
    ...               - 1.0 * df["female"] + rng.normal(size=200))
    >>> result = sp.regress(
    ...     "wage ~ experience + female + experience:female", data=df
    ... )
    >>> m = sp.margins_at(result, data=df, at={"experience": [1, 5, 10, 15, 20]})
    >>> m.shape[0]
    5
    >>> m.columns.tolist()
    ['experience', 'margin', 'se', 'ci_lower', 'ci_upper']
    """
    ctx = _MarginsContext(result, data, alpha=alpha)
    for v in at:
        _require_model_variable(ctx, v, "margins_at")
    V = ctx.vcov()
    ppf, _ = _t_or_z(ctx.df)
    crit = ppf(1 - alpha / 2)

    at_vars = list(at.keys())
    at_values = [np.atleast_1d(at[v]).tolist() for v in at_vars]
    rows = []
    for point in itertools_product(*at_values):
        point_dict = dict(zip(at_vars, point))
        eta, X = ctx.index_at(point_dict)
        mu, slope = _mu_and_slope(ctx.link, eta)
        margin = float(ctx.mean(mu))
        gradient = np.asarray(ctx.mean(slope[:, None] * X), dtype=float)
        se = float(np.sqrt(max(float(gradient @ V @ gradient), 0.0)))
        row = dict(point_dict)
        row.update(
            {
                "margin": margin,
                "se": se,
                "ci_lower": margin - crit * se,
                "ci_upper": margin + crit * se,
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs.update(ctx.attrs())
    return out


# ---------------------------------------------------------------------------
# margins_at_plot: Visualise predictive margins
# ---------------------------------------------------------------------------


def margins_at_plot(
    margins_at_df: pd.DataFrame,
    x: Optional[str] = None,
    by: Optional[str] = None,
    ax: Any = None,
    figsize: Tuple[float, float] = (8, 5),
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Predicted Value",
    palette: Optional[List[str]] = None,
) -> Tuple[Any, Any]:
    """
    Plot predictive margins from ``margins_at()`` with confidence bands.

    Parameters
    ----------
    margins_at_df : pd.DataFrame
        Output from ``margins_at()``.
    x : str, optional
        Variable to place on the x-axis.  If *None*, inferred as the
        at-variable with the most unique values.
    by : str, optional
        Variable to produce separate lines for (legend grouping).
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    xlabel : str, optional
    ylabel : str
    palette : list of str, optional
        Colours for each ``by`` group.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame({
    ...     "experience": rng.integers(0, 20, size=200).astype(float),
    ...     "female": rng.integers(0, 2, size=200).astype(float),
    ... })
    >>> df["wage"] = (10.0 + 0.5 * df["experience"]
    ...               - 1.0 * df["female"] + rng.normal(size=200))
    >>> res = sp.regress(
    ...     "wage ~ experience + female + experience:female", data=df
    ... )
    >>> m = sp.margins_at(
    ...     res, data=df, at={"experience": [1, 5, 10, 15, 20]}
    ... )
    >>> fig, ax = sp.margins_at_plot(m, x="experience")
    >>> fig.savefig("margins_at.png")  # doctest: +SKIP
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.  Install: pip install matplotlib")

    # Detect at-variable columns (everything except margin/se/ci_*)
    meta_cols = {"margin", "se", "ci_lower", "ci_upper"}
    at_cols = [c for c in margins_at_df.columns if c not in meta_cols]

    if x is None:
        # Pick the at-variable with the most unique values
        x = max(at_cols, key=lambda c: margins_at_df[c].nunique())

    if by is None:
        remaining = [c for c in at_cols if c != x]
        if remaining:
            by = remaining[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    default_palette = [
        "#2C3E50",
        "#E74C3C",
        "#3498DB",
        "#2ECC71",
        "#9B59B6",
        "#F39C12",
        "#1ABC9C",
        "#E67E22",
    ]
    colors = palette or default_palette

    if by is not None:
        groups = margins_at_df[by].unique()
        for idx, grp in enumerate(groups):
            sub = margins_at_df[margins_at_df[by] == grp].sort_values(x)
            color = colors[idx % len(colors)]
            ax.plot(sub[x], sub["margin"], marker="o", color=color, label=f"{by}={grp}")
            ax.fill_between(
                sub[x], sub["ci_lower"], sub["ci_upper"], alpha=0.15, color=color
            )
        ax.legend()
    else:
        sub = margins_at_df.sort_values(x)
        ax.plot(sub[x], sub["margin"], marker="o", color=colors[0])
        ax.fill_between(
            sub[x], sub["ci_lower"], sub["ci_upper"], alpha=0.20, color=colors[0]
        )

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    ax.set_title(title or "Predictive Margins")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# contrast: Contrasts of predictive margins
# ---------------------------------------------------------------------------


def contrast(
    result: Any,
    data: pd.DataFrame,
    variable: str,
    method: str = "r",
    reference: Any = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Compute contrasts of predictive margins across levels of a variable.

    Equivalent to Stata's ``margins <var>, contrast(ar)`` / ``contrast(r)``.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result.
    data : pd.DataFrame
        Estimation data.
    variable : str
        Categorical variable whose levels are contrasted.
    method : str, default 'r'
        Contrast type:

        - ``'r'`` (reference): each level vs *reference* level.
        - ``'ar'`` (adjacent): each level vs the previous level.
        - ``'gw'`` (grand-mean weighted): each level vs the weighted
          grand mean of all levels.
    reference : scalar, optional
        Reference level when ``method='r'``.  Defaults to the smallest
        observed level.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    pd.DataFrame
        Columns: ``contrast_label``, ``contrast``, ``se``, ``z``,
        ``pvalue``, ``ci_lower``, ``ci_upper``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> group = rng.integers(0, 3, size=n).astype(float)
    >>> df = pd.DataFrame({
    ...     "group": group,
    ...     "experience": rng.normal(10, 3, size=n),
    ... })
    >>> df["wage"] = (10 + 2.0 * df["group"]
    ...               + 0.3 * df["experience"] + rng.normal(size=n))
    >>> result = sp.regress("wage ~ group + experience", data=df)
    >>> c = sp.contrast(result, data=df, variable="group",
    ...                 method="r", reference=0)
    >>> c["contrast_label"].tolist()
    ['1.0 vs 0', '2.0 vs 0']
    """
    ctx = _MarginsContext(result, data, alpha=alpha)
    _require_model_variable(ctx, variable, "contrast")
    V = ctx.vcov()
    ppf, sf = _t_or_z(ctx.df)
    crit = ppf(1 - alpha / 2)

    levels = sorted(ctx.frame[variable].dropna().unique())
    level_margins, level_grads = _level_margins(ctx, variable, levels)

    def _row(label: str, diff: float, grad_diff: np.ndarray) -> Dict[str, Any]:
        se = float(np.sqrt(max(float(grad_diff @ V @ grad_diff), 0.0)))
        z = diff / se if se > 0 else 0.0
        return {
            "contrast_label": label,
            "contrast": diff,
            "se": se,
            "z": z,
            "pvalue": float(2 * sf(abs(z))),
            "ci_lower": diff - crit * se,
            "ci_upper": diff + crit * se,
        }

    rows = []
    if method == "r":
        if reference is None:
            reference = levels[0]
        ref_key = next((lv for lv in levels if _same_level(lv, reference)), None)
        if ref_key is None:
            raise MethodIncompatibility(
                f"contrast: reference level {reference!r} is not observed in "
                f"{variable!r} (levels {levels}).",
            )
        for lev in levels:
            if lev == ref_key:
                continue
            rows.append(
                _row(
                    f"{lev} vs {reference}",
                    level_margins[lev] - level_margins[ref_key],
                    level_grads[lev] - level_grads[ref_key],
                )
            )
    elif method == "ar":
        for i in range(1, len(levels)):
            lev, prev = levels[i], levels[i - 1]
            rows.append(
                _row(
                    f"{lev} vs {prev}",
                    level_margins[lev] - level_margins[prev],
                    level_grads[lev] - level_grads[prev],
                )
            )
    elif method == "gw":
        col = ctx.frame[variable]
        shares = {
            lev: float(ctx.mean((col == lev).to_numpy(dtype=float))) for lev in levels
        }
        grand_margin = sum(shares[lev] * level_margins[lev] for lev in levels)
        grand_grad = sum(shares[lev] * level_grads[lev] for lev in levels)
        for lev in levels:
            rows.append(
                _row(
                    f"{lev} vs grand mean",
                    level_margins[lev] - grand_margin,
                    level_grads[lev] - grand_grad,
                )
            )
    else:
        raise ValueError(f"Unknown contrast method '{method}'. Use 'r', 'ar', or 'gw'.")

    out = pd.DataFrame(rows)
    out.attrs.update(ctx.attrs())
    return out


# ---------------------------------------------------------------------------
# pwcompare: Pairwise comparisons of predictive margins
# ---------------------------------------------------------------------------


def pwcompare(
    result: Any,
    data: pd.DataFrame,
    variable: str,
    adjust: str = "none",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Pairwise comparisons of predictive margins across all levels.

    Equivalent to Stata's ``pwcompare <var>``.

    Parameters
    ----------
    result : EconometricResults
        Fitted model result.
    data : pd.DataFrame
        Estimation data.
    variable : str
        Categorical variable whose levels are compared pairwise.
    adjust : str, default 'none'
        P-value adjustment method:

        - ``'none'``: unadjusted.
        - ``'bonferroni'``: Bonferroni correction.
        - ``'sidak'``: Sidak correction.
        - ``'holm'``: Holm step-down procedure.
    alpha : float, default 0.05
        Significance level for (adjusted) confidence intervals.

    Returns
    -------
    pd.DataFrame
        Columns: ``comparison``, ``diff``, ``se``, ``z``, ``pvalue``,
        ``pvalue_adj``, ``ci_lower``, ``ci_upper``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> group = rng.integers(0, 3, size=n).astype(float)
    >>> df = pd.DataFrame({
    ...     "group": group,
    ...     "experience": rng.normal(10, 3, size=n),
    ... })
    >>> df["wage"] = (10 + 2.0 * df["group"]
    ...               + 0.3 * df["experience"] + rng.normal(size=n))
    >>> result = sp.regress("wage ~ group + experience", data=df)
    >>> pw = sp.pwcompare(result, data=df, variable="group",
    ...                   adjust="bonferroni")
    >>> bool((pw["pvalue_adj"] >= pw["pvalue"]).all())
    True
    """
    ctx = _MarginsContext(result, data, alpha=alpha)
    _require_model_variable(ctx, variable, "pwcompare")
    V = ctx.vcov()
    ppf, sf = _t_or_z(ctx.df)

    levels = sorted(ctx.frame[variable].dropna().unique())
    level_margins, level_grads = _level_margins(ctx, variable, levels)

    pairs = [
        (levels[i], levels[j])
        for i in range(len(levels))
        for j in range(i + 1, len(levels))
    ]
    n_comp = len(pairs)
    rows = []
    for lev_a, lev_b in pairs:
        diff = level_margins[lev_b] - level_margins[lev_a]
        grad_diff = level_grads[lev_b] - level_grads[lev_a]
        se = float(np.sqrt(max(float(grad_diff @ V @ grad_diff), 0.0)))
        z = diff / se if se > 0 else 0.0
        rows.append(
            {
                "comparison": f"{lev_b} vs {lev_a}",
                "diff": diff,
                "se": se,
                "z": z,
                "pvalue": float(2 * sf(abs(z))),
            }
        )

    raw_pvals = [r["pvalue"] for r in rows]
    adj_pvals = _adjust_pvalues(raw_pvals, method=adjust, n_comparisons=n_comp)
    alpha_adj = _adjusted_alpha(alpha, adjust, n_comp)
    crit = ppf(1 - alpha_adj / 2)
    for r, padj in zip(rows, adj_pvals):
        r["pvalue_adj"] = padj
        r["ci_lower"] = r["diff"] - crit * r["se"]
        r["ci_upper"] = r["diff"] + crit * r["se"]

    out = pd.DataFrame(rows)
    out.attrs.update(ctx.attrs())
    return out


def _adjust_pvalues(
    pvals: Any,
    method: str,
    n_comparisons: int,
) -> List[float]:
    """Apply multiple-comparison correction to p-values."""
    pvals = np.asarray(pvals, dtype=float)
    if method == "none":
        return [float(p) for p in pvals.tolist()]
    elif method == "bonferroni":
        return [float(p) for p in np.minimum(pvals * n_comparisons, 1.0).tolist()]
    elif method == "sidak":
        # 1 - (1 - p)^m cancels to exactly 0 for p < ~1e-17; the log1p /
        # expm1 form keeps full precision (Stata reports 7.0e-26 there).
        sidak = -np.expm1(n_comparisons * np.log1p(-pvals))
        return [float(p) for p in np.minimum(sidak, 1.0).tolist()]
    elif method == "holm":
        n = len(pvals)
        order = np.argsort(pvals)
        adj = np.empty(n)
        for rank, idx in enumerate(order):
            adj[idx] = min(pvals[idx] * (n_comparisons - rank), 1.0)
        # Enforce monotonicity
        cum_max = 0.0
        for idx in order:
            cum_max = max(cum_max, adj[idx])
            adj[idx] = cum_max
        return [float(p) for p in adj.tolist()]
    else:
        raise ValueError(
            f"Unknown adjustment method '{method}'. "
            "Use 'none', 'bonferroni', 'sidak', or 'holm'."
        )


def _adjusted_alpha(alpha: float, method: str, n_comparisons: int) -> float:
    """Return adjusted significance level for CI construction."""
    if method == "none":
        return alpha
    elif method == "bonferroni":
        return alpha / n_comparisons
    elif method == "sidak":
        return float(-np.expm1(np.log1p(-alpha) / n_comparisons))
    elif method == "holm":
        # Conservative: use Bonferroni alpha for CIs
        return alpha / n_comparisons
    else:
        return alpha


# ---------------------------------------------------------------------------
# margins_table — adapter that wraps a margins DataFrame so it pipes
# straight into sp.regtable for publication-quality marginal-effects
# tables. Mirrors the R workflow ``modelsummary(avg_slopes(model))`` and
# closes the "estimator → marginal-effects table" gap that previously
# required users to hand-build add_rows.
# ---------------------------------------------------------------------------


class _MarginsResult:
    """Duck-typed result wrapping a ``margins`` DataFrame.

    Exposes the attributes ``_extract_model_data`` (in ``output/estimates``)
    looks for: ``params`` / ``std_errors`` / ``tvalues`` / ``pvalues`` /
    ``conf_int_lower`` / ``conf_int_upper`` / ``diagnostics`` /
    ``data_info`` / ``model_info``. Behaves like an ``EconometricResults``
    for the purposes of ``regtable``.
    """

    def __init__(
        self,
        margins_df: pd.DataFrame,
        *,
        n_obs: Optional[int] = None,
        method: str = "ame",
    ) -> None:
        if "variable" not in margins_df.columns:
            raise ValueError(
                "margins_table requires a DataFrame with a 'variable' column "
                "(produced by sp.margins). Got columns: "
                f"{list(margins_df.columns)}"
            )
        idx = margins_df["variable"].astype(str).tolist()
        # ``dy/dx`` may be NaN for binary/categorical rows that report a
        # discrete change instead — the helper below is defensive.
        col_dydx = "dy/dx" if "dy/dx" in margins_df.columns else "diff"
        col_p = "pvalue_adj" if "pvalue_adj" in margins_df.columns else "pvalue"
        self.params = pd.Series(margins_df[col_dydx].astype(float).values, index=idx)
        self.std_errors = pd.Series(margins_df["se"].astype(float).values, index=idx)
        # margins reports z-stat (z under normal); we name it tvalues so
        # the rest of regtable's machinery treats it as a t-style ratio
        # without special-casing.
        z_col = "z" if "z" in margins_df.columns else "t"
        if z_col in margins_df.columns:
            self.tvalues = pd.Series(margins_df[z_col].astype(float).values, index=idx)
        else:
            self.tvalues = pd.Series(np.nan, index=idx)
        self.pvalues = pd.Series(margins_df[col_p].astype(float).values, index=idx)
        if "ci_lower" in margins_df.columns:
            self.conf_int_lower = pd.Series(
                margins_df["ci_lower"].astype(float).values, index=idx
            )
            self.conf_int_upper = pd.Series(
                margins_df["ci_upper"].astype(float).values, index=idx
            )
        else:
            self.conf_int_lower = pd.Series(np.nan, index=idx)
            self.conf_int_upper = pd.Series(np.nan, index=idx)
        # Minimal diagnostics so ``stats=["N"]`` still works.
        self.diagnostics = {"N": n_obs} if n_obs is not None else {}
        self.data_info = {"nobs": n_obs} if n_obs is not None else {}
        # Tag so users can introspect (and we leave a breadcrumb in the
        # rendered table footer via ``method`` later if desired).
        self.model_info = {"model_type": f"Marginal effects ({method})"}
        # df_resid omitted — margins inference is z-based, regtable falls
        # back to standard normal which is correct here.

    def __repr__(self) -> str:
        n = len(self.params)
        return f"<MarginsResult: {n} marginal effect{'s' if n != 1 else ''}>"


def event_study_table(
    result: Any,
    *,
    regex: Optional[str] = None,
    label_fmt: str = "t={t}",
    include_reference: bool = False,
) -> _MarginsResult:
    """Adapter that turns an event-study fit into a regtable input.

    Two extraction paths:

    1. **CausalResult fast path** — when ``result.model_info`` carries an
       ``"event_study"`` DataFrame (the canonical shape produced by
       :func:`sp.event_study`), read its ``relative_time`` /
       ``estimate`` / ``se`` / ``ci_lower`` / ``ci_upper`` / ``pvalue``
       columns directly. Reference period (``estimate==0`` and
       ``se==0``) is dropped by default (set ``include_reference=True``
       to keep it visible).

    2. **Regex path** — when ``regex`` is provided, scan
       ``result.params.index`` for coefficients matching the pattern
       and use the first capture group as the relative time.

    Rows are sorted by relative time (so the rendered table reads
    ``t=-3, t=-2, …, t=+3`` regardless of the underlying coefficient
    name encoding).

    Parameters
    ----------
    result : CausalResult or EconometricResults
        Fitted model. The CausalResult fast path is used automatically
        when ``model_info['event_study']`` is present.
    regex : str, optional
        Pattern with one capture group that matches the relative time
        in coefficient names. Required when the CausalResult fast path
        is not applicable. Examples: ``r"^tau_(-?\\d+)$"``,
        ``r"^lag(\\d+)$"``, ``r"::(-?\\d+)$"``.
    label_fmt : str, default ``"t={t}"``
        Format string for the row label of each event-time bin. The
        ``{t}`` placeholder receives the integer relative time.
    include_reference : bool, default False
        Whether to render the reference period row (typically
        ``t=-1``) where the estimate is identically zero.

    Returns
    -------
    _MarginsResult
        Duck-typed result accepted directly by :func:`sp.regtable`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(7)
    >>> rows = []
    >>> for u in range(60):
    ...     tt = rng.choice([3, 4, 5, 100])  # 100 marks never-treated
    ...     for t in range(8):
    ...         post = 1.0 if (tt < 100 and t >= tt) else 0.0
    ...         y = 0.3 * u / 60 + 0.1 * t + 1.5 * post + rng.normal(scale=0.5)
    ...         rows.append({"unit": u, "time": t, "treat_time": tt, "y": y})
    >>> panel = pd.DataFrame(rows)
    >>> r = sp.event_study(panel, y="y", treat_time="treat_time",
    ...                    time="time", unit="unit", window=(-3, 3))
    >>> tbl = sp.event_study_table(r)
    >>> _ = sp.regtable(tbl, title="Event study", output="text")
    """
    mi = getattr(result, "model_info", {}) or {}
    es_df = mi.get("event_study") if isinstance(mi, dict) else None
    rows: List[Tuple[int, float, float, float, float, float]] = []

    if isinstance(es_df, pd.DataFrame) and "relative_time" in es_df.columns:
        for _, row in es_df.iterrows():
            t = int(row["relative_time"])
            est = float(row.get("estimate", row.get("dy/dx", np.nan)))
            se = float(row.get("se", np.nan))
            lo = float(row.get("ci_lower", np.nan))
            hi = float(row.get("ci_upper", np.nan))
            pv = float(row.get("pvalue", np.nan))
            if not include_reference and abs(est) < 1e-15 and abs(se) < 1e-15:
                continue
            rows.append((t, est, se, lo, hi, pv))
    elif regex is not None:
        params = getattr(result, "params", None)
        if params is None:
            raise ValueError(
                "event_study_table: result has no .params; cannot use "
                "regex extraction path."
            )
        std_errors = getattr(result, "std_errors", pd.Series())
        pvalues = getattr(result, "pvalues", pd.Series())
        ci_lo = getattr(result, "conf_int_lower", pd.Series())
        ci_hi = getattr(result, "conf_int_upper", pd.Series())
        rx = re.compile(regex)
        for name in params.index:
            m = rx.search(str(name))
            if not m:
                continue
            try:
                t = int(m.group(1))
            except (IndexError, ValueError):
                continue
            est = float(params.get(name, np.nan))
            se = (
                float(std_errors.get(name, np.nan))
                if name in std_errors.index
                else np.nan
            )
            pv = float(pvalues.get(name, np.nan)) if name in pvalues.index else np.nan
            lo = float(ci_lo.get(name, np.nan)) if name in ci_lo.index else np.nan
            hi = float(ci_hi.get(name, np.nan)) if name in ci_hi.index else np.nan
            rows.append((t, est, se, lo, hi, pv))
    else:
        raise ValueError(
            "event_study_table: result has no model_info['event_study'] "
            "DataFrame and no `regex` was provided. Pass regex=... to "
            "extract event-time coefficients from result.params.index."
        )

    if not rows:
        raise ValueError(
            "event_study_table: no event-time rows extracted. Check "
            "the regex pattern or include_reference=True."
        )

    rows.sort(key=lambda r: r[0])
    labels = [label_fmt.format(t=t) for t, *_ in rows]
    df = pd.DataFrame(
        {
            "variable": labels,
            "dy/dx": [r[1] for r in rows],
            "se": [r[2] for r in rows],
            "ci_lower": [r[3] for r in rows],
            "ci_upper": [r[4] for r in rows],
            "pvalue": [r[5] for r in rows],
        }
    )
    # Synthesise a z-stat for the ``tvalues`` slot (regtable uses it
    # only when se_type='t', and event studies almost always show
    # estimates with SE — but populate consistently).
    with np.errstate(divide="ignore", invalid="ignore"):
        df["z"] = df["dy/dx"] / df["se"]
    n_obs = None
    diag = getattr(result, "diagnostics", {}) or {}
    dinfo = getattr(result, "data_info", {}) or {}
    if isinstance(diag, dict) and diag.get("N") is not None:
        n_obs = diag.get("N")
    elif isinstance(dinfo, dict) and dinfo.get("nobs") is not None:
        n_obs = dinfo.get("nobs")
    return _MarginsResult(df, n_obs=n_obs, method="event-study")


def margins_table(
    result: Any,
    data: Optional[pd.DataFrame] = None,
    variables: Optional[List[str]] = None,
    at: Optional[Dict[str, Any]] = None,
    method: str = "ame",
    eps: float = 1e-5,
    alpha: float = 0.05,
) -> _MarginsResult:
    """Marginal-effects result that pipes straight into ``sp.regtable``.

    Thin adapter over :func:`margins`: computes the marginal effects
    (same kwargs), then wraps the resulting DataFrame in a
    :class:`_MarginsResult` exposing ``params`` / ``std_errors`` /
    ``tvalues`` / ``pvalues`` / ``conf_int_*`` so that
    ``sp.regtable(margins_table(model))`` produces a publication-quality
    marginal-effects table — closing the "estimator → margins table"
    gap that previously required users to hand-build ``add_rows``.

    Mirrors the R workflow ``modelsummary(avg_slopes(model))``.

    Parameters
    ----------
    result : EconometricResults
        Fitted model — same input ``sp.margins`` accepts.
    data, variables, at, method, eps, alpha
        Forwarded verbatim to :func:`margins`.

    Returns
    -------
    _MarginsResult
        A duck-typed result object usable directly as a positional
        argument to :func:`sp.regtable`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> z = rng.normal(size=n)
    >>> p = 1 / (1 + np.exp(-(0.5 * x - 0.4 * z)))
    >>> df = pd.DataFrame({
    ...     "y": (rng.uniform(size=n) < p).astype(int),
    ...     "x": x,
    ...     "z": z,
    ... })
    >>> m = sp.logit("y ~ x + z", data=df)
    >>> mt = sp.margins_table(m)
    >>> _ = sp.regtable(mt, output="text")
    >>> sp.regtable(mt, output="latex", filename="margins.tex")  # doctest: +SKIP
    """
    df = margins(
        result,
        data=data,
        variables=variables,
        at=at,
        method=method,
        eps=eps,
        alpha=alpha,
    )
    n_obs = None
    diag = getattr(result, "diagnostics", {}) or {}
    dinfo = getattr(result, "data_info", {}) or {}
    if isinstance(diag, dict) and diag.get("N") is not None:
        n_obs = diag.get("N")
    elif isinstance(dinfo, dict) and dinfo.get("nobs") is not None:
        n_obs = dinfo.get("nobs")
    return _MarginsResult(df, n_obs=n_obs, method=method)
