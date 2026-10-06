"""Path analysis and structural equation models.

A system of linear equations fitted jointly by normal-theory maximum
likelihood, written in the model syntax of R's ``lavaan``::

    m1 ~ a1*x + w            # regressions, with optional labels
    m2 ~ a2*x
    y  ~ c*x + b1*m1 + b2*m2
    m1 ~~ m2                 # a residual covariance
    indirect := a1*b1 + a2*b2          # a function of labelled paths
    total    := c + a1*b1 + a2*b2
    ability =~ test1 + test2 + test3   # a latent variable and its indicators
    y ~ 1                              # an intercept (a mean structure)

The implied covariance matrix of all the variables is
``Sigma = A Psi A'`` with ``A = (I - B)^{-1}``, ``B`` the matrix of path
coefficients and ``Psi`` the covariance of the disturbances (the sample
covariance for the exogenous variables, which are conditioned on). The
estimates minimise ``log|Sigma| + tr(S Sigma^{-1}) - log|S| - p``.

What a single-equation regression does not give: the indirect effects through
several mediators at once with a standard error for each and for their sum,
equality constraints across equations, correlated disturbances, and a test of
the restrictions the path diagram imposes; and latent variables measured by
several error-prone indicators, whose effects a regression on any one
indicator would attenuate.

The estimation is in :mod:`statspai.structural._sem_engine`. Multiple groups
and categorical indicators are not implemented.
"""

from __future__ import annotations

import ast
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from ._sem_engine import fit_sem

__all__ = ["path_analysis", "PathAnalysisResult"]


# --------------------------------------------------------------------- result
class PathAnalysisResult(ResultProtocolMixin):
    """Fitted path model.

    Attributes
    ----------
    params : pandas.DataFrame
        One row per parameter, in lavaan's layout: ``lhs``, ``op`` (``=~``
        loading, ``~`` regression, ``~~`` (co)variance, ``~1`` intercept,
        ``:=`` defined), ``rhs``, ``label``,
        ``est``, ``se``, ``z``, ``pvalue``, ``ci_lower``, ``ci_upper`` and
        ``std_all`` (the estimate with every variable standardised).
    fit : dict
        ``chisq``, ``df``, ``pvalue`` (the test of the model against the
        saturated one), ``baseline_chisq``, ``baseline_df``, ``cfi``, ``tli``,
        ``rmsea`` with its 90% interval, ``srmr``, ``logl``, ``aic``, ``bic``
        and ``npar``; with ``se='robust'`` also the Satorra-Bentler
        ``chisq_scaled``, ``pvalue_scaled`` and ``scaling_factor``.
    implied_cov, sample_cov : pandas.DataFrame
        Model-implied and sample covariance matrices of the observed
        variables (divisor ``n``).
    implied_mean : pandas.Series or None
        Model-implied means, when the model has a mean structure.
    latent_cov : pandas.DataFrame
        Model-implied covariance matrix of the latent variables (empty
        without any).
    factor_scores : pandas.DataFrame or None
        The expected value of each latent variable given a row's observed
        values (the regression method), for the rows used in the fit.
    vcov : pandas.DataFrame
        Covariance matrix of the free parameters.
    r2 : pandas.Series
        Share of each endogenous variable's variance the model explains
        (for an indicator, its reliability).
    n_obs : int
        Rows used. ``n_dropped`` counts the rest, and ``n_patterns`` the
        distinct patterns of missing values among the rows used (one with
        complete data).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=500)
    >>> m = 0.5 * x + rng.normal(size=500)
    >>> y = 0.4 * m + 0.2 * x + rng.normal(size=500)
    >>> df = pd.DataFrame({"x": x, "m": m, "y": y})
    >>> fit = sp.path_analysis("m ~ a*x\\ny ~ b*m + c*x\\nind := a*b", df)
    >>> type(fit).__name__, int(fit.fit["df"])
    ('PathAnalysisResult', 0)
    >>> list(fit.params["op"].unique())
    ['~', '~~', ':=']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        params: pd.DataFrame,
        fit: Dict[str, Any],
        implied_cov: pd.DataFrame,
        sample_cov: pd.DataFrame,
        vcov: pd.DataFrame,
        r2: pd.Series,
        n_obs: int,
        model: str,
        se: str,
        n_dropped: int = 0,
        latent_cov: Optional[pd.DataFrame] = None,
        implied_mean: Optional[pd.Series] = None,
        factor_scores: Optional[pd.DataFrame] = None,
        n_patterns: int = 1,
    ) -> None:
        self.params = params
        self.fit = fit
        self.implied_cov = implied_cov
        self.sample_cov = sample_cov
        self.vcov = vcov
        self.r2 = r2
        self.n_obs = n_obs
        self.model = model
        self.se = se
        self.n_dropped = n_dropped
        self.latent_cov = latent_cov if latent_cov is not None else pd.DataFrame()
        self.implied_mean = implied_mean
        self.factor_scores = factor_scores
        self.n_patterns = n_patterns

    def effect(self, name: str) -> pd.Series:
        """The row of a defined (``:=``) or labelled parameter."""
        hit = self.params[
            (self.params["label"] == name)
            | ((self.params["op"] == ":=") & (self.params["lhs"] == name))
        ]
        if hit.empty:
            raise MethodIncompatibility(
                f"path_analysis: no parameter is labelled or defined as {name!r}.",
                diagnostics={
                    "labels": sorted(set(self.params["label"]) - {""}),
                },
            )
        return hit.iloc[0]

    def summary(self) -> str:
        f = self.fit
        lines = [
            ("Structural equation model" if len(self.latent_cov) else "Path analysis")
            + " (maximum likelihood)",
            "=" * 66,
            f"  Observations : {self.n_obs}"
            + (f"   ({self.n_dropped} dropped, missing)" if self.n_dropped else "")
            + (
                f"   ({self.n_patterns} missing-data patterns)"
                if self.n_patterns > 1
                else ""
            ),
            f"  Free params  : {f['npar']}",
            f"  Std. errors  : {self.se}",
            f"  Model test   : chi2({f['df']}) = {f['chisq']:.3f}"
            + (f", p = {f['pvalue']:.4f}" if f["df"] > 0 else "  (saturated)"),
        ]
        if "chisq_scaled" in f and f["df"] > 0:
            lines.append(
                f"  Scaled test  : chi2({f['df']}) = {f['chisq_scaled']:.3f}, "
                f"p = {f['pvalue_scaled']:.4f}  (Satorra-Bentler)"
            )
        if f["df"] > 0:
            lines.append(
                f"  CFI {f['cfi']:.3f}   TLI {f['tli']:.3f}   "
                f"RMSEA {f['rmsea']:.3f} [{f['rmsea_ci_lower']:.3f}, "
                f"{f['rmsea_ci_upper']:.3f}]   SRMR {f['srmr']:.3f}"
            )
        lines.append("-" * 66)
        shown = self.params.copy()
        shown.insert(
            0,
            "parameter",
            (shown["lhs"] + " " + shown["op"] + " " + shown["rhs"]).str.strip(),
        )
        shown = shown.drop(columns=["lhs", "op", "rhs"])
        with pd.option_context("display.width", 200, "display.precision", 4):
            lines.append(shown.to_string(index=False, na_rep=""))
        lines.append("-" * 66)
        lines.append(
            "R-squared: " + ", ".join(f"{k} {v:.3f}" for k, v in self.r2.items())
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- parser
_NAME = r"[A-Za-z_][A-Za-z0-9_.]*"


def _safe_eval(expr: str, values: Dict[str, float]) -> float:
    """Arithmetic on labelled parameters; nothing else is evaluated."""

    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise MethodIncompatibility(
                    f"path_analysis: {node.id!r} in a ':=' definition is not "
                    "a label of the model.",
                    recovery_hint="Label the path first, e.g. y ~ b*m.",
                )
            return values[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            v = walk(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if isinstance(node.op, ast.Div):
                return a / b
            if isinstance(node.op, ast.Pow):
                return float(a**b)
        raise MethodIncompatibility(
            f"path_analysis: cannot evaluate {expr!r}; a definition may use "
            "labels, numbers, + - * / ** and parentheses.",
        )

    try:
        tree = ast.parse(expr.strip().replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise MethodIncompatibility(
            f"path_analysis: cannot parse the definition {expr!r}."
        ) from exc
    return walk(tree)


def _parse(model: str) -> Dict[str, Any]:
    Term = Tuple[str, str, Optional[str], Optional[float]]
    loadings: List[Term] = []
    regressions: List[Term] = []
    covariances: List[Term] = []
    intercepts: List[Tuple[str, Optional[str], Optional[float]]] = []
    defined: List[Tuple[str, str]] = []
    statements: List[str] = []
    for raw in model.replace(";", "\n").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            statements.append(line)
    if not statements:
        raise MethodIncompatibility("path_analysis: the model is empty.")

    def modifier(term: str) -> Tuple[str, Optional[str], Optional[float]]:
        """``(variable, label, fixed)``; a fixed value of NaN means ``NA*``."""
        term = term.strip()
        if "*" not in term:
            return term.replace(" ", ""), None, None
        mod, var = (t.strip() for t in term.split("*", 1))
        var = var.replace(" ", "")
        if mod == "NA":
            return var, None, float("nan")
        try:
            return var, None, float(mod)
        except ValueError:
            if not re.fullmatch(_NAME, mod):
                raise MethodIncompatibility(
                    f"path_analysis: cannot read the modifier {mod!r} in "
                    f"{term!r}; use a label, a number or NA.",
                    recovery_hint="start(), equal() and c() are not supported.",
                ) from None
            return var, mod, None

    def name_of(text: str, line: str) -> str:
        if not re.fullmatch(_NAME, text):
            raise MethodIncompatibility(
                f"path_analysis: cannot read the left-hand side of {line!r}."
            )
        return text

    for line in statements:
        if ":=" in line:
            name, expr = (t.strip() for t in line.split(":=", 1))
            if not re.fullmatch(_NAME, name):
                raise MethodIncompatibility(
                    f"path_analysis: {name!r} is not a valid name for a "
                    "defined parameter."
                )
            defined.append((name, expr))
        elif "=~" in line:
            lhs, rhs = (t.strip() for t in line.split("=~", 1))
            lhs = name_of(lhs, line)
            for term in rhs.split("+"):
                var, label, fixed = modifier(term)
                loadings.append((lhs, var, label, fixed))
        elif "~~" in line:
            lhs, rhs = (t.strip() for t in line.split("~~", 1))
            lhs = name_of(lhs, line)
            for term in rhs.split("+"):
                var, label, fixed = modifier(term)
                covariances.append((lhs, var, label, fixed))
        elif "~" in line:
            lhs, rhs = (t.strip() for t in line.split("~", 1))
            lhs = name_of(lhs, line)
            for term in rhs.split("+"):
                var, label, fixed = modifier(term)
                if var == "1":
                    intercepts.append((lhs, label, fixed))
                elif var == "0":
                    intercepts.append((lhs, label, 0.0))
                else:
                    regressions.append((lhs, var, label, fixed))
        else:
            raise MethodIncompatibility(
                f"path_analysis: cannot read {line!r}; expected '=~', '~', "
                "'~~' or ':='."
            )
    if not regressions and not loadings:
        raise MethodIncompatibility(
            "path_analysis: the model has neither a regression (y ~ x) nor a "
            "latent variable (f =~ a + b + c)."
        )
    return {
        "loadings": loadings,
        "regressions": regressions,
        "covariances": covariances,
        "intercepts": intercepts,
        "defined": defined,
    }


# ----------------------------------------------------------------- estimation
def path_analysis(
    model: str,
    data: pd.DataFrame,
    *,
    se: str = "standard",
    alpha: float = 0.05,
    meanstructure: bool = False,
    std_lv: bool = False,
    growth: bool = False,
    auto_cov_y: bool = False,
    missing: str = "listwise",
) -> PathAnalysisResult:
    """Fit a path model or a structural equation model by maximum likelihood.

    Parameters
    ----------
    model : str
        The model in lavaan syntax, one statement per line (or separated by
        ``;``):

        * ``y ~ x1 + b*x2`` regresses ``y`` on ``x1`` and ``x2``; ``b`` labels
          the second coefficient. Two parameters with the same label are
          constrained to be equal. ``0.5*x`` fixes a coefficient.
        * ``f =~ a + b + c`` defines the latent variable ``f``, measured by
          the columns ``a``, ``b`` and ``c``. The first loading is fixed at
          one to give ``f`` a scale; ``NA*a`` frees it (then fix the variance
          with ``f ~~ 1*f``, or pass ``std_lv=True``). A latent variable may
          appear in regressions like any other, and may itself be an
          indicator of a higher-order one.
        * ``y1 ~~ y2`` lets two disturbances covary; ``y ~~ 0.5*y`` fixes a
          variance.
        * ``y ~ 1`` asks for an intercept, and switches the mean structure
          on; ``y ~ 0*1`` fixes it at zero.
        * ``name := expression`` defines a function of labelled parameters,
          with a delta-method standard error.
        * ``x1:x2`` on the right-hand side is the product of two columns.
        * ``#`` starts a comment.
    data : pandas.DataFrame
        One numeric column per observed variable. See ``missing`` for rows
        with missing values.
    se : {'standard', 'robust'}, default 'standard'
        ``'standard'`` uses the expected information matrix and is correct
        under multivariate normality. ``'robust'`` is the Satorra-Bentler
        sandwich built from the fourth moments of the data, with a scaled
        test statistic, for when the disturbances are heavy-tailed or
        heteroskedastic (lavaan's ``estimator = "MLM"``).
    alpha : float, default 0.05
        One minus the confidence level.
    meanstructure : bool, default False
        Model the means as well as the covariances: an intercept for every
        observed endogenous variable, latent means at zero. On its own this
        changes no other estimate; it matters once intercepts are
        constrained.
    std_lv : bool, default False
        Give each latent variable a scale by fixing its (residual) variance
        at one instead of its first loading.
    growth : bool, default False
        A latent growth curve: the intercepts of the observed variables are
        fixed at zero and the means of the latent variables are free, so the
        latent means are the average starting level and the average slope
        (lavaan's ``growth()``).
    auto_cov_y : bool, default False
        Let the disturbances of the terminal outcomes (endogenous variables
        that predict nothing) covary without being asked. ``lavaan::sem``
        does this; Stata's ``sem`` does not, and neither does the default
        here, which fits the model as written.
    missing : {'listwise', 'fiml'}, default 'listwise'
        ``'listwise'`` drops every row with a missing value in a variable of
        the model. ``'fiml'`` is full-information maximum likelihood: each
        row contributes the likelihood of the values it does have, so no
        endogenous value is thrown away. It is consistent when values are
        missing at random given the observed ones, where listwise deletion
        needs them missing completely at random; and it is more precise
        either way. Rows missing an exogenous variable are still dropped,
        since the model is conditional on those. It implies a mean
        structure, and standard errors come from the observed information
        (lavaan's ``missing = "ml"``; Stata's ``method(mlmv)``).

    Returns
    -------
    PathAnalysisResult

    Notes
    -----
    Observed variables that only appear on a right-hand side are exogenous.
    Their means, variances and covariances are held at the sample values
    (lavaan's ``fixed.x = TRUE``), so the model is conditional on them.
    Latent variables with no equation of their own covary freely with one
    another, and are uncorrelated with the exogenous observed variables.
    Covariances use the divisor ``n``, as maximum likelihood does.

    The estimates, standard errors, standardised solution, test statistic
    and fit indices reproduce ``lavaan::sem(model, data)`` (``growth()``
    with ``growth=True``), with ``se='robust'`` its ``estimator = "MLM"``
    and with ``missing='fiml'`` its ``missing = "ml"``. The one default
    that differs is ``auto_cov_y``:
    a model with two or more terminal outcomes needs ``auto_cov_y=True`` to
    reproduce ``sem()``, or the covariance written out.

    A latent variable is identified by its indicators alone only with three
    or more of them (two, if it is related to something else in the model).
    The function refuses a model whose information matrix is singular. An
    estimated variance below zero is reported with a warning.

    A path coefficient is a causal effect only if the diagram is right: no
    omitted common causes of the variables it links, and arrows in the
    stated direction. The chi-square test can reject the diagram but cannot
    confirm it, and a saturated model (``df = 0``) is not tested at all. An
    indirect effect ``a*b`` additionally needs no unmeasured confounder of
    the mediator and the outcome; :func:`statspai.mediate_sensitivity`
    probes that. A latent variable removes the bias from measurement error
    in its indicators, on the assumption that the indicators are related
    only through it.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 800
    >>> x = rng.normal(size=n)
    >>> m1 = 0.5 * x + rng.normal(size=n)
    >>> m2 = -0.4 * x + rng.normal(size=n)
    >>> y = 0.3 * x + 0.6 * m1 - 0.5 * m2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"x": x, "m1": m1, "m2": m2, "y": y})
    >>> fit = sp.path_analysis('''
    ...     m1 ~ a1*x
    ...     m2 ~ a2*x
    ...     y  ~ c*x + b1*m1 + b2*m2
    ...     indirect := a1*b1 + a2*b2
    ...     total    := c + a1*b1 + a2*b2
    ... ''', df)
    >>> bool(abs(fit.effect("indirect")["est"] - 0.5) < 0.1)
    True

    A latent predictor measured three times. The regression of ``y`` on any
    single test is attenuated; the latent-variable slope is not.

    >>> skill = rng.normal(size=n)
    >>> tests = {f"t{j}": skill + rng.normal(size=n) for j in (1, 2, 3)}
    >>> df2 = pd.DataFrame({**tests, "y": 0.8 * skill + rng.normal(size=n)})
    >>> fit2 = sp.path_analysis("skill =~ t1 + t2 + t3\\ny ~ b*skill", df2)
    >>> bool(abs(fit2.effect("b")["est"] - 0.8) < 0.15)
    True
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("path_analysis: data must be a DataFrame.")
    se = str(se).lower()
    if se not in ("standard", "robust"):
        raise MethodIncompatibility(
            f"path_analysis: se must be 'standard' or 'robust', got {se!r}."
        )
    if not (0.0 < alpha < 1.0):
        raise MethodIncompatibility("path_analysis: alpha must be between 0 and 1.")
    how = {"ml": "fiml", "mlmv": "fiml"}.get(str(missing).lower(), str(missing).lower())
    if how not in ("listwise", "fiml"):
        raise MethodIncompatibility(
            f"path_analysis: missing must be 'listwise' or 'fiml', got {missing!r}."
        )
    spec = _parse(model)
    pieces = fit_sem(
        spec,
        data,
        se=se,
        alpha=alpha,
        meanstructure=bool(meanstructure),
        std_lv=bool(std_lv),
        growth=bool(growth),
        auto_cov_y=bool(auto_cov_y),
        evaluate=_safe_eval,
        missing=how,
    )
    return PathAnalysisResult(model=model, se=se, **pieces)
