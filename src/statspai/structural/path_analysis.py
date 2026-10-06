"""Path analysis: structural equation models among observed variables.

A system of linear regressions fitted jointly by normal-theory maximum
likelihood, written in the model syntax of R's ``lavaan``::

    m1 ~ a1*x + w            # regressions, with optional labels
    m2 ~ a2*x
    y  ~ c*x + b1*m1 + b2*m2
    m1 ~~ m2                 # a residual covariance
    indirect := a1*b1 + a2*b2          # a function of labelled paths
    total    := c + a1*b1 + a2*b2

The implied covariance matrix of all the variables is
``Sigma = A Psi A'`` with ``A = (I - B)^{-1}``, ``B`` the matrix of path
coefficients and ``Psi`` the covariance of the disturbances (the sample
covariance for the exogenous variables, which are conditioned on). The
estimates minimise ``log|Sigma| + tr(S Sigma^{-1}) - log|S| - p``.

What a single-equation regression does not give: the indirect effects through
several mediators at once with a standard error for each and for their sum,
equality constraints across equations, correlated disturbances, and a test of
the restrictions the path diagram imposes.

Latent variables (``=~``) and mean structures are not implemented.
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["path_analysis", "PathAnalysisResult"]


# --------------------------------------------------------------------- result
class PathAnalysisResult(ResultProtocolMixin):
    """Fitted path model.

    Attributes
    ----------
    params : pandas.DataFrame
        One row per parameter, in lavaan's layout: ``lhs``, ``op`` (``~``
        regression, ``~~`` (co)variance, ``:=`` defined), ``rhs``, ``label``,
        ``est``, ``se``, ``z``, ``pvalue``, ``ci_lower``, ``ci_upper`` and
        ``std_all`` (the estimate with every variable standardised).
    fit : dict
        ``chisq``, ``df``, ``pvalue`` (the test of the model against the
        saturated one), ``baseline_chisq``, ``baseline_df``, ``cfi``, ``tli``,
        ``rmsea`` with its 90% interval, ``srmr``, ``logl``, ``aic``, ``bic``
        and ``npar``; with ``se='robust'`` also the Satorra-Bentler
        ``chisq_scaled``, ``pvalue_scaled`` and ``scaling_factor``.
    implied_cov, sample_cov : pandas.DataFrame
        Model-implied and sample covariance matrices (divisor ``n``).
    vcov : pandas.DataFrame
        Covariance matrix of the free parameters.
    r2 : pandas.Series
        Share of each endogenous variable's variance the model explains.
    n_obs : int

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
            "Path analysis (maximum likelihood)",
            "=" * 66,
            f"  Observations : {self.n_obs}"
            + (f"   ({self.n_dropped} dropped, missing)" if self.n_dropped else ""),
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
            shown["lhs"] + " " + shown["op"] + " " + shown["rhs"],
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
    regressions: List[Tuple[str, str, Optional[str], Optional[float]]] = []
    covariances: List[Tuple[str, str, Optional[str], Optional[float]]] = []
    defined: List[Tuple[str, str]] = []
    statements: List[str] = []
    for raw in model.replace(";", "\n").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            statements.append(line)
    if not statements:
        raise MethodIncompatibility("path_analysis: the model is empty.")

    def modifier(term: str) -> Tuple[str, Optional[str], Optional[float]]:
        term = term.strip()
        if "*" not in term:
            return term, None, None
        mod, var = (t.strip() for t in term.split("*", 1))
        try:
            return var, None, float(mod)
        except ValueError:
            if not re.fullmatch(_NAME, mod):
                raise MethodIncompatibility(
                    f"path_analysis: cannot read the modifier {mod!r} in "
                    f"{term!r}; use a label or a number."
                ) from None
            return var, mod, None

    for line in statements:
        if "=~" in line:
            raise MethodIncompatibility(
                "path_analysis: latent variables (=~) are not implemented; "
                "the model must be among observed variables.",
                recovery_hint="Replace the latent variable by a scale score, "
                "or fit the measurement model in lavaan / semopy.",
            )
        if ":=" in line:
            name, expr = (t.strip() for t in line.split(":=", 1))
            if not re.fullmatch(_NAME, name):
                raise MethodIncompatibility(
                    f"path_analysis: {name!r} is not a valid name for a "
                    "defined parameter."
                )
            defined.append((name, expr))
        elif "~~" in line:
            lhs, rhs = (t.strip() for t in line.split("~~", 1))
            for term in rhs.split("+"):
                var, label, fixed = modifier(term)
                covariances.append((lhs, var, label, fixed))
        elif "~" in line:
            lhs, rhs = (t.strip() for t in line.split("~", 1))
            for term in rhs.split("+"):
                var, label, fixed = modifier(term)
                if var in ("1", "0"):
                    raise MethodIncompatibility(
                        "path_analysis: intercepts (y ~ 1) are not part of "
                        "the model; it is fitted to the covariance matrix.",
                        recovery_hint="Drop the '1'; means do not affect the "
                        "path coefficients.",
                    )
                regressions.append((lhs, var.replace(" ", ""), label, fixed))
        else:
            raise MethodIncompatibility(
                f"path_analysis: cannot read {line!r}; expected '~', '~~' or ':='."
            )
    if not regressions:
        raise MethodIncompatibility(
            "path_analysis: the model has no regression (y ~ x)."
        )
    return {"regressions": regressions, "covariances": covariances, "defined": defined}


# ----------------------------------------------------------------- estimation
def _vech_index(p: int) -> Tuple[np.ndarray, np.ndarray]:
    return np.tril_indices(p)


def path_analysis(
    model: str,
    data: pd.DataFrame,
    *,
    se: str = "standard",
    alpha: float = 0.05,
) -> PathAnalysisResult:
    """Fit a path model among observed variables by maximum likelihood.

    Parameters
    ----------
    model : str
        The model in lavaan syntax, one statement per line (or separated by
        ``;``):

        * ``y ~ x1 + b*x2`` regresses ``y`` on ``x1`` and ``x2``; ``b`` labels
          the second coefficient. Two paths with the same label are
          constrained to be equal. ``0.5*x`` fixes a coefficient.
        * ``y1 ~~ y2`` lets the disturbances of two endogenous variables
          covary. They are uncorrelated otherwise.
        * ``name := expression`` defines a function of labelled parameters,
          with a delta-method standard error.
        * ``x1:x2`` on the right-hand side is the product of two columns.
        * ``#`` starts a comment.
    data : pandas.DataFrame
        One numeric column per variable. Rows with a missing value in any
        variable of the model are dropped.
    se : {'standard', 'robust'}, default 'standard'
        ``'standard'`` uses the expected information matrix and is correct
        under multivariate normality. ``'robust'`` is the Satorra-Bentler
        sandwich built from the fourth moments of the data, with a scaled
        test statistic, for when the disturbances are heavy-tailed or
        heteroskedastic (lavaan's ``estimator = "MLM"``).
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    PathAnalysisResult

    Notes
    -----
    Variables that only appear on a right-hand side are exogenous. Their
    variances and covariances are held at the sample values (lavaan's
    ``fixed.x = TRUE``), so the model is conditional on them. Covariances
    use the divisor ``n``, as maximum likelihood does.

    With the default settings the estimates, standard errors, standardised
    solution, test statistic and fit indices reproduce
    ``lavaan::sem(model, data)``; with ``se='robust'`` they reproduce
    ``estimator = "MLM"``.

    A path coefficient is a causal effect only if the diagram is right: no
    omitted common causes of the variables it links, and arrows in the
    stated direction. The chi-square test can reject the diagram but cannot
    confirm it, and a saturated model (``df = 0``) is not tested at all. An
    indirect effect ``a*b`` additionally needs no unmeasured confounder of
    the mediator and the outcome; :func:`statspai.mediate_sensitivity`
    probes that.

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
    spec = _parse(model)

    # ---- variables ------------------------------------------------------
    endo: List[str] = []
    for lhs, _, _, _ in spec["regressions"]:
        if lhs not in endo:
            endo.append(lhs)
    rhs_vars: List[str] = []
    for _, var, _, _ in spec["regressions"]:
        if var not in rhs_vars:
            rhs_vars.append(var)
    exo = [v for v in rhs_vars if v not in endo]
    names = endo + exo
    q, k = len(endo), len(exo)
    p = q + k

    frame = pd.DataFrame(index=data.index)
    for v in names:
        if v in data.columns:
            col = data[v]
        elif ":" in v and all(part in data.columns for part in v.split(":")):
            col = data[v.split(":")[0]].astype(float)
            for part in v.split(":")[1:]:
                col = col * data[part].astype(float)
        else:
            raise MethodIncompatibility(
                f"path_analysis: variable {v!r} is not a column of data.",
                diagnostics={"columns": [str(c) for c in data.columns][:30]},
            )
        if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
            raise MethodIncompatibility(
                f"path_analysis: column {v!r} is not numeric.",
                recovery_hint="Encode it as numbers (a two-level factor as 0/1).",
            )
        frame[v] = col.astype(float)
    n_all = len(frame)
    frame = frame.dropna()
    n = len(frame)
    if n <= p + 1:
        raise DataInsufficient(f"path_analysis: {n} complete rows for {p} variables.")
    Z = frame.to_numpy()
    Zc = Z - Z.mean(axis=0)
    S = Zc.T @ Zc / n
    if np.linalg.matrix_rank(S) < p:
        raise DataInsufficient(
            "path_analysis: the variables are linearly dependent; the sample "
            "covariance matrix is singular."
        )
    idx = {v: i for i, v in enumerate(names)}

    # ---- parameter table ------------------------------------------------
    # each entry: kind ('b' | 'psi'), i, j, label, fixed value, free index
    entries: List[Dict[str, Any]] = []
    label_to_free: Dict[str, int] = {}
    n_free = 0

    def add(
        kind: str, i: int, j: int, label: Optional[str], fixed: Optional[float]
    ) -> None:
        nonlocal n_free
        entry: Dict[str, Any] = {
            "kind": kind,
            "i": i,
            "j": j,
            "label": label or "",
            "fixed": fixed,
        }
        if fixed is not None:
            entry["free"] = None
        elif label and label in label_to_free:
            entry["free"] = label_to_free[label]
        else:
            entry["free"] = n_free
            if label:
                label_to_free[label] = n_free
            n_free += 1
        entries.append(entry)

    seen = set()
    for lhs, var, label, fixed in spec["regressions"]:
        if lhs == var:
            raise MethodIncompatibility(
                f"path_analysis: {lhs!r} is regressed on itself."
            )
        if (lhs, var) in seen:
            raise MethodIncompatibility(
                f"path_analysis: the path {lhs} ~ {var} appears twice."
            )
        seen.add((lhs, var))
        add("b", idx[lhs], idx[var], label, fixed)
    cov_pairs = set()
    explicit_var: Dict[int, Tuple[Optional[str], Optional[float]]] = {}
    pending_cov: List[Tuple[int, int, Optional[str], Optional[float]]] = []
    for a, b, label, fixed in spec["covariances"]:
        for v in (a, b):
            if v not in idx:
                raise MethodIncompatibility(
                    f"path_analysis: {v!r} in '{a} ~~ {b}' is not a variable "
                    "of the model."
                )
        ia, ib = idx[a], idx[b]
        if ia >= q and ib >= q:
            continue  # exogenous (co)variances are held at the sample values
        if ia >= q or ib >= q:
            raise MethodIncompatibility(
                f"path_analysis: '{a} ~~ {b}' relates a disturbance to an "
                "exogenous variable, which would make that variable "
                "endogenous.",
                recovery_hint="Add an equation for it, or drop the covariance.",
            )
        if ia == ib:
            explicit_var[ia] = (label, fixed)
            continue
        pair = (max(ia, ib), min(ia, ib))
        if pair in cov_pairs:
            continue
        cov_pairs.add(pair)
        pending_cov.append((ia, ib, label, fixed))  # as written
    for i, j, label, fixed in pending_cov:
        add("psi", i, j, label, fixed)
    for i in range(q):
        label, fixed = explicit_var.get(i, (None, None))
        add("psi", i, i, label, fixed)

    df_model = p * (p + 1) // 2 - k * (k + 1) // 2 - n_free
    if df_model < 0:
        raise MethodIncompatibility(
            f"path_analysis: the model has {n_free} free parameters and only "
            f"{p * (p + 1) // 2 - k * (k + 1) // 2} moments; it is not "
            "identified.",
            recovery_hint="Remove a path or a residual covariance.",
        )

    Sxx = S[q:, q:]

    def matrices(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        B = np.zeros((p, p))
        Psi = np.zeros((p, p))
        Psi[q:, q:] = Sxx
        for e in entries:
            val = e["fixed"] if e["free"] is None else theta[e["free"]]
            if e["kind"] == "b":
                B[e["i"], e["j"]] = val
            else:
                Psi[e["i"], e["j"]] = val
                Psi[e["j"], e["i"]] = val
        return B, Psi

    def implied(theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        B, Psi = matrices(theta)
        A = np.linalg.inv(np.eye(p) - B)
        return A @ Psi @ A.T, A, Psi

    def jacobian(theta: np.ndarray) -> np.ndarray:
        """d Sigma / d theta, one p x p slice per free parameter."""
        Sigma, A, _ = implied(theta)
        out = np.zeros((n_free, p, p))
        for e in entries:
            if e["free"] is None:
                continue
            i, j = e["i"], e["j"]
            if e["kind"] == "b":
                # d Sigma = A E_ij Sigma + (A E_ij Sigma)'
                half = np.outer(A[:, i], Sigma[j, :])
                out[e["free"]] += half + half.T
            else:
                d = np.outer(A[:, i], A[:, j])
                out[e["free"]] += d if i == j else d + d.T
        return out

    sign, logdet_S = np.linalg.slogdet(S)

    def objective(theta: np.ndarray) -> Tuple[float, np.ndarray]:
        try:
            Sigma, _, _ = implied(theta)
            sgn, logdet = np.linalg.slogdet(Sigma)
            if sgn <= 0:
                return 1e10, np.zeros(n_free)
            Sinv = np.linalg.inv(Sigma)
        except np.linalg.LinAlgError:
            return 1e10, np.zeros(n_free)
        F = logdet + np.trace(S @ Sinv) - logdet_S - p
        G = Sinv @ (Sigma - S) @ Sinv
        grad = np.einsum("ij,kij->k", G, jacobian(theta))
        return float(F), grad

    theta0 = np.zeros(n_free)
    for e in entries:
        if e["free"] is not None and e["kind"] == "psi" and e["i"] == e["j"]:
            theta0[e["free"]] = S[e["i"], e["i"]]
    if n_free:
        res = optimize.minimize(
            objective, theta0, jac=True, method="BFGS",
            options={"gtol": 1e-10, "maxiter": 2000},
        )  # fmt: skip
        theta = res.x
        # Newton steps on the expected information tighten the solution to
        # machine precision, which BFGS alone does not reach.
        for _ in range(25):
            F, grad = objective(theta)
            Sigma, _, _ = implied(theta)
            Sinv = np.linalg.inv(Sigma)
            J = jacobian(theta)
            H = np.einsum("aij,jk,bkl,li->ab", J, Sinv, J, Sinv)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            cand = theta - step
            F_new, _ = objective(cand)
            if not np.isfinite(F_new) or F_new > F + 1e-12:
                break
            theta = cand
            if np.max(np.abs(step)) < 1e-12:
                break
        F_min, grad = objective(theta)
        if np.max(np.abs(grad)) > 1e-5:
            raise DataInsufficient(
                "path_analysis: the likelihood did not converge "
                f"(largest gradient {np.max(np.abs(grad)):.2e}). The model is "
                "probably not identified.",
                recovery_hint="Check for a feedback loop or a residual "
                "covariance that the data cannot separate from a path.",
            )
    else:  # every parameter fixed
        theta = theta0
        F_min, _ = objective(theta)
    F_min = max(F_min, 0.0)
    Sigma, A, Psi = implied(theta)
    Sinv = np.linalg.inv(Sigma)

    # ---- covariance of the estimates -------------------------------------
    J = jacobian(theta) if n_free else np.zeros((0, p, p))
    info = 0.5 * np.einsum("aij,jk,bkl,li->ab", J, Sinv, J, Sinv)  # per obs.
    rows_i, cols_i = _vech_index(p)
    fit: Dict[str, Any] = {}
    if n_free:
        try:
            info_inv = np.linalg.inv(info)
        except np.linalg.LinAlgError as exc:
            raise DataInsufficient(
                "path_analysis: the information matrix is singular; the "
                "model is not identified."
            ) from exc
    else:
        info_inv = np.zeros((0, 0))
    need_gamma = se == "robust"
    if need_gamma:
        # Fourth-moment covariance of the distinct elements of S, with the
        # exogenous variables conditioned on: the residuals of y on x carry
        # the sampling variation, the x block carries none.
        D = np.einsum("ni,nj->nij", Zc, Zc)[:, rows_i, cols_i]
        if k:
            beta_yx = np.linalg.solve(Sxx, S[q:, :q])  # k x q
            R = Zc.copy()
            R[:, :q] = Zc[:, :q] - Zc[:, q:] @ beta_yx
            # moments of (residual y, x); map back to the moments of (y, x)
            T = np.eye(p)
            T[:q, q:] = beta_yx.T
            Dr = np.einsum("ni,nj->nij", R, R)
            Dr[:, q:, q:] = Sxx  # no sampling variation in the x block
            Dfull = np.einsum("ai,nij,bj->nab", T, Dr, T)
            D = Dfull[:, rows_i, cols_i]
        Dc = D - D.mean(axis=0)
        Gamma = Dc.T @ Dc / n
        # weight on the distinct elements: off-diagonals count twice
        mult = np.where(rows_i == cols_i, 1.0, 2.0)
        Jv = J[:, rows_i, cols_i]  # n_free x p*
        Wv = np.zeros((len(rows_i), len(rows_i)))
        for a_, (i1, j1) in enumerate(zip(rows_i, cols_i)):
            for b_, (i2, j2) in enumerate(zip(rows_i, cols_i)):
                Wv[a_, b_] = (
                    0.5
                    * mult[a_]
                    * mult[b_]
                    * 0.5
                    * (Sinv[i1, i2] * Sinv[j1, j2] + Sinv[i1, j2] * Sinv[j1, i2])
                )
        if n_free:
            bread = info_inv
            meat = Jv @ Wv @ Gamma @ Wv @ Jv.T
            vcov = bread @ meat @ bread / n
            U = Wv - Wv @ Jv.T @ info_inv @ Jv @ Wv
        else:
            vcov = np.zeros((0, 0))
            U = Wv
    else:
        vcov = info_inv / n

    # ---- defined parameters and the table --------------------------------
    labels_free = dict(label_to_free)

    def label_values(th: np.ndarray) -> Dict[str, float]:
        out = {lab: float(th[i]) for lab, i in labels_free.items()}
        for e in entries:
            if e["label"] and e["free"] is None:
                out[e["label"]] = float(e["fixed"])
        return out

    sd = np.sqrt(np.diag(Sigma))
    z_crit = float(stats.norm.isf(alpha / 2.0))
    records: List[Dict[str, Any]] = []
    std_by_label: Dict[str, float] = {}

    def record(
        lhs: str,
        op: str,
        rhs: str,
        label: str,
        est: float,
        se_val: float,
        std_all: float,
    ) -> None:
        if se_val is None or not np.isfinite(se_val) or se_val <= 0:
            zval = pval = lo = hi = float("nan")
            se_out = 0.0 if se_val == 0 else float("nan")
            if se_val == 0:
                lo = hi = est
        else:
            se_out = float(se_val)
            zval = est / se_out
            pval = float(2 * stats.norm.sf(abs(zval)))
            lo, hi = est - z_crit * se_out, est + z_crit * se_out
        records.append(
            {
                "lhs": lhs, "op": op, "rhs": rhs, "label": label,
                "est": float(est), "se": se_out, "z": zval, "pvalue": pval,
                "ci_lower": lo, "ci_upper": hi, "std_all": float(std_all),
            }
        )  # fmt: skip

    def se_of(e: Dict[str, Any]) -> float:
        if e["free"] is None:
            return 0.0
        return float(math.sqrt(max(vcov[e["free"], e["free"]], 0.0)))

    for e in entries:
        est = e["fixed"] if e["free"] is None else theta[e["free"]]
        i, j = e["i"], e["j"]
        if e["kind"] == "b":
            std = est * sd[j] / sd[i]
            record(names[i], "~", names[j], e["label"], est, se_of(e), std)
        else:
            if i == j:
                std = est / (sd[i] * sd[j])  # share of variance unexplained
            else:  # correlation of the two disturbances
                std = est / math.sqrt(Psi[i, i] * Psi[j, j])
            record(names[i], "~~", names[j], e["label"], est, se_of(e), std)
        if e["label"]:
            # an equality-constrained label has one standardised value per
            # path; a definition uses the first, as lavaan does
            std_by_label.setdefault(e["label"], float(std))
    for a_ in range(k):
        for b_ in range(a_, k):
            i, j = q + a_, q + b_
            record(
                names[i], "~~", names[j], "", S[i, j], 0.0,
                S[i, j] / math.sqrt(S[i, i] * S[j, j]),
            )  # fmt: skip
    for name, expr in spec["defined"]:
        est = _safe_eval(expr, label_values(theta))
        grad = np.zeros(n_free)
        for lab, fi in labels_free.items():
            h = 1e-6 * max(1.0, abs(theta[fi]))
            up, dn = theta.copy(), theta.copy()
            up[fi] += h
            dn[fi] -= h
            grad[fi] = (
                _safe_eval(expr, label_values(up)) - _safe_eval(expr, label_values(dn))
            ) / (2 * h)
        var = float(grad @ vcov @ grad) if n_free else 0.0
        std = _safe_eval(expr, {**label_values(theta), **std_by_label})
        record(
            name, ":=", expr.replace(" ", ""), name, est, math.sqrt(max(var, 0)), std
        )
    params = pd.DataFrame.from_records(records)

    # ---- fit measures ---------------------------------------------------
    chisq = n * F_min
    logdet_Sigma = float(np.linalg.slogdet(Sigma)[1])
    logl_joint = -0.5 * n * (p * math.log(2 * math.pi) + logdet_Sigma
                             + float(np.trace(S @ Sinv)))  # fmt: skip
    logl_x = (
        -0.5 * n * (k * math.log(2 * math.pi) + float(np.linalg.slogdet(Sxx)[1]) + k)
        if k
        else 0.0
    )
    logl = logl_joint - logl_x
    base_df = p * (p + 1) // 2 - k * (k + 1) // 2 - q
    base_logdet = float(np.sum(np.log(np.diag(S)[:q]))) + (
        float(np.linalg.slogdet(Sxx)[1]) if k else 0.0
    )
    base_chisq = n * (base_logdet - logdet_S)
    d_m = max(chisq - df_model, 0.0)
    d_b = max(base_chisq - base_df, 0.0)
    cfi = 1.0 - d_m / max(d_m, d_b) if max(d_m, d_b) > 0 else 1.0
    if df_model > 0 and base_df > 0 and base_chisq / base_df != 1.0:
        tli = (base_chisq / base_df - chisq / df_model) / (base_chisq / base_df - 1.0)
    else:
        tli = 1.0
    rmsea = (
        math.sqrt(max((chisq - df_model) / (df_model * n), 0.0)) if df_model else 0.0
    )

    def ncp_for(target: float) -> float:
        """Noncentrality at which chisq sits at the given upper tail."""
        if df_model == 0 or stats.chi2.sf(chisq, df_model) >= target:
            return 0.0

        def gap(lam: float) -> float:
            return float(stats.ncx2.sf(chisq, df_model, lam)) - target

        hi = max(chisq, 1.0)
        while gap(hi) < 0:
            hi *= 2.0
        return float(optimize.brentq(gap, 1e-12, hi))

    if df_model:
        rm_lo = math.sqrt(ncp_for(0.05) / (df_model * n))
        rm_hi = math.sqrt(ncp_for(0.95) / (df_model * n))
    else:
        rm_lo = rm_hi = 0.0
    resid = (S - Sigma) / np.sqrt(np.outer(np.diag(S), np.diag(S)))
    srmr = math.sqrt(float(np.mean(resid[rows_i, cols_i] ** 2)))
    fit.update(
        {
            "chisq": float(chisq),
            "df": int(df_model),
            "pvalue": (
                float(stats.chi2.sf(chisq, df_model)) if df_model else float("nan")
            ),
            "baseline_chisq": float(base_chisq),
            "baseline_df": int(base_df),
            "cfi": float(cfi),
            "tli": float(tli),
            "rmsea": float(rmsea),
            "rmsea_ci_lower": float(rm_lo),
            "rmsea_ci_upper": float(rm_hi),
            "srmr": float(srmr),
            "logl": float(logl),
            "aic": float(-2 * logl + 2 * n_free),
            "bic": float(-2 * logl + math.log(n) * n_free),
            "npar": int(n_free),
        }
    )
    if need_gamma and df_model > 0:
        scale = float(np.trace(U @ Gamma)) / df_model
        fit["scaling_factor"] = scale
        fit["chisq_scaled"] = float(chisq / scale)
        fit["pvalue_scaled"] = float(stats.chi2.sf(chisq / scale, df_model))

    free_names: List[str] = [""] * n_free
    for e in entries:
        if e["free"] is not None and not free_names[e["free"]]:
            op = "~" if e["kind"] == "b" else "~~"
            free_names[e["free"]] = e["label"] or f"{names[e['i']]}{op}{names[e['j']]}"
    r2 = pd.Series(
        {names[i]: 1.0 - Psi[i, i] / Sigma[i, i] for i in range(q)}, name="r2"
    )
    return PathAnalysisResult(
        params=params,
        fit=fit,
        implied_cov=pd.DataFrame(Sigma, index=names, columns=names),
        sample_cov=pd.DataFrame(S, index=names, columns=names),
        vcov=pd.DataFrame(vcov, index=free_names, columns=free_names),
        r2=r2,
        n_obs=int(n),
        model=model,
        se=se,
        n_dropped=int(n_all - n),
    )
