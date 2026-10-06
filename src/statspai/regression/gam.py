"""
Generalized additive models with penalised B-splines.

``sp.gam("y ~ s(x1) + s(x2) + z", df)`` fits

    g(E[y]) = z'b + f1(x1) + f2(x2)

where each ``f`` is a cubic B-spline on evenly spaced knots whose wiggliness
is held down by a second-difference penalty on neighbouring coefficients (a
P-spline, Eilers and Marx [@eilers1996flexible]). One smoothing parameter
per term decides how far each curve may depart from a straight line; they
are chosen by generalized cross-validation, or by the unbiased risk
estimator when the scale is known (binomial, Poisson), as in Wood
[@wood2017generalized].

The basis, the penalty and the identifiability constraint are those of R
``mgcv::gam`` with ``s(x, bs = "ps")``, so the two can be compared number
for number at given smoothing parameters. mgcv's default basis is a thin
plate spline, which gives a slightly different curve from the same data.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.interpolate import BSpline

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility
from .glm import FAMILIES, _get_link

_KNOWN_SCALE = ("binomial", "poisson")
_SMOOTH = re.compile(r"^s\((.*)\)$")
_DEGREE = 3  # cubic B-splines
_DIFF = 2  # second-difference penalty


@dataclass
class _Smooth:
    """One ``s(x)`` term: its knots and the reparametrisation that removes
    the constant from its span."""

    var: str
    k: int
    knots: np.ndarray
    Q: np.ndarray  # k x (k - 1): columns span {c : colsums(X) c = 0}
    penalty: np.ndarray  # (k - 1) x (k - 1), in the constrained basis
    root: np.ndarray  # (k - 2) x (k - 1) with root'root = penalty
    cols: slice = slice(0, 0)
    by: Optional[str] = None  # the curve multiplies this column

    @property
    def label(self) -> str:
        return f"s({self.var})" if self.by is None else f"s({self.var}):{self.by}"

    def raw(self, x: np.ndarray) -> np.ndarray:
        basis = BSpline(self.knots, np.eye(self.k), _DEGREE, extrapolate=True)
        return np.asarray(basis(np.asarray(x, dtype=float)), dtype=float)

    def design(self, x: np.ndarray, by: Optional[np.ndarray] = None) -> np.ndarray:
        Z = np.asarray(self.raw(x) @ self.Q, dtype=float)
        if by is not None:
            Z = Z * np.asarray(by, dtype=float)[:, None]
        return Z


def _make_smooth(var: str, x: np.ndarray, k: int, by: Optional[str] = None) -> _Smooth:
    if k < _DEGREE + 2:
        raise MethodIncompatibility(
            f"gam: s({var}, k={k}) needs k >= {_DEGREE + 2} basis functions.",
            diagnostics={"k": k},
        )
    distinct = np.unique(x).size
    if distinct < k:
        raise DataInsufficient(
            f"gam: {var} has {distinct} distinct values, fewer than the "
            f"k = {k} basis functions of its smooth.",
            recovery_hint=f"Use s({var}, k=...) with a smaller k, or enter "
            f"{var} as a factor.",
            diagnostics={"distinct": int(distinct), "k": k},
        )
    lo, hi = float(np.min(x)), float(np.max(x))
    span = hi - lo
    lo, hi = lo - 0.001 * span, hi + 0.001 * span
    inner = k - (_DEGREE - 1)  # knots across the (padded) range of the data
    dx = (hi - lo) / (inner - 1)
    knots = lo + dx * np.arange(-_DEGREE, inner + _DEGREE)
    D = np.diff(np.eye(k), n=_DIFF, axis=0)
    S = D.T @ D
    raw = np.asarray(
        BSpline(knots, np.eye(k), _DEGREE, extrapolate=True)(x), dtype=float
    )
    if by is not None:
        # a curve that multiplies another column carries its own level
        # (the coefficient on that column), so nothing is removed
        return _Smooth(var=var, k=k, knots=knots, Q=np.eye(k), penalty=S, root=D, by=by)
    # sum-to-zero over the sample: f is identified apart from the intercept
    c = raw.sum(axis=0)[:, None]
    Qfull, _ = np.linalg.qr(c, mode="complete")
    Q = Qfull[:, 1:]
    return _Smooth(var=var, k=k, knots=knots, Q=Q, penalty=Q.T @ S @ Q, root=D @ Q)


def _split_terms(rhs: str) -> List[str]:
    terms, depth, start = [], 0, 0
    for i, ch in enumerate(rhs):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "+" and depth == 0:
            terms.append(rhs[start:i].strip())
            start = i + 1
    terms.append(rhs[start:].strip())
    return [t for t in terms if t]


def _parse_smooth(term: str, default_k: int) -> Tuple[str, int, Optional[str]]:
    inside = _SMOOTH.match(term)
    assert inside is not None
    parts = [p.strip() for p in inside.group(1).split(",")]
    var = parts[0]
    k = default_k
    by: Optional[str] = None
    for opt in parts[1:]:
        name, _, value = opt.partition("=")
        name, value = name.strip(), value.strip().strip("\"'")
        if name == "k":
            try:
                k = int(value)
            except ValueError as exc:
                raise MethodIncompatibility(
                    f"gam: k={value!r} in {term} is not an integer."
                ) from exc
        elif name == "by":
            if not re.fullmatch(r"[A-Za-z_]\w*", value):
                raise MethodIncompatibility(
                    f"gam: by={value!r} in {term} must name one column."
                )
            by = value
        elif name == "bs":
            if value != "ps":
                raise MethodIncompatibility(
                    f"gam: {term} asks for the {value!r} basis; only "
                    "P-splines (bs='ps') are implemented.",
                    recovery_hint="Drop bs=, or use bs='ps'.",
                )
        else:
            raise MethodIncompatibility(
                f"gam: option {opt!r} in {term} is not understood; a smooth "
                "takes a variable and, optionally, k= and by=.",
            )
    if not re.fullmatch(r"[A-Za-z_]\w*", var):
        raise MethodIncompatibility(
            f"gam: {term} must name one column; smooths of several variables "
            "or of expressions are not implemented.",
            recovery_hint="Create the transformed column first.",
        )
    return var, k, by


@dataclass
class GAMResult(ResultProtocolMixin):
    """Result of :func:`gam`.

    Attributes
    ----------
    params, std_errors : pd.Series
        Coefficients of the parametric terms and their Bayesian standard
        errors (the square roots of the diagonal of ``scale (X'WX +
        S)^-1``).
    smooth_terms : pd.DataFrame
        One row per ``s()`` term: ``edf`` (effective degrees of freedom; 1
        is a straight line, ``k - 1`` an unpenalised spline), ``lambda``
        (its smoothing parameter) and ``k``.
    edf : float
        Total effective degrees of freedom, parametric terms included.
    scale : float
        Estimated residual variance (Gaussian, gamma) or 1.
    gcv : float
        The value of the criterion named in ``criterion``: minus the
        restricted log likelihood up to a constant (``"reml"``), GCV, or
        UBRE when the scale is known.
    deviance, n_obs, family, link, formula
        As named.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = np.linspace(0, 1, 300)
    >>> df = pd.DataFrame({"x": x, "y": np.sin(6 * x) + rng.normal(0, 0.3, 300)})
    >>> fit = sp.gam("y ~ s(x)", df)
    >>> isinstance(fit, sp.GAMResult)
    True
    >>> list(fit.smooth_terms.columns)
    ['term', 'edf', 'lambda', 'k']
    >>> fit.partial("s(x)").columns.tolist()
    ['x', 'fit', 'se', 'lower', 'upper']
    """

    _citation_keys = ("eilers1996flexible", "wood2017generalized")
    params: pd.Series
    std_errors: pd.Series
    smooth_terms: pd.DataFrame
    edf: float
    scale: float
    gcv: float
    deviance: float
    n_obs: int
    family: str
    link: str
    formula: str
    criterion: str = "gcv"
    vce: str = "nonrobust"
    converged: bool = True
    alpha: float = 0.05
    fitted_values: Any = field(default=None, repr=False)
    residuals: Any = field(default=None, repr=False)
    _coef: Any = field(default=None, repr=False)
    _vcov: Any = field(default=None, repr=False)
    _smooths: Any = field(default=None, repr=False)
    _design_info: Any = field(default=None, repr=False)
    _par_names: Any = field(default=None, repr=False)
    _link_name: Any = field(default=None, repr=False)

    # -- design on new data -------------------------------------------
    def _design(self, data: pd.DataFrame) -> np.ndarray:
        from patsy import build_design_matrices

        if self._design_info is not None:
            P = np.asarray(
                build_design_matrices([self._design_info], data)[0], dtype=float
            )
        else:
            # plain numeric columns: the design is the columns themselves
            cols = [c for c in self._par_names if c != "Intercept"]
            absent = [c for c in cols if c not in data.columns]
            if absent:
                raise MethodIncompatibility(
                    f"gam: the new data lack column(s) {absent}.",
                    diagnostics={"missing": absent},
                )
            P = np.column_stack(
                [np.ones(len(data))] + [data[c].to_numpy(dtype=float) for c in cols]
            )
            P = P[:, [0] + [1 + cols.index(c) for c in cols]]
            order = [self._par_names.index(c) for c in ["Intercept"] + cols]
            P = P[:, np.argsort(order)]
        if P.shape[0] != len(data):
            raise MethodIncompatibility(
                "gam: the new data have missing values in the parametric terms.",
                recovery_hint="Drop or fill them before predicting.",
            )
        blocks = [P] + [
            sm.design(
                data[sm.var].to_numpy(dtype=float),
                None if sm.by is None else data[sm.by].to_numpy(dtype=float),
            )
            for sm in self._smooths
        ]
        return np.column_stack(blocks)

    def predict(
        self,
        data: pd.DataFrame,
        what: str = "mean",
        alpha: Optional[float] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Predictions on new data.

        ``what='mean'`` (the fitted mean) or ``'link'`` (the additive
        predictor) return an array. ``what='confidence'`` returns ``yhat``,
        ``se`` and an interval for the mean, formed on the link scale and
        mapped back. Outside the range of the data a smooth continues the
        cubic of its last segment; treat such predictions as extrapolation.
        """
        key = str(what).lower()
        if key not in ("mean", "link", "confidence"):
            raise MethodIncompatibility(
                f"gam: what={what!r} is not 'mean', 'link' or 'confidence'."
            )
        lnk = _get_link(self._link_name, FAMILIES[self.family]())
        M = self._design(data)
        eta = M @ self._coef
        if key == "link":
            return np.asarray(eta, dtype=float)
        mu = np.asarray(lnk.inverse(eta), dtype=float)
        if key == "mean":
            return mu
        level = self.alpha if alpha is None else float(alpha)
        se_eta = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", M, self._vcov, M), 0.0))
        z = float(stats.norm.ppf(1.0 - level / 2.0))
        return pd.DataFrame(
            {
                "yhat": mu,
                "se": se_eta / np.abs(lnk.deriv(mu)),
                "lower": lnk.inverse(eta - z * se_eta),
                "upper": lnk.inverse(eta + z * se_eta),
            },
            index=data.index,
        )

    def partial(
        self,
        term: str,
        grid: Optional[Sequence[float]] = None,
        alpha: Optional[float] = None,
    ) -> pd.DataFrame:
        """The estimated function of one smooth term, with a pointwise band.

        ``term`` is a label from ``smooth_terms`` (``"s(x)"``,
        ``"s(x):d"``) or, when it is not ambiguous, the variable name. A
        plain smooth is centred (it sums to zero over the sample). A
        ``by=`` smooth is the function that multiplies the ``by`` column,
        level included: with a 0/1 treatment ``d`` in ``s(x) + s(x,
        by=d)``, it is the effect of ``d`` as a function of ``x``. Both
        are on the scale of the link.
        """
        match = [sm for sm in self._smooths if sm.label == term]
        if not match:
            match = [sm for sm in self._smooths if sm.var == term]
            if len(match) > 1:
                raise MethodIncompatibility(
                    f"gam: {term!r} is smoothed more than once; name one of "
                    f"{[sm.label for sm in match]}.",
                )
        name = match[0].var if match else term
        if not match:
            raise MethodIncompatibility(
                f"gam: no smooth of {name!r}; the model has "
                f"{[sm.label for sm in self._smooths]}.",
            )
        sm = match[0]
        if grid is None:
            inner = sm.knots[_DEGREE : len(sm.knots) - _DEGREE]
            span = inner[-1] - inner[0]
            lo = inner[0] + 0.001 * span / 1.002
            hi = inner[-1] - 0.001 * span / 1.002
            xs = np.linspace(lo, hi, 100)
        else:
            xs = np.asarray(grid, dtype=float)
        Z = sm.design(xs)
        fit = Z @ self._coef[sm.cols]
        V = self._vcov[sm.cols, sm.cols]
        se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Z, V, Z), 0.0))
        level = self.alpha if alpha is None else float(alpha)
        z = float(stats.norm.ppf(1.0 - level / 2.0))
        return pd.DataFrame(
            {
                name: xs,
                "fit": fit,
                "se": se,
                "lower": fit - z * se,
                "upper": fit + z * se,
            }
        )

    def plot(self, term: Optional[str] = None, ax: Any = None) -> Any:
        """Plot one smooth (the first by default) with its pointwise band."""
        import matplotlib.pyplot as plt

        sm = self._smooths[0].label if term is None else term
        curve = self.partial(sm)
        x = curve.columns[0]
        fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 4))
        ax.fill_between(curve[x], curve["lower"], curve["upper"], alpha=0.25)
        ax.plot(curve[x], curve["fit"])
        ax.set_xlabel(x)
        ax.set_ylabel(f"s({x})")
        return fig, ax

    def summary(self) -> str:
        z = self.params / self.std_errors
        lines = [
            "=" * 64,
            f"Generalized additive model ({self.family}, {self.link} link)",
            "=" * 64,
            f"  Formula      : {self.formula}",
            f"  Observations : {self.n_obs}",
            f"  Total edf    : {self.edf:.3f}",
            f"  Scale        : {self.scale:.6g}",
            f"  {self.criterion.upper():<13s}: {self.gcv:.6g}",
            f"  Std. errors  : {self.vce}",
            "",
            "  Parametric terms",
            f"    {'':<22s}{'coef':>12s}{'se':>12s}{'z':>9s}",
        ]
        for name in self.params.index:
            lines.append(
                f"    {str(name):<22s}{self.params[name]:>12.5g}"
                f"{self.std_errors[name]:>12.5g}{z[name]:>9.2f}"
            )
        lines += [
            "",
            "  Smooth terms",
            f"    {'':<22s}{'edf':>12s}{'lambda':>12s}{'k':>9s}",
        ]
        for _, row in self.smooth_terms.iterrows():
            lines.append(
                f"    {row['term']:<22s}{row['edf']:>12.3f}{row['lambda']:>12.4g}"
                f"{int(row['k']):>9d}"
            )
        lines.append("=" * 64)
        text = "\n".join(lines)
        print(text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": {str(k): float(v) for k, v in self.params.items()},
            "std_errors": {str(k): float(v) for k, v in self.std_errors.items()},
            "smooth_terms": self.smooth_terms.to_dict(orient="records"),
            "edf": float(self.edf),
            "scale": float(self.scale),
            self.criterion: float(self.gcv),
            "deviance": float(self.deviance),
            "n_obs": int(self.n_obs),
            "family": self.family,
            "link": self.link,
            "formula": self.formula,
            "converged": bool(self.converged),
        }


def gam(
    formula: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    link: Optional[str] = None,
    k: int = 10,
    lambda_: Union[None, float, Sequence[float]] = None,
    method: str = "reml",
    gamma: float = 1.0,
    vce: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 100,
    tol: float = 1e-9,
    alpha: float = 0.05,
) -> GAMResult:
    """
    Generalized additive model with penalised B-spline smooths.

    Fits ``g(E[y]) = parametric terms + f1(x1) + f2(x2) + ...`` where each
    ``s(x)`` in the formula is a smooth function estimated from the data.
    Equivalent to R ``mgcv::gam(y ~ s(x1, bs = "ps") + ..., method =
    "GCV.Cp")``.

    Parameters
    ----------
    formula : str
        ``"y ~ s(x1) + s(x2, k=15) + z + C(g)"``. A term ``s(x)`` is a
        smooth of the numeric column ``x``; everything else is an ordinary
        formula term and enters linearly.
    data : pd.DataFrame
        Rows with a missing value in any variable of the model are dropped.
    family : {"gaussian", "binomial", "poisson", "gamma"}, default "gaussian"
    link : str, optional
        The family's canonical link when omitted.
    k : int, default 10
        Basis functions per smooth unless a term sets its own
        (``s(x, k=20)``). It caps how wiggly a curve can get; the penalty
        does the rest, so the choice matters little once it is large
        enough. If a term's ``edf`` comes out close to ``k - 1``, raise it.
    lambda_ : float or sequence of float, optional
        Smoothing parameters, one per smooth in the order of the formula
        (a single number is used for all). Larger is smoother; as it
        grows the curve tends to a straight line. When omitted they are
        chosen by ``method``.
    method : {"reml", "gcv"}, default "reml"
        How the smoothing parameters are chosen. ``"reml"`` maximises the
        restricted marginal likelihood of the model read as a mixed model
        (a Laplace approximation outside the Gaussian case). ``"gcv"``
        minimises generalized cross-validation, or the unbiased risk
        estimator when the scale is known (binomial, Poisson); it is R
        ``mgcv``'s default (``method = "GCV.Cp"``). The prediction-error
        criteria have shallow, sometimes several, minima and now and then
        pick a far too wiggly curve; REML penalises that more firmly and
        is the choice its own author recommends, which is why it is the
        default here.
    gamma : float, default 1.0
        Under ``method="gcv"``, the factor on the effective degrees of
        freedom in the criterion. Values above 1 (1.4 is the usual
        suggestion) ask for smoother curves. Ignored by REML.
    vce : {"nonrobust", "hc0", "robust"}, default "nonrobust"
        Covariance of the coefficients. ``"nonrobust"`` is the Bayesian
        ``scale (X'WX + S)^-1``. ``"hc0"`` is the sandwich ``B (sum u_i
        u_i') B`` with ``B = (X'WX + S)^-1`` and ``u_i`` the score of
        observation ``i``; it does not rely on the variance function
        being right. ``"robust"`` is the same times ``N / (N - 1)``, the
        package's convention for likelihood models. As the smoothing
        parameters grow these become the sandwich of the GLM with linear
        terms. (R ``vcov(gam, sandwich = TRUE)`` adds a finite-sample
        adjustment of its own and is about 1% larger at n = 455.)
    cluster : str, optional
        Column of cluster identifiers. The scores are summed within
        clusters before the sandwich is formed, with the factor ``G / (G
        - 1)``, as in ``sp.glm(cluster=)``. Use it for panels and grouped
        samples; the default covariance assumes independent rows.
    maxiter : int, default 100
        Iterations of penalised IRLS for non-Gaussian families.
    tol : float, default 1e-9
        Convergence tolerance on the relative change of the deviance.
    alpha : float, default 0.05
        One minus the level of the bands of ``predict`` and ``partial``.

    Returns
    -------
    GAMResult
        ``params`` and ``std_errors`` of the parametric terms,
        ``smooth_terms`` (edf and smoothing parameter of each curve),
        ``partial(term)`` for the estimated functions, ``predict`` and
        ``plot``.

    Notes
    -----
    Each smooth is a cubic B-spline on evenly spaced knots with a
    second-difference penalty and is constrained to sum to zero over the
    sample, so the intercept carries the level. Standard errors and bands
    are the Bayesian ones, ``scale (X'WX + S)^-1``, unless ``vce=`` or
    ``cluster=`` asks for a sandwich; either way they condition on the
    chosen smoothing parameters and are pointwise, not simultaneous.

    ``s(x, by=d)`` is a curve that multiplies the numeric column ``d``:
    the model gains ``d * f(x)``. With a 0/1 treatment, ``"y ~ s(x) +
    s(x, by=d)"`` fits one curve for the untreated and adds ``f(x)`` for
    the treated, so ``partial("s(x):d")`` is the difference between the
    two groups as a function of ``x``, with a band. Such a term is not
    centred (it contains the level shift), so ``d`` must not also enter
    linearly.

    ``lambda_`` multiplies the plain difference penalty ``D'D``. mgcv
    divides that matrix by a constant (``S.scale`` in its fitted object)
    before applying its ``sp``, so ``lambda_ = sp / S.scale`` reproduces a
    given mgcv fit.

    The additive structure is an assumption. Interactions between
    smoothed variables are not estimated, and a smooth of a confounder
    adjusts for it more flexibly than a linear term but no more credibly:
    selection on unobservables is untouched. For a treatment effect with
    flexible controls, ``sp.dml`` gives valid inference after the
    flexibility; the standard error of a parametric term here does not
    account for the search over smoothing parameters.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 500
    >>> x = rng.uniform(0, 1, n)
    >>> d = rng.integers(0, 2, n)
    >>> df = pd.DataFrame({"x": x, "d": d,
    ...                    "y": 0.5 * d + np.sin(2 * np.pi * x)
    ...                         + rng.normal(0, 0.3, n)})
    >>> fit = sp.gam("y ~ d + s(x)", df)
    >>> bool(abs(fit.params["d"] - 0.5) < 0.1)
    True
    >>> bool(3 < fit.smooth_terms["edf"].iloc[0] < 9)
    True

    References
    ----------
    [@eilers1996flexible]
    [@wood2017generalized]
    """
    fam_key = str(family).lower()
    if fam_key not in FAMILIES:
        raise MethodIncompatibility(
            f"gam: unknown family {family!r}. Choose from: "
            f"{', '.join(FAMILIES.keys())}.",
        )
    fam = FAMILIES[fam_key]()
    lnk = _get_link(link, fam)
    if "~" not in formula:
        raise MethodIncompatibility("gam: the formula needs a '~'.")
    lhs, rhs = formula.split("~", 1)
    terms = _split_terms(rhs)
    smooth_specs = [_parse_smooth(t, k) for t in terms if _SMOOTH.match(t)]
    linear = [t for t in terms if not _SMOOTH.match(t)]
    if not smooth_specs:
        raise MethodIncompatibility(
            "gam: the formula has no s() term.",
            recovery_hint="Use sp.glm or sp.regress for a model without smooths.",
        )
    labels = [(v, b) for v, _, b in smooth_specs]
    if len(set(labels)) != len(labels):
        raise MethodIncompatibility("gam: the same smooth appears twice.")
    names_s = list(
        dict.fromkeys(
            [v for v, _, _ in smooth_specs] + [b for _, _, b in smooth_specs if b]
        )
    )
    missing = [v for v in names_s if v not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"gam: smoothed column(s) {missing} not found in data.",
            diagnostics={"missing": missing},
        )
    for v in names_s:
        if not pd.api.types.is_numeric_dtype(data[v]):
            raise MethodIncompatibility(
                f"gam: {v} must be numeric to be smoothed or to multiply a " "smooth.",
                recovery_hint=f"Enter {v} as C({v}), or code a by= variable " "as 0/1.",
            )
    key_vce = str(vce).lower()
    if key_vce not in ("nonrobust", "hc0", "robust"):
        raise MethodIncompatibility(
            f"gam: vce={vce!r} is not 'nonrobust', 'hc0' or 'robust'.",
            diagnostics={"vce": vce},
        )
    if cluster is not None and cluster not in data.columns:
        raise MethodIncompatibility(
            f"gam: cluster column {cluster!r} not found in data.",
            diagnostics={"cluster": cluster},
        )
    frame = data.dropna(subset=names_s + ([cluster] if cluster else []))
    par_formula = f"{lhs.strip()} ~ {' + '.join(linear) if linear else '1'}"
    y_df, P_df = create_design_matrices(par_formula, frame)
    design_info = getattr(P_df, "design_info", None)
    par_names = [str(c) for c in P_df.columns]
    if "Intercept" not in par_names:
        raise MethodIncompatibility(
            "gam: the model needs an intercept; each smooth is centred and "
            "cannot supply the level.",
            recovery_hint="Remove '0 +' / '- 1' from the formula.",
        )
    rows = P_df.index
    y = np.asarray(y_df, dtype=float).reshape(len(rows), -1)[:, -1]
    P = np.asarray(P_df, dtype=float)
    n = len(y)

    smooths: List[_Smooth] = []
    blocks = [P]
    at = P.shape[1]
    for var, kk, by_name in smooth_specs:
        xv = frame.loc[rows, var].to_numpy(dtype=float)
        sm = _make_smooth(var, xv, kk, by=by_name)
        by_values = (
            None if by_name is None else frame.loc[rows, by_name].to_numpy(dtype=float)
        )
        Z = sm.design(xv, by_values)
        sm.cols = slice(at, at + Z.shape[1])
        at += Z.shape[1]
        smooths.append(sm)
        blocks.append(Z)
    M = np.column_stack(blocks)
    p = M.shape[1]
    if n <= p:
        raise DataInsufficient(
            f"gam: {n} complete rows for {p} coefficients.",
            recovery_hint="Lower k.",
        )
    J = len(smooths)
    known_scale = fam_key in _KNOWN_SCALE
    ones = np.ones(n)
    how = str(method).lower().replace("gcv.cp", "gcv").replace("ubre", "gcv")
    if how not in ("reml", "gcv"):
        raise MethodIncompatibility(
            f"gam: method={method!r} is not 'reml' or 'gcv'.",
            diagnostics={"method": method},
        )
    if not np.isfinite(gamma) or gamma <= 0:
        raise MethodIncompatibility(f"gam: gamma must be positive, got {gamma}.")
    # rank of each penalty and the dimension it leaves unpenalised
    ranks = [int(np.linalg.matrix_rank(sm.penalty)) for sm in smooths]
    null_dim = p - int(sum(ranks))

    def total_penalty(lams: np.ndarray) -> np.ndarray:
        S = np.zeros((p, p))
        for lam, sm in zip(lams, smooths):
            S[sm.cols, sm.cols] = lam * sm.penalty
        return S

    # Every solve goes through the QR factor of the design stacked on the
    # square root of the penalty, [sqrt(W) M; E] with E'E = S, never through
    # M'WM + S itself: the normal equations square the condition number,
    # and a smooth pushed to a straight line has a penalty of order 1e12.
    gaussian_identity = fam_key == "gaussian" and lnk.name == "identity"
    if gaussian_identity:
        Qm, Rm = np.linalg.qr(M)
        fm = Qm.T @ y

    def penalty_root(lams: np.ndarray) -> np.ndarray:
        rows = []
        for lam, sm in zip(lams, smooths):
            block = np.zeros((sm.root.shape[0], p))
            block[:, sm.cols] = np.sqrt(lam) * sm.root
            rows.append(block)
        return np.vstack(rows)

    def fit_at(lams: np.ndarray) -> Dict[str, Any]:
        S = total_penalty(lams)
        E = penalty_root(lams)
        pad = np.zeros(E.shape[0])
        if gaussian_identity:
            top, rhs = Rm, fm
            R = np.asarray(np.linalg.qr(np.vstack([top, E]), mode="r"))
            beta = np.linalg.lstsq(
                np.vstack([top, E]), np.concatenate([rhs, pad]), rcond=None
            )[0]
            mu = M @ beta
            ok = True
        else:
            mu = fam.initialize_mu(y)
            eta = lnk.link(mu)
            beta = np.zeros(p)
            dev_old = np.inf
            ok = False
            for _ in range(maxiter):
                gp = lnk.deriv(mu)
                sw = 1.0 / np.sqrt(fam.variance(mu) * gp**2)
                z = eta + (y - mu) * gp
                beta = np.linalg.lstsq(
                    np.vstack([M * sw[:, None], E]),
                    np.concatenate([sw * z, pad]),
                    rcond=None,
                )[0]
                eta = M @ beta
                mu = lnk.inverse(eta)
                dev = float(fam.deviance(y, mu, ones))
                if abs(dev - dev_old) <= tol * (abs(dev) + 0.1):
                    ok = True
                    break
                dev_old = dev
            gp = lnk.deriv(mu)
            sw = 1.0 / np.sqrt(fam.variance(mu) * gp**2)
            top = np.asarray(np.linalg.qr(M * sw[:, None], mode="r"))
            R = np.asarray(np.linalg.qr(np.vstack([top, E]), mode="r"))
        # (M'WM + S)^-1 = R^-1 R^-T; the hat matrix has trace ||top R^-1||^2
        Rinv = np.linalg.solve(R, np.eye(p))
        B = Rinv @ Rinv.T
        TR = top @ Rinv
        edf_each = np.einsum("ij,ij->j", TR @ Rinv.T, top)
        edf = float(np.sum(TR**2))
        logdet = 2.0 * float(np.sum(np.log(np.abs(np.diag(R)))))
        dev = float(fam.deviance(y, mu, ones))
        if known_scale:
            gcv = dev / n + 2.0 * gamma * edf / n - 1.0
            phi = 1.0
        else:
            gcv = n * dev / (n - gamma * edf) ** 2
            pearson = float(np.sum((y - mu) ** 2 / fam.variance(mu)))
            phi = pearson / (n - edf)
        # minus the restricted log likelihood, up to terms free of lambda:
        #   D_p / (2 phi) + log|X'WX + S| / 2 - log|S|_+ / 2
        # with D_p the deviance plus the penalty b'Sb, and phi profiled out
        # when it is unknown, which turns the first term into
        # (n - M0) log(D_p) / 2 (M0 = dimension left unpenalised).
        d_pen = dev + float(beta @ S @ beta)
        with np.errstate(divide="ignore"):
            log_pen = float(sum(r * np.log(lam) for r, lam in zip(ranks, lams)))
        if known_scale:
            reml = 0.5 * d_pen + 0.5 * logdet - 0.5 * log_pen
        else:
            reml = 0.5 * (n - null_dim) * np.log(d_pen) + 0.5 * logdet - 0.5 * log_pen
        score = reml if how == "reml" else gcv
        return {
            "beta": beta,
            "mu": mu,
            "edf_each": edf_each,
            "edf": edf,
            "dev": dev,
            "score": score,
            "gcv": gcv,
            "reml": reml,
            "phi": phi,
            "B": B,
            "ok": ok,
        }

    lams: np.ndarray
    if lambda_ is not None:
        lams = np.atleast_1d(np.asarray(lambda_, dtype=float)).ravel()
        if lams.size == 1:
            lams = np.repeat(lams, J)
        if lams.size != J or np.any(lams < 0) or not np.all(np.isfinite(lams)):
            raise MethodIncompatibility(
                f"gam: lambda_ needs {J} non-negative number(s), one per smooth.",
            )
        best = fit_at(lams)
    else:

        def objective(rho: np.ndarray) -> float:
            try:
                return float(fit_at(np.exp(rho))["score"])
            except np.linalg.LinAlgError:
                return np.inf

        # one smooth at a time over a coarse grid, then a joint polish
        rho = np.zeros(J)
        coarse = np.linspace(-8.0, 24.0, 17)
        for _ in range(2):
            for j in range(J):
                trial = []
                for r in coarse:
                    cand = rho.copy()
                    cand[j] = r
                    trial.append(objective(cand))
                rho[j] = coarse[int(np.argmin(trial))]
        res = optimize.minimize(
            objective,
            rho,
            method="L-BFGS-B",
            bounds=[(-12.0, 30.0)] * J,
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 500},
        )
        polish = optimize.minimize(
            objective,
            res.x,
            method="Nelder-Mead",
            options={"xatol": 1e-8, "fatol": 1e-15, "maxiter": 400 * J},
        )
        rho = polish.x if polish.fun < res.fun else res.x
        lams = np.exp(np.clip(rho, -12.0, 30.0))
        best = fit_at(lams)
    if not best["ok"]:
        raise ConvergenceFailure(
            f"gam: penalised IRLS did not converge in {maxiter} iterations.",
            recovery_hint="Raise maxiter, or check for separation in a "
            "binomial model.",
        )

    beta = best["beta"]
    vcov = best["phi"] * best["B"]
    vce_used = "nonrobust"
    if cluster is not None or key_vce != "nonrobust":
        mu_hat = best["mu"]
        scores = (
            M * ((y - mu_hat) / (fam.variance(mu_hat) * lnk.deriv(mu_hat)))[:, None]
        )
        if cluster is not None:
            codes = pd.factorize(frame.loc[rows, cluster])[0]
            G = int(codes.max()) + 1
            if G < 2:
                raise DataInsufficient("gam: cluster-robust SEs need two clusters.")
            summed = np.zeros((G, p))
            np.add.at(summed, codes, scores)
            meat = summed.T @ summed * (G / (G - 1.0))
            vce_used = f"cluster({cluster})"
        else:
            meat = scores.T @ scores
            if key_vce == "robust":
                meat = meat * (n / (n - 1.0))
            vce_used = key_vce
        vcov = best["B"] @ meat @ best["B"]
    se_all = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    npar = P.shape[1]
    table = pd.DataFrame(
        {
            "term": [sm.label for sm in smooths],
            "edf": [float(best["edf_each"][sm.cols].sum()) for sm in smooths],
            "lambda": [float(v) for v in lams],
            "k": [sm.k for sm in smooths],
        }
    )
    for _, row in table.iterrows():
        if row["edf"] > 0.95 * (row["k"] - 1):
            warnings.warn(
                f"gam: {row['term']} uses {row['edf']:.1f} of its "
                f"{int(row['k']) - 1} available degrees of freedom; the basis "
                "may be too small. Raise k for this term.",
                UserWarning,
                stacklevel=2,
            )
    return GAMResult(
        params=pd.Series(beta[:npar], index=par_names),
        std_errors=pd.Series(se_all[:npar], index=par_names),
        smooth_terms=table,
        edf=float(best["edf"]),
        scale=float(best["phi"]),
        gcv=float(best["score"]),
        deviance=float(best["dev"]),
        n_obs=int(n),
        family=fam_key,
        link=lnk.name,
        formula=formula,
        criterion="reml" if how == "reml" else ("ubre" if known_scale else "gcv"),
        vce=vce_used,
        converged=bool(best["ok"]),
        alpha=alpha,
        fitted_values=np.asarray(best["mu"], dtype=float),
        residuals=np.asarray(y - best["mu"], dtype=float),
        _coef=np.asarray(beta, dtype=float),
        _vcov=np.asarray(vcov, dtype=float),
        _smooths=smooths,
        _design_info=design_info,
        _par_names=par_names,
        _link_name=lnk.name,
    )
