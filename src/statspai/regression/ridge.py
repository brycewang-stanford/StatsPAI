"""
Ridge regression (Hoerl and Kennard 1970).

``sp.ridge`` fits ``min_b ||y - a - Xb||^2 + lambda ||b||^2`` on
standardised regressors over one or many penalties and picks one by
generalized cross-validation (Golub, Heath and Wahba 1979). The scaling,
the GCV score and the two plug-in penalties (HKB, LW) are those of R
``MASS::lm.ridge``, so results can be compared digit for digit.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import DataInsufficient, MethodIncompatibility


@dataclass
class RidgeResult(ResultProtocolMixin):
    """Result of :func:`ridge`.

    Attributes
    ----------
    params : pd.Series
        Coefficients at the chosen penalty, on the original scale of the
        regressors, intercept first (when the model has one).
    lambda_ : float
        The chosen penalty.
    path : pd.DataFrame
        One row per penalty: ``lambda``, ``gcv``, ``df`` (effective degrees
        of freedom, the trace of the hat matrix) and the coefficients.
    lambda_gcv, lambda_hkb, lambda_lw : float
        The GCV minimiser over the supplied grid, and the Hoerl-Kennard-
        Baldwin and Lawless-Wang plug-in penalties.
    n_obs : int
    selection : str
        How ``lambda_`` was chosen.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
    >>> df["y"] = df["a"] - df["b"] + rng.normal(size=100)
    >>> fit = sp.ridge("y ~ a + b + c", df, lambda_=[0.0, 5.0])
    >>> isinstance(fit, sp.RidgeResult)
    True
    >>> list(fit.path.columns)
    ['lambda', 'gcv', 'df', 'Intercept', 'a', 'b', 'c']
    >>> fit.predict(df.head(2)).shape
    (2,)
    """

    _citation_keys = ("hoerl1970ridge", "golub1979generalized")
    params: pd.Series
    lambda_: float
    path: pd.DataFrame
    lambda_gcv: float
    lambda_hkb: float
    lambda_lw: float
    n_obs: int
    selection: str = "gcv"
    formula: Optional[str] = None
    _design_info: Any = field(default=None, repr=False)
    _names: List[str] = field(default_factory=list, repr=False)
    _fitted: Any = field(default=None, repr=False)

    @property
    def coef(self) -> pd.Series:
        return self.params

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Fitted values, on ``data`` when given."""
        if data is None:
            return np.asarray(self._fitted, dtype=float)
        if self._design_info is not None:
            from patsy import build_design_matrices

            X = np.asarray(
                build_design_matrices([self._design_info], data)[0], dtype=float
            )
        else:
            cols = [c for c in self._names if c != "Intercept"]
            X = data[cols].to_numpy(dtype=float)
            if "Intercept" in self._names:
                X = np.column_stack([np.ones(len(X)), X])
        return np.asarray(X @ self.params.to_numpy(), dtype=float)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "Ridge regression",
            "=" * 60,
            f"  Observations      : {self.n_obs}",
            f"  Penalty (lambda)  : {self.lambda_:.6g}  [{self.selection}]",
            f"  GCV minimiser     : {self.lambda_gcv:.6g} (over the grid)",
            f"  HKB / LW plug-ins : {self.lambda_hkb:.6g} / {self.lambda_lw:.6g}",
            "",
            "  Coefficients (original scale)",
        ]
        for name, value in self.params.items():
            lines.append(f"    {str(name):<24s} {value: .6g}")
        lines.append("=" * 60)
        text = "\n".join(lines)
        print(text)
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "params": {str(k): float(v) for k, v in self.params.items()},
            "lambda": float(self.lambda_),
            "lambda_gcv": float(self.lambda_gcv),
            "lambda_hkb": float(self.lambda_hkb),
            "lambda_lw": float(self.lambda_lw),
            "n_obs": int(self.n_obs),
            "selection": self.selection,
        }


_RIDGE_RULES = ("gcv", "hkb", "lw")


def _ridge_design(formula, data, y, x):
    """Outcome, regressors (no intercept column), names and design info."""
    if formula is not None:
        y_df, X_df = create_design_matrices(formula, data)
        names = [str(c) for c in X_df.columns]
        X = np.asarray(X_df, dtype=float)
        is_const = [n == "Intercept" for n in names]
        has_int = any(is_const)
        keep = [i for i, c in enumerate(is_const) if not c]
        return (
            np.asarray(y_df, dtype=float).reshape(-1),
            X[:, keep],
            [names[i] for i in keep],
            has_int,
            getattr(X_df, "design_info", None),
        )
    if y is None or x is None or data is None:
        raise MethodIncompatibility(
            "ridge: provide a formula or (y, x).",
            recovery_hint="Pass formula='y ~ x1 + x2' and data, or y='y', "
            "x=['x1', 'x2'] and data.",
        )
    cols = [x] if isinstance(x, str) else list(x)
    frame = data[[y] + cols].dropna()
    return (
        frame[y].to_numpy(dtype=float),
        frame[cols].to_numpy(dtype=float),
        [str(c) for c in cols],
        True,
        None,
    )


def _ridge_penalties(lambda_, select):
    """Resolve ``lambda_`` / ``select`` into (grid or None, select, given)."""
    if select not in _RIDGE_RULES:
        raise MethodIncompatibility(
            f"ridge: select={select!r} is not 'gcv', 'hkb' or 'lw'.",
            recovery_hint="Use select='gcv', 'hkb' or 'lw'.",
        )
    if isinstance(lambda_, str):
        if lambda_ not in _RIDGE_RULES:
            raise MethodIncompatibility(
                f"ridge: lambda_={lambda_!r} is not a number, a sequence of "
                "numbers, or one of the rules 'gcv', 'hkb', 'lw'.",
                recovery_hint="Pass penalties as numbers, or a rule name.",
            )
        return None, lambda_
    if lambda_ is None:
        return None, select
    grid = np.atleast_1d(np.asarray(lambda_, dtype=float)).reshape(-1)
    if grid.size == 0:
        raise MethodIncompatibility(
            "ridge: lambda_ is an empty sequence.",
            recovery_hint="Pass at least one penalty, or None for the default grid.",
        )
    if not np.all(np.isfinite(grid)) or np.any(grid < 0):
        raise MethodIncompatibility(
            "ridge: penalties must be finite and >= 0.",
            recovery_hint="Drop negative or non-finite penalties.",
        )
    return grid, select


def ridge(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[Sequence[str]] = None,
    lambda_: Union[None, str, float, Sequence[float]] = None,
    select: str = "gcv",
) -> RidgeResult:
    """Ridge regression with penalty chosen by generalized cross-validation.

    Minimises ``||y - a - Xb||^2 + lambda ||b||^2`` after centring ``y`` and
    scaling each regressor to unit root-mean-square deviation; the
    intercept is not penalised and the coefficients are reported on the
    original scale. The scaling, the GCV score and the two plug-in
    penalties reproduce the outputs of R ``MASS::lm.ridge(formula, data,
    lambda)`` with ``select()`` (checked digit for digit; clean-room
    implementation, ``MASS`` used as a black box, see
    ``docs/dev/cleanroom/ridge.md``).

    Parameters
    ----------
    formula : str, optional
        Model formula, e.g. ``"medv ~ crim + rm"``.
    data : pd.DataFrame
    y, x : str and list of str, optional
        Outcome and regressors, as an alternative to ``formula``.
    lambda_ : float, sequence of float, str or None
        Penalty or grid of penalties (on the standardised scale, as
        ``lm.ridge``). ``None`` builds a grid of 100 values, log-spaced
        from ``1e-4`` to ``1e4`` times the mean squared singular value of
        the standardised design, plus zero. One of the rule names
        ``"gcv"``, ``"hkb"``, ``"lw"`` is the default grid with that
        ``select``.
    select : {"gcv", "hkb", "lw"}, default "gcv"
        Which penalty the returned coefficients use when more than one is
        available: the GCV minimiser over the grid, or the HKB or LW
        plug-in (a refit at that value).

    Returns
    -------
    RidgeResult
        ``params`` at the chosen penalty, the whole ``path`` (GCV, effective
        degrees of freedom and coefficients per penalty) and the three
        candidate penalties.

    Notes
    -----
    ``lambda`` here multiplies the squared norm of the coefficients of the
    *standardised* regressors and is not divided by ``n``. ``glmnet`` and
    scikit-learn scale the loss differently, so the same number means a
    different amount of shrinkage there.

    The GCV score is ``RSS / (n - df)^2`` with ``df`` the trace of the hat
    matrix of the slopes: Golub, Heath and Wahba's criterion divided by
    ``n``, so it has the same minimiser. The plug-ins use the
    ``lambda = 0`` slopes ``b0`` (standardised scale) and
    ``s2 = RSS / (n - r - 1)`` (``n - r`` without an intercept; ``r`` the
    rank of the regressors, ``p`` when they are linearly independent):
    Hoerl-Kennard-Baldwin ``(p - 2) s2 / b0'b0`` and Lawless-Wang
    ``(p - 2) s2 n / b0'Z'Z b0``. They are NaN with two or fewer
    regressors.

    With linearly dependent regressors least squares is not unique. At
    ``lambda = 0`` (also the first row of the default path) the
    coefficients are then the minimum-norm solution on the standardised
    scale, with a warning; any positive penalty gives the usual unique
    ridge solution. A regressor that does not vary is refused.

    Ridge trades variance for bias; it does not select variables and has no
    standard errors here. For inference after shrinkage use ``sp.rlasso``
    or double machine learning.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 5))
    >>> df = pd.DataFrame(X, columns=list("abcde"))
    >>> df["y"] = X @ np.array([1.0, 0.5, 0.0, 0.0, -0.5]) + rng.normal(size=200)
    >>> fit = sp.ridge("y ~ a + b + c + d + e", df, lambda_=[0.0, 1.0, 10.0])
    >>> len(fit.path) == 3 and bool(fit.lambda_ in (0.0, 1.0, 10.0))
    True
    >>> ols = sp.ridge("y ~ a + b + c + d + e", df, lambda_=0.0).params
    >>> shrunk = sp.ridge("y ~ a + b + c + d + e", df, lambda_=50.0).params
    >>> bool(abs(shrunk["a"]) < abs(ols["a"]))
    True

    References
    ----------
    [@hoerl1970ridge],
    [@golub1979generalized]
    """
    grid, rule = _ridge_penalties(lambda_, select)
    yv, X, names, has_int, design_info = _ridge_design(formula, data, y, x)
    n, p = X.shape
    if p == 0:
        raise MethodIncompatibility(
            "ridge: the model has no regressors to penalise.",
            recovery_hint="Add at least one regressor to the formula.",
        )
    if n < 2:
        raise DataInsufficient(
            f"ridge: {n} complete observation(s); at least 2 are needed.",
            recovery_hint="Check for missing values in the outcome and regressors.",
        )

    # 1. centre (with an intercept) and scale to unit root mean square.
    xbar = X.mean(axis=0) if has_int else np.zeros(p)
    ybar = float(yv.mean()) if has_int else 0.0
    Xc = X - xbar
    yc = yv - ybar
    scale = np.sqrt(np.mean(Xc**2, axis=0))
    flat = scale <= 1e-10 * np.maximum(np.max(np.abs(X), axis=0), np.finfo(float).tiny)
    if np.any(flat):
        bad = [names[j] for j in np.flatnonzero(flat)]
        raise MethodIncompatibility(
            f"ridge: regressor(s) {bad} do not vary.",
            recovery_hint="Drop the constant column(s).",
        )
    Z = Xc / scale

    # 2. SVD; singular values below the rank tolerance are the null space.
    U, d, Vt = np.linalg.svd(Z, full_matrices=False)
    tol = max(n, p) * np.finfo(float).eps * (d[0] if d.size else 0.0)
    live = d > tol
    U, d, Vt = U[:, live], d[live], Vt[live]
    rank = int(d.size)
    uty = U.T @ yc

    def slopes_at(lam: float) -> np.ndarray:
        return Vt.T @ ((d / (d**2 + lam)) * uty)

    def rss_of(b: np.ndarray) -> float:
        r = yc - Z @ b
        return float(r @ r)

    if grid is None:
        m = float(np.mean(np.linalg.svd(Z, compute_uv=False) ** 2))
        grid = np.concatenate([[0.0], np.logspace(-4.0, 4.0, 100) * m])
    if p >= n and np.any(grid == 0.0):
        raise MethodIncompatibility(
            "ridge: lambda = 0 is least squares, which is not unique with at "
            "least as many regressors as observations.",
            recovery_hint="Use strictly positive penalties.",
        )
    if rank < p and np.any(grid == 0.0):
        warnings.warn(
            f"ridge: the regressors have rank {rank}, fewer than their number "
            f"({p}), so least squares (lambda = 0) is not unique. The "
            "lambda = 0 coefficients are the minimum-norm solution on the "
            "standardised scale, not a least-squares estimate of each "
            "coefficient.",
            UserWarning,
            stacklevel=2,
        )

    def original_scale(b: np.ndarray) -> np.ndarray:
        slope = b / scale
        return np.concatenate([[ybar - xbar @ slope], slope]) if has_int else slope

    # 3. the path.
    rows = []
    for lam in grid:
        b = slopes_at(float(lam))
        dof = float(np.sum(d**2 / (d**2 + lam)))
        gcv = rss_of(b) / (n - dof) ** 2
        # Reported df: trace of the full hat matrix, intercept included.
        rows.append(
            [float(lam), gcv, dof + (1.0 if has_int else 0.0), *original_scale(b)]
        )
    coef_names = (["Intercept"] if has_int else []) + names
    path = pd.DataFrame(rows, columns=["lambda", "gcv", "df", *coef_names])

    # 5. plug-in penalties from the lambda = 0 fit.
    b0 = slopes_at(0.0)
    resid_df = n - rank - (1 if has_int else 0)
    if p > 2 and resid_df > 0:
        s2 = rss_of(b0) / resid_df
        lam_hkb = (p - 2) * s2 / float(b0 @ b0)
        zb = Z @ b0
        lam_lw = (p - 2) * s2 * n / float(zb @ zb)
    else:
        lam_hkb = lam_lw = float("nan")

    lam_gcv = float(path["lambda"].iloc[int(np.argmin(path["gcv"].to_numpy()))])
    if len(grid) == 1:
        chosen, selection = float(grid[0]), "given"
    elif rule == "gcv":
        chosen, selection = lam_gcv, "gcv"
    else:
        chosen = lam_hkb if rule == "hkb" else lam_lw
        if not np.isfinite(chosen):
            raise MethodIncompatibility(
                f"ridge: the {rule.upper()} penalty needs more than two "
                "regressors and a positive residual degrees of freedom.",
                recovery_hint="Use select='gcv' or supply the penalty.",
            )
        selection = rule

    coef = original_scale(slopes_at(chosen))
    design = np.column_stack([np.ones(n), X]) if has_int else X
    return RidgeResult(
        params=pd.Series(coef, index=coef_names),
        lambda_=chosen,
        path=path,
        lambda_gcv=lam_gcv,
        lambda_hkb=float(lam_hkb),
        lambda_lw=float(lam_lw),
        n_obs=int(n),
        selection=selection,
        formula=formula,
        _design_info=design_info,
        _names=coef_names,
        _fitted=design @ coef,
    )
