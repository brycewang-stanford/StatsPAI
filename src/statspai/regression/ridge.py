"""
Ridge regression (Hoerl and Kennard 1970).

``sp.ridge`` fits ``min_b ||y - a - Xb||^2 + lambda ||b||^2`` on
standardised regressors over one or many penalties and picks one by
generalized cross-validation (Golub, Heath and Wahba 1979). The scaling,
the GCV score and the two plug-in penalties (HKB, LW) are those of R
``MASS::lm.ridge``, so results can be compared digit for digit.
"""

from __future__ import annotations

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


def ridge(
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[Sequence[str]] = None,
    lambda_: Union[None, str, float, Sequence[float]] = None,
    select: str = "gcv",
) -> RidgeResult:
    """
    Ridge regression with penalty chosen by generalized cross-validation.

    Minimises ``||y - a - Xb||^2 + lambda ||b||^2`` after centring ``y`` and
    scaling each regressor to unit root-mean-square deviation; the
    intercept is not penalised and the coefficients are reported on the
    original scale. Equivalent to R ``MASS::lm.ridge(formula, data, lambda)``
    with ``select()``.

    Parameters
    ----------
    formula : str, optional
        Model formula, e.g. ``"medv ~ crim + rm"``.
    data : pd.DataFrame
    y, x : str and list of str, optional
        Outcome and regressors, as an alternative to ``formula``.
    lambda_ : float, sequence of float or None
        Penalty or grid of penalties (on the standardised scale, as
        ``lm.ridge``). ``None`` builds a grid of 100 values, log-spaced
        from ``1e-4`` to ``1e4`` times the mean squared singular value of
        the standardised design, plus zero.
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
    [@hoerl1970ridge]
    [@golub1979generalized]
    """
    if data is None:
        raise MethodIncompatibility("ridge: data is required.")
    key = str(select).lower()
    if key not in ("gcv", "hkb", "lw"):
        raise MethodIncompatibility(
            f"ridge: select={select!r} is not 'gcv', 'hkb' or 'lw'.",
            diagnostics={"select": select},
        )
    design_info = None
    if formula is not None:
        y_df, X_df = create_design_matrices(formula, data)
        design_info = getattr(X_df, "design_info", None)
        names = [str(c) for c in X_df.columns]
        yv = np.asarray(y_df, dtype=float).reshape(len(X_df), -1)[:, -1]
        Xall = np.asarray(X_df, dtype=float)
    elif y is not None and x is not None:
        cols = [y] + list(x)
        frame = data[cols].dropna()
        yv = frame[y].to_numpy(dtype=float)
        Xall = np.column_stack([np.ones(len(frame)), frame[list(x)].to_numpy(float)])
        names = ["Intercept"] + list(x)
    else:
        raise MethodIncompatibility("ridge: provide a formula or (y, x).")

    has_const = "Intercept" in names
    slope = [i for i, nm in enumerate(names) if nm != "Intercept"]
    Xr = Xall[:, slope]
    n, p = Xr.shape
    if p == 0:
        raise MethodIncompatibility("ridge: the model has no regressors.")
    if n <= 1:
        raise DataInsufficient("ridge: at least two observations are needed.")

    # MASS::lm.ridge scaling: centre when there is an intercept, then divide
    # each column by its root-mean-square (divisor n, not n - 1)
    xm = Xr.mean(axis=0) if has_const else np.zeros(p)
    ym = float(yv.mean()) if has_const else 0.0
    Xc = Xr - xm
    yc = yv - ym
    scale = np.sqrt(np.mean(Xc**2, axis=0))
    if np.any(scale <= 0):
        bad = [names[slope[i]] for i in np.flatnonzero(scale <= 0)]
        raise MethodIncompatibility(
            f"ridge: regressor(s) {bad} do not vary.",
            recovery_hint="Drop the constant column(s).",
        )
    Xs = Xc / scale
    U, d, Vt = np.linalg.svd(Xs, full_matrices=False)
    rhs = U.T @ yc
    keep = d > d.max() * max(n, p) * np.finfo(float).eps
    ls_coef = Vt.T[:, keep] @ (rhs[keep] / d[keep])
    ls_fit = Xs @ ls_coef
    df_resid = n - int(keep.sum()) - int(has_const)
    s2 = float(np.sum((yc - ls_fit) ** 2) / df_resid) if df_resid > 0 else np.nan
    hkb = float((p - 2) * s2 / np.sum(ls_coef**2)) if p > 2 else np.nan
    lw = float((p - 2) * s2 * n / np.sum(ls_fit**2)) if p > 2 else np.nan

    if lambda_ is None:
        base = float(np.mean(d**2))
        grid = np.concatenate([[0.0], base * np.logspace(-4, 4, 100)])
    else:
        grid = np.atleast_1d(np.asarray(lambda_, dtype=float)).ravel()
    if np.any(~np.isfinite(grid)) or np.any(grid < 0):
        raise MethodIncompatibility("ridge: penalties must be finite and >= 0.")
    if p >= n and np.any(grid == 0):
        raise MethodIncompatibility(
            "ridge: lambda = 0 is least squares, which is not unique with "
            "at least as many regressors as observations.",
            recovery_hint="Use strictly positive penalties.",
        )

    def fit_at(lams: np.ndarray) -> Dict[str, np.ndarray]:
        div = d[:, None] ** 2 + lams[None, :]
        a = (d * rhs)[:, None] / div
        coef_s = Vt.T @ a  # p x k, standardised scale
        resid = yc[:, None] - Xs @ coef_s
        edf = np.sum(d[:, None] ** 2 / div, axis=0)
        gcv = np.sum(resid**2, axis=0) / (n - edf) ** 2
        coef = coef_s / scale[:, None]
        inter = ym - xm @ coef
        return {"coef": coef, "inter": inter, "gcv": gcv, "edf": edf}

    out = fit_at(grid)
    best = int(np.argmin(out["gcv"]))
    lam_gcv = float(grid[best])
    slope_names = [names[i] for i in slope]
    path = pd.DataFrame(
        {
            "lambda": grid,
            "gcv": out["gcv"],
            "df": out["edf"] + int(has_const),
        }
    )
    if has_const:
        path["Intercept"] = out["inter"]
    for j, nm in enumerate(slope_names):
        path[nm] = out["coef"][j]

    if len(grid) == 1:
        lam, pick, how = float(grid[0]), out, "given"
        col = 0
    elif key == "gcv":
        lam, pick, how, col = lam_gcv, out, "gcv", best
    else:
        lam = hkb if key == "hkb" else lw
        if not np.isfinite(lam):
            raise MethodIncompatibility(
                f"ridge: the {key.upper()} penalty needs more than two "
                "regressors and a positive residual degrees of freedom.",
            )
        pick, how, col = fit_at(np.array([lam])), key, 0
    values = ([float(pick["inter"][col])] if has_const else []) + [
        float(v) for v in pick["coef"][:, col]
    ]
    ordered = (["Intercept"] if has_const else []) + slope_names
    params = pd.Series(values, index=ordered)
    # predict() multiplies the design in its own column order
    params = params.reindex(names)
    res = RidgeResult(
        params=params,
        lambda_=lam,
        path=path,
        lambda_gcv=lam_gcv,
        lambda_hkb=hkb,
        lambda_lw=lw,
        n_obs=int(n),
        selection=how,
        formula=formula,
        _design_info=design_info,
        _names=names,
    )
    res._fitted = np.asarray(Xall @ params.to_numpy(), dtype=float)
    return res
