"""Prediction with many regressors: ridge, lasso and principal components.

One entry point, :func:`shrinkage`, for the estimators a predictive
regression with many predictors is fitted with, and for the m-fold
cross-validation that picks their tuning parameter and estimates the mean
squared prediction error.

Conventions, shared by every method:

* The predictors are standardised (mean zero, unit standard deviation with
  the ``n - 1`` divisor) and the outcome is demeaned on the estimation
  sample; coefficients are reported for the standardised predictors and the
  intercept is the mean of the outcome.
* Ridge minimises ``SSR + penalty * sum(b_j ** 2)`` and the lasso
  ``SSR + penalty * sum(|b_j|)``: the penalty multiplies the sum of squared
  residuals as it stands, not divided by the sample size. In scikit-learn's
  parameterisation that is ``Ridge(alpha=penalty)`` and
  ``Lasso(alpha=penalty / (2 * n))``.
* In cross-validation the means and standard deviations are recomputed on
  each training fold, so nothing of the held-out fold enters its own
  prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

_METHODS = ("ols", "ridge", "lasso", "pcr")


@dataclass
class ShrinkageResult(ResultProtocolMixin):
    """Fit returned by :func:`statspai.shrinkage`.

    Attributes
    ----------
    method : str
    params : pd.Series
        Coefficients of the standardised predictors.
    intercept : float
        Mean of the outcome on the estimation sample.
    penalty : float or None
        The ridge / lasso penalty of the reported fit.
    n_components : int or None
        Number of principal components of the reported fit.
    cv : pd.DataFrame or None
        One row per candidate tuning value: the value, the cross-validated
        mean squared prediction error and its square root.
    cv_rmspe : float or None
        Cross-validated root MSPE of the reported fit.
    rmspe_in : float
        In-sample root mean squared error (no degrees-of-freedom correction).
    n_obs, n_predictors, n_nonzero : int

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(120, 8))
    >>> df = pd.DataFrame(X, columns=[f"x{j}" for j in range(8)])
    >>> df["y"] = X[:, 0] - X[:, 1] + rng.normal(size=120)
    >>> fit = sp.shrinkage(df, "y", [f"x{j}" for j in range(8)], method="ridge")
    >>> type(fit).__name__
    'ShrinkageResult'
    >>> fit.predict(df.head(2)).shape
    (2,)
    >>> bool(fit.cv_rmspe > 0 and fit.penalty in set(fit.cv["penalty"]))
    True
    """

    method: str
    params: pd.Series
    intercept: float
    penalty: Optional[float]
    n_components: Optional[int]
    cv: Optional[pd.DataFrame]
    cv_rmspe: Optional[float]
    rmspe_in: float
    n_obs: int
    n_predictors: int
    n_nonzero: int
    y: str
    x: List[str]
    n_folds: Optional[int] = None
    selected_by_cv: bool = False
    _center: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)
    _scale: np.ndarray = field(default_factory=lambda: np.ones(0), repr=False)

    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Predicted outcome for the rows of ``data`` (which must hold the
        predictors, in their original units)."""
        if data is None:
            raise MethodIncompatibility(
                "predict() needs the data to predict for.",
                recovery_hint="Pass the estimation frame or a hold-out frame.",
            )
        missing = [c for c in self.x if c not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"predict(): columns {missing} are not in the data.",
                diagnostics={"missing_columns": missing},
            )
        X = data[self.x].to_numpy(dtype=float)
        Z = (X - self._center) / self._scale
        return np.asarray(self.intercept + Z @ self.params.to_numpy(dtype=float))

    def rmspe(self, data: pd.DataFrame) -> float:
        """Root mean squared prediction error on ``data``, a frame holding
        the outcome and the predictors (a hold-out sample)."""
        if self.y not in data.columns:
            raise MethodIncompatibility(
                f"rmspe(): the outcome {self.y!r} is not in the data."
            )
        err = data[self.y].to_numpy(dtype=float) - self.predict(data)
        ok = np.isfinite(err)
        if not ok.any():
            raise MethodIncompatibility("rmspe(): no complete rows in the data.")
        return float(np.sqrt(np.mean(err[ok] ** 2)))

    def summary(self) -> str:
        label = {
            "ols": "OLS",
            "ridge": "Ridge regression",
            "lasso": "Lasso",
            "pcr": "Principal components regression",
        }[self.method]
        lines = [
            f"Predictive regression: {label}",
            "=" * 60,
            f"Outcome: {self.y}   Observations: {self.n_obs}   "
            f"Predictors: {self.n_predictors}",
        ]
        if self.penalty is not None:
            how = f"{self.n_folds}-fold CV" if self.selected_by_cv else "given"
            lines.append(f"Penalty: {self.penalty:.6g} ({how})")
        if self.n_components is not None:
            how = f"{self.n_folds}-fold CV" if self.selected_by_cv else "given"
            lines.append(f"Principal components: {self.n_components} ({how})")
        if self.method == "lasso":
            lines.append(f"Nonzero coefficients: {self.n_nonzero}")
        lines.append(f"In-sample RMSPE: {self.rmspe_in:.4f}")
        if self.cv_rmspe is not None:
            lines.append(f"{self.n_folds}-fold CV RMSPE: {self.cv_rmspe:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "outcome": self.y,
            "n_obs": self.n_obs,
            "n_predictors": self.n_predictors,
            "n_nonzero": self.n_nonzero,
            "penalty": self.penalty,
            "n_components": self.n_components,
            "n_folds": self.n_folds,
            "selected_by_cv": self.selected_by_cv,
            "rmspe_in": self.rmspe_in,
            "cv_rmspe": self.cv_rmspe,
            "intercept": self.intercept,
            "params": {str(k): float(v) for k, v in self.params.items()},
        }

    def plot(self, ax: Any = None) -> Any:
        """Cross-validated RMSPE against the tuning parameter."""
        if self.cv is None:
            raise MethodIncompatibility(
                "plot(): this fit was not cross-validated over a grid."
            )
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        col = self.cv.columns[0]
        ax.plot(self.cv[col], self.cv["rmspe"], color="steelblue")
        if col == "penalty":
            ax.set_xscale("log")
        chosen = self.penalty if col == "penalty" else self.n_components
        ax.axvline(chosen, color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("penalty" if col == "penalty" else "principal components")
        ax.set_ylabel("cross-validated RMSPE")
        return ax


def _standardise(
    X: np.ndarray, y: np.ndarray, scale: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    center = X.mean(axis=0)
    if scale:
        sd = X.std(axis=0, ddof=1)
    else:
        sd = np.ones(X.shape[1])
    ybar = float(y.mean())
    return (X - center) / sd, y - ybar, center, sd, ybar


def _path(
    method: str,
    Z: np.ndarray,
    yc: np.ndarray,
    grid: np.ndarray,
    lasso_tol: float = 1e-8,
) -> np.ndarray:
    """Coefficients for every value in ``grid``; shape ``(len(grid), k)``."""
    n, k = Z.shape
    if method == "lasso":
        from sklearn.linear_model import lasso_path

        order = np.argsort(-grid)  # lasso_path wants decreasing penalties
        # tol is scikit-learn's duality-gap tolerance, relative to y'y
        _, coefs, _ = lasso_path(
            Z,
            yc,
            alphas=grid[order] / (2.0 * n),
            precompute=k <= n,
            tol=lasso_tol,
            max_iter=100_000,
        )
        out = np.empty((len(grid), k))
        out[order] = coefs.T
        return out
    U, d, Vt = np.linalg.svd(Z, full_matrices=False)
    uy = U.T @ yc
    tol = d.max() * max(n, k) * np.finfo(float).eps if d.size else 0.0
    if method == "ridge":
        # b(l) = V diag(d / (d^2 + l)) U'y
        shrink = d[None, :] / (d[None, :] ** 2 + grid[:, None])
        return np.asarray((shrink * uy[None, :]) @ Vt)
    keep = d > tol
    load = np.where(keep, uy / np.where(keep, d, 1.0), 0.0)
    if method == "ols":
        return np.asarray(load[None, :] @ Vt)  # minimum-norm least squares
    # pcr: the first p components, p = grid
    out = np.empty((len(grid), k))
    cumulative = np.cumsum(load[:, None] * Vt, axis=0)
    for i, p in enumerate(grid.astype(int)):
        out[i] = cumulative[min(p, len(d)) - 1]
    return out


def _folds(
    n: int, n_folds: int, shuffle: bool, seed: Optional[int]
) -> List[np.ndarray]:
    index: np.ndarray = np.arange(n)
    if shuffle:
        index = np.random.default_rng(seed).permutation(n)
    return [np.sort(part) for part in np.array_split(index, n_folds)]


def shrinkage(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    method: str = "ridge",
    *,
    penalty: Union[None, float, Sequence[float]] = None,
    n_components: Union[None, int, Sequence[int]] = None,
    n_folds: int = 10,
    shuffle: bool = False,
    seed: Optional[int] = None,
    standardize: bool = True,
) -> ShrinkageResult:
    """Ridge, lasso and principal-components prediction, cross-validated.

    A predictive regression with many predictors. The tuning parameter
    (penalty, or number of components) is chosen by m-fold cross-validation,
    which also estimates the mean squared prediction error.

    Parameters
    ----------
    data : pandas.DataFrame
        The estimation sample. Rows with a missing outcome or predictor are
        dropped.
    y : str
        Outcome.
    x : sequence of str
        Predictors, in their original units.
    method : {'ridge', 'lasso', 'pcr', 'ols'}, default 'ridge'
        ``'ridge'`` minimises ``SSR + penalty * sum(b ** 2)``; ``'lasso'``
        minimises ``SSR + penalty * sum(|b|)``; ``'pcr'`` regresses the
        outcome on the first ``n_components`` principal components of the
        standardised predictors; ``'ols'`` is least squares (the
        minimum-norm solution when the predictors are collinear), for the
        cross-validated benchmark.
    penalty : float or sequence of float, optional
        Ridge / lasso penalty. A single value is used as given; a sequence
        is searched by cross-validation; ``None`` searches a default grid
        (ridge: 61 values, log-spaced over ``n * 10 ** [-3, 3]``; lasso: 100
        values, log-spaced from the smallest penalty that sets every
        coefficient to zero down to a thousandth of it).
    n_components : int or sequence of int, optional
        For ``'pcr'``: the number of components, or the candidates to
        search. ``None`` searches ``1 .. min(k, n_train - 1)``.
    n_folds : int, default 10
        Folds for cross-validation. The cross-validated MSPE is the mean
        squared error of the out-of-fold predictions over all observations.
    shuffle : bool, default False
        ``False`` cuts the sample into consecutive blocks in the order of
        the rows; ``True`` assigns rows to folds at random (``seed``).
    seed : int, optional
    standardize : bool, default True
        Scale each predictor to unit standard deviation. ``False`` only
        centres them, which makes the penalty depend on their units.

    Returns
    -------
    ShrinkageResult
        ``params`` (coefficients of the standardised predictors),
        ``intercept``, the chosen ``penalty`` / ``n_components``, the
        cross-validation table ``cv``, ``cv_rmspe``, ``rmspe_in``, and the
        methods ``predict(data)`` and ``rmspe(holdout)``.

    Notes
    -----
    Means and standard deviations are recomputed on every training fold.
    Standardising once on the whole sample before splitting, as some
    textbook code does, lets the held-out fold enter its own prediction;
    the two differ little when folds are large.

    :func:`statspai.ridge` is the other ridge in the package: the
    ``MASS::lm.ridge`` parameterisation with the penalty chosen by
    generalised cross-validation, for looking at a ridge trace. This
    function is for prediction, with m-fold cross-validation and a
    hold-out error.

    The lasso is solved by scikit-learn's coordinate descent
    (``lasso_path``): at its default tolerance (``1e-4``) while the grid is
    searched, at ``1e-8`` for the reported fit. Ridge and principal
    components use the singular value decomposition of the standardised
    predictors.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 30))
    >>> df = pd.DataFrame(X, columns=[f"x{j}" for j in range(30)])
    >>> df["y"] = X[:, 0] - 2 * X[:, 1] + rng.normal(size=200)
    >>> cols = [f"x{j}" for j in range(30)]
    >>> fit = sp.shrinkage(df[:150], "y", cols, method="lasso")
    >>> fit.n_nonzero < 30
    True
    >>> holdout_rmspe = fit.rmspe(df[150:])
    """
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"shrinkage: unknown method {method!r}.",
            recovery_hint=f"Choose one of {list(_METHODS)}.",
            diagnostics={"method": method, "valid": list(_METHODS)},
        )
    x = [x] if isinstance(x, str) else list(x)
    missing = [c for c in [y] + x if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"shrinkage: columns {missing} are not in the data.",
            diagnostics={"missing_columns": missing},
        )
    if not x:
        raise MethodIncompatibility("shrinkage: no predictors given.")
    if method != "pcr" and n_components is not None:
        raise MethodIncompatibility(
            "shrinkage: n_components applies to method='pcr' only."
        )
    if method not in ("ridge", "lasso") and penalty is not None:
        raise MethodIncompatibility(
            "shrinkage: penalty applies to method='ridge' / 'lasso' only."
        )

    frame = data[[y] + x].apply(pd.to_numeric, errors="coerce").dropna()
    yv = frame[y].to_numpy(dtype=float)
    X = frame[x].to_numpy(dtype=float)
    n, k = X.shape
    if n < 3:
        raise MethodIncompatibility(
            f"shrinkage: {n} complete rows are too few to fit a model."
        )
    constant = [c for c, s in zip(x, X.std(axis=0)) if s == 0]
    if constant:
        raise MethodIncompatibility(
            f"shrinkage: predictors {constant} do not vary.",
            recovery_hint="Drop them; a constant cannot be standardised.",
            diagnostics={"constant_columns": constant},
        )

    Z, yc, center, sd, ybar = _standardise(X, yv, standardize)

    # ---- candidate tuning values -------------------------------------
    grid: Optional[np.ndarray] = None
    key = None
    if method in ("ridge", "lasso"):
        key = "penalty"
        if penalty is None:
            if method == "ridge":
                grid = n * np.logspace(-3, 3, 61)
            else:
                top = 2.0 * float(np.max(np.abs(Z.T @ yc)))
                grid = np.logspace(np.log10(top), np.log10(top) - 3, 100)
        else:
            grid = np.atleast_1d(np.asarray(penalty, dtype=float))
            if grid.ndim != 1 or not np.all(np.isfinite(grid)) or np.any(grid < 0):
                raise MethodIncompatibility(
                    "shrinkage: penalty must be non-negative and finite."
                )
            if method == "lasso" and np.any(grid == 0):
                raise MethodIncompatibility(
                    "shrinkage: a lasso penalty of zero is least squares; "
                    "use method='ols'."
                )
    elif method == "pcr":
        key = "n_components"
        if n_components is None:
            smallest_train = n - int(np.ceil(n / max(n_folds, 2)))
            grid = np.arange(1, min(k, max(smallest_train - 1, 1)) + 1, dtype=float)
        else:
            grid = np.atleast_1d(np.asarray(n_components, dtype=float))
            bad = (grid < 1) | (grid > min(k, n - 1)) | (grid != np.round(grid))
            if grid.ndim != 1 or np.any(bad):
                raise MethodIncompatibility(
                    "shrinkage: n_components must be whole numbers between "
                    f"1 and {min(k, n - 1)}.",
                    diagnostics={"n_components": grid.tolist()},
                )
    single = (
        grid is not None
        and len(grid) == 1
        and (penalty is not None or n_components is not None)
    )

    # ---- cross-validation --------------------------------------------
    cv_table: Optional[pd.DataFrame] = None
    cv_rmspe: Optional[float] = None
    chosen = 0
    run_cv = n_folds is not None and n_folds >= 2
    if run_cv:
        if n_folds > n:
            raise MethodIncompatibility(
                f"shrinkage: n_folds={n_folds} exceeds the {n} observations."
            )
        search = np.zeros(1) if grid is None else grid
        sse = np.zeros(len(search))
        for held in _folds(n, int(n_folds), shuffle, seed):
            train = np.setdiff1d(np.arange(n), held, assume_unique=True)
            Zt, yt, c_t, s_t, yb_t = _standardise(X[train], yv[train], standardize)
            if np.any(s_t == 0):
                raise MethodIncompatibility(
                    "shrinkage: a predictor is constant within a training "
                    "fold, so it cannot be standardised there.",
                    recovery_hint="Use shuffle=True, fewer folds, or drop "
                    "predictors that vary in a few rows only.",
                )
            # the search runs at scikit-learn's default tolerance; the
            # reported fit below is solved tightly
            B = _path(method, Zt, yt, search, lasso_tol=1e-4)
            pred = yb_t + ((X[held] - c_t) / s_t) @ B.T
            sse += ((yv[held][:, None] - pred) ** 2).sum(axis=0)
        mspe = sse / n
        if grid is not None:
            cv_table = pd.DataFrame({key: grid, "mspe": mspe, "rmspe": np.sqrt(mspe)})
            if key == "n_components":
                cv_table[key] = cv_table[key].astype(int)
            chosen = int(np.argmin(mspe))
        cv_rmspe = float(np.sqrt(mspe[chosen]))
    elif grid is not None and len(grid) > 1:
        raise MethodIncompatibility(
            "shrinkage: several candidate values need cross-validation "
            "(n_folds >= 2) to choose among them."
        )

    # ---- the reported fit, on the whole sample -------------------------
    final = np.zeros(1) if grid is None else grid[[chosen]]
    beta = _path(method, Z, yc, final)[0]
    resid = yc - Z @ beta
    result = ShrinkageResult(
        method=method,
        params=pd.Series(beta, index=x),
        intercept=ybar,
        penalty=float(final[0]) if key == "penalty" else None,
        n_components=int(final[0]) if key == "n_components" else None,
        cv=cv_table if not single else None,
        cv_rmspe=cv_rmspe,
        rmspe_in=float(np.sqrt(np.mean(resid**2))),
        n_obs=int(n),
        n_predictors=int(k),
        n_nonzero=int(np.count_nonzero(beta)),
        y=y,
        x=x,
        n_folds=int(n_folds) if run_cv else None,
        selected_by_cv=bool(run_cv and grid is not None and not single),
        _center=center,
        _scale=sd,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.selection.shrinkage",
            params={"y": y, "method": method, "n_folds": n_folds},
            data=data,
        )
    except Exception:  # pragma: no cover
        pass
    return result
