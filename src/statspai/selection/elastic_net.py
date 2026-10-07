"""Elastic net with the conventions of R ``glmnet``.

Lasso, ridge and everything between them for a Gaussian or a binomial
outcome, on a path of penalties, with K-fold cross-validation that reports
``lambda.min`` and ``lambda.1se``. The objective, the standardisation, the
penalty path and the cross-validation summaries are those of ``glmnet`` /
``cv.glmnet`` (Friedman, Hastie and Tibshirani 2010), so a penalty chosen
in one program means the same thing in the other, and with the same fold
assignment the two select the same penalty.

Written from the paper and the package documentation; ``glmnet`` is GPL and
its source was not read. The conventions that the documentation leaves
open (how the Gaussian outcome is scaled, where the path stops) were
recovered by comparing outputs and are recorded where they are used.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility

__all__ = ["glmnet", "GlmnetResult"]

_PMIN = 1e-5  # glmnet's guard on fitted probabilities
_BIG = 9.9e35  # stands in for an infinite penalty


def _soft(z: float, t: float) -> float:
    if z > t:
        return z - t
    if z < -t:
        return z + t
    return 0.0


def _cd_gaussian(
    gram: np.ndarray,
    xty: np.ndarray,
    b: np.ndarray,
    l1: np.ndarray,
    l2: np.ndarray,
    tol: float,
    max_iter: int,
) -> Tuple[np.ndarray, bool]:
    """Coordinate descent on ``b'Gb/2 - b'c + sum(l1 |b|) + sum(l2 b^2)/2``.

    ``gram = X'WX / n`` and ``xty = X'Wy / n``. Cycles over all coordinates
    until the largest squared change, scaled by the curvature, is below
    ``tol``.
    """
    p = len(b)
    grad = xty - gram @ b  # c - G b
    diag = np.diag(gram).copy()
    for _ in range(max_iter):
        dmax = 0.0
        for j in range(p):
            if diag[j] <= 0.0:
                continue
            old = b[j]
            z = grad[j] + diag[j] * old
            new = _soft(z, l1[j]) / (diag[j] + l2[j])
            if new != old:
                delta = new - old
                b[j] = new
                grad -= gram[:, j] * delta
                dmax = max(dmax, diag[j] * delta * delta)
        if dmax < tol:
            return b, True
    return b, False


def _standardise(
    X: np.ndarray, standardize: bool, intercept: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0) if intercept else np.zeros(X.shape[1])
    if standardize:
        sd = np.sqrt(np.mean((X - X.mean(axis=0)) ** 2, axis=0))
    else:
        sd = np.ones(X.shape[1])
    return mean, sd, (X - mean) / np.where(sd > 0, sd, 1.0)


class _Path:
    """One penalty path on one sample."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family: str,
        alpha: float,
        vp: np.ndarray,
        standardize: bool,
        intercept: bool,
        tol: float,
        max_iter: int,
    ) -> None:
        self.n, self.p = X.shape
        self.family, self.alpha, self.vp = family, alpha, vp
        self.intercept, self.tol, self.max_iter = intercept, tol, max_iter
        self.mean, self.sd, self.Z = _standardise(X, standardize, intercept)
        self.const = self.sd <= 0
        self.y = y
        self.converged = True
        if family == "gaussian":
            # glmnet scales a Gaussian outcome to unit variance (divisor n)
            # and states lambda on the original scale. The lasso term is
            # equivariant to that, the ridge term is not: in original units
            # the ridge part of the penalty is lambda / sd(y).
            self.ym = float(y.mean()) if intercept else 0.0
            self.ys = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
            if self.ys <= 0:
                raise DataInsufficient(
                    "glmnet: the outcome is constant.",
                    recovery_hint="There is nothing to fit.",
                )
            yc = (y - self.ym) / self.ys
            self.gram = self.Z.T @ self.Z / self.n
            self.xty = self.Z.T @ yc / self.n
            self.null_dev = float(np.mean(yc**2))
            self.a0_null = 0.0
        else:
            self.ys = 1.0
            if intercept:
                pbar = float(np.clip(y.mean(), _PMIN, 1 - _PMIN))
                self.a0_null = float(np.log(pbar / (1 - pbar)))
            else:
                self.a0_null = 0.0
            self.null_dev = float(self._deviance(np.full(self.n, self.a0_null)))
        # The path starts at the fit with an infinite penalty: penalised
        # coefficients at zero, unpenalised ones (penalty factor 0) free.
        # The largest finite penalty that still gives that fit is read off
        # its gradient.
        big = np.where(vp > 0, _BIG, 0.0)
        self.b_inf, self.a0_inf, _ = self._solve(
            np.zeros(self.p), self.a0_null, alpha * big, (1.0 - alpha) * big
        )
        if family == "gaussian":
            g0 = np.abs(self.xty - self.gram @ self.b_inf)
        else:
            prob = 1.0 / (1.0 + np.exp(-(self.a0_inf + self.Z @ self.b_inf)))
            g0 = np.abs(self.Z.T @ (y - prob)) / self.n
        pen = vp > 0
        ratio = np.where(pen & ~self.const, g0 / np.where(pen, vp, 1.0), 0.0)
        self.lambda_max = float(ratio.max() / max(alpha, 1e-3)) * self.ys

    def _solve(
        self, b: np.ndarray, a0: float, l1: np.ndarray, l2: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        if self.family == "gaussian":
            b, ok = _cd_gaussian(
                self.gram, self.xty, b, l1, l2, self.tol, self.max_iter
            )
            return b, 0.0, ok
        return self._irls(b, a0, l1, l2)

    def _dev_ratio(self, b: np.ndarray, a0: float) -> float:
        if self.family == "gaussian":
            rss = self.null_dev - 2 * b @ self.xty + b @ self.gram @ b
            return 1.0 - float(rss / self.null_dev)
        return 1.0 - self._deviance(a0 + self.Z @ b) / self.null_dev

    def _deviance(self, eta: np.ndarray) -> float:
        prob = 1.0 / (1.0 + np.exp(-eta))
        prob = np.clip(prob, 1e-15, 1 - 1e-15)
        y = self.y
        return float(-2.0 * np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob)))

    def fit(self, lambdas: np.ndarray, auto: bool) -> Dict[str, np.ndarray]:
        """Coefficients on the original scale for a decreasing ``lambdas``.

        ``auto`` marks a path built here rather than supplied. As in
        ``glmnet``, its first point is the fit with an infinite penalty
        (for the ridge that is not the fit at the penalty it is labelled
        with), and it stops once the fit has stopped improving: after at
        least five penalties, when the deviance explained exceeds 0.999 or
        its increase falls below 1e-5, relative to its level for a
        Gaussian outcome and in absolute terms for a binomial one.
        """
        b = self.b_inf.copy()
        a0 = self.a0_inf
        betas: List[np.ndarray] = []
        a0s: List[float] = []
        devs: List[float] = []
        for k, lam in enumerate(lambdas):
            if auto and k == 0:
                ok = True
            else:
                ls = lam / self.ys
                b, a0, ok = self._solve(
                    b,
                    a0,
                    ls * self.alpha * self.vp,
                    ls * (1.0 - self.alpha) * self.vp,
                )
            dev = self._dev_ratio(b, a0)
            self.converged = self.converged and ok
            betas.append(b.copy())
            a0s.append(a0)
            devs.append(dev)
            if auto and k >= 4:
                gain = dev - devs[-2]
                floor = 1e-5 * dev if self.family == "gaussian" else 1e-5
                if dev > 0.999 or gain < floor:
                    break
        B = np.array(betas)
        scale = np.where(self.sd > 0, self.sd, 1.0)
        coef = B * self.ys / scale
        coef[:, self.const] = 0.0
        if self.family == "gaussian":
            inter = self.ym - coef @ self.mean if self.intercept else np.zeros(len(B))
        else:
            inter = np.array(a0s) - coef @ self.mean
        return {
            "beta": coef,
            "a0": np.asarray(inter, dtype=float),
            "dev": np.array(devs),
        }

    def _irls(
        self, b: np.ndarray, a0: float, l1: np.ndarray, l2: np.ndarray
    ) -> Tuple[np.ndarray, float, bool]:
        """Penalised logistic regression by proximal Newton steps."""
        Z, y, n = self.Z, self.y, self.n
        ok = False
        for _ in range(100):
            eta = a0 + Z @ b
            prob = 1.0 / (1.0 + np.exp(-eta))
            w = prob * (1 - prob)
            lo, hi = prob < _PMIN, prob > 1 - _PMIN
            prob = np.where(lo, 0.0, np.where(hi, 1.0, prob))
            w = np.where(lo | hi, _PMIN, w)
            z = eta + (y - prob) / w
            sw = w.sum()
            if self.intercept:
                zbar = float(w @ z / sw)
                xbar = (Z * w[:, None]).sum(axis=0) / sw
            else:
                zbar, xbar = 0.0, np.zeros(self.p)
            Zc = Z - xbar
            gram = (Zc * w[:, None]).T @ Zc / n
            xty = (Zc * w[:, None]).T @ (z - zbar) / n
            new, inner = _cd_gaussian(
                gram, xty, b.copy(), l1, l2, self.tol * 1e-2, self.max_iter
            )
            new_a0 = zbar - float(xbar @ new) if self.intercept else 0.0
            step = float(np.max(np.abs(Z @ (new - b) + (new_a0 - a0))))
            b, a0 = new, new_a0
            if step < 1e-10 and inner:
                ok = True
                break
        return b, a0, ok


@dataclass
class GlmnetResult(ResultProtocolMixin):
    """Result of :func:`glmnet`.

    Attributes
    ----------
    params : pandas.Series
        Coefficients at the selected penalty, on the scale of the data.
    intercept : float
    penalty : float
        The selected ``lambda``.
    lambda_min, lambda_1se : float
        The penalty with the smallest cross-validated error, and the
        largest one within a standard error of it (``nan`` without
        cross-validation).
    path : pandas.DataFrame
        One row per penalty: ``lambda``, ``df`` (non-zero coefficients),
        ``dev_ratio`` (share of deviance explained).
    coefficients : pandas.DataFrame
        Coefficients along the path, one row per penalty, ``intercept``
        first.
    cv : pandas.DataFrame or None
        ``lambda``, ``cvm`` (mean cross-validated error), ``cvsd`` (its
        standard error), ``cvup``, ``cvlo``, ``nzero``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 8))
    >>> df = pd.DataFrame(X, columns=[f"x{j}" for j in range(8)])
    >>> df["y"] = 2 * X[:, 0] - X[:, 1] + rng.normal(size=200)
    >>> fit = sp.glmnet(df, "y", [f"x{j}" for j in range(8)], seed=0)
    >>> isinstance(fit, sp.GlmnetResult)
    True
    >>> bool(fit.lambda_1se >= fit.lambda_min)
    True
    >>> fit.predict(df.head(3)).shape
    (3,)
    """

    _citation_keys = ("friedman2010regularization", "zou2005regularization")

    params: pd.Series
    intercept: float
    penalty: float
    lambda_min: float
    lambda_1se: float
    path: pd.DataFrame
    coefficients: pd.DataFrame
    cv: Optional[pd.DataFrame]
    family: str
    alpha: float
    x: List[str]
    n_obs: int
    model_info: Dict[str, Any] = field(default_factory=dict)

    def _at(self, s: Union[None, str, float]) -> Tuple[float, np.ndarray]:
        if s is None:
            lam = self.penalty
        elif isinstance(s, str):
            key = s.replace(".", "_").lower()
            if key not in ("lambda_min", "lambda_1se"):
                raise MethodIncompatibility(
                    f"glmnet: s must be 'lambda.min', 'lambda.1se' or a "
                    f"penalty on the path, got {s!r}.",
                    recovery_hint="Use s='lambda.min'.",
                )
            lam = getattr(self, key)
            if not np.isfinite(lam):
                raise MethodIncompatibility(
                    f"glmnet: {s} needs cross-validation.",
                    recovery_hint="Fit with cv=True.",
                )
        else:
            lam = float(s)
        lams = self.path["lambda"].to_numpy()
        i = int(np.argmin(np.abs(np.log(lams) - np.log(lam))))
        if abs(lams[i] / lam - 1) > 1e-8:
            raise MethodIncompatibility(
                f"glmnet: {lam!r} is not a penalty on the fitted path.",
                recovery_hint="Refit with lambda_=[...] holding the penalty "
                "you want; coefficients are not interpolated.",
            )
        row = self.coefficients.iloc[i].to_numpy()
        return float(row[0]), row[1:]

    def coef(self, s: Union[None, str, float] = None) -> pd.Series:
        """Coefficients (intercept first) at ``s``: ``'lambda.min'``,
        ``'lambda.1se'``, a penalty on the path, or the selected one."""
        a0, b = self._at(s)
        return pd.Series(np.r_[a0, b], index=["intercept", *self.x])

    def predict(
        self,
        data: pd.DataFrame,
        s: Union[None, str, float] = None,
        type: str = "response",
    ) -> np.ndarray:
        """Predictions at ``s``. ``type='link'`` is the linear predictor;
        ``'response'`` is the mean (a probability for a binomial fit)."""
        a0, b = self._at(s)
        eta = a0 + data[self.x].to_numpy(dtype=float) @ b
        if type == "link" or self.family == "gaussian":
            return np.asarray(eta, dtype=float)
        if type != "response":
            raise MethodIncompatibility(
                "glmnet: type must be 'response' or 'link'.",
                recovery_hint="Use type='response'.",
            )
        return np.asarray(1.0 / (1.0 + np.exp(-eta)), dtype=float)

    @property
    def n_nonzero(self) -> int:
        return int(np.sum(self.params.to_numpy() != 0))

    def summary(self) -> str:
        kind = {1.0: "lasso", 0.0: "ridge"}.get(self.alpha, "elastic net")
        lines = [
            f"glmnet: {kind} (alpha = {self.alpha:g}), {self.family}",
            "-" * 52,
            f"Observations      : {self.n_obs}",
            f"Predictors        : {len(self.x)}",
            f"Penalties on path : {len(self.path)}",
        ]
        if self.cv is not None:
            lines += [
                f"lambda.min        : {self.lambda_min:.6g}",
                f"lambda.1se        : {self.lambda_1se:.6g}",
                f"CV error at min   : {float(self.cv['cvm'].min()):.6g}",
            ]
        lines += [
            f"Selected lambda   : {self.penalty:.6g}"
            f"  ({self.model_info.get('rule', '')})",
            f"Non-zero          : {self.n_nonzero}",
            "",
            self.params[self.params != 0].round(6).to_string(),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def glmnet(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    *,
    alpha: float = 1.0,
    lambda_: Union[None, float, Sequence[float]] = None,
    family: str = "gaussian",
    penalty_factor: Optional[Sequence[float]] = None,
    cv: bool = True,
    n_folds: int = 10,
    foldid: Union[None, str, Sequence[int]] = None,
    rule: str = "min",
    n_lambda: int = 100,
    lambda_min_ratio: Optional[float] = None,
    standardize: bool = True,
    intercept: bool = True,
    seed: Optional[int] = None,
    tol: float = 1e-18,
    max_iter: int = 100_000,
) -> GlmnetResult:
    """Lasso, ridge and elastic net with the conventions of R ``glmnet``.

    Minimises, over the intercept and the coefficients,

    ``-loglik / n + lambda * sum_j v_j (alpha |b_j| + (1 - alpha) b_j^2 / 2)``

    (for a Gaussian outcome ``-loglik / n`` is ``RSS / (2 n)``) along a
    decreasing path of ``lambda``, and chooses ``lambda`` by K-fold
    cross-validation.

    Parameters
    ----------
    data : pandas.DataFrame
        Rows with a missing outcome or predictor are dropped.
    y : str
        Outcome. For ``family='binomial'`` it must be coded 0/1.
    x : sequence of str
        Predictors, in their original units.
    alpha : float, default 1.0
        Mixing weight: 1 is the lasso, 0 the ridge.
    lambda_ : float or sequence of float, optional
        Penalties to fit. ``None`` builds ``glmnet``'s path: ``n_lambda``
        values, log-spaced from the smallest penalty that sets every
        penalised coefficient to zero down to ``lambda_min_ratio`` of it,
        stopped early once the deviance explained no longer improves.
    family : {'gaussian', 'binomial'}
    penalty_factor : sequence of float, optional
        Multiplier ``v_j`` of the penalty of each predictor. 0 leaves a
        predictor unpenalised; weights ``1 / |b_j|`` from a first-stage fit
        give the adaptive lasso. Rescaled to sum to the number of
        predictors, as in ``glmnet``.
    cv : bool, default True
        Cross-validate. With ``False`` the last penalty of the path is
        selected and ``lambda_min`` / ``lambda_1se`` are ``nan``.
    n_folds : int, default 10
    foldid : sequence of int or str, optional
        Fold label of each row (or a column holding it). Without it rows
        are assigned at random with ``seed``.
    rule : {'min', '1se'}, default 'min'
        Which penalty ``params``, ``coef()`` and ``predict()`` use by
        default.
    n_lambda : int, default 100
    lambda_min_ratio : float, optional
        Default ``1e-4`` with more rows than predictors, else ``0.01``.
    standardize : bool, default True
        Scale predictors to unit standard deviation (divisor ``n``) before
        penalising. Coefficients are always returned in original units.
    intercept : bool, default True
    seed : int, optional
    tol : float, default 1e-18
        Convergence threshold of coordinate descent, on the largest
        squared coefficient change (standardised units). Much tighter than
        ``glmnet``'s default ``thresh = 1e-7``, so that the fit is the
        minimiser to about nine digits. ``glmnet`` at its default is up
        to 6e-4 from that in the coefficients of the test designs; at
        ``thresh = 1e-14`` it agrees to 1e-6.
    max_iter : int, default 100000

    Returns
    -------
    GlmnetResult

    Notes
    -----
    Conventions shared with ``glmnet``:

    * Predictors are scaled by the standard deviation with divisor ``n``.
    * A Gaussian outcome is scaled to unit variance before the penalty is
      applied and ``lambda`` is reported on the original scale. The lasso
      part does not notice; the ridge part does, so in original units the
      ridge term is ``lambda / sd(y) * (1 - alpha) / 2 * b^2``. This is why
      ``glmnet(alpha = 0)`` is not the textbook ridge at the same
      ``lambda``.
    * The first point of a path built here is the fit with an infinite
      penalty, and the path stops early once the deviance explained no
      longer improves. For the ridge, where no finite penalty sets the
      coefficients to zero, the first point is therefore the null model
      (coefficients of order 1e-36, all counted in ``df``) although it
      carries a finite label.
    * Cross-validation refits the whole path, standardisation included,
      on each training set. With a path built here every training set
      builds its own and is read at the full-sample penalties by linear
      interpolation in ``lambda``; with ``lambda_`` supplied every
      training set is fitted at exactly those penalties. The
      error is the mean squared error (Gaussian) or the binomial deviance
      with probabilities held in ``[1e-5, 1 - 1e-5]``, averaged within
      fold; ``cvsd`` is the standard error of the fold means, weighted by
      fold size.

    :func:`statspai.shrinkage` is the other penalised regression in the
    package: ridge, lasso and principal components with the penalty stated
    on the residual sum of squares, for comparing predictors by
    cross-validated error. Use this function when the penalty has to mean
    what it means in ``glmnet``, or for the elastic net, a binomial
    outcome, penalty factors, or ``lambda.1se``.

    Inference after selection is a different problem: see
    :func:`statspai.rlasso_effect` and :func:`statspai.dml`.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 8))
    >>> cols = [f"x{j}" for j in range(8)]
    >>> df = pd.DataFrame(X, columns=cols)
    >>> df["y"] = 2 * X[:, 0] - X[:, 1] + rng.normal(size=200)
    >>> fit = sp.glmnet(df, "y", cols, alpha=0.5, seed=0)
    >>> bool(fit.params["x0"] > 1.5)
    True
    >>> sparse = fit.coef("lambda.1se")
    >>> bool((sparse != 0).sum() <= (fit.coef("lambda.min") != 0).sum())
    True

    References
    ----------
    [@friedman2010regularization],
    [@zou2005regularization],
    [@zou2006adaptive]
    """
    if family not in ("gaussian", "binomial"):
        raise MethodIncompatibility(
            f"glmnet: family must be 'gaussian' or 'binomial', got {family!r}.",
            recovery_hint="Other families are not implemented.",
        )
    if not 0.0 <= float(alpha) <= 1.0:
        raise MethodIncompatibility("glmnet: alpha must lie in [0, 1].")
    if rule not in ("min", "1se"):
        raise MethodIncompatibility(
            f"glmnet: rule must be 'min' or '1se', got {rule!r}.",
            recovery_hint="Use rule='min'.",
        )
    x = [x] if isinstance(x, str) else list(x)
    if not x:
        raise MethodIncompatibility("glmnet: at least one predictor is required.")
    cols = [y, *x] + ([foldid] if isinstance(foldid, str) else [])
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"glmnet: column(s) not in data: {missing}.",
            diagnostics={"missing_columns": missing},
        )
    keep = data[[y, *x]].notna().all(axis=1).to_numpy()
    df = data.loc[keep]
    try:
        X = df[x].to_numpy(dtype=float)
        yv = df[y].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "glmnet: the outcome and predictors must be numeric.",
            recovery_hint="Encode categorical predictors as indicators.",
        ) from exc
    n, p = X.shape
    if n < 3:
        raise DataInsufficient("glmnet: fewer than three complete rows.")
    if family == "binomial" and not np.isin(np.unique(yv), (0.0, 1.0)).all():
        raise MethodIncompatibility(
            "glmnet: a binomial outcome must be coded 0/1.",
            recovery_hint="Recode the outcome.",
        )
    if family == "binomial" and len(np.unique(yv)) < 2:
        raise DataInsufficient("glmnet: the binomial outcome has one class.")
    if penalty_factor is None:
        vp = np.ones(p)
    else:
        vp = np.asarray(penalty_factor, dtype=float).ravel()
        if len(vp) != p or (vp < 0).any() or not np.isfinite(vp).all():
            raise MethodIncompatibility(
                f"glmnet: penalty_factor needs {p} finite non-negative values.",
                recovery_hint="One multiplier per predictor, in the order of x.",
            )
        if vp.sum() <= 0:
            raise MethodIncompatibility(
                "glmnet: every penalty factor is zero; that is an "
                "unpenalised regression.",
                recovery_hint="Use sp.regress or sp.logit.",
            )
        vp = vp * p / vp.sum()

    def path_of(Xm: np.ndarray, ym: np.ndarray) -> _Path:
        return _Path(
            Xm, ym, family, float(alpha), vp, standardize, intercept, tol, max_iter
        )

    full = path_of(X, yv)
    user_path = lambda_ is not None
    if user_path:
        lambdas = np.sort(np.atleast_1d(np.asarray(lambda_, dtype=float)))[::-1]
        if (lambdas < 0).any() or not np.isfinite(lambdas).all():
            raise MethodIncompatibility("glmnet: lambda_ must be non-negative.")
    else:
        ratio = (
            float(lambda_min_ratio)
            if lambda_min_ratio is not None
            else (1e-4 if n > p else 1e-2)
        )
        lambdas = full.lambda_max * np.exp(
            np.linspace(0.0, np.log(ratio), int(n_lambda))
        )
    fit = full.fit(lambdas, auto=not user_path)
    k = len(fit["dev"])
    lambdas = lambdas[:k]
    converged = full.converged

    cv_table: Optional[pd.DataFrame] = None
    lam_min = lam_1se = float("nan")
    if cv and k > 1:
        if foldid is None:
            rng = np.random.default_rng(seed)
            folds = rng.permutation(np.arange(n) % int(n_folds))
        else:
            raw = (
                data.loc[keep, foldid].to_numpy()
                if isinstance(foldid, str)
                else np.asarray(foldid)
            )
            if not isinstance(foldid, str) and len(raw) == len(data):
                raw = raw[keep]
            if len(raw) != n:
                raise MethodIncompatibility(
                    f"glmnet: foldid needs one label per row ({n}).",
                    recovery_hint="Pass a column name or an aligned vector.",
                )
            folds = pd.factorize(raw, sort=True)[0]
        K = int(folds.max()) + 1
        if K < 3:
            raise MethodIncompatibility(
                "glmnet: cross-validation needs at least three folds.",
                recovery_hint="Use n_folds >= 3.",
            )
        fold_err = np.full((K, k), np.nan)
        sizes = np.bincount(folds, minlength=K).astype(float)
        for f in range(K):
            test = folds == f
            if family == "binomial" and len(np.unique(yv[~test])) < 2:
                raise DataInsufficient(
                    "glmnet: a training fold holds one class only.",
                    recovery_hint="Use fewer folds or stratified fold labels.",
                )
            sub = path_of(X[~test], yv[~test])
            if user_path:
                out = sub.fit(lambdas, auto=False)
                a0_f, beta_f = out["a0"], out["beta"]
            else:
                # cv.glmnet lets each training set build its own path and
                # reads it at the full-sample penalties: linear
                # interpolation in lambda between the two neighbouring
                # points of the fold's path, held flat beyond its ends.
                own = sub.lambda_max * np.exp(
                    np.linspace(0.0, np.log(ratio), int(n_lambda))
                )
                out = sub.fit(own, auto=True)
                grid = own[: len(out["dev"])][::-1]  # increasing
                a0_f = np.interp(lambdas, grid, out["a0"][::-1])
                beta_f = np.column_stack(
                    [np.interp(lambdas, grid, out["beta"][::-1, j]) for j in range(p)]
                )
            converged = converged and sub.converged
            eta = a0_f[:, None] + beta_f @ X[test].T  # k x n_test
            if family == "gaussian":
                fold_err[f] = np.mean((yv[test][None, :] - eta) ** 2, axis=1)
            else:
                prob = np.clip(1.0 / (1.0 + np.exp(-eta)), _PMIN, 1 - _PMIN)
                yt = yv[test][None, :]
                fold_err[f] = -2.0 * np.mean(
                    yt * np.log(prob) + (1 - yt) * np.log(1 - prob), axis=1
                )
        wts = sizes / sizes.sum()
        cvm = wts @ fold_err
        cvsd = np.sqrt(wts @ (fold_err - cvm) ** 2 / (K - 1))
        i_min = int(np.argmin(cvm))
        lam_min = float(lambdas[i_min])
        within = cvm <= cvm[i_min] + cvsd[i_min]
        lam_1se = float(lambdas[within].max())
        cv_table = pd.DataFrame(
            {
                "lambda": lambdas,
                "cvm": cvm,
                "cvsd": cvsd,
                "cvup": cvm + cvsd,
                "cvlo": cvm - cvsd,
                "nzero": (fit["beta"] != 0).sum(axis=1),
            }
        )
    if not converged:
        warnings.warn(
            "glmnet: coordinate descent did not converge at some penalty; "
            "raise max_iter or drop the smallest penalties.",
            ConvergenceWarning,
            stacklevel=2,
        )
    if cv_table is None:
        chosen, how = float(lambdas[-1]), "last penalty of the path"
    elif rule == "min":
        chosen, how = lam_min, "lambda.min"
    else:
        chosen, how = lam_1se, "lambda.1se"
    i = int(np.argmin(np.abs(lambdas - chosen)))
    coefs = pd.DataFrame(
        np.column_stack([fit["a0"], fit["beta"]]), columns=["intercept", *x]
    )
    path = pd.DataFrame(
        {
            "lambda": lambdas,
            "df": (fit["beta"] != 0).sum(axis=1),
            "dev_ratio": fit["dev"],
        }
    )
    return GlmnetResult(
        params=pd.Series(fit["beta"][i], index=x),
        intercept=float(fit["a0"][i]),
        penalty=chosen,
        lambda_min=lam_min,
        lambda_1se=lam_1se,
        path=path,
        coefficients=coefs,
        cv=cv_table,
        family=family,
        alpha=float(alpha),
        x=list(x),
        n_obs=int(n),
        model_info={
            "rule": how,
            "n_folds": None if cv_table is None else int(folds.max()) + 1,
            "penalty_factor": vp,
            "standardize": bool(standardize),
            "intercept": bool(intercept),
            "lambda_max": float(full.lambda_max),
            "converged": bool(converged),
            "null_deviance": float(full.null_dev),
        },
    )
