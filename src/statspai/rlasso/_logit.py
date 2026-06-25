"""Logistic rigorous Lasso — a faithful port of R ``hdm::rlassologit``.

``hdm::rlassologit`` is the binary-outcome analogue of :func:`rlasso`: a
*logistic* rigorous (post-)Lasso whose penalized fit is glmnet's binomial
lasso at a single, data-driven penalty level
``λ₀ = (c/2)·√n·Φ⁻¹(1 − γ/(2p))`` (``λ = λ₀/(2n)``, ``c = 1.1`` for
post-Lasso, ``γ = 0.1/log n``).

Because hdm delegates the penalized fit to ``glmnet(family="binomial",
alpha=1, lambda=λ, standardize=TRUE)``, a faithful port must reproduce
glmnet's binomial lasso at that single ``λ``.  :func:`_glmnet_logit_lasso`
implements glmnet's algorithm directly — IRLS outer loop, weighted
coordinate-descent inner loop, ``1/n``-scaled deviance objective,
population-variance standardization and the ``pmin`` probability clamp —
and agrees with R ``glmnet`` 4.1 to ~1e-6 on the coefficients **with the
selected support matching exactly**.  Exact support is what matters: for
the default ``post=True`` the final coefficients come from an
*unpenalized* logistic refit on the selected set (:func:`_glm_logit`),
which matches R's ``glm`` to ~1e-8 whenever the support matches.

Numerical contract (vs ``hdm`` 0.3.2 / ``glmnet`` 4.1, see
``tests/reference_parity/test_rlassologit_parity.py``): selected support
**exact**; post-Lasso coefficients / intercept / residuals to ~1e-6;
plain-Lasso (``post=False``) coefficients to ~1e-6 (glmnet's own
convergence tolerance).

The underlying logistic-lasso solver follows Friedman, Hastie &
Tibshirani's coordinate-descent algorithm [@friedman2010regularization]
(the engine behind R's ``glmnet``); the post-selection theory is that of
Belloni, Chernozhukov & Wei on inference for generalized linear models
with many controls [@belloni2016post].

References
----------
Belloni, A., Chernozhukov, V. and Wei, Y. (2016). "Post-Selection
    Inference for Generalized Linear Models With Many Controls."
    *Journal of Business & Economic Statistics*, 34(4), 606-619.
    [@belloni2016post]

Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
    High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
    [@chernozhukov2016hdm]

Friedman, J., Hastie, T. and Tibshirani, R. (2010). "Regularization
    Paths for Generalized Linear Models via Coordinate Descent."
    *Journal of Statistical Software*, 33(1), 1-22.
    [@friedman2010regularization]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .._result_serialize import ResultProtocolMixin

# ═══════════════════════════════════════════════════════════════════════
#  glmnet binomial lasso at a single lambda
# ═══════════════════════════════════════════════════════════════════════


def _glmnet_logit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    lam: float,
    intercept: bool = True,
    standardize: bool = True,
    max_outer: int = 200,
    max_inner: int = 200,
    thresh: float = 1e-10,
    pmin: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """glmnet's binomial lasso at a single ``λ`` — returns ``(beta, a0)``.

    Reproduces ``glmnet(x, y, family="binomial", alpha=1, lambda=lam,
    standardize=standardize, intercept=intercept)`` on the *original*
    coefficient scale.  Minimises
    ``(1/n) Σ_i [−y_i η_i + log(1+e^{η_i})] + λ Σ_j |β_j|`` via IRLS +
    weighted coordinate descent on the (population-variance) standardized
    design.
    """
    n, p = X.shape
    xbar = X.mean(axis=0)
    if standardize:
        xs = np.sqrt(((X - xbar) ** 2).mean(axis=0))
        xs = np.where(xs > 0, xs, 1.0)
    else:
        xs = np.ones(p)
    Xs = (X - xbar) / xs

    beta = np.zeros(p)
    ybar = float(y.mean())
    ybar = min(max(ybar, pmin), 1.0 - pmin)
    b0 = float(np.log(ybar / (1.0 - ybar))) if intercept else 0.0
    eta = b0 + Xs @ beta

    for _ in range(max_outer):
        pvec = 1.0 / (1.0 + np.exp(-eta))
        pvec = np.clip(pvec, pmin, 1.0 - pmin)
        v = pvec * (1.0 - pvec)
        v = np.maximum(v, pmin)
        z = eta + (y - pvec) / v
        w = v / n  # observation weights normalised to 1/n

        beta_outer = beta.copy()
        b0_outer = b0
        for _ in range(max_inner):
            r = z - (b0 + Xs @ beta)
            max_delta = 0.0
            for j in range(p):
                rj = r + Xs[:, j] * beta[j]
                num = float(np.sum(w * Xs[:, j] * rj))
                den = float(np.sum(w * Xs[:, j] ** 2))
                if den <= 0:
                    new_b = 0.0
                else:
                    new_b = np.sign(num) * max(abs(num) - lam, 0.0) / den
                d = new_b - beta[j]
                if d != 0.0:
                    r = rj - Xs[:, j] * new_b
                    beta[j] = new_b
                    max_delta = max(max_delta, abs(d))
            if intercept:
                b0 = float(np.sum(w * (z - Xs @ beta)) / np.sum(w))
            if max_delta < thresh:
                break
        eta = b0 + Xs @ beta
        if np.max(np.abs(beta - beta_outer)) < thresh and abs(b0 - b0_outer) < thresh:
            break

    beta_orig = beta / xs
    a0 = float(b0 - np.sum(xbar * beta_orig)) if intercept else 0.0
    return beta_orig, a0


# ═══════════════════════════════════════════════════════════════════════
#  Unpenalized logistic regression (post-Lasso refit) — matches R's glm
# ═══════════════════════════════════════════════════════════════════════


def _glm_logit(
    X: np.ndarray,
    y: np.ndarray,
    intercept: bool = True,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Unpenalized logistic MLE by IRLS (Newton) — mirrors ``glm`` binomial.

    Returns ``(slopes, intercept, fitted)``.
    """
    n, p = X.shape
    A = np.column_stack([np.ones(n), X]) if intercept else X
    k = A.shape[1]
    coef = np.zeros(k)
    if intercept:
        ybar = float(np.clip(y.mean(), 1e-6, 1 - 1e-6))
        coef[0] = np.log(ybar / (1.0 - ybar))
    for _ in range(max_iter):
        eta = A @ coef
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        z = eta + (y - mu) / w
        WA = A * w[:, None]
        H = A.T @ WA
        g = A.T @ (w * z)
        try:
            new_coef = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:  # pragma: no cover - rank-deficient guard
            new_coef = np.linalg.lstsq(H, g, rcond=None)[0]
        if np.max(np.abs(new_coef - coef)) < tol:
            coef = new_coef
            break
        coef = new_coef
    fitted = 1.0 / (1.0 + np.exp(-(A @ coef)))
    if intercept:
        return coef[1:], float(coef[0]), fitted
    return coef, 0.0, fitted


# ═══════════════════════════════════════════════════════════════════════
#  Penalty level (hdm::rlassologit)
# ═══════════════════════════════════════════════════════════════════════


def _rlassologit_lambda(
    n: int, p: int, post: bool, penalty: Optional[Dict[str, Any]]
) -> Tuple[float, float]:
    """Return ``(lambda0, lambda)`` exactly as ``hdm::rlassologit`` does.

    Subtlety: hdm's *default* penalty list already carries ``c = 1.1``
    explicitly, so ``exists("c")`` is always true on a default call and
    the ``post=FALSE -> c = 0.5`` switch fires only when the user passes a
    ``penalty`` list that omits ``c``.  Hence ``penalty=None`` uses
    ``c = 1.1`` regardless of ``post`` (unlike ``rlasso``).
    """
    if penalty is None:
        c: float = 1.1
        gamma = 0.1 / np.log(n)
        lam_start = None
    else:
        user = dict(penalty)
        if "c" in user:
            c = user["c"]
        else:
            c = 1.1 if post else 0.5
        gamma = user.get("gamma", None)
        if gamma is None:
            gamma = 0.1 / np.log(n)
        lam_start = user.get("lambda", None)
    if lam_start is not None:
        lam = float(lam_start) / (2.0 * n)
        lambda0 = lam * (2.0 * n)
    else:
        lambda0 = c / 2.0 * np.sqrt(n) * stats.norm.ppf(1.0 - gamma / (2.0 * p))
        lam = lambda0 / (2.0 * n)
    return float(lambda0), float(lam)


# ═══════════════════════════════════════════════════════════════════════
#  Fit object
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RLassoLogitFit(ResultProtocolMixin):
    """Result of :func:`rlassologit` — mirrors ``hdm`` ``rlassologit``."""

    _citation_keys: ClassVar[Tuple[str, ...]] = ("chernozhukov2016hdm",)

    beta: np.ndarray  # slope coefficients (original scale), length p
    intercept: float
    index: np.ndarray  # bool, length p — selected support
    coefficients: np.ndarray  # [intercept, beta] if intercept else beta
    residuals: np.ndarray  # response residuals y - p̂
    sigma: float
    lambda0: float
    lambda_: float
    n: int
    p: int
    post: bool
    intercept_flag: bool
    colnames: List[str] = field(default_factory=list)

    @property
    def n_selected(self) -> int:
        return int(self.index.sum())

    @property
    def selected(self) -> List[str]:
        return [self.colnames[j] for j in np.where(self.index)[0]]

    def predict(self, newX: np.ndarray, type: str = "response") -> np.ndarray:
        """Predict on ``newX`` — ``type='response'`` (probabilities) or
        ``'link'`` (log-odds).  Mirrors ``hdm::predict.rlassologit``."""
        Xn = np.asarray(newX, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        eta = Xn @ self.beta + (self.intercept if self.intercept_flag else 0.0)
        if type == "link":
            out: np.ndarray = np.asarray(eta, dtype=float)
            return out
        if type == "response":
            resp: np.ndarray = 1.0 / (1.0 + np.exp(-eta))
            return resp
        raise ValueError(f"type must be 'response' or 'link', got {type!r}")

    def summary(self) -> str:
        head = [
            "Logistic Rigorous Lasso (hdm::rlassologit port)",
            "-" * 56,
            f"  n = {self.n}   p = {self.p}   post = {self.post}",
            f"  selected: {self.n_selected}   λ₀ = {self.lambda0:.4f}",
        ]
        if self.n_selected:
            head.append("  support:")
            for j in np.where(self.index)[0][:20]:
                head.append(f"    {self.colnames[j]:<16}{self.beta[j]:>12.5f}")
        else:
            head.append("  (no covariates selected)")
        return "\n".join(head)


# ═══════════════════════════════════════════════════════════════════════
#  rlassologit  (hdm::rlassologit.default)
# ═══════════════════════════════════════════════════════════════════════


def rlassologit(
    X: np.ndarray,
    y: np.ndarray,
    post: bool = True,
    intercept: bool = True,
    penalty: Optional[Dict[str, Any]] = None,
    control: Optional[Dict[str, Any]] = None,
    colnames: Optional[List[str]] = None,
) -> RLassoLogitFit:
    """Logistic rigorous (post-)Lasso — a faithful port of ``hdm::rlassologit``.

    Parameters
    ----------
    X : (n, p) array of candidate covariates.
    y : (n,) binary outcome in {0, 1}.
    post : bool, default True
        If ``True``, refit the selected support by *unpenalized* logistic
        regression (post-Lasso); else keep the glmnet-penalized fit.
    intercept : bool, default True
        Include an intercept.
    penalty : dict, optional
        Overrides for ``c`` (slack; default 1.1 for ``post=True``, else
        0.5), ``gamma`` (default ``0.1/log n``) and ``lambda`` (raw
        penalty; bypasses the data-driven level).
    control : dict, optional
        ``threshold`` — coefficients below it are zeroed (default None).
    colnames : list of str, optional
        Column names (default ``V1..Vp``).

    Returns
    -------
    RLassoLogitFit

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((300, 20))
    >>> p = 1 / (1 + np.exp(-(X[:, 0] - X[:, 1])))
    >>> y = (rng.uniform(size=300) < p).astype(float)
    >>> fit = sp.rlassologit(X, y)
    >>> fit.n_selected >= 1
    True
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape

    # Fail loudly on non-binary input (CLAUDE.md §7): a logistic model is
    # only defined for a 0/1 response.  Without this guard, continuous or
    # {1,2}-coded labels slide silently into the IRLS solver and produce
    # overflow garbage instead of an error.
    classes = np.unique(y[np.isfinite(y)])
    if not np.all(np.isin(classes, (0.0, 1.0))):
        shown = classes[:5].tolist()
        raise ValueError(
            "rlassologit requires a binary 0/1 response y; got "
            f"{classes.size} distinct value(s) {shown}"
            f"{'...' if classes.size > 5 else ''}. Encode the outcome as 0/1 "
            "(e.g. (y == positive_label).astype(float))."
        )
    if classes.size < 2:
        raise ValueError(
            "rlassologit needs both classes present in y; got a single class "
            f"({classes.tolist()}) — a logistic model is not identified."
        )

    if colnames is None:
        colnames = [f"V{j + 1}" for j in range(p)]
    ctrl = dict(control) if control else {}
    threshold = ctrl.get("threshold", None)

    lambda0, lam = _rlassologit_lambda(n, p, post, penalty)

    beta_glmnet, a0_glmnet = _glmnet_logit_lasso(X, y, lam, intercept=intercept)
    coefTemp = np.where(np.abs(beta_glmnet) > 0, beta_glmnet, 0.0)
    ind1 = np.abs(coefTemp) > 0

    if ind1.sum() == 0:
        # hdm early return: nothing selected.
        if intercept:
            ybar = float(np.clip(y.mean(), 1e-12, 1 - 1e-12))
            a0 = float(np.log(ybar / (1.0 - ybar)))
            res = y - y.mean()
        else:
            a0 = 0.0
            res = y - 0.5
        beta_full = np.zeros(p)
        coefs = np.concatenate([[a0], beta_full]) if intercept else beta_full
        return RLassoLogitFit(
            beta=beta_full,
            intercept=a0 if intercept else 0.0,
            index=np.zeros(p, dtype=bool),
            coefficients=coefs,
            residuals=res,
            sigma=float(np.sqrt(np.var(res, ddof=1))),
            lambda0=lambda0,
            lambda_=lam,
            n=n,
            p=p,
            post=post,
            intercept_flag=intercept,
            colnames=list(colnames),
        )

    if post:
        X1 = X[:, ind1]
        slopes, a0, fitted = _glm_logit(X1, y, intercept=intercept)
        e1 = y - fitted
        coefTemp[ind1] = slopes
    else:
        eta = a0_glmnet + X @ beta_glmnet if intercept else X @ beta_glmnet
        fitted = 1.0 / (1.0 + np.exp(-eta))
        e1 = y - fitted
        a0 = a0_glmnet if intercept else 0.0

    if threshold is not None:
        coefTemp[np.abs(coefTemp) < threshold] = 0.0

    coefs = np.concatenate([[a0], coefTemp]) if intercept else coefTemp.copy()
    return RLassoLogitFit(
        beta=coefTemp,
        intercept=float(a0) if intercept else 0.0,
        index=ind1,
        coefficients=coefs,
        residuals=e1,
        sigma=float(np.sqrt(np.var(e1, ddof=1))),
        lambda0=lambda0,
        lambda_=lam,
        n=n,
        p=p,
        post=post,
        intercept_flag=intercept,
        colnames=list(colnames),
    )
