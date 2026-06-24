"""Rigorous (data-driven) Lasso — a faithful port of R ``hdm::rlasso``.

This module is a *line-for-line* re-implementation of the ``rlasso``
estimator from the **hdm** package (Chernozhukov, Hansen & Spindler,
2016, *The R Journal*), itself the reference implementation of the
rigorous Lasso of Belloni, Chernozhukov & Hansen (2014) and the optimal
instrument selection of Belloni, Chen, Chernozhukov & Hansen (2012).

The defining features versus a vanilla ``sklearn`` Lasso are:

1. **Data-driven penalty level** ``λ₀ = 2·c·√n·Φ⁻¹(1 − γ/(2p))`` with
   ``c = 1.1`` and ``γ = 0.1/log(n)`` — an *a-priori*, theory-justified
   choice that needs no cross-validation and delivers near-oracle rates.
2. **Per-coefficient penalty loadings** ``Ψⱼ`` that absorb
   heteroskedasticity (BCH 2012, Algorithm 1), refined by iteration on
   the regression residuals.
3. **Post-Lasso**: re-estimate by OLS on the selected support to remove
   shrinkage bias.

Numerical contract
------------------
Every internal step mirrors ``hdm`` exactly so that ``rlasso`` returns
the **same** selected support, coefficients, penalty level, loadings and
residuals as ``hdm::rlasso`` (verified to ~1e-6 in
``tests/reference_parity/test_rlasso_parity.py``).  In particular:

- the design is **centered but not standardized** inside the Lasso (the
  loadings carry the scale);
- the coordinate-descent solver is hdm's ``LassoShooting.fit`` minimizing
  ``‖y − Xβ‖² + Σⱼ λⱼ|βⱼ|`` (note: *no* ``1/n`` and *no* ``1/2``);
- the loadings are seeded from the residuals of an OLS on the
  ``min(5, p)`` covariates most correlated with the response
  (``init_values``).

References
----------
Belloni, A., Chen, D., Chernozhukov, V. and Hansen, C. (2012). "Sparse
    Models and Methods for Optimal Instruments With an Application to
    Eminent Domain." *Econometrica*, 80(6), 2369-2429.
    [@belloni2012sparse]

Belloni, A., Chernozhukov, V. and Hansen, C. (2014). "Inference on
    Treatment Effects After Selection Among High-Dimensional Controls."
    *Review of Economic Studies*, 81(2), 608-650.
    [@belloni2014inference]

Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
    High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
    [@chernozhukov2016hdm]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .._result_serialize import ResultProtocolMixin

# ═══════════════════════════════════════════════════════════════════════
#  Penalty / control defaults (mirror hdm::rlasso.default)
# ═══════════════════════════════════════════════════════════════════════


def _default_penalty(
    n: int, post: bool, user: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve the ``penalty`` list exactly as ``hdm::rlasso.default`` does.

    hdm default is
    ``list(homoscedastic=FALSE, X.dependent.lambda=FALSE,
           lambda.start=NULL, c=1.1, gamma=0.1/log(n))``.
    When ``post`` is ``FALSE`` and the user did not set ``c`` (or passed
    the bare default), hdm switches to ``c = 0.5``.
    """
    user = dict(user) if user else {}
    pen: Dict[str, Any] = {
        "homoscedastic": user.get("homoscedastic", False),
        "X.dependent.lambda": user.get("X.dependent.lambda", False),
        "lambda.start": user.get("lambda.start", None),
        "c": user.get("c", 1.1),
        "gamma": user.get("gamma", 0.1 / np.log(n)),
    }
    if "numSim" in user:
        pen["numSim"] = user["numSim"]
    # hdm: post == FALSE & !exists("c", penalty)  ->  c = 0.5
    if not post and "c" not in user:
        pen["c"] = 0.5
    return pen


def _default_control(user: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    user = dict(user) if user else {}
    return {
        "numIter": user.get("numIter", 15),
        "tol": user.get("tol", 1e-5),
        "threshold": user.get("threshold", None),
    }


# ═══════════════════════════════════════════════════════════════════════
#  init_values  (hdm::init_values)
# ═══════════════════════════════════════════════════════════════════════


def _init_values(
    X: np.ndarray, y: np.ndarray, number: int = 5, intercept: bool = True
) -> Dict[str, np.ndarray]:
    """OLS on the ``number`` covariates most correlated with ``y``.

    Faithful to ``hdm::init_values``: returns the residuals (used to seed
    the penalty loadings) and a length-``p`` coefficient vector (used as
    the warm start for the coordinate-descent solver).
    """
    n, kx = X.shape
    yc = y - y.mean()
    Xc = X - X.mean(axis=0)
    denom_y = float(np.sqrt(yc @ yc))
    col = np.sqrt((Xc**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs((yc @ Xc) / (denom_y * col))
    # R: order(corr, decreasing = TRUE) — NA sorted last, ties stable.
    key = np.where(np.isnan(corr), -np.inf, corr)
    order = np.argsort(-key, kind="stable")
    index = order[: min(number, kx)]

    coefficients = np.zeros(kx)
    Xsub = X[:, index]
    if intercept:
        A = np.column_stack([np.ones(n), Xsub])
    else:
        A = Xsub
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    slopes = sol[1:] if intercept else sol
    coefficients[index] = slopes
    coefficients[np.isnan(coefficients)] = 0.0
    residuals = y - A @ sol
    return {"residuals": residuals, "coefficients": coefficients, "index": index}


# ═══════════════════════════════════════════════════════════════════════
#  LassoShooting.fit  (hdm coordinate descent)
# ═══════════════════════════════════════════════════════════════════════


def _lasso_shooting(
    X: np.ndarray,
    y: np.ndarray,
    lam: np.ndarray,
    XX: np.ndarray,
    Xy: np.ndarray,
    beta_start: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    opt_tol: float = 1e-5,
    zero_threshold: float = 1e-6,
) -> np.ndarray:
    """Coordinate-descent solver, mirroring ``hdm::LassoShooting.fit``.

    Minimizes ``‖y − Xβ‖² + Σⱼ λⱼ|βⱼ|`` (per-coordinate penalty ``λⱼ``).
    """
    p = X.shape[1]
    beta: np.ndarray
    if beta_start is None:
        beta = _init_values(X, y, intercept=False)["coefficients"].astype(float).copy()
    else:
        beta = beta_start.astype(float).copy()
    XX2 = 2.0 * XX
    Xy2 = 2.0 * np.asarray(Xy, dtype=float).ravel()
    lam = np.asarray(lam, dtype=float).ravel()

    m = 1
    while m < max_iter:
        beta_old: np.ndarray = beta.copy()
        for j in range(p):
            S0 = float(XX2[j, :] @ beta - XX2[j, j] * beta[j] - Xy2[j])
            if np.isnan(S0):
                beta[j] = 0.0
                continue
            if S0 > lam[j]:
                beta[j] = (lam[j] - S0) / XX2[j, j]
            elif S0 < -lam[j]:
                beta[j] = (-lam[j] - S0) / XX2[j, j]
            else:
                beta[j] = 0.0
        if np.abs(beta - beta_old).sum() < opt_tol:
            break
        m += 1

    beta[np.abs(beta) < zero_threshold] = 0.0
    return beta


# ═══════════════════════════════════════════════════════════════════════
#  lambdaCalculation  (hdm::lambdaCalculation)
# ═══════════════════════════════════════════════════════════════════════


def _lambda_calculation(
    penalty: Dict[str, Any],
    y: np.ndarray,
    X: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Data-driven penalty level and loadings (``hdm::lambdaCalculation``).

    Implements all four (homoscedastic × X.dependent) branches plus the
    ``"none"`` (user ``lambda.start``) branch.  The two X-dependent
    branches use Monte-Carlo simulation; they match hdm *in distribution*
    only (R's Mersenne-Twister stream is not reproduced).
    """
    n, p = X.shape
    homo = penalty.get("homoscedastic", False)
    xdep = penalty.get("X.dependent.lambda", False)
    c = penalty.get("c", 1.1)
    gamma = penalty.get("gamma", 0.1)
    y = np.asarray(y, dtype=float).ravel()

    if homo is True and xdep is False:
        lambda0 = 2.0 * c * np.sqrt(n) * stats.norm.ppf(1.0 - gamma / (2.0 * p))
        Ups0 = float(np.sqrt(np.var(y, ddof=1)))
        lam = np.full(p, lambda0 * Ups0)
    elif homo is True and xdep is True:
        num_sim = penalty.get("numSim", 5000)
        if rng is None:
            rng = np.random.default_rng()
        psi = (X**2).mean(axis=0)
        tX = X / np.sqrt(psi)
        sim = np.empty(num_sim)
        for ell in range(num_sim):
            g = rng.standard_normal(n)
            sim[ell] = n * np.max(2.0 * np.abs((tX * g[:, None]).mean(axis=0)))
        lambda0 = c * np.quantile(sim, 1.0 - gamma)
        Ups0 = float(np.sqrt(np.var(y, ddof=1)))
        lam = np.full(p, lambda0 * Ups0)
    elif homo is False and xdep is False:
        lambda0 = 2.0 * c * np.sqrt(n) * stats.norm.ppf(1.0 - gamma / (2.0 * p))
        Ups0 = (1.0 / np.sqrt(n)) * np.sqrt((y**2) @ (X**2))
        lam = lambda0 * Ups0
    elif homo is False and xdep is True:
        num_sim = penalty.get("numSim", 5000)
        if rng is None:
            rng = np.random.default_rng()
        xehat = X * y[:, None]
        psi = (xehat**2).mean(axis=0)
        tXe = xehat / np.sqrt(psi)
        sim = np.empty(num_sim)
        for ell in range(num_sim):
            g = rng.standard_normal(n)
            sim[ell] = n * np.max(2.0 * np.abs((tXe * g[:, None]).mean(axis=0)))
        lambda0 = c * np.quantile(sim, 1.0 - gamma)
        Ups0 = (1.0 / np.sqrt(n)) * np.sqrt((y**2) @ (X**2))
        lam = lambda0 * Ups0
    elif homo == "none":
        lstart = penalty.get("lambda.start", None)
        if lstart is None:
            raise ValueError('For method "none" lambda.start must be provided')
        lambda0 = float(lstart)
        Ups0 = (1.0 / np.sqrt(n)) * np.sqrt((y**2) @ (X**2))
        lam = lambda0 * Ups0
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported penalty combination: {penalty!r}")

    lstart = penalty.get("lambda.start", None)
    if lstart is not None and homo != "none":
        lam = (
            np.full(p, float(lstart))
            if np.ndim(lstart) == 0
            else np.asarray(lstart, float)
        )

    return {"lambda0": lambda0, "lambda": np.asarray(lam, float), "Ups0": Ups0}


def _update_loadings(
    penalty: Dict[str, Any],
    e1: np.ndarray,
    X: np.ndarray,
    Psi: np.ndarray,
    lambda0: float,
    n: int,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """In-loop loading/penalty refresh (the per-branch updates in rlasso)."""
    homo = penalty.get("homoscedastic", False)
    xdep = penalty.get("X.dependent.lambda", False)
    e1 = np.asarray(e1, dtype=float).ravel()
    s1 = float(np.sqrt(np.var(e1, ddof=1)))

    if homo is True:
        Ups1 = s1 * Psi
        lam = lambda0 * Ups1
    elif homo is False and xdep is False:
        Ups1 = (1.0 / np.sqrt(n)) * np.sqrt((e1**2) @ (X**2))
        lam = lambda0 * Ups1
    elif homo is False and xdep is True:
        lc = _lambda_calculation(penalty, y=e1, X=X, rng=rng)
        Ups1 = lc["Ups0"]
        lam = lc["lambda"]
    elif homo == "none":
        Ups1 = (1.0 / np.sqrt(n)) * np.sqrt((e1**2) @ (X**2))
        lam = lambda0 * Ups1
    else:  # pragma: no cover
        Ups1 = (1.0 / np.sqrt(n)) * np.sqrt((e1**2) @ (X**2))
        lam = lambda0 * Ups1
    return np.asarray(Ups1, float), np.asarray(lam, float), s1


# ═══════════════════════════════════════════════════════════════════════
#  Fit object
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class RLassoFit(ResultProtocolMixin):
    """Result of :func:`rlasso` — mirrors the fields of an ``hdm`` ``rlasso``."""

    #: Verified paper.bib keys (CLAUDE.md §10).
    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "belloni2014inference",
        "chernozhukov2016hdm",
    )

    beta: np.ndarray  # slope coefficients on the (centered) columns, length p
    intercept: float
    index: np.ndarray  # bool, length p — selected support
    coefficients: np.ndarray  # [intercept, beta] if intercept else beta
    residuals: np.ndarray
    sigma: float
    loadings: np.ndarray
    lambda0: float
    lambda_: np.ndarray
    iter: int
    n: int
    p: int
    post: bool
    intercept_flag: bool
    meanx: np.ndarray
    mu: float
    colnames: List[str] = field(default_factory=list)

    @property
    def n_selected(self) -> int:
        return int(self.index.sum())

    @property
    def selected(self) -> List[str]:
        return [self.colnames[j] for j in np.where(self.index)[0]]

    def predict(self, newX: np.ndarray) -> np.ndarray:
        """Predict on the original (uncentered) scale — ``hdm::predict.rlasso``."""
        Xn = np.asarray(newX, dtype=float)
        if Xn.ndim == 1:
            Xn = Xn.reshape(-1, 1)
        if self.intercept_flag:
            out: np.ndarray = Xn @ self.beta + self.intercept
        else:
            out = Xn @ self.beta
        return out

    def summary(self) -> str:
        head = [
            "Rigorous Lasso (hdm::rlasso port)",
            "-" * 56,
            f"  n = {self.n}   p = {self.p}   post = {self.post}",
            f"  selected: {self.n_selected}   "
            f"λ₀ = {self.lambda0:.4f}   σ̂ = {self.sigma:.4f}   "
            f"iter = {self.iter}",
        ]
        if self.n_selected:
            sel = np.where(self.index)[0]
            head.append("  support:")
            for j in sel[:20]:
                head.append(f"    {self.colnames[j]:<16}{self.beta[j]:>12.5f}")
            if self.n_selected > 20:
                head.append(f"    ... (+{self.n_selected - 20} more)")
        else:
            head.append("  (no covariates selected)")
        return "\n".join(head)


# ═══════════════════════════════════════════════════════════════════════
#  rlasso  (hdm::rlasso.default)
# ═══════════════════════════════════════════════════════════════════════


def rlasso(
    X: np.ndarray,
    y: np.ndarray,
    post: bool = True,
    intercept: bool = True,
    penalty: Optional[Dict[str, Any]] = None,
    control: Optional[Dict[str, Any]] = None,
    colnames: Optional[List[str]] = None,
    rng: Optional[np.random.Generator] = None,
) -> RLassoFit:
    """Rigorous Lasso / post-Lasso — a faithful port of ``hdm::rlasso``.

    Parameters
    ----------
    X : (n, p) array
        Design matrix of candidate covariates (``p`` may exceed ``n``).
    y : (n,) array
        Response.
    post : bool, default True
        If ``True``, re-estimate the selected support by OLS (post-Lasso).
    intercept : bool, default True
        Center ``X`` and ``y`` and report an intercept on the original
        scale.
    penalty : dict, optional
        Overrides for ``homoscedastic`` (``True`` / ``False`` / ``"none"``),
        ``X.dependent.lambda`` (bool), ``c`` (slack, default 1.1),
        ``gamma`` (default ``0.1/log(n)``), ``lambda.start`` and
        ``numSim``.  Defaults reproduce hdm exactly.
    control : dict, optional
        Overrides for ``numIter`` (default 15), ``tol`` (default 1e-5)
        and ``threshold`` (default ``None``).
    colnames : list of str, optional
        Names for the columns of ``X`` (default ``V1..Vp``).
    rng : numpy Generator, optional
        Only used when ``X.dependent.lambda`` simulation is requested.

    Returns
    -------
    RLassoFit

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 20))
    >>> beta = np.zeros(20); beta[:3] = [1.0, -1.0, 0.5]
    >>> y = X @ beta + 0.5 * rng.standard_normal(100)
    >>> fit = sp.rlasso(X, y, post=True)  # rigorous post-Lasso
    >>> int(fit.index.sum()) >= 1  # at least one control kept
    True
    >>> fit.beta.shape
    (20,)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, dtype=float).ravel()
    n, p = X.shape
    if colnames is None:
        colnames = [f"V{j + 1}" for j in range(p)]

    pen = _default_penalty(n, post, penalty)
    ctrl = _default_control(control)

    if intercept:
        meanx = X.mean(axis=0)
        Xc = X - meanx
        mu = float(y.mean())
        yc = y - mu
    else:
        meanx = np.zeros(p)
        Xc = X
        mu = 0.0
        yc = y

    Psi = (Xc**2).mean(axis=0)
    XX = Xc.T @ Xc
    Xy = Xc.T @ yc

    startingval = _init_values(Xc, yc)["residuals"]
    pencalc = _lambda_calculation(pen, y=startingval, X=Xc, rng=rng)
    lam = pencalc["lambda"].copy()
    lambda0 = pencalc["lambda0"]
    Ups1 = np.atleast_1d(pencalc["Ups0"]).astype(float)

    s0 = float(np.sqrt(np.var(yc, ddof=1)))
    mm = 1
    coefTemp = np.zeros(p)
    ind1 = np.zeros(p, dtype=bool)
    e1 = yc - yc.mean()

    while mm <= ctrl["numIter"]:
        lam_used = lam / 2.0 if (mm == 1 and post) else lam
        coefTemp = _lasso_shooting(Xc, yc, lam_used, XX, Xy)
        coefTemp = np.nan_to_num(coefTemp, nan=0.0)
        ind1 = np.abs(coefTemp) > 0
        x1 = Xc[:, ind1]

        if x1.shape[1] == 0:
            # hdm early-return: nothing selected.
            intercept_value = float((yc + mu).mean()) if intercept else float(yc.mean())
            beta_full = np.zeros(p)
            resid = yc - yc.mean()
            coefs = (
                np.concatenate([[intercept_value], beta_full])
                if intercept
                else beta_full
            )
            return RLassoFit(
                beta=beta_full,
                intercept=intercept_value if intercept else float("nan"),
                index=np.zeros(p, dtype=bool),
                coefficients=coefs,
                residuals=resid,
                sigma=float(np.var(yc, ddof=1)),
                loadings=Ups1,
                lambda0=lambda0,
                lambda_=lam,
                iter=mm,
                n=n,
                p=p,
                post=post,
                intercept_flag=intercept,
                meanx=meanx,
                mu=mu,
                colnames=list(colnames),
            )

        if post:
            coefT, *_ = np.linalg.lstsq(x1, yc, rcond=None)
            coefT = np.nan_to_num(coefT, nan=0.0)
            e1 = yc - x1 @ coefT
            coefTemp[ind1] = coefT
        else:
            e1 = yc - x1 @ coefTemp[ind1]

        Ups1, lam, s1 = _update_loadings(pen, e1, Xc, Psi, lambda0, n, rng=rng)

        mm += 1
        if abs(s0 - s1) < ctrl["tol"]:
            break
        s0 = s1

    threshold = ctrl["threshold"]
    if threshold is not None:
        coefTemp[np.abs(coefTemp) < threshold] = 0.0

    if intercept:
        intercept_value = float(mu - np.sum(meanx * coefTemp))
        coefs = np.concatenate([[intercept_value], coefTemp])
    else:
        intercept_value = float("nan")
        coefs = coefTemp.copy()

    s1 = float(np.sqrt(np.var(e1, ddof=1)))
    return RLassoFit(
        beta=coefTemp,
        intercept=intercept_value,
        index=ind1,
        coefficients=coefs,
        residuals=e1,
        sigma=s1,
        loadings=np.asarray(Ups1, float),
        lambda0=lambda0,
        lambda_=lam,
        iter=mm,
        n=n,
        p=p,
        post=post,
        intercept_flag=intercept,
        meanx=meanx,
        mu=mu,
        colnames=list(colnames),
    )
