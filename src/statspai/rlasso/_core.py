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
from ..exceptions import DataInsufficient, MethodIncompatibility

# ═══════════════════════════════════════════════════════════════════════
#  Penalty / control defaults (mirror hdm::rlasso.default)
# ═══════════════════════════════════════════════════════════════════════


def _default_penalty(
    n: int,
    post: bool,
    user: Optional[Dict[str, Any]],
    n_clusters: Optional[int] = None,
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
        # Cluster-Lasso: the effective sample size for the moderate-deviation
        # bound is the number of clusters (Belloni et al. 2016; lassopack).
        "gamma": user.get(
            "gamma", 0.1 / np.log(n if n_clusters is None else n_clusters)
        ),
    }
    if "numSim" in user:
        pen["numSim"] = user["numSim"]
    # hdm: post == FALSE & !exists("c", penalty)  ->  c = 0.5
    if not post and "c" not in user:
        pen["c"] = 0.5
    return pen


def _cluster_codes(cluster: Any, n: int) -> Tuple[np.ndarray, int]:
    """Integer codes ``0..G-1`` for a cluster identifier of length ``n``."""
    arr = np.asarray(cluster)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise MethodIncompatibility(
            f"cluster must be one-dimensional with one entry per row ({n}); "
            f"got shape {arr.shape}."
        )
    if arr.dtype.kind in "fc" and np.isnan(arr.astype(float)).any():
        raise DataInsufficient("cluster contains missing values.")
    _, codes = np.unique(arr, return_inverse=True)
    n_clusters = int(codes.max()) + 1 if n else 0
    if n_clusters < 2:
        raise DataInsufficient("cluster must identify at least two clusters.")
    return codes.astype(np.intp), n_clusters


def _score_norm(
    X: np.ndarray, e: np.ndarray, codes: Optional[np.ndarray]
) -> np.ndarray:
    """``sqrt(sum_i (x_ij e_i)^2)``, or its cluster analogue.

    With ``codes`` the scores are summed within cluster before squaring:
    ``sqrt(sum_g (sum_{i in g} x_ij e_i)^2)``, the cluster-Lasso loading of
    Belloni, Chernozhukov, Hansen and Kozbur (2016), which keeps the
    within-cluster cross products that the heteroskedastic loading drops.
    """
    if codes is None:
        out: np.ndarray = np.sqrt((e**2) @ (X**2))
        return out
    n_clusters = int(codes.max()) + 1
    sums = np.zeros((n_clusters, X.shape[1]))
    np.add.at(sums, codes, X * e[:, None])
    out = np.sqrt((sums**2).sum(axis=0))
    return out


def _require_finite(X: np.ndarray, y: np.ndarray, function: str) -> None:
    """Refuse missing or infinite values.

    A NaN in one column made every penalty loading NaN, the coordinate
    descent then set every coefficient to zero, and the fit reported
    "nothing selected" (so ``rlasso_effect`` silently dropped all controls).
    """
    if y.shape[0] != X.shape[0]:
        raise MethodIncompatibility(
            f"{function}: X has {X.shape[0]} rows but y has {y.shape[0]}."
        )
    bad_y = int((~np.isfinite(y)).sum())
    bad_x = int((~np.isfinite(X)).any(axis=1).sum())
    if bad_y or bad_x:
        raise DataInsufficient(
            f"{function}: {bad_y} missing/infinite value(s) in y and {bad_x} "
            "row(s) with a missing/infinite value in X.",
            recovery_hint="Drop or impute the incomplete rows first.",
            diagnostics={"rows_bad_y": bad_y, "rows_bad_X": bad_x},
        )


_ITERATIONS = ("hdm", "lassopack")


def _default_control(
    user: Optional[Dict[str, Any]], iteration: str = "hdm"
) -> Dict[str, Any]:
    """Resolve ``control``; the defaults depend on the iteration scheme.

    ``"hdm"``: up to 15 Lasso fits, stop when the residual SD (``ddof=1``)
    moves by less than ``1e-5``. ``"lassopack"``: Stata rlasso's
    ``maxpsiiter(2)`` and ``tolpsi(1e-4)``, so at most two Lasso fits.
    """
    user = dict(user) if user else {}
    iteration = user.get("iteration", iteration)
    if iteration not in _ITERATIONS:
        raise MethodIncompatibility(
            f"control['iteration'] must be one of {_ITERATIONS}, got {iteration!r}.",
            recovery_hint="Use 'hdm' (R hdm) or 'lassopack' (Stata rlasso).",
        )
    lp = iteration == "lassopack"
    return {
        "iteration": iteration,
        "numIter": user.get("numIter", 2 if lp else 15),
        "tol": user.get("tol", 1e-4 if lp else 1e-5),
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
    codes: Optional[np.ndarray] = None,
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
        Ups0 = (1.0 / np.sqrt(n)) * _score_norm(X, y, codes)
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
        Ups0 = (1.0 / np.sqrt(n)) * _score_norm(X, y, codes)
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
    codes: Optional[np.ndarray] = None,
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
        Ups1 = (1.0 / np.sqrt(n)) * _score_norm(X, e1, codes)
        lam = lambda0 * Ups1
    elif homo is False and xdep is True:
        lc = _lambda_calculation(penalty, y=e1, X=X, rng=rng)
        Ups1 = lc["Ups0"]
        lam = lc["lambda"]
    elif homo == "none":
        Ups1 = (1.0 / np.sqrt(n)) * _score_norm(X, e1, codes)
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
    #: Levels of each factor removed by ``absorb=`` (empty without it).
    absorb_levels: Tuple[int, ...] = ()
    #: Penalty-loading iteration used: ``"hdm"`` or ``"lassopack"``.
    iteration: str = "hdm"

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
    cluster: Optional[Any] = None,
    absorb: Optional[Any] = None,
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
        Overrides for ``numIter`` (default 15), ``tol`` (default 1e-5),
        ``threshold`` (default ``None``) and ``iteration``. ``iteration``
        is ``"hdm"`` (the default without ``absorb``) or ``"lassopack"``
        (the default with it). ``"lassopack"`` follows Stata's ``rlasso``:
        the first Lasso uses the full penalty (hdm halves it when
        ``post=True``), the loadings are refreshed from the post-Lasso
        residuals, at most ``numIter = 2`` Lasso fits are run
        (``maxpsiiter``) and the iteration stops early when the residual
        RMSE moves by less than ``tol = 1e-4`` (``tolpsi``). The two
        schemes can stop at different supports because the loading
        iteration may have more than one fixed point. ``"lassopack"``
        needs the default heteroskedastic or cluster loadings.
    colnames : list of str, optional
        Names for the columns of ``X`` (default ``V1..Vp``).
    rng : numpy Generator, optional
        Only used when ``X.dependent.lambda`` simulation is requested.
    cluster : (n,) array-like, optional
        Cluster identifier. Switches to the cluster-Lasso of Belloni,
        Chernozhukov, Hansen and Kozbur (2016) [@belloni2016inference]:
        the loading of column ``j`` becomes
        ``sqrt(sum_g (sum_{i in g} x_ij e_i)^2 / n)``, which keeps the
        within-cluster dependence of the scores, and ``gamma`` defaults
        to ``0.1 / log(G)`` with ``G`` the number of clusters (the
        ``rlasso, cluster()`` rule of Stata's lassopack). Under positive
        within-cluster correlation the loadings are larger than the
        heteroskedastic ones, so fewer variables are selected; ignoring
        the clustering over-selects. In a fixed-effects panel pass the
        within-transformed ``X`` and ``y`` with ``intercept=False``.
        Requires the default heteroskedastic loadings
        (``homoscedastic=False``, no ``X.dependent.lambda``).
    absorb : array-like, optional
        Fixed effects to absorb: one identifier per row, a 2-D array with
        one column per factor, or a list of such arrays. ``X`` and ``y``
        are within-transformed (projected off the dummies of every factor)
        and the Lasso is run without an intercept, as Stata's ``rlasso,
        fe`` does [@belloni2016inference]; the penalty level still uses
        all ``n`` rows. ``intercept`` is ignored, and ``control`` defaults
        to ``iteration="lassopack"``. With one factor this reproduces
        ``rlasso ..., fe``; with two, ``rlasso ..., fe partial(i.<f2>)``.
        Coefficients, residuals and ``predict`` are on the within scale.

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
    _require_finite(X, y, "rlasso")
    if colnames is None:
        colnames = [f"V{j + 1}" for j in range(p)]

    absorb_levels: Tuple[int, ...] = ()
    default_iteration = "hdm"
    if absorb is not None:
        from ._absorb import absorb_codes, absorbed_levels, within

        fe_codes = absorb_codes(absorb, n)
        absorb_levels = absorbed_levels(fe_codes)
        X = within(X, fe_codes)
        y = within(y, fe_codes)
        intercept = False
        default_iteration = "lassopack"

    codes: Optional[np.ndarray] = None
    n_clusters: Optional[int] = None
    if cluster is not None:
        codes, n_clusters = _cluster_codes(cluster, n)
    ctrl = _default_control(control, default_iteration)
    lassopack = ctrl["iteration"] == "lassopack"
    # hdm switches c to 0.5 without post-Lasso; lassopack keeps c = 1.1.
    pen = _default_penalty(n, post or lassopack, penalty, n_clusters=n_clusters)
    if lassopack and (
        pen["homoscedastic"] is not False or pen["X.dependent.lambda"] is True
    ):
        raise MethodIncompatibility(
            "control iteration='lassopack' supports the heteroskedastic and "
            "cluster loadings only.",
            recovery_hint="Drop the homoscedastic / X.dependent.lambda penalty, "
            "or use control={'iteration': 'hdm'}.",
        )
    if codes is not None and (
        pen["homoscedastic"] is True or pen["X.dependent.lambda"] is True
    ):
        raise MethodIncompatibility(
            "cluster= needs the heteroskedastic loadings: it cannot be "
            "combined with penalty={'homoscedastic': True} or "
            "{'X.dependent.lambda': True}."
        )

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

    if lassopack:
        lp = _lassopack_iterate(Xc, yc, pen, ctrl, post, XX, Xy, codes)
        coefTemp, ind1, e1, Ups1, lam, lambda0, mm = lp
    else:
        coefTemp, ind1, e1, Ups1, lam, lambda0, mm, early = _hdm_iterate(
            Xc, yc, pen, ctrl, post, XX, Xy, Psi, rng, codes, n, p
        )
        if early:
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
                absorb_levels=absorb_levels,
                iteration=ctrl["iteration"],
            )
    return _package(
        coefTemp,
        ind1,
        e1,
        Ups1,
        lam,
        lambda0,
        mm,
        ctrl,
        intercept,
        meanx,
        mu,
        n,
        p,
        post,
        colnames,
        absorb_levels,
    )


def _hdm_iterate(
    Xc: np.ndarray,
    yc: np.ndarray,
    pen: Dict[str, Any],
    ctrl: Dict[str, Any],
    post: bool,
    XX: np.ndarray,
    Xy: np.ndarray,
    Psi: np.ndarray,
    rng: Optional[np.random.Generator],
    codes: Optional[np.ndarray],
    n: int,
    p: int,
) -> tuple:
    """hdm's penalty-loading loop (``rlasso.default``), unchanged."""
    startingval = _init_values(Xc, yc)["residuals"]
    pencalc = _lambda_calculation(pen, y=startingval, X=Xc, rng=rng, codes=codes)
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
            return coefTemp, ind1, e1, Ups1, lam, lambda0, mm, True

        if post:
            coefT, *_ = np.linalg.lstsq(x1, yc, rcond=None)
            coefT = np.nan_to_num(coefT, nan=0.0)
            e1 = yc - x1 @ coefT
            coefTemp[ind1] = coefT
        else:
            e1 = yc - x1 @ coefTemp[ind1]

        Ups1, lam, s1 = _update_loadings(
            pen, e1, Xc, Psi, lambda0, n, rng=rng, codes=codes
        )

        mm += 1
        if abs(s0 - s1) < ctrl["tol"]:
            break
        s0 = s1

    return coefTemp, ind1, e1, Ups1, lam, lambda0, mm, False


def _lassopack_init_residuals(
    X: np.ndarray, y: np.ndarray, number: int = 5
) -> np.ndarray:
    """Residuals of an OLS of ``y`` on its ``number`` most correlated columns.

    lassopack's ``InitialResiduals`` for a model without a constant (or with
    data that were demeaned or within-transformed): the correlations are
    uncentred and the OLS has no intercept.
    """
    kx = X.shape[1]
    num = float(np.sqrt(y @ y))
    den = np.sqrt((X**2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.abs((y @ X) / (num * den))
    key = np.where(np.isnan(corr), -np.inf, corr)
    index = np.argsort(-key, kind="stable")[: min(number, kx)]
    b, *_ = np.linalg.lstsq(X[:, index], y, rcond=None)
    out: np.ndarray = y - X[:, index] @ b
    return out


def _lassopack_fit(
    Xc: np.ndarray,
    yc: np.ndarray,
    lam: np.ndarray,
    XX: np.ndarray,
    Xy: np.ndarray,
    post: bool,
) -> tuple:
    """One Lasso fit and the residuals the next loading update uses."""
    coef = np.nan_to_num(_lasso_shooting(Xc, yc, lam, XX, Xy), nan=0.0)
    ind = np.abs(coef) > 0
    if not ind.any():
        return coef, ind, yc.copy()
    x1 = Xc[:, ind]
    if post:
        coefT, *_ = np.linalg.lstsq(x1, yc, rcond=None)
        coef[ind] = np.nan_to_num(coefT, nan=0.0)
    return coef, ind, yc - x1 @ coef[ind]


def _lassopack_iterate(
    Xc: np.ndarray,
    yc: np.ndarray,
    pen: Dict[str, Any],
    ctrl: Dict[str, Any],
    post: bool,
    XX: np.ndarray,
    Xy: np.ndarray,
    codes: Optional[np.ndarray],
) -> tuple:
    """Stata lassopack's loading loop (``RLasso`` in ``lassoutils``).

    Re-implemented from the documented behaviour (help file and verbose
    output of ``rlasso``), not from its source. Initial loadings from the
    residuals of the five most correlated columns; first Lasso with the full
    penalty; then, while fewer than ``numIter`` fits have been run and the
    RMSE of the residuals moved by more than ``tol``, refresh the loadings
    from the latest (post-)Lasso residuals and refit. The reported loadings
    are the ones the final fit used.
    """
    n = Xc.shape[0]
    # On centred data (intercept, or within-transformed) this is the centred
    # correlation and OLS; without an intercept lassopack uses neither.
    v = _lassopack_init_residuals(Xc, yc)
    s1 = float(np.sqrt(np.mean(v**2)))
    pencalc = _lambda_calculation(pen, y=v, X=Xc, codes=codes)
    lambda0 = pencalc["lambda0"]
    Ups = np.atleast_1d(pencalc["Ups0"]).astype(float)
    lam = pencalc["lambda"].copy()
    coef, ind, e = _lassopack_fit(Xc, yc, lam, XX, Xy, post)
    it = 1
    delta = np.inf
    while it < ctrl["numIter"] and delta > ctrl["tol"]:
        s0 = s1
        s1 = float(np.sqrt(np.mean(e**2)))
        delta = abs(s1 - s0)
        Ups = _score_norm(Xc, e, codes) / np.sqrt(n)
        lam = lambda0 * Ups
        coef, ind, e = _lassopack_fit(Xc, yc, lam, XX, Xy, post)
        it += 1
    return coef, ind, e, Ups, lam, lambda0, it


def _package(
    coefTemp: np.ndarray,
    ind1: np.ndarray,
    e1: np.ndarray,
    Ups1: np.ndarray,
    lam: np.ndarray,
    lambda0: float,
    mm: int,
    ctrl: Dict[str, Any],
    intercept: bool,
    meanx: np.ndarray,
    mu: float,
    n: int,
    p: int,
    post: bool,
    colnames: List[str],
    absorb_levels: Tuple[int, ...],
) -> RLassoFit:
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
        absorb_levels=absorb_levels,
        iteration=ctrl["iteration"],
    )
