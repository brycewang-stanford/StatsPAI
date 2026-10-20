"""Prediction-interval inference for ``sp.scpi`` (private).

Clean-room implementation of the uncertainty quantification of
Cattaneo, Feng and Titiunik [@cattaneo2021prediction] as documented for the
``scpi`` software [@cattaneo2025scpi], for one treated unit, the outcome as
the only feature and no covariate adjustment. The R package ``scpi``
(GPL-3) was used only as a black box to pin conventions the papers leave
open; no reference source was read. The specification, with the black-box
evidence for every convention, is ``docs/dev/cleanroom/scpi.md``.

The prediction error of the synthetic control splits into an in-sample part
(the weights are estimated) and an out-of-sample part (the post-treatment
shock):

* in-sample: Gaussian draws ``G ~ N(0, Sigma)`` of the score, and for each
  draw, horizon and side the extreme value of ``p_t'(w - w_hat)`` over the
  weights that satisfy the simulated basic inequality and a locally relaxed
  version of the constraint set; the bounds are quantiles over draws;
* out-of-sample: sub-Gaussian, location-scale or quantile-regression bounds
  on the post-treatment error, estimated from the pre-treatment residuals.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _scpi_solvers as _sv

ACTIVE_TOL = 1e-6  # a weight counts as non-zero from this magnitude on
RHO_FLOOR = 1e-3  # a data-driven rho below this falls back to rho_max
IQR_TO_SD = 1.34  # Gaussian IQR rule of thumb used for the scale
HC_TYPES = ("HC0", "HC1", "HC2", "HC3", "HC4")


# ---------------------------------------------------------------------- #
#  Ingredients
# ---------------------------------------------------------------------- #


def _degrees_of_freedom(spec: Dict[str, Any], w: np.ndarray, B: np.ndarray) -> float:
    """Effective degrees of freedom of the weight estimator.

    ``ols``: J; ``lasso``: number of non-zero weights [@cattaneo2025scpi,
    Section 4.3]; ``simplex``: that number minus one; ``ridge`` and
    ``L1-L2``: ``sum_j s_j^2 / (s_j^2 + lambda)`` with ``s`` the singular
    values of ``B`` and ``lambda`` the shrinkage parameter of the radius rule
    (0 when the radius was given by the user).
    """
    name = spec["name"]
    J = B.shape[1]
    d0 = int(np.sum(np.abs(w) >= ACTIVE_TOL))
    if name == "ols":
        return float(J)
    if name == "lasso":
        return float(d0)
    if name == "simplex":
        return float(d0 - 1)
    lam = spec.get("lambda")
    if lam is None:
        return 0.0
    s = np.linalg.svd(B, compute_uv=False)
    return float(np.sum(s**2 / (s**2 + float(lam))))


def _regularisation(
    kind: Any, rho_max: float, res: np.ndarray, B: np.ndarray, d0: int
) -> Tuple[float, Optional[str]]:
    """Sparsity threshold rho [@cattaneo2025scpi, Section 4.3].

    ``rho = sqrt(d0 log(J) log(T0)) C / sqrt(T0)`` with ``C = sigma_u /
    min_j sigma_bj`` (``'type-1'``) or ``max_j sigma_bj sigma_u / min_j
    sigma_bj^2`` (``'type-2'``, the default), capped at ``rho_max``;
    ``sigma_u`` is the population and ``sigma_bj`` the sample standard
    deviation. A user number is used as given. Returns ``(rho, note)``.
    """
    if kind is not None and not isinstance(kind, str):
        return float(kind), None
    kind = "type-2" if kind is None else kind
    if kind not in ("type-1", "type-2"):
        raise MethodIncompatibility(
            f"rho must be 'type-1', 'type-2' or a positive number, got {kind!r}.",
            recovery_hint="Use rho='type-2' (default), rho='type-1' or a number.",
            diagnostics={"rho": kind},
        )
    T0, J = B.shape
    sd_b = np.std(B, axis=0, ddof=1)
    if sd_b.min() <= 1e-12 * max(1.0, float(np.abs(B).max())):
        raise DataInsufficient(
            "sp.scpi: a donor has no variation in the pre-treatment period, "
            "so the regularisation parameter rho cannot be computed.",
            recovery_hint="Drop constant donors or pass a numeric rho=.",
            diagnostics={"constant_donors": np.flatnonzero(sd_b <= 0).tolist()},
        )
    sd_u = float(np.std(res))
    if kind == "type-1":
        C = sd_u / sd_b.min()
    else:
        C = sd_b.max() * sd_u / sd_b.min() ** 2
    rho = float(np.sqrt(d0 * np.log(J) * np.log(T0)) * C / np.sqrt(T0))
    rho = min(rho, float(rho_max))
    if not rho >= RHO_FLOOR:
        return float(rho_max), "low"
    return rho, None


def _hc_scale(u_sigma: str, Z: np.ndarray, TT: int, df: float) -> np.ndarray:
    """Heteroskedasticity-robust scale factors (MacKinnon's HC0-HC4).

    The leverage is the diagonal of the hat matrix of ``Z`` (the donors'
    pre-treatment outcomes); ``df`` is the effective degrees of freedom.
    """
    if u_sigma not in HC_TYPES:
        raise MethodIncompatibility(
            f"u_sigma must be one of {list(HC_TYPES)}, got {u_sigma!r}.",
            recovery_hint="Use u_sigma='HC1' (default) or one of HC0-HC4.",
        )
    if u_sigma == "HC0":
        return np.ones(TT)
    if u_sigma == "HC1":
        return np.full(TT, TT / (TT - df))
    lev = np.einsum("ij,ji->i", Z, np.linalg.pinv(Z.T @ Z) @ Z.T)
    if u_sigma == "HC2":
        return np.asarray(1.0 / (1.0 - lev), dtype=float)
    if u_sigma == "HC3":
        return np.asarray(1.0 / (1.0 - lev) ** 2, dtype=float)
    with np.errstate(divide="ignore"):
        delta = np.minimum(4.0, TT * lev / df) if df > 0 else np.full(TT, 4.0)
    return np.asarray((1.0 - lev) ** (-delta), dtype=float)


def _sqrtm_psd(S: np.ndarray) -> np.ndarray:
    """Symmetric square root of a positive semi-definite matrix (via SVD)."""
    U, d, _ = np.linalg.svd(0.5 * (S + S.T))
    return np.asarray((U * np.sqrt(d)) @ U.T, dtype=float)


def _local_geometry(
    spec: Dict[str, Any],
    w: np.ndarray,
    rho: Any,
    rho_max: float,
    res: np.ndarray,
    B: np.ndarray,
) -> Dict[str, Any]:
    """Constraint set used in the simulation [@cattaneo2025scpi, Section 4.3].

    An inequality constraint counts as binding when it is within rho of
    holding with equality; binding linear constraints are kept at their
    estimated value, slack ones at their original form:

    * simplex / L1-L2: ``w_j >= lb_j`` with ``lb_j = w_hat_j`` if
      ``w_hat_j < rho`` and 0 otherwise, ``sum(w) = Q``; for L1-L2 also
      ``||w||_2 <= Q2*``, with ``Q2*`` from the ridge rule below using
      ``min(rho, rho_max)``;
    * lasso: ``||w||_1 <= Q*`` with ``Q* = ||w_hat||_1 + rho`` when
      ``||w_hat||_1 > Q - rho`` and ``Q* = Q`` otherwise;
    * ridge: ``||w||_2 <= Q*`` with ``Q* = ||w_hat||_2^2 + rho`` when
      ``||w_hat||_2^2 > Q - rho`` and ``Q* = Q`` otherwise (R's outputs:
      the squared norm is compared with Q, see the specification);
    * ols: unrestricted.

    ``rho`` is computed here when it is a rule name or ``None``.
    """
    d0 = int(np.sum(np.abs(w) >= ACTIVE_TOL))
    rho_v, note = _regularisation(rho, rho_max, res, B, d0)
    name = spec["name"]
    J = len(w)
    out: Dict[str, Any] = {"rho": rho_v, "rho_note": note, "w": w, "Q2_star": None}
    if name in ("simplex", "L1-L2"):
        lb = np.where(w < rho_v, w, 0.0)
        out.update(kind="simplex", lb=lb, Q_star=float(spec["Q"]))
        if name == "L1-L2":
            # the L2 ball follows the ridge rule with rho capped at rho_max
            Q2 = float(spec["Q2"])
            r2 = min(rho_v, float(rho_max))
            sq = float(w @ w)
            Q2s = sq + r2 if sq > Q2 - r2 else Q2
            out.update(Q2=Q2s, Q2_star=Q2s)
    elif name == "lasso":
        Q = float(spec["Q"])
        l1 = float(np.abs(w).sum())
        Qs = l1 + rho_v if l1 > Q - rho_v else Q
        out.update(kind="l1", Q=Qs, Q_star=Qs, lb=np.full(J, -np.inf))
    elif name == "ridge":
        Q = float(spec["Q"])
        sq = float(w @ w)
        Qs = sq + rho_v if sq > Q - rho_v else Q
        out.update(kind="l2", Q=Qs, Q_star=Qs, lb=np.full(J, -np.inf))
    else:
        out.update(kind="none", Q_star=None, lb=np.full(J, -np.inf))
    return out


def _active_donors(
    spec: Dict[str, Any], w: np.ndarray, rho: float
) -> Tuple[np.ndarray, bool]:
    """Donors entering the residual designs.

    For the L1-type constraints (simplex, lasso, L1-L2) the donors whose
    weight exceeds rho in absolute value (the largest one if none does); for
    ridge and ols every donor. The flag reports the "none exceeds" case.
    """
    if spec["name"] in ("ridge", "ols"):
        return np.ones(len(w), bool), False
    keep = np.abs(w) > rho
    if keep.any():
        return keep, False
    keep = np.zeros(len(w), bool)
    keep[int(np.argmax(np.abs(w)))] = True
    return keep, True


def _design(Bk: np.ndarray, order: int, T0: int) -> Tuple[np.ndarray, int]:
    """Residual design ``[B_active, 1]`` (order 1) or ``[1]`` (order 0).

    Falls back to the constant when fewer than 11 observations per column
    remain (``T0 <= columns + 10``). Returns ``(design, order_used)``.
    """
    ones = np.ones((Bk.shape[0], 1))
    if order >= 1:
        D = np.column_stack([Bk, ones])
        if T0 > D.shape[1] + 10:
            return D, order
    return ones, 0


def _quantile(x: np.ndarray, q: float, axis: int = 0) -> np.ndarray:
    """Type-7 (linear) quantile ignoring NaN; NaN where nothing is finite."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(np.nanquantile(x, q, axis=axis), dtype=float)


# ---------------------------------------------------------------------- #
#  Out-of-sample error
# ---------------------------------------------------------------------- #


def _out_of_sample(
    res: np.ndarray, De: np.ndarray, De_post: np.ndarray, alpha: float
) -> Dict[str, Any]:
    """Moments and bounds of the post-treatment error.

    Conditional mean: least squares of the pre-treatment residuals on the
    design. Conditional variance: the smaller of an exponential
    (log-squared-residual) regression and the squared interquartile range
    divided by 1.34, both from restricted regression quantiles. Quantile
    bounds: restricted regression quantiles on the design, or the type-7
    sample quantiles when the design is a constant.
    Bounds: sub-Gaussian
    ``mean -/+ sqrt(2 var log(2/alpha))``; location-scale with the empirical
    quantiles of the standardised residuals; and regression quantiles.
    """
    fit, mean_post = _sv.lm_fitted(De, res, De_post)
    dev2 = (res - fit) ** 2
    tiny = np.finfo(float).tiny
    log_dev2 = np.log(np.maximum(dev2, tiny))
    lfit, lpost = _sv.lm_fitted(De, log_dev2, De_post)
    var_exp_post, var_exp_in = np.exp(np.asarray(lpost)), np.exp(lfit)
    taus = [0.25, 0.75, alpha / 2, 1 - alpha / 2]
    qpost = De_post @ _sv.rrq(De, res, taus)
    var_iqr = ((qpost[:, 1] - qpost[:, 0]) / IQR_TO_SD) ** 2
    if De.shape[1] == 1:  # constant design: bounds at the sample (type-7) quantiles
        qpost[:, 2:4] = np.quantile(res, taus[2:4])
    var_post = np.minimum(var_exp_post, var_iqr)
    half = np.sqrt(2.0 * var_post * np.log(2.0 / alpha))
    eps = (res - fit) / np.sqrt(var_exp_in)
    q_lo, q_hi = _quantile(eps, alpha / 2), _quantile(eps, 1 - alpha / 2)
    sd = np.sqrt(var_post)
    return {
        "mean": mean_post,
        "var": var_post,
        "subgaussian": np.column_stack([mean_post - half, mean_post + half]),
        "ls": np.column_stack([mean_post + sd * q_lo, mean_post + sd * q_hi]),
        "qreg": qpost[:, 2:4].copy(),
    }


# ---------------------------------------------------------------------- #
#  In-sample simulation
# ---------------------------------------------------------------------- #


def _solve_one(
    c: np.ndarray, M: np.ndarray, G: np.ndarray, geom: Dict[str, Any], fallback: list
) -> float:
    """``c'y`` at the optimum of min c'y (NaN when unbounded or failed)."""
    y = _sv.insample_solve(c, M, G, geom)
    if isinstance(y, str):
        return np.nan
    if y is None:
        y = _sv.insample_slsqp(c, M, G, geom)
        fallback[0] += 1
        if y is None:
            return np.nan
    return float(c @ y)


def _simulate(
    rows: np.ndarray,
    M: np.ndarray,
    zeta: np.ndarray,
    geom: Dict[str, Any],
    fallback: list,
) -> np.ndarray:
    """Per-draw lower and upper in-sample errors for each row of ``rows``.

    Column ``t`` holds ``-max p_t'y`` and column ``H + t`` holds
    ``-min p_t'y`` over the simulated feasible set (``H`` rows).
    """
    H = rows.shape[0]
    S = zeta.shape[1]
    out = np.full((S, 2 * H), np.nan)
    for s in range(S):
        G = zeta[:, s]
        for t in range(H):
            p = rows[t]
            hi = _solve_one(-p, M, G, geom, fallback)  # = -max p'y
            lo = _solve_one(p, M, G, geom, fallback)  # = min p'y
            out[s, t] = hi
            out[s, H + t] = -lo
    return out


# ---------------------------------------------------------------------- #
#  Driver
# ---------------------------------------------------------------------- #


def scpi_inference(
    A: Any,
    B: Any,
    P: Any,
    w: Any,
    spec: Any,
    sims: int = 200,
    draws: Optional[Any] = None,
    rng: Optional[Any] = None,
    u_missp: bool = True,
    u_sigma: str = "HC1",
    u_order: int = 1,
    u_alpha: float = 0.05,
    e_order: int = 1,
    e_alpha: float = 0.05,
    rho: Optional[Any] = None,
    rho_max: float = 0.2,
    aggregate: bool = False,
) -> Dict[str, Any]:
    """Prediction-interval ingredients and bounds for given data and weights.

    ``A`` (T0), ``B`` (T0 x J) and ``P`` (T1 x J) are the pre-treatment
    outcome of the treated unit, the donors' pre-treatment outcomes and their
    post-treatment outcomes; ``w`` the estimated weights and ``spec`` their
    constraint (``name``, ``Q``, ``Q2``, ``lambda``). ``draws`` (J x sims)
    are standard normal draws (generated from ``rng`` when omitted). With
    ``aggregate=True`` the bounds for the average over post-treatment periods
    are added under ``"aggregate"``.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    P = np.asarray(P, float)
    w = np.asarray(w, float)
    T0, J = B.shape
    T1 = P.shape[0]
    if u_order > 1 or e_order > 1:
        raise NotImplementedError(
            "u_order / e_order > 1 (polynomial residual designs) are not implemented."
        )
    if draws is not None:
        Z = np.asarray(draws, float)
        if Z.shape != (J, sims):
            raise MethodIncompatibility(
                f"draws must have shape ({J}, {sims}) (donors x sims), got {Z.shape}.",
                recovery_hint="Pass a donors x sims matrix of standard normal draws.",
            )
    else:
        gen = rng if rng is not None else np.random.default_rng()
        Z = gen.standard_normal((J, sims))

    res = A - B @ w
    geom = _local_geometry(spec, w, rho, rho_max, res, B)
    rho_v = geom["rho"]
    if geom["rho_note"] == "low":
        warnings.warn(
            "sp.scpi: the data-driven rho was too low (below 1e-3, e.g. an exact "
            f"pre-treatment fit); rho_max = {rho_max} is used instead.",
            RuntimeWarning,
            stacklevel=3,
        )
    keep, none_above = _active_donors(spec, w, rho_v)
    if none_above:
        warnings.warn(
            f"sp.scpi: rho was too high ({rho_v:.4g}): no weight exceeds it, so "
            "only the largest weight is kept active in the residual designs.",
            RuntimeWarning,
            stacklevel=3,
        )

    df = _degrees_of_freedom(spec, w, B)
    if df >= T0:
        warnings.warn(
            f"sp.scpi: the weights use {df:.4g} degrees of freedom, more degrees "
            f"of freedom than observations (T0 = {T0}); df is capped at T0 - 1.",
            RuntimeWarning,
            stacklevel=3,
        )
        df = float(T0 - 1)

    # conditional moments of the pseudo-residuals
    Bk = B[:, keep]
    if u_missp:
        Du, u_order_used = _design(Bk, u_order, T0)
        u_mean, _ = _sv.lm_fitted(Du, res)
        u_T, u_params = T0, Du.shape[1]
    else:
        u_mean = np.zeros(T0)
        u_order_used, u_T, u_params = u_order, 0, 0
    vc = _hc_scale(u_sigma, B, T0, df)
    u_var = vc * (res - u_mean) ** 2
    Sigma = (B.T * u_var) @ B / T0**2

    # in-sample simulation
    zeta = _sqrtm_psd(Sigma) @ Z
    M = B.T @ B / T0
    fallback = [0]
    rows = np.vstack([P, P.mean(axis=0)]) if aggregate else P
    vs_all = _simulate(rows, M, zeta, geom, fallback)
    H = rows.shape[0]
    lower_all, upper_all = vs_all[:, :H], vs_all[:, H:]
    lower, upper = lower_all[:, :T1], upper_all[:, :T1]
    vsig = np.hstack([lower, upper])
    n_bad = int(np.isnan(vsig).sum())
    if n_bad == vsig.size:
        warnings.warn(
            "sp.scpi: Every in-sample simulation problem was unbounded or failed "
            "(e.g. more donors than pre-treatment periods without a norm "
            "constraint); the in-sample bounds are NaN.",
            RuntimeWarning,
            stacklevel=3,
        )
    else:
        _sv.warn_failed(n_bad, vsig.size)
    failed = 100.0 * np.vstack([np.isnan(lower).mean(0), np.isnan(upper).mean(0)])
    a1, a2 = float(u_alpha), float(e_alpha)
    insample = np.column_stack([_quantile(lower, a1 / 2), _quantile(upper, 1 - a1 / 2)])

    # out-of-sample error
    De, e_order_used = _design(Bk, e_order, T0)
    Pk = P[:, keep]
    De_post = (
        np.column_stack([Pk, np.ones(T1)]) if e_order_used >= 1 else np.ones((T1, 1))
    )
    oos = _out_of_sample(res, De, De_post, a2)
    e_bounds = {"subgaussian": oos["subgaussian"], "ls": oos["ls"], "qreg": oos["qreg"]}
    bounds = {"insample": insample}
    for k, eb in e_bounds.items():
        bounds[k] = insample + eb

    # simultaneous bounds over the post-treatment horizon
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo_min = np.nanmin(lower, axis=1)
        hi_max = np.nanmax(upper, axis=1)
    m_lo, m_hi = _quantile(lo_min, a1 / 2), _quantile(hi_max, 1 - a1 / 2)
    scale = np.sqrt(np.log(T1 + 1.0))
    bounds["joint"] = np.column_stack(
        [
            m_lo + scale * e_bounds["subgaussian"][:, 0],
            m_hi + scale * e_bounds["subgaussian"][:, 1],
        ]
    )

    out: Dict[str, Any] = {
        "rho": rho_v,
        "Q_star": geom["Q_star"],
        "Q2_star": geom["Q2_star"],
        "lb": geom["lb"],
        "df": df,
        "u_mean": u_mean,
        "u_var": u_var,
        "u_T": u_T,
        "u_params": u_params,
        "u_order": u_order_used,
        "Sigma": Sigma,
        "e_mean": oos["mean"],
        "e_var": oos["var"],
        "e_T": T0,
        "e_params": De.shape[1],
        "e_order": e_order_used,
        "e_bounds": e_bounds,
        "bounds": bounds,
        "vsig": vsig,
        "failed_sims": failed,
        "n_slsqp_fallback": int(fallback[0]),
        "sims": int(sims),
        "active_donors": keep,
    }
    if aggregate:
        a_lo, a_hi = lower_all[:, T1], upper_all[:, T1]
        agg_in = np.array([_quantile(a_lo, a1 / 2), _quantile(a_hi, 1 - a1 / 2)])
        row = De_post.mean(axis=0, keepdims=True)
        agg_oos = _out_of_sample(res, De, row, a2)
        agg: Dict[str, Any] = {"insample": agg_in}
        for k in ("subgaussian", "ls", "qreg"):
            eb = agg_oos[k][0]
            agg["e_" + k] = eb
            agg[k] = agg_in + eb
        agg["e_mean"] = float(agg_oos["mean"][0])
        agg["e_var"] = float(agg_oos["var"][0])
        out["aggregate"] = agg
    return out
