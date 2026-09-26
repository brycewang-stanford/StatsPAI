"""Spatial panel data estimators — SAR / SEM / SDM with entity fixed effects.

Balanced panel:  ``y_it = ρ W y_it + X_it β + μ_i + ε_it`` (SAR-FE)
               ``y_it = X_it β + μ_i + u_it;  u_it = λ W u_it + ε_it`` (SEM-FE)
               ``y_it = ρ W y_it + X_it β + W X_it θ + μ_i + ε_it`` (SDM-FE)

Within-transform removes ``μ_i``; concentrated ML on the demeaned variables
gives ρ (or λ), β, σ². Log-determinant is ``T * log|I_N − ρ W|`` (small-N
eigenvalue path).

Two-way fixed effects (``effects='twoways'``) demean along both dimensions.
Random effects are not implemented here (splm's ``spreml``) — open a
separate sub-spec if needed.

References
----------
Elhorst, J.P. (2014). *Spatial Econometrics: From Cross-Sectional Data to
  Spatial Panels*. Springer.
Lee, L.-F. & Yu, J. (2010). "Estimation of spatial autoregressive panel
  data models with fixed effects." *JoE*, 154(2), 165-185.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from ..._aliases import accepts_formula_first
from ..._result_serialize import ResultProtocolMixin
from ...exceptions import ConvergenceFailure, MethodIncompatibility
from ..models.ml import _coerce_W

EffectKind = Literal["fe", "twoways"]
ModelKind = Literal["sar", "sem", "sdm"]


# --------------------------------------------------------------------- #
#  Balanced-panel reshaping
# --------------------------------------------------------------------- #


def _balanced_panel_matrix(
    data: pd.DataFrame, entity: str, time: str, var: str
) -> np.ndarray:
    """Return (N, T) matrix: rows = entities (sorted), columns = time (sorted)."""
    pivot = data.pivot(index=entity, columns=time, values=var)
    if pivot.isna().any().any():
        raise ValueError(
            "Panel is unbalanced — every (entity, time) cell must be present."
        )
    return np.asarray(pivot.to_numpy(dtype=float), dtype=float)


def _within_transform(arr: np.ndarray, effects: EffectKind) -> np.ndarray:
    """Demean an (N, T) array within entities (and within time if twoways).

    Two-way: y_tilde = y_it - y_bar_i - y_bar_t + y_bar  (all from original arr).
    """
    if effects == "twoways":
        return np.asarray(
            arr
            - arr.mean(axis=1, keepdims=True)
            - arr.mean(axis=0, keepdims=True)
            + arr.mean(),
            dtype=float,
        )
    return np.asarray(arr - arr.mean(axis=1, keepdims=True), dtype=float)


# --------------------------------------------------------------------- #
#  Result dataclass
# --------------------------------------------------------------------- #


@dataclass
class SpatialPanelResult(ResultProtocolMixin):
    params: pd.Series  # [x1, x2, …, ρ or λ]
    std_errors: pd.Series
    model: ModelKind
    effects: EffectKind
    spatial_param: str  # "rho" or "lambda"
    spatial_param_value: float
    sigma2: float
    log_likelihood: float
    residuals: np.ndarray  # (N, T)
    N: int
    T: int

    def summary(self) -> str:
        lines = [
            f"Spatial Panel ({self.model.upper()}, {self.effects})",
            "-" * 50,
            f"Entities N : {self.N}",
            f"Periods  T : {self.T}",
            f"NT         : {self.N * self.T}",
            f"σ²         : {self.sigma2: .6f}",
            f"LogLik     : {self.log_likelihood: .4f}",
            "",
            "Coefficients:",
        ]
        for name in self.params.index:
            coef = self.params[name]
            se = self.std_errors[name]
            t = coef / se if se and np.isfinite(se) and se > 0 else np.nan
            lines.append(f"  {name:<15s}  {coef: .4f}   (se={se: .4f}, t={t: .3f})")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------- #


@accepts_formula_first()
def spatial_panel(
    data: pd.DataFrame,
    formula: str,
    entity: str,
    time: str,
    W: Any,
    model: ModelKind = "sar",
    effects: EffectKind = "fe",
    row_normalize: bool = True,
    vce: str = "information",
    twoways_lag: str = "splm",
) -> SpatialPanelResult:
    """Fit a spatial panel model by concentrated ML (Elhorst 2014).

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel; must be balanced.
    formula : str
        ``"y ~ x1 + x2"``. Constant is dropped (absorbed by the entity FE).
    entity, time : str
        Column names identifying entity and time dimensions.
    W : sparse matrix, ndarray, or :class:`statspai.spatial.weights.W`
        N × N spatial weights matrix aligned with ``sorted(data[entity].unique())``.
    model : {"sar", "sem", "sdm"}
    effects : {"fe", "twoways"}
        ``"fe"`` = entity FE only. ``"twoways"`` = entity + time demeaning.
    row_normalize : bool, default True
        Row-standardise ``W``.
    vce : {"information", "oim"}, default "information"
        ``"information"``: inverse of the analytic information matrix of
        ``(sigma^2, rho, beta)`` evaluated at the estimates, as
        ``splm::spml(model = "within")`` reports. ``"oim"``: inverse of the
        observed (analytic) Hessian of the full log-likelihood -- the
        ``vce(oim)`` default of Stata's ``xsmle``. The two differ in the
        ``rho``-``rho`` term (expected ``tr(WA'WA)`` + ``beta'X'(WA)'WAX beta``
        against observed ``(Wy)'(Wy)``) and so in every SE.
    twoways_lag : {"splm", "within"}, default "splm"
        Spatial lag regressor of SAR / SDM under ``effects="twoways"``
        (ignored otherwise: under entity effects the two coincide).
        ``"splm"``: ``W (Q y)``, the lag of the two-way demeaned ``y``, as
        ``splm::spml`` computes it (and as ``xsmle``'s log-likelihood
        evaluator does). ``"within"``: ``Q (W y)``, the two-way within
        transform of the lag, which is what the concentrated likelihood of
        the model with unit and time dummies (Lee & Yu 2010, eq. 21, the
        "direct approach") contains. For a row-standardised ``W`` the two
        differ by a per-period constant ``(1/N) 1' W Q y_t`` unless ``W``
        is also column-stochastic, so ``"splm"``'s SSE carries an extra
        ``rho^2 N sum_t m_t^2`` term. Both are direct-approach estimators
        with an O(1/N) bias in ``rho`` that does not vanish as ``T`` grows
        (see ``docs/dev/campaign_phase3``). ``"within"`` supports
        ``vce="oim"`` only.

    Returns
    -------
    SpatialPanelResult
        ``params`` / ``std_errors`` indexed ``[x..., rho]`` (``[x..., W_x...,
        rho]`` for SDM, ``[x..., lambda]`` for SEM), ``sigma2 = e'e / NT``
        and the full log-likelihood.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> N, T = 8, 6
    >>> W = np.zeros((N, N))  # ring contiguity weights
    >>> for i in range(N):
    ...     W[i, (i - 1) % N] = 1.0
    ...     W[i, (i + 1) % N] = 1.0
    >>> rows = []
    >>> for i in range(N):
    ...     fe = rng.normal()
    ...     for t in range(T):
    ...         x = rng.normal()
    ...         y = fe + 0.5 * x + rng.normal(0, 0.3)
    ...         rows.append({"id": i, "year": t, "y": y, "x": x})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.spatial_panel(
    ...     df, "y ~ x", entity="id", time="year", W=W, model="sar",
    ... )
    >>> res.spatial_param
    'rho'
    >>> (res.N, res.T)
    (8, 6)
    >>> bool(np.isfinite(res.params["x"]))
    True
    """
    if twoways_lag not in ("splm", "within"):
        raise MethodIncompatibility(
            f"twoways_lag must be 'splm' or 'within'; got {twoways_lag!r}"
        )
    if (
        twoways_lag == "within"
        and effects == "twoways"
        and model != "sem"
        and vce != "oim"
    ):
        raise MethodIncompatibility(
            "twoways_lag='within' is implemented with vce='oim' only; the "
            "expected information of splm assumes the W(Qy) lag",
            recovery_hint=(
                "Use vce='oim' with twoways_lag='within', or twoways_lag='splm'."
            ),
        )
    if vce not in ("information", "oim"):
        raise MethodIncompatibility(f"vce must be 'information' or 'oim'; got {vce!r}")
    if "~" not in formula:
        raise ValueError("formula must be of the form 'y ~ x1 + x2'")
    dep, indep_str = [s.strip() for s in formula.split("~", 1)]
    indep = [v.strip() for v in indep_str.split("+") if v.strip()]

    entities = sorted(data[entity].unique())
    times = sorted(data[time].unique())
    N = len(entities)
    T = len(times)

    # (N, T) matrices for dependent + each independent
    Y = _balanced_panel_matrix(data, entity, time, dep)
    X_mats = [_balanced_panel_matrix(data, entity, time, v) for v in indep]

    M = _coerce_W(W, n_expected=N, row_normalize=row_normalize)
    if M.shape != (N, N):
        raise ValueError(f"W shape {M.shape} does not match N entities = {N}")

    # Within transform
    Y_w = _within_transform(Y, effects)
    X_w = [_within_transform(x, effects) for x in X_mats]

    # Stack to NT vectors / matrix
    X_stack = np.column_stack([x.flatten(order="F") for x in X_w])

    if model == "sdm":
        # Append the within transform of W @ X (lag each period's raw X, then
        # demean like any other regressor). Lagging the demeaned X instead
        # is the same thing under entity effects but not under two-way
        # effects, where W does not commute with the cross-sectional
        # demeaning unless W is also column-stochastic.
        WX_mats = [_within_transform(np.asarray(M @ x), effects) for x in X_mats]
        WX_stack = np.column_stack([x.flatten(order="F") for x in WX_mats])
        X_stack = np.column_stack([X_stack, WX_stack])
        indep_names = list(indep) + [f"W_{v}" for v in indep]
    else:
        indep_names = list(indep)

    W_dense = M.toarray()
    eigvals = np.real(np.linalg.eigvals(W_dense))
    # Admissible interval of spdep/splm ``jacobianSetup(method = "eigen")``:
    # (1 / min eigenvalue, 1 / max eigenvalue).
    lo = 1.0 / float(eigvals.min())
    hi = 1.0 / float(eigvals.max())
    span = hi - lo
    lo, hi = lo + 1e-10 * span, hi - 1e-10 * span

    def _apply_spatial(rho_or_lam: float, target: np.ndarray) -> np.ndarray:
        """Apply (I - θ W) to each period column of an (N, T) matrix."""
        # target is (N, T); premultiply by (I - θ W)
        out = target - rho_or_lam * (W_dense @ target)
        return np.asarray(out, dtype=float)

    def _ldet(theta: float) -> float:
        return float(np.sum(np.log(np.abs(1 - theta * eigvals))))

    def _maximise(neg_ll, score) -> float:
        """Maximise the concentrated log-likelihood.

        A bounded Brent search locates the maximum, but on a flat likelihood
        a value search cannot resolve the maximiser beyond ~sqrt(eps)
        relative (splm's ``stats::optimize`` has the same limit). The root
        of the analytic concentrated score is then polished with ``brentq``,
        which pins the maximiser to machine precision.
        """
        from scipy.optimize import brentq

        opt = minimize_scalar(
            neg_ll,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 2000},
        )
        if not opt.success:
            raise ConvergenceFailure(
                f"spatial panel ML did not converge: {opt.message}"
            )
        x = float(opt.x)
        if min(x - lo, hi - x) < 1e-6 * span:
            import warnings

            warnings.warn(
                "spatial parameter on the boundary of its admissible interval; "
                "results should not be used",
                RuntimeWarning,
                stacklevel=3,
            )
            return x
        delta = 1e-4 * span
        a_, b_ = max(lo, x - delta), min(hi, x + delta)
        sa, sb = score(a_), score(b_)
        if np.sign(sa) != np.sign(sb):
            x = float(
                brentq(
                    score, a_, b_, xtol=1e-15, rtol=4 * np.finfo(float).eps, maxiter=500
                )
            )
        return x

    NT = N * T
    if model in ("sar", "sdm"):
        XtX_inv = np.linalg.inv(X_stack.T @ X_stack)

        if effects == "twoways" and twoways_lag == "within":
            WY_w = _within_transform(W_dense @ Y, effects)
        else:
            WY_w = W_dense @ Y_w

        def _lagged(rho: float) -> np.ndarray:
            return np.asarray(Y_w - rho * WY_w, dtype=float)

        def neg_ll(rho: float) -> float:
            Y_star = _lagged(rho)
            y_star_vec = Y_star.flatten(order="F")
            beta = XtX_inv @ (X_stack.T @ y_star_vec)
            e = y_star_vec - X_stack @ beta
            sigma2 = float(e @ e) / NT
            if sigma2 <= 0:
                return 1e20
            # log |I - ρ W| = sum log|1 - ρ λ_i|; panel brings factor T
            return float(
                -(
                    -NT / 2 * np.log(2 * np.pi * sigma2)
                    + T * _ldet(rho)
                    - (e @ e) / (2 * sigma2)
                )
            )

        wy_vec = WY_w.flatten(order="F")

        def score(rho: float) -> float:
            # d/drho of the concentrated log-likelihood; by the envelope
            # theorem dSSE/drho = -2 e'(Wy) with beta profiled out.
            y_star_vec = _lagged(rho).flatten(order="F")
            e = y_star_vec - X_stack @ (XtX_inv @ (X_stack.T @ y_star_vec))
            sse = float(e @ e)
            return NT * float(e @ wy_vec) / sse - T * float(
                np.sum(eigvals / (1 - rho * eigvals))
            )

        rho_hat = _maximise(neg_ll, score)
        Y_star = _lagged(rho_hat)
        y_star_vec = Y_star.flatten(order="F")
        beta = XtX_inv @ (X_stack.T @ y_star_vec)
        e = y_star_vec - X_stack @ beta
        sigma2 = float(e @ e) / NT
        # Asymptotic variance from the analytic information matrix of
        # (sigma^2, rho, beta) -- the matrix ``splm``'s ``splaglm`` inverts
        # (Elhorst 2014, Lee & Yu 2010). The (beta, rho) block is not zero,
        # so beta's variance must come from the joint inverse.
        WA = W_dense @ np.linalg.inv(np.eye(N) - rho_hat * W_dense)
        p = X_stack.shape[1]
        X3 = X_stack.reshape((N, T, p), order="F")  # [i, t, j]
        WAX = np.einsum("ik,ktj->itj", WA, X3).reshape((NT, p), order="F")
        xWAx = X_stack.T @ WAX
        xWAWAx = WAX.T @ WAX
        trWA = float(np.trace(WA))
        V_rr = T * (float(np.trace(WA @ WA)) + float(np.sum(WA * WA)))
        V_rr += float(beta @ xWAWAx @ beta) / sigma2
        info = np.zeros((p + 2, p + 2))
        info[0, 0] = NT / (2 * sigma2**2)
        info[0, 1] = info[1, 0] = T * trWA / sigma2
        info[1, 1] = V_rr
        cross = (xWAx @ beta) / sigma2
        info[1, 2:] = cross
        info[2:, 1] = cross
        info[2:, 2:] = (X_stack.T @ X_stack) / sigma2
        if vce == "oim":
            # Negative observed Hessian of the full log-likelihood in
            # (sigma^2, rho, beta); e = (I - rho W) y - X beta.
            Wy = wy_vec
            info = np.zeros((p + 2, p + 2))
            info[0, 0] = -NT / (2 * sigma2**2) + float(e @ e) / sigma2**3
            info[0, 1] = info[1, 0] = float(Wy @ e) / sigma2**2
            info[0, 2:] = info[2:, 0] = (X_stack.T @ e) / sigma2**2
            info[1, 1] = T * float(np.trace(WA @ WA)) + float(Wy @ Wy) / sigma2
            info[1, 2:] = info[2:, 1] = (X_stack.T @ Wy) / sigma2
            info[2:, 2:] = (X_stack.T @ X_stack) / sigma2
        avar = np.linalg.inv(info)
        se_beta = np.sqrt(np.diag(avar)[2:])
        se_rho = float(np.sqrt(avar[1, 1]))
        spatial_name = "rho"
        spatial_value = rho_hat
        loglik = -float(neg_ll(rho_hat))

    else:  # SEM-FE: (I - λW) premultiplied to both sides

        def _filtered(lam: float):
            y_s = _apply_spatial(lam, Y_w).flatten(order="F")
            X_s = np.column_stack(
                [_apply_spatial(lam, x).flatten(order="F") for x in X_w]
            )
            return y_s, X_s

        def neg_ll_sem(lam: float) -> float:
            y_star_vec, X_star_stack = _filtered(lam)
            beta, *_ = np.linalg.lstsq(X_star_stack, y_star_vec, rcond=None)
            e = y_star_vec - X_star_stack @ beta
            sigma2 = float(e @ e) / NT
            if sigma2 <= 0:
                return 1e20
            return float(
                -(
                    -NT / 2 * np.log(2 * np.pi * sigma2)
                    + T * _ldet(lam)
                    - (e @ e) / (2 * sigma2)
                )
            )

        y_vec = Y_w.flatten(order="F")
        X_raw = np.column_stack([x.flatten(order="F") for x in X_w])

        def score_sem(lam: float) -> float:
            # Envelope theorem: dSSE/dlambda = -2 e'(W u), u = y - X beta(lambda)
            y_star_vec, X_star_stack = _filtered(lam)
            beta, *_ = np.linalg.lstsq(X_star_stack, y_star_vec, rcond=None)
            e = y_star_vec - X_star_stack @ beta
            u = (y_vec - X_raw @ beta).reshape((N, T), order="F")
            Wu = (W_dense @ u).flatten(order="F")
            return NT * float(e @ Wu) / float(e @ e) - T * float(
                np.sum(eigvals / (1 - lam * eigvals))
            )

        lam_hat = _maximise(neg_ll_sem, score_sem)
        y_star_vec, X_star_stack = _filtered(lam_hat)
        XtX_inv = np.linalg.inv(X_star_stack.T @ X_star_stack)
        beta = XtX_inv @ (X_star_stack.T @ y_star_vec)
        e = y_star_vec - X_star_stack @ beta
        sigma2 = float(e @ e) / NT
        se_beta = np.sqrt(np.diag(sigma2 * XtX_inv))
        # (sigma^2, lambda) block of the information matrix; beta's block is
        # separable (``splm``'s ``sperrorlm``).
        WB = W_dense @ np.linalg.inv(np.eye(N) - lam_hat * W_dense)
        info2 = np.array(
            [
                [NT / (2 * sigma2**2), T * float(np.trace(WB)) / sigma2],
                [
                    T * float(np.trace(WB)) / sigma2,
                    T * (float(np.trace(WB @ WB)) + float(np.sum(WB * WB))),
                ],
            ]
        )
        se_rho = float(np.sqrt(np.linalg.inv(info2)[1, 1]))
        if vce == "oim":
            # Negative observed Hessian in (sigma^2, lambda, beta);
            # e = B (y - X beta), B = I - lambda W, u = y - X beta.
            u = (y_vec - X_raw @ beta).reshape((N, T), order="F")
            Wu = (W_dense @ u).flatten(order="F")
            WX = np.column_stack([(W_dense @ x).flatten(order="F") for x in X_w])
            p = X_star_stack.shape[1]
            H = np.zeros((p + 2, p + 2))
            H[0, 0] = -NT / (2 * sigma2**2) + float(e @ e) / sigma2**3
            H[0, 1] = H[1, 0] = float(Wu @ e) / sigma2**2
            H[0, 2:] = H[2:, 0] = (X_star_stack.T @ e) / sigma2**2
            H[1, 1] = T * float(np.trace(WB @ WB)) + float(Wu @ Wu) / sigma2
            cross = (WX.T @ e + X_star_stack.T @ Wu) / sigma2
            H[1, 2:] = H[2:, 1] = cross
            H[2:, 2:] = (X_star_stack.T @ X_star_stack) / sigma2
            avar = np.linalg.inv(H)
            se_beta = np.sqrt(np.diag(avar)[2:])
            se_rho = float(np.sqrt(avar[1, 1]))
        spatial_name = "lambda"
        spatial_value = lam_hat
        loglik = -float(neg_ll_sem(lam_hat))

    names = list(indep_names) + [spatial_name]
    params_vec = np.append(beta, spatial_value)
    se_vec = np.append(se_beta, se_rho)
    residuals = e.reshape((N, T), order="F")
    return SpatialPanelResult(
        params=pd.Series(params_vec, index=names),
        std_errors=pd.Series(se_vec, index=names),
        model=model,
        effects=effects,
        spatial_param=spatial_name,
        spatial_param_value=spatial_value,
        sigma2=sigma2,
        log_likelihood=loglik,
        residuals=residuals,
        N=N,
        T=T,
    )
