r"""Model-averaging double/debiased machine learning (Ahrens et al. 2025).

Standard DML picks a single nuisance learner for the outcome regression
:math:`\ell_0(X) = E[Y|X]` and the treatment conditional mean
:math:`m_0(X) = E[D|X]`. Getting that choice wrong degrades the
:math:`\sqrt n`-rate consistency of the target parameter
:math:`\theta_0`, so applied researchers commonly run DML under several
candidates and inspect agreement.

Ahrens, Hansen, Schaffer and Wiemann (2025, *JAE*) formalise this as
**stacking with cross-fitting**: combine candidate nuisance learners
via constrained least squares (CLS) on cross-fitted predictions, then
plug the stacked nuisance into the standard PLR moment equation.

This module implements three variants:

* ``weight_rule="short_stacking"`` — the paper's *short-stacking* recipe
  (default). For each nuisance, solve

  .. math::

     \min_{w_1,\dots,w_J}\; \sum_{i=1}^n
        \Bigl(Y_i - \sum_{j=1}^J w_j\,\hat\ell^{(j)}_{I^c_{k(i)}}(X_i)\Bigr)^2
        \quad\text{s.t.}\quad w_j\ge 0,\ \sum_j w_j = 1

  where :math:`\hat\ell^{(j)}_{I^c_k}` is the cross-fitted prediction
  from candidate ``j`` (Ahrens et al. 2025, eq. 7). The stacked
  out-of-fold prediction is :math:`\hat\ell^{\mathrm{stack}}_i =
  \sum_j \hat w_j \hat\ell^{(j)}_{I^c_{k(i)}}(X_i)`. Same for
  :math:`\hat m^{\mathrm{stack}}`. The PLR estimator is

  .. math::

     \hat\theta = \frac{\sum_i (Y_i - \hat\ell^{\mathrm{stack}}_i)
                              (D_i - \hat m^{\mathrm{stack}}_i)}
                       {\sum_i (D_i - \hat m^{\mathrm{stack}}_i)^2}

  with the standard PLR sandwich variance — the moment equation is
  Neyman-orthogonal so the variance does not require a between-candidate
  covariance correction.

* ``weight_rule="single_best"`` — :math:`w_j \in \{0,1\}` (Ahrens et al.
  2025, footnote 8): pick the candidate with the lowest cross-fitted
  nuisance MSE. Asymptotically equivalent to the best learner under
  van der Laan & Dudoit (2003) conditions.

* ``weight_rule="inverse_risk"`` / ``"equal"`` — convenience baselines
  that ARE NOT in Ahrens et al. (2025). They compute per-candidate
  :math:`\hat\theta_k` first, then weight by :math:`1/\mathrm{MSE}_k`
  (or uniformly), and report the influence-function-based variance of
  the weighted average. Use ``"short_stacking"`` if you want the paper's
  approach; ``"inverse_risk"`` is kept for backwards compatibility and
  as a quick check against more rigorous stacking.

Pooled stacking (eq. 8 of the paper) and conventional per-fold stacking
(eq. 6) are not yet implemented — short-stacking dominates on the small-
:math:`J` regime that fits ``sp.dml_model_averaging`` use cases (the
paper recommends short-stacking when :math:`J \ll n`, §3).

References
----------
Ahrens, A., Hansen, C.B., Schaffer, M.E. and Wiemann, T. (2025).
    "Model Averaging and Double Machine Learning."
    *Journal of Applied Econometrics*, 40(3), 249-269.
    DOI 10.1002/jae.3103. [@ahrens2025model]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.results import CausalResult


__all__ = ["dml_model_averaging", "model_averaging_dml", "DMLAveragingResult"]


def _default_candidates() -> List[Tuple[Any, Any, str]]:
    """Return a reasonable default roster of (g, m, label) triples."""
    from sklearn.linear_model import LassoCV, RidgeCV
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

    return [
        (LassoCV(cv=5), LassoCV(cv=5), "lasso"),
        (RidgeCV(), RidgeCV(), "ridge"),
        (RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1),
         RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1), "rf"),
        (GradientBoostingRegressor(n_estimators=200, random_state=0),
         GradientBoostingRegressor(n_estimators=200, random_state=0), "gbm"),
    ]


class DMLAveragingResult(CausalResult):
    """CausalResult extended with per-candidate and weight details.

    Attributes stored in ``model_info``:

    * ``candidates``  — list of candidate labels.
    * ``theta_k``     — per-candidate :math:`\\hat\\theta` (only meaningful
      for the non-stacking weight rules).
    * ``se_k``        — per-candidate SE.
    * ``mse_k``       — per-candidate nuisance risk (g + m).
    * ``weights``     — averaging or stacking weights.
    * ``weights_g`` / ``weights_m`` — separate stacking weights per
      nuisance under ``weight_rule="short_stacking"``.
    * ``weight_rule`` — how the weights were computed.
    """


def _fit_candidate_plr(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    ml_g: Any,
    ml_m: Any,
    n_folds: int,
    seed: int,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit one PLR candidate; return (yhat, dhat, y_resid, d_resid, mse_g, mse_m).

    Both the cross-fitted predictions ``yhat / dhat`` AND the residuals
    are needed: residuals feed into the per-candidate ``θ̂_k`` for the
    inverse-risk / equal / single_best weight rules; predictions feed
    into the CLS short-stacking weight rule.

    When ``sample_weight`` is supplied the nuisance learners are fit
    with ``sample_weight=`` (falling back to unweighted fit if the
    learner doesn't accept it) and the reported MSE is the weighted MSE.
    """
    from sklearn.base import clone
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    yhat = np.zeros(n)
    dhat = np.zeros(n)

    def _fit(learner, Xfit, yfit, wfit):
        clf = clone(learner)
        if wfit is None:
            clf.fit(Xfit, yfit)
            return clf
        try:
            clf.fit(Xfit, yfit, sample_weight=wfit)
        except TypeError:
            import warnings
            warnings.warn(
                f"{type(learner).__name__}.fit does not accept "
                f"sample_weight; falling back to unweighted nuisance fit.",
                RuntimeWarning,
                stacklevel=4,
            )
            clf.fit(Xfit, yfit)
        return clf

    for tr, te in kf.split(X):
        wtr = sample_weight[tr] if sample_weight is not None else None
        g = _fit(ml_g, X[tr], Y[tr], wtr)
        yhat[te] = g.predict(X[te])

        m = _fit(ml_m, X[tr], D[tr], wtr)
        dhat[te] = m.predict(X[te])

    y_resid = Y - yhat
    d_resid = D - dhat
    if sample_weight is None:
        mse_g = float(np.mean(y_resid ** 2))
        mse_m = float(np.mean(d_resid ** 2))
    else:
        W = float(np.sum(sample_weight))
        mse_g = float(np.sum(sample_weight * y_resid ** 2) / W)
        mse_m = float(np.sum(sample_weight * d_resid ** 2) / W)
    return yhat, dhat, y_resid, d_resid, mse_g, mse_m


def _solve_cls_weights(
    target: np.ndarray,
    predictions: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Solve constrained least squares for stacking weights.

    minimise ``Σ w_obs · (target - predictions @ w)²`` s.t. ``w_j ≥ 0,
    Σ w_j = 1`` — where ``w_obs`` defaults to 1 (unweighted CLS).

    Implementation: scipy.optimize.minimize with SLSQP. The CLS problem
    is convex with linear constraints, so SLSQP converges in O(K)
    iterations for the small K (~4-15 candidates) of the model-averaging
    use case. Falls back to ``np.argmin(per-candidate MSE)`` if the
    optimiser fails to converge.
    """
    from scipy.optimize import minimize

    K = predictions.shape[1]
    sw = (
        np.ones(len(target)) if sample_weight is None
        else np.asarray(sample_weight, dtype=float)
    )

    def loss(w):
        r = target - predictions @ w
        return float(np.sum(sw * r * r))

    def grad(w):
        r = target - predictions @ w
        return -2.0 * predictions.T @ (sw * r)

    w0 = np.full(K, 1.0 / K)
    bounds = [(0.0, 1.0)] * K
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0),
                    "jac": lambda w: np.ones(K)}]
    res = minimize(
        loss, w0, jac=grad, method="SLSQP", bounds=bounds,
        constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not res.success:
        # Fallback: pick the column with lowest weighted MSE.
        weighted_sse = np.sum(sw[:, None] * (target[:, None] - predictions) ** 2, axis=0)
        w = np.zeros(K)
        w[int(np.argmin(weighted_sse))] = 1.0
        return w
    w = np.clip(res.x, 0.0, None)
    s = w.sum()
    return w / s if s > 0 else np.full(K, 1.0 / K)


def dml_model_averaging(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    candidates: Optional[List[Tuple[Any, Any, str]]] = None,
    n_folds: int = 5,
    seed: int = 0,
    weight_rule: str = "short_stacking",
    alpha: float = 0.05,
    sample_weight: Optional[Any] = None,
) -> DMLAveragingResult:
    """Model-averaging / stacking DML-PLR estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    treat : str
        Continuous-or-binary treatment column.
    covariates : list of str
        Covariate columns ``X``.
    candidates : list of (ml_g, ml_m, label), optional
        Candidate nuisance learners.  ``ml_g`` regresses ``y`` on ``X``;
        ``ml_m`` regresses ``treat`` on ``X``.  Defaults to a Lasso/Ridge/
        RandomForest/GradientBoosting roster.
    n_folds : int, default 5
        Cross-fitting folds per candidate.
    seed : int, default 0
    weight_rule : {"short_stacking", "single_best", "inverse_risk", "equal"}
        How to combine candidate nuisance predictions or estimates.

        * ``"short_stacking"`` *(default; Ahrens et al. 2025 eq. 7)*  —
          solve constrained least squares on cross-fitted predictions
          for each nuisance separately (``ŷ`` and ``D̂``), produce
          stacked nuisances, plug into the PLR moment equation.
        * ``"single_best"`` — Ahrens et al. (2025, fn. 8): pick the
          candidate with lowest joint nuisance MSE.
        * ``"inverse_risk"`` — :math:`w_k \\propto 1/(\\text{MSE}_g +
          \\text{MSE}_m)`. Convenience baseline; **not** in the paper.
        * ``"equal"`` — :math:`w_k = 1/K`. Convenience baseline; **not**
          in the paper.

        For the non-stacking rules (``inverse_risk`` / ``equal`` /
        ``single_best``) the function computes per-candidate
        :math:`\\hat\\theta_k` and reports the weighted average with a
        between-candidate-covariance-corrected SE; for
        ``"short_stacking"`` it reports the standard PLR sandwich SE on
        the stacked-nuisance score (Neyman orthogonality is preserved).
    alpha : float, default 0.05
        Two-sided CI level.
    sample_weight : np.ndarray | pd.Series | str, optional
        Per-observation weights. If supplied, every nuisance fit uses
        ``sample_weight=`` (with a graceful fallback warning if the
        learner does not accept it), the CLS stacking objective becomes
        weighted least squares, and the PLR moment + sandwich variance
        use weighted sums. The MSE used for ``inverse_risk`` /
        ``single_best`` weighting is also the weighted MSE.

    Returns
    -------
    DMLAveragingResult
        With the weighted :math:`\\hat\\theta`, SE, CI, and per-candidate
        diagnostics under ``result.model_info``.

    Examples
    --------
    >>> import statspai as sp
    >>> r = sp.dml_model_averaging(df, y="y", treat="d",
    ...                             covariates=[f"x{j}" for j in range(10)])
    >>> r.summary()
    >>> r.model_info["weights_g"]   # CLS stacking weights for ℓ̂(X) = E[Y|X]
    {"lasso": 0.42, "ridge": 0.0, "rf": 0.0, "gbm": 0.58}
    >>> r.model_info["weights_m"]   # CLS stacking weights for m̂(X) = E[D|X]
    {"lasso": 0.0, "ridge": 0.31, "rf": 0.0, "gbm": 0.69}
    """
    from scipy import stats as sp_stats

    for c in [y, treat] + list(covariates):
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found in data")
    valid_rules = {"short_stacking", "single_best", "inverse_risk", "equal"}
    if weight_rule not in valid_rules:
        raise ValueError(
            f"weight_rule must be one of {sorted(valid_rules)}, "
            f"got {weight_rule!r}"
        )
    if len(covariates) == 0:
        raise ValueError("At least one covariate required")

    cand = list(candidates) if candidates is not None else _default_candidates()
    if len(cand) == 0:
        raise ValueError("No candidate nuisance models supplied")

    # Drop rows with any missing value in y, treat, covariates *and*
    # sample_weight so the dropna mask aligns. NaNs would otherwise
    # silently poison the cross-fitted residuals (sklearn learners would
    # either raise or produce NaN predictions, after which
    # ``denom < 1e-12`` cannot detect the problem).
    cols = [y, treat] + list(covariates)
    work = data[cols].copy()
    if sample_weight is not None:
        if isinstance(sample_weight, str):
            if sample_weight not in data.columns:
                raise ValueError(
                    f"sample_weight column '{sample_weight}' not in data"
                )
            work["__sw__"] = data[sample_weight].astype(float).values
        else:
            arr = np.asarray(sample_weight, dtype=float)
            if arr.ndim != 1 or len(arr) != len(data):
                raise ValueError(
                    f"sample_weight must be 1-D of length {len(data)}; "
                    f"got shape {arr.shape}"
                )
            work["__sw__"] = arr
    clean = work.dropna()
    n_dropped = len(data) - len(clean)
    Y = clean[y].to_numpy(dtype=float)
    D = clean[treat].to_numpy(dtype=float)
    X = clean[list(covariates)].to_numpy(dtype=float)
    if "__sw__" in clean.columns:
        sw = clean["__sw__"].to_numpy(dtype=float)
        if np.any(sw < 0):
            raise ValueError("sample_weight must be non-negative")
        if not np.isfinite(sw).all():
            raise ValueError("sample_weight contains non-finite values")
        if sw.sum() <= 0:
            raise ValueError("sample_weight has zero total mass")
    else:
        sw = None
    n = len(Y)
    if n == 0:
        raise ValueError(
            "No rows remain after dropping missing values in y / treat / "
            "covariates. Check the input data."
        )
    if n != len(D) or n != X.shape[0]:  # pragma: no cover — defensive
        raise ValueError("Inconsistent row counts between y, treat, covariates")

    # --- Stage 1: fit every candidate, collect cross-fitted predictions and
    # per-candidate diagnostics. We need both predictions (for stacking)
    # and residuals (for per-candidate θ̂_k under non-stacking rules).
    yhat_mat: List[np.ndarray] = []
    dhat_mat: List[np.ndarray] = []
    thetas: List[float] = []
    ses: List[float] = []
    mses: List[float] = []
    labels: List[str] = []
    resids: List[Tuple[np.ndarray, np.ndarray, float]] = []

    for (ml_g, ml_m, label) in cand:
        yhat, dhat, y_r, d_r, mse_g, mse_m = _fit_candidate_plr(
            Y, D, X, ml_g, ml_m, n_folds, seed, sample_weight=sw,
        )
        if sw is None:
            denom = float(np.sum(d_r ** 2))
        else:
            denom = float(np.sum(sw * d_r ** 2))
        if denom < 1e-12:
            # Candidate produced a degenerate first stage (m̂ ≈ D in
            # mean square). Skip it so it does not poison the stacking
            # design matrix.
            continue
        if sw is None:
            theta_k = float(np.sum(d_r * y_r) / denom)
            psi = (y_r - theta_k * d_r) * d_r
            J = -np.mean(d_r ** 2)
            var_k = float(np.mean(psi ** 2) / (J ** 2) / n)
        else:
            theta_k = float(np.sum(sw * d_r * y_r) / denom)
            psi = (y_r - theta_k * d_r) * d_r
            # Weighted Z-estimator variance (sandwich) for candidate k.
            num = float(np.sum((sw ** 2) * (psi ** 2)))
            var_k = num / (denom ** 2)
        ses.append(np.sqrt(max(var_k, 0.0)))
        thetas.append(theta_k)
        mses.append(mse_g + mse_m)
        labels.append(label)
        resids.append((y_r, d_r, theta_k))
        yhat_mat.append(yhat)
        dhat_mat.append(dhat)

    if not labels:
        raise RuntimeError("No candidate produced a finite estimate")

    thetas_arr = np.array(thetas)
    ses_arr = np.array(ses)
    mses_arr = np.array(mses)
    yhat_arr = np.column_stack(yhat_mat)   # n × K
    dhat_arr = np.column_stack(dhat_mat)   # n × K

    z = sp_stats.norm.ppf(1 - alpha / 2)
    weights_g: Optional[np.ndarray] = None
    weights_m: Optional[np.ndarray] = None

    # --- Stage 2: combine. Two paths.
    if weight_rule == "short_stacking":
        # Solve CLS for each nuisance independently (paper §3, eq. 7).
        weights_g = _solve_cls_weights(Y, yhat_arr, sample_weight=sw)
        weights_m = _solve_cls_weights(D, dhat_arr, sample_weight=sw)
        y_resid_stack = Y - yhat_arr @ weights_g
        d_resid_stack = D - dhat_arr @ weights_m
        if sw is None:
            denom = float(np.sum(d_resid_stack ** 2))
        else:
            denom = float(np.sum(sw * d_resid_stack ** 2))
        if denom < 1e-12:
            raise RuntimeError(
                "short_stacking: stacked first stage is degenerate "
                f"(Σ d_resid² ≈ {denom:.2e}). All candidate m̂ predict D "
                "near-perfectly; consider richer covariates or a "
                "different roster."
            )
        if sw is None:
            theta_avg = float(np.sum(y_resid_stack * d_resid_stack) / denom)
            psi_stack = (y_resid_stack - theta_avg * d_resid_stack) * d_resid_stack
            J_stack = -float(np.mean(d_resid_stack ** 2))
            var_avg = (
                float(np.mean(psi_stack ** 2) / (J_stack ** 2) / n)
                if abs(J_stack) > 1e-10 else float("nan")
            )
        else:
            theta_avg = float(np.sum(sw * y_resid_stack * d_resid_stack) / denom)
            psi_stack = (y_resid_stack - theta_avg * d_resid_stack) * d_resid_stack
            # Weighted Z-estimator variance (sandwich): same recipe as
            # weighted PLR — Σ w² ψ² / (Σ w d²)².
            num = float(np.sum((sw ** 2) * (psi_stack ** 2)))
            var_avg = num / (denom ** 2)
        se_avg = float(np.sqrt(max(var_avg, 0.0)))
        # Reported "weights" = stacking weights for m (the moment-equation
        # denominator's nuisance) — the more identification-relevant set.
        # Keep both ``weights_g``/``weights_m`` for transparency.
        w = weights_m
    else:
        # Per-candidate-θ̂ averaging (legacy / convenience).
        if weight_rule == "equal":
            w = np.ones_like(thetas_arr) / len(thetas_arr)
        elif weight_rule == "single_best":
            w = np.zeros_like(thetas_arr)
            w[int(np.argmin(mses_arr))] = 1.0
        else:  # inverse_risk
            inv = 1.0 / np.clip(mses_arr, 1e-12, None)
            w = inv / inv.sum()

        theta_avg = float(np.sum(w * thetas_arr))

        # Variance: cross-candidate influence-function covariance.
        # Unweighted: each candidate k's IF is φ_k,i = ψ_k,i / J_k where
        # J_k = -E[d_resid_k²] and ψ_k,i = (y_resid_k,i - θ̂_k d_resid_k,i)
        # · d_resid_k,i. Storing φ/√n in ``phi_matrix`` makes
        # ``phi_matrix.T @ phi_matrix`` ≈ (1/n) Σ φ_k φ_l, then dividing
        # by n gives Var(θ̂_avg).
        # Weighted: per-candidate weighted Z-estimator gives
        # Var(θ̂_k) = Σ w_i² ψ_k,i² / (Σ w_i d_resid_k,i²)². For the
        # cross-product we use Σ_kl = Σ w_i² ψ_k,i ψ_l,i / (denom_k · denom_l).
        K_cand = len(thetas_arr)
        if sw is None:
            phi_matrix = np.zeros((n, K_cand))
            for k, (y_r, d_r, theta_k) in enumerate(resids):
                J_k = -np.mean(d_r ** 2)
                phi_matrix[:, k] = (y_r - theta_k * d_r) * d_r / (J_k * np.sqrt(n))
            cov_scaled = phi_matrix.T @ phi_matrix
            var_avg = float(w @ cov_scaled @ w) / n
        else:
            psi_matrix = np.zeros((n, K_cand))
            denoms = np.zeros(K_cand)
            for k, (y_r, d_r, theta_k) in enumerate(resids):
                psi_matrix[:, k] = (y_r - theta_k * d_r) * d_r
                denoms[k] = float(np.sum(sw * d_r ** 2))
            # Σ_kl = Σ_i sw_i² ψ_k,i ψ_l,i  /  (denom_k · denom_l)
            scaled_psi = (sw[:, None]) * psi_matrix
            sigma_kl = scaled_psi.T @ scaled_psi  # = Σ_i sw_i² ψ_k,i ψ_l,i
            sigma_kl = sigma_kl / np.outer(denoms, denoms)
            var_avg = float(w @ sigma_kl @ w)
        se_avg = float(np.sqrt(max(var_avg, 0.0)))

    ci = (theta_avg - z * se_avg, theta_avg + z * se_avg)
    pvalue = (
        float(2 * (1 - sp_stats.norm.cdf(abs(theta_avg / se_avg))))
        if se_avg > 0 else float("nan")
    )

    model_info: Dict[str, Any] = {
        "method": "Model-averaging DML (PLR)",
        "candidates": labels,
        "theta_k": dict(zip(labels, thetas_arr.tolist())),
        "se_k": dict(zip(labels, ses_arr.tolist())),
        "mse_k": dict(zip(labels, mses_arr.tolist())),
        "weights": dict(zip(labels, w.tolist())),
        "weight_rule": weight_rule,
        "n_folds": n_folds,
        "n_obs": int(n),
        "n_dropped_missing": int(n_dropped),
        "alpha": alpha,
        "citation": (
            "Ahrens, A., Hansen, C.B., Schaffer, M.E. and Wiemann, T. (2025). "
            "Model Averaging and Double Machine Learning. "
            "Journal of Applied Econometrics 40(3):249-269. DOI 10.1002/jae.3103."
        ),
    }
    if weights_g is not None and weights_m is not None:
        model_info["weights_g"] = dict(zip(labels, weights_g.tolist()))
        model_info["weights_m"] = dict(zip(labels, weights_m.tolist()))

    return DMLAveragingResult(
        method="DML (PLR) with model averaging",
        estimand="ATE",
        estimate=theta_avg,
        se=se_avg,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(n),
        model_info=model_info,
    )


# R-style alias
model_averaging_dml = dml_model_averaging
