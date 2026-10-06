"""Inference for forests fitted with the GRF engine.

Every statistic here uses *out-of-bag* CATE predictions for the training
rows, observation weights (``equalize_cluster_weights``), and cluster-robust
variances when the forest was fitted with clusters -- the three things an
in-sample, i.i.d. treatment gets wrong on panel or grouped data.

* :func:`cluster_robust_vcov` -- the heteroskedasticity- and cluster-robust
  covariance of a weighted least-squares fit, with the HC0/HC1/HC2/HC3
  conventions of the R ``sandwich`` package's ``vcovCL`` (Zeileis, Koell
  and Graham 2020), including the Bell-McCaffrey cluster adjustment for
  HC2/HC3.
* :func:`calibration_blp` -- the best-linear-predictor calibration test of
  the forest's CATE (Chernozhukov, Demirer, Duflo and Fernandez-Val), in the
  form used by ``grf::test_calibration``: residualized outcome on the
  residualized treatment times the mean and the demeaned OOB prediction.
* :func:`best_linear_projection` -- regression of doubly-robust scores on
  covariates.
* :func:`average_effect` -- AIPW ATE / ATT / ATC and the overlap-weighted
  partially linear effect, with cluster-robust standard errors.
* FE forests: :func:`fe_residuals` and the FE analogue of the calibration
  test (residualization on unit and period effects replaces the nuisance
  centring).  Doubly-robust averages are not defined for FE forests.

References
----------
[@athey2019generalized], [@chernozhukov2025generic], [@zeileis2020sandwich]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #


def is_grf_forest(forest: Any) -> bool:
    return (
        getattr(forest, "_engine", None) is not None
        and getattr(forest, "_oob_tau", None) is not None
    )


def require_finite_oob(forest: Any, context: str) -> None:
    tau = np.asarray(forest._oob_tau, dtype=float)
    missing = int(np.sum(~np.isfinite(tau)))
    if missing:
        raise DataInsufficient(
            f"{context}: {missing} training row(s) have no out-of-bag CATE "
            "prediction, so in-sample inference is undefined.",
            recovery_hint="Refit with more trees (n_estimators).",
            diagnostics={"n_rows_without_oob_prediction": missing},
        )


def is_fe_forest(forest: Any) -> bool:
    return is_grf_forest(forest) and getattr(forest, "fe", None) is not None


def training_cate(forest: Any, X: Optional[np.ndarray]) -> Tuple[np.ndarray, bool]:
    """CATE for inference rows: OOB when ``X`` is the training design.

    Returns ``(tau, is_training)``.  ``X`` counts as the training design when
    it is ``None`` or numerically identical to the fitted covariates, so a
    caller passing the training matrix back in still gets honest OOB
    predictions rather than predictions from trees that saw those rows.
    """
    X_train = np.asarray(forest._X_original, dtype=float)
    if X is None:
        return np.asarray(forest._oob_tau, dtype=float), True
    X_ = np.asarray(X, dtype=float)
    if X_.shape == X_train.shape and np.array_equal(X_, X_train):
        return np.asarray(forest._oob_tau, dtype=float), True
    return np.asarray(forest.effect(X_), dtype=float).ravel(), False


def _cluster_codes(forest: Any, n: int) -> np.ndarray:
    clusters = getattr(forest, "_clusters", None)
    if clusters is None:
        return np.arange(n, dtype=np.int64)
    return np.asarray(clusters, dtype=np.int64)


def _weights(forest: Any, n: int) -> np.ndarray:
    w = getattr(forest, "_observation_weight", None)
    if w is None or len(w) != n:
        return np.full(n, 1.0 / n)
    return np.asarray(w, dtype=float)


def cluster_robust_vcov(
    design: np.ndarray,
    resid: np.ndarray,
    *,
    weights: Optional[np.ndarray] = None,
    clusters: Optional[np.ndarray] = None,
    vcov_type: str = "HC3",
) -> np.ndarray:
    """Sandwich covariance of a (weighted) least-squares fit.

    Matches ``sandwich::vcovCL(lm(..., weights = weights), cluster =
    clusters, type = vcov_type)`` with ``cadjust = TRUE``; ``clusters=None``
    treats each row as its own cluster.  ``vcov_type`` is one of ``HC0``,
    ``HC1``, ``HC2``, ``HC3``.
    """
    X = np.asarray(design, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    e = np.asarray(resid, dtype=float).ravel()
    n, k = X.shape
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).ravel()
    vt = vcov_type.upper()
    if vt not in ("HC0", "HC1", "HC2", "HC3"):
        raise MethodIncompatibility(
            f"cluster_robust_vcov(): unsupported vcov_type {vcov_type!r}.",
            recovery_hint="Use 'HC0', 'HC1', 'HC2' or 'HC3'.",
        )
    XtWX = X.T @ (X * w[:, None])
    bread = np.linalg.pinv(XtWX)
    we = w * e
    if clusters is None:
        codes = np.arange(n)
    else:
        _, codes = np.unique(np.asarray(clusters), return_inverse=True)
        codes = codes.ravel()
    G = int(codes.max()) + 1
    if G < 2:
        raise DataInsufficient(
            "cluster_robust_vcov(): at least two clusters are required.",
            recovery_hint="Use data from more than one cluster.",
        )
    if vt in ("HC0", "HC1"):
        scores = X * we[:, None]
        if G < n:
            summed = np.zeros((G, k))
            np.add.at(summed, codes, scores)
        else:
            summed = scores
        meat = (G / (G - 1)) * (summed.T @ summed)
        if vt == "HC1":
            meat *= (n - 1) / (n - k)
    else:
        if G == n:
            h = w * np.einsum("ij,jk,ik->i", X, bread, X)
            denom = np.sqrt(1.0 - h) if vt == "HC2" else (1.0 - h)
            adj = we / denom
            scores = X * adj[:, None]
            meat = scores.T @ scores
        else:
            order = np.argsort(codes, kind="mergesort")
            bounds = np.flatnonzero(np.diff(codes[order])) + 1
            u = np.zeros((G, k))
            for g, idx in enumerate(np.split(order, bounds)):
                Xg = X[idx]
                H = Xg @ bread @ Xg.T * w[idx][None, :]
                M = np.eye(idx.size) - H
                if vt == "HC3":
                    adj = np.linalg.solve(M, we[idx])
                else:
                    vals, vecs = np.linalg.eig(M)
                    inv_sqrt = (vecs * (vals.astype(complex) ** -0.5)) @ np.linalg.inv(
                        vecs
                    )
                    adj = np.real(inv_sqrt @ we[idx])
                u[g] = Xg.T @ adj
            meat = u.T @ u
    return np.asarray(bread @ meat @ bread)


def _coef_table(
    beta: np.ndarray,
    vcov: np.ndarray,
    names: list,
    alpha: float,
    null: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    null_arr = np.zeros_like(beta) if null is None else np.asarray(null, dtype=float)
    z = float(stats.norm.ppf(1 - alpha / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (beta - null_arr) / se
    out = pd.DataFrame({"coef": beta, "se": se, "null": null_arr, "t": t}, index=names)
    out["p"] = 2 * stats.norm.sf(np.abs(out["t"].to_numpy()))
    out["ci_low"] = beta - z * se
    out["ci_high"] = beta + z * se
    return out


# --------------------------------------------------------------------------- #
#  Residualization for FE forests
# --------------------------------------------------------------------------- #


def fe_residuals(forest: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Full-sample within-transformed ``(Y - Y_hat, W - W_hat)`` of an FE forest."""
    from . import _grf_engine as engine

    unit = np.asarray(forest._fe_unit, dtype=np.int64)
    time = np.asarray(forest._fe_time, dtype=np.int64)
    n = unit.size
    Yc = np.asarray(forest._Y_original, dtype=float) - forest._m_insample
    Wc = np.asarray(forest._T_original, dtype=float) - forest._e_insample
    G = np.ones(n)
    idx = np.arange(n, dtype=np.int64)
    n_u = int(unit.max()) + 1
    n_t = int(time.max()) + 1
    outs = []
    for V in (Yc, Wc):
        out = np.zeros(n)
        sweeps = engine._fe_residualize(
            idx,
            np.ascontiguousarray(V),
            G,
            unit,
            time,
            np.zeros(n_u),
            np.zeros(n_u),
            np.zeros(n_t),
            np.zeros(n_t),
            out,
            10000,
            1e-12,
        )
        if sweeps < 0:
            raise DataInsufficient(
                "Two-way within transformation did not converge.",
                recovery_hint="Check the panel for disconnected unit-period sets.",
            )
        outs.append(out)
    return outs[0], outs[1]


# --------------------------------------------------------------------------- #
#  Calibration test
# --------------------------------------------------------------------------- #


def calibration_blp(
    forest: Any, alpha: float = 0.05, vcov_type: str = "HC3"
) -> pd.DataFrame:
    """Best-linear-predictor calibration test on OOB predictions.

    Regresses ``Y - Y_hat`` (FE forests: its within transformation) on
    ``(W - W_hat) * mean(tau)`` and ``(W - W_hat) * (tau - mean(tau))``
    without an intercept, using observation weights and a cluster-robust
    ``vcov_type`` covariance.  ``beta_mean`` near 1 says the average
    prediction is calibrated; ``beta_differential`` significantly above 0
    says the forest's ranking captures real heterogeneity, and its value is
    the slope by which predictions should be rescaled around their mean.

    For FE forests the regression uses globally within-transformed variables
    (the residualization of Aytug 2026, arXiv:2607.22896, Sec. 4).  Under
    the null of a homogeneous effect that regression is correctly
    specified, so the heterogeneity test keeps its size; with heterogeneous
    effects ``beta_differential`` is a best-linear-predictor slope, not an
    exact de-attenuation factor.
    """
    require_finite_oob(forest, "calibration_test()")
    n = int(len(forest._Y_original))
    tau = np.asarray(forest._oob_tau, dtype=float)
    w = _weights(forest, n)
    clusters = _cluster_codes(forest, n)
    if is_fe_forest(forest):
        target, w_res = fe_residuals(forest)
    else:
        target = np.asarray(forest._Y_original, dtype=float) - forest._m_insample
        w_res = np.asarray(forest._T_original, dtype=float) - forest._e_insample
    # The regression, covariance and grf's reporting rule (t against 0,
    # one-sided p from Student t with n - 2 df) are the pure operator
    # ``forest_inference.grf_calibration``; this function only chooses the
    # GRF-engine inputs (OOB predictions, weights, clusters, FE residuals).
    from .forest_inference import grf_calibration

    zeros = np.zeros(n)
    out = grf_calibration(
        Y=target,
        W=w_res,
        Y_hat=zeros,
        W_hat=zeros,
        tau_hat=tau,
        vcov_type=vcov_type,
        alpha=alpha,
        weights=w,
        clusters=clusters if getattr(forest, "_clusters", None) is not None else None,
    )
    clustered = getattr(forest, "_clusters", None) is not None
    out.attrs["method"] = (
        "BLP calibration on out-of-bag predictions, "
        f"{vcov_type} {'cluster-' if clustered else ''}robust SE"
    )
    return out


# --------------------------------------------------------------------------- #
#  Doubly-robust scores and averages
# --------------------------------------------------------------------------- #


def _require_dr(forest: Any, context: str) -> None:
    if is_fe_forest(forest):
        raise MethodIncompatibility(
            f"{context} is not defined for causal forests with fixed effects: "
            "the doubly-robust score needs a propensity E[W | X], which a "
            "within-unit design does not have.",
            recovery_hint=(
                "Report the distribution of oob_effect(), calibrate it with "
                "sp.calibration_test(), or use sp.did_forest() for "
                "group-time averages with valid standard errors."
            ),
            alternative_functions=["sp.did_forest", "sp.calibration_test"],
        )


def continuous_debiasing_weights(forest: Any) -> np.ndarray:
    """``(W - W_hat) / V_hat(X)`` for a non-binary treatment.

    ``V_hat`` is an out-of-bag regression forest of ``(W - W_hat)^2`` on the
    effect modifiers (500 trees, same clusters and seed), the Riesz
    representer used by grf's ``get_scores`` for continuous treatments.
    Cached on the forest.
    """
    cached = getattr(forest, "_continuous_debias_weights", None)
    if cached is not None:
        return np.asarray(cached)
    from . import _grf_engine as engine
    from ._grf_family import with_stream

    W = np.asarray(forest._T_original, dtype=float)
    resid = W - np.asarray(forest._e_insample, dtype=float)
    X = np.asarray(forest._X_original, dtype=float)
    seed = (
        0 if getattr(forest, "random_state", None) is None else int(forest.random_state)
    )
    vforest = engine.train_forest(
        X,
        resid**2,
        kind=engine.KIND_REGRESSION,
        num_trees=500,
        ci_group_size=1,
        clusters=getattr(forest, "_clusters", None),
        seed=with_stream({"seed": seed}, "var_w")["seed"],
        n_jobs=getattr(forest, "n_jobs", 1),
    )
    V_hat, _ = vforest.predict_oob(X)
    if not np.all(np.isfinite(V_hat)) or np.any(V_hat <= 0):
        raise DataInsufficient(
            "Doubly-robust scores for a continuous treatment need a positive "
            "out-of-bag estimate of Var(W | X) for every row.",
            recovery_hint="Use target_sample='overlap', or a larger sample.",
        )
    weights = resid / V_hat
    try:
        forest._continuous_debias_weights = weights
    except AttributeError:  # pragma: no cover - frozen namespaces in tests
        pass
    return np.asarray(weights)


def dr_scores(
    forest: Any, tau: np.ndarray, clip: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Doubly-robust scores; returns (scores, e, m, W).

    Binary treatment: the AIPW score with propensity ``e``.  Continuous
    treatment: ``tau + (W - W_hat) / V_hat(X) * (Y - Y_hat - tau (W - W_hat))``.
    """
    Y = np.asarray(forest._Y_original, dtype=float)
    W = np.asarray(forest._T_original, dtype=float)
    m = np.asarray(forest._m_insample, dtype=float)
    e = np.asarray(forest._e_insample, dtype=float)
    if not np.all(np.isin(np.unique(W), (0.0, 1.0))):
        debias = continuous_debiasing_weights(forest)
        scores = tau + debias * (Y - (m + tau * (W - e)))
        return scores, e, m, W
    if clip > 0:
        e = np.clip(e, clip, 1.0 - clip)
    if np.any((e <= 0) | (e >= 1)):
        raise DataInsufficient(
            "Doubly-robust scores need estimated propensities strictly "
            "between 0 and 1.",
            recovery_hint="Use clip > 0 or target_sample='overlap'.",
        )
    mu = m + (W - e) * tau
    scores = tau + (W - e) / (e * (1.0 - e)) * (Y - mu)
    return scores, e, m, W


def _clustered_mean_var(
    values: np.ndarray, w: np.ndarray, clusters: Optional[np.ndarray]
) -> float:
    """grf's variance of a weighted mean of (centred) scores."""
    if clusters is None:
        pos = w > 0
        n_pos = int(pos.sum())
        return float(np.sum(w**2 * values**2) / np.sum(w) ** 2 * n_pos / (n_pos - 1))
    G = int(clusters.max()) + 1
    summed = np.bincount(clusters, weights=w * values, minlength=G)
    cluster_w = np.bincount(clusters, weights=w, minlength=G)
    n_adj = int(np.sum(cluster_w > 0))
    return float(np.sum(summed**2) / np.sum(w) ** 2 * n_adj / (n_adj - 1))


def average_effect(
    forest: Any,
    target_sample: str,
    alpha: float,
    clip: float,
) -> Dict[str, Any]:
    """AIPW averages on the training sample with cluster-robust SEs."""
    _require_dr(forest, "average_treatment_effect()")
    require_finite_oob(forest, "average_treatment_effect()")
    n = int(len(forest._Y_original))
    tau = np.asarray(forest._oob_tau, dtype=float)
    w = _weights(forest, n)
    clustered = getattr(forest, "_clusters", None) is not None
    clusters = _cluster_codes(forest, n) if clustered else None
    z = float(stats.norm.ppf(1 - alpha / 2))
    W = np.asarray(forest._T_original, dtype=float)
    binary = bool(np.all(np.isin(np.unique(W), (0.0, 1.0))))

    if target_sample == "overlap":
        Y_res = np.asarray(forest._Y_original, dtype=float) - forest._m_insample
        W_res = W - forest._e_insample
        design = np.column_stack([np.ones(n), W_res])
        XtWX = design.T @ (design * w[:, None])
        beta = np.linalg.solve(XtWX, design.T @ (w * Y_res))
        resid = Y_res - design @ beta
        V = cluster_robust_vcov(
            design,
            resid,
            weights=w,
            clusters=clusters,
            vcov_type="HC1" if clustered else "HC3",
        )
        estimate, se = float(beta[1]), float(np.sqrt(V[1, 1]))
        e = np.asarray(forest._e_insample, dtype=float)
        estimand, ess, method = (
            "ATO",
            float(np.sum(w) ** 2 / np.sum(w**2)),
            "partially_linear",
        )
    else:
        if not binary and target_sample != "all":
            raise MethodIncompatibility(
                "average_treatment_effect(): ATT/ATC averages need a binary "
                "treatment.",
                recovery_hint="Use target_sample='all' or 'overlap'.",
            )
        scores, e, m, _ = dr_scores(forest, tau, clip)
        if target_sample == "all":
            estimate = float(np.sum(w * scores) / np.sum(w))
            se = float(np.sqrt(_clustered_mean_var(scores - estimate, w, clusters)))
            estimand, ess = "ATE", float(np.sum(w) ** 2 / np.sum(w**2))
        else:
            treated = W == 1
            control = ~treated
            if not treated.any() or not control.any():
                raise DataInsufficient(
                    "average_treatment_effect(): both arms must be non-empty.",
                    recovery_hint="Pass a sample with treated and control units.",
                )
            idx = treated if target_sample == "treated" else control
            tau_raw = float(np.sum(w[idx] * tau[idx]) / np.sum(w[idx]))
            tau_var = float(
                np.sum(w[idx] ** 2 * (tau[idx] - tau_raw) ** 2) / np.sum(w[idx]) ** 2
            )
            if target_sample == "treated":
                g_c = e[control] / (1.0 - e[control])
                g_t = np.ones(int(treated.sum()))
            else:
                g_c = np.ones(int(control.sum()))
                g_t = (1.0 - e[treated]) / e[treated]
            gamma = np.zeros(n)
            gamma[control] = g_c / np.sum(w[control] * g_c) * np.sum(w)
            gamma[treated] = g_t / np.sum(w[treated] * g_t) * np.sum(w)
            Y = np.asarray(forest._Y_original, dtype=float)
            mu0 = m - e * tau
            mu1 = m + (1.0 - e) * tau
            corr = W * gamma * (Y - mu1) - (1.0 - W) * gamma * (Y - mu0)
            dr = float(np.sum(w * corr) / np.sum(w))
            estimate = tau_raw + dr
            se = float(np.sqrt(tau_var + _clustered_mean_var(corr, w, clusters)))
            estimand = "ATT" if target_sample == "treated" else "ATC"
            ess = float(idx.sum())
        method = "aipw" if binary else "aipw_continuous"
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": target_sample,
        "estimand": estimand,
        "method": method,
        "effective_sample_size": ess,
        "n": n,
        "n_clusters": None if clusters is None else int(clusters.max()) + 1,
        "alpha": float(alpha),
        "pscore_min": float(np.min(e)),
        "pscore_max": float(np.max(e)),
        "cate_source": "out_of_bag",
    }


def best_linear_projection(
    forest: Any,
    A: Optional[np.ndarray],
    names: list,
    alpha: float,
    clip: float,
    vcov_type: str = "HC3",
    covariates: Any = "none",
) -> pd.DataFrame:
    """Regress AIPW scores on ``(1, A)`` with a cluster-robust covariance.

    FE forests regress imputation scores on ``(1, A)`` over treated cells
    instead (:func:`._fe_imputation.best_linear_projection_fe`).
    """
    if is_fe_forest(forest):
        from ._fe_imputation import best_linear_projection_fe

        return best_linear_projection_fe(
            forest, A, names[1:], alpha=alpha, covariates=covariates
        )
    _require_dr(forest, "best_linear_projection()")
    require_finite_oob(forest, "best_linear_projection()")
    n = int(len(forest._Y_original))
    tau = np.asarray(forest._oob_tau, dtype=float)
    scores, *_ = dr_scores(forest, tau, clip)
    w = _weights(forest, n)
    design = np.ones((n, 1)) if A is None else np.column_stack([np.ones(n), A])
    XtWX = design.T @ (design * w[:, None])
    beta = np.linalg.lstsq(XtWX, design.T @ (w * scores), rcond=None)[0]
    resid = scores - design @ beta
    V = cluster_robust_vcov(
        design,
        resid,
        weights=w,
        clusters=(
            _cluster_codes(forest, n)
            if getattr(forest, "_clusters", None) is not None
            else None
        ),
        vcov_type=vcov_type,
    )
    out = _coef_table(beta, V, names, alpha)
    out = out.drop(columns=["null"]).rename(
        columns={"ci_low": "ci_lower", "ci_high": "ci_upper"}
    )
    out.attrs["method"] = f"AIPW scores on OOB CATE, {vcov_type} robust SE"
    out.attrs["vcov"] = pd.DataFrame(V, index=names, columns=names)
    return out
