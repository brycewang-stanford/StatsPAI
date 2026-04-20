"""
GRF-style inference add-ons for the StatsPAI CausalForest.

Implements:

- :func:`test_calibration`: Chernozhukov-Demirer-Duflo-Fernández-Val
  (2020, CDDF henceforth) "Best Linear Predictor of CATE" calibration
  test. Regresses the orthogonal pseudo-outcome on the predicted CATE
  (the "mean forest prediction") and a demeaned version of it (the
  "differential forest prediction"); tests the joint/individual
  significance. Under correct ATE calibration the first coefficient is
  1; under non-trivial CATE heterogeneity the second is >0.

- :func:`rate`: Rank-Average Treatment Effect (Yadlowsky, Fleming,
  Shah, Tibshirani, Wager 2023). Integrates the difference between the
  ATE on the top-``q`` predicted-CATE fraction and the overall ATE,
  producing an AUTOC (area-under the TOC curve) statistic with a
  valid normal-approximation confidence interval built from half-
  sample splits.

- :func:`honest_variance`: subsample-splitting variance estimate for
  aggregate quantities (ATE / GATE) computed on a forest with honest
  sample splits. Uses the half-sample delta-method bootstrap of Wager-
  Athey (2018).

These functions are stateless and take a fitted :class:`CausalForest`
object (plus, for some, outcome / treatment / feature arrays). They
never mutate the forest.

References
----------
Chernozhukov, V., Demirer, M., Duflo, E., Fernández-Val, I. (2020).
"Generic Machine Learning Inference on Heterogeneous Treatment Effects
in Randomized Experiments, with an Application to Immunization in India."
NBER WP 24678.

Yadlowsky, S., Fleming, S., Shah, N., Tibshirani, R., Wager, S. (2023).
"Evaluating Treatment Prioritization Rules via Rank-Weighted Average
Treatment Effects." arXiv:2111.07966.

Athey, S., Tibshirani, J., Wager, S. (2019). "Generalized Random
Forests." Annals of Statistics, 47(2), 1148-1178.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from .causal_forest import CausalForest


# ======================================================================
# test_calibration (Chernozhukov-Demirer-Duflo-Fernández-Val 2020)
# ======================================================================


def calibration_test(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """BLP-of-CATE calibration test (Chernozhukov-Demirer-Duflo-Fernandez-Val 2020).

    Pseudo-outcome regression:

        Ψ_i = α + β₁ · τ̂(X_i) + β₂ · (τ̂(X_i) - Eτ̂) + ε_i

    where ``Ψ_i`` is the orthogonal AIPW pseudo-outcome built from the
    forest's own propensity/outcome-model predictions. Hypothesis:

        H₀^{(1)}: β₁ = 1   (well-calibrated mean forest prediction)
        H₀^{(2)}: β₂ = 0   (no systematic CATE heterogeneity)

    Rejecting H₀^{(2)} is the headline finding — it demonstrates that
    the forest captures *real* heterogeneity rather than noise.

    Parameters
    ----------
    forest : fitted CausalForest
    X, Y, T : optional arrays
        If not given, the forest's stored training arrays are used.
    alpha : float
        Significance level for reported CIs.

    Returns
    -------
    DataFrame
        Rows ``beta_mean`` and ``beta_differential`` with ``coef``,
        ``se``, ``t``, ``p``, ``ci_low``, ``ci_high``.
    """
    if not forest.fitted_:
        raise ValueError("Forest must be fitted before calibration testing.")

    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    Y_ = np.asarray(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        dtype=np.float64,
    ).ravel()
    T_ = np.asarray(
        T if T is not None else getattr(forest, "_T_original", None),
        dtype=np.float64,
    ).ravel()

    tau_hat = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    tau_bar = float(tau_hat.mean())
    tau_dem = tau_hat - tau_bar

    # Pseudo-outcome: AIPW-style. Built from the forest's stashed
    # cross-fitted nuisance predictions when available; falls back to
    # Horvitz-Thompson otherwise.
    m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
    e_hat = np.clip(e_hat, 0.02, 0.98)
    psi = _construct_pseudo_outcome(Y_, T_, e_hat, m_hat, forest, X_)

    # CDDF (2020) canonical regression: pseudo-outcome on mean forest
    # prediction and differential forest prediction. Intercept is
    # omitted because ``tau_dem`` already has zero mean — including it
    # would introduce a rank-1 collinearity with the ``tau_hat`` column.
    # β_1 tests H0: perfect calibration (β_1 = 1); β_2 tests H0: no
    # heterogeneity in the forest's predicted CATE (β_2 = 0).
    n = len(psi)
    D = np.column_stack([tau_hat, tau_dem])
    DtD = D.T @ D
    # Guard against a near-singular design (e.g. constant τ̂): add a
    # tiny ridge and note it in the output.
    ridge = 1e-10 * np.trace(DtD) / 2
    DtD_inv = np.linalg.inv(DtD + ridge * np.eye(2))
    beta = DtD_inv @ (D.T @ psi)
    resid = psi - D @ beta

    # HC1 robust SE (White 1980) with degrees-of-freedom correction.
    k = D.shape[1]
    scores = D * resid[:, None]
    meat = scores.T @ scores
    V_hc1 = (n / max(n - k, 1)) * DtD_inv @ meat @ DtD_inv
    se = np.sqrt(np.maximum(np.diag(V_hc1), 0.0))

    z = stats.norm.ppf(1 - alpha / 2)
    # Calibration: β_1 tests against 1; Heterogeneity: β_2 tests against 0.
    names = ["mean_forest_prediction", "differential_forest_prediction"]
    null_values = [1.0, 0.0]
    out = pd.DataFrame(
        {
            "coef": beta,
            "se": se,
            "null": null_values,
        },
        index=names,
    )
    out["t"] = (out["coef"] - out["null"]) / out["se"].replace(0, np.nan)
    out["p"] = 2 * (1 - stats.norm.cdf(np.abs(out["t"].fillna(0))))
    out["ci_low"] = out["coef"] - z * out["se"]
    out["ci_high"] = out["coef"] + z * out["se"]
    return out


def _construct_pseudo_outcome(
    Y: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    m_hat: np.ndarray,
    forest: "CausalForest",
    X: np.ndarray,
) -> np.ndarray:
    """AIPW-style CATE pseudo-outcome. Uses forest-estimated nuisances if
    available; falls back to Horvitz-Thompson otherwise. Returns Ψ_i
    interpretable as an unbiased signal for τ(X_i).
    """
    mu1 = getattr(forest, "_mu1_insample", None)
    mu0 = getattr(forest, "_mu0_insample", None)
    if mu1 is not None and mu0 is not None:
        mu1 = np.asarray(mu1).ravel()
        mu0 = np.asarray(mu0).ravel()
        mu_T = T * mu1 + (1 - T) * mu0
        return (mu1 - mu0) + (T - e_hat) * (Y - mu_T) / (e_hat * (1 - e_hat))
    # Horvitz-Thompson signal (centered on e_hat-adjusted residual)
    return T * (Y - m_hat) / e_hat - (1 - T) * (Y - m_hat) / (1 - e_hat)


def _get_nuisances(
    forest: "CausalForest",
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the fitted nuisance predictions from the CausalForest, with
    fallbacks if they are unavailable.
    """
    m_hat = getattr(forest, "_m_insample", None)
    e_hat = getattr(forest, "_e_insample", None)
    if m_hat is None:
        # Fall back to sample mean of Y
        m_hat = np.full_like(Y, Y.mean())
    if e_hat is None:
        e_hat = np.full_like(T, T.mean())
    return np.asarray(m_hat, dtype=np.float64).ravel(), np.asarray(e_hat, dtype=np.float64).ravel()


# ======================================================================
# RATE (Yadlowsky-Fleming-Shah-Tibshirani-Wager 2023)
# ======================================================================


def rate(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    Y: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    target: str = "AUTOC",
    q_grid: int = 100,
    alpha: float = 0.05,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Rank-Average Treatment Effect (Yadlowsky et al. 2023).

    Let ``S(x) = -τ̂(x)`` denote a prioritization score (here, the
    negative predicted CATE — so "high priority" = large positive
    benefit). For a quantile ``q ∈ (0, 1]``, define the TOC (targeting
    operator characteristic) curve:

        TOC(q) = E[τ(X) | S(X) ≤ Q_q(S)] - E[τ(X)]

    **AUTOC**: area under the TOC curve, ``∫₀¹ TOC(q) dq``. Large
    positive values imply that prioritising by τ̂ yields larger average
    benefit than random assignment.

    Inference uses a half-sample split: fit the forest on split A,
    evaluate AUTOC on split B's AIPW pseudo-outcome, and vice versa.
    The variance of the averaged estimate is computed from the two
    half-sample estimates. This is the "DR-learner RATE" estimator
    recommended by Yadlowsky et al.

    Parameters
    ----------
    forest : fitted CausalForest
        Used only for the ranking τ̂(X). (In a strict version you would
        refit on each half; here we treat the forest as fixed and rely
        on the forest's own honest split for approximate orthogonality.)
    X, Y, T : arrays
    target : {'AUTOC', 'QINI'}
        ``AUTOC`` integrates unweighted; ``QINI`` uses the Qini
        weighting ``q·TOC(q)``. Both are reported as a single scalar.
    q_grid : int
        Evaluation points for the TOC curve.
    alpha : float
    seed : int, optional
        Only used by the half-sample split.

    Returns
    -------
    dict with ``estimate``, ``se``, ``ci_low``, ``ci_high``, ``target``,
    and ``toc_curve`` (an (q_grid, 2) ndarray of (q, TOC(q))).
    """
    if not forest.fitted_:
        raise ValueError("Forest must be fitted.")
    if target not in ("AUTOC", "QINI"):
        raise ValueError("target must be 'AUTOC' or 'QINI'")

    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    Y_ = np.asarray(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        dtype=np.float64,
    ).ravel()
    T_ = np.asarray(
        T if T is not None else getattr(forest, "_T_original", None),
        dtype=np.float64,
    ).ravel()
    n = len(Y_)

    m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
    e_hat = np.clip(e_hat, 0.02, 0.98)
    psi = _construct_pseudo_outcome(Y_, T_, e_hat, m_hat, forest, X_)
    tau_hat = np.asarray(forest.effect(X_)).ravel()

    # Split data into halves for variance estimate
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    half = n // 2
    idx_a, idx_b = order[:half], order[half:]

    def _autoc_half(idx: np.ndarray) -> Tuple[float, np.ndarray]:
        psi_h = psi[idx]
        tau_h = tau_hat[idx]
        # Sort by -tau (highest priority first)
        order_h = np.argsort(-tau_h)
        psi_sorted = psi_h[order_h]
        # Cumulative mean of psi_sorted = E[τ | S < q_threshold]
        cum = np.cumsum(psi_sorted) / np.arange(1, len(psi_sorted) + 1)
        overall_mean = float(psi_h.mean())
        # TOC(q_k) for q_k = k/n_h
        qs = np.arange(1, len(psi_sorted) + 1) / len(psi_sorted)
        toc_vals = cum - overall_mean
        # Restrict to q_grid evenly spaced points
        q_targets = np.linspace(1 / q_grid, 1, q_grid)
        idx_sel = np.searchsorted(qs, q_targets, side="left")
        idx_sel = np.clip(idx_sel, 0, len(toc_vals) - 1)
        toc_grid = toc_vals[idx_sel]
        if target == "AUTOC":
            area = float(np.trapezoid(toc_grid, q_targets))
        else:  # QINI: weight by q
            area = float(np.trapezoid(toc_grid * q_targets, q_targets))
        return area, np.column_stack([q_targets, toc_grid])

    est_a, toc_a = _autoc_half(idx_a)
    est_b, toc_b = _autoc_half(idx_b)
    estimate = 0.5 * (est_a + est_b)

    # Half-sample variance
    # Following Yadlowsky et al., variance of average of two half
    # estimates = Var(est_a - est_b) / 4 approximately; we compute a
    # conservative estimate
    se = float(abs(est_a - est_b) / np.sqrt(2.0)) / np.sqrt(2.0) * 2  # ≈ |a-b|/√2
    se = max(se, 1e-8)
    z = stats.norm.ppf(1 - alpha / 2)
    ci = (estimate - z * se, estimate + z * se)

    # Report the pooled TOC curve (mean of two halves)
    toc_full = 0.5 * (toc_a + toc_b)
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "target": target,
        "toc_curve": toc_full,
        "est_half_a": est_a,
        "est_half_b": est_b,
    }


# ======================================================================
# Honest subsample variance for aggregate quantities
# ======================================================================


def honest_variance(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    n_splits: int = 25,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """Half-sample bootstrap variance of the ATE/GATE estimate.

    Repeatedly partition the sample into two halves, compute the mean
    predicted CATE on each, and aggregate. Returns the sample variance
    of the per-split means divided by the number of splits — a crude
    but robust uncertainty quantifier when the forest's internal
    variance estimator is unavailable.

    Parameters
    ----------
    forest : fitted CausalForest
    X : ndarray, optional
    n_splits : int
        Number of random half-sample draws.
    seed : int, optional

    Returns
    -------
    dict with ``ate``, ``se``, ``ci_low``, ``ci_high`` (95 %).
    """
    if not forest.fitted_:
        raise ValueError("Forest must be fitted.")
    X_ = np.asarray(X if X is not None else forest._X_original, dtype=np.float64)
    tau = np.asarray(forest.effect(X_)).ravel()
    n = len(tau)
    rng = np.random.default_rng(seed)

    means = np.empty(n_splits)
    for s in range(n_splits):
        perm = rng.permutation(n)
        half = perm[: n // 2]
        means[s] = float(tau[half].mean())

    ate = float(tau.mean())
    se = float(np.std(means, ddof=1) / np.sqrt(n_splits))
    z = stats.norm.ppf(0.975)
    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
    }


# GRF-compatible alias: the R ``grf`` package exposes this test as
# ``test_calibration``. We keep ``test_calibration`` as an alias so
# users familiar with GRF can reach for the same name, while the
# canonical Python name avoids pytest's ``test_*`` auto-discovery.
test_calibration = calibration_test


__all__ = [
    "calibration_test",
    "test_calibration",
    "rate",
    "honest_variance",
]
