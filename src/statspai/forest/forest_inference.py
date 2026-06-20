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
- :func:`average_treatment_effect`: GRF-style ATE/ATT/ATC/ATO aggregation
  of CATE predictions with effective sample size and normal CIs.
- :func:`forest_diagnostics`: overlap and CATE-distribution diagnostics.

These functions are stateless and take a fitted :class:`CausalForest`
object (plus, for some, outcome / treatment / feature arrays). They
never mutate the forest.

References
----------
Chernozhukov, V., Demirer, M., Duflo, E., Fernández-Val, I. (2020).
"Generic Machine Learning Inference on Heterogeneous Treatment Effects
in Randomized Experiments, with an Application to Immunization in India."
NBER WP 24678. [@chernozhukov2020generic]

Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., Wager, S. (2021).
"Evaluating Treatment Prioritization Rules via Rank-Weighted Average
Treatment Effects." arXiv:2111.07966.

Athey, S., Tibshirani, J., Wager, S. (2019). "Generalized Random
Forests." Annals of Statistics, 47(2), 1148-1178. [@athey2019surrogate]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient, MethodIncompatibility

if TYPE_CHECKING:
    from .causal_forest import CausalForest


def _require_fitted_forest(forest: "CausalForest", context: str) -> None:
    """Raise a StatsPAI taxonomy error when an inference helper is unfitted."""
    if not getattr(forest, "fitted_", False):
        raise MethodIncompatibility(
            f"{context} requires a fitted forest.",
            recovery_hint="Call fit() before running forest inference.",
        )


def _validate_alpha(alpha: float, context: str) -> float:
    """Validate a confidence/significance level."""
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: alpha must be a finite scalar.",
            recovery_hint="Use an alpha value in the open interval (0, 1).",
            diagnostics={"alpha": alpha},
        ) from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise MethodIncompatibility(
            f"{context}: alpha must be in the open interval (0, 1).",
            recovery_hint="Use an alpha value such as 0.05.",
            diagnostics={"alpha": alpha},
        )
    return alpha_value


def _prepare_forest_features(
    forest: "CausalForest",
    X: Optional[np.ndarray],
    context: str,
) -> np.ndarray:
    """Validate inference features with the forest's fitted schema."""
    if X is None:
        return np.asarray(forest._X_original, dtype=np.float64)
    if callable(getattr(forest, "_prepare_effect_matrix", None)):
        return np.asarray(
            forest._prepare_effect_matrix(X, context=context),
            dtype=np.float64,
        )
    try:
        return np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: X must be numeric.",
            recovery_hint="Pass X shaped (n_samples, n_features).",
        ) from exc


def _prepare_forest_vector(
    values: Any,
    name: str,
    expected_rows: int,
    context: str,
) -> np.ndarray:
    """Validate a numeric inference vector aligned with X."""
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be numeric.",
            recovery_hint=f"Pass a numeric {name} vector aligned with X.",
        ) from exc
    if arr.shape[0] != expected_rows:
        raise MethodIncompatibility(
            f"{context}: {name} must have the same row count as X.",
            recovery_hint="Align X, Y, and T before running forest inference.",
            diagnostics={
                f"n_{name.lower()}": int(arr.shape[0]),
                "n_x": int(expected_rows),
            },
        )
    if expected_rows == 0:
        raise DataInsufficient(
            f"{context}: no rows were supplied.",
            recovery_hint="Pass at least one inference row.",
        )
    if not np.isfinite(arr).all():
        raise MethodIncompatibility(
            f"{context}: {name} contains NaN or infinite values.",
            recovery_hint=f"Drop or impute non-finite {name} rows.",
        )
    return arr


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

    Examples
    --------
    ``sp.test_calibration`` is an alias of this function.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> ct = sp.calibration_test(cf)
    >>> ct.index.tolist()
    ['mean_forest_prediction', 'differential_forest_prediction']
    >>> sp.test_calibration is sp.calibration_test
    True
    """
    _require_fitted_forest(forest, "calibration_test()")
    alpha_value = _validate_alpha(alpha, "calibration_test()")

    X_ = _prepare_forest_features(forest, X, "calibration_test()")
    Y_ = _prepare_forest_vector(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        "Y",
        X_.shape[0],
        "calibration_test()",
    )
    T_ = _prepare_forest_vector(
        T if T is not None else getattr(forest, "_T_original", None),
        "T",
        X_.shape[0],
        "calibration_test()",
    )
    if X_.shape[0] < 3:
        raise DataInsufficient(
            "calibration_test() requires at least 3 rows.",
            recovery_hint="Use a larger sample for the calibration regression.",
        )

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

    z = stats.norm.ppf(1 - alpha_value / 2)
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
        return np.asarray(
            (mu1 - mu0) + (T - e_hat) * (Y - mu_T) / (e_hat * (1 - e_hat))
        )
    # Horvitz-Thompson signal (centered on e_hat-adjusted residual)
    return np.asarray(T * (Y - m_hat) / e_hat - (1 - T) * (Y - m_hat) / (1 - e_hat))


def _get_nuisances(
    forest: "CausalForest",
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the fitted nuisance predictions from the CausalForest, with
    fallbacks if they are unavailable.
    """
    m_hat_raw = getattr(forest, "_m_insample", None)
    e_hat_raw = getattr(forest, "_e_insample", None)
    if m_hat_raw is not None and len(np.asarray(m_hat_raw).ravel()) == len(Y):
        m_hat = np.asarray(m_hat_raw, dtype=np.float64).ravel()
    else:
        # Fall back to sample mean of Y
        m_hat = np.full_like(Y, Y.mean())
    if e_hat_raw is not None and len(np.asarray(e_hat_raw).ravel()) == len(T):
        e_hat = np.asarray(e_hat_raw, dtype=np.float64).ravel()
    else:
        e_hat = np.full_like(T, T.mean())
    return m_hat, e_hat


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

    Let ``S(x) = τ̂(x)`` denote a prioritisation score (higher = higher
    priority). Define the TOC (targeting operator characteristic) curve:

        TOC(q) = E[τ(X) | S(X) ≥ Q_{1-q}(S)] - E[τ(X)]

    i.e. the expected CATE among the top-``q`` fraction minus the
    population ATE. Two scalar summaries are supported:

    - **AUTOC**: ``∫₀¹ TOC(q) dq`` — the *unweighted* area under the
      TOC curve. Emphasises prioritisation performance uniformly
      across the quantile range.
    - **QINI**:  ``∫₀¹ q · TOC(q) dq`` — down-weights narrow top
      fractions; closer to the classical uplift / Qini coefficient.

    Estimation
    ----------
    The DR-RATE estimator from Yadlowsky et al. uses an AIPW pseudo-
    outcome ``Ψ_i`` (computed from the forest's own cross-fitted
    nuisance predictions) and reduces AUTOC / Qini to a weighted sum:

        AUTOC_hat = (1/n) Σ_i Ψ_i · w_{AUTOC}(R_i / n) - Ψ̄
        QINI_hat  = (1/n) Σ_i Ψ_i · w_{QINI}(R_i / n)  - (1/2) Ψ̄

    where ``R_i`` is the descending rank of ``S(X_i)`` and the weights
    are closed-form rank kernels. This representation makes the
    estimator a sample mean of per-observation contributions φ_i, so
    the variance admits the standard influence-function form

        Var(AUTOC_hat) = (1/(n(n-1))) Σ_i (φ_i - φ̄)²

    which replaces the conservative half-sample estimator used in the
    earlier draft of this function.

    Parameters
    ----------
    forest : fitted CausalForest
    X, Y, T : arrays, optional
        If omitted, falls back to the forest's stored training arrays.
    target : {'AUTOC', 'QINI'}
    q_grid : int
        Number of quantile grid points used to report the TOC curve.
        Does not affect the point estimate or SE (those are computed
        from ranks exactly).
    alpha : float
    seed : int, optional
        Ignored; kept for API backwards compatibility.

    Returns
    -------
    dict with keys ``estimate``, ``se``, ``ci_low``, ``ci_high``,
    ``target``, ``toc_curve`` (``(q_grid, 2)``), ``n``, ``method``.

    References
    ----------
    Yadlowsky, S., Fleming, S., Shah, N., Brunskill, E., Wager, S.
    (2021). "Evaluating Treatment Prioritization Rules via Rank-
    Weighted Average Treatment Effects." arXiv:2111.07966.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> res = sp.rate(cf, target="AUTOC")
    >>> sorted(res.keys())
    ['ci_high', 'ci_low', 'estimate', 'method', 'n', 'se', \
'target', 'toc_curve']
    >>> res["toc_curve"].shape
    (100, 2)
    """
    _require_fitted_forest(forest, "rate()")
    try:
        target_key = target.upper().strip()
    except AttributeError as exc:
        raise MethodIncompatibility(
            "rate(): target must be a string.",
            recovery_hint="Use target='AUTOC' or target='QINI'.",
            diagnostics={"target": target},
        ) from exc
    if target_key not in ("AUTOC", "QINI"):
        raise MethodIncompatibility(
            "rate(): target must be 'AUTOC' or 'QINI'.",
            recovery_hint="Use a supported RATE summary target.",
            diagnostics={"target": target},
        )
    if (
        isinstance(q_grid, bool)
        or not isinstance(q_grid, (int, np.integer))
        or int(q_grid) < 1
    ):
        raise MethodIncompatibility(
            "rate(): q_grid must be a positive integer.",
            recovery_hint="Use q_grid >= 1.",
            diagnostics={"q_grid": q_grid},
        )
    q_grid_value = int(q_grid)
    alpha_value = _validate_alpha(alpha, "rate()")

    X_ = _prepare_forest_features(forest, X, "rate()")
    Y_ = _prepare_forest_vector(
        Y if Y is not None else getattr(forest, "_Y_original", None),
        "Y",
        X_.shape[0],
        "rate()",
    )
    T_ = _prepare_forest_vector(
        T if T is not None else getattr(forest, "_T_original", None),
        "T",
        X_.shape[0],
        "rate()",
    )
    n = len(Y_)
    if n < 2:
        raise DataInsufficient(
            "rate() requires at least 2 rows.",
            recovery_hint="Use a larger sample for RATE inference.",
        )

    m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
    e_hat = np.clip(e_hat, 0.02, 0.98)
    psi = _construct_pseudo_outcome(Y_, T_, e_hat, m_hat, forest, X_)
    tau_hat = np.asarray(forest.effect(X_)).ravel()
    psi_bar = float(psi.mean())

    # Descending rank by τ̂ (rank 1 = highest priority).  Ties: stable
    # ordering is fine; any rank permutation among ties leaves the
    # weighted sum invariant because the kernel only depends on the
    # fractional rank.
    desc_order = np.argsort(-tau_hat, kind="mergesort")
    rank = np.empty(n, dtype=np.float64)
    rank[desc_order] = np.arange(1, n + 1)  # 1-based rank
    u = rank / n  # fractional rank ∈ (0, 1]

    # Closed-form rank kernels. Derivation (AUTOC): AUTOC_hat rewrites
    # as the mean over k ∈ {1,..,n} of the top-k mean of Ψ minus Ψ̄.
    # Reordering the double sum gives the weight on observation i as
    # w_i = (1/n) · Σ_{k=R_i}^{n} (1/k) = (H_n - H_{R_i - 1}) / n.
    # The harmonic-number formula is exact for the empirical estimator.
    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])
    # For observation i with rank R_i, weight = H_n - H_{R_i - 1}.
    R_int = rank.astype(np.int64)
    w_autoc = H[n] - H[R_int - 1]  # shape (n,)
    # Qini kernel: w_{QINI}(u) = 1 - u (weights decrease linearly with
    # the descending rank). Using the fractional rank keeps the sample
    # average well-calibrated to the continuous population integral.
    w_qini = 1.0 - u

    # Rewrite both estimators as a single sample mean
    #     θ̂ = (1/n) Σ_i Ψ_i · (w_i - c)
    # with c = 1 for AUTOC (because AUTOC_hat = mean(Ψ·w) - Ψ̄) and
    # c = 1/2 for Qini. The per-observation contribution
    #     φ_i = Ψ_i · (w_i - c)
    # depends only on obs i (treating the ranks as conditioning), so
    # its sample variance over n gives the correct influence-function
    # variance of θ̂ — no whole-sample subtraction is needed.
    if target_key == "AUTOC":
        phi = psi * (w_autoc - 1.0)
    else:  # QINI
        phi = psi * (w_qini - 0.5)
    estimate = float(phi.mean())

    # Influence-function variance: Var_hat(φ) / n, using the n-1
    # denominator for the centred sum of squares.
    phi_centered = phi - phi.mean()
    var_est = (
        float(phi_centered @ phi_centered) / (n * (n - 1)) if n > 1 else float("nan")
    )
    se = float(np.sqrt(max(var_est, 0.0)))
    z = stats.norm.ppf(1 - alpha_value / 2)
    ci = (estimate - z * se, estimate + z * se)

    # TOC curve for diagnostics (not used in the IF variance path).
    psi_sorted = psi[desc_order]
    cum = np.cumsum(psi_sorted) / np.arange(1, n + 1)
    toc_all = cum - psi_bar  # length-n array, evaluated at q_k = k/n
    q_targets = np.linspace(1.0 / q_grid_value, 1.0, q_grid_value)
    idx_sel = np.clip((q_targets * n).astype(np.int64) - 1, 0, n - 1)
    toc_grid = np.column_stack([q_targets, toc_all[idx_sel]])

    return {
        "estimate": estimate,
        "se": se,
        "ci_low": ci[0],
        "ci_high": ci[1],
        "target": target_key,
        "toc_curve": toc_grid,
        "n": n,
        "method": "Influence-function SE (Yadlowsky et al. 2023)",
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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> hv = sp.honest_variance(cf, n_splits=25, seed=0)
    >>> bool(hv["se"] >= 0 and hv["ci_low"] <= hv["ate"] <= hv["ci_high"])
    True
    """
    _require_fitted_forest(forest, "honest_variance()")
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, (int, np.integer))
        or int(n_splits) < 2
    ):
        raise MethodIncompatibility(
            "honest_variance(): n_splits must be an integer >= 2.",
            recovery_hint="Use at least two half-sample splits.",
            diagnostics={"n_splits": n_splits},
        )
    n_splits_value = int(n_splits)
    X_ = _prepare_forest_features(forest, X, "honest_variance()")
    tau = np.asarray(forest.effect(X_)).ravel()
    n = len(tau)
    if n < 2:
        raise DataInsufficient(
            "honest_variance() requires at least 2 CATE rows.",
            recovery_hint="Pass at least two rows of effect modifiers.",
        )
    rng = np.random.default_rng(seed)

    means = np.empty(n_splits_value)
    for s in range(n_splits_value):
        perm = rng.permutation(n)
        half = perm[: n // 2]
        means[s] = float(tau[half].mean())

    ate = float(tau.mean())
    se = float(np.std(means, ddof=1) / np.sqrt(n_splits_value))
    z = stats.norm.ppf(0.975)
    return {
        "ate": ate,
        "se": se,
        "ci_low": ate - z * se,
        "ci_high": ate + z * se,
    }


def average_treatment_effect(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    target_sample: str = "all",
    alpha: float = 0.05,
    clip: float = 0.01,
) -> Dict[str, Any]:
    """Aggregate CATE predictions into ATE/ATT/ATC/ATO targets.

    This mirrors the most-used ``grf::average_treatment_effect`` targets:
    ``"all"`` (ATE), ``"treated"`` (ATT), ``"control"`` (ATC), and
    ``"overlap"`` (ATO, weighted by ``e(X)(1-e(X))``).

    The estimate is the **doubly-robust AIPW influence-function mean**
    (the estimator grf reports), not a plug-in average of the CATE
    predictions.  Using the forest's own cross-fitted nuisances
    :math:`\\hat m(X)=\\hat E[Y\\mid X]` and :math:`\\hat e(X)=\\hat
    E[T\\mid X]`, the ATE score is

    .. math::
        \\Gamma_i = \\hat\\tau(X_i)
            + \\frac{T_i-\\hat e(X_i)}{\\hat e(X_i)(1-\\hat e(X_i))}
              \\bigl(Y_i-\\hat m(X_i)-(T_i-\\hat e(X_i))\\hat\\tau(X_i)\\bigr),

    and the ATT/ATC scores use the analogous Robins doubly-robust
    weighting.  ``se`` is the influence-function standard error
    :math:`\\mathrm{sd}(\\Gamma)/\\sqrt n`.  When the score cannot be
    formed (out-of-sample ``X`` with no stored nuisances) the function
    falls back to the plug-in CATE average and sets ``method='plug_in'``.

    Parameters
    ----------
    clip : float, default 0.01
        Propensity scores are clipped to ``[clip, 1-clip]`` before the
        inverse-propensity term to stabilise the score under near-overlap
        violations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> ate = sp.average_treatment_effect(cf, target_sample="all")
    >>> ate["estimand"]
    'ATE'
    >>> att = sp.average_treatment_effect(cf, target_sample="treated")
    >>> att["estimand"]
    'ATT'
    """
    if not getattr(forest, "fitted_", False):
        raise MethodIncompatibility(
            "average_treatment_effect() requires a fitted forest.",
            recovery_hint="Call fit() before aggregating treatment effects.",
        )

    try:
        target_key = target_sample.lower().strip()
    except AttributeError as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): target_sample must be a string.",
            recovery_hint=("Use one of 'all', 'treated', 'control', or 'overlap'."),
            diagnostics={"target_sample": target_sample},
        ) from exc
    aliases = {
        "ate": "all",
        "all": "all",
        "att": "treated",
        "treated": "treated",
        "atc": "control",
        "control": "control",
        "ato": "overlap",
        "overlap": "overlap",
    }
    target = aliases.get(target_key)
    if target is None:
        raise MethodIncompatibility(
            "target_sample must be one of 'all', 'treated', 'control', " "or 'overlap'",
            recovery_hint="Use a supported GRF aggregation target.",
            diagnostics={"target_sample": target_sample},
        )
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): alpha must be a finite scalar.",
            recovery_hint="Use an alpha value in the open interval (0, 1).",
            diagnostics={"alpha": alpha},
        ) from exc
    if not np.isfinite(alpha_value) or not 0.0 < alpha_value < 1.0:
        raise MethodIncompatibility(
            "average_treatment_effect(): alpha must be in the open interval (0, 1).",
            recovery_hint="Use an alpha value such as 0.05.",
            diagnostics={"alpha": alpha},
        )
    try:
        clip_value = float(clip)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): clip must be a finite scalar.",
            recovery_hint="Use a propensity clip in the interval [0, 0.5).",
            diagnostics={"clip": clip},
        ) from exc
    if not np.isfinite(clip_value) or not 0.0 <= clip_value < 0.5:
        raise MethodIncompatibility(
            "average_treatment_effect(): clip must be in the interval [0, 0.5).",
            recovery_hint="Use a small propensity clip such as 0.01.",
            diagnostics={"clip": clip},
        )

    use_insample = X is None and T is None
    if X is None:
        X_ = np.asarray(forest._X_original, dtype=np.float64)
    elif callable(getattr(forest, "_prepare_effect_matrix", None)):
        X_ = np.asarray(
            forest._prepare_effect_matrix(
                X,
                context="average_treatment_effect()",
            ),
            dtype=np.float64,
        )
    else:
        try:
            X_ = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "average_treatment_effect(): X must be numeric.",
                recovery_hint="Pass X shaped (n_samples, n_features).",
            ) from exc
    tau = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    try:
        T_ = np.asarray(
            T if T is not None else getattr(forest, "_T_original", None),
            dtype=np.float64,
        ).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): T must be numeric.",
            recovery_hint="Pass a numeric treatment vector aligned with X.",
        ) from exc
    if len(T_) != len(tau):
        raise MethodIncompatibility(
            "average_treatment_effect(): T must match the number of effect rows.",
            recovery_hint="Pass treatment values aligned with the CATE rows.",
            diagnostics={"n_t": int(len(T_)), "n_effects": int(len(tau))},
        )
    if not np.isfinite(T_).all():
        raise MethodIncompatibility(
            "average_treatment_effect(): T contains NaN or infinite values.",
            recovery_hint="Drop or impute non-finite treatment rows.",
        )
    if target == "treated" and not np.any(T_ == 1):
        raise DataInsufficient(
            "average_treatment_effect(): no treated observations for ATT.",
            recovery_hint="Use target_sample='all' or pass at least one T == 1 row.",
        )
    if target == "control" and not np.any(T_ == 0):
        raise DataInsufficient(
            "average_treatment_effect(): no control observations for ATC.",
            recovery_hint="Use target_sample='all' or pass at least one T == 0 row.",
        )
    try:
        Y_ = np.asarray(
            getattr(forest, "_Y_original", None),
            dtype=np.float64,
        ).ravel()
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "average_treatment_effect(): stored outcomes must be numeric.",
            recovery_hint="Refit the forest with numeric outcomes.",
        ) from exc

    # Outcome and propensity nuisances.  When aggregating on the training
    # sample we reuse the forest's own cross-fitted (cv=3) out-of-fold
    # nuisances m̂ = Ê[Y|X] and ê = Ê[T|X] -- the same quantities grf
    # uses for ``average_treatment_effect`` -- so the ATE/ATT scores are
    # honest doubly-robust influence functions rather than a plug-in mean
    # of the (regularisation-shrunk) CATE predictions.
    m_insample = getattr(forest, "_m_insample", None)
    e_insample = getattr(forest, "_e_insample", None)
    m_hat: np.ndarray
    e_hat: np.ndarray
    if use_insample and m_insample is not None and e_insample is not None:
        m_hat = np.asarray(m_insample, dtype=np.float64).ravel()
        e_hat = np.asarray(e_insample, dtype=np.float64).ravel()
    else:
        m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
        m_hat = np.asarray(m_hat, dtype=np.float64).ravel()
        e_hat = np.asarray(e_hat, dtype=np.float64).ravel()
    e_hat = np.clip(e_hat, clip_value, 1.0 - clip_value)
    if len(e_hat) != len(tau) or len(m_hat) != len(tau) or len(Y_) != len(tau):
        # AIPW score is unavailable (out-of-sample without nuisances or a
        # length mismatch); fall back to the plug-in CATE average and flag it.
        return _plug_in_average(tau, T_, e_hat, target, alpha)

    n = int(len(tau))
    z = float(stats.norm.ppf(1 - alpha_value / 2))
    # Reconstruct the per-arm outcome regressions from (m̂, ê, τ̂):
    #   m = e·μ1 + (1-e)·μ0,  μ1 - μ0 = τ  ⇒  μ0 = m - e·τ,  μ1 = m + (1-e)·τ.
    mu0 = m_hat - e_hat * tau
    mu1 = m_hat + (1.0 - e_hat) * tau
    m_full = m_hat + (T_ - e_hat) * tau  # = E[Y|X,T] under the model

    if target == "all":
        estimand = "ATE"
        psi = tau + (T_ - e_hat) / (e_hat * (1.0 - e_hat)) * (Y_ - m_full)
        estimate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(n))
        ess = float(n)
    elif target == "treated":
        estimand = "ATT"
        p1 = float(max(T_.mean(), 1e-8))
        psi = (T_ * (Y_ - mu0) - (1.0 - T_) * (e_hat / (1.0 - e_hat)) * (Y_ - mu0)) / p1
        estimate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(n))
        ess = float(T_.sum())
    elif target == "control":
        estimand = "ATC"
        p0 = float(max((1.0 - T_).mean(), 1e-8))
        psi = ((1.0 - T_) * (mu1 - Y_) - T_ * ((1.0 - e_hat) / e_hat) * (mu1 - Y_)) / p0
        estimate = float(psi.mean())
        se = float(psi.std(ddof=1) / np.sqrt(n))
        ess = float((1.0 - T_).sum())
    else:  # overlap (ATO): overlap-weighted average of the AIPW pointwise scores
        estimand = "ATO"
        w = e_hat * (1.0 - e_hat)
        if float(w.sum()) <= 0:
            raise DataInsufficient(
                f"No observations contribute to target_sample={target_sample!r}.",
                recovery_hint="Choose a target with support in the supplied sample.",
            )
        psi_pt = tau + (T_ - e_hat) / (e_hat * (1.0 - e_hat)) * (Y_ - m_full)
        estimate = float(np.average(psi_pt, weights=w))
        norm_w = w / w.sum()
        se = float(np.sqrt(np.sum((norm_w**2) * (psi_pt - estimate) ** 2)))
        ess = float((w.sum() ** 2) / np.sum(w**2))

    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": target,
        "estimand": estimand,
        "method": "aipw",
        "effective_sample_size": ess,
        "n": n,
        "alpha": alpha_value,
        "pscore_min": float(e_hat.min()),
        "pscore_max": float(e_hat.max()),
    }


def _plug_in_average(
    tau: np.ndarray,
    T_: np.ndarray,
    e_hat: np.ndarray,
    target: str,
    alpha: float,
) -> Dict[str, Any]:
    """Fallback weighted average of CATE predictions (no AIPW score).

    Used only when the doubly-robust influence function cannot be formed
    (out-of-sample ``X`` with no stored nuisances, or a length mismatch).
    """
    if target == "all":
        weights = np.ones_like(tau)
        estimand = "ATE"
    elif target == "treated":
        weights = (T_ == 1).astype(float)
        estimand = "ATT"
    elif target == "control":
        weights = (T_ == 0).astype(float)
        estimand = "ATC"
    else:
        weights = e_hat * (1.0 - e_hat)
        estimand = "ATO"
    if float(weights.sum()) <= 0:
        raise DataInsufficient(
            "No observations contribute to the requested target_sample.",
            recovery_hint="Choose a target with support in the supplied sample.",
        )
    estimate = float(np.average(tau, weights=weights))
    norm_w = weights / weights.sum()
    se = float(np.sqrt(np.sum((norm_w**2) * (tau - estimate) ** 2)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    ess = float((weights.sum() ** 2) / np.sum(weights**2))
    return {
        "estimate": estimate,
        "se": se,
        "ci_low": estimate - z * se,
        "ci_high": estimate + z * se,
        "target_sample": target,
        "estimand": estimand,
        "method": "plug_in",
        "effective_sample_size": ess,
        "n": int(len(tau)),
        "alpha": float(alpha),
        "pscore_min": float(e_hat.min()) if len(e_hat) else float("nan"),
        "pscore_max": float(e_hat.max()) if len(e_hat) else float("nan"),
    }


def forest_diagnostics(
    forest: "CausalForest",
    X: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    propensity_bounds: Tuple[float, float] = (0.05, 0.95),
) -> Dict[str, object]:
    """Return overlap and CATE-distribution diagnostics for a fitted forest.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.binomial(1, 0.5, size=n)
    >>> tau = 1.0 + X[:, 0]  # heterogeneous effect
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> df = pd.DataFrame({
    ...     "y": Y, "d": T,
    ...     "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
    ... })
    >>> cf = sp.causal_forest(
    ...     data=df, formula="y ~ d | x0 + x1 + x2",
    ...     n_estimators=50, random_state=0,
    ... )
    >>> diag = sp.forest_diagnostics(cf)
    >>> bool(diag["n_treated"] + diag["n_control"] == diag["n"])
    True
    """
    _require_fitted_forest(forest, "forest_diagnostics()")
    try:
        low_raw, high_raw = propensity_bounds
        low = float(low_raw)
        high = float(high_raw)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "forest_diagnostics(): propensity_bounds must contain two scalars.",
            recovery_hint="Use bounds such as (0.05, 0.95).",
            diagnostics={"propensity_bounds": propensity_bounds},
        ) from exc
    if not np.isfinite(low) or not np.isfinite(high) or not 0 <= low < high <= 1:
        raise MethodIncompatibility(
            "propensity_bounds must satisfy 0 <= low < high <= 1.",
            recovery_hint="Use bounds such as (0.05, 0.95).",
            diagnostics={"propensity_bounds": propensity_bounds},
        )

    X_ = _prepare_forest_features(forest, X, "forest_diagnostics()")
    tau = np.asarray(forest.effect(X_), dtype=np.float64).ravel()
    if X is not None and T is None:
        raise MethodIncompatibility(
            "forest_diagnostics(): T is required when X is supplied.",
            recovery_hint="Pass treatment values aligned with the diagnostic X rows.",
        )
    T_ = _prepare_forest_vector(
        T if T is not None else getattr(forest, "_T_original", np.zeros(len(tau))),
        "T",
        len(tau),
        "forest_diagnostics()",
    )
    Y_ = np.asarray(
        getattr(forest, "_Y_original", np.zeros(len(tau))),
        dtype=np.float64,
    )
    Y_ = Y_.ravel()
    if len(Y_) != len(tau):
        Y_ = np.zeros(len(tau), dtype=np.float64)
    _m_hat, e_hat = _get_nuisances(forest, X_, Y_, T_)
    e_hat = np.clip(np.asarray(e_hat, dtype=np.float64).ravel(), 0.0, 1.0)
    if len(e_hat) != len(tau):
        e_hat = np.full(len(tau), float(np.mean(T_)))

    overlap = (e_hat >= low) & (e_hat <= high)
    warnings = []
    if e_hat.min() < low or e_hat.max() > high:
        warnings.append(
            "propensity scores outside requested overlap bounds; report "
            "ATE/ATT with caution or use target_sample='overlap'"
        )
    if float(np.std(tau)) < 1e-8:
        warnings.append("predicted CATE is nearly constant; heterogeneity is weak")
    if not getattr(forest, "honest", True):
        warnings.append("forest was fitted with honest=False")

    return {
        "n": int(len(tau)),
        "n_treated": int(np.sum(T_ == 1)),
        "n_control": int(np.sum(T_ == 0)),
        "cate_mean": float(np.mean(tau)),
        "cate_sd": float(np.std(tau, ddof=1)) if len(tau) > 1 else 0.0,
        "cate_min": float(np.min(tau)),
        "cate_max": float(np.max(tau)),
        "cate_iqr": float(np.subtract(*np.percentile(tau, [75, 25]))),
        "pscore_min": float(np.min(e_hat)),
        "pscore_max": float(np.max(e_hat)),
        "overlap_low": float(low),
        "overlap_high": float(high),
        "overlap_share": float(np.mean(overlap)),
        "n_low_pscore": int(np.sum(e_hat < low)),
        "n_high_pscore": int(np.sum(e_hat > high)),
        "warnings": warnings,
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
    "average_treatment_effect",
    "forest_diagnostics",
]
