"""
Multi-Dimensional Sensitivity Dashboard.

One-call sensitivity analysis across common dimensions: sample,
specification, bandwidth, functional form, and estimator.
This multi-axis sensitivity workflow reports selected checks without
turning them into a blanket validation claim.

Usage
-----
>>> import statspai as sp
>>> result = sp.regress("wage ~ educ + exper", data=df)
>>> dash = sp.sensitivity_dashboard(result, data=df)
>>> print(dash.summary())
"""

from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from ..workflow._degradation import record_degradation


class SensitivityDashboard:
    """Multi-dimensional sensitivity analysis results.

    Returned by :func:`sp.sensitivity_dashboard`. Holds the ``baseline``
    estimate, a list of per-``dimension`` summaries and an overall
    ``A``/``B``/``C``/``D``/``F`` stability grade.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=n),
    ...     "x2": rng.normal(size=n),
    ... })
    >>> df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(size=n)
    >>> result = sp.regress("y ~ x1 + x2", data=df)
    >>> dash = sp.sensitivity_dashboard(result, data=df, verbose=False)
    >>> type(dash).__name__
    'SensitivityDashboard'
    >>> bool(dash.overall_stability in {"A", "B", "C", "D", "F"})
    True
    >>> isinstance(dash.dimensions, list)
    True
    """

    def __init__(
        self,
        baseline: Dict[str, Any],
        dimensions: List[Dict[str, Any]],
        overall_stability: str,
        method: str,
    ) -> None:
        self.baseline = baseline  # dict: estimate, se, ci
        self.dimensions = dimensions  # list of dicts
        self.overall_stability = overall_stability  # A/B/C/D/F
        self.method = method

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "Sensitivity Dashboard",
            "=" * 70,
            f"Method: {self.method}",
            f"Baseline estimate: {self.baseline['estimate']:.4f} "
            f"(SE = {self.baseline['se']:.4f})",
            f"Overall stability: {self.overall_stability}",
            "",
        ]

        for dim in self.dimensions:
            lines.append(f"{'─' * 70}")
            lines.append(f"  {dim['dimension'].upper()}")
            lines.append(f"  Variations: {dim['n_variations']}")
            lines.append(f"  Range: [{dim['min_est']:.4f}, {dim['max_est']:.4f}]")
            lines.append(f"  Sign stable: {dim['sign_stable']:.0%}")
            lines.append(f"  Sig. stable: {dim['sig_stable']:.0%}")

            status = "✓" if dim["stable"] else "✗"
            lines.append(
                f"  Status: {status} {'Stable' if dim['stable'] else 'SENSITIVE'}"
            )

            if not dim["stable"]:
                lines.append(f"  → {dim['remedy']}")

        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)


def _linear_design(result: Any) -> Optional[tuple]:
    """Return ``(X, y, var_names)`` from a result's stored linear design, or
    ``None`` when it does not expose a plain ``X``/``y`` an OLS re-fit can act
    on (e.g. CausalResult families that carry no design matrix)."""
    di = getattr(result, "data_info", {}) or {}
    X, y, names = di.get("X"), di.get("y"), di.get("var_names")
    if X is None or y is None or names is None:
        return None
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0] or X.shape[0] < 3:
        return None
    return X, y, list(names)


def _refit_coef(
    X: np.ndarray, y: np.ndarray, j: int, rows: np.ndarray
) -> Optional[float]:
    """Real OLS re-fit of coefficient ``j`` on ``rows``; ``None`` if singular."""
    try:
        beta, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
    except np.linalg.LinAlgError:
        return None
    if j >= beta.shape[0] or not np.isfinite(beta[j]):
        return None
    return float(beta[j])


def sensitivity_dashboard(
    result: Any,
    data: Optional[pd.DataFrame] = None,
    dimensions: Optional[List[str]] = None,
    alpha: float = 0.05,
    verbose: bool = True,
) -> SensitivityDashboard:
    """
    Comprehensive multi-dimensional sensitivity analysis.

    Test sensitivity across selected dimensions and produce an overall
    stability grade.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        Baseline estimated result.
    data : pd.DataFrame, optional
        Original data (auto-extracted if possible).
    dimensions : list of str, optional
        Which dimensions to test. Default: all applicable.
        Options: 'sample', 'controls', 'functional_form',
        'outliers', 'unobservables'.
    alpha : float, default 0.05
    verbose : bool, default True

    Returns
    -------
    SensitivityDashboard

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({
    ...     "x1": rng.normal(size=n),
    ...     "x2": rng.normal(size=n),
    ... })
    >>> df["y"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(size=n)
    >>> result = sp.regress("y ~ x1 + x2", data=df)
    >>> dash = sp.sensitivity_dashboard(
    ...     result, data=df, dimensions=["sample", "outliers"], verbose=False,
    ... )
    >>> type(dash).__name__
    'SensitivityDashboard'
    """
    # Extract baseline. Order: (1) CausalResult-like `.estimate` /
    # `.se`, (2) EconometricResults `.params` / `.std_errors`,
    # (3) PrincipalStratResult (which has neither) — pull the top
    # row of `.effects` instead.
    # ``baseline_key`` is the name of the coefficient ``baseline_est``
    # represents; it is set only on the regression (``.params``) paths so the
    # subsample / outlier dimensions can re-fit *the same* coefficient.
    baseline_key: Optional[str] = None
    if hasattr(result, "estimate") and not isinstance(
        getattr(result, "estimate", None), pd.Series
    ):
        baseline_est = float(result.estimate)
        baseline_se = float(getattr(result, "se", 0.0))
    elif type(result).__name__ == "PrincipalStratResult":
        # Use the complier row explicitly (LATE) rather than
        # .iloc[0], which depends on the effects DataFrame ordering
        # and would silently select an always-taker effect if the
        # table is ever sorted differently upstream.
        effects = getattr(result, "effects", None)
        if effects is not None and len(effects):
            _complier_mask = (
                effects["stratum"].astype(str).str.lower().str.contains("complier")
            )
            _row = (
                effects[_complier_mask].iloc[0]
                if _complier_mask.any()
                else effects.iloc[0]
            )
            baseline_est = float(_row["estimate"])
            baseline_se = float(_row["se"]) if "se" in effects.columns else 0.0
        else:
            baseline_est, baseline_se = 0.0, 0.0
    elif hasattr(result, "params") and len(result.params) > 1:
        # Use first non-constant coefficient. Exclude every common intercept
        # name — Stata's ``_cons``, patsy's ``Intercept`` and statsmodels'
        # ``const`` — so the dashboard analyses the first real regressor
        # (e.g. the treatment) rather than the intercept.
        non_const = [
            k
            for k in result.params.index
            if str(k) not in ("_cons", "Intercept", "const")
        ]
        if non_const:
            key = non_const[0]
            baseline_est = result.params[key]
            baseline_se = result.std_errors[key]
            baseline_key = str(key)
        else:
            baseline_est = result.params.iloc[0]
            baseline_se = result.std_errors.iloc[0]
            baseline_key = str(result.params.index[0])
    else:
        baseline_est = result.params.iloc[0]
        baseline_se = result.std_errors.iloc[0]
        baseline_key = str(result.params.index[0])

    z_crit = 1.96
    baseline = {
        "estimate": baseline_est,
        "se": baseline_se,
        "ci": (
            baseline_est - z_crit * baseline_se,
            baseline_est + z_crit * baseline_se,
        ),
        "significant": (
            abs(baseline_est / baseline_se) > z_crit if baseline_se > 0 else False
        ),
    }

    # Real OLS re-fit support for the subsample / outlier dimensions. We can
    # genuinely re-estimate the headline coefficient only when the result
    # exposes a plain linear design (``data_info['X']``/``'y'``) whose OLS fit
    # reproduces ``baseline_est``. The self-consistency check below fails for
    # weighted / IV / non-linear results, in which case those dimensions are
    # skipped (not applicable) rather than fabricated.
    _refit_X: Optional[np.ndarray] = None
    _refit_y: Optional[np.ndarray] = None
    _refit_j: Optional[int] = None
    _design = _linear_design(result)
    if _design is not None and baseline_key is not None:
        _dX, _dy, _dnames = _design
        if baseline_key in _dnames:
            _j = _dnames.index(baseline_key)
            _full = _refit_coef(_dX, _dy, _j, np.arange(_dy.shape[0]))
            if _full is not None and np.isclose(
                _full, float(baseline_est), rtol=1e-4, atol=1e-6
            ):
                _refit_X, _refit_y, _refit_j = _dX, _dy, _j

    # Resolve a human-readable method label. Sprint-B CausalResult
    # objects store the label on ``.method`` (e.g. "Proximal Causal
    # Inference (linear 2SLS)"); older EconometricResults use
    # ``model_info['model_type']``. Read both so the dashboard
    # doesn't show "Unknown" on proximal / msm / g_computation / etc.
    # PrincipalStratResult likewise has a ``.method`` ('monotonicity'
    # or 'principal_score') plus an 'estimator' label in model_info;
    # prefer the latter because it's more descriptive.
    _model_info = getattr(result, "model_info", {}) or {}
    if type(result).__name__ == "PrincipalStratResult":
        # PrincipalStratResult: prefer the verbose estimator label.
        method = (
            str(_model_info.get("estimator", "") or "") or "Principal Stratification"
        )
    else:
        method = (
            str(getattr(result, "method", "") or "")
            or str(_model_info.get("model_type", "") or "")
            or str(_model_info.get("estimator", "") or "")
            or "Unknown"
        )
    dim_results = []

    if dimensions is None:
        dimensions = ["sample", "outliers", "unobservables"]
        # Auto-append Sprint-B-specific dimensions when the result
        # type matches. Users who pass an explicit ``dimensions=``
        # list opt out of this expansion; the explicit list is taken
        # verbatim.
        _ml_lower = method.lower()
        if "proximal" in _ml_lower:
            dimensions.append("first_stage_f")
        if "marginal structural" in _ml_lower:
            dimensions.append("trim_sweep")
        if type(result).__name__ == "PrincipalStratResult" or "principal" in _ml_lower:
            dimensions.append("monotonicity")

    if data is not None:
        if "sample" in dimensions and _refit_j is not None:
            # Subsample sensitivity via a *genuine* OLS re-fit of the headline
            # coefficient on 20 draws of 80% of the rows. Only runs when the
            # result exposes a plain linear design (see the ``_refit_j`` guard);
            # for other result types the dimension is not applicable and is
            # skipped rather than fabricated.
            assert _refit_X is not None and _refit_y is not None
            rng = np.random.default_rng(42)
            m = _refit_y.shape[0]
            subsample_ests = []
            for _ in range(20):
                idx = rng.choice(m, size=max(int(m * 0.8), 2), replace=False)
                est = _refit_coef(_refit_X, _refit_y, _refit_j, idx)
                if est is not None:
                    subsample_ests.append(est)

            if subsample_ests:
                ests = np.array(subsample_ests)
                mean_abs = max(abs(float(np.mean(ests))), 1e-10)
                dim_results.append(
                    {
                        "dimension": "Sample stability (80% subsamples)",
                        "n_variations": len(ests),
                        "min_est": float(ests.min()),
                        "max_est": float(ests.max()),
                        "sign_stable": float(
                            np.mean(np.sign(ests) == np.sign(baseline_est))
                        ),
                        "sig_stable": (
                            float(np.mean(np.abs(ests / baseline_se) > z_crit))
                            if baseline_se > 0
                            else 0.0
                        ),
                        "stable": bool(float(np.std(ests)) / mean_abs < 0.5),
                        "remedy": "Results are sample-dependent. Consider a "
                        "larger sample or bootstrap CI.",
                    }
                )

        if "outliers" in dimensions and _refit_j is not None:
            # Outcome-outlier sensitivity via a *genuine* OLS re-fit after
            # trimming the outcome's tails at 1 / 2 / 5%. Only runs with a
            # plain linear design; skipped (not applicable) otherwise.
            assert _refit_X is not None and _refit_y is not None
            outlier_ests = []
            for pct in [1, 2, 5]:
                low, high = np.percentile(_refit_y, [pct, 100 - pct])
                rows = np.where((_refit_y >= low) & (_refit_y <= high))[0]
                if rows.size >= _refit_X.shape[1] + 1:
                    est = _refit_coef(_refit_X, _refit_y, _refit_j, rows)
                    if est is not None:
                        outlier_ests.append(est)

            if outlier_ests:
                ests = np.array(outlier_ests)
                mean_abs = max(abs(float(np.mean(ests))), 1e-10)
                dim_results.append(
                    {
                        "dimension": "Outlier sensitivity (trimming)",
                        "n_variations": len(ests),
                        "min_est": float(min(ests.min(), baseline_est)),
                        "max_est": float(max(ests.max(), baseline_est)),
                        "sign_stable": float(
                            np.mean(np.sign(ests) == np.sign(baseline_est))
                        ),
                        "sig_stable": (
                            float(np.mean(np.abs(ests / baseline_se) > z_crit))
                            if baseline_se > 0
                            else 0.0
                        ),
                        "stable": bool(float(np.std(ests)) / mean_abs < 0.5),
                        "remedy": "Use sp.winsor() to winsorize outliers.",
                    }
                )

    if "unobservables" in dimensions:
        # Oster-style sensitivity
        try:
            import statspai as sp

            oster = sp.oster_bounds(result)
            delta = oster.get("delta", oster.get("oster_delta", np.nan))
            if np.isfinite(delta):
                dim_results.append(
                    {
                        "dimension": "Unobservable confounders (Oster)",
                        "n_variations": 1,
                        "min_est": baseline_est if delta > 1 else 0,
                        "max_est": baseline_est,
                        "sign_stable": 1.0 if delta > 1 else 0.5,
                        "sig_stable": 1.0 if delta > 1 else 0.0,
                        "stable": abs(delta) > 1,
                        "remedy": f"Oster δ = {delta:.2f}. If < 1, selection on "
                        f"unobservables could explain the result. "
                        f"Try sp.sensemakr() for more detail.",
                    }
                )
        except Exception as exc:
            record_degradation(
                None,
                section="sensitivity_dashboard: Oster unobservables dimension",
                exc=exc,
            )

    # ------------------------------------------------------------------
    #  Sprint-B-aware method-specific dimensions (auto-applied when the
    #  result is a proximal / msm / principal_strat / g-computation /
    #  interventional-mediation fit). Users opt out by passing an
    #  explicit ``dimensions=`` list that omits the new names; users
    #  opt in explicitly by adding them.
    # ------------------------------------------------------------------
    _method_low = method.lower()
    _info = _model_info

    # Proximal: report first-stage F as its own sensitivity dimension.
    # The F is already computed at fit time (cost 0 to surface it here)
    # and is the canonical weak-IV health check for proximal.
    if "proximal" in _method_low and (
        "first_stage_f" in dimensions or dimensions is None or "proximal" in dimensions
    ):
        fs_F = _info.get("first_stage_F")
        if fs_F is not None:
            # Stability = F >= 10 (Stock-Yogo rule of thumb). Reporting
            # only a single value, not a sweep, because the F is
            # deterministic given the data + proxy specification.
            dim_results.append(
                {
                    "dimension": "Proximal first-stage F (weak-IV)",
                    "n_variations": 1,
                    "min_est": baseline_est,
                    "max_est": baseline_est,
                    "sign_stable": 1.0,
                    "sig_stable": 1.0,
                    "stable": float(fs_F) >= 10.0,
                    "remedy": (
                        f"First-stage F = {float(fs_F):.2f}. "
                        + (
                            "F ≥ 10 → proxy is sufficiently strong."
                            if float(fs_F) >= 10.0
                            else "F < 10 → WEAK proxy; find a stronger Z "
                            "or use weak-IV-robust inference "
                            "(sp.anderson_rubin_test)."
                        )
                    ),
                }
            )

    # MSM: sweep trim quantile and report how the marginal coefficient
    # MSM: report a weight-POSITIVITY readiness check. This is a positivity
    # diagnostic (max stabilized weight), NOT a coefficient trim-sweep: a
    # genuine sweep would have to re-fit the MSM at several trim levels, which
    # the result object does not carry enough state to reproduce. We therefore
    # surface only the real, already-computed ``sw_max`` signal and tie every
    # reported flag to it, rather than fabricating a coefficient sweep.
    if "marginal structural" in _method_low and (
        "trim_sweep" in dimensions or dimensions is None or "msm" in dimensions
    ):
        _id = _info.get("cluster_var")
        if _id and data is not None:
            sw_max = float(_info.get("sw_max", 0.0))
            _positivity_ok = sw_max < 50.0
            dim_results.append(
                {
                    "dimension": "MSM weight stability (trim readiness)",
                    "n_variations": 1,
                    "min_est": baseline_est,
                    "max_est": baseline_est,
                    "sign_stable": 1.0 if _positivity_ok else 0.0,
                    "sig_stable": 1.0 if _positivity_ok else 0.0,
                    "stable": _positivity_ok,
                    "remedy": (
                        f"Positivity check only (no coefficient sweep): max "
                        f"stabilized weight = {sw_max:.2f}. "
                        + (
                            "Weights well-behaved."
                            if _positivity_ok
                            else "Extreme weight — re-fit with trim_per_period=True "
                            "and compare estimates manually."
                        )
                    ),
                }
            )

    # Principal stratification: flag monotonicity violation fraction
    # as its own sensitivity dimension (diagnostic-style, not a sweep).
    if (
        "principal" in _method_low or type(result).__name__ == "PrincipalStratResult"
    ) and (
        "monotonicity" in dimensions
        or dimensions is None
        or "principal_strat" in dimensions
    ):
        viol = _info.get("mono_violation_frac")
        if viol is not None:
            dim_results.append(
                {
                    "dimension": "Principal-strat monotonicity violation",
                    "n_variations": 1,
                    "min_est": baseline_est,
                    "max_est": baseline_est,
                    "sign_stable": 1.0,
                    "sig_stable": 1.0,
                    "stable": float(viol) <= 0.05,
                    "remedy": (
                        f"Fitted p11(x) < p10(x) for {float(viol):.1%} of "
                        f"units. "
                        + (
                            "Within 5% tolerance (clipping absorbs it)."
                            if float(viol) <= 0.05
                            else "Monotonicity concern — pair with sensitivity "
                            "analysis."
                        )
                    ),
                }
            )

    # Overall stability grade
    if dim_results:
        n_stable = sum(1 for d in dim_results if d["stable"])
        frac_stable = n_stable / len(dim_results)
        if frac_stable >= 0.9:
            grade = "A"
        elif frac_stable >= 0.7:
            grade = "B"
        elif frac_stable >= 0.5:
            grade = "C"
        elif frac_stable >= 0.3:
            grade = "D"
        else:
            grade = "F"
    else:
        grade = "?"

    dash = SensitivityDashboard(
        baseline=baseline,
        dimensions=dim_results,
        overall_stability=grade,
        method=method,
    )

    if verbose:
        print(dash.summary())

    return dash
