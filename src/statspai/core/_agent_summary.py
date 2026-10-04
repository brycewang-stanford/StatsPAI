"""
Agent-native structured views of StatsPAI result objects.

This module hosts the logic behind ``result.to_agent_summary()`` and
``result.violations()`` for both :class:`EconometricResults` and
:class:`CausalResult`.  Kept separate from ``results.py`` to avoid
bloating that file and to let the per-method rules evolve
independently of the core data model.

Design principles
-----------------

* **Non-invasive.** Neither method alters the underlying result
  object.  Callers can invoke them any number of times.
* **Structured, not prose.**  ``to_agent_summary()`` returns a
  plain ``dict`` suitable for ``json.dumps`` or direct consumption
  by an LLM tool loop.  ``violations()`` returns a list of dicts.
* **Pattern-matching, not fitting.**  Violation detection only
  *inspects* diagnostics that the estimator already stored
  (``pretrend_test``, ``rhat``, ``first_stage_f``, …). It never
  re-runs a test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ====================================================================== #
#  Thresholds (literature-based rules of thumb)
# ====================================================================== #

#: Pre-trend test p-value below which we flag a DID parallel-trends
#: concern.  Deliberately *not* 0.05: agents should treat 0.10 as
#: "warrants follow-up" per Roth (2022) on low-power pre-trend tests.
_PRETREND_ALPHA = 0.10

#: Stock-Yogo 5% bias threshold (single endogenous regressor, 2SLS).
_WEAK_IV_F = 10.0

#: Gelman-Rubin / MCMC convergence thresholds (mirrors PyMC / arviz).
_RHAT_MAX = 1.01
_ESS_MIN = 400
_DIVERGENCES_MAX = 0

#: Maximum tolerable standardized mean difference after matching (SMD). 0.10 is
#: the conventional "well balanced" target (Rosenbaum-Rubin); a match leaving
#: more than this on a covariate is not ideal but often acceptable.
_SMD_MAX = 0.10

#: Standardized mean difference above which residual imbalance is a *concern*
#: worth flagging loudly (Stuart 2010; What Works Clearinghouse). Set above
#: _SMD_MAX so the warning fires on genuinely poor matches, not borderline ones.
_SMD_IMBALANCE_MAX = 0.25

#: Propensity score overlap: treated weight share below this → bad
#: common support.
_OVERLAP_MIN = 0.05

#: DML / AIPW propensity overlap: share of units whose cross-fitted propensity
#: sits at or beyond the trimming bound (near 0 or 1). Above this the IRM /
#: AIPW estimate leans on a few near-degenerate-weight units and is unstable.
#: 0.05 fires on genuinely poor overlap (strong confounding) while clearing
#: moderate confounding, which trims essentially nothing.
_DML_OVERLAP_EXTREME_SHARE = 0.05

#: Absolute logit/probit slope coefficient above which (quasi-)complete
#: separation is the likely cause: an odds ratio of e^15 ≈ 3.3M is not a real
#: effect, it is the MLE diverging when a predictor perfectly splits the outcome
#: (Albert-Anderson 1984; Heinze-Schemper 2002). Real coefficients are O(1-5).
_LOGIT_SEPARATION_COEF = 15.0

#: Pearson dispersion (χ²/df) above which a Poisson fit is over-dispersed —
#: its variance exceeds its mean, so model-based SEs are too small. 1.5 is a
#: conservative bar (equidispersion is 1.0) that clears clean Poisson data.
_POISSON_DISPERSION_MAX = 1.5

#: Excess-zero margin: observed zero share minus the share the fitted count
#: model predicts. Above this the data has more zeros than Poisson/NB explains,
#: so a zero-inflated model (ZIP/ZINB) is warranted. 0.05 clears equidispersed
#: data (observed ≈ predicted) and fires on genuine zero inflation.
_COUNT_EXCESS_ZERO_MARGIN = 0.05

#: Heckman: |rho| this close to the ±1 boundary signals a weak exclusion
#: restriction / near-collinear inverse-Mills term — the two-step estimates are
#: unstable and hypersensitive to the specification.
_HECKMAN_RHO_BOUNDARY = 0.99

#: Heckman: inverse-Mills (lambda) p-value above which there is no detectable
#: selection — the correction is adding variance for nothing, so OLS on the
#: selected sample is consistent and more efficient.
_HECKMAN_SELECTION_P = 0.10

#: Tobit: censored share (percent) above which the likelihood is dominated by
#: the censoring point and the latent-model coefficients are fragile.
_TOBIT_CENSOR_PCT_MAX = 90.0

#: Cox: proportional-hazards test p-value below which the PH assumption is
#: rejected for some covariate (its hazard ratio changes over time).
_COX_PH_ALPHA = 0.05

#: Minimum number of clusters for cluster-robust SE to be reliable. Below this,
#: the CRVE is downward-biased and t-tests over-reject; the wild cluster
#: bootstrap (Cameron-Gelbach-Miller 2008; MacKinnon-Webb 2017) is the standard
#: remedy. 30 is the conservative end of the common 30-50 rule of thumb.
_FEW_CLUSTERS_MIN = 30

#: What to do about it, in the words the size study supports
#: (``tests/reliability/few_clusters.py``, 2,000 replications per cell):
#: with clusters of similar size and half of them treated the wild cluster
#: bootstrap rejects a true null 5% to 7% of the time from 6 clusters up;
#: with two treated clusters it almost never rejects (0.0% to 6%); with
#: one cluster holding half the sample it over-rejects (up to 13%) while
#: CR3 stays at 3% to 5%.
FEW_CLUSTERS_HINT = (
    "Report sp.wild_cluster_bootstrap (or sp.wild_cluster_ci_inv for CIs): "
    "near nominal size with few clusters of similar size when the regressor "
    "varies across many of them. It is not a remedy when only one or two "
    "clusters are treated (it then almost never rejects) or when one cluster "
    "holds much of the sample (vce='cr3' held its size there); see "
    "tests/reliability/few_clusters_results.json."
)


def effective_n_clusters(keys: Any) -> float:
    """Effective number of clusters by size: ``(sum n_g)^2 / sum n_g^2``.

    The inverse Herfindahl index of the cluster shares. It equals the
    number of clusters when they are the same size and falls toward 1 as
    one of them dominates. In ``tests/reliability/few_clusters.py`` CR1
    rejects a true null about 6% of the time when this is 36 or more, 8%
    at 26, 10% at 18, and 36% when one of 40 clusters holds half the
    sample (effective number 4), so the count of clusters alone does not
    say whether cluster-robust inference is safe.
    """
    counts = pd.Series(np.asarray(keys)).value_counts().to_numpy(dtype=float)
    if counts.size == 0:
        return 0.0
    return float(counts.sum() ** 2 / (counts**2).sum())


#: Few *treated* clusters. The cluster-robust variance estimates the treated
#: side's contribution from as many draws as there are treated clusters, so
#: the over-rejection is governed by that count and not by the total (Conley
#: and Taber 2011; Ferman and Pinto 2019). Ten is the point below which the
#: placebo-distribution methods of ``sp.did_few_treated`` are the safer
#: report; with one treated cluster the CRVE rejects a true null most of the
#: time whatever the number of controls.
_FEW_TREATED_MIN = 10


#: Kish effective sample size below which a weighted HC0 / HC1 interval
#: is visibly too short (``tests/reliability/extreme_weights.py``: 93%
#: coverage at 82, 88% at 53, 84% at 17, while HC3 stays at 93% to 94%).
_FEW_EFFECTIVE_OBS = 100


def kish_effective_n(weights: Any) -> float:
    """Kish effective sample size ``(sum w)^2 / sum w^2`` of a weight vector."""
    w = np.asarray(weights, dtype=float).ravel()
    denom = float((w**2).sum())
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0


def warn_if_weights_extreme(
    weights: Any,
    variance: str,
    stacklevel: int = 3,
    *,
    entry: str = "sp.regress",
    robust_option: str = "robust='hc1' (Stata [pw=])",
    small_sample_option: Optional[str] = "vce='hc3'",
    classical_reading: str = "analytic weights (Stata [aw=])",
    classical_assumption: str = "precisions (error variance proportional to 1 / w)",
    evidence: str = "tests/reliability/extreme_weights_results.json",
) -> float:
    """Kish effective sample size; warn where the studies found intervals short.

    ``variance`` is ``"classical"``, a heteroskedasticity-robust label
    (``"hc0"`` / ``"hc1"`` / ``"hc2"`` / ``"robust"``), or anything else (no
    warning). Two findings, of ``tests/reliability/extreme_weights.py`` for
    ``sp.regress`` and ``extreme_weights_models.py`` for fixed effects and
    Poisson (2,000 replications a cell):

    * the classical weighted variance is right when the weights are
      precisions or frequencies and badly wrong when they are sampling
      weights: 78% coverage of a 95% interval at a Kish ratio of 0.4, 34%
      to 47% at 0.05 to 0.09 (OLS); 45% to 80% (fixed effects) and 14% to
      68% (Poisson) in the same range;
    * robust variances are right in large effective samples and too short
      in small ones (OLS HC1 84% at a Kish size of 17, Poisson 84% at 17
      and 88% at 53), where HC3 holds for OLS.

    The keyword arguments name the entry point's own options, so the
    hint is one the caller can act on; ``small_sample_option=None`` says
    the entry point has no small-sample correction to offer.
    """
    import warnings

    from ..exceptions import AssumptionWarning

    w = np.asarray(weights, dtype=float).ravel()
    n = int(w.size)
    n_eff = kish_effective_n(w)
    if n == 0:
        return n_eff
    ratio = n_eff / n
    diagnostics = {"n_obs": n, "n_effective_weights": n_eff, "kish_ratio": ratio}
    if variance == "classical" and ratio < 0.5:
        small = (
            f", or {small_sample_option} when the effective sample is small"
            if small_sample_option
            else ""
        )
        warnings.warn(
            AssumptionWarning(
                f"The weights are dispersed (Kish effective sample size "
                f"{n_eff:.0f} of {n}) and the standard errors are the "
                "classical weighted ones, which assume the weights are "
                f"{classical_assumption}. If they "
                "are sampling weights these intervals are far too short.",
                recovery_hint=(
                    "For sampling or inverse-probability weights pass "
                    f"{robust_option}{small}. Keep the default only for "
                    f"{classical_reading}. See {evidence}."
                ),
                diagnostics=diagnostics,
                alternative_functions=[entry],
            ),
            stacklevel=stacklevel,
        )
    elif (
        variance in ("hc0", "hc1", "hc2", "robust")
        and n_eff < _FEW_EFFECTIVE_OBS
        and ratio < 0.5
    ):
        label = "robust" if variance == "robust" else variance.upper()
        hint = (
            f"Use {small_sample_option}, which held its coverage in the same "
            f"designs; see {evidence}."
            if small_sample_option
            else (
                f"{entry} has no small-sample correction for this case: "
                "report the interval as a lower bound on the uncertainty, or "
                f"trim the weights. See {evidence}."
            )
        )
        warnings.warn(
            AssumptionWarning(
                f"A few observations carry most of the weight (Kish effective "
                f"sample size {n_eff:.0f} of {n}): {label} "
                "intervals are too short in that case.",
                recovery_hint=hint,
                diagnostics=diagnostics,
                alternative_functions=[entry],
            ),
            stacklevel=stacklevel,
        )
    return n_eff


def weighted_effective_n_clusters(weights: Any, keys: Any) -> float:
    """Kish effective number of clusters, by the weight each cluster carries.

    ``(sum_g W_g)^2 / sum_g W_g^2`` with ``W_g`` the total weight of
    cluster ``g``. With equal weights this is :func:`effective_n_clusters`.
    """
    totals = (
        pd.Series(np.asarray(weights, dtype=float).ravel())
        .groupby(pd.Series(np.asarray(keys)).to_numpy())
        .sum()
        .to_numpy()
    )
    return kish_effective_n(totals)


def warn_if_weighted_clusters_few(
    weights: Any, keys: Any, cluster: Any, stacklevel: int = 3
) -> float:
    """Weight-effective number of clusters; warn when the weights leave few.

    In ``tests/reliability/extreme_weights_models.py`` the cluster-robust
    interval of a weighted fixed-effects regression covered 94% with 82
    clusters in effect, 91% with 23, 87% with 17 and 78% with 7, although
    50 or 200 clusters were present and equal in size. The warning fires
    when the count and the size-effective count are both at least
    ``_FEW_CLUSTERS_MIN`` (otherwise the existing warnings already speak)
    and the weight-effective count is below it.
    """
    import warnings

    from ..exceptions import AssumptionWarning

    series = pd.Series(np.asarray(keys))
    n_clusters = int(series.nunique())
    effective = weighted_effective_n_clusters(weights, series.to_numpy())
    if (
        n_clusters >= _FEW_CLUSTERS_MIN
        and effective_n_clusters(series) >= _FEW_CLUSTERS_MIN
        and effective < _FEW_CLUSTERS_MIN
    ):
        warnings.warn(
            AssumptionWarning(
                f"{n_clusters} clusters for cluster='{cluster}', but the "
                f"weights concentrate on a few of them: the weight-effective "
                f"number is {effective:.1f} (< {_FEW_CLUSTERS_MIN}). "
                "Cluster-robust intervals are too short in that case.",
                recovery_hint=(
                    "Report the weight-effective count next to the estimate "
                    "and read the interval as too short; trimming the weights "
                    "is the usual remedy. See "
                    "tests/reliability/extreme_weights_models_results.json."
                ),
                diagnostics={
                    "n_clusters": n_clusters,
                    "n_clusters_effective_weights": effective,
                    "threshold": _FEW_CLUSTERS_MIN,
                },
            ),
            stacklevel=stacklevel,
        )
    return effective


def warn_if_clusters_unequal(keys: Any, cluster: Any, stacklevel: int = 3) -> float:
    """Effective number of clusters; warn when the count hides how few.

    Returns :func:`effective_n_clusters` of ``keys``. When there are at
    least ``_FEW_CLUSTERS_MIN`` clusters but fewer than that in effect, an
    :class:`~statspai.exceptions.AssumptionWarning` is raised: the
    few-cluster warning keyed on the count stays silent there, and in
    ``tests/reliability/few_clusters.py`` cluster-robust t-tests reject a
    true null 36% of the time (pooled regression) and 22% (fixed-effects
    panel) with 40 clusters of which one holds half the sample.
    """
    import warnings

    from ..exceptions import AssumptionWarning

    series = pd.Series(np.asarray(keys))
    n_clusters = int(series.nunique())
    effective = effective_n_clusters(series)
    if n_clusters >= _FEW_CLUSTERS_MIN and effective < _FEW_CLUSTERS_MIN:
        top_share = float(series.value_counts(normalize=True).iloc[0])
        warnings.warn(
            AssumptionWarning(
                f"{n_clusters} clusters for cluster='{cluster}', but unequal "
                f"in size: the effective number is {effective:.1f} "
                f"(< {_FEW_CLUSTERS_MIN}; the largest holds "
                f"{100 * top_share:.0f}% of the rows). Cluster-robust "
                "t-tests over-reject in that case as they do with few "
                "clusters.",
                recovery_hint=FEW_CLUSTERS_HINT,
                diagnostics={
                    "n_clusters": n_clusters,
                    "n_clusters_effective": effective,
                    "largest_cluster_share": top_share,
                    "threshold": _FEW_CLUSTERS_MIN,
                },
                alternative_functions=[
                    "sp.wild_cluster_bootstrap",
                    "sp.wild_cluster_ci_inv",
                ],
            ),
            stacklevel=stacklevel,
        )
    return effective


def few_treated_cluster_columns(
    X: Any, names: Any, keys: Any, max_columns: int = 60
) -> List[Dict[str, Any]]:
    """Cluster-level 0/1 regressors with few clusters on one side.

    A regressor that is constant within every cluster and takes the
    values 0 and 1 is a cluster-level treatment. When fewer than
    ``_FEW_TREATED_MIN`` clusters carry one of the two
    values, and they are under a quarter of all clusters, the
    cluster-robust variance of its coefficient rests on that many draws:
    in ``tests/reliability/few_clusters.py``, with two treated clusters
    out of 40, CR1 rejects a true null 31% of the time and the wild
    cluster bootstrap never does.

    Returns one record per such column (name, clusters at 1, clusters at
    0). Designs wider than ``max_columns`` are not scanned, and the level
    dummies of a categorical term (``C(state)[T.CA]``) are skipped: a set
    of cluster indicators is not a treatment.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] > max_columns:
        return []
    codes = pd.factorize(np.asarray(keys))[0]
    if codes.min(initial=0) < 0:
        return []
    n_clusters = int(codes.max()) + 1 if codes.size else 0
    if n_clusters < 2:
        return []
    out: List[Dict[str, Any]] = []
    for j, name in enumerate(names):
        if "[T." in str(name) or str(name).startswith("C("):
            continue
        col = X[:, j]
        lo, hi = col.min(), col.max()
        if lo != 0.0 or hi != 1.0:
            continue
        if not np.all((col == 0.0) | (col == 1.0)):
            continue
        ones = np.bincount(codes, weights=col, minlength=n_clusters)
        sizes = np.bincount(codes, minlength=n_clusters)
        if not np.all((ones == 0) | (ones == sizes)):
            continue  # varies within some cluster
        treated = int((ones > 0).sum())
        control = n_clusters - treated
        # Few on one side in absolute terms and as a share: five of ten
        # is the balanced few-cluster case, which the count warning covers
        # and where the bootstrap does work.
        if (
            min(treated, control) < _FEW_TREATED_MIN
            and 4 * min(treated, control) < n_clusters
        ):
            out.append(
                {
                    "variable": str(name),
                    "clusters_at_one": treated,
                    "clusters_at_zero": control,
                }
            )
    return out


#: Synthetic-control pre-fit quality: pre-treatment RMSPE divided by the
#: pre-period SD of the treated outcome. Deliberately conservative (0.6 ⇒ the
#: synthetic unit explains < ~64% of pre-period variance) so it clears the
#: canonical *good* example — California Prop-99 sits at ~0.42 — and only fires
#: on genuinely poor fits (an unmatchable treated trend lands ~2.7). A poor
#: pre-fit means the synthetic control does not track the treated unit before
#: treatment, so the post-period gap cannot be read as a treatment effect.
_SYNTH_PREFIT_RATIO_MAX = 0.6


# ====================================================================== #
#  CausalResult helpers
# ====================================================================== #


def _safe_get(obj: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Get ``obj[keys[0]][keys[1]]...`` or ``default`` if any step
    is missing / not a dict."""
    cur: Any = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _as_float(x: Any) -> Optional[float]:
    """Coerce to float or return ``None`` on failure / NaN."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _max_covariate_smd(mi: Dict[str, Any]) -> Optional[float]:
    """Largest absolute post-matching standardized mean difference across
    covariates. Handles the three shapes estimators use: the scalar convention
    (``mi["balance"]["max_smd_after"]``), the per-variable balance table
    ``sp.match`` stores (a DataFrame with ``variable`` / ``smd`` columns), and
    the ``{covariate: smd}`` dict weighting estimators store under
    ``std_mean_diff_after`` (e.g. ``sp.cbps``). The propensity score / distance
    entries are excluded — they are not covariates."""
    _skip = ("propensity_score", "distance")
    scalar = _as_float(_safe_get(mi, "balance", "max_smd_after"))
    if scalar is not None:
        return scalar
    bal = mi.get("balance")
    if isinstance(bal, pd.DataFrame) and {"variable", "smd"}.issubset(bal.columns):
        cov = bal[~bal["variable"].isin(_skip)]
        if not cov.empty:
            return _as_float(cov["smd"].abs().max())
    smd_dict = mi.get("std_mean_diff_after")
    if isinstance(smd_dict, dict) and smd_dict:
        vals = [
            abs(v)
            for k, v in smd_dict.items()
            if str(k) not in _skip and _as_float(v) is not None
        ]
        if vals:
            return _as_float(max(vals))
    return None


def _propensity_extreme_share(mi: Dict[str, Any]) -> Optional[float]:
    """Fraction of units with a near-degenerate propensity (at or beyond the
    overlap bound), read from whichever convention the estimator stored:

    * the cross-fitted propensity array + ``trimming_threshold`` (DML IRM), or
    * the pre-computed ``propensity_diagnostics['clip_share']`` (TMLE).

    Returns ``None`` when no propensity distribution is available (e.g. AIPW
    stores only the mean), so overlap simply is not asserted rather than
    guessed."""
    trim = _as_float(mi.get("trimming_threshold")) or 0.01
    psc = mi.get("_pscore")
    if psc is not None:
        try:
            arr = np.asarray(psc, dtype=float)
            arr = arr[np.isfinite(arr)]
        except (TypeError, ValueError):  # pragma: no cover - defensive
            arr = np.empty(0)
        if arr.size:
            return float(np.mean((arr < trim) | (arr > 1.0 - trim)))
    return _as_float(_safe_get(mi, "propensity_diagnostics", "clip_share"))


def causal_violations(result: Any) -> List[Dict[str, Any]]:
    """Detect assumption / diagnostic violations on a ``CausalResult``.

    Each violation is a dict with keys ``kind`` / ``severity`` /
    ``test`` / ``value`` / ``threshold`` / ``message`` /
    ``recovery_hint`` / ``alternatives``.

    ``severity`` is one of ``"error"`` (identifying assumption
    clearly rejected), ``"warning"`` (borderline / low-power signal),
    or ``"info"`` (worth mentioning but unlikely to change the
    conclusion).
    """
    from .next_steps import _detect_family  # lazy to avoid cycle

    mi: Dict[str, Any] = result.model_info or {}
    # Detect the family from the result's ``method`` attribute, falling back to
    # ``model_info`` keys (``model_type`` / ``method``). Several estimators
    # (e.g. IV) leave ``result.method`` unset but record an identifying
    # ``model_type`` such as ``"IV-2SLS"`` — without this fallback their
    # family-gated checks (weak IV, RD manipulation, …) silently never fire.
    method_family = _detect_family((getattr(result, "method", None) or "").lower())
    if method_family == "generic":
        for _key in ("model_type", "method"):
            _fam = _detect_family(str(mi.get(_key, "")).lower())
            if _fam != "generic":
                method_family = _fam
                break
    out: List[Dict[str, Any]] = []

    # --- DID: parallel trends ------------------------------------------
    pretrend_p = _as_float(_safe_get(mi, "pretrend_test", "pvalue"))
    if pretrend_p is not None and pretrend_p < _PRETREND_ALPHA:
        out.append(
            {
                "kind": "assumption",
                "severity": "error" if pretrend_p < 0.05 else "warning",
                "test": "pretrend",
                "value": pretrend_p,
                "threshold": _PRETREND_ALPHA,
                "message": (
                    f"Pre-trend joint test p = {pretrend_p:.3g} "
                    f"< {_PRETREND_ALPHA} — parallel trends is likely violated."
                ),
                "recovery_hint": (
                    "Run sp.sensitivity_rr(result) for Rambachan & Roth (2023) "
                    "honest CIs, and consider sp.callaway_santanna or "
                    "sp.did_imputation (robust to heterogeneous effects)."
                ),
                "alternatives": [
                    "sp.sensitivity_rr",
                    "sp.callaway_santanna",
                    "sp.did_imputation",
                ],
            }
        )

    # --- IV: weak instruments -------------------------------------------
    first_f = (
        _as_float(_safe_get(mi, "first_stage_f"))
        or _as_float(_safe_get(mi, "first_stage", "f_stat"))
        or _as_float(mi.get("weak_iv_f"))
    )
    if first_f is not None and first_f < _WEAK_IV_F and method_family == "iv":
        out.append(
            {
                "kind": "assumption",
                "severity": "warning",
                "test": "weak_instrument",
                "value": first_f,
                "threshold": _WEAK_IV_F,
                "message": (
                    f"First-stage F = {first_f:.2f} < {_WEAK_IV_F} (Stock-Yogo "
                    "5% bias) — weak instrument bias is likely."
                ),
                "recovery_hint": (
                    "Use sp.anderson_rubin_ci (weak-IV-robust) or "
                    "sp.iv(..., method='liml') which has smaller weak-IV bias."
                ),
                "alternatives": ["sp.anderson_rubin_ci", "sp.iv"],
            }
        )

    # --- Cluster-robust inference: too few clusters ---------------------
    # Gated on the estimator having recorded ``n_clusters`` (i.e. cluster-robust
    # SEs were actually requested) rather than on family, since the small-G
    # problem applies wherever a CRVE is used.
    n_clusters = _as_float(mi.get("n_clusters"))
    if n_clusters is not None and n_clusters < _FEW_CLUSTERS_MIN:
        out.append(
            {
                "kind": "inference",
                "severity": "warning",
                "test": "few_clusters",
                "value": int(n_clusters),
                "threshold": _FEW_CLUSTERS_MIN,
                "message": (
                    f"Only {int(n_clusters)} clusters (< {_FEW_CLUSTERS_MIN}) — "
                    "cluster-robust SEs are downward-biased and t-tests "
                    "over-reject (Cameron-Gelbach-Miller 2008)."
                ),
                "recovery_hint": (
                    "Report sp.wild_cluster_bootstrap (or sp.wild_cluster_ci_inv "
                    "for confidence intervals), which has correct size with few "
                    "clusters."
                ),
                "alternatives": [
                    "sp.wild_cluster_bootstrap",
                    "sp.wild_cluster_ci_inv",
                ],
            }
        )

    # --- Synthetic control: poor pre-treatment fit ----------------------
    # Gated on synth-specific keys rather than family detection. The ratio is
    # scale-free (pre-RMSPE / pre-period SD of the treated outcome); see
    # _SYNTH_PREFIT_RATIO_MAX for the calibration.
    pre_rmspe = _as_float(mi.get("pre_treatment_rmse"))
    if (
        pre_rmspe is not None
        and "n_donors" in mi
        and mi.get("Y_treated") is not None
        and mi.get("times") is not None
        and mi.get("treatment_time") is not None
    ):
        try:
            y_treated = np.asarray(mi["Y_treated"], dtype=float)
            times = np.asarray(mi["times"])
            pre = y_treated[times < mi["treatment_time"]]
            pre_sd = float(np.std(pre, ddof=1)) if pre.size >= 2 else 0.0
        except (TypeError, ValueError):  # pragma: no cover - defensive
            pre_sd = 0.0
        ratio = pre_rmspe / pre_sd if pre_sd > 0 else None
        if ratio is not None and ratio > _SYNTH_PREFIT_RATIO_MAX:
            out.append(
                {
                    "kind": "assumption",
                    "severity": "warning",
                    "test": "synth_prefit",
                    "value": ratio,
                    "threshold": _SYNTH_PREFIT_RATIO_MAX,
                    "message": (
                        f"Pre-treatment fit is poor: RMSPE / pre-period SD = "
                        f"{ratio:.2f} > {_SYNTH_PREFIT_RATIO_MAX}. The synthetic "
                        "control does not track the treated unit before "
                        "treatment, so the post-period gap is unreliable."
                    ),
                    "recovery_hint": (
                        "Improve the donor pool / predictors, compare estimators "
                        "with sp.synth_compare, try sp.augsynth (ridge-augmented), "
                        "and gauge robustness with sp.synth_sensitivity."
                    ),
                    "alternatives": [
                        "sp.synth_compare",
                        "sp.augsynth",
                        "sp.synth_sensitivity",
                    ],
                }
            )

    # --- Matching / weighting: covariate balance -----------------------
    # Gated on the estimator having recorded post-adjustment balance (only
    # matching / weighting estimators do), not on family detection — CBPS and
    # entropy balancing report balance but are not tagged "matching".
    smd_max = _max_covariate_smd(mi)
    if smd_max is not None and smd_max > _SMD_IMBALANCE_MAX:
        out.append(
            {
                "kind": "assumption",
                "severity": "warning",
                "test": "balance",
                "value": smd_max,
                "threshold": _SMD_IMBALANCE_MAX,
                "message": (
                    f"Max standardized mean difference after matching = "
                    f"{smd_max:.3f} > {_SMD_IMBALANCE_MAX} — notable residual "
                    "imbalance, so the matched comparison is still confounded."
                ),
                "recovery_hint": (
                    "Tighten the caliper, add interactions/polynomials to the "
                    "propensity model, or reweight with sp.ebalance "
                    "(entropy balancing) / sp.cbps, and re-check sp.love_plot."
                ),
                "alternatives": ["sp.ebalance", "sp.cbps", "sp.love_plot"],
            }
        )

    # --- Matching / IPW: overlap ----------------------------------------
    overlap = _as_float(_safe_get(mi, "overlap", "min_share"))
    if overlap is not None and overlap < _OVERLAP_MIN:
        out.append(
            {
                "kind": "assumption",
                "severity": "error",
                "test": "overlap",
                "value": overlap,
                "threshold": _OVERLAP_MIN,
                "message": (
                    f"Propensity score overlap min share = {overlap:.3f} "
                    f"< {_OVERLAP_MIN} — thin common support."
                ),
                "recovery_hint": (
                    "Apply Crump (2009) trimming via sp.trimming or narrow "
                    "the estimand to ATT on the overlap region."
                ),
                "alternatives": ["sp.trimming"],
            }
        )

    # --- DML / AIPW / TMLE: propensity overlap --------------------------
    # A large share of units with a near-degenerate propensity (at/beyond the
    # overlap bound) means unstable inverse-propensity weights, so the estimate
    # rests on a handful of influential observations. Read the share from
    # whichever convention the estimator stored: the cross-fitted propensity
    # array + trimming bound (DML IRM) or the pre-computed clipped share (TMLE).
    extreme = _propensity_extreme_share(mi)
    if extreme is not None and extreme > _DML_OVERLAP_EXTREME_SHARE:
        out.append(
            {
                "kind": "assumption",
                "severity": "warning",
                "test": "dml_overlap",
                "value": extreme,
                "threshold": _DML_OVERLAP_EXTREME_SHARE,
                "message": (
                    f"{extreme:.1%} of units have a near-degenerate propensity "
                    "score (at or beyond the overlap bound) — weak overlap, so "
                    "the estimate leans on a few near-degenerate-weight units."
                ),
                "recovery_hint": (
                    "Narrow the estimand to the overlap region with "
                    "sp.trimming (Crump 2009), reweight with "
                    "sp.overlap_weights (Li et al. 2018), or improve the "
                    "propensity model (sp.cbps)."
                ),
                "alternatives": [
                    "sp.trimming",
                    "sp.overlap_weights",
                    "sp.cbps",
                ],
            }
        )

    # --- Heckman: selection correction ---------------------------------
    # Gated on the two-step selection keys the estimator stores.
    rho = _as_float(mi.get("rho"))
    lambda_p = _as_float(mi.get("lambda_pvalue"))
    if rho is not None and lambda_p is not None:
        if abs(rho) > _HECKMAN_RHO_BOUNDARY:
            out.append(
                {
                    "kind": "numerical",
                    "severity": "warning",
                    "test": "heckman_rho_boundary",
                    "value": rho,
                    "threshold": _HECKMAN_RHO_BOUNDARY,
                    "message": (
                        f"Selection correlation rho = {rho:.3f} sits on the ±1 "
                        "boundary — a weak exclusion restriction is making the "
                        "two-step estimates unstable."
                    ),
                    "recovery_hint": (
                        "Strengthen the exclusion restriction (a selection-only "
                        "instrument), or compare against sp.regress on the "
                        "selected sample."
                    ),
                    "alternatives": ["sp.regress", "sp.ipw"],
                }
            )
        elif lambda_p > _HECKMAN_SELECTION_P:
            out.append(
                {
                    "kind": "assumption",
                    "severity": "info",
                    "test": "heckman_no_selection",
                    "value": lambda_p,
                    "threshold": _HECKMAN_SELECTION_P,
                    "message": (
                        f"Inverse-Mills term is not significant (p = {lambda_p:.3g})"
                        " — no detectable selection, so the correction only adds "
                        "variance."
                    ),
                    "recovery_hint": (
                        "OLS on the selected sample (sp.regress) is consistent "
                        "here and more efficient than the Heckman two-step."
                    ),
                    "alternatives": ["sp.regress"],
                }
            )

    # --- Tobit: extreme censoring --------------------------------------
    censor_pct = _as_float(mi.get("censor_pct"))
    if (
        censor_pct is not None
        and "n_censored" in mi
        and censor_pct > _TOBIT_CENSOR_PCT_MAX
    ):
        out.append(
            {
                "kind": "data",
                "severity": "warning",
                "test": "extreme_censoring",
                "value": censor_pct,
                "threshold": _TOBIT_CENSOR_PCT_MAX,
                "message": (
                    f"{censor_pct:.0f}% of observations are censored — the "
                    "likelihood is dominated by the limit and the Tobit slope "
                    "estimates are fragile."
                ),
                "recovery_hint": (
                    "Model the censoring explicitly (sp.heckman two-part), or "
                    "report bounds; treat the point estimates with caution."
                ),
                "alternatives": ["sp.heckman"],
            }
        )

    # --- Bayesian: convergence ------------------------------------------
    rhat = _as_float(mi.get("rhat_max") or _safe_get(mi, "diagnostics", "rhat_max"))
    if rhat is not None and rhat > _RHAT_MAX:
        out.append(
            {
                "kind": "convergence",
                "severity": "error",
                "test": "rhat",
                "value": rhat,
                "threshold": _RHAT_MAX,
                "message": (
                    f"Max R-hat = {rhat:.3f} > {_RHAT_MAX} — MCMC has not mixed."
                ),
                "recovery_hint": (
                    "Increase ``tune`` (≥ 4000), check for divergences, "
                    "reparameterize (non-centered), or verify priors."
                ),
                "alternatives": [],
            }
        )

    ess = _as_float(
        mi.get("ess_bulk_min") or _safe_get(mi, "diagnostics", "ess_bulk_min")
    )
    if ess is not None and ess < _ESS_MIN:
        out.append(
            {
                "kind": "convergence",
                "severity": "warning",
                "test": "ess_bulk",
                "value": ess,
                "threshold": _ESS_MIN,
                "message": (
                    f"Min bulk effective sample size = {ess:.0f} < {_ESS_MIN}."
                ),
                "recovery_hint": "Increase draws, or rerun with more chains.",
                "alternatives": [],
            }
        )

    divs = mi.get("divergences") or _safe_get(mi, "diagnostics", "divergences")
    divs_val = _as_float(divs)
    if divs_val is not None and divs_val > _DIVERGENCES_MAX:
        out.append(
            {
                "kind": "convergence",
                "severity": "error",
                "test": "divergences",
                "value": divs_val,
                "threshold": _DIVERGENCES_MAX,
                "message": (
                    f"{int(divs_val)} post-warmup divergent transitions — "
                    "posterior geometry is problematic."
                ),
                "recovery_hint": (
                    "Raise ``target_accept`` to 0.95+ and/or reparameterize."
                ),
                "alternatives": [],
            }
        )

    # --- RD: manipulation (density test; key kept as 'mccrary') ----------
    mccrary_p = _as_float(_safe_get(mi, "mccrary", "pvalue"))
    if mccrary_p is not None and mccrary_p < 0.05 and method_family == "rd":
        out.append(
            {
                "kind": "assumption",
                "severity": "error",
                "test": "mccrary_density",
                "value": mccrary_p,
                "threshold": 0.05,
                "message": (
                    f"Density test (rddensity) p = {mccrary_p:.3g} < 0.05 — "
                    "running variable may be manipulated at the cutoff."
                ),
                "recovery_hint": (
                    "Inspect the density with sp.rdplotdensity; "
                    "manipulation undermines RD identification, so also consider "
                    "excluding a donut window around the cutoff or reporting "
                    "partial-identification bounds."
                ),
                "alternatives": ["sp.rddensity"],
            }
        )

    # --- NaN / degenerate estimate --------------------------------------
    est = _as_float(result.estimate)
    se = _as_float(result.se)
    if est is None:
        out.append(
            {
                "kind": "numerical",
                "severity": "error",
                "test": "estimate_finite",
                "value": result.estimate,
                "threshold": None,
                "message": "Point estimate is NaN or ±inf.",
                "recovery_hint": "Check data for perfect collinearity / zero variance.",
                "alternatives": [],
            }
        )
    if se is None or (se is not None and se <= 0):
        out.append(
            {
                "kind": "numerical",
                "severity": "error",
                "test": "se_positive",
                "value": result.se,
                "threshold": 0,
                "message": "Standard error is non-positive / NaN.",
                "recovery_hint": (
                    "Check sandwich / cluster setup; inspect influence functions."
                ),
                "alternatives": [],
            }
        )

    return out


def causal_agent_summary(result: Any) -> Dict[str, Any]:
    """Return a JSON-ready structured summary of a ``CausalResult``.

    Payload (all keys always present; empty containers when N/A):

    * ``method`` / ``method_family`` — estimator identity
    * ``estimand`` — ``"ATT"`` / ``"ATE"`` / ``"LATE"`` / etc.
    * ``point`` — dict with ``estimate`` / ``se`` / ``ci`` / ``pvalue``
      / ``alpha``
    * ``n_obs`` — sample size
    * ``diagnostics`` — the scalar-valued entries from
      ``model_info`` (DataFrames/arrays are replaced with a
      ``"<type>(shape)"`` placeholder so the output stays JSON-ready)
    * ``violations`` — output of :func:`causal_violations`
    * ``next_steps`` — output of ``result.next_steps(print_result=False)``
    * ``citation_key`` — key into :attr:`CausalResult._CITATIONS`
    """
    from .next_steps import _detect_family

    method = result.method or ""
    family = _detect_family(method.lower())

    est = _as_float(result.estimate)
    se = _as_float(result.se)
    pval = _as_float(result.pvalue)
    ci_lo = _as_float(result.ci[0]) if result.ci else None
    ci_hi = _as_float(result.ci[1]) if result.ci else None

    # Flatten scalar diagnostics so the payload stays JSON-safe.
    mi = result.model_info or {}
    scalar_diagnostics: Dict[str, Any] = {}
    for key, val in mi.items():
        if isinstance(val, (str, int, float, bool)) or val is None:
            scalar_diagnostics[key] = val
        elif isinstance(val, (pd.DataFrame, pd.Series, np.ndarray)):
            scalar_diagnostics[key] = (
                f"<{type(val).__name__} shape={getattr(val, 'shape', '?')}>"
            )
        elif isinstance(val, dict):
            # One level deep is enough for most diagnostic subtrees.
            nested = {}
            for k2, v2 in val.items():
                if isinstance(v2, (str, int, float, bool)) or v2 is None:
                    nested[k2] = v2
            if nested:
                scalar_diagnostics[key] = nested

    try:
        next_steps = result.next_steps(print_result=False)
    except Exception:  # pragma: no cover - defensive
        next_steps = []

    return {
        "kind": "causal_result",
        "method": method,
        "method_family": family,
        "estimand": result.estimand,
        "point": {
            "estimate": est,
            "se": se,
            "pvalue": pval,
            "ci": [ci_lo, ci_hi] if (ci_lo is not None and ci_hi is not None) else None,
            "alpha": _as_float(result.alpha),
        },
        "n_obs": int(result.n_obs) if result.n_obs is not None else None,
        "diagnostics": scalar_diagnostics,
        "violations": causal_violations(result),
        "next_steps": next_steps,
        "citation_key": getattr(result, "_citation_key", None),
    }


# ====================================================================== #
#  EconometricResults helpers
# ====================================================================== #


def econometric_violations(result: Any) -> List[Dict[str, Any]]:
    """Detect common violations on an :class:`EconometricResults`."""
    out: List[Dict[str, Any]] = []
    diag: Dict[str, Any] = getattr(result, "diagnostics", None) or {}
    mi: Dict[str, Any] = getattr(result, "model_info", None) or {}
    model_type = (mi.get("model_type", "") or "").lower()

    # IV weak-instrument check
    first_f = (
        _as_float(mi.get("first_stage_f"))
        or _as_float(_safe_get(mi, "first_stage", "f_stat"))
        or _as_float(diag.get("first_stage_f"))
    )
    is_iv = any(k in model_type for k in ("iv", "2sls", "liml", "gmm"))
    if is_iv and first_f is not None and first_f < _WEAK_IV_F:
        out.append(
            {
                "kind": "assumption",
                "severity": "warning",
                "test": "weak_instrument",
                "value": first_f,
                "threshold": _WEAK_IV_F,
                "message": (
                    f"First-stage F = {first_f:.2f} < {_WEAK_IV_F} — weak "
                    "instrument bias likely."
                ),
                "recovery_hint": (
                    "Use sp.anderson_rubin_ci or sp.iv(..., method='liml')."
                ),
                "alternatives": ["sp.anderson_rubin_ci", "sp.iv"],
            }
        )

    # Too few clusters for reliable cluster-robust inference. Gated on the
    # estimator having recorded ``n_clusters`` (cluster-robust SEs were
    # requested), mirroring the fit-time warning so the two never disagree.
    n_clusters = _as_float(mi.get("n_clusters"))
    if n_clusters is not None and n_clusters < _FEW_CLUSTERS_MIN:
        out.append(
            {
                "kind": "inference",
                "severity": "warning",
                "test": "few_clusters",
                "value": int(n_clusters),
                "threshold": _FEW_CLUSTERS_MIN,
                "message": (
                    f"Only {int(n_clusters)} clusters (< {_FEW_CLUSTERS_MIN}) — "
                    "cluster-robust SEs are downward-biased and t-tests "
                    "over-reject (Cameron-Gelbach-Miller 2008)."
                ),
                "recovery_hint": FEW_CLUSTERS_HINT,
                "alternatives": [
                    "sp.wild_cluster_bootstrap",
                    "sp.wild_cluster_ci_inv",
                ],
            }
        )

    # Logit / probit (quasi-)complete separation. A slope coefficient this
    # large is the MLE diverging because a predictor perfectly splits the
    # outcome, not a real effect — the point estimate and its SE are unusable.
    family = (mi.get("family", "") or "").lower()
    is_binary_glm = (
        "logit" in model_type or "probit" in model_type or family == "binomial"
    )
    if is_binary_glm:
        params = getattr(result, "params", None)
        if params is not None:
            try:
                slopes = params.drop(
                    [
                        i
                        for i in params.index
                        if str(i).lower() in ("intercept", "const")
                    ],
                    errors="ignore",
                )
                max_abs = float(np.max(np.abs(slopes.values))) if len(slopes) else 0.0
            except (AttributeError, TypeError, ValueError, KeyError):
                max_abs = 0.0
            if max_abs > _LOGIT_SEPARATION_COEF:
                out.append(
                    {
                        "kind": "numerical",
                        "severity": "error",
                        "test": "separation",
                        "value": max_abs,
                        "threshold": _LOGIT_SEPARATION_COEF,
                        "message": (
                            f"A coefficient of {max_abs:.1f} indicates "
                            "(quasi-)complete separation — a predictor perfectly "
                            "splits the outcome and the MLE has diverged, so the "
                            "estimate and its SE are meaningless."
                        ),
                        "recovery_hint": (
                            "Use penalised logistic regression (Firth), drop or "
                            "combine the separating predictor, or report an exact "
                            "/ profile-likelihood interval."
                        ),
                        "alternatives": ["sp.logit", "sp.rlasso"],
                    }
                )

    # Poisson over-dispersion. Compute the Pearson dispersion from the stored
    # fitted values (not a re-fit) — variance far above the mean means Poisson
    # SEs are too small and a negative-binomial / robust fit is warranted.
    if "poisson" in model_type or family == "poisson":
        di = getattr(result, "data_info", None) or {}
        mu = di.get("fitted_values")
        yv = di.get("y")
        df_resid = _as_float(di.get("df_resid"))
        if mu is not None and yv is not None and df_resid and df_resid > 0:
            try:
                mu_a = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
                y_a = np.asarray(yv, dtype=float)
                dispersion = float(np.sum((y_a - mu_a) ** 2 / mu_a) / df_resid)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                dispersion = None
            if dispersion is not None and dispersion > _POISSON_DISPERSION_MAX:
                out.append(
                    {
                        "kind": "assumption",
                        "severity": "warning",
                        "test": "overdispersion",
                        "value": dispersion,
                        "threshold": _POISSON_DISPERSION_MAX,
                        "message": (
                            f"Pearson dispersion = {dispersion:.2f} > "
                            f"{_POISSON_DISPERSION_MAX} — the variance exceeds the "
                            "mean, so Poisson standard errors are too small and "
                            "t-tests over-reject."
                        ),
                        "recovery_hint": (
                            "Refit with sp.nbreg (negative binomial) for "
                            "over-dispersion, or keep Poisson point estimates "
                            "with robust='hc1' SEs (quasi-Poisson)."
                        ),
                        "alternatives": ["sp.nbreg", "sp.zinb", "sp.poisson"],
                    }
                )

    # Count models: excess zeros. Compare the observed zero share to the share
    # the fitted model predicts (Poisson e^-mu, or the NB2 formula using the
    # stored dispersion) — a large gap means a zero-inflated model is needed.
    is_count = (
        "poisson" in model_type
        or "negbin" in model_type
        or family in ("poisson", "negative binomial")
    )
    if is_count:
        di = getattr(result, "data_info", None) or {}
        mu = di.get("fitted_values")
        yv = di.get("y")
        if mu is not None and yv is not None:
            try:
                mu_a = np.clip(np.asarray(mu, dtype=float), 1e-10, None)
                y_a = np.asarray(yv, dtype=float)
                alpha = _as_float(mi.get("dispersion"))
                if ("negbin" in model_type or family == "negative binomial") and (
                    alpha is not None and alpha > 1e-6
                ):
                    pred0 = float(np.mean((1.0 + alpha * mu_a) ** (-1.0 / alpha)))
                else:
                    pred0 = float(np.mean(np.exp(-mu_a)))
                obs0 = float(np.mean(y_a == 0))
            except (TypeError, ValueError):
                obs0 = pred0 = 0.0
            if obs0 - pred0 > _COUNT_EXCESS_ZERO_MARGIN:
                out.append(
                    {
                        "kind": "assumption",
                        "severity": "warning",
                        "test": "excess_zeros",
                        "value": obs0 - pred0,
                        "threshold": _COUNT_EXCESS_ZERO_MARGIN,
                        "message": (
                            f"Observed zeros ({obs0:.1%}) far exceed the "
                            f"{pred0:.1%} the model predicts — the data are "
                            "zero-inflated, so a plain count model understates "
                            "the zero process."
                        ),
                        "recovery_hint": (
                            "Fit a zero-inflated model: sp.zip_model (Poisson) or "
                            "sp.zinb (negative binomial), or a hurdle model."
                        ),
                        "alternatives": ["sp.zip_model", "sp.zinb"],
                    }
                )

    # Cox: proportional-hazards assumption (from the stored ph_test p-value).
    ph_p = _as_float(_safe_get(mi, "ph_test", "min_pvalue"))
    if "cox" in model_type and ph_p is not None and ph_p < _COX_PH_ALPHA:
        worst = _safe_get(mi, "ph_test", "worst_variable")
        out.append(
            {
                "kind": "assumption",
                "severity": "warning",
                "test": "proportional_hazards",
                "value": ph_p,
                "threshold": _COX_PH_ALPHA,
                "message": (
                    f"Proportional-hazards test rejects (p = {ph_p:.3g}"
                    + (f" for '{worst}'" if worst else "")
                    + ") — the hazard ratio is not constant over time, so the "
                    "reported Cox coefficient is a time-averaged blur."
                ),
                "recovery_hint": (
                    "Inspect result.ph_test(); add a time interaction / stratify "
                    "on the offending covariate, or model it with sp.aft."
                ),
                "alternatives": ["sp.aft", "sp.cox"],
            }
        )

    # Non-positive SE
    ses = getattr(result, "std_errors", None)
    try:
        if ses is not None and (ses <= 0).any():
            bad = [str(k) for k in ses.index[ses <= 0]]
            out.append(
                {
                    "kind": "numerical",
                    "severity": "error",
                    "test": "se_positive",
                    "value": bad,
                    "threshold": 0,
                    "message": f"Non-positive SE on: {bad}",
                    "recovery_hint": (
                        "Inspect collinearity (sp.estat(result, 'vif')) and "
                        "sandwich / cluster setup."
                    ),
                    "alternatives": [],
                }
            )
    except Exception:  # pragma: no cover - defensive
        pass

    return out


def _positional(arr: Any, i: int) -> Optional[float]:
    """Read element ``i`` from a Series or ndarray, best-effort."""
    if arr is None:
        return None
    # Series: prefer iloc so we don't depend on label alignment.
    if hasattr(arr, "iloc"):
        try:
            return _as_float(arr.iloc[i])
        except Exception:
            return None
    try:
        return _as_float(arr[i])
    except Exception:
        return None


def econometric_agent_summary(result: Any) -> Dict[str, Any]:
    """JSON-ready structured summary of an :class:`EconometricResults`."""
    params = getattr(result, "params", None)
    ses = getattr(result, "std_errors", None)
    pvals = getattr(result, "pvalues", None)
    tvals = getattr(result, "tvalues", None)

    coefs: List[Dict[str, Any]] = []
    if params is not None and hasattr(params, "index"):
        for i, name in enumerate(params.index):
            coefs.append(
                {
                    "term": str(name),
                    "estimate": _positional(params, i),
                    "std_error": _positional(ses, i),
                    "statistic": _positional(tvals, i),
                    "p_value": _positional(pvals, i),
                }
            )

    mi = getattr(result, "model_info", None) or {}
    data_info = getattr(result, "data_info", None) or {}
    diag = getattr(result, "diagnostics", None) or {}

    scalar_diagnostics = {
        k: v
        for k, v in diag.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }

    try:
        next_steps = result.next_steps(print_result=False)
    except Exception:  # pragma: no cover - defensive
        next_steps = []

    return {
        "kind": "econometric_result",
        "model_type": mi.get("model_type", ""),
        "robust": mi.get("robust", "nonrobust"),
        "n_obs": (
            int(data_info.get("nobs", 0)) if data_info.get("nobs") is not None else None
        ),
        "df_resid": data_info.get("df_resid"),
        "dependent_var": data_info.get("dependent_var", ""),
        "coefficients": coefs,
        "diagnostics": scalar_diagnostics,
        "violations": econometric_violations(result),
        "next_steps": next_steps,
    }


def note_weight_diagnostics(
    model_info: Dict[str, Any],
    weights: Any,
    variance: str,
    *,
    entry: str,
    robust_option: str,
    classical_reading: str,
    classical_assumption: str = "precisions (error variance proportional to 1 / w)",
    cluster_keys: Any = None,
    cluster: Any = None,
    stacklevel: int = 4,
) -> None:
    """Record (and warn on) the weight diagnostics of a weighted fit.

    For the entry points that have no HC3: ``sp.panel``, ``sp.hdfe_ols``,
    ``sp.feols``, ``sp.poisson``, ``sp.ppmlhdfe``. Writes
    ``n_effective_weights`` and, for a one-way clustered fit,
    ``n_clusters_effective_weights`` into ``model_info``.
    """
    model_info["n_effective_weights"] = warn_if_weights_extreme(
        weights,
        variance,
        stacklevel=stacklevel,
        entry=entry,
        robust_option=robust_option,
        small_sample_option=None,
        classical_reading=classical_reading,
        classical_assumption=classical_assumption,
        evidence="tests/reliability/extreme_weights_models_results.json",
    )
    if variance == "cluster" and cluster_keys is not None:
        model_info["n_clusters_effective_weights"] = warn_if_weighted_clusters_few(
            weights, cluster_keys, cluster, stacklevel=stacklevel
        )
