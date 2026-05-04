"""
Resampling-stability checks for estimator recommendations.

The ``recommend()`` engine is a rule-based decision tree: given data
shape and research design, it returns ranked methods as a *prior*.
This module runs each recommended method on resamples of the observed
data and produces a composite **stability_score** in [0, 100]:

1. **Bootstrap stability** — coefficient of variation across B bootstraps.
   Low CV → estimate is stable under row/cluster resampling.
2. **Placebo pass rate** — share of permuted-treatment runs with p>0.10.
   Note: unconditional permutation destroys confounder structure and
   therefore has limited power for selection-on-observables designs.
3. **Subsample agreement** — sign agreement across K random 50% splits.
   High agreement → robust to sample composition.

WHAT THIS SCORE IS NOT
----------------------
The composite is a **stability** score, not a validity score. It does
NOT establish:

- that identifying assumptions hold (unconfoundedness, parallel trends,
  exclusion restrictions, continuity at the cutoff, ...);
- that the method is unbiased for the estimand of interest;
- that the method beats alternatives on MSE against ground truth.

A biased estimator on observational data can be perfectly stable, pass
the (unconditional) placebo test, and agree across subsamples — and
score near 100. Treat this score as "does the method behave predictably
on this dataset," not as empirical evidence that it is correct.

Design notes
------------
- Opt-in only (``verify=False`` default in ``recommend()``).
- Time budget enforced; B automatically reduced if base method is slow.
- Failures per-method are caught — one bad fit does not poison the run.
- Independent RNG streams for bootstrap / placebo / subsample so that
  results are reproducible even when the bootstrap budget is exhausted
  early.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..workflow._degradation import record_degradation

__all__ = ["verify_recommendation"]


_ID_KEYS = ("id", "entity", "i", "unit")
_TREAT_KEYS = ("treatment", "treat", "g", "x_endog")


def _get_id_col(rec: Dict[str, Any]) -> Optional[str]:
    params = rec.get("params", {})
    for k in _ID_KEYS:
        v = params.get(k)
        if isinstance(v, str):
            return v
    return None


def _get_treat_col(rec: Dict[str, Any], data: pd.DataFrame) -> Optional[str]:
    # Explicit raw treatment override wins (used by CSA/SA where params.g
    # points to a derived cohort column that won't exist in raw bootstraps).
    raw = rec.get("raw_treat")
    if isinstance(raw, str) and raw in data.columns:
        return raw
    params = rec.get("params", {})
    for k in _TREAT_KEYS:
        v = params.get(k)
        if isinstance(v, str) and v in data.columns:
            return v
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            if v[0] in data.columns:
                return v[0]
    # Formula style: try to parse "y ~ treat + ..." or "y ~ (endog ~ z)"
    formula = params.get("formula", "")
    if "~" in formula:
        rhs = formula.split("~", 1)[1]
        # IV syntax: "y ~ exog + (endog ~ z)" — prefer endogenous
        if "(" in rhs and "~" in rhs:
            endog_part = rhs.split("(", 1)[1].split("~", 1)[0]
            tokens = [t.strip() for t in endog_part.replace("+", " ").split()]
            for t in tokens:
                if t in data.columns:
                    return t
        tokens = [t.strip() for t in rhs.replace("+", " ").replace("(", " ").replace(")", " ").split()]
        for t in tokens:
            if t in data.columns:
                return t
    return None


def _extract_estimate(result: Any, treat_name: Optional[str] = None) -> Optional[float]:
    """Extract the treatment coefficient from a StatsPAI result object.

    Handles CausalResult (scalar ``estimate``/``att``/``ate``) and
    EconometricResults (pandas Series ``params`` indexed by variable).
    """
    # CausalResult path: scalar attributes
    for attr in ("att", "ate", "estimate", "coef", "beta"):
        val = getattr(result, attr, None)
        if val is None:
            continue
        if np.isscalar(val):
            return float(val)
        try:
            arr = np.asarray(val).ravel()
            if arr.size == 1:
                return float(arr[0])
        except Exception:
            pass

    # EconometricResults path: params is a pandas Series indexed by name
    params = getattr(result, "params", None)
    if isinstance(params, pd.Series) and len(params):
        if treat_name and treat_name in params.index:
            return float(params[treat_name])
        # Otherwise take first non-intercept-looking entry
        for name in params.index:
            low = str(name).lower()
            if low in ("intercept", "const", "(intercept)"):
                continue
            return float(params[name])
        return float(params.iloc[-1])
    return None


def _extract_pvalue(result: Any, treat_name: Optional[str] = None) -> Optional[float]:
    """Extract the p-value for the treatment coefficient."""
    for attr in ("pvalue", "p_value"):
        val = getattr(result, attr, None)
        if val is None:
            continue
        if np.isscalar(val):
            return float(val)
        try:
            arr = np.asarray(val).ravel()
            if arr.size == 1:
                return float(arr[0])
        except Exception:
            pass

    pvalues = getattr(result, "pvalues", None)
    if isinstance(pvalues, pd.Series) and len(pvalues):
        if treat_name and treat_name in pvalues.index:
            return float(pvalues[treat_name])
        for name in pvalues.index:
            low = str(name).lower()
            if low in ("intercept", "const", "(intercept)"):
                continue
            return float(pvalues[name])
    return None


def _run_method(rec: Dict[str, Any], data: pd.DataFrame) -> Optional[float]:
    """Execute a recommendation on ``data``, return point estimate.

    Returns None on any failure (the method itself is responsible for
    raising; we catch and return None so a single bad fit does not
    abort the whole verification).
    """
    import statspai as sp

    func = getattr(sp, rec["function"], None)
    if func is None or not callable(func):
        return None

    params = dict(rec.get("params", {}))
    prep = rec.get("prep")
    prepared = prep(data) if callable(prep) else data
    params["data"] = prepared

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = func(**params)
    except Exception as exc:
        # Cross-method verification deliberately tolerates per-method
        # failures so a single bad fit doesn't abort the cross-check.
        # But silent skip would let "all methods crashed identically"
        # masquerade as "all methods agreed at None" — surface the
        # failure so users can spot pattern (e.g. shared signature bug).
        record_degradation(
            None,
            section="verify_recommendation: per-method fit",
            exc=exc,
            detail=f"function=sp.{rec.get('function', '?')}",
        )
        return None

    treat_name = _get_treat_col(rec, prepared)
    return _extract_estimate(result, treat_name=treat_name)


def _bootstrap_stability(
    rec: Dict[str, Any],
    data: pd.DataFrame,
    B: int,
    rng: np.random.Generator,
    deadline: float,
) -> Dict[str, Any]:
    """CV of point estimate across B bootstrap resamples (cluster-aware)."""
    estimates = []
    n = len(data)
    id_col = _get_id_col(rec)

    for _ in range(B):
        if time.perf_counter() > deadline:
            break
        if id_col and id_col in data.columns:
            units = data[id_col].unique()
            sampled = rng.choice(units, size=len(units), replace=True)
            boot = pd.concat(
                [data[data[id_col] == u] for u in sampled], ignore_index=True
            )
        else:
            idx = rng.integers(0, n, n)
            boot = data.iloc[idx].reset_index(drop=True)

        est = _run_method(rec, boot)
        if est is not None and np.isfinite(est):
            estimates.append(est)

    if len(estimates) < 3:
        return {"n_success": len(estimates), "cv": np.nan, "score": 0.0}

    arr = np.asarray(estimates)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    # Near-zero-mean handling: the coefficient of variation explodes when
    # the true effect is near zero (a CORRECT null-effect estimator is
    # not "unstable" just because it estimates zero stably). Fall back
    # to a scale-free dispersion metric — IQR / (|median| + data MAD) —
    # so a well-behaved null-effect scores high rather than zero.
    if abs(mean) > 1e-3 * (np.abs(arr).max() + 1e-12):
        cv = std / abs(mean)
        metric = "cv"
    else:
        q1, med, q3 = np.quantile(arr, [0.25, 0.5, 0.75])
        iqr = float(q3 - q1)
        mad = float(np.median(np.abs(arr - med))) + 1e-12
        # Scale dispersion by MAD (robust scale of estimates themselves)
        cv = iqr / (abs(med) + mad)
        metric = "iqr_over_medmad"
    # Score: CV of 0 → 100, CV of 1.0 → 0, clamped
    score = max(0.0, min(100.0, 100.0 * (1.0 - min(cv, 1.0))))
    return {
        "n_success": len(estimates),
        "cv": cv,
        "score": score,
        "metric": metric,
    }


def _placebo_pass(
    rec: Dict[str, Any],
    data: pd.DataFrame,
    rng: np.random.Generator,
    n_reps: int = 5,
) -> Dict[str, Any]:
    """Permutation test: shuffle treatment, expect null.

    Returns share of placebo runs where p>0.10 (correctly non-significant).

    Three distinct outcomes:
    - ``applicable=False`` (score=NaN): treatment column not identifiable
      from the recommendation — placebo cannot run at all. This drops
      the component from the composite average (NaN-safe).
    - ``applicable=True, total=0`` (score=0): placebo ran but no p-value
      could be extracted from any rep. We score 0 rather than NaN so
      methods that silently lack a p-value are penalised rather than
      escaping the placebo check via NaN renormalisation.
    - ``applicable=True, total>0`` (score in [0,100]): normal path.
    """
    treat_col = _get_treat_col(rec, data)
    if not treat_col or treat_col not in data.columns:
        # Genuinely not applicable — e.g. RD, where "treatment" isn't a
        # permutable column. Return NaN so the composite drops it.
        return {
            "score": np.nan, "passed": 0, "total": 0, "applicable": False
        }

    id_col = _get_id_col(rec)
    n_pass = 0
    n_total = 0
    import statspai as sp

    func = getattr(sp, rec["function"], None)
    if func is None:
        return {
            "score": np.nan, "passed": 0, "total": 0, "applicable": False
        }

    last_exc: Optional[BaseException] = None
    n_crashes = 0
    for _ in range(n_reps):
        permuted = data.copy()
        if id_col and id_col in data.columns:
            units = permuted[id_col].unique()
            perm = rng.permutation(units)
            mapping = dict(zip(units, perm))
            # Shuffle the treatment assignment at the unit level
            unit_treat = permuted.groupby(id_col)[treat_col].first()
            new_treat = permuted[id_col].map(
                {u: unit_treat[mapping[u]] for u in units}
            )
            permuted[treat_col] = new_treat.values
        else:
            permuted[treat_col] = rng.permutation(permuted[treat_col].values)

        params = dict(rec.get("params", {}))
        prep = rec.get("prep")
        prep_data = prep(permuted) if callable(prep) else permuted
        params["data"] = prep_data
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = func(**params)
            p_val = _extract_pvalue(result, treat_name=treat_col)
            if p_val is None:
                continue
            n_total += 1
            if np.isfinite(p_val) and p_val > 0.10:
                n_pass += 1
        except Exception as exc:
            # Per-rep failures are intentionally tolerated (one bad
            # permutation shouldn't kill the placebo battery), but
            # remember the last failure so we can surface it when ALL
            # reps crash — otherwise "every fit raised" looks identical
            # to "method genuinely failed placebo" (both score=0).
            last_exc = exc
            n_crashes += 1
            continue

    if n_total:
        score = 100.0 * n_pass / n_total
    else:
        # Placebo is applicable (treatment column resolved, method
        # callable) but no p-value could be extracted. Penalise this
        # case with score=0 so it doesn't silently escape via NaN.
        score = 0.0
        # If ALL reps crashed (vs. some returned None p-values), the
        # score=0 is hiding a likely shared bug — surface it as a
        # single degradation warning rather than 5 silent skips.
        if n_crashes == n_reps and last_exc is not None:
            from ..workflow._degradation import record_degradation
            record_degradation(
                None,
                section="verify_recommendation: placebo all-reps crashed",
                exc=last_exc,
                detail=(
                    f"function=sp.{rec.get('function', '?')}, "
                    f"n_reps={n_reps}, last error from rep #{n_reps}"
                ),
            )
    return {
        "score": score, "passed": n_pass, "total": n_total,
        "applicable": True,
    }


def _subsample_agreement(
    rec: Dict[str, Any],
    data: pd.DataFrame,
    K: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Sign agreement across K random 50% subsamples."""
    estimates = []
    id_col = _get_id_col(rec)
    for _ in range(K):
        if id_col and id_col in data.columns:
            units = data[id_col].unique()
            keep = rng.choice(units, size=max(2, len(units) // 2), replace=False)
            sub = data[data[id_col].isin(keep)].copy()
        else:
            sub = data.sample(frac=0.5, random_state=int(rng.integers(1 << 30))).copy()
        est = _run_method(rec, sub)
        if est is not None and np.isfinite(est):
            estimates.append(est)

    if len(estimates) < 2:
        return {"score": np.nan, "n_success": len(estimates)}

    signs = np.sign(estimates)
    most_common = max(np.sum(signs > 0), np.sum(signs < 0), np.sum(signs == 0))
    score = 100.0 * most_common / len(estimates)
    return {"score": score, "n_success": len(estimates)}


def verify_recommendation(
    rec: Dict[str, Any],
    data: pd.DataFrame,
    B: int = 50,
    K_subsample: int = 5,
    n_placebo: int = 5,
    budget_s: float = 30.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Empirically verify a single recommendation.

    Parameters
    ----------
    rec : dict
        A single entry from ``RecommendationResult.recommendations``.
    data : pd.DataFrame
        The original dataset.
    B : int
        Number of bootstrap replications (auto-reduced if over budget).
    K_subsample : int
        Number of 50% subsample splits.
    n_placebo : int
        Number of permutation placebo runs.
    budget_s : float
        Wall-clock budget per recommendation (seconds).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict with keys: ``score`` (0-100 stability composite, NOT a
    validity score), ``stability``, ``placebo``, ``subsample``,
    ``elapsed_s``, ``B_used``.

    Notes
    -----
    The score measures resampling behavior, not identification
    validity. A biased estimator can score high. See module docstring.
    """
    # Independent RNG streams per component so that non-determinism in
    # bootstrap timing does not leak into placebo/subsample draws.
    rng_boot = np.random.default_rng(seed)
    rng_placebo = np.random.default_rng(seed + 1)
    rng_sub = np.random.default_rng(seed + 2)
    t_start = time.perf_counter()

    # First probe base runtime so we can cap B within budget
    t0 = time.perf_counter()
    base_est = _run_method(rec, data)
    t_base = time.perf_counter() - t0

    # If we can't even run the method on the full data, bail out with a
    # clear signal rather than silently scoring zero.
    if base_est is None:
        return {
            "score": np.nan,
            "error": "method_execution_failed",
            "elapsed_s": time.perf_counter() - t_start,
            "B_used": 0,
            "base_runtime_s": t_base,
        }

    # Reserve ~30% of budget for placebo + subsample; rest for bootstrap
    boot_budget = budget_s * 0.7
    B_used = B
    if t_base > 0:
        max_B = max(5, int(boot_budget / t_base))
        B_used = min(B, max_B)

    deadline = t_start + boot_budget
    stability = _bootstrap_stability(rec, data, B_used, rng_boot, deadline)
    placebo = _placebo_pass(rec, data, rng_placebo, n_reps=n_placebo)
    subsample = _subsample_agreement(rec, data, K=K_subsample, rng=rng_sub)

    # Combine: weights reflect our confidence in each signal
    weights = []
    scores = []
    for w, s in [(0.4, stability["score"]),
                 (0.3, placebo["score"]),
                 (0.3, subsample["score"])]:
        if np.isfinite(s):
            weights.append(w)
            scores.append(s)
    combined = (
        float(np.average(scores, weights=weights))
        if weights
        else np.nan
    )

    return {
        "score": combined,
        "stability": stability,
        "placebo": placebo,
        "subsample": subsample,
        "elapsed_s": time.perf_counter() - t_start,
        "B_used": B_used,
        "base_runtime_s": t_base,
    }
