"""
Clinical diagnostic-test performance metrics.

Standard clinical-epi primitives for evaluating a binary diagnostic
test against reference labels:

- Sensitivity (true-positive rate) and specificity (true-negative rate)
- Positive / negative predictive values (with prevalence-adjusted forms)
- Likelihood ratios (LR+, LR-)
- ROC curve + AUC (Mann-Whitney, ties counted one half) with Hanley-McNeil
  or DeLong SE
- Cohen's kappa (inter-rater agreement, linear or quadratic weighted)

All functions accept either:
  - A 2x2 confusion matrix (``tp, fn, fp, tn``), or
  - Vectors of binary / continuous predictions plus reference labels.

References
----------
Altman, D.G. & Bland, J.M. (1994). "Diagnostic tests 1: Sensitivity and
specificity." *BMJ*, 308(6943), 1552. [@altman1994statistics]

Cohen, J. (1960). "A coefficient of agreement for nominal scales."
*Educational and Psychological Measurement*, 20(1), 37-46. [@cohen1960coefficient]

Hanley, J.A. & McNeil, B.J. (1982). "The meaning and use of the area
under a receiver operating characteristic (ROC) curve." *Radiology*,
143(1), 29-36. [@hanley1982meaning]
"""

from __future__ import annotations

import inspect as _inspect
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

__all__ = [
    "DiagnosticTestResult",
    "ROCResult",
    "KappaResult",
    "diagnostic_test",
    "sensitivity_specificity",
    "roc_curve",
    "auc",
    "cohen_kappa",
]


# --------------------------------------------------------------------------- #
#  Sensitivity / specificity / PPV / NPV / LR
# --------------------------------------------------------------------------- #


@dataclass
class DiagnosticTestResult(ResultProtocolMixin):
    """Container for binary diagnostic-test performance metrics.

    Returned by :func:`sensitivity_specificity` / :func:`diagnostic_test`.
    Holds sensitivity and specificity (with Wilson-score or exact CIs), predictive
    values (``ppv``, ``npv``), likelihood ratios (``lr_pos``, ``lr_neg``),
    ``prevalence``, and the raw confusion cells ``tp``/``fp``/``fn``/``tn``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.sensitivity_specificity(tp=90, fn=10, fp=5, tn=95)
    >>> isinstance(res, sp.DiagnosticTestResult)
    True
    >>> res.sensitivity
    0.9
    >>> res.tp
    90
    """

    sensitivity: float
    sensitivity_ci: tuple[float, float]
    specificity: float
    specificity_ci: tuple[float, float]
    ppv: float
    npv: float
    lr_pos: float
    lr_neg: float
    prevalence: float
    tp: int
    fp: int
    fn: int
    tn: int

    def summary(self) -> str:
        sl, su = self.sensitivity_ci
        pl, pu = self.specificity_ci
        return (
            "Diagnostic Test Performance\n"
            f"  Sensitivity  = {self.sensitivity:.4f}   95% CI [{sl:.4f}, {su:.4f}]\n"
            f"  Specificity  = {self.specificity:.4f}   95% CI [{pl:.4f}, {pu:.4f}]\n"
            f"  PPV          = {self.ppv:.4f}\n"
            f"  NPV          = {self.npv:.4f}\n"
            f"  LR+          = {self.lr_pos:.3f}\n"
            f"  LR-          = {self.lr_neg:.3f}\n"
            f"  Prevalence   = {self.prevalence:.4f}\n"
            f"  Confusion: TP={self.tp}  FP={self.fp}  FN={self.fn}  TN={self.tn}"
        )


def _wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = float(stats.norm.ppf(1 - alpha / 2))
    phat = k / n
    denom = 1 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))) / denom
    return float(centre - half), float(centre + half)


def _exact_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Clopper-Pearson interval from beta quantiles."""
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.isf(alpha / 2, k + 1, n - k))
    return lo, hi


def sensitivity_specificity(
    y_true: Any = None,
    y_pred: Any = None,
    *,
    tp: Optional[int] = None,
    fn: Optional[int] = None,
    fp: Optional[int] = None,
    tn: Optional[int] = None,
    alpha: float = 0.05,
    ci_method: str = "wilson",
) -> DiagnosticTestResult:
    """Sensitivity and specificity with Wilson-score (or exact) CIs.

    Parameters
    ----------
    y_true, y_pred : array-like, optional
        Reference and predicted binary labels (0/1).
    tp, fn, fp, tn : int, optional
        Pre-computed confusion cells.  Use instead of ``y_true``/
        ``y_pred`` when you already have counts.
    alpha : float, default 0.05
    ci_method : {"wilson", "exact"}, default "wilson"
        Interval for sensitivity and specificity: Wilson score (R
        ``epiR::epi.tests(method="wilson")``) or Clopper-Pearson exact
        (``epi.tests`` default, Stata ``diagt``).

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.sensitivity_specificity(tp=90, fn=10, fp=5, tn=95)
    >>> res.sensitivity  # 90 / (90 + 10)
    0.9
    >>> res.specificity  # 95 / (95 + 5)
    0.95
    >>> res.ppv  # 90 / (90 + 5)
    0.9473684210526315

    References
    ----------
    [@altman1994statistics]
    """
    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    elif tp is None or fn is None or fp is None or tn is None:
        raise ValueError("Provide (y_true, y_pred) OR all of (tp, fn, fp, tn).")

    tp_i = int(tp)
    fn_i = int(fn)
    fp_i = int(fp)
    tn_i = int(tn)

    pos = tp_i + fn_i
    neg = tn_i + fp_i
    sens = tp_i / pos if pos > 0 else float("nan")
    spec = tn_i / neg if neg > 0 else float("nan")
    if ci_method not in ("wilson", "exact"):
        raise MethodIncompatibility("ci_method must be 'wilson' or 'exact'")
    _ci = _wilson_ci if ci_method == "wilson" else _exact_ci
    sens_ci = _ci(tp_i, pos, alpha) if pos > 0 else (0.0, 1.0)
    spec_ci = _ci(tn_i, neg, alpha) if neg > 0 else (0.0, 1.0)

    ppv = tp_i / (tp_i + fp_i) if (tp_i + fp_i) > 0 else float("nan")
    npv = tn_i / (tn_i + fn_i) if (tn_i + fn_i) > 0 else float("nan")
    lr_pos = sens / (1 - spec) if spec < 1 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")
    prevalence = pos / (pos + neg) if (pos + neg) > 0 else float("nan")

    return DiagnosticTestResult(
        sensitivity=float(sens),
        sensitivity_ci=sens_ci,
        specificity=float(spec),
        specificity_ci=spec_ci,
        ppv=float(ppv),
        npv=float(npv),
        lr_pos=float(lr_pos),
        lr_neg=float(lr_neg),
        prevalence=float(prevalence),
        tp=tp_i,
        fp=fp_i,
        fn=fn_i,
        tn=tn_i,
    )


def diagnostic_test(*args: Any, **kwargs: Any) -> DiagnosticTestResult:
    """Alias for :func:`sensitivity_specificity`.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.diagnostic_test(tp=90, fn=10, fp=5, tn=95)
    >>> res.sensitivity  # 90 / (90 + 10)
    0.9
    >>> res.specificity  # 95 / (95 + 5)
    0.95
    """
    return sensitivity_specificity(*args, **kwargs)


# Expose sensitivity_specificity's real parameter list on the thin alias so
# introspection (help(), sp.function_schema, agent tooling) is informative.
diagnostic_test.__signature__ = _inspect.signature(  # type: ignore[attr-defined]
    sensitivity_specificity
)


# --------------------------------------------------------------------------- #
#  ROC curve + AUC
# --------------------------------------------------------------------------- #


@dataclass
class ROCResult(ResultProtocolMixin):
    """Container for ROC-curve coordinates and AUC inference.

    Returned by :func:`roc_curve`.  Holds the sweep ``thresholds`` and the
    corresponding true/false positive rates (``tpr``, ``fpr``), plus the
    ``auc`` with its standard error (``auc_se``; Hanley-McNeil or DeLong,
    see ``se_method``) and CI (``auc_ci``).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # perfectly separable
    >>> roc = sp.roc_curve(y, s)
    >>> isinstance(roc, sp.ROCResult)
    True
    >>> roc.auc
    1.0
    """

    thresholds: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    auc: float
    auc_se: float
    auc_ci: tuple[float, float]
    se_method: str = "delong"

    def summary(self) -> str:
        lo, hi = self.auc_ci
        return (
            "ROC Curve\n"
            f"  AUC = {self.auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
            f"SE = {self.auc_se:.4f}"
        )


def roc_curve(
    y_true: Any,
    scores: Any,
    *,
    alpha: float = 0.05,
    se_method: str = "delong",
    weights: Any = None,
) -> ROCResult:
    """ROC curve with the AUC and its standard error.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    scores : array-like of continuous predictions (higher = more "positive")
    alpha : float, default 0.05
    weights : array-like, optional
        Non-negative observation weights. The rates become weighted
        shares and the AUC the weighted Mann-Whitney probability
        ``sum_ij w_i w_j [1(s_i > s_j) + 1(s_i = s_j)/2] / (W+ W-)``.
        The main use is as a balance check: with ``y_true`` the exposure,
        ``scores`` the propensity score and ``weights`` the propensity
        weights, an AUC near 0.5 says the score no longer separates the
        two groups in the weighted sample. No standard error is reported
        with weights (``auc_se`` and ``auc_ci`` are NaN): the weights are
        themselves estimated from the scores, and the DeLong and Hanley
        variances do not account for that.
    se_method : {"delong", "hanley", "hanley-empirical"}, default "delong"
        ``"hanley"``: the Hanley-McNeil (1982) variance with the
        exponential approximations ``Q1 = A/(2-A)``, ``Q2 = 2A^2/(1+A)``.
        ``"hanley-empirical"``: the same variance with ``Q1`` and ``Q2``
        estimated from the data (ties weighted 1/3), which is what Stata
        ``roctab, hanley`` reports. ``"delong"``: the DeLong-DeLong-
        Clarke-Pearson placement-value variance (R ``pROC::var`` /
        ``ci.auc`` default and Stata ``roctab``'s default).

        .. versionchanged:: 1.30.0
           The default moved from ``"hanley"`` (an approximation no
           reference uses by default) to ``"delong"``, the default of both
           ``pROC`` and ``roctab``. Pass ``se_method="hanley"`` for the old
           standard error.

    Notes
    -----
    The curve is traced over the distinct score values: ``thresholds`` holds
    them in decreasing order and ``tpr`` / ``fpr`` the rates of the rule
    "positive when score >= threshold". The AUC is therefore the
    Mann-Whitney probability ``P(S+ > S-) + P(S+ = S-)/2`` exactly, with
    tied scores counted as half. The confidence interval is
    ``AUC +/- z se`` truncated to [0, 1] (as ``pROC::ci.auc`` does).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # perfectly separable
    >>> roc = sp.roc_curve(y, s)
    >>> roc.auc
    1.0
    >>> bool(roc.auc_ci[0] <= roc.auc <= roc.auc_ci[1])
    True

    References
    ----------
    [@hanley1982meaning]
    """
    if se_method not in ("hanley", "hanley-empirical", "delong"):
        raise MethodIncompatibility(
            "se_method must be 'hanley', 'hanley-empirical' or 'delong'"
        )
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape.")
    if not np.all(np.isin(y, (0, 1))):
        raise MethodIncompatibility(
            "y_true must be coded 0/1.", recovery_hint="Recode the outcome to 0/1."
        )

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative labels for ROC.")
    if weights is not None:
        return _weighted_roc(y, s, np.asarray(weights, dtype=float), se_method)

    # One ROC point per distinct score (ties move TPR and FPR together).
    thr = np.unique(s)[::-1]
    idx = np.searchsorted(-thr, -s)  # position of each score in thr
    tp_at = np.bincount(idx[y == 1], minlength=len(thr))
    fp_at = np.bincount(idx[y == 0], minlength=len(thr))
    tpr = np.cumsum(tp_at) / n_pos
    fpr = np.cumsum(fp_at) / n_neg

    # Mann-Whitney AUC via placement values (ties count one half).
    s_pos = np.sort(s[y == 1])
    s_neg = np.sort(s[y == 0])
    pos = s[y == 1]
    neg = s[y == 0]
    v10 = (
        np.searchsorted(s_neg, pos, side="left")
        + 0.5
        * (
            np.searchsorted(s_neg, pos, side="right")
            - np.searchsorted(s_neg, pos, side="left")
        )
    ) / n_neg
    v01 = (
        (n_pos - np.searchsorted(s_pos, neg, side="right"))
        + 0.5
        * (
            np.searchsorted(s_pos, neg, side="right")
            - np.searchsorted(s_pos, neg, side="left")
        )
    ) / n_pos
    auc_val = float(v10.mean())

    if se_method in ("hanley", "hanley-empirical"):
        if se_method == "hanley":
            q1 = auc_val / (2 - auc_val)
            q2 = 2 * auc_val**2 / (1 + auc_val)
        else:
            # By distinct score, ascending: negatives / positives at the
            # level, negatives strictly below, positives strictly above.
            neg_at = fp_at[::-1].astype(float)
            pos_at = tp_at[::-1].astype(float)
            neg_below = np.cumsum(neg_at) - neg_at
            pos_above = n_pos - np.cumsum(pos_at)
            q2 = float(
                np.sum(pos_at * (neg_below * (neg_below + neg_at) + neg_at**2 / 3))
                / (n_pos * n_neg**2)
            )
            q1 = float(
                np.sum(neg_at * (pos_above * (pos_above + pos_at) + pos_at**2 / 3))
                / (n_neg * n_pos**2)
            )
        var = (
            auc_val * (1 - auc_val)
            + (n_pos - 1) * (q1 - auc_val**2)
            + (n_neg - 1) * (q2 - auc_val**2)
        ) / (n_pos * n_neg)
    else:
        s10 = float(np.var(v10, ddof=1)) if n_pos > 1 else float("nan")
        s01 = float(np.var(v01, ddof=1)) if n_neg > 1 else float("nan")
        var = s10 / n_pos + s01 / n_neg
    se = float(np.sqrt(max(var, 0.0)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    ci = (
        float(max(0.0, auc_val - z * se)),
        float(min(1.0, auc_val + z * se)),
    )

    return ROCResult(
        thresholds=thr,
        tpr=tpr,
        fpr=fpr,
        auc=auc_val,
        auc_se=se,
        auc_ci=ci,
        se_method=se_method,
    )


def _weighted_roc(
    y: np.ndarray, s: np.ndarray, w: np.ndarray, se_method: str
) -> ROCResult:
    """Weighted ROC curve and AUC; no standard error (see ``roc_curve``)."""
    if w.shape != s.shape:
        raise MethodIncompatibility("weights and scores must have the same shape.")
    if not np.isfinite(w).all() or (w < 0).any():
        raise MethodIncompatibility("weights must be finite and non-negative.")
    w_pos, w_neg = float(w[y == 1].sum()), float(w[y == 0].sum())
    if w_pos <= 0 or w_neg <= 0:
        raise MethodIncompatibility("Each label needs positive total weight.")
    thr = np.unique(s)[::-1]
    idx = np.searchsorted(-thr, -s)
    tp_at = np.bincount(idx[y == 1], weights=w[y == 1], minlength=len(thr))
    fp_at = np.bincount(idx[y == 0], weights=w[y == 0], minlength=len(thr))
    tpr = np.cumsum(tp_at) / w_pos
    fpr = np.cumsum(fp_at) / w_neg
    # By distinct score, descending: negatives strictly below each level
    # count in full, negatives at the level count one half.
    neg_below = w_neg - np.cumsum(fp_at)
    auc_val = float(np.sum(tp_at * (neg_below + 0.5 * fp_at)) / (w_pos * w_neg))
    return ROCResult(
        thresholds=thr,
        tpr=tpr,
        fpr=fpr,
        auc=auc_val,
        auc_se=float("nan"),
        auc_ci=(float("nan"), float("nan")),
        se_method=se_method,
    )


def auc(y_true: Any, scores: Any, weights: Any = None) -> float:
    """Shortcut: just return the AUC (weighted when ``weights`` is given).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])  # perfectly separable
    >>> sp.auc(y, s)
    1.0
    """
    return roc_curve(y_true, scores, weights=weights).auc


def ndcg(y_true: Any, scores: Any, k: float = 0.01) -> float:
    """Normalised discounted cumulative gain of the top-ranked cases.

    For a rare outcome (accounting fraud, default) the question is not how
    well a score ranks everyone, which is what the AUC answers, but how many
    true cases sit at the very top of the list an examiner would work
    through. NDCG at ``k`` looks at the ``k`` highest scores only, credits a
    true case at rank ``i`` with ``1 / log2(i + 1)``, and divides by the
    credit of a perfect ranking, so 1 means the top of the list is all true
    cases (or all of them, when there are fewer than ``k``).

    Parameters
    ----------
    y_true : array-like
        0/1 outcome.
    scores : array-like
        Predicted score; higher means more likely.
    k : float, default 0.01
        A fraction in (0, 1): the top ``round(n * k)`` cases, 0.01 being
        the top 1%. A number of 1 or more is the count itself.

    Returns
    -------
    float
        NDCG at ``k`` in [0, 1]; 0 when there is no true case at all.

    Notes
    -----
    Tied scores are ranked in the order of the data, as R's ``sort`` does.
    With many ties at the cutoff the value depends on that order; break
    them before reading much into it.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    >>> s = np.array([.9, .8, .7, .6, .5, .4, .3, .2, .1, .0])
    >>> round(sp.ndcg(y, s, k=3), 4)        # hits at ranks 1 and 3 of top 3
    0.9197
    >>> sp.ndcg(y, s, k=1)
    1.0

    References
    ----------
    jarvelin2002cumulated, bao2020detecting
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise MethodIncompatibility(
            "ndcg: y_true and scores must be one-dimensional and equally long."
        )
    if np.isnan(s).any() or pd.isna(y).any():
        raise MethodIncompatibility("ndcg: missing values in y_true or scores.")
    rel = (y.astype(float) == 1).astype(float)
    n = len(rel)
    if not k > 0:
        raise MethodIncompatibility(f"ndcg: k must be positive, got {k!r}.")
    kn = int(round(n * k)) if k < 1 else int(k)
    kn = min(kn, n)
    if kn < 1:
        raise MethodIncompatibility(
            f"ndcg: k={k!r} selects no case out of {n}.",
            recovery_hint="Raise k, or give a count of 1 or more.",
        )
    order = np.argsort(-s, kind="stable")
    discount = 1.0 / np.log2(np.arange(1, kn + 1) + 1.0)
    dcg = float(np.sum(rel[order][:kn] * discount))
    ideal = float(np.sum(discount[: int(min(kn, rel.sum()))]))
    return dcg / ideal if ideal > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Cohen's kappa (inter-rater agreement)
# --------------------------------------------------------------------------- #


@dataclass
class KappaResult(ResultProtocolMixin):
    """Container for Cohen's kappa inter-rater agreement.

    Returned by :func:`cohen_kappa`.  Holds ``kappa`` with its standard
    error and CI, the observed and expected agreement, the number of
    ``n_categories``, the ``weights`` scheme, and the z/p inference.  The
    :meth:`interpretation` method maps ``kappa`` to a Landis-Koch label.

    Examples
    --------
    >>> import statspai as sp
    >>> a = [0, 1, 2, 0, 1, 2]
    >>> b = [0, 1, 2, 0, 1, 2]  # perfect agreement
    >>> k = sp.cohen_kappa(a, b)
    >>> isinstance(k, sp.KappaResult)
    True
    >>> k.kappa
    1.0
    >>> k.interpretation()
    'almost perfect agreement'
    """

    kappa: float
    se: float
    ci: tuple[float, float]
    observed_agreement: float
    expected_agreement: float
    n_categories: int
    weights: str
    z: float
    p_value: float

    def interpretation(self) -> str:
        k = self.kappa
        if k < 0:
            return "worse than chance"
        if k < 0.20:
            return "slight agreement (Landis-Koch)"
        if k < 0.40:
            return "fair agreement"
        if k < 0.60:
            return "moderate agreement"
        if k < 0.80:
            return "substantial agreement"
        return "almost perfect agreement"

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Cohen's Kappa ({self.weights})\n"
            f"  kappa   = {self.kappa:.4f}   SE = {self.se:.4f}   "
            f"95% CI [{lo:.4f}, {hi:.4f}]\n"
            f"  P_observed = {self.observed_agreement:.4f}   "
            f"P_expected = {self.expected_agreement:.4f}\n"
            f"  Categories = {self.n_categories}   "
            f"z = {self.z:.3f}, p = {self.p_value:.4g}\n"
            f"  Interpretation: {self.interpretation()}"
        )


def cohen_kappa(
    rater_a: Any,
    rater_b: Any,
    *,
    weights: str = "unweighted",
    alpha: float = 0.05,
) -> KappaResult:
    """Cohen's (1960) kappa for two raters.

    Parameters
    ----------
    rater_a, rater_b : array-like
        Same-length sequences of category labels from two raters.
    weights : {"unweighted", "linear", "quadratic"}
        Weighting scheme for disagreements across an ordered category
        scale.  "unweighted" recovers the classic Cohen kappa.

    Examples
    --------
    >>> import statspai as sp
    >>> a = [0, 1, 2, 0, 1, 2]
    >>> b = [0, 1, 2, 0, 1, 2]  # perfect agreement
    >>> k = sp.cohen_kappa(a, b)
    >>> k.kappa
    1.0
    >>> k.n_categories
    3
    >>> c = [0, 1, 2, 0, 2, 1]  # two disagreements vs a
    >>> kc = sp.cohen_kappa(a, c)
    >>> bool(kc.kappa < 1.0)
    True

    References
    ----------
    [@cohen1960coefficient]
    """
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    if a.shape != b.shape:
        raise ValueError("Raters must have the same length.")
    if weights not in ("unweighted", "linear", "quadratic"):
        raise ValueError("weights must be 'unweighted', 'linear', or 'quadratic'.")

    cats = np.unique(np.concatenate([a, b]))
    K = len(cats)
    cat_idx = {c: i for i, c in enumerate(cats)}
    ia = np.array([cat_idx[x] for x in a])
    ib = np.array([cat_idx[x] for x in b])
    n = len(a)

    # Confusion matrix of joint ratings
    conf = np.zeros((K, K), dtype=float)
    for i, j in zip(ia, ib):
        conf[i, j] += 1
    marg_a = conf.sum(axis=1)
    marg_b = conf.sum(axis=0)

    if weights == "unweighted":
        w = 1.0 - np.eye(K)
    else:
        w = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                diff = abs(i - j) / (K - 1) if K > 1 else 0
                if weights == "linear":
                    w[i, j] = diff
                else:  # quadratic
                    w[i, j] = diff**2

    # P_observed and P_expected
    po = 1 - float(np.sum(w * conf) / n)
    expected = np.outer(marg_a, marg_b) / n
    pe = 1 - float(np.sum(w * expected) / n)

    if pe >= 1:
        kappa = float("nan")
        se = float("nan")
    else:
        kappa = (po - pe) / (1 - pe)
        # Standard error (Fleiss 1981 approximation for unweighted)
        var_num = 0.0
        for i in range(K):
            for j in range(K):
                if weights == "unweighted":
                    w_bar_i = 1 - marg_a[i] / n
                    w_bar_j = 1 - marg_b[j] / n
                    var_num += (
                        conf[i, j]
                        / n
                        * (
                            (1 - int(i == j))
                            - (w_bar_i + w_bar_j) * (1 - kappa)
                            - kappa
                            + pe
                        )
                        ** 2
                    )
                else:
                    w_ij = 1 - w[i, j]
                    w_bar_i = 1 - np.sum(w[i, :] * marg_b) / n
                    w_bar_j = 1 - np.sum(w[:, j] * marg_a) / n
                    var_num += (conf[i, j] / n) * (
                        w_ij - (w_bar_i + w_bar_j) * (1 - kappa)
                    ) ** 2
        var = (var_num - (kappa - pe * (1 - kappa)) ** 2) / (n * (1 - pe) ** 2)
        se = float(np.sqrt(max(var, 0.0)))

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (kappa - z_crit * se, kappa + z_crit * se)
    z = kappa / se if se > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))

    return KappaResult(
        kappa=float(kappa),
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        observed_agreement=po,
        expected_agreement=pe,
        n_categories=K,
        weights=weights,
        z=float(z),
        p_value=p,
    )
