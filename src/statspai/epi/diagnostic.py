"""
Clinical diagnostic-test performance metrics.

Standard clinical-epi primitives for evaluating a binary diagnostic
test against a gold standard:

- Sensitivity (true-positive rate) and specificity (true-negative rate)
- Positive / negative predictive values (with prevalence-adjusted forms)
- Likelihood ratios (LR+, LR-)
- ROC curve + AUC (trapezoidal integration) with DeLong-style SE
- Cohen's kappa (inter-rater agreement, linear or quadratic weighted)

All functions accept either:
  - A 2x2 confusion matrix (``tp, fn, fp, tn``), or
  - Vectors of binary / continuous predictions + gold labels.

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

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


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
class DiagnosticTestResult:
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
    denom = 1 + z ** 2 / n
    centre = (phat + z ** 2 / (2 * n)) / denom
    half = (z * np.sqrt(phat * (1 - phat) / n + z ** 2 / (4 * n ** 2))) / denom
    return float(centre - half), float(centre + half)


def sensitivity_specificity(
    y_true=None,
    y_pred=None,
    *,
    tp: Optional[int] = None,
    fn: Optional[int] = None,
    fp: Optional[int] = None,
    tn: Optional[int] = None,
    alpha: float = 0.05,
) -> DiagnosticTestResult:
    """Sensitivity and specificity with Wilson-score CIs.

    Parameters
    ----------
    y_true, y_pred : array-like, optional
        Gold standard and predicted binary labels (0/1).
    tp, fn, fp, tn : int, optional
        Pre-computed confusion cells.  Use instead of ``y_true``/
        ``y_pred`` when you already have counts.
    alpha : float, default 0.05
    """
    if y_true is not None and y_pred is not None:
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int)
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    elif not all(v is not None for v in (tp, fn, fp, tn)):
        raise ValueError(
            "Provide (y_true, y_pred) OR all of (tp, fn, fp, tn)."
        )

    pos = tp + fn
    neg = tn + fp
    sens = tp / pos if pos > 0 else float("nan")
    spec = tn / neg if neg > 0 else float("nan")
    sens_ci = _wilson_ci(tp, pos, alpha) if pos > 0 else (0.0, 1.0)
    spec_ci = _wilson_ci(tn, neg, alpha) if neg > 0 else (0.0, 1.0)

    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
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
        tp=tp, fp=fp, fn=fn, tn=tn,
    )


def diagnostic_test(*args, **kwargs) -> DiagnosticTestResult:
    """Alias for :func:`sensitivity_specificity`."""
    return sensitivity_specificity(*args, **kwargs)


# --------------------------------------------------------------------------- #
#  ROC curve + AUC
# --------------------------------------------------------------------------- #


@dataclass
class ROCResult:
    thresholds: np.ndarray
    tpr: np.ndarray
    fpr: np.ndarray
    auc: float
    auc_se: float
    auc_ci: tuple[float, float]

    def summary(self) -> str:
        lo, hi = self.auc_ci
        return (
            "ROC Curve\n"
            f"  AUC = {self.auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   "
            f"SE = {self.auc_se:.4f}"
        )


def roc_curve(
    y_true,
    scores,
    *,
    alpha: float = 0.05,
) -> ROCResult:
    """ROC curve with Hanley-McNeil (1982) AUC standard error.

    Parameters
    ----------
    y_true : array-like of {0, 1}
    scores : array-like of continuous predictions (higher = more "positive")
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if y.shape != s.shape:
        raise ValueError("y_true and scores must have the same shape.")

    # Sweep thresholds from high to low
    order = np.argsort(-s)
    y_sorted = y[order]
    s_sorted = s[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need both positive and negative labels for ROC.")

    cum_tp = np.cumsum(y_sorted == 1)
    cum_fp = np.cumsum(y_sorted == 0)
    tpr = cum_tp / n_pos
    fpr = cum_fp / n_neg

    # Trapezoidal AUC
    fpr_ext = np.concatenate([[0.0], fpr, [1.0]])
    tpr_ext = np.concatenate([[0.0], tpr, [1.0]])
    auc_val = float(np.trapezoid(tpr_ext, fpr_ext))

    # Hanley-McNeil SE
    q1 = auc_val / (2 - auc_val)
    q2 = 2 * auc_val ** 2 / (1 + auc_val)
    var = (
        auc_val * (1 - auc_val)
        + (n_pos - 1) * (q1 - auc_val ** 2)
        + (n_neg - 1) * (q2 - auc_val ** 2)
    ) / (n_pos * n_neg)
    se = float(np.sqrt(max(var, 0.0)))
    z = float(stats.norm.ppf(1 - alpha / 2))
    ci = (
        float(max(0.0, auc_val - z * se)),
        float(min(1.0, auc_val + z * se)),
    )

    return ROCResult(
        thresholds=s_sorted,
        tpr=tpr,
        fpr=fpr,
        auc=auc_val,
        auc_se=se,
        auc_ci=ci,
    )


def auc(y_true, scores) -> float:
    """Shortcut: just return the AUC."""
    return roc_curve(y_true, scores).auc


# --------------------------------------------------------------------------- #
#  Cohen's kappa (inter-rater agreement)
# --------------------------------------------------------------------------- #


@dataclass
class KappaResult:
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
    rater_a,
    rater_b,
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
    """
    a = np.asarray(rater_a)
    b = np.asarray(rater_b)
    if a.shape != b.shape:
        raise ValueError("Raters must have the same length.")
    if weights not in ("unweighted", "linear", "quadratic"):
        raise ValueError(
            "weights must be 'unweighted', 'linear', or 'quadratic'."
        )

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
                    w[i, j] = diff ** 2

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
                    w_bar_star = 1 - (marg_a[i] + marg_b[j]) / n
                    var_num += conf[i, j] / n * (
                        (1 - int(i == j)) - (w_bar_i + w_bar_j) * (1 - kappa)
                        - kappa + pe
                    ) ** 2
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
    p = float(2 * (1 - stats.norm.cdf(abs(z))))

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
