"""
Core fairness metrics and counterfactual-fairness diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


__all__ = [
    "counterfactual_fairness",
    "orthogonal_to_bias",
    "demographic_parity",
    "equalized_odds",
    "fairness_audit",
    "FairnessResult",
    "FairnessAudit",
]


# -------------------------------------------------------------------------
# Result objects
# -------------------------------------------------------------------------


@dataclass
class FairnessResult:
    """Single fairness diagnostic."""
    metric: str
    value: float
    per_group: Dict[Any, float] = field(default_factory=dict)
    threshold: Optional[float] = None
    passes: Optional[bool] = None
    notes: str = ""

    def summary(self) -> str:
        lines = [
            f"Fairness metric: {self.metric}",
            f"  gap value : {self.value:.6f}",
        ]
        if self.threshold is not None:
            lines.append(f"  threshold : {self.threshold:.6f}")
            lines.append(f"  verdict   : {'PASS' if self.passes else 'FAIL'}")
        if self.per_group:
            lines.append("  per-group :")
            for g, v in self.per_group.items():
                lines.append(f"    {g!r:>12s}: {v:.6f}")
        if self.notes:
            lines.append(f"  notes     : {self.notes}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<FairnessResult: {self.metric} = {self.value:.4f}>"


@dataclass
class FairnessAudit:
    """One-shot dashboard of fairness diagnostics."""
    demographic_parity: FairnessResult
    equalized_odds: Optional[FairnessResult]
    counterfactual_fairness: Optional[FairnessResult]
    n: int
    protected_attribute: str

    def summary(self) -> str:
        bar = "=" * 64
        parts = [
            bar,
            f"Fairness Audit — protected = {self.protected_attribute!r}, n = {self.n}",
            bar,
            self.demographic_parity.summary(),
        ]
        if self.equalized_odds is not None:
            parts += ["", self.equalized_odds.summary()]
        if self.counterfactual_fairness is not None:
            parts += ["", self.counterfactual_fairness.summary()]
        parts.append(bar)
        return "\n".join(parts)

    def __repr__(self) -> str:
        return (
            f"<FairnessAudit: n={self.n}, "
            f"DP={self.demographic_parity.value:.4f}>"
        )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _check_binary(arr: np.ndarray, name: str) -> np.ndarray:
    vals = set(np.unique(arr))
    if not vals.issubset({0, 1, 0.0, 1.0, True, False}):
        raise ValueError(
            f"`{name}` must be binary 0/1; got unique values {sorted(vals)}."
        )
    return arr.astype(int)


def _column(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise ValueError(f"Column {col!r} not found in data.")
    arr = df[col].to_numpy()
    if pd.isna(arr).any():
        raise ValueError(
            f"Column {col!r} contains NaN; drop or impute before fairness audit."
        )
    return arr


# -------------------------------------------------------------------------
# Group-level fairness metrics
# -------------------------------------------------------------------------


def demographic_parity(
    data: pd.DataFrame,
    *,
    predictions: str,
    protected: str,
    threshold: float = 0.1,
) -> FairnessResult:
    """Demographic-parity gap: ``max_{a,b} | P(Y_hat=1 | A=a) − P(Y_hat=1 | A=b) |``.

    Parameters
    ----------
    data : DataFrame
    predictions : str
        Column of binary classifier outputs (0/1).
    protected : str
        Column containing the protected attribute ``A``. May be multi-valued.
    threshold : float, default 0.1
        A gap below this value counts as a "pass". Follows the 80%-rule of
        thumb (EEOC disparate-impact guideline).

    Notes
    -----
    Demographic parity is the weakest fairness criterion — it ignores
    ground-truth labels and can be trivially satisfied by a random
    classifier. Use together with :func:`equalized_odds`.
    """
    yhat = _check_binary(_column(data, predictions), predictions)
    a = _column(data, protected)
    if len(np.unique(a)) < 2:
        raise ValueError(
            f"`protected` column has only one level ({np.unique(a)}). "
            "Need at least 2 groups for demographic parity."
        )
    per_group: Dict[Any, float] = {}
    for g in np.unique(a):
        mask = a == g
        per_group[g] = float(yhat[mask].mean()) if mask.any() else float("nan")
    rates = np.array(list(per_group.values()))
    gap = float(rates.max() - rates.min())
    return FairnessResult(
        metric="demographic_parity",
        value=gap,
        per_group=per_group,
        threshold=threshold,
        passes=bool(gap <= threshold),
        notes=(
            "Group-specific positive-prediction rates. "
            "Gap is max - min across groups."
        ),
    )


def equalized_odds(
    data: pd.DataFrame,
    *,
    predictions: str,
    labels: str,
    protected: str,
    threshold: float = 0.1,
) -> FairnessResult:
    """Hardt-Price-Srebro equalized-odds gap.

    Returns the larger of the TPR gap and FPR gap across groups:

        gap = max( max |TPR_a − TPR_b|, max |FPR_a − FPR_b| )

    Parameters
    ----------
    data : DataFrame
    predictions : str
        Binary classifier output.
    labels : str
        Binary ground-truth outcome.
    protected : str
        Protected attribute column.
    threshold : float, default 0.1
    """
    yhat = _check_binary(_column(data, predictions), predictions)
    y = _check_binary(_column(data, labels), labels)
    a = _column(data, protected)
    tprs: Dict[Any, float] = {}
    fprs: Dict[Any, float] = {}
    for g in np.unique(a):
        mask = a == g
        y_g = y[mask]
        yhat_g = yhat[mask]
        if (y_g == 1).sum() > 0:
            tprs[g] = float((yhat_g[y_g == 1] == 1).mean())
        if (y_g == 0).sum() > 0:
            fprs[g] = float((yhat_g[y_g == 0] == 1).mean())
    if len(tprs) < 2 or len(fprs) < 2:
        raise ValueError(
            "Equalized odds requires at least one positive and one negative "
            "label in each group."
        )
    tpr_vals = np.array(list(tprs.values()))
    fpr_vals = np.array(list(fprs.values()))
    tpr_gap = float(tpr_vals.max() - tpr_vals.min())
    fpr_gap = float(fpr_vals.max() - fpr_vals.min())
    gap = max(tpr_gap, fpr_gap)
    per_group = {f"TPR[{g}]": v for g, v in tprs.items()}
    per_group.update({f"FPR[{g}]": v for g, v in fprs.items()})
    return FairnessResult(
        metric="equalized_odds",
        value=gap,
        per_group=per_group,
        threshold=threshold,
        passes=bool(gap <= threshold),
        notes=f"TPR_gap = {tpr_gap:.4f}, FPR_gap = {fpr_gap:.4f}.",
    )


# -------------------------------------------------------------------------
# Counterfactual fairness (Kusner et al. 2018)
# -------------------------------------------------------------------------


def counterfactual_fairness(
    data: pd.DataFrame,
    predictor: Callable[[pd.DataFrame], np.ndarray],
    *,
    protected: str,
    scm_intervention: Callable[[pd.DataFrame, Any], pd.DataFrame],
    alternative_values: Optional[Sequence[Any]] = None,
    threshold: float = 0.05,
) -> FairnessResult:
    """Kusner-Loftus-Russell-Silva counterfactual-fairness test.

    For each unit, compare the factual prediction with the counterfactual
    prediction obtained under a user-supplied SCM intervention that sets
    the protected attribute ``A`` to alternative values.

    Parameters
    ----------
    data : DataFrame
        Observed covariates (including the protected attribute).
    predictor : Callable(DataFrame) -> ndarray
        Deployed predictor. Must accept a DataFrame and return an array of
        the same length.
    protected : str
        Protected attribute column — its counterfactual is constructed by
        ``scm_intervention``.
    scm_intervention : Callable(DataFrame, value) -> DataFrame
        User-supplied causal model. Given the original data and an
        alternative value of ``A``, returns a modified DataFrame
        representing the counterfactual world (with downstream descendants
        updated under the intervention). The caller owns the SCM.
    alternative_values : sequence, optional
        Values of ``A`` to intervene on. Defaults to all unique values of
        ``data[protected]`` other than the observed one.
    threshold : float, default 0.05

    Returns
    -------
    FairnessResult
        ``value`` is the average absolute counterfactual change,
        ``mean_i max_{a'} | f(X_i^{a'}) - f(X_i^{a_i}) |``.

    Notes
    -----
    This is a *Level-3* (counterfactual) fairness test and is only as
    credible as the SCM the user supplies. A DAG + structural equations
    must be specified outside this function — we just wrap the mechanics.

    References
    ----------
    Kusner, Loftus, Russell, Silva (2018).
    """
    if protected not in data.columns:
        raise ValueError(f"`protected` column {protected!r} not in data.")
    y_obs = np.asarray(predictor(data), dtype=float)
    if y_obs.shape[0] != len(data):
        raise ValueError(
            "`predictor(data)` must return one value per row; got "
            f"{y_obs.shape[0]} for {len(data)} rows."
        )
    observed_a = data[protected].to_numpy()
    if alternative_values is None:
        alternative_values = [v for v in np.unique(observed_a)]
        if len(alternative_values) < 2:
            raise ValueError(
                f"Protected attribute {protected!r} has only one level; "
                "counterfactual fairness is undefined."
            )

    # For each alternative value, intervene and predict.
    max_abs_diff = np.zeros(len(data), dtype=float)
    per_alt: Dict[Any, float] = {}
    for a_alt in alternative_values:
        df_cf = scm_intervention(data, a_alt)
        if not isinstance(df_cf, pd.DataFrame):
            raise TypeError(
                "`scm_intervention` must return a pandas DataFrame; got "
                f"{type(df_cf).__name__}."
            )
        if len(df_cf) != len(data):
            raise ValueError(
                "`scm_intervention` returned a DataFrame with different "
                f"length ({len(df_cf)} vs {len(data)})."
            )
        y_cf = np.asarray(predictor(df_cf), dtype=float)
        # Only count units whose observed A differs from a_alt.
        differs = observed_a != a_alt
        diff = np.abs(y_cf - y_obs)
        max_abs_diff = np.where(differs, np.maximum(max_abs_diff, diff), max_abs_diff)
        per_alt[a_alt] = float(diff[differs].mean()) if differs.any() else 0.0

    value = float(max_abs_diff.mean())
    return FairnessResult(
        metric="counterfactual_fairness",
        value=value,
        per_group=per_alt,
        threshold=threshold,
        passes=bool(value <= threshold),
        notes=(
            f"Mean over units of max_{{a'}} |f(X^{{a'}}) - f(X^{{obs}})|. "
            f"Interventions on {len(alternative_values)} alternative values."
        ),
    )


# -------------------------------------------------------------------------
# Orthogonal-to-Bias preprocessing (Chen & Zhu 2024, arXiv:2403.17852)
# [@chen2024counterfactual]
# -------------------------------------------------------------------------


def orthogonal_to_bias(
    data: pd.DataFrame,
    *,
    features: Sequence[str],
    protected: str,
    method: str = "residualize",
) -> pd.DataFrame:
    """OB preprocessing — remove the part of ``features`` correlated with ``A``.

    For each feature ``X_j``, regress ``X_j ~ A`` (one-hot A if multi-valued)
    and replace the feature by its residual. The residualized features are
    by construction uncorrelated with ``A`` in-sample. Training a predictor
    on the residualized features is a simple relaxation of counterfactual
    fairness that requires no explicit SCM.

    Parameters
    ----------
    data : DataFrame
    features : sequence of str
        Numeric columns to residualize.
    protected : str
        Protected attribute column (numeric or categorical).
    method : {'residualize'}, default 'residualize'

    Returns
    -------
    DataFrame
        Copy of ``data`` with the ``features`` columns replaced by
        residualized versions. Other columns (including ``protected``)
        unchanged.

    References
    ----------
    Chen & Zhu (arXiv:2403.17852v3, 2024).
    """
    if method != "residualize":
        raise ValueError(f"Unknown method {method!r}; only 'residualize' supported.")
    if protected not in data.columns:
        raise ValueError(f"Protected column {protected!r} not in data.")
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise ValueError(f"Feature columns not in data: {missing}")
    out = data.copy()
    A_raw = data[protected]
    # One-hot encode categorical/object protected attribute.
    if A_raw.dtype.kind in "OUSb" or isinstance(A_raw.dtype, pd.CategoricalDtype):
        A_oh = pd.get_dummies(A_raw, drop_first=True).to_numpy(dtype=float)
    else:
        a_arr = A_raw.to_numpy(dtype=float)
        if len(np.unique(a_arr)) > 2:
            # Treat multi-level numeric as categorical to avoid linearity assumption.
            A_oh = pd.get_dummies(A_raw.astype("category"), drop_first=True).to_numpy(dtype=float)
        else:
            A_oh = a_arr.reshape(-1, 1)
    n = A_oh.shape[0]
    X_design = np.column_stack([np.ones(n), A_oh])
    for f in features:
        col = data[f].to_numpy(dtype=float)
        if np.isnan(col).any():
            raise ValueError(f"Feature {f!r} contains NaN; drop or impute first.")
        beta, *_ = np.linalg.lstsq(X_design, col, rcond=None)
        resid = col - X_design @ beta
        out[f] = resid
    return out


# -------------------------------------------------------------------------
# Dashboard
# -------------------------------------------------------------------------


def fairness_audit(
    data: pd.DataFrame,
    *,
    predictions: str,
    protected: str,
    labels: Optional[str] = None,
    predictor: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
    scm_intervention: Optional[Callable[[pd.DataFrame, Any], pd.DataFrame]] = None,
    alternative_values: Optional[Sequence[Any]] = None,
    threshold: float = 0.1,
) -> FairnessAudit:
    """Run all applicable fairness diagnostics on a classifier's output.

    Parameters
    ----------
    data : DataFrame
    predictions : str
        Binary classifier output column.
    protected : str
    labels : str, optional
        If present, adds equalized-odds diagnostic.
    predictor, scm_intervention, alternative_values
        If ``predictor`` and ``scm_intervention`` are both supplied, also
        runs counterfactual-fairness.
    """
    dp = demographic_parity(
        data, predictions=predictions, protected=protected, threshold=threshold,
    )
    eo = None
    if labels is not None:
        eo = equalized_odds(
            data, predictions=predictions, labels=labels,
            protected=protected, threshold=threshold,
        )
    cf = None
    if predictor is not None and scm_intervention is not None:
        cf = counterfactual_fairness(
            data, predictor=predictor, protected=protected,
            scm_intervention=scm_intervention,
            alternative_values=alternative_values,
            threshold=threshold / 2.0,  # tighter threshold for CF
        )
    return FairnessAudit(
        demographic_parity=dp,
        equalized_odds=eo,
        counterfactual_fairness=cf,
        n=len(data),
        protected_attribute=protected,
    )
