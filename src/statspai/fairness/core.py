"""
Core fairness metrics and counterfactual-fairness diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

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
            f"<FairnessAudit: n={self.n}, " f"DP={self.demographic_parity.value:.4f}>"
        )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _format_values(values: Sequence[Any]) -> str:
    return ", ".join(repr(v) for v in values)


def _require_dataframe(data: pd.DataFrame, *, function: str) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"`data` must be a pandas DataFrame, got {type(data).__name__}.",
            recovery_hint=(
                "Pass a pandas DataFrame containing the prediction, protected, "
                "and optional label columns."
            ),
            diagnostics={"function": function, "type": type(data).__name__},
        )
    if data.empty:
        raise DataInsufficient(
            "`data` must contain at least one row.",
            recovery_hint="Provide non-empty audit data after any missing-value filtering.",
            diagnostics={"function": function, "n_rows": 0},
        )
    return data


def _require_column_name(name: Any, *, argument: str) -> str:
    if not isinstance(name, str) or not name:
        raise MethodIncompatibility(
            f"`{argument}` must be a non-empty column name string.",
            recovery_hint=f"Pass the name of an existing DataFrame column for `{argument}`.",
            diagnostics={"argument": argument, "type": type(name).__name__},
        )
    return name


def _require_threshold(value: Any, *, argument: str = "threshold") -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise MethodIncompatibility(
            f"`{argument}` must be a finite non-negative number.",
            recovery_hint=f"Pass a numeric `{argument}` such as 0.05 or 0.1.",
            diagnostics={"argument": argument, "value": value},
        )
    out = float(value)
    if not np.isfinite(out) or out < 0:
        raise MethodIncompatibility(
            f"`{argument}` must be a finite non-negative number.",
            recovery_hint=f"Pass a numeric `{argument}` such as 0.05 or 0.1.",
            diagnostics={"argument": argument, "value": value},
        )
    return out


def _coerce_columns(columns: Sequence[str] | str, *, argument: str) -> List[str]:
    if isinstance(columns, str):
        out = [columns]
    else:
        try:
            out = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{argument}` must be a column name or a sequence of column names.",
                recovery_hint=f"Pass `{argument}` as 'x' or ['x1', 'x2'].",
                diagnostics={"argument": argument, "type": type(columns).__name__},
            ) from exc
    if not out:
        raise MethodIncompatibility(
            f"`{argument}` must contain at least one column name.",
            recovery_hint=f"Pass at least one numeric feature column in `{argument}`.",
            diagnostics={"argument": argument},
        )
    return [_require_column_name(col, argument=argument) for col in out]


def _check_binary(arr: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != 1:
        raise MethodIncompatibility(
            f"`{name}` must be a one-dimensional binary column.",
            recovery_hint=f"Pass a single 0/1 column for `{name}`.",
            diagnostics={"column": name, "shape": arr.shape},
        )
    if arr.size == 0:
        raise DataInsufficient(
            f"`{name}` has no observations.",
            recovery_hint="Provide non-empty audit data after missing-value filtering.",
            diagnostics={"column": name},
        )
    try:
        vals = set(np.unique(arr))
    except TypeError as exc:
        raise MethodIncompatibility(
            f"`{name}` must be binary 0/1.",
            recovery_hint=f"Coerce `{name}` to numeric 0/1 values before auditing.",
            diagnostics={"column": name, "dtype": str(arr.dtype)},
        ) from exc
    if not vals.issubset({0, 1, 0.0, 1.0, True, False}):
        raise MethodIncompatibility(
            f"`{name}` must be binary 0/1; got unique values {_format_values(sorted(vals, key=repr))}.",
            recovery_hint=f"Coerce `{name}` to numeric 0/1 values before auditing.",
            diagnostics={"column": name, "unique_values": [repr(v) for v in vals]},
        )
    return arr.astype(int)


def _column(df: pd.DataFrame, col: str) -> np.ndarray:
    col = _require_column_name(col, argument="column")
    if col not in df.columns:
        raise MethodIncompatibility(
            f"Column {col!r} not found in data.",
            recovery_hint="Check the column names passed to the fairness diagnostic.",
            diagnostics={"column": col, "available_columns": list(df.columns)},
        )
    arr = df[col].to_numpy()
    if pd.isna(arr).any():
        raise MethodIncompatibility(
            f"Column {col!r} contains NaN; drop or impute before fairness audit.",
            recovery_hint="Drop missing rows or impute this column before running the diagnostic.",
            diagnostics={"column": col},
        )
    return np.asarray(arr)


def _finite_numeric_vector(
    values: Any, *, name: str, n_expected: Optional[int] = None
) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be numeric.",
            recovery_hint=f"Return or pass a numeric vector for `{name}`.",
            diagnostics={"name": name, "type": type(values).__name__},
        ) from exc
    if arr.ndim == 0:
        raise MethodIncompatibility(
            f"`{name}` must be a one-dimensional numeric vector.",
            recovery_hint=f"Return one value per row for `{name}`.",
            diagnostics={"name": name, "shape": arr.shape},
        )
    if arr.ndim > 1:
        if n_expected is not None and arr.size == n_expected and 1 in arr.shape:
            arr = arr.reshape(-1)
        else:
            raise MethodIncompatibility(
                f"`{name}` must be a one-dimensional numeric vector.",
                recovery_hint=f"Return one value per row for `{name}`.",
                diagnostics={"name": name, "shape": arr.shape},
            )
    if n_expected is not None and arr.shape[0] != n_expected:
        raise MethodIncompatibility(
            f"`{name}` must return one value per row; got {arr.shape[0]} for {n_expected} rows.",
            recovery_hint="Make the predictor return an array with length equal to the input DataFrame.",
            diagnostics={
                "name": name,
                "n_expected": n_expected,
                "n_observed": arr.shape[0],
            },
        )
    if not np.all(np.isfinite(arr)):
        raise NumericalInstability(
            f"`{name}` contains non-finite values.",
            recovery_hint="Check the predictor or feature preprocessing for NaN/Inf outputs.",
            diagnostics={"name": name},
        )
    return arr


def _prediction_vector(
    predictor: Callable[[pd.DataFrame], np.ndarray],
    data: pd.DataFrame,
    *,
    name: str,
) -> np.ndarray:
    try:
        raw = predictor(data)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` failed while evaluating the predictor: {exc}",
            recovery_hint="Check that the predictor accepts the DataFrame columns supplied here.",
            diagnostics={"name": name, "error_type": type(exc).__name__},
        ) from exc
    return _finite_numeric_vector(raw, name=name, n_expected=len(data))


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

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "group": rng.integers(0, 2, 200),
    ...     "pred": rng.integers(0, 2, 200),
    ... })
    >>> res = sp.demographic_parity(df, predictions="pred", protected="group")
    >>> res.metric
    'demographic_parity'
    >>> sorted(int(g) for g in res.per_group)  # group labels (keys of per_group)
    [0, 1]
    >>> isinstance(res.passes, bool)
    True
    """
    data = _require_dataframe(data, function="demographic_parity")
    predictions = _require_column_name(predictions, argument="predictions")
    protected = _require_column_name(protected, argument="protected")
    threshold = _require_threshold(threshold)
    yhat = _check_binary(_column(data, predictions), predictions)
    a = _column(data, protected)
    if len(np.unique(a)) < 2:
        raise DataInsufficient(
            f"`protected` column has only one level ({np.unique(a)}). "
            "Need at least 2 groups for demographic parity.",
            recovery_hint="Provide audit data with at least two protected groups.",
            diagnostics={
                "function": "demographic_parity",
                "protected": protected,
                "n_groups": int(len(np.unique(a))),
            },
        )
    per_group: Dict[Any, float] = {}
    for g in np.unique(a):
        mask = a == g
        per_group[g] = float(yhat[mask].mean()) if mask.any() else float("nan")
    rates = np.array(list(per_group.values()))
    gap = float(rates.max() - rates.min())
    _result = FairnessResult(
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
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.fairness.demographic_parity",
            params={
                "predictions": predictions,
                "protected": protected,
                "threshold": threshold,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


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

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "group": rng.integers(0, 2, 200),
    ...     "label": rng.integers(0, 2, 200),
    ...     "pred": rng.integers(0, 2, 200),
    ... })
    >>> res = sp.equalized_odds(df, predictions="pred", labels="label",
    ...                         protected="group")
    >>> res.metric
    'equalized_odds'
    >>> isinstance(res.passes, bool)
    True

    References
    ----------
    hardt2016equality
    """
    data = _require_dataframe(data, function="equalized_odds")
    predictions = _require_column_name(predictions, argument="predictions")
    labels = _require_column_name(labels, argument="labels")
    protected = _require_column_name(protected, argument="protected")
    threshold = _require_threshold(threshold)
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
        raise DataInsufficient(
            "Equalized odds requires at least one positive and one negative "
            "label in each group.",
            recovery_hint="Use data with both outcome classes represented in at least two protected groups.",
            diagnostics={
                "function": "equalized_odds",
                "protected": protected,
                "n_tpr_groups": len(tprs),
                "n_fpr_groups": len(fprs),
            },
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

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "group": rng.integers(0, 2, 200),
    ...     "x1": rng.normal(size=200),
    ... })
    >>> def predictor(d):
    ...     return (0.5 * d["x1"] + 0.3 * d["group"]).to_numpy()
    >>> def scm(d, a):  # intervene: set the protected attribute to `a`
    ...     d2 = d.copy()
    ...     d2["group"] = a
    ...     return d2
    >>> res = sp.counterfactual_fairness(df, predictor, protected="group",
    ...                                  scm_intervention=scm)
    >>> res.metric
    'counterfactual_fairness'
    >>> res.value >= 0.0
    True

    References
    ----------
    kusner2017counterfactual
    """
    data = _require_dataframe(data, function="counterfactual_fairness")
    protected = _require_column_name(protected, argument="protected")
    threshold = _require_threshold(threshold)
    if not callable(predictor):
        raise MethodIncompatibility(
            "`predictor` must be callable.",
            recovery_hint="Pass a function that accepts a DataFrame and returns one numeric prediction per row.",
            diagnostics={"argument": "predictor", "type": type(predictor).__name__},
        )
    if not callable(scm_intervention):
        raise MethodIncompatibility(
            "`scm_intervention` must be callable.",
            recovery_hint=(
                "Pass a function of (data, protected_value) returning a counterfactual DataFrame."
            ),
            diagnostics={
                "argument": "scm_intervention",
                "type": type(scm_intervention).__name__,
            },
        )
    if protected not in data.columns:
        raise MethodIncompatibility(
            f"`protected` column {protected!r} not in data.",
            recovery_hint="Check the `protected` column name passed to counterfactual_fairness.",
            diagnostics={
                "protected": protected,
                "available_columns": list(data.columns),
            },
        )
    y_obs = _prediction_vector(predictor, data, name="predictor(data)")
    observed_a = data[protected].to_numpy()
    if alternative_values is None:
        alternative_values = [v for v in np.unique(observed_a)]
        if len(alternative_values) < 2:
            raise DataInsufficient(
                f"Protected attribute {protected!r} has only one level; "
                "counterfactual fairness is undefined.",
                recovery_hint="Provide at least two protected-attribute values or explicit alternatives.",
                diagnostics={
                    "function": "counterfactual_fairness",
                    "protected": protected,
                    "n_levels": int(len(alternative_values)),
                },
            )
    else:
        try:
            alternative_values = list(alternative_values)
        except TypeError as exc:
            raise MethodIncompatibility(
                "`alternative_values` must be a non-empty sequence.",
                recovery_hint="Pass explicit protected-attribute alternatives such as [0, 1].",
                diagnostics={
                    "argument": "alternative_values",
                    "type": type(alternative_values).__name__,
                },
            ) from exc
        if not alternative_values:
            raise DataInsufficient(
                "`alternative_values` must contain at least one value.",
                recovery_hint="Pass explicit protected-attribute alternatives such as [0, 1].",
                diagnostics={"function": "counterfactual_fairness"},
            )

    # For each alternative value, intervene and predict.
    max_abs_diff: np.ndarray = np.zeros(len(data), dtype=float)
    per_alt: Dict[Any, float] = {}
    for a_alt in alternative_values:
        df_cf = scm_intervention(data, a_alt)
        if not isinstance(df_cf, pd.DataFrame):
            raise MethodIncompatibility(
                "`scm_intervention` must return a pandas DataFrame; got "
                f"{type(df_cf).__name__}.",
                recovery_hint="Return a DataFrame with the same row count as the factual data.",
                diagnostics={"returned_type": type(df_cf).__name__},
            )
        if len(df_cf) != len(data):
            raise MethodIncompatibility(
                "`scm_intervention` returned a DataFrame with different "
                f"length ({len(df_cf)} vs {len(data)}).",
                recovery_hint="Return one counterfactual row for every factual row.",
                diagnostics={"n_expected": len(data), "n_observed": len(df_cf)},
            )
        y_cf = _prediction_vector(predictor, df_cf, name="predictor(counterfactual)")
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

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "group": rng.integers(0, 2, 200),
    ...     "x1": rng.normal(size=200),
    ...     "x2": rng.normal(size=200),
    ... })
    >>> ob = sp.orthogonal_to_bias(df, features=["x1", "x2"], protected="group")
    >>> # residualized features are uncorrelated with the protected attribute
    >>> bool(abs(np.corrcoef(ob["x1"], df["group"])[0, 1]) < 1e-8)
    True
    >>> ob.shape == df.shape
    True

    References
    ----------
    chen2024counterfactual
    """
    data = _require_dataframe(data, function="orthogonal_to_bias")
    features = _coerce_columns(features, argument="features")
    protected = _require_column_name(protected, argument="protected")
    if method != "residualize":
        raise MethodIncompatibility(
            f"Unknown method {method!r}; only 'residualize' supported.",
            recovery_hint="Use method='residualize'.",
            diagnostics={"method": method, "valid_methods": ["residualize"]},
        )
    if protected not in data.columns:
        raise MethodIncompatibility(
            f"Protected column {protected!r} not in data.",
            recovery_hint="Check the `protected` column name.",
            diagnostics={
                "protected": protected,
                "available_columns": list(data.columns),
            },
        )
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"Feature columns not in data: {missing}",
            recovery_hint="Pass feature names that exist in the DataFrame.",
            diagnostics={"missing": missing, "available_columns": list(data.columns)},
        )
    out = data.copy()
    _column(data, protected)
    A_raw = data[protected]
    # One-hot encode categorical/object protected attribute.
    if A_raw.dtype.kind in "OUSb" or isinstance(A_raw.dtype, pd.CategoricalDtype):
        A_oh = pd.get_dummies(A_raw, drop_first=True).to_numpy(dtype=float)
    else:
        a_arr = _finite_numeric_vector(A_raw.to_numpy(), name=protected)
        if len(np.unique(a_arr)) > 2:
            # Treat multi-level numeric as categorical to avoid linearity assumption.
            A_oh = pd.get_dummies(A_raw.astype("category"), drop_first=True).to_numpy(
                dtype=float
            )
        else:
            A_oh = a_arr.reshape(-1, 1)
    n = A_oh.shape[0]
    X_design = np.column_stack([np.ones(n), A_oh])
    for f in features:
        col = _finite_numeric_vector(data[f].to_numpy(), name=f, n_expected=len(data))
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

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({
    ...     "group": rng.integers(0, 2, 200),
    ...     "label": rng.integers(0, 2, 200),
    ...     "pred": rng.integers(0, 2, 200),
    ... })
    >>> audit = sp.fairness_audit(df, predictions="pred", protected="group",
    ...                           labels="label")
    >>> audit.n
    200
    >>> audit.demographic_parity.metric
    'demographic_parity'
    >>> audit.equalized_odds.metric
    'equalized_odds'
    """
    data = _require_dataframe(data, function="fairness_audit")
    predictions = _require_column_name(predictions, argument="predictions")
    protected = _require_column_name(protected, argument="protected")
    if labels is not None:
        labels = _require_column_name(labels, argument="labels")
    threshold = _require_threshold(threshold)
    dp = demographic_parity(
        data,
        predictions=predictions,
        protected=protected,
        threshold=threshold,
    )
    eo = None
    if labels is not None:
        eo = equalized_odds(
            data,
            predictions=predictions,
            labels=labels,
            protected=protected,
            threshold=threshold,
        )
    cf = None
    if predictor is not None and scm_intervention is not None:
        cf = counterfactual_fairness(
            data,
            predictor=predictor,
            protected=protected,
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
