"""ANCOVA and pre/post non-equivalent group designs.

Both estimators reduce to a covariate-adjusted OLS regression of the outcome on
a treatment indicator. They reuse :func:`statspai.regression.regress` (so robust
/ clustered SEs come from one place) and return the unified
:class:`~statspai.core.results.CausalResult`. The value they add over a raw
``regress`` call is the *design framing*: a binary treatment is encoded for you,
the treatment coefficient is reported as the average treatment effect, and the
identifying assumptions (and the regression-to-the-mean caveat for change
scores) are surfaced in ``model_info``.
"""

from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from ..core.results import CausalResult

__all__ = ["ancova", "negd"]


def _unique_col(data: pd.DataFrame, base: str) -> str:
    """A column name not already present in ``data``."""
    name = base
    i = 0
    while name in data.columns:
        i += 1
        name = f"{base}_{i}"
    return name


def _encode_treatment(
    data: pd.DataFrame, group: str, group_value: Any
) -> "tuple[pd.Series, Any, Any]":
    """Return a 0/1 treated indicator plus the (control, treated) levels."""
    col = data[group]
    uniques = pd.unique(col.dropna())
    if len(uniques) < 2:
        raise ValueError(
            f"group column {group!r} must have two levels (treated vs control);"
            f" found {len(uniques)}."
        )
    if len(uniques) > 2 and group_value is None:
        raise ValueError(
            f"group column {group!r} has {len(uniques)} levels. Pass "
            "group_value=<treated level> to define the treated group, or "
            "collapse it to a binary indicator first."
        )
    if group_value is not None:
        if group_value not in set(uniques):
            raise ValueError(
                f"group_value={group_value!r} is not a level of {group!r} "
                f"(levels: {sorted(map(str, uniques))})."
            )
        treated_level = group_value
        others = [u for u in uniques if u != group_value]
        control_level = others[0] if len(others) == 1 else "rest"
    else:
        numeric = pd.api.types.is_numeric_dtype(col)
        as_set = set(np.asarray(uniques).tolist())
        if numeric and as_set <= {0, 1}:
            treated_level, control_level = 1, 0
        else:
            ordered = sorted(uniques, key=lambda v: str(v))
            control_level, treated_level = ordered[0], ordered[1]
    treated = (col == treated_level).astype(float)
    return treated, control_level, treated_level


def _term(name: str, is_categorical: bool) -> str:
    return f"C({name})" if is_categorical else name


def _fit_adjusted(
    data: pd.DataFrame,
    *,
    outcome: str,
    group: str,
    covariates: Optional[Sequence[str]],
    robust: str,
    cluster: Optional[str],
    alpha: float,
    group_value: Any,
    method_label: str,
    extra_info: dict,
) -> CausalResult:
    """Shared OLS fit: outcome ~ treated + covariates, treated coef = ATE."""
    from ..regression import regress

    covariates = list(covariates or [])
    needed = [outcome, group] + covariates + ([cluster] if cluster else [])
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"columns not found in data: {missing}.")

    treated, control_level, treated_level = _encode_treatment(data, group, group_value)
    work = data.copy()
    tcol = _unique_col(work, "_sp_treated")
    work[tcol] = treated.to_numpy()

    rhs = [tcol] + [
        _term(c, not pd.api.types.is_numeric_dtype(work[c])) for c in covariates
    ]
    formula = f"{outcome} ~ " + " + ".join(rhs)

    fit = regress(formula, data=work, robust=robust, cluster=cluster)
    tidy = fit.tidy(conf_level=1.0 - alpha)
    row = tidy.loc[tidy["term"] == tcol]
    if row.empty:
        raise RuntimeError(
            "ANCOVA could not locate the treatment coefficient after fitting; "
            "this usually means the treatment indicator was collinear with a "
            "covariate."
        )
    row = row.iloc[0]

    model_info = {
        "design": method_label,
        "group_column": group,
        "control_level": control_level,
        "treated_level": treated_level,
        "covariates": covariates,
        "robust": robust,
        "formula": formula,
        "assumptions": [
            "Selection on observables: treatment is independent of potential "
            "outcomes given the included covariates.",
            "Correct functional form (linear, additive covariate adjustment).",
            "No unmeasured confounding of treatment and outcome.",
        ],
    }
    model_info.update(extra_info)

    return CausalResult(
        method=method_label,
        estimand="ATE",
        estimate=float(row["estimate"]),
        se=float(row["std_error"]),
        pvalue=float(row["p_value"]),
        ci=(float(row["conf_low"]), float(row["conf_high"])),
        alpha=alpha,
        n_obs=int(fit.data_info.get("nobs", len(work))),
        model_info=model_info,
    )


def ancova(
    data: pd.DataFrame,
    outcome: str,
    group: str,
    covariates: Optional[Sequence[str]] = None,
    *,
    robust: str = "hc1",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    group_value: Any = None,
) -> CausalResult:
    """Covariate-adjusted comparison of group means (ANCOVA).

    Fits ``outcome ~ treated + covariates`` by OLS and reports the treatment
    coefficient as the (covariate-adjusted) average treatment effect. This is
    the recommended adjusted comparison when groups are not randomised but
    treatment is plausibly ignorable given the covariates.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per unit.
    outcome : str
        Outcome column.
    group : str
        Treatment-group column. A 0/1 numeric column is used directly; any
        other two-level column is encoded (pass ``group_value`` to name the
        treated level explicitly).
    covariates : sequence of str, optional
        Adjustment covariates. Non-numeric covariates enter as categorical
        (``C(col)``).
    robust : str, default 'hc1'
        Heteroskedasticity-robust SE type forwarded to :func:`sp.regress`.
    cluster : str, optional
        Cluster-robust SE column.
    alpha : float, default 0.05
        Significance level for the reported confidence interval.
    group_value : any, optional
        The level of ``group`` that denotes treatment.

    Returns
    -------
    CausalResult
        ``estimand='ATE'``; ``model_info`` carries the design, group levels,
        covariates and identifying assumptions.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> res = sp.ancova(df, outcome='post', group='treated',
    ...                 covariates=['baseline', 'age'])  # doctest: +SKIP
    >>> res.estimate  # doctest: +SKIP
    """
    return _fit_adjusted(
        data,
        outcome=outcome,
        group=group,
        covariates=covariates,
        robust=robust,
        cluster=cluster,
        alpha=alpha,
        group_value=group_value,
        method_label="ANCOVA (covariate-adjusted)",
        extra_info={},
    )


def negd(
    data: pd.DataFrame,
    group: str,
    *,
    pre: str,
    post: str,
    covariates: Optional[Sequence[str]] = None,
    method: str = "ancova",
    robust: str = "hc1",
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    group_value: Any = None,
) -> CausalResult:
    """Pre/post non-equivalent group design (NEGD).

    A treated and a non-randomised comparison group are each measured before and
    after the intervention. Two estimators are offered:

    - ``method='ancova'`` (default): regress ``post`` on treatment and ``pre``
      (plus any covariates). Conditions on baseline; robust to baseline
      imbalance and generally preferred (Lord's paradox).
    - ``method='change_score'``: regress the change ``post - pre`` on treatment
      (plus covariates). Identifies under parallel pre/post trends and is more
      vulnerable to regression to the mean when baseline differs across groups.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per unit (wide format: a ``pre`` and a ``post`` column).
    group : str
        Treatment-group column (see :func:`ancova` for encoding).
    pre, post : str
        Baseline and follow-up outcome columns.
    covariates : sequence of str, optional
        Additional adjustment covariates.
    method : {'ancova', 'change_score'}, default 'ancova'
        Estimator (see above).
    robust, cluster, alpha, group_value
        As in :func:`ancova`.

    Returns
    -------
    CausalResult
        ``estimand='ATE'``. ``model_info`` records the method and, for
        change-score, a regression-to-the-mean caveat.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> res = sp.negd(  # doctest: +SKIP
    ...     df, group='treated', pre='score0', post='score1',
    ... )
    """
    if method not in ("ancova", "change_score"):
        raise ValueError(f"method must be 'ancova' or 'change_score'; got {method!r}.")
    covariates = list(covariates or [])

    if method == "ancova":
        return _fit_adjusted(
            data,
            outcome=post,
            group=group,
            covariates=[pre] + covariates,
            robust=robust,
            cluster=cluster,
            alpha=alpha,
            group_value=group_value,
            method_label="NEGD (pre/post ANCOVA)",
            extra_info={"pre": pre, "post": post, "negd_method": "ancova"},
        )

    for col in (pre, post):
        if col not in data.columns:
            raise ValueError(f"column {col!r} not found in data.")
    work = data.copy()
    delta = _unique_col(work, "_sp_change")
    work[delta] = work[post].to_numpy(dtype=float) - work[pre].to_numpy(dtype=float)
    warnings.warn(
        "negd(method='change_score'): change-score (gain) analysis identifies "
        "the effect only under equal pre/post trends across groups and is "
        "sensitive to regression to the mean when baseline differs between "
        "groups. Prefer method='ancova' unless trends are known parallel.",
        UserWarning,
        stacklevel=2,
    )
    res = _fit_adjusted(
        work,
        outcome=delta,
        group=group,
        covariates=covariates,
        robust=robust,
        cluster=cluster,
        alpha=alpha,
        group_value=group_value,
        method_label="NEGD (change-score)",
        extra_info={
            "pre": pre,
            "post": post,
            "negd_method": "change_score",
            "regression_to_mean_warning": True,
        },
    )
    return res
