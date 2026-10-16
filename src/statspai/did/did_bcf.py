"""DiD with a BCF-style forest on long-differenced outcomes.

``sp.did_bcf`` is StatsPAI's own estimator.  For each treatment cohort
``g`` it forms, for cohort ``g`` and never-treated units alike, the long
difference between the unit's average outcome over the cohort's post
periods (``t >= g``) and over its pre periods (``t < g``), and fits the
BCF-style ensemble of :func:`statspai.bcf` (a prognostic forest on controls
plus a treatment-effect booster on treated residuals, with a bootstrap for
uncertainty -- not an MCMC posterior) to that long difference.  Conditional
parallel trends make the conditional mean contrast of the long difference
the cohort's conditional ATT averaged over its post periods.

It is related to, but is not, the DiD-BCF model of [@souto2025forests],
which specifies a Bayesian causal forest for outcome *levels*.  For
heterogeneous effects with validated group-time inference use
:func:`statspai.did_forest`.

Standard errors
---------------
* No covariates: exact influence-function standard errors of the
  difference in mean long differences, combined across cohorts unit by
  unit (never-treated units enter several cohorts' comparisons).
* With covariates: each cohort's standard error is the bootstrap standard
  deviation of the treated-unit average of the BCF CATE draws; cohorts
  share control units, so the overall standard error uses the conservative
  bound ``sum_g w_g se_g`` (perfect positive correlation).
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility


def _unit_cohorts(df: pd.DataFrame, treat: str, time: str, id: str) -> pd.Series:
    """First treatment period per unit from a cohort column or a 0/1 indicator."""
    by_unit = df.groupby(id)[treat]
    if (by_unit.nunique(dropna=False) <= 1).all():
        cohort = pd.to_numeric(by_unit.first(), errors="coerce")
        return cohort.where(cohort.notna() & (cohort != 0), np.inf)
    values = pd.to_numeric(df[treat], errors="coerce")
    if not values.isin([0, 1]).all():
        raise MethodIncompatibility(
            "did_bcf(): treat varies within unit but is not a 0/1 indicator.",
            recovery_hint=(
                "Pass the first-treatment period (constant per unit, 0 for "
                "never treated) or an absorbing 0/1 treatment indicator."
            ),
        )
    ordered = df.assign(_d=values).sort_values([id, time])
    if (ordered.groupby(id)["_d"].diff().fillna(0) < 0).any():
        raise MethodIncompatibility(
            "did_bcf(): the treatment indicator switches off within a unit; "
            "the design must be staggered adoption (absorbing treatment).",
            recovery_hint="Use an estimator for non-absorbing treatments.",
            alternative_functions=["sp.did_multiplegt_dyn"],
        )
    first = ordered[ordered["_d"] == 1].groupby(id)[time].min()
    return first.reindex(ordered[id].unique()).fillna(np.inf).astype(float)


def did_bcf(
    data: pd.DataFrame,
    y: str,
    treat: str,
    time: str,
    id: str,
    covariates: Optional[List[str]] = None,
    n_trees: int = 50,
    alpha: float = 0.05,
    seed: int = 0,
    n_bootstrap: int = 100,
) -> CausalResult:
    """
    DiD with a BCF-style forest on long-differenced outcomes.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    y : str
        Outcome column.
    treat : str
        First-treatment period (constant within unit; 0 or NaN for never
        treated), or an absorbing 0/1 treatment indicator from which the
        first-treatment period is inferred.
    time : str
        Numeric period column.
    id : str
        Unit identifier.
    covariates : list of str, optional
        Effect modifiers; unit averages are used, so pass baseline
        (pre-treatment) characteristics.
    n_trees : int, default 50
        Trees of the treatment-effect booster.
    alpha : float, default 0.05
    seed : int
    n_bootstrap : int, default 100
        Bootstrap replications of the BCF fit per cohort (covariate path).

    Returns
    -------
    CausalResult
        Overall ATT (cohort-size weighted) with per-cohort estimates in
        ``model_info['catt_by_cohort']`` and ``model_info['se_by_cohort']``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(30):
    ...     g = 0 if i < 15 else 3
    ...     for t in range(1, 6):
    ...         post = int(g > 0 and t >= g)
    ...         y = 1.0 + 0.2 * t + 1.5 * post + rng.normal(0, 0.5)
    ...         rows.append({"id": i, "time": t, "g": g, "y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.did_bcf(df, y="y", treat="g", time="time", id="id", seed=0)
    >>> res.estimand
    'ATT'
    >>> bool(np.isfinite(res.estimate))
    True

    References
    ----------
    [@hahn2020bayesian], [@souto2025forests]
    """
    from scipy import stats

    cov = list(covariates or [])
    check_columns(
        data,
        {"y": y, "treat": treat, "time": time, "id": id, "covariates": cov},
        function="sp.did_bcf",
    )
    df = data[[y, treat, time, id] + cov].dropna(subset=[y, time, id, *cov])
    df = df.reset_index(drop=True)
    df[time] = pd.to_numeric(df[time])
    cohort = _unit_cohorts(df, treat, time, id)
    unit_ids = cohort.index.to_numpy()
    periods = np.sort(df[time].unique())
    wide = (
        df.pivot_table(index=id, columns=time, values=y, aggfunc="mean")
        .reindex(index=unit_ids, columns=periods)
        .to_numpy(dtype=float)
    )
    X_units = (
        df.groupby(id)[cov].mean().reindex(unit_ids).to_numpy(dtype=float)
        if cov
        else None
    )
    cohort_arr = cohort.to_numpy(dtype=float)
    never = ~np.isfinite(cohort_arr)
    if not never.any():
        raise DataInsufficient(
            "did_bcf(): no never-treated units to compare against.",
            recovery_hint="Use sp.did_forest(control_group='notyettreated').",
            alternative_functions=["sp.did_forest"],
        )

    catt: Dict[float, float] = {}
    se_by: Dict[float, float] = {}
    n_by: Dict[float, int] = {}
    skipped: Dict[float, str] = {}
    contrib: Dict[float, np.ndarray] = {}
    for g in sorted(c for c in np.unique(cohort_arr) if np.isfinite(c)):
        pre = periods < g
        post = periods >= g
        if not pre.any() or not post.any():
            skipped[float(g)] = "no pre-treatment or no post-treatment period"
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dy = np.nanmean(wide[:, post], axis=1) - np.nanmean(wide[:, pre], axis=1)
        treated = (cohort_arr == g) & np.isfinite(dy)
        control = never & np.isfinite(dy)
        n1, n0 = int(treated.sum()), int(control.sum())
        if n1 < 2 or n0 < 2:
            skipped[float(g)] = f"too few units (treated {n1}, control {n0})"
            continue
        if X_units is None:
            m1, m0 = dy[treated].mean(), dy[control].mean()
            catt[float(g)] = float(m1 - m0)
            inf = np.zeros(unit_ids.size)
            inf[treated] = (dy[treated] - m1) / n1
            inf[control] = -(dy[control] - m0) / n0
            contrib[float(g)] = inf
            se_by[float(g)] = float(np.sqrt(np.sum(inf**2)))
        else:
            from ..bcf import bcf as bcf_fit

            idx = np.flatnonzero(treated | control)
            block = pd.DataFrame(X_units[idx], columns=cov)
            block["_dy"] = dy[idx]
            block["_d"] = treated[idx].astype(int)
            res = bcf_fit(
                data=block,
                y="_dy",
                treat="_d",
                covariates=cov,
                n_trees_tau=n_trees,
                n_bootstrap=n_bootstrap,
                n_folds=3,
                random_state=seed,
            )
            tau = np.asarray(res.model_info["cate"], dtype=float)
            tr = block["_d"].to_numpy() == 1
            catt[float(g)] = float(tau[tr].mean())
            draws = np.asarray(res._bootstrap_cate, dtype=float)[:, tr].mean(axis=1)
            se_by[float(g)] = float(np.std(draws, ddof=1))
        n_by[float(g)] = n1

    if not catt:
        raise DataInsufficient(
            "did_bcf(): no cohort could be estimated.",
            recovery_hint="Check the treatment timing and never-treated units.",
            diagnostics={"skipped_cohorts": skipped},
        )
    if skipped:
        warnings.warn(
            f"did_bcf(): skipped cohorts {sorted(skipped)}; see "
            "model_info['skipped_cohorts'].",
            AssumptionWarning,
            stacklevel=2,
        )
    weights = {g: n_by[g] / sum(n_by.values()) for g in catt}
    att = float(sum(weights[g] * catt[g] for g in catt))
    if X_units is None:
        total = sum(weights[g] * contrib[g] for g in catt)
        n_units = int(np.sum(np.abs(total) > 0))
        se = float(np.sqrt(np.sum(total**2) * n_units / max(n_units - 1, 1)))
        se_method = "influence function (difference in mean long differences)"
    else:
        se = float(sum(weights[g] * se_by[g] for g in catt))
        se_method = (
            "bootstrap per cohort; overall = sum_g w_g se_g (conservative, "
            "cohorts share control units)"
        )

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (att - z_crit * se, att + z_crit * se)
    pvalue = float(2 * stats.norm.sf(abs(att / se))) if se > 0 else float("nan")

    model_info: dict[str, Any] = {
        "estimator": "DiD with BCF-style forest on long differences (StatsPAI)",
        "n_trees": n_trees,
        "n_covariates": len(cov),
        "catt_by_cohort": catt,
        "se_by_cohort": se_by,
        "cohort_weights": weights,
        "skipped_cohorts": skipped,
        "control_group": "never treated",
        "se_method": se_method,
    }

    _result = CausalResult(
        method="DiD-BCF (BCF-style forest on long differences)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=int(sum(n_by.values()) + never.sum()),
        model_info=model_info,
        _citation_key="did_bcf",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.did_bcf",
            params={
                "y": y,
                "treat": treat,
                "time": time,
                "id": id,
                "covariates": list(covariates) if covariates else None,
                "n_trees": n_trees,
                "alpha": alpha,
                "seed": seed,
                "n_bootstrap": n_bootstrap,
            },
            data=data,
            overwrite=False,
        )
    except (ImportError, AttributeError, TypeError) as exc:  # pragma: no cover
        _result.model_info["provenance_error"] = f"{type(exc).__name__}: {exc}"
    return _result


# Citation
CausalResult._CITATIONS["did_bcf"] = (
    "@article{souto2025forests,\n"
    "  title={Forests for Differences: Robust Causal Inference Beyond "
    "Parametric DiD},\n"
    "  author={Souto, Hugo Gobato and Neto, Francisco Louzada},\n"
    "  journal={arXiv preprint arXiv:2505.09706},\n"
    "  year={2025}\n"
    "}"
)
