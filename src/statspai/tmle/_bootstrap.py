"""Nonparametric bootstrap for :func:`statspai.tmle`.

The influence-function standard error treats the fitted nuisances as
known. That is correct to first order and anti-conservative when inverse
propensity weights are large. Resampling rows and rerunning the whole fit
(nuisance models, targeting, plug-in) carries the nuisance estimation into
the spread of the estimates.
"""

import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, StatsPAIError

_LOG_SCALE = ("RR", "OR")


def _pvalue(estimate: float, draws: np.ndarray, log_scale: bool) -> float:
    if log_scale:
        ok = draws > 0
        se = float(np.std(np.log(draws[ok]), ddof=1))
        z = np.log(estimate) / se if se > 0 else np.nan
    else:
        se = float(np.std(draws, ddof=1))
        z = estimate / se if se > 0 else np.nan
    return float(2 * sp_stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan


def bootstrap_inference(result: CausalResult, est: Any) -> None:
    """Replace the inference of ``result`` by bootstrap inference, in place.

    ``est`` is the fitted :class:`~statspai.tmle.tmle.TMLE`. Rows are
    resampled with replacement (whole clusters when ``est.cluster`` is
    set), each resample is fitted with the same settings, and

    * ``se`` becomes the standard deviation of the resampled estimates,
    * ``ci`` their percentile interval,
    * ``pvalue`` the normal p-value of ``estimate / se`` (of the log
      estimate for a ratio or an odds ratio).

    The rows of ``result.detail``, when present, get the same treatment.
    A resample that cannot be fitted (one arm missing, say) is skipped and
    counted; more than a tenth of them failing raises.
    """
    from .tmle import TMLE

    design = [c for c in (est.weights, est.cluster) if c is not None]
    cols = list(dict.fromkeys([est.y, est.treat] + list(est.covariates) + design))
    keep = est.data[cols].notna().all(axis=1).to_numpy()
    clean = est.data[cols].dropna().reset_index(drop=True)
    n = len(clean)
    # CV-TMLE: a copy of a row stays in that row's fold, so the model that
    # predicts it has seen neither it nor its copies.
    folds = None if est.fold_indices is None else np.asarray(est.fold_indices)[keep]
    rng = np.random.default_rng(est.random_state)
    members: List[np.ndarray] = []
    if est.cluster is not None:
        codes = pd.factorize(clean[est.cluster])[0]
        members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]

    draws: List[float] = []
    detail_draws: Dict[str, List[float]] = {}
    failed = 0
    for _ in range(est.n_boot):
        if members:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
            sample = clean.iloc[idx].reset_index(drop=True)
            # A cluster drawn twice is two clusters in the resample.
            sample[est.cluster] = np.repeat(
                np.arange(len(pick)), [len(members[g]) for g in pick]
            )
        else:
            idx = rng.integers(0, n, n)
            sample = clean.iloc[idx].reset_index(drop=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rep = TMLE(
                    data=sample,
                    y=est.y,
                    treat=est.treat,
                    covariates=est.covariates,
                    outcome_library=est.outcome_library,
                    propensity_library=est.propensity_library,
                    n_folds=est.n_folds,
                    estimand=est.estimand,
                    alpha=est.alpha,
                    propensity_bounds=est.propensity_bounds,
                    random_state=est.random_state,
                    fluctuation=est.fluctuation,
                    q_bound=est.q_bound,
                    fold_indices=None if folds is None else folds[idx],
                    weights=est.weights,
                    cluster=est.cluster,
                ).fit()
        except (ValueError, np.linalg.LinAlgError, StatsPAIError):
            failed += 1
            continue
        if not np.isfinite(rep.estimate):
            failed += 1
            continue
        draws.append(float(rep.estimate))
        if rep.detail is not None:
            for name, value in zip(rep.detail["parameter"], rep.detail["estimate"]):
                detail_draws.setdefault(name, []).append(float(value))

    if failed > 0.1 * est.n_boot or len(draws) < 20:
        raise DataInsufficient(
            f"tmle: {failed} of {est.n_boot} bootstrap resamples could not be "
            "fitted; the standard error would rest on a selected subset.",
            recovery_hint=(
                "Check that both treatment arms are well represented (and, "
                "with cluster=, in enough clusters), or use "
                "se_method='influence'."
            ),
            diagnostics={"n_boot": est.n_boot, "n_failed": failed},
        )
    if failed:
        warnings.warn(
            f"tmle: {failed} of {est.n_boot} bootstrap resamples could not be "
            "fitted and were skipped.",
            UserWarning,
            stacklevel=3,
        )

    boot = np.asarray(draws)
    lo_q, hi_q = est.alpha / 2, 1 - est.alpha / 2
    info = result.model_info
    info["se_influence"] = float(result.se)
    info["ci_influence"] = tuple(float(v) for v in result.ci)
    result.se = float(np.std(boot, ddof=1))
    result.ci = (float(np.quantile(boot, lo_q)), float(np.quantile(boot, hi_q)))
    result.pvalue = _pvalue(result.estimate, boot, est.estimand in _LOG_SCALE)
    info["se_method"] = "bootstrap" if est.cluster is None else "cluster_bootstrap"
    info["n_boot"] = est.n_boot
    info["n_boot_failed"] = failed
    info["bootstrap_estimates"] = boot

    if result.detail is not None:
        table = result.detail.copy()
        for i, name in enumerate(table["parameter"]):
            d = np.asarray(detail_draws.get(name, []))
            if d.size < 20:
                table.loc[i, ["se", "ci_lower", "ci_upper", "pvalue", "se_log"]] = (
                    np.nan
                )
                continue
            log_scale = name in _LOG_SCALE
            table.loc[i, "se"] = float(np.std(d, ddof=1))
            table.loc[i, "ci_lower"] = float(np.quantile(d, lo_q))
            table.loc[i, "ci_upper"] = float(np.quantile(d, hi_q))
            table.loc[i, "pvalue"] = _pvalue(
                float(table.loc[i, "estimate"]), d, log_scale
            )
            if log_scale:
                table.loc[i, "se_log"] = float(np.std(np.log(d[d > 0]), ddof=1))
        result.detail = table
