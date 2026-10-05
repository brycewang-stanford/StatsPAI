"""Debiased synthetic control with a t-test for the average effect.

The average post-period gap of a synthetic control is biased when the
weights are estimated on few pre-periods, and placebo or conformal
inference tests sharp nulls. Chernozhukov, Wüthrich and Zhu (2026) estimate
the average effect over the post-period with a K-fold cross-fitting scheme
over time: the weights are fitted on the pre-periods outside a block, the
gap on that block estimates the bias of those weights, and the bias is
subtracted from the post-period gap. The K debiased estimates are
approximately independent and normal, so their mean has a t distribution
with ``K - 1`` degrees of freedom after self-normalisation.

References
----------
chernozhukov2026debiasing
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["synth_ttest"]


@accepts_aliases(id="unit", y="outcome")
def synth_ttest(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    n_folds: int = 3,
    alpha: float = 0.05,
    penalization: float = 0.0,
) -> CausalResult:
    """Debiased synthetic control ATT with a t-test (Chernozhukov, Wüthrich & Zhu).

    Let ``r = min(floor(T0 / K), T1)`` and take the last ``K r`` pre-periods
    as ``K`` consecutive blocks ``H_1, ..., H_K`` of ``r`` periods. For each
    block the simplex-constrained synthetic control weights ``w_k`` are
    fitted on the pre-periods outside ``H_k`` and

    ``tau_k = mean_post(Y1 - Y0 w_k) - mean_{H_k}(Y1 - Y0 w_k)``.

    The estimate is the mean of the ``tau_k``; its standard error is
    ``sqrt(1 + K r / T1) * sd(tau_k) / sqrt(K)`` and the reference
    distribution is Student's t with ``K - 1`` degrees of freedom. This is
    the ``inference_method = "ttest"`` procedure of the authors' R package
    ``scinference``.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel, one row per unit and period.
    outcome : str
        Outcome column.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treated period.
    n_folds : int, default 3
        Number of blocks ``K`` (at least 2). The interval uses ``K - 1``
        degrees of freedom, so ``K = 2`` (the default of ``scinference``)
        gives a Cauchy reference distribution and very wide intervals;
        a larger ``K`` leaves fewer pre-periods per block.
    alpha : float, default 0.05
        Level of the confidence interval.
    penalization : float, default 0.0
        Ridge penalty on the weights.

    Returns
    -------
    CausalResult
        ``estimate`` is the debiased average effect on the treated unit over
        the post-period, with its ``se``, t-based ``ci`` and ``pvalue``.
        ``detail`` has one row per fold (``tau``, the post-period gap and
        the bias estimated on the held-out block); ``model_info`` holds
        ``df``, ``block_size`` and the plain synthetic control estimate
        ``att_sc`` fitted on all pre-periods, for comparison.

    Notes
    -----
    The target is the average effect over the post-period, and the method
    needs both the pre-period and the post-period to be long: each block
    and the post-period supply an average whose error has to be small. With
    stationary errors the K estimates are approximately independent; the
    factor ``1 + K r / T1`` accounts for the post-period average they share.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> res = sp.synth_ttest(df, outcome='packspercapita', unit='state',
    ...                      time='year', treated_unit='California',
    ...                      treatment_time=1989)
    >>> int(res.model_info['df'])
    2

    References
    ----------
    chernozhukov2026debiasing
    """
    from ._core import solve_simplex_weights

    n_folds = int(n_folds)
    if n_folds < 2:
        raise MethodIncompatibility("synth_ttest: n_folds must be at least 2.")
    missing = [c for c in (outcome, unit, time) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"synth_ttest: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    if data.duplicated(subset=[time, unit]).any():
        raise MethodIncompatibility(
            "synth_ttest needs one row per (unit, time); found duplicates.",
            recovery_hint="Aggregate or drop duplicate (unit, time) rows.",
        )
    pivot = data.pivot(index=time, columns=unit, values=outcome).sort_index()
    if treated_unit not in pivot.columns:
        raise MethodIncompatibility(
            f"synth_ttest: treated_unit {treated_unit!r} is not a value of {unit!r}."
        )
    times = pivot.index.values
    pre = times < treatment_time
    post = ~pre
    T0, T1 = int(pre.sum()), int(post.sum())
    if T1 < 1:
        raise DataInsufficient(
            "synth_ttest: no post-treatment period.",
            diagnostics={"n_post_periods": T1},
        )
    r = min(T0 // n_folds, T1)
    if r < 1 or T0 - r < 2:
        raise DataInsufficient(
            f"synth_ttest: {T0} pre-periods are too few for n_folds={n_folds}.",
            recovery_hint="Use fewer folds or a longer pre-period.",
            diagnostics={"n_pre_periods": T0, "n_folds": n_folds},
        )

    y1 = pivot[treated_unit].to_numpy(dtype=float)
    donors = [c for c in pivot.columns if c != treated_unit]
    Y0 = pivot[donors].to_numpy(dtype=float)
    if np.isnan(y1).any():
        raise DataInsufficient("synth_ttest: the treated unit has missing outcomes.")
    complete = ~np.isnan(Y0).any(axis=0)
    if not complete.any():
        raise DataInsufficient("synth_ttest: no donor has complete outcomes.")
    if not complete.all():
        import warnings

        warnings.warn(
            f"Dropped {int((~complete).sum())} donor(s) with missing outcomes.",
            UserWarning,
            stacklevel=2,
        )
        Y0 = Y0[:, complete]
        donors = [d for d, ok in zip(donors, complete) if ok]

    y1_pre, Y0_pre = y1[pre], Y0[pre]
    y1_post, Y0_post = y1[post], Y0[post]
    offset = T0 - r * n_folds
    rows = []
    for k in range(n_folds):
        hold = np.arange(offset + k * r, offset + (k + 1) * r)
        keep = np.ones(T0, dtype=bool)
        keep[hold] = False
        w = solve_simplex_weights(y1_pre[keep], Y0_pre[keep], penalization)
        gap_post = float(np.mean(y1_post - Y0_post @ w))
        bias = float(np.mean(y1_pre[hold] - Y0_pre[hold] @ w))
        rows.append(
            {
                "fold": k + 1,
                "tau": gap_post - bias,
                "post_gap": gap_post,
                "bias": bias,
                "holdout_start": times[pre][hold[0]],
                "holdout_end": times[pre][hold[-1]],
            }
        )
    detail = pd.DataFrame(rows)
    taus = detail["tau"].to_numpy()
    estimate = float(np.mean(taus))
    se = float(
        np.sqrt(1.0 + n_folds * r / T1) * np.std(taus, ddof=1) / np.sqrt(n_folds)
    )
    df = n_folds - 1
    if se > 0:
        t_stat = estimate / se
        pvalue = float(2 * stats.t.sf(abs(t_stat), df))
    else:
        t_stat, pvalue = float("nan"), float("nan")
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    ci = (estimate - crit * se, estimate + crit * se)

    w_all = solve_simplex_weights(y1_pre, Y0_pre, penalization)
    model_info = {
        "inference_method": "debiased K-fold t-test",
        "n_folds": n_folds,
        "df": df,
        "block_size": int(r),
        "t_stat": float(t_stat),
        "n_donors": len(donors),
        "n_pre_periods": T0,
        "n_post_periods": T1,
        "att_sc": float(np.mean(y1_post - Y0_post @ w_all)),
        "weights": dict(zip(donors, w_all)),
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
    }
    return CausalResult(
        method="Debiased Synthetic Control t-test (Chernozhukov et al. 2026)",
        estimand="ATT",
        estimate=estimate,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(y1),
        detail=detail,
        model_info=model_info,
        _citation_key="synth_ttest",
    )


# Kept in sync with paper.bib (key chernozhukov2026debiasing).
CausalResult._CITATIONS["synth_ttest"] = (
    "@article{chernozhukov2026debiasing,\n"
    "  title={Debiasing and $t$-Tests for Synthetic Control Inference on "
    "Average Causal Effects},\n"
    '  author={Chernozhukov, Victor and W{\\"u}thrich, Kaspar and Zhu, Yinchu},\n'
    "  journal={Journal of Political Economy},\n"
    "  volume={134},\n"
    "  number={9},\n"
    "  pages={2740--2777},\n"
    "  year={2026},\n"
    "  doi={10.1086/742424}\n"
    "}"
)
