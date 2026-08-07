"""Heterogeneous-adoption DiD with quasi-untreated groups (did_had).

de Chaisemartin, Ciccia, D'Haultfœuille & Knau, "Difference-in-Differences
Estimators When No Unit Remains Untreated" (arXiv:2405.04465).

The design
----------
Every group is untreated in period 1, and every treated group adopts at
the **same** period F, with heterogeneous dose. Timing does not vary — if
it does, this is the wrong estimator and ``sp.did_multiplegt_dyn`` is the
right one.

The problem is that **no group stays untreated**, so there is no control
group to supply the counterfactual outcome evolution. What the design
does offer, when doses are concentrated near zero, is *quasi-untreated*
groups: groups whose dose is small enough to be treated as effectively
zero. The counterfactual is then the intercept of a local polynomial
regression of the outcome evolution on the dose, evaluated at dose 0.

That makes the estimand a DiD/RD hybrid: a difference in outcome
evolutions, where one arm is an RD-style boundary extrapolation rather
than an observed group.

The estimator, per event-study horizon ℓ
----------------------------------------
    ΔY_ℓ = Y_{F-1+ℓ} − Y_{F-1}                    outcome evolution
    D_ℓ  = dose at F-1+ℓ  (or cumulative, see ``dynamic``)

    μ̂    = local polynomial fit of ΔY_ℓ on D_ℓ at D = 0
    β̂_ℓ  = (E[ΔY_ℓ] − μ̂) / E[D_ℓ]

The average evolution, minus what a dose-zero group would have done,
rescaled by the average dose. Standard errors and the bias correction
come from the same fit — see :func:`statspai.lprobust_at_point`.

.. warning::
   The reported interval is **not symmetric** around the point estimate.
   ``did_had`` reports the *conventional* estimate with a
   *bias-corrected* interval, centred at ``β̂ − B̂`` where
   ``B̂ = −(τ_us − τ_bc)/E[D]``. Reading the interval as
   ``estimate ± z·se`` will not reproduce it, and that is deliberate:
   the point estimate is the interpretable quantity, the interval is the
   one with correct coverage.

Quasi-untreated groups have to exist
------------------------------------
Everything above is meaningless if no group has a dose near zero, so the
paper's §3.3 test is reported alongside every effect. It compares the
smallest positive dose to the gap between the smallest and second
smallest; both statistics converge to a ratio of iid Exponential(1)
variables, giving ``p = 1/(1 + T)``. It needs no bandwidth and is
computed here in closed form.

An identical T and p across every horizon indicates the treatment
changes only once.

References
----------
- de Chaisemartin, C., Ciccia, D., D'Haultfœuille, X. and Knau, F.
  "Difference-in-Differences Estimators When No Unit Remains Untreated."
  arXiv preprint arXiv:2405.04465. [@dechaisemartin2024nounit]
- Calonico, S., Cattaneo, M. D. and Farrell, M. H. (2019), for the local
  polynomial engine. [@calonico2019nprobust]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..nonparametric.lprobust import lprobust_at_point

__all__ = ["did_had", "quasi_untreated_test"]


def quasi_untreated_test(dose: Sequence[float]) -> Dict[str, float]:
    """Test that quasi-untreated groups exist (dCDH et al., §3.3).

    The whole design rests on some group having a dose close enough to
    zero to stand in for an untreated one. This checks it, in closed
    form, with no bandwidth involved::

        T = D(1) / (D(2) − D(1))        smallest positive dose over the
                                        gap to the second smallest
        p = 1 / (1 + T)

    ``T`` converges to a ratio of two iid Exponential(1) variables, whose
    CDF is ``t/(1+t)`` — hence the p-value. A **large** T means the
    smallest dose is far from zero relative to the spacing, i.e. no
    credible quasi-untreated group, and drives p toward 0.

    Parameters
    ----------
    dose : sequence of float
        Doses at the horizon being tested. Non-positive values are
        dropped: an exact zero is a genuine stayer, not a quasi-stayer,
        and the statistic is about how close the *positive* doses get.

    Returns
    -------
    dict with ``statistic``, ``pvalue``, ``d_min``, ``d_second``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.quasi_untreated_test([0.001, 0.9, 1.4, 2.0])
    >>> res['pvalue'] > 0.5   # tiny smallest dose -> quasi-untreated exist
    True
    """
    d = np.asarray(dose, dtype=float)
    d = np.sort(d[np.isfinite(d) & (d > 0)])
    if d.size < 2:
        raise ValueError(
            "the quasi-untreated test needs at least two distinct positive "
            f"doses, got {d.size}."
        )
    gap = d[1] - d[0]
    if gap <= 0:
        # Tied smallest doses: the spacing is degenerate, so T is +inf and
        # the smallest dose is as close to the next as it can be.
        return {
            "statistic": float("inf"),
            "pvalue": 0.0,
            "d_min": float(d[0]),
            "d_second": float(d[1]),
        }
    t = float(d[0] / gap)
    return {
        "statistic": t,
        "pvalue": float(1.0 / (1.0 + t)),
        "d_min": float(d[0]),
        "d_second": float(d[1]),
    }


def _panel_matrices(data: pd.DataFrame, y: str, group: str, time: str, treatment: str):
    """Reshape to group x period matrices and locate the common adoption date."""
    for col in (y, group, time, treatment):
        if col not in data.columns:
            raise ValueError(f"Column {col!r} not found in data.")

    df = data[[group, time, y, treatment]].copy()
    df = df.sort_values([group, time])

    wide_y = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")
    wide_d = df.pivot_table(
        index=group, columns=time, values=treatment, aggfunc="first"
    )
    periods = list(wide_y.columns)

    # Each group's first treatment CHANGE. Groups that never change are
    # stayers -- dose 0 throughout -- and belong in the estimation as the
    # most quasi-untreated groups there are, not dropped.
    d = wide_d.to_numpy(dtype=float)
    changed = np.diff(d, axis=1) != 0
    first_change = np.full(d.shape[0], np.nan)
    for i in range(d.shape[0]):
        idx = np.flatnonzero(changed[i])
        if idx.size:
            first_change[i] = periods[idx[0] + 1]

    switchers = first_change[np.isfinite(first_change)]
    if switchers.size == 0:
        raise ValueError(
            "no group ever changes treatment, so there is no adoption date "
            "and no effect to estimate."
        )
    f_period = switchers[0]
    if not np.allclose(switchers, f_period):
        found = sorted({float(v) for v in switchers})
        raise ValueError(
            "did_had requires every treated group to adopt at the SAME "
            f"period, but adoption dates {found[:6]} were found. With "
            "variation in treatment timing use sp.did_multiplegt_dyn "
            "instead — the heterogeneous-adoption estimator is not valid "
            "here."
        )
    return wide_y, wide_d, periods, float(f_period)


def did_had(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    treatment: str,
    *,
    effects: int = 1,
    placebo: int = 0,
    bandwidth: Union[float, Sequence[float], None] = None,
    kernel: str = "epanechnikov",
    alpha: float = 0.05,
    dynamic: bool = False,
    trends_lin: bool = False,
) -> CausalResult:
    """Heterogeneous-adoption DiD using quasi-untreated groups.

    Parameters
    ----------
    data : DataFrame
        Long panel, one row per group-period.
    y, group, time, treatment : str
        Column names. ``treatment`` is the dose, not an indicator.
    effects : int, default 1
        Number of event-study effects. Effect ℓ is the effect at period
        ``F-1+ℓ``, i.e. ℓ periods after adoption.
    placebo : int, default 0
        Number of placebo estimates, built symmetrically: the ``F-1`` to
        ``F-1+ℓ`` evolution is replaced by ``F-1`` to ``F-1-ℓ``, with the
        dose taken from the matching post period.
    bandwidth : float or sequence of float, optional
        Bandwidth for the local polynomial fit at dose zero — one value,
        or one per reported horizon (placebos first, then effects).

        **Required for now.** Stata's default ``bw_method('mse-dpi')``
        selector is not yet implemented; see Notes.
    kernel : {'epanechnikov', 'triangular', 'uniform', 'gaussian'}
        Default epanechnikov, matching ``did_had``.
    alpha : float, default 0.05
        1 − α confidence level.
    dynamic : bool, default False
        Scale effect ℓ by the average **cumulative** dose from F to
        ``F-1+ℓ`` instead of the dose at ``F-1+ℓ``. The current-dose
        normalization is right under a static model, the cumulative one
        under a dynamic model where past treatment still matters.
    trends_lin : bool, default False
        Allow group-specific linear trends, estimated from each group's
        ``F-2`` to ``F-1`` evolution and subtracted. Costs one placebo,
        and needs at least three pre-treatment periods.

    Returns
    -------
    CausalResult
        ``.detail`` has one row per horizon with ``estimate``, ``se``,
        ``ci_lower``, ``ci_upper``, ``bandwidth``, ``n_in_bw`` and the
        quasi-untreated test. ``.estimate`` is effect 1.

    Notes
    -----
    **The interval is not symmetric around the estimate.** It is centred
    at ``estimate − bias``; see the module docstring.

    **Bandwidth selection is not implemented.** ``did_had`` defaults to
    ``mse-dpi``, whose implementation lives in a compiled Mata library
    (``nprobust``'s ``nprobust_lp_mse_dpi``) with no readable source, so
    reproducing it means re-deriving the selector from Calonico,
    Cattaneo & Farrell (2019) rather than porting it. Until that is done
    and pinned, ``bandwidth`` must be supplied explicitly — an estimator
    that silently used a *different* bandwidth from the reference would
    produce numbers that look right and are not.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=60, n_periods=6, seed=0)  # doctest: +SKIP
    >>> res = sp.did_had(df, 'y', 'unit', 'time', 'dose',
    ...                  effects=2, bandwidth=0.5)        # doctest: +SKIP

    References
    ----------
    dechaisemartin2024nounit, calonico2019nprobust
    """
    if effects < 1:
        raise ValueError(f"effects must be >= 1, got {effects}")
    if placebo < 0:
        raise ValueError(f"placebo must be >= 0, got {placebo}")
    if bandwidth is None:
        raise ValueError(
            "did_had requires an explicit bandwidth=. Stata's default "
            "bw_method('mse-dpi') selector is not implemented yet — its "
            "source is a compiled Mata routine — and guessing a bandwidth "
            "would silently change every estimate. Pass the bandwidth you "
            "want, or the one Stata's did_had reports in its BW column."
        )

    wide_y, wide_d, periods, f_period = _panel_matrices(data, y, group, time, treatment)
    pos = {p: k for k, p in enumerate(periods)}
    if f_period not in pos:
        raise ValueError(f"adoption period {f_period} is absent from {time}.")
    f_idx = pos[f_period]

    if f_idx < 1:
        raise ValueError(
            "did_had needs at least one pre-treatment period; adoption "
            f"occurs at the first observed period ({f_period})."
        )

    y_mat = wide_y.to_numpy(dtype=float)
    d_mat = wide_d.to_numpy(dtype=float)
    base = y_mat[:, f_idx - 1]  # Y_{F-1}

    lin_trend = None
    if trends_lin:
        if f_idx < 2:
            raise ValueError(
                "trends_lin needs at least three pre-treatment periods so "
                "each group's F-2 to F-1 evolution can proxy its trend; "
                f"adoption at {f_period} leaves too few."
            )
        lin_trend = base - y_mat[:, f_idx - 2]

    max_effects = len(periods) - f_idx
    if effects > max_effects:
        raise ValueError(
            f"effects={effects} exceeds the {max_effects} estimable from "
            f"this panel (adoption at {f_period}, last period {periods[-1]})."
        )
    max_placebo = f_idx - 1 - (1 if trends_lin else 0)
    if placebo > max_placebo:
        raise ValueError(
            f"placebo={placebo} exceeds the {max_placebo} estimable"
            + (" once trends_lin consumes one" if trends_lin else "")
            + f" (adoption at {f_period})."
        )

    horizons = [-k for k in range(placebo, 0, -1)] + list(range(1, effects + 1))
    bws = _resolve_bandwidths(bandwidth, len(horizons))

    rows: List[Dict[str, Any]] = []
    z = float(stats.norm.ppf(1 - alpha / 2))

    for h_rel, bw in zip(horizons, bws):
        if h_rel > 0:
            t_idx = f_idx + h_rel - 1  # period F-1+l
            dy = y_mat[:, t_idx] - base
            if lin_trend is not None:
                dy = dy - h_rel * lin_trend
        else:
            k = -h_rel
            t_idx = f_idx - 1 - k  # period F-1-l
            dy = y_mat[:, t_idx] - base
            if lin_trend is not None:
                dy = dy + k * lin_trend
            # The placebo carries the dose of the SYMMETRIC post period,
            # so it asks the same question of the pre-period evolution.
            t_idx = f_idx + k - 1

        if dynamic:
            dose = d_mat[:, f_idx : t_idx + 1].sum(axis=1)
        else:
            dose = d_mat[:, t_idx]

        ok = np.isfinite(dy) & np.isfinite(dose)
        dy_k, dose_k = dy[ok], dose[ok]
        if dy_k.size < 5:
            raise ValueError(
                f"horizon {h_rel}: only {dy_k.size} groups have both an "
                "outcome evolution and a dose; too few to fit."
            )

        mean_dose = float(dose_k.mean())
        if abs(mean_dose) < 1e-12:
            raise ValueError(
                f"horizon {h_rel}: the average dose is {mean_dose:.3g}, so "
                "the effect per unit of treatment is not identified — "
                "every group is (quasi-)untreated at this horizon."
            )

        fit = lprobust_at_point(dose_k, dy_k, 0.0, h=bw, b=bw, kernel=kernel)
        beta = (float(dy_k.mean()) - fit.tau_us) / mean_dose
        bias = -fit.bias / mean_dose
        se = fit.se_rb / mean_dose
        qug = quasi_untreated_test(dose_k)

        rows.append(
            {
                "relative_time": h_rel,
                "type": "effect" if h_rel > 0 else "placebo",
                "estimate": beta,
                "se": se,
                # Centred at beta - bias, NOT at beta -- see the module
                # docstring. Reading it as estimate +/- z*se is wrong.
                "ci_lower": beta - bias - z * se,
                "ci_upper": beta - bias + z * se,
                "bias": bias,
                "bandwidth": float(bw),
                "n_groups": int(dy_k.size),
                "n_in_bw": int((dose_k <= bw).sum()),
                "qug_statistic": qug["statistic"],
                "qug_pvalue": qug["pvalue"],
                "mean_dose": mean_dose,
            }
        )

    detail = pd.DataFrame(rows)
    headline = detail[detail["relative_time"] == 1].iloc[0]

    return CausalResult(
        estimate=float(headline["estimate"]),
        se=float(headline["se"]),
        # The interval is bias-corrected, so it is NOT estimate +/- z*se.
        ci=(float(headline["ci_lower"]), float(headline["ci_upper"])),
        alpha=alpha,
        pvalue=(
            float(2 * (1 - stats.norm.cdf(abs(headline["estimate"] / headline["se"]))))
            if headline["se"] > 0
            else 1.0
        ),
        n_obs=int(len(data)),
        method="did_had",
        estimand="WAS (weighted average of slopes) at F-1+1",
        detail=detail,
        model_info={
            "estimator": "did_had (de Chaisemartin, Ciccia, D'Haultfoeuille & Knau)",
            "adoption_period": f_period,
            "effects": effects,
            "placebo": placebo,
            "kernel": kernel,
            "dynamic": dynamic,
            "trends_lin": trends_lin,
            "bandwidth": list(map(float, bws)),
            "n_groups": int(y_mat.shape[0]),
            "quasi_untreated_test": {
                "statistic": rows[-1]["qug_statistic"],
                "pvalue": rows[-1]["qug_pvalue"],
            },
            "ci_is_bias_corrected": True,
        },
    )


def _resolve_bandwidths(
    bandwidth: Union[float, Sequence[float]], n: int
) -> List[float]:
    """One bandwidth per horizon, from a scalar or a sequence."""
    if np.isscalar(bandwidth):
        vals = [float(bandwidth)] * n
    else:
        vals = [float(v) for v in bandwidth]  # type: ignore[union-attr]
        if len(vals) != n:
            raise ValueError(
                f"bandwidth has {len(vals)} entries but {n} horizons are "
                "reported (placebos first, then effects). Pass one value "
                "or exactly that many."
            )
    for v in vals:
        if not np.isfinite(v) or v <= 0:
            raise ValueError(f"each bandwidth must be positive and finite, got {v}")
    return vals
