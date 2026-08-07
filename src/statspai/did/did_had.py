"""Heterogeneous-adoption DiD with quasi-untreateded groups (did_had).

de Chaisemartin, Ciccia, D'Haultfœuille & Knau, "Difference-in-Differences
Estimators When No Unit Remains Untreated" (arXiv:2405.04465).

The design
----------
Every group is untreated in period 1, and every treated group adopts at
the **same** period F, with heterogeneous dose. Timing does not vary — if
it does, this is the wrong estimator and ``sp.did_multiplegt_dyn`` is the
right one.

The problem is that **no group stays untreateded**, so there is no control
group to supply the counterfactual outcome evolution. What the design
does offer, when doses are concentrated near zero, is *quasi-untreateded*
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
from ..nonparametric.lprobust import lpbwselect_mse_dpi, lprobust_at_point

__all__ = ["did_had", "quasi_untreated_test", "yatchew_linearity_test"]


def quasi_untreated_test(dose: Sequence[float]) -> Dict[str, float]:
    """Test that quasi-untreateded groups exist (dCDH et al., §3.3).

    The whole design rests on some group having a dose close enough to
    zero to stand in for an untreated one. This checks it, in closed
    form, with no bandwidth involved::

        T = D(1) / (D(2) − D(1))        smallest positive dose over the
                                        gap to the second smallest
        p = 1 / (1 + T)

    ``T`` converges to a ratio of two iid Exponential(1) variables, whose
    CDF is ``t/(1+t)`` — hence the p-value. A **large** T means the
    smallest dose is far from zero relative to the spacing, i.e. no
    credible quasi-untreateded group, and drives p toward 0.

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
    >>> res['pvalue'] > 0.5   # tiny smallest dose -> quasi-untreateded exist
    True
    """
    d = np.asarray(dose, dtype=float)
    d = np.sort(d[np.isfinite(d) & (d > 0)])
    if d.size < 2:
        raise ValueError(
            "the quasi-untreateded test needs at least two distinct positive "
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


def _panel_matrices(data: pd.DataFrame, y: str, group: str, time: str, treat: str):
    """Reshape to group x period matrices and locate the common adoption date."""
    for col in (y, group, time, treat):
        if col not in data.columns:
            raise ValueError(f"Column {col!r} not found in data.")

    df = data[[group, time, y, treat]].copy()
    df = df.sort_values([group, time])

    wide_y = df.pivot_table(index=group, columns=time, values=y, aggfunc="first")
    wide_d = df.pivot_table(index=group, columns=time, values=treat, aggfunc="first")
    periods = list(wide_y.columns)

    # Each group's first treatment CHANGE. Groups that never change are
    # stayers -- dose 0 throughout -- and belong in the estimation as the
    # most quasi-untreateded groups there are, not dropped.
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
    treat: str,
    *,
    effects: int = 1,
    placebo: int = 0,
    bandwidth: Union[float, Sequence[float], str] = "mse-dpi",
    kernel: str = "epanechnikov",
    alpha: float = 0.05,
    dynamic: bool = False,
    trends_lin: bool = False,
    yatchew: bool = False,
) -> CausalResult:
    """Heterogeneous-adoption DiD using quasi-untreateded groups.

    Parameters
    ----------
    data : DataFrame
        Long panel, one row per group-period.
    y, group, time, treat : str
        Column names. ``treat`` is the dose, not an indicator.
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
    yatchew : bool, default False
        Report the Yatchew differencing test alongside each horizon.
        Effects are tested for **linearity** in the dose (order 1),
        placebos for **mean independence** of the pre-period evolution
        from the future dose (order 0).

        Theorem 5 of the paper: with (quasi-)untreated groups, plain OLS
        of the evolution on the dose is unbiased for the same estimand
        *iff* that conditional expectation is linear. Failing to reject
        therefore licenses the far simpler estimator; rejecting says the
        nonparametric machinery is doing real work.
    trends_lin : bool, default False
        Allow group-specific linear trends, estimated from each group's
        ``F-2`` to ``F-1`` evolution and subtracted. Costs one placebo,
        and needs at least three pre-treat periods.

    Returns
    -------
    CausalResult
        ``.detail`` has one row per horizon with ``estimate``, ``se``,
        ``ci_lower``, ``ci_upper``, ``bandwidth``, ``n_in_bw`` and the
        quasi-untreateded test. ``.estimate`` is effect 1.

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
    ...                  effects=2)                       # doctest: +SKIP

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

    wide_y, wide_d, periods, f_period = _panel_matrices(data, y, group, time, treat)
    pos = {p: k for k, p in enumerate(periods)}
    if f_period not in pos:
        raise ValueError(f"adoption period {f_period} is absent from {time}.")
    f_idx = pos[f_period]

    if f_idx < 1:
        raise ValueError(
            "did_had needs at least one pre-treat period; adoption "
            f"occurs at the first observed period ({f_period})."
        )

    y_mat = wide_y.to_numpy(dtype=float)
    d_mat = wide_d.to_numpy(dtype=float)
    base = y_mat[:, f_idx - 1]  # Y_{F-1}

    lin_trend = None
    if trends_lin:
        if f_idx < 2:
            raise ValueError(
                "trends_lin needs at least three pre-treat periods so "
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
    # 'mse-dpi' is selected per horizon, from that horizon's own (dose,
    # evolution) pair -- exactly as Stata re-runs lprobust for each
    # effect and placebo, which is why its BW column varies down the table.
    auto_bw = isinstance(bandwidth, str)
    if auto_bw:
        if bandwidth != "mse-dpi":
            raise ValueError(
                f"bandwidth must be 'mse-dpi', a float, or one float per "
                f"horizon; got {bandwidth!r}."
            )
        bws: List[Optional[float]] = [None] * len(horizons)
    else:
        bws = list(_resolve_bandwidths(bandwidth, len(horizons)))

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

        if bw is None:
            # lprobust's rho defaults to 1, so it sets b = h/rho = h
            # rather than using the selector's own b. did_had inherits
            # that, so the bias bandwidth here is h, not the b the
            # selector would report.
            bw = lpbwselect_mse_dpi(dose_k, dy_k, 0.0, kernel=kernel)["h"]
        fit = lprobust_at_point(dose_k, dy_k, 0.0, h=bw, b=bw, kernel=kernel)
        beta = (float(dy_k.mean()) - fit.tau_us) / mean_dose
        bias = -fit.bias / mean_dose
        se = fit.se_rb / mean_dose
        qug = quasi_untreated_test(dose_k)

        yat = None
        if yatchew:
            # Effects test LINEARITY (order 1); placebos test mean
            # INDEPENDENCE of the pre-period evolution from the future
            # dose (order 0). That asymmetry is the reference's, and it
            # is the right one: the placebo null is not "linear in dose".
            yat = yatchew_linearity_test(
                dose_k, dy_k, order=1 if h_rel > 0 else 0, het_robust=True
            )

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
                "yatchew_statistic": yat["statistic"] if yat else np.nan,
                "yatchew_pvalue": yat["pvalue"] if yat else np.nan,
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
            "bandwidth": [float(r["bandwidth"]) for r in rows],
            "bandwidth_method": "mse-dpi" if auto_bw else "supplied",
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


def yatchew_linearity_test(
    x: Sequence[float],
    y: Sequence[float],
    *,
    order: int = 1,
    het_robust: bool = True,
) -> Dict[str, float]:
    """Yatchew differencing test that E[y | x] is a polynomial of ``order``.

    Why ``did_had`` reports it: Theorem 5 of de Chaisemartin et al. says
    that in a design with (quasi-)untreated groups, the plain OLS
    coefficient from regressing the outcome evolution on the dose is
    unbiased for the same estimand **if and only if** that conditional
    expectation is linear. So a non-rejection licenses the far simpler
    OLS estimator, and a rejection says the nonparametric machinery is
    doing real work rather than decorating an OLS number.

    The test compares two variance estimates of the same residual
    variance:

    * ``s2_lin`` — residual variance from the order-``order`` polynomial
      fit, consistent **only if** that model is right;
    * ``s2_diff`` — ``0.5 * mean((y_i − y_{i−1})²)`` over the x-sorted
      data, consistent whatever the shape.

    Under the null they agree; under the alternative the parametric one
    is inflated. The test is therefore one-sided.

    ``order=0`` tests mean-independence rather than linearity, which is
    what the *placebo* horizons need: there the null is that the
    pre-period evolution does not depend on the future dose.

    Parameters
    ----------
    x, y : sequence of float
        Running variable and outcome, same length.
    order : int, default 1
        Polynomial order under the null. 1 = linear, 0 = constant.
    het_robust : bool, default True
        Use the heteroskedasticity-robust statistic of the paper's
        Appendix E. ``did_had`` always does.

    Returns
    -------
    dict with ``s2_lin``, ``s2_diff``, ``statistic``, ``pvalue``, ``n``.

    Examples
    --------
    >>> import numpy as np, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 1, 400)
    >>> lin = sp.yatchew_linearity_test(x, 2 * x + rng.normal(0, .1, 400))
    >>> bool(lin['pvalue'] > 0.05)      # truly linear: not rejected
    True

    References
    ----------
    yatchew1999elementary, dechaisemartin2024nounit
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same length, got {x.shape}, {y.shape}")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 3:
        raise ValueError(
            f"the differencing test needs at least 3 observations, got {n}"
        )
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    # Parametric fit. order=0 leaves the intercept alone, which makes the
    # null "y is mean-independent of x" rather than "linear in x".
    cols = [np.ones(n)] + [x**j for j in range(1, order + 1)]
    design = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    s2_lin = float(np.var(resid, ddof=1))

    # Nonparametric benchmark: first differences along sorted x, with
    # ties in x broken by y ascending -- `sort D_vars_XX Y_XX` in the
    # reference. Not cosmetic: the differences are taken between
    # ADJACENT rows, so the order within a tie group changes s2_diff.
    # With 25 tied zero doses on the did_had fixture, sorting by x alone
    # moved s2_diff by 3.7e-3 and the statistic by 0.36.
    srt = np.lexsort((y, x))
    dy = np.diff(y[srt])
    s2_diff = float(0.5 * np.mean(dy**2))

    if het_robust:
        e = resid[srt]
        denom = float(np.mean((e[1:] * e[:-1]) ** 2))
        stat = float(np.sqrt(n) * (s2_lin - s2_diff) / np.sqrt(denom))
    else:
        stat = float(np.sqrt(n) * (s2_lin / s2_diff - 1.0))

    return {
        "s2_lin": s2_lin,
        "s2_diff": s2_diff,
        "statistic": stat,
        "pvalue": float(1.0 - stats.norm.cdf(stat)),
        "n": int(n),
    }
