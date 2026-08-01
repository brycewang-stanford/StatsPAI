"""Callaway & Li (2019) quantile treatment effect on the treated, panel data.

The QTT is

.. math::

    QTT(\\tau) = F^{-1}_{Y_t(1)|D=1}(\\tau) - F^{-1}_{Y_t(0)|D=1}(\\tau)

The first term is observed.  The second is not: we never see untreated
outcomes for the treated group in period ``t``.  A mean DiD recovers only
``E[Y_t(0)|D=1]``, which pins the *location* of that counterfactual
distribution but not its shape -- and the shape is the entire object of
interest here.

Callaway & Li close the gap with a **copula stability** assumption: the
dependence between a unit's period-``t`` change and its period-``t-1``
level, among the treated, is the same as the dependence between its
period-``t-1`` change and its period-``t-2`` level.  That is why the
estimator needs **three** periods where a mean DiD needs two.  The
assumption restricts dependence, not marginals, so it is not testable at
``t`` (the relevant change is counterfactual) -- but it *is* testable on the
untreated group, where both copulas are observed.  :func:`panel_qtet`
performs that check and reports it in ``model_info['copula_check']``.

Algorithm (no covariates), matching ``qte::panel.qtet``
-------------------------------------------------------
For each treated unit ``i``:

1. ``u_i  = F_{Y_{t-2}|D=1}(Y_{i,t-2})`` -- the unit's rank two periods back.
2. ``q1_i = F^{-1}_{Y_{t-1}|D=1}(u_i)`` -- that rank's period-``t-1`` level.
3. ``v_i  = F_{dY_{t-1}|D=1}(dY_{i,t-1})`` -- the rank of its lagged change.
4. ``q2_i = F^{-1}_{dY_t|D=0}(v_i)`` -- that rank's period-``t`` change, taken
   from the *untreated* change distribution (the distributional DiD step).
5. Counterfactual outcome ``Y_{i,t}(0) = q1_i + q2_i``.

``QTT(tau)`` is then the difference of the ``tau``-quantiles of the observed
treated outcomes and of that counterfactual sample.

Quantile convention: ``numpy``'s default linear interpolation, which is R's
``type = 7`` -- the convention ``qte::panel.qtet`` inherits from
``quantile.ecdf``.  A hand-rolled R replication of the steps above reproduces
``panel.qtet`` to ``max |difference| = 0`` on ``lalonde.psid.panel``, so the
port is anchored on an exact algorithmic match rather than a tolerance.

References
----------
callaway2019quantile
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .qte import QTEResult

__all__ = ["panel_qtet"]


def _ecdf_at(sample: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``F(x) = mean(sample <= x)``, matching R's ``ecdf``."""
    s = np.sort(np.asarray(sample, dtype=float))
    return np.searchsorted(s, np.asarray(points, dtype=float), side="right") / len(s)


def _quantile_at(sample: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Type-7 quantiles -- numpy's default, and R's default."""
    return np.asarray(np.quantile(np.asarray(sample, dtype=float), probs))


def _counterfactual(
    y_t1: np.ndarray,
    y_t1_m1: np.ndarray,
    y_t1_m2: np.ndarray,
    dy_t0: np.ndarray,
) -> np.ndarray:
    """Callaway-Li counterfactual sample for the treated in period ``t``.

    Parameters
    ----------
    y_t1, y_t1_m1, y_t1_m2 : ndarray
        Treated-group outcomes at ``t``, ``t-1``, ``t-2``, aligned by unit.
        ``y_t1`` is unused in the construction (it is the observed arm) and is
        accepted only to keep the caller's alignment explicit.
    dy_t0 : ndarray
        Untreated-group changes ``Y_t - Y_{t-1}``.
    """
    u = _ecdf_at(y_t1_m2, y_t1_m2)  # rank at t-2
    q1 = _quantile_at(y_t1_m1, u)  # -> level at t-1
    dy_treated_lag = y_t1_m1 - y_t1_m2
    v = _ecdf_at(dy_treated_lag, dy_treated_lag)  # rank of the lagged change
    q2 = _quantile_at(dy_t0, v)  # -> untreated change at t
    return np.asarray(q1 + q2)


def _copula_stability_check(
    y_t0: np.ndarray,
    y_t0_m1: np.ndarray,
    y_t0_m2: np.ndarray,
) -> Dict[str, float]:
    """Testable implication of copula stability, on the UNTREATED group.

    Callaway-Li assume ``C(dY_t, Y_{t-1} | D=1) = C(dY_{t-1}, Y_{t-2} | D=1)``.
    That is untestable for the treated at ``t``.  For the untreated, both
    copulas are observed, so a large discrepancy there is evidence the
    stability assumption is implausible in this data.

    Reported as Spearman rank correlations (a copula functional: invariant to
    monotone transformations of the marginals).
    """
    rho_t = float(stats.spearmanr(y_t0 - y_t0_m1, y_t0_m1).statistic)
    rho_tm1 = float(stats.spearmanr(y_t0_m1 - y_t0_m2, y_t0_m2).statistic)
    return {
        "spearman_t": rho_t,
        "spearman_tmin1": rho_tm1,
        "difference": rho_t - rho_tm1,
    }


def _coherence_check(
    arr: Dict[str, np.ndarray],
    cf: np.ndarray,
    did_cf_mean: float,
) -> Dict[str, Any]:
    """Is the rank map actually measure-preserving on this outcome?

    Steps 1-2 of the algorithm map each treated unit's ``t-2`` rank into the
    ``t-1`` distribution.  With a *continuous* outcome that is measure
    preserving, so ``mean(q1) == mean(Y_{t-1}|D=1)`` and hence
    ``mean(cf) == mean(Y_{t-1}|D=1) + mean(dY_t|D=0)``, the distributional-DiD
    counterfactual mean.

    With **mass points** it is not.  On ``qte::lalonde.psid.panel``, 131 of 185
    treated units have ``re74 == 0``; they all receive the same rank and are
    mapped to the same ``t-1`` value, so the counterfactual mean comes out at
    8,786 against a DiD value of 4,023 -- and the QTT curve inherits that
    distortion.  R's ``panel.qtet`` has the identical behaviour (our quantiles
    agree with it to 1e-12); it simply never surfaces the discrepancy because
    it reports a plain mean DiD as the ATT.

    Returns the pieces needed to warn about it.
    """
    scale = max(abs(did_cf_mean), np.std(arr["y_t1_m1"]), 1.0)
    gap = abs(float(np.mean(cf)) - did_cf_mean)
    tie_fracs = [
        float(np.max(np.bincount(np.unique(v, return_inverse=True)[1])) / len(v))
        for v in (arr["y_t1_m2"], arr["y_t1_m1"])
    ]
    return {
        "counterfactual_mean": float(np.mean(cf)),
        "did_counterfactual_mean": did_cf_mean,
        "mean_gap": gap,
        "mean_gap_relative": gap / scale,
        "means_agree": bool(gap / scale < 0.1),
        "tie_fraction_max": float(max(tie_fracs)),
    }


def _prepare(
    data: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    t: Any,
    tmin1: Any,
    tmin2: Any,
) -> Tuple[Dict[str, np.ndarray], int, int]:
    """Balanced treated/untreated arrays for the three periods, unit-aligned."""
    cols = [y, treat, unit, time]
    df = data[cols].dropna()
    periods = [tmin2, tmin1, t]
    df = df[df[time].isin(periods)]

    # Keep only units observed in all three periods.
    counts = df.groupby(unit)[time].nunique()
    keep = counts[counts == 3].index
    dropped = int(df[unit].nunique() - len(keep))
    if dropped:
        warnings.warn(
            f"panel_qtet: dropped {dropped} unit(s) not observed in all three "
            f"periods ({tmin2}, {tmin1}, {t}). Callaway-Li requires a balanced "
            "three-period panel; an unbalanced one would silently change the "
            "copula being estimated.",
            UserWarning,
            stacklevel=3,
        )
    df = df[df[unit].isin(keep)]
    if df.empty:
        raise ValueError(
            f"No unit is observed in all three periods {tmin2}, {tmin1}, {t}."
        )

    # Treatment status is taken from period t (a unit treated at t is 'treated').
    status = df[df[time] == t].set_index(unit)[treat]
    if not np.all(np.isin(status.to_numpy(), (0, 1))):
        raise ValueError("panel_qtet requires a binary (0/1) treatment.")

    out: Dict[str, np.ndarray] = {}
    for grp, name in ((1, "t1"), (0, "t0")):
        units = status[status == grp].index
        sub = df[df[unit].isin(units)]
        for per, tag in ((t, ""), (tmin1, "_m1"), (tmin2, "_m2")):
            block = sub[sub[time] == per].sort_values(unit)
            out[f"y_{name}{tag}"] = block[y].to_numpy(float)
    n1 = len(out["y_t1"])
    n0 = len(out["y_t0"])
    if n1 < 2 or n0 < 2:
        raise ValueError(
            f"panel_qtet needs at least 2 treated and 2 untreated units in a "
            f"balanced panel; found {n1} treated and {n0} untreated."
        )
    return out, n1, n0


def panel_qtet(
    data: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    t: Any,
    tmin1: Any,
    tmin2: Any,
    quantiles: Optional[List[float]] = None,
    alpha: float = 0.05,
    se: str = "bootstrap",
    n_boot: int = 200,
    seed: int = 0,
) -> QTEResult:
    """Callaway & Li (2019) quantile treatment effect on the treated.

    Recovers the counterfactual *distribution* of untreated outcomes for the
    treated group using a copula stability assumption, and returns the
    quantile difference against the observed treated distribution.

    Requires a **balanced three-period panel**: the third period is what
    identifies the copula, and is the reason this needs more data than a mean
    DiD.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    y, treat, unit, time : str
        Column names. ``treat`` is read at period ``t``.
    t, tmin1, tmin2 : Any
        **Values of the ``time`` column** for the post period, the pre period
        and the pre-pre period. Passing them in the wrong order does not
        error -- it silently estimates a different contrast -- so they are
        required positionally rather than defaulted.
    quantiles : list of float, optional
        Defaults to ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    alpha : float, default 0.05
    se : {'bootstrap', 'none'}
        Unit-level bootstrap. There is no analytic option: the estimator
        composes several empirical quantile functions and its influence
        function is not implemented here.
    n_boot : int, default 200
    seed : int

    Returns
    -------
    QTEResult
        ``model_info`` carries ``'counterfactual'`` (the constructed
        counterfactual sample), ``'copula_check'`` (see Notes) and the group
        sizes.

    Notes
    -----
    **Assumptions.** Distributional DiD plus copula stability: among the
    treated, the dependence between the period-``t`` change and the
    period-``t-1`` level equals that between the period-``t-1`` change and
    the period-``t-2`` level. The second is untestable at ``t``, but the same
    stability is checkable on the *untreated* group, where both copulas are
    observed. ``model_info['copula_check']`` reports Spearman rank
    correlations for both; a large ``'difference'`` is evidence against the
    assumption and triggers a warning.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(400):
    ...     a = rng.normal(0, 1)
    ...     treated = i >= 200
    ...     y0 = a + rng.normal(0, 1)
    ...     y1 = a + rng.normal(0, 1)
    ...     y2 = a + rng.normal(0, 1) + (2.0 if treated else 0.0)
    ...     for per, val in ((0, y0), (1, y1), (2, y2)):
    ...         rows.append((i, per, val, int(treated)))
    >>> df = pd.DataFrame(rows, columns=["id", "per", "y", "d"])
    >>> res = sp.panel_qtet(df, y="y", treat="d", unit="id", time="per",
    ...                     t=2, tmin1=1, tmin2=0,
    ...                     quantiles=[0.25, 0.5, 0.75], se="none")
    >>> bool(np.all(np.abs(res.effects - 2.0) < 0.5))  # true QTT = 2.0
    True

    References
    ----------
    callaway2019quantile
    """
    if se not in ("bootstrap", "none"):
        raise ValueError(f"se must be 'bootstrap' or 'none', got {se!r}")
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    taus = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("quantiles must lie strictly inside (0, 1).")
    if len({t, tmin1, tmin2}) != 3:
        raise ValueError(
            f"t, tmin1 and tmin2 must be three distinct periods, got "
            f"{t!r}, {tmin1!r}, {tmin2!r}."
        )

    arr, n1, n0 = _prepare(data, y, treat, unit, time, t, tmin1, tmin2)

    cf = _counterfactual(
        arr["y_t1"], arr["y_t1_m1"], arr["y_t1_m2"], arr["y_t0"] - arr["y_t0_m1"]
    )
    effects = _quantile_at(arr["y_t1"], taus) - _quantile_at(cf, taus)

    # The headline ATT is the plain mean DiD, matching qte::panel.qtet. It is
    # reported rather than mean(Y_t) - mean(cf) because the mean does not need
    # copula stability: distributional DiD alone pins it. See the coherence
    # check below for why the two can diverge.
    did_cf_mean = float(
        np.mean(arr["y_t1_m1"]) + (np.mean(arr["y_t0"]) - np.mean(arr["y_t0_m1"]))
    )
    att = float(
        np.mean(arr["y_t1"])
        - np.mean(arr["y_t1_m1"])
        - (np.mean(arr["y_t0"]) - np.mean(arr["y_t0_m1"]))
    )

    coherence = _coherence_check(arr, cf, did_cf_mean)
    if coherence["tie_fraction_max"] > 0.2 or not coherence["means_agree"]:
        warnings.warn(
            "panel_qtet: the copula construction looks distorted on this "
            f"outcome. The rank map ties {coherence['tie_fraction_max']:.0%} of "
            "treated units to a single value (mass points in the outcome), and "
            f"the counterfactual mean is {np.mean(cf):,.1f} against the "
            f"distributional-DiD value of {did_cf_mean:,.1f}. Those two should "
            "agree -- the rank map is meant to be measure-preserving -- so a "
            "gap means the estimated QTT curve is unreliable even though the "
            "reported ATT (a plain mean DiD) is not. Callaway-Li assumes a "
            "continuous outcome; consider sp.cic bounds for discrete or "
            "mass-point outcomes.",
            UserWarning,
            stacklevel=2,
        )

    copula = _copula_stability_check(arr["y_t0"], arr["y_t0_m1"], arr["y_t0_m2"])
    if abs(copula["difference"]) > 0.2:
        warnings.warn(
            "panel_qtet: copula stability looks doubtful. On the UNTREATED "
            f"group the Spearman correlation between the change and the "
            f"lagged level is {copula['spearman_t']:.3f} at t but "
            f"{copula['spearman_tmin1']:.3f} at t-1 "
            f"(difference {copula['difference']:+.3f}). Callaway-Li assume "
            "this dependence is stable for the TREATED; a large shift among "
            "the untreated is evidence against it.",
            UserWarning,
            stacklevel=2,
        )

    # ---- standard errors: unit-level bootstrap ------------------------ #
    if se == "none":
        se_arr = np.full(len(taus), np.nan)
        ci_lo = np.full(len(taus), np.nan)
        ci_hi = np.full(len(taus), np.nan)
    else:
        rng = np.random.default_rng(seed)
        boot = np.full((n_boot, len(taus)), np.nan)
        for b in range(n_boot):
            i1 = rng.integers(0, n1, size=n1)
            i0 = rng.integers(0, n0, size=n0)
            try:
                cf_b = _counterfactual(
                    arr["y_t1"][i1],
                    arr["y_t1_m1"][i1],
                    arr["y_t1_m2"][i1],
                    arr["y_t0"][i0] - arr["y_t0_m1"][i0],
                )
                boot[b] = _quantile_at(arr["y_t1"][i1], taus) - _quantile_at(cf_b, taus)
            except Exception:  # noqa: BLE001 - counted below
                continue
        n_ok = np.isfinite(boot).sum(axis=0)
        se_arr = np.where(n_ok >= 2, np.nanstd(boot, axis=0, ddof=1), np.nan)
        z = float(stats.norm.ppf(1 - alpha / 2))
        ci_lo, ci_hi = effects - z * se_arr, effects + z * se_arr
        if (n_ok < n_boot).any():
            warnings.warn(
                f"panel_qtet: {int((n_ok < n_boot).sum())}/{len(taus)} "
                "quantile(s) had bootstrap replications fail; SEs use the "
                "remainder and are NaN where fewer than two survived.",
                RuntimeWarning,
                stacklevel=2,
            )

    result = QTEResult(
        quantiles=taus,
        effects=effects,
        se=se_arr,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=att,
        method="Panel QTT (Callaway & Li, 2019)",
        n_obs=n1 + n0,
        alpha=alpha,
        model_info={
            "n_treated": n1,
            "n_untreated": n0,
            "periods": {"t": t, "tmin1": tmin1, "tmin2": tmin2},
            "counterfactual": cf,
            "copula_check": copula,
            "coherence_check": coherence,
            "se_method": se,
            "n_boot": n_boot if se == "bootstrap" else None,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.qte.panel_qtet",
            params={
                "y": y,
                "treat": treat,
                "unit": unit,
                "time": time,
                "t": t,
                "tmin1": tmin1,
                "tmin2": tmin2,
                "quantiles": list(taus),
                "alpha": alpha,
                "se": se,
                "n_boot": n_boot,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover - provenance is best-effort metadata
        pass
    return result
