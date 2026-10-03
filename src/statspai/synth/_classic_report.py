"""Placebo and leave-one-out reports of the classic synthetic control fit.

``sp.synth(method='classic')`` fits the treated unit and, with
``placebo=True``, every control unit as a pretend treated one. The tables
here are read off those fits, or need a handful of further fits:

* the unit table of pre- and post-treatment mean squared prediction errors
  and the period-by-period placebo p-values, optionally leaving out pretend
  units the method fits badly (``placebo_cutoff``);
* a pretend treatment date (``placebo_time``);
* the range of the synthetic path when one donor that carries weight is
  dropped (``loo``).

They are the reports of the Stata command ``synth2`` and use the names the
regression control method (``method='rcm'``) already uses.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "add_fit_statistics",
    "add_placebo_tables",
    "placebo_time_table",
    "loo_table",
]

#: A donor "carries weight" when its weight is 0.001 after rounding to three
#: decimals, the precision Stata's synth stores weights at.
LOO_MIN_WEIGHT = 5e-4


def _gaps(info: Dict[str, Any]) -> pd.DataFrame:
    return info["gap_table"]


def add_fit_statistics(info: Dict[str, Any]) -> None:
    """Pre-treatment R-squared of the synthetic path (1 - SSE / SST)."""
    gaps = _gaps(info)
    pre = ~gaps["post_treatment"].to_numpy(dtype=bool)
    actual = gaps["treated"].to_numpy(dtype=float)[pre]
    gap = gaps["gap"].to_numpy(dtype=float)[pre]
    sst = float(((actual - actual.mean()) ** 2).sum())
    info["pre_r2"] = float(1.0 - (gap @ gap) / sst) if sst > 0 else float("nan")


def add_placebo_tables(info: Dict[str, Any], cutoff: Optional[float]) -> None:
    """Unit table and period-by-period p-values from the in-space placebos.

    The p-values count the treated unit among the units compared, so the
    smallest attainable value is one over their number.
    """
    if "placebo_gaps" not in info:
        raise MethodIncompatibility(
            "sp.synth(method='classic'): placebo_cutoff needs the in-space "
            "placebo fits.",
            recovery_hint="Pass placebo=True.",
        )
    if cutoff is not None and not (np.isfinite(cutoff) and cutoff >= 1.0):
        raise MethodIncompatibility(
            "sp.synth(method='classic'): placebo_cutoff must be at least 1 "
            "(a multiple of the treated unit's pre-treatment MSPE).",
            diagnostics={"placebo_cutoff": cutoff},
        )
    gaps = _gaps(info)
    post = gaps["post_treatment"].to_numpy(dtype=bool)
    tau = gaps["gap"].to_numpy(dtype=float)
    cloud = np.asarray(info["placebo_gaps"], dtype=float)  # periods x units
    units = list(info["placebo_units"])
    treated = info["treated_unit"]

    def row(unit: Any, gap: np.ndarray) -> Dict[str, Any]:
        return {
            "unit": unit,
            "pre_mspe": float(np.mean(gap[~post] ** 2)),
            "post_mspe": float(np.mean(gap[post] ** 2)),
        }

    table = pd.DataFrame(
        [row(treated, tau)] + [row(u, cloud[:, j]) for j, u in enumerate(units)]
    ).set_index("unit")
    table["ratio"] = table["post_mspe"] / table["pre_mspe"]
    own_pre = float(table["pre_mspe"].iloc[0])
    table["pre_mspe_relative"] = table["pre_mspe"] / own_pre
    relative = table["pre_mspe_relative"].to_numpy()[1:]
    keep = np.ones(len(units), dtype=bool) if cutoff is None else relative <= cutoff
    ratios = table["ratio"].to_numpy()
    own = ratios[0]
    kept_ratios = np.concatenate([[own], ratios[1:][keep]])

    compared = np.column_stack(
        [tau[post]] + [cloud[post, j] for j in np.flatnonzero(keep)]
    )
    effect = tau[post][:, None]
    effects = pd.DataFrame(
        {
            "time": gaps["time"].to_numpy()[post],
            "effect": tau[post],
            "p_two_sided": (np.abs(compared) >= np.abs(effect)).mean(axis=1),
            "p_right": (compared >= effect).mean(axis=1),
            "p_left": (compared <= effect).mean(axis=1),
        }
    )
    info.update(
        placebo_table=table,
        placebo_pvalue=float(np.mean(ratios >= own)),
        placebo_pvalue_cutoff=float(np.mean(kept_ratios >= own)),
        placebo_excluded=[u for u, k in zip(units, keep) if not k],
        placebo_cutoff=cutoff,
        placebo_effects=effects,
    )


def placebo_time_table(
    info: Dict[str, Any], placebo_time: Any, refit: Callable[..., Any]
) -> None:
    """Refit as if the treatment had started at ``placebo_time``."""
    times = np.asarray(info["times"])
    if not (
        placebo_time in set(times.tolist()) and placebo_time < info["treatment_time"]
    ):
        raise MethodIncompatibility(
            "sp.synth(method='classic'): placebo_time must be a period of the "
            "data before treatment_time.",
            diagnostics={"placebo_time": placebo_time},
        )
    if int(np.sum(times < placebo_time)) < 2:
        raise DataInsufficient(
            "sp.synth(method='classic'): placebo_time leaves fewer than two "
            "periods to fit.",
            recovery_hint="Choose a later placebo_time.",
        )
    run = refit(treatment_time=placebo_time).model_info
    gaps = run["gap_table"]
    shown = gaps["post_treatment"].to_numpy(dtype=bool)
    info["placebo_time"] = pd.DataFrame(
        {
            "time": gaps["time"].to_numpy()[shown],
            "actual": gaps["treated"].to_numpy()[shown],
            "synthetic": gaps["synthetic"].to_numpy()[shown],
            "effect": gaps["gap"].to_numpy()[shown],
        }
    ).reset_index(drop=True)
    info["placebo_time_weights"] = run["weights"]


def loo_table(info: Dict[str, Any], refit: Callable[..., Any]) -> None:
    """Range of the synthetic path when one weighted donor is dropped."""
    weights = info["weights"]
    dropped: List[Any] = weights.loc[
        weights["weight"] >= LOO_MIN_WEIGHT, "unit"
    ].tolist()
    if not dropped:
        raise DataInsufficient(
            "sp.synth(method='classic'): no donor carries weight, so there is "
            "nothing to leave out.",
        )
    if info["n_donors"] < 3:
        raise DataInsufficient(
            "sp.synth(method='classic'): loo needs at least three donors.",
        )
    gaps = _gaps(info)
    paths, refit_weights = {}, {}
    for unit in dropped:
        run = refit(drop_unit=unit).model_info
        paths[unit] = np.asarray(run["Y_synth"], dtype=float)
        refit_weights[unit] = run["weights"].set_index("unit")["weight"]
    wide = pd.DataFrame(paths, index=gaps["time"].to_numpy())
    actual = gaps["treated"].to_numpy(dtype=float)
    low, high = wide.min(axis=1).to_numpy(), wide.max(axis=1).to_numpy()
    info["loo"] = pd.DataFrame(
        {
            "time": gaps["time"].to_numpy(),
            "actual": actual,
            "synthetic": gaps["synthetic"].to_numpy(),
            "synthetic_min": low,
            "synthetic_max": high,
            "effect": gaps["gap"].to_numpy(),
            "effect_min": actual - high,
            "effect_max": actual - low,
            "post": gaps["post_treatment"].to_numpy(dtype=bool),
        }
    )
    info["loo_units"] = dropped
    info["loo_paths"] = wide
    info["loo_weights"] = pd.DataFrame(refit_weights).fillna(0.0)
