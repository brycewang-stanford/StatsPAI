"""Detect non-absorbing (reverting) treatment in a panel.

Most of the modern DiD family — Callaway-Sant'Anna, Sun-Abraham,
Borusyak-Jaravel-Spiess, Wooldridge ETWFE, stacked DiD — is built on a
*cohort* variable ``g``: the period a unit is first treated. That
representation is lossless only when treatment is **absorbing**, i.e. once on
it never turns off. When treatment reverts, collapsing the time-varying
indicator to "first treated at g" silently discards the reversal, and the
estimator then treats post-reversal periods as though the unit were still
treated. The estimate is biased toward zero with no error and no warning.

The failure is easy to reproduce: on a 150-unit panel where a third of units
switch on at t=4 and back off at t=7, ``sp.callaway_santanna`` returns 0.706
against a true ATT of 1.5 — a 53% error, silently.

Because the cohort-based estimators never see the time-varying indicator,
they cannot detect this themselves. This module gives the check a home so
routing layers (``sp.recommend``) and users can run it on the raw panel
*before* choosing an estimator.

Estimators that genuinely handle reversal: ``sp.did_multiplegt_dyn``
(de Chaisemartin & D'Haultfœuille) and ``sp.lp_did`` (Dube, Girardi, Jordà &
Taylor), both of which take the time-varying indicator directly.

References
----------
de Chaisemartin, C. and D'Haultfœuille, X. (2024). "Difference-in-Differences
Estimators of Intertemporal Treatment Effects." *Review of Economics and
Statistics*. [@dechaisemartin2024difference]
"""

from __future__ import annotations

from typing import List, NamedTuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["AbsorbingCheck", "check_absorbing"]

#: Reverting units listed individually before the report truncates.
_MAX_LISTED = 20


class AbsorbingCheck(NamedTuple):
    """Whether treatment is absorbing, and where it is not."""

    is_absorbing: bool
    n_units: int
    n_reverting_units: int
    n_reversals: int
    share_reverting: float
    reverting_units: List
    first_reversal_period: object
    treatment_is_binary: bool

    def summary(self) -> str:
        """One-line reading suitable for a warning or a report row."""
        if self.is_absorbing:
            return (
                f"treatment is absorbing across all {self.n_units} units "
                "(no reversals)"
            )
        return (
            f"treatment is NOT absorbing: {self.n_reverting_units} of "
            f"{self.n_units} units ({self.share_reverting:.1%}) turn "
            f"treatment off again, {self.n_reversals} reversal(s) in total, "
            f"first at period {self.first_reversal_period}. Cohort-based "
            "estimators (callaway_santanna, sun_abraham, did_imputation, "
            "etwfe, stacked_did) collapse this to 'first treated at g' and "
            "silently discard the reversal; use did_multiplegt_dyn or "
            "lp_did instead."
        )


def check_absorbing(
    data: pd.DataFrame,
    unit: str,
    time: str,
    treatment: str,
    strict: bool = False,
) -> AbsorbingCheck:
    """Check whether a time-varying treatment indicator is absorbing.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel.
    unit, time, treatment : str
        Column names. ``treatment`` must be the **time-varying** 0/1
        indicator, not a cohort / first-treatment column — a cohort column
        cannot express reversal, so checking one is meaningless.
    strict : bool, default False
        Raise :class:`MethodIncompatibility` instead of returning when
        treatment reverts. Useful as a guard in front of a cohort-based
        estimator.

    Returns
    -------
    AbsorbingCheck
        ``is_absorbing`` plus the location and extent of any reversals.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rows = []
    >>> for u in range(6):
    ...     for t in range(1, 7):
    ...         on = 1 if (u < 3 and 3 <= t < 5) else 0   # units 0-2 revert
    ...         rows.append({"i": u, "t": t, "d": on})
    >>> chk = sp.check_absorbing(pd.DataFrame(rows), "i", "t", "d")
    >>> chk.is_absorbing
    False
    >>> chk.n_reverting_units
    3

    References
    ----------
    dechaisemartin2024difference
    """
    for col in (unit, time, treatment):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"column {col!r} not found in data.",
                recovery_hint="Pass the long-panel unit / time / treatment "
                "column names.",
                diagnostics={"missing": col, "columns": list(data.columns)[:20]},
            )

    df = data[[unit, time, treatment]].dropna()
    if df.empty:
        raise DataInsufficient(
            "no complete unit / time / treatment rows to check.",
            recovery_hint="Check for missing values in the treatment column.",
            diagnostics={"n_rows": int(len(data))},
        )

    d = pd.to_numeric(df[treatment], errors="coerce")
    if d.isna().any():
        raise MethodIncompatibility(
            f"treatment column {treatment!r} is not numeric.",
            recovery_hint="Pass a 0/1 time-varying treatment indicator.",
            diagnostics={"n_non_numeric": int(d.isna().sum())},
        )
    df = df.assign(_d=d.astype(float))
    binary = bool(np.isin(df["_d"].unique(), [0.0, 1.0]).all())

    df = df.sort_values([unit, time])
    # A reversal is any within-unit step down in treatment status.
    step = df.groupby(unit, sort=False)["_d"].diff()
    reverted = step < 0

    n_reversals = int(reverted.sum())
    reverting = df.loc[reverted, unit].unique().tolist()
    n_units = int(df[unit].nunique())
    first_period = None
    if n_reversals:
        first_period = df.loc[reverted, time].min()

    check = AbsorbingCheck(
        is_absorbing=n_reversals == 0,
        n_units=n_units,
        n_reverting_units=len(reverting),
        n_reversals=n_reversals,
        share_reverting=(len(reverting) / n_units) if n_units else 0.0,
        reverting_units=reverting[:_MAX_LISTED],
        first_reversal_period=first_period,
        treatment_is_binary=binary,
    )

    if strict and not check.is_absorbing:
        raise MethodIncompatibility(
            check.summary(),
            recovery_hint="Use sp.did_multiplegt_dyn or sp.lp_did, which "
            "accept a time-varying treatment and handle reversal.",
            diagnostics={
                "n_reverting_units": check.n_reverting_units,
                "n_units": check.n_units,
                "n_reversals": check.n_reversals,
                "first_reversal_period": check.first_reversal_period,
            },
        )
    return check
