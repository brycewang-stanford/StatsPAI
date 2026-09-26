"""Diagnostic checks for target trial emulation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .._aliases import accepts_aliases


@dataclass
class ImmortalTimeDiagnostic:
    n_total: int
    n_flagged: int
    fraction_flagged: float
    flagged_ids: list
    explanation: str


@accepts_aliases(time="time_col")
def immortal_time_check(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    treatment_start_col: str,
    eligibility_time_col: str,
) -> ImmortalTimeDiagnostic:
    """Flag subjects whose follow-up begins *before* treatment
    initiation — the textbook recipe for immortal time bias.

    Parameters
    ----------
    data : pd.DataFrame
    id_col : str
    time_col : str
        Follow-up time column.
    treatment_start_col : str
        Time of treatment initiation (NaN / ``-inf`` if never treated).
    eligibility_time_col : str
        The protocol's defined time zero.

    Returns
    -------
    ImmortalTimeDiagnostic

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> data = pd.DataFrame({
    ...     "id": [1, 2, 3, 4],
    ...     "fu_time": [12.0, 8.0, 24.0, 6.0],
    ...     "tx_start": [3.0, 0.0, 5.0, 1.0],
    ...     "elig_time": [0.0, 0.0, 0.0, 2.0],  # id 4: treated before eligible
    ... })
    >>> diag = sp.immortal_time_check(
    ...     data, id_col="id", time_col="fu_time",
    ...     treatment_start_col="tx_start", eligibility_time_col="elig_time")
    >>> diag.n_flagged
    1
    >>> diag.flagged_ids
    [4]
    """
    df = data.copy()
    df["_elig"] = pd.to_numeric(df[eligibility_time_col], errors="coerce")
    df["_tx"] = pd.to_numeric(df[treatment_start_col], errors="coerce")
    df["_flag"] = df["_tx"] < df["_elig"]
    flagged = df.loc[df["_flag"].fillna(False), id_col].unique().tolist()
    n = len(df)
    k = len(flagged)
    return ImmortalTimeDiagnostic(
        n_total=int(n),
        n_flagged=int(k),
        fraction_flagged=float(k / n) if n else 0.0,
        flagged_ids=flagged,
        explanation=(
            "Subjects listed here had treatment_start < eligibility_time. "
            "Including their pre-treatment time as follow-up induces "
            "immortal time bias. Either realign time zero to treatment "
            "initiation or clone-censor-weight before analysis."
        ),
    )
