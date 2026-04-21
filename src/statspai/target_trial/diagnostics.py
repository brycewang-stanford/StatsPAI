"""Diagnostic checks for target trial emulation."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ImmortalTimeDiagnostic:
    n_total: int
    n_flagged: int
    fraction_flagged: float
    flagged_ids: list
    explanation: str


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
