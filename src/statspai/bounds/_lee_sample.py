"""Estimation sample of ``sp.lee_bounds``, defined as Stata ``leebounds``.

``leebounds`` (Tauchmann, v1.5) drops a row from every part of the
computation (retention rates, trimming, variance) when its treatment or
selection indicator is missing, or when it is selected but its outcome is
missing. Before 2026-10-19 ``sp.lee_bounds`` kept such rows: a missing
outcome among the selected turned both bounds into NaN, a missing selection
indicator raised an unrelated ``ValueError``, and indicators coded other than
0/1 crashed with ``IndexError``.
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

#: Column holding ``selection=None``'s indicator (outcome observed).
SELECTED_COL = "__lee_selected__"


def _require_01(values: pd.Series, name: str, role: str) -> None:
    levels = set(np.unique(values.to_numpy(dtype=float)))
    if not levels <= {0.0, 1.0}:
        shown = sorted(levels)[:6]
        raise MethodIncompatibility(
            f"lee_bounds: {role} indicator {name!r} must be coded 0/1, "
            f"found values {shown}.",
            recovery_hint=(
                f"Recode {name!r} to 0/1 (1 = {'treated' if role == 'treatment' else 'outcome observed'})."
            ),
        )


def lee_estimation_sample(
    data: pd.DataFrame, y: str, treat: str, selection: Optional[str]
) -> Tuple[pd.DataFrame, str]:
    """Rows ``leebounds`` uses: indicators observed, outcome observed if selected.

    ``selection=None`` is ``leebounds`` without ``select()``: a unit is
    selected when its outcome is observed. Returns the sample and the name
    of its selection column.
    """
    df = data.copy()
    if selection is None:
        observed = df[y].notna()
        if observed.all():
            raise DataInsufficient(
                f"lee_bounds: no value of {y!r} is missing, so there is no "
                "selection to bound (Stata leebounds: 'all observations "
                "selected').",
                recovery_hint="Pass selection= with the retention indicator.",
            )
        df[SELECTED_COL] = observed.astype(int)
        selection = SELECTED_COL
    n0 = len(df)
    df = df[df[treat].notna() & df[selection].notna()]
    n_ind = n0 - len(df)
    _require_01(df[treat], treat, "treatment")
    _require_01(df[selection], selection, "selection")
    sel_no_y = (df[selection].to_numpy(dtype=float) == 1) & df[y].isna().to_numpy()
    n_y = int(sel_no_y.sum())
    if n_y:
        df = df[~sel_no_y]
    if n_ind or n_y:
        parts = []
        if n_ind:
            parts.append(f"{n_ind} with a missing treatment or selection indicator")
        if n_y:
            parts.append(f"{n_y} selected but with a missing outcome")
        warnings.warn(
            "lee_bounds: dropped " + " and ".join(parts) + " (as Stata "
            "leebounds does).",
            UserWarning,
            stacklevel=3,
        )
    return df, selection
