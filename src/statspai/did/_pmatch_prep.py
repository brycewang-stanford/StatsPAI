"""Panel preparation for ``sp.panel_match``.

Turns a long data frame into balanced unit x period arrays (missing
unit-periods become rows of missing values), parses the refinement
covariate specifications (``'x'``, ``'lag(x, k)'``, ``'lag(x, a:b)'``) and
builds the refinement design: missing cells are set to zero and, for every
column that has missing cells somewhere in the balanced panel, a 0/1
"missing" indicator column is appended.

Clean-room module: written from Imai, Kim and Wang (2023) and the
PanelMatch documentation; no PanelMatch source was consulted.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility

_LAG_RE = re.compile(r"^\s*lag\(\s*([^,()\s]+)\s*,\s*(\d+)\s*(?::\s*(\d+)\s*)?\)\s*$")


@dataclass
class PMPanel:
    """Balanced panel: every array is (n_units, n_periods)."""

    units: np.ndarray
    times: np.ndarray
    D: np.ndarray
    Y: np.ndarray
    raw: Dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return self.D.shape[0]

    @property
    def n_periods(self) -> int:
        return self.D.shape[1]


@dataclass(frozen=True)
class CovSpec:
    """One covariate term: a base column and the lags it enters with."""

    text: str
    base: str
    lags: Tuple[int, ...]  # (0,) for the contemporaneous value


def parse_covariates(
    specs: Optional[Sequence[str]], columns: Sequence[str]
) -> List[CovSpec]:
    """Covariate terms: a column of the data, or ``lag(x, k)`` /
    ``lag(x, a:b)`` of one."""
    cols = set(columns)
    out: List[CovSpec] = []
    for s in specs or []:
        if not isinstance(s, str):
            raise MethodIncompatibility(
                f"covariate {s!r} is not a string",
                recovery_hint="Pass column names or 'lag(x, k)' / 'lag(x, a:b)'.",
            )
        m = _LAG_RE.match(s)
        if s in cols:
            out.append(CovSpec(s, s, (0,)))
        elif m:
            a = int(m.group(2))
            b = int(m.group(3)) if m.group(3) is not None else a
            if b < a:
                raise MethodIncompatibility(
                    f"lag window in {s!r} is decreasing",
                    recovery_hint="Write the window as 'lag(x, a:b)' with a <= b.",
                )
            out.append(CovSpec(s, m.group(1), tuple(range(a, b + 1))))
        else:
            raise MethodIncompatibility(
                f"covariate term {s!r} is neither a column nor lag(x, k)",
                recovery_hint=(
                    "Use a column name or 'lag(x, k)' / 'lag(x, a:b)'; build "
                    "transformed columns in the data frame first."
                ),
            )
    return out


def _time_codes(values: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Map the time column to consecutive integer codes 0..T-1."""
    uniq = pd.Index(pd.unique(values.dropna())).sort_values()
    consecutive = False
    if pd.api.types.is_numeric_dtype(uniq.dtype) and len(uniq) > 0:
        arr = uniq.to_numpy(dtype=float)
        consecutive = bool(np.all(np.mod(arr, 1) == 0) and np.all(np.diff(arr) == 1))
    if not consecutive:
        warnings.warn(
            "panel_match: the time variable is not consecutive integers; "
            "observed periods are ranked 1..T (a gap in calendar time is "
            "treated as adjacent periods).",
            UserWarning,
            stacklevel=4,
        )
    codes = uniq.get_indexer(values)
    return codes, uniq.to_numpy()


def build_panel(
    data: pd.DataFrame,
    y: str,
    treat: str,
    id: str,
    time: str,
    variables: Sequence[str],
) -> PMPanel:
    needed = [y, treat, id, time] + [v for v in variables]
    missing = sorted({c for c in needed if c not in data.columns})
    if missing:
        raise ColumnNotFound.from_missing(
            missing, list(data.columns), context="panel_match"
        )
    df = data[list(dict.fromkeys(needed))].copy()
    if df[id].isna().any() or df[time].isna().any():
        raise MethodIncompatibility(
            "id or time has missing values",
            recovery_hint="Drop rows with a missing unit or period identifier.",
        )
    if df.duplicated([id, time]).any():
        raise MethodIncompatibility(
            "unit-period pairs are repeated",
            recovery_hint="Keep one row per unit and period.",
        )
    d = pd.to_numeric(df[treat], errors="coerce")
    bad = d.notna() & ~d.isin([0, 1])
    if bad.any() or (df[treat].notna() & d.isna()).any():
        raise MethodIncompatibility(
            "treatment must be coded 0/1 (missing allowed)",
            recovery_hint="Recode the treatment to a 0/1 indicator.",
        )
    tcode, tlab = _time_codes(df[time])
    ucode, ulab = pd.factorize(df[id], sort=True)
    N, T = len(ulab), len(tlab)

    def grid(values: np.ndarray) -> np.ndarray:
        out = np.full((N, T), np.nan)
        out[ucode, tcode] = values
        return out

    raw: Dict[str, np.ndarray] = {}
    for v in dict.fromkeys(variables):
        col = df[v]
        if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
            vals = col.to_numpy(dtype=float)
        else:
            codes, _ = pd.factorize(col)
            vals = np.where(codes < 0, np.nan, codes).astype(float)
        raw[v] = grid(vals)
    raw[y] = grid(pd.to_numeric(df[y], errors="coerce").to_numpy(dtype=float))
    return PMPanel(
        units=np.asarray(ulab),
        times=np.asarray(tlab),
        D=grid(d.to_numpy(dtype=float)),
        Y=raw[y],
        raw=raw,
    )


def lagged(a: np.ndarray, k: int) -> np.ndarray:
    """Within-unit lag by ``k`` periods on a balanced (N, T) array."""
    if k == 0:
        return a
    out = np.full_like(a, np.nan)
    if k < a.shape[1]:
        out[:, k:] = a[:, :-k]
    return out


def refinement_design(
    panel: PMPanel, specs: Sequence[CovSpec]
) -> Tuple[np.ndarray, List[str]]:
    """(N, T, K) refinement covariates with zero-filled missing cells and
    missingness indicators for the columns that have any missing cell."""
    cols: List[np.ndarray] = []
    names: List[str] = []
    for s in specs:
        base = panel.raw[s.base]
        for k in s.lags:
            cols.append(lagged(base, k))
            names.append(s.base if k == 0 and s.text == s.base else f"{s.base}_l{k}")
    if not cols:
        return np.zeros(panel.D.shape + (0,)), []
    indicators, ind_names = [], []
    for c, n in zip(cols, names):
        miss = np.isnan(c)
        if miss.any() and not miss.all():
            indicators.append(miss.astype(float))
            ind_names.append(f"{n}_missing")
    filled = [np.where(np.isnan(c), 0.0, c) for c in cols]
    Z = np.stack(filled + indicators, axis=2)
    return Z, names + ind_names


def check_lag(lag: int, panel: PMPanel) -> None:
    if lag + 1 > panel.n_periods:
        raise DataInsufficient(
            f"lag={lag} needs at least {lag + 1} periods; the panel has "
            f"{panel.n_periods}",
            recovery_hint="Use a shorter treatment history (smaller lag).",
        )
