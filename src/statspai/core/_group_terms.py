"""Interacted grouping terms for ``absorb=`` / ``cluster=`` arguments.

``fixest`` writes one group per observed level combination as ``a^b``;
Stata writes the same absorbed effect as ``a#b`` (``reghdfe``/``ppmlhdfe``
``absorb(ind#year)``) and needs ``egen group()`` for the cluster
equivalent.  Estimators that take column names for fixed effects or
clusters call :func:`resolve_group_terms` once, up front, so a user never
has to build ``groupby([...]).ngroup()`` columns by hand.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import pandas as pd

_SEP = re.compile(r"\s*(?:\^|#)\s*")


def split_group_term(term: str) -> List[str]:
    """``"ind^year"`` / ``"i.ind#i.year"`` -> ``["ind", "year"]``."""
    parts = [p.strip() for p in _SEP.split(str(term).strip()) if p.strip()]
    return [p[2:] if p.startswith("i.") else p for p in parts]


def is_group_term(term: str) -> bool:
    return len(split_group_term(term)) > 1


def group_term_atoms(terms: Sequence[str]) -> List[str]:
    """Every underlying column named by ``terms`` (in order, de-duplicated)."""
    out: List[str] = []
    for t in terms:
        out.extend(split_group_term(t))
    return list(dict.fromkeys(out))


def resolve_group_terms(
    data: pd.DataFrame, terms: Sequence[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """Materialise interacted terms as integer group columns.

    Returns ``(data, names)``: ``names`` lists one column per term, the
    plain name for a single column and ``"a^b"`` for an interaction (added
    to a copy of ``data``; rows with a missing atom get a missing group).
    ``data`` is returned unchanged when no term is interacted.
    """
    names: List[str] = []
    new_cols = {}
    for t in terms:
        atoms = split_group_term(t)
        if len(atoms) == 1:
            names.append(atoms[0])
            continue
        missing = [a for a in atoms if a not in data.columns]
        if missing:
            raise KeyError(f"interacted term {t!r}: column(s) {missing} not in data")
        label = "^".join(atoms)
        codes = data.groupby(atoms, dropna=False, sort=True).ngroup()
        codes = codes.where(data[atoms].notna().all(axis=1))
        new_cols[label] = codes
        names.append(label)
    if new_cols:
        data = data.assign(**new_cols)
    return data, names
