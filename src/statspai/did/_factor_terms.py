"""Stata factor-variable terms (``i.year#i.nodecity``) for DID covariates.

``sp.etwfe`` / ``sp.jwdid`` take covariate *columns*.  A Stata user writes
``exovar(i.year#i.nodecity)`` and would otherwise have to build the 20
year-by-node-city dummies by hand.  :func:`expand_factor_terms` turns such
terms into columns of a copy of the data, following Stata's notation:

* ``i.v`` -- one indicator per level of ``v`` except the base (lowest)
  level, Stata's default ``ib(first)``; ``c.v`` -- ``v`` itself; a bare
  name inside ``#`` is a factor.
* ``a#b`` -- the products of the components' columns, over **every** level
  of each factor: without its main effects in the term Stata keeps all
  cells (``c.x#i.f`` is one slope per level of ``f``) and omits whichever
  are collinear with the rest of the model -- the estimator does the same
  here.
* ``a##b`` -- the full factorial ``a + b + a#b``; with the main effects in,
  the interaction cells of the base levels are omitted, as in Stata.
* A row with a missing value in any component is missing in the product,
  so the estimator drops it -- Stata's listwise deletion.

A term that is already a column name is left alone, so existing calls
keep their meaning.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

__all__ = ["expand_factor_terms", "factor_xvar", "is_factor_term"]


def is_factor_term(term: Any, columns: Any) -> bool:
    """True when ``term`` uses factor notation rather than naming a column."""
    if not isinstance(term, str) or term in columns:
        return False
    return "#" in term or term.startswith(("i.", "c."))


def _component(
    data: pd.DataFrame,
    token: str,
    in_interaction: bool,
    term: str,
    context: str,
    drop_base: bool,
) -> List[Tuple[str, pd.Series]]:
    """Columns of one component: ``(label, float series)`` pairs."""
    token = token.strip()
    if token.startswith("c."):
        kind, var = "c", token[2:]
    elif token.startswith("i."):
        kind, var = "i", token[2:]
    elif in_interaction:
        kind, var = "i", token  # Stata: a bare name in `#` is a factor
    else:
        kind, var = "", token
    if var not in data.columns:
        raise MethodIncompatibility(
            f"{context}: {term!r} refers to {var!r}, which is not a column.",
            recovery_hint="Name existing columns in i.<var> / c.<var> terms.",
            diagnostics={"term": term, "variable": var},
        )
    col = data[var]
    if kind == "c":
        return [(f"c.{var}", pd.to_numeric(col, errors="coerce").astype(float))]
    if kind == "":
        raise MethodIncompatibility(
            f"{context}: {term!r} is neither a column nor a factor term.",
            recovery_hint="Use i.<var>, c.<var>, a#b or a##b.",
            diagnostics={"term": term},
        )
    levels = sorted(pd.unique(col.dropna()))
    if len(levels) < 2:
        raise MethodIncompatibility(
            f"{context}: i.{var} in {term!r} has fewer than two levels.",
            recovery_hint="Drop the term; a constant factor adds no column.",
            diagnostics={"term": term, "levels": [str(v) for v in levels]},
        )
    missing = col.isna().to_numpy()
    out = []
    for lev in levels[1:] if drop_base else levels:
        ind = (col == lev).to_numpy(dtype=float)
        ind[missing] = np.nan
        label = f"{_level_label(lev)}.{var}"
        out.append((label, pd.Series(ind, index=data.index)))
    return out


def _level_label(level: Any) -> str:
    if isinstance(level, (float, np.floating)) and float(level).is_integer():
        return str(int(level))
    return str(level)


def _interaction(
    data: pd.DataFrame,
    tokens: Sequence[str],
    term: str,
    context: str,
    drop_base: bool,
) -> List[Tuple[str, pd.Series]]:
    parts = [
        _component(data, t, len(tokens) > 1, term, context, drop_base) for t in tokens
    ]
    out = []
    for combo in product(*parts):
        label = "#".join(lab for lab, _ in combo)
        values = combo[0][1]
        for _, s in combo[1:]:
            values = values * s
        out.append((label, values))
    return out


def _term_columns(
    data: pd.DataFrame, term: str, context: str
) -> List[Tuple[str, pd.Series]]:
    if "##" in term:
        tokens = [t for t in term.split("##") if t.strip()]
        cols: List[Tuple[str, pd.Series]] = []
        # Full factorial: every non-empty subset of the components, in order.
        n = len(tokens)
        for mask in range(1, 2**n):
            subset = [tokens[j] for j in range(n) if mask >> j & 1]
            if len(subset) == 1:
                tok = subset[0].strip()
                if not tok.startswith(("i.", "c.")):
                    tok = f"i.{tok}"
                cols.extend(_interaction(data, [tok], term, context, True))
            else:
                cols.extend(_interaction(data, subset, term, context, True))
        return cols
    tokens = [t for t in term.split("#") if t.strip()]
    # A lone ``i.v`` omits its base level; a ``#`` product keeps every cell.
    return _interaction(data, tokens, term, context, drop_base=len(tokens) == 1)


def expand_factor_terms(
    data: pd.DataFrame,
    terms: Optional[Union[str, Sequence[str]]],
    *,
    context: str = "etwfe",
) -> Tuple[pd.DataFrame, Optional[List[str]]]:
    """Expand Stata factor terms in ``terms`` into columns of ``data``.

    Returns ``(data, names)``: ``data`` is a copy with the generated
    columns added (the input itself when nothing needed expanding) and
    ``names`` the column list to pass on, in order.  Plain column names
    pass through unchanged.

    Examples
    --------
    >>> import pandas as pd
    >>> from statspai.did._factor_terms import expand_factor_terms
    >>> df = pd.DataFrame({"year": [1, 2, 3, 1], "node": [0, 1, 1, 0]})
    >>> _, names = expand_factor_terms(df, ["i.year#i.node"])
    >>> names
    ['1.year#0.node', '1.year#1.node', '2.year#0.node', '2.year#1.node', '3.year#0.node', '3.year#1.node']
    >>> expand_factor_terms(df, ["i.year"])[1]
    ['2.year', '3.year']
    """
    if terms is None:
        return data, None
    term_list = [terms] if isinstance(terms, str) else list(terms)
    if not any(is_factor_term(t, data.columns) for t in term_list):
        return data, term_list
    new_cols: Dict[str, pd.Series] = {}
    names: List[str] = []
    for t in term_list:
        if not is_factor_term(t, data.columns):
            names.append(t)
            continue
        for label, values in _term_columns(data, t, context):
            if label in data.columns or label in new_cols:
                if label in new_cols:
                    continue  # the same column from an overlapping term
                raise MethodIncompatibility(
                    f"{context}: generated column {label!r} from {t!r} clashes "
                    "with an existing column.",
                    recovery_hint="Rename the existing column.",
                    diagnostics={"term": t, "column": label},
                )
            new_cols[label] = values
            names.append(label)
    out = pd.concat([data, pd.DataFrame(new_cols, index=data.index)], axis=1)
    return out, names


def factor_xvar(
    data: pd.DataFrame, xvar: Any, *, context: str = "etwfe"
) -> Tuple[pd.DataFrame, Any]:
    """Resolve ``i.v`` / ``c.v`` in an ``xvar`` specification.

    ``i.v`` makes ``v`` categorical (level dummies, per-level ATTs with
    ``etwfe_emfx(by_xvar=True)``); ``c.v`` keeps it continuous.
    Interactions are not a moderator design ``xvar`` supports.
    """
    if xvar is None:
        return data, None
    items = [xvar] if isinstance(xvar, str) else list(xvar)
    if not any(is_factor_term(x, data.columns) for x in items):
        return data, xvar
    out = data.copy()
    resolved = []
    for x in items:
        if not is_factor_term(x, data.columns):
            resolved.append(x)
            continue
        if "#" in x:
            raise MethodIncompatibility(
                f"{context}: xvar={x!r} -- interactions are not supported as "
                "treatment-effect moderators.",
                recovery_hint="Pass the components as separate xvar terms.",
                diagnostics={"xvar": x},
            )
        var = x[2:]
        if var not in data.columns:
            raise MethodIncompatibility(
                f"{context}: xvar={x!r} refers to {var!r}, which is not a column.",
                recovery_hint="Name an existing column.",
                diagnostics={"xvar": x},
            )
        if x.startswith("i."):
            out[var] = data[var].astype("category")
        else:
            out[var] = pd.to_numeric(data[var], errors="coerce").astype(float)
        resolved.append(var)
    return out, (resolved[0] if isinstance(xvar, str) else resolved)
