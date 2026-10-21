"""Integer columns inside generated formula terms.

patsy evaluates the code of a term such as ``I(age**3)`` or
``np.power(inc, 2)`` on the column values in their stored dtype. numpy
integer arithmetic wraps around without a warning, so ``I(age**3)`` on an
``int8`` column with ages above 5 is garbage, and ``I(pop**3)`` overflows
even ``int64`` once ``pop`` passes about two million. Stata does all
arithmetic in double precision and R's ``^`` returns a double, so the same
formula is right there.

:func:`float_generated_term_columns` casts every integer column that a
generated term reads to ``float64`` before patsy sees the data. A bare
column (``age``) and an interaction (``age:educ``) are already converted
to float by patsy, and ``C(g)`` keeps its integer levels (a cast would
rename ``C(g)[T.2]`` to ``C(g)[T.2.0]``), so neither is touched. A column
read both by ``C(...)`` and by a generated term stays integer (``int64``)
and is checked for a magnitude at which a cube could overflow.

:func:`warn_wrapped_integer_columns` flags a narrow integer column (the
``int8`` / ``int16`` / ``int32`` storage of a ``.dta`` or parquet file)
whose values reach both ends of its type's range: the signature of a
product or a power that was computed in that type before the call.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "float_generated_term_columns",
    "float_generated_code_columns",
    "warn_wrapped_integer_columns",
    "formula_factor_codes",
]

_NAME = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\b(?!\s*\()")
_QUOTED_NAME = re.compile(r"""\bQ\(\s*(["'])(.*?)\1\s*\)""")
_STRING = re.compile(r"""(["']).*?\1""")
# Inside a generated term these only make sense on integers; leave the
# columns of such a term alone rather than make it raise on floats.
_BITWISE = re.compile(r"[&|~]|<<|>>")
_BARE = re.compile(r"^\s*[A-Za-z_]\w*\s*$")
_CATEGORICAL = re.compile(r"^\s*C\s*\(")
_QUOTED_ONLY = re.compile(r"""^\s*Q\(\s*(["']).*?\1\s*\)\s*$""")
# |x|**3 stays below 2**63 when |x| < 2**21.
_INT64_CUBE_SAFE = 2**21


def _is_int(dtype: Any) -> bool:
    return isinstance(dtype, np.dtype) and dtype.kind in "iu"


def formula_factor_codes(formula: Any) -> List[str]:
    """The code of every factor in a formula string or patsy ``DesignInfo``.

    Returns ``[]`` when the formula does not parse; the caller's own parse
    then reports the problem.
    """
    infos = getattr(formula, "factor_infos", None)
    if infos is not None:
        return [str(getattr(f, "code", f)) for f in infos]
    if not isinstance(formula, str):
        return []
    from patsy import ModelDesc, PatsyError

    try:
        desc = ModelDesc.from_formula(formula)
    except (PatsyError, SyntaxError, ValueError, TypeError):
        # the caller's own parse reports the malformed formula
        return []
    codes = []
    for term in list(desc.lhs_termlist) + list(desc.rhs_termlist):
        for factor in term.factors:
            codes.append(str(getattr(factor, "code", factor)))
    return codes


def _names_in(code: str) -> Set[str]:
    names = {m.group(2) for m in _QUOTED_NAME.finditer(code)}
    stripped = _STRING.sub(" ", _QUOTED_NAME.sub(" ", code))
    names |= set(_NAME.findall(stripped))
    return names


def _split_codes(codes: Iterable[str]) -> Tuple[Set[str], Set[str]]:
    """Names read by generated terms, and names read by ``C(...)``."""
    generated: Set[str] = set()
    categorical: Set[str] = set()
    for code in codes:
        if _BARE.match(code) or _QUOTED_ONLY.match(code):
            continue
        if _CATEGORICAL.match(code):
            categorical |= _names_in(code)
            continue
        if _BITWISE.search(_STRING.sub(" ", code)):
            continue
        generated |= _names_in(code)
    return generated, categorical


def float_generated_term_columns(data: pd.DataFrame, formula: Any) -> pd.DataFrame:
    """Cast the integer columns read by generated terms to ``float64``.

    ``formula`` is a formula string or a patsy ``DesignInfo`` (prediction
    on new data). The frame is copied only when a column is cast.
    """
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        return data
    if not any(_is_int(t) for t in data.dtypes):
        return data
    return float_generated_code_columns(data, formula_factor_codes(formula))


def float_generated_code_columns(
    data: pd.DataFrame, codes: Iterable[str]
) -> pd.DataFrame:
    """:func:`float_generated_term_columns` on factor codes already parsed.

    For callers whose formula grammar is not patsy's (the fixest wrapper
    hands over the codes of each ``~`` part, with ``i(...)`` written as
    ``C(...)`` so that its levels keep their integer names).
    """
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        return data
    if not any(_is_int(t) for t in data.dtypes):
        return data
    generated, categorical = _split_codes(codes)
    if not generated:
        return data
    cast = {}
    for col in generated:
        if col not in data.columns:
            continue
        dtype = data[col].dtype
        if not (isinstance(dtype, np.dtype) and dtype.kind in "iu"):
            continue
        values = data[col].to_numpy()
        if col in categorical:
            if values.size and np.abs(values.astype(np.float64)).max() >= (
                _INT64_CUBE_SAFE
            ):
                from ..exceptions import StatsPAIWarning

                warnings.warn(
                    f"Column {col!r} is used both as a factor (C(...)) and "
                    "inside a generated term, so it stays an integer column, "
                    "and its values are large enough for a power of it to "
                    "overflow 64-bit integers. Cast it with "
                    f"df[{col!r}].astype(float) for the generated term, or "
                    "build the term as its own column.",
                    StatsPAIWarning,
                    stacklevel=3,
                )
            continue
        cast[col] = values.astype(np.float64)
    if cast:
        data = data.assign(**cast)
    return data


def warn_wrapped_integer_columns(data: pd.DataFrame, formula: Any) -> None:
    """Warn about narrow integer columns that look wrapped around.

    A column stored as ``int8`` / ``int16`` / ``int32`` whose values reach
    within 10% of both ends of the type's range is what a product or a
    power computed in that type looks like after it wrapped. The check
    reads only the columns the formula names.
    """
    if not isinstance(data, pd.DataFrame) or not data.columns.is_unique:
        return
    if not any(_is_int(t) and t.itemsize < 8 for t in data.dtypes):
        return
    names: Set[str] = set()
    for code in formula_factor_codes(formula):
        names |= _names_in(code)
    suspect = []
    for col in sorted(names):
        if col not in data.columns:
            continue
        dtype = data[col].dtype
        if not (isinstance(dtype, np.dtype) and dtype.kind == "i"):
            continue
        if dtype.itemsize >= 8:
            continue
        values = data[col].to_numpy()
        if not values.size:
            continue
        info = np.iinfo(dtype)
        if values.min() <= 0.9 * info.min and values.max() >= 0.9 * info.max:
            suspect.append(f"{col} ({dtype}, {values.min()} .. {values.max()})")
    if suspect:
        from ..exceptions import StatsPAIWarning

        warnings.warn(
            "These integer columns reach both ends of their storage type's "
            f"range, which is what a product or power looks like after it "
            f"wrapped around in that type: {', '.join(suspect)}. If you "
            "computed them yourself (e.g. df['age3'] = df['age']**3 on an "
            "int8 column read from a .dta file), recompute them in float64 "
            "(df['age'].astype(float)**3), or write the term in the formula "
            "(I(age**3)), which StatsPAI evaluates in float64.",
            StatsPAIWarning,
            stacklevel=3,
        )
