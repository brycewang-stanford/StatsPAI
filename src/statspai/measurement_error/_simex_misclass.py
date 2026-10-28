"""Misclassification SIMEX (MC-SIMEX) for a binary or categorical variable.

A variable with ``k`` levels is observed through a misclassification
matrix ``P`` with ``P[j, i] = Pr(observed = j | true = i)`` (columns sum to
one, as in R ``simex::mcsimex``). MC-SIMEX misclassifies the observed
variable again with ``P ** lambda``, so that the total misclassification
is ``P ** (1 + lambda)``, and extrapolates to ``lambda = -1`` like SIMEX.
``P ** lambda`` is ``V diag(e ** lambda) V^{-1}`` from the eigen
decomposition, rounded to seven significant decimals like R's
``zapsmall``. For ``k = 2`` it is a misclassification matrix for every
``lambda >= 0`` when ``P[0, 0] + P[1, 1] > 1``; for ``k > 2`` the
condition is R's ``check.mc.matrix``: the matrix logarithm (zapsmall-ed,
diagonal set to one) has no negative entry, i.e. ``log P`` is a generator.

References
----------
[@kuchenhoff2006general]
"""

from __future__ import annotations

import re
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def mc_matrix(name: str, value: Any) -> np.ndarray:
    """Validate one 2 x 2 misclassification matrix (rows, columns = 0, 1)."""
    if isinstance(value, pd.DataFrame):
        try:
            value = value.loc[[0, 1], [0, 1]]
        except KeyError:
            value = value.to_numpy()
    try:
        P = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        P = np.full((1, 1), np.nan)
    hint = (
        "Pass misclassification={'x': [[p00, p01], [p10, p11]]} with "
        "P[j, i] = Pr(observed j | true i); each column sums to one."
    )
    if P.shape != (2, 2) or not np.all(np.isfinite(P)):
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} must be a "
            "finite 2 x 2 matrix.",
            recovery_hint=hint,
        )
    if np.any(P < 0) or np.any(P > 1) or not np.allclose(P.sum(axis=0), 1.0, atol=1e-8):
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} must have "
            "entries in [0, 1] and columns that sum to one.",
            recovery_hint=hint,
            diagnostics={"column_sums": P.sum(axis=0).tolist()},
        )
    if not P[0, 0] + P[1, 1] > 1.0:
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} has "
            "P[0, 0] + P[1, 1] <= 1; its fractional powers are not "
            "misclassification matrices and the error cannot be undone.",
            recovery_hint="A classifier no better than chance carries no "
            "information; check the orientation of the matrix (columns are "
            "the true values).",
        )
    return P


def mc_power(P: np.ndarray, lam: float) -> np.ndarray:
    """``P ** lam`` by eigen decomposition, rounded like R ``zapsmall``."""
    evalue, evec = np.linalg.eig(P)
    out = np.real(evec @ np.diag(evalue.astype(complex) ** lam) @ np.linalg.inv(evec))
    mx = float(np.max(np.abs(out)))
    digits = max(0, 7 - int(np.ceil(np.log10(mx)))) if mx > 0 else 7
    rounded: np.ndarray = np.round(out, digits)
    return rounded


def misclassify(values: np.ndarray, P_lam: np.ndarray, rng: Any) -> np.ndarray:
    """Draw new level codes: ``Pr(new = j | old = i) = P_lam[j, i]``.

    For a 0/1 variable this is ``Pr(new = 1 | old = i) = P_lam[1, i]``.
    With ``k > 2`` levels the codes are ``0 .. k-1`` and each column of
    ``P_lam`` is renormalised (the rounding of ``mc_power`` can leave it a
    few 1e-8 off one), as R's ``sample(prob=)`` does.
    """
    if P_lam.shape == (2, 2):
        prob_one = P_lam[1, values.astype(int)]
        out: np.ndarray = (rng.random(values.size) < prob_one).astype(float)
        return out
    cum = np.cumsum(np.clip(P_lam, 0.0, None)[:, values.astype(int)], axis=0)
    cum = cum / cum[-1]
    u = rng.random(values.size)
    codes: np.ndarray = (u[None, :] >= cum[:-1]).sum(axis=0).astype(float)
    return codes


def _zapsmall(a: np.ndarray) -> np.ndarray:
    """R ``zapsmall(a)`` with the default ``digits = 7``."""
    mx = float(np.max(np.abs(a)))
    digits = max(0, 7 - int(np.ceil(np.log10(mx)))) if mx > 0 else 7
    out: np.ndarray = np.round(a, digits)
    return out


def mc_matrix_levels(name: str, value: Any) -> Tuple[np.ndarray, List[Any]]:
    """Validate a ``k x k`` misclassification matrix and return its levels.

    ``k = 2`` goes through :func:`mc_matrix` (levels 0, 1). For ``k > 2`` a
    DataFrame is read by its labels (rows are put in the column order) and
    an array by position, with levels ``0 .. k-1``.
    """
    shape = np.shape(value.to_numpy() if isinstance(value, pd.DataFrame) else value)
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] <= 2:
        return mc_matrix(name, value), [0, 1]
    hint = (
        "Pass misclassification={'x': P} with a k x k matrix P[j, i] = "
        "Pr(observed j | true i) whose columns sum to one; a DataFrame is "
        "read by its labels, an array by the levels 0 .. k-1."
    )
    if isinstance(value, pd.DataFrame):
        levels = list(value.columns)
        if set(value.index) != set(levels) or len(set(levels)) != len(levels):
            raise MethodIncompatibility(
                f"sp.simex: the misclassification matrix of {name!r} must have "
                "the same labels on its rows and columns.",
                recovery_hint=hint,
                diagnostics={
                    "index": list(map(str, value.index)),
                    "columns": list(map(str, value.columns)),
                },
            )
        value = value.loc[levels, levels].to_numpy()
    else:
        levels = list(range(shape[0]))
    try:
        P = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        P = np.full(shape, np.nan)
    if not np.all(np.isfinite(P)):
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} must be " "finite.",
            recovery_hint=hint,
        )
    if np.any(P < 0) or np.any(P > 1) or not np.allclose(P.sum(axis=0), 1.0, atol=1e-8):
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} must have "
            "entries in [0, 1] and columns that sum to one.",
            recovery_hint=hint,
            diagnostics={"column_sums": P.sum(axis=0).tolist()},
        )
    evalue, evec = np.linalg.eig(P)
    if np.any(np.abs(np.imag(evalue)) > 0) or np.any(np.real(evalue) <= 0):
        raise MethodIncompatibility(
            f"sp.simex: the misclassification matrix of {name!r} has complex "
            "or non-positive eigenvalues, so it has no real matrix logarithm "
            "and its fractional powers are not misclassification matrices.",
            recovery_hint="MC-SIMEX needs a matrix close enough to the "
            "identity (strongly diagonal); check its orientation (columns "
            "are the true values).",
            diagnostics={"eigenvalues": [complex(e) for e in evalue]},
        )
    evec = np.real(evec)
    log_p = _zapsmall(evec @ np.diag(np.log(np.real(evalue))) @ np.linalg.inv(evec))
    np.fill_diagonal(log_p, 1.0)
    if not np.all(log_p > -np.finfo(float).eps):
        raise MethodIncompatibility(
            f"sp.simex: the matrix logarithm of the misclassification matrix "
            f"of {name!r} has negative off-diagonal entries, so its fractional "
            "powers are not misclassification matrices (R check.mc.matrix).",
            recovery_hint="MC-SIMEX cannot undo this misclassification; a "
            "matrix closer to the identity, or one obtained by embedding a "
            "continuous-time process, passes the check.",
            diagnostics={"min_off_diagonal_log": float(log_p.min())},
        )
    return P, levels


def _level_labels(level: Any) -> List[str]:
    """How a level can be printed inside a dummy name such as ``[T.1]``."""
    out = [str(level)]
    if isinstance(level, (int, float, np.integer, np.floating)) and not isinstance(
        level, bool
    ):
        f = float(level)
        out.append(repr(f))
        if f.is_integer():
            out.append(str(int(f)))
    return out


def _same_level(raw: np.ndarray, level: Any) -> np.ndarray:
    numeric = isinstance(level, (int, float, np.integer, np.floating))
    if numeric and pd.api.types.is_numeric_dtype(raw):
        same: np.ndarray = np.asarray(raw, dtype=float) == float(level)
    else:
        same = np.asarray([str(v) for v in raw]) == str(level)
    return same


def factor_columns(
    name: str,
    levels: List[Any],
    names: List[str],
    df: pd.DataFrame,
    X: np.ndarray,
    rows: np.ndarray,
) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Dummy columns of a categorical term and the level codes of the data.

    The term enters the formula as ``C(x)``, ``i.x`` or a categorical
    column, so the design has one indicator per level except the reference
    (``C(x)[T.1]``, ``x[T.b]``). Returns ``[(column, level code)]`` and the
    observed variable as codes ``0 .. k-1`` in the order of ``levels``;
    each simulated data set rebuilds the indicators from new codes.
    """
    hint = (
        f"Enter {name!r} as a factor, e.g. 'y ~ C({name}) + z', with levels "
        "that match the rows and columns of its misclassification matrix."
    )
    pattern = re.compile(
        r"^(?:C\(\s*"
        + re.escape(name)
        + r"\s*\)|"
        + re.escape(name)
        + r")\[(?:T\.)?(.+)\]$"
    )
    dummies: List[Tuple[int, int]] = []
    for j, term in enumerate(names):
        m = pattern.match(term)
        if not m:
            continue
        label = m.group(1)
        hit = [c for c, lvl in enumerate(levels) if label in _level_labels(lvl)]
        if len(hit) != 1:
            raise MethodIncompatibility(
                f"sp.simex: the dummy {term!r} matches no level of the "
                f"misclassification matrix of {name!r} ({levels}).",
                recovery_hint=hint,
            )
        dummies.append((j, hit[0]))
    if not dummies:
        raise MethodIncompatibility(
            f"sp.simex: {name!r} is not a term of the model {names}; a variable "
            f"with a {len(levels)} x {len(levels)} misclassification matrix "
            "must enter as a factor.",
            recovery_hint=hint,
            diagnostics={"terms": names},
        )
    used = {j for j, _ in dummies}
    others = [
        t
        for j, t in enumerate(names)
        if j not in used
        and re.search(r"(?<![\w.])" + re.escape(name) + r"(?![\w.])", t)
    ]
    if others:
        raise MethodIncompatibility(
            f"sp.simex: {name!r} also appears in {others}; MC-SIMEX rebuilds "
            "only its own indicators.",
            recovery_hint="Fit the model with the factor as a main effect only.",
        )
    if name not in df.columns:
        raise MethodIncompatibility(
            f"sp.simex: column {name!r} is not in the data.", recovery_hint=hint
        )
    raw = df[name].to_numpy()[rows]
    codes = np.full(raw.shape[0], -1.0)
    for c, lvl in enumerate(levels):
        codes[_same_level(raw, lvl)] = float(c)
    if np.any(codes < 0):
        bad = sorted({str(v) for v in raw[codes < 0]})[:5]
        raise MethodIncompatibility(
            f"sp.simex: {name!r} takes values {bad} that are not levels of its "
            f"misclassification matrix ({levels}).",
            recovery_hint="Label the matrix (a DataFrame) with the variable's "
            "values, or code the variable 0 .. k-1.",
        )
    missing = [levels[c] for c in range(len(levels)) if not np.any(codes == c)]
    uncovered = [c for c in range(len(levels)) if c not in {lv for _, lv in dummies}]
    if missing or len(uncovered) > 1:
        raise MethodIncompatibility(
            f"sp.simex: every level of {name!r} must be observed and all but "
            f"the reference must have an indicator; unobserved {missing}.",
            recovery_hint="Drop the empty level from the matrix only if it "
            "cannot be reached by misclassification; otherwise MC-SIMEX "
            "cannot represent it.",
        )
    for j, c in dummies:
        if not np.array_equal(X[:, j], (codes == c).astype(float)):
            raise MethodIncompatibility(
                f"sp.simex: the column {names[j]!r} is not the indicator of "
                f"level {levels[c]!r} of {name!r}.",
                recovery_hint="Use treatment (dummy) coding, the default of " "C(x).",
            )
    return dummies, codes
