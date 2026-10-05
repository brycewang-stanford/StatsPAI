"""Dependency-free argument validators shared across estimator families.

Deliberately importing nothing beyond numpy and ``..exceptions``. The
first home for ``require_bool_flag`` was ``core/_vcov.py``, which reads
as the natural place for a standard-error validator — but ``_vcov``
imports ``._numba_kernels`` at module level, so routing five estimator
modules through it dragged numba into a plain ``import statspai`` and
tripped the cold-import budget. A validator has no business costing an
import; hence this module.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..exceptions import MethodIncompatibility


def require_bool_flag(value, *, argument: str = "robust") -> bool:
    """Reject a non-boolean where the signature promises an on/off flag.

    ``robust`` is a *boolean* switch on eight estimators and a *string*
    HC-type selector on the regression family (``robust="HC1"``).
    ``statspai._house_style.ROBUST_BOOL_HINTS`` names that split the
    highest-impact hazard in the signature surface. It was also a silent
    one: every bool-typed site accepted the string form, read it as truthy,
    and returned its default sandwich.

    ``robust="cluster"`` is the case that costs someone a result. Clustering
    is a separate ``cluster=`` argument on all of these estimators, so the
    call returned *unclustered* standard errors and gave no sign the request
    had been dropped — output that is correct for what was computed and is
    not what was asked for.

    One implementation, per CLAUDE.md §4; the DiD and dynamic-panel families
    alias this rather than carrying their own copies.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{argument}` must be boolean.",
            recovery_hint=(
                f"Pass `{argument}=True` or `{argument}=False`. A standard-error "
                f"*type* string such as 'HC1' is not accepted here; for clustered "
                f"standard errors pass `cluster=<column>` instead."
            ),
            diagnostics={"argument": argument, "type": type(value).__name__},
        )
    return bool(value)


def validate_fold_indices(
    fold_indices: Any,
    n_input: int,
    *,
    context: str,
    keep: Any = None,
    n_folds: Optional[int] = None,
    binary_target: Any = None,
) -> np.ndarray:
    """Validate a caller-supplied cross-fitting partition.

    The convention is the one ``sp.dml(fold_indices=...)`` normalises to:
    one integer label per input row, the labels being exactly
    ``0, 1, ..., K-1`` with ``K >= 2``, and row ``i`` held out when fold
    ``fold_indices[i]`` is predicted. The vector is aligned with the rows
    of the *input* data; ``keep`` (a boolean mask of the rows the
    estimator retains after dropping missing values) subsets it afterwards,
    so a caller never has to pre-compute which rows survive.

    Parameters
    ----------
    fold_indices : array-like of int, length ``n_input``
    n_input : int
        Number of rows in the data the vector is aligned with.
    context : str
        Prefix for error messages (e.g. ``"sp.tmle"``).
    keep : array-like of bool, optional
        Rows retained by the estimator.
    n_folds : int, optional
        When given, the partition must define exactly this many folds.
    binary_target : array-like, optional
        A 0/1 vector (already subset by ``keep``) that every training
        complement must contain both classes of, e.g. the treatment for a
        propensity model.

    Returns
    -------
    ndarray of int
        The fold labels of the retained rows.
    """
    from ..exceptions import DataInsufficient

    raw = np.asarray(fold_indices)
    if raw.ndim != 1 or raw.shape[0] != int(n_input):
        raise MethodIncompatibility(
            f"{context}: fold_indices must be 1-D of length {int(n_input)} "
            f"(one label per row of data); got shape {raw.shape}.",
            recovery_hint="Pass one integer fold label per input row.",
            diagnostics={"shape": list(raw.shape), "n_input": int(n_input)},
        )
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iuf":
        raise MethodIncompatibility(
            f"{context}: fold_indices must be integer labels 0..K-1; got "
            f"dtype {raw.dtype}.",
            recovery_hint="Pass an integer array such as np.array([0, 1, 2, ...]).",
            diagnostics={"dtype": str(raw.dtype)},
        )
    if raw.dtype.kind == "f":
        if not np.all(np.isfinite(raw)):
            raise MethodIncompatibility(
                f"{context}: fold_indices contain missing or non-finite values.",
                recovery_hint="Assign every row to a fold.",
            )
        if not np.all(raw == np.round(raw)):
            raise MethodIncompatibility(
                f"{context}: fold_indices must be integer-valued.",
                recovery_hint="Pass integer fold labels 0..K-1.",
            )
    codes = raw.astype(np.int64)
    labels = np.unique(codes)
    k = int(labels.size)
    if k < 2 or not np.array_equal(labels, np.arange(k)):
        raise MethodIncompatibility(
            f"{context}: fold_indices must use the labels 0..K-1 with K >= 2; "
            f"got labels {labels[:10].tolist()}"
            f"{'...' if labels.size > 10 else ''}.",
            recovery_hint="Relabel the folds as consecutive integers from 0.",
            diagnostics={"labels": labels[:50].tolist()},
        )
    if n_folds is not None and k != int(n_folds):
        raise MethodIncompatibility(
            f"{context}: fold_indices define {k} folds but n_folds={n_folds}.",
            recovery_hint=f"Pass n_folds={k} together with this partition.",
            diagnostics={"k_from_fold_indices": k, "n_folds": n_folds},
        )
    if keep is not None:
        codes = codes[np.asarray(keep, dtype=bool)]
    counts = np.bincount(codes, minlength=k)
    if np.any(counts == 0):
        raise DataInsufficient(
            f"{context}: after dropping rows with missing values, fold(s) "
            f"{np.flatnonzero(counts == 0).tolist()} of fold_indices are empty.",
            recovery_hint="Supply a partition of the complete-case rows.",
            diagnostics={"fold_sizes": counts.tolist()},
        )
    if binary_target is not None:
        target = np.asarray(binary_target)
        for fold in range(k):
            classes = np.unique(target[codes != fold])
            if classes.size < 2:
                raise DataInsufficient(
                    f"{context}: the training complement of fold {fold} "
                    f"contains a single treatment class, so the nuisance "
                    f"models cannot be fitted out of fold.",
                    recovery_hint=(
                        "Use a partition whose training sets each contain "
                        "treated and control units (e.g. stratify on the "
                        "treatment)."
                    ),
                    diagnostics={
                        "fold": fold,
                        "classes_in_train": classes.tolist(),
                    },
                )
    return codes


def require_binary_treatment(
    values: Any, *, function: str, argument: str = "treat"
) -> None:
    """Refuse a treatment column that is not coded 0/1.

    The two-group estimators and tables select ``== 1`` and ``== 0``. A
    column with other codes (arm labels, a dose, three arms) used to leave
    both groups empty or partly filled, and the output was a table of NaN
    or an estimate for a subset, with nothing said.
    """
    import pandas as pd

    observed = pd.unique(pd.Series(values).dropna())
    try:
        ok = len(observed) > 0 and all(v in (0, 1) for v in observed)
    except TypeError:  # unhashable or uncomparable codes
        ok = False
    if ok:
        return
    shown = sorted(observed, key=repr)[:6]
    raise MethodIncompatibility(
        f"{function}: {argument} must be coded 0/1; found "
        f"{len(observed)} distinct values, e.g. {shown}.",
        recovery_hint=(
            "For several arms, compare one arm with the control at a time on "
            "the rows of those two arms, with the arm recoded to 1 and the "
            "control to 0. For a dose, use an estimator for continuous "
            "treatments."
        ),
        diagnostics={
            "n_values": int(len(observed)),
            "values": [repr(v) for v in shown],
        },
    )


def treatment_as_float(
    values: Any, *, function: str, argument: str = "treat"
) -> np.ndarray:
    """A treatment column as floats, or a named error when it holds labels.

    ``np.asarray(labels, dtype=float)`` fails with ``could not convert
    string to float: 'high'``, which names neither the function nor the
    column nor the way out. Which label is the treated one is the
    caller's to say, so the column is not recoded here.
    """
    import pandas as pd
    from pandas.api.types import is_bool_dtype, is_numeric_dtype

    column = pd.Series(values)
    if is_bool_dtype(column) or is_numeric_dtype(column):
        return np.asarray(column.to_numpy(dtype=float), dtype=float)
    try:
        return np.asarray(column.to_numpy(dtype=float), dtype=float)
    except (TypeError, ValueError):
        pass
    levels = sorted(pd.unique(column.dropna()), key=repr)
    shown = [repr(v) for v in levels[:6]]
    if len(levels) == 2:
        hint = (
            f"Say which label is the treated one and recode, e.g. "
            f"df[col] = (df[col] == {levels[1]!r}).astype(int)."
        )
    else:
        hint = (
            "Recode the treated arm to 1 and the control arm to 0; with "
            "several arms, compare one arm with the control at a time."
        )
    raise MethodIncompatibility(
        f"{function}: {argument} holds labels, not numbers "
        f"({len(levels)} distinct: {', '.join(shown)}).",
        recovery_hint=hint,
        diagnostics={"n_values": int(len(levels)), "values": shown},
    )
