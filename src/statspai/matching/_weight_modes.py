"""Stata weight-regime semantics for matched-sample regressions.

Stata's ``regress`` accepts several weight kinds that share a point estimate
but *not* a variance.  Matched-sample work only ever needs two of them, and
the difference between them is a standing source of confusion when a PSM-DID
is ported from Stata:

``aweight`` (analytic weights)
    "This row is a mean of ``w`` underlying observations."  The estimator is
    WLS; the residual degrees of freedom stay at ``n_rows - k``.  Stata
    rescales the weights to sum to ``n_rows``, which leaves
    ``sigma2 * (X'WX)^-1`` unchanged — the reported variance is invariant to
    the scale of ``w``.

``fweight`` (frequency weights)
    "This row *stands for* ``w`` identical rows."  The residual degrees of
    freedom become ``sum(w) - k`` and every ``N``-dependent correction (the
    ``(N-1)/(N-k)`` factor in the cluster sandwich included) uses ``sum(w)``
    rather than the row count.  Stata therefore rejects non-integer ``w``.

Because ``fweight`` *is defined as* row replication, the honest way to
implement it is to replicate the rows and run the unweighted regression.
That is what :func:`expand_frequency_weights` does, and it is not an
approximation: verified against Stata 18 MP on the fixture behind
``tests/reference_parity/test_psmdid_weight_parity.py``, physically expanding
the matched sample and running ``reg y d post did`` reproduces
``reg y d post did [fweight=_weight]`` bit-for-bit — coefficient
``1.55116250172257e+00``, SE ``2.14796919061755e-01``, ``df_r = 496`` — and
likewise under ``cluster(id)`` (SE ``1.08275840314657e-01``, ``df_r = 184``,
``G = 185``).  Deriving Stata's per-VCE degrees-of-freedom corrections by
hand would be both more code and more ways to be wrong.

The cost is memory: the expanded frame has ``sum(w)`` rows.  For a matched
PSM sample ``sum(w)`` is on the order of twice the number of treated units
per period, so this is cheap in the setting it exists for.

References
----------
Leuven, E. and Sianesi, B. (2003). PSMATCH2: Stata module to perform full
    Mahalanobis and propensity score matching, common support graphing, and
    covariate imbalance testing.  Statistical Software Components S432001,
    Boston College Department of Economics.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

#: Weight regimes accepted by the matched-sample regression helpers.
WEIGHT_MODES = ("aweight", "fweight", "none")

#: Largest expanded frame we will materialise before refusing.  A matched
#: PSM-DID sample that expands past this is almost certainly a misuse (e.g.
#: passing a probability weight where a frequency weight was meant).
MAX_EXPANDED_ROWS = 50_000_000

# Tolerance for calling a float weight an integer.  Weights arrive as float64
# accumulations of 1/k shares, so exact equality is too strict: with k = 1 a
# control matched 3 times can land on 2.9999999999999996.
_INTEGER_ATOL = 1e-9


def resolve_weight_mode(weight: Any) -> str:
    """Validate and normalise a ``weight=`` argument.

    Parameters
    ----------
    weight : str
        One of ``'aweight'``, ``'fweight'``, ``'none'``.

    Returns
    -------
    str
        The normalised mode.

    Raises
    ------
    MethodIncompatibility
        If ``weight`` is not one of the supported regimes.
    """
    if not isinstance(weight, str):
        raise MethodIncompatibility(
            f"weight must be a string, one of {WEIGHT_MODES}; got "
            f"{type(weight).__name__}.",
            diagnostics={"weight": repr(weight)},
            recovery_hint="Pass weight='aweight' (default), 'fweight', or 'none'.",
        )
    mode = weight.strip().lower()
    if mode not in WEIGHT_MODES:
        raise MethodIncompatibility(
            f"weight must be one of {WEIGHT_MODES}; got {weight!r}.",
            diagnostics={"weight": weight, "supported": list(WEIGHT_MODES)},
            recovery_hint=(
                "Use weight='aweight' for Stata [aweight=_weight] semantics "
                "(WLS, df = n_rows - k), weight='fweight' for Stata "
                "[fweight=_weight] semantics (df = sum(w) - k), or "
                "weight='none' for the unweighted matched-sample regression."
            ),
        )
    return mode


def integerize_weights(
    w: np.ndarray,
    *,
    atol: float = _INTEGER_ATOL,
) -> Optional[np.ndarray]:
    """Round ``w`` to integers if every entry is integral within ``atol``.

    Returns ``None`` when any weight is genuinely fractional, which is the
    signal that Stata would refuse the ``fweight`` regime.
    """
    w = np.asarray(w, dtype=float)
    nearest: np.ndarray = np.rint(w)
    if not np.all(np.abs(w - nearest) <= atol):
        return None
    counts: np.ndarray = nearest.astype(np.int64)
    return counts


def expand_frequency_weights(
    frame: pd.DataFrame,
    weight_col: str,
    *,
    context: str = "fweight regression",
) -> pd.DataFrame:
    """Replicate each row ``frame[weight_col]`` times (Stata ``fweight``).

    This is the *definition* of a frequency weight, so the unweighted
    regression on the returned frame is exactly the ``[fweight=]``
    regression on the input — coefficients, ``sigma2``, residual degrees of
    freedom, t / p values, and the clustered sandwich alike.  Replicated rows
    keep their original cluster label, so the number of clusters is
    unchanged; only ``N`` grows.

    Parameters
    ----------
    frame : DataFrame
        Rows to expand.  The index is not preserved (replication makes it
        non-unique by construction); the result is re-indexed 0..N-1.
    weight_col : str
        Column holding the frequency weights.  Must be positive and integral.
    context : str
        Used in error messages so the caller's option name shows up.

    Returns
    -------
    DataFrame
        The expanded frame, with ``weight_col`` left in place (every row now
        carries weight 1 in effect, but dropping the column would surprise
        callers who select on it).

    Raises
    ------
    MethodIncompatibility
        If the weights are non-integer, non-positive, or expand to an
        implausible number of rows.  Stata raises the analogous
        "may not use noninteger frequency weights" error.
    """
    if weight_col not in frame.columns:
        raise MethodIncompatibility(
            f"{context}: weight column {weight_col!r} is not in the frame.",
            diagnostics={"weight_col": weight_col, "columns": list(frame.columns)},
            recovery_hint="Check the weight column name.",
        )

    raw = frame[weight_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(raw)):
        raise MethodIncompatibility(
            f"{context}: frequency weights must all be finite; "
            f"{int(np.sum(~np.isfinite(raw)))} are missing or infinite.",
            diagnostics={"n_nonfinite": int(np.sum(~np.isfinite(raw)))},
            recovery_hint=(
                "Restrict to the matched sample before the regression, or use "
                "weight='aweight'."
            ),
        )

    counts = integerize_weights(raw)
    if counts is None:
        frac = raw[np.abs(raw - np.rint(raw)) > _INTEGER_ATOL]
        raise MethodIncompatibility(
            f"{context}: Stata frequency weights must be integers, but "
            f"{frac.size} weight(s) are fractional (e.g. {float(frac[0]):.6g}). "
            "Matching with k > 1 neighbours splits a treated unit's weight "
            "into 1/k shares, so the resulting _weight is not a frequency.",
            diagnostics={
                "n_fractional": int(frac.size),
                "example": float(frac[0]),
            },
            recovery_hint=(
                "Use weight='aweight' (the default) — it is the correct "
                "regime for fractional matching weights and is what Stata's "
                "[aweight=_weight] computes. weight='fweight' is only "
                "meaningful for 1:1 matching, where _weight counts how many "
                "times a control was reused."
            ),
        )

    if np.any(counts <= 0):
        raise MethodIncompatibility(
            f"{context}: frequency weights must be strictly positive; "
            f"{int(np.sum(counts <= 0))} row(s) have weight <= 0.",
            diagnostics={"n_nonpositive": int(np.sum(counts <= 0))},
            recovery_hint=(
                "Drop unmatched rows (missing _weight) before the regression."
            ),
        )

    total = int(counts.sum())
    if total > MAX_EXPANDED_ROWS:
        raise MethodIncompatibility(
            f"{context}: expanding by the frequency weights would produce "
            f"{total:,} rows (limit {MAX_EXPANDED_ROWS:,}).",
            diagnostics={"expanded_rows": total, "limit": MAX_EXPANDED_ROWS},
            recovery_hint=(
                "This usually means _weight holds probability or sampling "
                "weights rather than frequencies. Use weight='aweight'."
            ),
        )

    out = frame.loc[frame.index.repeat(counts)].reset_index(drop=True)
    return out


def weight_regime_info(
    mode: str,
    w: Optional[np.ndarray],
) -> Dict[str, Any]:
    """Describe the applied regime for ``model_info`` / diagnostics."""
    info: Dict[str, Any] = {"weight": mode}
    if mode == "none" or w is None:
        info["weight_semantics"] = "unweighted"
        return info
    w = np.asarray(w, dtype=float)
    finite = w[np.isfinite(w)]
    if mode == "aweight":
        info[
            "weight_semantics"
        ] = "Stata [aweight=_weight]: WLS with df_resid = n_rows - k"
    else:
        info[
            "weight_semantics"
        ] = "Stata [fweight=_weight]: rows replicated, df_resid = sum(w) - k"
    info["weight_sum"] = float(finite.sum())
    info["weight_is_integer"] = integerize_weights(finite) is not None
    return info
