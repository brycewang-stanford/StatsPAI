"""
Further estimands for ``sp.noisy_covariates`` (Agarwal-Singh, Appendix E).

Each estimand is a dictionary ``b`` and a counterfactual moment
``m = m(W, b)``, the derivative of the functional along ``b``
(Appendix I, Algorithm I.1):

* ``"elasticity"`` (Example E.4): ``theta = E[d gamma(D, X) / dD]``
  with the interacted quadratic dictionary
  ``b(D, X) = (X, D X, D^2 X)`` (``X`` includes the constant), so
  ``m = (0, X, 2 D X)``. With logs of ``y`` and ``treat`` it is the
  average elasticity.
* ``"policy_effect"`` (Example E.3): ``theta = E[gamma{t(X)} - gamma(X)]``
  for the affine counterfactual ``t(x) = t1 * x + t2`` with the linear
  dictionary, so ``m = (t1 - 1) * X + t2`` (zero on the constant). An
  uncorrupted regressor ``D`` may enter the dictionary and the policy.
* ``"partially_linear_iv"`` (Example E.6): ratio of the partially
  linear coefficients of ``y`` and ``treat`` on a continuous
  instrument; it reuses the partially linear dictionary of the main
  module.

``user_folds`` turns a user's fold labels into codes, for reproducing an
external cross-fitting split (the authors' replication code).

References
----------
[@agarwal2026causal]
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

FUNCTIONALS = ("elasticity", "policy_effect")


def _with_constant(X: np.ndarray, add_constant: bool) -> np.ndarray:
    if add_constant:
        return np.column_stack([np.ones(X.shape[0]), X])
    return X


def functional_dictionary(
    estimand: str, d: Optional[np.ndarray], X: np.ndarray, add_constant: bool
) -> np.ndarray:
    """``b(D, X)`` for the estimands in :data:`FUNCTIONALS`."""
    Xc = _with_constant(X, add_constant)
    if estimand == "elasticity":
        dd = np.asarray(d, dtype=float)[:, None]
        return np.column_stack([Xc, dd * Xc, dd**2 * Xc])
    if d is None:
        return Xc
    return np.column_stack([Xc, d])


def functional_moment(
    estimand: str,
    d: Optional[np.ndarray],
    X: np.ndarray,
    add_constant: bool,
    policy: Optional[Dict[str, np.ndarray]],
) -> np.ndarray:
    """Rows ``m_i`` with ``m_i' beta`` the functional of ``b' beta`` at unit i."""
    Xc = _with_constant(X, add_constant)
    if estimand == "elasticity":
        dd = np.asarray(d, dtype=float)[:, None]
        return np.column_stack([np.zeros_like(Xc), Xc, 2.0 * dd * Xc])
    if policy is None:
        raise MethodIncompatibility(
            "policy_effect needs a policy.",
            recovery_hint="Pass policy={'column': shift}.",
        )
    mx = (policy["scale"] - 1.0) * X + policy["shift"]
    parts = [mx]
    if add_constant:
        parts.insert(0, np.zeros((X.shape[0], 1)))
    if d is not None:
        parts.append(((policy["scale_d"] - 1.0) * d + policy["shift_d"])[:, None])
    return np.column_stack(parts)


def parse_policy(
    policy: Any, covariates: Sequence[str], treat: Optional[str]
) -> Dict[str, Any]:
    """Validate ``policy=`` into per-column scale and shift arrays.

    ``policy`` maps a column to a shift ``c`` (``x -> x + c``) or to a
    pair ``(a, c)`` (``x -> a x + c``). Columns not named are unchanged.
    """
    if not isinstance(policy, Mapping) or not policy:
        raise MethodIncompatibility(
            "estimand='policy_effect' needs policy= as a non-empty dict "
            "{column: shift} or {column: (scale, shift)}.",
            recovery_hint="E.g. policy={'z3': 1.0} moves z3 up by one unit; "
            "policy={'z3': (0.5, 0.0)} halves it.",
        )
    p = len(covariates)
    scale, shift = np.ones(p), np.zeros(p)
    scale_d, shift_d = 1.0, 0.0
    pos = {c: j for j, c in enumerate(covariates)}
    record: Dict[str, Tuple[float, float]] = {}
    for col, spec in policy.items():
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            a, c = float(spec[0]), float(spec[1])
        elif isinstance(spec, (int, float, np.integer, np.floating)) and not (
            isinstance(spec, bool)
        ):
            a, c = 1.0, float(spec)
        else:
            raise MethodIncompatibility(
                f"policy[{col!r}] must be a number (shift) or a (scale, shift) "
                f"pair; got {spec!r}.",
                recovery_hint="Use policy={'col': 1.0} or {'col': (2.0, 0.0)}.",
            )
        if not (np.isfinite(a) and np.isfinite(c)):
            raise MethodIncompatibility(
                f"policy[{col!r}] must be finite.",
                recovery_hint="Pass finite scale and shift values.",
            )
        if col in pos:
            scale[pos[col]], shift[pos[col]] = a, c
        elif treat is not None and col == treat:
            scale_d, shift_d = a, c
        else:
            raise MethodIncompatibility(
                f"policy names {col!r}, which is neither a covariate nor treat.",
                recovery_hint="The policy can move the covariates in "
                "covariates= and, when given, the treat column.",
            )
        record[str(col)] = (a, c)
    if all(a == 1.0 and c == 0.0 for a, c in record.values()):
        raise MethodIncompatibility(
            "policy= leaves every column unchanged; the effect is zero by "
            "construction.",
            recovery_hint="Give a non-zero shift or a scale other than 1.",
        )
    return {
        "scale": scale,
        "shift": shift,
        "scale_d": scale_d,
        "shift_d": shift_d,
        "record": record,
    }


def user_folds(
    data: pd.DataFrame, folds: Any, keep: np.ndarray, arm: Optional[np.ndarray]
) -> np.ndarray:
    """Integer fold codes from a user column / array, on the kept rows."""
    if isinstance(folds, str):
        raw = data[folds].to_numpy()
    else:
        raw = np.asarray(folds)
        if raw.ndim != 1 or raw.shape[0] != data.shape[0]:
            raise MethodIncompatibility(
                f"folds must have one label per row of data ({data.shape[0]}); "
                f"got shape {raw.shape}.",
                recovery_hint="Pass a column name or a 1-D array aligned " "with data.",
            )
    raw = raw[keep]
    if pd.isna(raw).any():
        raise MethodIncompatibility(
            "folds has missing labels.",
            recovery_hint="Give every row a fold label.",
        )
    _, codes = np.unique(raw, return_inverse=True)
    codes = codes.astype(int)
    if codes.max() < 1:
        raise MethodIncompatibility(
            "folds must contain at least two distinct labels.",
            recovery_hint="Cross-fitting needs a held-out fold.",
        )
    for f in range(codes.max() + 1):
        train = codes != f
        if arm is not None:
            n1 = int(arm[train].sum())
            if min(n1, int(train.sum()) - n1) < 2:
                raise DataInsufficient(
                    f"the training rows outside fold {f} lack one arm.",
                    recovery_hint="Each training fold needs both arms; "
                    "stratify the folds on the treatment.",
                )
        elif int(train.sum()) < 2:
            raise DataInsufficient(
                "a training fold has fewer than two rows.",
                recovery_hint="Use larger folds.",
            )
    return np.asarray(codes)
