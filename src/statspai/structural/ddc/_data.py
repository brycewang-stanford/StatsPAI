"""Input handling for ``sp.ddc``: states, actions, utility features, transitions."""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ...exceptions import (
    AssumptionWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)


def encode_panel(
    data: pd.DataFrame,
    choice: str,
    state: str,
    id: Optional[str],
    time: Optional[str],
    actions: Optional[Sequence[Any]],
    n_states: Optional[int],
) -> Dict[str, Any]:
    """Integer state / action codes, in panel order when ``id`` is given."""
    cols = [c for c in (choice, state, id, time) if c is not None]
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise DataInsufficient(
            f"columns not found in data: {missing}",
            recovery_hint="pass existing column names for choice / state / id / time",
        )
    if data[cols].isna().to_numpy().any():
        raise DataInsufficient(
            "missing values in the choice, state, id or time columns",
            recovery_hint="drop or impute the incomplete rows before calling sp.ddc",
        )
    if len(data) == 0:
        raise DataInsufficient(
            "no observations", recovery_hint="pass a non-empty panel"
        )
    df = data[cols].reset_index(drop=True)
    if id is not None:
        order = [id] + ([time] if time is not None else [])
        df = df.sort_values(order, kind="mergesort").reset_index(drop=True)

    s_raw = df[state].to_numpy()
    if not np.all(np.equal(np.mod(s_raw, 1), 0)) or np.min(s_raw) < 0:
        raise MethodIncompatibility(
            "the state column must hold non-negative integer codes 0, 1, ..., S-1",
            recovery_hint=(
                "discretise the state first, e.g. with pd.cut(...).cat.codes"
            ),
        )
    s = s_raw.astype(np.int64)
    S = int(s.max()) + 1 if n_states is None else int(n_states)
    if s.max() >= S:
        raise MethodIncompatibility(
            f"state code {int(s.max())} is outside 0..{S - 1}",
            recovery_hint="raise n_states or recode the state column",
        )

    labels = list(pd.unique(df[choice])) if actions is None else list(actions)
    if actions is None:
        labels = sorted(labels)
    if len(labels) < 2:
        raise MethodIncompatibility(
            "a dynamic discrete choice model needs at least two actions",
            recovery_hint=(
                "check the choice column; with one action nothing is identified"
            ),
        )
    lookup = {lab: j for j, lab in enumerate(labels)}
    try:
        a = np.array([lookup[v] for v in df[choice].to_numpy()], dtype=np.int64)
    except KeyError as exc:
        raise MethodIncompatibility(
            f"choice value {exc.args[0]!r} is not in actions={labels}",
            recovery_hint="list every observed action in `actions`",
        ) from None

    nxt = None
    if id is not None:
        g = df[id].to_numpy()
        same = np.r_[g[1:] == g[:-1], False]
        if time is not None:
            t = df[time].to_numpy()
            dupl = np.r_[same[:-1] & (t[1:] == t[:-1]), False]
            if dupl.any():
                raise DataInsufficient(
                    "duplicate (id, time) rows",
                    recovery_hint="keep one observation per unit and period",
                )
            consecutive = np.r_[same[:-1] & (t[1:] - t[:-1] == 1), False]
        else:
            consecutive = same
        nxt = np.where(consecutive, np.r_[s[1:], -1], -1)

    counts = np.zeros((S, len(labels)))
    np.add.at(counts, (s, a), 1.0)
    return {
        "s": s,
        "a": a,
        "next_state": nxt,
        "ids": None if id is None else df[id].to_numpy(),
        "S": S,
        "labels": labels,
        "counts": counts,
    }


def rust_features(S: int, cost: str, cost_scale: float) -> Tuple[np.ndarray, List[str]]:
    """Rust (1987) engine replacement: action 0 keeps, action 1 replaces.

    ``u(x, keep) = -c(x)``, ``u(x, replace) = -RC - c(0)`` with
    ``c(x) = theta_1 (scale * x)`` (+ ``theta_2 (scale * x)^2``).
    """
    x = cost_scale * np.arange(S, dtype=float)
    if cost == "linear":
        names = ["RC", "theta_1"]
        cols = [x]
    elif cost == "quadratic":
        names = ["RC", "theta_1", "theta_2"]
        cols = [x, x**2]
    else:
        raise MethodIncompatibility(
            f"cost={cost!r} is not supported",
            recovery_hint="use cost='linear' or cost='quadratic'",
        )
    phi = np.zeros((S, 2, len(names)))
    for k, c in enumerate(cols, start=1):
        phi[:, 0, k] = -c
        phi[:, 1, k] = -c[0]
    phi[:, 1, 0] = -1.0
    return phi, names


def build_features(
    utility: Any,
    S: int,
    J: int,
    cost: str,
    cost_scale: float,
    param_names: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    if isinstance(utility, str):
        if utility != "rust":
            raise MethodIncompatibility(
                f"utility={utility!r} is not a built-in specification",
                recovery_hint="use 'rust', an array of shape (S, J, K) or a callable",
            )
        if J != 2:
            raise MethodIncompatibility(
                "utility='rust' needs exactly two actions (keep, replace)",
                recovery_hint="pass actions=[keep_label, replace_label]",
            )
        phi, names = rust_features(S, cost, cost_scale)
    elif callable(utility):
        fn: Callable = utility
        rows = [
            [np.atleast_1d(np.asarray(fn(s, j), dtype=float)) for j in range(J)]
            for s in range(S)
        ]
        phi = np.array(rows, dtype=float)
        names = None
    else:
        phi = np.asarray(utility, dtype=float)
        names = None
    if phi.ndim != 3 or phi.shape[:2] != (S, J):
        raise MethodIncompatibility(
            f"utility features must have shape (S, J, K) = ({S}, {J}, K); "
            f"got {phi.shape}",
            recovery_hint="build one feature vector per (state, action) pair",
        )
    if not np.all(np.isfinite(phi)):
        raise MethodIncompatibility(
            "utility features contain NaN or inf",
            recovery_hint="check the feature array",
        )
    K = phi.shape[2]
    if names is None:
        names = (
            list(param_names)
            if param_names is not None
            else [f"theta_{k}" for k in range(K)]
        )
    if len(names) != K:
        raise MethodIncompatibility(
            f"{len(names)} param_names for {K} utility parameters",
            recovery_hint="give one name per column of the feature array",
        )
    # Only utility differences across actions are identified.
    D = (phi[:, 1:, :] - phi[:, :1, :]).reshape(-1, K)
    if np.linalg.matrix_rank(D, tol=1e-10 * max(1.0, np.abs(D).max())) < K:
        raise IdentificationFailure(
            "utility parameters are not identified: some combination of the "
            "features is the same for every action",
            recovery_hint=(
                "drop features that do not vary across actions; only utility "
                "differences between actions are identified"
            ),
        )
    return phi, list(names)


def _absorbing_fill(C: np.ndarray, label: str, diag: Dict[str, Any]) -> np.ndarray:
    n = C.sum(axis=1)
    empty = np.flatnonzero(n == 0)
    F = np.zeros_like(C)
    F[n > 0] = C[n > 0] / n[n > 0, None]
    F[empty, empty] = 1.0
    if empty.size:
        diag.setdefault("transition_rows_without_data", {})[label] = empty.tolist()
    return F


def estimate_transitions(
    enc: Dict[str, Any],
    mode: str,
    reset_actions: Sequence[int],
    diag: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """First-stage transition estimates from consecutive panel rows."""
    nxt = enc["next_state"]
    if nxt is None:
        raise MethodIncompatibility(
            "estimating transitions needs the panel structure",
            recovery_hint="pass id= (and time=), or supply transition matrices",
        )
    s, a, S, J = enc["s"], enc["a"], enc["S"], len(enc["labels"])
    ok = nxt >= 0
    if ok.sum() == 0:
        raise DataInsufficient(
            "no consecutive observations to estimate transitions from",
            recovery_hint="each unit needs at least two consecutive periods",
        )
    params: Dict[str, Any] = {"n_transitions": int(ok.sum())}
    if mode == "estimate":
        F = np.zeros((J, S, S))
        for j in range(J):
            C = np.zeros((S, S))
            sel = ok & (a == j)
            np.add.at(C, (s[sel], nxt[sel]), 1.0)
            F[j] = _absorbing_fill(C, str(enc["labels"][j]), diag)
        if diag.get("transition_rows_without_data"):
            warnings.warn(
                AssumptionWarning(
                    "some (action, state) pairs are never followed by an observed "
                    "transition; those rows were set to stay put",
                    recovery_hint=(
                        "use transition='increments' or supply transition matrices"
                    ),
                    diagnostics={"rows": diag["transition_rows_without_data"]},
                ),
                stacklevel=3,
            )
        return F, params
    # Rust's first stage: one distribution of increments, pooled over states,
    # measured from the state the action leads to (0 after a reset).
    reset = np.isin(a, np.asarray(list(reset_actions), dtype=np.int64))
    base = np.where(reset, 0, s)
    inc = nxt[ok] - base[ok]
    if inc.min() < 0:
        raise MethodIncompatibility(
            "negative state increments; transition='increments' assumes the state "
            "only moves up (and restarts at 0 after a reset action)",
            recovery_hint="use transition='estimate' or supply transition matrices",
        )
    support = np.arange(inc.max() + 1)
    p = np.bincount(inc, minlength=support.size) / inc.size
    params.update({"increments": support.tolist(), "increment_probs": p.tolist()})
    F = np.zeros((J, S, S))
    for j in range(J):
        for x in range(S):
            start = 0 if j in reset_actions else x
            for d, pd_ in zip(support, p):
                F[j, x, min(start + d, S - 1)] += pd_
    return F, params


def check_transitions(F: Any, S: int, J: int, labels: List[Any]) -> np.ndarray:
    if isinstance(F, dict):
        try:
            F = np.stack([np.asarray(F[lab], dtype=float) for lab in labels])
        except KeyError as exc:
            raise MethodIncompatibility(
                f"no transition matrix for action {exc.args[0]!r}",
                recovery_hint="give one (S, S) matrix per action label",
            ) from None
    F = np.asarray(F, dtype=float)
    if F.shape != (J, S, S):
        raise MethodIncompatibility(
            f"transition must have shape (J, S, S) = ({J}, {S}, {S}); got {F.shape}",
            recovery_hint="stack one row-stochastic (S, S) matrix per action",
        )
    if np.any(F < -1e-12) or np.max(np.abs(F.sum(axis=2) - 1.0)) > 1e-8:
        raise MethodIncompatibility(
            "transition matrices must be row-stochastic",
            recovery_hint="each row must be non-negative and sum to one",
        )
    return np.asarray(np.clip(F, 0.0, None))
