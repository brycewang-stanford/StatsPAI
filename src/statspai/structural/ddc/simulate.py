"""``sp.dgp_ddc``: simulate an engine-replacement panel with known parameters."""

from __future__ import annotations

from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from ...exceptions import MethodIncompatibility
from ._core import solve_value
from ._data import rust_features

__all__ = ["dgp_ddc"]


def dgp_ddc(
    n_units: int = 100,
    n_periods: int = 100,
    n_states: int = 90,
    RC: float = 10.0,
    theta_1: float = 2.5,
    theta_2: Optional[float] = None,
    beta: float = 0.9999,
    increment_probs: Sequence[float] = (0.35, 0.60, 0.05),
    cost_scale: float = 0.001,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate a Rust (1987)-style engine replacement panel.

    Each unit starts at state 0. In every period it keeps (``replace = 0``,
    flow utility ``-c(s)``) or replaces (``replace = 1``, ``-RC - c(0)``),
    choosing by the model's choice probabilities at the true parameters;
    the state then moves up by a draw from ``increment_probs`` (measured
    from 0 after a replacement), capped at ``n_states - 1``.

    Parameters
    ----------
    n_units, n_periods : int
        Panel size.
    n_states : int, default 90
        Number of discrete states (mileage bins).
    RC, theta_1 : float
        Replacement cost and linear maintenance cost parameter.
    theta_2 : float, optional
        Quadratic cost parameter; ``None`` gives the linear model.
    beta : float, default 0.9999
        Discount factor in ``[0, 1)``.
    increment_probs : sequence of float
        Probabilities of moving up 0, 1, 2, ... states per period.
    cost_scale : float, default 0.001
        ``c(s) = theta_1 * cost_scale * s (+ theta_2 (cost_scale s)^2)``.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns ``bus``, ``t``, ``mileage`` (state code) and ``replace``.
        ``.attrs`` holds the true parameters, the transition matrices
        (``(2, S, S)`` nested list) and the true choice probabilities
        (``(S, 2)`` nested list). They are lists, not arrays, because pandas
        compares ``attrs`` when it combines frames.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_ddc(n_units=50, n_periods=40, seed=1)
    >>> sorted(df.columns)
    ['bus', 'mileage', 'replace', 't']

    References
    ----------
    [@rust1987optimal]
    """
    p = np.asarray(increment_probs, dtype=float)
    if p.ndim != 1 or p.size == 0 or np.any(p < 0) or abs(p.sum() - 1) > 1e-10:
        raise MethodIncompatibility(
            "increment_probs must be non-negative and sum to one",
            recovery_hint="e.g. increment_probs=(0.35, 0.60, 0.05)",
        )
    if not 0 <= beta < 1:
        raise MethodIncompatibility(
            "beta must be in [0, 1)", recovery_hint="pass a discount factor below one"
        )
    if n_units < 1 or n_periods < 1 or n_states < 2:
        raise MethodIncompatibility(
            "need n_units >= 1, n_periods >= 1 and n_states >= 2",
            recovery_hint="increase the panel dimensions",
        )
    cost: Literal["linear", "quadratic"] = "linear" if theta_2 is None else "quadratic"
    phi, _ = rust_features(n_states, cost, cost_scale)
    theta = np.array([RC, theta_1] + ([] if theta_2 is None else [theta_2]))
    F = np.zeros((2, n_states, n_states))
    for x in range(n_states):
        for d, pd_ in enumerate(p):
            F[0, x, min(x + d, n_states - 1)] += pd_
            F[1, x, min(d, n_states - 1)] += pd_
    _, _, P, _ = solve_value(phi @ theta, F, beta)

    rng = np.random.default_rng(seed)
    s: np.ndarray = np.zeros(n_units, dtype=np.int64)
    rows_s = np.empty((n_periods, n_units), dtype=np.int64)
    rows_a = np.empty((n_periods, n_units), dtype=np.int64)
    for t in range(n_periods):
        a = (rng.random(n_units) < P[s, 1]).astype(np.int64)
        rows_s[t], rows_a[t] = s, a
        inc = rng.choice(p.size, size=n_units, p=p)
        s = np.minimum(np.where(a == 1, 0, s) + inc, n_states - 1).astype(np.int64)
    df = pd.DataFrame(
        {
            "bus": np.tile(np.arange(n_units), n_periods),
            "t": np.repeat(np.arange(n_periods), n_units),
            "mileage": rows_s.ravel(),
            "replace": rows_a.ravel(),
        }
    ).sort_values(["bus", "t"], kind="mergesort", ignore_index=True)
    names = ["RC", "theta_1"] + ([] if theta_2 is None else ["theta_2"])
    df.attrs.update(
        {
            "true_params": dict(zip(names, theta.tolist())),
            "beta": beta,
            "increment_probs": p.tolist(),
            "cost_scale": cost_scale,
            "n_states": n_states,
            "transition": F.tolist(),
            "true_ccp": P.tolist(),
        }
    )
    return df
