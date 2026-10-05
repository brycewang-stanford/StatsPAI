"""Order selection for ARIMA models: the search of Hyndman and Khandakar
(2008).

The orders of differencing are settled by unit-root and seasonal-strength
rules before any likelihood is compared; the remaining orders and the
constant are chosen by AICc, either over a neighbourhood that moves with
the best model so far (stepwise) or over every model up to a total order.

The module is written from the published algorithm. It only decides
*which* models to fit; fitting is the caller's ``fit`` function.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

Key = Tuple[int, int, int, int, bool]  # p, q, P, Q, constant
FitFn = Callable[[int, int, int, int, bool], float]


def search(
    fit: FitFn,
    *,
    seasonal: bool,
    allow_constant: bool,
    max_p: int,
    max_q: int,
    max_P: int,
    max_Q: int,
    max_order: int = 5,
    stepwise: bool = True,
    max_models: int = 94,
) -> Tuple[Optional[Key], List[Tuple[Key, float]]]:
    """Return the best ``(p, q, P, Q, constant)`` and the models tried.

    ``fit`` returns the AICc of a model, ``inf`` when it cannot be fitted
    or is rejected.
    """
    tried: Dict[Key, float] = {}

    def score(p: int, q: int, P: int, Q: int, c: bool) -> float:
        if p < 0 or q < 0 or P < 0 or Q < 0:
            return np.inf
        if p > max_p or q > max_q or P > max_P or Q > max_Q:
            return np.inf
        if c and not allow_constant:
            return np.inf
        key = (p, q, P, Q, bool(c))
        if key not in tried:
            if len(tried) >= max_models:
                return np.inf
            tried[key] = float(fit(p, q, P, Q, bool(c)))
        return tried[key]

    if not stepwise:
        ps = range(max_P + 1) if seasonal else range(1)
        qs = range(max_Q + 1) if seasonal else range(1)
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                for P in ps:
                    for Q in qs:
                        if p + q + P + Q > max_order:
                            continue
                        for c in (True, False) if allow_constant else (False,):
                            tried.setdefault((p, q, P, Q, c), float(fit(p, q, P, Q, c)))
    else:
        c0 = allow_constant
        s1 = 1 if seasonal else 0
        start = [
            (min(2, max_p), min(2, max_q), min(s1, max_P), min(s1, max_Q), c0),
            (0, 0, 0, 0, c0),
            (min(1, max_p), 0, min(s1, max_P), 0, c0),
            (0, min(1, max_q), 0, min(s1, max_Q), c0),
        ]
        if allow_constant:
            start.append((0, 0, 0, 0, False))
        best: Optional[Key] = None
        best_val = np.inf
        for key in start:
            val = score(*key)
            if val < best_val:
                best, best_val = key, val
        improved = best is not None
        while improved and best is not None:
            improved = False
            p, q, P, Q, c = best
            moves: List[Key] = []
            if seasonal:
                moves += [
                    (p, q, P - 1, Q, c),
                    (p, q, P, Q - 1, c),
                    (p, q, P + 1, Q, c),
                    (p, q, P, Q + 1, c),
                    (p, q, P - 1, Q - 1, c),
                    (p, q, P - 1, Q + 1, c),
                    (p, q, P + 1, Q - 1, c),
                    (p, q, P + 1, Q + 1, c),
                ]
            moves += [
                (p - 1, q, P, Q, c),
                (p, q - 1, P, Q, c),
                (p + 1, q, P, Q, c),
                (p, q + 1, P, Q, c),
                (p - 1, q - 1, P, Q, c),
                (p - 1, q + 1, P, Q, c),
                (p + 1, q - 1, P, Q, c),
                (p + 1, q + 1, P, Q, c),
            ]
            if allow_constant:
                moves.append((p, q, P, Q, not c))
            for key in moves:
                val = score(*key)
                if val < best_val:
                    best, best_val = key, val
                    improved = True
                    break
    finite = [(k, v) for k, v in tried.items() if np.isfinite(v)]
    if not finite:
        return None, list(tried.items())
    finite.sort(key=lambda kv: kv[1])
    return finite[0][0], sorted(tried.items(), key=lambda kv: kv[1])
