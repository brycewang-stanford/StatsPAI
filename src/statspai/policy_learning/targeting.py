"""Budget-constrained treatment targeting from CATE estimates.

Turns "who benefits most?" estimates into an explicit assignment rule
under a capacity constraint — the "from report-writing to
decision-making" step: rank units by predicted effect, treat down the
ranking until the budget runs out, and report the expected gain
against the treat-all and random-assignment baselines.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

__all__ = ["policy_targeting"]


def policy_targeting(
    cate: Any,
    *,
    budget: Optional[int] = None,
    frac: Optional[float] = None,
    min_effect: float = 0.0,
) -> Dict[str, Any]:
    """Rank-and-treat policy under a budget constraint.

    Parameters
    ----------
    cate : array-like, CausalResult, or fitted CATE model
        Per-unit effect estimates — a raw array, a ``metalearner()`` /
        ``tarnet()`` result, or a fitted ``causal_forest()`` model
        (training-sample effects are used).
    budget : int, optional
        Maximum number of units that can be treated.  Mutually
        exclusive with ``frac``.
    frac : float, optional
        Maximum *fraction* of units that can be treated (in ``(0, 1]``).
    min_effect : float, default 0.0
        Never treat a unit whose predicted effect is at or below this
        threshold, even with budget left over — treating predicted
        non-responders wastes budget and can do harm.

    Returns
    -------
    dict
        ``policy`` (0/1 array in input order), ``n_treated``,
        ``threshold`` (smallest predicted effect among the treated),
        ``expected_gain`` under the policy, and the ``expected_gain_*``
        baselines (``treat_all``, ``random`` at the same budget), plus a
        one-row ``summary`` DataFrame.  Gains are sums of predicted
        effects — validate against ``sp.policy_value`` with doubly
        robust scores before deployment.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> tau = np.array([2.0, 1.0, 0.5, -0.5, -2.0])
    >>> out = sp.policy_targeting(tau, budget=2)
    >>> out["policy"].tolist()
    [1, 1, 0, 0, 0]
    >>> out["expected_gain"]
    3.0
    >>> out["expected_gain_treat_all"]
    1.0
    >>> # budget larger than the number of positive effects: the
    >>> # min_effect guard stops at 3 treated units
    >>> sp.policy_targeting(tau, budget=5)["n_treated"]
    3
    """
    from ..metalearners.diagnostics import _extract_cate

    tau = _extract_cate(cate)
    n = len(tau)
    if n == 0:
        raise MethodIncompatibility(
            "policy_targeting() received an empty CATE array.",
            recovery_hint="Pass one effect estimate per unit.",
        )
    if not np.all(np.isfinite(tau)):
        raise MethodIncompatibility(
            "policy_targeting() requires finite CATE estimates; got "
            f"{int(np.sum(~np.isfinite(tau)))} non-finite value(s).",
            recovery_hint="Drop or re-estimate rows with NaN effects.",
        )

    if budget is not None and frac is not None:
        raise MethodIncompatibility(
            "Pass either budget= or frac=, not both.",
            recovery_hint="Drop one of the two constraints.",
        )
    if frac is not None:
        if not 0.0 < float(frac) <= 1.0:
            raise MethodIncompatibility(
                f"frac must be in (0, 1]; got {frac!r}.",
                recovery_hint="Use e.g. frac=0.5 for a half-capacity budget.",
            )
        budget = int(np.floor(float(frac) * n))
    if budget is None:
        budget = n
    budget = int(budget)
    if budget < 0:
        raise MethodIncompatibility(
            f"budget must be non-negative; got {budget}.",
            recovery_hint="Pass the number of units you can treat.",
        )
    budget = min(budget, n)

    # Rank by predicted effect (stable ⇒ deterministic under ties),
    # stop at the budget or at the min_effect guard, whichever binds.
    order = np.argsort(-tau, kind="stable")
    eligible = tau[order] > float(min_effect)
    n_treated = int(min(budget, int(eligible.sum())))
    chosen = order[:n_treated]

    policy = np.zeros(n, dtype=int)
    policy[chosen] = 1

    expected_gain = float(tau[chosen].sum())
    gain_all = float(tau.sum())
    gain_random = float(budget * tau.mean())
    threshold = float(tau[chosen].min()) if n_treated > 0 else float("nan")

    summary = pd.DataFrame(
        {
            "n": [n],
            "budget": [budget],
            "n_treated": [n_treated],
            "threshold": [threshold],
            "expected_gain": [expected_gain],
            "expected_gain_treat_all": [gain_all],
            "expected_gain_random": [gain_random],
        }
    )
    return {
        "policy": policy,
        "n_treated": n_treated,
        "budget": budget,
        "threshold": threshold,
        "min_effect": float(min_effect),
        "expected_gain": expected_gain,
        "expected_gain_treat_all": gain_all,
        "expected_gain_random": gain_random,
        "summary": summary,
    }
