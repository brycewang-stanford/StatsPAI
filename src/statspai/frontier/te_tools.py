"""Post-estimation helpers for technical efficiency scores.

* :func:`te_summary`  — descriptive summary (count, mean, sd, quartiles).
* :func:`te_rank`     — efficiency ranking with optional bootstrap CI.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def te_summary(result, method: Optional[str] = None) -> pd.DataFrame:
    """Return a small descriptive DataFrame of TE scores (summary stats only).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> log_k = rng.normal(0, 1, n)
    >>> log_l = rng.normal(0, 1, n)
    >>> u = rng.exponential(0.3, n)  # inefficiency
    >>> log_y = 0.4 * log_k + 0.5 * log_l + rng.normal(0, 0.1, n) - u
    >>> df = pd.DataFrame({"log_y": log_y, "log_k": log_k, "log_l": log_l})
    >>> res = sp.frontier(df, y="log_y", x=["log_k", "log_l"])
    >>> s = sp.te_summary(res)
    >>> list(s.columns)[:3]
    ['n', 'mean', 'std']
    >>> s.index.tolist()
    ['efficiency']
    """
    te = result.efficiency(method=method)
    s = pd.DataFrame(
        {
            "n": [te.size],
            "mean": [float(te.mean())],
            "std": [float(te.std(ddof=1))],
            "min": [float(te.min())],
            "q25": [float(te.quantile(0.25))],
            "median": [float(te.median())],
            "q75": [float(te.quantile(0.75))],
            "max": [float(te.max())],
            "frac_above_0_9": [float((te > 0.9).mean())],
            "frac_below_0_5": [float((te < 0.5).mean())],
        },
        index=["efficiency"],
    )
    return s


def te_rank(
    result,
    method: Optional[str] = None,
    with_ci: bool = False,
    alpha: float = 0.05,
    B: int = 500,
    seed: Optional[int] = 0,
) -> pd.DataFrame:
    """Return efficiency scores sorted descending, with rank column.

    If ``with_ci=True``, calls :meth:`FrontierResult.efficiency_ci` for
    parametric-bootstrap bounds.  For very large samples prefer a small B.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> log_k = rng.normal(0, 1, n)
    >>> log_l = rng.normal(0, 1, n)
    >>> u = rng.exponential(0.3, n)  # inefficiency
    >>> log_y = 0.4 * log_k + 0.5 * log_l + rng.normal(0, 0.1, n) - u
    >>> df = pd.DataFrame({"log_y": log_y, "log_k": log_k, "log_l": log_l})
    >>> res = sp.frontier(df, y="log_y", x=["log_k", "log_l"])
    >>> ranked = sp.te_rank(res)
    >>> list(ranked.columns)
    ['efficiency', 'rank']
    >>> int(ranked["rank"].min())
    1
    """
    te = result.efficiency(method=method)
    df = te.to_frame(name="efficiency")
    df["rank"] = df["efficiency"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("rank")
    if with_ci:
        ci = result.efficiency_ci(alpha=alpha, B=B, method=method, seed=seed)
        df = df.join(ci[["lower", "upper"]], how="left")
    return df


__all__ = ["te_summary", "te_rank"]
