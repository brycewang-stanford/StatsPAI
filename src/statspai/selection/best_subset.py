"""
Best-subset selection for linear regression.

For every model size, the subset of candidate regressors with the smallest
residual sum of squares, found exactly by branch and bound (Furnival and
Wilson 1974) rather than greedily; then one size is chosen by AIC, BIC,
adjusted R-squared or Mallows' Cp. The counterpart of R
``leaps::regsubsets(method = "exhaustive")``.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from .stepwise import SelectionResult

_MAX_CANDIDATES = 30


def best_subset(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    criterion: Literal["bic", "aic", "adjr2", "cp"] = "bic",
    max_vars: Optional[int] = None,
    force: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> SelectionResult:
    """
    Exhaustive best-subset selection for OLS.

    Finds, for each number of regressors, the subset with the smallest
    residual sum of squares, and returns the size that optimises
    ``criterion``. Unlike ``sp.stepwise`` the search is exact: no model of
    any size with a smaller RSS exists among the candidates.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    x : sequence of str
        Candidate regressors (numeric columns; expand factors into dummies
        first). At most 30; the search takes about a second at 25
        correlated candidates and grows quickly after that.
    criterion : {"bic", "aic", "adjr2", "cp"}, default "bic"
        ``aic`` and ``bic`` are those of the normal linear model, ``-2
        loglik + 2k`` and ``+ k log(n)`` with ``k`` counting the intercept
        (the same values ``sp.stepwise`` reports); ``cp`` is Mallows' Cp
        with the error variance of the model on all candidates.
    max_vars : int, optional
        Largest model size to consider (default: all candidates).
    force : sequence of str, optional
        Regressors kept in every model, such as a treatment indicator. They
        are not counted in ``size`` but do count in the criteria.
    verbose : bool, default False
        Print the table of best models by size.

    Returns
    -------
    SelectionResult
        ``selected`` and ``dropped`` for the chosen size; ``history`` has
        one row per size with ``variables``, ``rss``, ``r_squared``,
        ``adj_r_squared``, ``aic``, ``bic`` and ``cp``; ``coefficients``
        are the OLS estimates of the chosen model.

    Notes
    -----
    The coefficients, standard errors and p-values of a model picked by
    searching over subsets do not have their usual sampling distribution:
    t-statistics are inflated and R-squared is optimistic (Freedman's
    paradox: pure noise regressors screened at 25% produce a "significant"
    final model). Use the result for prediction, and for inference on a
    coefficient of interest prefer a specification fixed in advance or
    post-double-selection (``sp.rlasso_effect``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 6))
    >>> df = pd.DataFrame(X, columns=list("abcdef"))
    >>> df["y"] = 2 * df["a"] - df["c"] + rng.normal(size=300)
    >>> res = sp.best_subset(df, "y", list("abcdef"))
    >>> sorted(res.selected)
    ['a', 'c']
    >>> list(res.history["size"])
    [1, 2, 3, 4, 5, 6]

    References
    ----------
    [@furnival1974regressions]
    """
    crit = str(criterion).lower()
    if crit not in ("bic", "aic", "adjr2", "cp"):
        raise MethodIncompatibility(
            f"best_subset: criterion={criterion!r} is not 'bic', 'aic', "
            "'adjr2' or 'cp'.",
            diagnostics={"criterion": criterion},
        )
    cand = list(dict.fromkeys(x))
    forced = list(dict.fromkeys(force or []))
    cand = [c for c in cand if c not in forced]
    missing = [c for c in [y] + cand + forced if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"best_subset: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    p = len(cand)
    if p == 0:
        raise MethodIncompatibility("best_subset: no candidate regressors.")
    if p > _MAX_CANDIDATES:
        raise MethodIncompatibility(
            f"best_subset: {p} candidates; the exact search is limited to "
            f"{_MAX_CANDIDATES}.",
            recovery_hint="Screen first with sp.lasso_select or sp.stepwise.",
            diagnostics={"n_candidates": p},
        )
    frame = data[[y] + forced + cand].dropna()
    n = len(frame)
    q = 1 + len(forced)  # always-in columns: intercept and forced regressors
    if n <= q + p:
        raise DataInsufficient(
            f"best_subset: {n} complete rows for {q + p} coefficients.",
        )
    yv = frame[y].to_numpy(dtype=float)
    Z = np.column_stack([np.ones(n)] + [frame[c].to_numpy(dtype=float) for c in forced])
    Xc = frame[cand].to_numpy(dtype=float)
    # partial the always-in columns out of y and the candidates; subset RSS
    # is then a quadratic form in the residualised Gram matrix
    Qz, _ = np.linalg.qr(Z)
    yr = yv - Qz @ (Qz.T @ yv)
    Xr = Xc - Qz @ (Qz.T @ Xc)
    norms = np.sqrt(np.sum(Xr**2, axis=0))
    if np.any(norms <= 1e-10 * max(1.0, float(np.max(norms)))):
        bad = [cand[i] for i in np.flatnonzero(norms <= 1e-10 * max(1.0, norms.max()))]
        raise MethodIncompatibility(
            f"best_subset: candidate(s) {bad} are constant or collinear with "
            "the forced regressors.",
            recovery_hint="Drop them from x.",
        )
    Xn = Xr / norms
    G = Xn.T @ Xn
    g = Xn.T @ yr
    yy = float(yr @ yr)

    def rss_of(idx: Tuple[int, ...]) -> float:
        if not idx:
            return yy
        ix = np.asarray(idx)
        try:
            L = np.linalg.cholesky(G[np.ix_(ix, ix)])
        except np.linalg.LinAlgError:
            return np.inf  # an exactly collinear subset is never the best
        z = np.linalg.solve(L, g[ix])
        return float(yy - z @ z)

    full = tuple(range(p))
    rss_full = rss_of(full)
    if not np.isfinite(rss_full):
        raise MethodIncompatibility(
            "best_subset: the candidates are perfectly collinear.",
            recovery_hint="Drop the redundant column(s) from x.",
        )
    best_rss: Dict[int, float] = {k: np.inf for k in range(0, p + 1)}
    best_set: Dict[int, Tuple[int, ...]] = {}
    best_rss[p], best_set[p] = rss_full, full
    best_rss[0], best_set[0] = yy, ()

    def search(fixed: Tuple[int, ...], free: Tuple[int, ...]) -> None:
        """Models that keep ``fixed`` and drop at least one of ``free``.

        The children drop one free variable each. They are ordered by RSS,
        largest first, and the child that dropped the i-th may go on to
        drop only the later ones: the most damaging deletions get the
        largest subtrees, which the bound then prunes whole. Children are
        visited from the smallest RSS up, so good models tighten the bound
        early.
        """
        kids = sorted(
            ((rss_of(fixed + tuple(u for u in free if u != v)), v) for v in free),
            reverse=True,
        )
        ordered = tuple(v for _, v in kids)
        m = len(ordered)
        size = len(fixed) + m - 1
        for i in range(m - 1, -1, -1):
            r = kids[i][0]
            if r < best_rss[size]:
                best_rss[size] = r
                best_set[size] = fixed + ordered[:i] + ordered[i + 1 :]
            # the child's descendants have sizes len(fixed) + i .. size - 1
            # and none of them can have a smaller RSS than the child
            lowest = len(fixed) + i
            if i < m - 1 and any(r < best_rss[k] for k in range(lowest, size)):
                search(fixed + ordered[:i], ordered[i + 1 :])

    search((), full)

    top = p if max_vars is None else max(1, min(int(max_vars), p))
    tss_total = float(np.sum((yv - yv.mean()) ** 2))
    sigma2_full = rss_full / (n - q - p)
    rows: List[Dict[str, object]] = []
    for size in range(1, top + 1):
        idx = best_set[size]
        rss = best_rss[size]
        k = q + size
        ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1.0)
        r2 = 1.0 - rss / tss_total
        rows.append(
            {
                "size": size,
                "variables": [cand[j] for j in sorted(idx)],
                "rss": rss,
                "r_squared": r2,
                "adj_r_squared": 1.0 - (1.0 - r2) * (n - 1) / (n - k),
                "aic": -2 * ll + 2 * k,
                "bic": -2 * ll + k * np.log(n),
                "cp": rss / sigma2_full - n + 2 * k,
            }
        )
    history = pd.DataFrame(rows)
    col = {"bic": "bic", "aic": "aic", "adjr2": "adj_r_squared", "cp": "cp"}[crit]
    pick = int(history[col].idxmax() if crit == "adjr2" else history[col].idxmin())
    chosen = list(history.loc[pick, "variables"])
    # report candidates in the order the caller gave them
    selected = forced + [c for c in cand if c in chosen]
    dropped = [c for c in cand if c not in chosen]
    Xf = np.column_stack([np.ones(n)] + [frame[c].to_numpy(float) for c in selected])
    beta = np.linalg.lstsq(Xf, yv, rcond=None)[0]
    final = {
        "n": int(n),
        "k": int(Xf.shape[1]),
        "r_squared": float(history.loc[pick, "r_squared"]),
        "adj_r_squared": float(history.loc[pick, "adj_r_squared"]),
        "aic": float(history.loc[pick, "aic"]),
        "bic": float(history.loc[pick, "bic"]),
    }
    res = SelectionResult(
        selected=selected,
        dropped=dropped,
        history=history,
        final_model=final,
        method=f"best_subset_{crit}",
        coefficients=dict(zip(["Intercept"] + selected, map(float, beta))),
    )
    if verbose:
        print(history.drop(columns=["variables"]).to_string(index=False))
    return res
