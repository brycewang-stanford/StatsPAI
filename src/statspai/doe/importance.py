"""Which regressors matter, straight from data: ``sp.factor_importance``.

FIRST (Huang and Joseph 2025) ranks and selects the columns of a data set
by their total Sobol' index, the share of the variance of the outcome that
is lost when a column is dropped from the others, without fitting a model.
The expected conditional variance ``E[Var(y | X_S)]`` of a set of columns
``S`` is estimated from nearest neighbours in those columns: observations
that are close in ``X_S`` should have close outcomes if ``S`` explains
``y``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility


@dataclass
class FactorImportanceResult(ResultProtocolMixin):
    """Model-free importance of the columns of a data set.

    Attributes
    ----------
    importance : Series
        Total-index importance per factor; exactly zero for a factor
        that was not selected.
    selected : list of str
        Factors with positive importance, most important first.
    noise_share : float
        Share of the variance of the outcome that the selected factors
        leave unexplained.
    model_info : dict

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
    >>> df["y"] = 2 * df["a"] + np.sin(2 * df["b"]) + 0.1 * rng.normal(size=500)
    >>> res = sp.factor_importance(df, "y")
    >>> sorted(res.selected)
    ['a', 'b']
    """

    importance: pd.Series
    selected: List[str]
    noise_share: float
    model_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        info = self.model_info
        lines = [
            "Factor importance (FIRST, total Sobol' indices from data)",
            "=" * 58,
            f"Observations: {info['n_obs']}    Factors: {self.importance.size}    "
            f"Selected: {len(self.selected)}",
            self.importance.sort_values(ascending=False).to_string(
                float_format=lambda v: f"{v:.4f}"
            ),
            f"Share of variance left unexplained: {self.noise_share:.4f}",
        ]
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Bars of the importance, largest first."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.6))
        imp = self.importance.sort_values(ascending=False)
        ax.bar(range(imp.size), imp.to_numpy(), **kwargs)
        ax.set_xticks(range(imp.size))
        ax.set_xticklabels(imp.index, rotation=45, ha="right")
        ax.set_ylabel("total-index importance")
        return ax


def factor_importance(
    data: pd.DataFrame,
    y: str,
    factors: Optional[Sequence[str]] = None,
    n_neighbors: Optional[int] = None,
    n_forward: int = 2,
    standardize: bool = True,
    n_mc: Optional[int] = None,
    seed: Optional[int] = None,
) -> FactorImportanceResult:
    """Rank and select regressors by total Sobol' index, without a model.

    A screen to run before modelling: which columns carry information
    about the outcome at all, through whatever functional form and
    whatever interactions. Unlike a correlation it sees non-linear
    effects and interactions, and unlike a variable importance from a
    forest it does not depend on a fitted model.

    Parameters
    ----------
    data : DataFrame
    y : str
        The outcome. Numeric, or a 0 / 1 indicator.
    factors : list of str, optional
        Candidate columns. Default: every other column. Categorical
        columns are one-hot encoded for the neighbour search and count as
        one factor.
    n_neighbors : int, optional
        Neighbours (the observation included) used for each conditional
        variance. Default 2, or 3 when the outcome takes two values.
    n_forward : int, default 2
        Passes of forward selection. In a pass, the factor that lowers
        the conditional variance most is added, and candidates that do
        not lower it are dropped from that pass. A second pass gives the
        dropped ones another chance once more factors are in. Set it to
        the number of factors for a complete forward selection.
    standardize : bool, default True
        Scale numeric factors to unit variance before distances are
        measured.
    n_mc : int, optional
        Evaluate the conditional variances at a random subset of this
        many observations (neighbours are still sought among all).
        Default: all observations.
    seed : int, optional
        For the subset when ``n_mc`` is given.

    Returns
    -------
    FactorImportanceResult
        ``importance``, ``selected``, ``noise_share``, ``summary()``,
        ``plot()``.

    Notes
    -----
    With ``v(S)`` the nearest-neighbour estimate of ``E[Var(y | X_S)]``
    and ``S`` the selected set after forward selection and backward
    elimination, the importance of a selected factor ``i`` is
    ``(v(S without i) - v(S)) / (Var(y) - v(S))``: the share of the
    explained variance that is lost without it. Factors outside ``S``
    get zero. Without subsampling the result equals that of the R
    package ``first``.

    One deliberate difference from ``first``: for a set made of
    categorical factors only, all observations of a cell are tied
    nearest neighbours, and which ``k`` of them a tree search returns is
    arbitrary. Here the conditional variance is then the variance within
    the cell. On data with ``y = 2 a + 3 [g = v] + noise`` (the share of
    ``a`` in the explained variance is 0.67) this gives 0.62 for ``a``
    where ``first`` gives 0.55.

    The selection is greedy. Two factors that matter only through their
    product, with no effect of either on its own, can be missed: in the
    population neither lowers the conditional variance when added
    alone, so whether the first of them enters is decided by sampling
    noise.

    Read the numbers as a ranking and a selection. With correlated
    factors a total index is small for each of two factors that carry the
    same information, since either can stand in for the other. Nearest
    neighbours degrade as the number of selected factors grows; with more
    than about ten relevant factors the conditional variances are biased
    upward and weak factors are missed. The measure is about prediction:
    a factor with a large index is not thereby a cause of the outcome.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame(rng.uniform(-1, 1, size=(800, 5)),
    ...                   columns=["x1", "x2", "x3", "x4", "x5"])
    >>> df["y"] = (df["x1"] * (1 + df["x2"]) + df["x3"] ** 2
    ...            + 0.05 * rng.normal(size=800))
    >>> res = sp.factor_importance(df, "y")
    >>> sorted(res.selected)
    ['x1', 'x2', 'x3']

    References
    ----------
    huang2025factor; sobol2001global
    """
    from scipy.spatial import cKDTree

    if y not in data.columns:
        raise ColumnNotFound(f"Outcome column {y!r} is not in the data.")
    cols = [c for c in data.columns if c != y] if factors is None else list(factors)
    miss = [str(c) for c in cols if c not in data.columns]
    if miss:
        raise ColumnNotFound(f"Not in the data: {', '.join(miss)}.")
    if not cols:
        raise MethodIncompatibility("There is no candidate factor.")
    use = data[[y] + cols].dropna()
    notes: List[str] = []
    if len(use) < len(data):
        notes.append(f"{len(data) - len(use)} rows with missing values dropped.")
    yy = use[y].to_numpy(dtype=float)
    N = yy.size
    if N < 20:
        raise DataInsufficient(f"{N} complete observations are too few.")
    vy = float(np.var(yy, ddof=1))
    if vy <= 0:
        raise MethodIncompatibility("The outcome does not vary.")
    k = (
        int(n_neighbors)
        if n_neighbors is not None
        else (3 if np.unique(yy).size == 2 else 2)
    )
    if not 2 <= k <= N:
        raise MethodIncompatibility("n_neighbors must be at least 2.")
    if n_forward < 1:
        raise MethodIncompatibility("n_forward must be at least 1.")
    blocks: List[np.ndarray] = []
    labels: Dict[int, np.ndarray] = {}
    for pos, c in enumerate(cols):
        col = use[c]
        if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
            blocks.append(pd.get_dummies(col).to_numpy(dtype=float))
            labels[pos] = pd.factorize(col)[0]
        else:
            num = col.to_numpy(dtype=float)[:, None]
            sd = float(num.std(ddof=1))
            if sd <= 0:
                raise MethodIncompatibility(f"Factor {c!r} is constant.")
            blocks.append((num - num.mean()) / sd if standardize else num)
    rng = np.random.default_rng(seed)
    at = np.arange(N)
    if n_mc is not None and int(n_mc) < N:
        at = np.sort(rng.choice(N, size=max(int(n_mc), 20), replace=False))
        notes.append(
            f"Conditional variances evaluated at {at.size} of {N} observations."
        )
    cache: Dict[frozenset, float] = {}

    def v(S: Sequence[int]) -> float:
        key = frozenset(S)
        if not key:
            return vy
        if key not in cache:
            if all(j in labels for j in key):
                # only categorical factors: every observation of a cell is
                # an equally near neighbour, so the conditional variance is
                # the variance within the cell, not that of k arbitrary
                # members of it
                cell = pd.Series(yy[at]).groupby([labels[j][at] for j in sorted(key)])
                var_c, n_c = cell.var(ddof=1).fillna(0.0), cell.size()
                cache[key] = float((var_c * n_c).sum() / n_c.sum())
            else:
                Z = np.hstack([blocks[j] for j in sorted(key)])
                _, idx = cKDTree(Z).query(Z[at], k=k)
                cache[key] = float(np.mean(np.var(yy[idx], axis=1, ddof=1)))
        return cache[key]

    p = len(cols)
    S: List[int] = []
    cur = vy
    for _ in range(int(n_forward)):
        cand = [j for j in range(p) if j not in S]
        added = False
        while cand:
            vals = {j: v(S + [j]) for j in cand}
            # early dropping: a candidate that does not lower the
            # conditional variance is out for the rest of this pass
            cand = [j for j in cand if vals[j] < cur]
            if not cand:
                break
            best = min(cand, key=lambda j: vals[j])
            S.append(best)
            cur = vals[best]
            cand.remove(best)
            added = True
        if not added:
            break
    while len(S) > 1:
        vals = {j: v([s for s in S if s != j]) for j in S}
        best = min(vals, key=lambda j: vals[j])
        if vals[best] > cur:
            break
        S.remove(best)
        cur = vals[best]
    imp = np.zeros(p)
    explained = vy - cur
    if S and explained > 0:
        for j in S:
            imp[j] = max(v([s for s in S if s != j]) - cur, 0.0) / explained
    names = [str(c) for c in cols]
    series = pd.Series(imp, index=names, name="importance")
    chosen = [nm for nm in series.sort_values(ascending=False).index if series[nm] > 0]
    if not S:
        notes.append("No factor lowers the conditional variance of the outcome.")
    return FactorImportanceResult(
        importance=series,
        selected=chosen,
        noise_share=float(cur / vy),
        model_info={
            "n_obs": int(N),
            "n_neighbors": k,
            "n_forward": int(n_forward),
            "standardize": bool(standardize),
            "notes": notes,
        },
    )
