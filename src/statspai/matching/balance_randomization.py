"""Covariate balance of a matched sample against a randomized benchmark.

A matched comparison is meant to resemble, on the observed covariates, the
experiment that was not run. :func:`balance_vs_randomization` makes the
resemblance literal: it re-randomizes the treatment labels of the matched
sample many times, computes the same balance tests in each simulated
completely randomized experiment, and reports how often randomization
would have balanced the covariates better than the match did.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import require_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility


@dataclass
class BalanceRandomizationResult(ResultProtocolMixin):
    """Result of :func:`balance_vs_randomization`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per covariate and three summary rows (``min_p``,
        ``truncated_product``, ``n_below_alpha``). ``actual`` is the value
        in the matched sample, ``sim_median`` its median over the simulated
        randomized experiments, and ``share_better`` the share of those
        experiments that were better balanced on that row.
    sim : pandas.DataFrame
        The simulated values, one row per simulated experiment.
    n_sim : int
        Number of simulated experiments.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"z": np.tile([1, 0], 100),
    ...                    "age": rng.normal(50, 10, 200),
    ...                    "female": rng.integers(0, 2, 200)})
    >>> res = sp.balance_vs_randomization(df, "z", ["age", "female"],
    ...                                   n_sim=200, random_state=1)
    >>> res.table.index.tolist()
    ['age', 'female', 'min_p', 'truncated_product', 'n_below_alpha']
    """

    _citation_keys = ("hansen2008covariate",)

    table: pd.DataFrame
    sim: pd.DataFrame
    n_sim: int
    test: str
    trunc: float
    alpha: float
    n_treated: int
    n_control: int
    tests_used: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Covariate balance compared with complete randomization",
            "======================================================",
            f"Treated / control : {self.n_treated} / {self.n_control}",
            f"Simulated experiments : {self.n_sim}",
            "",
            self.table.to_string(float_format=lambda v: f"{v:.4f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        worst = float(self.table.loc["min_p", "share_better"])
        return (
            f"BalanceRandomizationResult(n_sim={self.n_sim}, "
            f"share_better[min_p]={worst:.3f})"
        )


def _welch_p(x: np.ndarray, zmat: np.ndarray) -> np.ndarray:
    n1 = zmat.sum(axis=1)
    n0 = zmat.shape[1] - n1
    s1 = zmat @ x
    q1 = zmat @ (x * x)
    s0 = x.sum() - s1
    q0 = (x * x).sum() - q1
    m1, m0 = s1 / n1, s0 / n0
    v1 = np.maximum(q1 - n1 * m1**2, 0.0) / (n1 - 1)
    v0 = np.maximum(q0 - n0 * m0**2, 0.0) / (n0 - 1)
    se2 = v1 / n1 + v0 / n0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (m1 - m0) / np.sqrt(se2)
        dof = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v0 / n0) ** 2 / (n0 - 1))
        p = 2.0 * stats.t.sf(np.abs(t), dof)
    return np.where(se2 > 0, p, 1.0)


def _wilcoxon_p(x: np.ndarray, zmat: np.ndarray) -> np.ndarray:
    """Two-sided rank-sum p-value by the rule of R's ``wilcox.test``: exact
    when both groups have fewer than 50 observations and there are no
    ties, otherwise the normal approximation with a continuity correction
    and the variance corrected for ties."""
    n = x.size
    n1 = int(zmat[0].sum())
    n0 = n - n1
    ranks = stats.rankdata(x, method="average")
    _, counts = np.unique(x, return_counts=True)
    ties = counts.max() > 1
    w = zmat @ ranks - n1 * (n1 + 1) / 2.0
    if n1 < 50 and n0 < 50 and not ties:
        out = np.empty(len(w))
        for i, row in enumerate(zmat.astype(bool)):
            out[i] = stats.mannwhitneyu(
                x[row], x[~row], alternative="two-sided", method="exact"
            ).pvalue
        return out
    sigma2 = n1 * n0 / 12.0 * ((n + 1) - np.sum(counts**3 - counts) / (n * (n - 1)))
    if sigma2 <= 0:
        return np.ones(len(w))
    centred = w - n1 * n0 / 2.0
    zstat = (centred - np.sign(centred) * 0.5) / np.sqrt(sigma2)
    return np.asarray(np.minimum(1.0, 2.0 * stats.norm.sf(np.abs(zstat))))


def _chisq_p(x: np.ndarray, zmat: np.ndarray) -> np.ndarray:
    """Pearson chi-square test of the covariate-by-group table, with the
    Yates correction when the table is 2 x 2 (R's ``chisq.test``)."""
    levels, codes = np.unique(x, return_inverse=True)
    k = len(levels)
    if k < 2:
        return np.ones(zmat.shape[0])
    onehot = np.eye(k)[codes]
    n = x.size
    total = onehot.sum(axis=0)
    n1 = zmat.sum(axis=1, keepdims=True)
    obs1 = zmat @ onehot
    obs0 = total[None, :] - obs1
    exp1 = n1 * total[None, :] / n
    exp0 = (n - n1) * total[None, :] / n
    d1, d0 = np.abs(obs1 - exp1), np.abs(obs0 - exp0)
    if k == 2:
        shrink = np.minimum(0.5, np.minimum(d1, d0).min(axis=1, keepdims=True))
        d1, d0 = d1 - shrink, d0 - shrink
    chi2 = (d1**2 / exp1).sum(axis=1) + (d0**2 / exp0).sum(axis=1)
    return np.asarray(stats.chi2.sf(chi2, k - 1))


def balance_vs_randomization(
    data: pd.DataFrame,
    treat: str,
    covariates: Sequence[str],
    *,
    n_sim: int = 1000,
    test: str = "auto",
    max_levels: int = 2,
    trunc: float = 0.2,
    alpha: float = 0.05,
    random_state: Optional[int] = None,
) -> BalanceRandomizationResult:
    """Compare covariate balance in a matched sample with randomization.

    For each covariate a two-sample test compares the treated and control
    groups of the matched sample. The treatment labels are then permuted
    ``n_sim`` times, as if the same individuals had been assigned at
    random with the same group sizes, and the tests are repeated. A match
    whose p-values look like those of the simulated experiments, or
    larger, has balanced the observed covariates about as well as
    randomization would have [@hansen2008covariate; @pimentel2015large].

    The p-values serve as a yardstick here. They are not tests of a
    hypothesis anyone holds: a matched sample was not randomized, and
    matching commonly balances observed covariates *better* than
    randomization does.

    Parameters
    ----------
    data : DataFrame
        The matched sample, one row per individual.
    treat : str
        0/1 treatment indicator.
    covariates : list of str
        Numeric covariates to assess.
    n_sim : int, default 1000
        Number of simulated randomized experiments.
    test : {"auto", "wilcoxon", "t"}, default "auto"
        ``"wilcoxon"`` is the two-sample rank-sum test, ``"t"`` the
        unequal-variance t-test. ``"auto"`` uses the chi-square test for a
        covariate with at most ``max_levels`` distinct values and the
        rank-sum test otherwise.
    max_levels : int, default 2
        See ``test``.
    trunc : float, default 0.2
        Truncation point for the product of the p-values, which summarises
        balance across covariates; ``1`` gives Fisher's product.
    alpha : float, default 0.05
        Threshold for counting small p-values.
    random_state : int, optional
        Seed for the permutations.

    Returns
    -------
    BalanceRandomizationResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"z": np.tile([1, 0], 100),
    ...                    "age": rng.normal(50, 10, 200),
    ...                    "female": rng.integers(0, 2, 200)})
    >>> res = sp.balance_vs_randomization(df, "z", ["age", "female"],
    ...                                   n_sim=200, random_state=1)
    >>> res.tests_used
    {'age': 'wilcoxon', 'female': 'chi2'}
    >>> bool(0 <= res.table.loc["min_p", "share_better"] <= 1)
    True

    References
    ----------
    [@hansen2008covariate], [@pimentel2015large], [@yu2021evaluating],
    [@zaykin2002truncated]
    """
    if test not in {"auto", "wilcoxon", "t"}:
        raise MethodIncompatibility("test must be 'auto', 'wilcoxon' or 't'")
    if not 0 < trunc <= 1:
        raise MethodIncompatibility("trunc must satisfy 0 < trunc <= 1")
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must lie strictly between 0 and 1")
    if int(n_sim) < 1:
        raise MethodIncompatibility("n_sim must be a positive integer")
    cols = list(covariates)
    if not cols:
        raise MethodIncompatibility("balance_vs_randomization: covariates is empty")
    require_columns(data, [treat] + cols, function="balance_vs_randomization")
    d = data[[treat] + cols].dropna()
    z = d[treat].to_numpy(dtype=float)
    if not np.all((z == 0) | (z == 1)):
        raise MethodIncompatibility(
            "balance_vs_randomization: the treat column must be coded 0/1"
        )
    n1, n0 = int(z.sum()), int((1 - z).sum())
    if min(n1, n0) < 2:
        raise DataInsufficient(
            "balance_vs_randomization: need at least two treated and two "
            "control individuals.",
            recovery_hint="Check the treat column.",
            diagnostics={"n_treated": n1, "n_control": n0},
            alternative_functions=[],
        )

    rng = np.random.default_rng(random_state)
    zsim = rng.permuted(np.tile(z, (int(n_sim), 1)), axis=1)
    zall = np.vstack([z[None, :], zsim])

    pvals = np.empty((zall.shape[0], len(cols)))
    used = {}
    for j, c in enumerate(cols):
        x = d[c].to_numpy(dtype=float)
        if test == "t":
            kind = "t"
        elif test == "wilcoxon" or np.unique(x).size > max_levels:
            kind = "wilcoxon"
        else:
            kind = "chi2"
        used[c] = kind
        fn = {"t": _welch_p, "wilcoxon": _wilcoxon_p, "chi2": _chisq_p}[kind]
        pvals[:, j] = fn(x, zall)

    min_p = pvals.min(axis=1)
    product = np.where(pvals <= trunc, pvals, 1.0).prod(axis=1)
    count = (pvals <= alpha).sum(axis=1).astype(float)
    full = np.column_stack([pvals, min_p, product, count])
    names: List[str] = cols + ["min_p", "truncated_product", "n_below_alpha"]
    actual, sim = full[0], full[1:]
    better = (sim > actual[None, :]).mean(axis=0)
    better[-1] = float((sim[:, -1] < actual[-1]).mean())
    table = pd.DataFrame(
        {
            "actual": actual,
            "sim_median": np.median(sim, axis=0),
            "share_better": better,
        },
        index=names,
    )
    return BalanceRandomizationResult(
        table=table,
        sim=pd.DataFrame(sim, columns=names),
        n_sim=int(n_sim),
        test=test,
        trunc=float(trunc),
        alpha=float(alpha),
        n_treated=n1,
        n_control=n0,
        tests_used=used,
    )


__all__: Tuple[str, ...] = ("balance_vs_randomization", "BalanceRandomizationResult")
