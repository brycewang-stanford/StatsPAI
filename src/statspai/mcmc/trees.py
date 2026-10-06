"""
Bayesian additive regression trees: ``sp.bart``.

    y_i = sum_{j=1}^{m} g(x_i; T_j, M_j) + e_i,     e_i ~ N(0, sigma^2)

A sum of many small regression trees, each kept small by its prior so
that no single tree explains much. The posterior is sampled by Bayesian
backfitting: one tree at a time is changed by a birth or death move
against the residual of the others, with the leaf means integrated out,
and then its leaf means are redrawn.

The engine is written from the model as published. It does not use or
port the code of any BART package.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats

from .._result_serialize import ResultProtocolMixin
from ..core.utils import create_design_matrices
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._core import design_for, rtruncnorm

_KERNEL: Dict[str, Any] = {}

# Node states in the heap layout (children of i are 2i + 1 and 2i + 2)
_ABSENT = -2
_LEAF = -1


def _depth(node: int) -> int:
    d = 0
    while node > 0:
        node = (node - 1) // 2
        d += 1
    return d


def _bounds(
    var: np.ndarray, cut: np.ndarray, node: int, v: int, ncut_v: int
) -> Tuple[int, int]:
    """Cut indices ``lo .. hi - 1`` of variable ``v`` still available at
    ``node`` given the splits of its ancestors."""
    lo = 0
    hi = ncut_v
    child = node
    while child > 0:
        parent = (child - 1) // 2
        if var[parent] == v:
            if child == 2 * parent + 1:
                if cut[parent] < hi:
                    hi = cut[parent]
            else:
                if cut[parent] + 1 > lo:
                    lo = cut[parent] + 1
        child = parent
    return lo, hi


def _n_avail(
    var: np.ndarray, cut: np.ndarray, node: int, ncut: np.ndarray, bounds: Any
) -> int:
    count = 0
    for v in range(ncut.shape[0]):
        lo, hi = bounds(var, cut, node, v, ncut[v])
        if hi > lo:
            count += 1
    return count


def _logm(n: float, s: float, sigma2: float, tau2: float) -> float:
    """Log marginal likelihood of a leaf, up to terms common to all trees."""
    denom = sigma2 + n * tau2
    return 0.5 * math.log(sigma2 / denom) + tau2 * s * s / (2.0 * sigma2 * denom)


def _sweep(
    y: np.ndarray,
    xcut: np.ndarray,
    ncut: np.ndarray,
    var: np.ndarray,
    cut: np.ndarray,
    val: np.ndarray,
    leaves: np.ndarray,
    nleaf: np.ndarray,
    leaf_of: np.ndarray,
    tree_fit: np.ndarray,
    total: np.ndarray,
    sigma2: float,
    tau2: float,
    base: float,
    power: float,
    maxdepth: int,
    min_leaf: float,
    depth: Any,
    bounds: Any,
    n_avail: Any,
    logm: Any,
) -> None:
    """One backfitting pass over all trees, in place."""
    m = var.shape[0]
    n = y.shape[0]
    p = ncut.shape[0]
    size = var.shape[1]
    cnt = np.zeros(size)
    sm = np.zeros(size)
    grow_list = np.empty(size, dtype=np.int64)
    nog_list = np.empty(size, dtype=np.int64)
    for j in range(m):
        vj = var[j]
        cj = cut[j]
        nl = nleaf[j]
        # leaves that can still be split, and parents of two leaves
        b = 0
        w2 = 0
        for q in range(nl):
            node = leaves[j, q]
            if depth(node) < maxdepth and n_avail(vj, cj, node, ncut, bounds) > 0:
                grow_list[b] = node
                b += 1
            if node % 2 == 1 and vj[node + 1] == _LEAF:
                nog_list[w2] = (node - 1) // 2
                w2 += 1
        if nl == 1:
            # a root without any available rule cannot move
            p_grow = 1.0 if b > 0 else -1.0
        elif b > 0:
            p_grow = 0.5
        else:
            p_grow = 0.0
        if p_grow < 0.0:
            pass
        elif np.random.random() < p_grow:
            eta = grow_list[int(np.random.random() * b)]
            # a variable with cuts left, then one of its cuts
            na = n_avail(vj, cj, eta, ncut, bounds)
            pick = int(np.random.random() * na)
            v = 0
            lo = 0
            hi = 0
            seen = 0
            for vv in range(p):
                lo, hi = bounds(vj, cj, eta, vv, ncut[vv])
                if hi > lo:
                    if seen == pick:
                        v = vv
                        break
                    seen += 1
            c = lo + int(np.random.random() * (hi - lo))
            left = 2 * eta + 1
            right = left + 1
            n_l = 0.0
            s_l = 0.0
            n_r = 0.0
            s_r = 0.0
            for i in range(n):
                if leaf_of[j, i] == eta:
                    r = y[i] - total[i] + tree_fit[j, i]
                    if xcut[i, v] <= c:
                        n_l += 1.0
                        s_l += r
                    else:
                        n_r += 1.0
                        s_r += r
            d = depth(eta)
            # the tree as it would be after the split
            vj[eta] = v
            cj[eta] = c
            vj[left] = _LEAF
            vj[right] = _LEAF
            g_l = d + 1 < maxdepth and n_avail(vj, cj, left, ncut, bounds) > 0
            g_r = d + 1 < maxdepth and n_avail(vj, cj, right, ncut, bounds) > 0
            ps = base * (1.0 + d) ** (-power)
            pc = base * (2.0 + d) ** (-power)
            log_prior = math.log(ps) - math.log(1.0 - ps)
            if g_l:
                log_prior += math.log(1.0 - pc)
            if g_r:
                log_prior += math.log(1.0 - pc)
            sib_leaf = 0
            if eta > 0:
                sib = eta + 1 if eta % 2 == 1 else eta - 1
                if vj[sib] == _LEAF:
                    sib_leaf = 1
            w2_new = w2 + 1 - sib_leaf
            b_new = b - 1 + (1 if g_l else 0) + (1 if g_r else 0)
            p_prune_new = 0.5 if b_new > 0 else 1.0
            log_a = (
                logm(n_l, s_l, sigma2, tau2)
                + logm(n_r, s_r, sigma2, tau2)
                - logm(n_l + n_r, s_l + s_r, sigma2, tau2)
                + log_prior
                + math.log(p_prune_new)
                + math.log(b)
                - math.log(p_grow)
                - math.log(w2_new)
            )
            # a split that leaves a child with too few observations has
            # zero prior mass
            if n_l < min_leaf or n_r < min_leaf:
                log_a = -np.inf
            if math.log(np.random.random()) < log_a:
                for q in range(nl):
                    if leaves[j, q] == eta:
                        leaves[j, q] = left
                        break
                leaves[j, nl] = right
                nleaf[j] = nl + 1
                for i in range(n):
                    if leaf_of[j, i] == eta:
                        leaf_of[j, i] = left if xcut[i, v] <= c else right
            else:
                vj[eta] = _LEAF
                vj[left] = _ABSENT
                vj[right] = _ABSENT
        else:
            eta = nog_list[int(np.random.random() * w2)]
            left = 2 * eta + 1
            right = left + 1
            n_l = 0.0
            s_l = 0.0
            n_r = 0.0
            s_r = 0.0
            for i in range(n):
                if leaf_of[j, i] == left:
                    n_l += 1.0
                    s_l += y[i] - total[i] + tree_fit[j, i]
                elif leaf_of[j, i] == right:
                    n_r += 1.0
                    s_r += y[i] - total[i] + tree_fit[j, i]
            d = depth(eta)
            g_l = d + 1 < maxdepth and n_avail(vj, cj, left, ncut, bounds) > 0
            g_r = d + 1 < maxdepth and n_avail(vj, cj, right, ncut, bounds) > 0
            ps = base * (1.0 + d) ** (-power)
            pc = base * (2.0 + d) ** (-power)
            log_prior = math.log(ps) - math.log(1.0 - ps)
            if g_l:
                log_prior += math.log(1.0 - pc)
            if g_r:
                log_prior += math.log(1.0 - pc)
            b_new = b - (1 if g_l else 0) - (1 if g_r else 0) + 1
            p_grow_new = 1.0 if eta == 0 else 0.5
            log_a = (
                logm(n_l + n_r, s_l + s_r, sigma2, tau2)
                - logm(n_l, s_l, sigma2, tau2)
                - logm(n_r, s_r, sigma2, tau2)
                - log_prior
                + math.log(p_grow_new)
                + math.log(w2)
                - math.log(1.0 - p_grow)
                - math.log(b_new)
            )
            if math.log(np.random.random()) < log_a:
                vj[eta] = _LEAF
                vj[left] = _ABSENT
                vj[right] = _ABSENT
                k = 0
                for q in range(nl):
                    node = leaves[j, q]
                    if node != left and node != right:
                        leaves[j, k] = node
                        k += 1
                leaves[j, k] = eta
                nleaf[j] = k + 1
                for i in range(n):
                    if leaf_of[j, i] == left or leaf_of[j, i] == right:
                        leaf_of[j, i] = eta
        # leaf means given the tree
        nl = nleaf[j]
        for q in range(nl):
            cnt[leaves[j, q]] = 0.0
            sm[leaves[j, q]] = 0.0
        for i in range(n):
            node = leaf_of[j, i]
            cnt[node] += 1.0
            sm[node] += y[i] - total[i] + tree_fit[j, i]
        for q in range(nl):
            node = leaves[j, q]
            denom = sigma2 + cnt[node] * tau2
            val[j, node] = (
                tau2 * sm[node] / denom
                + math.sqrt(sigma2 * tau2 / denom) * np.random.normal()
            )
        for i in range(n):
            new = val[j, leaf_of[j, i]]
            total[i] += new - tree_fit[j, i]
            tree_fit[j, i] = new


def _predict(
    xcut: np.ndarray, var: np.ndarray, cut: np.ndarray, val: np.ndarray
) -> np.ndarray:
    n = xcut.shape[0]
    out = np.zeros(n)
    for j in range(var.shape[0]):
        for i in range(n):
            node = 0
            while var[j, node] >= 0:
                if xcut[i, var[j, node]] <= cut[j, node]:
                    node = 2 * node + 1
                else:
                    node = 2 * node + 2
            out[i] += val[j, node]
    return out


def _seed(s: int) -> None:
    np.random.seed(s)


def _kernels() -> Dict[str, Any]:
    if "sweep" not in _KERNEL:
        from numba import njit  # type: ignore[import-untyped]

        depth = njit(cache=True)(_depth)
        bounds = njit(cache=True)(_bounds)
        n_avail = njit(cache=True)(_n_avail)
        logm = njit(cache=True)(_logm)
        _KERNEL.update(
            depth=depth,
            bounds=bounds,
            n_avail=n_avail,
            logm=logm,
            sweep=njit(cache=True)(_sweep),
            predict=njit(cache=True)(_predict),
            seed=njit(cache=True)(_seed),
        )
    return _KERNEL


class _Forest:
    """State of the sum of trees and one backfitting sweep over it."""

    def __init__(
        self,
        xcut: np.ndarray,
        ncut: np.ndarray,
        n_trees: int,
        tau2: float,
        base: float,
        power: float,
        maxdepth: int,
        min_leaf: int = 0,
    ) -> None:
        self.k = _kernels()
        n = xcut.shape[0]
        size = 2 ** (maxdepth + 1) - 1
        self.xcut = np.ascontiguousarray(xcut, dtype=np.int64)
        self.ncut = np.ascontiguousarray(ncut, dtype=np.int64)
        self.var = np.full((n_trees, size), _ABSENT, dtype=np.int64)
        self.var[:, 0] = _LEAF
        self.cut = np.zeros((n_trees, size), dtype=np.int64)
        self.val = np.zeros((n_trees, size))
        self.leaves = np.zeros((n_trees, size), dtype=np.int64)
        self.nleaf = np.ones(n_trees, dtype=np.int64)
        self.leaf_of = np.zeros((n_trees, n), dtype=np.int64)
        self.tree_fit = np.zeros((n_trees, n))
        self.total = np.zeros(n)
        self.tau2, self.base, self.power, self.maxdepth = tau2, base, power, maxdepth
        self.min_leaf = float(min_leaf)

    def sweep(self, y: np.ndarray, sigma2: float) -> None:
        k = self.k
        k["sweep"](
            y,
            self.xcut,
            self.ncut,
            self.var,
            self.cut,
            self.val,
            self.leaves,
            self.nleaf,
            self.leaf_of,
            self.tree_fit,
            self.total,
            float(sigma2),
            self.tau2,
            self.base,
            self.power,
            self.maxdepth,
            self.min_leaf,
            k["depth"],
            k["bounds"],
            k["n_avail"],
            k["logm"],
        )

    def pack(self) -> Tuple[np.ndarray, ...]:
        """The existing nodes only: (tree, node, variable, cut, value)."""
        tr, nd = np.nonzero(self.var > _ABSENT)
        return (
            tr.astype(np.int32),
            nd.astype(np.int32),
            self.var[tr, nd].astype(np.int32),
            self.cut[tr, nd].astype(np.int32),
            self.val[tr, nd].astype(np.float32),
        )


def _cutpoints(x: np.ndarray, numcut: int) -> np.ndarray:
    u = np.unique(x)
    if u.size <= 1:
        return np.empty(0)
    mids = 0.5 * (u[1:] + u[:-1])
    if mids.size <= numcut:
        return np.asarray(mids)
    q = np.quantile(x, np.linspace(0, 1, numcut + 2)[1:-1])
    return np.asarray(np.unique(q))


@dataclass
class BARTResult(ResultProtocolMixin):
    """Result of :func:`bart`.

    Attributes
    ----------
    fitted : pd.DataFrame
        Posterior mean, standard deviation and interval of the regression
        function at the sample points (of the success probability for
        ``family='binary'``).
    sigma : pd.Series
        Draws of the error standard deviation (absent for binary).
    variable_importance : pd.Series
        Share of all splitting rules that use each regressor, averaged
        over draws. A screening device, not an effect size.
    params : pd.Series
        ``sigma`` (posterior mean) and the average number of leaves per
        tree.
    n_trees, formula, family, n_obs, level

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.uniform(-2, 2, 200)})
    >>> df["y"] = np.sin(2 * df["x"]) + 0.2 * rng.normal(size=200)
    >>> fit = sp.bart("y ~ x", df, n_trees=30, draws=200, burnin=100, seed=1)
    >>> isinstance(fit, sp.BARTResult)
    True
    >>> fit.predict(pd.DataFrame({"x": [0.0, 1.0]})).shape
    (2, 4)
    """

    fitted: pd.DataFrame
    params: pd.Series
    variable_importance: pd.Series
    formula: str
    family: str
    n_trees: int
    n_obs: int
    level: float = 0.95
    sigma: Optional[pd.Series] = None
    model_info: Dict[str, Any] = field(default_factory=dict)
    _state: Dict[str, Any] = field(default_factory=dict, repr=False)

    _citation_keys = ("chipman2010bart",)

    @property
    def coef(self) -> pd.Series:
        return self.params

    def _summarise(self, f: np.ndarray, index: Any, level: float) -> pd.DataFrame:
        lo = (1.0 - level) / 2.0
        return pd.DataFrame(
            {
                "mean": f.mean(axis=0),
                "sd": f.std(axis=0, ddof=1),
                "lower": np.quantile(f, lo, axis=0),
                "upper": np.quantile(f, 1.0 - lo, axis=0),
            },
            index=index,
        )

    def predict_draws(self, newdata: pd.DataFrame) -> np.ndarray:
        """Posterior draws of the regression function at ``newdata``:
        an array with one row per draw."""
        st = self._state
        X = design_for(st["design_info"], st["all_names"], newdata)[:, st["keep"]]
        xcut = np.column_stack(
            [np.searchsorted(c, X[:, v], side="left") for v, c in enumerate(st["cuts"])]
        ).astype(np.int64)
        k = _kernels()
        m, size = st["shape"]
        out = np.empty((len(st["packed"]), X.shape[0]))
        var = np.full((m, size), _ABSENT, dtype=np.int64)
        cut = np.zeros((m, size), dtype=np.int64)
        val = np.zeros((m, size))
        for g, (tr, nd, vv, cc, va) in enumerate(st["packed"]):
            var[:] = _ABSENT
            var[tr, nd] = vv
            cut[tr, nd] = cc
            val[tr, nd] = va
            out[g] = k["predict"](xcut, var, cut, val)
        f = st["center"] + st["scale"] * out
        return np.asarray(special.ndtr(f)) if self.family == "binary" else f

    def predict(
        self, newdata: Optional[pd.DataFrame] = None, level: Optional[float] = None
    ) -> pd.DataFrame:
        """Posterior mean and interval of the regression function.

        Parameters
        ----------
        newdata : DataFrame, optional
            Default: the sample.
        level : float, optional

        Returns
        -------
        DataFrame
            Columns ``mean``, ``sd``, ``lower``, ``upper``. For
            ``family='binary'`` these describe the success probability.
        """
        lv = self.level if level is None else float(level)
        if not 0.0 < lv < 1.0:
            raise MethodIncompatibility(f"level must be in (0, 1); got {lv}.")
        if newdata is None:
            if lv == self.level:
                return self.fitted.copy()
            return self._summarise(self._state["train_draws"], self.fitted.index, lv)
        return self._summarise(self.predict_draws(newdata), newdata.index, lv)

    def summary(self) -> str:
        lines = [
            f"Bayesian additive regression trees ({self.family})    {self.formula}",
            f"Observations: {self.n_obs}    Trees: {self.n_trees}    "
            f"Draws: {self.model_info.get('draws')}",
            "",
            self.params.to_string(float_format=lambda v: f"{v:.5g}"),
            "",
            "Share of splitting rules by regressor",
            self.variable_importance.sort_values(ascending=False)
            .head(15)
            .to_string(float_format=lambda v: f"{v:.3f}"),
        ]
        for note in self.model_info.get("notes", []):
            lines.append("")
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def to_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {
            "formula": self.formula,
            "family": self.family,
            "n_obs": int(self.n_obs),
            "n_trees": int(self.n_trees),
            "params": {str(k): float(v) for k, v in self.params.items()},
            "variable_importance": {
                str(k): float(v) for k, v in self.variable_importance.items()
            },
            "model_info": {
                k: v
                for k, v in self.model_info.items()
                if not isinstance(v, np.ndarray)
            },
        }


def bart(
    formula: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    n_trees: int = 200,
    k: float = 2.0,
    base: float = 0.95,
    power: float = 2.0,
    nu: float = 3.0,
    q: float = 0.9,
    numcut: int = 100,
    max_depth: int = 8,
    min_leaf: int = 5,
    draws: int = 1000,
    burnin: int = 500,
    thin: int = 1,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> BARTResult:
    """Bayesian additive regression trees.

    A flexible regression that finds nonlinearities and interactions on
    its own and returns posterior bands for the regression function.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 + C(group)"``. Regressors enter the trees one
        column of the design matrix at a time; do not add polynomial or
        interaction terms.
    data : DataFrame
    family : {'gaussian', 'binary'}, default 'gaussian'
        ``'binary'`` is a probit link for a 0/1 outcome.
    n_trees : int, default 200
    k : float, default 2
        The prior puts the regression function within the range of the
        outcome with probability about ``2 Phi(k) - 1``. Larger values
        shrink more.
    base, power : float, default 0.95 and 2
        A node at depth ``d`` splits with prior probability
        ``base * (1 + d) ** -power``.
    nu, q : float, default 3 and 0.9
        Prior of the error variance: scaled inverse chi-square with
        ``nu`` degrees of freedom, placing probability ``q`` below the
        residual variance of a linear fit.
    numcut : int, default 100
        Candidate cutpoints per regressor (midpoints of the observed
        values, or quantiles when there are more than ``numcut``).
    max_depth : int, default 8
        Hard cap on the depth of a tree. The prior makes deeper trees
        vanishingly rare.
    min_leaf : int, default 5
        A split is allowed only if both children keep at least this many
        observations. Without it trees isolate single observations and
        the error variance is underestimated.
    draws, burnin, thin : int
    seed : int, optional
    level : float, default 0.95

    Returns
    -------
    BARTResult
        ``fitted`` and ``predict()`` give the posterior mean and band of
        the regression function, ``variable_importance`` the share of
        splitting rules per regressor, ``sigma`` the error scale.

    Notes
    -----
    The sampler uses birth and death moves only. The bands are for the
    regression function, not for new observations; add ``sigma`` for
    those.

    BART bands tend to be somewhat narrow where the function changes
    quickly and at the edge of the data. Treat them as approximate.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x1": rng.uniform(size=300), "x2": rng.uniform(size=300)})
    >>> df["y"] = 10 * np.sin(np.pi * df.x1 * df.x2) + rng.normal(size=300)
    >>> fit = sp.bart("y ~ x1 + x2", df, n_trees=50, draws=200, burnin=200, seed=1)
    >>> truth = 10 * np.sin(np.pi * df.x1 * df.x2)
    >>> bool(np.sqrt(np.mean((fit.fitted["mean"] - truth) ** 2)) < 1.0)
    True

    References
    ----------
    chipman2010bart
    """
    fam = str(family).lower()
    if fam in ("normal", "continuous"):
        fam = "gaussian"
    if fam in ("probit", "binomial"):
        fam = "binary"
    if fam not in ("gaussian", "binary"):
        raise MethodIncompatibility(
            f"family must be 'gaussian' or 'binary'; got {family!r}."
        )
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    if n_trees < 1 or draws < 10 or burnin < 0 or thin < 1 or numcut < 1:
        raise MethodIncompatibility(
            "n_trees and numcut must be at least 1, draws at least 10, "
            "burnin non-negative and thin at least 1."
        )
    if not (0.0 < base < 1.0) or power < 0 or k <= 0 or nu <= 0 or not 0 < q < 1:
        raise MethodIncompatibility(
            "base and q must be in (0, 1), power non-negative, k and nu positive."
        )
    if not 1 <= int(max_depth) <= 12:
        raise MethodIncompatibility("max_depth must be between 1 and 12.")
    if min_leaf < 0:
        raise MethodIncompatibility("min_leaf must be non-negative.")
    y_df, X_df = create_design_matrices(formula, data)
    y = np.asarray(y_df, dtype=float).reshape(-1)
    Xall = np.asarray(X_df, dtype=float)
    all_names = [str(c) for c in X_df.columns]
    keep = [j for j in range(Xall.shape[1]) if np.ptp(Xall[:, j]) > 0]
    if not keep:
        raise MethodIncompatibility("The formula has no regressor that varies.")
    X = Xall[:, keep]
    names = [all_names[j] for j in keep]
    n, p = X.shape
    if n < 10:
        raise DataInsufficient(f"{n} observations are too few.")
    cuts = [_cutpoints(X[:, v], int(numcut)) for v in range(p)]
    ncut = np.array([c.size for c in cuts], dtype=np.int64)
    xcut = np.column_stack(
        [np.searchsorted(c, X[:, v], side="left") for v, c in enumerate(cuts)]
    )
    notes: List[str] = []
    if fam == "binary":
        if not np.isin(y, (0.0, 1.0)).all():
            raise MethodIncompatibility("For family='binary' the outcome must be 0/1.")
        if y.min() == y.max():
            raise DataInsufficient("The outcome does not vary.")
        center = float(stats.norm.ppf(y.mean()))
        scale = 1.0
        tau2 = (3.0 / (k * math.sqrt(n_trees))) ** 2
        sigma2 = 1.0
        lam = 0.0
    else:
        if y.max() == y.min():
            raise MethodIncompatibility("The outcome does not vary.")
        scale = float(y.max() - y.min())
        center = float(0.5 * (y.max() + y.min()))
        ys = (y - center) / scale
        tau2 = (0.5 / (k * math.sqrt(n_trees))) ** 2
        # sigma prior calibrated on a linear fit
        Z = np.column_stack([np.ones(n), X])
        if n > Z.shape[1] + 1 and np.linalg.matrix_rank(Z) == Z.shape[1]:
            res = ys - Z @ np.linalg.lstsq(Z, ys, rcond=None)[0]
            s2_hat = float(res @ res / (n - Z.shape[1]))
        else:
            s2_hat = float(ys.var(ddof=1))
            notes.append(
                "the error-variance prior is calibrated on the variance of "
                "the outcome (a linear fit is not available)"
            )
        lam = s2_hat * stats.chi2.ppf(1.0 - q, nu) / nu
        sigma2 = s2_hat
    forest = _Forest(
        xcut,
        ncut,
        int(n_trees),
        tau2,
        float(base),
        float(power),
        int(max_depth),
        int(min_leaf),
    )
    rng = np.random.default_rng(seed)
    forest.k["seed"](int(rng.integers(0, 2**31 - 1)))
    n_iter = int(burnin) + int(draws) * int(thin)
    train = np.empty((int(draws), n), dtype=np.float32)
    sig = np.empty(int(draws))
    counts = np.zeros((int(draws), p))
    leaves_avg = np.empty(int(draws))
    packed = []
    z = ys if fam == "gaussian" else np.where(y > 0, 0.5, -0.5)
    kept = 0
    for it in range(n_iter):
        if fam == "binary":
            mu = center + forest.total
            lower = np.where(y > 0, 0.0, -np.inf)
            upper = np.where(y > 0, np.inf, 0.0)
            z = rtruncnorm(rng, mu, 1.0, lower, upper) - center
        forest.sweep(z, sigma2)
        if fam == "gaussian":
            sse = float(((z - forest.total) ** 2).sum())
            sigma2 = (nu * lam + sse) / rng.chisquare(nu + n)
        if it >= burnin and (it - burnin) % thin == 0:
            train[kept] = forest.total
            sig[kept] = math.sqrt(sigma2) * scale
            used = forest.var[forest.var >= 0]
            counts[kept] = np.bincount(used, minlength=p)
            leaves_avg[kept] = forest.nleaf.mean()
            packed.append(forest.pack())
            kept += 1
    f_train = center + scale * train.astype(float)
    if fam == "binary":
        f_train = special.ndtr(f_train)
    lo = (1.0 - level) / 2.0
    fitted = pd.DataFrame(
        {
            "mean": f_train.mean(axis=0),
            "sd": f_train.std(axis=0, ddof=1),
            "lower": np.quantile(f_train, lo, axis=0),
            "upper": np.quantile(f_train, 1.0 - lo, axis=0),
        },
        index=X_df.index,
    )
    tot = counts.sum(axis=1, keepdims=True)
    share = np.divide(counts, tot, out=np.zeros_like(counts), where=tot > 0).mean(
        axis=0
    )
    params = pd.Series(
        {
            "sigma": float(sig.mean()) if fam == "gaussian" else 1.0,
            "leaves_per_tree": float(leaves_avg.mean()),
        }
    )
    if fam == "gaussian":
        from .diagnostics import mcmc_ess

        ess = float(np.asarray(mcmc_ess(pd.DataFrame({"sigma": sig}))).ravel()[0])
        if ess < 10:
            notes.append(
                f"the error scale has an effective sample size of {ess:.0f}; "
                "increase burnin and draws"
            )
    else:
        ess = float("nan")
    res = BARTResult(
        fitted=fitted,
        params=params,
        variable_importance=pd.Series(share, index=names, name="share"),
        formula=formula,
        family=fam,
        n_trees=int(n_trees),
        n_obs=n,
        level=level,
        sigma=pd.Series(sig, name="sigma") if fam == "gaussian" else None,
        model_info={
            "draws": int(draws),
            "burnin": int(burnin),
            "thin": int(thin),
            "k": float(k),
            "base": float(base),
            "power": float(power),
            "nu": float(nu),
            "q": float(q),
            "min_leaf": int(min_leaf),
            "sigma_ess": ess,
            "notes": notes,
            "regressors": names,
        },
        _state={
            "design_info": getattr(X_df, "design_info", None),
            "all_names": all_names,
            "keep": keep,
            "cuts": cuts,
            "packed": packed,
            "shape": forest.var.shape,
            "center": center,
            "scale": scale,
            "train_draws": f_train,
        },
    )
    for note in notes:
        if "effective sample size" in note:
            warnings.warn(note, ConvergenceWarning, stacklevel=2)
    return res
