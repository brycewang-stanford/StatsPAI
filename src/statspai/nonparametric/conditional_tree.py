"""Conditional inference trees.

A tree that decides whether to split with a hypothesis test. In each node
the association between the outcome and every covariate is measured by a
linear statistic whose mean and covariance under permutation of the
outcome are known in closed form (Hothorn, Hornik and Zeileis 2006). The
covariate with the smallest p-value is selected; when no p-value, adjusted
for the number of covariates tested, is below ``alpha``, the node is not
split. Selecting the variable by a test and only then searching for the
cut point removes the bias of exhaustive-search trees toward covariates
with many possible splits, and makes pruning unnecessary.

Written from the paper and the documentation of the R packages ``party``
and ``partykit``, which are GPL and whose source was not read. The
conventions the paper leaves open (which covariates count toward the
Bonferroni correction, how small a child may be) were recovered by
comparing outputs and are recorded where they are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["ctree", "CTreeResult"]

_MAX_LEVELS = 12  # unordered covariate: 2^(K-1) - 1 candidate splits
_RANK_TOL = 1e-10


def _bad(message: str, hint: str) -> MethodIncompatibility:
    return MethodIncompatibility(f"ctree: {message}", recovery_hint=hint)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class _Covariate:
    name: str
    kind: str  # 'ordered' or 'nominal'
    values: np.ndarray  # float for ordered, integer codes for nominal
    levels: Optional[List[Any]] = None  # nominal levels; ordered-factor levels


def _is_nominal(col: pd.Series) -> bool:
    if isinstance(col.dtype, pd.CategoricalDtype):
        return not col.dtype.ordered
    if pd.api.types.is_bool_dtype(col.dtype):
        return False
    return not pd.api.types.is_numeric_dtype(col.dtype)


def _encode_covariate(col: pd.Series, name: str) -> _Covariate:
    if col.isna().any():
        raise DataInsufficient(
            f"ctree: covariate '{name}' has missing values.",
            recovery_hint="Drop or impute the rows first; surrogate splits "
            "are not implemented.",
        )
    if _is_nominal(col):
        cat = col.astype("category")
        levels = list(cat.cat.categories)
        if len(levels) > _MAX_LEVELS:
            raise _bad(
                f"covariate '{name}' has {len(levels)} unordered levels; "
                f"at most {_MAX_LEVELS} are searched exhaustively.",
                "Merge levels, or pass an ordered categorical.",
            )
        return _Covariate(name, "nominal", cat.cat.codes.to_numpy(), levels)
    if isinstance(col.dtype, pd.CategoricalDtype):  # ordered factor
        return _Covariate(
            name,
            "ordered",
            col.cat.codes.to_numpy(dtype=float),
            list(col.cat.categories),
        )
    return _Covariate(name, "ordered", col.to_numpy(dtype=float))


def _covariate_values(cov: _Covariate, col: pd.Series) -> np.ndarray:
    """Encode a column of new data the way ``cov`` was encoded."""
    if col.isna().any():
        raise DataInsufficient(
            f"ctree: covariate '{cov.name}' has missing values.",
            recovery_hint="Drop or impute the rows before predicting.",
        )
    if cov.levels is None:
        return np.asarray(col.to_numpy(dtype=float))
    codes = pd.Categorical(col, categories=cov.levels).codes
    if (codes < 0).any():
        unseen = sorted(set(col[codes < 0].astype(str)))
        raise _bad(
            f"covariate '{cov.name}' has levels not seen in fitting: " f"{unseen[:5]}.",
            "Predict only for levels present in the training data.",
        )
    return np.asarray(codes, dtype=float if cov.kind == "ordered" else int)


# ---------------------------------------------------------------------------
# Permutation moments
# ---------------------------------------------------------------------------


def _pinv_rank(a: np.ndarray) -> Tuple[np.ndarray, int]:
    """Moore-Penrose inverse of a symmetric PSD matrix, and its rank."""
    w, v = np.linalg.eigh((a + a.T) / 2.0)
    keep = w > _RANK_TOL * max(float(w.max()), 0.0)
    if not keep.any():
        return np.zeros_like(a), 0
    return (v[:, keep] / w[keep]) @ v[:, keep].T, int(keep.sum())


def _association(
    g: np.ndarray, h: np.ndarray, vh_pinv: np.ndarray, vh_rank: int
) -> Tuple[float, int]:
    """Quadratic-form statistic of the linear statistic ``T = g'h``.

    Under permutation of the rows of ``h``, ``T`` has mean
    ``(sum g) E[h]'`` and covariance ``V[h] (x) A`` with
    ``A = n/(n-1) sum g g' - 1/(n-1) (sum g)(sum g)'`` (Hothorn et al.
    2006, eq. 2), so the statistic is ``tr(A^+ D V[h]^+ D')`` with
    ``D`` the centred statistic, chi-squared on ``rank(A) rank(V[h])``
    degrees of freedom.
    """
    n = g.shape[0]
    gs = g.sum(axis=0)
    d = g.T @ h - np.outer(gs, h.mean(axis=0))
    a = (n * (g.T @ g) - np.outer(gs, gs)) / (n - 1.0)
    a_pinv, a_rank = _pinv_rank(a)
    if a_rank == 0 or vh_rank == 0:
        return np.nan, 0
    return float(np.sum((a_pinv @ d) * (d @ vh_pinv))), a_rank * vh_rank


def _split_statistics(
    left: np.ndarray,
    hsum_left: np.ndarray,
    h_mean: np.ndarray,
    vh: np.ndarray,
    vh_pinv: np.ndarray,
    n: int,
    split_stat: str,
) -> np.ndarray:
    """Two-sample statistic of each candidate split.

    ``left`` holds the size of the left child and ``hsum_left`` the sum
    of ``h`` in it, one row per candidate. The centred sum has covariance
    ``V[h] n_L n_R / (n - 1)``.
    """
    d = hsum_left - np.outer(left, h_mean)
    scale = left * (n - left) / (n - 1.0)
    if split_stat == "quadratic":
        return np.asarray(np.sum((d @ vh_pinv) * d, axis=1) / scale)
    var = np.diag(vh)
    ok = var > _RANK_TOL * max(float(var.max()), 0.0)
    z = np.abs(d[:, ok]) / np.sqrt(np.outer(scale, var[ok]))
    return np.asarray(z.max(axis=1))


# ---------------------------------------------------------------------------
# The tree
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    id: int
    depth: int
    n: int
    value: np.ndarray  # mean of h in the node
    parent: int = 0
    variable: Optional[str] = None
    threshold: float = np.nan
    left_levels: Optional[Tuple[int, ...]] = None
    statistic: float = np.nan
    p_value: float = np.nan
    left: int = 0
    right: int = 0
    majority_left: bool = True

    @property
    def is_leaf(self) -> bool:
        return self.variable is None


@dataclass
class _Settings:
    alpha: float
    bonferroni: bool
    min_split: int
    min_bucket: int
    min_prob: float
    max_depth: Optional[int]
    split_stat: str


class _Grower:
    def __init__(
        self, covs: List[_Covariate], h: np.ndarray, settings: _Settings
    ) -> None:
        self.covs = covs
        self.h = h
        self.s = settings
        self.nodes: List[_Node] = []
        self.tests: List[Dict[str, Any]] = []
        self.where = np.zeros(h.shape[0], dtype=int)

    def grow(self, idx: np.ndarray, depth: int, parent: int) -> int:
        h = self.h[idx]
        node = _Node(len(self.nodes) + 1, depth, idx.size, h.mean(axis=0), parent)
        self.nodes.append(node)
        split = None
        deep = self.s.max_depth is not None and depth >= self.s.max_depth
        if idx.size >= self.s.min_split and not deep:
            split = self._find_split(node, idx, h)
        if split is None:
            self.where[idx] = node.id
            return node.id
        go_left = split
        node.majority_left = bool(go_left.sum() >= (~go_left).sum())
        node.left = self.grow(idx[go_left], depth + 1, node.id)
        node.right = self.grow(idx[~go_left], depth + 1, node.id)
        return node.id

    def _find_split(
        self, node: _Node, idx: np.ndarray, h: np.ndarray
    ) -> Optional[np.ndarray]:
        n = idx.size
        hc = h - h.mean(axis=0)
        vh = hc.T @ hc / n
        vh_pinv, vh_rank = _pinv_rank(vh)
        if vh_rank == 0:  # the outcome is constant in the node
            return None
        rows = []
        for j, cov in enumerate(self.covs):
            x = cov.values[idx]
            if cov.kind == "nominal":
                present = np.unique(x)
                g = (x[:, None] == present[None, :]).astype(float)
            else:
                g = x[:, None]
            stat, df = _association(g, h, vh_pinv, vh_rank)
            if df == 0:  # constant in the node: not tested, not counted
                continue
            rows.append((j, stat, df, float(stats.chi2.logsf(stat, df))))
        if not rows:
            return None
        # partykit counts the covariates that vary in the node, so a
        # covariate used up by an earlier split stops costing anything.
        m = len(rows) if self.s.bonferroni else 1
        tested = []
        for j, stat, df, logp in rows:
            p_raw = float(np.exp(logp))
            p_adj = float(-np.expm1(m * np.log1p(-p_raw))) if m > 1 else p_raw
            if p_raw < 1e-10:  # 1 - (1 - p)^m loses the digits of a tiny p
                p_adj = min(1.0, m * p_raw)
            tested.append((logp, -stat, j, p_adj))
            self.tests.append(
                {
                    "node": node.id,
                    "variable": self.covs[j].name,
                    "statistic": stat,
                    "df": df,
                    "p_value_raw": p_raw,
                    "p_value": p_adj,
                }
            )
        # Smallest p first; among p-values that underflow, largest statistic.
        for logp, neg_stat, j, p_adj in sorted(tested):
            if not p_adj < self.s.alpha:
                break
            found = self._best_split(self.covs[j], idx, h, vh, vh_pinv)
            if found is None:  # no admissible cut: try the next covariate
                continue
            go_left, threshold, levels = found
            node.variable = self.covs[j].name
            node.threshold = threshold
            node.left_levels = levels
            node.statistic = -neg_stat
            node.p_value = p_adj
            return go_left
        return None

    def _best_split(
        self,
        cov: _Covariate,
        idx: np.ndarray,
        h: np.ndarray,
        vh: np.ndarray,
        vh_pinv: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, float, Optional[Tuple[int, ...]]]]:
        n = idx.size
        x = cov.values[idx]
        smallest = max(self.s.min_bucket, int(np.ceil(self.s.min_prob * n)))
        h_mean = h.mean(axis=0)
        if cov.kind == "ordered":
            order = np.argsort(x, kind="stable")
            xs = x[order]
            cuts = np.flatnonzero(xs[1:] > xs[:-1]) + 1  # size of the left child
            cuts = cuts[(cuts >= smallest) & (n - cuts >= smallest)]
            if cuts.size == 0:
                return None
            csum = np.cumsum(h[order], axis=0)
            stat = _split_statistics(
                cuts.astype(float),
                csum[cuts - 1],
                h_mean,
                vh,
                vh_pinv,
                n,
                self.s.split_stat,
            )
            best = int(np.argmax(stat))  # first maximum: the smallest cut
            threshold = float(xs[cuts[best] - 1])
            return x <= threshold, threshold, None
        present = np.unique(x)
        counts = np.array([(x == lv).sum() for lv in present], dtype=float)
        sums = np.array([h[x == lv].sum(axis=0) for lv in present])
        k = present.size
        subsets = [
            (0, *rest)  # the first level stays left: each partition once
            for size in range(0, k - 1)
            for rest in combinations(range(1, k), size)
        ]
        left = np.array([counts[list(sub)].sum() for sub in subsets])
        keep = (left >= smallest) & (n - left >= smallest)
        if not keep.any():
            return None
        hsum = np.array([sums[list(sub)].sum(axis=0) for sub in subsets])
        stat = _split_statistics(
            left[keep], hsum[keep], h_mean, vh, vh_pinv, n, self.s.split_stat
        )
        chosen = [s for s, ok in zip(subsets, keep) if ok][int(np.argmax(stat))]
        levels = tuple(int(present[i]) for i in chosen)
        return np.isin(x, levels), np.nan, levels


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CTreeResult(ResultProtocolMixin):
    """Result of :func:`ctree`.

    Attributes
    ----------
    nodes : pandas.DataFrame
        One row per node, numbered depth first with the left child before
        the right one (the numbering of ``partykit``): ``parent``,
        ``depth``, ``n``, ``variable`` and ``split`` (inner nodes),
        ``statistic`` and ``p_value`` of the test that selected the
        variable, ``prediction``, and ``error`` (share misclassified, or
        the within-node variance for a numeric outcome).
    tests : pandas.DataFrame
        Every test that was run: ``node``, ``variable``, ``statistic``,
        ``df``, ``p_value_raw`` and ``p_value`` (adjusted for the number
        of covariates tested in the node).
    fitted : numpy.ndarray
        Prediction for each row of the data.
    node_id : numpy.ndarray
        Terminal node of each row.
    task : {'classification', 'regression'}
    classes : list or None
        Outcome classes, in the column order of ``predict(type='prob')``.
    n_leaves, n_obs : int

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.uniform(size=400), "z": rng.normal(size=400)})
    >>> df["y"] = np.where(df["x"] > 0.5, 2.0, 0.0) + rng.normal(size=400)
    >>> fit = sp.ctree(df, y="y", covariates=["x", "z"])
    >>> isinstance(fit, sp.CTreeResult)
    True
    >>> fit.nodes.loc[1, "variable"]
    'x'
    >>> fit.predict(df.head(3)).shape
    (3,)
    """

    _citation_keys = ("hothorn2006unbiased",)

    nodes: pd.DataFrame
    tests: pd.DataFrame
    fitted: np.ndarray
    node_id: np.ndarray
    task: str
    classes: Optional[List[Any]]
    y: str
    covariates: List[str]
    n_obs: int
    model_info: Dict[str, Any] = field(default_factory=dict)
    _tree: List[_Node] = field(default_factory=list, repr=False)
    _covs: List[_Covariate] = field(default_factory=list, repr=False)

    @property
    def n_leaves(self) -> int:
        return int(self.nodes["variable"].isna().sum())

    def _route(self, data: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.covariates if c not in data.columns]
        if missing:
            raise _bad(
                f"columns missing from the data: {missing}.",
                "Pass the covariates the tree was fitted on.",
            )
        cols = {c.name: _covariate_values(c, data[c.name]) for c in self._covs}
        kinds = {c.name: c.kind for c in self._covs}
        out = np.ones(len(data), dtype=int)
        for node in self._tree:  # parents come before their children
            if node.is_leaf:
                continue
            here = out == node.id
            if not here.any():
                continue
            assert node.variable is not None
            x = cols[node.variable][here]
            if kinds[node.variable] == "ordered":
                go_left = x <= node.threshold
            else:
                assert node.left_levels is not None
                go_left = np.isin(x, node.left_levels)
                # A level the node never saw follows the larger child.
                seen = np.isin(x, self.model_info["levels_seen"][node.id])
                go_left = np.where(seen, go_left, node.majority_left)
            out[here] = np.where(go_left, node.left, node.right)
        return out

    def predict(self, data: pd.DataFrame, type: str = "response") -> np.ndarray:
        """Predict for new data.

        ``type='response'``: the majority class of the terminal node, or
        its mean for a numeric outcome. ``'prob'``: class shares in the
        terminal node, one column per class. ``'node'``: the terminal
        node itself.
        """
        if type not in ("response", "prob", "node"):
            raise _bad(
                f"type={type!r} is not available.",
                "Use 'response', 'prob' or 'node'.",
            )
        where = self._route(data)
        if type == "node":
            return where
        value = np.array([self._tree[i - 1].value for i in where])
        if self.task == "regression":
            if type == "prob":
                raise _bad(
                    "type='prob' needs a classification tree.",
                    "Use type='response'.",
                )
            return np.asarray(value[:, 0])
        if type == "prob":
            return np.asarray(value)
        assert self.classes is not None
        labels = np.asarray(self.classes, dtype=object)
        return np.asarray(labels[value.argmax(axis=1)])

    def _label(self, node: _Node) -> Tuple[str, str]:
        assert node.variable is not None
        if node.left_levels is None:
            cov = next(c for c in self._covs if c.name == node.variable)
            if cov.levels is not None:  # ordered factor
                cut = cov.levels[int(node.threshold)]
                return f"{node.variable} <= {cut}", f"{node.variable} > {cut}"
            return (
                f"{node.variable} <= {node.threshold:.6g}",
                f"{node.variable} > {node.threshold:.6g}",
            )
        cov = next(c for c in self._covs if c.name == node.variable)
        assert cov.levels is not None
        seen = self.model_info["levels_seen"][node.id]
        left = [str(cov.levels[i]) for i in node.left_levels]
        right = [str(cov.levels[i]) for i in seen if i not in node.left_levels]
        return (
            f"{node.variable} in {{{', '.join(left)}}}",
            f"{node.variable} in {{{', '.join(right)}}}",
        )

    def summary(self) -> str:
        head = [
            f"Conditional inference tree ({self.task})",
            "-" * 52,
            f"Outcome        : {self.y}",
            f"Observations   : {self.n_obs}",
            f"Inner nodes    : {len(self.nodes) - self.n_leaves}",
            f"Terminal nodes : {self.n_leaves}",
            f"Test           : {self.model_info['test']}, "
            f"alpha = {self.model_info['alpha']:g}",
            "",
            "[1] root",
        ]
        lines: List[str] = []

        def walk(node: _Node, indent: str) -> None:
            for child, label in zip((node.left, node.right), self._label(node)):
                kid = self._tree[child - 1]
                text = f"{indent}|   [{kid.id}] {label}"
                if kid.is_leaf:
                    row = self.nodes.loc[kid.id]
                    pred = row["prediction"]
                    shown = f"{pred:.6g}" if self.task == "regression" else str(pred)
                    measure = (
                        f"err = {100 * row['error']:.1f}%"
                        if self.task == "classification"
                        else f"var = {row['error']:.6g}"
                    )
                    text += f": {shown} (n = {kid.n}, {measure})"
                elif np.isfinite(kid.p_value):
                    text += f"  (p = {kid.p_value:.3g})"
                lines.append(text)
                if not kid.is_leaf:
                    walk(kid, indent + "|   ")

        root = self._tree[0]
        if root.is_leaf:
            head[-1] += f": {self.nodes.loc[1, 'prediction']} (n = {root.n})"
        else:
            head[-1] += f"  (p = {root.p_value:.3g})"
            walk(root, "")
        return "\n".join(head + lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def ctree(
    data: pd.DataFrame,
    y: str,
    covariates: Sequence[str],
    *,
    task: str = "auto",
    alpha: float = 0.05,
    test: str = "bonferroni",
    min_split: int = 20,
    min_bucket: int = 7,
    min_prob: float = 0.01,
    max_depth: Optional[int] = None,
    split_stat: str = "quadratic",
) -> CTreeResult:
    """Conditional inference tree (Hothorn, Hornik and Zeileis 2006).

    In each node every covariate is tested for association with the
    outcome by a permutation test in its asymptotic chi-squared form. The
    covariate with the smallest p-value is split on, at the cut point
    with the largest two-sample statistic. A node whose smallest adjusted
    p-value is not below ``alpha`` becomes a leaf, so the size of the
    tree is set by the significance level and there is no pruning step.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome column.
    covariates : sequence of str
        Numeric, boolean and ordered-categorical columns are split at a
        cut point. String and unordered-categorical columns are split
        into two sets of levels (at most 12 levels).
    task : {'auto', 'classification', 'regression'}
        ``'auto'``: classification for a string, categorical or boolean
        outcome, regression for a numeric one. A numeric 0/1 outcome
        gives the same tree either way; classification predicts the
        majority class, regression the share.
    alpha : float, default 0.05
        Significance level a split must reach (``1 - mincriterion`` in
        ``party``).
    test : {'bonferroni', 'univariate'}
        ``'bonferroni'``: the p-value is adjusted to
        ``1 - (1 - p)^m`` for the ``m`` covariates that vary in the node.
        ``'univariate'``: no adjustment.
    min_split : int, default 20
        A node with fewer observations is not split.
    min_bucket : int, default 7
    min_prob : float, default 0.01
        A child must hold at least ``max(min_bucket, min_prob * n)``
        observations, ``n`` being the size of the node.
    max_depth : int, optional
        Depth at which growth stops; the root has depth 0.
    split_stat : {'quadratic', 'maximum'}
        Statistic that ranks cut points: the quadratic form (the default
        of ``partykit``) or the largest standardised component (the
        paper). They coincide for a numeric or a two-class outcome.

    Returns
    -------
    CTreeResult

    Notes
    -----
    Variable selection uses the quadratic-form statistic
    ``c = (t - mu)' Sigma^+ (t - mu)`` of the linear statistic
    ``t = sum_i g(x_i) h(y_i)'``, where ``h`` is the outcome itself or
    its class indicators and ``g`` is the covariate itself or its level
    indicators. Its conditional mean and covariance under permutation
    are those of eq. 2 of the paper, and ``c`` is referred to a
    chi-squared distribution with ``rank(Sigma)`` degrees of freedom.
    The maximum-type test statistic and Monte Carlo p-values of the R
    packages are not implemented.

    The p-values are those of the tests that grew the tree. They are
    not valid for inference on the fitted tree, whose splits were chosen
    by looking at the same data.

    Rows with a missing outcome or covariate are refused; surrogate
    splits are not implemented. There are no case weights.

    With the same data and settings the tree, the test statistics and
    the p-values equal those of ``partykit::ctree`` (and of
    ``party::ctree``, the implementation used by Varian 2014).

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({"age": rng.uniform(1, 70, 500),
    ...                    "noise": rng.normal(size=500)})
    >>> df["survived"] = (rng.uniform(size=500)
    ...                   < np.where(df["age"] < 20, 0.7, 0.35)).astype(int)
    >>> fit = sp.ctree(df, y="survived", covariates=["age", "noise"],
    ...                task="classification")
    >>> fit.nodes.loc[1, "variable"]
    'age'
    >>> fit.predict(df.head(2), type="prob").shape
    (2, 2)

    References
    ----------
    [@hothorn2006unbiased],
    [@varian2014big]
    """
    covariates = list(covariates)
    if not covariates:
        raise _bad("no covariates were given.", "Pass at least one column.")
    absent = [c for c in [y, *covariates] if c not in data.columns]
    if absent:
        raise _bad(f"columns not in the data: {absent}.", "Check the names.")
    if y in covariates:
        raise _bad(f"the outcome '{y}' is also a covariate.", "Remove it.")
    if not 0.0 < alpha < 1.0:
        raise _bad(f"alpha={alpha!r} is not a level.", "Use a value in (0, 1).")
    if test not in ("bonferroni", "univariate"):
        raise _bad(
            f"test={test!r} is not available.", "Use 'bonferroni' or " "'univariate'."
        )
    if split_stat not in ("quadratic", "maximum"):
        raise _bad(
            f"split_stat={split_stat!r} is not available.",
            "Use 'quadratic' or 'maximum'.",
        )
    if task not in ("auto", "classification", "regression"):
        raise _bad(
            f"task={task!r} is not available.",
            "Use 'auto', 'classification' or 'regression'.",
        )
    if min_split < 2 or min_bucket < 1 or not 0.0 <= min_prob < 0.5:
        raise _bad(
            "min_split, min_bucket or min_prob is out of range.",
            "Need min_split >= 2, min_bucket >= 1, 0 <= min_prob < 0.5.",
        )
    if max_depth is not None and max_depth < 0:
        raise _bad(f"max_depth={max_depth!r} is negative.", "Use None or >= 0.")

    yc = data[y]
    if yc.isna().any():
        raise DataInsufficient(
            f"ctree: the outcome '{y}' has missing values.",
            recovery_hint="Drop those rows first.",
        )
    n = len(data)
    if n < 2:
        raise DataInsufficient(
            "ctree: fewer than two observations.",
            recovery_hint="A tree needs data to test on.",
        )
    numeric = pd.api.types.is_numeric_dtype(yc.dtype) and not (
        pd.api.types.is_bool_dtype(yc.dtype)
    )
    if task == "auto":
        task = "regression" if numeric else "classification"
    classes: Optional[List[Any]] = None
    if task == "regression":
        if not numeric:
            raise _bad(
                f"the outcome '{y}' is not numeric.",
                "Use task='classification'.",
            )
        h = yc.to_numpy(dtype=float)[:, None]
    else:
        cat = yc.astype("category")
        classes = list(cat.cat.categories)
        h = np.eye(len(classes))[cat.cat.codes.to_numpy()]

    covs = [_encode_covariate(data[c], c) for c in covariates]
    settings = _Settings(
        alpha,
        test == "bonferroni",
        int(min_split),
        int(min_bucket),
        float(min_prob),
        max_depth,
        split_stat,
    )
    grower = _Grower(covs, h, settings)
    grower.grow(np.arange(n), 0, 0)
    tree = grower.nodes

    levels_seen: Dict[int, List[int]] = {}
    rows = []
    for node in tree:
        mine = h[_members(tree, node, grower.where)]
        if task == "classification":
            assert classes is not None
            prediction: Any = classes[int(node.value.argmax())]
            error = float(1.0 - node.value.max())
        else:
            prediction = float(node.value[0])
            error = float(mine.var())
        split: Any = None
        if not node.is_leaf:
            cov = next(c for c in covs if c.name == node.variable)
            if node.left_levels is None:
                split = node.threshold
            else:
                assert cov.levels is not None
                split = [cov.levels[i] for i in node.left_levels]
                rows_in = _members(tree, node, grower.where)
                levels_seen[node.id] = sorted(
                    int(v) for v in np.unique(cov.values[rows_in])
                )
        rows.append(
            {
                "node": node.id,
                "parent": node.parent,
                "depth": node.depth,
                "n": node.n,
                "variable": node.variable,
                "split": split,
                "statistic": node.statistic,
                "p_value": node.p_value,
                "prediction": prediction,
                "error": error,
            }
        )
    nodes = pd.DataFrame(rows).set_index("node")
    tests = pd.DataFrame(
        grower.tests,
        columns=["node", "variable", "statistic", "df", "p_value_raw", "p_value"],
    )
    fitted = nodes.loc[grower.where, "prediction"].to_numpy()
    return CTreeResult(
        nodes=nodes,
        tests=tests,
        fitted=fitted,
        node_id=grower.where.copy(),
        task=task,
        classes=classes,
        y=y,
        covariates=covariates,
        n_obs=n,
        model_info={
            "alpha": float(alpha),
            "test": test,
            "split_stat": split_stat,
            "min_split": int(min_split),
            "min_bucket": int(min_bucket),
            "min_prob": float(min_prob),
            "max_depth": max_depth,
            "levels_seen": levels_seen,
        },
        _tree=tree,
        _covs=covs,
    )


def _members(tree: List[_Node], node: _Node, where: np.ndarray) -> np.ndarray:
    """Rows whose terminal node lies under ``node``."""
    leaves, stack = [], [node]
    while stack:
        cur = stack.pop()
        if cur.is_leaf:
            leaves.append(cur.id)
        else:
            stack += [tree[cur.left - 1], tree[cur.right - 1]]
    return np.asarray(np.isin(where, leaves))
