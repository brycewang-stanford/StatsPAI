"""Representative points of a distribution: ``sp.support_points``,
``sp.split_data``.

Support points (Mak and Joseph 2018) are the ``n`` points whose empirical
distribution is closest, in energy distance, to a target distribution. A
small set of them stands in for a large sample: to push an input
distribution through an expensive model, to thin a long MCMC chain, or to
pick a test set that looks like the whole data set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility
from ._common import pair_sqdist


def helmert_frame(data: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    """Numeric matrix with mean 0 and sd 1; categories in Helmert contrasts.

    Constant columns are dropped. This is the preprocessing of the R
    packages ``SPlit`` and ``twinning``.
    """
    blocks: List[np.ndarray] = []
    for c in columns:
        col = data[c]
        if pd.api.types.is_bool_dtype(col) or not pd.api.types.is_numeric_dtype(col):
            codes, levels = pd.factorize(col, sort=True)
            k = len(levels)
            H = np.zeros((k, max(k - 1, 0)))
            for j in range(1, k):
                H[:j, j - 1] = -1.0
                H[j, j - 1] = float(j)
            blocks.append(H[codes])
        else:
            blocks.append(col.to_numpy(dtype=float)[:, None])
    X = np.column_stack(blocks) if blocks else np.zeros((data.shape[0], 0))
    if not np.all(np.isfinite(X)):
        raise MethodIncompatibility("The data have missing or infinite values.")
    sd = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros(X.shape[1])
    keep = sd > 0
    if not keep.any():
        raise MethodIncompatibility("No column varies.")
    return np.asarray((X[:, keep] - X[:, keep].mean(axis=0)) / sd[keep])


def _sp_ccp(
    Y: np.ndarray,
    n: int,
    rng: np.random.Generator,
    max_iter: int,
    tol: float,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int, bool]:
    """Convex-concave iterations of Mak and Joseph (2018) on a fixed sample."""
    N, p = Y.shape
    w = np.full(N, 1.0 / N) if weights is None else weights / weights.sum()
    start = rng.choice(N, size=n, replace=N < n, p=w)
    X = Y[start] + 1e-3 * rng.standard_normal((n, p))
    block = max(1, int(4_000_000 // max(N, 1)))
    eps = 1e-12
    scale = float(np.sqrt(p))
    done = False
    it = 0
    for it in range(1, max_iter + 1):
        dxx = np.sqrt(pair_sqdist(X))
        np.fill_diagonal(dxx, np.inf)
        # repulsion among the points
        rep = (X[:, None, :] - X[None, :, :]) / dxx[:, :, None]
        num = rep.sum(axis=1) / n
        den = np.zeros(n)
        for s in range(0, n, block):
            d = np.sqrt(pair_sqdist(X[s : s + block], Y))
            inv = w[None, :] / np.maximum(d, eps)
            num[s : s + block] += inv @ Y
            den[s : s + block] = inv.sum(axis=1)
        new = num / den[:, None]
        move = float(np.abs(new - X).max())
        X = new
        if move < tol * scale:
            done = True
            break
    return X, it, done


def _nearest_distinct(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """For each row of ``P`` in turn, the nearest row of ``Y`` not yet taken."""
    from scipy.spatial import cKDTree

    N = Y.shape[0]
    tree = cKDTree(Y)
    taken = np.zeros(N, dtype=bool)
    out = np.empty(P.shape[0], dtype=int)
    for i, pt in enumerate(P):
        k = 1
        while True:
            _, idx = tree.query(pt, k=min(k, N))
            idx = np.atleast_1d(idx)
            free = idx[~taken[idx]]
            if free.size:
                out[i] = free[0]
                taken[free[0]] = True
                break
            if k >= N:
                raise DataInsufficient("More points than data rows.")
            k *= 2
    return out


@dataclass
class SupportPointsResult(ResultProtocolMixin):
    """Points that represent a distribution.

    Attributes
    ----------
    points : DataFrame
        The ``n`` points, in the units of the data.
    index : array or None
        With ``subsample=True``, the rows of the data that were picked.
    energy : float
        Energy distance between the points and the sample, up to the
        term that involves the sample alone (so it can be negative);
        lower is better.
    energy_random : float
        The same for random subsamples of that size, averaged.
    model_info : dict

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.normal(size=(2000, 2)), columns=["a", "b"])
    >>> rep = sp.support_points(df, 20, seed=1)
    >>> rep.points.shape
    (20, 2)
    >>> bool(rep.energy < rep.energy_random)
    True
    """

    points: pd.DataFrame
    index: Optional[np.ndarray]
    energy: float
    energy_random: float
    model_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        info = self.model_info
        lines = [
            "Support points",
            "=" * 50,
            f"Points: {self.points.shape[0]}    Dimensions: {self.points.shape[1]}"
            f"    Sample: {info['n_sample']}",
            f"Energy distance (up to a constant): {self.energy:.6g}",
            f"Random subsample of that size:      {self.energy_random:.6g}",
            f"Iterations: {info['iterations']}"
            + ("" if info["converged"] else " (not converged)"),
        ]
        if self.index is not None:
            lines.append("The points are rows of the data (subsample=True).")
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _as_sample(
    data: Any, columns: Optional[Sequence[str]], n: int, seed: Optional[int]
) -> Tuple[pd.DataFrame, bool]:
    """A sample as a DataFrame; draws one when distributions are given."""
    if isinstance(data, dict):
        from scipy.stats import qmc

        names = [str(k) for k in data]
        for k, v in data.items():
            if not hasattr(v, "ppf"):
                raise MethodIncompatibility(
                    f"{k!r}: a dict maps each name to a frozen scipy.stats "
                    "distribution (something with .ppf)."
                )
        N = 1 << int(np.ceil(np.log2(max(8192, 200 * n))))
        u = qmc.Sobol(len(names), scramble=True, seed=seed).random(N)
        u = np.clip(u, 1e-12, 1 - 1e-12)
        cols = {
            nm: np.asarray(data[k].ppf(u[:, j]))
            for j, (k, nm) in enumerate(zip(data, names))
        }
        return pd.DataFrame(cols), True
    if isinstance(data, pd.DataFrame):
        if columns is not None:
            miss = [c for c in columns if c not in data.columns]
            if miss:
                raise ColumnNotFound(f"Not in the data: {', '.join(map(str, miss))}.")
            return data[list(columns)], False
        return data, False
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    return pd.DataFrame(arr, columns=[f"x{j + 1}" for j in range(arr.shape[1])]), False


def support_points(
    data: Any,
    n: int,
    columns: Optional[Sequence[str]] = None,
    weights: Any = None,
    subsample: bool = False,
    standardize: bool = True,
    seed: Optional[int] = None,
    n_starts: int = 1,
    max_iter: int = 500,
    tol: float = 1e-4,
    max_sample: int = 50000,
) -> SupportPointsResult:
    """The ``n`` points that best represent a sample or a distribution.

    A compact stand-in for a distribution: where a random subsample of
    ``n`` points reproduces expectations with error of order
    ``n^(-1/2)``, support points do markedly better, so far fewer model
    evaluations are needed to propagate input uncertainty, and a thinned
    posterior sample keeps the shape of the full one.

    Parameters
    ----------
    data : DataFrame, array, or dict of distributions
        A sample from the target distribution (data, MCMC draws, draws
        of model inputs), one row per draw. Or ``{name: scipy.stats
        frozen distribution}`` for independent inputs with known
        distributions; a large quasi-random sample is then drawn
        internally.
    n : int
        Number of points.
    columns : list of str, optional
        Columns to use. Default: all, which must be numeric.
    weights : str or array, optional
        Weights of the draws (importance weights, survey weights).
    subsample : bool, default False
        Return rows of the data: each support point is replaced by the
        nearest row not yet taken. Use it when the points must be real
        observations.
    standardize : bool, default True
        Scale every column by its standard deviation before measuring
        distances. Energy distance is not invariant to the units.
    seed : int, optional
    n_starts : int, default 1
        Random starts; the lowest energy is kept.
    max_iter : int, default 500
    tol : float, default 1e-4
        Stop when no coordinate (in standard deviations) moves by more
        than ``tol * sqrt(dimension)``.
    max_sample : int, default 50000
        Larger samples are thinned at random to this size for the
        optimisation; the energy is still evaluated on all rows.

    Returns
    -------
    SupportPointsResult
        ``points``, ``index``, ``energy``, ``energy_random``,
        ``summary()``.

    Notes
    -----
    The points minimise ``2 mean|x_i - Y| - mean|x_i - x_j|`` over the
    sample ``Y`` by the convex-concave iteration of Mak and Joseph
    (2018). The problem is not convex, so different starts give
    different, almost equally good, point sets. On the energy criterion
    the result is on par with ``support::sp`` in R.

    An average of a function over the points estimates its expectation.
    The points carry equal weight.

    Examples
    --------
    Twenty points for a bivariate normal with correlation 0.5, from a
    sample:

    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=5000)
    >>> rep = sp.support_points(pd.DataFrame(z, columns=["x1", "x2"]), 20, seed=1)
    >>> rep.points.shape
    (20, 2)

    From known input distributions:

    >>> from scipy import stats
    >>> rep = sp.support_points({"r": stats.norm(0.1, 0.016),
    ...                          "k": stats.uniform(990, 120)}, 30, seed=1)
    >>> list(rep.points.columns)
    ['r', 'k']

    References
    ----------
    mak2018support; szekely2013energy
    """
    n = int(n)
    if n < 1:
        raise MethodIncompatibility(f"n must be at least 1; got {n}.")
    frame, drawn = _as_sample(data, columns, n, seed)
    w = None
    if weights is not None:
        if drawn:
            raise MethodIncompatibility("weights apply to a sample, not to a dict.")
        w = np.asarray(
            data[weights] if isinstance(weights, str) else weights, dtype=float
        )
        if isinstance(weights, str) and weights in frame.columns:
            frame = frame.drop(columns=[weights])
        if w.shape != (frame.shape[0],) or np.any(w < 0) or w.sum() <= 0:
            raise MethodIncompatibility(
                "weights must be non-negative, one per row, not all zero."
            )
        if int((w > 0).sum()) < 2 * n:
            raise DataInsufficient(
                f"Only {int((w > 0).sum())} draws have positive weight; "
                f"{n} points need several times as many."
            )
    non_num = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_num:
        raise MethodIncompatibility(
            f"Non-numeric columns: {', '.join(map(str, non_num))}. Support "
            "points are locations in a numeric space; to pick representative "
            "rows of mixed data use sp.split_data."
        )
    Y = frame.to_numpy(dtype=float)
    if not np.all(np.isfinite(Y)):
        raise MethodIncompatibility("The sample has missing or infinite values.")
    N, p = Y.shape
    if N < 2 * n:
        raise DataInsufficient(
            f"{N} draws are too few for {n} points; the sample should be "
            "several times larger than n."
        )
    if subsample and drawn:
        raise MethodIncompatibility("subsample=True needs data, not distributions.")
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        flat = [str(c) for c, s in zip(frame.columns, sd) if s <= 0]
        raise MethodIncompatibility(f"Constant columns: {', '.join(flat)}.")
    if not standardize:
        sd = np.ones(p)
    Z = (Y - mu) / sd
    rng = np.random.default_rng(seed)
    notes: List[str] = []
    Zopt, wopt = Z, w
    if N > max_sample:
        pick = rng.choice(N, size=int(max_sample), replace=False)
        Zopt = Z[pick]
        wopt = None if w is None else w[pick]
        notes.append(
            f"Optimised on a random {max_sample} of the {N} draws; the energy "
            "is evaluated on all of them."
        )

    def energy(P: np.ndarray) -> float:
        if w is None:
            return _energy_unit(Z, P)
        cross = float((np.sqrt(pair_sqdist(Z, P)).mean(axis=1) * w).sum() / w.sum())
        return float(2.0 * cross - np.sqrt(pair_sqdist(P)).sum() / P.shape[0] ** 2)

    best: Optional[Tuple[float, np.ndarray, int, bool]] = None
    for _ in range(max(int(n_starts), 1)):
        P, it, ok = _sp_ccp(Zopt, n, rng, int(max_iter), float(tol), wopt)
        e = energy(P)
        if best is None or e < best[0]:
            best = (e, P, it, ok)
    assert best is not None
    e, P, it, ok = best
    index = None
    if subsample:
        index = _nearest_distinct(Z, P)
        P = Z[index]
        e = energy(P)
    if n * N <= 20_000_000:
        rand = []
        for _ in range(20):
            ridx = rng.choice(
                N, size=n, replace=False, p=None if w is None else w / w.sum()
            )
            rand.append(energy(Z[ridx]))
        e_rand = float(np.mean(rand))
    else:
        e_rand = float("nan")
    if not ok:
        notes.append(f"Not converged in {max_iter} iterations; raise max_iter.")
    if index is not None:
        points = frame.iloc[index].copy()
    else:
        points = pd.DataFrame(mu + P * sd, columns=frame.columns)
    return SupportPointsResult(
        points=points,
        index=index,
        energy=float(e),
        energy_random=e_rand,
        model_info={
            "n_sample": int(N),
            "iterations": int(it),
            "converged": bool(ok),
            "standardize": bool(standardize),
            "sample_drawn": bool(drawn),
            "notes": notes,
        },
    )


def _energy_unit(Z: np.ndarray, P: np.ndarray) -> float:
    """Energy of ``P`` against an already standardised sample ``Z``."""
    cross = 0.0
    for s in range(0, Z.shape[0], 8192):
        cross += float(np.sqrt(pair_sqdist(Z[s : s + 8192], P)).sum())
    cross /= Z.shape[0] * P.shape[0]
    return float(2.0 * cross - np.sqrt(pair_sqdist(P)).sum() / P.shape[0] ** 2)


def twin_indices(Z: np.ndarray, r: int, start: int) -> np.ndarray:
    """Twinning (Vakayil and Joseph 2022): rows of the smaller twin.

    From the current row take its ``r`` nearest remaining rows (itself
    included); the row itself joins the smaller twin and the whole group
    leaves the pool. Continue from the remaining row nearest to the
    farthest member of the group.
    """
    from scipy.spatial import cKDTree

    N = Z.shape[0]
    alive = np.ones(N, dtype=bool)
    ids = np.arange(N)
    tree = cKDTree(Z)
    tree_size = N
    n_alive = N
    out: List[int] = []

    def nearest(point: np.ndarray, k: int) -> np.ndarray:
        """The ``k`` nearest alive rows, nearest first."""
        nonlocal tree, ids, tree_size
        if n_alive * 2 < tree_size and n_alive > 0:
            ids = np.flatnonzero(alive)
            tree = cKDTree(Z[ids])
            tree_size = ids.size
        want = k
        while True:
            q = min(max(2 * want, 8), tree_size)
            _, loc = tree.query(point, k=q)
            cand = ids[np.atleast_1d(loc)]
            cand = cand[alive[cand]]
            if cand.size >= k or q >= tree_size:
                return np.asarray(cand[:k])
            want = q

    u = int(start)
    while n_alive > 0:
        group = nearest(Z[u], min(r, n_alive))
        # the query row is at distance zero; ties with duplicates aside,
        # it comes first
        if u not in group:
            group = np.r_[u, group[: max(len(group) - 1, 0)]]
        out.append(u)
        far = int(group[-1])
        alive[group] = False
        n_alive -= len(group)
        if n_alive == 0:
            break
        u = int(nearest(Z[far], 1)[0])
    return np.asarray(out, dtype=int)


@dataclass
class DataSplitResult(ResultProtocolMixin):
    """A split of a data set into two parts with the same distribution.

    Unpacks as ``train, test = sp.split_data(df)``.

    Attributes
    ----------
    train, test : DataFrame
    test_index : ndarray
        Positions (0-based) of the test rows in the data.
    energy : float
        Energy distance of the test set to the full data, up to a
        constant; lower is more representative.
    energy_random : float
        The same for random splits of that size, averaged.
    model_info : dict

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200)})
    >>> df["y"] = df["x"] ** 2 + rng.normal(size=200)
    >>> train, test = sp.split_data(df, test_size=0.2, seed=1)
    >>> train.shape[0], test.shape[0]
    (160, 40)
    """

    train: pd.DataFrame
    test: pd.DataFrame
    test_index: np.ndarray
    energy: float
    energy_random: float
    model_info: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        yield self.train
        yield self.test

    def summary(self) -> str:
        info = self.model_info
        lines = [
            f"Data split by {info['method']}",
            "=" * 50,
            f"Train: {self.train.shape[0]}    Test: {self.test.shape[0]}",
            f"Energy distance of the test set to the data: {self.energy:.6g}",
            f"Random split of that size:                  {self.energy_random:.6g}",
        ]
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    columns: Optional[Sequence[str]] = None,
    method: str = "auto",
    start: Optional[int] = None,
    seed: Optional[int] = None,
    max_iter: int = 500,
    tol: float = 1e-4,
) -> DataSplitResult:
    """Split data into a training and a test set that resemble each other.

    A random split leaves the test set unrepresentative by chance: a
    region of the covariates, or a tail of the outcome, can be missing
    from it, and the measured test error then depends on the luck of the
    draw. This split is chosen so that the test set has, as nearly as
    possible, the joint distribution of the whole data set, outcome
    included.

    Parameters
    ----------
    data : DataFrame
        One row per observation, outcome and features together.
    test_size : float, default 0.2
        Share of the rows in the test set.
    columns : list of str, optional
        Columns that define the distribution. Default: all. Categorical
        columns are expanded into Helmert contrasts.
    method : {'auto', 'support', 'twinning'}, default 'auto'
        ``'support'`` (SPlit, Joseph and Vakayil 2022) computes support
        points of the data and takes the nearest rows. ``'twinning'``
        (Vakayil and Joseph 2022) walks through the data taking one row
        out of each group of ``r = 1 / test_size`` neighbours; it is far
        faster on large data and needs ``1 / test_size`` to be a whole
        number. ``'auto'`` uses twinning when it applies and there are
        more than 2,000 rows, otherwise support points.
    start : int, optional
        Twinning only: the row (position) where the walk starts. With it
        the split is deterministic.
    seed : int, optional
    max_iter, tol
        Passed to the support-points iteration.

    Returns
    -------
    DataSplitResult
        ``train``, ``test``, ``test_index``, ``energy``,
        ``energy_random``; unpacks into ``train, test``.

    Notes
    -----
    The split does not depend on any model. It is meant for assessing
    prediction error of a model fitted once. Do not use it where the
    independence of the two parts is what a procedure relies on
    (sample splitting for honest inference, cross-fitting): there the
    split must be random.

    With ``method='twinning'`` and the same ``start`` the test rows are
    those returned by ``twinning::twin`` in R.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> df = pd.DataFrame({"x": rng.normal(size=300)})
    >>> df["y"] = np.sin(2 * df["x"]) + 0.3 * rng.normal(size=300)
    >>> split = sp.split_data(df, test_size=0.2, seed=1)
    >>> split.test.shape
    (60, 2)
    >>> bool(split.energy < split.energy_random)
    True

    References
    ----------
    joseph2022split; vakayil2022data; mak2018support
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("data must be a DataFrame.")
    if not 0.0 < test_size < 1.0:
        raise MethodIncompatibility(f"test_size must be in (0, 1); got {test_size}.")
    cols = list(data.columns) if columns is None else list(columns)
    miss = [c for c in cols if c not in data.columns]
    if miss:
        raise ColumnNotFound(f"Not in the data: {', '.join(map(str, miss))}.")
    N = data.shape[0]
    n_test = int(round(test_size * N))
    if n_test < 1 or n_test >= N:
        raise DataInsufficient(
            f"test_size={test_size} leaves {n_test} of {N} rows for testing."
        )
    if data[cols].isna().any().any():
        raise MethodIncompatibility(
            "The columns that define the split have missing values."
        )
    Z = helmert_frame(data, cols)
    kind = str(method).lower()
    if kind not in ("auto", "support", "twinning", "split"):
        raise MethodIncompatibility(
            f"method must be 'auto', 'support' or 'twinning'; got {method!r}."
        )
    kind = "support" if kind == "split" else kind
    inv = 1.0 / test_size
    whole = abs(inv - round(inv)) < 1e-9
    if kind == "auto":
        kind = "twinning" if whole and N > 2000 else "support"
    rng = np.random.default_rng(seed)
    notes: List[str] = []
    info: Dict[str, Any] = {"notes": notes}
    if kind == "twinning":
        if not whole:
            raise MethodIncompatibility(
                f"Twinning needs 1 / test_size to be a whole number; got "
                f"{inv:.4g}. Use method='support'."
            )
        r = int(round(inv))
        u1 = int(rng.integers(N)) if start is None else int(start)
        if not 0 <= u1 < N:
            raise MethodIncompatibility(f"start must be a row position in 0..{N - 1}.")
        idx = twin_indices(Z, r, u1)
        info.update(method="twinning", r=r, start=u1)
    else:
        if start is not None:
            raise MethodIncompatibility("start applies to method='twinning'.")
        if N * n_test > 60_000_000:
            raise MethodIncompatibility(
                f"{N} rows with {n_test} test rows are too many for support "
                "points; use method='twinning'."
            )
        P, it, ok = _sp_ccp(Z, n_test, rng, int(max_iter), float(tol))
        idx = _nearest_distinct(Z, P)
        if not ok:
            notes.append(f"Support points not converged in {max_iter} iterations.")
        info.update(method="support points (SPlit)", iterations=int(it))
    idx = np.sort(idx)
    mask = np.zeros(N, dtype=bool)
    mask[idx] = True
    energy = _energy_unit(Z, Z[idx])
    if N * len(idx) <= 20_000_000:
        e_rand = float(
            np.mean(
                [
                    _energy_unit(Z, Z[rng.choice(N, size=len(idx), replace=False)])
                    for _ in range(20)
                ]
            )
        )
    else:
        e_rand = float("nan")
    info["columns"] = [str(c) for c in cols]
    return DataSplitResult(
        train=data.loc[~mask],
        test=data.loc[mask],
        test_index=idx,
        energy=float(energy),
        energy_random=e_rand,
        model_info=info,
    )
