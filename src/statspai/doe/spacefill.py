"""Space-filling designs: ``sp.space_filling`` and ``sp.design_augment``.

A Latin hypercube design puts one run in each of ``n`` equal slices of
every factor, so no run is wasted when only a few factors matter. Among
Latin hypercubes the function searches, by simulated annealing over swaps
within a column, for the one that is best on a chosen criterion; for the
distance-based criteria the levels are then moved continuously.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._common import DesignResult, pair_sqdist, resolve_factors, to_unit
from .criteria import all_criteria

_METHODS = ("maxpro", "maximin", "uniform", "lhs", "sobol", "halton", "random")
_ALIASES = {
    "maxprolhd": "maxpro",
    "maximinlhd": "maximin",
    "maximinlhs": "maximin",
    "uniformlhd": "uniform",
    "latin": "lhs",
    "latinhypercube": "lhs",
    "lhd": "lhs",
    "mc": "random",
    "montecarlo": "random",
}


def _method(name: str) -> str:
    key = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
    key = _ALIASES.get(key, key)
    if key not in _METHODS:
        raise MethodIncompatibility(
            f"method must be one of {', '.join(_METHODS)}; got {name!r}."
        )
    return key


def _random_lhd(n: int, p: int, rng: np.random.Generator) -> np.ndarray:
    return np.column_stack([(rng.permutation(n) + 0.5) / n for _ in range(p)])


class _PairCriterion:
    """A criterion that is a sum over pairs of runs, to be minimised.

    The state is a matrix with one entry per pair that is additive over
    the factors (log pair terms for MaxPro and the discrepancy, squared
    distances for maximin), so a swap within one column updates two rows
    at a cost linear in the number of runs.
    """

    def __init__(
        self,
        kind: str,
        p: int,
        r: float,
        delta: float,
        n: int,
        offset: Optional[np.ndarray] = None,
    ) -> None:
        self.kind, self.p, self.r, self.delta = kind, p, r, delta
        # fixed log pair terms of factors the search does not move
        self.offset = np.zeros((n, n)) if offset is None else offset
        # log pair terms are shifted by a constant so that sums stay finite
        if kind == "maxpro":
            self.shift = -p * np.log(1.0 / (3.0 * n) ** 2 + delta)
        elif kind == "maximin":
            self.shift = 0.5 * r * np.log(n)
        else:
            self.shift = 0.0

    def piece(self, diff: np.ndarray) -> np.ndarray:
        """Contribution of one factor to the state, from its differences."""
        if self.kind == "maxpro":
            return np.asarray(-np.log(diff * diff + self.delta))
        if self.kind == "maximin":
            return np.asarray(diff * diff)
        a = np.abs(diff)
        return np.asarray(np.log(1.5 - a * (1.0 - a)))

    def terms(self, state: np.ndarray) -> np.ndarray:
        if self.kind == "maximin":
            return np.asarray(np.exp(-0.5 * self.r * np.log(state) - self.shift))
        return np.asarray(np.exp(state - self.shift))

    def state(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        S = self.offset.copy()
        for col in X.T:
            S += self.piece(col[:, None] - col[None, :])
        S[np.diag_indices(n)] = np.inf if self.kind == "maximin" else -np.inf
        return S

    def total(self, X: np.ndarray) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            return 0.5 * float(self.terms(self.state(X)).sum())


def _anneal(
    X: np.ndarray, crit: _PairCriterion, iterations: int, rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    """Swap two levels within a column; accept by the Metropolis rule on logs."""
    try:
        from ._spacefill_core import MAXIMIN, MAXPRO, UNIFORM, anneal
    except ImportError:  # numba missing or broken: the slow path
        with np.errstate(divide="ignore", invalid="ignore"):
            return _anneal_loop(X, crit, iterations, rng)
    code = {"maxpro": MAXPRO, "maximin": MAXIMIN, "uniform": UNIFORM}[crit.kind]
    best_X, _ = anneal(
        np.ascontiguousarray(X, dtype=np.float64),
        code,
        float(crit.r),
        float(crit.delta),
        float(crit.shift),
        int(iterations),
        int(rng.integers(2**31 - 1)),
        np.ascontiguousarray(crit.offset, dtype=np.float64),
    )
    # the running total drifts by rounding; report the exact value
    return best_X, crit.total(best_X)


def _anneal_loop(
    X: np.ndarray, crit: _PairCriterion, iterations: int, rng: np.random.Generator
) -> Tuple[np.ndarray, float]:
    n, p = X.shape
    X = X.copy()
    S = crit.state(X)
    T = crit.terms(S)
    cur = 0.5 * float(T.sum())
    best, best_X = cur, X.copy()
    if n < 3 or iterations <= 0:
        return best_X, best
    rows = T.sum(axis=1)
    ii = rng.integers(n, size=iterations + 64)
    jj = (ii + 1 + rng.integers(n - 1, size=iterations + 64)) % n
    kk = rng.integers(p, size=iterations + 64)
    uu = rng.random(iterations + 64)

    def move(q: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        i, j, k = ii[q], jj[q], kk[q]
        col = X[:, k]
        change = crit.piece(col[j] - col) - crit.piece(col[i] - col)
        change[i] = change[j] = 0.0
        si, sj = S[i] + change, S[j] - change
        ti, tj = crit.terms(si), crit.terms(sj)
        d = float(np.add.reduce(ti) + np.add.reduce(tj) - rows[i] - rows[j])
        return si, sj, ti, tj, d

    # starting temperature from the size of typical moves, on the log scale
    trial = []
    for q in range(iterations, iterations + 64):
        d = move(q)[4]
        if cur + d > 0 and np.isfinite(d):
            trial.append(abs(np.log1p(d / cur)))
    t0 = max(0.5 * float(np.median(trial)) if trial else 1e-2, 1e-10)
    cool = 1e-3 ** (1.0 / max(iterations - 1, 1))
    temp = t0
    for q in range(iterations):
        si, sj, ti, tj, d = move(q)
        new = cur + d
        if new > 0 and np.isfinite(new):
            dlog = np.log(new / cur)
            if dlog <= 0 or uu[q] < np.exp(-dlog / temp):
                i, j, k = ii[q], jj[q], kk[q]
                X[i, k], X[j, k] = X[j, k], X[i, k]
                S[i], S[:, i], S[j], S[:, j] = si, si, sj, sj
                rows += (ti - T[i]) + (tj - T[j])
                T[i], T[:, i], T[j], T[:, j] = ti, ti, tj, tj
                rows[i], rows[j] = np.add.reduce(ti), np.add.reduce(tj)
                cur = new
                if cur < best:
                    best, best_X = cur, X.copy()
        temp *= cool
    # the running total drifts by rounding; report the exact value
    return best_X, crit.total(best_X)


def _polish(
    X0: np.ndarray,
    kind: str,
    r: float,
    delta: float,
    offset: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Move the levels continuously within the cube (L-BFGS-B, exact gradient)."""
    n, p = X0.shape
    iu = np.triu_indices(n, k=1)

    def fun(v: np.ndarray) -> Tuple[float, np.ndarray]:
        X = v.reshape(n, p)
        diff = X[:, None, :] - X[None, :, :]
        sq = diff * diff
        if kind == "maxpro":
            den = sq + delta
            den[np.arange(n), np.arange(n), :] = 1.0
            lg = -np.log(np.maximum(den, 1e-300)).sum(axis=2)
            if offset is not None:
                lg = lg + offset
        else:
            d2 = sq.sum(axis=2)
            d2[np.arange(n), np.arange(n)] = 1.0
            lg = -0.5 * r * np.log(np.maximum(d2, 1e-300))
        lg[np.arange(n), np.arange(n)] = -np.inf
        m = lg[iu].max()
        q = np.exp(lg - m)
        tot = q[iu].sum()
        if kind == "maxpro":
            g = (q[:, :, None] * (-2.0 * diff / np.maximum(den, 1e-300))).sum(axis=1)
        else:
            g = (q / np.maximum(d2, 1e-300))[:, :, None] * diff
            g = -r * g.sum(axis=1)
        return float(m + np.log(tot)), (g / tot).ravel()

    f0, _ = fun(X0.ravel())
    sol = optimize.minimize(
        fun,
        X0.ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * (n * p),
        options={"maxiter": 500, "maxfun": 2000},
    )
    return sol.x.reshape(n, p) if np.isfinite(sol.fun) and sol.fun < f0 else X0


def _feasible(
    cand: np.ndarray,
    constraint: Callable[..., Any],
    names: List[str],
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Rows of ``cand`` (unit cube) that satisfy the constraint, in real units."""
    frame = pd.DataFrame(lo + cand * (hi - lo), columns=names)
    try:
        ok = np.asarray(constraint(frame))
    except (TypeError, ValueError, KeyError, AttributeError, IndexError):
        ok = np.zeros(0)  # the constraint takes one run at a time
    if ok.shape != (cand.shape[0],):
        ok = np.array([bool(constraint(row)) for _, row in frame.iterrows()])
    return ok.astype(bool)


def _greedy(
    exist: np.ndarray, cand: np.ndarray, n_new: int, criterion: str, delta: float
) -> List[int]:
    """Add candidates one at a time, each the best given the runs so far."""
    N, p = cand.shape
    chosen: List[int] = []
    taken = np.zeros(N, dtype=bool)
    if criterion == "maxpro":
        score = np.zeros(N)
        # scale by the largest term so that the running sums stay finite
        for x in exist:
            diff = cand - x
            with np.errstate(divide="ignore"):
                score += np.exp(-np.log(diff * diff + delta).sum(axis=1))
        for _ in range(n_new):
            s = np.where(taken, np.inf, score)
            i = int(np.argmin(s))
            if not np.isfinite(s[i]):
                raise DataInsufficient(
                    "Every remaining candidate repeats a level of a run already "
                    "in the design, so the MaxPro criterion is infinite. Use "
                    "more candidates, or delta > 0."
                )
            chosen.append(i)
            taken[i] = True
            diff = cand - cand[i]
            with np.errstate(divide="ignore"):
                score += np.exp(-np.log(diff * diff + delta).sum(axis=1))
    else:
        near = np.full(N, np.inf)
        for start in range(0, exist.shape[0], 512):
            near = np.minimum(
                near, pair_sqdist(cand, exist[start : start + 512]).min(axis=1)
            )
        for _ in range(n_new):
            s = np.where(taken, -np.inf, near)
            i = int(np.argmax(s))
            chosen.append(i)
            taken[i] = True
            near = np.minimum(near, ((cand - cand[i]) ** 2).sum(axis=1))
    return chosen


def _sobol(n: int, p: int, seed: Optional[int]) -> np.ndarray:
    from scipy.stats import qmc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(qmc.Sobol(p, scramble=True, seed=seed).random(n))


def space_filling(
    n: int,
    factors: Any,
    method: str = "maxpro",
    seed: Optional[int] = None,
    n_starts: int = 3,
    iterations: Optional[int] = None,
    polish: Optional[bool] = None,
    constraint: Optional[Callable[..., Any]] = None,
    n_candidates: Optional[int] = None,
    delta: float = 0.0,
    r: Optional[float] = None,
    qualitative: Optional[Mapping[str, Any]] = None,
) -> DesignResult:
    """A space-filling design: runs spread evenly over the factor region.

    The design to use when little is known about the response surface:
    a simulation study over a parameter region, the training runs of a
    surrogate model, starting values for a multi-start optimiser, or the
    draws of a simulated-moments estimator.

    Parameters
    ----------
    n : int
        Number of runs.
    factors : int, list of str, or dict
        The number of factors, their names, or ``{name: (lower, upper)}``.
        Without bounds the design is on the unit cube.
    method : str, default 'maxpro'
        ``'maxpro'``
            Maximum projection design (Joseph, Gul and Ba 2015): fills
            the full region and every lower-dimensional projection.
            The default, because usually only some factors matter.
        ``'maximin'``
            Latin hypercube with the largest minimum distance between
            runs (Morris and Mitchell 1995).
        ``'uniform'``
            Latin hypercube with the smallest wrap-around discrepancy;
            the one to use for numerical integration.
        ``'lhs'``
            A random Latin hypercube (McKay, Beckman and Conover 1979).
        ``'sobol'``, ``'halton'``
            Scrambled low-discrepancy sequences. Extensible: the first
            ``m`` runs of a longer sequence are a sequence themselves.
        ``'random'``
            Independent uniform draws, the baseline.
    seed : int, optional
    n_starts : int, default 3
        Independent searches; the best design is returned.
    iterations : int, optional
        Swaps tried per search. Default ``min(400000, max(20000, 1000 n p))``.
    polish : bool, optional
        Move the levels continuously after the search. Default ``True``
        for ``'maxpro'`` (the design is then no longer on the Latin
        hypercube midpoints but still has ``n`` distinct levels per
        factor) and ``False`` otherwise.
    constraint : callable, optional
        Takes a DataFrame of candidate runs in the units of the factors
        and returns one boolean per row (or takes one row and returns a
        boolean). When given, the design is built by adding feasible
        candidates one at a time (``sp.design_augment``), since a Latin
        hypercube does not exist on an irregular region; ``method`` must
        be ``'maxpro'`` or ``'maximin'``.
    n_candidates : int, optional
        Size of the candidate set under ``constraint``.
    delta : float, default 0
        Added to squared differences in the MaxPro criterion.
    r : float, optional
        Power of the reciprocal-distance criterion that stands in for
        maximin during the search. Default ``2 p``.

    qualitative : dict, optional
        ``{name: [levels]}`` for factors without an order (a model
        variant, a solver, a region). Each level gets the same number of
        runs, as nearly as ``n`` allows, and the quantitative factors are
        arranged so that the runs fill the space within every level as
        well as overall (Joseph, Gul and Ba 2020). ``method`` must be
        ``'maxpro'``. ``criteria['maxpro_qq']`` is the criterion that
        was minimised; the other measures refer to the quantitative
        factors alone.

    Returns
    -------
    DesignResult
        ``design`` (the runs as a DataFrame), ``criteria``, ``summary()``,
        ``plot()``.

    Notes
    -----
    The search is stochastic and finds a good design, not a certified
    optimum. On the criterion value the designs are on par with those of
    the R packages ``SFDesign`` and ``MaxPro``.

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.space_filling(20, {"beta": (0.9, 0.99), "sigma": (1, 5)}, seed=1)
    >>> d.design.shape
    (20, 2)
    >>> bool(d.design["beta"].between(0.9, 0.99).all())
    True
    >>> lhs = sp.space_filling(20, 2, method="lhs", seed=1)
    >>> bool(d.criteria["maxpro"] < lhs.criteria["maxpro"])
    True

    References
    ----------
    joseph2015maximum; joseph2020designing; morris1995exploratory;
    mckay1979comparison; joseph2025experimental
    """
    kind = _method(method)
    names, lo, hi = resolve_factors(factors)
    p = len(names)
    n = int(n)
    if n < 2:
        raise DataInsufficient(f"A design needs at least two runs; got n={n}.")
    if delta < 0:
        raise MethodIncompatibility(f"delta must be non-negative; got {delta}.")
    rng = np.random.default_rng(seed)
    rr = float(2 * p if r is None else r)
    info: Dict[str, Any] = {"notes": []}
    nominal: Dict[str, np.ndarray] = {}
    offset = None
    if qualitative:
        if kind != "maxpro" or constraint is not None:
            raise MethodIncompatibility(
                "Qualitative factors are supported with method='maxpro' and "
                "without a constraint only."
            )
        clash = [str(k) for k in qualitative if str(k) in names]
        if clash:
            raise MethodIncompatibility(
                f"{clash[0]!r} is both quantitative and qualitative."
            )
        sets = {str(k): list(v) for k, v in qualitative.items()}
        for nm, lv in sets.items():
            if len(lv) < 2 or len(set(map(str, lv))) != len(lv):
                raise MethodIncompatibility(
                    f"Qualitative factor {nm!r} needs at least two distinct levels."
                )
        # every combination of levels equally often, in random run order
        combos = list(itertools.product(*[range(len(v)) for v in sets.values()]))
        reps = int(np.ceil(n / len(combos)))
        codes = np.array((combos * reps), dtype=int)
        codes = codes[rng.permutation(len(codes))[:n]] if n % len(combos) else codes[:n]
        if n % len(combos):
            info["notes"].append(
                f"{n} runs are not a multiple of the {len(combos)} level "
                "combinations: the levels are not perfectly balanced."
            )
        offset = np.zeros((n, n))
        for j, (nm, lv) in enumerate(sets.items()):
            diff = (codes[:, j][:, None] != codes[:, j][None, :]).astype(float)
            offset -= 2.0 * np.log(diff + 1.0 / len(lv))
            nominal[nm] = np.array(lv, dtype=object)[codes[:, j]]
    if constraint is not None:
        if kind not in ("maxpro", "maximin"):
            raise MethodIncompatibility(
                "With a constraint the design is built from candidates; "
                "method must be 'maxpro' or 'maximin'."
            )
        N = int(n_candidates) if n_candidates else max(2**13, 400 * n)
        N = 1 << int(np.ceil(np.log2(N)))
        cand = _sobol(N, p, int(rng.integers(2**31)))
        cand = cand[_feasible(cand, constraint, names, lo, hi)]
        if cand.shape[0] < 5 * n:
            raise DataInsufficient(
                f"Only {cand.shape[0]} of {N} candidates satisfy the constraint; "
                f"at least {5 * n} are needed for {n} runs. Raise n_candidates."
            )
        first = int(np.argmin(((cand - cand.mean(axis=0)) ** 2).sum(axis=1)))
        rest = np.delete(cand, first, axis=0)
        idx = _greedy(cand[[first]], rest, n - 1, kind, delta)
        X = np.vstack([cand[[first]], rest[idx]])
        info.update(feasible_candidates=int(cand.shape[0]), candidates=N)
        info["notes"].append(
            "Built by adding feasible candidates one at a time; not a Latin "
            "hypercube."
        )
        label = f"{kind} design on a constrained region"
    elif kind == "random":
        X, label = rng.random((n, p)), "random design"
    elif kind == "lhs":
        X, label = _random_lhd(n, p, rng), "random Latin hypercube"
    elif kind in ("sobol", "halton"):
        from scipy.stats import qmc

        s = int(rng.integers(2**31))
        if kind == "sobol":
            X = _sobol(n, p, s)
        else:
            X = np.asarray(qmc.Halton(p, scramble=True, seed=s).random(n))
        label = f"scrambled {kind.capitalize()} points"
    else:
        its = (
            int(iterations)
            if iterations is not None
            else int(min(400000, max(20000, 1000 * n * p)))
        )
        crit = _PairCriterion(kind, p, rr, delta, n, offset)
        best_X, best_f = None, np.inf
        for _ in range(max(int(n_starts), 1)):
            Xs, fs = _anneal(_random_lhd(n, p, rng), crit, its, rng)
            if fs < best_f:
                best_X, best_f = Xs, fs
        assert best_X is not None
        X = best_X
        do_polish = (kind == "maxpro") if polish is None else bool(polish)
        if do_polish and kind == "uniform":
            raise MethodIncompatibility(
                "polish is available for 'maxpro' and 'maximin' only."
            )
        if do_polish and n * n * p <= 4_000_000:
            X = _polish(X, kind, rr, delta, offset)
        elif do_polish:
            info["notes"].append("Too large to polish; the Latin hypercube is kept.")
            do_polish = False
        label = {
            "maxpro": "maximum projection design",
            "maximin": "maximin Latin hypercube",
            "uniform": "uniform Latin hypercube",
        }[kind] + (" (polished)" if do_polish and kind == "maximin" else "")
        info.update(iterations=its, n_starts=int(n_starts), polished=do_polish)
    info["method"] = kind
    frame = pd.DataFrame(lo + X * (hi - lo), columns=names)
    measures = all_criteria(X, delta=delta, r=r, fill=p <= 8)
    if nominal:
        assert offset is not None
        from .criteria import _log_maxpro_terms

        iu = np.triu_indices(n, k=1)
        lg = _log_maxpro_terms(X, delta) + offset[iu]
        top = lg.max()
        measures["maxpro_qq"] = float(
            np.exp((top + np.log(np.exp(lg - top).mean())) / (p + len(nominal)))
        )
        for nm, col in nominal.items():
            frame[nm] = col
        label += f" with {len(nominal)} qualitative factor(s)"
        info["qualitative"] = list(nominal)
    return DesignResult(
        design=frame,
        unit=X,
        method=label,
        criteria=measures,
        lower=lo,
        upper=hi,
        model_info=info,
    )


def design_augment(
    design: Any,
    n_new: int,
    candidates: Any = None,
    criterion: str = "maxpro",
    bounds: Optional[Mapping[str, Tuple[float, float]]] = None,
    constraint: Optional[Callable[..., Any]] = None,
    n_candidates: Optional[int] = None,
    delta: float = 0.0,
    seed: Optional[int] = None,
) -> DesignResult:
    """Add runs to an existing design so that the whole stays space-filling.

    For a second batch after the first has been run, for validation runs
    that keep away from the training runs, or for a design on a region of
    irregular shape (pass only feasible candidates).

    Parameters
    ----------
    design : DataFrame, array or DesignResult
        The runs already made. They are kept as they are.
    n_new : int
        Runs to add.
    candidates : DataFrame or array, optional
        The points the new runs are chosen from, in the units of the
        design. Default: a scrambled Sobol' set over the region.
    criterion : {'maxpro', 'maximin'}, default 'maxpro'
        ``'maxpro'`` adds, each time, the candidate with the smallest sum
        over existing runs of ``1 / prod_l (x_l - c_l)^2``; ``'maximin'``
        the candidate farthest from its nearest run.
    bounds : dict, optional
        ``{factor: (lower, upper)}``. Needed when the design is not on
        the unit cube and is not a DesignResult.
    constraint : callable, optional
        Filters the default candidates; see ``sp.space_filling``.
    n_candidates : int, optional
        Size of the default candidate set. Default
        ``max(8192, 400 (n + n_new))`` rounded up to a power of two.
    delta : float, default 0
        Added to squared differences in the MaxPro criterion. With 0 a
        candidate that repeats a level of an existing run is never
        chosen.
    seed : int, optional
        For the default candidate set.

    Returns
    -------
    DesignResult
        The augmented design, old runs first. ``model_info['new_rows']``
        lists the positions of the added runs, ``model_info
        ['candidate_index']`` their rows in ``candidates``.

    Notes
    -----
    The search is greedy: each run is the best addition given those
    before it, which is what makes the result usable in batches. Given
    the same candidates the ``'maxpro'`` choice is the one made by
    ``MaxPro::MaxProAugment`` in R.

    Examples
    --------
    >>> import statspai as sp
    >>> first = sp.space_filling(10, 2, seed=1)
    >>> both = sp.design_augment(first, 10, seed=2)
    >>> both.design.shape
    (20, 2)
    >>> bool((both.design.iloc[:10].to_numpy() == first.design.to_numpy()).all())
    True

    References
    ----------
    joseph2015maximum; johnson1990minimax
    """
    crit = str(criterion).lower()
    if crit not in ("maxpro", "maximin"):
        raise MethodIncompatibility(
            f"criterion must be 'maxpro' or 'maximin'; got {criterion!r}."
        )
    n_new = int(n_new)
    if n_new < 1:
        raise MethodIncompatibility(f"n_new must be at least 1; got {n_new}.")
    if delta < 0:
        raise MethodIncompatibility(f"delta must be non-negative; got {delta}.")
    X, names, lo, hi = to_unit(design, bounds)
    n, p = X.shape
    info: Dict[str, Any] = {"notes": [], "criterion": crit}
    if candidates is None:
        N = int(n_candidates) if n_candidates else max(2**13, 400 * (n + n_new))
        N = 1 << int(np.ceil(np.log2(N)))
        cand = _sobol(N, p, seed)
        if constraint is not None:
            cand = cand[_feasible(cand, constraint, names, lo, hi)]
        info["candidates"] = "scrambled Sobol' set"
    else:
        if constraint is not None:
            raise MethodIncompatibility(
                "constraint filters the default candidates; with candidates= "
                "pass only the feasible ones."
            )
        C = (
            candidates.to_numpy(dtype=float)
            if isinstance(candidates, pd.DataFrame)
            else np.asarray(candidates, dtype=float)
        )
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[1] != p:
            raise MethodIncompatibility(
                f"candidates have {C.shape[1]} columns, the design has {p}."
            )
        if isinstance(candidates, pd.DataFrame) and isinstance(
            getattr(design, "design", design), pd.DataFrame
        ):
            cn = [str(c) for c in candidates.columns]
            if set(cn) != set(names):
                raise MethodIncompatibility(
                    "candidates and the design do not name the same factors."
                )
            C = candidates[names].to_numpy(dtype=float)
        cand = (C - lo) / (hi - lo)
        info["candidates"] = "supplied"
    if cand.shape[0] < n_new:
        raise DataInsufficient(
            f"{cand.shape[0]} candidates are fewer than the {n_new} runs to add."
        )
    idx = _greedy(X, cand, n_new, crit, delta)
    out = np.vstack([X, cand[idx]])
    info["new_rows"] = list(range(n, n + n_new))
    info["candidate_index"] = [int(i) for i in idx]
    return DesignResult(
        design=pd.DataFrame(lo + out * (hi - lo), columns=names),
        unit=out,
        method=f"design augmented by {crit}",
        criteria=all_criteria(out, delta=delta, fill=p <= 8),
        lower=lo,
        upper=hi,
        model_info=info,
    )
