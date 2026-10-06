"""Global sensitivity analysis of a model: ``sp.sobol_indices``,
``sp.morris_screening``.

Which inputs of a model drive its output? For a structural model, a
simulator or a fitted surrogate ``y = f(x_1, ..., x_p)`` with uncertain
inputs, the variance of ``y`` is split among the inputs. The first-order
index of an input is the share of the variance explained by that input
alone; its total index adds every interaction it takes part in. An input
with a total index near zero can be fixed at any value without changing
the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility


def _inputs(factors: Any) -> Tuple[List[str], List[Any]]:
    """Names and, per input, a (lower, upper) pair or a frozen distribution."""
    if isinstance(factors, (int, np.integer)) and not isinstance(factors, bool):
        p = int(factors)
        if p < 1:
            raise MethodIncompatibility("At least one input is needed.")
        return [f"x{j + 1}" for j in range(p)], [(0.0, 1.0)] * p
    if isinstance(factors, dict):
        names, spec = [], []
        for k, v in factors.items():
            names.append(str(k))
            if hasattr(v, "ppf"):
                spec.append(v)
                continue
            try:
                lo, hi = float(v[0]), float(v[1])
            except (TypeError, IndexError, ValueError) as exc:
                raise MethodIncompatibility(
                    f"{k!r}: give (lower, upper) or a frozen scipy.stats "
                    "distribution."
                ) from exc
            if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                raise MethodIncompatibility(
                    f"{k!r}: bounds must satisfy lower < upper."
                )
            spec.append((lo, hi))
        if not names:
            raise MethodIncompatibility("At least one input is needed.")
        return names, spec
    if isinstance(factors, str):
        raise MethodIncompatibility(
            "factors is a number, a list of names or a dict of bounds / "
            "distributions."
        )
    names = [str(k) for k in factors]
    if not names or len(set(names)) != len(names):
        raise MethodIncompatibility("Input names must be non-empty and distinct.")
    return names, [(0.0, 1.0)] * len(names)


def _from_unit(U: np.ndarray, spec: Sequence[Any]) -> np.ndarray:
    out = np.empty_like(U)
    for j, s in enumerate(spec):
        if hasattr(s, "ppf"):
            out[:, j] = s.ppf(np.clip(U[:, j], 1e-12, 1 - 1e-12))
        else:
            out[:, j] = s[0] + U[:, j] * (s[1] - s[0])
    return out


def _evaluate(
    func: Callable[..., Any], X: np.ndarray, names: List[str], pass_as: str
) -> np.ndarray:
    arg = pd.DataFrame(X, columns=names) if pass_as == "frame" else X
    if pass_as == "rows":
        y = np.array([float(func(row)) for row in X])
    else:
        y = np.asarray(func(arg), dtype=float)
        if y.ndim == 2 and 1 in y.shape:
            y = y.reshape(-1)
    if y.shape != (X.shape[0],):
        raise MethodIncompatibility(
            f"func returned shape {y.shape} for {X.shape[0]} runs; it must "
            "return one number per run. For a function of a single run use "
            "pass_as='rows'."
        )
    if not np.all(np.isfinite(y)):
        raise MethodIncompatibility(
            f"func returned {int((~np.isfinite(y)).sum())} missing or infinite "
            "values. A variance decomposition needs the output everywhere on "
            "the input region."
        )
    return y


@dataclass
class SobolResult(ResultProtocolMixin):
    """Variance-based sensitivity indices.

    Attributes
    ----------
    indices : DataFrame
        Indexed by input: ``first`` and ``total`` indices, with
        ``_lower`` / ``_upper`` bootstrap limits when ``n_boot > 0``.
    variance : float
        Variance of the output.
    mean : float
    n_evaluations : int
    model_info : dict

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.sobol_indices(lambda X: X[:, 0] + 2 * X[:, 1], 3,
    ...                        n=1024, pass_as="array", seed=0, n_boot=0)
    >>> [round(float(v), 1) for v in res.indices["first"]]
    [0.2, 0.8, 0.0]
    """

    indices: pd.DataFrame
    variance: float
    mean: float
    n_evaluations: int
    model_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        info = self.model_info
        lines = [
            "Sobol' sensitivity indices",
            "=" * 60,
            f"Model evaluations: {self.n_evaluations}    Output variance: "
            f"{self.variance:.6g}",
            self.indices.to_string(float_format=lambda v: f"{v:.4f}"),
            f"Sum of first-order indices: {self.indices['first'].sum():.4f} "
            "(1 means no interactions)",
        ]
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Bars of the first-order index and of the interaction part."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 3.8))
        first = self.indices["first"].clip(lower=0)
        rest = (self.indices["total"] - first).clip(lower=0)
        pos = np.arange(first.size)
        ax.bar(pos, first, label="first order", **kwargs)
        ax.bar(pos, rest, bottom=first, label="interactions", alpha=0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels(self.indices.index)
        ax.set_ylabel("share of variance")
        ax.legend(frameon=False)
        return ax


def _sobol_estimates(
    yA: np.ndarray, yB: np.ndarray, yAB: np.ndarray, estimator: str
) -> Tuple[np.ndarray, np.ndarray, float]:
    """First-order and total indices from the three sets of evaluations."""
    var = float(np.var(yA, ddof=1))
    total = 0.5 * np.mean((yA[:, None] - yAB) ** 2, axis=0) / var
    if estimator == "jansen":
        first = 1.0 - 0.5 * np.mean((yB[:, None] - yAB) ** 2, axis=0) / var
    else:
        first = np.mean(yB[:, None] * (yAB - yA[:, None]), axis=0) / var
    return first, total, var


def sobol_indices(
    func: Callable[..., Any],
    factors: Any,
    n: int = 1024,
    estimator: str = "jansen",
    sampling: str = "sobol",
    pass_as: str = "frame",
    n_boot: int = 200,
    level: float = 0.95,
    seed: Optional[int] = None,
    A: Any = None,
    B: Any = None,
) -> SobolResult:
    """First-order and total Sobol' indices of a model's inputs.

    Tells which inputs (parameters, calibrated values, assumptions) the
    output of a model is sensitive to over the whole range of the
    inputs, interactions included. Use it on a structural or simulation
    model before estimation to see which parameters the moments can
    identify, after estimation to report which assumptions drive a
    counterfactual, or on a fitted flexible model (``sp.gp_regress``, a
    forest) to rank the regressors.

    Parameters
    ----------
    func : callable
        The model. By default it receives a DataFrame with one row per
        run and one column per input and returns one number per run.
    factors : int, list of str, or dict
        The inputs: their number (uniform on [0, 1]), their names, or
        ``{name: (lower, upper)}`` for uniform inputs and ``{name:
        scipy.stats frozen distribution}`` for others. Inputs are
        independent.
    n : int, default 1024
        Base sample size. The model is evaluated ``n (p + 2)`` times.
        With ``sampling='sobol'`` it is rounded up to a power of two.
    estimator : {'jansen', 'saltelli'}, default 'jansen'
        Estimator of the first-order index: Jansen (1999), or Saltelli
        et al. (2010); see Notes for which to prefer. The total index is
        Jansen's in both cases.
    sampling : {'sobol', 'random'}, default 'sobol'
        Scrambled Sobol' points (smaller error for a given ``n``) or
        independent draws.
    pass_as : {'frame', 'array', 'rows'}, default 'frame'
        What ``func`` receives: a DataFrame, a 2-D array, or one 1-D
        array per call (slow; for a function of a single run).
    n_boot : int, default 200
        Bootstrap replications for the intervals; 0 for none.
    level : float, default 0.95
    seed : int, optional
    A, B : DataFrame or array, optional
        Two independent samples of the inputs to use instead of drawing
        them, in the units of the inputs (then ``n``, ``sampling`` and
        any distributions in ``factors`` are not used).

    Returns
    -------
    SobolResult
        ``indices`` (``first``, ``total`` and limits per input),
        ``variance``, ``summary()``, ``plot()``.

    Notes
    -----
    With ``A`` and ``B`` two samples and ``AB_i`` the sample ``A`` with
    column ``i`` taken from ``B``,

    ``total_i = mean (f(A) - f(AB_i))^2 / (2 V)`` and
    ``first_i = 1 - mean (f(B) - f(AB_i))^2 / (2 V)`` (Jansen), or
    ``first_i = mean f(B) (f(AB_i) - f(A)) / V`` (Saltelli et al.),

    with ``V`` the sample variance of ``f(A)``. ``sensitivity::
    soboljansen`` in R computes the same quantities with the sums divided
    by ``2 n - 1`` instead of ``2 n``; after that rescaling the two agree
    to rounding error.

    Neither first-order estimator dominates. Jansen's is the more
    accurate for an input that explains most of the variance, Saltelli's
    for inputs that explain little: on the eight-input borehole function
    with ``n = 512`` Sobol' points the root mean squared errors are 0.005
    against 0.012 for the dominant input (index 0.83) and about 0.008
    against 0.0001 for the three inert ones. Read small Jansen
    first-order indices with their intervals, or switch estimator.

    Estimates can fall slightly below zero or sum to more than one by
    sampling error; that is the Monte Carlo noise to read the indices
    against. The intervals are percentile bootstrap intervals over the
    base sample; with Sobol' points they are conservative, since the
    bootstrap treats the points as independent draws.

    The decomposition assumes independent inputs. With dependent inputs
    the indices no longer add up to shares of the variance.

    Examples
    --------
    ``y = x1 + 2 x2 + x1 x3`` with independent uniform inputs:

    >>> import statspai as sp
    >>> f = lambda d: d["x1"] + 2 * d["x2"] + d["x1"] * d["x3"]
    >>> res = sp.sobol_indices(f, ["x1", "x2", "x3"], n=2048, seed=1, n_boot=0)
    >>> bool(res.indices.loc["x2", "first"] > res.indices.loc["x1", "first"])
    True
    >>> bool(res.indices.loc["x3", "total"] > res.indices.loc["x3", "first"])
    True

    References
    ----------
    sobol2001global; jansen1999analysis; saltelli2010variance;
    harenberg2019uncertainty
    """
    est = str(estimator).lower()
    if est not in ("jansen", "saltelli"):
        raise MethodIncompatibility(
            f"estimator must be 'jansen' or 'saltelli'; got {estimator!r}."
        )
    if pass_as not in ("frame", "array", "rows"):
        raise MethodIncompatibility(
            f"pass_as must be 'frame', 'array' or 'rows'; got {pass_as!r}."
        )
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    names, spec = _inputs(factors)
    p = len(names)
    notes: List[str] = []
    if (A is None) != (B is None):
        raise MethodIncompatibility("Give both A and B, or neither.")
    if A is not None:
        XA = np.asarray(
            (
                A[names].to_numpy(dtype=float)
                if isinstance(A, pd.DataFrame) and set(names) <= set(A.columns)
                else (A.to_numpy(dtype=float) if isinstance(A, pd.DataFrame) else A)
            ),
            dtype=float,
        )
        XB = np.asarray(
            (
                B[names].to_numpy(dtype=float)
                if isinstance(B, pd.DataFrame) and set(names) <= set(B.columns)
                else (B.to_numpy(dtype=float) if isinstance(B, pd.DataFrame) else B)
            ),
            dtype=float,
        )
        if XA.ndim != 2 or XA.shape != XB.shape or XA.shape[1] != p:
            raise MethodIncompatibility(
                f"A and B must have the same shape with {p} columns."
            )
        how = "supplied"
    else:
        n = int(n)
        if n < 16:
            raise DataInsufficient(f"n={n} is too small; use at least 16.")
        how = str(sampling).lower()
        if how == "sobol":
            from scipy.stats import qmc

            m = int(np.ceil(np.log2(n)))
            U = qmc.Sobol(2 * p, scramble=True, seed=seed).random_base2(m)
        elif how == "random":
            U = np.random.default_rng(seed).random((n, 2 * p))
        else:
            raise MethodIncompatibility(
                f"sampling must be 'sobol' or 'random'; got {sampling!r}."
            )
        XA, XB = _from_unit(U[:, :p], spec), _from_unit(U[:, p:], spec)
    n = XA.shape[0]
    yA = _evaluate(func, XA, names, pass_as)
    yB = _evaluate(func, XB, names, pass_as)
    yAB = np.empty((n, p))
    for j in range(p):
        X = XA.copy()
        X[:, j] = XB[:, j]
        yAB[:, j] = _evaluate(func, X, names, pass_as)
    if np.var(yA, ddof=1) <= 0:
        raise MethodIncompatibility(
            "The output does not vary over the inputs; there is no variance "
            "to decompose."
        )
    first, total, var = _sobol_estimates(yA, yB, yAB, est)
    tab = pd.DataFrame({"first": first, "total": total}, index=names)
    if n_boot and n_boot > 0:
        rng = np.random.default_rng(None if seed is None else seed + 1)
        bf = np.empty((int(n_boot), p))
        bt = np.empty((int(n_boot), p))
        for b in range(int(n_boot)):
            i = rng.integers(n, size=n)
            bf[b], bt[b], _ = _sobol_estimates(yA[i], yB[i], yAB[i], est)
        a = 0.5 * (1.0 - level)
        tab["first_lower"], tab["first_upper"] = np.quantile(bf, [a, 1 - a], axis=0)
        tab["total_lower"], tab["total_upper"] = np.quantile(bt, [a, 1 - a], axis=0)
        tab = tab[
            [
                "first",
                "first_lower",
                "first_upper",
                "total",
                "total_lower",
                "total_upper",
            ]
        ]
    if (tab["total"] < tab["first"] - 0.05).any():
        notes.append(
            "A total index below the first-order index: the sample is too "
            "small for these estimates; raise n."
        )
    return SobolResult(
        indices=tab,
        variance=var,
        mean=float(np.mean(np.r_[yA, yB])),
        n_evaluations=int(n * (p + 2)),
        model_info={
            "estimator": est,
            "sampling": how,
            "n": int(n),
            "level": float(level),
            "n_boot": int(n_boot or 0),
            "notes": notes,
        },
    )


@dataclass
class MorrisResult(ResultProtocolMixin):
    """Elementary-effects screening.

    Attributes
    ----------
    effects : DataFrame
        Indexed by input: ``mu`` (mean elementary effect), ``mu_star``
        (mean absolute effect, the importance measure), ``sigma``
        (standard deviation: non-linearity or interaction), and
        ``share`` (``mu_star`` as a share of the total).
    elementary_effects : DataFrame
        One row per trajectory, one column per input.
    design : DataFrame
        The runs, in the units of the inputs.
    n_evaluations : int
    model_info : dict

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.morris_screening(lambda X: 3 * X[:, 0] + 0 * X[:, 1], 2,
    ...                           r=4, pass_as="array", seed=0)
    >>> [round(float(v), 6) for v in res.effects["mu_star"]]
    [3.0, 0.0]
    """

    effects: pd.DataFrame
    elementary_effects: pd.DataFrame
    design: pd.DataFrame
    n_evaluations: int
    model_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        info = self.model_info
        lines = [
            "Morris elementary-effects screening",
            "=" * 60,
            f"Trajectories: {info['r']}    Model evaluations: {self.n_evaluations}",
            self.effects.to_string(float_format=lambda v: f"{v:.4g}"),
            "mu_star ranks the inputs; sigma large relative to mu_star "
            "signals non-linearity or interaction.",
        ]
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """``sigma`` against ``mu_star``, one labelled point per input."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(self.effects["mu_star"], self.effects["sigma"], **kwargs)
        for nm, row in self.effects.iterrows():
            ax.annotate(
                str(nm),
                (row["mu_star"], row["sigma"]),
                textcoords="offset points",
                xytext=(4, 4),
            )
        ax.set_xlabel("mu* (mean absolute elementary effect)")
        ax.set_ylabel("sigma")
        return ax


def morris_screening(
    func: Optional[Callable[..., Any]],
    factors: Any,
    r: int = 10,
    levels: int = 4,
    grid_jump: Optional[int] = None,
    pass_as: str = "frame",
    seed: Optional[int] = None,
    design: Any = None,
    y: Any = None,
) -> MorrisResult:
    """Screen the inputs of a model with one-factor-at-a-time trajectories.

    A cheap first pass over a model with many inputs: ``r (p + 1)`` runs
    are enough to sort the inputs into those that do nothing, those with
    a roughly linear effect, and those whose effect is non-linear or
    depends on other inputs. Sobol' indices (``sp.sobol_indices``) need
    hundreds of runs per input for the same question.

    Parameters
    ----------
    func : callable or None
        The model; see ``sp.sobol_indices`` for what it receives.
        ``None`` when ``design`` and ``y`` are given.
    factors : int, list of str, or dict
        The inputs: their number, their names, or ``{name: (lower,
        upper)}``. Effects are measured per full range of an input.
    r : int, default 10
        Number of trajectories. Each takes ``p + 1`` runs.
    levels : int, default 4
        Grid points per input.
    grid_jump : int, optional
        Grid steps moved at a time. Default ``levels // 2``; with an even
        number of levels every level is then visited equally often
        (Morris 1991). With an odd number the middle levels are visited
        more often.
    pass_as : {'frame', 'array', 'rows'}, default 'frame'
    seed : int, optional
    design : DataFrame or array, optional
        Runs made already, in the units of the inputs: ``p + 1``
        consecutive rows per trajectory, consecutive rows differing in
        exactly one input.
    y : array, optional
        The outputs of ``design``.

    Returns
    -------
    MorrisResult
        ``effects`` (``mu``, ``mu_star``, ``sigma``, ``share``),
        ``elementary_effects``, ``design``, ``summary()``, ``plot()``.

    Notes
    -----
    The elementary effect of input ``i`` on a trajectory is the change
    in output over the change in that input (on the unit scale) for the
    step that moves it. ``mu_star`` is the mean absolute effect
    (Campolongo, Cariboni and Saltelli 2007), ``sigma`` the standard
    deviation over trajectories.

    With few trajectories the ranking of unimportant inputs is noisy;
    what is reliable is the separation of the clearly important from the
    rest. Given the same design and outputs the statistics equal those
    of ``sensitivity::morris`` in R.

    Examples
    --------
    >>> import statspai as sp
    >>> f = lambda d: 5 * d["a"] + d["b"] ** 2 + 0 * d["c"]
    >>> res = sp.morris_screening(f, ["a", "b", "c"], r=8, seed=1)
    >>> res.effects["mu_star"].idxmax(), float(res.effects.loc["c", "mu_star"])
    ('a', 0.0)
    >>> res.n_evaluations
    32

    References
    ----------
    morris1991factorial; campolongo2007effective
    """
    names, spec = _inputs(factors)
    if any(hasattr(s, "ppf") for s in spec):
        raise MethodIncompatibility(
            "Morris screening moves inputs on a grid over a range; give "
            "(lower, upper) bounds, not distributions."
        )
    p = len(names)
    lo = np.array([s[0] for s in spec])
    hi = np.array([s[1] for s in spec])
    info: Dict[str, Any] = {"notes": []}
    if design is not None:
        if y is None:
            raise MethodIncompatibility("With design= give the outputs as y=.")
        X = np.asarray(
            (
                design[names].to_numpy(dtype=float)
                if isinstance(design, pd.DataFrame)
                and set(names) <= set(design.columns)
                else (
                    design.to_numpy(dtype=float)
                    if isinstance(design, pd.DataFrame)
                    else design
                )
            ),
            dtype=float,
        )
        yy = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[1] != p or X.shape[0] % (p + 1) != 0:
            raise MethodIncompatibility(
                f"design must have {p} columns and a multiple of {p + 1} rows."
            )
        if yy.shape != (X.shape[0],):
            raise MethodIncompatibility("y must have one value per run of design.")
        r = X.shape[0] // (p + 1)
        if r < 2:
            raise DataInsufficient(
                "One trajectory gives one elementary effect per input and no "
                "spread; at least two trajectories are needed."
            )
        info["design"] = "supplied"
    else:
        if func is None:
            raise MethodIncompatibility("Give func, or design and y.")
        r, levels = int(r), int(levels)
        if r < 2:
            raise DataInsufficient("At least two trajectories are needed.")
        if levels < 2:
            raise MethodIncompatibility("levels must be at least 2.")
        jump = max(levels // 2, 1) if grid_jump is None else int(grid_jump)
        if not 1 <= jump < levels:
            raise MethodIncompatibility(
                f"grid_jump must be between 1 and {levels - 1}."
            )
        rng = np.random.default_rng(seed)
        delta = jump / (levels - 1)
        U = np.empty((r * (p + 1), p))
        for t in range(r):
            base = rng.integers(0, levels - jump, size=p) / (levels - 1)
            up = rng.random(p) < 0.5
            # start from the low or the high end of each step
            x = np.where(up, base, base + delta)
            U[t * (p + 1)] = x
            for s, j in enumerate(rng.permutation(p), start=1):
                x = x.copy()
                x[j] += delta if up[j] else -delta
                U[t * (p + 1) + s] = x
        X = lo + U * (hi - lo)
        yy = _evaluate(func, X, names, pass_as)
        info.update(design="generated", levels=levels, grid_jump=jump)
    Uu = (X - lo) / (hi - lo)
    ee = np.full((r, p), np.nan)
    for t in range(r):
        blk = slice(t * (p + 1), (t + 1) * (p + 1))
        dx, dy = np.diff(Uu[blk], axis=0), np.diff(yy[blk])
        moved = np.abs(dx) > 1e-12
        if not np.all(moved.sum(axis=1) == 1) or not np.all(moved.sum(axis=0) == 1):
            raise MethodIncompatibility(
                f"Trajectory {t + 1} is not one-factor-at-a-time: each step "
                "must move exactly one input and each input must move once."
            )
        step, j = np.nonzero(moved)
        ee[t, j] = dy[step] / dx[step, j]
    eff = pd.DataFrame(
        {
            "mu": ee.mean(axis=0),
            "mu_star": np.abs(ee).mean(axis=0),
            "sigma": ee.std(axis=0, ddof=1),
        },
        index=names,
    )
    tot = float(eff["mu_star"].sum())
    eff["share"] = eff["mu_star"] / tot if tot > 0 else 0.0
    info["r"] = int(r)
    return MorrisResult(
        effects=eff,
        elementary_effects=pd.DataFrame(ee, columns=names),
        design=pd.DataFrame(X, columns=names),
        n_evaluations=int(X.shape[0]),
        model_info=info,
    )
