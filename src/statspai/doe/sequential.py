"""Sequential designs driven by a Gaussian process: ``sp.sequential_design``.

When each run is expensive (a structural model solved at one parameter
vector, a long simulation), the next run should be chosen in the light of
those already made. A Gaussian process fitted to the runs so far predicts
the response everywhere with an uncertainty; the next run goes where the
expected improvement over the best value is largest (to optimise) or where
the prediction is least certain (to learn the whole surface).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import ConvergenceWarning, DataInsufficient, MethodIncompatibility
from ._common import pair_sqdist, resolve_factors
from .sensitivity import _evaluate

_GOALS = ("minimize", "maximize", "emulate")


@dataclass
class SequentialDesignResult(ResultProtocolMixin):
    """Runs chosen one at a time with a Gaussian process.

    Attributes
    ----------
    design : DataFrame
        All runs: the factors, the response ``y`` and ``stage``
        (``'initial'`` or ``'sequential'``).
    best : dict
        The best run (for ``goal='minimize'`` / ``'maximize'``).
    trace : DataFrame
        One row per added run: the response, the value of the criterion
        that selected it and the best response so far.
    fit : GPResult
        The Gaussian process fitted to all runs, on factors scaled to
        [0, 1]. Use ``result.predict(newdata)`` for predictions in the
        units of the factors.
    goal : str
    model_info : dict

    Examples
    --------
    >>> import statspai as sp
    >>> f = lambda d: (d["x"] - 0.3) ** 2
    >>> res = sp.sequential_design(f, {"x": (0, 1)}, n_new=6, seed=1)
    >>> abs(res.best["x"] - 0.3) < 0.05
    True
    """

    design: pd.DataFrame
    best: Dict[str, float]
    trace: pd.DataFrame
    fit: Any
    goal: str
    model_info: Dict[str, Any] = field(default_factory=dict)

    def predict(self, newdata: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Surrogate prediction (``mean``, ``sd``, ``lower``, ``upper``) at
        new points given in the units of the factors."""
        info = self.model_info
        names = list(info["surrogate_inputs"].values())
        miss = [nm for nm in names if nm not in newdata.columns]
        if miss:
            raise MethodIncompatibility(f"newdata lacks {', '.join(miss)}.")
        lo, hi = np.asarray(info["lower"]), np.asarray(info["upper"])
        U = (newdata[names].to_numpy(dtype=float) - lo) / (hi - lo)
        frame = pd.DataFrame(
            U, columns=list(info["surrogate_inputs"]), index=newdata.index
        )
        return self.fit.predict(frame, **kwargs)

    def summary(self) -> str:
        info = self.model_info
        n0 = int((self.design["stage"] == "initial").sum())
        lines = [
            f"Sequential design ({self.goal})",
            "=" * 50,
            f"Initial runs: {n0}    Added: {self.design.shape[0] - n0}",
        ]
        if self.goal != "emulate":
            lines.append(
                "Best run: " + ", ".join(f"{k} = {v:.6g}" for k, v in self.best.items())
            )
            last = float(self.trace["criterion"].iloc[-1]) if len(self.trace) else 0.0
            lines.append(
                f"Expected improvement at the last added run: {last:.3g} "
                "(small relative to the spread of y means little is left to gain)"
            )
        else:
            last = float(self.trace["criterion"].iloc[-1]) if len(self.trace) else 0.0
            lines.append(f"Largest predictive sd before the last run: {last:.4g}")
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Best response so far against the number of runs."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 3.8))
        y = self.design["y"].to_numpy()
        run = np.arange(1, y.size + 1)
        if self.goal == "maximize":
            ax.plot(run, np.maximum.accumulate(y), **kwargs)
        elif self.goal == "minimize":
            ax.plot(run, np.minimum.accumulate(y), **kwargs)
        ax.scatter(run, y, s=14, color="0.5")
        ax.axvline((self.design["stage"] == "initial").sum() + 0.5, ls=":", color="0.5")
        ax.set_xlabel("run")
        ax.set_ylabel("y")
        return ax


def _alc(gp: Any, cand: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Average reduction of the predictive variance over reference points
    when a candidate is added: ``mean_r cov(r, c)^2 / var(c)``, with the
    posterior covariance of the process at the current hyperparameters."""
    from scipy import linalg

    from ..mcmc.gp import _kernel

    st = gp._state
    kind, X, length, sf = gp.kernel, st["X"], st["length"], st["signal_var"]
    a = linalg.solve_triangular(
        st["chol"], _kernel(kind, X, cand, length, sf), lower=True
    )
    b = linalg.solve_triangular(
        st["chol"], _kernel(kind, X, ref, length, sf), lower=True
    )
    cov = _kernel(kind, ref, cand, length, sf) - b.T @ a
    var = np.maximum(sf - (a * a).sum(axis=0), 1e-12 * sf)
    return np.asarray((cov * cov).mean(axis=0) / var)


def sequential_design(
    func: Callable[..., Any],
    factors: Any,
    n_new: int = 20,
    goal: str = "minimize",
    n_init: Optional[int] = None,
    design: Optional[pd.DataFrame] = None,
    y: Any = None,
    kernel: str = "rbf",
    noisy: bool = False,
    n_candidates: Optional[int] = None,
    pass_as: str = "frame",
    seed: Optional[int] = None,
    criterion: str = "variance",
) -> SequentialDesignResult:
    """Choose the runs of an expensive function one at a time.

    Bayesian optimisation and active learning in one function: minimise
    (or maximise) a function that is costly to evaluate with as few
    evaluations as possible, or learn it everywhere to build a fast
    surrogate. Typical uses are the objective of a simulated-moments or
    indirect-inference estimator, a tuning parameter chosen by an
    expensive cross-validation, or a simulation model that is to be
    replaced by an emulator.

    Parameters
    ----------
    func : callable
        The function. By default it receives a DataFrame with one row
        per run and returns one number per run.
    factors : dict
        ``{name: (lower, upper)}``.
    n_new : int, default 20
        Runs to add after the initial design.
    goal : {'minimize', 'maximize', 'emulate'}, default 'minimize'
        ``'minimize'`` / ``'maximize'`` add the run with the largest
        expected improvement (Jones, Schonlau and Welch 1998);
        ``'emulate'`` the run that most reduces the uncertainty of the
        surrogate over the region (see ``criterion``).
    n_init : int, optional
        Size of the initial maximum projection design. Default
        ``max(5 p, 6)``.
    design, y : DataFrame and array, optional
        Runs already made and their responses, used instead of an
        initial design.
    kernel : {'rbf', 'matern52', 'matern32'}, default 'rbf'
    noisy : bool, default False
        ``False``: the function returns the same value when called twice
        at the same point, and the surrogate interpolates. ``True``: the
        output has noise (a simulation with fresh random numbers), a
        noise variance is estimated and the improvement is measured
        against the best *predicted* value at the runs.
    n_candidates : int, optional
        Candidate points scored per step. Default ``min(20000, 2000 p)``.
    pass_as : {'frame', 'array', 'rows'}, default 'frame'
        See ``sp.sobol_indices``.
    seed : int, optional
    criterion : {'variance', 'alc'}, default 'variance'
        For ``goal='emulate'``. ``'variance'`` adds the run where the
        predictive variance is largest; ``'alc'`` the run that lowers the
        predictive variance averaged over the region the most. Neither
        is better throughout: with 10 initial and 20 added runs the root
        mean squared prediction error on the two-input Branin function
        was 0.18 (variance) against 0.36 (alc), and on a three-input
        test function 0.19 against 0.05, the first figure driven by one
        poor run in four.

    Returns
    -------
    SequentialDesignResult
        ``design``, ``best``, ``trace``, ``fit``, ``predict()``,
        ``summary()``, ``plot()``.

    Notes
    -----
    Each step refits the Gaussian process (``sp.gp_regress``), scores a
    fresh quasi-random candidate set, plus points near the best runs
    when optimising, and evaluates the function at the winner. The
    candidates are finite, so the optimum is located up to their
    spacing; finish with a local optimiser started from ``best`` when
    more digits are needed.

    Expected improvement balances looking near good values with looking
    where little is known. It finds the neighbourhood of a global
    optimum in few runs; it is not a convergence proof. A Gaussian
    process is a poor surrogate for a function with jumps or with more
    than about ten inputs.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> f = lambda d: np.sin(3 * d["a"]) + (d["b"] - 0.5) ** 2
    >>> res = sp.sequential_design(f, {"a": (0, 3), "b": (0, 1)},
    ...                            n_new=12, seed=1)
    >>> bool(res.best["y"] < -0.98)
    True

    References
    ----------
    jones1998efficient; sacks1989design; joseph2025experimental
    """
    from ..mcmc.gp import gp_regress
    from .spacefill import _sobol, space_filling

    g = str(goal).lower()
    g = {
        "min": "minimize",
        "max": "maximize",
        "minimise": "minimize",
        "maximise": "maximize",
        "emulation": "emulate",
    }.get(g, g)
    if g not in _GOALS:
        raise MethodIncompatibility(
            f"goal must be one of {', '.join(_GOALS)}; got {goal!r}."
        )
    rule = str(criterion).lower()
    if rule not in ("alc", "variance"):
        raise MethodIncompatibility(
            f"criterion must be 'variance' or 'alc'; got {criterion!r}."
        )
    if not isinstance(factors, dict):
        raise MethodIncompatibility("factors is a {name: (lower, upper)} dict.")
    names, lo, hi = resolve_factors(factors)
    taken = [nm for nm in names if nm in ("y", "stage")]
    if taken:
        raise MethodIncompatibility(
            f"A factor cannot be called {taken[0]!r}: the result table uses "
            "the columns 'y' (the response) and 'stage'. Rename the factor."
        )
    p = len(names)
    n_new = int(n_new)
    if n_new < 0:
        raise MethodIncompatibility("n_new must be non-negative.")
    rng = np.random.default_rng(seed)
    if design is not None:
        if y is None:
            raise MethodIncompatibility("With design= give the responses as y=.")
        miss = [nm for nm in names if nm not in design.columns]
        if miss:
            raise MethodIncompatibility(f"design lacks the factors {', '.join(miss)}.")
        X = design[names].to_numpy(dtype=float)
        yy = np.asarray(y, dtype=float).reshape(-1)
        if yy.shape != (X.shape[0],) or not np.all(np.isfinite(yy)):
            raise MethodIncompatibility("y must hold one finite value per run.")
        if np.any(X < lo - 1e-9) or np.any(X > hi + 1e-9):
            raise MethodIncompatibility("design has runs outside the bounds.")
    else:
        n0 = max(5 * p, 6) if n_init is None else int(n_init)
        first = space_filling(
            n0,
            dict(zip(names, zip(lo, hi))),
            seed=int(rng.integers(2**31)),
            n_starts=1,
        )
        X = first.design.to_numpy(dtype=float)
        yy = _evaluate(func, X, names, pass_as)
    if X.shape[0] < 5:
        raise DataInsufficient("At least five initial runs are needed.")
    n_start = X.shape[0]
    cols = [f"f{j}" for j in range(p)]
    formula = "y ~ " + " + ".join(cols)
    N = int(n_candidates) if n_candidates else int(min(20000, 2000 * p))
    N = 1 << int(np.ceil(np.log2(max(N, 64))))
    sign = -1.0 if g == "maximize" else 1.0
    notes: List[str] = []
    trace: List[Dict[str, float]] = []

    def fit_gp(Xc: np.ndarray, yc: np.ndarray) -> Any:
        U = (Xc - lo) / (hi - lo)
        frame = pd.DataFrame(U, columns=cols)
        frame["y"] = yc
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return gp_regress(
                formula,
                frame,
                kernel=kernel,
                interpolate=not noisy,
                restarts=2,
                seed=int(rng.integers(2**31)),
            )

    for step in range(n_new):
        if np.ptp(yy) <= 0:
            raise MethodIncompatibility(
                "The function returned the same value at every run; there is "
                "nothing to model."
            )
        gp = fit_gp(X, yy)
        U = (X - lo) / (hi - lo)
        cand = _sobol(N, p, int(rng.integers(2**31)))
        if g != "emulate":
            top = U[np.argsort(sign * yy)[: min(3, len(yy))]]
            local = np.vstack(
                [
                    np.clip(t + s * rng.standard_normal((150, p)), 0.0, 1.0)
                    for t in top
                    for s in (0.05, 0.01)
                ]
            )
            cand = np.vstack([cand, local])
        cand = cand[pair_sqdist(cand, U).min(axis=1) > 1e-10]
        cframe = pd.DataFrame(cand, columns=cols)
        if g == "emulate" and rule == "variance":
            score = gp.predict(cframe)["sd"].to_numpy()
        elif g == "emulate":
            score = _alc(gp, cand, _sobol(512, p, int(rng.integers(2**31))))
        else:
            if noisy:
                at_runs = gp.predict(pd.DataFrame(U, columns=cols))["mean"].to_numpy()
                ref = float(at_runs.min() if g == "minimize" else at_runs.max())
            else:
                ref = float(yy.min() if g == "minimize" else yy.max())
            score = gp.expected_improvement(
                cframe, minimize=(g == "minimize"), best=ref
            ).to_numpy()
        j = int(np.argmax(score))
        xn = lo + cand[j] * (hi - lo)
        yn = float(_evaluate(func, xn[None, :], names, pass_as)[0])
        X = np.vstack([X, xn])
        yy = np.r_[yy, yn]
        best_now = float(yy.min() if g != "maximize" else yy.max())
        trace.append(
            {"run": X.shape[0], "y": yn, "criterion": float(score[j]), "best": best_now}
        )
    gp = fit_gp(X, yy) if np.ptp(yy) > 0 else None
    out = pd.DataFrame(X, columns=names)
    out["y"] = yy
    out["stage"] = ["initial"] * n_start + ["sequential"] * (X.shape[0] - n_start)
    if g == "emulate":
        best: Dict[str, float] = {}
    else:
        if noisy and gp is not None:
            U = (X - lo) / (hi - lo)
            m = gp.predict(pd.DataFrame(U, columns=cols))["mean"].to_numpy()
            ib = int(np.argmin(sign * m))
            notes.append(
                "Noisy function: the best run is the one with the best "
                "predicted value, and 'y_predicted' is that prediction."
            )
            best = {nm: float(v) for nm, v in zip(names, X[ib])}
            best["y"] = float(yy[ib])
            best["y_predicted"] = float(m[ib])
        else:
            ib = int(np.argmin(sign * yy))
            best = {nm: float(v) for nm, v in zip(names, X[ib])}
            best["y"] = float(yy[ib])
    return SequentialDesignResult(
        design=out,
        best=best,
        trace=pd.DataFrame(trace, columns=["run", "y", "criterion", "best"]),
        fit=gp,
        goal=g,
        model_info={
            "n_candidates": N,
            "kernel": kernel,
            "noisy": bool(noisy),
            "surrogate_inputs": dict(zip(cols, names)),
            "lower": [float(v) for v in lo],
            "upper": [float(v) for v in hi],
            "notes": notes,
        },
    )
