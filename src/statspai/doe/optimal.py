"""Model-based optimal designs: ``sp.doe_optimal``.

Given a model, where should the runs be made so that its parameters are
estimated as precisely as possible? With information matrix ``M`` of a
design, a D-optimal design maximises ``det M`` (smallest confidence
ellipsoid), an A-optimal design minimises ``trace M^-1`` (smallest average
variance of the estimates) and an I-optimal design minimises the average
prediction variance over the region.

Two kinds of answer. An *approximate* design is a set of support points
with weights, the limit as the number of runs grows; its optimality can be
certified by the equivalence theorem. An *exact* design is ``n`` runs
chosen from a candidate set.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from ._common import resolve_factors

_PARAM = re.compile(r"\{\s*([A-Za-z_]\w*)\s*\}")
_CRITERIA = ("D", "A", "I")
_FAMILIES = ("gaussian", "binomial", "poisson")


class _Model:
    """Rows of the (weighted) regressor matrix at any set of points."""

    def __init__(self, model: str, family: str, sample: pd.DataFrame) -> None:
        self.family = family
        text = model.split("~", 1)[1].strip() if "~" in model else model.strip()
        self.nonlinear = bool(_PARAM.search(text))
        if not self.nonlinear and family != "gaussian":
            raise MethodIncompatibility(
                "For a binomial or Poisson response the information depends on "
                "the coefficients: write the linear predictor with its "
                "parameters in braces, e.g. '{b0} + {b1} * x', and give "
                "params= or prior=."
            )
        if self.nonlinear:
            self.names: List[str] = []
            for nm in _PARAM.findall(text):
                if nm not in self.names:
                    self.names.append(nm)
            self._private = {n: f"_doe_parameter_{n}" for n in self.names}
            self._expr = _PARAM.sub(lambda m: self._private[m.group(1)], text)
            self.terms = [f"d/d{n}" for n in self.names]
        else:
            from patsy import PatsyError, dmatrix

            from ..core.utils import r_formula_idioms

            try:
                F = dmatrix(r_formula_idioms(text), sample, return_type="dataframe")
            except (PatsyError, NameError, KeyError, TypeError, ValueError) as exc:
                raise MethodIncompatibility(
                    f"Cannot build the model matrix of {text!r}: {exc}"
                ) from exc
            self._info = F.design_info
            self.terms = [str(c) for c in F.columns]
            self.names = []

    def _eta(self, big: pd.DataFrame) -> np.ndarray:
        from ..agent._translation._stata_expr import StataExprError, evaluate

        try:
            val = evaluate(self._expr, big, {})
        except StataExprError as exc:
            raise MethodIncompatibility(
                f"Cannot evaluate the model: {exc}. Operators + - * / ^ and "
                "the functions exp, ln, sqrt, abs ... are read; factors are "
                "named as in factors=, parameters are in braces."
            ) from exc
        return np.asarray(val, dtype=float)

    def rows(
        self, data: pd.DataFrame, thetas: Optional[np.ndarray], weighted: bool = True
    ) -> List[np.ndarray]:
        """One regressor matrix per parameter vector.

        For a binomial or Poisson response the rows are scaled by the
        square root of the variance function, so that ``G'G`` is the
        information matrix; ``weighted=False`` gives the plain derivatives
        of the linear predictor.

        The parameters enter the expression as columns, so every
        parameter vector is evaluated in the same pass.
        """
        if not self.nonlinear:
            from patsy import build_design_matrices

            return [
                np.asarray(build_design_matrices([self._info], data)[0], dtype=float)
            ]
        assert thetas is not None
        S, m = thetas.shape[0], data.shape[0]
        big = pd.DataFrame({c: np.tile(data[c].to_numpy(), S) for c in data.columns})
        base = {}
        for j, nm in enumerate(self.names):
            base[nm] = np.repeat(thetas[:, j].astype(float), m)
            big[self._private[nm]] = base[nm]
        cols = []
        for nm in self.names:
            h = np.finfo(float).eps ** (1.0 / 3.0) * np.maximum(np.abs(base[nm]), 1.0)
            big[self._private[nm]] = base[nm] + h
            up = self._eta(big)
            big[self._private[nm]] = base[nm] - h
            down = self._eta(big)
            big[self._private[nm]] = base[nm]
            cols.append((up - down) / (2.0 * h))
        G = np.column_stack(cols)
        if self.family != "gaussian" and weighted:
            eta = self._eta(big)
            if self.family == "binomial":
                mu = 1.0 / (1.0 + np.exp(-eta))
                wt = mu * (1.0 - mu)
            else:
                wt = np.exp(eta)
            G = G * np.sqrt(wt)[:, None]
        if not np.all(np.isfinite(G)):
            raise NumericalInstability(
                "The model or its derivatives are not finite somewhere on the "
                "candidate set at the given parameter values."
            )
        return [G[i * m : (i + 1) * m] for i in range(S)]


def _sensitivity(
    Gs: Sequence[np.ndarray],
    pis: np.ndarray,
    w: np.ndarray,
    crit: str,
    Bs: Sequence[np.ndarray],
) -> Tuple[np.ndarray, float, float]:
    """Directional derivative at every candidate, its design average, and the
    criterion (log det for D, trace for A and I), averaged over parameters."""
    G = np.asarray(Gs)  # parameter vectors x candidates x terms
    k = G.shape[2]
    M = np.einsum("snk,n,snl->skl", G, w, G)
    ridge = 1e-12 * np.maximum(np.trace(M, axis1=1, axis2=2), 1e-300) / k
    Minv = np.linalg.inv(M + ridge[:, None, None] * np.eye(k))
    GM = G @ Minv
    if crit == "D":
        phi = np.einsum("s,snk,snk->n", pis, GM, G)
        sign, logdet = np.linalg.slogdet(M)
        value = float(pis @ np.where(sign > 0, logdet, -np.inf))
    elif crit == "A":
        phi = np.einsum("s,snk,snk->n", pis, GM, GM)
        value = float(pis @ np.trace(Minv, axis1=1, axis2=2))
    else:
        B = np.asarray(Bs)
        phi = np.einsum("s,snk,snk->n", pis, GM @ B, GM)
        value = float(pis @ np.einsum("skl,slk->s", B, Minv))
    return phi, float(w @ phi), value


def _multiplicative(
    Gs: Sequence[np.ndarray],
    pis: np.ndarray,
    crit: str,
    Bs: Sequence[np.ndarray],
    max_iter: int,
    target: float,
) -> Tuple[np.ndarray, float, int]:
    """Weights of the optimal approximate design on a finite candidate set."""
    N = Gs[0].shape[0]
    w = np.full(N, 1.0 / N)
    bound = 0.0
    it = 0
    for it in range(1, max_iter + 1):
        phi, avg, _ = _sensitivity(Gs, pis, w, crit, Bs)
        top = float(phi.max())
        bound = avg / top if crit == "D" else 2.0 - top / avg
        if bound >= target:
            break
        w = w * (phi / avg if crit == "D" else np.sqrt(np.maximum(phi, 0.0)))
        w /= w.sum()
    return w, float(bound), it


def _merge(U: np.ndarray, w: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Collapse neighbouring support points into their weighted mean."""
    keep = np.flatnonzero(w > 1e-4 * w.max())
    order = keep[np.argsort(-w[keep])]
    centres: List[np.ndarray] = []
    members: List[List[int]] = []
    for i in order:
        for c, m in zip(centres, members):
            if np.max(np.abs(U[i] - c)) <= tol:
                m.append(int(i))
                break
        else:
            centres.append(U[i].copy())
            members.append([int(i)])
    pts = np.array([np.average(U[m], axis=0, weights=w[m]) for m in members])
    wts = np.array([w[m].sum() for m in members])
    # an absolute floor, relaxed when even the largest weight is small
    # (2^p corners with weight 2^-p each)
    big = wts > min(1e-3, 1e-2 * wts.max())
    return pts[big], wts[big] / wts[big].sum()


def _grid(p: int, n_per: int) -> np.ndarray:
    axes = [np.linspace(0.0, 1.0, n_per)] * p
    return np.array(list(itertools.product(*axes)))


def _exchange(
    Gs: Sequence[np.ndarray],
    pis: np.ndarray,
    n: int,
    crit: str,
    Bs: Sequence[np.ndarray],
    rng: np.random.Generator,
    n_starts: int,
    kicks: int = 8,
) -> Tuple[np.ndarray, float]:
    """Exact design: replace one run at a time by the best candidate."""
    N, k = Gs[0].shape
    GG, BB = np.asarray(Gs), np.asarray(Bs)
    ridge = 1e-9 * np.maximum(np.einsum("snk,snk->s", GG, GG) / N, 1e-300)

    def value(idx: np.ndarray) -> float:
        w = np.bincount(idx, minlength=N) / float(len(idx))
        return _sensitivity(Gs, pis, w, crit, Bs)[2]

    def climb(idx: np.ndarray) -> np.ndarray:
        for _pass in range(100):
            changed = False
            for pos in range(n):
                rest = np.delete(idx, pos)
                Gr = GG[:, rest, :]
                M = np.einsum("snk,snl->skl", Gr, Gr)
                Minv = np.linalg.inv(M + ridge[:, None, None] * np.eye(k))
                GM = GG @ Minv
                q = np.einsum("snk,snk->sn", GM, GG)
                if crit == "D":
                    gain = pis @ np.log1p(np.maximum(q, 0.0))
                elif crit == "A":
                    gain = pis @ (np.einsum("snk,snk->sn", GM, GM) / (1.0 + q))
                else:
                    gain = pis @ (np.einsum("snk,snk->sn", GM @ BB, GM) / (1.0 + q))
                j = int(np.argmax(gain))
                if gain[j] > gain[idx[pos]] * (1.0 + 1e-9) + 1e-12:
                    idx[pos] = j
                    changed = True
            if not changed:
                break
        return idx

    def score_of(idx: np.ndarray) -> float:
        val = value(idx)
        sc = val if crit == "D" else -val
        return sc if np.isfinite(sc) else -np.inf

    best_idx, best_val = None, -np.inf
    n_kick = max(1, n // 5)
    for _ in range(max(int(n_starts), 1)):
        idx = climb(rng.choice(N, size=n, replace=N < n))
        score = score_of(idx)
        # a local optimum of one-run exchanges: replace a few runs at
        # random and climb again, keeping the better design
        for _kick in range(kicks):
            trial = idx.copy()
            trial[rng.choice(n, size=n_kick, replace=False)] = rng.choice(
                N, size=n_kick
            )
            trial = climb(trial)
            sc = score_of(trial)
            if sc > score:
                idx, score = trial, sc
        if score > best_val:
            best_idx, best_val = idx.copy(), score
    if best_idx is None:
        raise NumericalInstability(
            "No design with a non-singular information matrix was found. "
            "Check that the model has no more parameters than runs and that "
            "the candidates support every term."
        )
    return np.sort(best_idx), (best_val if crit == "D" else -best_val)


@dataclass
class ModelDesignResult(ResultProtocolMixin):
    """An optimal design for a model.

    Attributes
    ----------
    design : DataFrame
        The support points with a ``weight`` column (approximate design)
        or an ``n`` column of replicate counts (exact design).
    criterion : str
    criterion_value : float
        ``det(M)^(1/k)`` for D (larger is better), ``trace(M^-1) / k``
        for A and the average prediction variance for I (smaller is
        better), with ``M`` the information matrix per run.
    efficiency : float
        For an approximate design, a lower bound on its efficiency from
        the equivalence theorem (1 certifies optimality on the candidate
        set). For an exact design, its efficiency relative to the
        optimal approximate design.
    information : DataFrame
        Information matrix per run (at the first parameter vector).
    terms : list of str
    model_info : dict

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.doe_optimal("x + I(x**2)", {"x": (0, 2)})
    >>> d.design.round(3).values.tolist()
    [[0.0, 0.333], [1.0, 0.333], [2.0, 0.333]]
    """

    design: pd.DataFrame
    criterion: str
    criterion_value: float
    efficiency: float
    information: pd.DataFrame
    terms: List[str]
    model_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_runs(self) -> Optional[int]:
        return int(self.design["n"].sum()) if "n" in self.design.columns else None

    def to_frame(self) -> pd.DataFrame:
        """One row per run for an exact design, the support points otherwise."""
        if "n" not in self.design.columns:
            return self.design.copy()
        rep = self.design.loc[self.design.index.repeat(self.design["n"])]
        return rep.drop(columns="n").reset_index(drop=True)

    def summary(self) -> str:
        info = self.model_info
        exact = "n" in self.design.columns
        lines = [
            f"{self.criterion}-optimal design"
            + (f", {self.n_runs} runs" if exact else " (approximate)"),
            "=" * 56,
            f"Model terms ({len(self.terms)}): " + ", ".join(self.terms),
            self.design.to_string(float_format=lambda v: f"{v:.4g}"),
            f"Criterion value: {self.criterion_value:.6g}",
        ]
        if exact:
            lines.append(
                f"Efficiency relative to the optimal approximate design: "
                f"{self.efficiency:.4f}"
            )
        else:
            lines.append(
                f"Efficiency lower bound (equivalence theorem): {self.efficiency:.4f}"
            )
        if info.get("parameters") == "prior":
            lines.append(f"Averaged over {info['n_prior']} draws from the prior.")
        elif info.get("parameters") == "local":
            lines.append("Locally optimal at the given parameter values.")
        for note in info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def doe_optimal(
    model: str,
    factors: Optional[Mapping[str, Tuple[float, float]]] = None,
    n: Optional[int] = None,
    criterion: str = "D",
    candidates: Optional[pd.DataFrame] = None,
    params: Optional[Mapping[str, float]] = None,
    prior: Optional[Mapping[str, Any]] = None,
    family: str = "gaussian",
    grid: Optional[int] = None,
    n_prior: int = 32,
    n_starts: Optional[int] = None,
    max_iter: int = 3000,
    seed: Optional[int] = None,
) -> ModelDesignResult:
    """The design that estimates a given model most precisely.

    For choosing the runs of an experiment when the model to be fitted
    is known in advance: which prices to test for a demand curve, which
    doses for a dose-response curve, which attribute combinations for a
    regression with interactions.

    Parameters
    ----------
    model : str
        A linear model as a formula in the factors, ``"x1 + x2 + x1:x2 +
        I(x1**2)"``; or a nonlinear model with its parameters in braces,
        ``"{a} + exp(-{b} * x)"`` (the syntax of ``sp.nls``).
    factors : dict, optional
        ``{name: (lower, upper)}``. The candidates are a grid over this
        box; the support points found on it are then moved continuously.
    n : int, optional
        Number of runs of an exact design. Omitted, the approximate
        design: support points and weights.
    criterion : {'D', 'A', 'I'}, default 'D'
        For ``'I'`` the prediction variance is averaged uniformly over
        the box of ``factors``, or over the rows of ``candidates``. With
        a binomial or Poisson response it is the variance of the
        estimated linear predictor.
    candidates : DataFrame, optional
        The admissible runs, one column per factor, instead of
        ``factors``. Use it for an irregular region or for qualitative
        factors (which a formula handles with ``C(name)``).
    params : dict, optional
        Values of the parameters of a nonlinear model at which the
        design is optimal (a *locally* optimal design).
    prior : dict, optional
        ``{parameter: (lower, upper)}`` for a uniform prior, or a frozen
        ``scipy.stats`` distribution. The criterion is averaged over the
        prior (a pseudo-Bayesian design), which protects against a wrong
        guess of the parameters. Parameters left out are fixed at their
        value in ``params``.
    family : {'gaussian', 'binomial', 'poisson'}, default 'gaussian'
        For a binomial (logit) or Poisson (log) response ``model`` is
        the linear predictor, written with braces.
    grid : int, optional
        Grid points per factor. Default 201, 41, 15 for one, two, three
        factors; 4,096 quasi-random points plus the corners for four to
        ten factors. Beyond ten, pass ``candidates=``.
    n_prior : int, default 32
        Quasi-random draws from the prior.
    n_starts : int, optional
        Random starts of the exchange algorithm for an exact design.
        Default: between 2 and 20, fewer for large problems.
    max_iter : int, default 3000
        Iterations of the weight algorithm.
    seed : int, optional

    Returns
    -------
    ModelDesignResult
        ``design``, ``criterion_value``, ``efficiency``,
        ``information``, ``summary()``, ``to_frame()``.

    Notes
    -----
    Approximate designs are computed with the multiplicative algorithm
    on the candidate set (monotone for D; Yu 2010), then the support is
    merged and moved continuously off the grid. ``efficiency`` is
    the equivalence-theorem bound ``k / max_x d(x)`` for D, with ``d``
    the standardised prediction variance and ``k`` the number of
    parameters (Kiefer and Wolfowitz 1960), and ``2 - max_x phi(x) /
    trace`` for A and I. It is a statement about the candidate set.

    Exact designs are found by an exchange algorithm (one run at a time
    is replaced by the best candidate; at a local optimum a few runs are
    replaced at random and the search resumes) from several random
    starts. That gives a good design without a guarantee: the problem has
    many local optima of nearly equal value. ``efficiency`` compares the
    design with the approximate optimum, an upper bound on what ``n``
    runs can achieve. On the examples of ``AlgDesign::optFederov`` in R
    the criterion values coincide.

    The information matrix of a nonlinear model depends on the unknown
    parameters. A locally optimal design is only as good as the guess;
    it typically has as many support points as parameters and leaves no
    way to check the model. Averaging over a prior spreads the runs.

    An optimal design is optimal for the model given. It does not guard
    against that model being wrong: with real doubt about the functional
    form, add space-filling runs (``sp.design_augment``).

    Examples
    --------
    A quadratic in one factor: a third of the runs at each end and at
    the centre.

    >>> import statspai as sp
    >>> d = sp.doe_optimal("x + I(x**2)", {"x": (0, 2)})
    >>> d.design.round(3).values.tolist()
    [[0.0, 0.333], [1.0, 0.333], [2.0, 0.333]]

    Twelve runs for a two-factor model with interaction:

    >>> d = sp.doe_optimal("a * b", {"a": (0, 1), "b": (0, 1)}, n=12, seed=1)
    >>> d.design["n"].tolist()
    [3, 3, 3, 3]

    Exponential decay with a guess of the rate: observe at 0 and at
    ``1 / rate``.

    >>> d = sp.doe_optimal("{a} * exp(-{b} * t)", {"t": (0, 10)},
    ...                    params={"a": 1, "b": 0.5})
    >>> d.design["t"].round(2).tolist()
    [0.0, 2.0]

    References
    ----------
    kiefer1960equivalence; fedorov1972theory; yu2010monotonic;
    chaloner1995bayesian
    """
    crit = str(criterion).upper()
    if crit not in _CRITERIA:
        raise MethodIncompatibility(
            f"criterion must be one of {', '.join(_CRITERIA)}; got {criterion!r}."
        )
    fam = str(family).lower()
    if fam not in _FAMILIES:
        raise MethodIncompatibility(
            f"family must be one of {', '.join(_FAMILIES)}; got {family!r}."
        )
    if (factors is None) == (candidates is None):
        raise MethodIncompatibility("Give either factors= or candidates=.")
    rng = np.random.default_rng(seed)
    notes: List[str] = []
    if candidates is not None:
        if not isinstance(candidates, pd.DataFrame) or candidates.shape[0] < 1:
            raise MethodIncompatibility("candidates must be a non-empty DataFrame.")
        cand_df = candidates.drop_duplicates().reset_index(drop=True)
        names = [str(c) for c in cand_df.columns]
        U = None
        step = 0.0
    else:
        names, lo, hi = resolve_factors(factors)
        p = len(names)
        if grid is not None and int(grid) < 2:
            raise MethodIncompatibility("grid must be at least 2.")
        if p <= 3 or grid is not None:
            per = int(grid) if grid is not None else {1: 201, 2: 41, 3: 15}[p]
            if float(per) ** p > 200_000:
                raise MethodIncompatibility(
                    f"A grid of {per} points on each of {p} factors has "
                    f"{float(per) ** p:.3g} candidates. Lower grid, or pass "
                    "candidates=."
                )
            U = _grid(p, per)
            step = 1.0 / (per - 1)
        else:
            from scipy.stats import qmc

            if p > 10:
                raise MethodIncompatibility(
                    f"{p} factors: pass candidates= (for instance the runs "
                    "of a sp.factorial_design or a sp.space_filling design). "
                    "An approximate design over a box in that many "
                    "dimensions has thousands of support points."
                )
            S = qmc.Sobol(p, scramble=True, seed=int(rng.integers(2**31))).random(4096)
            U = np.vstack([S, _grid(p, 2)])
            step = 0.5 * 4096 ** (-1.0 / p)
        cand_df = pd.DataFrame(lo + U * (hi - lo), columns=names)
    mdl = _Model(model, fam, cand_df)
    if mdl.nonlinear:
        odd = [nm for nm in names if not nm.isidentifier() or nm.startswith("_")]
        if odd:
            raise MethodIncompatibility(
                "In a model with parameters in braces the factors are read by "
                "name inside an expression, so their names must be plain "
                f"identifiers that do not start with an underscore; got "
                f"{', '.join(repr(o) for o in odd)}. Rename them."
            )
    k = len(mdl.terms)
    # parameter vectors
    thetas: Optional[np.ndarray]
    if mdl.nonlinear:
        fixed = {str(a): float(b) for a, b in (params or {}).items()}
        pr = {str(a): b for a, b in (prior or {}).items()}
        unknown = [nm for nm in list(fixed) + list(pr) if nm not in mdl.names]
        if unknown:
            raise MethodIncompatibility(
                f"Not parameters of the model: {', '.join(unknown)}."
            )
        missing = [nm for nm in mdl.names if nm not in fixed and nm not in pr]
        if missing:
            raise MethodIncompatibility(
                f"No value for {', '.join(missing)}: the information of a "
                "nonlinear model depends on its parameters. Give params= "
                "(a guess) or prior= (a range)."
            )
        if pr:
            from scipy.stats import qmc

            S = 1 << int(np.ceil(np.log2(max(int(n_prior), 2))))
            u = qmc.Sobol(
                len(pr), scramble=True, seed=int(rng.integers(2**31))
            ).random(S)
            draws = np.empty((S, len(mdl.names)))
            col = 0
            for j, nm in enumerate(mdl.names):
                if nm in pr:
                    spec = pr[nm]
                    if hasattr(spec, "ppf"):
                        draws[:, j] = spec.ppf(np.clip(u[:, col], 1e-9, 1 - 1e-9))
                    else:
                        try:
                            a, b = float(spec[0]), float(spec[1])
                        except (TypeError, IndexError, ValueError) as exc:
                            raise MethodIncompatibility(
                                f"prior[{nm!r}] is (lower, upper) or a frozen "
                                "scipy.stats distribution."
                            ) from exc
                        draws[:, j] = a + u[:, col] * (b - a)
                    col += 1
                else:
                    draws[:, j] = fixed[nm]
            thetas = draws
            mode = "prior"
        else:
            thetas = np.array([[fixed[nm] for nm in mdl.names]])
            mode = "local"
    else:
        if params or prior:
            raise MethodIncompatibility(
                "params= and prior= are for models with parameters in braces; "
                "the design of a linear model does not depend on its coefficients."
            )
        thetas = None
        mode = "linear"
    n_theta = 1 if thetas is None else int(thetas.shape[0])
    pis = np.full(n_theta, 1.0 / n_theta)

    Gs = mdl.rows(cand_df, thetas)
    if any(np.linalg.matrix_rank(G) < k for G in Gs):
        raise MethodIncompatibility(
            f"The {k} model terms are not linearly independent on the "
            "candidate set, so no design can estimate them all. Drop a term "
            "or widen the region."
        )
    # The I criterion averages the variance of the predicted (linear
    # predictor) surface over the region: uniformly over the box when
    # factors= is given, over the rows of candidates= otherwise. The
    # candidate set of a box over-represents its boundary, so the box is
    # sampled separately.
    if crit == "I" and U is not None:
        from scipy.stats import qmc

        pts = qmc.Sobol(len(names), scramble=True, seed=int(rng.integers(2**31)))
        region = pd.DataFrame(lo + pts.random(8192) * (hi - lo), columns=names)
    else:
        region = cand_df
    Bs_region = [F.T @ F / F.shape[0] for F in mdl.rows(region, thetas, weighted=False)]

    def approximate() -> Tuple[pd.DataFrame, np.ndarray, float, int, float]:
        w, bound, its = _multiplicative(
            Gs, pis, crit, Bs_region, max_iter, 1 - 1e-6 if U is None else 0.9995
        )
        if U is None:
            keep = w > min(1e-3, 1e-2 * w.max())
            support = cand_df.loc[keep].reset_index(drop=True)
            wts = w[keep] / w[keep].sum()
            val, bound = certify(support, wts)[:2]
            return support, wts, val, its, bound
        pts, wts = _merge(U, w, 1.5 * step)
        pts, wts = polish(pts, wts)
        for _ in range(8):
            support = pd.DataFrame(lo + pts * (hi - lo), columns=names)
            val, bound, worst = certify(support, wts)
            if bound >= 0.9995 or worst is None:
                break
            # the equivalence theorem fails most at this candidate: give it
            # a little weight and let the polish decide
            pts = np.vstack([pts, U[worst]])
            wts = np.r_[0.95 * wts, 0.05]
            pts, wts = polish(pts, wts)
        support = pd.DataFrame(lo + pts * (hi - lo), columns=names)
        val, bound, _ = certify(support, wts)
        # points added along the way can end next to an existing one or
        # with next to no weight: join neighbours, drop those, and polish
        # once more, unless that costs efficiency
        if pts.shape[0] > 1:
            p2, w2 = _merge(pts, wts, 0.75 * step)
            heavy = w2 >= min(0.005, 0.02 * w2.max())
            p2, w2 = p2[heavy], w2[heavy] / w2[heavy].sum()
            if p2.shape[0] < pts.shape[0]:
                p2, w2 = polish(p2, w2)
                s2 = pd.DataFrame(lo + p2 * (hi - lo), columns=names)
                v2, b2, _ = certify(s2, w2)
                if b2 >= bound - 2e-4:
                    support, wts, val, bound = s2, w2, v2, b2
        return support, wts, val, its, bound

    def certify(
        support: pd.DataFrame, wts: np.ndarray
    ) -> Tuple[float, float, Optional[int]]:
        """Criterion, equivalence-theorem bound, and the candidate where the
        bound is attained (None when that is a support point)."""
        both = pd.concat([cand_df, support], ignore_index=True)
        Gb = mdl.rows(both, thetas)
        wb = np.r_[np.zeros(cand_df.shape[0]), wts]
        phi, avg, val = _sensitivity(Gb, pis, wb, crit, Bs_region)
        top = float(phi.max())
        bound = avg / top if crit == "D" else 2.0 - top / avg
        arg = int(np.argmax(phi))
        return val, float(min(bound, 1.0)), arg if arg < cand_df.shape[0] else None

    def polish(pts: np.ndarray, wts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Move the support points off the grid and re-weight them."""
        from scipy import optimize

        m, p_ = pts.shape
        if m * (p_ + 1) > 60:
            # too many coordinates to move: keep the points, redo the weights
            note = "Too many support points to move off the grid; only weights."
            if note not in notes:
                notes.append(note)
            # merged points are weighted means of neighbours: put each back
            # on its nearest candidate
            from ._common import pair_sqdist

            assert U is not None
            near = np.unique(pair_sqdist(pts, U).argmin(axis=1))
            pts = U[near]
            frame = pd.DataFrame(lo + pts * (hi - lo), columns=names)
            w2, _, _ = _multiplicative(
                mdl.rows(frame, thetas), pis, crit, Bs_region, max_iter, 1 - 1e-7
            )
            keep = w2 > 1e-5
            return pts[keep], w2[keep] / w2[keep].sum()

        def unpack(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            a = z[m * p_ :]
            ww = np.exp(a - a.max())
            return z[: m * p_].reshape(m, p_), ww / ww.sum()

        def objective(z: np.ndarray) -> float:
            x, ww = unpack(z)
            frame = pd.DataFrame(lo + x * (hi - lo), columns=names)
            try:
                val = _sensitivity(mdl.rows(frame, thetas), pis, ww, crit, Bs_region)[2]
            except (NumericalInstability, np.linalg.LinAlgError):
                return 1e30
            out = -val if crit == "D" else val
            return float(out) if np.isfinite(out) else 1e30

        z0 = np.r_[pts.ravel(), np.log(np.maximum(wts, 1e-8))]
        f0 = objective(z0)
        bnds = [(0.0, 1.0)] * (m * p_) + [(-25.0, 5.0)] * m
        sol = optimize.minimize(
            objective,
            z0,
            method="L-BFGS-B",
            bounds=bnds,
            options={"maxiter": 200, "eps": 1e-7},
        )
        if not (np.isfinite(sol.fun) and sol.fun <= f0):
            return pts, wts
        x, ww = unpack(sol.x)
        # points that met during the polish are one point
        return _merge(x, ww, 1e-4)

    support, wts, val_approx, its, bound = approximate()
    info: Dict[str, Any] = {
        "parameters": mode,
        "n_prior": n_theta if mode == "prior" else 0,
        "family": fam,
        "candidates": int(cand_df.shape[0]),
        "iterations": int(its),
        "notes": notes,
    }

    def report(value: float) -> float:
        return (
            float(np.exp(value / k))
            if crit == "D"
            else float(value / k if crit == "A" else value)
        )

    if n is None:
        design = support.copy()
        design["weight"] = wts
        order = np.lexsort([design[c].to_numpy() for c in names[::-1]])
        design = design.iloc[order].reset_index(drop=True)
        frame_w, eff, value = support, bound, val_approx
        w_final = wts
        if bound < 0.999:
            notes.append(
                "The equivalence-theorem bound is below 0.999: raise max_iter "
                "or grid."
            )
    else:
        n = int(n)
        if n < k:
            raise DataInsufficient(
                f"{n} runs cannot estimate {k} parameters; n must be at least {k}."
            )
        # candidates: the first set plus the refined support of the optimum
        pool = pd.concat([cand_df, support], ignore_index=True).drop_duplicates()
        pool = pool.reset_index(drop=True)
        Gp = mdl.rows(pool, thetas)
        if n_starts is None:
            cost = float(n) * pool.shape[0] * k * k * n_theta
            starts = int(np.clip(2e9 / (40.0 * cost), 2, 20))
        else:
            starts = int(n_starts)
        info["n_starts"] = starts
        idx, value = _exchange(Gp, pis, n, crit, Bs_region, rng, starts)
        counts = pd.Series(idx).value_counts().sort_index()
        design = pool.loc[counts.index].reset_index(drop=True)
        design["n"] = counts.to_numpy()
        order = np.lexsort([design[c].to_numpy() for c in names[::-1]])
        design = design.iloc[order].reset_index(drop=True)
        frame_w = design[names]
        w_final = design["n"].to_numpy() / float(n)
        if crit == "D":
            eff = float(np.exp((value - val_approx) / k))
        else:
            eff = float(val_approx / value)
        eff = min(eff, 1.0)
    G0 = mdl.rows(frame_w.reset_index(drop=True), thetas)[0]
    M0 = G0.T @ (np.asarray(w_final)[:, None] * G0)
    return ModelDesignResult(
        design=design,
        criterion=crit,
        criterion_value=report(value),
        efficiency=float(eff),
        information=pd.DataFrame(M0, index=mdl.terms, columns=mdl.terms),
        terms=list(mdl.terms),
        model_info=info,
    )
