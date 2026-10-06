"""Designs for mixture experiments: ``sp.mixture_design``.

In a mixture experiment the factors are the shares of the components of a
blend (a portfolio, a budget, a diet, a time allocation), so they are
non-negative and sum to one and the experimental region is a simplex.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from ._common import DesignResult


def _simplex_sample(q: int, N: int, seed: Optional[int]) -> np.ndarray:
    """Quasi-random points uniform on the simplex (spacings of sorted uniforms)."""
    from scipy.stats import qmc

    N = 1 << int(np.ceil(np.log2(N)))
    u = np.sort(qmc.Sobol(q - 1, scramble=True, seed=seed).random(N), axis=1)
    edges = np.column_stack([np.zeros(N), u, np.ones(N)])
    return np.asarray(np.diff(edges, axis=1))


def mixture_design(
    components: Any,
    kind: str = "simplex_lattice",
    degree: int = 2,
    n: Optional[int] = None,
    lower: Optional[Sequence[float]] = None,
    constraint: Optional[Callable[..., Any]] = None,
    total: float = 1.0,
    seed: Optional[int] = None,
) -> DesignResult:
    """A design for an experiment whose factors are shares that sum to one.

    Parameters
    ----------
    components : int or list of str
        The number of components or their names.
    kind : {'simplex_lattice', 'simplex_centroid', 'space_filling'}
        ``'simplex_lattice'``
            Every blend whose shares are multiples of ``1 / degree``
            (Scheffe 1958); supports a polynomial of that degree.
        ``'simplex_centroid'``
            The pure components, the equal blends of every two, of
            every three, ... up to all (Scheffe 1963);
            ``2^q - 1`` runs.
        ``'space_filling'``
            ``n`` runs spread evenly over the simplex, or over the part
            of it that satisfies ``constraint``: the support points of
            the uniform distribution on that region. The choice when the
            region is irregular or no polynomial is assumed.
    degree : int, default 2
        For the lattice.
    n : int, optional
        Number of runs, for ``'space_filling'``.
    lower : list of float, optional
        Lower bounds on the shares. The design is built on the
        sub-simplex they leave (in pseudo-components) and mapped back.
    constraint : callable, optional
        For ``'space_filling'``: takes a DataFrame of candidate blends
        and returns one boolean per row.
    total : float, default 1.0
        What the components sum to (100 for percentages).
    seed : int, optional

    Returns
    -------
    DesignResult
        ``design`` holds the shares; rows sum to ``total``.

    Notes
    -----
    Space-filling criteria in ``criteria`` are omitted: they are defined
    on a cube, not a simplex.

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.mixture_design(3, degree=2)
    >>> d.design.shape
    (6, 3)
    >>> d.design.sum(axis=1).round(12).unique().tolist()
    [1.0]
    >>> sp.mixture_design(3, kind="simplex_centroid").n_runs
    7

    References
    ----------
    scheffe1958experiments; scheffe1963simplex; mak2018support
    """
    if isinstance(components, (int, np.integer)) and not isinstance(components, bool):
        names = [f"x{j + 1}" for j in range(int(components))]
    elif isinstance(components, str):
        raise MethodIncompatibility("components is a number or a list of names.")
    else:
        names = [str(c) for c in components]
    q = len(names)
    if q < 2 or len(set(names)) != q:
        raise MethodIncompatibility("A mixture needs at least two distinct components.")
    if total <= 0:
        raise MethodIncompatibility(f"total must be positive; got {total}.")
    key = str(kind).lower().replace("-", "_").replace(" ", "_")
    key = {
        "lattice": "simplex_lattice",
        "sld": "simplex_lattice",
        "scd": "simplex_centroid",
        "centroid": "simplex_centroid",
        "spacefilling": "space_filling",
    }.get(key, key)
    lo = np.zeros(q) if lower is None else np.asarray(lower, dtype=float) / total
    if lo.shape != (q,) or np.any(lo < 0) or lo.sum() >= 1:
        raise MethodIncompatibility(
            "lower gives one non-negative bound per component, summing to "
            "less than the total."
        )
    info: Dict[str, Any] = {"notes": [], "kind": key}
    if key == "simplex_lattice":
        m = int(degree)
        if m < 1:
            raise MethodIncompatibility(f"degree must be at least 1; got {degree}.")
        rows = [c for c in itertools.product(range(m + 1), repeat=q) if sum(c) == m]
        Z = np.array(sorted(rows, reverse=True), dtype=float) / m
        label = f"{{{q}, {m}}} simplex-lattice design"
    elif key == "simplex_centroid":
        if q > 12:
            raise MethodIncompatibility(
                f"A simplex-centroid design in {q} components has {2**q - 1} runs."
            )
        pts: List[np.ndarray] = []
        for size in range(1, q + 1):
            for combo in itertools.combinations(range(q), size):
                z = np.zeros(q)
                z[list(combo)] = 1.0 / size
                pts.append(z)
        Z = np.array(pts)
        label = "simplex-centroid design"
    elif key == "space_filling":
        from .support import _sp_ccp

        if n is None or int(n) < 2:
            raise MethodIncompatibility("kind='space_filling' needs n >= 2 runs.")
        n = int(n)
        cand = _simplex_sample(q, max(8192, 400 * n), seed)
        real = total * (lo + (1.0 - lo.sum()) * cand)
        if constraint is not None:
            ok = np.asarray(constraint(pd.DataFrame(real, columns=names))).astype(bool)
            if ok.shape != (cand.shape[0],):
                raise MethodIncompatibility(
                    "constraint must return one boolean per candidate row."
                )
            cand = cand[ok]
            info["feasible_share"] = float(ok.mean())
        if cand.shape[0] < 20 * n:
            raise DataInsufficient(
                f"Only {cand.shape[0]} candidate blends satisfy the constraint; "
                f"{20 * n} are needed for {n} runs."
            )
        rng = np.random.default_rng(seed)
        # every update is an affine combination of points of the simplex
        # plane, so the iterates stay on it
        P, _, ok_conv = _sp_ccp(cand, n, rng, 500, 1e-5)
        Z = P
        Z = np.clip(Z, 0.0, None)
        Z /= Z.sum(axis=1, keepdims=True)
        if not ok_conv:
            info["notes"].append("Support points not converged in 500 iterations.")
        label = "space-filling mixture design (support points)"
    else:
        raise MethodIncompatibility(
            "kind must be 'simplex_lattice', 'simplex_centroid' or "
            f"'space_filling'; got {kind!r}."
        )
    if key != "space_filling" and constraint is not None:
        raise MethodIncompatibility("constraint applies to kind='space_filling'.")
    X = total * (lo + (1.0 - lo.sum()) * Z)
    return DesignResult(
        design=pd.DataFrame(X, columns=names),
        unit=Z,
        method=label,
        criteria={},
        lower=total * lo,
        upper=np.full(q, total),
        model_info=info,
    )
