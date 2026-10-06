"""Shared pieces of ``statspai.doe``: factor specifications, distances, results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

FactorSpec = Union[int, Sequence[str], Mapping[str, Tuple[float, float]]]


def resolve_factors(factors: Any) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Names and lower / upper bounds from an int, a list of names or a dict."""
    if isinstance(factors, (int, np.integer)) and not isinstance(factors, bool):
        p = int(factors)
        if p < 1:
            raise MethodIncompatibility(f"At least one factor is needed; got {p}.")
        return [f"x{j + 1}" for j in range(p)], np.zeros(p), np.ones(p)
    if isinstance(factors, Mapping):
        names = [str(k) for k in factors]
        try:
            lo = np.array([float(factors[k][0]) for k in factors])
            hi = np.array([float(factors[k][1]) for k in factors])
        except (TypeError, IndexError, ValueError) as exc:
            raise MethodIncompatibility(
                "A factor dict maps each name to a (lower, upper) pair."
            ) from exc
        if not names:
            raise MethodIncompatibility("At least one factor is needed.")
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
            raise MethodIncompatibility(
                "Each factor needs finite bounds with lower < upper."
            )
        return names, lo, hi
    if isinstance(factors, str):
        raise MethodIncompatibility(
            "factors is a number, a list of names or a {name: (lower, upper)} dict."
        )
    try:
        names = [str(k) for k in factors]
    except TypeError as exc:
        raise MethodIncompatibility(
            "factors is a number, a list of names or a {name: (lower, upper)} dict."
        ) from exc
    if not names or len(set(names)) != len(names):
        raise MethodIncompatibility("Factor names must be non-empty and distinct.")
    return names, np.zeros(len(names)), np.ones(len(names))


def to_unit(
    design: Any, bounds: Optional[Mapping[str, Tuple[float, float]]] = None
) -> Tuple[np.ndarray, List[str], np.ndarray, np.ndarray]:
    """A design as an array on the unit cube, with names and bounds.

    Without ``bounds`` the design is taken to be on the unit cube already.
    """
    if isinstance(design, DesignResult):
        return (
            design.unit.copy(),
            list(design.design.columns),
            design.lower,
            design.upper,
        )
    if isinstance(design, pd.DataFrame):
        names = [str(c) for c in design.columns]
        X = design.to_numpy(dtype=float)
    else:
        X = np.asarray(design, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        names = [f"x{j + 1}" for j in range(X.shape[1])]
    if X.ndim != 2 or X.shape[0] < 1 or X.shape[1] < 1:
        raise MethodIncompatibility("A design is a runs-by-factors table.")
    if not np.all(np.isfinite(X)):
        raise MethodIncompatibility("The design has missing or infinite entries.")
    p = X.shape[1]
    if bounds is None:
        lo, hi = np.zeros(p), np.ones(p)
        if X.min() < -1e-12 or X.max() > 1 + 1e-12:
            raise MethodIncompatibility(
                "The design is not on the unit cube. Pass bounds={name: (lower, "
                "upper)} so that it can be scaled."
            )
    else:
        bn, lo, hi = resolve_factors(bounds)
        if len(bn) != p:
            raise MethodIncompatibility(
                f"bounds names {len(bn)} factors, the design has {p}."
            )
        if isinstance(design, pd.DataFrame):
            if set(bn) != set(names):
                raise MethodIncompatibility(
                    "bounds and the design do not name the same factors."
                )
            order = [bn.index(nm) for nm in names]
            lo, hi = lo[order], hi[order]
        else:
            names = bn
        X = (X - lo) / (hi - lo)
        if X.min() < -1e-9 or X.max() > 1 + 1e-9:
            raise MethodIncompatibility("The design has runs outside the bounds.")
    return X, names, lo, hi


def pair_sqdist(A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
    """Squared Euclidean distances between the rows of ``A`` and of ``B``."""
    if B is None:
        B = A
    d2 = (A * A).sum(axis=1)[:, None] + (B * B).sum(axis=1)[None, :] - 2.0 * A @ B.T
    np.maximum(d2, 0.0, out=d2)
    return np.asarray(d2)


def upper(M: np.ndarray) -> np.ndarray:
    """The entries above the diagonal."""
    return np.asarray(M[np.triu_indices(M.shape[0], k=1)])


@dataclass
class DesignResult(ResultProtocolMixin):
    """An experimental design: the runs, how it was built and how good it is.

    Attributes
    ----------
    design : DataFrame
        One row per run, one column per factor, in the units of the
        factors.
    unit : ndarray
        The same runs scaled to the unit cube.
    method : str
    criteria : dict
        Space-filling measures of the design (see ``sp.design_criteria``).
    model_info : dict

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.space_filling(8, 2, method="lhs", seed=1)
    >>> d.design.shape
    (8, 2)
    """

    design: pd.DataFrame
    unit: np.ndarray
    method: str
    criteria: Dict[str, float] = field(default_factory=dict)
    lower: np.ndarray = field(default_factory=lambda: np.zeros(0))
    upper: np.ndarray = field(default_factory=lambda: np.zeros(0))
    model_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return int(self.design.shape[0])

    @property
    def n_factors(self) -> int:
        return int(self.design.shape[1])

    def to_frame(self) -> pd.DataFrame:
        """The runs as a DataFrame."""
        return self.design.copy()

    def summary(self) -> str:
        lines = [
            f"Design: {self.method}",
            "=" * 50,
            f"Runs: {self.n_runs}    Factors: {self.n_factors}",
        ]
        for k, v in self.criteria.items():
            lines.append(f"{k:<26}{v:>14.6g}")
        for note in self.model_info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Scatter of the first two factors, or a scatter matrix for more."""
        import matplotlib.pyplot as plt

        if self.n_factors <= 2:
            if ax is None:
                _, ax = plt.subplots(figsize=(4.5, 4.5))
            cols = list(self.design.columns)
            xs = self.design[cols[0]]
            ys = self.design[cols[1]] if len(cols) > 1 else np.zeros(self.n_runs)
            ax.scatter(xs, ys, **{"s": 30, **kwargs})
            ax.set_xlabel(cols[0])
            if len(cols) > 1:
                ax.set_ylabel(cols[1])
            ax.set_title(self.method)
            return ax
        return pd.plotting.scatter_matrix(self.design, **{"alpha": 0.9, **kwargs})
