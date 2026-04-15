"""Sparse-backed spatial weights object."""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy import sparse


class W:
    """Spatial weights matrix with CSR-sparse backing.

    Parameters
    ----------
    neighbors : dict[int, list[int]]
        Mapping observation id -> list of neighbour ids.
    weights : dict[int, list[float]], optional
        Matching weight values. If ``None``, binary (1.0) weights are used.
    id_order : sequence, optional
        Explicit ordering of observation ids. Defaults to ``sorted(neighbors)``.
    """

    _VALID_TRANSFORMS = {"O", "B", "R", "V", "D"}

    def __init__(
        self,
        neighbors: Mapping[int, Sequence[int]],
        weights: Optional[Mapping[int, Sequence[float]]] = None,
        id_order: Optional[Sequence[int]] = None,
    ) -> None:
        if not isinstance(neighbors, Mapping):
            raise TypeError("`neighbors` must be a dict-like mapping id -> list")
        self._id_order = list(id_order) if id_order is not None else sorted(neighbors)
        self._id_to_idx = {i: k for k, i in enumerate(self._id_order)}
        self._neighbors = {i: list(neighbors.get(i, [])) for i in self._id_order}
        if weights is None:
            self._weights = {i: [1.0] * len(v) for i, v in self._neighbors.items()}
        else:
            self._weights = {i: list(weights[i]) for i in self._id_order}
        self._transform = "O"
        self._sparse: Optional[sparse.csr_matrix] = None

    @property
    def n(self) -> int:
        return len(self._id_order)

    @property
    def neighbors(self) -> Dict[int, List[int]]:
        return {i: list(v) for i, v in self._neighbors.items()}

    @property
    def islands(self) -> List[int]:
        return [i for i, v in self._neighbors.items() if len(v) == 0]

    @property
    def sparse(self) -> sparse.csr_matrix:
        if self._sparse is None:
            self._sparse = self._build_sparse()
        return self._sparse

    @property
    def transform(self) -> str:
        return self._transform

    @transform.setter
    def transform(self, value: str) -> None:
        value = value.upper()
        if value not in self._VALID_TRANSFORMS:
            raise ValueError(f"transform must be one of {self._VALID_TRANSFORMS}")
        if value == self._transform:
            return
        base_weights = {i: [1.0] * len(v) for i, v in self._neighbors.items()}
        if value == "R":
            for i, ws in base_weights.items():
                s = sum(ws)
                if s > 0:
                    base_weights[i] = [w / s for w in ws]
        elif value == "V":
            for i, ws in base_weights.items():
                s = sum(ws)
                if s > 0:
                    base_weights[i] = [w / s * np.sqrt(len(ws)) for w in ws]
        elif value == "D":
            total = sum(sum(ws) for ws in base_weights.values())
            if total > 0:
                base_weights = {i: [w / total for w in ws] for i, ws in base_weights.items()}
        self._weights = base_weights
        self._transform = value
        self._sparse = None

    def _build_sparse(self) -> sparse.csr_matrix:
        rows, cols, data = [], [], []
        for i, nbrs in self._neighbors.items():
            row = self._id_to_idx[i]
            ws = self._weights[i]
            for j, w in zip(nbrs, ws):
                if j in self._id_to_idx:
                    rows.append(row)
                    cols.append(self._id_to_idx[j])
                    data.append(float(w))
        return sparse.csr_matrix(
            (data, (rows, cols)), shape=(self.n, self.n)
        )

    def full(self) -> np.ndarray:
        return self.sparse.toarray()

    def to_libpysal(self):
        try:
            from libpysal.weights import W as _LPW
        except ImportError as e:
            raise ImportError(
                "libpysal is required for to_libpysal(). "
                "Install with `pip install libpysal`."
            ) from e
        return _LPW(self._neighbors, self._weights, id_order=self._id_order)

    @classmethod
    def from_libpysal(cls, w) -> "W":
        return cls(dict(w.neighbors), dict(w.weights), id_order=list(w.id_order))
