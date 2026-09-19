"""Sparse-backed spatial weights object."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

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

    Notes
    -----
    ``w.transform`` re-weights the weights *as constructed* (never a
    previous transform): ``"O"`` original, ``"B"`` binary, ``"R"`` row
    standardised (spdep style ``"W"``), ``"V"`` variance stabilising
    (spdep ``"S"``: ``w_ij / sqrt(sum_j w_ij^2)`` rescaled to sum ``n``),
    ``"D"`` globally standardised to sum 1 (spdep ``"U"``). Each matches
    ``spdep::nb2listw(style=, glist=)`` and libpysal's ``transform``.

    Examples
    --------
    >>> import statspai as sp
    >>> # A 4-node chain graph 0-1-2-3
    >>> neighbors = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    >>> w = sp.W(neighbors)
    >>> w.n
    4
    >>> w.islands
    []
    >>> w.transform = "R"          # row-standardize the weights
    >>> float(w.full()[1].sum())   # each row now sums to 1
    1.0
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
        # Weights as constructed; every transform is computed from these
        # (spdep ``nb2listw(glist = ...)`` / libpysal ``transformations["O"]``).
        self._original = {i: list(v) for i, v in self._weights.items()}
        self._transform = "O"
        self._sparse: Optional[Any] = None

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
    def sparse(self) -> Any:
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
        orig = self._original
        if value == "O":
            base_weights = {i: list(ws) for i, ws in orig.items()}
        elif value == "B":
            base_weights = {i: [1.0] * len(ws) for i, ws in orig.items()}
        elif value == "R":
            # Row-standardise the ORIGINAL weights (spdep style "W"):
            # w_ij / sum_j w_ij. Kernel / inverse-distance weights keep their
            # relative magnitudes; only binary weights become 1 / k_i.
            base_weights = {}
            for i, ws in orig.items():
                s_ = float(sum(ws))
                base_weights[i] = [w / s_ for w in ws] if s_ > 0 else [0.0] * len(ws)
        elif value == "V":
            # Variance-stabilising (spdep style "S", Tiefelsdorf et al. 1999):
            # w_ij / sqrt(sum_j w_ij^2), then rescaled so all weights sum to n.
            scaled = {}
            for i, ws in orig.items():
                q = float(np.sqrt(sum(w * w for w in ws)))
                scaled[i] = [w / q for w in ws] if q > 0 else [0.0] * len(ws)
            big_q = float(sum(sum(ws) for ws in scaled.values()))
            if not big_q > 0:
                raise ValueError("variance-stabilising transform: all weights are zero")
            nq = self.n / big_q
            base_weights = {i: [w * nq for w in ws] for i, ws in scaled.items()}
        else:  # "D": globally standardised to sum 1 (spdep style "U")
            total = float(sum(sum(ws) for ws in orig.values()))
            if not total > 0:
                raise ValueError("global standardisation: all weights are zero")
            base_weights = {i: [w / total for w in ws] for i, ws in orig.items()}
        self._weights = base_weights
        self._transform = value
        self._sparse = None

    def _build_sparse(self) -> Any:
        rows, cols, data = [], [], []
        for i, nbrs in self._neighbors.items():
            row = self._id_to_idx[i]
            ws = self._weights[i]
            for j, w in zip(nbrs, ws):
                if j in self._id_to_idx:
                    rows.append(row)
                    cols.append(self._id_to_idx[j])
                    data.append(float(w))
        return sparse.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))

    def full(self) -> np.ndarray:
        return np.asarray(self.sparse.toarray(), dtype=float)

    def to_libpysal(self) -> Any:
        try:
            from libpysal.weights import W as _LPW
        except ImportError as e:
            raise ImportError(
                "libpysal is required for to_libpysal(). "
                "Install with `pip install libpysal`."
            ) from e
        return _LPW(self._neighbors, self._weights, id_order=self._id_order)

    @classmethod
    def from_libpysal(cls, w: Any) -> "W":
        return cls(dict(w.neighbors), dict(w.weights), id_order=list(w.id_order))
