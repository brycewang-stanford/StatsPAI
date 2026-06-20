"""``sp.fast.within`` — first-class HDFE residualizer.

Builds the within-projection once for a given FE structure, then lets
the user sweep many y / X vectors through it without re-factorising or
re-detecting singletons. This is the primitive that Phase 3 unlocks for
DML, IV, Lasso, and any downstream estimator that wants to FE-residualise
its inputs without committing to a particular regression API.

Compared to the existing ``sp.Absorber`` (Stata-style ``reghdfe`` API),
``sp.fast.within`` is a thinner wrapper that:

- Calls the Phase 1 Rust ``sp.fast.demean`` kernel directly.
- Returns convergence info on every transform (not buried as state).
- Has DataFrame-friendly accessors (``transform_columns``).

The two co-exist: ``sp.Absorber`` is the stable production path with
LSMR / LSQR backends and weighted support; ``sp.fast.within`` is the
high-throughput AP-only path for the new Rust kernel.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility
from .demean import (
    _demean_core,
    DemeanInfo,
    _detect_singletons,
)
from ._validation import (
    nonempty_sample,
    nonnegative_finite_float,
    positive_int,
)


class WithinTransformer:
    """Reusable within-transform for a fixed FE structure.

    Build once with ``sp.fast.within(data, fe)``, then call
    :meth:`transform` for any X you want residualised against the same
    FEs. Singleton pruning happens at construction; ``keep_mask``
    surfaces which input rows survived.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Source DataFrame. Required when ``fe`` is given as a list of
        column names. Otherwise can be omitted.
    fe : DataFrame | ndarray (n, K) | list of column names | list of arrays
        Fixed-effect specification. Strings, ints, categoricals are all
        accepted (factorised internally).
    drop_singletons : bool, default True
        Iteratively prune rows whose FE level appears once.
    accel : {"aitken", "none"}, default "aitken"
    max_iter : int, default 1000
    tol : float, default 1e-8
    accel_period : int, default 5
    backend : {"auto", "rust", "numpy"}, default "auto"
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        fe: Union[pd.DataFrame, np.ndarray, Sequence, str] = None,
        *,
        drop_singletons: bool = True,
        accel: str = "aitken",
        max_iter: int = 1_000,
        tol: float = 1e-8,
        tol_abs: float = 0.0,
        accel_period: int = 5,
        backend: str = "auto",
    ) -> None:
        if fe is None:
            raise MethodIncompatibility("fast.within: `fe` is required")
        if accel not in ("aitken", "none"):
            raise MethodIncompatibility(
                f"fast.within: accel={accel!r}; expected 'aitken' or 'none'"
            )
        if backend not in ("auto", "rust", "numpy", "jax"):
            raise MethodIncompatibility(
                "fast.within: backend must be one of 'auto', 'rust', 'numpy', or 'jax'"
            )
        max_iter = positive_int(max_iter, name="max_iter", context="fast.within")
        accel_period = positive_int(
            accel_period, name="accel_period", context="fast.within"
        )
        tol = nonnegative_finite_float(tol, name="tol", context="fast.within")
        tol_abs = nonnegative_finite_float(
            tol_abs, name="tol_abs", context="fast.within"
        )

        # Resolve FE columns into a list of 1-D arrays
        fe_arrays: List[np.ndarray] = []
        fe_names: List[str] = []
        if isinstance(fe, pd.DataFrame):
            if fe.shape[1] < 1:
                raise MethodIncompatibility(
                    "fast.within: `fe` must contain at least one fixed-effect column"
                )
            for col in fe.columns:
                fe_arrays.append(fe[col].to_numpy())
                fe_names.append(str(col))
        elif isinstance(fe, np.ndarray):
            if fe.ndim == 1:
                fe_arrays.append(fe)
                fe_names.append("fe0")
            elif fe.ndim == 2:
                if fe.shape[1] < 1:
                    raise MethodIncompatibility(
                        "fast.within: `fe` must contain at least one fixed-effect column"
                    )
                for k in range(fe.shape[1]):
                    fe_arrays.append(fe[:, k])
                    fe_names.append(f"fe{k}")
            else:
                raise MethodIncompatibility(
                    "fast.within: fe ndarray must be 1-D or 2-D"
                )
        elif isinstance(fe, str):
            if data is None:
                raise MethodIncompatibility(
                    "fast.within: pass `data=` when `fe` is a column name"
                )
            if fe not in data.columns:
                raise MethodIncompatibility(
                    f"fast.within: fixed-effect column {fe!r} is not in data"
                )
            fe_arrays.append(data[fe].to_numpy())
            fe_names.append(fe)
        elif isinstance(fe, (list, tuple)):
            if len(fe) < 1:
                raise MethodIncompatibility(
                    "fast.within: `fe` must contain at least one fixed-effect column"
                )
            # Could be a list of column names or a list of arrays.
            if all(isinstance(c, str) for c in fe):
                if data is None:
                    raise MethodIncompatibility(
                        "fast.within: pass `data=` when `fe` is column names"
                    )
                missing = [col for col in fe if col not in data.columns]
                if missing:
                    raise MethodIncompatibility(
                        f"fast.within: fixed-effect columns not in data: {missing}"
                    )
                for col in fe:
                    fe_arrays.append(data[col].to_numpy())
                    fe_names.append(col)
            else:
                for k, c in enumerate(fe):
                    arr = np.asarray(c)
                    if arr.ndim != 1:
                        raise MethodIncompatibility(
                            f"fast.within: fixed-effect array {k} must be 1-D"
                        )
                    fe_arrays.append(arr)
                    fe_names.append(getattr(c, "name", f"fe{k}"))
        else:
            raise MethodIncompatibility(
                f"fast.within: fe of type {type(fe).__name__} is not supported"
            )

        n = fe_arrays[0].shape[0]
        nonempty_sample(n, context="fast.within")
        for name, a in zip(fe_names[1:], fe_arrays[1:]):
            if a.shape[0] != n:
                raise MethodIncompatibility(
                    f"fast.within: fixed-effect column {name!r} has length "
                    f"{a.shape[0]} but expected {n}"
                )

        # Factorise to int64
        fe_codes_raw: List[np.ndarray] = []
        for a in fe_arrays:
            codes, _ = pd.factorize(a, sort=False, use_na_sentinel=True)
            if (codes < 0).any():
                raise MethodIncompatibility("fast.within: NaN in FE column")
            fe_codes_raw.append(codes.astype(np.int64))

        # Singleton drop
        if drop_singletons:
            keep = _detect_singletons(fe_codes_raw, n)
        else:
            keep = np.ones(n, dtype=bool)

        n_kept = int(keep.sum())
        if n_kept < 1:
            raise DataInsufficient(
                "fast.within: no rows remain after singleton pruning"
            )

        # Re-densify post-prune
        fe_codes_kept: List[np.ndarray] = []
        counts_list: List[np.ndarray] = []
        n_fe_post: List[int] = []
        for c in fe_codes_raw:
            ck = c[keep]
            dense, uniq = pd.factorize(ck, sort=False)
            dense = dense.astype(np.int64)
            G = len(uniq)
            fe_codes_kept.append(dense)
            counts_list.append(np.bincount(dense, minlength=G).astype(np.float64))
            n_fe_post.append(G)

        self.fe_names: List[str] = fe_names
        self._fe_codes: List[np.ndarray] = fe_codes_kept
        self._counts_list: List[np.ndarray] = counts_list
        self.keep_mask: np.ndarray = keep
        self.n: int = n
        self.n_kept: int = n_kept
        self.n_dropped: int = n - n_kept
        self.n_fe: List[int] = n_fe_post
        self._accel = accel
        self._max_iter = max_iter
        self._tol = tol
        self._tol_abs = tol_abs
        self._accel_period = accel_period
        self._backend = backend

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def transform(
        self,
        X: Union[np.ndarray, pd.Series, pd.DataFrame],
        *,
        already_masked: bool = False,
    ) -> Tuple[np.ndarray, DemeanInfo]:
        """Residualise X against the cached FE structure.

        Returns ``(X_dem, info)`` where ``X_dem`` has shape
        ``(n_kept, ...)`` and ``info`` has per-column convergence stats.
        """
        if isinstance(X, pd.Series):
            X = X.to_numpy()
        elif isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim not in (1, 2):
            raise MethodIncompatibility(
                f"fast.within.transform: X must be 1-D or 2-D, got ndim={X.ndim}"
            )
        if not np.isfinite(X).all():
            raise MethodIncompatibility(
                "fast.within.transform: X contains non-finite values"
            )

        if already_masked:
            if X.shape[0] != self.n_kept:
                raise MethodIncompatibility(
                    f"fast.within.transform: already_masked=True requires "
                    f"{self.n_kept} rows, got {X.shape[0]}"
                )
        else:
            if X.shape[0] != self.n:
                if X.shape[0] == self.n_kept:
                    already_masked = True
                else:
                    raise MethodIncompatibility(
                        f"fast.within.transform: X has {X.shape[0]} rows but "
                        f"cached n={self.n} (or n_kept={self.n_kept})"
                    )
        if not already_masked:
            X = X[self.keep_mask]

        # Round 4 audit fix: bypass ``_fast_demean``'s factorise +
        # singleton-detect pipeline (we already paid that cost in
        # ``__init__``) and call the kernel core directly. For
        # repeated transforms this is the entire point of the
        # ``WithinTransformer`` cache.
        if X.ndim == 1:
            X_2d = X.reshape(-1, 1).copy()
            squeeze = True
        else:
            X_2d = X.copy()
            squeeze = False

        accelerate = self._accel == "aitken"
        X_dem, iters_out, converged_out, max_dx_out, backend_used = _demean_core(
            X_2d,
            self._fe_codes,
            self._counts_list,
            max_iter=self._max_iter,
            tol=self._tol,
            tol_abs=self._tol_abs,
            accelerate=accelerate,
            accel_period=self._accel_period,
            backend=self._backend,
        )
        info = DemeanInfo(
            n=self.n,
            n_kept=self.n_kept,
            n_dropped=self.n_dropped,
            iters=iters_out,
            converged=converged_out,
            max_dx=max_dx_out,
            keep_mask=self.keep_mask,
            backend=backend_used,
            accel=self._accel,
            n_fe=self.n_fe,
        )
        if squeeze:
            return np.asarray(X_dem).ravel(), info
        return np.ascontiguousarray(X_dem), info

    def transform_columns(
        self,
        data: pd.DataFrame,
        columns: Sequence[str],
        *,
        already_masked: bool = False,
    ) -> pd.DataFrame:
        """Residualise specified DataFrame columns; return a new DataFrame.

        Useful when feeding the within-residualised X into a downstream
        estimator that prefers DataFrames (DML, Lasso, etc.).
        """
        columns = list(columns)
        if not columns:
            raise MethodIncompatibility(
                "fast.within.transform_columns: columns must be non-empty"
            )
        missing = [col for col in columns if col not in data.columns]
        if missing:
            raise MethodIncompatibility(
                f"fast.within.transform_columns: columns not in data: {missing}"
            )
        X = data[columns].to_numpy(dtype=np.float64)
        Xd, _info = self.transform(X, already_masked=already_masked)
        if Xd.ndim == 1:
            Xd = Xd.reshape(-1, 1)
        # Index alignment: kept rows from the original index
        if already_masked:
            idx = data.index
        else:
            idx = data.index[self.keep_mask]
        return pd.DataFrame(Xd, index=idx, columns=columns)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover  - cosmetic
        return (
            f"WithinTransformer(K={len(self._fe_codes)}, n={self.n}, "
            f"n_kept={self.n_kept}, fe={list(zip(self.fe_names, self.n_fe))})"
        )


def within(
    data: Optional[pd.DataFrame] = None,
    fe: Union[pd.DataFrame, np.ndarray, Sequence] = None,
    **kwargs: Any,
) -> WithinTransformer:
    """Build a ``WithinTransformer`` from a DataFrame + FE spec.

    Convenience constructor. See :class:`WithinTransformer` for arguments.
    """
    return WithinTransformer(data=data, fe=fe, **kwargs)


__all__ = ["within", "WithinTransformer"]
