"""
Spatial regression models via concentrated Maximum Likelihood.

Implements SAR, SEM, and SDM with:
- Row-normalised or raw spatial weights
- ML estimation via eigenvalue decomposition (Ord 1975)
- LM tests for spatial dependence (Anselin 1988)
- Direct / indirect / total effect decomposition for SAR/SDM (LeSage & Pace 2009)

References
----------
Anselin, L. (1988). *Spatial Econometrics: Methods and Models*. Kluwer.
Ord, K. (1975). "Estimation Methods for Models of Spatial Interaction."
    *JASA*, 70(349), 120-126.
LeSage, J. and Pace, R.K. (2009).
    *Introduction to Spatial Econometrics*. CRC Press. [@anselin1988spatial]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ...core.results import EconometricResults

# ====================================================================== #
#  Public API
# ====================================================================== #


def sar(
    W: np.ndarray,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Spatial Autoregressive (Lag) Model:  Y = ρWY + Xβ + ε.

    Parameters
    ----------
    W : (n, n) array
        Spatial weights matrix (contiguity or distance-based).
    data : pd.DataFrame
    formula : str
        ``"y ~ x1 + x2"`` style formula.
    row_normalize : bool
        If True, row-normalise W so each row sums to 1.
    alpha : float
        Significance level.

    Returns
    -------
    EconometricResults
        Includes spatial autoregressive coefficient ρ (rho) as a parameter.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 80
    >>> coords = rng.uniform(0, 10, size=(n, 2))
    >>> W = sp.knn_weights(coords, k=5).sparse.toarray()  # or any (n,n) array
    >>> df = pd.DataFrame({"income": rng.normal(size=n),
    ...                    "education": rng.normal(size=n)})
    >>> df["crime"] = 2 - 0.5 * df["income"] + rng.normal(size=n)
    >>> result = sp.sar(W, data=df, formula='crime ~ income + education')
    >>> list(result.params.index)
    ['const', 'income', 'education', 'rho']
    """
    model = SpatialModel(
        W, data, formula, model_type="sar", row_normalize=row_normalize, alpha=alpha
    )
    _result = model.fit()
    try:
        from ...output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.spatial.sar",
            params={
                "formula": formula,
                "row_normalize": row_normalize,
                "alpha": alpha,
                "W_shape": list(W.shape) if hasattr(W, "shape") else None,
            },
            data=data,
            overwrite=False,
        )
    except (AttributeError, TypeError, KeyError, ValueError):  # pragma: no cover
        # Provenance is decoration; narrow so a bug in it still surfaces.
        pass
    return _result


def sem(
    W: np.ndarray,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Spatial Error Model:  Y = Xβ + u,  u = λWu + ε.

    Parameters
    ----------
    W, data, formula, row_normalize, alpha : as in :func:`sar`.

    Returns
    -------
    EconometricResults
        Includes spatial error parameter λ (lambda).
    """
    model = SpatialModel(
        W, data, formula, model_type="sem", row_normalize=row_normalize, alpha=alpha
    )
    _result = model.fit()
    try:
        from ...output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.spatial.sem",
            params={
                "formula": formula,
                "row_normalize": row_normalize,
                "alpha": alpha,
                "W_shape": list(W.shape) if hasattr(W, "shape") else None,
            },
            data=data,
            overwrite=False,
        )
    except (AttributeError, TypeError, KeyError, ValueError):  # pragma: no cover
        # Provenance is decoration; narrow so a bug in it still surfaces.
        pass
    return _result


def sdm(
    W: np.ndarray,
    data: pd.DataFrame,
    formula: str,
    row_normalize: bool = True,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Spatial Durbin Model:  Y = ρWY + Xβ + WXθ + ε.

    Nests both SAR and SEM as special cases. Use LR tests to determine
    whether SDM simplifies to SAR (θ = 0) or SEM (θ + ρβ = 0).

    Parameters
    ----------
    W, data, formula, row_normalize, alpha : as in :func:`sar`.

    Returns
    -------
    EconometricResults
        Includes ρ (rho), β (own effects), θ (spatial lag effects),
        and direct/indirect/total effect decomposition.
    """
    model = SpatialModel(
        W, data, formula, model_type="sdm", row_normalize=row_normalize, alpha=alpha
    )
    _result = model.fit()
    try:
        from ...output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.spatial.sdm",
            params={
                "formula": formula,
                "row_normalize": row_normalize,
                "alpha": alpha,
                "W_shape": list(W.shape) if hasattr(W, "shape") else None,
            },
            data=data,
            overwrite=False,
        )
    except (AttributeError, TypeError, KeyError, ValueError):  # pragma: no cover
        # Provenance is decoration; narrow so a bug in it still surfaces.
        pass
    return _result


# ====================================================================== #
#  SpatialModel class
# ====================================================================== #


class SpatialModel:
    """
    Unified spatial regression estimator.

    Uses concentrated ML: profile out β and σ², optimise over ρ (or λ)
    in one dimension using eigenvalues of W.

    The engine behind :func:`sp.sar` / :func:`sp.sem` / :func:`sp.sdm`.
    Construct it with a dense ``(n, n)`` weights matrix ``W``, the data
    and a formula, then call :meth:`fit` to obtain an
    ``EconometricResults`` carrying the spatial parameter (``rho`` for
    SAR / SDM, ``lambda`` for SEM).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 50
    >>> W = np.zeros((n, n))            # nearest-neighbour chain weights
    >>> for i in range(n - 1):
    ...     W[i, i + 1] = W[i + 1, i] = 1.0
    >>> df = pd.DataFrame({"x": rng.normal(size=n)})
    >>> df["y"] = 1.0 + 0.5 * df["x"] + rng.normal(size=n)
    >>> model = sp.SpatialModel(W, df, "y ~ x", model_type="sar")
    >>> res = model.fit()
    >>> type(res).__name__
    'EconometricResults'
    >>> model.model_type
    'sar'
    """

    def __init__(
        self,
        W: np.ndarray,
        data: pd.DataFrame,
        formula: str,
        model_type: str = "sar",
        row_normalize: bool = True,
        alpha: float = 0.05,
    ) -> None:
        self.model_type = model_type.lower()
        self.alpha = alpha
        self._W_input = W
        self._data = data
        self._formula = formula
        self._row_normalize = row_normalize

        # Parse formula
        if "~" not in formula:
            raise ValueError("Formula must contain '~'")
        dep, indep = formula.split("~", 1)
        self._dep_var = dep.strip()
        self._indep_vars = [v.strip() for v in indep.split("+") if v.strip()]

        # Prepare data
        self.y = data[self._dep_var].values.astype(np.float64)
        X_raw = data[self._indep_vars].values.astype(np.float64)
        self.n = len(self.y)
        # Add constant
        self.X = np.column_stack([np.ones(self.n), X_raw])
        self.k = self.X.shape[1]
        self._var_names = ["const"] + self._indep_vars

        # Weights matrix
        W = np.asarray(W, dtype=np.float64)
        if W.shape != (self.n, self.n):
            raise ValueError(f"W must be ({self.n}, {self.n}), got {W.shape}")
        if row_normalize:
            row_sums = W.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            W = W / row_sums
        self.W = W

        # Eigenvalues of W (for concentrated log-likelihood bounds)
        self._eigenvalues = np.real(np.linalg.eigvals(W))
        self._rho_min = (
            1.0 / min(self._eigenvalues[self._eigenvalues < 0])
            if any(self._eigenvalues < 0)
            else -0.99
        )
        self._rho_max = (
            1.0 / max(self._eigenvalues[self._eigenvalues > 0])
            if any(self._eigenvalues > 0)
            else 0.99
        )
        # Safety margin
        self._rho_min = max(self._rho_min * 0.99, -0.99)
        self._rho_max = min(self._rho_max * 0.99, 0.99)

    def fit(self) -> EconometricResults:
        """Fit by maximum likelihood with :func:`sp.sar` / :func:`sp.sem` /
        :func:`sp.sdm`, so the class and the functions report one set of
        estimates and standard errors.

        Notes
        -----
        Through 1.40.2 the class kept its own concentrated-likelihood
        fit with the information-matrix formulas the functions had
        replaced: the ``rho`` SE omitted ``tr(G'G)``, the ``beta`` SEs
        treated ``rho`` as known, the ``lambda`` SE used
        ``tr(G^2) + tr(G)^2 / n``, and ``alpha`` never reached the
        intervals.
        """
        from . import ml

        fitters = {"sar": ml.sar, "sem": ml.sem, "sdm": ml.sdm}
        if self.model_type not in fitters:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        return fitters[self.model_type](
            self._W_input,
            self._data,
            self._formula,
            row_normalize=self._row_normalize,
            alpha=self.alpha,
        )
