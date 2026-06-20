"""
Survey design specification.

A ``SurveyDesign`` carries metadata about the sampling design (weights,
strata, PSU clusters) and provides convenience methods that automatically
apply design-corrected estimation.  Modelled after R's ``svydesign()`` and
Stata's ``svyset``.
"""

from __future__ import annotations

from typing import Optional, Union, List

import numpy as np
import pandas as pd

from .estimators import svymean, svytotal, svyglm, SurveyResult


class SurveyDesign:
    """
    Declare a complex survey design.

    Parameters
    ----------
    data : pd.DataFrame
        Survey microdata.
    weights : str or array-like
        Sampling weights (inverse probability).  If *str*, column name in
        *data*.
    strata : str or None
        Stratification variable (column name).
    cluster : str or None
        Primary sampling unit (PSU) variable (column name).
    fpc : str or None
        Finite population correction — column of stratum population sizes
        or sampling fractions.  If values are < 1 they are treated as
        fractions; otherwise as population counts.
    nest : bool
        If True, PSU ids are nested within strata (re-label internally).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame({
    ...     "stratum": rng.integers(0, 3, size=n),
    ...     "psu": rng.integers(0, 30, size=n),
    ...     "wt": rng.uniform(0.5, 2.0, size=n),
    ...     "income": rng.normal(50, 10, size=n),
    ... })
    >>> design = sp.SurveyDesign(df, weights="wt", strata="stratum",
    ...                          cluster="psu")
    >>> design.n
    300
    >>> m = design.mean("income")
    >>> type(m).__name__
    'SurveyResult'
    """

    def __init__(
        self,
        data: pd.DataFrame,
        weights: Union[str, np.ndarray],
        strata: Optional[str] = None,
        cluster: Optional[str] = None,
        fpc: Optional[str] = None,
        nest: bool = False,
    ):
        self.data = data.copy()
        n_obs = len(data)

        # Resolve weights
        if isinstance(weights, str):
            if weights not in data.columns:
                raise ValueError(f"weights='{weights}' is not a column in data")
            self._weight_col = weights
            self.weights = data[weights].values.astype(np.float64)
        else:
            self._weight_col = "__weight__"
            self.weights = np.asarray(weights, dtype=np.float64)
            if self.weights.shape[0] != n_obs:
                raise ValueError(
                    f"weights length ({self.weights.shape[0]}) must match "
                    f"data length ({n_obs})"
                )
            self.data[self._weight_col] = self.weights

        if not np.all(np.isfinite(self.weights)):
            raise ValueError("All sampling weights must be finite")
        if np.any(self.weights <= 0):
            raise ValueError("All sampling weights must be strictly positive")

        # Strata
        self.strata_col = strata
        if strata is not None:
            if strata not in data.columns:
                raise ValueError(f"strata='{strata}' is not a column in data")
            self.strata = data[strata].values
        else:
            self.strata = np.ones(n_obs, dtype=int)

        # PSU / cluster
        self.cluster_col = cluster
        if cluster is not None:
            if cluster not in data.columns:
                raise ValueError(f"cluster='{cluster}' is not a column in data")
            psu = data[cluster].values
            if nest and strata is not None:
                # Make PSU ids unique within strata
                psu = np.array([f"{s}__{c}" for s, c in zip(self.strata, psu)])
            self.cluster_ids = psu
        else:
            # Each row is its own PSU
            self.cluster_ids = np.arange(n_obs)

        # Finite population correction
        self.fpc_col = fpc
        if fpc is not None:
            if fpc not in data.columns:
                raise ValueError(f"fpc='{fpc}' is not a column in data")
            raw = data[fpc].values.astype(np.float64)
            if not np.all(np.isfinite(raw)) or np.any(raw <= 0):
                raise ValueError("fpc values must be finite and strictly positive")
            if np.all(raw < 1):
                self.fpc_values = raw  # sampling fractions
            else:
                # Convert population sizes to fractions per stratum
                strata_n = (
                    pd.Series(self.strata)
                    .groupby(self.strata)
                    .transform("count")
                    .values
                )
                self.fpc_values = strata_n / raw
        else:
            self.fpc_values = None

        self.n = n_obs

    # ------------------------------------------------------------------ #
    #  Convenience methods
    # ------------------------------------------------------------------ #

    def mean(
        self,
        variables: Union[str, List[str]],
        alpha: float = 0.05,
    ) -> SurveyResult:
        """Design-corrected weighted mean(s)."""
        return svymean(variables, design=self, alpha=alpha)

    def total(
        self,
        variables: Union[str, List[str]],
        alpha: float = 0.05,
    ) -> SurveyResult:
        """Design-corrected weighted total(s)."""
        return svytotal(variables, design=self, alpha=alpha)

    def glm(
        self,
        formula: str,
        family: str = "gaussian",
        alpha: float = 0.05,
    ) -> SurveyResult:
        """Survey-weighted generalised linear model."""
        return svyglm(formula, design=self, family=family, alpha=alpha)

    def __repr__(self) -> str:
        parts = [f"SurveyDesign(n={self.n}"]
        if self.strata_col:
            n_strata = len(np.unique(self.strata))
            parts.append(f"strata={self.strata_col}[{n_strata}]")
        if self.cluster_col:
            n_psu = len(np.unique(self.cluster_ids))
            parts.append(f"cluster={self.cluster_col}[{n_psu}]")
        parts.append(f"weights={self._weight_col}")
        return ", ".join(parts) + ")"


def svydesign(
    data: pd.DataFrame,
    weights: Union[str, np.ndarray],
    strata: Optional[str] = None,
    cluster: Optional[str] = None,
    fpc: Optional[str] = None,
    nest: bool = False,
) -> SurveyDesign:
    """
    Create a survey design object — functional interface.

    Parameters are identical to :class:`SurveyDesign`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame({
    ...     "region": rng.integers(0, 3, size=n),
    ...     "psu_id": rng.integers(0, 30, size=n),
    ...     "pw": rng.uniform(0.5, 2.0, size=n),
    ...     "income": rng.normal(50, 10, size=n),
    ...     "age": rng.normal(40, 12, size=n),
    ... })
    >>> design = sp.svydesign(data=df, weights='pw', strata='region',
    ...                       cluster='psu_id')
    >>> type(design).__name__
    'SurveyDesign'
    >>> m = design.mean('income')
    >>> g = design.glm('income ~ age')
    """
    return SurveyDesign(
        data=data,
        weights=weights,
        strata=strata,
        cluster=cluster,
        fpc=fpc,
        nest=nest,
    )
