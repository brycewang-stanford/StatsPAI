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

        # Resolve weights
        if isinstance(weights, str):
            self._weight_col = weights
            self.weights = data[weights].values.astype(np.float64)
        else:
            self._weight_col = "__weight__"
            self.weights = np.asarray(weights, dtype=np.float64)
            self.data[self._weight_col] = self.weights

        if np.any(self.weights <= 0):
            raise ValueError("All sampling weights must be strictly positive")

        # Strata
        self.strata_col = strata
        if strata is not None:
            self.strata = data[strata].values
        else:
            self.strata = np.ones(len(data), dtype=int)

        # PSU / cluster
        self.cluster_col = cluster
        if cluster is not None:
            psu = data[cluster].values
            if nest and strata is not None:
                # Make PSU ids unique within strata
                psu = np.array(
                    [f"{s}__{c}" for s, c in zip(self.strata, psu)]
                )
            self.cluster_ids = psu
        else:
            # Each row is its own PSU
            self.cluster_ids = np.arange(len(data))

        # Finite population correction
        self.fpc_col = fpc
        if fpc is not None:
            raw = data[fpc].values.astype(np.float64)
            if np.all(raw < 1):
                self.fpc_values = raw  # sampling fractions
            else:
                # Convert population sizes to fractions per stratum
                strata_n = pd.Series(self.strata).groupby(self.strata).transform("count").values
                self.fpc_values = strata_n / raw
        else:
            self.fpc_values = None

        self.n = len(data)

    # ------------------------------------------------------------------ #
    #  Convenience methods
    # ------------------------------------------------------------------ #

    def mean(
        self, variables: Union[str, List[str]], alpha: float = 0.05,
    ) -> SurveyResult:
        """Design-corrected weighted mean(s)."""
        return svymean(variables, design=self, alpha=alpha)

    def total(
        self, variables: Union[str, List[str]], alpha: float = 0.05,
    ) -> SurveyResult:
        """Design-corrected weighted total(s)."""
        return svytotal(variables, design=self, alpha=alpha)

    def glm(
        self,
        formula: str,
        family: str = "gaussian",
        alpha: float = 0.05,
    ):
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
    >>> import statspai as sp
    >>> design = sp.svydesign(data=df, weights='pw', strata='region',
    ...                       cluster='psu_id')
    >>> design.mean('income')
    >>> design.glm('income ~ age + education')
    """
    return SurveyDesign(
        data=data, weights=weights, strata=strata,
        cluster=cluster, fpc=fpc, nest=nest,
    )
