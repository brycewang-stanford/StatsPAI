"""
Survey design specification.

A ``SurveyDesign`` carries metadata about the sampling design (weights,
strata, PSU clusters) and provides convenience methods that automatically
apply design-corrected estimation.  Modelled after R's ``svydesign()`` and
Stata's ``svyset``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility
from .estimators import SurveyResult, svyglm, svymean, svytotal


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
        Finite population correction, as in R ``survey::svydesign(fpc=)``
        and Stata ``svyset, fpc()``: either the population number of PSUs
        in the stratum (all values >= 1, at least one > 1) or the
        first-stage sampling fraction (all values <= 1).  For an
        element-sampled design (``cluster=None``) the PSUs are the rows,
        so the count form is the population number of elements.  Mixing
        the two forms, or a count smaller than the number of sampled PSUs
        in the stratum, raises ``ValueError``.
    nest : bool
        If True, PSU ids are relabelled to be unique within strata.  PSUs
        are always identified *within* their stratum for variance and
        degrees-of-freedom purposes (Stata ``svyset`` convention), so with
        ``nest=False`` ids that repeat across strata only trigger a
        warning (R's ``svydesign`` refuses them without ``nest=TRUE``).
    lonely_psu : {"remove", "certainty", "adjust", "average", "fail"}
        How a stratum with a single sampled PSU enters the variance, with
        the semantics of R's ``options(survey.lonely.psu=)`` for a
        single-stage design:

        * ``"remove"`` (default) / ``"certainty"`` -- the stratum
          contributes zero (Stata ``singleunit(certainty)``);
        * ``"adjust"`` -- the single PSU total is centred at the grand
          mean of PSU totals instead of its stratum mean (Stata
          ``singleunit(centered)``);
        * ``"average"`` -- the variance summed over strata with >1 PSU is
          scaled by #strata / #strata-with->1-PSU (Stata
          ``singleunit(scaled)``);
        * ``"fail"`` -- raise ``ValueError`` (R's default).

        Unless ``lonely_psu`` is given explicitly, a lonely stratum emits
        a ``UserWarning`` (Stata's default reports a missing SE instead).

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
    ...                          cluster="psu", nest=True)
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
        lonely_psu: Optional[str] = None,
    ):
        self.data = data.copy()
        n_obs = len(data)
        self._init_kwargs = {
            "strata": strata,
            "cluster": cluster,
            "fpc": fpc,
            "nest": nest,
            "lonely_psu": lonely_psu,
        }
        #: set by :meth:`calibrate`: the auxiliary design matrix and the
        #: weights its linearisation regression uses (see ``_design_vcov``).
        self._calibration: Optional[dict] = None

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

        # PSUs are identified within strata (Stata svyset; R nest=TRUE).
        self._strata_codes = pd.factorize(pd.Series(self.strata), sort=True)[0]
        psu_key = pd.MultiIndex.from_arrays(
            [pd.Series(self.strata), pd.Series(self.cluster_ids)]
        )
        self._psu_codes = pd.factorize(psu_key, sort=True)[0]
        if np.any(self._strata_codes < 0) or np.any(self._psu_codes < 0):
            raise MethodIncompatibility(
                "strata / cluster columns must not contain missing values"
            )
        if cluster is not None and strata is not None and not nest:
            crossed = (
                pd.DataFrame({"s": self.strata, "c": self.cluster_ids})
                .drop_duplicates()
                .groupby("c")["s"]
                .nunique()
            )
            if (crossed > 1).any():
                warnings.warn(
                    f"PSU ids in '{cluster}' repeat across strata; they are "
                    "treated as distinct PSUs within each stratum (Stata "
                    "svyset convention; R svydesign requires nest=TRUE). "
                    "Pass nest=True to silence this warning.",
                    UserWarning,
                    stacklevel=3,
                )
        # number of sampled PSUs in the row's stratum
        n_psu_by_stratum = (
            pd.Series(self._psu_codes).groupby(self._strata_codes).nunique()
        )
        self._n_psu_h = n_psu_by_stratum.reindex(self._strata_codes).to_numpy(
            dtype=np.float64
        )

        # Lonely-PSU rule
        _lonely_ok = ("remove", "certainty", "adjust", "average", "fail")
        self._lonely_psu_explicit = lonely_psu is not None
        if lonely_psu is None:
            lonely_psu = "remove"
        if lonely_psu not in _lonely_ok:
            raise MethodIncompatibility(
                f"lonely_psu must be one of {_lonely_ok}; got {lonely_psu!r}"
            )
        self.lonely_psu = lonely_psu

        # Finite population correction (R survey:::as.fpc semantics):
        # fpc_values holds the first-stage sampling fraction n_h / N_h per row.
        self.fpc_col = fpc
        if fpc is not None:
            if fpc not in data.columns:
                raise ValueError(f"fpc='{fpc}' is not a column in data")
            raw = data[fpc].values.astype(np.float64)
            if np.any(np.isnan(raw)) or np.any(raw <= 0):
                raise MethodIncompatibility(
                    "fpc values must be non-missing and strictly positive"
                )
            is_popsize = bool(np.any(raw > 1))
            if is_popsize != bool(np.all(raw >= 1)):
                raise MethodIncompatibility(
                    "fpc must be all population counts (>= 1) or all "
                    "sampling fractions (<= 1)"
                )
            if is_popsize:
                if np.any(raw < self._n_psu_h):
                    raise MethodIncompatibility(
                        "fpc implies more than 100% sampling in some strata: "
                        "the population count is smaller than the number of "
                        "sampled PSUs"
                    )
                self.fpc_values = np.where(np.isinf(raw), 0.0, self._n_psu_h / raw)
            else:
                self.fpc_values = raw  # sampling fractions
            varies = (
                pd.Series(self.fpc_values).groupby(self._strata_codes).nunique() > 1
            )
            if varies.any():
                warnings.warn(
                    f"fpc '{fpc}' varies within strata; each PSU uses the value "
                    "on its first row (as R survey does).",
                    UserWarning,
                    stacklevel=3,
                )
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
        **kwargs,
    ) -> SurveyResult:
        """Design-corrected weighted mean(s); ``kwargs`` go to ``sp.svymean``."""
        return svymean(variables, design=self, alpha=alpha, **kwargs)

    def total(
        self,
        variables: Union[str, List[str]],
        alpha: float = 0.05,
        **kwargs,
    ) -> SurveyResult:
        """Design-corrected weighted total(s); ``kwargs`` go to ``sp.svytotal``."""
        return svytotal(variables, design=self, alpha=alpha, **kwargs)

    def glm(
        self,
        formula: str,
        family: str = "gaussian",
        alpha: float = 0.05,
        **kwargs,
    ) -> SurveyResult:
        """Survey-weighted GLM; ``kwargs`` go to ``sp.svyglm``."""
        return svyglm(formula, design=self, family=family, alpha=alpha, **kwargs)

    def calibrate(
        self,
        *,
        margins: Optional[Dict[str, Dict]] = None,
        totals: Optional[Dict[str, float]] = None,
        variance: str = "greg",
        max_iter: int = 100,
        tol: float = 1e-10,
    ) -> "SurveyDesign":
        """Calibrated design whose standard errors account for calibration.

        Raking to ``margins`` (as :func:`sp.rake`) or linear GREG
        calibration to ``totals`` (as :func:`sp.linear_calibration`).
        Unlike feeding the calibrated weights to a new design -- which
        treats them as fixed -- the returned design keeps the auxiliary
        variables, and every estimator's linearisation scores are replaced
        by their residuals from the weighted regression on them before the
        design variance is formed. Estimates of the calibration totals
        themselves then have zero variance, as they should.

        Parameters
        ----------
        margins : dict, optional
            ``{column: {category: population count}}`` -- raking. Weights
            are put on the population scale (sum = the margins' total), as R
            ``rake`` / ``calibrate`` and Stata ``svycal``. Proportions are
            accepted; the weights then keep the design-weight total.
        totals : dict, optional
            ``{column: population total}`` -- linear (chi-squared distance)
            calibration, no intercept added (include a column of ones with
            total N for one).
        variance : {"greg", "stata"}, default "greg"
            Weights of the residualising regression. ``"greg"`` uses the
            design weights and multiplies the residuals by the calibrated
            weights -- the g-weighted residual variance of Särndal,
            Swensson & Wretman, as R ``survey::calibrate`` computes it.
            ``"stata"`` uses the calibrated weights throughout, as Stata
            ``svyset, rake()`` / ``regress()``. The two are asymptotically
            equivalent and differ at O(1/n) (0.5 % on the reference data).
        max_iter, tol
            Raking controls (see :func:`sp.rake`).

        Returns
        -------
        SurveyDesign

        Examples
        --------
        >>> import numpy as np, pandas as pd
        >>> import statspai as sp
        >>> rng = np.random.default_rng(0)
        >>> n = 400
        >>> df = pd.DataFrame({
        ...     "psu": rng.integers(0, 40, n), "stratum": rng.integers(0, 4, n),
        ...     "d": rng.uniform(20, 40, n), "sex": rng.choice(["F", "M"], n),
        ... })
        >>> df["y"] = 5 + (df["sex"] == "M") + rng.normal(size=n)
        >>> des = sp.svydesign(df, weights="d", strata="stratum",
        ...                    cluster="psu", nest=True)
        >>> cal = des.calibrate(margins={"sex": {"F": 6000, "M": 5000}})
        >>> round(float(cal.weights.sum()))
        11000
        >>> bool(cal.mean("y").std_error.iloc[0] < des.mean("y").std_error.iloc[0])
        True
        """
        from .calibration import linear_calibration, rake

        if (margins is None) == (totals is None):
            raise MethodIncompatibility(
                "calibrate: pass exactly one of margins= (raking) or totals= "
                "(linear calibration)."
            )
        if variance not in ("greg", "stata"):
            raise MethodIncompatibility(
                f"calibrate: variance must be 'greg' or 'stata', got {variance!r}."
            )
        d = self.weights
        data = self.data.assign(__design_w__=d)
        if margins is not None:
            res = rake(data, margins, weight="__design_w__", max_iter=max_iter, tol=tol)
            sums = [float(sum(t.values())) for t in margins.values()]
            scale = sums[0] if min(sums) > 1.0 + 1e-12 else float(d.sum())
            w_new = res.calibrated_weights * scale
            cols: List[np.ndarray] = []
            for col, targets in margins.items():
                vals = self.data[col].to_numpy()
                cols.extend((vals == c).astype(float) for c in targets)
            X = np.column_stack(cols)
            method = "raking"
        else:
            assert totals is not None  # exactly one of margins / totals
            res = linear_calibration(data, totals, weight="__design_w__")
            w_new = res.calibrated_weights
            X = self.data[list(totals)].to_numpy(dtype=float)
            method = "linear"
        if not np.all(w_new > 0):
            raise MethodIncompatibility(
                "calibrate: linear calibration produced non-positive weights "
                f"({int((w_new <= 0).sum())} rows); the design cannot use them.",
                recovery_hint="Use raking (margins=), or bounded calibration.",
            )
        kw: Dict[str, Any] = dict(self._init_kwargs)
        base = self.data.drop(columns=["__weight__"], errors="ignore")
        new = SurveyDesign(base, weights=w_new, **kw)
        new._calibration = {
            "X": X,
            "reg_weights": d.copy() if variance == "greg" else w_new.copy(),
            "method": method,
            "variance": variance,
            "converged": bool(res.converged),
        }
        return new

    def __repr__(self) -> str:
        parts = [f"SurveyDesign(n={self.n}"]
        if self.strata_col:
            n_strata = len(np.unique(self.strata))
            parts.append(f"strata={self.strata_col}[{n_strata}]")
        if self.cluster_col:
            n_psu = len(np.unique(self.cluster_ids))
            parts.append(f"cluster={self.cluster_col}[{n_psu}]")
        parts.append(f"weights={self._weight_col}")
        if self._calibration is not None:
            parts.append(
                f"calibrated={self._calibration['method']}"
                f"[{self._calibration['variance']}]"
            )
        return ", ".join(parts) + ")"


def svydesign(
    data: pd.DataFrame,
    weights: Union[str, np.ndarray],
    strata: Optional[str] = None,
    cluster: Optional[str] = None,
    fpc: Optional[str] = None,
    nest: bool = False,
    lonely_psu: Optional[str] = None,
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
    ...                       cluster='psu_id', nest=True)
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
        lonely_psu=lonely_psu,
    )
