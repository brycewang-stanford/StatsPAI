"""
Synthetic Control Method — Unified Entry Point.

Provides ``synth()`` as a single dispatcher for 20 SCM variants:

* **classic** �� Abadie, Diamond & Hainmueller (2010)
* **penalized / ridge** — Ridge-penalised SCM
* **demeaned / detrended** — Ferman & Pinto (2021)
* **unconstrained / elastic_net** — Doudchenko & Imbens (2016)
* **augmented / ascm** — Ben-Michael, Feller & Rothstein (2021)
* **sdid** — Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)
* **factor / gsynth** — Xu (2017)
* **staggered** — Ben-Michael, Feller & Rothstein (2022)
* **mc / matrix_completion** — Athey, Bayati et al. (2021)
* **discos / distributional** — Gunsilius (2023)
* **multi_outcome** — Sun (2023)
* **scpi / prediction_interval** — Cattaneo, Feng & Titiunik (2021)
* **bayesian** — Bayesian SCM with MCMC posterior (Vives & Martinez 2024)
* **bsts / causal_impact** — Bayesian Structural Time Series (Brodersen et al. 2015)
* **penscm / abadie_lhour** — Penalized SCM (Abadie & L'Hour 2021)
* **fdid / forward_did** — Forward DID (Li 2024)
* **cluster** — Cluster SCM (Rho et al. 2025, arXiv:2503.21629)
* **sparse / lasso** — Sparse SCM (Amjad, Shah & Shen 2018)
* **kernel** — Kernel-based nonlinear SCM
* **kernel_ridge** — Kernel ridge regression SCM

Inference can be switched independently via ``inference=``:

* **placebo** (default) ��� in-space permutation
* **conformal** — Chernozhukov, Wüthrich & Zhu (2021)
* **bootstrap / jackknife** — for SDID
* **bayesian posterior** — MCMC credible intervals
* **bsts posterior** — Kalman-based uncertainty
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult


def synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any = None,
    treatment_time: Any = None,
    method: str = "augmented",
    covariates: Optional[List[str]] = None,
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
    inference: Optional[str] = None,
    treatment: Optional[str] = None,
    **kwargs,
) -> CausalResult:
    """
    Unified Synthetic Control estimator with multiple method variants.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any, optional
        Identifier of the treated unit. Required for all methods
        except ``'staggered'``.
    treatment_time : any, optional
        First treatment period (inclusive). Required for all methods
        except ``'staggered'``.
    method : str, default 'classic'
        SCM variant:

        * ``'classic'`` — Standard SCM (Abadie et al. 2010).
        * ``'penalized'`` / ``'ridge'`` — SCM with ridge penalty.
        * ``'demeaned'`` — De-meaned SCM (Ferman & Pinto 2021).
        * ``'detrended'`` — De-trended SCM (Ferman & Pinto 2021).
        * ``'unconstrained'`` — No sign/sum constraints
          (Doudchenko & Imbens 2016).
        * ``'elastic_net'`` — Elastic-net regularised weights.
        * ``'augmented'`` / ``'ascm'`` — Augmented SCM
          (Ben-Michael et al. 2021).
        * ``'sdid'`` — Synthetic DID (Arkhangelsky et al. 2021).
        * ``'factor'`` / ``'gsynth'`` — Factor model (Xu 2017).
        * ``'staggered'`` — Staggered adoption (Ben-Michael et al. 2022).
        * ``'mc'`` / ``'matrix_completion'`` — Matrix completion
          (Athey et al. 2021).
        * ``'discos'`` / ``'distributional'`` — Distributional SCM
          (Gunsilius 2023).
        * ``'multi_outcome'`` / ``'multi'`` — Multiple outcomes SCM
          (Sun 2023). Requires ``outcomes`` kwarg.
        * ``'scpi'`` / ``'prediction_interval'`` — SCM with prediction
          intervals (Cattaneo et al. 2021).
        * ``'bayesian'`` — Bayesian SCM with MCMC posterior
          (Vives & Martinez 2024).
        * ``'bsts'`` / ``'causal_impact'`` — Bayesian Structural
          Time Series (Brodersen et al. 2015).
        * ``'penscm'`` / ``'abadie_lhour'`` — Penalized SCM with
          pairwise discrepancy (Abadie & L'Hour 2021).
        * ``'fdid'`` / ``'forward_did'`` — Forward DID
          (Li 2024).
        * ``'cluster'`` — Cluster SCM (Rho et al. 2025, arXiv:2503.21629). [@rho2025clustersc]
        * ``'sparse'`` / ``'lasso'`` — Sparse SCM
          (Amjad, Shah & Shen 2018).
        * ``'kernel'`` — Kernel-based nonlinear SCM.
        * ``'kernel_ridge'`` — Kernel ridge regression SCM.
    covariates : list of str, optional
        Additional covariates to match on.
    penalization : float, default 0.0
        Ridge penalty for donor weights.
    placebo : bool, default True
        Run placebo inference.
    alpha : float, default 0.05
        Significance level.
    inference : str, optional
        Override default inference: ``'placebo'``, ``'conformal'``,
        ``'bootstrap'``, ``'jackknife'``.
    treatment : str, optional
        Binary treatment column (required for ``method='staggered'``).
    **kwargs
        Method-specific arguments passed through to the variant.

    Returns
    -------
    CausalResult
        A unified result object. Fields common to all 20 backends:

        * ``estimate`` : float — ATT (post-treatment average effect).
        * ``se`` : float — standard error (``NaN`` if ``placebo=False``
          and the method has no analytic SE).
        * ``pvalue`` : float — two-sided; floor ``1/(J+1)`` for permutation.
        * ``ci`` : tuple[float, float] — ``(1-alpha)`` confidence interval.
        * ``detail`` : pd.DataFrame — one row per post-treatment period with
          columns ``time, treated, counterfactual, effect``.
        * ``model_info`` : dict — method-specific diagnostics. Keys present
          for most methods: ``pre_rmspe``, ``post_rmspe``, ``weights``,
          ``n_donors``, ``n_pre_periods``, ``n_post_periods``. Extra keys
          are method-specific — see each variant's own docstring
          (``help(sp.bayesian_synth)``, ``help(sp.mc_synth)``, ...).

    Notes
    -----
    Run ``sp.synth_compare(...)`` to run every method at once and compare
    point estimates, pre-RMSPE, and placebo p-values side by side.

    Examples
    --------
    Classic SCM:

    >>> result = sp.synth(df, outcome='gdp', unit='state', time='year',
    ...                   treated_unit='California', treatment_time=1989)

    De-meaned:

    >>> result = sp.synth(..., method='demeaned')

    Unconstrained (negative weights):

    >>> result = sp.synth(..., method='unconstrained')

    Factor model:

    >>> result = sp.synth(..., method='gsynth', n_factors=3)

    Conformal inference:

    >>> result = sp.synth(..., inference='conformal')

    Staggered adoption:

    >>> result = sp.synth(df, outcome='gdp', unit='state', time='year',
    ...                   treatment='treated', method='staggered')

    Matrix completion:

    >>> result = sp.synth(..., method='mc')

    Distributional synthetic controls:

    >>> result = sp.synth(..., method='discos')

    Multiple outcomes:

    >>> result = sp.synth(df, outcome='gdp', unit='state', time='year',
    ...                   treated_unit='California', treatment_time=1989,
    ...                   method='multi_outcome',
    ...                   outcomes=['gdp', 'employment', 'investment'])

    Prediction intervals:

    >>> result = sp.synth(..., method='scpi')

    See Also
    --------
    sdid, augsynth, gsynth, demeaned_synth, robust_synth,
    staggered_synth, conformal_synth
    """
    method = method.lower().strip()

    # --- Conformal inference override ---
    if inference == "conformal":
        from .conformal import conformal_synth
        return conformal_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            scm_method=method if method in ("classic", "ridge") else "classic",
            penalization=penalization, alpha=alpha, **kwargs,
        )

    # --- Dispatch ---
    if method in ("classic", "classic_adh", "adh", "penalized", "ridge"):
        if method in ("penalized", "ridge") and penalization == 0.0:
            penalization = kwargs.pop("l2_penalty", 0.01)
        special_predictors = kwargs.pop("special_predictors", None)
        v_method = kwargs.pop("v_method", "auto")
        standardize_predictors = kwargs.pop("standardize_predictors", True)
        n_random_starts = kwargs.pop("n_random_starts", 4)
        model = SyntheticControl(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates,
            special_predictors=special_predictors,
            v_method=v_method,
            standardize_predictors=standardize_predictors,
            n_random_starts=n_random_starts,
            penalization=penalization, alpha=alpha,
        )
        return model.fit(placebo=placebo)

    if method in ("demeaned", "detrended"):
        from .demeaned import demeaned_synth
        return demeaned_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, variant=method,
            penalization=penalization, placebo=placebo, alpha=alpha,
            **kwargs,
        )

    if method in ("unconstrained", "elastic_net"):
        from .robust import robust_synth
        return robust_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, variant=method,
            placebo=placebo, alpha=alpha, **kwargs,
        )

    if method in ("augmented", "ascm"):
        from .augsynth import augsynth
        return augsynth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, alpha=alpha, **kwargs,
        )

    if method == "sdid":
        from .sdid import sdid as _sdid
        se_method = inference or "placebo"
        return _sdid(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            method="sdid", covariates=covariates,
            se_method=se_method, alpha=alpha, **kwargs,
        )

    if method in ("factor", "gsynth"):
        from .gsynth import gsynth as _gsynth
        return _gsynth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, placebo=placebo, alpha=alpha,
            **kwargs,
        )

    if method == "staggered":
        from .staggered import staggered_synth
        if treatment is None:
            raise ValueError(
                "method='staggered' requires the `treatment` parameter "
                "(binary treatment indicator column name)"
            )
        return staggered_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treatment=treatment, penalization=penalization,
            placebo=placebo, alpha=alpha, **kwargs,
        )

    if method in ("mc", "matrix_completion"):
        from .mc import mc_synth
        return mc_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            alpha=alpha, placebo=placebo, **kwargs,
        )

    if method in ("discos", "distributional"):
        from .discos import discos
        return discos(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            alpha=alpha, placebo=placebo, **kwargs,
        )

    if method in ("multi_outcome", "multi"):
        from .multi_outcome import multi_outcome_synth
        outcomes = kwargs.pop("outcomes", None)
        if outcomes is None:
            outcomes = [outcome]
        return multi_outcome_synth(
            data=data, outcomes=outcomes, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            penalization=penalization, alpha=alpha, placebo=placebo,
            **kwargs,
        )

    if method in ("scpi", "prediction_interval"):
        from .scpi import scpi
        return scpi(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            alpha=alpha, **kwargs,
        )

    # --- New methods (v0.9) ---

    if method == "bayesian":
        from .bayesian import bayesian_synth
        return bayesian_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, alpha=alpha, **kwargs,
        )

    if method in ("bsts", "causal_impact"):
        from .bsts import bsts_synth
        return bsts_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, alpha=alpha, **kwargs,
        )

    if method in ("penscm", "abadie_lhour", "pairwise"):
        from .penscm import penalized_synth
        return penalized_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, placebo=placebo, alpha=alpha,
            **kwargs,
        )

    if method in ("fdid", "forward_did"):
        from .fdid import fdid as _fdid
        return _fdid(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            placebo=placebo, alpha=alpha, **kwargs,
        )

    if method == "cluster":
        from .cluster import cluster_synth
        return cluster_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, placebo=placebo, alpha=alpha,
            **kwargs,
        )

    if method in ("sparse", "lasso"):
        from .sparse import sparse_synth
        return sparse_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, placebo=placebo, alpha=alpha,
            **kwargs,
        )

    if method == "kernel":
        from .kernel import kernel_synth
        return kernel_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, placebo=placebo, alpha=alpha,
            **kwargs,
        )

    if method == "kernel_ridge":
        from .kernel import kernel_ridge_synth
        return kernel_ridge_synth(
            data=data, outcome=outcome, unit=unit, time=time,
            treated_unit=treated_unit, treatment_time=treatment_time,
            covariates=covariates, placebo=placebo, alpha=alpha, **kwargs,
        )

    raise ValueError(
        f"Unknown method {method!r}. Choose from: 'classic', 'penalized', "
        f"'ridge', 'demeaned', 'detrended', 'unconstrained', 'elastic_net', "
        f"'augmented', 'ascm', 'sdid', 'factor', 'gsynth', 'staggered', "
        f"'mc', 'discos', 'multi_outcome', 'scpi', "
        f"'bayesian', 'bsts', 'causal_impact', 'penscm', 'abadie_lhour', "
        f"'fdid', 'forward_did', 'cluster', 'sparse', 'lasso', "
        f"'kernel', 'kernel_ridge'."
    )


SpecialPredictor = Tuple[str, Any, str]  # (col, period_spec, op)


class SyntheticControl:
    """
    Canonical Synthetic Control estimator (Abadie, Diamond & Hainmueller
    2010) with nested V-W optimization.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel ``(unit, time, outcome, ...)``.
    outcome, unit, time : str
        Column names.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period (inclusive).
    covariates : list of str, optional
        Column names whose pre-treatment means are used as predictors for
        the V-weighted matching problem.
    special_predictors : list of tuple, optional
        R/Stata ``Synth``-style predictor specifications. Each entry is
        ``(column, period_spec, op)`` where ``period_spec`` is a scalar
        year, a list of years, or a ``slice(start, stop)`` (inclusive),
        and ``op`` is ``'mean'`` or ``'sum'``. When omitted together with
        ``covariates``, the pre-treatment outcome vector itself is used as
        the predictor (V has no identifying power and is fixed to the
        identity, following Kaul et al. 2015).
    v_method : {'auto', 'nested', 'equal'}, default 'auto'
        ``'auto'`` → nested V-W when covariates / special predictors are
        supplied, equal V otherwise. ``'nested'`` forces the outer V
        optimisation even when only Y lags are used (note: the outer
        problem is then under-identified, per Kaul et al. 2015). Equal
        V reduces to the outcome-only simplex LS estimator.
    standardize_predictors : bool, default True
        Rescale predictors to unit range before the V optimization.
    n_random_starts : int, default 4
        Additional random Dirichlet starts for the outer V optimiser.
    penalization : float, default 0.0
        Ridge penalty on donor weights.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treated_unit: Any,
        treatment_time: Any,
        covariates: Optional[List[str]] = None,
        special_predictors: Optional[List[SpecialPredictor]] = None,
        v_method: str = "auto",
        standardize_predictors: bool = True,
        n_random_starts: int = 4,
        penalization: float = 0.0,
        alpha: float = 0.05,
    ):
        self.data = data
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treated_unit = treated_unit
        self.treatment_time = treatment_time
        self.covariates = covariates or []
        self.special_predictors = special_predictors or []
        self.v_method = v_method
        self.standardize_predictors = standardize_predictors
        self.n_random_starts = n_random_starts
        self.penalization = penalization
        self.alpha = alpha

        self._validate()
        self._prepare_matrices()

    # ------------------------------------------------------------------
    # Validation & data prep
    # ------------------------------------------------------------------

    def _validate(self):
        for col in [self.outcome, self.unit, self.time] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        if self.treated_unit not in self.data[self.unit].values:
            raise ValueError(
                f"Treated unit '{self.treated_unit}' not found in '{self.unit}'"
            )

    def _prepare_matrices(self):
        """Pivot data into (T x J) outcome matrix and build predictor tables."""
        pivot = self.data.pivot_table(
            index=self.time, columns=self.unit, values=self.outcome,
        )

        if self.treated_unit not in pivot.columns:
            raise ValueError(f"Treated unit '{self.treated_unit}' missing after pivot")

        self.times = pivot.index.values
        self.pre_mask = self.times < self.treatment_time
        self.post_mask = self.times >= self.treatment_time

        if self.pre_mask.sum() < 2:
            raise ValueError("Need at least 2 pre-treatment periods")
        if self.post_mask.sum() < 1:
            raise ValueError("Need at least 1 post-treatment period")

        self.Y_treated = pivot[self.treated_unit].values  # (T,)
        donor_cols = [c for c in pivot.columns if c != self.treated_unit]
        self.donor_units = donor_cols
        self.Y_donors = pivot[donor_cols].values  # (T, J)

        # Drop donors with any NaN in pre-period
        pre_donors = self.Y_donors[self.pre_mask]
        valid = ~np.any(np.isnan(pre_donors), axis=0)
        if valid.sum() == 0:
            raise ValueError("No valid donor units (all have NaN in pre-period)")
        self.Y_donors = self.Y_donors[:, valid]
        self.donor_units = [
            self.donor_units[i] for i in range(len(self.donor_units)) if valid[i]
        ]

        # Build predictor table X (K, 1+J)
        all_units = [self.treated_unit] + list(self.donor_units)
        self._predictor_names: List[str] = []
        predictor_rows: List[np.ndarray] = []

        # 1. Simple covariate means over the pre-treatment window
        if self.covariates:
            pre_data = self.data[self.data[self.time] < self.treatment_time]
            for col in self.covariates:
                row = []
                for u in all_units:
                    vals = pre_data.loc[pre_data[self.unit] == u, col]
                    row.append(vals.mean() if len(vals) else np.nan)
                predictor_rows.append(np.asarray(row, dtype=float))
                self._predictor_names.append(f"{col}[mean]")

        # 2. Special predictors (R/Stata-style)
        if self.special_predictors:
            for col, period_spec, op in self.special_predictors:
                if col not in self.data.columns:
                    raise ValueError(
                        f"special_predictor column '{col}' not in data"
                    )
                row = []
                years = self._resolve_period_spec(period_spec)
                spec_df = self.data[self.data[self.time].isin(years)]
                for u in all_units:
                    vals = spec_df.loc[spec_df[self.unit] == u, col]
                    if len(vals) == 0:
                        row.append(np.nan)
                    elif op == "mean":
                        row.append(vals.mean())
                    elif op == "sum":
                        row.append(vals.sum())
                    else:
                        raise ValueError(
                            f"special_predictor op must be 'mean' or 'sum', "
                            f"got '{op}'"
                        )
                predictor_rows.append(np.asarray(row, dtype=float))
                label = (
                    f"{col}[{years[0]}]" if len(years) == 1
                    else f"{col}[{years[0]}-{years[-1]},{op}]"
                )
                self._predictor_names.append(label)

        if predictor_rows:
            X_full = np.vstack(predictor_rows)           # (K, J+1)
            if np.any(~np.isfinite(X_full)):
                raise ValueError(
                    "Predictor matrix contains NaN/Inf — check data coverage."
                )
            self.X_treated = X_full[:, 0]                # (K,)
            self.X_donors = X_full[:, 1:]                # (K, J)
            self._has_predictors = True
        else:
            # No covariates / special predictors → use pre-period Y
            # as predictors.  V optimisation is under-identified here
            # (Kaul et al. 2015), so we'll fix V = I in _solve_weights.
            self.X_treated = self.Y_treated[self.pre_mask].copy()
            self.X_donors = self.Y_donors[self.pre_mask].copy()
            self._predictor_names = [
                f"{self.outcome}[{t}]" for t in self.times[self.pre_mask]
            ]
            self._has_predictors = False

    def _resolve_period_spec(self, period_spec: Any) -> List[Any]:
        """Expand scalar / list / slice specs into a concrete year list."""
        all_times = list(self.times)
        if isinstance(period_spec, slice):
            start = period_spec.start if period_spec.start is not None else all_times[0]
            stop = period_spec.stop if period_spec.stop is not None else all_times[-1]
            return [t for t in all_times if start <= t <= stop]
        if isinstance(period_spec, (list, tuple, np.ndarray)):
            return list(period_spec)
        return [period_spec]

    # ------------------------------------------------------------------
    # Weight optimization
    # ------------------------------------------------------------------

    def _solve_weights(
        self,
        Y_treated_pre: np.ndarray,
        Y_donors_pre: np.ndarray,
        X_treated: np.ndarray,
        X_donors: np.ndarray,
        run_nested: bool,
    ) -> Dict[str, Any]:
        """
        Nested V-W solver.

        * ``run_nested=True`` — full ADH(2010) outer V + inner W loop.
        * ``run_nested=False`` — equal V, solve inner W once.  Used when
          predictors are just the pre-period outcomes (V unidentified,
          per Kaul et al. 2015).

        Returns dict with keys ``w``, ``v``, ``loss``, ``inner_loss``,
        ``scale``, ``n_starts``, ``converged``.
        """
        from ._core import (
            solve_synth_weights_adh,
            _inner_w_given_v,
            standardize_predictors,
        )

        if run_nested:
            return solve_synth_weights_adh(
                X_treated, X_donors, Y_treated_pre, Y_donors_pre,
                standardize=self.standardize_predictors,
                n_random_starts=self.n_random_starts,
                penalization=self.penalization,
            )

        # Equal V path (also used as a fast fallback)
        if self.standardize_predictors:
            X1s, X0s, scale = standardize_predictors(X_treated, X_donors)
        else:
            X1s, X0s = X_treated, X_donors
            scale = np.ones(X_treated.shape[0])
        K = X_treated.shape[0]
        V = np.ones(K)
        w = _inner_w_given_v(V, X1s, X0s, penalization=self.penalization)
        r_outer = Y_treated_pre - Y_donors_pre @ w
        r_inner = X1s - X0s @ w
        return {
            "w": w,
            "v": V,
            "loss": float(r_outer @ r_outer),
            "inner_loss": float(np.sum(V * r_inner ** 2)),
            "scale": scale,
            "n_starts": 1,
            "converged": True,
        }

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _should_run_nested(self) -> bool:
        """Decide whether to run the outer V optimization."""
        if self.v_method == "equal":
            return False
        if self.v_method == "nested":
            return True
        # 'auto': run nested iff predictors come from covariates /
        # special predictors, not purely pre-outcome lags.
        return self._has_predictors

    def fit(self, placebo: bool = True) -> CausalResult:
        """
        Fit the Synthetic Control model.

        Parameters
        ----------
        placebo : bool, default True
            Run in-space placebo tests across all donor units.

        Returns
        -------
        CausalResult
        """
        Y_pre_treated = self.Y_treated[self.pre_mask]
        Y_pre_donors = self.Y_donors[self.pre_mask]

        run_nested = self._should_run_nested()
        solver_out = self._solve_weights(
            Y_pre_treated, Y_pre_donors,
            self.X_treated, self.X_donors,
            run_nested=run_nested,
        )
        weights = solver_out["w"]
        V = solver_out["v"]

        # Synthetic control trajectory
        Y_synth = self.Y_donors @ weights  # (T,)

        # Treatment effect (gap)
        gap = self.Y_treated - Y_synth  # (T,)
        gap_post = gap[self.post_mask]
        gap_pre = gap[self.pre_mask]

        # Average treatment effect on treated (post-period)
        att = float(np.mean(gap_post))

        # Pre-treatment MSPE (fit quality)
        pre_mspe = float(np.mean(gap_pre**2))

        # --- Placebo inference ---
        post_mspe = float(np.mean(gap_post**2))
        ratio_treated = (np.sqrt(post_mspe) / np.sqrt(pre_mspe)
                         if pre_mspe > 1e-10 else np.inf)

        placebo_result: Dict[str, Any] = {}
        if placebo and len(self.donor_units) >= 2:
            placebo_result = self._run_placebos()

        placebo_atts = placebo_result.get('atts', [])

        # P-value from placebo distribution
        if len(placebo_atts) > 0:
            placebo_ratios = np.array(placebo_result['ratios'])

            # One-sided p-value: fraction of placebos with ratio >= treated
            pvalue = float(np.mean(placebo_ratios >= ratio_treated))
            # Ensure at least 1/(J+1) if treated is most extreme
            pvalue = max(pvalue, 1 / (len(placebo_ratios) + 1))

            se = float(np.std(placebo_atts)) if len(placebo_atts) > 1 else 0.0
        else:
            pvalue = np.nan
            se = float(np.std(gap_post)) / max(np.sqrt(len(gap_post)), 1)

        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (att - z_crit * se, att + z_crit * se)

        # --- Weight table ---
        weight_df = pd.DataFrame({
            'unit': self.donor_units,
            'weight': weights,
        }).sort_values('weight', ascending=False).reset_index(drop=True)
        weight_df = weight_df[weight_df['weight'] > 1e-6]

        # --- Gap table ---
        gap_df = pd.DataFrame({
            'time': self.times,
            'treated': self.Y_treated,
            'synthetic': Y_synth,
            'gap': gap,
            'post_treatment': self.post_mask,
        })

        # --- Predictor V table (ADH diagnostic) ---
        v_df = pd.DataFrame({
            'predictor': self._predictor_names,
            'v_weight': V,
        })

        # --- Predictor balance table (treated vs synthetic on matching vars) ---
        X_synth = self.X_donors @ weights
        predictor_balance_df = pd.DataFrame({
            'predictor': self._predictor_names,
            'treated': self.X_treated,
            'synthetic': X_synth,
            'donor_mean': self.X_donors.mean(axis=1),
        })

        # --- Weight-concentration diagnostics ---
        w_active = weights[weights > 1e-6]
        hhi = float(np.sum(weights ** 2))
        n_effective = 1.0 / hhi if hhi > 0 else 0.0

        # --- Model info ---
        model_info: Dict[str, Any] = {
            'n_donors': len(self.donor_units),
            'n_pre_periods': int(self.pre_mask.sum()),
            'n_post_periods': int(self.post_mask.sum()),
            'pre_treatment_mspe': round(pre_mspe, 6),
            'pre_treatment_rmse': round(np.sqrt(pre_mspe), 6),
            'penalization': self.penalization,
            'treatment_time': self.treatment_time,
            'treated_unit': self.treated_unit,
            'weights': weight_df,
            'v_weights': v_df,
            'predictor_balance': predictor_balance_df,
            'gap_table': gap_df,
            'Y_synth': Y_synth,
            'Y_treated': self.Y_treated,
            'times': self.times,
            'v_method': 'nested' if run_nested else 'equal',
            'n_starts': solver_out['n_starts'],
            'converged': solver_out['converged'],
            'n_active_donors': int(len(w_active)),
            'weight_hhi': round(hhi, 4),
            'effective_n_donors': round(n_effective, 2),
        }

        if len(placebo_atts) > 0:
            model_info['placebo_atts'] = placebo_atts
            model_info['placebo_pre_mspes'] = placebo_result['pre_mspes']
            model_info['placebo_ratios'] = placebo_result['ratios']
            model_info['placebo_gaps'] = placebo_result['gaps']
            model_info['placebo_units'] = placebo_result['units']
            model_info['treated_ratio'] = ratio_treated
            model_info['n_placebos'] = len(placebo_atts)

        return CausalResult(
            method='Synthetic Control Method',
            estimand='ATT',
            estimate=att,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=len(self.Y_treated),
            detail=weight_df,
            model_info=model_info,
            _citation_key='synth',
        )

    def _run_placebos(self) -> Dict[str, Any]:
        """
        Run placebo SCM for each donor unit (in-space placebo).

        Placebo predictor matrices are rebuilt per unit so covariates /
        special predictors are re-averaged on the right unit set.

        Returns
        -------
        dict with keys
            atts, pre_mspes, post_mspes, ratios : list
            gaps : np.ndarray (T, n_placebos)
            units : list
        """
        atts: List[float] = []
        pre_mspes: List[float] = []
        post_mspes: List[float] = []
        ratios: List[float] = []
        gap_trajectories: List[np.ndarray] = []
        units: List[Any] = []

        all_units_data = np.column_stack([
            self.Y_treated[:, np.newaxis], self.Y_donors
        ])
        # Placebo predictor matrix: column 0 = treated, 1..J = donors
        X_all = np.column_stack([self.X_treated[:, None], self.X_donors])

        run_nested = self._should_run_nested()

        for i, placebo_unit in enumerate(self.donor_units):
            idx_placebo = i + 1  # treated at column 0
            Y_placebo = all_units_data[:, idx_placebo]
            donor_idx = [
                j for j in range(all_units_data.shape[1]) if j != idx_placebo
            ]
            Y_placebo_donors = all_units_data[:, donor_idx]

            Y_pre_p = Y_placebo[self.pre_mask]
            Y_pre_d = Y_placebo_donors[self.pre_mask]

            # Swap predictor columns accordingly
            X_placebo = X_all[:, idx_placebo]
            X_placebo_donors = X_all[:, donor_idx]
            # When no covariates were given, X = pre-outcome of the
            # placebo unit (which just swapped).
            if not self._has_predictors:
                X_placebo = Y_pre_p
                X_placebo_donors = Y_pre_d

            try:
                sol = self._solve_weights(
                    Y_pre_p, Y_pre_d,
                    X_placebo, X_placebo_donors,
                    run_nested=run_nested,
                )
                w = sol["w"]
                synth_p = Y_placebo_donors @ w
                gap_p = Y_placebo - synth_p

                pre_mspe_p = float(np.mean(gap_p[self.pre_mask]**2))
                post_mspe_p = float(np.mean(gap_p[self.post_mask]**2))
                att_p = float(np.mean(gap_p[self.post_mask]))
                ratio_p = (np.sqrt(post_mspe_p) / np.sqrt(pre_mspe_p)
                           if pre_mspe_p > 1e-10 else 0.0)

                atts.append(att_p)
                pre_mspes.append(pre_mspe_p)
                post_mspes.append(post_mspe_p)
                ratios.append(ratio_p)
                gap_trajectories.append(gap_p)
                units.append(placebo_unit)
            except Exception:
                continue

        gaps = (np.column_stack(gap_trajectories)
                if gap_trajectories else np.empty((len(self.times), 0)))

        return {
            'atts': atts,
            'pre_mspes': pre_mspes,
            'post_mspes': post_mspes,
            'ratios': ratios,
            'gaps': gaps,
            'units': units,
        }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def synthplot(
    result: CausalResult,
    type: str = 'trajectory',
    ax=None,
    figsize: tuple = (10, 7),
    title: Optional[str] = None,
):
    """
    Standard synthetic control plots.

    Parameters
    ----------
    result : CausalResult
        Result from ``synth()`` or ``sdid()``.
    type : str, default 'trajectory'
        Plot type:
        - 'trajectory': treated vs synthetic over time
        - 'gap': treatment effect (gap) over time
        - 'both': two-panel (trajectory + gap)
    ax : matplotlib Axes, optional
        Only used for 'trajectory' or 'gap'. Ignored for 'both'.
    figsize : tuple
    title : str, optional

    Returns
    -------
    (fig, ax) or (fig, axes) for 'both'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    mi = result.model_info
    gap_df = mi.get('gap_table')
    if gap_df is None:
        raise ValueError("No gap_table in model_info. Use synth() result.")

    times = gap_df['time'].values
    treated = gap_df['treated'].values
    synthetic = gap_df['synthetic'].values
    gap = gap_df['gap'].values
    treatment_time = mi.get('treatment_time')
    treated_unit = mi.get('treated_unit', 'Treated')

    if type == 'both':
        fig, axes = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.3),
                                 sharex=True)
        # Top: trajectory
        _trajectory_panel(axes[0], times, treated, synthetic,
                          treatment_time, treated_unit)
        # Bottom: gap
        _gap_panel(axes[1], times, gap, treatment_time)
        fig.suptitle(title or f'Synthetic Control: {treated_unit}',
                     fontsize=14, y=1.01)
        fig.tight_layout()
        return fig, axes

    if type == 'gap':
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        _gap_panel(ax, times, gap, treatment_time)
        ax.set_title(title or f'Gap Plot: {treated_unit}', fontsize=13)
        fig.tight_layout()
        return fig, ax

    # Default: trajectory
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    _trajectory_panel(ax, times, treated, synthetic,
                      treatment_time, treated_unit)
    ax.set_title(title or f'Synthetic Control: {treated_unit}', fontsize=13)
    fig.tight_layout()
    return fig, ax


def _trajectory_panel(ax, times, treated, synthetic,
                      treatment_time, treated_unit):
    ax.plot(times, treated, color='#2C3E50', linewidth=2,
            label=str(treated_unit))
    ax.plot(times, synthetic, color='#E74C3C', linewidth=2,
            linestyle='--', label='Synthetic')
    if treatment_time is not None:
        ax.axvline(x=treatment_time, color='gray', linestyle=':',
                   linewidth=1, alpha=0.7, label='Treatment')
    ax.set_ylabel('Outcome', fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _gap_panel(ax, times, gap, treatment_time):
    ax.plot(times, gap, color='#2C3E50', linewidth=2)
    ax.fill_between(times, 0, gap, alpha=0.15, color='#3498DB')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    if treatment_time is not None:
        ax.axvline(x=treatment_time, color='gray', linestyle=':',
                   linewidth=1, alpha=0.7)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Gap (Treated − Synthetic)', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ------------------------------------------------------------------
# Citation
# ------------------------------------------------------------------

# Add to CausalResult citation registry
CausalResult._CITATIONS['synth'] = (
    "@article{abadie2010synthetic,\n"
    "  title={Synthetic Control Methods for Comparative Case Studies: "
    "Estimating the Effect of California's Tobacco Control Program},\n"
    "  author={Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={105},\n"
    "  number={490},\n"
    "  pages={493--505},\n"
    "  year={2010},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
