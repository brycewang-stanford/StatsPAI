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

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility


def _require_dataframe(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise MethodIncompatibility(
            f"`{name}` must be a pandas DataFrame.",
            diagnostics={name: value.__class__.__name__},
        )
    return value


def _require_column_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodIncompatibility(
            f"`{name}` must be a non-empty column name.",
            diagnostics={name: repr(value)},
        )
    return value


def _require_string_option(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise MethodIncompatibility(
            f"`{name}` must be a string option.",
            diagnostics={name: repr(value)},
        )
    out = value.lower().strip()
    if not out:
        raise MethodIncompatibility(
            f"`{name}` must be a non-empty string option.",
            diagnostics={name: repr(value)},
        )
    return out


def _require_open_unit_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{name}` must be a number in (0, 1).",
            diagnostics={name: repr(value)},
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be a number in (0, 1).",
            diagnostics={name: repr(value)},
        ) from exc
    if not np.isfinite(out) or not 0.0 < out < 1.0:
        raise MethodIncompatibility(
            f"`{name}` must be in (0, 1).",
            diagnostics={name: out},
        )
    return out


def _require_nonnegative_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{name}` must be a non-negative finite number.",
            diagnostics={name: repr(value)},
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be a non-negative finite number.",
            diagnostics={name: repr(value)},
        ) from exc
    if not np.isfinite(out) or out < 0.0:
        raise MethodIncompatibility(
            f"`{name}` must be a non-negative finite number.",
            diagnostics={name: out},
        )
    return out


def _require_int_at_least(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        )
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        ) from exc
    if out != value or out < minimum:
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        )
    return out


def _coerce_column_list(
    columns: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> List[str]:
    if isinstance(columns, str):
        out = [columns]
    else:
        try:
            out = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{name}` must be a column name or list of column names.",
                diagnostics={name: repr(columns)},
            ) from exc
    if not allow_empty and not out:
        raise MethodIncompatibility(
            f"`{name}` must contain at least one column name.",
            diagnostics={name: out},
        )
    bad = [c for c in out if not isinstance(c, str) or not c]
    if bad:
        raise MethodIncompatibility(
            f"`{name}` must contain only non-empty string column names.",
            diagnostics={name: out, "invalid_columns": bad},
        )
    return out


def _coerce_optional_column_list(columns: Any, name: str) -> Optional[List[str]]:
    if columns is None:
        return None
    return _coerce_column_list(columns, name, allow_empty=True)


def synth(
    data: pd.DataFrame,
    outcome: Optional[str] = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    treated_unit: Any = None,
    treatment_time: Any = None,
    method: str = "classic",
    covariates: Optional[List[str]] = None,
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
    inference: Optional[str] = None,
    treatment: Optional[str] = None,
    y: Optional[str] = None,
    **kwargs: Any,
) -> CausalResult:
    """Public ``sp.synth`` entry point — see ``_dispatch_synth_impl`` for
    the full docstring on methods and parameters.

    Thin wrapper around the multi-branch dispatcher that attaches a
    :class:`Provenance` record to the returned result so downstream
    ``replication_pack`` / Quarto appendix / table footers can pick up
    the call (function name, args, data hash) without each individual
    SCM backend having to opt in. The 20-method dispatcher itself
    lives in :func:`_dispatch_synth_impl`.

    References
    ----------
    Abadie, A., Diamond, A. and Hainmueller, J. (2010). Synthetic control
    methods for comparative case studies. *Journal of the American
    Statistical Association*. [@abadie2010synthetic]

    Examples
    --------
    Classic Abadie-Diamond-Hainmueller SCM on the bundled Proposition 99
    panel (39 states x 31 years):

    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> result = sp.synth(df, outcome='packspercapita', unit='state',
    ...                   time='year', treated_unit='California',
    ...                   treatment_time=1989, method='classic')
    >>> round(result.estimate, 1)  # post-1989 ATT, packs per capita
    -18.2

    Switch estimator via ``method=`` (same call signature):

    >>> result = sp.synth(df, outcome='packspercapita', unit='state',
    ...                   time='year', treated_unit='California',
    ...                   treatment_time=1989, method='sdid')

    To fit several SCM variants at once and get a recommendation, use
    :func:`sp.synth_compare` with the same arguments.
    """
    from ..core._param_aliases import resolve_alias

    outcome = resolve_alias("outcome", outcome, "y", y)
    if outcome is None:
        raise TypeError("synth() missing required argument: 'outcome' (alias 'y')")
    if unit is None or time is None:
        raise TypeError("synth() missing required arguments: 'unit' and 'time'")
    _result = _dispatch_synth_impl(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        method=method,
        covariates=covariates,
        penalization=penalization,
        placebo=placebo,
        alpha=alpha,
        inference=inference,
        treatment=treatment,
        **kwargs,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.synth",
            params={
                "outcome": outcome,
                "unit": unit,
                "time": time,
                "treated_unit": treated_unit,
                "treatment_time": treatment_time,
                "method": method,
                "covariates": covariates,
                "penalization": penalization,
                "placebo": placebo,
                "alpha": alpha,
                "inference": inference,
                "treatment": treatment,
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    in (
                        "n_factors",
                        "outcomes",
                        "v_method",
                        "se_method",
                        "se_type",
                        "weights",
                        "l2_penalty",
                    )
                },
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover — provenance must never break fit
        pass
    return _result


def _dispatch_synth_impl(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any = None,
    treatment_time: Any = None,
    method: str = "classic",
    covariates: Optional[List[str]] = None,
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
    inference: Optional[str] = None,
    treatment: Optional[str] = None,
    **kwargs: Any,
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
    data = _require_dataframe(data, "data")
    outcome = _require_column_name(outcome, "outcome")
    unit = _require_column_name(unit, "unit")
    time = _require_column_name(time, "time")
    method = _require_string_option(method, "method")
    covariates = _coerce_optional_column_list(covariates, "covariates")
    penalization = _require_nonnegative_float(penalization, "penalization")
    alpha = _require_open_unit_float(alpha, "alpha")
    if inference is not None:
        inference = _require_string_option(inference, "inference")

    # --- Conformal inference override ---
    if inference == "conformal":
        from .conformal import conformal_synth

        return conformal_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            scm_method=method if method in ("classic", "ridge") else "classic",
            penalization=penalization,
            alpha=alpha,
            **kwargs,
        )

    # --- Dispatch ---
    if method in ("classic", "classic_adh", "adh", "penalized", "ridge"):
        backend = kwargs.pop("backend", "native")
        backend_norm = _require_string_option(backend, "backend").replace("-", "_")
        if backend_norm in {"synth", "r", "synth_r"}:
            if method not in ("classic", "classic_adh", "adh"):
                raise NotImplementedError(
                    "The Synth R reference backend is only available for "
                    "method='classic'. Use backend='native' for penalized SCM."
                )
            if covariates:
                raise NotImplementedError(
                    "The Synth R reference backend currently supports the "
                    "outcome-lag specification used in the parity harness; "
                    "use backend='native' for covariates."
                )
            if kwargs.get("special_predictors") is not None:
                raise NotImplementedError(
                    "The Synth R reference backend constructs the standard "
                    "pre-outcome special predictors automatically; use "
                    "backend='native' for custom special_predictors."
                )
            return _synth_r_backend(
                data=data,
                outcome=outcome,
                unit=unit,
                time=time,
                treated_unit=treated_unit,
                treatment_time=treatment_time,
                alpha=alpha,
            )
        if backend_norm != "native":
            raise MethodIncompatibility(
                "Unknown backend. Use 'native', 'synth', or 'r'.",
                diagnostics={"backend": backend},
                alternative_functions=["sp.synth"],
            )

        if method in ("penalized", "ridge") and penalization == 0.0:
            penalization = _require_nonnegative_float(
                kwargs.pop("l2_penalty", 0.01), "l2_penalty"
            )
        special_predictors = kwargs.pop("special_predictors", None)
        v_method = kwargs.pop("v_method", "auto")
        standardize_predictors = kwargs.pop("standardize_predictors", True)
        n_random_starts = kwargs.pop("n_random_starts", 4)
        model = SyntheticControl(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            special_predictors=special_predictors,
            v_method=v_method,
            standardize_predictors=standardize_predictors,
            n_random_starts=n_random_starts,
            penalization=penalization,
            alpha=alpha,
        )
        return model.fit(placebo=placebo)

    if method in ("demeaned", "detrended"):
        from .demeaned import demeaned_synth

        return demeaned_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            variant=cast(Literal["demeaned", "detrended"], method),
            penalization=penalization,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method in ("unconstrained", "elastic_net"):
        from .robust import robust_synth

        return robust_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            variant=cast(Literal["unconstrained", "elastic_net"], method),
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method in ("augmented", "ascm"):
        from .augsynth import augsynth

        return augsynth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            alpha=alpha,
            **kwargs,
        )

    if method == "sdid":
        from .sdid import sdid as _sdid

        se_method = cast(
            Literal["placebo", "bootstrap", "jackknife"],
            inference or "placebo",
        )
        return _sdid(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            method="sdid",
            covariates=covariates,
            se_method=se_method,
            alpha=alpha,
            **kwargs,
        )

    if method in ("factor", "gsynth"):
        from .gsynth import gsynth as _gsynth

        return _gsynth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method == "staggered":
        from .staggered import staggered_synth

        if treatment is None:
            raise MethodIncompatibility(
                "method='staggered' requires the `treatment` parameter "
                "(binary treatment indicator column name)",
                diagnostics={"method": method, "treatment": treatment},
                alternative_functions=["sp.staggered_synth", "sp.synth"],
            )
        treatment = _require_column_name(treatment, "treatment")
        return staggered_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treatment=treatment,
            penalization=penalization,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method in ("mc", "matrix_completion"):
        from .mc import mc_synth

        return mc_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            alpha=alpha,
            placebo=placebo,
            **kwargs,
        )

    if method in ("discos", "distributional"):
        from .discos import discos

        return discos(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            alpha=alpha,
            placebo=placebo,
            **kwargs,
        )

    if method in ("multi_outcome", "multi"):
        from .multi_outcome import multi_outcome_synth

        outcomes = kwargs.pop("outcomes", None)
        if outcomes is None:
            outcomes = [outcome]
        return multi_outcome_synth(
            data=data,
            outcomes=outcomes,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            penalization=penalization,
            alpha=alpha,
            placebo=placebo,
            **kwargs,
        )

    if method in ("scpi", "prediction_interval"):
        from .scpi import scpi

        return scpi(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            alpha=alpha,
            **kwargs,
        )

    # --- New methods (v0.9) ---

    if method == "bayesian":
        from .bayesian import bayesian_synth

        return bayesian_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            alpha=alpha,
            **kwargs,
        )

    if method in ("bsts", "causal_impact"):
        from .bsts import bsts_synth

        return bsts_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            alpha=alpha,
            **kwargs,
        )

    if method in ("penscm", "abadie_lhour", "pairwise"):
        from .penscm import penalized_synth

        return penalized_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method in ("fdid", "forward_did"):
        from .fdid import fdid as _fdid

        return _fdid(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method == "cluster":
        from .cluster import cluster_synth

        return cluster_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method in ("sparse", "lasso"):
        from .sparse import sparse_synth

        return sparse_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method == "kernel":
        from .kernel import kernel_synth

        return kernel_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    if method == "kernel_ridge":
        from .kernel import kernel_ridge_synth

        return kernel_ridge_synth(
            data=data,
            outcome=outcome,
            unit=unit,
            time=time,
            treated_unit=treated_unit,
            treatment_time=treatment_time,
            covariates=covariates,
            placebo=placebo,
            alpha=alpha,
            **kwargs,
        )

    raise MethodIncompatibility(
        f"Unknown method {method!r}. Choose from: 'classic', 'penalized', "
        f"'ridge', 'demeaned', 'detrended', 'unconstrained', 'elastic_net', "
        f"'augmented', 'ascm', 'sdid', 'factor', 'gsynth', 'staggered', "
        f"'mc', 'discos', 'multi_outcome', 'scpi', "
        f"'bayesian', 'bsts', 'causal_impact', 'penscm', 'abadie_lhour', "
        f"'fdid', 'forward_did', 'cluster', 'sparse', 'lasso', "
        f"'kernel', 'kernel_ridge'.",
        diagnostics={"method": method},
        alternative_functions=["sp.synth_compare", "sp.synth"],
    )


SpecialPredictor = Tuple[str, Any, str]  # (col, period_spec, op)


def _find_rscript() -> str:
    """Return a usable Rscript executable, including common macOS paths."""
    candidates = [
        shutil.which("Rscript"),
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise ConvergenceFailure(
        "The Synth backend requires Rscript plus the R packages "
        "'Synth' and 'jsonlite'. Install R or use backend='native'.",
        recovery_hint="Install Rscript and Synth/jsonlite, or set backend='native'.",
        alternative_functions=["sp.synth"],
    )


def _synth_r_backend(
    *,
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    alpha: float,
) -> CausalResult:
    """Delegate classical SCM estimation to R ``Synth`` for parity."""
    if treated_unit is None or treatment_time is None:
        raise MethodIncompatibility(
            "treated_unit and treatment_time are required",
            diagnostics={
                "treated_unit": treated_unit,
                "treatment_time": treatment_time,
            },
        )
    if not pd.api.types.is_numeric_dtype(data[time]):
        raise MethodIncompatibility(
            "The Synth R backend requires a numeric time column so it can "
            "construct pre-treatment special predictors.",
            diagnostics={"time": time, "dtype": str(data[time].dtype)},
            alternative_functions=["sp.synth"],
        )

    unit_col = "statspai_unit"
    time_col = "statspai_time"
    outcome_col = "statspai_outcome"
    panel_df = (
        pd.DataFrame(
            {
                unit_col: data[unit],
                time_col: data[time],
                outcome_col: data[outcome],
            }
        )
        .sort_values([unit_col, time_col])
        .reset_index(drop=True)
    )

    r_script = r"""
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("expected 4 arguments: input output treated_unit treatment_time")
}
input_path <- args[[1]]
output_path <- args[[2]]
treated_unit <- args[[3]]
treatment_time <- as.numeric(args[[4]])

suppressPackageStartupMessages({
  library(Synth)
  library(jsonlite)
})

df <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
df$unit_num <- as.integer(as.factor(df$statspai_unit))
units <- unique(df[, c("statspai_unit", "unit_num")])
units <- units[order(units$unit_num), ]
treated_id <- units$unit_num[units$statspai_unit == treated_unit]
controls <- setdiff(units$unit_num, treated_id)

pre_years <- sort(unique(df$statspai_time[df$statspai_time < treatment_time]))
post_years <- sort(unique(df$statspai_time[df$statspai_time >= treatment_time]))
special_preds <- lapply(pre_years, function(yr) {
  list("statspai_outcome", yr, "mean")
})

dp <- Synth::dataprep(
  foo = df,
  predictors = NULL,
  predictors.op = "mean",
  dependent = "statspai_outcome",
  unit.variable = "unit_num",
  time.variable = "statspai_time",
  special.predictors = special_preds,
  treatment.identifier = treated_id,
  controls.identifier = controls,
  time.predictors.prior = pre_years,
  time.optimize.ssr = pre_years,
  time.plot = c(pre_years, post_years),
  unit.names.variable = "statspai_unit"
)

sy <- Synth::synth(data.prep.obj = dp, optimxmethod = "BFGS", verbose = FALSE)
w_unit <- as.numeric(sy$solution.w)
donor_names <- units$statspai_unit[units$unit_num %in% controls]
names(w_unit) <- donor_names

Y0_synth <- dp$Y0plot %*% sy$solution.w
Y1_treat <- dp$Y1plot
gap <- Y1_treat - Y0_synth
post_idx <- which(rownames(Y1_treat) %in% as.character(post_years))
pre_idx <- which(rownames(Y1_treat) %in% as.character(pre_years))
avg_post_gap <- mean(gap[post_idx])
pre_rmse <- sqrt(mean(gap[pre_idx]^2))

weights <- lapply(sort(donor_names), function(nm) {
  list(unit = nm, weight = unname(w_unit[nm]))
})

payload <- list(
  estimate = as.numeric(avg_post_gap),
  pre_rmse = as.numeric(pre_rmse),
  n_obs = nrow(df),
  n_time_periods = length(unique(df$statspai_time)),
  n_donors = length(controls),
  n_pre_periods = length(pre_years),
  n_post_periods = length(post_years),
  weights = weights
)
jsonlite::write_json(
  payload,
  output_path,
  auto_unbox = TRUE,
  null = "null",
  na = "null",
  digits = 16
)
"""

    rscript = _find_rscript()
    with tempfile.TemporaryDirectory(prefix="statspai_synth_") as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "panel.csv"
        script_path = tmp_path / "synth_backend.R"
        out_path = tmp_path / "result.json"
        panel_df.to_csv(data_path, index=False)
        script_path.write_text(r_script, encoding="utf-8")

        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(data_path),
                str(out_path),
                str(treated_unit),
                str(float(treatment_time)),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise ConvergenceFailure(
                "R Synth backend failed. stderr:\n" f"{proc.stderr.strip()}",
                recovery_hint=(
                    "Check the R Synth/jsonlite installation or rerun with "
                    "backend='native'."
                ),
                diagnostics={"returncode": proc.returncode},
                alternative_functions=["sp.synth"],
            )
        payload = json.loads(out_path.read_text(encoding="utf-8"))

    weight_df = (
        pd.DataFrame(payload["weights"])
        .sort_values("weight", ascending=False)
        .reset_index(drop=True)
    )
    pre_rmse = float(payload["pre_rmse"])
    model_info = {
        "backend": "synth",
        "r_package": "Synth",
        "validation_tier": "reference_backend_bridge",
        "reference_backend": "Synth",
        "validation_note": (
            "This result delegates to Synth::dataprep and Synth::synth. It "
            "is useful for exact reference-package numbers, but it is not "
            "counted as native Python parity evidence because the reference "
            "backend is the estimator itself."
        ),
        "optimxmethod": "BFGS",
        "n_donors": int(payload["n_donors"]),
        "n_pre_periods": int(payload["n_pre_periods"]),
        "n_post_periods": int(payload["n_post_periods"]),
        "pre_treatment_rmse": pre_rmse,
        "pre_treatment_mspe": pre_rmse**2,
        "treatment_time": treatment_time,
        "treated_unit": treated_unit,
        "weights": weight_df,
    }

    return CausalResult(
        method="Synthetic Control Method (R Synth reference backend)",
        estimand="ATT",
        estimate=float(payload["estimate"]),
        se=float("nan"),
        pvalue=float("nan"),
        ci=(float("nan"), float("nan")),
        alpha=alpha,
        n_obs=int(payload["n_time_periods"]),
        detail=weight_df,
        model_info=model_info,
        _citation_key="synth",
    )


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

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.california_prop99()
    >>> sc = sp.SyntheticControl(
    ...     df, outcome='packspercapita', unit='state', time='year',
    ...     treated_unit='California', treatment_time=1989,
    ... )
    >>> res = sc.fit(placebo=False)
    >>> res.method
    'Synthetic Control Method'
    >>> bool(res.estimate < 0)  # cigarette sales fell after Prop 99
    True
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
        self.data = _require_dataframe(data, "data")
        self.outcome = _require_column_name(outcome, "outcome")
        self.unit = _require_column_name(unit, "unit")
        self.time = _require_column_name(time, "time")
        if treated_unit is None or treatment_time is None:
            raise MethodIncompatibility(
                "treated_unit and treatment_time are required",
                diagnostics={
                    "treated_unit": treated_unit,
                    "treatment_time": treatment_time,
                },
                alternative_functions=["sp.synth"],
            )
        self.treated_unit = treated_unit
        self.treatment_time = treatment_time
        self.covariates = _coerce_optional_column_list(covariates, "covariates") or []
        self.special_predictors = special_predictors or []
        self.v_method = _require_string_option(v_method, "v_method")
        if self.v_method not in {"auto", "equal", "nested"}:
            raise MethodIncompatibility(
                "v_method must be one of 'auto', 'equal', or 'nested'.",
                diagnostics={"v_method": v_method},
            )
        if not isinstance(standardize_predictors, (bool, np.bool_)):
            raise MethodIncompatibility(
                "standardize_predictors must be True or False.",
                diagnostics={"standardize_predictors": repr(standardize_predictors)},
            )
        self.standardize_predictors = bool(standardize_predictors)
        self.n_random_starts = _require_int_at_least(
            n_random_starts, "n_random_starts", 0
        )
        self.penalization = _require_nonnegative_float(penalization, "penalization")
        self.alpha = _require_open_unit_float(alpha, "alpha")

        self._validate()
        self._prepare_matrices()

    # ------------------------------------------------------------------
    # Validation & data prep
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        for col in [self.outcome, self.unit, self.time] + self.covariates:
            if col not in self.data.columns:
                raise MethodIncompatibility(
                    f"Column '{col}' not found in data",
                    diagnostics={
                        "missing_column": col,
                        "available_columns": list(self.data.columns),
                    },
                )
        if self.treated_unit not in self.data[self.unit].values:
            raise DataInsufficient(
                f"Treated unit '{self.treated_unit}' not found in '{self.unit}'",
                recovery_hint="Check treated_unit or filter the data to include it.",
                diagnostics={
                    "treated_unit": self.treated_unit,
                    "unit": self.unit,
                },
            )

    def _prepare_matrices(self) -> None:
        """Pivot data into (T x J) outcome matrix and build predictor tables."""
        pivot = self.data.pivot_table(
            index=self.time,
            columns=self.unit,
            values=self.outcome,
        )

        if self.treated_unit not in pivot.columns:
            raise DataInsufficient(
                f"Treated unit '{self.treated_unit}' missing after pivot",
                diagnostics={"treated_unit": self.treated_unit},
            )

        self.times = pivot.index.values
        try:
            self.pre_mask = self.times < self.treatment_time
            self.post_mask = self.times >= self.treatment_time
        except TypeError as exc:
            raise MethodIncompatibility(
                "treatment_time must be comparable with the time column.",
                diagnostics={
                    "treatment_time": repr(self.treatment_time),
                    "time": self.time,
                },
            ) from exc

        if self.pre_mask.sum() < 2:
            raise DataInsufficient(
                "Need at least 2 pre-treatment periods",
                recovery_hint="Choose an earlier treatment_time or add pre-period rows.",
                diagnostics={"n_pre_periods": int(self.pre_mask.sum())},
            )
        if self.post_mask.sum() < 1:
            raise DataInsufficient(
                "Need at least 1 post-treatment period",
                recovery_hint="Choose a later treatment_time or add post-period rows.",
                diagnostics={"n_post_periods": int(self.post_mask.sum())},
            )

        self.Y_treated = pivot[self.treated_unit].values  # (T,)
        donor_cols = [c for c in pivot.columns if c != self.treated_unit]
        self.donor_units = donor_cols
        self.Y_donors = pivot[donor_cols].values  # (T, J)

        # Drop donors with any NaN in pre-period
        pre_donors = self.Y_donors[self.pre_mask]
        valid = ~np.any(np.isnan(pre_donors), axis=0)
        if valid.sum() == 0:
            raise DataInsufficient(
                "No valid donor units (all have NaN in pre-period)",
                recovery_hint="Add donor units with observed pre-treatment outcomes.",
                diagnostics={"n_donors": len(self.donor_units)},
            )
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
                    raise MethodIncompatibility(
                        f"special_predictor column '{col}' not in data",
                        diagnostics={
                            "missing_column": col,
                            "available_columns": list(self.data.columns),
                        },
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
                        raise MethodIncompatibility(
                            f"special_predictor op must be 'mean' or 'sum', "
                            f"got '{op}'",
                            diagnostics={"op": op},
                        )
                predictor_rows.append(np.asarray(row, dtype=float))
                label = (
                    f"{col}[{years[0]}]"
                    if len(years) == 1
                    else f"{col}[{years[0]}-{years[-1]},{op}]"
                )
                self._predictor_names.append(label)

        if predictor_rows:
            X_full = np.vstack(predictor_rows)  # (K, J+1)
            if np.any(~np.isfinite(X_full)):
                raise DataInsufficient(  # pragma: no cover
                    "Predictor matrix contains NaN/Inf — check data coverage.",
                    recovery_hint=(
                        "Add complete pre-treatment covariate or special "
                        "predictor coverage for the treated and donor units."
                    ),
                )
            self.X_treated = X_full[:, 0]  # (K,)
            self.X_donors = X_full[:, 1:]  # (K, J)
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
            _inner_w_given_v,
            solve_synth_weights_adh,
            standardize_predictors,
        )

        if run_nested:
            return solve_synth_weights_adh(
                X_treated,
                X_donors,
                Y_treated_pre,
                Y_donors_pre,
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
        loss = float(r_outer @ r_outer)
        inner_loss = float(np.sum(V * r_inner**2))
        return {
            "w": w,
            "v": V,
            "loss": loss,
            "inner_loss": inner_loss,
            "scale": scale,
            "n_starts": 1,
            "converged": True,
            "best_start": "equal",
            "start_diagnostics": [
                {
                    "start": "equal",
                    "success": True,
                    "loss": loss,
                    "inner_loss": inner_loss,
                    "weights": w,
                    "v": V,
                }
            ],
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
        return bool(self._has_predictors)

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
            Y_pre_treated,
            Y_pre_donors,
            self.X_treated,
            self.X_donors,
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
        ratio_treated = (
            np.sqrt(post_mspe) / np.sqrt(pre_mspe) if pre_mspe > 1e-10 else np.inf
        )

        placebo_result: Dict[str, Any] = {}
        if placebo and len(self.donor_units) >= 2:
            placebo_result = self._run_placebos()

        placebo_atts = placebo_result.get("atts", [])

        # P-value from placebo distribution
        if len(placebo_atts) > 0:
            placebo_ratios = np.array(placebo_result["ratios"])

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
        weight_df = (
            pd.DataFrame(
                {
                    "unit": self.donor_units,
                    "weight": weights,
                }
            )
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )
        weight_df = weight_df[weight_df["weight"] > 1e-6]

        # --- Gap table ---
        gap_df = pd.DataFrame(
            {
                "time": self.times,
                "treated": self.Y_treated,
                "synthetic": Y_synth,
                "gap": gap,
                "post_treatment": self.post_mask,
            }
        )

        # --- Predictor V table (ADH diagnostic) ---
        v_df = pd.DataFrame(
            {
                "predictor": self._predictor_names,
                "v_weight": V,
            }
        )

        # --- Predictor balance table (treated vs synthetic on matching vars) ---
        X_synth = self.X_donors @ weights
        predictor_balance_df = pd.DataFrame(
            {
                "predictor": self._predictor_names,
                "treated": self.X_treated,
                "synthetic": X_synth,
                "donor_mean": self.X_donors.mean(axis=1),
            }
        )

        # --- Weight-concentration diagnostics ---
        w_active = weights[weights > 1e-6]
        hhi = float(np.sum(weights**2))
        n_effective = 1.0 / hhi if hhi > 0 else 0.0

        # --- Multi-start solver diagnostics ---
        start_diagnostics = solver_out.get("start_diagnostics", [])
        start_diag_rows = []
        start_weight_rows = []
        start_v_rows = []
        finite_losses = [
            float(item["loss"])
            for item in start_diagnostics
            if np.isfinite(float(item.get("loss", np.inf)))
        ]
        best_start_loss = min(finite_losses) if finite_losses else np.nan
        loss_tol = max(1e-6, abs(best_start_loss) * 1e-4) if finite_losses else np.nan
        weight_class_l1_tol = 1e-4
        near_best_weights = []
        for item in start_diagnostics:
            loss = float(item.get("loss", np.inf))
            start_name = str(item.get("start", ""))
            start_weights = np.asarray(item.get("weights", []), dtype=float)
            start_v = np.asarray(item.get("v", []), dtype=float)
            near_best = bool(np.isfinite(loss) and loss <= best_start_loss + loss_tol)
            if near_best and start_weights.size == len(self.donor_units):
                near_best_weights.append(start_weights)
            start_diag_rows.append(
                {
                    "start": start_name,
                    "success": bool(item.get("success", False)),
                    "loss": loss,
                    "inner_loss": float(item.get("inner_loss", np.nan)),
                    "near_best": near_best,
                    "n_active_donors": (
                        int(np.sum(start_weights > 1e-6)) if start_weights.size else 0
                    ),
                    "weight_hhi": (
                        float(np.nansum(start_weights**2))
                        if start_weights.size
                        else np.nan
                    ),
                }
            )
            if start_weights.size == len(self.donor_units):
                for donor, weight in zip(self.donor_units, start_weights):
                    start_weight_rows.append(
                        {
                            "start": start_name,
                            "unit": donor,
                            "weight": float(weight),
                            "loss": loss,
                            "near_best": near_best,
                        }
                    )
            if start_v.size == len(self._predictor_names):
                for predictor, v_weight in zip(self._predictor_names, start_v):
                    start_v_rows.append(
                        {
                            "start": start_name,
                            "predictor": predictor,
                            "v_weight": float(v_weight),
                            "loss": loss,
                            "near_best": near_best,
                        }
                    )

        solver_start_diagnostics = pd.DataFrame(start_diag_rows)
        solver_start_weights = pd.DataFrame(start_weight_rows)
        solver_start_v_weights = pd.DataFrame(start_v_rows)
        near_best_weight_l1_max = 0.0
        weight_class_reps: list[np.ndarray] = []
        for candidate in near_best_weights:
            if weight_class_reps:
                near_best_weight_l1_max = max(
                    near_best_weight_l1_max,
                    max(
                        float(np.sum(np.abs(candidate - rep)))
                        for rep in weight_class_reps
                    ),
                )
            if not any(
                float(np.sum(np.abs(candidate - rep))) <= weight_class_l1_tol
                for rep in weight_class_reps
            ):
                weight_class_reps.append(candidate)

        # --- Model info ---
        model_info: Dict[str, Any] = {
            "backend": "native",
            "validation_tier": "identification_dependent_native",
            "reference_backend": "Synth",
            "validation_note": (
                "Native classical SCM is certified on uniquely identified "
                "synthetic-control DGPs. Empirical applications with "
                "non-unique V/W solutions, including the Basque parity row, "
                "are treated as T4 non-uniqueness disclosures; use "
                "backend='synth' when exact R Synth numbers are required."
            ),
            "n_donors": len(self.donor_units),
            "n_pre_periods": int(self.pre_mask.sum()),
            "n_post_periods": int(self.post_mask.sum()),
            "pre_treatment_mspe": round(pre_mspe, 6),
            "pre_treatment_rmse": round(np.sqrt(pre_mspe), 6),
            "penalization": self.penalization,
            "treatment_time": self.treatment_time,
            "treated_unit": self.treated_unit,
            "weights": weight_df,
            "v_weights": v_df,
            "predictor_balance": predictor_balance_df,
            "gap_table": gap_df,
            "Y_synth": Y_synth,
            "Y_treated": self.Y_treated,
            "times": self.times,
            "v_method": "nested" if run_nested else "equal",
            "n_starts": solver_out["n_starts"],
            "converged": solver_out["converged"],
            "solver_best_start": solver_out.get("best_start"),
            "solver_start_diagnostics": solver_start_diagnostics,
            "solver_start_weights": solver_start_weights,
            "solver_start_v_weights": solver_start_v_weights,
            "solver_best_loss": best_start_loss,
            "solver_near_best_loss_tol": loss_tol,
            "solver_near_best_start_count": int(len(near_best_weights)),
            "solver_weight_class_l1_tol": weight_class_l1_tol,
            "solver_near_best_weight_l1_max": round(near_best_weight_l1_max, 6),
            "solver_near_best_weight_class_count": int(len(weight_class_reps)),
            "weight_solution_nonunique": bool(
                len(weight_class_reps) > 1 and near_best_weight_l1_max > 1e-3
            ),
            "n_active_donors": int(len(w_active)),
            "weight_hhi": round(hhi, 4),
            "effective_n_donors": round(n_effective, 2),
        }

        if len(placebo_atts) > 0:
            model_info["placebo_atts"] = placebo_atts
            model_info["placebo_pre_mspes"] = placebo_result["pre_mspes"]
            model_info["placebo_ratios"] = placebo_result["ratios"]
            model_info["placebo_gaps"] = placebo_result["gaps"]
            model_info["placebo_units"] = placebo_result["units"]
            model_info["treated_ratio"] = ratio_treated
            model_info["n_placebos"] = len(placebo_atts)

        return CausalResult(
            method="Synthetic Control Method",
            estimand="ATT",
            estimate=att,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=len(self.Y_treated),
            detail=weight_df,
            model_info=model_info,
            _citation_key="synth",
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

        all_units_data = np.column_stack([self.Y_treated[:, np.newaxis], self.Y_donors])
        # Placebo predictor matrix: column 0 = treated, 1..J = donors
        X_all = np.column_stack([self.X_treated[:, None], self.X_donors])

        run_nested = self._should_run_nested()

        for i, placebo_unit in enumerate(self.donor_units):
            idx_placebo = i + 1  # treated at column 0
            Y_placebo = all_units_data[:, idx_placebo]
            donor_idx = [j for j in range(all_units_data.shape[1]) if j != idx_placebo]
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
                    Y_pre_p,
                    Y_pre_d,
                    X_placebo,
                    X_placebo_donors,
                    run_nested=run_nested,
                )
                w = sol["w"]
                synth_p = Y_placebo_donors @ w
                gap_p = Y_placebo - synth_p

                pre_mspe_p = float(np.mean(gap_p[self.pre_mask] ** 2))
                post_mspe_p = float(np.mean(gap_p[self.post_mask] ** 2))
                att_p = float(np.mean(gap_p[self.post_mask]))
                ratio_p = (
                    np.sqrt(post_mspe_p) / np.sqrt(pre_mspe_p)
                    if pre_mspe_p > 1e-10
                    else 0.0
                )

                atts.append(att_p)
                pre_mspes.append(pre_mspe_p)
                post_mspes.append(post_mspe_p)
                ratios.append(ratio_p)
                gap_trajectories.append(gap_p)
                units.append(placebo_unit)
            except Exception:
                continue

        gaps = (
            np.column_stack(gap_trajectories)
            if gap_trajectories
            else np.empty((len(self.times), 0))
        )

        return {
            "atts": atts,
            "pre_mspes": pre_mspes,
            "post_mspes": post_mspes,
            "ratios": ratios,
            "gaps": gaps,
            "units": units,
        }


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def synthplot(
    result: CausalResult,
    type: str = "trajectory",
    ax: Any = None,
    figsize: tuple[float, float] = (10, 7),
    title: Optional[str] = None,
) -> tuple[Any, Any]:
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
    gap_df = mi.get("gap_table")
    if gap_df is None:
        raise MethodIncompatibility(
            "No gap_table in model_info. Use synth() result.",
            alternative_functions=["sp.synth"],
        )

    times = gap_df["time"].values
    treated = gap_df["treated"].values
    synthetic = gap_df["synthetic"].values
    gap = gap_df["gap"].values
    treatment_time = mi.get("treatment_time")
    treated_unit = mi.get("treated_unit", "Treated")

    if type == "both":
        fig, axes = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 1.3), sharex=True
        )
        # Top: trajectory
        _trajectory_panel(
            axes[0], times, treated, synthetic, treatment_time, treated_unit
        )
        # Bottom: gap
        _gap_panel(axes[1], times, gap, treatment_time)
        fig.suptitle(title or f"Synthetic Control: {treated_unit}", fontsize=14, y=1.01)
        fig.tight_layout()
        return fig, axes

    if type == "gap":
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()
        _gap_panel(ax, times, gap, treatment_time)
        ax.set_title(title or f"Gap Plot: {treated_unit}", fontsize=13)
        fig.tight_layout()
        return fig, ax

    # Default: trajectory
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    _trajectory_panel(ax, times, treated, synthetic, treatment_time, treated_unit)
    ax.set_title(title or f"Synthetic Control: {treated_unit}", fontsize=13)
    fig.tight_layout()
    return fig, ax


def _trajectory_panel(
    ax: Any,
    times: Any,
    treated: Any,
    synthetic: Any,
    treatment_time: Any,
    treated_unit: Any,
) -> None:
    ax.plot(times, treated, color="#2C3E50", linewidth=2, label=str(treated_unit))
    ax.plot(
        times,
        synthetic,
        color="#E74C3C",
        linewidth=2,
        linestyle="--",
        label="Synthetic",
    )
    if treatment_time is not None:
        ax.axvline(
            x=treatment_time,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.7,
            label="Treatment",
        )
    ax.set_ylabel("Outcome", fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _gap_panel(ax: Any, times: Any, gap: Any, treatment_time: Any) -> None:
    ax.plot(times, gap, color="#2C3E50", linewidth=2)
    ax.fill_between(times, 0, gap, alpha=0.15, color="#3498DB")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    if treatment_time is not None:
        ax.axvline(
            x=treatment_time, color="gray", linestyle=":", linewidth=1, alpha=0.7
        )
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Gap (Treated - Synthetic)", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ------------------------------------------------------------------
# Citation
# ------------------------------------------------------------------

# Add to CausalResult citation registry
CausalResult._CITATIONS["synth"] = (
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
