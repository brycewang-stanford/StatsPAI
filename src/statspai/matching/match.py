"""
Matching estimators for observational causal inference.

Unified interface supporting orthogonal design choices:

- **distance**: how to measure unit similarity
  - ``'propensity'`` — logit propensity score (Rosenbaum & Rubin 1983)
  - ``'mahalanobis'`` — Mahalanobis distance (Rubin 1980)
  - ``'euclidean'`` — normalized Euclidean distance
  - ``'exact'`` — exact covariate values (no approximation)

- **method**: how to use those distances
  - ``'nearest'`` — k-nearest-neighbor matching
  - ``'stratify'`` — subclassification / stratification
  - ``'cem'`` — coarsened exact matching (Iacus, King & Porro 2012)

- **bias_correction**: Abadie-Imbens (2011) regression adjustment for
  matching discrepancies in nearest-neighbor matching.

Backward compatible: ``method='psm'``, ``method='mahalanobis'``, and
``method='cem'`` still work and map to the new parameter space.

References
----------
Rosenbaum, P.R. and Rubin, D.B. (1983). Biometrika, 70(1), 41-55.
Abadie, A. and Imbens, G.W. (2006). Econometrica, 74(1), 235-267.
Abadie, A. and Imbens, G.W. (2011). JBES, 29(1), 1-11.
Iacus, S.M., King, G., and Porro, G. (2012). Political Analysis, 20(1), 1-24.
King, G. and Nielsen, R. (2019). Political Analysis, 27(4), 435-454.
Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University Press.
    Ch. 5: Matching and Subclassification. https://mixtape.scunning.com/
    [@rosenbaum1983central]
"""

import operator
import warnings
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._matched_frame import (
    COL_WEIGHT,
    abadie_imbens_se,
    attach_matched_frame,
    build_ate_matched_frame,
    build_matched_frame,
    build_stratum_matched_frame,
    common_support_mask,
    matched_columns,
    psmatch2_se,
)
from ._stata_lpoly import psmatch2_llr_smoothed_outcome

# ======================================================================
# Legacy method aliases → (distance, method) pairs
# ======================================================================
_LEGACY_MAP = {
    "psm": ("propensity", "nearest"),
    "mahalanobis": ("mahalanobis", "nearest"),
    "cem": (None, "cem"),
}

_VALID_DISTANCES = ("propensity", "mahalanobis", "euclidean", "exact")
_VALID_METHODS = ("nearest", "stratify", "cem", "kernel", "radius", "llr")

#: Below this weighted variance of the propensity gaps the local linear
#: design is collinear and no slope is identified; the treated unit is
#: dropped off support, mirroring psmatch2's `cap reg` failure branch.
_LLR_MIN_VARIANCE = 1e-24

# Kernel functions K(u) used by kernel / radius matching, matching the
# definitions in Stata psmatch2.ado (Leuven & Sianesi 2003).  Each returns
# the (un-normalised) weight for |u| <= 1, and 0 outside the bandwidth
# (the 'normal' kernel has unbounded support).  The leading constants
# (e.g. 0.75 for Epanechnikov) cancel under the per-treated normalisation,
# so psmatch2 omits them — we follow suit for digit-for-digit parity.
_VALID_KERNELS = ("epan", "normal", "biweight", "uniform", "tricube")


def _positive_int(value: Any, *, name: str, context: str) -> int:
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a positive integer"
        ) from exc
    if isinstance(value, bool) or parsed < 1:
        raise MethodIncompatibility(f"{context}: {name} must be a positive integer")
    return int(parsed)


def _open_unit_float(value: Any, *, name: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        ) from exc
    if not np.isfinite(parsed) or not (0.0 < parsed < 1.0):
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and in the open interval (0, 1)"
        )
    return parsed


def _positive_float(value: Any, *, name: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be finite and positive"
        ) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise MethodIncompatibility(f"{context}: {name} must be finite and positive")
    return parsed


def _coerce_column_list(value: Any, *, name: str, context: str) -> List[str]:
    if isinstance(value, str):
        return [value]
    try:
        cols = list(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a column name or a list of column names"
        ) from exc
    if not all(isinstance(col, str) for col in cols):
        raise MethodIncompatibility(
            f"{context}: {name} must contain only column-name strings"
        )
    return cols


def _kernel_weight(u: np.ndarray, kernel: str) -> np.ndarray:
    """Un-normalised kernel weight K(u); 0 outside [-1, 1] (save 'normal')."""
    au = np.abs(u)
    inside = au <= 1.0
    if kernel == "epan":
        w = np.where(inside, 1.0 - u**2, 0.0)
    elif kernel == "biweight":
        w = np.where(inside, (1.0 - u**2) ** 2, 0.0)
    elif kernel == "uniform":
        w = np.where(inside, 1.0, 0.0)
    elif kernel == "tricube":
        w = np.where(inside, (1.0 - au**3) ** 3, 0.0)
    elif kernel == "normal":
        # standard normal density, unbounded support (no truncation)
        w = np.exp(-0.5 * u**2) / np.sqrt(2.0 * np.pi)
    else:  # pragma: no cover — guarded by _validate
        raise MethodIncompatibility(f"unknown kernel '{kernel}'")
    return w


# ======================================================================
# Public API
# ======================================================================


def match(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    *,
    # --- new orthogonal API ---
    distance: Optional[str] = None,
    method: str = "nearest",
    # --- matching parameters ---
    estimand: str = "ATT",
    n_matches: int = 1,
    caliper: Optional[float] = None,
    caliper_scale: str = "raw",
    replace: bool = True,
    ties: str = "first",
    tie_tolerance: float = 0.0,
    m_order: str = "smallest_min_dist",
    mahalanobis_cov: str = "pooled",
    bias_correction: bool = False,
    # --- propensity score specification ---
    ps_poly: int = 1,
    # --- common support ---
    common_support: str = "none",
    # --- kernel / radius matching ---
    kernel: str = "epan",
    bwidth: float = 0.06,
    se_method: str = "auto",
    ai_matches: int = 1,
    bootstrap_reps: int = 200,
    bootstrap_seed: Optional[int] = None,
    llr_stata_compat: bool = False,
    # --- stratification parameters ---
    n_strata: int = 5,
    # --- CEM parameters ---
    n_bins: Optional[int] = None,
    # --- inference ---
    alpha: float = 0.05,
) -> CausalResult:
    """
    Estimate treatment effect using matching.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Outcome variable.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Variables to match on.
    distance : str, optional
        Distance metric: 'propensity', 'mahalanobis', 'euclidean', 'exact'.
        Default is 'propensity' for method='nearest'/'stratify'.
    method : str, default 'nearest'
        Matching algorithm: 'nearest', 'stratify', 'cem', 'kernel',
        'radius', 'llr'.  Legacy values 'psm', 'mahalanobis' also accepted.

        ``'llr'`` is local linear regression matching (Heckman, Ichimura &
        Todd 1997): each treated unit's counterfactual is the intercept of a
        kernel-weighted degree-1 regression of the control outcome on the
        propensity gap.  Note that Stata ``psmatch2 ..., llr`` with its
        *default* Epanechnikov kernel does not run this estimator — it
        substitutes nearest-neighbour matching on an ``lpoly``-smoothed
        outcome.  Pass ``kernel='tricube'`` to reproduce psmatch2's own LLR
        routine; ``kernel='epan'`` runs genuine LLR here and warns about the
        divergence.
    estimand : str, default 'ATT'
        Target estimand: 'ATT' or 'ATE'.
    n_matches : int, default 1
        Number of nearest-neighbor matches per unit.
    caliper : float, optional
        Maximum distance for a valid match.
    replace : bool, default True
        Match with replacement (nearest-neighbor only).
    bias_correction : bool, default False
        Apply Abadie-Imbens (2011) bias correction via regression
        adjustment on the matching discrepancy.
    ps_poly : int, default 1
        Polynomial degree for the propensity score logit model.
        ``ps_poly=1`` uses linear terms only.
        ``ps_poly=2`` adds all squared terms and pairwise interactions.
        ``ps_poly=3`` adds cubic terms as well.
        Higher-order specifications are standard practice; see
        Cunningham (2021, Ch. 5) for worked examples with
        ``age + age^2 + age^3 + educ + educ^2 + educ*re74``.
    common_support : {'none', 'minmax'}, default 'none'
        Common-support trimming for nearest-neighbor matching.
        ``'none'`` (default) matches every treated unit and leaves the
        point estimate unchanged.  ``'minmax'`` mirrors Stata
        ``psmatch2 , common``: treated units whose propensity score falls
        outside the [min, max] range of the control scores are dropped
        before matching and the ATT is taken over the on-support treated.
        The matched-sample frame (``result.matched_data``) records the
        common-support flag in ``_support`` either way.
    kernel : str, default 'epan'
        Kernel type for ``method='kernel'`` — one of ``'epan'``,
        ``'normal'``, ``'biweight'``, ``'uniform'``, ``'tricube'`` (matches
        Stata ``psmatch2 , kerneltype()``).  Ignored for other methods.
    bwidth : float, default 0.06
        Kernel bandwidth on the propensity score for ``method='kernel'``
        (Stata's ``bwidth()`` default).  For ``method='radius'`` the
        bandwidth is taken from ``caliper`` instead.
    se_method : {'auto', 'ai', 'psmatch2', 'abadie_imbens', 'bootstrap'}, default 'auto'
        Standard-error estimator, measured over 36 designs x 1000
        replications by ``benchmarks/matching_se_coverage.py`` (the ratios
        below are reported SE / true sampling SD; 1.00 is correctly sized):

        ==================  ===========  ==================
        option              SE ratio     coverage (nom. .95)
        ==================  ===========  ==================
        ``'ai'``            0.56 - 0.91  0.71 - 0.92
        ``'psmatch2'``      1.50 - 1.69  0.994 - 1.000
        ``'abadie_imbens'`` 0.95 - 1.04  0.905 - 0.956
        ``'bootstrap'``     0.95 - 1.23  0.933 - 1.000
        ==================  ===========  ==================

        ``'ai'`` is the simple matched-pair SE (the historical default for
        nearest-neighbour matching). **It is anti-conservative**: it treats
        matched pairs as independent and ignores the extra variance from
        reusing controls under matching with replacement, and never reaches
        nominal coverage on the grid above. ``'psmatch2'`` is Stata
        psmatch2's homoskedastic analytic ATT SE
        ``sqrt(var1/N1 + var0*Σw²/N1²)``; it errs the other way, so use it
        to reproduce Stata rather than to size a test. ``'abadie_imbens'``
        is the Abadie-Imbens (2006) heteroskedasticity-robust SE (Stata
        ``psmatch2 , ai(J)``), with ``J = ai_matches`` within-arm matches.
        It is the only option measured to be correctly sized across the
        grid and is the recommended choice for nearest-neighbour
        inference.
        ``'bootstrap'`` is an arm-stratified nonparametric bootstrap that
        re-estimates the propensity score in every replication, so unlike the
        analytic options it accounts for the sampling variability of the
        fitted score.  It is the default (and the only valid choice) for
        ``method='llr'``, where local linear weights can be negative and
        Stata psmatch2 itself reports no standard error.  Abadie & Imbens
        (2008) show the bootstrap is *not* valid for nearest-neighbour
        matching with a fixed number of matches; requesting it there warns.
        ``'auto'`` (the default) resolves to ``'abadie_imbens'`` for
        nearest-neighbour matching, ``'psmatch2'`` for kernel / radius, and
        ``'bootstrap'`` for ``'llr'`` -- in each case the estimator measured
        to be correctly sized for that matching scheme.

        .. versionchanged:: 1.22
           ``'auto'`` resolved to ``'ai'`` before 1.22, so **default
           nearest-neighbour standard errors change** (they get larger, by
           roughly 1/0.56 to 1/0.91). Point estimates are untouched. Pass
           ``se_method='ai'`` to recover the old numbers.
    ai_matches : int, default 1
        Number of within-arm matches ``J`` used by the
        ``se_method='abadie_imbens'`` conditional-variance estimate
        (Stata's ``ai(J)``).
    bootstrap_reps : int, default 200
        Number of bootstrap replications for ``se_method='bootstrap'``.
    bootstrap_seed : int, optional
        Seed for the bootstrap resampler.  Pass one for reproducible
        standard errors.
    n_strata : int, default 5
        Number of strata for method='stratify'.
    n_bins : int, optional
        Number of bins per covariate for method='cem'.
        Default uses Sturges' rule.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> # Propensity score matching (default)
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'])

    >>> # Mahalanobis distance + bias correction
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'],
    ...                   distance='mahalanobis', bias_correction=True)

    >>> # Exact matching
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'],
    ...                   distance='exact')

    >>> # Propensity score stratification (5 strata)
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'],
    ...                   method='stratify', n_strata=5)

    >>> # CEM
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'],
    ...                   method='cem')

    >>> # Quadratic PS model (Cunningham 2021, Ch. 5 style)
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu', 'exp'],
    ...                   ps_poly=2)

    >>> # Without-replacement matching
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'],
    ...                   replace=False)

    >>> # Legacy API still works
    >>> result = sp.match(df, y='wage', treat='training',
    ...                   covariates=['age', 'edu'], method='psm')
    """
    estimator = MatchEstimator(
        data=data,
        y=y,
        treat=treat,
        covariates=covariates,
        distance=distance,
        method=method,
        estimand=estimand,
        n_matches=n_matches,
        caliper=caliper,
        caliper_scale=caliper_scale,
        replace=replace,
        ties=ties,
        tie_tolerance=tie_tolerance,
        m_order=m_order,
        mahalanobis_cov=mahalanobis_cov,
        bias_correction=bias_correction,
        ps_poly=ps_poly,
        common_support=common_support,
        kernel=kernel,
        bwidth=bwidth,
        se_method=se_method,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
        llr_stata_compat=llr_stata_compat,
        ai_matches=ai_matches,
        n_strata=n_strata,
        n_bins=n_bins,
        alpha=alpha,
    )
    _result = estimator.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.matching.match",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(estimator.covariates),
                "distance": distance,
                "method": method,
                "estimand": estimand,
                "n_matches": n_matches,
                "caliper": caliper,
                "caliper_scale": caliper_scale,
                "replace": replace,
                "ties": ties,
                "tie_tolerance": tie_tolerance,
                "m_order": m_order,
                "mahalanobis_cov": mahalanobis_cov,
                "bias_correction": bias_correction,
                "ps_poly": ps_poly,
                "common_support": common_support,
                "kernel": kernel,
                "bwidth": bwidth,
                "se_method": se_method,
                "bootstrap_reps": bootstrap_reps,
                "ai_matches": ai_matches,
                "n_strata": n_strata,
                "n_bins": n_bins,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# MatchEstimator
# ======================================================================


class MatchEstimator:
    """Unified matching estimator supporting multiple distance × method combinations.

    This is the object-oriented backend behind :func:`match`. Most users
    should call :func:`sp.match`; construct ``MatchEstimator`` directly only
    when you want to hold the configured estimator and call ``.fit()``
    yourself. ``.fit()`` returns a ``CausalResult``.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data.
    y : str
        Outcome column.
    treat : str
        Binary (0/1) treatment column.
    covariates : list of str
        Variables to match on.
    distance : str, optional
        ``'propensity'``, ``'mahalanobis'``, ``'euclidean'`` or ``'exact'``.
    method : str, default 'nearest'
        ``'nearest'``, ``'stratify'`` or ``'cem'`` (legacy ``'psm'`` /
        ``'mahalanobis'`` are also accepted).
    estimand : str, default 'ATT'
        ``'ATT'`` or ``'ATE'``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> age = rng.normal(40, 8, n)
    >>> edu = rng.normal(12, 2, n)
    >>> ps = 1 / (1 + np.exp(-(0.05 * (age - 40) + 0.1 * (edu - 12))))
    >>> training = rng.binomial(1, ps)
    >>> wage = 20 + 0.3 * age + 0.5 * edu + 4.0 * training + rng.normal(0, 3, n)
    >>> df = pd.DataFrame({"wage": wage, "training": training,
    ...                    "age": age, "edu": edu})
    >>> est = sp.MatchEstimator(df, y="wage", treat="training",
    ...                         covariates=["age", "edu"], distance="propensity")
    >>> result = est.fit()
    >>> type(result).__name__
    'CausalResult'
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        *,
        distance: Optional[str] = None,
        method: str = "nearest",
        estimand: str = "ATT",
        n_matches: int = 1,
        caliper: Optional[float] = None,
        caliper_scale: str = "raw",
        replace: bool = True,
        ties: str = "first",
        tie_tolerance: float = 0.0,
        m_order: str = "smallest_min_dist",
        mahalanobis_cov: str = "pooled",
        bias_correction: bool = False,
        ps_poly: int = 1,
        common_support: str = "none",
        kernel: str = "epan",
        bwidth: float = 0.06,
        se_method: str = "auto",
        ai_matches: int = 1,
        bootstrap_reps: int = 200,
        bootstrap_seed: Optional[int] = None,
        llr_stata_compat: bool = False,
        n_strata: int = 5,
        n_bins: Optional[int] = None,
        alpha: float = 0.05,
    ):
        context = "match"
        if not isinstance(data, pd.DataFrame):
            raise MethodIncompatibility(
                "match: data must be a pandas DataFrame",
                recovery_hint=(
                    "Pass a pandas DataFrame with named outcome, treatment, "
                    "and covariate columns."
                ),
                diagnostics={"type": type(data).__name__},
            )
        self.data = data.copy()
        self.y = y
        self.treat = treat
        self.covariates = _coerce_column_list(
            covariates,
            name="covariates",
            context=context,
        )
        self.estimand = str(estimand).upper()
        self.n_matches = _positive_int(
            n_matches,
            name="n_matches",
            context=context,
        )
        self.caliper = (
            None
            if caliper is None
            else _positive_float(caliper, name="caliper", context=context)
        )
        self.caliper_scale = str(caliper_scale).lower()
        if self.caliper_scale not in ("raw", "sd"):
            raise ValueError(
                f"match: caliper_scale must be 'raw' or 'sd', got " f"{caliper_scale!r}"
            )
        self.replace = replace
        self.ties = str(ties).lower()
        if self.ties not in ("first", "all"):
            raise ValueError(f"match: ties must be 'first' or 'all', got {ties!r}")
        self.tie_tolerance = float(tie_tolerance)
        if self.tie_tolerance < 0:
            raise ValueError("match: tie_tolerance must be >= 0")
        self._pscore_cache: Optional[np.ndarray] = None
        self.m_order = str(m_order).lower()
        _VALID_M_ORDER = (
            "smallest_min_dist",
            "data",
            "closest",
            "farthest",
            "smallest",
            "largest",
        )
        if self.m_order not in _VALID_M_ORDER:
            raise ValueError(
                f"match: m_order must be one of {_VALID_M_ORDER}, got " f"{m_order!r}"
            )
        self.mahalanobis_cov = str(mahalanobis_cov).lower()
        if self.mahalanobis_cov not in ("pooled", "total"):
            raise ValueError(
                f"match: mahalanobis_cov must be 'pooled' or 'total', got "
                f"{mahalanobis_cov!r}"
            )
        self.bias_correction = bias_correction
        self.ps_poly = _positive_int(ps_poly, name="ps_poly", context=context)
        self.common_support = str(common_support).lower()
        self.kernel = str(kernel).lower()
        self.bwidth = _positive_float(bwidth, name="bwidth", context=context)
        self.se_method = str(se_method).lower()
        self.ai_matches = _positive_int(
            ai_matches,
            name="ai_matches",
            context=context,
        )
        self.bootstrap_reps = _positive_int(
            bootstrap_reps,
            name="bootstrap_reps",
            context=context,
        )
        self.bootstrap_seed = bootstrap_seed
        self.llr_stata_compat = bool(llr_stata_compat)
        # Set by fit() before the SE stage so _bootstrap_se can report bias.
        self._point_estimate: float = float("nan")
        self.n_strata = _positive_int(
            n_strata,
            name="n_strata",
            context=context,
        )
        self.n_bins = (
            None
            if n_bins is None
            else _positive_int(n_bins, name="n_bins", context=context)
        )
        self.alpha = _open_unit_float(alpha, name="alpha", context=context)
        # Filled by _fit_nearest so fit() can build the matched-sample frame.
        self._assignment: Optional[dict[str, Any]] = None
        # ATE only: the control->treated matching leg, needed for the
        # Abadie-Imbens ATE weight 1 + K_M(i).
        self._ate_reverse: Optional[Tuple[List[Any], List[Any]]] = None
        # Filled by _fit_stratify / _fit_cem so fit() can build the cell-based
        # matched frame (pscore, cell label, and which cells were retained).
        self._stratum_assignment: Optional[dict[str, Any]] = None

        # Resolve legacy method names
        method_lower = str(method).lower()
        if method_lower in _LEGACY_MAP:
            resolved_dist, resolved_method = _LEGACY_MAP[method_lower]
            self.distance = resolved_dist if distance is None else str(distance).lower()
            self.method = resolved_method
        else:
            self.method = method_lower
            self.distance = str(distance).lower() if distance else None

        # Set default distance for methods that need one
        if self.distance is None:
            if self.method in ("nearest", "stratify", "kernel", "radius", "llr"):
                self.distance = "propensity"
            elif self.method == "cem":
                self.distance = None  # CEM doesn't use distance

        self._validate()

    def _validate(self) -> None:
        required = [self.y, self.treat] + self.covariates
        missing = [col for col in required if col not in self.data.columns]
        if missing:
            raise MethodIncompatibility(
                f"match: columns not found in data: {missing}",
                recovery_hint=(
                    "Check y, treat, and covariates column names before matching."
                ),
                diagnostics={"missing_columns": missing},
            )

        if self.method not in _VALID_METHODS:
            raise MethodIncompatibility(
                f"match: method must be one of {_VALID_METHODS} "
                f"(or legacy: 'psm', 'mahalanobis'), got '{self.method}'"
            )
        if self.distance is not None and self.distance not in _VALID_DISTANCES:
            raise MethodIncompatibility(
                f"match: distance must be one of {_VALID_DISTANCES}, "
                f"got '{self.distance}'"
            )
        if self.estimand not in ("ATT", "ATE"):
            raise MethodIncompatibility(
                f"match: estimand must be 'ATT' or 'ATE', got '{self.estimand}'"
            )

        if self.common_support not in ("none", "minmax"):
            raise MethodIncompatibility(
                "match: common_support must be 'none' or 'minmax', "
                f"got '{self.common_support}'"
            )

        treat_vals = self.data[self.treat].dropna().unique()
        if not set(treat_vals).issubset({0, 1, 0.0, 1.0}):
            raise MethodIncompatibility(
                f"Treatment must be binary (0/1), got values: {treat_vals}",
                recovery_hint=(
                    "Matching assumes a binary treatment. For multi-valued "
                    "treatments use sp.multi_treatment; for continuous use "
                    "sp.dose_response."
                ),
                diagnostics={"treat_values": sorted(map(str, treat_vals))[:10]},
                alternative_functions=["sp.multi_treatment", "sp.dose_response"],
            )

        # Exact matching only supports ATT
        if self.distance == "exact" and self.estimand == "ATE":
            raise MethodIncompatibility(
                "match: exact matching only supports estimand='ATT'"
            )

        # Stratification only works with propensity distance
        if self.method == "stratify" and self.distance != "propensity":
            raise MethodIncompatibility(
                "match: method='stratify' requires distance='propensity'"
            )

        # Kernel / radius matching are propensity-score based.
        if self.method in ("kernel", "radius", "llr") and self.distance != "propensity":
            raise MethodIncompatibility(
                f"match: method='{self.method}' requires distance='propensity'"
            )
        if self.method in ("kernel", "radius", "llr") and self.estimand != "ATT":
            raise MethodIncompatibility(
                f"match: method='{self.method}' currently supports "
                "estimand='ATT' only"
            )
        if self.method == "kernel" and self.kernel not in _VALID_KERNELS:
            raise MethodIncompatibility(
                f"match: kernel must be one of {_VALID_KERNELS}, "
                f"got '{self.kernel}'"
            )
        if self.method == "radius" and self.caliper is None:
            raise MethodIncompatibility(
                "match: radius matching requires caliper > 0 " "(the radius bandwidth)"
            )

        if self.se_method not in (
            "auto",
            "ai",
            "psmatch2",
            "abadie_imbens",
            "abadie_imbens_pop",
            "bootstrap",
        ):
            raise MethodIncompatibility(
                "match: se_method must be 'auto', 'ai', 'psmatch2', "
                "'abadie_imbens', 'abadie_imbens_pop', or 'bootstrap', got "
                f"'{self.se_method}'"
            )
        if self.llr_stata_compat and self.method != "llr":
            raise MethodIncompatibility(
                "match: llr_stata_compat=True only applies to method='llr'; "
                f"got method={self.method!r}.",
                recovery_hint="Pass method='llr', or drop llr_stata_compat.",
            )
        if (
            self.method == "llr"
            and self.se_method == "psmatch2"
            and not self.llr_stata_compat
        ):
            raise MethodIncompatibility(
                "match: se_method='psmatch2' is not defined for "
                "method='llr'. Local linear weights can be negative, which "
                "the analytic formula sqrt(var1/N1 + var0*sum(w^2)/N1^2) "
                "assumes away; Stata psmatch2 itself reports seatt = . for "
                "llr. Use se_method='bootstrap' (the default for llr).",
                recovery_hint="Use se_method='bootstrap' for method='llr'.",
            )
        if self.estimand != "ATT" and self.se_method == "psmatch2":
            raise MethodIncompatibility(
                "match: se_method='psmatch2' is only defined for estimand='ATT'"
            )

    # ==================================================================
    # Main fit
    # ==================================================================

    def fit(self) -> CausalResult:
        """Fit matching estimator and return results."""
        cols = [self.y, self.treat] + self.covariates
        clean = self.data[cols].dropna()
        T = clean[self.treat].values.astype(int)
        Y = clean[self.y].values.astype(float)
        X = clean[self.covariates].values.astype(float)
        row_order = self._stable_index_order(clean.index)

        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]

        if len(idx_t) == 0 or len(idx_c) == 0:
            from statspai.exceptions import DataInsufficient

            raise DataInsufficient(
                "Need both treated and control observations",
                recovery_hint=(
                    "All observations have the same treatment value; "
                    "re-check the treatment column."
                ),
                diagnostics={
                    "n_treated": int(len(idx_t)),
                    "n_control": int(len(idx_c)),
                },
                alternative_functions=[],
            )

        # Dispatch — each returns (att, se, balance, extra_info)
        extra_info: dict[str, Any] = {}
        if self.method == "cem":
            att, se, balance, extra_info = self._fit_cem(Y, X, T, idx_t, idx_c)
            method_label = "Matching (CEM)"
        elif self.method == "stratify":
            att, se, balance, extra_info = self._fit_stratify(Y, X, T, idx_t, idx_c)
            method_label = "Matching (PS Stratification)"
        elif self.method == "llr" and self.llr_stata_compat:
            # Stata compatibility: psmatch2 does not run LLR here, it runs
            # nearest-neighbour matching on an lpoly-smoothed outcome.
            att, se, balance = self._fit_nearest(Y, X, T, idx_t, idx_c, row_order)
            method_label = "Matching (psmatch2 llr-compat: lpoly + nearest)"
        elif self.method in ("kernel", "radius", "llr"):
            att, se, balance, extra_info = self._fit_kernel(Y, X, T, idx_t, idx_c)
            kt = "Radius" if self.method == "radius" else f"Kernel:{self.kernel}"
            method_label = f"Matching ({kt})"
        elif self.distance == "exact":
            att, se, balance, extra_info = self._fit_exact(Y, X, T, idx_t, idx_c)
            method_label = "Matching (Exact)"
        else:
            att, se, balance = self._fit_nearest(Y, X, T, idx_t, idx_c, row_order)
            dist_name = str(self.distance).capitalize()
            bc_tag = ", BC" if self.bias_correction else ""
            method_label = f"Matching ({dist_name}{bc_tag})"

        # PSM warning
        if self.distance == "propensity" and self.method == "nearest":
            warnings.warn(
                "PSM can increase imbalance and bias (King & Nielsen 2019). "
                "Consider distance='mahalanobis' or method='cem'.",
                UserWarning,
                stacklevel=3,
            )

        model_info = {
            "distance": self.distance,
            "method": self.method,
            "estimand": self.estimand,
            "n_treated": int(len(idx_t)),
            "n_control": int(len(idx_c)),
            "n_matches": self.n_matches,
            "caliper": self.caliper,
            "replace": self.replace,
            "bias_correction": self.bias_correction,
            "ps_poly": self.ps_poly,
            "common_support": self.common_support,
            "balance": balance,
            **extra_info,
        }

        # Assemble the Stata psmatch2-style matched-sample frame for the
        # assignment-producing paths (nearest / kernel / radius).  This is
        # pure bookkeeping over the assignment already used for the point
        # estimate.  When ``se_method`` resolves to ``'psmatch2'`` the
        # analytic Lechner SE is read back off this frame.
        self._point_estimate = float(att)
        matched_data = None
        if self._assignment is not None and self.estimand == "ATT":
            a = self._assignment
            emit_neighbors = a.get("neighbors", True)
            frame = build_matched_frame(
                index=clean.index,
                treated=a["treated"],
                pscore=a["pscore"],
                idx_t=a["idx_t"],
                idx_c=a["idx_c"],
                matches=a["matches"],
                weights=a["weights"],
                n_matches=self.n_matches,
                support=a["support"],
                outcome=a.get("outcome_frame", a["outcome"]),
                neighbors=emit_neighbors,
            )
            matched_data = attach_matched_frame(self.data, frame)
            model_info["matched_columns"] = matched_columns(
                self.n_matches, with_outcome=True, neighbors=emit_neighbors
            )
            model_info["matched_frame_estimand"] = "ATT"
            model_info["matched_frame_weight_kind"] = "att_frequency"
            # n_on_support = all on-support obs; n_treated_on_support = the
            # treated subset (what the psmatch2 summary reports).
            model_info["n_on_support"] = int(np.sum(a["support"]))
            model_info["n_treated_on_support"] = int(np.sum(a["support"][a["idx_t"]]))
            model_info["n_matched_treated"] = int(
                np.sum([len(m) > 0 for m in a["matches"]])
            )

            # The reported ATT averages over the treated units that actually
            # found a match. When some did not, that average is over a
            # *non-random* subset -- selected by matching order, caliper
            # width, or an exhausted control pool -- and saying so is the
            # difference between a documented restriction and a silent one.
            self._warn_incomplete_treated_coverage(a, model_info)

            # Resolve and (optionally) override the SE with the digit-exact
            # Stata psmatch2 analytic / Abadie-Imbens robust standard error.
            se_method = self._resolve_se_method()
            model_info["se_method"] = se_method
            if se_method == "psmatch2":
                se_p = psmatch2_se(
                    a["outcome"],
                    a["treated"],
                    a["support"],
                    frame[COL_WEIGHT].to_numpy(dtype=float),
                )
                if np.isfinite(se_p):
                    se = se_p
            elif se_method == "abadie_imbens":
                model_info["ai_matches"] = self.ai_matches
                se_ai = abadie_imbens_se(
                    a["outcome"],
                    a["treated"],
                    a["pscore"],
                    a["support"],
                    frame[COL_WEIGHT].to_numpy(dtype=float),
                    n_ai_matches=self.ai_matches,
                )
                if np.isfinite(se_ai):
                    se = se_ai
            elif se_method == "abadie_imbens_pop":
                se_pop = self._ai_population_se(
                    a["outcome"], a["matches"], a["weights"]
                )
                if np.isfinite(se_pop):
                    se = se_pop
            elif se_method == "bootstrap":
                se_bs, bs_info = self._bootstrap_se(clean)
                model_info.update(bs_info)
                if np.isfinite(se_bs):
                    se = se_bs
        elif self._assignment is not None and self._ate_reverse is not None:
            # ATE: both arms are matched, so the frame carries the
            # Abadie-Imbens weight 1 + K_M(i) rather than a control frequency.
            a = self._assignment
            m_ct, w_ct = self._ate_reverse
            frame = build_ate_matched_frame(
                index=clean.index,
                treated=a["treated"],
                pscore=a["pscore"],
                idx_t=a["idx_t"],
                idx_c=a["idx_c"],
                matches_tc=a["matches"],
                weights_tc=a["weights"],
                matches_ct=m_ct,
                weights_ct=w_ct,
                n_matches=self.n_matches,
                support=a["support"],
                outcome=a["outcome"],
            )
            matched_data = attach_matched_frame(self.data, frame)
            model_info["matched_columns"] = matched_columns(
                self.n_matches, with_outcome=True, neighbors=True
            )
            model_info["matched_frame_estimand"] = "ATE"
            model_info["matched_frame_weight_kind"] = "ate_signed"
            model_info["matched_data_note"] = (
                "estimand='ATE' matches both arms, so _weight is the "
                "Abadie-Imbens (2006) weight 1 + K_M(i) and enters the "
                "estimator with the sign (2*treated - 1): "
                "ATE = mean((2*_treated - 1) * _weight * y). It is NOT a "
                "frequency weight -- do not pass it to a [fweight=] "
                "regression. Use estimand='ATT' for the psmatch2 frequency "
                "semantics."
            )
        elif self._stratum_assignment is not None:
            # Stratification / CEM: cell comparisons, not ordered neighbours.
            s = self._stratum_assignment
            frame = build_stratum_matched_frame(
                index=clean.index,
                treated=s["treated"],
                pscore=s["pscore"],
                stratum=s["stratum"],
                keep=s["keep"],
                outcome=s["outcome"],
            )
            matched_data = attach_matched_frame(self.data, frame)
            model_info["matched_columns"] = matched_columns(
                self.n_matches, with_outcome=True, stratum=True
            )
            model_info["matched_frame_estimand"] = self.estimand
            model_info["matched_frame_weight_kind"] = "att_frequency"
            model_info["n_on_support"] = int(np.sum(s["keep"]))
            model_info["n_treated_on_support"] = int(
                np.sum(s["keep"] & (s["treated"] == 1))
            )
            if self.estimand != "ATT":
                model_info["matched_data_note"] = (
                    "_weight encodes the ATT cell weighting (treated = 1, "
                    "control = n_treated_in_cell / n_control_in_cell); the "
                    "reported estimate uses the ATE cell weighting, so the "
                    "two will not agree. Use estimand='ATT' to have them "
                    "match."
                )
        else:  # pragma: no cover - every fitted path is covered above
            model_info["matched_data_note"] = (
                f"No matched-sample frame is available for method="
                f"{self.method!r} with estimand={self.estimand!r}. This "
                "combination does not produce a unit-level matching weight."
            )

        # An explicit ``se_method='ai'`` is the caller's own choice, but it
        # is the one option measured never to reach nominal coverage, so it
        # does not pass silently.
        if self.se_method == "ai" and self.method == "nearest":
            warnings.warn(
                "sp.match: se_method='ai' is the simple matched-pair SE. It "
                "treats matched pairs as independent and ignores the extra "
                "variance from reusing controls under matching with "
                "replacement. Measured over 36 designs x 1000 replications "
                "(benchmarks/matching_se_coverage.py) it runs 0.56-0.91x the "
                "true sampling SD and never reaches nominal coverage: 0.71 "
                "to 0.92 against a nominal 0.95. The default "
                "se_method='auto' now resolves to 'abadie_imbens', which is "
                "correctly sized over the same grid.",
                UserWarning,
                stacklevel=2,
            )

        # Inference (after the SE is finalized)
        t_stat = att / se if se > 0 else 0.0
        pvalue = float(2 * (1 - stats.norm.cdf(abs(t_stat))))
        z = stats.norm.ppf(1 - self.alpha / 2)
        ci = (att - z * se, att + z * se)

        result = CausalResult(
            method=method_label,
            estimand=self.estimand,
            estimate=att,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=len(clean),
            detail=balance,
            model_info=model_info,
            _citation_key="matching",
        )
        # Expose the matched frame both as a convenience attribute and in
        # model_info (the latter survives serialisation / provenance).
        setattr(result, "matched_data", matched_data)
        if matched_data is not None:
            result.model_info["matched_data"] = matched_data
        return result

    # ==================================================================
    # Nearest-neighbor matching (propensity / mahalanobis / euclidean)
    # ==================================================================

    def _fit_nearest(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        idx_t: np.ndarray,
        idx_c: np.ndarray,
        row_order: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, pd.DataFrame]:
        """Nearest-neighbor matching with configurable distance metric."""
        if row_order is None:
            row_order = np.arange(len(T), dtype=float)

        # For propensity distance, estimate PS once with actual treatment
        pscore = (
            self._logit_propensity(X, T, poly=self.ps_poly)
            if self.distance == "propensity"
            else None
        )
        # PS is always needed downstream (balance table + matched frame +
        # common-support flag), even when the distance metric is not PS.
        if pscore is None:
            pscore = self._logit_propensity(X, T, poly=self.ps_poly)

        # Common-support flag over the full estimation sample.  With
        # common_support='none' every unit is on support and the matching
        # below is byte-identical to the historical implementation.
        support = common_support_mask(pscore, T, rule=self.common_support)

        # psmatch2 llr-compat: Stata replaces the *control* outcome with an
        # lpoly(deg 1) fit on the on-support controls, then matches on the
        # propensity score as usual.  A treated unit's counterfactual is the
        # SMOOTHED outcome of its match, while its own outcome stays raw --
        # so the estimator compares a raw level to a smoothed one.  Y_eff
        # carries that mix; the raw Y is kept for the analytic SE, which
        # psmatch2 also computes off the unsmoothed outcome.
        Y_eff = Y
        if self.llr_stata_compat:
            smoothed = psmatch2_llr_smoothed_outcome(
                Y, pscore, T, support, bandwidth=float(self.bwidth), degree=1
            )
            Y_eff = np.where(T == 1, Y, smoothed)
            if not np.all(np.isfinite(Y_eff[T == 0])):
                bad = int(np.sum(~np.isfinite(Y_eff[T == 0])))
                warnings.warn(
                    f"sp.match(llr_stata_compat=True): the lpoly smoother "
                    f"left {bad} control outcome(s) undefined (no donor "
                    f"inside the bandwidth). Those controls cannot serve as "
                    f"matches. Increase bwidth.",
                    UserWarning,
                    stacklevel=3,
                )

        # Targets actually fed to the matcher.  Under 'minmax' we drop the
        # off-support treated *before* matching so the ATT is taken over the
        # on-support treated (Stata psmatch2 `common`).
        if self.common_support == "minmax":
            t_use = idx_t[support[idx_t]]
        else:
            t_use = idx_t

        # Build distance matrix for the (used-treated × control) block
        dist_mat = self._compute_distance_matrix(X, t_use, idx_c, pscore)

        # A caliper given in 'sd' units is expressed in standard deviations
        # of the distance measure (the MatchIt `std.caliper = TRUE`
        # convention); 'raw' leaves it on the distance scale (Stata
        # psmatch2). Resolve it once, against the full-sample distance
        # measure, so both matching directions use the same width.
        self._pscore_cache = pscore
        caliper = self._resolve_caliper(pscore)

        if self.estimand == "ATT":
            matches, weights = self._nn_match_from_dist(
                dist_mat,
                caliper,
                target_order=row_order[t_use],
                pool_order=row_order[idx_c],
                pscore_target=None if pscore is None else pscore[t_use],
            )
            att = self._compute_effect(Y_eff, t_use, idx_c, X, matches, weights)
            se = self._ai_se(Y, X, T, t_use, idx_c, matches, weights)
            assign_matches, assign_weights = matches, weights
        else:
            # ATE: match both directions, reuse the same propensity scores.
            # Common-support trimming is ATT-specific; ATE uses all units.
            dist_ct = self._compute_distance_matrix(X, idx_c, idx_t, pscore)
            m_tc, w_tc = self._nn_match_from_dist(
                dist_mat,
                caliper,
                target_order=row_order[t_use],
                pool_order=row_order[idx_c],
                pscore_target=None if pscore is None else pscore[t_use],
            )
            m_ct, w_ct = self._nn_match_from_dist(
                dist_ct,
                caliper,
                target_order=row_order[idx_c],
                pool_order=row_order[idx_t],
                pscore_target=None if pscore is None else pscore[idx_c],
            )
            att_part = self._compute_effect(Y, t_use, idx_c, X, m_tc, w_tc)
            atc_part = self._compute_effect(Y, idx_c, idx_t, X, m_ct, w_ct)
            n_t, n_c = len(t_use), len(idx_c)
            att = (n_t * att_part + n_c * (-atc_part)) / (n_t + n_c)
            se = self._ai_se(Y, X, T, t_use, idx_c, m_tc, w_tc)
            assign_matches, assign_weights = m_tc, w_tc
            # Keep the control->treated leg too: the ATE matched frame needs
            # both directions to form the Abadie-Imbens weight 1 + K_M(i).
            self._ate_reverse = (m_ct, w_ct)

        balance = self._balance_table(X, T, pscore)

        # Record the treated→control assignment so fit() can assemble the
        # psmatch2-style matched-sample frame.  Expand back to the full
        # treated index: off-support / unmatched treated get empty matches.
        full_matches = [np.array([], dtype=int)] * len(idx_t)
        full_weights = [np.array([], dtype=float)] * len(idx_t)
        pos_in_full = {int(p): k for k, p in enumerate(idx_t)}
        for j, t_pos in enumerate(t_use):
            k = pos_in_full[int(t_pos)]
            full_matches[k] = assign_matches[j]
            full_weights[k] = assign_weights[j]

        self._assignment = {
            "pscore": pscore,
            "treated": T,
            "idx_t": idx_t,
            "idx_c": idx_c,
            "matches": full_matches,
            "weights": full_weights,
            "support": support,
            "outcome": Y,
            # `_y` is the matched partner's outcome as psmatch2 records it,
            # which under llr-compat is the smoothed value.
            "outcome_frame": Y_eff,
            "neighbors": True,
        }

        return att, se, balance

    def _warn_incomplete_treated_coverage(
        self,
        assignment: dict[str, Any],
        model_info: dict[str, Any],
    ) -> None:
        """Warn when the ATT does not cover every on-support treated unit.

        Three mechanisms drop treated units, and all three are silent
        without this: a caliper that excludes every donor, common-support
        trimming, and -- for ``replace=False`` -- an exhausted control pool.
        The last is the dangerous one, because nothing about the returned
        number hints that most of the treated sample never entered it.
        """
        matches = assignment["matches"]
        support = np.asarray(assignment["support"], dtype=bool)
        idx_t = np.asarray(assignment["idx_t"], dtype=int)
        idx_c = np.asarray(assignment["idx_c"], dtype=int)

        on_support = support[idx_t]
        n_target = int(np.sum(on_support))
        if n_target == 0:
            return

        n_full = 0
        n_none = 0
        for i, m in enumerate(matches):
            if not on_support[i]:
                continue
            got = len(m)
            if got == 0:
                n_none += 1
            elif got >= self.n_matches:
                n_full += 1

        n_partial = n_target - n_full - n_none
        model_info["n_treated_unmatched"] = n_none
        model_info["n_treated_partially_matched"] = n_partial

        if n_none == 0 and n_partial == 0:
            return

        n_control = int(np.sum(support[idx_c]))
        needed = self.n_matches * n_target
        pool_exhausted = (not self.replace) and needed > n_control

        parts = []
        if n_none:
            parts.append(
                f"{n_none} of {n_target} on-support treated units found no "
                f"match at all and are EXCLUDED from the reported ATT "
                f"({100.0 * n_none / n_target:.0f}% of the treated sample)"
            )
        if n_partial:
            parts.append(
                f"{n_partial} matched fewer than the requested "
                f"n_matches={self.n_matches}"
            )
        cause = (
            (
                f" Cause: replace=False needs n_matches x n_treated = "
                f"{needed} control units but only {n_control} are on "
                f"support, so the pool is exhausted and the units matched "
                f"last get the worst partners (or none). Use replace=True, "
                f"reduce n_matches, or accept the restricted estimand."
            )
            if pool_exhausted
            else (
                " Cause: the caliper or common-support rule left those "
                "treated units without an admissible donor. Widen the "
                "caliper, or accept the restricted estimand."
            )
        )
        warnings.warn(
            "sp.match: "
            + "; ".join(parts)
            + "."
            + cause
            + " The reported effect is an ATT over the matched subset, which "
            "is selected non-randomly.",
            UserWarning,
            stacklevel=3,
        )

    def _bootstrap_se(self, clean: pd.DataFrame) -> Tuple[float, dict[str, Any]]:
        """Arm-stratified nonparametric bootstrap standard error.

        Each replication resamples units **with replacement within treatment
        arm** (so both arms stay non-empty and the arm sizes are fixed at
        their observed values), then re-runs the *whole* pipeline on the
        draw — including re-estimating the propensity score.  Re-estimating
        the score is the point: the analytic matching standard errors all
        condition on the fitted propensity score and so ignore the sampling
        variability it contributes.

        This is the estimator Stata users reach for via ``psmatch2``'s
        bootstrap prefix, and the only inference psmatch2 itself offers for
        local linear regression matching (it reports ``seatt = .`` there).

        .. warning::

           Abadie & Imbens (2008) show the nonparametric bootstrap is *not*
           generally valid for nearest-neighbour matching with a fixed
           number of matches, because the estimator is not smooth in the
           sampling distribution.  It is used here for the smooth
           kernel-class estimators (``kernel``, ``radius``, ``llr``), where
           that objection does not bite.  Requesting it for
           ``method='nearest'`` emits a warning.

        Returns
        -------
        (float, dict)
            The bootstrap SE and a ``model_info`` fragment recording the
            number of successful replications.

        References
        ----------
        Abadie, A. and Imbens, G.W. (2008). On the Failure of the Bootstrap
            for Matching Estimators. *Econometrica*, 76(6), 1537-1557.
        """
        if self.method == "nearest":
            warnings.warn(
                "sp.match: the nonparametric bootstrap is not generally "
                "valid for nearest-neighbour matching with a fixed number "
                "of matches (Abadie & Imbens 2008) -- the estimator is not "
                "smooth enough in the sampling distribution. Prefer "
                "se_method='abadie_imbens'.",
                UserWarning,
                stacklevel=3,
            )

        rng = np.random.default_rng(self.bootstrap_seed)
        t_rows = np.where(clean[self.treat].to_numpy() == 1)[0]
        c_rows = np.where(clean[self.treat].to_numpy() == 0)[0]

        estimates: List[float] = []
        n_failed = 0
        for _ in range(int(self.bootstrap_reps)):
            take = np.concatenate(
                [
                    rng.choice(t_rows, size=len(t_rows), replace=True),
                    rng.choice(c_rows, size=len(c_rows), replace=True),
                ]
            )
            draw = clean.iloc[take].reset_index(drop=True)
            try:
                with warnings.catch_warnings():
                    # A replication that trips a data warning is not itself
                    # noteworthy; a replication that *fails* is counted below.
                    warnings.simplefilter("ignore")
                    rep = MatchEstimator(
                        data=draw,
                        y=self.y,
                        treat=self.treat,
                        covariates=self.covariates,
                        distance=self.distance,
                        method=self.method,
                        estimand=self.estimand,
                        n_matches=self.n_matches,
                        caliper=self.caliper,
                        caliper_scale=self.caliper_scale,
                        replace=self.replace,
                        ties=self.ties,
                        tie_tolerance=self.tie_tolerance,
                        m_order=self.m_order,
                        mahalanobis_cov=self.mahalanobis_cov,
                        bias_correction=self.bias_correction,
                        ps_poly=self.ps_poly,
                        common_support=self.common_support,
                        kernel=self.kernel,
                        bwidth=self.bwidth,
                        llr_stata_compat=self.llr_stata_compat,
                        se_method="ai",  # never recurse into the bootstrap
                        ai_matches=self.ai_matches,
                        n_strata=self.n_strata,
                        n_bins=self.n_bins,
                        alpha=self.alpha,
                    ).fit()
                est = float(rep.estimate)
            except Exception:
                # A resample can legitimately be degenerate (e.g. no control
                # inside the bandwidth). Count it and carry on; a systematic
                # failure surfaces in the diagnostics below.
                n_failed += 1
                continue
            if np.isfinite(est):
                estimates.append(est)
            else:
                n_failed += 1

        info: dict[str, Any] = {
            "bootstrap_reps": int(self.bootstrap_reps),
            "bootstrap_reps_successful": len(estimates),
            "bootstrap_reps_failed": n_failed,
            "bootstrap_seed": self.bootstrap_seed,
        }
        if len(estimates) < 2:
            warnings.warn(
                f"sp.match: the bootstrap produced only {len(estimates)} "
                f"usable replication(s) out of {self.bootstrap_reps}; the "
                "standard error is left at its analytic fallback. Check "
                "common support and the bandwidth.",
                UserWarning,
                stacklevel=3,
            )
            return float("nan"), info

        if n_failed > 0.1 * self.bootstrap_reps:
            warnings.warn(
                f"sp.match: {n_failed} of {self.bootstrap_reps} bootstrap "
                "replications failed to produce an estimate. The standard "
                "error is computed from the survivors and may be optimistic.",
                UserWarning,
                stacklevel=3,
            )

        arr = np.asarray(estimates, dtype=float)
        info["bootstrap_bias"] = float(np.mean(arr)) - float(self._point_estimate)
        return float(np.std(arr, ddof=1)), info

    def _resolve_se_method(self) -> str:
        """Resolve ``se_method='auto'`` to a concrete estimator.

        Nearest-neighbour matching keeps the historical Abadie-Imbens
        matched-pair SE; kernel / radius matching (which has no matched-pair
        structure) uses Stata psmatch2's analytic SE.
        """
        if self.se_method != "auto":
            return self.se_method
        if self.method == "nearest":
            # Abadie-Imbens (2006) is the only estimator measured to be
            # correctly sized here (0.95-1.04x the sampling SD, coverage
            # 0.905-0.956 over 36 designs x 1000 reps; see
            # benchmarks/matching_se_coverage.py). The historical default
            # 'ai' never reached nominal coverage on that grid.
            return "abadie_imbens"
        if self.method == "llr":
            if self.llr_stata_compat:
                # The compat path is ordinary nearest-neighbour matching on a
                # smoothed outcome, so the weights are non-negative and the
                # analytic formula applies -- which is what Stata reports.
                return "psmatch2"
            # Stata psmatch2 reports seatt = . for genuine LLR: the analytic
            # formula assumes non-negative matching weights, which local
            # linear weights violate. Bootstrap is the honest default.
            return "bootstrap"
        if self.method in ("kernel", "radius"):
            return "psmatch2"
        return "ai"

    # ==================================================================
    # Kernel / radius matching (Heckman-Ichimura-Todd 1997; psmatch2)
    # ==================================================================

    def _fit_kernel(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        idx_t: np.ndarray,
        idx_c: np.ndarray,
    ) -> Tuple[float, float, pd.DataFrame, dict[str, Any]]:
        """Kernel / radius propensity-score matching (Stata psmatch2).

        Each treated unit is matched to *all* on-support controls, weighted
        by a kernel of the propensity-score distance::

            w_ij = K(|p_i - p_j| / h) / Σ_k K(|p_i - p_k| / h)

        so the controls' contributions to a treated unit sum to 1.  The
        matched-control mean outcome is ``_y_i = Σ_j w_ij Y_j`` and
        ``ATT = mean_i (Y_i - _y_i)`` over the on-support treated.  A treated
        unit with no control inside the bandwidth (all kernel weights zero)
        is dropped off support.

        ``method='radius'`` is the special case of a uniform kernel with the
        bandwidth set to ``caliper`` (Stata: "radius matching is like kernel
        matching with a uniform kernel").
        """
        pscore = self._logit_propensity(X, T, poly=self.ps_poly)

        # Common-support trimming (Stata `common`) precedes kernel matching.
        support = common_support_mask(pscore, T, rule=self.common_support)

        kerneltype = "uniform" if self.method == "radius" else self.kernel
        if self.method == "radius":
            assert self.caliper is not None
            bw = float(self.caliper)
        else:
            bw = float(self.bwidth)

        is_llr = self.method == "llr"
        if is_llr and kerneltype == "epan":
            # Stata psmatch2 does NOT run local linear regression here: with
            # the (default) Epanechnikov kernel and a propensity metric it
            # rewrites the request as nearest-neighbour matching on an
            # lpoly-smoothed outcome (psmatch2.ado, "do nearest neighbor if
            # llr with tricube").  The two produce materially different
            # numbers.  We run the estimator the option names, and say so.
            warnings.warn(
                "sp.match(method='llr', kernel='epan'): Stata psmatch2 does "
                "not perform local linear regression matching for this "
                "combination -- it silently substitutes nearest-neighbour "
                "matching on an lpoly-smoothed outcome, so its ATT will "
                "differ. StatsPAI runs genuine LLR. To reproduce a Stata "
                "`psmatch2 ..., llr` number exactly, use kernel='tricube' "
                "(or another non-Epanechnikov kernel), which is the only way "
                "psmatch2 reaches its own LLR routine.",
                UserWarning,
                stacklevel=3,
            )

        ps_c = pscore[idx_c]
        Y_c = Y[idx_c]
        # On-support controls form the donor pool (controls are always on
        # support under psmatch2's comsup, but we honour the flag anyway).
        c_on = support[idx_c]
        pool = np.where(c_on)[0]  # positions into idx_c

        full_matches = [np.array([], dtype=int) for _ in idx_t]
        full_weights = [np.array([], dtype=float) for _ in idx_t]
        effects = []
        for i, t_pos in enumerate(idx_t):
            if not support[t_pos] or len(pool) == 0:
                support[t_pos] = False
                continue
            d = np.abs(pscore[t_pos] - ps_c[pool]) / bw
            k = _kernel_weight(d, kerneltype)
            ksum = k.sum()
            if ksum <= 0:
                # No control within the bandwidth -> drop off support.
                support[t_pos] = False
                continue
            nz = k > 0
            w = k[nz] / ksum
            cpos = pool[nz]  # positions into idx_c

            if is_llr:
                # Local linear regression matching (Heckman, Ichimura & Todd
                # 1997), reproducing psmatch2's `_Match_llr`.  With kernel
                # weights K normalised to sum 1 and signed propensity gaps
                # d_j = p_j - p_i,
                #
                #     w_ij = K_j (V + dbar^2 - dbar * d_j) / V,
                #     dbar = sum_j K_j d_j,  V = sum_j K_j d_j^2 - dbar^2
                #
                # which satisfies sum_j w_ij = 1 and sum_j w_ij d_j = 0 — the
                # local-linear property that makes the estimate exact for a
                # linear conditional mean.  Note w_ij may be NEGATIVE; that is
                # inherent to local linear weighting, and it is why the
                # psmatch2 analytic SE does not apply.
                d_signed = ps_c[cpos] - pscore[t_pos]
                dbar = float(np.sum(w * d_signed))
                v = float(np.sum(w * d_signed**2) - dbar**2)
                # V == 0 means the local design is collinear (every donor at
                # the same propensity score), so no linear term is
                # identified.  Stata's `cap reg` fails and it drops the unit
                # off support; do the same rather than dividing by ~0.
                if not np.isfinite(v) or v <= _LLR_MIN_VARIANCE:
                    support[t_pos] = False
                    continue
                w = w * (v + dbar**2 - dbar * d_signed) / v

            full_matches[i] = cpos
            full_weights[i] = w
            yhat = float(np.sum(w * Y_c[cpos]))
            effects.append(Y[t_pos] - yhat)

        if len(effects) == 0:
            raise DataInsufficient(
                f"match: {self.method} matching found no treated unit with a "
                f"control within bandwidth {bw}.",
                recovery_hint="Increase bwidth/caliper or inspect common support.",
                diagnostics={"method": self.method, "bandwidth": bw},
            )

        att = float(np.mean(effects))
        # Placeholder SE; fit() replaces it with the psmatch2 analytic SE
        # (se_method resolves to 'psmatch2' for kernel/radius).
        se = (
            float(np.std(effects, ddof=1) / np.sqrt(len(effects)))
            if len(effects) > 1
            else 0.0
        )

        balance = self._balance_table(X, T, pscore)

        self._assignment = {
            "pscore": pscore,
            "treated": T,
            "idx_t": idx_t,
            "idx_c": idx_c,
            "matches": full_matches,
            "weights": full_weights,
            "support": support,
            "outcome": Y,
            "neighbors": False,
        }

        extra = {
            "kernel": kerneltype,
            "bwidth": bw,
            "n_on_support": int(np.sum(support)),
            "n_matched_treated": int(np.sum([len(m) > 0 for m in full_matches])),
        }
        return att, se, balance, extra

    @staticmethod
    def _stable_index_order(index: Any) -> np.ndarray:
        """Numeric rank of DataFrame index labels for deterministic tie-breaking."""
        labels = np.asarray(pd.Index(index))
        n = len(labels)
        try:
            order = np.argsort(labels, kind="mergesort")
        except TypeError:
            order = np.array(
                sorted(
                    range(n),
                    key=lambda i: (
                        type(labels[i]).__name__,
                        repr(labels[i]),
                        i,
                    ),
                ),
                dtype=int,
            )
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        return np.asarray(ranks, dtype=float)

    def _compute_distance_matrix(
        self,
        X: np.ndarray,
        idx_from: np.ndarray,
        idx_to: np.ndarray,
        pscore: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute distance matrix between two groups."""
        X_from = X[idx_from]
        X_to = X[idx_to]

        if self.distance == "propensity":
            # Use pre-estimated propensity scores (estimated once with actual T)
            if pscore is None:
                raise ValueError("pscore is required when distance='propensity'")
            ps_from = pscore[idx_from].reshape(-1, 1)
            ps_to = pscore[idx_to].reshape(-1, 1)
            return np.asarray(cdist(ps_from, ps_to, metric="euclidean"), dtype=float)

        elif self.distance == "mahalanobis":
            cov = self._mahalanobis_cov_matrix(X)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            try:
                VI = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                VI = np.linalg.pinv(cov)
            return np.asarray(
                cdist(X_from, X_to, metric="mahalanobis", VI=VI),
                dtype=float,
            )

        else:  # euclidean (normalized)
            sd = np.std(X, axis=0, ddof=1)
            sd[sd == 0] = 1.0
            return np.asarray(
                cdist(X_from / sd, X_to / sd, metric="euclidean"),
                dtype=float,
            )

    # ==================================================================
    # Exact matching
    # ==================================================================

    def _fit_exact(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        idx_t: np.ndarray,
        idx_c: np.ndarray,
    ) -> Tuple[float, float, pd.DataFrame, dict[str, Any]]:
        """Exact matching: only match units with identical covariate values."""
        # Build string keys for each observation
        keys_t = self._covariate_keys(X, idx_t)
        keys_c = self._covariate_keys(X, idx_c)

        # Index control units by key
        control_map: dict[Tuple[Any, ...], List[int]] = {}
        for i, key in enumerate(keys_c):
            control_map.setdefault(key, []).append(i)

        effects = []
        n_matched = 0
        for i, key in enumerate(keys_t):
            if key not in control_map:
                continue
            c_indices = control_map[key]
            y_t = Y[idx_t[i]]
            y_c_mean = np.mean(Y[idx_c[c_indices]])
            effects.append(y_t - y_c_mean)
            n_matched += 1

        if n_matched == 0:
            raise DataInsufficient(
                "match: exact matching found no treated units with exact matches.",
                recovery_hint=(
                    "Consider distance='mahalanobis', method='cem', or coarser "
                    "covariates."
                ),
            )

        att = float(np.mean(effects))
        se = (
            float(np.std(effects, ddof=1) / np.sqrt(n_matched))
            if n_matched > 1
            else 0.0
        )

        pscore = self._logit_propensity(X, T, poly=self.ps_poly)
        balance = self._balance_table(X, T, pscore)
        extra = {
            "n_matched_treated": n_matched,
            "n_unmatched_treated": len(keys_t) - n_matched,
        }
        return att, se, balance, extra

    @staticmethod
    def _covariate_keys(
        X: np.ndarray,
        indices: np.ndarray,
    ) -> List[Tuple[Any, ...]]:
        """Create hashable keys for exact matching."""
        return [tuple(X[i]) for i in indices]

    # ==================================================================
    # Subclassification / propensity score stratification
    # ==================================================================

    def _fit_stratify(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        idx_t: np.ndarray,
        idx_c: np.ndarray,
    ) -> Tuple[float, float, pd.DataFrame, dict[str, Any]]:
        """
        Propensity score stratification (Rosenbaum & Rubin 1984).

        Partition the sample into strata by propensity score quantiles,
        compute within-stratum treatment effects, then weight by the
        proportion of treated (ATT) or total (ATE) units per stratum.
        """
        pscore = self._logit_propensity(X, T, poly=self.ps_poly)

        # Create strata from propensity score quantiles
        boundaries = np.quantile(pscore, np.linspace(0, 1, self.n_strata + 1))
        boundaries[0] -= 1e-6
        boundaries[-1] += 1e-6
        strata = np.digitize(pscore, boundaries) - 1
        strata = np.clip(strata, 0, self.n_strata - 1)

        # Collect per-stratum effects, weights, and variance components
        strata_results = []  # list of (tau, weight, var_t, var_c)

        for s in range(self.n_strata):
            in_s = strata == s
            t_in = in_s & (T == 1)
            c_in = in_s & (T == 0)
            n_t_s = t_in.sum()
            n_c_s = c_in.sum()

            if n_t_s == 0 or n_c_s == 0:
                continue

            tau_s = Y[t_in].mean() - Y[c_in].mean()

            if self.estimand == "ATT":
                w_s = float(n_t_s)
            else:
                w_s = float(n_t_s + n_c_s)

            # Within-stratum variance components
            vt = np.var(Y[t_in], ddof=1) / n_t_s if n_t_s >= 2 else 0.0
            vc = np.var(Y[c_in], ddof=1) / n_c_s if n_c_s >= 2 else 0.0

            strata_results.append((tau_s, w_s, vt, vc))

        if len(strata_results) == 0:
            raise DataInsufficient(
                "match: no strata contain both treated and control units"
            )

        effects = np.array([r[0] for r in strata_results])
        raw_weights = np.array([r[1] for r in strata_results])
        weights = raw_weights / raw_weights.sum()

        att = float(effects @ weights)

        # SE: sum of weighted within-stratum sampling variances
        within_var = 0.0
        for (_, _, vt, vc), w_s in zip(strata_results, weights):
            within_var += w_s**2 * (vt + vc)

        se = float(np.sqrt(within_var))

        balance = self._balance_table(X, T, pscore)

        # Record the cell structure so fit() can build the matched frame.
        # A unit is "kept" iff its stratum contained both arms, i.e. iff it
        # entered one of the within-stratum comparisons above.
        keep = np.zeros(len(T), dtype=bool)
        for s in range(self.n_strata):
            in_s = strata == s
            if (in_s & (T == 1)).any() and (in_s & (T == 0)).any():
                keep |= in_s
        self._stratum_assignment = {
            "pscore": pscore,
            "treated": T,
            "stratum": strata,
            "keep": keep,
            "outcome": Y,
        }

        extra = {
            "n_strata": self.n_strata,
            "n_effective_strata": len(strata_results),
        }
        return att, se, balance, extra

    # ==================================================================
    # CEM
    # ==================================================================

    def _fit_cem(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        idx_t: np.ndarray,
        idx_c: np.ndarray,
    ) -> Tuple[float, float, pd.DataFrame, dict[str, Any]]:
        """Coarsened Exact Matching (Iacus, King & Porro 2012)."""
        n, k = X.shape

        # Coarsen each covariate
        n_bins = self.n_bins
        if n_bins is None:
            n_bins = max(int(np.ceil(np.log2(n) + 1)), 3)  # Sturges' rule

        strata = np.zeros(n, dtype=object)
        for j in range(k):
            col = X[:, j]
            bins = np.linspace(col.min() - 1e-10, col.max() + 1e-10, n_bins + 1)
            digitized = np.digitize(col, bins)
            if j == 0:
                strata = digitized.astype(str)
            else:
                strata = np.char.add(np.char.add(strata, "_"), digitized.astype(str))

        # Match within strata
        matched_t = []
        matched_c = []
        weights_c = []

        for s in np.unique(strata):
            in_s = strata == s
            t_in = np.where(in_s & (T == 1))[0]
            c_in = np.where(in_s & (T == 0))[0]
            if len(t_in) > 0 and len(c_in) > 0:
                matched_t.extend(t_in.tolist())
                matched_c.extend(c_in.tolist())
                w = len(t_in) / len(c_in)
                weights_c.extend([w] * len(c_in))

        if len(matched_t) == 0:
            raise DataInsufficient(
                "match: CEM found no strata with both treated and control units"
            )

        Y_t = Y[matched_t]
        Y_c = Y[matched_c]
        w_c = np.array(weights_c)

        att = float(np.mean(Y_t) - np.average(Y_c, weights=w_c))

        var_t = np.var(Y_t, ddof=1) / len(Y_t) if len(Y_t) > 1 else 0
        var_c = (
            (
                np.average((Y_c - np.average(Y_c, weights=w_c)) ** 2, weights=w_c)
                / len(Y_c)
            )
            if len(Y_c) > 1
            else 0
        )
        se = float(np.sqrt(var_t + var_c))

        pscore = self._logit_propensity(X, T, poly=self.ps_poly)
        balance = self._balance_table(X, T, pscore)

        # Record the coarsened cells so fit() can build the matched frame.
        # `keep` marks the units in cells that held both arms — exactly the
        # units that entered the comparison above.
        keep = np.zeros(n, dtype=bool)
        keep[np.asarray(matched_t, dtype=int)] = True
        keep[np.asarray(matched_c, dtype=int)] = True
        self._stratum_assignment = {
            "pscore": pscore,
            "treated": T,
            "stratum": strata,
            "keep": keep,
            "outcome": Y,
        }

        n_matched_t = len(set(matched_t))
        extra = {
            "n_matched_treated": n_matched_t,
            "n_matched_control": len(set(matched_c)),
            "n_unmatched_treated": len(idx_t) - n_matched_t,
            "n_bins": n_bins,
        }
        return att, se, balance, extra

    # ==================================================================
    # Propensity score estimation
    # ==================================================================

    @staticmethod
    def _expand_poly(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expand covariate matrix with polynomial and interaction terms.

        - degree=1: linear terms only (identity).
        - degree=2: add X^2 and all pairwise X_i * X_j interactions.
        - degree=3: add X^3 as well.

        This follows the common practice in propensity score estimation
        of including higher-order terms (Cunningham 2021, Ch. 5;
        Dehejia & Wahba 1999).
        """
        if degree <= 1:
            return X
        cols = [X]
        n, k = X.shape
        # Squared terms
        cols.append(X**2)
        # Pairwise interactions
        if k > 1:
            for i in range(k):
                for j in range(i + 1, k):
                    cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        # Cubic terms
        if degree >= 3:
            cols.append(X**3)
        return np.asarray(np.column_stack(cols), dtype=float)

    @staticmethod
    def _logit_propensity(
        X: np.ndarray,
        T: np.ndarray,
        poly: int = 1,
    ) -> np.ndarray:
        """
        Logistic regression propensity score via Newton-Raphson (IRLS).

        Parameters
        ----------
        X : ndarray, shape (n, k)
        T : ndarray, shape (n,)
        poly : int, default 1
            Polynomial expansion degree for the design matrix.
            ``poly=2`` adds squared terms and pairwise interactions,
            following the standard specification in Cunningham (2021, Ch. 5)
            and Dehejia & Wahba (1999).
        """
        X_poly = MatchEstimator._expand_poly(X, poly)
        n = X_poly.shape[0]
        X_aug = np.column_stack([np.ones(n), X_poly])
        k_aug = X_aug.shape[1]

        beta = np.zeros(k_aug)
        for _ in range(25):
            linear = np.clip(X_aug @ beta, -500, 500)
            p = 1 / (1 + np.exp(-linear))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            # Vectorized IRLS: W is diagonal, so X'WX = (X * w)' X
            w = p * (1 - p)
            grad = X_aug.T @ (T - p)
            H = (X_aug * w[:, None]).T @ X_aug
            try:
                delta = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H, grad, rcond=None)[0]
            beta += delta
            if np.max(np.abs(delta)) < 1e-8:
                break

        linear = np.clip(X_aug @ beta, -500, 500)
        pscore = 1 / (1 + np.exp(-linear))
        return np.asarray(np.clip(pscore, 1e-6, 1 - 1e-6), dtype=float)

    # ==================================================================
    # NN matching helpers
    # ==================================================================

    def _ai_population_se(
        self,
        Y: np.ndarray,
        matches: List[np.ndarray],
        weights: List[np.ndarray],
    ) -> float:
        """Abadie-Imbens (2006) *population* ATT variance.

        This is the estimand ``Matching::Match`` reports by default
        (``Var.calc = 0``, ``sample = FALSE``). Writing ``tau_i`` for the
        matched effect of treated unit *i*, ``K_j`` for the total matching
        weight control *j* receives and ``KK_j`` for the sum of its squared
        weights::

            sigma2 = 0.5 * sum_ik w_ik (Y_i - Y_c_ik - tau)^2 / sum_ik w_ik
            V      = [ sigma2 * sum_j (K_j^2 - KK_j)
                       + sum_i (tau_i - tau)^2 ] / N_1^2

        The first term is the penalty for re-using controls (it vanishes when
        every control is used at most once); the second is the contribution of
        treatment-effect heterogeneity, which is what makes this a
        *population* rather than a *sample* variance.

        This is a different estimand from ``se_method='abadie_imbens'``, which
        reproduces Stata's ``psmatch2 , ai()``: that one targets the *sample*
        ATT and lets sigma^2(X) vary with the covariates. Neither is more
        correct; they answer different questions, and on ``MatchIt::lalonde``
        they differ by ~9%.

        References
        ----------
        Abadie, A. and Imbens, G.W. (2006). Large Sample Properties of
            Matching Estimators for Average Treatment Effects.
            *Econometrica*, 74(1), 235-267.
        """
        if self._assignment is None:  # pragma: no cover - defensive
            return float("nan")
        idx_target = np.asarray(self._assignment["idx_t"], dtype=int)
        idx_pool = np.asarray(self._assignment["idx_c"], dtype=int)
        Y = np.asarray(Y, dtype=float)

        K = np.zeros(len(idx_pool))
        KK = np.zeros(len(idx_pool))
        tau_i: List[float] = []
        pair_resid: List[np.ndarray] = []
        pair_w: List[np.ndarray] = []

        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            w = np.asarray(w, dtype=float)
            y_t = Y[idx_target[i]]
            y_c = Y[idx_pool[m]]
            tau_i.append(float(y_t - np.sum(w * y_c)))
            np.add.at(K, m, w)
            np.add.at(KK, m, w**2)
            pair_resid.append(y_t - y_c)
            pair_w.append(w)

        n1 = len(tau_i)
        if n1 < 2:
            return float("nan")
        tau = np.asarray(tau_i, dtype=float)
        tau_hat = float(tau.mean())

        resid = np.concatenate(pair_resid) - tau_hat
        wcat = np.concatenate(pair_w)
        sigma2 = 0.5 * float(np.sum(wcat * resid**2) / np.sum(wcat))

        reuse = float(np.sum(K**2 - KK))
        hetero = float(np.sum((tau - tau_hat) ** 2))
        var = (sigma2 * reuse + hetero) / (n1**2)
        return float(np.sqrt(var)) if var >= 0 else float("nan")

    def _tie_scale(self) -> float:
        """Variance the squared distances are divided by in the tie test.

        For a propensity-score distance this is ``var(pscore)``, which makes
        ``tie_tolerance`` directly comparable to ``Matching::Match``'s
        ``distance.tolerance`` (that package weights the squared distance by
        the inverse covariance of the matching variables). Other distances are
        already standardised, so the scale is 1.
        """
        if self.distance == "propensity" and self._pscore_cache is not None:
            v = float(np.var(np.asarray(self._pscore_cache, dtype=float), ddof=1))
            return v if v > 0 else 1.0
        return 1.0

    def _resolve_caliper(self, pscore: Optional[np.ndarray]) -> Optional[float]:
        """Caliper width on the distance scale.

        ``caliper_scale='raw'`` (default, Stata ``psmatch2 , caliper()``)
        uses the value as given. ``'sd'`` multiplies it by the standard
        deviation of the distance measure, which is the ``MatchIt``
        ``std.caliper = TRUE`` convention.
        """
        if self.caliper is None:
            return None
        if self.caliper_scale == "raw":
            return float(self.caliper)
        if pscore is None:
            raise MethodIncompatibility(
                "match: caliper_scale='sd' scales the caliper by the "
                "standard deviation of the propensity score, which "
                f"distance={self.distance!r} does not produce.",
                recovery_hint=(
                    "Use caliper_scale='raw' with a non-propensity "
                    "distance, or set distance='propensity'."
                ),
                diagnostics={"distance": self.distance},
            )
        sd = float(np.std(np.asarray(pscore, dtype=float), ddof=1))
        return float(self.caliper) * sd

    def _mahalanobis_cov_matrix(self, X: np.ndarray) -> np.ndarray:
        """Covariance matrix defining the Mahalanobis metric.

        ``'pooled'`` (default) is the pooled *within-group* covariance
        ``S_w = [(n1-1)S1 + (n0-1)S0] / (n1 + n0 - 2)``, i.e. the metric of
        Rubin (1980) that ``MatchIt`` also uses. ``'total'`` is the
        full-sample covariance.

        The distinction matters: the full-sample covariance is inflated
        along the direction in which the groups' means differ, which is
        exactly the direction matching needs to resolve most finely, so
        using it shrinks the distances that should discriminate most.
        """
        T = self.data[self.treat].to_numpy()
        if self.mahalanobis_cov == "total":
            return np.atleast_2d(np.cov(X.T))
        x1 = X[T == 1]
        x0 = X[T == 0]
        n1, n0 = len(x1), len(x0)
        if n1 < 2 or n0 < 2:  # pragma: no cover - guarded upstream
            return np.atleast_2d(np.cov(X.T))
        s1 = np.atleast_2d(np.cov(x1.T))
        s0 = np.atleast_2d(np.cov(x0.T))
        return ((n1 - 1) * s1 + (n0 - 1) * s0) / (n1 + n0 - 2)

    def _target_processing_order(
        self,
        dist: np.ndarray,
        target_order: np.ndarray,
        pscore_target: Optional[np.ndarray],
    ) -> np.ndarray:
        """Static processing order for without-replacement greedy matching.

        Returns ``None`` for the dynamic orders ('closest' / 'farthest'),
        which cannot be precomputed because each assignment changes which
        controls remain available.
        """
        mo = self.m_order
        if mo in ("closest", "farthest"):
            return None
        if mo == "data":
            return np.argsort(target_order, kind="stable")
        if mo == "smallest_min_dist":
            return np.lexsort((target_order, np.min(dist, axis=1)))
        if mo in ("smallest", "largest"):
            if pscore_target is None:
                raise MethodIncompatibility(
                    f"match: m_order={mo!r} orders treated units by the "
                    "propensity score, which requires distance='propensity'.",
                    recovery_hint=(
                        "Use m_order='closest'/'farthest'/'data' with a "
                        "non-propensity distance, or set distance='propensity'."
                    ),
                    diagnostics={"m_order": mo, "distance": self.distance},
                )
            key = pscore_target if mo == "smallest" else -pscore_target
            return np.lexsort((target_order, key))
        raise ValueError(f"match: unknown m_order {mo!r}")  # pragma: no cover

    def _nn_match_from_dist(
        self,
        dist: np.ndarray,
        caliper: Optional[float] = None,
        target_order: Optional[np.ndarray] = None,
        pool_order: Optional[np.ndarray] = None,
        pscore_target: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        k-NN matching from a precomputed distance matrix.

        When ``self.replace=False``, each control unit can be used at most
        once across all treated units.  Treated units are processed in
        order of their minimum distance (best match first) so the greedy
        assignment favours the closest pairs.

        Equal-distance ties are resolved by the source DataFrame index:
        lower-index pool units are selected first, and without-replacement
        target processing falls back to target index order. This makes
        matching deterministic across BLAS/NumPy backends and independent of
        incidental row order when index labels preserve unit identity.

        References: Cunningham (2021, Ch. 5) discusses with- vs.
        without-replacement matching and the bias–variance trade-off.
        """
        n_target = dist.shape[0]
        matches: List[np.ndarray] = [np.array([], dtype=int) for _ in range(n_target)]
        weights: List[np.ndarray] = [np.array([], dtype=float) for _ in range(n_target)]
        if target_order is None:
            target_order = np.arange(n_target, dtype=float)
        if pool_order is None:
            pool_order = np.arange(dist.shape[1], dtype=float)

        def _nearest_indices(d: np.ndarray, k: int) -> np.ndarray:
            finite = np.isfinite(d)
            if not np.any(finite):
                return np.array([], dtype=int)
            candidates = np.where(finite)[0]
            order = np.lexsort((pool_order[candidates], d[candidates]))
            return np.asarray(candidates[order[:k]], dtype=int)

        # Without replacement each control can be used once, so the result
        # depends on the order treated units are processed in -- materially
        # so (see the `m_order` documentation on `sp.match`). The order is
        # a user choice, not an implementation detail.
        if not self.replace:
            used: set[int] = set()
            static_order = self._target_processing_order(
                dist, target_order, pscore_target
            )

            def _assign(i: int) -> None:
                d = dist[i].copy()
                if caliper is not None:
                    d[d > caliper] = np.inf
                for u in used:
                    d[u] = np.inf
                k = min(self.n_matches, int(np.sum(np.isfinite(d))))
                if k == 0:
                    matches[i] = np.array([], dtype=int)
                    weights[i] = np.array([], dtype=float)
                    return
                idx = _nearest_indices(d, k)
                matches[i] = idx
                weights[i] = np.ones(k, dtype=float) / k
                used.update(idx.tolist())

            if static_order is not None:
                for i in static_order:
                    _assign(int(i))
                return matches, weights

            # Dynamic orders: at every step re-evaluate which *remaining*
            # treated unit currently has the closest (or farthest) available
            # control, and match that one next.
            remaining = set(range(n_target))
            take_min = self.m_order == "closest"
            while remaining:
                best_i = None
                best_d = None
                for i in sorted(remaining):
                    d = dist[i].copy()
                    if caliper is not None:
                        d[d > caliper] = np.inf
                    for u in used:
                        d[u] = np.inf
                    if not np.any(np.isfinite(d)):
                        continue
                    dv = float(np.min(d))
                    if best_d is None or (dv < best_d if take_min else dv > best_d):
                        best_d, best_i = dv, i
                if best_i is None:
                    break
                _assign(best_i)
                remaining.discard(best_i)
            for i in remaining:
                matches[i] = np.array([], dtype=int)
                weights[i] = np.array([], dtype=float)

            return matches, weights

        # With replacement (default): simple k-NN per target
        for i in range(n_target):
            d = dist[i].copy()
            if caliper is not None:
                d[d > caliper] = np.inf

            k = min(self.n_matches, int(np.sum(np.isfinite(d))))
            if k == 0:
                matches[i] = np.array([], dtype=int)
                weights[i] = np.array([], dtype=float)
                continue

            idx = _nearest_indices(d, k)
            if self.ties == "all":
                idx = self._extend_with_ties(d, idx)
            matches[i] = idx
            weights[i] = np.ones(len(idx), dtype=float) / len(idx)

        return matches, weights

    def _extend_with_ties(self, d: np.ndarray, idx: np.ndarray) -> np.ndarray:
        """Add every control tied with the k-th nearest already selected.

        Dropping a control equidistant with one that was kept makes the
        estimate depend on row order, which is why ``Matching::Match`` pools
        them by default. The comparison is on *squared* distances scaled by
        ``_tie_scale`` (the variance of the distance measure), and the pooled
        units share the target's weight equally.
        """
        if idx.size == 0:
            return idx
        cutoff = float(d[idx[-1]])
        if not np.isfinite(cutoff):  # pragma: no cover - defensive
            return idx
        scale = self._tie_scale()
        d2 = (d**2) / scale
        tied = np.flatnonzero(
            np.isfinite(d) & (d2 <= (cutoff**2) / scale + self.tie_tolerance)
        )
        if tied.size <= idx.size:
            return idx
        return np.asarray(np.union1d(idx, tied), dtype=int)

    # ==================================================================
    # Effect computation (with optional bias correction)
    # ==================================================================

    def _compute_effect(
        self,
        Y: np.ndarray,
        idx_target: np.ndarray,
        idx_pool: np.ndarray,
        X: np.ndarray,
        matches: List[np.ndarray],
        weights: List[np.ndarray],
    ) -> float:
        """
        Compute matching estimate, optionally with Abadie-Imbens (2011)
        bias correction.

        Bias correction estimates mu_0(x) via OLS on the matched control
        group, then adjusts each matched pair:
            tau_i^BC = (Y_i - Y_j) - (mu_hat(X_i) - mu_hat(X_j))
        """
        effects: List[float] = []
        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            y_target = Y[idx_target[i]]
            y_matched = Y[idx_pool[m]]
            effects.append(y_target - np.average(y_matched, weights=w))

        if len(effects) == 0:
            return 0.0

        raw_att = float(np.mean(effects))

        if not self.bias_correction:
            return raw_att

        # --- Abadie-Imbens (2011) bias correction ---
        # Estimate conditional mean function on pool group via OLS
        X_pool = X[idx_pool]
        Y_pool = Y[idx_pool]
        X_pool_aug = np.column_stack([np.ones(len(X_pool)), X_pool])

        try:
            beta_pool = np.linalg.lstsq(X_pool_aug, Y_pool, rcond=None)[0]
        except np.linalg.LinAlgError:
            return raw_att  # fallback to uncorrected

        # Compute bias correction for each matched pair
        corrections = []
        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            x_target = np.concatenate([[1], X[idx_target[i]]])
            x_matched = np.column_stack([np.ones(len(m)), X[idx_pool[m]]])
            mu_target = x_target @ beta_pool
            mu_matched = np.average(x_matched @ beta_pool, weights=w)
            corrections.append(mu_target - mu_matched)

        if len(corrections) == 0:
            return raw_att

        bias = float(np.mean(corrections))
        return raw_att - bias

    # ==================================================================
    # Standard errors
    # ==================================================================

    def _ai_se(
        self,
        Y: np.ndarray,
        X: np.ndarray,
        T: np.ndarray,
        idx_t: np.ndarray,
        idx_c: np.ndarray,
        matches: List[np.ndarray],
        weights: List[np.ndarray],
    ) -> float:
        """Simple matched-pair standard error.

        Returns ``std(per-pair treatment effects, ddof=1) / sqrt(n_pairs)``.

        NOTE: despite the historical ``'ai'`` label this is NOT the
        Abadie-Imbens (2006) variance -- it treats matched pairs as
        independent and ignores the extra variance from reusing controls
        under matching *with replacement*, so it is anti-conservative
        (empirically ~0.68x the true sampling SD; ~81% coverage at a nominal
        95% level). For the rigorous Abadie-Imbens (2006) conditional-variance
        standard error pass ``se_method='abadie_imbens'`` (implemented in
        ``matching/_matched_frame.py``).
        """
        effects = []
        for i, (m, w) in enumerate(zip(matches, weights)):
            if len(m) == 0:
                continue
            y_t = Y[idx_t[i]]
            y_c = Y[idx_c[m]]
            effects.append(float(y_t - np.average(y_c, weights=w)))

        if len(effects) < 2:
            return 0.0

        effects_arr = np.asarray(effects, dtype=float)
        n_eff = len(effects_arr)
        return float(np.std(effects_arr, ddof=1) / np.sqrt(n_eff))

    # ==================================================================
    # Balance diagnostics
    # ==================================================================

    def _balance_table(
        self,
        X: np.ndarray,
        T: np.ndarray,
        pscore: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Standardized mean differences (SMD) before matching."""
        idx_t = T == 1
        idx_c = T == 0
        rows = []

        for j, name in enumerate(self.covariates):
            x_t = X[idx_t, j]
            x_c = X[idx_c, j]
            mean_t = np.mean(x_t)
            mean_c = np.mean(x_c)
            sd_pool = np.sqrt((np.var(x_t, ddof=1) + np.var(x_c, ddof=1)) / 2)
            smd = (mean_t - mean_c) / sd_pool if sd_pool > 0 else 0
            rows.append(
                {
                    "variable": name,
                    "mean_treated": round(mean_t, 4),
                    "mean_control": round(mean_c, 4),
                    "smd": round(smd, 4),
                }
            )

        if pscore is not None:
            ps_t = pscore[idx_t]
            ps_c = pscore[idx_c]
            sd_ps = np.sqrt((np.var(ps_t, ddof=1) + np.var(ps_c, ddof=1)) / 2)
            smd_ps = (np.mean(ps_t) - np.mean(ps_c)) / sd_ps if sd_ps > 0 else 0
            rows.append(
                {
                    "variable": "propensity_score",
                    "mean_treated": round(float(np.mean(ps_t)), 4),
                    "mean_control": round(float(np.mean(ps_c)), 4),
                    "smd": round(float(smd_ps), 4),
                }
            )

        return pd.DataFrame(rows)


# ======================================================================
# Citation
# ======================================================================

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------


def balanceplot(
    result: CausalResult,
    threshold: float = 0.1,
    ax: Any = None,
    figsize: tuple = (8, None),
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Love plot: covariate balance visualization (SMD dot plot).

    Displays standardized mean differences (SMD) for each covariate.
    The standard threshold for good balance is |SMD| < 0.1.

    Parameters
    ----------
    result : CausalResult
        Result from ``match()`` or ``ebalance()``.
    threshold : float, default 0.1
        SMD threshold lines.
    ax : matplotlib Axes, optional
    figsize : tuple
        Height auto-scales with number of covariates if None.
    title : str, optional

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> age = rng.normal(40, 8, n)
    >>> edu = rng.normal(12, 2, n)
    >>> ps = 1 / (1 + np.exp(-(0.05 * (age - 40) + 0.1 * (edu - 12))))
    >>> training = rng.binomial(1, ps)
    >>> wage = 20 + 0.3 * age + 0.5 * edu + 4.0 * training + rng.normal(0, 3, n)
    >>> df = pd.DataFrame({"wage": wage, "training": training,
    ...                    "age": age, "edu": edu})
    >>> result = sp.match(df, y="wage", treat="training",
    ...                   covariates=["age", "edu"])
    >>> fig, ax = sp.balanceplot(result)
    >>> fig.savefig("balance.png")  # doctest: +SKIP
    >>> type(ax).__name__
    'Axes'
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    balance = result.detail
    if balance is None or "smd" not in balance.columns:
        raise ValueError("No balance table. Use match() result.")

    n_vars = len(balance)
    if figsize[1] is None:
        figsize = (figsize[0], max(4, n_vars * 0.4 + 1))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    variables = balance["variable"].values
    smd = balance["smd"].values
    y_pos = np.arange(n_vars)

    # Color by balance quality
    colors = ["#27AE60" if abs(s) < threshold else "#E74C3C" for s in smd]

    ax.scatter(smd, y_pos, c=colors, s=60, zorder=5, edgecolors="white", linewidth=0.5)
    ax.barh(y_pos, smd, height=0.02, color="#BDC3C7", zorder=2)

    # Threshold lines
    ax.axvline(x=threshold, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=-threshold, color="#E74C3C", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables, fontsize=10)
    ax.set_xlabel("Standardized Mean Difference (SMD)", fontsize=11)
    ax.set_title(title or "Covariate Balance (Love Plot)", fontsize=13)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig, ax


def psplot(
    data: pd.DataFrame,
    treat: str,
    covariates: List[str],
    *,
    n_bins: int = 40,
    ax: Any = None,
    figsize: tuple = (8, 5),
    title: Optional[str] = None,
    labels: tuple = ("Control", "Treated"),
    colors: tuple = ("#3498DB", "#E74C3C"),
    trim: Optional[float] = None,
) -> Tuple[Any, Any]:
    """
    Propensity score distribution plot (common support diagnostic).

    Overlays histograms of the estimated propensity score for treated
    and control groups, so the user can visually assess whether the
    common support (overlap) assumption holds.

    Parameters
    ----------
    data : pd.DataFrame
    treat : str
        Binary treatment column.
    covariates : list of str
        Covariates used to estimate the propensity score.
    n_bins : int, default 40
        Number of histogram bins.
    ax : matplotlib Axes, optional
    figsize : tuple
    title : str, optional
    labels : tuple of str
        Labels for (control, treated).
    colors : tuple of str
        Colors for (control, treated).
    trim : float, optional
        If set, draw vertical lines at (trim, 1-trim) to show
        the recommended trimming region.

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> D = rng.binomial(1, 1 / (1 + np.exp(-(x1 + 0.5 * x2))))
    >>> df = pd.DataFrame({"D": D, "x1": x1, "x2": x2})
    >>> fig, ax = sp.psplot(df, treat="D", covariates=["x1", "x2"])
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required.")

    df = data[[treat] + covariates].dropna()
    T = df[treat].values.astype(int)
    X = df[covariates].values.astype(float)

    pscore = MatchEstimator._logit_propensity(X, T)
    ps_c = pscore[T == 0]
    ps_t = pscore[T == 1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    bins = np.linspace(0, 1, n_bins + 1)

    # Control: mirrored downward
    ax.hist(
        ps_c,
        bins=bins,
        alpha=0.6,
        color=colors[0],
        label=labels[0],
        density=True,
        edgecolor="white",
        linewidth=0.3,
    )
    # Treated: upward
    ax.hist(
        ps_t,
        bins=bins,
        alpha=0.6,
        color=colors[1],
        label=labels[1],
        density=True,
        edgecolor="white",
        linewidth=0.3,
    )

    # Trimming region
    if trim is not None:
        ax.axvline(
            x=trim,
            color="#8E44AD",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"Trim [{trim:.2f}, {1 - trim:.2f}]",
        )
        ax.axvline(x=1 - trim, color="#8E44AD", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Propensity Score", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title or "Propensity Score Distribution (Common Support)", fontsize=13)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    return fig, ax


CausalResult._CITATIONS["matching"] = (
    "@article{abadie2006large,\n"
    "  title={Large Sample Properties of Matching Estimators for "
    "Average Treatment Effects},\n"
    "  author={Abadie, Alberto and Imbens, Guido W},\n"
    "  journal={Econometrica},\n"
    "  volume={74},\n"
    "  number={1},\n"
    "  pages={235--267},\n"
    "  year={2006},\n"
    "  publisher={Wiley}\n"
    "}"
)
