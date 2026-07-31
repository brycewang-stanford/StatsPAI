"""Orchestration: specification -> design -> estimate -> inference -> tests.

This is the single place where a user-facing dynamic-panel call is turned
into numbers.  ``sp.xtabond`` is a thin presentation layer over
:func:`fit_dynamic_panel`.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import IdentificationFailure
from ._data import add_time_dummies, build_panel_arrays
from ._diagnostics import (
    arellano_bond_ar_test,
    check_instrument_count,
    difference_in_hansen,
    overid_test,
)
from ._estimate import (
    gmm_solve,
    level_sigma2,
    moment_covariance,
    onestep_weight,
    safe_inv,
)
from ._inference import robust_sandwich, windmeijer_correction
from ._moments import build_design
from ._spec import DynPanelSpec, GMMBlock, IVBlock, normalize_lag_range, parse_terms

__all__ = ["fit_dynamic_panel", "DynPanelFit"]

LagRange = Optional[Tuple[Optional[int], Optional[int]]]


class DynPanelFit(dict):
    """Plain result bag (a ``dict`` so it serialises without ceremony)."""


def fit_dynamic_panel(
    data,
    y: str,
    x: Optional[Sequence[str]] = None,
    id: str = "id",
    time: str = "time",
    lags: int = 1,
    gmm_lags: Tuple[int, Optional[int]] = (2, None),
    twostep: bool = False,
    robust: bool = True,
    predetermined: Optional[Sequence[str]] = None,
    endogenous: Optional[Sequence[str]] = None,
    predetermined_lags: LagRange = None,
    endogenous_lags: LagRange = None,
    collapse: bool = False,
    time_dummies: bool = False,
    method: str = "difference",
    transform: str = "fd",
    constant: Optional[bool] = None,
    stacklevel: int = 3,
) -> DynPanelFit:
    """Fit Arellano-Bond difference GMM or Blundell-Bond system GMM.

    See ``sp.xtabond`` for the documented user-facing semantics; this
    function deliberately returns raw arrays and scalars so that the
    ``CausalResult`` packaging, the ``sp.panel`` dispatcher and the
    postestimation surface can each present them their own way.
    """
    if lags < 1:
        raise ValueError("lags must be >= 1 (a dynamic panel needs a lagged Y).")
    if method not in ("difference", "system"):
        raise ValueError(f"method must be 'difference' or 'system', got {method!r}.")
    system = method == "system"
    if constant is None:
        # The constant differences away, so it is only identified once the
        # level equation is stacked in; xtabond2 includes it by default there.
        constant = system
    if constant and not system:
        raise NotImplementedError(
            "constant=True requires method='system': in a pure first-differenced "
            "GMM the intercept is differenced away, and Stata's `xtabond` default "
            "recovers it from an extra level moment that StatsPAI does not "
            "implement. Use method='system' (which reports _cons) or accept the "
            "`noconstant` parameterisation."
        )

    x_terms = parse_terms(x)
    pre_terms = parse_terms(predetermined)
    endo_terms = parse_terms(endogenous)

    exog_vars = list(dict.fromkeys(t.var for t in x_terms))
    pre_vars = list(dict.fromkeys(t.var for t in pre_terms))
    endo_vars = list(dict.fromkeys(t.var for t in endo_terms))
    overlap = (set(pre_vars) | set(endo_vars)) & set(exog_vars)
    if overlap:
        raise ValueError(
            f"variable(s) {sorted(overlap)} appear both as strictly exogenous "
            "and as predetermined/endogenous. A variable belongs to exactly "
            "one instrument class."
        )
    both = set(pre_vars) & set(endo_vars)
    if both:
        raise ValueError(
            f"variable(s) {sorted(both)} are declared both predetermined and "
            "endogenous; pick one."
        )

    needed = [y] + exog_vars + pre_vars + endo_vars
    panel = build_panel_arrays(data, id, time, needed)

    dummy_names: List[str] = []
    if time_dummies:
        dummy_names = add_time_dummies(panel, drop_first=1)

    horizon = panel.n_periods
    y_lag_min, y_lag_max = normalize_lag_range(gmm_lags, default_min=2, horizon=horizon)
    if y_lag_min < 2:
        raise ValueError(
            "gmm_lags minimum must be >= 2: only lags of at least 2 of the "
            "dependent variable are orthogonal to the first-differenced error."
        )
    pre_min, pre_max = normalize_lag_range(
        predetermined_lags, default_min=1, horizon=horizon
    )
    endo_min, endo_max = normalize_lag_range(
        endogenous_lags, default_min=2, horizon=horizon
    )

    all_x_terms = (
        list(x_terms)
        + list(pre_terms)
        + list(endo_terms)
        + [t for name in dummy_names for t in parse_terms([name])]
    )

    # One GMM block per instrumented variable for the transformed equation;
    # system GMM mirrors each with a level block whose instrument is the
    # lagged *difference* (Blundell-Bond).
    windows = (
        [(y, y_lag_min, y_lag_max)]
        + [(v, pre_min, pre_max) for v in pre_vars]
        + [(v, endo_min, endo_max) for v in endo_vars]
    )
    gmm_blocks = [
        GMMBlock(v, lo, hi, collapse=collapse, equation="diff") for v, lo, hi in windows
    ]
    if system:
        gmm_blocks += [
            GMMBlock(v, lo, hi, collapse=collapse, equation="level")
            for v, lo, hi in windows
        ]

    # xtabond2's iv() default is equation(both): one column carrying the
    # difference on transformed rows and the level on level rows.
    iv_eq = "both" if system else "diff"
    iv_terms = list(x_terms) + [t for name in dummy_names for t in parse_terms([name])]
    iv_blocks = [IVBlock(t, equation=iv_eq) for t in iv_terms]

    spec = DynPanelSpec(
        y=y,
        y_lags=lags,
        x_terms=all_x_terms,
        gmm_blocks=gmm_blocks,
        iv_blocks=iv_blocks,
        transform=transform,
        level_equation=system,
        constant=constant,
    )
    design = build_design(panel, spec)

    k = design.n_params
    m = design.n_instruments
    if m < k:
        raise IdentificationFailure(
            f"Under-identified: {m} instruments for {k} parameters. Widen "
            "gmm_lags, drop collapse=True, or add periods."
        )
    if design.n_rows < k + 1:
        raise ValueError(
            f"Not enough observations after differencing ({design.n_rows} rows "
            f"for {k} parameters)."
        )

    gap_note = _warn_on_internal_gaps(panel, y, stacklevel=stacklevel + 1)
    instrument_note = check_instrument_count(
        m, design.n_units_used, stacklevel=stacklevel + 1
    )

    W, Z, dy = design.W, design.Z, design.dy
    unit_rows = design.unit_rows

    A1 = onestep_weight(design)
    beta1, Minv1, WZ = gmm_solve(W, Z, dy, A1)
    resid1 = dy - W @ beta1
    Omega1 = moment_covariance(Z, resid1, unit_rows)
    V1_robust = robust_sandwich(Minv1, WZ, A1, Omega1)
    sigma2 = level_sigma2(resid1, design, k)

    # The efficient (two-step) solve is computed even when the reported
    # estimate is one-step, because the heteroskedasticity-robust Hansen J is
    # defined at the two-step optimum. xtabond2 does the same, which is why it
    # prints both Sargan and Hansen for a one-step fit.
    #
    # When the user did *not* ask for two-step, a rank deficiency here is
    # about the Hansen J and nothing else, so the generic "two-step weight
    # matrix is singular" text would misattribute the problem. Catch it and
    # re-raise one warning that says what is actually unreliable.
    hansen_note: Optional[str] = None
    with warnings.catch_warnings(record=True) as aux:
        warnings.simplefilter("always")
        A2 = safe_inv(Omega1, "two-step weight matrix Z'êê'Z")
        beta2, Minv2, _ = gmm_solve(W, Z, dy, A2)
    resid2 = dy - W @ beta2
    if aux:
        if twostep:
            for record in aux:
                warnings.warn(record.message, stacklevel=stacklevel)
        else:
            hansen_note = (
                "The heteroskedasticity-robust Hansen J is evaluated at the "
                "two-step optimum, and that solve is rank-deficient here "
                f"({aux[0].message}). The reported Hansen statistic is "
                "unreliable; the one-step estimate and its standard errors "
                "are unaffected."
            )
            warnings.warn(hansen_note, stacklevel=stacklevel)

    if twostep:
        if not robust:
            warnings.warn(
                "Two-step GMM standard errors are downward biased in finite "
                "samples; robust=True (Windmeijer correction) is recommended.",
                stacklevel=stacklevel,
            )
        beta, resid = beta2, resid2
        weight_final, Minv_final = A2, Minv2
        if robust:
            vcov = windmeijer_correction(
                W, Z, WZ, resid1, resid2, A2, Minv2, V1_robust, unit_rows
            )
        else:
            vcov = Minv2
    else:
        beta, resid = beta1, resid1
        weight_final, Minv_final = A1, Minv1
        vcov = V1_robust if robust else sigma2 * Minv1

    var_diag = np.diag(vcov)
    if np.any(var_diag <= 0):
        warnings.warn(
            "Non-positive coefficient variance encountered — the model may be "
            "under-identified or the instrument set rank-deficient; the "
            "affected standard errors are unreliable.",
            stacklevel=stacklevel,
        )
    se = np.sqrt(np.maximum(var_diag, 0.0))

    robust_ar = robust or twostep
    # The Arellano-Bond variance carries a (W'q)' Avar(beta) (W'q) term whose
    # natural expansion uses the *uncorrected* robust sandwich; swap in the VCE
    # actually reported so the test and the coefficients agree on Avar(beta).
    naive_vcov = (
        robust_sandwich(
            Minv_final, WZ, weight_final, moment_covariance(Z, resid, unit_rows)
        )
        if robust_ar
        else None
    )
    ar_kwargs = dict(
        unit_rows=unit_rows,
        periods=design.row_period,
        Z=Z,
        W=W,
        weight=weight_final,
        Minv=Minv_final,
        robust=robust_ar,
        sigma2=sigma2,
        # The Arellano-Bond test is always about serial correlation in the
        # *first-differenced* errors, so under system GMM only the
        # transformed rows contribute residual pairs.
        eq_mask=design.row_eq == 0,
        coef_vcov=vcov if robust_ar else None,
        naive_vcov=naive_vcov,
    )
    ar1 = arellano_bond_ar_test(resid, order=1, **ar_kwargs)
    ar2 = arellano_bond_ar_test(resid, order=2, **ar_kwargs)

    overid_df = m - k
    sargan = overid_test(Z, resid1, A1, overid_df, scale=sigma2)
    hansen = overid_test(Z, resid2, A2, overid_df, scale=1.0)
    diff_hansen = difference_in_hansen(design, hansen["stat"], k, Omega1)

    return DynPanelFit(
        beta=beta,
        se=se,
        vcov=vcov,
        names=design.regressor_names,
        n_obs=design.n_rows_level if system else design.n_rows_diff,
        n_obs_diff=design.n_rows_diff,
        n_obs_level=design.n_rows_level,
        n_obs_total=design.n_rows,
        n_units=design.n_units_used,
        method=method,
        constant=bool(constant),
        n_instruments=m,
        n_params=k,
        instrument_labels=design.instrument_labels,
        gmm_lags=(y_lag_min, None if y_lag_max >= horizon else y_lag_max),
        collapse=collapse,
        time_dummies=list(dummy_names),
        sigma2=sigma2,
        resid=resid,
        design=design,
        ar1=ar1,
        ar2=ar2,
        sargan=sargan,
        hansen=hansen,
        difference_in_hansen=diff_hansen,
        gap_warning=gap_note,
        instrument_warning=instrument_note,
        hansen_warning=hansen_note,
    )


def _warn_on_internal_gaps(panel, y: str, stacklevel: int) -> Optional[str]:
    """Flag units missing an *interior* period of the dependent variable.

    What is known, precisely (measured on a hole-punched ``abdata``):

    * the **design** agrees with Stata exactly — same equations, same
      sample size, same per-unit row counts, same instrument count, and a
      *just-identified* fit (one instrument, so the weight matrix cancels)
      reproduces ``xtabond2`` to 2e-15;
    * the **one-step weight matrix** does not. StatsPAI uses ``H = M M'``
      for the actual differencing operator — 2 on the diagonal, -1 between
      calendar-adjacent equations, nothing across a hole — which is the
      textbook a-priori covariance of the differenced errors. Stata's
      gapped-panel convention is different and undocumented; the resulting
      coefficients differ by roughly 2-6%.

    Since the disagreement is confined to the weight matrix, both estimators
    remain consistent and only their finite-sample efficiency differs. But a
    user reporting "identical to Stata" would be wrong, so say so.
    """
    obs = panel.observed(y)
    idx = np.arange(panel.n_periods)
    has_gap = False
    for row in obs:
        present = idx[row]
        if present.size and (present[-1] - present[0] + 1) != present.size:
            has_gap = True
            break
    if not has_gap:
        return None
    msg = (
        "Panel has internal time gaps in the dependent variable. The design "
        "(sample, equations, instruments) matches Stata exactly, but the "
        "one-step weight matrix uses a different gap convention, so "
        "coefficients differ from Stata's xtabond / xtabond2 by roughly "
        "2-6%. Both remain consistent — only finite-sample efficiency "
        "differs — but machine-precision cross-software parity holds only "
        "for gap-free panels. Consider orthogonal=True: forward orthogonal "
        "deviations lose one observation per gap instead of two."
    )
    warnings.warn(msg, stacklevel=stacklevel)
    return msg
