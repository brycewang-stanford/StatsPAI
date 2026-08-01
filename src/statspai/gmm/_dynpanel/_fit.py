"""Orchestration: specification -> design -> estimate -> inference -> tests.

This is the single place where a user-facing dynamic-panel call is turned
into numbers.  ``sp.xtabond`` is a thin presentation layer over
:func:`fit_dynamic_panel`.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import IdentificationFailure
from ._data import add_time_dummies, build_panel_arrays, unit_cluster_codes
from ._diagnostics import (
    _lagged_residual_vector,
    ar_test_cross_basis,
    arellano_bond_ar_test,
    check_instrument_count,
    difference_in_hansen,
    overid_test,
)
from ._estimate import (
    gmm_solve,
    group_index,
    level_sigma2,
    moment_covariance,
    onestep_weight,
    safe_inv,
)
from ._inference import robust_sandwich, windmeijer_correction
from ._moments import build_design
from ._spec import (
    DynPanelSpec,
    GMMBlock,
    IVBlock,
    Term,
    normalize_lag_range,
    parse_terms,
)

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
    cluster: Optional[str] = None,
    steps: Optional[object] = None,
    iter_tol: float = 1e-10,
    iter_maxiter: int = 100,
    ah_instrument: str = "levels",
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
    if method not in ("difference", "system", "ah"):
        raise ValueError(
            f"method must be 'difference', 'system' or 'ah', got {method!r}."
        )
    if ah_instrument not in ("levels", "differences"):
        raise ValueError(
            "ah_instrument must be 'levels' (instrument L2.y) or 'differences' "
            f"(instrument D.L2.y), got {ah_instrument!r}."
        )
    system = method == "system"
    anderson_hsiao = method == "ah"
    steps = _resolve_steps(steps, twostep)
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
    # ``cluster=id`` IS the default unit grouping, so it must not be pivoted
    # as a variable (the frame would carry the index column twice).
    cluster_is_unit = cluster is not None and cluster == id
    if cluster is not None and not cluster_is_unit and cluster not in needed:
        needed = needed + [cluster]
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
    ah_extra_iv: List[Term] = []
    ah_blocks: List[GMMBlock] = []
    if anderson_hsiao:
        # Anderson-Hsiao uses ONE pooled instrument for the differenced lagged
        # dependent variable instead of Arellano-Bond's block-diagonal set:
        # y_{t-2} in levels, or its first difference. Both are expressible in
        # the same moment vocabulary -- the levels variant is a collapsed GMM
        # block with a single lag, the differences variant a standard IV
        # column -- so this is a different moment set, not a new estimator.
        if ah_instrument == "levels":
            ah_blocks = [
                GMMBlock(y, y_lag_min, y_lag_min, collapse=True, equation="diff")
            ]
        else:
            ah_extra_iv.append(Term(y, y_lag_min))
        windows = windows[1:]  # the dependent variable is handled above

    gmm_blocks = ah_blocks + [
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
    iv_terms = (
        list(ah_extra_iv)
        + list(x_terms)
        + [t for name in dummy_names for t in parse_terms([name])]
    )
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
    if cluster is not None and not cluster_is_unit:
        design.set_clusters(unit_cluster_codes(panel, cluster))

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

    if steps != 1 and design.n_clusters <= m:
        raise IdentificationFailure(
            f"Multi-step GMM with cluster={cluster!r} has {design.n_clusters} "
            f"clusters for {m} moment conditions. The efficient weight matrix "
            "is the inverse of a clustered moment covariance whose rank cannot "
            "exceed the number of clusters, so it is singular here and the "
            "estimate would be an artefact of whichever generalized inverse is "
            "used. Either keep the one-step estimator (its cluster-robust "
            "standard errors are valid at any cluster count), or reduce the "
            "moment count with collapse=True / a tighter gmm_lags window until "
            "it is below the cluster count."
        )

    gap_note = _warn_on_internal_gaps(panel, y, stacklevel=stacklevel + 1)
    instrument_note = check_instrument_count(
        m, design.n_units_used, stacklevel=stacklevel + 1
    )

    W, Z, dy = design.W, design.Z, design.dy
    # The sandwich meat and the AR-test variance are summed over clusters;
    # with no cluster= these are the units, which is the default convention.
    meat_rows = design.meat_rows
    meat_index = design.group_index()

    A1 = onestep_weight(design)
    beta1, Minv1, WZ = gmm_solve(W, Z, dy, A1)
    resid1 = dy - W @ beta1
    Omega1 = moment_covariance(Z, resid1, meat_rows, index=meat_index)
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
        if steps != 1:
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

    n_steps_run = 1 if steps == 1 else 2
    converged: Optional[bool] = None
    if steps == "cue":
        beta, weight_final, Minv_final, converged, n_steps_run = _cue(
            W,
            Z,
            dy,
            meat_rows,
            beta2,
            iter_tol,
            iter_maxiter,
            stacklevel,
            index=meat_index,
        )
        resid = dy - W @ beta
        vcov = (
            robust_sandwich(
                Minv_final,
                WZ,
                weight_final,
                moment_covariance(Z, resid, meat_rows, index=meat_index),
            )
            if robust
            else Minv_final
        )
    elif steps == 1:
        beta, resid = beta1, resid1
        weight_final, Minv_final = A1, Minv1
        vcov = V1_robust if robust else sigma2 * Minv1
    else:
        if not robust:
            warnings.warn(
                "Multi-step GMM standard errors are downward biased in finite "
                "samples; robust=True (Windmeijer correction) is recommended.",
                stacklevel=stacklevel,
            )
        # Steps 3+ repeat the two-step recursion: re-estimate the moment
        # covariance at the current residuals, re-solve, repeat. `steps=2` is
        # the classic efficient two-step; 'iterated' runs it to a fixed point,
        # where beta and the weight it implies are mutually consistent.
        prev_resid, beta, resid = resid1, beta2, resid2
        weight_final, Minv_final = A2, Minv2
        shift = 0.0
        if steps != 2:
            limit = iter_maxiter if steps == "iterated" else int(steps)
            while n_steps_run < limit:
                Omega = moment_covariance(Z, resid, meat_rows, index=meat_index)
                A = safe_inv(Omega, f"step-{n_steps_run + 1} weight matrix")
                new_beta, Minv_new, _ = gmm_solve(W, Z, dy, A)
                shift = float(np.max(np.abs(new_beta - beta)))
                prev_resid = resid
                beta, Minv_final, weight_final = new_beta, Minv_new, A
                resid = dy - W @ beta
                n_steps_run += 1
                if steps == "iterated" and shift < iter_tol:
                    converged = True
                    break
            if steps == "iterated" and converged is None:
                converged = False
                warnings.warn(
                    f"Iterated GMM did not converge in {iter_maxiter} steps "
                    f"(last coefficient change {shift:.3g} > tol {iter_tol:g}). "
                    "The reported estimate is the last iterate, not a fixed "
                    "point; raise iter_maxiter or loosen iter_tol.",
                    stacklevel=stacklevel,
                )
        if robust:
            # Windmeijer (2005) is derived for the two-step estimator. For a
            # deeper recursion the same correction is applied with the
            # penultimate iterate playing the role of step 1 — exactly right
            # at steps=2 and its natural extension beyond.
            vcov = windmeijer_correction(
                W,
                Z,
                WZ,
                prev_resid,
                resid,
                weight_final,
                Minv_final,
                V1_robust,
                meat_rows,
                index=meat_index,
            )
        else:
            vcov = Minv_final

    var_diag = np.diag(vcov)
    if np.any(var_diag <= 0):
        warnings.warn(
            "Non-positive coefficient variance encountered — the model may be "
            "under-identified or the instrument set rank-deficient; the "
            "affected standard errors are unreliable.",
            stacklevel=stacklevel,
        )
    se = np.sqrt(np.maximum(var_diag, 0.0))

    # The Arellano-Bond test is a statement about serial correlation in the
    # *first-differenced* errors -- that is what Stata and xtabond2 print,
    # whatever transform produced the estimate. Under forward orthogonal
    # deviations the estimation rows hold FOD residuals, so the test is run
    # on an auxiliary first-differenced design evaluated at the fitted
    # coefficients. Running it on the FOD residuals instead flips the sign
    # of the statistic.
    ar_design = design
    ar_resid = resid
    if transform == "fod":
        ar_spec = replace(spec, transform="fd")
        ar_design = build_design(panel, ar_spec)
        if cluster is not None and not cluster_is_unit:
            ar_design.set_clusters(unit_cluster_codes(panel, cluster))
        ar_resid = ar_design.dy - ar_design.W @ beta

    robust_ar = robust or twostep
    # The Arellano-Bond variance carries a (W'q)' Avar(beta) (W'q) term whose
    # natural expansion uses the *uncorrected* robust sandwich; swap in the VCE
    # actually reported so the test and the coefficients agree on Avar(beta).
    if ar_design is design:
        ar_W, ar_Z = W, Z
        ar_weight, ar_Minv = weight_final, Minv_final
        # The Arellano-Bond variance sums (q_i' e_i)^2 and Z_i' e_i (q_i' e_i)
        # over *units*, never over clusters -- xtabond2's `_ARTests` loops
        # `for (i = N; i; i--)` for both. Only the coefficient-variance term
        # picks up the clustering, through `coef_vcov` below.
        ar_meat = design.unit_rows
        ar_index = group_index(ar_meat, design.n_rows)
    else:
        # The auxiliary design has its own row space, so its influence
        # adjustment uses its own one-step weight. The coefficient-variance
        # term still comes from the reported VCE.
        ar_meat = ar_design.unit_rows
        ar_index = group_index(ar_meat, ar_design.n_rows)
        ar_W, ar_Z = ar_design.W, ar_design.Z
        ar_weight = onestep_weight(ar_design)
        _, ar_Minv, _ = gmm_solve(ar_W, ar_Z, ar_design.dy, ar_weight)

    naive_vcov = (
        robust_sandwich(
            ar_Minv,
            ar_W.T @ ar_Z,
            ar_weight,
            moment_covariance(ar_Z, ar_resid, ar_meat, index=ar_index),
        )
        if robust_ar
        else None
    )
    ar_kwargs = dict(
        unit_rows=ar_meat,
        periods=ar_design.row_period,
        Z=ar_Z,
        W=ar_W,
        weight=ar_weight,
        Minv=ar_Minv,
        robust=robust_ar,
        sigma2=sigma2,
        # The Arellano-Bond test is always about serial correlation in the
        # *first-differenced* errors, so under system GMM only the
        # transformed rows contribute residual pairs.
        eq_mask=ar_design.row_eq == 0,
        units=ar_design.row_unit,
        coef_vcov=vcov if robust_ar else None,
        naive_vcov=naive_vcov,
    )
    # Every AR configuration is now exact against Stata / xtabond2, so no
    # provenance caveat is attached. The field is kept so downstream readers
    # have a stable key to check.
    ar_note: Optional[str] = None

    if ar_design is not design:
        # Forward orthogonal deviations: the test lives on first differences
        # while the estimator ran on the deviations, so the two bases are
        # combined unit by unit exactly as xtabond2's `_ARTests` does.
        ar_mask = ar_design.row_eq == 0
        n_units_total = (
            int(max(int(ar_design.row_unit.max()), int(design.row_unit.max()))) + 1
        )
        XZ_est = W.T @ Z
        ar1, ar2 = (
            ar_test_cross_basis(
                q=_lagged_residual_vector(
                    ar_resid,
                    ar_design.unit_rows,
                    ar_design.row_period,
                    order,
                    ar_mask,
                    ar_design.row_unit,
                )[ar_mask],
                resid_ar=ar_resid[ar_mask],
                X_ar=ar_design.W[ar_mask],
                units_ar=ar_design.row_unit[ar_mask],
                Z_est=Z,
                resid_est=resid,
                units_est=design.row_unit,
                XZ_est=XZ_est,
                weight=weight_final,
                Minv=Minv_final,
                coef_vcov=vcov,
                n_units=n_units_total,
            )
            for order in (1, 2)
        )
    else:
        ar1 = arellano_bond_ar_test(ar_resid, order=1, **ar_kwargs)
        ar2 = arellano_bond_ar_test(ar_resid, order=2, **ar_kwargs)

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
        n_clusters=design.n_clusters,
        cluster=cluster,
        steps=n_steps_run,
        steps_requested=steps,
        converged=converged,
        ah_instrument=ah_instrument if anderson_hsiao else None,
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
        ar_note=ar_note,
        instrument_warning=instrument_note,
        hansen_warning=hansen_note,
    )


def _warn_on_internal_gaps(panel, y: str, stacklevel: int) -> Optional[str]:
    """Flag units missing an *interior* period of the dependent variable.

    This is an efficiency advisory, not a parity caveat. First differencing
    destroys two equations per hole (the difference *at* the missing period
    and the one *after* it); forward orthogonal deviations destroy one, so
    ``orthogonal=True`` is usually the better transform on a holed panel.

    Parity with Stata is exact here. An earlier version of this warning
    claimed the one-step weight matrix used a different gap convention and
    that coefficients differed from ``xtabond2`` by 2-6%. That was wrong,
    and the mistake is worth recording because it was a *reference* bug that
    read as an estimator bug: the fixture wrote the instrument set as
    ``gmm(L.n, lag(k k))``, and on a gapped panel that is not the same
    moment set as ``gmm(n, lag(k+1 k+1))``. Stata materialises the
    expression ``L.n`` row by row, so it is missing wherever the preceding
    row is absent, and ``xtabond2`` then lags that already-holed series —
    the instrument ends up needing both period ``t-k-1`` and period ``t-k``
    to exist. The two forms coincide on gap-free panels, which is why every
    other spec agreed.

    Against the level form, which is what ``gmm_lags`` actually names,
    StatsPAI matches ``xtabond2`` on holed panels to 1e-12 across
    one-step, two-step Windmeijer, FOD, system GMM and multi-regressor
    designs — see the ``I*`` specs in
    ``tests/reference_parity/_fixtures/_generate_dynpanel_stata.do``.
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
        "Panel has internal time gaps in the dependent variable. First "
        "differencing loses two equations per gap; consider orthogonal=True, "
        "since forward orthogonal deviations lose one. This is an efficiency "
        "note only — the estimator is unaffected otherwise, and matches "
        "Stata's xtabond2 to machine precision on gapped panels. (When "
        "cross-checking in Stata, write the instrument set on the level, "
        "gmm(y, lag(a b)); the lagged-expression form gmm(L.y, lag(a-1 b-1)) "
        "is a different moment set once the panel has holes.)"
    )
    warnings.warn(msg, stacklevel=stacklevel)
    return msg


def _resolve_steps(steps, twostep: bool):
    """Normalise ``steps`` / ``twostep`` to ``1``, an int, ``'iterated'`` or ``'cue'``.

    ``twostep`` predates ``steps`` and stays the documented switch for the
    common case; ``steps`` generalises it.  Passing both raises rather than
    resolving by a silent precedence rule — a caller writing
    ``twostep=True, steps=1`` has contradicted themselves.
    """
    if steps is None:
        return 2 if twostep else 1
    if twostep:
        raise ValueError(
            "pass either twostep= or steps=, not both: twostep=True is steps=2."
        )
    if isinstance(steps, str):
        if steps not in ("iterated", "cue"):
            raise ValueError(
                "steps must be a positive integer, 'iterated' or 'cue', got "
                f"{steps!r}."
            )
        return steps
    steps = int(steps)
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}.")
    return steps


def _cue(W, Z, dy, meat_rows, start, tol, maxiter, stacklevel, index=None):
    """Continuously-updated GMM (Hansen, Heaton & Yaron 1996).

    Minimises ``g(β)' Ω(β)^{-1} g(β)`` with the weight re-evaluated at every
    trial ``β`` rather than held at a first-step estimate.  That removes the
    two-step estimator's dependence on preliminary residuals — the very
    dependence the Windmeijer correction exists to patch — at the cost of a
    non-quadratic, generally non-convex objective, so this is a numerical
    optimisation rather than a solve.

    The two-step estimate is used only as a starting point; the objective
    itself is invariant to the starting weight, which is the property worth
    testing against.

    References
    ----------
    Hansen, L.P., Heaton, J. and Yaron, A. (1996). Finite-sample properties
    of some alternative GMM estimators. *Journal of Business & Economic
    Statistics* 14(3), 262-280. [@hansen1996finite]
    """
    from scipy.optimize import minimize

    def objective(beta: np.ndarray) -> float:
        resid = dy - W @ beta
        Omega = moment_covariance(Z, resid, meat_rows, index=index)
        g = Z.T @ resid
        try:
            weight = np.linalg.inv(Omega)
        except np.linalg.LinAlgError:
            weight = np.linalg.pinv(Omega)
        return float(g @ weight @ g)

    result = minimize(
        objective,
        np.asarray(start, dtype=float),
        method="Nelder-Mead",
        options={"xatol": tol, "fatol": tol, "maxiter": maxiter * 200},
    )
    beta = np.asarray(result.x, dtype=float)
    if not result.success:
        warnings.warn(
            "Continuously-updated GMM did not converge "
            f"({result.message}). The reported estimate is the last iterate; "
            "the CUE objective is non-convex, so a different starting point "
            "or a larger iter_maxiter may help.",
            stacklevel=stacklevel,
        )
    resid = dy - W @ beta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weight = safe_inv(
            moment_covariance(Z, resid, meat_rows, index=index), "CUE weight"
        )
    WZ = W.T @ Z
    Minv = safe_inv(WZ @ weight @ WZ.T, "CUE moment matrix")
    return beta, weight, Minv, bool(result.success), int(result.nit)
