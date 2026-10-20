"""Orchestration: specification -> design -> estimate -> inference -> tests.

This is the single place where a user-facing dynamic-panel call is turned
into numbers.  ``sp.xtabond`` is a thin presentation layer over
:func:`fit_dynamic_panel`.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ...exceptions import IdentificationFailure
from ._data import add_time_dummies, build_panel_arrays, unit_cluster_codes
from ._diagnostics import (
    check_instrument_count,
    difference_in_hansen,
    overid_test,
    serial_correlation_test,
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
    h: int = 3,
    iv_equation: Optional[str] = None,
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
    if h not in (1, 2, 3):
        raise ValueError(f"h must be 1, 2 or 3 (xtabond2's h()), got {h!r}.")
    if iv_equation not in (None, "diff", "level", "both"):
        raise ValueError(
            "iv_equation must be 'diff', 'level' or 'both', got " f"{iv_equation!r}."
        )
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

    def build(dummy_names: List[str], dummy_iv: Optional[List[str]] = None) -> Any:
        if dummy_iv is None:
            dummy_iv = dummy_names
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
            GMMBlock(v, lo, hi, collapse=collapse, equation="diff")
            for v, lo, hi in windows
        ]
        if system:
            gmm_blocks += [
                GMMBlock(v, lo, hi, collapse=collapse, equation="level")
                for v, lo, hi in windows
            ]

        # xtabond2's iv() default is equation(both): one column carrying the
        # difference on transformed rows and the level on level rows.
        iv_eq = (
            iv_equation
            if (system and iv_equation is not None)
            else ("both" if system else "diff")
        )
        iv_terms = (
            list(ah_extra_iv)
            + list(x_terms)
            + [t for name in dummy_iv for t in parse_terms([name])]
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
        design.spec = spec
        design.h = h
        if cluster is not None and not cluster_is_unit:
            design.set_clusters(unit_cluster_codes(panel, cluster))
        return design

    design = build(dummy_names)
    if dummy_names:
        # The dummies were made for every period of the panel. The lags and
        # the differencing remove the first periods from the transformed
        # equation, and a dummy for a period with no transformed row is not
        # identified by it. If the ones that are left cover every period
        # that has such a row, they sum to the constant there and one more
        # has to go. Stata's xtdpd checks this on the same rows, omits the
        # last dummy from the regressors and keeps it among the instruments;
        # so do we.
        period_of = {name: p + 1 for p, name in enumerate(dummy_names)}
        present = set(np.unique(design.row_period[design.row_eq == 0]).tolist())
        kept = [n for n in dummy_names if period_of[n] in present]
        dummy_iv = list(kept)
        if constant and len(kept) == len(present):
            kept = kept[:-1]
        if len(kept) < len(dummy_names):
            design = build(kept, dummy_iv)
        while kept and np.linalg.matrix_rank(design.W) < design.n_params:
            kept = kept[:-1]
            design = build(kept, dummy_iv)
        dummy_names = kept

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
            # When the weight is a generalized inverse, the moments it set
            # to zero are also left out of the one-step covariance that
            # enters the correction. Stata's xtdpd and xtabond2 get there by
            # inverting the weight back; with a weight of full rank the two
            # are the same matrix.
            V1_step = V1_robust
            kept = np.diag(weight_final) != 0
            if not kept.all():
                Omega_kept = np.zeros_like(weight_final)
                at = np.ix_(kept, kept)
                Omega_kept[at] = moment_covariance(
                    Z, prev_resid, meat_rows, index=meat_index
                )[at]
                V1_step = robust_sandwich(Minv1, WZ, A1, Omega_kept)
            vcov = windmeijer_correction(
                W,
                Z,
                WZ,
                prev_resid,
                resid,
                weight_final,
                Minv_final,
                V1_step,
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

    ar1, ar2, ar_note = _serial_correlation_tests(
        panel,
        design,
        beta,
        resid,
        vcov,
        weight_final,
        Minv_final,
        classical=(steps == 1 and not robust),
        sigma2=sigma2,
    )

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


def _serial_correlation_tests(
    panel, design, beta, resid, vcov, weight, Minv, *, classical, sigma2
):
    """AR(1) and AR(2) tests on the first-differenced residuals.

    The test is always run in first differences (Roodman 2009, Sec. 3.5:
    deviations residuals are all mutually related), so after estimation in
    forward orthogonal deviations the residuals are recomputed on the
    first-differenced version of the same specification. Level-equation
    rows of system GMM do not enter. Returns ``(ar1, ar2, note)``.
    """
    if design.transform == "fd":
        test, resid_test = design, resid
    else:
        test = build_design(panel, replace(design.spec, transform="fd"))
        resid_test = test.dy - test.W @ beta

    # The AR statistic sums over panel units even when the coefficient
    # covariance is clustered more coarsely (black box: xtabond2 with
    # cluster() matches this to 1e-12; the clustering enters through Avar).
    est_groups = design.unit_rows
    unit_group = {}
    for j, rows in enumerate(est_groups):
        for u in np.unique(design.row_unit[rows]).tolist():
            unit_group[u] = j
    n_groups = len(est_groups)
    test_units = test.row_unit
    for u in np.unique(test_units).tolist():
        if u not in unit_group:
            unit_group[u] = n_groups
            n_groups += 1
    groups_test = np.array([unit_group[u] for u in test_units.tolist()], dtype=np.int64)
    groups_est = np.array(
        [unit_group[u] for u in design.row_unit.tolist()], dtype=np.int64
    )

    blocks = (
        _classical_blocks(design, test, sigma2, panel.n_periods) if classical else None
    )

    common = dict(
        resid_test=resid_test,
        X_test=test.W,
        units_test=test.row_unit,
        periods_test=test.row_period,
        use_test=test.row_eq == 0,
        groups_test=groups_test,
        Z=design.Z,
        resid_est=resid,
        groups_est=groups_est,
        XZ=design.W.T @ design.Z,
        weight=weight,
        Minv=Minv,
        coef_vcov=vcov,
        n_groups=n_groups,
        classical_blocks=blocks,
    )
    ar1 = serial_correlation_test(1, **common)
    ar2 = serial_correlation_test(2, **common)
    return ar1, ar2, None


def _level_operator(design, rows, n_periods: int) -> np.ndarray:
    """Rows of the transform as weights on the unit's level errors.

    First differences: ``e_p - e_{p-1}``. Forward orthogonal deviations:
    the deviation operator on the balanced period grid (the one ``H`` is
    built from; black box: xtabond2's classical AR statistic after
    ``orthogonal`` matches this, to 1e-10, and not the unit-specific
    operator). Level-equation rows get zero rows: like xtabond2, the
    classical AR variance ignores their covariance with the differences.
    """
    out = np.zeros((rows.size, n_periods))
    operator = design.transform_operator
    for j, r in enumerate(rows):
        if design.row_eq[r] != 0:
            continue
        p = int(design.row_period[r])
        if operator is None:
            out[j, p] = 1.0
            if p >= 1:
                out[j, p - 1] = -1.0
        else:
            out[j] = operator[p - 1]
    return out


def _classical_blocks(design, test, sigma2: float, n_periods: int):
    """Per-unit error covariances for the classical AR variance.

    Under i.i.d. level errors with variance ``sigma2``, the covariance of
    two transformed errors is ``sigma2 * a'b`` for their level weights
    ``a``, ``b``. With ``h=1`` the one-step weight assumes ``H = I`` for the
    transformed errors, and the AR variance takes the same assumption:
    ``sigma2`` between a row and itself (same unit and period), zero
    otherwise (black box: xtabond2's ``h(1)`` statistics follow this
    structure).
    """
    blocks = []
    test_use = test.row_eq == 0
    test_rows_of = {}
    for rows in test.unit_rows:
        rows = rows[test_use[rows]]
        if rows.size:
            test_rows_of[int(test.row_unit[rows[0]])] = rows
    for est_rows in design.unit_rows:
        unit = int(design.row_unit[est_rows[0]])
        test_rows = test_rows_of.get(unit)
        if test_rows is None:
            continue
        if design.h == 1:
            same = (design.row_eq[est_rows] == 0)[:, None] & (
                design.row_period[est_rows][:, None]
                == test.row_period[test_rows][None, :]
            )
            c_et = sigma2 * same.astype(float)
            c_tt = sigma2 * np.eye(test_rows.size)
        else:
            a = _level_operator(design, est_rows, n_periods)
            b = _level_operator(test, test_rows, n_periods)
            c_et, c_tt = sigma2 * (a @ b.T), sigma2 * (b @ b.T)
        blocks.append((est_rows, test_rows, c_et, c_tt))
    return blocks


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
