"""
Traditional OLS event study (lead/lag) estimator.

Estimates dynamic treatment effects via relative-time dummies in a
two-way fixed effects framework.  Unlike the Callaway-Sant'Anna or
Sun-Abraham estimators (which correct for heterogeneous treatment timing),
this implements the **classic event study** that is standard in applied
economics when treatment timing is uniform or the researcher wants the
conventional specification.

Model
-----
Y_{it} = α_i + λ_t + Σ_{k≠-1} β_k · 1{t − g_i = k} + X_{it}'γ + ε_{it}

where g_i is unit i's treatment time, k indexes relative time, and
k = −1 is the omitted reference period.

References
----------
Freyaldenhoven, S., Hansen, C. and Shapiro, J.M. (2019).
"Pre-event Trends in the Panel Event-Study Design."
*American Economic Review*, 109(9), 3307-3338. [@freyaldenhoven2019event]

Roth, J. (2022).
"Pretest with Caution: Event-Study Estimates After Testing for Parallel
Trends." *American Economic Review: Insights*, 4(3), 305-322. [@roth2022pretest]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility

RefPeriodSpec = Union[int, Tuple[str, int], Sequence[int]]

_INTERVAL_OPS = ("<=", "<", ">=", ">")


def _resolve_ref_set(
    ref_period: RefPeriodSpec,
    min_lag: int,
    max_lag: int,
) -> Tuple[List[int], Any]:
    """Normalise ``ref_period`` into an explicit set of omitted relative times.

    Accepts a plain ``int`` (classic behaviour), an interval spec such as
    ``("<=", -50)`` / ``(">=", 20)``, or an explicit span ``[-3, -2, -1]``.

    Returns ``(sorted_ref_times, canonical_spec)``.
    """
    candidates = list(range(min_lag, max_lag + 1))
    window_repr = f"window=({min_lag}, {max_lag})"

    def _fail(msg: str, fix: str) -> "MethodIncompatibility":
        return MethodIncompatibility(
            f"event_study: {msg}",
            recovery_hint=fix,
            diagnostics={
                "ref_period": repr(ref_period),
                "window": (min_lag, max_lag),
            },
        )

    # --- interval form: ("<=", v) / (">=", v) ------------------------------ #
    is_interval = (
        isinstance(ref_period, (tuple, list))
        and len(ref_period) == 2
        and isinstance(ref_period[0], str)
    )
    if is_interval:
        op, raw = ref_period[0], ref_period[1]
        if op not in _INTERVAL_OPS:
            raise _fail(
                f"unknown interval operator {op!r} in ref_period={ref_period!r}.",
                "Use one of "
                f"{list(_INTERVAL_OPS)}, e.g. ref_period=('<=', {min_lag}).",
            )
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
            raise _fail(
                f"interval bound {raw!r} in ref_period={ref_period!r} must be an "
                "integer relative period.",
                f"e.g. ref_period=('{op}', {min_lag}).",
            )
        bound = int(raw)
        if op == "<=":
            ref_times = [t for t in candidates if t <= bound]
        elif op == "<":
            ref_times = [t for t in candidates if t < bound]
        elif op == ">=":
            ref_times = [t for t in candidates if t >= bound]
        else:
            ref_times = [t for t in candidates if t > bound]
        if not ref_times:
            raise _fail(
                f"ref_period={ref_period!r} selects no relative period inside "
                f"{window_repr}: the bound {bound} lies outside the window.",
                "Widen the window or move the bound inside it, e.g. "
                f"ref_period=('{op}', "
                f"{min_lag if op in ('<=', '<') else max_lag}).",
            )
        canonical: Any = (op, bound)
    # --- explicit span: [-3, -2, -1] --------------------------------------- #
    elif isinstance(ref_period, (list, tuple, set, np.ndarray)):
        raw_list = list(ref_period)
        if not raw_list:
            raise _fail(
                "ref_period is an empty sequence; there is nothing to omit.",
                "Pass an int (ref_period=-1), an interval "
                "(ref_period=('<=', -50)), or a non-empty span "
                "(ref_period=[-3, -2, -1]).",
            )
        bad = [
            v
            for v in raw_list
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
        ]
        if bad:
            raise _fail(
                f"ref_period span contains non-integer entries {bad!r}.",
                "Pass integer relative periods, e.g. ref_period=[-3, -2, -1].",
            )
        ref_times = sorted({int(v) for v in raw_list})
        outside = [t for t in ref_times if t not in candidates]
        if outside:
            raise _fail(
                f"ref_period span {ref_times!r} contains period(s) {outside!r} "
                f"outside {window_repr}.",
                f"Either widen the window (e.g. window=("
                f"{min(min_lag, min(ref_times))}, "
                f"{max(max_lag, max(ref_times))})) or drop those periods, e.g. "
                f"ref_period={[t for t in ref_times if t in candidates] or [-1]}.",
            )
        canonical = ref_times
    # --- plain int (classic) ----------------------------------------------- #
    else:
        if isinstance(ref_period, (bool, np.bool_)) or not isinstance(
            ref_period, (int, np.integer)
        ):
            raise _fail(
                f"ref_period={ref_period!r} must be an int, an interval such as "
                "('<=', -50), or a span such as [-3, -2, -1].",
                "e.g. ref_period=-1.",
            )
        r = int(ref_period)
        if r not in candidates:
            raise _fail(
                f"ref_period={r} lies outside {window_repr}, so there is no "
                "coefficient to omit and the design is unidentified.",
                f"Either pick a period inside the window (e.g. ref_period=-1) "
                f"or widen it, e.g. window=({min(min_lag, r)}, "
                f"{max(max_lag, r)}).",
            )
        ref_times = [r]
        canonical = r

    if len(ref_times) == len(candidates):
        raise _fail(
            f"ref_period={ref_period!r} omits every relative period in "
            f"{window_repr}; nothing is left to estimate.",
            "Shrink the reference span or widen the window, e.g. "
            f"window=({min_lag}, {max_lag}) with ref_period=-1.",
        )
    return ref_times, canonical


def _build_bins(
    min_lag: int,
    max_lag: int,
    bin_width: Optional[int],
) -> Dict[int, Tuple[int, int]]:
    """Map each relative period in the window to its ``(start, end)`` bin.

    Bins are anchored at the treatment boundary so that ``tau = -1`` and
    ``tau = 0`` can never fall in the same bin: post-treatment bins tile
    ``[0, k-1], [k, 2k-1], ...`` and pre-treatment bins tile
    ``[-k, -1], [-2k, -k-1], ...``.
    """
    candidates = list(range(min_lag, max_lag + 1))
    if bin_width is None:
        return {t: (t, t) for t in candidates}

    if isinstance(bin_width, (bool, np.bool_)) or not isinstance(
        bin_width, (int, np.integer)
    ):
        raise MethodIncompatibility(
            f"event_study: bin_width={bin_width!r} must be a positive integer.",
            recovery_hint="e.g. bin_width=10 for decade bins, or "
            "bin_width=None for one coefficient per period.",
            diagnostics={"bin_width": repr(bin_width)},
        )
    k = int(bin_width)
    if k <= 0:
        raise MethodIncompatibility(
            f"event_study: bin_width={k} must be strictly positive; a "
            "non-positive bin width does not define any grouping.",
            recovery_hint="e.g. bin_width=10 for decade bins, or "
            "bin_width=None for one coefficient per period.",
            diagnostics={"bin_width": k},
        )

    out: Dict[int, Tuple[int, int]] = {}
    for t in candidates:
        if t >= 0:
            start = (t // k) * k
        else:
            # -1 -> [-k, -1]; -k-1 -> [-2k, -k-1]
            start = -(((-t - 1) // k) + 1) * k
        end = start + k - 1
        # Clip the outermost bins to the window so labels never advertise
        # periods that carry no data.
        out[t] = (max(start, min_lag), min(end, max_lag))
    return out


def event_study(
    data: pd.DataFrame,
    y: str,
    treat_time: str,
    time: str,
    unit: str,
    window: Tuple[int, int] = (-4, 4),
    ref_period: RefPeriodSpec = -1,
    covariates: Optional[List[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    weights: Optional[str] = None,
    bin_width: Optional[int] = None,
    expose_pre_vcov: bool = False,
) -> CausalResult:
    """
    Traditional OLS event study with entity and time fixed effects.

    Generates relative-time dummies around the treatment date, omits a
    reference period (default: t = −1), and estimates with TWFE + optional
    clustering.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data in long format.
    y : str
        Outcome variable.
    treat_time : str
        Column with each unit's treatment time (period when treatment
        starts).  Units never treated should have ``NaN`` or a value
        outside the data range.
    time : str
        Calendar time column (integer or datetime coercible to integer).
    unit : str
        Unit identifier column.
    window : (int, int), default (-4, 4)
        Relative time window (min_lag, max_lag). Periods outside this
        window are binned into the endpoints.
    ref_period : int, (str, int) or sequence of int, default -1
        The omitted reference. Three forms are accepted:

        * ``int`` -- a single omitted period, e.g. ``-1`` (classic).
        * ``(op, bound)`` -- an interval, e.g. ``("<=", -50)`` or
          ``(">=", 20)``. Every relative period in the window satisfying
          the comparison is pooled into the omitted base. Valid operators
          are ``"<="``, ``"<"``, ``">="``, ``">"``.
        * sequence of int -- an explicit span, e.g. ``[-3, -2, -1]``,
          meaning that whole span is the omitted reference.

        Composing an interval reference with ``bin_width`` is the standard
        long-horizon specification (e.g. decade bins with ``tau <= -50`` as
        the omitted base).
    bin_width : int, optional
        Group relative time into bins of this width instead of estimating
        one coefficient per period. Bins are anchored at the treatment
        boundary, so post-treatment bins tile ``[0, k-1], [k, 2k-1], ...``
        and pre-treatment bins tile ``[-k, -1], [-2k, -k-1], ...``;
        ``tau = -1`` and ``tau = 0`` therefore never share a bin. The
        output frame carries ``bin_start`` / ``bin_end`` / ``bin_label``
        columns, and ``relative_time`` is the bin's left edge.
    covariates : list of str, optional
        Additional time-varying controls.
    cluster : str, optional
        Cluster variable for standard errors (default: ``unit``).
    alpha : float, default 0.05
        Significance level.
    weights : str, optional
        Column name for analytical weights (e.g. population weights).
        Equivalent to Stata's ``[aweight=...]``.
    expose_pre_vcov : bool, default False
        Whether to publish the true pre-period covariance matrix into
        ``model_info['vcv_pre']``. When ``True``, downstream honest-DiD and
        pre-trend tools (:func:`pretrends_test`, :func:`pretrends_power`,
        :func:`sensitivity_rr`, :func:`honest_did`) use the full covariance
        of the pre-treatment coefficients. When ``False`` (the default) that
        key is withheld, so those tools fall back to treating the
        pre-coefficients as independent (diagonal covariance) and emit a
        warning saying so. The full covariance is the statistically correct
        input; the diagonal default is retained temporarily for numerical
        continuity with earlier releases and will become the default in a
        future version (flagged as a correctness fix). The full covariance of
        the event-time coefficients is always available in
        ``model_info['vcov']`` / ``['vcov_full']`` regardless of this flag —
        it changes only which path the *pre-trend* tools take.

    Returns
    -------
    CausalResult
        With event study estimates in ``model_info['event_study']``
        (DataFrame with columns: relative_time, estimate, se, ci_lower,
        ci_upper) and a pre-trend test in ``model_info['pretrend_test']``.

        Call ``result.event_study_plot()`` to visualize.

    References
    ----------
    Freyaldenhoven, S., Hansen, C. and Shapiro, J. M. (2019). Pre-event
    Trends in the Panel Event-Study Design. Working paper (Federal
    Reserve Bank of Philadelphia). doi:10.21799/frbp.wp.2019.27
    [@freyaldenhoven2019event]

    Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects
    in event studies with heterogeneous treatment effects. *Journal of
    Econometrics*, 225(2), 175-199. [@sun2021estimating]

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=80, n_periods=8, staggered=True, seed=0)
    >>> result = sp.event_study(df, y='y', treat_time='first_treat',
    ...                         time='time', unit='unit')
    >>> bool('relative_time' in result.model_info['event_study'].columns)
    True
    >>> fig, ax = result.event_study_plot()

    >>> # Narrower relative-time window
    >>> result = sp.event_study(df, y='y', treat_time='first_treat',
    ...                         time='time', unit='unit', window=(-3, 3))
    >>> bool(result.model_info['pretrend_test']['pvalue'] >= 0)
    True
    """
    df = data.copy()
    min_lag, max_lag = int(window[0]), int(window[1])
    if min_lag > max_lag:
        raise MethodIncompatibility(
            f"event_study: window=({min_lag}, {max_lag}) is empty because the "
            "lower bound exceeds the upper bound.",
            recovery_hint=f"e.g. window=({max_lag}, {min_lag}).",
            diagnostics={"window": (min_lag, max_lag)},
        )

    # --- Resolve the omitted reference and the relative-time binning ------- #
    ref_times, ref_canonical = _resolve_ref_set(ref_period, min_lag, max_lag)
    bin_of = _build_bins(min_lag, max_lag, bin_width)

    ref_set = set(ref_times)
    # Group the window's periods into bins and check the reference span is
    # bin-aligned: a bin that is only partly omitted has no coherent meaning.
    bins_members: Dict[Tuple[int, int], List[int]] = {}
    for t in range(min_lag, max_lag + 1):
        bins_members.setdefault(bin_of[t], []).append(t)
    ref_bins = []
    for b, members in bins_members.items():
        in_ref = [t for t in members if t in ref_set]
        if not in_ref:
            continue
        if len(in_ref) != len(members):
            raise MethodIncompatibility(
                f"event_study: the reference span {sorted(ref_set)} cuts bin "
                f"[{b[0]}, {b[1]}] in half (it omits {sorted(in_ref)} but "
                f"keeps {sorted(set(members) - ref_set)}). A partially-omitted "
                "bin has no coherent interpretation.",
                recovery_hint=(
                    "Align the reference to the bin edges, e.g. "
                    f"ref_period=('<=', {b[1]}) or "
                    f"ref_period={list(range(b[0], b[1] + 1))}, or drop "
                    "bin_width."
                ),
                diagnostics={
                    "bin": b,
                    "ref_period": repr(ref_period),
                    "bin_width": bin_width,
                },
            )
        ref_bins.append(b)
    if len(ref_bins) == len(bins_members):
        raise MethodIncompatibility(
            f"event_study: ref_period={ref_period!r} with bin_width="
            f"{bin_width!r} omits every bin in window=({min_lag}, {max_lag}); "
            "nothing is left to estimate.",
            recovery_hint="Shrink the reference span or widen the window.",
            diagnostics={"n_bins": len(bins_members), "ref_period": repr(ref_period)},
        )
    ref_bins_sorted = sorted(ref_bins)
    est_bins = sorted(b for b in bins_members if b not in set(ref_bins))

    # --- Compute relative time ---
    df["__treat_time__"] = df[treat_time]
    df["__time__"] = df[time]
    df["__unit__"] = df[unit]

    # Convert time to numeric if needed. Use ``pd.api.types.is_numeric_dtype``
    # (not ``np.issubdtype(... .dtype, np.number)``) so pandas extension dtypes
    # are handled: under pandas>=3.0 a string column is a ``StringDtype``, which
    # ``np.issubdtype`` cannot interpret (raises ``TypeError``). The truth value
    # is identical to the old check for every numpy numeric dtype, so this only
    # fixes the string/extension path — no numeric behaviour changes.
    if not pd.api.types.is_numeric_dtype(df["__time__"]):
        time_map = {t: i for i, t in enumerate(sorted(df["__time__"].unique()))}
        df["__time_num__"] = df["__time__"].map(time_map)
        if not pd.api.types.is_numeric_dtype(df["__treat_time__"]):
            df["__treat_time_num__"] = df["__treat_time__"].map(time_map)
        else:
            df["__treat_time_num__"] = df["__treat_time__"]
    else:
        df["__time_num__"] = df["__time__"].astype(float)
        df["__treat_time_num__"] = df["__treat_time__"].astype(float)

    # Relative time
    df["__rel_time__"] = df["__time_num__"] - df["__treat_time_num__"]

    # Never-treated units get NaN rel_time — they only contribute via FE
    never_treated = df["__treat_time_num__"].isna()

    # --- Bin endpoints ---
    df.loc[~never_treated, "__rel_time_binned__"] = df.loc[
        ~never_treated, "__rel_time__"
    ].clip(
        lower=min_lag,
        upper=max_lag,
    )

    # --- Create dummies (one per estimated bin) ---
    # ``rel_periods`` keeps its historical meaning -- the ordered list of
    # estimated coefficient labels -- and equals the bin left edges. With
    # bin_width=None every bin is a single period, so this reduces EXACTLY to
    # the previous ``sorted(set(range(...)) - {ref_period})``.
    rel_periods = [b[0] for b in est_bins]
    dummy_cols = []
    for b in est_bins:
        start, end = b
        col = f"__rel_{start}_{end}__"
        df[col] = 0.0
        mask = (~never_treated) & (
            df["__rel_time_binned__"].between(start, end, inclusive="both")
        )
        df.loc[mask, col] = 1.0
        dummy_cols.append(col)

    # --- Build OLS with entity + time FE via demeaning ---
    cov_cols = covariates or []

    # Demean by entity and time (Frisch-Waugh for TWFE)
    all_y_x_cols = [y] + dummy_cols + cov_cols
    dropna_cols = all_y_x_cols + ["__unit__", "__time_num__"]
    if weights is not None:
        dropna_cols.append(weights)
    df_clean = df.dropna(subset=dropna_cols).copy()

    # Prepare weights array (before demeaning)
    if weights is not None:
        w_raw = df_clean[weights].values.astype(float)
        if np.any(w_raw < 0):
            raise ValueError(f"Weights column '{weights}' contains negative values.")
        n_clean = len(df_clean)
        w_arr = w_raw * (n_clean / w_raw.sum())
    else:
        w_arr = None

    Y, X_mat, col_names = _demean_twfe(
        df_clean,
        y,
        dummy_cols + cov_cols,
        "__unit__",
        "__time_num__",
        w=w_arr,
    )

    n, k = X_mat.shape

    # --- OLS (possibly weighted) ---
    if w_arr is not None:
        sqrt_w = np.sqrt(w_arr)
        Xw = X_mat * sqrt_w[:, np.newaxis]
        Yw = Y * sqrt_w
    else:
        Xw = X_mat
        Yw = Y

    try:
        XtX_inv = np.linalg.inv(Xw.T @ Xw)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(Xw.T @ Xw)

    beta = XtX_inv @ Xw.T @ Yw
    resid = Y - X_mat @ beta  # residuals in original scale

    # --- Standard errors (clustered by default) ---
    cluster_var = cluster or unit
    cluster_ids = df_clean[cluster_var].values
    se, vcov = _cluster_se(Xw, resid, XtX_inv, cluster_ids, w=w_arr)

    # --- Build event study table ---
    def _bin_label(start: int, end: int) -> str:
        """Unambiguous bin label: a point period prints as-is."""
        return str(start) if start == end else f"[{start}, {end}]"

    es_rows = []
    for i, b in enumerate(est_bins):
        k_val = b[0]
        coef = float(beta[i])
        se_i = float(se[i])
        t_crit = sp_stats.norm.ppf(1 - alpha / 2)
        es_rows.append(
            {
                "relative_time": k_val,
                "bin_start": b[0],
                "bin_end": b[1],
                "bin_label": _bin_label(b[0], b[1]),
                "is_reference": False,
                # ``att`` is the canonical event-study coefficient name shared by
                # the whole DID family (see did._core.EVENT_STUDY_COLUMNS); the
                # downstream plotters / exporters / pretrend tools key on it.
                # ``estimate`` is kept as a backward-compatible alias.
                "att": coef,
                "estimate": coef,
                "se": se_i,
                "ci_lower": coef - t_crit * se_i,
                "ci_upper": coef + t_crit * se_i,
                "pvalue": (
                    float(2 * (1 - sp_stats.norm.cdf(abs(coef / se_i))))
                    if se_i > 0
                    else 1.0
                ),
            }
        )

    # Add the omitted reference bin(s) (zero by definition)
    for b in ref_bins_sorted:
        es_rows.append(
            {
                "relative_time": b[0],
                "bin_start": b[0],
                "bin_end": b[1],
                "bin_label": _bin_label(b[0], b[1]),
                "is_reference": True,
                "att": 0.0,
                "estimate": 0.0,
                "se": 0.0,
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "pvalue": 1.0,
            }
        )
    event_study_df = (
        pd.DataFrame(es_rows).sort_values("relative_time").reset_index(drop=True)
    )

    # --- Covariance matrix of the event-study coefficients ---------------- #
    # ``vcov`` above covers [event-time dummies, covariates]; slice off the
    # covariate block so ``vcov_event`` is coefficient-aligned with
    # ``rel_periods``.  ``vcv_pre`` is the pre-period submatrix laid out in the
    # same row order as ``event_study_df`` restricted to relative_time < 0
    # (reference period included as an exact-zero row/column, since it is
    # normalised to 0 by construction).  ``pretrends._pre_vcv`` reads that key
    # and accepts either the (K, K) estimated-only block or this
    # (K_all, K_all) form.
    n_ev = len(rel_periods)
    vcov_event = vcov[:n_ev, :n_ev]

    pre_times = [int(t) for t in event_study_df["relative_time"].tolist() if int(t) < 0]
    pos_of = {t: i for i, t in enumerate(rel_periods)}
    K_all_pre = len(pre_times)
    vcv_pre = np.zeros((K_all_pre, K_all_pre), dtype=float)
    for a, t_a in enumerate(pre_times):
        if t_a not in pos_of:  # reference period row stays all-zero
            continue
        for b, t_b in enumerate(pre_times):
            if t_b not in pos_of:
                continue
            vcv_pre[a, b] = vcov_event[pos_of[t_a], pos_of[t_b]]

    # --- Pre-trend test (joint F-test on pre-treatment coefficients) ---
    pre_indices = [i for i, k_val in enumerate(rel_periods) if k_val < 0]
    pretrend_result = _joint_f_test(
        beta,
        XtX_inv,
        pre_indices,
        resid,
        n,
        k,
        w=w_arr,
    )

    # --- Overall ATT (average of post-treatment coefficients) ---
    post = event_study_df[event_study_df["relative_time"] >= 0]
    post_nonref = post[~post["is_reference"]]
    att = float(post_nonref["estimate"].mean()) if len(post_nonref) > 0 else 0.0
    att_se = (
        float(np.sqrt(np.mean(post_nonref["se"] ** 2) / len(post_nonref)))
        if len(post_nonref) > 0
        else 0.0
    )
    att_p = float(2 * (1 - sp_stats.norm.cdf(abs(att / att_se)))) if att_se > 0 else 1.0

    n_clusters = len(np.unique(cluster_ids))

    _result = CausalResult(
        method="OLS Event Study (TWFE)",
        estimand="ATT",
        estimate=att,
        se=att_se,
        pvalue=att_p,
        ci=(
            att - sp_stats.norm.ppf(1 - alpha / 2) * att_se,
            att + sp_stats.norm.ppf(1 - alpha / 2) * att_se,
        ),
        alpha=alpha,
        n_obs=n,
        detail=event_study_df,
        model_info={
            "model_type": "DID Event Study",
            "event_study": event_study_df,
            "pretrend_test": pretrend_result,
            # Full cluster-robust covariance of the event-time coefficients
            # (covariate block excluded). Exposed for inspection: nothing
            # downstream reads these two keys to change a result, so they are
            # always available and move no published numbers.
            "vcov": vcov_event,
            "vcov_full": vcov,
            # ``vcv_pre`` is the ONE key that flips pretrends_test /
            # pretrends_power / sensitivity_rr / honest_did from the historical
            # diagonal (independent pre-coefficients) approximation onto the
            # true pre-period covariance — a genuine correctness fix, but one
            # that MOVES numbers published under earlier releases. Per the
            # maintainer's decision it stays opt-in during the live JOSS
            # review: by default we withhold it, so those tools take their
            # diagonal path AND emit the loud _pre_vcv warning. Set
            # ``expose_pre_vcov=True`` to opt into the accurate covariance now.
            # TODO(post-JOSS): flip this default to always-on and log it as a
            # ⚠️ correctness fix in CHANGELOG + MIGRATION.
            "vcv_pre": vcv_pre if expose_pre_vcov else None,
            "vcov_event_times": list(rel_periods),
            "vcov_pre_times": list(pre_times),
            "ref_period": ref_canonical,
            "ref_periods": list(ref_times),
            "ref_bins": [list(b) for b in ref_bins_sorted],
            "bin_width": bin_width,
            "bins": [list(b) for b in est_bins],
            "window": window,
            "n_clusters": n_clusters,
            "cluster_var": cluster_var,
            "weights": weights,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.event_study",
            params={
                "y": y,
                "treat_time": treat_time,
                "time": time,
                "unit": unit,
                "window": list(window),
                "ref_period": (
                    list(ref_canonical)
                    if isinstance(ref_canonical, (list, tuple))
                    else ref_canonical
                ),
                "bin_width": bin_width,
                "covariates": list(covariates) if covariates else None,
                "cluster": cluster,
                "alpha": alpha,
                "weights": weights,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ====================================================================== #
#  Internal helpers
# ====================================================================== #


def _demean_twfe(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
    unit_col: str,
    time_col: str,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Demean Y and X by entity and time means (within transformation).

    If *w* is provided, uses weighted means for demeaning (WLS-FE).
    """
    cols = [y_col] + x_cols
    data_mat = df[cols].values.astype(np.float64)

    # Entity means
    unit_ids = df[unit_col].values
    unique_units = np.unique(unit_ids)
    for u in unique_units:
        mask = unit_ids == u
        if w is not None:
            wm = w[mask]
            ws = wm.sum()
            if ws > 0:
                wmean = (wm[:, np.newaxis] * data_mat[mask]).sum(axis=0) / ws
            else:
                wmean = data_mat[mask].mean(axis=0)
            data_mat[mask] -= wmean
        else:
            data_mat[mask] -= data_mat[mask].mean(axis=0)

    # Time means (on already entity-demeaned data)
    time_ids = df[time_col].values
    unique_times = np.unique(time_ids)
    for t in unique_times:
        mask = time_ids == t
        if w is not None:
            wm = w[mask]
            ws = wm.sum()
            if ws > 0:
                wmean = (wm[:, np.newaxis] * data_mat[mask]).sum(axis=0) / ws
            else:
                wmean = data_mat[mask].mean(axis=0)
            data_mat[mask] -= wmean
        else:
            data_mat[mask] -= data_mat[mask].mean(axis=0)

    Y = data_mat[:, 0]
    X = data_mat[:, 1:]
    return Y, X, x_cols


def _cluster_se(
    X: np.ndarray,
    resid: np.ndarray,
    XtX_inv: np.ndarray,
    cluster_ids: np.ndarray,
    w: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cluster-robust standard errors **and** the full covariance matrix.

    *X* should already be the weighted design matrix (Xw) if weights are used.
    *resid* should be unweighted residuals; weighting is applied here via *w*.

    Returns
    -------
    (se, vcov)
        ``se`` is ``sqrt(diag(vcov))`` (clipped at zero); ``vcov`` is the
        full (k, k) cluster-robust covariance matrix.  The covariance
        matrix is returned so that downstream pre-trend / honest-DiD tools
        can use the *true* joint distribution of the event-study
        coefficients rather than assuming independence across relative
        periods (they share a reference period and the same fixed
        effects, so the off-diagonal terms are large).
    """
    n, k = X.shape
    unique_clusters = np.unique(cluster_ids)
    G = len(unique_clusters)

    meat = np.zeros((k, k))
    for c in unique_clusters:
        mask = cluster_ids == c
        if w is not None:
            score_c = (X[mask] * (np.sqrt(w[mask]) * resid[mask])[:, None]).sum(axis=0)
        else:
            score_c = (X[mask] * resid[mask, None]).sum(axis=0)
        meat += np.outer(score_c, score_c)

    correction = (G / (G - 1)) * ((n - 1) / (n - k))
    vcov = correction * XtX_inv @ meat @ XtX_inv
    # Symmetrise: the sandwich is symmetric in exact arithmetic, but the
    # matrix products introduce O(1e-16) asymmetry that makes downstream
    # Cholesky / inverse calls fussy.
    vcov = 0.5 * (vcov + vcov.T)
    se = np.asarray(np.sqrt(np.maximum(np.diag(vcov), 0)), dtype=float)
    return se, np.asarray(vcov, dtype=float)


def _joint_f_test(
    beta: np.ndarray,
    XtX_inv: np.ndarray,
    indices: List[int],
    resid: np.ndarray,
    n: int,
    k: int,
    w: Optional[np.ndarray] = None,
) -> dict:
    """Joint F-test for subset of coefficients being zero."""
    if not indices:
        return {"statistic": 0.0, "pvalue": 1.0, "df": 0}

    q = len(indices)
    beta_sub = beta[np.array(indices)]

    # Submatrix of variance
    idx = np.array(indices)
    V_sub = XtX_inv[np.ix_(idx, idx)]
    if w is not None:
        sigma2 = np.sum(w * resid**2) / (n - k)
    else:
        sigma2 = np.sum(resid**2) / (n - k)

    try:
        V_inv = np.linalg.inv(sigma2 * V_sub)
        f_stat = float(beta_sub @ V_inv @ beta_sub / q)
    except np.linalg.LinAlgError:
        f_stat = 0.0

    pvalue = float(1 - sp_stats.f.cdf(f_stat, q, n - k))

    return {
        "statistic": round(f_stat, 4),
        "pvalue": round(pvalue, 4),
        "df": q,
        "test": "Joint F-test on pre-treatment coefficients",
    }
