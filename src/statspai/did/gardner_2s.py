r"""
Gardner (2022) two-stage DID estimator (a.k.a. ``did2s``).

The **two-stage DID** method of Gardner (2022) recovers the ATT under staggered
treatment adoption by a two-step regression:

1. **Stage 1 — Fit FE model on untreated rows only.**
   Using observations where the unit is *not yet* treated, regress the outcome
   on unit and time fixed effects (plus any covariates):

       Y_it = alpha_i + lambda_t + X_it' beta + e_it    for (i, t) untreated.

2. **Stage 2 — Residualise + regress on treatment.**
   Construct the residualised outcome  Y_tilde_it = Y_it - (predicted from
   Stage 1), and fit a pooled regression, without intercept, on treatment
   dummies (either a single ATT or an event-study by relative time):

       Y_tilde_it = tau * D_it + u_it.

The default standard errors (``vce='analytic'``) are the two-stage corrected
clustered variance of Butts & Gardner (2022): the influence function of the
Stage-2 coefficients is adjusted for the estimation of the Stage-1 fixed
effects,

    IF_i = (X2'X2)^{-1} [ X2'X1 (X10'X10)^{-1} x10_i e1_i  -  x2_i e2_i ],

where ``X1`` is the Stage-1 design on every row, ``X10`` the same design with
treated rows zeroed, ``e1`` the Stage-1 residual (zero on treated rows), ``X2``
the Stage-2 design and ``e2`` the Stage-2 residual; the variance is the sum
over clusters of the outer product of the summed contributions, with no
small-sample factor. This is the construction of R ``did2s`` 1.2.1 and Stata
``did2s`` v0.5, which StatsPAI reproduces to ~1e-8 relative on the
castle-doctrine panel (``tests/test_gardner_did2s_reference.py``).
``vce='bootstrap'`` resamples whole clusters and re-runs both stages. The
estimator closely parallels the Borusyak-Jaravel-Spiess (2024) imputation
estimator numerically, but the two-step regression framing makes event studies
and covariate interactions trivial.

References
----------
Gardner, J. (2022).  "Two-stage differences in differences."
    arXiv:2207.05943. [@gardner2022twostage]
Butts, K. and Gardner, J. (2022).  "did2s: Two-Stage Difference-in-Differences."
    *R Journal*, 14(3), 162-173. [@butts2022stage]
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from ..core._bootstrap import bootstrap_se as _bootstrap_se
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["gardner_did", "did_2stage"]


def _gardner_cluster_bootstrap(
    df: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: List[str],
    event_study: bool,
    horizon: Optional[List[int]],
    cluster: str,
    names: List[str],
    n_boot: int,
    seed: int,
    weights: Optional[str],
) -> Tuple[Dict[str, float], float]:
    """Pairs-cluster bootstrap of the *full* Gardner two-step procedure.

    Resamples whole clusters and re-runs both stages on every replicate,
    so the bootstrap distribution carries the Stage-1 estimation error as
    well as the Stage-2 one. Returns ``(per_coef_se, overall_att_se)``.
    """
    rng = np.random.default_rng(seed)
    clusters = pd.unique(df[cluster])
    n_g = len(clusters)
    rows_by_cluster = {c: df[df[cluster] == c] for c in clusters}
    ctrl = controls if controls else None

    n_boot_int = int(n_boot)
    # Pre-size to the full attempt count so failed replicates stay NaN and
    # bootstrap_se can surface the failure rate loudly (CLAUDE.md §3.7)
    # instead of silently computing the SD over survivors only.
    boot_overall = np.full(n_boot_int, np.nan, dtype=float)
    boot_coefs: Dict[str, np.ndarray] = {
        nm: np.full(n_boot_int, np.nan, dtype=float) for nm in names
    }
    for b in range(n_boot_int):
        drawn = rng.choice(n_g, size=n_g, replace=True)
        parts = []
        for j, ci in enumerate(drawn):
            sub = rows_by_cluster[clusters[ci]].copy()
            # Fresh, globally-unique ids so a cluster drawn twice (and the
            # units inside it) do not collide in the re-estimation.
            sub[group] = sub[group].astype(str) + f"__b{j}"
            sub["__bcl"] = j
            parts.append(sub)
        bd = pd.concat(parts, ignore_index=True)
        try:
            r = gardner_did(
                bd,
                y=y,
                group=group,
                time=time,
                first_treat=first_treat,
                controls=ctrl,
                event_study=event_study,
                horizon=horizon,
                cluster="__bcl",
                vce="none",
                weights=weights,
            )
        except Exception:
            continue  # replicate stays NaN; bootstrap_se tracks the failure
        if np.isfinite(r.estimate):
            boot_overall[b] = float(r.estimate)
        es = r.model_info.get("event_study") if event_study else None
        if es:
            for nm in names:
                boot_coefs[nm][b] = float(es["coef"].get(nm, np.nan))

    overall_se = _bootstrap_se(boot_overall, label="did.gardner.overall")
    if not event_study:
        # Static mode: the single coefficient *is* the overall ATT (the
        # per-name arrays are only filled from the event-study dict).
        return {names[0]: overall_se}, overall_se
    se_dict: Dict[str, float] = {}
    for nm in names:
        se_dict[nm] = _bootstrap_se(boot_coefs[nm], label=f"did.gardner[{nm}]")
    return se_dict, overall_se


def _cluster_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    cluster: np.ndarray,
) -> np.ndarray:
    """Liang-Zeger cluster-robust variance for an OLS coefficient vector.

    Carries the ``G/(G-1) (n-1)/(n-k)`` factor. Used only by the legacy
    ``vce='stage2'`` mode, which treats the Stage-1 fit as known.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    clusters = np.unique(cluster)
    G = len(clusters)
    meat = np.zeros((k, k))
    for g in clusters:
        mask = cluster == g
        s = X[mask].T @ resid[mask]
        meat += np.outer(s, s)
    if G > 1 and n > k:
        dof = G / (G - 1) * (n - 1) / (n - k)
    else:
        dof = 1.0
    return np.asarray(dof * xtx_inv @ meat @ xtx_inv, dtype=float)


def _stage2_bin_se(y_k: np.ndarray, cl_k: np.ndarray) -> float:
    """Legacy Stage-2-only cluster SE of an (unweighted) within-bin mean."""
    uniq = np.unique(cl_k)
    coef_k = float(np.mean(y_k))
    if len(uniq) > 1:
        sq = 0.0
        for g in uniq:
            sq += float(np.sum(y_k[cl_k == g] - coef_k)) ** 2
        return float(np.sqrt(max(sq / (len(y_k) ** 2), 0.0)))
    return float(np.std(y_k, ddof=1) / np.sqrt(len(y_k)))


def _did2s_vcov(
    A_un_w: sparse.spmatrix,
    e1_w: np.ndarray,
    A_full_w: sparse.spmatrix,
    X2_w: np.ndarray,
    e2_w: np.ndarray,
    untreated_mask: np.ndarray,
    cluster: np.ndarray,
) -> np.ndarray:
    """Butts-Gardner (2022) two-stage corrected clustered covariance.

    Every input is already scaled by ``sqrt(weight)`` (the identity for an
    unweighted fit), so the algebra below is the R ``did2s`` one verbatim:

    * ``IF_ss``  = (X2'X2)^{-1} x2_i e2_i                    (Stage-2 term)
    * ``IF_fs``  = (X2'X2)^{-1} Γ' x10_i e1_i,  Γ = (X10'X10)^{-1} X1'X2
                                                            (Stage-1 term)
    * ``V``      = Σ_g s_g s_g',  s_g = Σ_{i∈g} (IF_fs,i − IF_ss,i)

    ``X10`` is the Stage-1 design with treated rows zeroed, which is the
    untreated-row block ``A_un_w`` here; ``e1`` is zero on treated rows by
    construction, so only untreated rows carry a Stage-1 term. No
    small-sample factor is applied (``did2s`` applies none).

    Parameters
    ----------
    A_un_w : (n_un, p) Stage-1 design on the untreated rows.
    e1_w : (n_un,) Stage-1 residuals on the untreated rows.
    A_full_w : (n, p) Stage-1 design evaluated on every row (``X1``).
    X2_w : (n, k) Stage-2 design.
    e2_w : (n,) Stage-2 residuals.
    untreated_mask : (n,) bool, True where the row entered Stage 1.
    cluster : (n,) cluster labels.
    """
    n, k = X2_w.shape
    cl_codes, cl_idx = np.unique(cluster, return_inverse=True)
    n_clusters = len(cl_codes)
    if n_clusters < 2:
        raise DataInsufficient(
            "gardner_did: the cluster-robust variance needs at least two "
            f"clusters, got {n_clusters}; the summed influence functions of a "
            "single cluster are identically zero. Pass a finer cluster= "
            "(the default clusters on the unit) or vce='none'.",
            diagnostics={"n_clusters": int(n_clusters)},
        )

    xtx = X2_w.T @ X2_w
    try:
        m_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        m_inv = np.linalg.pinv(xtx)

    # Γ = (X10'X10)^{-1} X1'X2. Cholesky when the Stage-1 design has full
    # column rank; otherwise the minimum-norm solution, which is what
    # did2s::robust_solve_XtX falls back to (pseudo-inverse). Rows of X1 in
    # the row space of X10 receive the same adjustment either way.
    # Sparse normal equations of the Stage-1 design (the dense Cholesky
    # of an (N+T) x (N+T) matrix built from dense dummies did not scale).
    atx2 = np.asarray(A_full_w.T @ X2_w)
    gamma = _sparse_normal_solve(sparse.csr_matrix(A_un_w), atx2)

    # Per-observation influence contributions, one row per observation:
    #   IF_i = e1_i a_i' Γ M  -  e2_i x2_i' M      (M symmetric)
    IF = -(X2_w * e2_w[:, None]) @ m_inv
    IF[untreated_mask] += np.asarray(sparse.diags(e1_w) @ A_un_w @ gamma) @ m_inv

    S = np.zeros((n_clusters, k), dtype=float)
    np.add.at(S, cl_idx, IF)
    return np.asarray(S.T @ S, dtype=float)


def _build_fe_design(
    unit: np.ndarray,
    time: np.ndarray,
    X: Optional[np.ndarray],
    *,
    u_levels: Optional[np.ndarray] = None,
    t_levels: Optional[np.ndarray] = None,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Build (intercept, unit dummies minus one, time dummies minus one, X).

    When ``u_levels``/``t_levels`` are provided, builds the design against that
    reference set of levels (useful for prediction on a different sample).
    Returned sparse: the dense dummy matrix was ``n x (N + T)`` and the
    dense least-squares on it failed ("SVD did not converge") by
    n = 20,000 panel rows.
    """
    if u_levels is None:
        u_levels = np.unique(unit)
    if t_levels is None:
        t_levels = np.unique(time)

    n = len(unit)
    rows = np.arange(n)

    def _dummies(codes_src: np.ndarray, levels: np.ndarray) -> sparse.csr_matrix:
        # Levels after the first (dropped) level get a column each.
        idx = np.searchsorted(levels, codes_src)
        ok = (idx < len(levels)) & (
            levels[np.minimum(idx, len(levels) - 1)] == codes_src
        )
        keep = ok & (idx > 0)
        return sparse.csr_matrix(
            (np.ones(int(keep.sum())), (rows[keep], idx[keep] - 1)),
            shape=(n, max(len(levels) - 1, 0)),
        )

    parts = [
        sparse.csr_matrix(np.ones((n, 1))),
        _dummies(np.asarray(unit), np.asarray(u_levels)),
        _dummies(np.asarray(time), np.asarray(t_levels)),
    ]
    if X is not None and X.size > 0:
        parts.append(sparse.csr_matrix(X))
    A = sparse.hstack(parts, format="csr")
    return A, u_levels, t_levels


def _sparse_normal_solve(
    A: sparse.csr_matrix, B: np.ndarray, *, keep: Optional[np.ndarray] = None
) -> np.ndarray:
    """``(A'A)^+ B`` for a sparse FE design, one row per column of ``A``.

    Columns of ``A`` that are identically zero (a unit seen only in treated
    rows) get zero rows -- the minimum-norm answer ``lstsq`` gave them --
    and are dropped before a sparse LU of the column-equilibrated normal
    matrix. Falls back to a dense least-squares solve if that matrix is
    singular (collinear FE beyond the zero columns).
    """
    p = A.shape[1]
    if keep is None:
        keep = np.asarray(A.getnnz(axis=0) > 0).ravel()
    Ak = A[:, keep]
    scale = np.sqrt(np.asarray(Ak.multiply(Ak).sum(axis=0)).ravel())
    scale[scale == 0] = 1.0
    D = sparse.diags(1.0 / scale)
    As = Ak @ D
    M = (As.T @ As).tocsc()
    B2 = np.asarray(B, dtype=float).reshape(p, -1)
    out = np.zeros_like(B2)
    rhs = D @ B2[keep]
    try:
        from scipy.sparse.linalg import splu

        sol = splu(M).solve(rhs)
        if not np.all(np.isfinite(sol)):
            raise RuntimeError("non-finite sparse solve")
    except (RuntimeError, ValueError):
        dense = As.toarray()
        sol = np.linalg.lstsq(dense.T @ dense, rhs, rcond=None)[0]
    out[keep] = D @ sol
    return out[:, 0] if B.ndim == 1 else out


@accepts_aliases(_strict=True, id="group", unit="group", covariates="controls")
def gardner_did(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    first_treat: str,
    controls: Optional[List[str]] = None,
    event_study: bool = False,
    horizon: Optional[List[int]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    vce: str = "analytic",
    n_boot: int = 199,
    boot_seed: int = 0,
    weights: Optional[str] = None,
) -> CausalResult:
    """Gardner (2022) two-stage DID estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    y : str
        Outcome column name.
    group : str
        Unit (panel-id) column.
    time : str
        Time column.
    first_treat : str
        First-treatment-period column.  Never-treated units should be encoded
        as ``0``, ``NaN``, or ``+inf``.
    controls : list of str, optional
        Additional covariates included in Stage 1.
    event_study : bool, default False
        If True, Stage 2 reports coefficients by relative time
        ``k = t - first_treat_i``.
    horizon : list of int, optional
        Relative-time leads/lags to report when ``event_study=True``;
        defaults to ``range(-5, 6)`` intersected with available support.
        Each reported coefficient is the (weighted) mean of the residualised
        outcome at that relative time; the Stage-2 regression has no
        intercept, so the set of horizons requested does not change any
        individual coefficient or its standard error.
    cluster : str, optional
        Cluster variable for the standard errors.  Defaults to ``group``.
        At least two clusters are required.
    alpha : float, default 0.05
        Two-sided CI level.
    vce : {'analytic', 'bootstrap', 'none'}, default 'analytic'
        Standard-error mode. ``'analytic'`` is the two-stage corrected
        clustered variance of Butts & Gardner (2022): the Stage-2 influence
        function is adjusted for the estimation of the Stage-1 fixed effects
        (see the module docstring), which is what R ``did2s`` and Stata
        ``did2s`` report and what StatsPAI reproduces to ~1e-8 relative.
        ``'bootstrap'`` resamples whole clusters and re-runs the full
        two-step procedure. ``'none'`` skips inference (``se``, ``ci`` and
        ``pvalue`` are NaN); it is what the bootstrap uses internally.
        Point estimates are identical in every mode.
    n_boot : int, default 199
        Number of cluster-bootstrap replications when ``vce='bootstrap'``.
    boot_seed : int, default 0
        Seed for the cluster bootstrap (deterministic results).
    weights : str, optional
        Column of strictly positive estimation weights applied to both
        stages (weighted least squares), the counterpart of R
        ``did2s(weights=)`` and Stata ``did2s [aw=]``. In event-study mode
        the overall ATT is the weight-share average of the post-treatment
        coefficients.

    Returns
    -------
    CausalResult
        ``.estimate`` is the overall ATT; ``.model_info['event_study']``
        carries the event-study dict when requested, and
        ``.model_info['vcov']`` / ``['cell_labels']`` the covariance of the
        Stage-2 coefficients (``vce='analytic'`` only).  Supplies
        ``.summary()``, ``.cite()``, and is compatible with ``sp.outreg2()``.

    Notes
    -----
    Identification requires the usual staggered-DID conditions (parallel
    trends, no anticipation) plus a linear two-way FE + additive covariate
    structure for the untreated potential outcome.

    In event-study mode ``.estimate`` is the weight-share average of the
    post-treatment coefficients (the ``did2s`` aggregated-ATT convention),
    which equals the static ATT whenever every post-treatment horizon is in
    ``horizon``; its analytic SE is the delta-method SE ``sqrt(a' V a)`` on
    the full Stage-2 covariance, which is *not* the static-ATT SE because the
    saturated Stage-2 residuals differ from the single-dummy ones.

    References
    ----------
    Gardner, J. (2022). Two-stage differences in differences.
    arXiv:2207.05943. [@gardner2022twostage]
    Butts, K. and Gardner, J. (2022). did2s: Two-Stage
    Difference-in-Differences. *R Journal*, 14(3), 162-173. [@butts2022stage]

    Examples
    --------
    Staggered panel with a never-treated group (``first_treat = 0``).
    ``sp.did_2stage`` is an alias of this function.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n_units, n_periods = 40, 8
    >>> unit = np.repeat(np.arange(n_units), n_periods)
    >>> time = np.tile(np.arange(1, n_periods + 1), n_units)
    >>> first = np.where(unit < 20, 5, 0)  # 0 = never treated
    >>> d = ((first > 0) & (time >= first)).astype(float)
    >>> y = (0.5 * unit + 0.3 * time + 2.0 * d
    ...      + rng.normal(0, 0.5, unit.size))
    >>> df = pd.DataFrame(
    ...     {"y": y, "unit": unit, "time": time, "g": first}
    ... )
    >>> res = sp.gardner_did(
    ...     df, y="y", group="unit", time="time", first_treat="g"
    ... )
    >>> round(res.estimate, 2)  # true ATT = 2.0
    2.03
    """
    if vce not in ("analytic", "stage2", "bootstrap", "none"):
        raise ValueError(
            "vce must be 'analytic', 'stage2', 'bootstrap', or 'none'; " f"got {vce!r}"
        )
    if vce == "stage2" and weights is not None:
        raise MethodIncompatibility(
            "gardner_did: vce='stage2' is the unweighted pre-correction SE kept "
            "only to reproduce earlier output; it has no weighted form. Use "
            "vce='analytic' (the did2s-corrected variance) with weights=.",
            diagnostics={"vce": "stage2"},
        )
    if controls is None:
        controls = []
    df = data.copy()
    required = [y, group, time, first_treat] + list(controls)
    if weights is not None:
        required.append(weights)
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in data")

    df[y] = pd.to_numeric(df[y], errors="coerce")
    subset = [y, group, time, first_treat]
    if weights is not None:
        df[weights] = pd.to_numeric(df[weights], errors="coerce")
        subset.append(weights)
    df = df.dropna(subset=subset).reset_index(drop=True)

    if cluster is None:
        cluster = group
    elif cluster not in df.columns:
        raise ValueError(f"cluster column '{cluster}' not found")

    n = len(df)
    if weights is not None:
        w = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise MethodIncompatibility(
                f"weights column '{weights}' must be finite and strictly "
                "positive; drop zero-weight rows before calling gardner_did",
                diagnostics={"weights": weights},
            )
    else:
        w = np.ones(n, dtype=float)
    sw = np.sqrt(w)

    ft = df[first_treat].to_numpy(dtype=float)
    t_arr = df[time].to_numpy(dtype=float)
    treated_now = np.isfinite(ft) & (ft > 0) & (t_arr >= ft)
    df["_D"] = treated_now.astype(float)

    # ── Stage 1: FE + covariate regression on untreated rows ────── #
    untreated_mask = ~treated_now
    if untreated_mask.sum() < 10:
        raise ValueError("Not enough untreated observations for Stage 1 (<10).")

    unit_all = df[group].to_numpy()
    time_all = df[time].to_numpy()
    X_all = df[controls].to_numpy(dtype=float) if controls else np.zeros((n, 0))

    # Stage-1 fit: design against ALL units/times seen in the data; rows from
    # untreated subset only.  Units that appear only in treated rows simply
    # get a zero dummy — they contribute nothing to the untreated fit but can
    # still be predicted (via intercept + time FE only).
    u_levels = np.unique(unit_all)
    t_levels = np.unique(time_all)

    A_un, _, _ = _build_fe_design(
        unit_all[untreated_mask],
        time_all[untreated_mask],
        X_all[untreated_mask] if controls else None,
        u_levels=u_levels,
        t_levels=t_levels,
    )
    y_un = df.loc[untreated_mask, y].to_numpy(dtype=float)
    sw_un = sw[untreated_mask]
    A_un_w = A_un if weights is None else sparse.diags(sw_un) @ A_un

    coefs = _sparse_normal_solve(A_un_w, A_un_w.T @ (y_un * sw_un))

    # Predict counterfactual Y(0) for all rows using Stage-1 coefficients.
    A_full, _, _ = _build_fe_design(
        unit_all,
        time_all,
        X_all if controls else None,
        u_levels=u_levels,
        t_levels=t_levels,
    )
    y_all_arr = df[y].to_numpy(dtype=float)
    y_hat_0 = A_full @ coefs
    y_tilde = y_all_arr - y_hat_0
    # Stage-1 residuals; zero on treated rows in did2s, so only the
    # untreated block is carried.
    e1_un = y_tilde[untreated_mask]

    # ── Stage 2: regress ỹ (no intercept) on treatment dummies ──── #
    # Overall ATT: ỹ ~ 0 + D, so the coefficient is the (weighted) mean of
    # ỹ over treated rows. Event study: ỹ ~ 0 + Σ_k 1{rel = k}, so every
    # coefficient is the (weighted) within-bin mean — the did2s construction
    # (``second_stage = ~ i(rel, ref = ...)`` with fixest's ``~ 0 +``). No
    # reference category enters, which is what keeps the leads free of the
    # contamination a dummy regression with intercept would introduce.
    cl = df[cluster].to_numpy()
    if event_study:
        rel_time = np.where(
            np.isfinite(ft) & (ft > 0),
            t_arr - ft,
            np.inf,  # never-treated → excluded
        )
        if horizon is None:
            support = np.unique(rel_time[np.isfinite(rel_time)])
            horizon = [int(k) for k in support if -5 <= int(k) <= 5]
            if 0 not in horizon:
                horizon.append(0)
            horizon = sorted(set(horizon))
        horizon = [int(k) for k in horizon]
        names = [f"D_k{k:+d}" for k in horizon]
        bin_masks = [rel_time == k for k in horizon]
        count_list = [int(m.sum()) for m in bin_masks]
        supported = [j for j, n_k in enumerate(count_list) if n_k > 0]
        X2 = (
            np.column_stack([bin_masks[j].astype(float) for j in supported])
            if supported
            else np.zeros((n, 0))
        )
    else:
        names = ["ATT"]
        count_list = [int(treated_now.sum())]
        if count_list[0] == 0:
            raise DataInsufficient(
                "gardner_did: no treated observations (first_treat never "
                "reached within the sample), the ATT is undefined."
            )
        supported = [0]
        X2 = df["_D"].to_numpy(dtype=float).reshape(-1, 1)

    k2 = X2.shape[1]
    X2_w = X2 if weights is None else X2 * sw[:, None]
    if k2 > 0:
        coef2, *_ = np.linalg.lstsq(X2_w, y_tilde * sw, rcond=None)
    else:
        coef2 = np.zeros(0, dtype=float)
    e2 = y_tilde - X2 @ coef2
    wsum_supported = X2_w.T @ sw if k2 > 0 else np.zeros(0)  # Σ w per column

    est = np.full(len(names), np.nan, dtype=float)
    est[supported] = coef2
    coef_dict = dict(zip(names, est))
    count_dict = dict(zip(names, count_list))
    supported_names = [names[j] for j in supported]

    # ── Inference ─────────────────────────────────────────────────── #
    V: Optional[np.ndarray] = None
    se_dict: Dict[str, float] = {nm: float("nan") for nm in names}
    boot_overall_se: Optional[float] = None
    n_clusters = int(pd.unique(cl).size)
    if vce == "analytic" and k2 > 0:
        A_full_w = A_full if weights is None else sparse.diags(sw) @ A_full
        V = _did2s_vcov(
            A_un_w,
            e1_un * sw_un,
            A_full_w,
            X2_w,
            e2 * sw,
            untreated_mask,
            cl,
        )
        se_supported = np.sqrt(np.clip(np.diag(V), 0.0, None))
        for nm, s in zip(supported_names, se_supported):
            se_dict[nm] = float(s)
    elif vce == "stage2" and k2 > 0:
        warnings.warn(
            "gardner_did: vce='stage2' clusters the Stage-2 residuals only and "
            "ignores the variance from estimating the Stage-1 fixed effects, "
            "so it understates uncertainty (about 26% low on mpdta; "
            "empirically ~0.78 coverage at a nominal 95% level). It is kept "
            "only to reproduce earlier output; the default vce='analytic' is "
            "the did2s-corrected two-stage variance.",
            UserWarning,
            stacklevel=2,
        )
        if event_study:
            for j in supported:
                m = bin_masks[j]
                se_dict[names[j]] = _stage2_bin_se(y_tilde[m], cl[m])
        else:
            design2 = np.column_stack([np.ones(n), X2])
            coef_s2, *_ = np.linalg.lstsq(design2, y_tilde, rcond=None)
            V_s2 = _cluster_vcov(design2, y_tilde - design2 @ coef_s2, cl)
            se_dict[names[0]] = float(np.sqrt(max(V_s2[1, 1], 0.0)))
    elif vce == "bootstrap":
        se_dict, boot_overall_se = _gardner_cluster_bootstrap(
            df,
            y,
            group,
            time,
            first_treat,
            controls,
            event_study,
            horizon if event_study else None,
            cluster,
            names,
            n_boot,
            boot_seed,
            weights,
        )

    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci = {
        k: (coef_dict[k] - z * se_dict[k], coef_dict[k] + z * se_dict[k]) for k in names
    }

    if event_study:
        # Treated-weight-share average of the post-treatment coefficients
        # (the did2s aggregated-ATT convention); equals the static ATT when
        # every post horizon is in ``horizon``. An *unweighted* mean disagrees
        # with it whenever the horizons have unbalanced support.
        horizons_int: List[int] = list(horizon or [])
        post_pos = [p for p, j in enumerate(supported) if horizons_int[j] >= 0]
        if post_pos:
            shares = wsum_supported[post_pos] / wsum_supported[post_pos].sum()
            att_overall = float(np.dot(shares, coef2[post_pos]))
            if vce == "bootstrap" and boot_overall_se is not None:
                att_se = float(boot_overall_se)
            elif V is not None:
                V_post = V[np.ix_(post_pos, post_pos)]
                att_se = float(np.sqrt(max(float(shares @ V_post @ shares), 0.0)))
            elif vce == "stage2":
                # Legacy: horizons treated as independent.
                ses_post = np.array(
                    [se_dict[supported_names[p]] for p in post_pos], dtype=float
                )
                att_se = float(np.sqrt(np.sum((shares * ses_post) ** 2)))
            else:
                att_se = float("nan")
        else:
            att_overall, att_se = float("nan"), float("nan")
    else:
        att_overall = float(coef_dict["ATT"])
        att_se = float(se_dict["ATT"])

    pvalue = (
        float(2 * sp_stats.norm.sf(abs(att_overall / att_se)))
        if att_se > 0
        else float("nan")
    )

    n_units = int(df[group].nunique())
    n_treated_units = int(df.loc[treated_now, group].nunique())

    es_vcov_df = None
    if event_study and V is not None:
        full = np.full((len(names), len(names)), np.nan)
        full[np.ix_(supported, supported)] = V
        es_vcov_df = pd.DataFrame(full, index=names, columns=names)
    se_convention = {
        "analytic": (
            "did2s corrected two-stage clustered variance (Gardner 2022): "
            "Stage-1 estimation error propagated, no small-sample factor; "
            "matches R did2s::did2s and Stata did2s"
        ),
        "stage2": (
            "legacy Stage-2-only cluster sandwich with G/(G-1)(n-1)/(n-k); "
            "ignores Stage-1 estimation error (understates)"
        ),
        "bootstrap": "pairs-cluster bootstrap of the full two-step procedure",
        "none": "no inference (internal fast path of the bootstrap)",
    }[vce]
    model_info = {
        "method": "Gardner 2022 two-stage DID",
        "vce": vce,
        "se_convention": se_convention,
        "weights": weights,
        "n_obs": n,
        "n_units": n_units,
        "n_treated_units": n_treated_units,
        "n_clusters": n_clusters,
        "alpha": alpha,
        "stage1_n": int(untreated_mask.sum()),
        "vcov": V,
        "cell_labels": supported_names if V is not None else None,
        "event_study": (
            {
                "horizon": names,
                "coef": coef_dict,
                "se": se_dict,
                "ci": ci,
                "n_obs": count_dict,
                "vcov": es_vcov_df,
            }
            if event_study
            else None
        ),
        "citation": (
            "Gardner, J. (2022). Two-stage differences in differences. "
            "arXiv:2207.05943. Butts & Gardner (2022), The R Journal."
        ),
    }

    _result = CausalResult(
        method="Gardner 2022 two-stage DID (did2s)",
        estimand="ATT",
        estimate=att_overall,
        se=att_se,
        pvalue=pvalue,
        ci=(att_overall - z * att_se, att_overall + z * att_se),
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.did.gardner_did",
            params={
                "y": y,
                "group": group,
                "time": time,
                "first_treat": first_treat,
                "controls": controls,
                "event_study": event_study,
                "horizon": horizon,
                "cluster": cluster,
                "alpha": alpha,
                "vce": vce,
                "weights": weights,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# Convenience alias aligned with the R package ``did2s``.
did_2stage = gardner_did
