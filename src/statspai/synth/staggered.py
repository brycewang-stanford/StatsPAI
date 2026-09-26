"""
Staggered Adoption Synthetic Control (partially pooled SCM).

Implements the partially pooled synthetic control estimator of
Ben-Michael, Feller & Rothstein (2022) for units that adopt treatment at
different times, following the conventions of the authors' R
implementation ``augsynth::multisynth`` (version 0.2.0):

* Outcomes are aligned in **event time**. For treated unit (or cohort)
  ``j`` adopting at period ``g_j``, the pre-treatment fit uses the last
  ``n_lags`` periods before ``g_j``; the effect is reported for event
  times ``0 .. n_leads - 1`` (truncated at the end of the panel).
* Donors for ``j`` are units whose adoption index is strictly greater
  than ``g_j + n_leads`` (never-treated units always qualify), so no donor
  is treated inside ``j``'s effect window.
* Weights solve, for ``nu`` in ``[0, 1]``::

      (1 - nu) / (2 J s_sep)  sum_j ||x_j - X_j' g_j||^2 / L_j
    + nu / (2 J^2 L s_pool)  || sum_j (x_j - X_j' g_j) ||^2
    + lambda / 2  sum_j ||g_j||^2,     g_j >= 0,  sum(g_j) = n_j

  where ``x_j`` stacks the (summed) pre-period outcomes of ``j``,
  ``L_j = min(g_j, n_lags)``, ``L = n_lags`` and ``s_sep`` / ``s_pool``
  are the squared individual / pooled imbalances of the ``nu = 0`` fit.
  ``nu = 0`` is a separate SCM per unit; ``nu = 1`` balances only the
  average; ``nu = 'auto'`` is multisynth's default heuristic (pooled
  imbalance / mean individual imbalance of the ``nu = 0`` fit).
* ``fixedeff=True`` de-means every unit by its average over ``j``'s
  pre-treatment periods before fitting (multisynth's ``fixedeff``).
* The overall ATT is the ``n_j``-weighted mean over treated units
  (``method='separate'``) or cohorts (``'pooled'``) of each one's average
  effect across its event-time window — multisynth's ``Average`` row.

References
----------
Ben-Michael, E., Feller, A. and Rothstein, J. (2022).
"Synthetic Controls with Staggered Adoption."
*Journal of the Royal Statistical Society: Series B*, 84(2), 351-381. [@benmichael2022synthetic]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

# Former per-unit solver, kept importable for existing callers; the
# estimator itself uses the multisynth QP below.
from ._core import placebo_rank_pvalue
from ._core import solve_simplex_weights as _solve_weights  # noqa: F401


@accepts_aliases(id="unit", treat="treatment", y="outcome")
def staggered_synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treatment: str,
    method: Literal["separate", "pooled"] = "separate",
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
    *,
    nu: Union[float, Literal["auto"]] = 0.0,
    fixedeff: bool = False,
    n_leads: Optional[int] = None,
    n_lags: Optional[int] = None,
    se_method: Literal["placebo", "jackknife"] = "placebo",
) -> CausalResult:
    """
    Staggered Adoption Synthetic Control (Ben-Michael, Feller & Rothstein).

    Parameters
    ----------
    data : pd.DataFrame
        Balanced long-format panel.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treatment : str
        Binary, absorbing treatment indicator (0/1). A unit's adoption
        period is the first period with ``treatment == 1``.
    method : {'separate', 'pooled'}, default 'separate'
        * ``'separate'`` — one set of weights per treated unit
          (multisynth ``time_cohort = FALSE``).
        * ``'pooled'`` — one set of weights per adoption cohort, fitted to
          the cohort's summed outcomes (multisynth ``time_cohort = TRUE``).
    penalization : float, default 0.0
        Ridge penalty ``lambda`` on the weights (multisynth ``lambda``), in
        the scale of the objective in the module docstring.
    placebo : bool, default True
        Compute inference (see ``se_method``). ``False`` returns a point
        estimate with NaN standard error / p-value.
    alpha : float, default 0.05
        Significance level.
    nu : float in [0, 1] or 'auto', default 0.0
        Pooling parameter. ``0`` fits each unit (cohort) separately;
        ``'auto'`` uses multisynth's default heuristic.
    fixedeff : bool, default False
        De-mean outcomes by unit before fitting (multisynth ``fixedeff``;
        note multisynth's own default is ``TRUE``).
    n_leads : int, optional
        Number of post-treatment event times in each effect window.
        ``None`` uses every period through the end of the panel (the
        longest window), which restricts donors to never-treated units.
    n_lags : int, optional
        Number of pre-treatment periods to balance (``None`` = all
        periods before the last adoption date, as in multisynth).
    se_method : {'placebo', 'jackknife'}, default 'placebo'
        * ``'placebo'`` — in-space placebos: every never-treated unit is
          assigned each cohort's adoption date and fitted from the other
          never-treated units; SE is the standard deviation of the
          placebo ATTs and the p-value is the placebo rank of ``|ATT|``.
        * ``'jackknife'`` — leave-one-unit-out jackknife over all units
          with ``nu`` held at its full-sample value, exactly as
          ``summary(multisynth, inf_type = "jackknife")``; normal-theory
          p-value and CI.

    Returns
    -------
    CausalResult
        ``model_info`` holds ``unit_effects`` (per treated unit or cohort),
        ``cohort_effects``, ``event_study`` (event-time ATT), the weight
        matrix, ``nu`` and the pooled / individual imbalance measures.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=30, n_periods=10, staggered=True, seed=0)
    >>> result = sp.staggered_synth(
    ...     df, outcome='y', unit='unit', time='time',
    ...     treatment='treated', placebo=False,
    ... )
    >>> bool(result.estimate is not None)
    True
    >>> _ = result.summary()

    References
    ----------
    benmichael2022synthetic
    """
    if method not in ("separate", "pooled"):
        raise MethodIncompatibility(
            f"method must be 'separate' or 'pooled', got {method!r}"
        )
    if se_method not in ("placebo", "jackknife"):
        raise MethodIncompatibility(
            f"se_method must be 'placebo' or 'jackknife', got {se_method!r}"
        )
    if not (nu == "auto" or (isinstance(nu, (int, float)) and 0.0 <= nu <= 1.0)):
        raise MethodIncompatibility(f"nu must be in [0, 1] or 'auto', got {nu!r}")
    if penalization < 0:
        raise MethodIncompatibility("penalization must be non-negative")

    panel = _prepare_panel(data, outcome, unit, time, treatment)
    Y, trt, units, times = panel.Y, panel.trt, panel.units, panel.times
    T = Y.shape[1]
    finite = np.isfinite(trt)
    d = int(np.max(trt[finite]))  # periods before the last adoption date
    horizon_max = int(T - np.min(trt[finite]))
    leads = horizon_max if n_leads is None else int(n_leads)
    if leads < 1:
        raise MethodIncompatibility("n_leads must be >= 1")
    leads = min(leads, horizon_max)
    lags = d if n_lags is None else int(n_lags)
    if lags < 1:
        raise MethodIncompatibility("n_lags must be >= 1")
    lags = min(lags, d)
    pooled = method == "pooled"

    fit = _fit_multisynth(Y, trt, d, leads, lags, nu, penalization, fixedeff, pooled)
    att = fit.att

    # --- Inference ---
    se = np.nan
    pvalue = np.nan
    placebo_atts: List[float] = []
    jack: List[float] = []
    z_crit = stats.norm.ppf(1 - alpha / 2)
    if placebo and se_method == "jackknife":
        n = Y.shape[0]
        for i in range(n):
            keep = np.arange(n) != i
            fit_i = _fit_multisynth(
                Y[keep],
                trt[keep],
                d,
                leads,
                lags,
                fit.nu,
                penalization,
                fixedeff,
                pooled,
            )
            jack.append(fit_i.att)
        jack_arr = np.asarray(jack)
        se = float(np.sqrt((n - 1) / n * np.sum((jack_arr - jack_arr.mean()) ** 2)))
        if se > 0:
            pvalue = float(2 * stats.norm.sf(abs(att) / se))
    elif placebo:
        placebo_atts = _placebo_atts(Y, trt, leads, lags, penalization, fixedeff, fit)
        if len(placebo_atts) > 1:
            se = float(np.std(placebo_atts, ddof=1))
            pvalue = placebo_rank_pvalue(abs(att), np.abs(placebo_atts))
    ci = (att - z_crit * se, att + z_crit * se)

    # --- Detail tables ---
    group_label = [
        units[fit.which[j][0]] if not pooled else times[int(fit.grps[j])]
        for j in range(len(fit.grps))
    ]
    unit_df = pd.DataFrame(
        {
            "unit": group_label,
            "cohort_time": [times[int(g)] for g in fit.grps],
            "att": fit.group_att,
            "n_pre": [int(g) for g in fit.grps],
            "n_post": fit.horizons,
            "n_units": fit.n1,
        }
    )
    cohort_rows = []
    for g in sorted(set(int(v) for v in fit.grps)):
        sel = [j for j in range(len(fit.grps)) if int(fit.grps[j]) == g]
        w = np.array([fit.n1[j] for j in sel], dtype=float)
        a = np.array([fit.group_att[j] for j in sel])
        cohort_rows.append(
            {
                "cohort_time": times[g],
                "att": float(np.sum(w * a) / np.sum(w)),
                "n_units": int(np.sum(w)),
            }
        )
    cohort_df = pd.DataFrame(cohort_rows)

    weight_cols = {}
    for j, lab in enumerate(group_label):
        weight_cols[lab] = fit.weights[:, j]
    weights_df = pd.DataFrame(weight_cols, index=pd.Index(units, name=unit))

    cohorts: Dict[Any, List[Any]] = {}
    for i in np.flatnonzero(finite):
        cohorts.setdefault(times[int(trt[i])], []).append(units[i])

    model_info: Dict[str, Any] = {
        "method": method,
        "nu": fit.nu,
        "fixedeff": fixedeff,
        "n_leads": leads,
        "n_lags": lags,
        "n_treated_units": int(finite.sum()),
        "n_control_units": int((~finite).sum()),
        "n_cohorts": len(cohorts),
        "cohort_times": sorted(cohorts.keys()),
        "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
        "unit_effects": unit_df,
        "cohort_effects": cohort_df,
        "event_study": fit.event_study,
        "weights": weights_df,
        "global_l2": fit.global_l2,
        "ind_l2": fit.ind_l2,
        "penalization": penalization,
        "se_method": se_method if placebo else None,
        "all_times": list(times),
    }
    if placebo_atts:
        model_info["placebo_atts"] = placebo_atts
        model_info["n_placebos"] = len(placebo_atts)
    if jack:
        model_info["jackknife_atts"] = jack

    return CausalResult(
        method="Staggered Synthetic Control (Ben-Michael et al. 2022)",
        estimand="ATT",
        estimate=att,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=len(data),
        detail=unit_df,
        model_info=model_info,
        _citation_key="staggered_synth",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass
class _Panel:
    Y: np.ndarray  # (n, T) outcomes, units sorted
    trt: np.ndarray  # (n,) adoption index (np.inf = never treated)
    units: List[Any]
    times: List[Any]


@dataclass
class _Fit:
    att: float
    group_att: List[float]
    horizons: List[int]
    grps: np.ndarray
    which: List[np.ndarray]
    n1: List[int]
    weights: np.ndarray  # (n, J)
    nu: float
    global_l2: float
    ind_l2: float
    event_study: pd.DataFrame


def _prepare_panel(
    data: pd.DataFrame, outcome: str, unit: str, time: str, treatment: str
) -> _Panel:
    for col in (outcome, unit, time, treatment):
        if col not in data.columns:
            raise MethodIncompatibility(f"column {col!r} not found in data")
    if data.duplicated([unit, time]).any():
        raise MethodIncompatibility("staggered_synth needs one row per (unit, time)")
    y_wide = data.pivot(index=unit, columns=time, values=outcome).sort_index()
    d_wide = data.pivot(index=unit, columns=time, values=treatment).sort_index()
    d_wide = d_wide.reindex(columns=y_wide.columns)
    if y_wide.isna().any().any() or d_wide.isna().any().any():
        raise MethodIncompatibility(
            "staggered_synth needs a balanced panel without NaN",
            recovery_hint="Balance the panel and drop missing outcomes.",
        )
    D = d_wide.to_numpy(dtype=np.float64)
    if not np.isin(D, (0.0, 1.0)).all():
        raise MethodIncompatibility("treatment must be binary 0/1")
    n, T = D.shape
    trt = np.full(n, np.inf)
    for i in range(n):
        on = np.flatnonzero(D[i] == 1)
        if on.size:
            g = int(on[0])
            if not np.all(D[i, g:] == 1):
                raise MethodIncompatibility(
                    f"treatment of unit {y_wide.index[i]!r} switches off after "
                    "adoption; staggered_synth needs absorbing treatment"
                )
            trt[i] = g
    if not np.isfinite(trt).any():
        raise ValueError("No treated units found")
    if np.isfinite(trt).all():
        raise ValueError("No never-treated (pure control) units found")
    if np.any(trt == 0):
        raise DataInsufficient(
            "Some units are treated in the first period (no pre-treatment "
            "outcome); remove them",
            recovery_hint="Drop units treated in the first period.",
        )
    return _Panel(
        Y=y_wide.to_numpy(dtype=np.float64),
        trt=trt,
        units=list(y_wide.index),
        times=list(y_wide.columns),
    )


def _groups(trt: np.ndarray, pooled: bool) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Treated units (separate) or cohorts (pooled), in multisynth order."""
    fin = np.flatnonzero(np.isfinite(trt))
    if pooled:
        grps: List[float] = []
        for i in fin:
            if trt[i] not in grps:
                grps.append(trt[i])
        which = [np.flatnonzero(trt == g) for g in grps]
    else:
        grps = [trt[i] for i in fin]
        which = [np.array([i]) for i in fin]
    return np.asarray(grps, dtype=float), which


def _fit_multisynth(
    Y: np.ndarray,
    trt: np.ndarray,
    d: int,
    n_leads: int,
    n_lags: int,
    nu: Union[float, str],
    lam: float,
    fixedeff: bool,
    pooled: bool,
) -> _Fit:
    n, T = Y.shape
    grps, which = _groups(trt, pooled)
    J = len(grps)
    n1 = [len(w) for w in which]
    max_dim = min(d, n_lags)

    xs: List[np.ndarray] = []  # padded target sums (max_dim,)
    Ms: List[np.ndarray] = []  # padded donor matrices (max_dim, n0_j)
    donors: List[np.ndarray] = []
    ndims: List[int] = []
    for j in range(J):
        g = int(grps[j])
        B = Y[:, :d].copy()
        if fixedeff:
            B = B - Y[:, :g].mean(axis=1, keepdims=True)
        ndim = min(g, n_lags)
        don = np.flatnonzero(trt > n_leads + g)
        if don.size == 0:
            raise DataInsufficient(
                f"treated unit/cohort adopting at index {g} has no eligible "
                "donor (units untreated through its effect window); reduce "
                "n_leads",
                recovery_hint="Reduce n_leads.",
            )
        x = B[which[j], g - ndim : g].sum(axis=0)
        M = B[np.ix_(don, np.arange(g - ndim, g))].T
        xs.append(np.concatenate([np.zeros(max_dim - ndim), x]))
        Ms.append(np.vstack([np.zeros((max_dim - ndim, don.size)), M]))
        donors.append(don)
        ndims.append(ndim)

    sep = _solve_qp(xs, Ms, n1, ndims, 0.0, lam, n_lags, 1.0, 1.0)
    imb = _imbalance(xs, Ms, sep, d, max_dim)
    global_l2, avg_l2, ind_l2 = _imbalance_norms(imb, d, n_lags)
    if nu == "auto":
        glbl = global_l2 * np.sqrt(d)
        nu_val = float(glbl / avg_l2) if avg_l2 > 0 else 0.0
    else:
        nu_val = float(nu)
    gam = _solve_qp(xs, Ms, n1, ndims, nu_val, lam, n_lags, global_l2**2, ind_l2**2)
    imb_f = _imbalance(xs, Ms, gam, d, max_dim)
    g_l2, _, i_l2 = _imbalance_norms(imb_f, d, n_lags)

    W = np.zeros((n, J))
    group_att: List[float] = []
    horizons: List[int] = []
    es_num: Dict[int, float] = {}
    es_den: Dict[int, float] = {}
    for j in range(J):
        g = int(grps[j])
        w = np.clip(gam[j] / n1[j], 0.0, None)
        w = w / w.sum()
        W[donors[j], j] = w
        tr = Y[which[j]].mean(axis=0)
        Yd = Y[donors[j]]
        if fixedeff:
            mu0 = tr[:g].mean() - w @ Yd[:, :g].mean(axis=1) + w @ Yd
        else:
            mu0 = w @ Yd
        tau = tr - mu0
        h = min(n_leads, T - g)
        horizons.append(h)
        group_att.append(float(np.mean(tau[g : g + h])))
        for e in range(-g, h):
            es_num[e] = es_num.get(e, 0.0) + n1[j] * tau[g + e]
            es_den[e] = es_den.get(e, 0.0) + n1[j]
    n1_arr = np.asarray(n1, dtype=float)
    att = float(np.sum(n1_arr * np.asarray(group_att)) / n1_arr.sum())
    ev = sorted(es_num)
    event_study = pd.DataFrame(
        {
            "event_time": ev,
            "att": [es_num[e] / es_den[e] for e in ev],
            "n_treated": [int(es_den[e]) for e in ev],
        }
    )
    return _Fit(
        att=att,
        group_att=group_att,
        horizons=horizons,
        grps=grps,
        which=which,
        n1=n1,
        weights=W,
        nu=nu_val,
        global_l2=float(g_l2),
        ind_l2=float(i_l2),
        event_study=event_study,
    )


def _solve_qp(
    xs: List[np.ndarray],
    Ms: List[np.ndarray],
    n1: List[int],
    ndims: List[int],
    nu: float,
    lam: float,
    n_lags: int,
    norm_pool: float,
    norm_sep: float,
) -> List[np.ndarray]:
    """Minimise the multisynth objective over the stacked weights.

    The objective is the convex quadratic ``0.5 z'Hz + c'z`` subject to
    ``z >= 0`` and one adding-up constraint per treated unit / cohort.
    SLSQP locates the optimal face; an active-set polish then solves the
    KKT system on that face exactly and checks dual feasibility, so the
    returned weights are accurate to machine precision rather than to the
    first-order solver's tolerance.
    """
    J = len(xs)
    sizes = [M.shape[1] for M in Ms]
    cuts = np.cumsum([0] + sizes)
    N = int(cuts[-1])
    c_sep = (1.0 - nu) / (norm_sep * J)
    c_pool = nu / (norm_pool * J**2 * n_lags)

    Mbig = np.zeros((xs[0].shape[0] * J, N))
    for j in range(J):
        Mbig[j * xs[0].shape[0] : (j + 1) * xs[0].shape[0], cuts[j] : cuts[j + 1]] = Ms[
            j
        ]
    H = np.zeros((N, N))
    c = np.zeros(N)
    x_sum = np.sum(xs, axis=0)
    for j in range(J):
        sl = slice(cuts[j], cuts[j + 1])
        H[sl, sl] += c_sep * (Ms[j].T @ Ms[j]) / ndims[j]
        c[sl] -= c_sep * (Ms[j].T @ xs[j]) / ndims[j]
        if c_pool > 0:
            c[sl] -= c_pool * (Ms[j].T @ x_sum)
            for k in range(J):
                sk = slice(cuts[k], cuts[k + 1])
                H[sl, sk] += c_pool * (Ms[j].T @ Ms[k])
    H += lam * np.eye(N)
    A = np.zeros((J, N))
    for j in range(J):
        A[j, cuts[j] : cuts[j + 1]] = 1.0
    b = np.asarray(n1, dtype=float)

    z = _active_set_qp(H, c, A, b, cuts)
    return [z[cuts[j] : cuts[j + 1]] for j in range(J)]


def _active_set_qp(
    H: np.ndarray,
    c: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    cuts: np.ndarray,
) -> np.ndarray:
    """Primal active-set solver for ``min .5 z'Hz + c'z, Az = b, z >= 0``.

    ``A`` has one row per block (``cuts`` delimits the blocks) with ones on
    the block. Lawson-Hanson style: keep a feasible iterate, solve the
    equality-constrained problem on the free set exactly (KKT system), step
    back to the boundary when that solution leaves the orthant, and free the
    coordinate with the most negative reduced cost once it does not. The
    result is exact up to the linear-algebra round-off.
    """
    N = H.shape[0]
    J = A.shape[0]
    z = np.zeros(N)
    free = np.zeros(N, dtype=bool)
    for j in range(J):
        sl = np.arange(cuts[j], cuts[j + 1])
        # start each block at its best single donor (vertex of the simplex)
        vals = 0.5 * b[j] ** 2 * np.diag(H)[sl] + b[j] * c[sl]
        k = sl[int(np.argmin(vals))]
        z[k] = b[j]
        free[k] = True
    scale_c = max(1.0, float(np.abs(c).max()), float(np.abs(H).max()))
    for _ in range(20 * N + 100):
        F = np.flatnonzero(free)
        nF = F.size
        K = np.zeros((nF + J, nF + J))
        K[:nF, :nF] = H[np.ix_(F, F)]
        K[:nF, nF:] = A[:, F].T
        K[nF:, :nF] = A[:, F]
        sol = np.linalg.lstsq(K, np.concatenate([-c[F], b]), rcond=None)[0]
        zF, mu = sol[:nF], sol[nF:]
        if np.any(zF < 0.0):
            cur = z[F]
            neg = np.flatnonzero(zF < 0.0)
            ratios = cur[neg] / (cur[neg] - zF[neg])
            step = float(np.min(ratios))
            z[F] = cur + step * (zF - cur)
            blocking = F[neg[int(np.argmin(ratios))]]
            z[blocking] = 0.0
            hit = F[z[F] <= 1e-14 * float(b.max())]
            z[hit] = 0.0
            free[hit] = False
            free[blocking] = False
            # keep every block non-empty
            for j in range(J):
                blk = slice(cuts[j], cuts[j + 1])
                if not free[blk].any():
                    k = cuts[j] + int(np.argmax(z[blk]))
                    free[k] = True
            continue
        z = np.zeros(N)
        z[F] = zF
        s_red = H @ z + c + A.T @ mu
        s_red[F] = 0.0
        k = int(np.argmin(s_red))
        if s_red[k] >= -1e-13 * scale_c:
            return z
        free[k] = True
    raise ConvergenceFailure("multisynth QP active-set solver did not converge")


def _imbalance(
    xs: List[np.ndarray],
    Ms: List[np.ndarray],
    gam: List[np.ndarray],
    d: int,
    max_dim: int,
) -> np.ndarray:
    """(d, J) pre-period imbalance, zero-padded at the top like multisynth."""
    cols = []
    for j in range(len(xs)):
        r = xs[j] - Ms[j] @ gam[j]
        cols.append(np.concatenate([np.zeros(d - max_dim), r]))
    return np.column_stack(cols)


def _imbalance_norms(
    imb: np.ndarray, d: int, n_lags: int
) -> Tuple[float, float, float]:
    """multisynth's global_l2, avg_l2 and ind_l2 (V = identity)."""
    last = imb[d - n_lags :, :]
    avg = last.mean(axis=1)
    global_l2 = float(np.sqrt(avg @ avg) / np.sqrt(d))
    col_ss = np.sum(last**2, axis=0)
    avg_l2 = float(np.mean(np.sqrt(col_ss)))
    nnz = np.sum(last != 0, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ind_l2 = float(np.sqrt(np.mean(col_ss / nnz)))
    return global_l2, avg_l2, ind_l2


def _placebo_atts(
    Y: np.ndarray,
    trt: np.ndarray,
    n_leads: int,
    n_lags: int,
    lam: float,
    fixedeff: bool,
    fit: _Fit,
) -> List[float]:
    """In-space placebos: each never-treated unit gets every cohort's date."""
    never = np.flatnonzero(~np.isfinite(trt))
    if never.size < 3:
        return []
    T = Y.shape[1]
    cohort_w: Dict[int, float] = {}
    for j, g in enumerate(fit.grps):
        cohort_w[int(g)] = cohort_w.get(int(g), 0.0) + fit.n1[j]
    out: List[float] = []
    Yn = Y[never]
    for p in range(never.size):
        num = 0.0
        den = 0.0
        for g, wg in sorted(cohort_w.items()):
            trt_p = np.full(never.size, np.inf)
            trt_p[p] = g
            lead_p = min(n_leads, T - g)
            fit_p = _fit_multisynth(
                Yn,
                trt_p,
                g,
                lead_p,
                min(n_lags, g),
                0.0,
                lam,
                fixedeff,
                False,
            )
            num += wg * fit_p.att
            den += wg
        out.append(num / den)
    return out


# Citation
CausalResult._CITATIONS["staggered_synth"] = (
    "@article{benmichael2022synthetic,\n"
    "  title={Synthetic Controls with Staggered Adoption},\n"
    "  author={Ben-Michael, Eli and Feller, Avi and Rothstein, Jesse},\n"
    "  journal={Journal of the Royal Statistical Society: Series B},\n"
    "  volume={84},\n"
    "  number={2},\n"
    "  pages={351--381},\n"
    "  year={2022},\n"
    "  publisher={Wiley}\n"
    "}"
)
