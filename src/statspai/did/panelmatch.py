"""Matching on treatment history for panel data (R ``PanelMatch``).

``sp.panel_match`` implements the matching estimator of Imai, Kim and
Wang (2023) [@imai2023matching] for a binary treatment that can switch on
and off: each treated observation (a switch into treatment) is matched to
the units with the same treatment history over the previous ``lag``
periods that stay untreated, the match is refined on covariates, and the
effect at each lead is a weighted difference-in-differences averaged over
the matched sets.

The numbers follow R ``PanelMatch`` 3.1.5 (matched sets, refinement
weights, point estimates, the conditional and unconditional standard
errors, covariate balance and the placebo test agree to 1e-10 on the
reference fixtures); see ``docs/dev/modules/did.md`` for the places where
we depart from it on purpose.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..core._call_record import records_call
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._panel_match_estimate import (
    analytic_se,
    bootstrap_draws,
    covariate_balance,
    unit_sums,
)
from ._panel_match_refine import covariate_tensor, parse_covariates, refine
from ._panel_match_sets import Grid, MatchedSet, build_matched_sets

__all__ = ["panel_match"]

CausalResult._CITATIONS["panel_match"] = (
    "@article{imai2023matching,\n"
    "  title={Matching Methods for Causal Inference with Time-Series "
    "Cross-Sectional Data},\n"
    "  author={Imai, Kosuke and Kim, In Song and Wang, Erik H.},\n"
    "  journal={American Journal of Political Science},\n"
    "  volume={67},\n"
    "  number={3},\n"
    "  pages={587--605},\n"
    "  year={2023},\n"
    "  doi={10.1111/ajps.12685}\n"
    "}"
)

_REFINEMENTS = (
    "mahalanobis",
    "ps_weight",
    "ps_match",
    "cbps_weight",
    "cbps_match",
    "none",
)
_QOIS = ("att", "atc", "ate", "art")
_SE_METHODS = ("bootstrap", "conditional", "unconditional")


def _grid(
    data: pd.DataFrame, y: str, treat: str, id: str, time: str, extra: Sequence[str]
) -> Grid:
    """Balanced (unit x period) arrays; absent unit-periods are missing."""
    t_raw = pd.to_numeric(data[time], errors="coerce")
    if t_raw.isna().any():
        raise MethodIncompatibility(
            f"panel_match: the time column {time!r} must be numeric periods.",
            recovery_hint="Recode dates to consecutive integers (e.g. years).",
        )
    periods = np.sort(t_raw.unique())
    steps = np.diff(periods)
    consecutive = bool(np.all(periods == np.round(periods))) and bool(
        np.all(steps == 1)
    )
    if not consecutive:
        warnings.warn(
            f"panel_match: {time!r} is not a run of consecutive integers; "
            "its observed values are treated as consecutive periods "
            "(R PanelMatch does the same).",
            UserWarning,
            stacklevel=3,
        )
    t_pos = np.searchsorted(periods, t_raw.to_numpy())
    unit_ids = np.sort(data[id].unique())
    u_pos = np.searchsorted(unit_ids, data[id].to_numpy())
    N, P = len(unit_ids), len(periods)

    def wide(col: str) -> np.ndarray:
        out = np.full((N, P), np.nan)
        out[u_pos, t_pos] = pd.to_numeric(data[col], errors="coerce").to_numpy(float)
        return out

    columns = {c: wide(c) for c in dict.fromkeys([y, *extra])}
    return Grid(unit_ids, periods, wide(treat), wide(y), columns)


def _run(
    grid: Grid,
    D: np.ndarray,
    *,
    qoi: str,
    lag: int,
    leads: List[int],
    refinement: str,
    C: Optional[np.ndarray],
    size_match: int,
    match_missing: bool,
    forbid_treatment_reversal: bool,
    restrict_control_period: Optional[int],
    exact_columns: Dict[str, np.ndarray],
    diagonal: bool,
    placebo: bool,
):
    g = Grid(grid.unit_ids, grid.periods, D, grid.outcome, grid.columns)
    sets, empty = build_matched_sets(
        g,
        lag=lag,
        max_lead=max(leads),
        qoi=qoi,
        match_missing=match_missing,
        forbid_treatment_reversal=forbid_treatment_reversal,
        restrict_control_period=restrict_control_period,
        exact_columns=exact_columns,
        placebo=placebo,
    )
    if not sets:
        raise DataInsufficient(
            f"panel_match: no treated observation for qoi={qoi!r} has a "
            "control with the same treatment history.",
            recovery_hint="Lower lag= (shorter histories match more often).",
            diagnostics={"treated_without_match": len(empty)},
        )
    refine(
        sets,
        g,
        method=refinement,
        C=C,
        lag=lag,
        size_match=size_match,
        diagonal=diagonal,
        D=D,
    )
    return sets, empty


def _sets_frame(
    sets: List[MatchedSet], empty: List[MatchedSet], grid: Grid
) -> pd.DataFrame:
    rows = []
    for s in list(sets) + list(empty):
        w = s.weights if s.weights is not None else np.zeros(0)
        for c, wc in zip(s.controls, w):
            rows.append(
                (
                    grid.unit_ids[s.unit],
                    grid.periods[s.period],
                    grid.unit_ids[c],
                    float(wc),
                )
            )
        if s.empty:
            rows.append((grid.unit_ids[s.unit], grid.periods[s.period], None, np.nan))
    return pd.DataFrame(
        rows, columns=["treated_unit", "time", "control_unit", "weight"]
    )


@records_call("panel_match")
def panel_match(
    data: pd.DataFrame,
    y: str,
    treat: str,
    id: str,
    time: str,
    *,
    lag: int = 4,
    lead: Union[int, Sequence[int]] = 0,
    qoi: str = "att",
    refinement: str = "mahalanobis",
    covariates: Optional[Sequence[str]] = None,
    size_match: int = 10,
    match_missing: bool = True,
    exact_match: Optional[Sequence[str]] = None,
    forbid_treatment_reversal: bool = False,
    restrict_control_period: Optional[int] = None,
    diagonal_covariance: bool = False,
    se_method: str = "bootstrap",
    n_boot: int = 1000,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    placebo_test: bool = False,
    balance_covariates: Optional[Sequence[str]] = None,
) -> CausalResult:
    """Matching estimator for panel data with a treatment that switches on
    and off (Imai, Kim and Wang 2023; R ``PanelMatch``).

    Parameters
    ----------
    data : pandas.DataFrame
        Long panel, one row per unit and period.
    y : str
        Outcome.
    treat : str
        0/1 treatment status in each period; may switch on and off.
    id : str
        Unit identifier.
    time : str
        Integer period. Missing unit-periods are treated as rows of
        missing values (the panel is balanced internally).
    lag : int, default 4
        Length ``L`` of the treatment history matched exactly
        (periods ``t-L .. t-1``).
    lead : int or sequence of int, default 0
        Leads ``F`` at which the effect is estimated (``0`` is the
        treatment period). ``lead=range(4)`` gives four estimates.
    qoi : {'att', 'atc', 'ate', 'art'}, default 'att'
        ``'att'`` effect of switching into treatment on the switchers;
        ``'atc'`` effect of treatment on observations that stay out;
        ``'ate'`` the two combined by their numbers of matched sets;
        ``'art'`` effect of switching out of treatment on those who do.
    refinement : str, default 'mahalanobis'
        ``'mahalanobis'``, ``'ps_weight'``, ``'ps_match'``,
        ``'cbps_weight'``, ``'cbps_match'`` or ``'none'`` (equal weights
        within each matched set).
    covariates : list of str, optional
        Refinement covariates: column names, or ``'lag(x, k)'`` /
        ``'lag(x, a:b)'`` for within-unit lags (R
        ``covs.formula = ~ I(lag(x, 1:4))``). Required unless
        ``refinement='none'``.
    size_match : int, default 10
        Number of controls kept by the ``*_match`` and Mahalanobis
        refinements.
    match_missing : bool, default True
        Treat a missing treatment value as its own history state. With
        ``False`` a history with a missing value matches nothing.
    exact_match : list of str, optional
        Variables a control must equal the treated unit on in every period
        ``t-L .. t``.
    forbid_treatment_reversal : bool, default False
        Keep only treated units that stay treated, and controls that stay
        untreated, through the last lead (ATT / ART only).
    restrict_control_period : int, optional
        Keep only treated observations untreated in each of the
        ``restrict_control_period`` periods before ``t`` (ATT / ART only).
    diagonal_covariance : bool, default False
        Mahalanobis distance with the diagonal of the covariance only.
    se_method : {'bootstrap', 'conditional', 'unconditional'}
        Unit bootstrap (``n_boot`` draws, percentile interval) or the
        analytic variances of Imai, Kim and Wang (normal interval). The
        ATE has only the bootstrap. As in R PanelMatch, all three grow
        with period shocks common to every unit although the estimate does
        not change (they are reproduced, not altered; see
        ``docs/dev/modules/did.md``).
    n_boot : int, default 1000
    seed : int, optional
        Seed of the bootstrap.
    alpha : float, default 0.05
    placebo_test : bool, default False
        Also estimate pre-treatment "effects" ``t-L .. t-2`` (outcome change
        from ``t-k`` to ``t-1``) on sets whose units observe the outcome
        over the whole history window; stored in
        ``model_info['placebo_test']``.
    balance_covariates : list of str, optional
        Columns for the balance table; defaults to the base variables of
        ``covariates`` and the outcome.

    Returns
    -------
    CausalResult
        ``estimate`` is the effect at the single lead, or the average over
        the leads when several are given (R ``pooled = TRUE``). ``detail``
        has one row per lead (``estimate``, ``se``, ``ci_lower``,
        ``ci_upper``). ``model_info`` holds ``matched_sets`` (treated unit,
        period, control, weight; an empty match has no control),
        ``n_matched_sets``, ``n_unmatched``, ``balance`` /
        ``balance_unrefined`` (standardised differences at ``t-L .. t``),
        ``bootstrap_draws`` and ``placebo_test`` when requested.

    Notes
    -----
    The standard errors resample (or treat as independent) units, so they
    allow arbitrary dependence within a unit over time. The analytic
    variances for several leads pooled are this function's extension: R
    ``PanelMatch`` gives the pooled average only with the bootstrap. The
    estimator applies the same formula to the averaged weights.

    CBPS refinements use ``sp.cbps``'s over-identified ATT fit, which
    reaches a lower GMM objective than R ``CBPS`` 0.24 (see
    ``sp.cbps``); refinement weights therefore differ slightly from R
    ``PanelMatch``'s ``CBPS.weight`` / ``CBPS.match``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(40):
    ...     d = 0
    ...     for t in range(10):
    ...         d = int(rng.random() < (0.7 if d else 0.2))
    ...         rows.append((i, t, d, 1.0 * d + 0.1 * t + rng.normal()))
    >>> df = pd.DataFrame(rows, columns=["id", "t", "d", "y"])
    >>> res = sp.panel_match(df, "y", "d", id="id", time="t", lag=2,
    ...                      refinement="none", lead=[0, 1],
    ...                      se_method="conditional")
    >>> res.detail[["lead", "estimate", "se"]].shape
    (2, 3)

    References
    ----------
    imai2023matching
    """
    leads = [int(lead)] if np.isscalar(lead) else [int(v) for v in lead]
    qoi = str(qoi).lower()
    refinement = str(refinement).lower().replace(".", "_")
    if not leads or min(leads) < 0:
        raise MethodIncompatibility(
            "panel_match: leads must be non-negative integers.",
            recovery_hint="Use placebo_test=True for pre-treatment periods.",
        )
    if qoi not in _QOIS:
        raise MethodIncompatibility(
            f"panel_match: qoi must be one of {_QOIS}, got {qoi!r}.",
            recovery_hint="qoi='att' for the effect on units switching into treatment.",
        )
    if refinement not in _REFINEMENTS:
        raise MethodIncompatibility(
            f"panel_match: refinement must be one of {_REFINEMENTS}, "
            f"got {refinement!r}.",
            recovery_hint=(
                "The marginal-structural-model refinements of R PanelMatch "
                "(ps.msm.weight, CBPS.msm.weight) are not implemented."
            ),
        )
    if se_method not in _SE_METHODS:
        raise MethodIncompatibility(
            f"panel_match: se_method must be one of {_SE_METHODS}.",
            recovery_hint="se_method='conditional' for a deterministic analytic SE.",
        )
    if qoi == "ate" and se_method != "bootstrap":
        raise MethodIncompatibility(
            "panel_match: the ATE has no analytic standard error.",
            recovery_hint="Use se_method='bootstrap' for qoi='ate'.",
        )
    if int(lag) < 1:
        raise MethodIncompatibility("panel_match: lag must be at least 1.")
    lag = int(lag)
    if (forbid_treatment_reversal or restrict_control_period is not None) and qoi in (
        "atc",
        "ate",
    ):
        raise MethodIncompatibility(
            "panel_match: forbid_treatment_reversal and restrict_control_period "
            "apply to qoi='att' and 'art' only.",
            recovery_hint="Drop the option or use qoi='att'.",
        )
    if (
        restrict_control_period is not None
        and not 1 <= int(restrict_control_period) <= lag
    ):
        raise MethodIncompatibility(
            "panel_match: restrict_control_period must lie in 1..lag."
        )
    if placebo_test and lag < 2:
        raise MethodIncompatibility("panel_match: the placebo test needs lag >= 2.")
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility("panel_match: alpha must lie in (0, 1).")
    if int(size_match) < 1:
        raise MethodIncompatibility("panel_match: size_match must be positive.")

    terms = parse_covariates(covariates or [])
    if refinement != "none" and not terms:
        raise MethodIncompatibility(
            f"panel_match: refinement={refinement!r} needs covariates=.",
            recovery_hint="Pass covariates=['x1', 'lag(y, 1:4)'] or refinement='none'.",
        )
    base_vars = list(dict.fromkeys(name for name, _ in terms))
    exact = list(exact_match or [])
    bal_vars = (
        list(balance_covariates) if balance_covariates is not None else base_vars + [y]
    )
    bal_vars = list(dict.fromkeys(bal_vars))
    needed = list(dict.fromkeys([y, treat, id, time, *base_vars, *exact, *bal_vars]))
    check_columns(
        data,
        {"columns": needed},
        message=f"panel_match: columns not in data: "
        f"{[c for c in needed if c not in data.columns]}.",
    )
    if data[id].isna().any():
        raise MethodIncompatibility(f"panel_match: {id!r} has missing values.")
    if data.duplicated([id, time]).any():
        raise MethodIncompatibility(
            f"panel_match: repeated ({id!r}, {time!r}) rows; one row per "
            "unit and period.",
            recovery_hint="Aggregate or drop duplicates first.",
        )
    d_obs = pd.to_numeric(data[treat], errors="coerce").dropna()
    if not d_obs.isin([0, 1]).all():
        raise MethodIncompatibility(
            f"panel_match: {treat!r} must be 0/1 (missing allowed).",
            recovery_hint=(
                "For a continuous or multi-valued treatment use "
                "sp.did_multiplegt_dyn."
            ),
        )

    exact_cols = {}
    for v in exact:
        codes = pd.Series(pd.factorize(data[v])[0], index=data.index).astype(float)
        exact_cols[v] = codes.where(data[v].notna())
    frame = data.copy()
    for v, codes in exact_cols.items():
        frame[f"__exact_{v}"] = codes
    grid = _grid(
        frame,
        y,
        treat,
        id,
        time,
        [*base_vars, *bal_vars, *[f"__exact_{v}" for v in exact]],
    )
    exact_grid = {v: grid.columns[f"__exact_{v}"] for v in exact}
    C = covariate_tensor(grid, terms) if refinement != "none" else None

    D_flip = 1.0 - grid.treat if qoi == "art" else grid.treat
    common = dict(
        lag=lag,
        leads=leads,
        refinement=refinement,
        C=C,
        size_match=int(size_match),
        match_missing=bool(match_missing),
        forbid_treatment_reversal=bool(forbid_treatment_reversal),
        restrict_control_period=restrict_control_period,
        exact_columns=exact_grid,
        diagonal=bool(diagonal_covariance),
    )
    parts = []  # (qoi_label, sets, empty, sign)
    if qoi == "ate":
        for q, sign in (("att", 1.0), ("atc", -1.0)):
            s, e = _run(grid, grid.treat, qoi=q, placebo=False, **common)
            parts.append((q, s, e, sign))
    else:
        s, e = _run(grid, D_flip, qoi=qoi, placebo=False, **common)
        parts.append((qoi, s, e, -1.0 if qoi == "atc" else 1.0))

    sums = []
    for q, s, e, sign in parts:
        A, B, nz = unit_sums(s, grid.outcome, -1, leads)
        sums.append((A, B, nz, sign))
    num = sum(sign * A.sum(axis=0) for A, B, nz, sign in sums)
    den = sum(B.sum() for A, B, nz, sign in sums)
    est_leads = num / den
    pooled = len(leads) > 1
    headline = float(np.mean(est_leads)) if pooled else float(est_leads[0])

    n_units = len(grid.unit_ids)
    model_info: Dict[str, Any] = {}
    if se_method == "bootstrap":
        rng = np.random.default_rng(seed)
        boot_parts = [(A, B, sign) for A, B, nz, sign in sums]
        treated_units = [np.unique([m.unit for m in s]) for q, s, e, sign in parts]
        draws = bootstrap_draws(boot_parts, treated_units, n_units, int(n_boot), rng)
        se_leads = np.nanstd(draws, axis=0, ddof=1)
        lo_q, hi_q = alpha / 2, 1 - alpha / 2
        ci_leads = np.nanquantile(draws, [lo_q, hi_q], axis=0).T
        pooled_draws = draws.mean(axis=1)
        se_head = (
            float(np.nanstd(pooled_draws, ddof=1)) if pooled else float(se_leads[0])
        )
        ci_head = (
            tuple(float(v) for v in np.nanquantile(pooled_draws, [lo_q, hi_q]))
            if pooled
            else (float(ci_leads[0, 0]), float(ci_leads[0, 1]))
        )
        model_info["bootstrap_draws"] = pd.DataFrame(
            draws, columns=[f"t+{f}" for f in leads]
        )
        p_head = (
            float(2 * stats.norm.sf(abs(headline / se_head))) if se_head > 0 else np.nan
        )
    else:
        A, B, nz, sign = sums[0]
        se_leads = analytic_se(A, B, nz, se_method)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_leads = np.column_stack([est_leads - z * se_leads, est_leads + z * se_leads])
        if pooled:
            # the pooled estimate is the same ratio with the lead-averaged
            # weights; a unit carries weight if it does at any lead
            Abar = A.mean(axis=1, keepdims=True)
            se_head = float(
                analytic_se(Abar, B, nz.any(axis=1, keepdims=True), se_method)[0]
            )
        else:
            se_head = float(se_leads[0])
        ci_head = (headline - z * se_head, headline + z * se_head)
        p_head = (
            float(2 * stats.norm.sf(abs(headline / se_head))) if se_head > 0 else np.nan
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        p_leads = 2 * stats.norm.sf(np.abs(est_leads / se_leads))
    detail = pd.DataFrame(
        {
            "lead": leads,
            "term": [f"t+{f}" for f in leads],
            "estimate": est_leads,
            "se": se_leads,
            "pvalue": p_leads,
            "ci_lower": ci_leads[:, 0],
            "ci_upper": ci_leads[:, 1],
        }
    )

    set_frames = []
    for q, s, e, sign in parts:
        f = _sets_frame(s, e, grid)
        f.insert(0, "qoi", q)
        set_frames.append(f)
    first_sets = parts[0][1]
    model_info.update(
        {
            "qoi": qoi,
            "lag": lag,
            "leads": leads,
            "refinement": refinement,
            "covariates": list(covariates or []),
            "se_method": se_method,
            "pooled": pooled,
            "matched_sets": pd.concat(set_frames, ignore_index=True),
            "n_matched_sets": int(sum(len(s) for q, s, e, sign in parts)),
            "n_unmatched": int(sum(len(e) for q, s, e, sign in parts)),
            "n_units": n_units,
            "n_periods": len(grid.periods),
            "mean_controls_per_set": float(
                np.mean([m.controls.size for m in first_sets])
            ),
        }
    )
    if qoi in ("att", "atc") and bal_vars:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model_info["balance"] = covariate_balance(
                first_sets, grid, bal_vars, lag, True
            )
            model_info["balance_unrefined"] = covariate_balance(
                first_sets, grid, bal_vars, lag, False
            )

    if placebo_test:
        if qoi == "ate":
            raise MethodIncompatibility(
                "panel_match: the placebo test is implemented for att, atc and art.",
                recovery_hint="Run qoi='att' and qoi='atc' separately.",
            )
        ps_sets, _ = _run(grid, D_flip, qoi=qoi, placebo=True, **common)
        # outcome change from t-k to t-1, k = lag .. 2
        cols_A, cols_nz = [], []
        for k in range(lag, 1, -1):
            Ak, B, nzk = unit_sums(ps_sets, grid.outcome, -k, [-1])
            cols_A.append(Ak[:, 0])
            cols_nz.append(nzk[:, 0])
        A = np.column_stack(cols_A)
        nz = np.column_stack(cols_nz)
        sign = -1.0 if qoi == "atc" else 1.0
        pl_est = sign * A.sum(axis=0) / B.sum()
        pl_se = analytic_se(
            A, B, nz, "unconditional" if se_method == "unconditional" else "conditional"
        )
        model_info["placebo_test"] = pd.DataFrame(
            {
                "term": [f"t-{k}" for k in range(lag, 1, -1)],
                "estimate": pl_est,
                "se": pl_se,
            }
        )
        model_info["placebo_test"].attrs["se_method"] = (
            "unconditional" if se_method == "unconditional" else "conditional"
        )

    labels = {"att": "ATT", "atc": "ATC", "ate": "ATE", "art": "ART"}
    return CausalResult(
        method=f"PanelMatch ({refinement} refinement, lag {lag})",
        estimand=labels[qoi],
        estimate=headline,
        se=se_head,
        pvalue=p_head,
        ci=ci_head,
        alpha=alpha,
        n_obs=int(len(data)),
        detail=detail,
        model_info=model_info,
        _citation_key="panel_match",
    )
