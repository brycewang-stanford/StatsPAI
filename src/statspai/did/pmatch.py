"""``sp.panel_match``: matching on treatment history for panel data.

Clean-room implementation (2026-10-20) of the matching estimator of
Imai, Kim and Wang (2023), written from the paper, the R ``PanelMatch``
documentation and the outputs of ``PanelMatch`` 3.1.5 on our fixtures
(used as a black box; it is GPL-3 and its source was not consulted).

Pipeline: ``_pmatch_prep`` (balanced arrays, covariate terms) ->
``_pmatch_sets`` (matched sets on treatment history) -> ``_pmatch_refine``
(Mahalanobis / propensity-score / CBPS weights) -> ``_pmatch_infer``
(estimates, standard errors, balance, placebo).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _pmatch_infer as inf
from ._pmatch_prep import build_panel, check_lag, parse_covariates, refinement_design
from ._pmatch_refine import refine
from ._pmatch_sets import MatchedSet, build_sets

_QOIS = ("att", "atc", "ate", "art")
_REFINEMENTS = (
    "mahalanobis",
    "ps_weight",
    "ps_match",
    "cbps_weight",
    "cbps_match",
    "none",
)
_SE_METHODS = ("bootstrap", "conditional", "unconditional")


def _refusal(msg: str, hint: str) -> MethodIncompatibility:
    return MethodIncompatibility(
        msg, recovery_hint=hint, alternative_functions=["did_multiplegt_dyn", "fect"]
    )


def _normalise_refinement(refinement: str) -> str:
    key = str(refinement).strip().lower().replace(".", "_")
    if key not in _REFINEMENTS:
        raise _refusal(
            f"refinement={refinement!r} is not supported",
            "Choose one of "
            + ", ".join(repr(r) for r in _REFINEMENTS)
            + " (marginal-structural-model weights are not implemented).",
        )
    return key


def _normalise_leads(lead: Union[int, Sequence[int]]) -> List[int]:
    leads = [lead] if np.isscalar(lead) else list(lead)
    if not leads:
        raise _refusal("lead is empty", "Pass a lead such as 0 or range(4).")
    out = []
    for f in leads:
        if not float(f).is_integer() or int(f) < 0:
            raise _refusal(
                f"lead {f!r} is not a non-negative integer",
                "Leads count periods from the treatment period: 0, 1, 2, ...",
            )
        out.append(int(f))
    return sorted(set(out))


def _validate(
    qoi: str,
    refinement: str,
    covariates,
    lag: int,
    size_match: int,
    se_method: str,
    forbid_treatment_reversal: bool,
    restrict_control_period: Optional[int],
    n_boot: int,
    alpha: float,
    placebo_test: bool,
) -> None:
    if qoi not in _QOIS:
        raise _refusal(
            f"qoi={qoi!r} is not one of {_QOIS}", "Use 'att', 'atc', 'ate' or 'art'."
        )
    if se_method not in _SE_METHODS:
        raise _refusal(
            f"se_method={se_method!r} is not one of {_SE_METHODS}",
            "Use 'bootstrap', 'conditional' or 'unconditional'.",
        )
    if not float(lag).is_integer() or int(lag) < 1:
        raise _refusal(f"lag={lag!r} must be a positive integer", "Use lag >= 1.")
    if int(size_match) < 1:
        raise _refusal("size_match must be at least 1", "Use size_match >= 1.")
    if forbid_treatment_reversal and qoi not in ("att", "art"):
        raise _refusal(
            "forbid_treatment_reversal applies to the ATT and ART only",
            "Drop forbid_treatment_reversal or use qoi='att' / 'art'.",
        )
    if restrict_control_period is not None:
        if qoi not in ("att", "art"):
            raise _refusal(
                "restrict_control_period applies to the ATT and ART only",
                "Drop restrict_control_period or use qoi='att' / 'art'.",
            )
        if not float(restrict_control_period).is_integer() or not (
            1 <= int(restrict_control_period) <= int(lag)
        ):
            raise _refusal(
                "restrict_control_period must be an integer in 1..lag",
                "Use a value between 1 and lag.",
            )
    if se_method == "bootstrap" and int(n_boot) < 2:
        raise _refusal(
            "n_boot must be at least 2", "Use n_boot >= 2 (1000 is the default)."
        )
    if not 0 < float(alpha) < 1:
        raise _refusal("alpha must lie in (0, 1)", "Use alpha=0.05 for 95% intervals.")
    if placebo_test and int(lag) < 2:
        raise _refusal(
            "the placebo test needs lag >= 2 (periods t-lag .. t-2)",
            "Increase lag or drop placebo_test.",
        )


def _sets_frame(families: Dict[str, List[MatchedSet]], units, times) -> pd.DataFrame:
    rows = []
    for name, sets in families.items():
        for s in sets:
            if s.empty:
                rows.append((name, units[s.unit], times[s.t], np.nan, np.nan))
                continue
            for c, w in zip(s.controls, s.weights):
                rows.append((name, units[s.unit], times[s.t], units[c], float(w)))
    return pd.DataFrame(
        rows, columns=["qoi", "treated_unit", "time", "control_unit", "weight"]
    )


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
        Keep only treated observations whose treatment stayed at its
        pre-switch value (0 for the ATT, 1 for the ART) in each of the
        ``restrict_control_period`` periods before ``t``.
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
        ``propensity_scores`` (score of every treated / control
        observation a propensity refinement used, as the probability of
        being the matched observation), ``bootstrap_draws`` and
        ``placebo_test`` when requested.

    Notes
    -----
    The standard errors resample (or treat as independent) units, so they
    allow arbitrary dependence within a unit over time. The analytic
    variances for several leads pooled are this function's extension: R
    ``PanelMatch`` gives the pooled average only with the bootstrap. The
    estimator applies the same formula to the averaged weights.

    Propensity refinements model the probability of being the matched
    (treated) observation among the observations of all sets at their
    period ``t``: weights are its odds (eq. 16), and ``*_match`` keeps the
    controls closest on the probability scale. For the ATC this differs
    from R ``PanelMatch`` 3.1.5, whose weights are the reciprocal odds and
    worsen balance; ``ps_match`` also keeps controls whose score equals the
    treated one, which R drops. See ``docs/dev/modules/did.md``.

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
    qoi = str(qoi).lower()
    ref_method = _normalise_refinement(refinement)
    if ref_method != "none" and not covariates:
        raise MethodIncompatibility(
            f"refinement={refinement!r} needs covariates",
            recovery_hint="Pass covariates=[...] or use refinement='none'.",
        )
    if qoi == "ate" and se_method != "bootstrap":
        raise MethodIncompatibility(
            "the ATE has only the bootstrap standard error",
            recovery_hint="Use se_method='bootstrap' with qoi='ate'.",
        )
    _validate(
        qoi,
        ref_method,
        covariates,
        lag,
        size_match,
        se_method,
        forbid_treatment_reversal,
        restrict_control_period,
        n_boot,
        alpha,
        placebo_test,
    )
    lag = int(lag)
    leads = _normalise_leads(lead)
    specs = parse_covariates(covariates, list(data.columns))
    exact_vars = list(exact_match or [])
    if balance_covariates is None:
        bal_vars = list(dict.fromkeys([s.base for s in specs] + [y]))
    else:
        bal_vars = list(balance_covariates)
    variables = list(dict.fromkeys([s.base for s in specs] + exact_vars + bal_vars))
    panel = build_panel(data, y, treat, id, time, variables)
    check_lag(lag, panel)
    Z, z_names = refinement_design(panel, specs)
    exact = [panel.raw[v] for v in exact_vars] or None

    kinds = ["att", "atc"] if qoi == "ate" else [qoi]

    def matched(outcome_history: bool):
        fams: Dict[str, List[MatchedSet]] = {}
        info: Dict[str, dict] = {}
        for k in kinds:
            sets = build_sets(
                panel,
                kind=k,
                lag=lag,
                leads=leads,
                match_missing=bool(match_missing),
                exact=exact,
                restrict_control_period=restrict_control_period,
                forbid_treatment_reversal=bool(forbid_treatment_reversal),
                outcome_history=outcome_history,
            )
            info[k] = refine(
                sets, ref_method, Z, lag, int(size_match), bool(diagonal_covariance)
            )
            fams[k] = sets
        return fams, info

    families, refine_info = matched(False)
    n_sets = {k: sum(not s.empty for s in v) for k, v in families.items()}
    if sum(n_sets.values()) == 0:
        raise DataInsufficient(
            "no treated observation has a matched set",
            recovery_hint=(
                "Shorten lag, set match_missing=True, drop exact_match / "
                "restrict_control_period, or use fewer leads."
            ),
            diagnostics={
                "n_treated_observations": sum(len(v) for v in families.values())
            },
        )

    rng = np.random.default_rng(seed)
    fam_objs = [inf.Family(k, v) for k, v in families.items()]
    pairs = [(f, -1) for f in leads]
    A, B, n_star = inf.contrasts(fam_objs, panel.Y, pairs)
    draws = (
        inf.run_bootstrap(A, B, int(n_boot), rng) if se_method == "bootstrap" else None
    )
    table = inf.inference_table(leads, A, B, n_star, se_method, alpha, draws)
    table = table.rename(columns={"term": "lead"})
    table.insert(1, "term", [f"t+{f}" for f in leads])
    table = table[["lead", "term", "estimate", "se", "pvalue", "ci_lower", "ci_upper"]]
    if len(leads) == 1:
        row = table.iloc[0]
        est, se, ci, pval = (
            float(row["estimate"]),
            float(row["se"]),
            (float(row["ci_lower"]), float(row["ci_upper"])),
            float(row["pvalue"]),
        )
    else:
        est, se, ci, pval = inf.pooled_summary(A, B, n_star, se_method, alpha, draws)

    model_info: Dict[str, object] = {
        "qoi": qoi,
        "refinement": ref_method,
        "lag": lag,
        "leads": leads,
        "covariates": [c.text for c in specs],
        "se_method": se_method,
        "matched_sets": _sets_frame(families, panel.units, panel.times),
        "n_matched_sets": int(sum(n_sets.values())),
        "n_matched_sets_by_qoi": dict(n_sets),
        "n_unmatched": int(sum(sum(s.empty for s in v) for v in families.values())),
        "n_units": panel.n_units,
        "n_periods": panel.n_periods,
        "n_units_contributing": n_star,
        "refinement_covariates": z_names,
        "refinement_fit": {
            k: {a: b for a, b in v.items() if a != "cells"}
            for k, v in refine_info.items()
        },
        "pooled": len(leads) > 1,
        "mean_controls_per_set": (
            float(np.mean([s.controls.size for s in families[kinds[0]] if not s.empty]))
            if n_sets[kinds[0]]
            else float("nan")
        ),
    }
    score_frames = [
        pd.DataFrame(
            {
                "qoi": k,
                "unit": panel.units[v["cells"][0]],
                "time": panel.times[v["cells"][1]],
                "pscore": v["cells"][2],
            }
        )
        for k, v in refine_info.items()
        if "cells" in v
    ]
    if score_frames:
        model_info["propensity_scores"] = pd.concat(score_frames, ignore_index=True)
    bal_arrays = {v: panel.raw[v] for v in bal_vars}
    first = kinds[0]
    model_info["balance"] = inf.balance_table(
        families[first], bal_arrays, lag, refined=True
    )
    model_info["balance_unrefined"] = inf.balance_table(
        families[first], bal_arrays, lag, refined=False
    )
    if qoi == "ate":
        model_info["balance_atc"] = inf.balance_table(
            families["atc"], bal_arrays, lag, refined=True
        )
        model_info["balance_atc_unrefined"] = inf.balance_table(
            families["atc"], bal_arrays, lag, refined=False
        )
    if draws is not None:
        model_info["bootstrap_draws"] = pd.DataFrame(
            draws, columns=[f"t+{f}" for f in leads]
        )
        model_info["n_boot"] = int(n_boot)

    if placebo_test:
        pl_fams, _ = matched(True)
        pl_objs = [inf.Family(k, v) for k, v in pl_fams.items()]
        if sum(not s.empty for v in pl_fams.values() for s in v) == 0:
            raise DataInsufficient(
                "no matched set observes the outcome over the whole history window",
                recovery_hint="Drop placebo_test or shorten lag.",
            )
        ks = list(range(lag, 1, -1))
        pl_pairs = [(-1, -k) for k in ks]
        pA, pB, pn = inf.contrasts(pl_objs, panel.Y, pl_pairs)
        pdraws = (
            inf.run_bootstrap(pA, pB, int(n_boot), rng)
            if se_method == "bootstrap"
            else None
        )
        model_info["placebo_test"] = inf.inference_table(
            [f"t-{k}" for k in ks], pA, pB, pn, se_method, alpha, pdraws
        )[["term", "estimate", "se", "pvalue", "ci_lower", "ci_upper"]]
        model_info["placebo_matched_sets"] = _sets_frame(
            pl_fams, panel.units, panel.times
        )

    return CausalResult(
        method=f"PanelMatch ({ref_method} refinement, lag {lag})",
        estimand=qoi.upper(),
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(data)),
        detail=table,
        model_info=model_info,
        _citation_key="imai2023matching",
    )
