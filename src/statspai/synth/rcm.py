"""Regression control method (the panel data approach of Hsiao, Ching and Wan).

The untreated outcome of the treated unit is predicted from the outcomes of
control units by a linear regression fitted on the pre-treatment periods:

    y_1t = a + sum_j b_j y_jt + e_t,        t before the treatment.

Unlike synthetic control the coefficients are unrestricted (they may be
negative and need not sum to one) and there is a constant, so the method
uses the correlation between units rather than a convex combination of
them. Which control units enter is chosen in two steps: for every model
size the best-fitting subset (exactly, or by forward / backward stepwise
selection), then the size by an information criterion.

Inference is by placebo: the procedure is repeated treating each control
unit as if it were treated (in space) or at a date before the treatment
(in time).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["rcm"]

_CRITERIA = ("aicc", "aic", "bic", "mbic")
_METHODS = ("best", "forward", "backward")


# ---------------------------------------------------------------- selection
def _sweep_in_all(G: np.ndarray, p: int) -> np.ndarray:
    """Sweep the first ``p`` pivots of the cross-product matrix ``G``."""
    M = G.astype(float).copy()
    for k in range(p):
        d = M[k, k]
        if abs(d) < 1e-12 * max(1.0, abs(G[k, k])):
            raise MethodIncompatibility(
                "sp.synth(method='rcm'): the control units are collinear in the pre-treatment "
                "periods.",
                recovery_hint="Drop duplicated control units, or use "
                "selection='forward'.",
            )
        column = M[:, k].copy()
        M -= np.outer(column, column) / d
        M[:, k] = column / d
        M[k, :] = column / d
        M[k, k] = -1.0 / d
    return M


def _rss(y: np.ndarray, X: np.ndarray, cols: Sequence[int]) -> float:
    Z = np.column_stack([np.ones(len(y))] + [X[:, c] for c in cols])
    e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(e @ e)


def _stepwise(y: np.ndarray, X: np.ndarray, forward: bool, max_k: int) -> Dict:
    """Best-of-step subsets: add (or drop) one control unit at a time."""
    p = X.shape[1]
    out: Dict[int, Tuple[float, Tuple[int, ...]]] = {}
    if forward:
        chosen: List[int] = []
        for _ in range(min(max_k, p)):
            trial = [(_rss(y, X, chosen + [j]), j) for j in range(p) if j not in chosen]
            rss, j = min(trial)
            chosen.append(j)
            out[len(chosen)] = (rss, tuple(sorted(chosen)))
    else:
        chosen = list(range(p))
        out[p] = (_rss(y, X, chosen), tuple(chosen))
        while len(chosen) > 1:
            trial = [(_rss(y, X, [c for c in chosen if c != j]), j) for j in chosen]
            rss, j = min(trial)
            chosen.remove(j)
            out[len(chosen)] = (rss, tuple(chosen))
    return out


def _candidates(
    y: np.ndarray, X: np.ndarray, selection: str, max_nodes: int
) -> Dict[int, Tuple[float, Tuple[int, ...]]]:
    """For each model size, the residual sum of squares and the members of
    the subset the selection method proposes."""
    T0, p = X.shape
    largest = min(p, T0 - 3)  # AICc needs T0 - K - 3 >= 1 ... and a fit
    if largest < 1:
        raise DataInsufficient(
            f"sp.synth(method='rcm'): {T0} pre-treatment periods are too few to fit a model.",
            recovery_hint="The method needs at least four pre-treatment periods.",
        )
    if selection == "forward":
        return _stepwise(y, X, True, largest)
    if p > T0 - 2:
        raise MethodIncompatibility(
            f"sp.synth(method='rcm'): selection={selection!r} starts from the model with all "
            f"{p} control units, which {T0} pre-treatment periods cannot fit.",
            recovery_hint="Use selection='forward', or restrict donors=.",
        )
    if selection == "backward":
        return _stepwise(y, X, False, largest)
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    full = np.column_stack([Xc, yc])
    swept = _sweep_in_all(full.T @ full, p)
    from ._rcm_kernels import best_subsets  # numba: imported on first use

    best, masks, nodes = best_subsets(swept, max_nodes)
    if nodes < 0:
        raise MethodIncompatibility(
            f"sp.synth(method='rcm'): the exact best-subset search over {p} control units did "
            f"not finish within {max_nodes} branches.",
            recovery_hint="Use selection='forward', or restrict donors=.",
        )
    return {
        k: (float(best[k]), tuple(int(j) for j in np.flatnonzero(masks[k])))
        for k in range(1, p + 1)
    }


def _criteria(rss: float, T0: int, k: int) -> Dict[str, float]:
    """AIC, AICc, BIC and MBIC of a model with ``k`` control units.

    ``AIC = T0 log(RSS / T0) + 2 (k + 2)`` and the small-sample correction
    of Hsiao, Ching and Wan; ``MBIC`` replaces the factor 2 by ``log(T0) *
    log(log(k + 1))``, the form the Stata command ``rcm`` uses.
    """
    base = T0 * np.log(rss / T0)
    aic = base + 2.0 * (k + 2)
    room = T0 - (k + 2) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        mbic = base + (k + 2) * np.log(T0) * np.log(np.log(k + 1.0))
    return {
        "aicc": aic + 2.0 * (k + 2) * (k + 3) / room if room >= 1 else np.nan,
        "aic": aic,
        "bic": base + (k + 2) * np.log(T0),
        "mbic": float(mbic),
    }


def _select(
    y: np.ndarray,
    X: np.ndarray,
    selection: str,
    criterion: str,
    max_nodes: int,
) -> Tuple[Tuple[int, ...], pd.DataFrame]:
    T0 = len(y)
    tss = float(((y - y.mean()) ** 2).sum())
    rows = []
    members: Dict[int, Tuple[int, ...]] = {}
    for k, (rss, cols) in sorted(_candidates(y, X, selection, max_nodes).items()):
        if rss <= 0 or T0 - k - 1 <= 0:
            continue
        row = {"K": k, **_criteria(rss, T0, k), "r2": 1.0 - rss / tss, "rss": rss}
        rows.append(row)
        members[k] = cols
    table = pd.DataFrame(rows).set_index("K")
    scores = table[criterion].dropna()
    if scores.empty:
        raise DataInsufficient(
            "sp.synth(method='rcm'): no model size leaves the degrees of freedom the "
            f"criterion {criterion!r} needs.",
            recovery_hint="Use criterion='bic', or a longer pre-treatment " "period.",
        )
    return members[int(scores.idxmin())], table


# ---------------------------------------------------------------- one unit
def _fit_unit(
    wide: pd.DataFrame,
    treated: Any,
    donors: List[Any],
    pre: np.ndarray,
    post: np.ndarray,
    selection: str,
    criterion: str,
    max_nodes: int,
) -> Dict[str, Any]:
    y = wide[treated].to_numpy(dtype=float)
    X = wide[donors].to_numpy(dtype=float)
    cols, table = _select(y[pre], X[pre], selection, criterion, max_nodes)
    Z = np.column_stack([np.ones(len(y))] + [X[:, c] for c in cols])
    Zp, yp = Z[pre], y[pre]
    bread = np.linalg.inv(Zp.T @ Zp)
    beta = bread @ Zp.T @ yp
    resid = yp - Zp @ beta
    T0, k = Zp.shape[0], len(cols)
    rss = float(resid @ resid)
    sigma2 = rss / (T0 - k - 1)
    predicted = Z @ beta
    effect = y - predicted
    return {
        "members": [donors[c] for c in cols],
        "table": table,
        "beta": beta,
        "se": np.sqrt(sigma2 * np.diag(bread)),
        "df_resid": T0 - k - 1,
        "predicted": predicted,
        "effect": effect,
        # mean squared prediction error of the fit, per residual degree of
        # freedom (the fit used k + 1 coefficients)
        "pre_mspe": sigma2,
        "post_mspe": float(np.mean(effect[post] ** 2)),
        "r2": 1.0 - rss / float(((yp - yp.mean()) ** 2).sum()),
        # root mean squared error of the fit, with its degrees of freedom
        "rmse": float(np.sqrt(sigma2)),
    }


# -------------------------------------------------------------- public API
def rcm(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    *,
    donors: Optional[Sequence[Any]] = None,
    pre_periods: Optional[Sequence[Any]] = None,
    post_periods: Optional[Sequence[Any]] = None,
    selection: str = "best",
    criterion: str = "aicc",
    placebo: bool = False,
    placebo_cutoff: Optional[float] = None,
    placebo_time: Optional[Any] = None,
    alpha: float = 0.05,
    max_nodes: int = 50_000_000,
) -> CausalResult:
    """Regression control method for one treated unit
    (``sp.synth(..., method='rcm')``).

    Parameters
    ----------
    data : pandas.DataFrame
        Long panel with one row per unit and period.
    outcome, unit, time : str
        Outcome, unit identifier and time variable.
    treated_unit : scalar
        The unit that receives the treatment.
    treatment_time : scalar
        First treated period. Earlier periods are the fitting sample.
    donors : sequence, optional
        Control units that may enter the model. Default: every other unit.
    pre_periods, post_periods : sequence, optional
        Periods used for fitting and for the effect. Default: all periods
        before, and from, ``treatment_time``.
    selection : {'best', 'forward', 'backward'}, default 'best'
        How the candidate model of each size is found. ``'best'`` is the
        exact best subset (branch and bound; with many control units it is
        slow, and it needs fewer control units than pre-treatment periods).
        ``'forward'`` adds one unit at a time and also works when there are
        more control units than periods; ``'backward'`` drops one at a time.
    criterion : {'aicc', 'aic', 'bic', 'mbic'}, default 'aicc'
        Which size is kept. ``'aicc'`` is the corrected AIC Hsiao, Ching and
        Wan use.
    placebo : bool, default False
        Repeat the procedure with each control unit as the pretend treated
        unit (the true treated unit joins its donor pool) and report placebo
        p-values.
    placebo_cutoff : float, optional
        Leave out pretend-treated units whose pre-treatment mean squared
        prediction error exceeds this multiple of the treated unit's when
        computing the period-by-period p-values.
    placebo_time : scalar, optional
        A pretend treatment date before ``treatment_time``: the model is
        fitted on the periods before it and the "effects" from it onwards
        are reported in ``model_info['placebo_time']``.
    alpha : float, default 0.05
        Kept on the result; the method has no analytic interval.
    max_nodes : int, default 50,000,000
        Limit on the branches of the exact search.

    Returns
    -------
    CausalResult
        ``estimate`` is the average effect over the post-treatment periods.
        ``detail`` lists, for every period, the actual outcome, the
        prediction and their difference (with placebo p-values when
        requested). ``model_info`` has ``selected`` (the control units of
        the model), ``coefficients`` (the pre-treatment regression),
        ``selection_table`` (criteria by model size), ``pre_r2``,
        ``pre_rmse`` and the placebo tables. With ``placebo=True``,
        ``pvalue`` is the share of units (the treated one included) whose
        post/pre mean-squared-error ratio is at least the treated unit's.

    Notes
    -----
    The prediction is a counterfactual only if the relation between the
    treated and the control units would have stayed the same without the
    treatment, and if the control units are unaffected by it. The fit is
    unrestricted, so with few pre-treatment periods it can overfit: compare
    the effects under ``selection='forward'`` and another ``criterion``,
    and run the placebo in time on a date with no intervention.

    The numbers reproduce the Stata command ``rcm`` (``method(best)`` /
    ``forward`` / ``backward`` with ``criterion(aicc | aic | bic | mbic)``).
    Its lasso option is not implemented.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> T, J = 40, 6
    >>> factor = rng.normal(size=T).cumsum()
    >>> rows = []
    >>> for j in range(J):
    ...     y = (0.5 + 0.2 * j) * factor + rng.normal(scale=0.3, size=T)
    ...     if j == 0:
    ...         y[30:] += 2.0
    ...     rows += [(j, t, y[t]) for t in range(T)]
    >>> df = pd.DataFrame(rows, columns=["unit", "t", "y"])
    >>> fit = sp.synth(df, "y", "unit", "t", treated_unit=0, treatment_time=30,
    ...                method="rcm")
    >>> bool(abs(fit.estimate - 2.0) < 1.0)
    True

    References
    ----------
    hsiao2012panel; yan2022rcm
    """
    selection, criterion = selection.lower(), criterion.lower()
    if selection not in _METHODS or criterion not in _CRITERIA:
        raise MethodIncompatibility(
            f"sp.synth(method='rcm'): selection must be one of {_METHODS} and criterion one "
            f"of {_CRITERIA}; got {selection!r}, {criterion!r}.",
            recovery_hint="Use selection='best', criterion='aicc'.",
        )
    for name in (outcome, unit, time):
        if name not in data.columns:
            raise MethodIncompatibility(
                f"sp.synth(method='rcm'): {name!r} is not a column of the data.",
                recovery_hint="Check outcome=, unit= and time=.",
            )
    wide = data.pivot(index=time, columns=unit, values=outcome).sort_index()
    if treated_unit not in wide.columns:
        raise MethodIncompatibility(
            f"sp.synth(method='rcm'): treated_unit={treated_unit!r} is not a value of {unit!r}.",
            recovery_hint="Pass one of the unit identifiers.",
        )
    pool = [u for u in wide.columns if u != treated_unit]
    if donors is not None:
        unknown = [u for u in donors if u not in pool]
        if unknown:
            raise MethodIncompatibility(
                f"sp.synth(method='rcm'): donors {unknown} are not control units of the data.",
                recovery_hint="List identifiers other than the treated unit.",
            )
        pool = list(donors)
    periods = wide.index
    pre = np.asarray(
        periods.isin(list(pre_periods))
        if pre_periods is not None
        else periods < treatment_time
    )
    post = np.asarray(
        periods.isin(list(post_periods))
        if post_periods is not None
        else periods >= treatment_time
    )
    used = wide.loc[pre | post, [treated_unit] + pool]
    if used.isna().any().any():
        missing = used.columns[used.isna().any()].tolist()
        raise DataInsufficient(
            f"sp.synth(method='rcm'): unit(s) {missing} have missing outcomes in the periods "
            "used.",
            recovery_hint="Drop those units from donors=, or fill the gaps.",
        )
    if pre.sum() < 4 or post.sum() < 1 or not pool:
        raise DataInsufficient(
            "sp.synth(method='rcm') needs at least four pre-treatment periods, one "
            "post-treatment period and one control unit.",
            recovery_hint="Check treatment_time and the donor pool.",
        )

    fit = _fit_unit(
        wide, treated_unit, pool, pre, post, selection, criterion, max_nodes
    )
    keep = pre | post
    y = wide[treated_unit].to_numpy(dtype=float)
    detail = pd.DataFrame(
        {
            "time": periods[keep],
            "actual": y[keep],
            "predicted": fit["predicted"][keep],
            "effect": fit["effect"][keep],
            "post": post[keep],
        }
    ).reset_index(drop=True)
    names = ["_cons"] + [str(m) for m in fit["members"]]
    tvalue = fit["beta"] / fit["se"]
    coefficients = pd.DataFrame(
        {
            "coef": fit["beta"],
            "se": fit["se"],
            "t": tvalue,
            "pvalue": 2.0 * stats.t.sf(np.abs(tvalue), fit["df_resid"]),
        },
        index=names,
    )
    att = float(np.mean(fit["effect"][post]))
    model_info: Dict[str, Any] = {
        "selection": selection,
        "criterion": criterion,
        "selected": list(fit["members"]),
        "n_selected": len(fit["members"]),
        "n_donors": len(pool),
        "coefficients": coefficients,
        "selection_table": fit["table"],
        "pre_r2": fit["r2"],
        "pre_rmse": fit["rmse"],
        "pre_mspe": fit["pre_mspe"],
        "post_mspe": fit["post_mspe"],
        "n_pre": int(pre.sum()),
        "n_post": int(post.sum()),
        "treated_unit": treated_unit,
        "treatment_time": treatment_time,
    }

    pvalue = float("nan")
    if placebo:
        rows = []
        effects = {}
        for fake in pool:
            # the treated unit moves into the donor pool of a pretend-treated
            # one, as in Abadie, Diamond and Hainmueller's placebo runs
            others = [u for u in pool if u != fake] + [treated_unit]
            run = _fit_unit(
                wide, fake, others, pre, post, selection, criterion, max_nodes
            )
            rows.append(
                {
                    "unit": fake,
                    "pre_mspe": run["pre_mspe"],
                    "post_mspe": run["post_mspe"],
                }
            )
            effects[fake] = run["effect"][post]
        table = pd.DataFrame(
            [
                {
                    "unit": treated_unit,
                    "pre_mspe": fit["pre_mspe"],
                    "post_mspe": fit["post_mspe"],
                }
            ]
            + rows
        ).set_index("unit")
        table["ratio"] = table["post_mspe"] / table["pre_mspe"]
        table["pre_mspe_relative"] = table["pre_mspe"] / fit["pre_mspe"]
        own = float(table.loc[treated_unit, "ratio"])
        pvalue = float(np.mean(table["ratio"].to_numpy() >= own))
        kept = [
            u
            for u in effects
            if placebo_cutoff is None
            or table.loc[u, "pre_mspe_relative"] <= placebo_cutoff
        ]
        ratios_kept = table.loc[[treated_unit] + kept, "ratio"].to_numpy()
        tau = fit["effect"][post]
        cloud = np.vstack([tau] + [effects[u] for u in kept])
        post_rows = detail["post"].to_numpy()
        for label, hit in (
            ("p_two_sided", np.abs(cloud) >= np.abs(tau)),
            ("p_right", cloud >= tau),
            ("p_left", cloud <= tau),
        ):
            column = np.full(len(detail), np.nan)
            column[post_rows] = hit.mean(axis=0)
            detail[label] = column
        model_info.update(
            placebo_units=table,
            placebo_pvalue=pvalue,
            placebo_pvalue_cutoff=float(np.mean(ratios_kept >= own)),
            placebo_excluded=[u for u in effects if u not in kept],
            placebo_cutoff=placebo_cutoff,
        )
    if placebo_time is not None:
        fake_pre = pre & np.asarray(periods < placebo_time)
        fake_post = np.asarray(periods >= placebo_time) & (pre | post)
        if fake_pre.sum() < 4:
            raise DataInsufficient(
                "sp.synth(method='rcm'): placebo_time leaves fewer than four periods to fit.",
                recovery_hint="Choose a later placebo_time.",
            )
        run = _fit_unit(
            wide, treated_unit, pool, fake_pre, fake_post, selection, criterion,
            max_nodes,
        )  # fmt: skip
        model_info["placebo_time"] = pd.DataFrame(
            {
                "time": periods[fake_post],
                "actual": y[fake_post],
                "predicted": run["predicted"][fake_post],
                "effect": run["effect"][fake_post],
            }
        ).reset_index(drop=True)
        model_info["placebo_time_selected"] = list(run["members"])

    return CausalResult(
        method="Regression control method (panel data approach)",
        estimand="ATT",
        estimate=att,
        se=float("nan"),
        pvalue=pvalue,
        ci=(float("nan"), float("nan")),
        alpha=alpha,
        n_obs=int(keep.sum()),
        detail=detail,
        model_info=model_info,
    )
