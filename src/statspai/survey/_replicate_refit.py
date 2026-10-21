"""``design=ReplicateDesign`` on estimators: refit with every replicate.

An estimator with a ``weights=`` argument handles a replicate-weight design
by calling itself ``1 + R`` times -- once with the full-sample weights and
once with each replicate's weights, rows of zero weight dropped -- and
installing the replicate covariance of the estimates
(:meth:`ReplicateDesign.variance`) with t inference on the design df. This
is what R ``survey`` does for ``svyglm`` / ``svycoxph`` /
``withReplicates`` on a ``svrepdesign`` and Stata ``svy brr:`` /
``svy jackknife:`` do for any estimation command.

The estimator calls :func:`replicate_refit` as its first statement::

    if design is not None and is_replicate_design(design):
        return replicate_refit(ipw, dict(locals()), design, function="sp.ipw")

Nothing estimator-specific lives here except how a result is read and
patched (:func:`_extract` / :func:`_patch`, by result type).
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..exceptions import MethodIncompatibility

#: columns of a result table that are statistics, not keys
_STAT_COLS = {
    "att",
    "estimate",
    "coef",
    "effect",
    "se",
    "std_err",
    "ci_lower",
    "ci_upper",
    "pvalue",
    "p_value",
    "tstat",
    "t",
    "z",
    "n",
    "n_obs",
    "weight",
    "weights",
    "share",
}
_EST_COLS = ("att", "estimate", "coef", "effect")


def replicate_refit(
    fn: Callable[..., Any],
    args: Dict[str, Any],
    design: Any,
    *,
    function: str,
    weights_arg: str = "weights",
    data_arg: str = "data",
    refuse: Tuple[str, ...] = ("cluster",),
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """Fit ``fn`` on the full sample and on every replicate; patch the SEs.

    ``args`` are the estimator's arguments (``dict(locals())`` at its first
    line); ``design`` is replaced by ``None`` and ``weights_arg`` by a
    column holding each replicate's weights. ``overrides`` are applied to
    every fit (e.g. a point-estimate-only mode: the estimator's own
    variance is replaced anyway).
    """
    data = args.get(data_arg)
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            f"{function}: design= needs data as a pandas DataFrame.",
            recovery_hint="Pass the data frame the design was built from.",
        )
    if len(data) != design.n or not data.index.equals(design.data.index):
        raise MethodIncompatibility(
            f"{function}: data has {len(data)} rows but the replicate design "
            f"was declared on {design.n} rows (or on a different index).",
            recovery_hint="Pass the data frame the design was built from "
            "(design.data); for a subpopulation keep all rows.",
            diagnostics={"n_data": len(data), "n_design": design.n},
        )
    given = [k for k in (weights_arg, *refuse) if args.get(k) not in (None, False)]
    if given:
        raise MethodIncompatibility(
            f"{function}: a replicate-weight design carries the weights and "
            "the variance; " + ", ".join(f"{k}=" for k in given) + " cannot "
            "be given as well.",
            recovery_hint="Drop " + ", ".join(f"{k}=" for k in given) + ".",
            diagnostics={"given": given},
        )
    col = "__svyrep_weight__"
    kwargs = dict(args)
    kwargs.pop("self", None)
    extra = kwargs.pop("kwargs", None)
    if isinstance(extra, dict):
        kwargs.update(extra)
    # None is every optional argument's "not given"; passing it explicitly
    # trips alias checks (``vce=None`` next to ``robust=``)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    kwargs.update(overrides or {})
    kwargs.pop("design", None)  # the refits are ordinary weighted fits
    kwargs[weights_arg] = col
    if "weights_as" in args and args.get("weights_as") is None:
        # sampling weights: avoids the frequency-weight warnings; the
        # variance it would choose is replaced below
        kwargs["weights_as"] = "sampling"

    def run(w: np.ndarray, quiet: bool) -> Any:
        keep = w > 0
        frame = data.assign(**{col: w})
        if not keep.all():
            frame = frame.loc[keep]
        kw = dict(kwargs)
        kw[data_arg] = frame
        if quiet:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return fn(**kw)
        return fn(**kw)

    base = run(design.weights, quiet=False)
    names, theta = _extract(base)
    reps = np.full((design.n_replicates, len(theta)), np.nan)
    failed: List[int] = []
    last_exc: Optional[Exception] = None
    for r in range(design.n_replicates):
        try:
            res_r = run(design.repweights[:, r], quiet=True)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            failed.append(r)
            last_exc = exc
            continue
        names_r, theta_r = _extract(res_r)
        reps[r] = _align(names, names_r, theta_r)
    if failed:
        raise MethodIncompatibility(
            f"{function}: the estimator failed on {len(failed)} of "
            f"{design.n_replicates} replicates (first error: "
            f"{type(last_exc).__name__}: {last_exc}). A replicate that drops "
            "a PSU can leave an arm or a category empty.",
            recovery_hint="Check that every replicate keeps both arms and "
            "every category, or simplify the model.",
            diagnostics={"failed_replicates": failed[:20]},
        )
    head = _headline_count(base)
    ok_rows = np.asarray(np.isfinite(reps).all(axis=0))
    if not ok_rows[:head].all():
        design.variance(theta[:head], reps[:, :head])  # raises with details
    V = np.full((len(theta), len(theta)), np.nan)
    idx = np.flatnonzero(ok_rows)
    V[np.ix_(idx, idx)] = design.variance(theta[idx], reps[:, idx])
    if (~ok_rows).any():
        warnings.warn(
            f"{function}: {int((~ok_rows).sum())} table entries (e.g. "
            "group-time cells) could not be estimated on every replicate; "
            "their standard errors are reported as missing.",
            UserWarning,
            stacklevel=3,
        )
    _patch(base, V, head, design, float(args.get("alpha", 0.05) or 0.05))
    return base


# ---------------------------------------------------------------------- #
#  Reading and patching results, by type
# ---------------------------------------------------------------------- #


def _tables(result: Any) -> List[Tuple[str, pd.DataFrame, str]]:
    """``(where, table, estimate column)`` of the result's effect tables."""
    out: List[Tuple[str, pd.DataFrame, str]] = []
    det = getattr(result, "detail", None)
    cands: List[Tuple[str, Any]] = [("detail", det)]
    mi = getattr(result, "model_info", None) or {}
    if isinstance(mi, dict):
        cands.append(("event_study", mi.get("event_study")))
    for where, tab in cands:
        if not isinstance(tab, pd.DataFrame) or "se" not in tab.columns:
            continue
        est = next((c for c in _EST_COLS if c in tab.columns), None)
        if est is not None and len(tab):
            out.append((where, tab, est))
    return out


def _keys(tab: pd.DataFrame, where: str) -> List[str]:
    cols = [c for c in tab.columns if str(c).lower() not in _STAT_COLS]
    if not cols:
        return [f"{where}[{i}]" for i in range(len(tab))]
    return [f"{where}{tuple(row)}" for row in tab[cols].itertuples(index=False)]


def _is_ltmle(result: Any) -> bool:
    return type(result).__name__ == "LTMLEResult"


def _extract(result: Any) -> Tuple[List[str], np.ndarray]:
    """Names and values of everything whose SE the replicates give."""
    from ..core.results import CausalResult, EconometricResults

    if _is_ltmle(result):
        return (
            ["ate", "psi_treated", "psi_control"],
            np.array([result.ate, result.psi_treated, result.psi_control], float),
        )
    if isinstance(result, CausalResult):
        names = ["__estimate__"]
        vals = [float(result.estimate)]
        for where, tab, est in _tables(result):
            names += _keys(tab, where)
            vals += list(tab[est].to_numpy(dtype=float))
        return names, np.asarray(vals, dtype=float)
    if isinstance(result, EconometricResults):
        p = result.params
        return [str(k) for k in p.index], p.to_numpy(dtype=float)
    raise MethodIncompatibility(
        f"replicate design: results of type {type(result).__name__} are not "
        "supported."
    )


def _headline_count(result: Any) -> int:
    from ..core.results import CausalResult

    if _is_ltmle(result):
        return 3
    if isinstance(result, CausalResult):
        return 1
    return len(result.params)


def _align(names: List[str], names_r: List[str], vals_r: np.ndarray) -> np.ndarray:
    if names_r == names:
        return vals_r
    lookup = dict(zip(names_r, vals_r))
    return np.array([lookup.get(k, np.nan) for k in names], dtype=float)


def _t_stats(est: np.ndarray, se: np.ndarray, df: float, alpha: float) -> Any:
    crit = float(sp_stats.t.ppf(1 - alpha / 2, df=df))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(se > 0, 2 * sp_stats.t.sf(np.abs(est / se), df=df), np.nan)
    return est - crit * se, est + crit * se, p


def _patch(
    result: Any, V: np.ndarray, head: int, design: Any, alpha_default: float
) -> None:
    from ..core.results import CausalResult, EconometricResults

    se_all = np.sqrt(np.maximum(np.diag(V), 0.0))
    se_all[~np.isfinite(np.diag(V))] = np.nan
    df = float(design.df)
    info = design.info()
    label = f"Replicate weights ({design.type}, {design.n_replicates} replicates)"
    if _is_ltmle(result):
        alpha = alpha_default
        lo, hi, p = _t_stats(np.array([result.ate]), se_all[:1], df, alpha)
        result.se = float(se_all[0])
        result.ci = (float(lo[0]), float(hi[0]))
        result.pvalue = float(p[0])
        result.detail = {
            **(result.detail or {}),
            "survey_design": info,
            "se_psi_treated": float(se_all[1]),
            "se_psi_control": float(se_all[2]),
            "inference": label + ", t(df)",
        }
        return
    if isinstance(result, CausalResult):
        alpha = float(result.alpha)
        lo, hi, p = _t_stats(np.array([result.estimate], float), se_all[:1], df, alpha)
        result.se = float(se_all[0])
        result.ci = (float(lo[0]), float(hi[0]))
        result.pvalue = float(p[0])
        result._influence_funcs = None
        pos = 1
        mi = dict(result.model_info or {})
        for where, tab, est in _tables(result):
            k = len(tab)
            new = tab.copy()
            s = se_all[pos : pos + k]
            b = new[est].to_numpy(dtype=float)
            lo, hi, p = _t_stats(b, s, df, alpha)
            new["se"] = s
            if "ci_lower" in new.columns:
                new["ci_lower"] = lo
            if "ci_upper" in new.columns:
                new["ci_upper"] = hi
            if "pvalue" in new.columns:
                new["pvalue"] = p
            if where == "detail":
                result.detail = new
            else:
                mi[where] = new
            pos += k
        for key in ("_cluster_ids", "_influence_funcs"):
            if key in mi:
                mi[key] = None
        mi["se_method"] = "replicate_weights"
        mi["inference_note"] = label + "; t intervals on the design df."
        mi["survey_design"] = info
        result.model_info = mi
        return
    if isinstance(result, EconometricResults):
        result.std_errors = pd.Series(se_all, index=result.params.index)
        dinfo = dict(result.data_info)
        dinfo.pop("inference", None)
        dinfo["df_inference"] = df
        dinfo["var_cov"] = V
        result.data_info = dinfo
        result._compute_statistics()
        mi = dict(result.model_info)
        for key in ("weights_as", "robust", "n_effective_weights"):
            mi.pop(key, None)
        mi["vcov_type"] = label
        mi["se_type"] = label
        mi["survey_design"] = info
        result.model_info = mi
        if getattr(result, "_hr_ci", None) is not None:
            ci = np.asarray(result.conf_int(), dtype=float)
            result._hr_ci = np.exp(ci)
        return
    raise MethodIncompatibility(  # pragma: no cover - _extract raised first
        f"replicate design: cannot patch {type(result).__name__}."
    )


def refuse_replicate(design: Any, function: str, why: Optional[str] = None) -> None:
    """Typed refusal for estimators without a replicate-weight path."""
    from .replicate import is_replicate_design

    if is_replicate_design(design):
        raise MethodIncompatibility(
            f"{function}: replicate-weight designs are not supported here"
            + (f" ({why})" if why else "")
            + ".",
            recovery_hint="Use a linearisation design (sp.svydesign with "
            "strata / PSUs), or an estimator that accepts replicate weights: "
            "sp.regress, sp.logit, sp.probit, sp.ipw, sp.aipw, sp.tmle, "
            "sp.cox, sp.did_2x2, sp.callaway_santanna, sp.ltmle, "
            "sp.svymean / svytotal / svyglm.",
            alternative_functions=["sp.svydesign"],
        )
