"""Calibrated placebo simulation: pick a DiD estimator on your own panel.

Every applied DiD paper picks an estimator, and the usual justification is a
citation.  A citation says nothing about how that estimator behaves on *this*
panel — its cohort structure, its serial correlation, its number of treated
units.  The recommendation in the recent comparative literature is to settle
the question empirically: inject a known effect into the researcher's own
data under a randomised adoption pattern, refit every candidate, and read off
bias, RMSE and coverage against a truth you control
[@ulloaperez2025comparative].

:func:`did_calibrated_simulation` is that loop:

1. **Calibrate.**  Subtract the estimated dynamic effect (the imputation /
   BJS fit) from the observed outcome, so the panel keeps its fixed effects,
   its error realisations and its serial correlation but carries an average
   effect of exactly zero at every horizon.
2. **Re-randomise.**  Draw a new adoption pattern — permute the observed
   cohort labels, or draw cohorts i.i.d. from their empirical distribution.
   Parallel trends then holds *by construction*, so dispersion across draws
   is sampling noise rather than a violated assumption.
3. **Inject.**  Add a known effect to the treated cells.
4. **Refit** every candidate and score it against that known effect.

What comes back is a table of bias, RMSE, coverage and rejection rate with
Monte Carlo standard errors, plus :meth:`DidSimulationStudy.best` for the
criterion the study cares about.

Scope
-----
* The target is the **overall ATT** (the average injected effect over treated
  observations).  Horizon-by-horizon study is not covered.
* **Covariates are not part of the harness.**  The placebo assignment is
  randomised, so unconditional parallel trends holds by construction and
  covariate adjustment cannot fix a bias that is not there.  A harness that
  scored covariate-adjusted estimators would need a confounded assignment,
  which is a different design.
* The reported "truth" is the treated-observation average of the injected
  effect.  With a constant ``effect`` (the default) every aggregation
  convention agrees exactly.  When ``effect`` varies across cohorts or
  horizons, estimators that target a different weighting of the same cells
  show an estimand difference that this table cannot distinguish from bias —
  ``model_info['heterogeneous_effect']`` flags that case.

References
----------
[@ulloaperez2025comparative] Ulloa-Perez, Bair, Navathe and Linn (2025),
    "Comparative Evaluation of Difference in Differences Methods for
    Staggered Adoption Interventions", arXiv:2508.14365.
[@borusyak2024revisiting] Borusyak, Jaravel and Spiess (2024), "Revisiting
    Event-Study Designs: Robust and Efficient Estimation", *Review of
    Economic Studies* — the imputation fit used by the calibration step.
"""

from __future__ import annotations

import inspect
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..exceptions import (
    AssumptionWarning,
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from ..workflow._degradation import record_degradation

__all__ = ["did_calibrated_simulation", "DidSimulationStudy"]

# Column names of the frame handed to the estimators.  The simulated panel is
# rebuilt from arrays, so these never collide with the caller's columns.
_Y, _ID, _T, _G, _D = "y", "id", "t", "g", "d"

_ASSIGNMENTS = ("resample_cohorts", "random_timing", "observed")
_RESAMPLES = ("units", "wild", "none")
_CALIBRATIONS = ("imputation", "none")

#: Short spellings accepted for ``estimators=``.
_ESTIMATOR_ALIASES = {
    "cs": "callaway_santanna",
    "sa": "sun_abraham",
    "bjs": "did_imputation",
    "did2s": "gardner_did",
    "dcdh": "did_multiplegt_dyn",
    "stacked": "stacked_did",
    "lpdid": "lp_did",
}

_DEFAULT_ESTIMATORS = (
    "twfe",
    "callaway_santanna",
    "sun_abraham",
    "did_imputation",
    "gardner_did",
    "etwfe",
)


# ======================================================================
# Estimator adapters
# ======================================================================
#
# Each adapter takes the simulated frame plus the design options and returns
# ``(estimate, se)`` for the overall ATT.  Failures propagate: the caller
# records them per draw rather than letting a broken estimator look good by
# quietly dropping its bad draws.


def _fit_twfe(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from ..fixest import feols

    fit = feols(f"{_Y} ~ {_D} | {_ID} + {_T}", data=df, vcov={"CRV1": _ID})
    r = fit[0] if isinstance(fit, list) else fit  # one formula, one fit
    return float(np.asarray(r.params[_D])), float(np.asarray(r.std_errors[_D]))


def _fit_cs(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .aggte import aggte
    from .callaway_santanna import callaway_santanna

    cs = callaway_santanna(
        df,
        y=_Y,
        g=_G,
        t=_T,
        i=_ID,
        estimator="reg",
        control_group=opt.control_group,
        base_period="universal",
    )
    a = aggte(cs, type="simple", bstrap=False, cband=False)
    return float(a.estimate), float(a.se)


def _fit_sa(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .sun_abraham import sun_abraham

    # Sun-Abraham compares with a cohort, never with not-yet-treated cells;
    # the calibration keeps at least two never-treated units, so that cohort
    # is the reference under either setting
    r = sun_abraham(df, y=_Y, g=_G, t=_T, i=_ID, control_group="nevertreated")
    return (
        float(r.model_info["att_fixest_att"]),
        float(r.model_info["se_fixest_att"]),
    )


def _fit_bjs(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .did_imputation import did_imputation

    r = did_imputation(df, y=_Y, group=_ID, time=_T, first_treat=_G, cluster=_ID)
    return float(r.estimate), float(r.se)


def _fit_did2s(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .gardner_2s import gardner_did

    r = gardner_did(df, y=_Y, group=_ID, time=_T, first_treat=_G, cluster=_ID)
    return float(r.estimate), float(r.se)


def _fit_etwfe(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .wooldridge_did import etwfe, etwfe_emfx

    fit = etwfe(
        df,
        y=_Y,
        group=_ID,
        time=_T,
        first_treat=_G,
        cluster=_ID,
        cgroup=opt.etwfe_cgroup,
    )
    m = etwfe_emfx(fit, type="simple")
    return float(m.estimate), float(m.se)


def _fit_dcdh(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .did_multiplegt_dyn import did_multiplegt_dyn

    r = did_multiplegt_dyn(
        df,
        y=_Y,
        group=_ID,
        time=_T,
        treatment=_D,
        dynamic=opt.max_horizon,
        placebo=0,
        aggregation="switchers",
        se_method="analytic",
        n_boot=0,
        cluster=_ID,
    )
    return float(r.estimate), float(r.se)


def _fit_stacked(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .stacked_did import stacked_did

    r = stacked_did(
        df,
        y=_Y,
        group=_ID,
        time=_T,
        first_treat=_G,
        window=(-opt.max_horizon, opt.max_horizon),
        cluster=_ID,
        never_treated_only=opt.control_group == "nevertreated",
    )
    return float(r.estimate), float(r.se)


def _fit_lpdid(df: pd.DataFrame, opt: "_Options") -> Tuple[float, float]:
    from .lp_did import lp_did

    r = lp_did(
        df,
        y=_Y,
        unit=_ID,
        time=_T,
        treatment=_D,
        horizons=(-opt.max_horizon, opt.max_horizon),
        clean_controls=(
            "never_treated"
            if opt.control_group == "nevertreated"
            else "not_yet_treated"
        ),
        cluster=_ID,
    )
    post = r.model_info["pooled"]["post"]
    return float(post["estimate"]), float(post["se"])


_ADAPTERS: Dict[str, Callable[[pd.DataFrame, "_Options"], Tuple[float, float]]] = {
    "twfe": _fit_twfe,
    "callaway_santanna": _fit_cs,
    "sun_abraham": _fit_sa,
    "did_imputation": _fit_bjs,
    "gardner_did": _fit_did2s,
    "etwfe": _fit_etwfe,
    "did_multiplegt_dyn": _fit_dcdh,
    "stacked_did": _fit_stacked,
    "lp_did": _fit_lpdid,
}


# ======================================================================
# Design containers
# ======================================================================


@dataclass
class _Options:
    """Everything a worker process needs besides the calibrated panel."""

    estimators: Tuple[str, ...]
    assignment: str
    resample: str
    effect: Any
    alpha: float
    control_group: str
    max_horizon: int

    @property
    def etwfe_cgroup(self) -> str:
        # sp.etwfe spells the never-treated option 'nevertreated'.
        return "nevertreated" if self.control_group == "nevertreated" else "notyet"


@dataclass
class _Panel:
    """The calibrated no-effect panel, in codes."""

    y0: np.ndarray  # calibrated outcome, one entry per row
    fitted: np.ndarray  # two-way FE fit of y0
    resid: np.ndarray  # y0 - fitted
    resid_mat: Optional[np.ndarray]  # units x periods, balanced panels only
    uid: np.ndarray  # unit code, 0 .. n_units-1
    tid: np.ndarray  # period code, 1 .. n_times
    cohort: np.ndarray  # per unit: 0 (never treated) or the adoption period
    n_units: int
    n_times: int
    balanced: bool
    notes: Dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Preparation, calibration, drawing
# ======================================================================


def _as_float(value: Any) -> Optional[float]:
    """``value`` as a float, or ``None`` when it is not numeric at all."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prepare(
    data: pd.DataFrame, y: str, id: str, time: str, cohort: str
) -> Tuple[_Panel, Dict[str, Any]]:
    """Validate the panel and recode it to ``0..n-1`` / ``1..T`` integers."""
    missing = [c for c in (y, id, time, cohort) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"Columns not found in `data`: {missing}.",
            diagnostics={"columns": list(data.columns)[:40]},
        )
    df = data[[y, id, time, cohort]].copy()
    if df[[y, id, time]].isna().any().any():
        raise MethodIncompatibility(
            "`y`, `id` and `time` must be complete; drop or impute the "
            "missing rows before calibrating a simulation on them.",
            diagnostics={"n_missing": int(df[[y, id, time]].isna().sum().sum())},
        )
    if df.duplicated([id, time]).any():
        raise MethodIncompatibility(
            "More than one row per unit x period; collapse to cell means " "first.",
            diagnostics={"n_duplicate_rows": int(df.duplicated([id, time]).sum())},
        )

    periods = np.sort(df[time].unique())
    n_times = int(periods.size)
    if n_times < 3:
        raise DataInsufficient(
            "A calibrated simulation needs at least 3 periods (one pre, one "
            f"post, one to spare); got {n_times}.",
            diagnostics={"n_times": n_times},
        )
    tid = pd.Categorical(df[time], categories=periods).codes.astype(int) + 1

    units = np.sort(df[id].unique())
    uid: np.ndarray = pd.Categorical(df[id], categories=units).codes.astype(int)
    n_units = int(units.size)

    # Cohort -> adoption period code; 0 / NaN / inf all mean never treated.
    raw = df[cohort].to_numpy()
    first = pd.Series(raw, index=df.index).groupby(uid).first().sort_index()
    varies = pd.Series(raw, index=df.index).groupby(uid).nunique(dropna=False)
    if int(varies.max()) > 1:
        raise MethodIncompatibility(
            "`cohort` must be constant within a unit (it is the period of "
            "first treatment, not a treatment indicator).",
            diagnostics={"n_units_varying": int((varies > 1).sum())},
        )
    code_of = {p: k + 1 for k, p in enumerate(periods)}
    cohort_codes: np.ndarray = np.zeros(n_units, dtype=int)
    unknown: List[Any] = []
    for u, raw_val in first.items():
        val: Any = raw_val
        # 0, NaN and inf are all the never-treated sentinel (the convention
        # shared with sp.did_imputation); 0 is caught before the numeric
        # comparisons below, which would otherwise read it as "adopted
        # before the panel starts".
        if val is None or bool(pd.isna(val)):
            continue
        fv = _as_float(val)
        if fv is not None and (fv == 0.0 or not np.isfinite(fv)):
            continue
        if val in code_of:
            cohort_codes[int(u)] = code_of[val]
        elif fv is not None and fv <= float(periods[0]):
            cohort_codes[int(u)] = 1  # treated before the panel starts
        elif fv is not None and fv > float(periods[-1]):
            cohort_codes[int(u)] = 0  # adopts after the panel ends
        else:
            unknown.append(val)
    if unknown:
        raise MethodIncompatibility(
            "`cohort` values are not periods of the panel: "
            f"{sorted(set(map(str, unknown)))[:5]}.",
            diagnostics={"n_unknown": len(unknown)},
        )

    notes: Dict[str, Any] = {}
    always = cohort_codes == 1
    if always.any():
        # A unit treated in the first period has no pre-period: every
        # estimator drops it, and it is not a valid control either.  Dropping
        # it here keeps the injected truth and the estimands on the same
        # cells, and keeps permuted assignments free of the same problem.
        keep_rows = ~always[uid]
        warnings.warn(
            f"{int(always.sum())} unit(s) are treated in the first period of "
            "the panel; they carry no pre-period and are dropped from the "
            "simulation (every estimator would drop them too).",
            AssumptionWarning,
            stacklevel=3,
        )
        notes["n_always_treated_dropped"] = int(always.sum())
        uid_old = uid[keep_rows]
        keep_units = np.flatnonzero(~always)
        remap = -np.ones(n_units, dtype=int)
        remap[keep_units] = np.arange(keep_units.size)
        uid = np.asarray(remap[uid_old])
        tid = tid[keep_rows]
        df = df.loc[keep_rows]
        cohort_codes = cohort_codes[keep_units]
        n_units = int(keep_units.size)

    if n_units < 4:
        raise DataInsufficient(
            f"Only {n_units} usable units; a simulation over re-randomised "
            "adoption patterns needs at least 4.",
            diagnostics={"n_units": n_units},
        )
    if int((cohort_codes == 0).sum()) < 2:
        # Two, not one: every redrawn assignment has to leave a comparison
        # group behind (see _estimable), and the calibration's Y(0) model
        # needs untreated cells in the last period.  `calibrate='none'` does
        # not relax this -- the estimators need the controls either way.
        raise DataInsufficient(
            "Fewer than two never-treated units. Every redrawn adoption "
            "pattern needs a comparison group, and the calibration's "
            "untreated-potential-outcome model needs units that are "
            "untreated in the last period.",
            diagnostics={
                "n_units": n_units,
                "n_never_treated": int((cohort_codes == 0).sum()),
            },
        )
    if not (cohort_codes > 0).any():
        raise DataInsufficient(
            "No treated unit: there is no adoption pattern to re-randomise.",
            diagnostics={"n_units": n_units},
        )

    yv = df[y].to_numpy(dtype=float)
    balanced = bool(uid.size == n_units * n_times)
    panel = _Panel(
        y0=yv,
        fitted=np.zeros_like(yv),
        resid=yv.copy(),
        resid_mat=None,
        uid=uid,
        tid=tid,
        cohort=cohort_codes,
        n_units=n_units,
        n_times=n_times,
        balanced=balanced,
        notes=notes,
    )
    return panel, notes


def _two_way_fit(values: np.ndarray, uid: np.ndarray, tid: np.ndarray) -> np.ndarray:
    """Fitted values of an OLS of ``values`` on unit and period dummies."""
    out = np.asarray(values, dtype=float).copy()
    n_u = int(uid.max()) + 1
    n_t = int(tid.max()) + 1
    u_count = np.bincount(uid, minlength=n_u).astype(float)
    t_count = np.bincount(tid, minlength=n_t).astype(float)
    u_count[u_count == 0] = 1.0
    t_count[t_count == 0] = 1.0
    for _ in range(500):
        prev = out.copy()
        out -= (np.bincount(uid, weights=out, minlength=n_u) / u_count)[uid]
        out -= (np.bincount(tid, weights=out, minlength=n_t) / t_count)[tid]
        if np.max(np.abs(out - prev)) < 1e-13:
            break
    else:  # pragma: no cover - alternating projections on a connected panel
        raise ConvergenceFailure(
            "The two-way within transform did not converge in 500 sweeps.",
            diagnostics={"n_sweeps": 500},
        )
    fitted: np.ndarray = np.asarray(values, dtype=float) - out
    return fitted


def _calibrate(panel: _Panel, method: str) -> Dict[str, Any]:
    """Remove the estimated dynamic effect, in place on ``panel.y0``."""
    info: Dict[str, Any] = {"calibrate": method}
    g_row = panel.cohort[panel.uid]
    treated = (g_row > 0) & (panel.tid >= g_row)
    info["n_treated_cells_observed"] = int(treated.sum())
    if method == "none":
        info["removed_effect_by_horizon"] = {}
        return info

    from .did_imputation import _fit_untreated_twfe_sparse

    frame = pd.DataFrame(
        {_Y: panel.y0, "__uid": panel.uid, "__tid": panel.tid - 1},
    )
    try:
        y0_hat, _, _, _ = _fit_untreated_twfe_sparse(
            frame,
            frame.loc[~treated],
            _Y,
            None,
            "__uid",
            "__tid",
            panel.n_units,
            panel.n_times,
        )
    except ValueError as exc:  # no untreated obs for some unit or period
        raise DataInsufficient(
            "The imputation fit used to calibrate the panel could not "
            f"identify its fixed effects: {exc}. Pass `calibrate='none'` if "
            "the outcome already carries no treatment effect.",
            diagnostics={"n_treated_cells": int(treated.sum())},
        ) from exc

    tau_cell = panel.y0 - y0_hat
    horizon = np.where(treated, panel.tid - g_row, -1)
    removed = np.zeros_like(panel.y0)
    by_horizon: Dict[int, float] = {}
    for h in np.unique(horizon[treated]):
        sel = treated & (horizon == h)
        val = float(tau_cell[sel].mean())
        by_horizon[int(h)] = val
        removed[sel] = val
    panel.y0 = panel.y0 - removed
    info["removed_effect_by_horizon"] = by_horizon
    info["removed_effect_overall"] = (
        float(removed[treated].mean()) if treated.any() else 0.0
    )
    return info


def _decompose(panel: _Panel, resample: str) -> None:
    """Split the calibrated outcome into a two-way fit plus a residual."""
    if resample == "none":
        return
    panel.fitted = _two_way_fit(panel.y0, panel.uid, panel.tid)
    panel.resid = panel.y0 - panel.fitted
    if resample == "units":
        if not panel.balanced:
            raise MethodIncompatibility(
                "`resample='units'` draws whole unit residual paths, which "
                "needs a balanced panel. Use `resample='wild'` (a unit-level "
                "Rademacher multiplier, fine on unbalanced panels) or "
                "`resample='none'` (randomisation only).",
                diagnostics={
                    "n_rows": int(panel.uid.size),
                    "n_units_x_periods": panel.n_units * panel.n_times,
                },
            )
        mat = np.full((panel.n_units, panel.n_times), np.nan)
        mat[panel.uid, panel.tid - 1] = panel.resid
        panel.resid_mat = mat


def _arity(effect: Callable[..., float]) -> int:
    params = [
        p
        for p in inspect.signature(effect).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
    ]
    return len(params)


def _tau_values(
    effect: Any, g: np.ndarray, t: np.ndarray, treated: np.ndarray
) -> np.ndarray:
    """Injected effect per row (0 on untreated cells)."""
    if not callable(effect):
        return np.where(treated, float(effect), 0.0)
    n_args = _arity(effect)
    key = g.astype(np.int64) * (t.max() + 1) + t.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    vals = np.empty(uniq.size, dtype=float)
    for k, code in enumerate(uniq):
        gg = int(code // (t.max() + 1))
        tt = int(code % (t.max() + 1))
        vals[k] = float(effect(tt - gg) if n_args == 1 else effect(gg, tt))
    return np.where(treated, vals[inv], 0.0)


def _estimable(cohort: np.ndarray, n_times: int) -> bool:
    """Enough never-treated units, and a treated cohort with a pre-period."""
    if int((cohort == 0).sum()) < 2:
        return False
    treated = cohort[cohort > 0]
    return bool(treated.size >= 2 and np.any((treated >= 2) & (treated <= n_times)))


def _assign(panel: _Panel, opt: _Options, rng: np.random.Generator) -> np.ndarray:
    base = panel.cohort
    if opt.assignment == "observed":
        return base.copy()
    for _ in range(200):
        if opt.assignment == "resample_cohorts":
            draw = rng.permutation(base)
        else:
            draw = rng.choice(base, size=base.size, replace=True)
        if _estimable(draw, panel.n_times):
            return draw
    raise DataInsufficient(
        "200 draws of the adoption pattern in a row left no estimable "
        f"design (assignment='{opt.assignment}'). The observed cohort mix is "
        "too thin to re-randomise; try assignment='observed'.",
        diagnostics={"n_never_treated": int((base == 0).sum())},
    )


def _draw(panel: _Panel, opt: _Options, seed: int) -> Tuple[pd.DataFrame, float, bool]:
    """One simulated panel plus the true ATT injected into it."""
    rng = np.random.default_rng(seed)
    if opt.resample == "none":
        y0 = panel.y0
    elif opt.resample == "wild":
        sign = rng.choice([-1.0, 1.0], size=panel.n_units)
        y0 = panel.fitted + panel.resid * sign[panel.uid]
    else:
        donor = rng.integers(0, panel.n_units, size=panel.n_units)
        assert panel.resid_mat is not None  # guaranteed by _decompose
        y0 = panel.fitted + panel.resid_mat[donor[panel.uid], panel.tid - 1]

    cohort = _assign(panel, opt, rng)
    g = cohort[panel.uid]
    treated = (g > 0) & (panel.tid >= g)
    tau = _tau_values(opt.effect, g, panel.tid, treated)
    frame = pd.DataFrame(
        {
            _Y: y0 + tau,
            _ID: panel.uid,
            _T: panel.tid,
            _G: g,
            _D: treated.astype(float),
        }
    )
    truth = float(tau[treated].mean()) if treated.any() else float("nan")
    hetero = bool(np.ptp(tau[treated]) > 1e-12) if treated.any() else False
    return frame, truth, hetero


def _fit_one(frame: pd.DataFrame, opt: _Options) -> Dict[str, Dict[str, Any]]:
    """Run every requested estimator on one simulated panel."""
    out: Dict[str, Dict[str, Any]] = {}
    for name in opt.estimators:
        t0 = _time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                est, se = _ADAPTERS[name](frame, opt)
                # An exact fit leaves a standard error of zero up to
                # rounding: exactly 0 from one estimator, 1e-17 from
                # another, and which one depends on the BLAS. Anything
                # below 1e-10 of the outcome's size is that.
                se_floor = 1e-10 * float(np.max(np.abs(frame[_Y].to_numpy())))
                if not np.isfinite(est) or not np.isfinite(se) or se <= se_floor:
                    # Some estimators answer a degenerate design with a
                    # point estimate and no standard error. Scoring that as
                    # a draw would credit the estimator with a number it
                    # does not stand behind.
                    raise NumericalInstability(
                        f"unusable fit: estimate={est!r}, se={se!r} (the "
                        "design left this estimator without a standard "
                        "error on this draw)"
                    )
                row: Dict[str, Any] = {"estimate": est, "se": se, "error": None}
            except Exception as exc:  # recorded per draw, never swallowed
                row = {
                    "estimate": float("nan"),
                    "se": float("nan"),
                    "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                }
        row["n_warnings"] = len(caught)
        row["seconds"] = _time.perf_counter() - t0
        out[name] = row
    return out


def _run_draw(args: Tuple[_Panel, _Options, int]) -> Dict[str, Any]:
    """One replication, top-level so a process pool can pickle it."""
    panel, opt, seed = args
    frame, truth, hetero = _draw(panel, opt, seed)
    return {
        "seed": seed,
        "truth": truth,
        "heterogeneous": hetero,
        "fits": _fit_one(frame, opt),
    }


# ======================================================================
# Result
# ======================================================================


class DidSimulationStudy(ResultProtocolMixin):
    """Per-estimator bias, RMSE and coverage from a calibrated simulation.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per estimator: ``bias``, ``rmse``, ``coverage``,
        ``reject_rate``, ``se_ratio`` and their Monte Carlo standard errors.
    draws : pandas.DataFrame
        Every replication x estimator: ``estimate``, ``se``, ``truth``,
        ``error`` (the estimate minus the truth), ``covered``, ``rejected``.
    failures : pandas.DataFrame
        The draws where an estimator raised, with the exception text.
    model_info : dict
        The design that generated the draws, plus the effect removed by the
        calibration step.
    degradations : list of dict
        One entry per estimator that failed on at least one draw.
    """

    _citation_keys = ("ulloaperez2025comparative",)

    def __init__(
        self,
        table: pd.DataFrame,
        draws: pd.DataFrame,
        failures: pd.DataFrame,
        model_info: Dict[str, Any],
    ) -> None:
        self.table = table
        self.draws = draws
        self.failures = failures
        self.model_info = model_info
        self.degradations: List[Dict[str, Any]] = []

    @property
    def n_sims(self) -> int:
        return int(self.model_info["n_sims"])

    def best(self, criterion: str = "rmse") -> str:
        """Name of the estimator that wins on ``criterion``.

        ``'rmse'`` and ``'abs_bias'`` are minimised; ``'coverage'`` is scored
        by distance from the nominal level.  Estimators that failed on every
        draw are never returned.
        """
        usable = self.table[self.table["n_ok"] > 0]
        if usable.empty:
            raise DataInsufficient(
                "Every estimator failed on every draw; see `.failures`.",
                diagnostics={"n_sims": self.n_sims},
            )
        if criterion == "rmse":
            key = usable["rmse"]
        elif criterion == "abs_bias":
            key = usable["bias"].abs()
        elif criterion == "coverage":
            key = (usable["coverage"] - (1.0 - self.model_info["alpha"])).abs()
        else:
            raise MethodIncompatibility(
                "criterion must be 'rmse', 'abs_bias' or 'coverage'; got "
                f"{criterion!r}.",
                diagnostics={"criterion": criterion},
            )
        return str(usable.loc[key.idxmin(), "estimator"])

    def summary(self) -> str:
        info = self.model_info
        lines = [
            "Calibrated placebo simulation for DiD estimator selection",
            "=" * 60,
            f"panel        : {info['n_units']} units x {info['n_times']} "
            f"periods ({'balanced' if info['balanced'] else 'unbalanced'})",
            f"calibration  : {info['calibrate']}   resample: {info['resample']}",
            f"assignment   : {info['assignment']}   control group: "
            f"{info['control_group']}",
            f"injected ATT : {info['truth_mean']:.6g}"
            + ("  (heterogeneous)" if info["heterogeneous_effect"] else ""),
            f"replications : {info['n_sims']}   level: " f"{1 - info['alpha']:.0%}",
            "",
        ]
        cols = [
            "estimator",
            "bias",
            "mc_se_bias",
            "rmse",
            "sd_estimate",
            "mean_se",
            "se_ratio",
            "coverage",
            "reject_rate",
            "n_ok",
        ]
        show = self.table[cols].copy()
        for c in cols[1:-1]:
            show[c] = show[c].map(lambda v: "     .   " if pd.isna(v) else f"{v: .4f}")
        lines.append(show.to_string(index=False))
        lines.append("")
        if (self.table["n_ok"] > 0).any():
            lines.append(
                f"lowest RMSE  : {self.best('rmse')}   "
                f"coverage closest to nominal: {self.best('coverage')}"
            )
        if not self.failures.empty:
            counts = self.failures.groupby("estimator").size()
            lines.append(
                "failed draws : "
                + ", ".join(f"{k} x{v}" for k, v in counts.items())
                + "  (see `.failures`)"
            )
        if info["heterogeneous_effect"]:
            lines.append(
                "note         : the injected effect varies across cells, so "
                "part of any 'bias' is an estimand difference, not an error."
            )
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (
            f"<DidSimulationStudy: {len(self.table)} estimators, "
            f"{self.n_sims} draws>"
        )


# ======================================================================
# Public entry point
# ======================================================================


@accepts_aliases(_strict=True, unit="id", first_treat="cohort", g="cohort")
def did_calibrated_simulation(
    data: pd.DataFrame,
    y: str,
    id: str,
    time: str,
    cohort: str,
    *,
    estimators: Sequence[str] = _DEFAULT_ESTIMATORS,
    effect: Union[float, Callable[..., float]] = 0.0,
    assignment: str = "resample_cohorts",
    calibrate: str = "imputation",
    resample: str = "units",
    control_group: str = "nevertreated",
    n_sims: int = 200,
    alpha: float = 0.05,
    seed: Optional[int] = 0,
    n_jobs: int = 1,
) -> DidSimulationStudy:
    """Score DiD estimators on a placebo simulation calibrated to your panel.

    .. versionadded:: 1.30.0

    The observed panel is stripped of its estimated dynamic effect, a new
    adoption pattern is drawn, a known effect is injected, and every
    candidate estimator is refit — ``n_sims`` times.  Because the placebo
    assignment is random, parallel trends holds by construction and the
    resulting bias, RMSE and coverage are properties of the estimators on
    *this* panel rather than of an assumption.

    Parameters
    ----------
    data : pandas.DataFrame
        Long panel, one row per unit x period.
    y : str
        Outcome column.
    id : str
        Unit identifier. ``unit=`` is accepted as an alias.
    time : str
        Period column.
    cohort : str
        Period of first treatment, constant within a unit; ``0``, ``NaN`` or
        ``inf`` marks a never-treated unit. ``first_treat=`` and ``g=`` are
        accepted as aliases.
    estimators : sequence of str, default the six panel estimators
        Any of ``'twfe'``, ``'callaway_santanna'``, ``'sun_abraham'``,
        ``'did_imputation'``, ``'gardner_did'``, ``'etwfe'``,
        ``'did_multiplegt_dyn'``, ``'stacked_did'``, ``'lp_did'``.  The short
        spellings ``'cs'``, ``'sa'``, ``'bjs'``, ``'did2s'``, ``'dcdh'``,
        ``'stacked'`` and ``'lpdid'`` are accepted.
    effect : float or callable, default 0.0
        The effect injected into treated cells.  ``0.0`` measures size and
        coverage under a true null.  A callable of one argument is evaluated
        at the horizon ``t - g``; of two arguments, at ``(cohort, period)``
        in the panel's own period codes (``1 .. n_times``).
    assignment : {'resample_cohorts', 'random_timing', 'observed'}
        How the adoption pattern is redrawn.  ``'resample_cohorts'`` permutes
        the observed cohort labels across units, so the cohort-size
        distribution is preserved exactly; ``'random_timing'`` draws cohorts
        i.i.d. from that distribution; ``'observed'`` keeps the real pattern,
        which only makes sense with ``resample`` switched on.
    calibrate : {'imputation', 'none'}
        ``'imputation'`` subtracts the horizon-averaged imputation (BJS)
        effect from the treated cells, leaving a panel with zero average
        effect at every horizon.  ``'none'`` takes the outcome as already
        effect-free.
    resample : {'units', 'wild', 'none'}
        Where the outcome noise comes from.  ``'units'`` draws whole unit
        residual paths with replacement (a cluster bootstrap that preserves
        serial correlation; balanced panels only); ``'wild'`` multiplies each
        unit's residual path by a Rademacher draw; ``'none'`` holds the
        outcome fixed so the only randomness is the assignment.
    control_group : {'nevertreated', 'notyettreated'}
        Passed through to the estimators that take one. Sun-Abraham has no
        not-yet-treated comparison and uses the never-treated cohort under
        either setting.
    n_sims : int, default 200
        Replications.  Monte Carlo standard errors are reported so this can
        be chosen against the precision actually needed.
    alpha : float, default 0.05
        Level for coverage and the rejection rate.
    seed : int or None, default 0
        Base seed; draw ``k`` uses ``seed + k``.
    n_jobs : int, default 1
        Replications to run in parallel.  ``-1`` uses every CPU.  Workers are
        spawned rather than forked, so with ``n_jobs > 1`` the call must sit
        behind an ``if __name__ == '__main__':`` guard in a script, and a
        callable ``effect`` must be picklable (a module-level function, not a
        lambda).

    Returns
    -------
    DidSimulationStudy
        ``.table`` (one row per estimator), ``.draws``, ``.failures``,
        ``.summary()`` and ``.best(criterion)``.

    Notes
    -----
    The target is the overall ATT, defined as the average injected effect
    over treated observations.  With a constant ``effect`` every aggregation
    convention agrees; with a heterogeneous one, estimators targeting a
    different weighting show an estimand difference that this table reports
    as bias (``model_info['heterogeneous_effect']`` flags it).

    Covariates are deliberately absent: the placebo assignment is randomised,
    so unconditional parallel trends holds and there is no confounding for a
    covariate to remove.

    The calibrated null is zero only up to the estimation error of the effect
    that was removed.  ``calibrate='imputation'`` subtracts an *estimated*
    horizon effect, so the panel's true remaining effect is zero plus a term
    of order ``n**-0.5`` which every replication shares.  That term shifts all
    estimators in the same direction, so differences between estimators are
    measured more precisely than any single estimator's absolute bias.  Read a
    common bias smaller than a few Monte Carlo standard errors as this, not as
    a finding about the estimators.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(40):
    ...     g = [3, 4, 5, 0][i % 4]
    ...     a = rng.normal()
    ...     for t in range(1, 7):
    ...         d = g > 0 and t >= g
    ...         rows.append({"i": i, "t": t, "g": g,
    ...                      "y": a + 0.3 * t + 1.0 * d + rng.normal(0, 0.5)})
    >>> df = pd.DataFrame(rows)
    >>> study = sp.did_calibrated_simulation(
    ...     df, y="y", id="i", time="t", cohort="g",
    ...     estimators=["callaway_santanna", "did_imputation"],
    ...     effect=0.5, n_sims=5, seed=1,
    ... )
    >>> list(study.table["estimator"])
    ['callaway_santanna', 'did_imputation']
    >>> study.model_info["truth_mean"]
    0.5
    >>> bool((study.table["n_ok"] == 5).all())
    True
    >>> study.best("rmse") in {"callaway_santanna", "did_imputation"}
    True

    References
    ----------
    [@ulloaperez2025comparative] Ulloa-Perez, Bair, Navathe and Linn (2025),
        "Comparative Evaluation of Difference in Differences Methods for
        Staggered Adoption Interventions", arXiv:2508.14365.
    [@borusyak2024revisiting] Borusyak, Jaravel and Spiess (2024),
        "Revisiting Event-Study Designs: Robust and Efficient Estimation",
        *Review of Economic Studies*.
    """
    names = _resolve_estimators(estimators)
    for arg, allowed in (
        (assignment, _ASSIGNMENTS),
        (calibrate, _CALIBRATIONS),
        (resample, _RESAMPLES),
    ):
        if arg not in allowed:
            raise MethodIncompatibility(
                f"{arg!r} is not one of {allowed}.",
                diagnostics={"allowed": list(allowed)},
            )
    if control_group not in ("nevertreated", "notyettreated"):
        raise MethodIncompatibility(
            "control_group must be 'nevertreated' or 'notyettreated'; got "
            f"{control_group!r}.",
            diagnostics={"control_group": control_group},
        )
    if not (0.0 < float(alpha) < 1.0):
        raise MethodIncompatibility(
            f"alpha must lie strictly between 0 and 1; got {alpha!r}.",
            diagnostics={"alpha": alpha},
        )
    if not isinstance(n_sims, (int, np.integer)) or int(n_sims) < 2:
        raise MethodIncompatibility(
            f"n_sims must be an integer >= 2; got {n_sims!r}.",
            diagnostics={"n_sims": n_sims},
        )
    if assignment == "observed" and resample == "none":
        raise MethodIncompatibility(
            "assignment='observed' with resample='none' redraws nothing: "
            "every replication would be the same panel. Randomise the "
            "assignment, or resample the residuals.",
            diagnostics={"assignment": assignment, "resample": resample},
        )
    if callable(effect) and _arity(effect) not in (1, 2):
        raise MethodIncompatibility(
            "A callable `effect` must take either one argument (the horizon "
            "t - g) or two (cohort, period).",
            diagnostics={"n_args": _arity(effect)},
        )

    panel, notes = _prepare(data, y, id, time, cohort)
    cal = _calibrate(panel, calibrate)
    _decompose(panel, resample)

    opt = _Options(
        estimators=tuple(names),
        assignment=assignment,
        resample=resample,
        effect=effect,
        alpha=float(alpha),
        control_group=control_group,
        max_horizon=_max_horizon(panel),
    )

    base_seed = 0 if seed is None else int(seed)
    jobs = [(panel, opt, base_seed + k) for k in range(int(n_sims))]
    workers = _resolve_n_jobs(n_jobs)
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            results = list(pool.map(_run_draw, jobs))
    else:
        results = [_run_draw(job) for job in jobs]

    study = _assemble(results, opt, panel, cal, notes, float(alpha), int(n_sims))
    return study


def _max_horizon(panel: _Panel) -> int:
    """The largest horizon any observed cohort reaches, capped for cost.

    ``stacked_did`` and ``lp_did`` take an event window rather than a set of
    (g, t) cells; a window wider than the data supports leaves their pooled
    regression with a handful of observations.
    """
    treated = panel.cohort[panel.cohort > 0]
    reach = int(panel.n_times - treated.min()) if treated.size else 1
    return int(max(1, min(reach, 12)))


def _resolve_n_jobs(value: Any) -> int:
    import os

    if value is None:
        return 1
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or (value < 1 and value != -1)
    ):
        raise MethodIncompatibility(
            "n_jobs must be a positive integer or -1 (all CPUs).",
            diagnostics={"n_jobs": repr(value)},
        )
    if value == -1:
        return max(1, os.cpu_count() or 1)
    return int(value)


def _resolve_estimators(estimators: Sequence[str]) -> List[str]:
    if isinstance(estimators, str):
        estimators = [estimators]
    if not len(list(estimators)):
        raise MethodIncompatibility(
            "`estimators` is empty; name at least one.",
            diagnostics={"known": sorted(_ADAPTERS)},
        )
    out: List[str] = []
    for raw in estimators:
        name = _ESTIMATOR_ALIASES.get(str(raw), str(raw))
        if name not in _ADAPTERS:
            raise MethodIncompatibility(
                f"Unknown estimator {raw!r}.",
                diagnostics={
                    "known": sorted(_ADAPTERS),
                    "aliases": dict(_ESTIMATOR_ALIASES),
                },
            )
        if name not in out:
            out.append(name)
    return out


def _assemble(
    results: List[Dict[str, Any]],
    opt: _Options,
    panel: _Panel,
    cal: Dict[str, Any],
    notes: Dict[str, Any],
    alpha: float,
    n_sims: int,
) -> DidSimulationStudy:
    from scipy import stats

    crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    rows: List[Dict[str, Any]] = []
    for rep, res in enumerate(results):
        for name, fit in res["fits"].items():
            est, se = fit["estimate"], fit["se"]
            err = est - res["truth"]
            ok = np.isfinite(est) and np.isfinite(se) and se > 0
            rows.append(
                {
                    "sim": rep,
                    "seed": res["seed"],
                    "estimator": name,
                    "estimate": est,
                    "se": se,
                    "truth": res["truth"],
                    "error": err,
                    "covered": (abs(err) <= crit * se) if ok else np.nan,
                    "rejected": (abs(est / se) > crit) if ok else np.nan,
                    "n_warnings": fit["n_warnings"],
                    "seconds": fit["seconds"],
                    "failure": fit["error"],
                }
            )
    draws = pd.DataFrame(rows)
    failures = draws.loc[draws["failure"].notna(), ["sim", "estimator", "failure"]]

    summary: List[Dict[str, Any]] = []
    for name in opt.estimators:
        sub = draws[draws["estimator"] == name]
        ok = sub[sub["failure"].isna() & np.isfinite(sub["estimate"])]
        n_ok = int(len(ok))
        if n_ok == 0:
            summary.append(
                {
                    "estimator": name,
                    "n_ok": 0,
                    "n_failed": int(len(sub)),
                    "bias": np.nan,
                    "mc_se_bias": np.nan,
                    "rmse": np.nan,
                    "sd_estimate": np.nan,
                    "mean_se": np.nan,
                    "se_ratio": np.nan,
                    "coverage": np.nan,
                    "mc_se_coverage": np.nan,
                    "reject_rate": np.nan,
                    "mean_estimate": np.nan,
                    "seconds": float(sub["seconds"].sum()),
                    "n_warnings": int(sub["n_warnings"].sum()),
                }
            )
            continue
        err = ok["error"].to_numpy(dtype=float)
        est = ok["estimate"].to_numpy(dtype=float)
        se = ok["se"].to_numpy(dtype=float)
        cov = ok["covered"].dropna()
        rej = ok["rejected"].dropna()
        sd_est = float(np.std(est, ddof=1)) if n_ok > 1 else np.nan
        mean_se = float(np.mean(se))
        p = float(cov.mean()) if len(cov) else np.nan
        summary.append(
            {
                "estimator": name,
                "n_ok": n_ok,
                "n_failed": int(len(sub) - n_ok),
                "bias": float(np.mean(err)),
                "mc_se_bias": (
                    float(np.std(err, ddof=1) / np.sqrt(n_ok)) if n_ok > 1 else np.nan
                ),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "sd_estimate": sd_est,
                "mean_se": mean_se,
                "se_ratio": (
                    float(mean_se / sd_est)
                    if sd_est and np.isfinite(sd_est) and sd_est > 0
                    else np.nan
                ),
                "coverage": p,
                "mc_se_coverage": (
                    float(np.sqrt(p * (1 - p) / len(cov)))
                    if len(cov) and np.isfinite(p)
                    else np.nan
                ),
                "reject_rate": float(rej.mean()) if len(rej) else np.nan,
                "mean_estimate": float(np.mean(est)),
                "seconds": float(sub["seconds"].sum()),
                "n_warnings": int(sub["n_warnings"].sum()),
            }
        )
    table = pd.DataFrame(summary)

    truths = np.array([r["truth"] for r in results], dtype=float)
    model_info: Dict[str, Any] = {
        "n_sims": n_sims,
        "alpha": alpha,
        "assignment": opt.assignment,
        "calibrate": cal["calibrate"],
        "resample": opt.resample,
        "control_group": opt.control_group,
        "estimators": list(opt.estimators),
        "n_units": panel.n_units,
        "n_times": panel.n_times,
        "balanced": panel.balanced,
        "truth_mean": float(np.nanmean(truths)),
        "truth_sd": float(np.nanstd(truths, ddof=1)) if truths.size > 1 else 0.0,
        "heterogeneous_effect": bool(any(r["heterogeneous"] for r in results)),
        "max_horizon": opt.max_horizon,
        "removed_effect_by_horizon": cal.get("removed_effect_by_horizon", {}),
        "removed_effect_overall": cal.get("removed_effect_overall", 0.0),
        "n_treated_cells_observed": cal.get("n_treated_cells_observed", 0),
    }
    model_info.update(notes)

    study = DidSimulationStudy(table, draws, failures, model_info)
    for name, group in failures.groupby("estimator"):
        record_degradation(
            study,
            section=f"estimator:{name}",
            exc=RuntimeError(str(group["failure"].iloc[0])),
            detail=(
                f"{len(group)} of {n_sims} replications failed; the reported "
                "moments condition on the draws that succeeded."
            ),
        )
    return study
