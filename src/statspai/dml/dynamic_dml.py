r"""Dynamic Double/Debiased ML: heterogeneous effects of a treatment sequence.

:func:`statspai.dml_panel` assumes one homogeneous effect and a treatment
that does not move the future state. When treatment is assigned repeatedly
and *changes the confounders that drive later treatment*, that model is not
merely inefficient, it is wrong: controlling for the later state blocks the
indirect path the treatment worked through, and not controlling for it
leaves the confounding in. Robins's g-methods
(:func:`statspai.msm`, :func:`statspai.gformula`,
:func:`statspai.ltmle`) solve the population-average version;
[lewis2021double] solve it in the Neyman-orthogonal, cross-fitted way that
also gives heterogeneity and admits machine-learnt nuisances.

Estimand
--------
With ``m`` periods per unit and the outcome read at the end,

.. math::
    Y_i = \sum_{t=0}^{m-1} \theta_t(X_i)\, T_{it} + \text{(state)} + \epsilon_i

where :math:`\theta_t` is the effect of *intervening* on the treatment at
period ``t``, holding the rest of the sequence at its intervened value and
letting the state evolve. It therefore contains the indirect path through
later states, which is what a policy maker who sets the whole sequence
cares about. ``theta.sum()`` is the effect of treating in every period
rather than none.

Identification (sequential ignorability)
----------------------------------------
``T_t`` must be as good as random given the recorded state at ``t``. The
state has to include **the treatment history**. On the simulation the
tests use, dropping the one lagged treatment moved the three period
effects by -15%, -15% and +37%, and not one interval covered the truth:
``T_t`` depends on ``T_{t-1}``, which predicts the later state, so the
residualisation is incomplete without it. ``lags=1`` (the default) adds
it for you; ``lags=0`` opts out, warns, and says so in ``diagnostics``.

Estimator
---------
Stage one, cross-fitted by unit: for every period ``t`` regress the final
outcome and every present-or-future treatment ``T_j`` (``j >= t``) on the
period-``t`` state, and keep the residuals ``R_t`` and ``B_{t,j}``.

Stage two: the moment conditions are triangular,

.. math::
    \mathbb{E}\Bigl[B_{t,t}\bigl(R_t - \sum_{j \ge t}\theta_j B_{t,j}\bigr)
    \Bigr] = 0, \qquad t = m-1, \dots, 0

which [lewis2021double] solve by back-substitution, one period at a time.
Solving the stacked system instead gives the same estimate and, for free,
the **joint** sandwich covariance -- so the standard error of a sequence
effect accounts for the correlation between periods rather than adding
variances. On the test design that is the difference between 0.044 and
0.073: the blips are negatively correlated, and the total is estimated
more precisely than any single period of it: 0.032 against 0.062.

Parity
------
Given the same folds and the same first-stage learners, this reproduces
``econml.panel.dml.DynamicDML`` [econml] -- the reference implementation,
written by the authors of the method -- to 1e-15 relative on both the
per-period estimates and their standard errors
(``tests/reference_parity/test_dynamic_dml_econml_parity.py``).

References
----------
[lewis2021double], [econml]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._input_validation import missing_named_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
)

__all__ = ["dynamic_dml", "DynamicDMLResult"]


@dataclass
class DynamicDMLResult(ResultProtocolMixin):
    """Output of :func:`dynamic_dml`.

    Attributes
    ----------
    periods : pd.DataFrame
        One row per period, indexed by the period label, with ``lag``
        (periods before the outcome), ``estimate``, ``se``, ``z``, ``p``,
        ``ci_low`` and ``ci_high``.
    estimate, se, ci_lower, ci_upper, p_value : float
        The total sequence effect -- treating in every period rather than
        none -- and its inference from the joint covariance.
    vcov : np.ndarray
        Joint ``(m, m)`` covariance of the per-period effects, the thing
        that makes any contrast across periods reportable.
    coef : pd.DataFrame or None
        Heterogeneity coefficients when ``modifiers=`` was given: one block
        per period, the linear projection of that period's effect on the
        modifiers.
    n_units, n_periods, n_folds : int
    period_labels : list
    lags : int
    method : str
        Always ``"dynamic_dml"``.
    diagnostics : dict
        ``first_stage_r2_outcome`` / ``first_stage_r2_treatment`` per
        period, ``state_columns``, ``lagged_treatment_included``,
        ``n_units_dropped``, ``independent_sum_se``, ``modifiers``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> rows = []
    >>> for i in range(250):
    ...     state, prev = rng.normal(), 0.0
    ...     hist = []
    ...     for period in range(3):
    ...         if period:
    ...             state = 0.6 * state + 0.7 * prev + rng.normal()
    ...         dose = 0.8 * state + 0.5 * prev + rng.normal()
    ...         hist.append((state, dose))
    ...         prev = dose
    ...     out = sum(0.5 * s + 0.4 * d for s, d in hist) + rng.normal()
    ...     for period, (s, d) in enumerate(hist):
    ...         rows.append({"id": i, "t": period, "y": out, "d": d, "w": s})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.dynamic_dml(df, y="y", treat="d", id="id", time="t",
    ...                      covariates=["w"], n_folds=3)
    >>> list(res.periods.columns)
    ['lag', 'estimate', 'se', 'z', 'p', 'ci_low', 'ci_high']
    >>> res.vcov.shape
    (3, 3)
    >>> # Any contrast uses the joint covariance, so summing periods does
    >>> # not overstate the uncertainty.
    >>> bool(res.se < res.diagnostics["independent_sum_se"])
    True
    >>> sorted(res.contrast([1, 1, 1]))
    ['ci_high', 'ci_low', 'estimate', 'p', 'se', 'z']
    >>> res.cumulative().shape
    (3, 7)
    """

    periods: pd.DataFrame
    estimate: float
    se: float
    ci_lower: float
    ci_upper: float
    p_value: float
    z_stat: float
    vcov: np.ndarray
    n_units: int
    n_periods: int
    n_folds: int
    period_labels: List[Any]
    lags: int
    alpha: float
    coef: Optional[pd.DataFrame] = None
    method: str = "dynamic_dml"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def contrast(self, weights: Any) -> Dict[str, float]:
        """Inference for ``w' theta``, using the joint covariance.

        ``weights=[1, 1, 1]`` is the total sequence effect;
        ``[0, 0, 1]`` the last period alone; ``[1, -1, 0]`` asks whether
        the first two periods differ.
        """
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != self.n_periods:
            raise MethodIncompatibility(
                f"contrast(): need one weight per period ({self.n_periods}), "
                f"got {w.size}.",
                recovery_hint="Pass a weight for every period in .period_labels.",
            )
        est = float(w @ self.periods["estimate"].to_numpy())
        var = float(w @ self.vcov @ w)
        se = float(np.sqrt(max(var, 0.0)))
        z = float(stats.norm.ppf(1 - self.alpha / 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            zstat = est / se if se > 0 else np.nan
        return {
            "estimate": est,
            "se": se,
            "ci_low": est - z * se,
            "ci_high": est + z * se,
            "z": zstat,
            "p": float(2 * stats.norm.sf(abs(zstat))) if se > 0 else float("nan"),
        }

    def cumulative(self) -> pd.DataFrame:
        """Effect of treating from each period to the end, with joint SEs."""
        rows = []
        for k in range(self.n_periods):
            w = np.zeros(self.n_periods)
            w[k:] = 1.0
            out = self.contrast(w)
            out["from_period"] = self.period_labels[k]
            out["n_periods_treated"] = self.n_periods - k
            rows.append(out)
        return pd.DataFrame(rows).set_index("from_period")

    def summary(self) -> str:
        lines = [
            "Dynamic Double/Debiased ML (Lewis & Syrgkanis 2021)",
            "=" * 62,
            f"  n units      : {self.n_units}",
            f"  n periods    : {self.n_periods}",
            f"  n folds      : {self.n_folds}   (split by unit)",
            f"  lagged treat.: {self.lags} lag(s) in the state",
            "",
            "  Effect of the treatment at each period on the final outcome",
            "  (intervening on the sequence; includes the path through later",
            "   states)",
            "",
        ]
        tab = self.periods
        lines.append(
            f"    {'period':>10}  {'lag':>3}  {'estimate':>10}  {'se':>8}"
            f"  {'95% CI':>22}"
        )
        for label, row in tab.iterrows():
            ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]"
            lines.append(
                f"    {str(label):>10}  {int(row['lag']):>3}  "
                f"{row['estimate']:>+10.4f}  {row['se']:>8.4f}  {ci:>22}"
            )
        lines += [
            "",
            f"  Total (treat every period) : {self.estimate:+.4f}"
            f"   SE = {self.se:.4f}",
            f"    95% CI                   : [{self.ci_lower:+.4f}, "
            f"{self.ci_upper:+.4f}]   p = {self.p_value:.4g}",
            "    (from the joint covariance, not a sum of variances)",
        ]
        if self.coef is not None:
            lines += ["", "  Heterogeneity (linear projection on modifiers):", ""]
            lines.append(self.coef.round(4).to_string())
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  Panel reshaping
# --------------------------------------------------------------------------- #


def _as_list(value: Any, name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    out = list(value)
    if not all(isinstance(c, str) for c in out):
        raise MethodIncompatibility(
            f"dynamic_dml(): {name} must be a column name or a list of them.",
            recovery_hint=f"Pass {name}=['col1', 'col2'].",
        )
    return out


def _wide_panel(
    data: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    needed: Sequence[str],
    periods: Optional[Sequence[Any]],
) -> tuple:
    """Balanced wide arrays, or a message saying exactly what is missing."""
    missing = [c for c in [y, treat, unit, time, *needed] if c not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            data.columns,
            context="dynamic_dml()",
            extra_diagnostics={"missing": missing},
        )
    frame = data.loc[:, [unit, time, y, treat, *dict.fromkeys(needed)]].copy()
    labels = sorted(frame[time].unique()) if periods is None else list(periods)
    frame = frame[frame[time].isin(labels)]
    if len(labels) < 2:
        raise DataInsufficient(
            "dynamic_dml(): needs at least two periods; a single period is "
            "sp.dml(model='plr').",
            recovery_hint="Use a panel with repeated treatment assignments.",
            diagnostics={"n_periods": len(labels)},
        )
    if frame.duplicated([unit, time]).any():
        n_dup = int(frame.duplicated([unit, time]).sum())
        raise MethodIncompatibility(
            f"dynamic_dml(): {n_dup} duplicated unit-period row(s); the "
            "estimator needs one observation per unit and period.",
            recovery_hint="Aggregate to one row per unit-period first.",
            diagnostics={"n_duplicated_rows": n_dup},
        )
    counts = frame.groupby(unit)[time].nunique()
    complete = counts[counts == len(labels)].index
    dropped = int(len(counts) - len(complete))
    if len(complete) == 0:
        raise DataInsufficient(
            f"dynamic_dml(): no unit is observed in all {len(labels)} "
            "periods, so no treatment sequence is complete.",
            recovery_hint=(
                "Restrict to a balanced window with periods=[...], or use "
                "sp.msm / sp.ltmle, which tolerate unbalanced histories."
            ),
            diagnostics={"n_periods": len(labels), "n_units": int(len(counts))},
        )
    frame = frame[frame[unit].isin(complete)]
    has_na = frame.isna().any(axis=1)
    if has_na.any():
        bad_units = frame.loc[has_na, unit].unique()
        frame = frame[~frame[unit].isin(bad_units)]
        dropped += len(bad_units)
        if frame.empty:
            raise DataInsufficient(
                "dynamic_dml(): every unit has a missing value somewhere in "
                "its sequence.",
                recovery_hint="Impute or drop the offending columns.",
            )
    frame = frame.sort_values([unit, time], kind="mergesort")
    units = frame[unit].to_numpy()
    n_units = int(pd.unique(units).size)
    m = len(labels)
    order = {lab: k for k, lab in enumerate(labels)}
    rank = frame[time].map(order).to_numpy()
    if not np.array_equal(
        rank.reshape(n_units, m), np.tile(np.arange(m), (n_units, 1))
    ):
        raise MethodIncompatibility(  # pragma: no cover - guarded by the above
            "dynamic_dml(): the panel did not reshape to one row per unit and "
            "period after balancing.",
            recovery_hint="Report this with a minimal example.",
        )

    def wide(col: str) -> np.ndarray:
        values: np.ndarray = frame[col].to_numpy(dtype=float)
        return values.reshape(n_units, m)

    return frame, labels, n_units, m, wide, dropped


# --------------------------------------------------------------------------- #
#  Estimator
# --------------------------------------------------------------------------- #


def _default_learner() -> Any:
    """Ridge with a small internal CV: linear nuisances are the common case
    in panels of this size, and an unregularised fit on a wide state is the
    quickest way to make the residuals useless."""
    from sklearn.linear_model import RidgeCV

    return RidgeCV(alphas=np.logspace(-3, 3, 13))


def _fit_predict(
    model: Any, state: np.ndarray, target: np.ndarray, folds: np.ndarray
) -> np.ndarray:
    """Cross-fitted out-of-fold prediction; folds are unit-level already."""
    from sklearn.base import clone

    pred = np.empty(len(target), dtype=float)
    scored = np.zeros(len(target), dtype=bool)
    for f in np.unique(folds):
        train, test = folds != f, folds == f
        fitted = clone(model).fit(state[train], target[train])
        pred[test] = np.asarray(fitted.predict(state[test]), dtype=float).ravel()
        scored[test] = True
    if not scored.all():  # pragma: no cover - every fold index is covered
        raise DataInsufficient("dynamic_dml(): a fold produced no predictions.")
    return pred


def _r2(target: np.ndarray, pred: np.ndarray) -> float:
    resid = target - pred
    denom = float(np.sum((target - target.mean()) ** 2))
    return float(1.0 - np.sum(resid**2) / denom) if denom > 0 else float("nan")


@accepts_aliases(_strict=True, unit="id", seed="random_state")
def dynamic_dml(
    data: pd.DataFrame,
    y: str,
    treat: str,
    *,
    id: str,
    time: str,
    covariates: Optional[Sequence[str]] = None,
    baseline: Optional[Sequence[str]] = None,
    modifiers: Optional[Sequence[str]] = None,
    lags: int = 1,
    model_y: Optional[Any] = None,
    model_t: Optional[Any] = None,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 0,
    periods: Optional[Sequence[Any]] = None,
    fold_ids: Optional[Any] = None,
) -> DynamicDMLResult:
    r"""Effect of a treatment *sequence*, period by period [lewis2021double].

    Use this when treatment is assigned repeatedly and moves the state that
    drives later treatment. Controlling for the later state blocks the
    indirect path; not controlling for it leaves the confounding in. The
    estimator residualises each period against the state *as of that
    period* and solves the resulting triangular moment conditions, so
    ``theta_t`` is the effect of intervening on period ``t``'s treatment
    with the rest of the sequence held at its intervened value -- the
    indirect path through later states included.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel, one row per unit-period. Units not observed in every
        period are dropped and counted in ``diagnostics``.
    y : str
        Outcome. Only its value in the **last** period is used: the
        estimand is the effect of the whole sequence on the final outcome.
    treat : str
        Treatment, continuous or binary.
    id, time : str
        Panel keys; ``unit=`` is accepted as an alias of ``id``, the
        spelling :func:`statspai.dml_panel` uses. ``time`` is sorted; pass
        ``periods=`` to choose a window or to fix a non-sortable order.
    covariates : list of str, optional
        Time-varying state, read at each period. This is where the
        confounders that the treatment itself moves belong.
    baseline : list of str, optional
        Time-invariant controls, added to every period's state (their
        first-period values are used).
    modifiers : list of str, optional
        Effect modifiers for heterogeneity, read at the **first** period
        (they are pre-treatment by construction). The final stage becomes
        linear in them and ``result.coef`` holds the projection.
    lags : int, default 1
        Lagged treatments added to each period's state. **The default is
        not cosmetic**: sequential ignorability needs the treatment
        history, and on the simulation behind the tests dropping the one
        lag moved the three period effects by -15%, -15% and +37% with no
        interval covering the truth. ``lags=0`` opts out, warns, and
        records it in ``diagnostics``.
    model_y, model_t : estimator, optional
        First-stage learners (scikit-learn API). Default
        ``RidgeCV(alphas=logspace(-3, 3, 13))``.
    n_folds : int, default 5
        Cross-fitting folds. Whole units are held out, never rows.
    alpha : float, default 0.05
    random_state : int, default 0
        Seeds the fold assignment.
    periods : sequence, optional
        The periods to use, in order. Defaults to every sorted value of
        ``time``.
    fold_ids : array-like, optional
        One fold index per unit (in sorted unit order), to reproduce an
        external split -- this is how the parity test matches ``econml``.

    Returns
    -------
    DynamicDMLResult
        ``.periods`` per-period effects, ``.estimate`` the total sequence
        effect, ``.vcov`` the joint covariance, ``.contrast(w)`` any
        linear combination, ``.cumulative()`` the treat-from-here-on
        profile.

    Notes
    -----
    Point estimates and standard errors reproduce
    ``econml.panel.dml.DynamicDML`` [econml] to 1e-15 relative given the
    same folds and learners; the joint covariance is the addition, and it
    is what makes ``.contrast`` and the total effect honest: on the test
    design the total's standard error is 0.032, against 0.062 from adding
    the per-period variances, because the period effects are negatively
    correlated. ``diagnostics['independent_sum_se']`` reports the wrong
    number alongside the right one so the gap is visible.

    Effects are identified only under sequential ignorability: given the
    recorded state, this period's treatment is as good as random. No test
    can confirm that; what the function can do is stop you from omitting
    the treatment history, which is the part people forget.

    See Also
    --------
    statspai.dml_panel : one homogeneous effect, no dynamic confounding.
    statspai.msm : population-average effect of a static regime via IPTW.
    statspai.ltmle : longitudinal TMLE for regime contrasts.

    References
    ----------
    [@lewis2021double], [@econml]

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(300):
    ...     w, t_prev = rng.normal(), 0.0
    ...     hist = []
    ...     for period in range(3):
    ...         if period:
    ...             w = 0.6 * w + 0.7 * t_prev + rng.normal()
    ...         d = 0.8 * w + 0.5 * t_prev + rng.normal()
    ...         hist.append((w, d))
    ...         t_prev = d
    ...     outcome = sum(0.5 * w_ + 0.4 * d_ for w_, d_ in hist) + rng.normal()
    ...     for period, (w_, d_) in enumerate(hist):
    ...         rows.append({"id": i, "t": period, "y": outcome,
    ...                      "d": d_, "w": w_})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.dynamic_dml(df, y="y", treat="d", id="id", time="t",
    ...                      covariates=["w"], n_folds=3)
    >>> res.periods.shape
    (3, 7)
    >>> bool(res.se > 0)
    True
    """
    cov_cols = _as_list(covariates, "covariates")
    base_cols = _as_list(baseline, "baseline")
    mod_cols = _as_list(modifiers, "modifiers")
    if not isinstance(lags, (int, np.integer)) or isinstance(lags, bool) or lags < 0:
        raise MethodIncompatibility(
            "dynamic_dml(): lags must be a non-negative integer.",
            recovery_hint="Use lags=1 (the default) unless you have a reason.",
            diagnostics={"lags": lags},
        )
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            "dynamic_dml(): alpha must be in (0, 1).", recovery_hint="Use alpha=0.05."
        )
    named = missing_named_columns(
        {
            "y": y,
            "treat": treat,
            "id": id,
            "time": time,
            "covariates": cov_cols,
            "baseline": base_cols,
            "modifiers": mod_cols,
        },
        data.columns,
    )
    if named:
        raise ColumnNotFound.from_missing(
            named,
            data.columns,
            context="dynamic_dml()",
            extra_diagnostics={"missing": list(named.values())},
        )
    frame, labels, n_units, m, wide, dropped = _wide_panel(
        data, y, treat, id, time, [*cov_cols, *base_cols, *mod_cols], periods
    )
    if lags >= m:
        raise MethodIncompatibility(
            f"dynamic_dml(): lags={lags} needs more than {m} periods.",
            recovery_hint=f"Use lags <= {m - 1}.",
            diagnostics={"lags": lags, "n_periods": m},
        )
    if n_units < n_folds:
        raise DataInsufficient(
            f"dynamic_dml(): {n_units} complete unit(s) cannot fill "
            f"{n_folds} folds.",
            recovery_hint="Lower n_folds, or widen the balanced window.",
            diagnostics={"n_units": n_units, "n_folds": n_folds},
        )
    if dropped:
        warnings.warn(
            f"dynamic_dml(): dropped {dropped} unit(s) that were not observed "
            f"in all {m} periods or had missing values. The estimand is the "
            "sequence effect among the units that stayed; if attrition is "
            "related to the treatment, that is a different population.",
            AssumptionWarning,
            stacklevel=2,
        )
    if lags == 0:
        warnings.warn(
            "dynamic_dml(lags=0): the treatment history is not in the state. "
            "Sequential ignorability then requires that covariates= already "
            "captures everything past treatment did. On the simulation "
            "behind the tests, omitting one lag moved the three period "
            "effects by -15%, -15% and +37%, and no interval covered the "
            "truth.",
            AssumptionWarning,
            stacklevel=2,
        )

    T_wide = wide(treat)
    Y_final = wide(y)[:, -1]
    state_parts: List[np.ndarray] = []
    state_names: List[List[str]] = []
    # Modifiers go into the state as well as the final stage: they drive the
    # outcome, so leaving them out of E[Y | state] throws away precision and
    # -- when the effect itself varies with them -- biases the period
    # estimates (measured at 3.2 standard errors on the test design before
    # this was fixed). ``econml``'s DynamicDML passes X to both stages too.
    for t in range(m):
        cols, names = [], []
        for c in cov_cols:
            cols.append(wide(c)[:, t])
            names.append(c)
        for c in dict.fromkeys([*base_cols, *mod_cols]):
            cols.append(wide(c)[:, 0])
            names.append(c)
        for k in range(1, lags + 1):
            cols.append(T_wide[:, t - k] if t - k >= 0 else np.zeros(n_units))
            names.append(f"{treat}_lag{k}")
        cols.append(np.ones(n_units))
        names.append("(intercept)")
        state_parts.append(np.column_stack(cols))
        state_names.append(names)

    if fold_ids is None:
        rng = np.random.default_rng(random_state)
        folds = rng.permutation(n_units) % int(n_folds)
    else:
        folds = np.asarray(fold_ids).ravel()
        if folds.size != n_units:
            raise MethodIncompatibility(
                f"dynamic_dml(): fold_ids must have one entry per complete "
                f"unit ({n_units}), got {folds.size}.",
                recovery_hint="Pass fold_ids aligned with sorted unit ids.",
            )
        n_folds = int(np.unique(folds).size)

    learner_y = _default_learner() if model_y is None else model_y
    learner_t = _default_learner() if model_t is None else model_t

    R = np.zeros((n_units, m))
    B = np.zeros((n_units, m, m))
    r2_y, r2_t = [], []
    for t in range(m):
        state = state_parts[t]
        pred = _fit_predict(learner_y, state, Y_final, folds)
        R[:, t] = Y_final - pred
        r2_y.append(_r2(Y_final, pred))
        for j in range(t, m):
            pred_t = _fit_predict(learner_t, state, T_wide[:, j], folds)
            B[:, t, j] = T_wide[:, j] - pred_t
            if j == t:
                r2_t.append(_r2(T_wide[:, j], pred_t))

    A = np.stack([B[:, t, t] for t in range(m)], axis=1)
    if mod_cols:
        Xmod = np.column_stack([np.ones(n_units)] + [wide(c)[:, 0] for c in mod_cols])
        mod_labels = ["(intercept)"] + list(mod_cols)
    else:
        Xmod = np.ones((n_units, 1))
        mod_labels = ["(intercept)"]
    k = Xmod.shape[1]

    # Stacked triangular moments, one block of k per period. Solving them
    # together rather than by back-substitution gives the same estimate and
    # the joint covariance with it.
    G = np.zeros((m * k, m * k))
    rhs = np.zeros(m * k)
    for t in range(m):
        inst = Xmod * A[:, [t]]  # (n, k)
        rhs[t * k : (t + 1) * k] = inst.T @ R[:, t] / n_units
        for j in range(t, m):
            reg = Xmod * B[:, t, [j]]
            G[t * k : (t + 1) * k, j * k : (j + 1) * k] = inst.T @ reg / n_units
    if np.linalg.matrix_rank(G) < G.shape[0]:
        raise DataInsufficient(
            "dynamic_dml(): the moment system is singular -- the residualised "
            "treatments carry no independent variation. Usually the state "
            "predicts the treatment perfectly (too many controls for the "
            "number of units) or a period has a constant treatment.",
            recovery_hint="Drop controls, or check treat for within-period variance.",
            diagnostics={"n_units": n_units, "n_moments": int(G.shape[0])},
        )
    beta = np.linalg.solve(G, rhs)
    scores = np.zeros((n_units, m * k))
    for t in range(m):
        inst = Xmod * A[:, [t]]
        fitted = np.zeros(n_units)
        for j in range(t, m):
            fitted += (Xmod @ beta[j * k : (j + 1) * k]) * B[:, t, j]
        scores[:, t * k : (t + 1) * k] = inst * (R[:, t] - fitted)[:, None]
    Ginv = np.linalg.inv(G)
    meat = scores.T @ scores / n_units
    V_full = Ginv @ meat @ Ginv.T / n_units

    # Per-period effect at the mean modifier values (the intercept block when
    # modifiers are centred is not the average effect, so evaluate at the mean).
    xbar = Xmod.mean(axis=0)
    L = np.zeros((m, m * k))
    for t in range(m):
        L[t, t * k : (t + 1) * k] = xbar
    theta = L @ beta
    V = L @ V_full @ L.T
    se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = np.where(se > 0, theta / se, np.nan)
    table = pd.DataFrame(
        {
            "lag": [m - 1 - t for t in range(m)],
            "estimate": theta,
            "se": se,
            "z": zstat,
            "p": 2 * stats.norm.sf(np.abs(zstat)),
            "ci_low": theta - z_crit * se,
            "ci_high": theta + z_crit * se,
        },
        index=pd.Index(labels, name="period"),
    )
    coef_table = None
    if mod_cols:
        coef_se = np.sqrt(np.clip(np.diag(V_full), 0.0, None))
        coef_table = pd.DataFrame(
            {
                "period": np.repeat(labels, k),
                "term": mod_labels * m,
                "estimate": beta,
                "se": coef_se,
            }
        ).set_index(["period", "term"])
        with np.errstate(divide="ignore", invalid="ignore"):
            cz = np.where(coef_se > 0, beta / coef_se, np.nan)
        coef_table["z"] = cz
        coef_table["p"] = 2 * stats.norm.sf(np.abs(cz))
        coef_table["ci_low"] = beta - z_crit * coef_se
        coef_table["ci_high"] = beta + z_crit * coef_se

    ones = np.ones(m)
    total = float(ones @ theta)
    total_se = float(np.sqrt(max(ones @ V @ ones, 0.0)))
    with np.errstate(divide="ignore", invalid="ignore"):
        total_z = total / total_se if total_se > 0 else float("nan")
    return DynamicDMLResult(
        periods=table,
        estimate=total,
        se=total_se,
        ci_lower=total - z_crit * total_se,
        ci_upper=total + z_crit * total_se,
        p_value=(
            float(2 * stats.norm.sf(abs(total_z))) if total_se > 0 else float("nan")
        ),
        z_stat=float(total_z),
        vcov=V,
        n_units=n_units,
        n_periods=m,
        n_folds=int(n_folds),
        period_labels=list(labels),
        lags=int(lags),
        alpha=alpha,
        coef=coef_table,
        diagnostics={
            "first_stage_r2_outcome": [float(v) for v in r2_y],
            "first_stage_r2_treatment": [float(v) for v in r2_t],
            "state_columns": state_names[-1],
            "lagged_treatment_included": bool(lags > 0),
            "n_units_dropped": int(dropped),
            "independent_sum_se": float(np.sqrt(np.sum(se**2))),
            "modifiers": list(mod_cols),
        },
    )
