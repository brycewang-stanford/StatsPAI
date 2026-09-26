r"""Synthetic control on Kaplan-Meier survival curves (cloglog scale).

Estimates the survival difference caused by treatment for a single treated
unit by building an Abadie-Diamond-Hainmueller-style synthetic control on
the complementary log-log (log-cumulative-hazard) scale:

1. Transform each unit's survival curve :math:`S_i(t)` to
   :math:`L_i(t) = \log(-\log S_i(t))` (values clipped to
   ``[1e-6, 1 - 1e-6]`` first).
2. Solve a nonnegative-weight, unit-simplex least-squares fit (exactly,
   by the shared active-set solver) that matches the treated unit's
   pre-treatment :math:`L_1(t)` by a convex combination of the donor
   :math:`L_j(t)`.
3. Apply the weights to the donors' post-treatment :math:`L_j(t)` and invert
   the link to obtain the counterfactual survival curve
   :math:`\hat S_1^{(0)}(t)`.
4. Report the gap :math:`S_1(t) - \hat S_1^{(0)}(t)` with a pointwise
   placebo band.

.. note::
   This is **not** the Synthetic Survival Control estimator of Han & Shah
   (arXiv:2511.14133, ``han2025synthetic``), which forms the counterfactual
   survival curve directly on the survival scale with unconstrained
   principal-component-regression weights. No public implementation of
   either estimator was found to compare against; the numbers here are
   checked by reference-free identities only.

References
----------
[@abadie2010synthetic]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin

__all__ = ["synth_survival", "SyntheticSurvivalResult"]


@dataclass
class SyntheticSurvivalResult(ResultProtocolMixin):
    """Output of :func:`synth_survival`.

    Exposes the fitted counterfactual survival curve (``s_synth``), the
    observed treated curve (``s_treated``), their gap trajectory (``gap``),
    the donor ``weights``, and a pointwise placebo band.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> months = np.arange(0, 13)
    >>> rows = []
    >>> for arm in ['treated_arm', 'ctrl_1', 'ctrl_2', 'ctrl_3']:
    ...     bump = 0.03 if arm == 'treated_arm' else 0.0
    ...     for m in months:
    ...         haz = 0.06 - bump * (m >= 6)
    ...         rows.append({'arm': arm, 'month': m,
    ...                      'km': float(np.exp(-haz * m)),
    ...                      'is_treated': arm == 'treated_arm'})
    >>> panel = pd.DataFrame(rows)
    >>> r = sp.synth_survival(
    ...     panel, unit='arm', time='month', survival='km',
    ...     treated='is_treated', treat_time=6, n_placebos=20, seed=0,
    ... )
    >>> isinstance(r, sp.SyntheticSurvivalResult)
    True
    >>> r.treated_unit
    'treated_arm'
    >>> int(len(r.time_grid))
    13
    """

    _citation_keys = ("abadie2010synthetic",)
    treated_unit: str
    time_grid: np.ndarray
    s_treated: np.ndarray
    s_synth: np.ndarray
    gap: np.ndarray
    weights: Dict[str, float]
    treat_time: float
    alpha: float
    ci_low: Optional[np.ndarray] = None
    ci_high: Optional[np.ndarray] = None
    pre_rmse: Optional[float] = None
    placebo_gaps: Optional[np.ndarray] = None

    def summary(self) -> str:
        post_mask = self.time_grid >= self.treat_time
        avg_gap = float(np.mean(self.gap[post_mask]))
        rows = [
            "Synthetic control on survival curves (cloglog scale)",
            "=" * 42,
            f"  Treated unit      : {self.treated_unit}",
            f"  Treatment time    : {self.treat_time}",
            f"  N grid points     : {len(self.time_grid)}",
            (
                f"  Pre-treat RMSE    : {self.pre_rmse:.4f}"
                if self.pre_rmse is not None
                else ""
            ),
            f"  Mean post-gap S(t): {avg_gap:+.4f}",
            "  Top-5 donor weights:",
        ]
        top = sorted(self.weights.items(), key=lambda kv: -abs(kv[1]))[:5]
        for name, w in top:
            rows.append(f"    {name:<20s} {w:.4f}")
        return "\n".join([r for r in rows if r])


def _cloglog(S: np.ndarray) -> np.ndarray:
    """Complementary log-log of a survival curve."""
    S = np.clip(S, 1e-6, 1 - 1e-6)
    return np.asarray(np.log(-np.log(S)))


def _inv_cloglog(L: np.ndarray) -> np.ndarray:
    """Inverse: ``exp(-exp(L))``."""
    return np.asarray(np.exp(-np.exp(np.clip(L, -50, 50))))


def _simplex_ls(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Minimise ||Y - X w||^2 over the unit simplex (exact).

    Delegates to the shared SCM solver. The former exponentiated-gradient
    loop (2000 steps, decaying rate) stopped short of the optimum, e.g.
    weights off by 0.03 and a 7% larger SSR on a 12-period, 8-donor panel.
    """
    from ._core import solve_simplex_weights

    return np.asarray(solve_simplex_weights(Y, X))


@accepts_aliases(id="unit")
def synth_survival(
    data: pd.DataFrame,
    unit: str,
    time: str,
    survival: str,
    treated: str,
    treat_time: float,
    alpha: float = 0.05,
    n_placebos: int = 100,
    seed: int = 0,
) -> SyntheticSurvivalResult:
    """Synthetic Survival Control estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long panel: one row per (unit, time) with a precomputed Kaplan-Meier
        survival probability in column ``survival``.  Each unit should have
        the *same* time grid (or be padded by forward/back-fill before
        calling — ragged grids are not accepted).
    unit : str
        Unit (panel-id) column.
    time : str
        Time grid column.
    survival : str
        Column containing the survival probability :math:`S_i(t)`
        (in :math:`(0,1)`).
    treated : str
        Column containing the name of the single treated unit.  Accepts
        either a boolean column or a dedicated string/int identifier.
    treat_time : float
        Time at which treatment starts (times >= ``treat_time`` are the
        post-treatment window).
    alpha : float, default 0.05
        Level of the pointwise placebo band.
    n_placebos : int, default 100
        Number of donors used as in-space placebos (a random subset when
        smaller than the donor pool; all donors otherwise). The band is
        *pointwise*: at each time the effect interval is
        ``[gap - q_{1-alpha/2}, gap - q_{alpha/2}]`` with ``q`` the placebo
        gaps' quantiles. It is not a uniform band.
    seed : int, default 0

    Returns
    -------
    SyntheticSurvivalResult
        Fitted counterfactual survival curve, gap trajectory, donor
        weights, and a pointwise placebo band.

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = [("treated_arm", m, float(np.exp(-0.04 * m)))
    ...         for m in range(12)]
    >>> for c in range(6):                       # six donor arms
    ...     rows += [(f"control_{c}", m, float(np.exp(-(0.05 + 0.01 * c) * m)))
    ...              for m in range(12)]
    >>> df = pd.DataFrame(rows, columns=["trial_arm", "month", "km_est"])
    >>> r = sp.synth_survival(
    ...     df, unit="trial_arm", time="month",
    ...     survival="km_est", treated="treated_arm", treat_time=6,
    ... )
    >>> _ = r.summary()
    """
    for col in [unit, time, survival]:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data")

    df = data.copy()

    # Detect treated unit
    if treated in df.columns:
        if df[treated].dtype == bool:
            treated_units = df.loc[df[treated], unit].unique()
        else:
            treated_units = df.loc[df[treated].astype(bool), unit].unique()
        if len(treated_units) != 1:
            raise ValueError(
                "Exactly one treated unit expected; got " f"{len(treated_units)}."
            )
        treated_unit = str(treated_units[0])
    else:
        # Interpret ``treated`` as the explicit unit name
        treated_unit = str(treated)

    wide = df.pivot(index=time, columns=unit, values=survival).sort_index()
    time_grid = wide.index.to_numpy(dtype=float)
    if treated_unit not in wide.columns:
        raise ValueError(
            f"Treated unit '{treated_unit}' not found among units: "
            f"{list(wide.columns)}"
        )

    # Transform to complementary log-log scale
    L_wide = wide.apply(_cloglog)
    donors = [c for c in wide.columns if c != treated_unit]
    if not donors:
        from statspai.exceptions import DataInsufficient

        raise DataInsufficient(
            "At least one donor unit required",
            recovery_hint=(
                "synth_survival needs at least one untreated (donor) unit. "
                "Check the unit / treatment_unit columns."
            ),
            diagnostics={"n_donors": 0},
            alternative_functions=[],
        )

    pre_mask = time_grid < treat_time
    if pre_mask.sum() < 2:
        raise ValueError(
            "At least two pre-treatment time points required "
            f"(got {int(pre_mask.sum())})"
        )

    Y_pre = L_wide[treated_unit].to_numpy()[pre_mask]
    X_pre = L_wide[donors].to_numpy()[pre_mask]
    weights = _simplex_ls(Y_pre, X_pre)

    L_synth = L_wide[donors].to_numpy() @ weights
    s_synth = _inv_cloglog(L_synth)
    s_treated = wide[treated_unit].to_numpy()
    gap = s_treated - s_synth
    pre_rmse = float(np.sqrt(np.mean((Y_pre - X_pre @ weights) ** 2)))

    # --- Pointwise placebo band via in-space placebos ------------- #
    rng = np.random.default_rng(seed)
    placebo_gaps = []
    candidate_donors = donors.copy()
    n_sample = min(n_placebos, len(candidate_donors))
    if n_sample > 0 and len(candidate_donors) > 1:
        chosen = rng.choice(candidate_donors, size=n_sample, replace=False)
        for placebo in chosen:
            others = [d for d in candidate_donors if d != placebo]
            Y_pre_p = L_wide[placebo].to_numpy()[pre_mask]
            X_pre_p = L_wide[others].to_numpy()[pre_mask]
            w_p = _simplex_ls(Y_pre_p, X_pre_p)
            L_synth_p = L_wide[others].to_numpy() @ w_p
            s_synth_p = _inv_cloglog(L_synth_p)
            placebo_gaps.append(wide[placebo].to_numpy() - s_synth_p)
    placebo_gaps_arr = np.asarray(placebo_gaps) if placebo_gaps else None

    if placebo_gaps_arr is not None and len(placebo_gaps_arr) >= 2:
        q_low = np.quantile(placebo_gaps_arr, alpha / 2, axis=0)
        q_high = np.quantile(placebo_gaps_arr, 1 - alpha / 2, axis=0)
        # Invert "gap - effect ~ placebo-gap distribution": the effect lies
        # in [gap - q_high, gap - q_low]. (The old gap + q_low / gap + q_high
        # was only right for a placebo distribution symmetric about 0.)
        ci_low = gap - q_high
        ci_high = gap - q_low
    else:
        ci_low = ci_high = None

    return SyntheticSurvivalResult(
        treated_unit=treated_unit,
        time_grid=time_grid,
        s_treated=s_treated,
        s_synth=s_synth,
        gap=gap,
        weights=dict(zip(donors, weights.tolist())),
        treat_time=float(treat_time),
        alpha=alpha,
        ci_low=ci_low,
        ci_high=ci_high,
        pre_rmse=pre_rmse,
        placebo_gaps=placebo_gaps_arr,
    )
