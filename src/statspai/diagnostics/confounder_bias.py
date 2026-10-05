"""Simple bias formulas for one unmeasured confounder.

Two questions about a reported effect and a hypothetical confounder that
was not adjusted for:

- :func:`confounder_adjust`: *if* the confounder has this association
  with the exposure and this effect on the outcome, what would the
  estimate have been had it been adjusted for?
- :func:`confounder_tip`: how strong would the confounder have to be to
  move the estimate (or a confidence limit) to the null?

They are the array formulas of Schlesselman (1978) and Lin, Psaty and
Kronmal (1998) for a binary or a normally distributed confounder that
does not modify the exposure effect, in the form the R package ``tipr``
uses. They complement the tools that need no such specification:
:func:`sp.evalue` (the minimum strength on the risk-ratio scale) and
:func:`sp.sensemakr` (partial R-squared).

References
----------
[@lin1998assessing] [@schlesselman1978assessing] [@mcgowan2022tipr]
[@vanderweele2017sensitivity]
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

_MEASURES = ("coef", "rr", "or", "hr")


def _ratio_scale(values: Any, measure: str, rare_outcome: bool) -> Any:
    """Approximate risk ratio behind an odds or hazard ratio.

    With a rare outcome both are close to the risk ratio. With a common
    outcome VanderWeele (2017) recommends ``sqrt(OR)`` and
    ``(1 - 0.5 ** sqrt(HR)) / (1 - 0.5 ** sqrt(1 / HR))``.
    """
    if values is None or rare_outcome or measure in ("coef", "rr"):
        return values
    v = np.asarray(values, dtype=float)
    if measure == "or":
        return np.sqrt(v)
    return (1 - 0.5 ** np.sqrt(v)) / (1 - 0.5 ** np.sqrt(1 / v))


def _vec(x: Any, name: str) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.atleast_1d(np.asarray(x, dtype=float))
    if not np.isfinite(arr).all():
        raise MethodIncompatibility(f"{name} must be finite")
    return arr


def _check(
    measure: str,
    effect: np.ndarray,
    outcome_effect: Optional[np.ndarray],
    exposed_prev: Optional[np.ndarray],
    unexposed_prev: Optional[np.ndarray],
    exposure_confounder_effect: Optional[np.ndarray],
) -> bool:
    """Validate the inputs; return True for a binary confounder."""
    if measure not in _MEASURES:
        raise MethodIncompatibility(
            f"measure must be one of {_MEASURES}; got {measure!r}"
        )
    binary = exposed_prev is not None or unexposed_prev is not None
    if binary and exposure_confounder_effect is not None:
        raise MethodIncompatibility(
            "Describe the confounder either by its prevalence in the two "
            "exposure groups (binary) or by exposure_confounder_effect "
            "(a mean difference; normal confounder), not both.",
        )
    for name, p in (("exposed_prev", exposed_prev), ("unexposed_prev", unexposed_prev)):
        if p is not None and ((p < 0) | (p > 1)).any():
            raise MethodIncompatibility(
                f"{name} is a prevalence and must lie in [0, 1]"
            )
    if measure != "coef":
        if (effect <= 0).any():
            raise MethodIncompatibility(f"a {measure.upper()} must be positive")
        if outcome_effect is not None and (outcome_effect <= 0).any():
            raise MethodIncompatibility(
                "confounder_outcome_effect is a ratio on this scale and must "
                "be positive"
            )
    return binary


def confounder_adjust(
    effect: Any,
    *,
    confounder_outcome_effect: Any,
    exposure_confounder_effect: Any = None,
    exposed_prev: Any = None,
    unexposed_prev: Any = None,
    measure: str = "coef",
    rare_outcome: bool = True,
) -> pd.DataFrame:
    """Adjust an observed effect for a specified unmeasured confounder.

    Parameters
    ----------
    effect : float or array-like
        The observed estimate. Pass the point estimate and both
        confidence limits together to see all three move.
    confounder_outcome_effect : float or array-like
        Effect of the confounder on the outcome, on the scale of
        ``measure``: a coefficient (per unit of the confounder) for
        ``'coef'``, a ratio otherwise.
    exposure_confounder_effect : float or array-like, optional
        Normal confounder with unit variance: the difference in its mean
        between the exposed and the unexposed.
    exposed_prev, unexposed_prev : float or array-like, optional
        Binary confounder: its prevalence among the exposed and among the
        unexposed. Give both, instead of ``exposure_confounder_effect``.
    measure : {'coef', 'rr', 'or', 'hr'}, default 'coef'
        ``'coef'``: a difference (linear-model coefficient, risk
        difference, difference in means). ``'rr'``, ``'or'``, ``'hr'``:
        risk, odds or hazard ratio.
    rare_outcome : bool, default True
        Odds and hazard ratios only. ``False`` converts the observed
        ratio and ``confounder_outcome_effect`` to approximate risk
        ratios first (``sqrt`` for an odds ratio; VanderWeele's formula
        for a hazard ratio), which is advisable when the outcome is
        common (above roughly 15%). The output is then on the risk-ratio
        scale and ``effect_observed`` shows the converted value.

    Returns
    -------
    pandas.DataFrame
        One row per combination of inputs (arrays are broadcast), with
        ``effect_adjusted``, ``effect_observed`` and the confounder
        specification used.

    Notes
    -----
    With ``d`` the mean difference of the confounder (``exposed_prev -
    unexposed_prev`` when binary) and ``g`` its effect on the outcome:

    - difference scale: ``adjusted = observed - g * d``;
    - ratio scale, normal confounder: ``adjusted = observed / g ** d``;
    - ratio scale, binary confounder:
      ``adjusted = observed * (1 + (g - 1) p0) / (1 + (g - 1) p1)``.

    The formulas assume one confounder, independent of the measured
    covariates given the exposure, whose effect on the outcome is the
    same in both exposure groups.

    Examples
    --------
    An observed coefficient of 6.58; a confounder one unit of which lowers
    the outcome by 2.3 and whose mean is 0.17 lower among the exposed:

    >>> import statspai as sp
    >>> out = sp.confounder_adjust(6.58, confounder_outcome_effect=-2.3,
    ...                            exposure_confounder_effect=-0.17)
    >>> round(float(out['effect_adjusted'].iloc[0]), 3)
    6.189

    A risk ratio of 1.5 and a binary confounder (40% vs 10%) that raises
    risk by 80%:

    >>> out = sp.confounder_adjust(1.5, confounder_outcome_effect=1.8,
    ...                            exposed_prev=0.4, unexposed_prev=0.1,
    ...                            measure='rr')
    >>> round(float(out['effect_adjusted'].iloc[0]), 4)
    1.2273

    References
    ----------
    [@lin1998assessing] [@schlesselman1978assessing] [@mcgowan2022tipr]
    """
    measure = str(measure).lower()
    b = _vec(effect, "effect")
    g = _vec(confounder_outcome_effect, "confounder_outcome_effect")
    d = _vec(exposure_confounder_effect, "exposure_confounder_effect")
    p1 = _vec(exposed_prev, "exposed_prev")
    p0 = _vec(unexposed_prev, "unexposed_prev")
    assert b is not None
    if g is None:
        raise MethodIncompatibility("confounder_outcome_effect is required")
    binary = _check(measure, b, g, p1, p0, d)
    if binary and (p1 is None or p0 is None):
        raise MethodIncompatibility(
            "a binary confounder needs exposed_prev and unexposed_prev"
        )
    if not binary and d is None:
        raise MethodIncompatibility(
            "give exposure_confounder_effect (normal confounder) or "
            "exposed_prev and unexposed_prev (binary confounder)"
        )
    b = np.asarray(_ratio_scale(b, measure, rare_outcome), dtype=float)
    g = np.asarray(_ratio_scale(g, measure, rare_outcome), dtype=float)

    cols = {}
    if binary:
        assert p1 is not None and p0 is not None
        b, g, p1, p0 = np.broadcast_arrays(b, g, p1, p0)
        if measure == "coef":
            adjusted = b - g * (p1 - p0)
        else:
            adjusted = b * (1 + (g - 1) * p0) / (1 + (g - 1) * p1)
        cols = {"exposed_prev": p1, "unexposed_prev": p0}
    else:
        assert d is not None
        b, g, d = np.broadcast_arrays(b, g, d)
        adjusted = b - g * d if measure == "coef" else b / g**d
        cols = {"exposure_confounder_effect": d}
    return pd.DataFrame(
        {
            "effect_adjusted": adjusted,
            "effect_observed": b,
            **cols,
            "confounder_outcome_effect": g,
        }
    )


def confounder_tip(
    effect: Any,
    *,
    confounder_outcome_effect: Any = None,
    exposure_confounder_effect: Any = None,
    exposed_prev: Any = None,
    unexposed_prev: Any = None,
    measure: str = "coef",
    rare_outcome: bool = True,
) -> pd.DataFrame:
    """Unmeasured confounder that would move an effect to the null.

    Leave exactly one property of the confounder unspecified and it is
    solved for; specify all of them and the result is how many
    independent confounders of that kind it would take.

    Parameters
    ----------
    effect : float or array-like
        The observed estimate to tip. To ask what would make a result
        "not significant", pass the confidence limit closest to the null
        rather than the point estimate.
    confounder_outcome_effect : float or array-like, optional
        Effect of the confounder on the outcome on the scale of
        ``measure``.
    exposure_confounder_effect : float or array-like, optional
        Normal confounder with unit variance: difference in its mean
        between the exposed and the unexposed.
    exposed_prev, unexposed_prev : float or array-like, optional
        Binary confounder: prevalence among the exposed and unexposed.
    measure : {'coef', 'rr', 'or', 'hr'}, default 'coef'
        Scale of ``effect``; the null is 0 for ``'coef'`` and 1 for the
        ratios.
    rare_outcome : bool, default True
        See :func:`confounder_adjust`.

    Returns
    -------
    pandas.DataFrame
        ``effect_observed``, the confounder specification with the solved
        value filled in, and ``n_unmeasured_confounders`` (1 when a
        property was solved for). A solved prevalence outside [0, 1]
        means no binary confounder with the stated properties can tip the
        effect; it is returned as NaN with a warning.

    Examples
    --------
    How large a mean difference would a confounder with a -7 effect on
    the outcome need, to explain away a coefficient of 6.58?

    >>> import statspai as sp
    >>> tip = sp.confounder_tip(6.58, confounder_outcome_effect=-7)
    >>> round(float(tip['exposure_confounder_effect'].iloc[0]), 2)
    -0.94

    How strongly would a binary confounder (50% vs 10%) have to raise
    risk to explain away a risk ratio of 1.2?

    >>> tip = sp.confounder_tip(1.2, exposed_prev=0.5, unexposed_prev=0.1,
    ...                         measure='rr')
    >>> round(float(tip['confounder_outcome_effect'].iloc[0]), 3)
    1.526

    References
    ----------
    [@lin1998assessing] [@mcgowan2022tipr] [@vanderweele2017sensitivity]
    """
    measure = str(measure).lower()
    b = _vec(effect, "effect")
    g = _vec(confounder_outcome_effect, "confounder_outcome_effect")
    d = _vec(exposure_confounder_effect, "exposure_confounder_effect")
    p1 = _vec(exposed_prev, "exposed_prev")
    p0 = _vec(unexposed_prev, "unexposed_prev")
    assert b is not None
    binary = _check(measure, b, g, p1, p0, d)
    b = np.asarray(_ratio_scale(b, measure, rare_outcome), dtype=float)
    if g is not None:
        g = np.asarray(_ratio_scale(g, measure, rare_outcome), dtype=float)
    n_conf: Any = 1.0

    with np.errstate(divide="ignore", invalid="ignore"):
        if binary:
            n_missing = sum(v is None for v in (g, p1, p0))
            if n_missing > 1:
                raise MethodIncompatibility(
                    "a binary confounder has three properties (exposed_prev, "
                    "unexposed_prev, confounder_outcome_effect); leave at most "
                    "one of them out"
                )
            if g is None:
                assert p1 is not None and p0 is not None
                b, p1, p0 = np.broadcast_arrays(b, p1, p0)
                if measure == "coef":
                    g = b / (p1 - p0)
                else:
                    g = 1 + (b - 1) / (p1 - b * p0)
                    g = np.where(g > 0, g, np.nan)
            elif p1 is None:
                assert p0 is not None
                b, g, p0 = np.broadcast_arrays(b, g, p0)
                if measure == "coef":
                    p1 = p0 + b / g
                else:
                    p1 = (b * (1 + (g - 1) * p0) - 1) / (g - 1)
            elif p0 is None:
                b, g, p1 = np.broadcast_arrays(b, g, p1)
                if measure == "coef":
                    p0 = p1 - b / g
                else:
                    p0 = ((1 + (g - 1) * p1) / b - 1) / (g - 1)
            else:
                b, g, p1, p0 = np.broadcast_arrays(b, g, p1, p0)
                if measure == "coef":
                    n_conf = b / (g * (p1 - p0))
                else:
                    n_conf = np.log(b) / np.log((1 + (g - 1) * p1) / (1 + (g - 1) * p0))
            p1 = np.asarray(p1, dtype=float)
            p0 = np.asarray(p0, dtype=float)
            bad = (
                ~np.isfinite(p1) | ~np.isfinite(p0) | (p1 < 0) | (p1 > 1)
                | (p0 < 0) | (p0 > 1) | ~np.isfinite(np.asarray(g, dtype=float))
            )  # fmt: skip
            if bad.any():
                warnings.warn(
                    "confounder_tip: no binary confounder with the stated "
                    "properties can move the effect to the null (the solved "
                    "value is not a valid prevalence or ratio); returning NaN "
                    "for those rows.",
                    stacklevel=2,
                )
                p1 = np.where(bad, np.nan, p1) if exposed_prev is None else p1
                p0 = np.where(bad, np.nan, p0) if unexposed_prev is None else p0
                if confounder_outcome_effect is None:
                    g = np.where(bad, np.nan, g)
            cols = {"exposed_prev": p1, "unexposed_prev": p0}
        else:
            if g is None and d is None:
                raise MethodIncompatibility(
                    "give confounder_outcome_effect, exposure_confounder_effect "
                    "(or the two prevalences of a binary confounder), or both"
                )
            if g is None:
                assert d is not None
                b, d = np.broadcast_arrays(b, d)
                g = b / d if measure == "coef" else b ** (1 / d)
            elif d is None:
                b, g = np.broadcast_arrays(b, g)
                d = b / g if measure == "coef" else np.log(b) / np.log(g)
            else:
                b, g, d = np.broadcast_arrays(b, g, d)
                n_conf = (
                    b / (g * d) if measure == "coef" else np.log(b) / (d * np.log(g))
                )
            cols = {"exposure_confounder_effect": d}

    n_conf = np.broadcast_to(np.asarray(n_conf, dtype=float), np.shape(b)).copy()
    if (n_conf < 0).any():
        warnings.warn(
            "confounder_tip: a confounder with these properties moves the "
            "effect away from the null, so no number of them tips it; "
            "n_unmeasured_confounders is set to 0 for those rows.",
            stacklevel=2,
        )
        n_conf = np.where(n_conf < 0, 0.0, n_conf)
    return pd.DataFrame(
        {
            "effect_observed": b,
            **cols,
            "confounder_outcome_effect": g,
            "n_unmeasured_confounders": n_conf,
        }
    )
