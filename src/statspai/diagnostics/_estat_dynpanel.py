"""``sp.estat`` handlers for dynamic-panel GMM fits.

Stata splits dynamic-panel postestimation across ``estat abond`` (the
Arellano-Bond serial-correlation tests), ``estat sargan`` (the
over-identification test) and, in ``xtabond2``, an always-printed
difference-in-Hansen block. ``sp.xtabond`` computes all three during the
fit — they are cheap and nobody should have to opt in to the tests that
decide whether the design holds — so these handlers *present* what is
already in ``model_info`` rather than recomputing anything.

That is deliberate: a postestimation command that re-derives a statistic
can silently disagree with the fit it is reporting on.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

__all__ = [
    "is_dynamic_panel_result",
    "estat_abond",
    "estat_sargan",
    "estat_difference_in_hansen",
]

_DYNPANEL_KEYS = ("ar1_z", "ar2_z", "n_instruments")


def is_dynamic_panel_result(result: Any) -> bool:
    """True when ``result`` came from ``sp.xtabond`` / ``sp.xtdpdsys``."""
    info = getattr(result, "model_info", None)
    if not isinstance(info, dict):
        return False
    return all(key in info for key in _DYNPANEL_KEYS)


def _require(result: Any, what: str) -> Dict[str, Any]:
    if not is_dynamic_panel_result(result):
        raise ValueError(
            f"estat '{what}' applies to dynamic-panel GMM fits "
            "(sp.xtabond / sp.xtdpdsys / sp.panel(method='ab'|'system'|'ah')); "
            f"got a {type(result).__name__} without the dynamic-panel "
            "diagnostics in model_info."
        )
    return result.model_info


def estat_abond(result: Any, alpha: float = 0.05) -> Dict[str, Any]:
    """Arellano-Bond test for serial correlation in the differenced errors.

    AR(1) is *expected* to reject: first-differencing induces MA(1) by
    construction, so failing to reject there is the surprising outcome.
    AR(2) rejecting is the damaging one — it means the level errors are
    serially correlated, which invalidates ``y_{t-2}`` as an instrument.
    The fix is a deeper lag window (``gmm_lags=(3, None)``), not a louder
    caveat.
    """
    info = _require(result, "abond")
    rows: List[Dict[str, Any]] = []
    for order in (1, 2):
        z = float(info.get(f"ar{order}_z", np.nan))
        p = float(info.get(f"ar{order}_p", np.nan))
        rows.append(
            {
                "order": order,
                "z": z,
                "pvalue": p,
                "reject": bool(np.isfinite(p) and p < alpha),
            }
        )
    ar2 = rows[1]
    if not np.isfinite(ar2["pvalue"]):
        verdict = "AR(2) could not be computed (too few usable period pairs)."
    elif ar2["reject"]:
        verdict = (
            f"AR(2) rejects at {alpha:.0%} (p = {ar2['pvalue']:.4f}): the level "
            "errors look serially correlated, so lag-2 instruments are not "
            "exogenous. Deepen gmm_lags."
        )
    else:
        verdict = (
            f"AR(2) does not reject (p = {ar2['pvalue']:.4f}): consistent with "
            "the instrument set."
        )
    return {
        "test": "abond",
        "label": "Arellano-Bond test for serial correlation",
        "rows": rows,
        "interpretation": verdict,
    }


def estat_sargan(result: Any, alpha: float = 0.05) -> Dict[str, Any]:
    """Over-identification: the Sargan statistic and the Hansen J.

    Both are reported because they answer the same question under
    different assumptions and can disagree: Sargan assumes homoskedastic
    errors, the Hansen J does not. Where they diverge, believe the J.

    A p-value close to 1 is not reassurance — with many instruments the J
    test loses power and is driven toward 1 regardless of validity, which
    is why the instrument count is reported alongside it.
    """
    info = _require(result, "sargan")
    n_instruments = int(info.get("n_instruments", 0))
    n_units = int(info.get("n_units", 0))
    rows = []
    for name, stat_key, df_key, p_key in (
        ("Sargan", "sargan_stat", "sargan_df", "sargan_p"),
        ("Hansen J", "hansen_stat", "hansen_df", "hansen_p"),
    ):
        stat = float(info.get(stat_key, np.nan))
        p = float(info.get(p_key, np.nan))
        rows.append(
            {
                "name": name,
                "statistic": stat,
                "df": int(info.get(df_key, 0)),
                "pvalue": p,
                "reject": bool(np.isfinite(p) and p < alpha),
                "robust_to_heteroskedasticity": name == "Hansen J",
            }
        )
    hansen_p = rows[1]["pvalue"]
    notes = []
    if np.isfinite(hansen_p) and hansen_p > 0.9 and n_units and n_instruments:
        notes.append(
            f"Hansen p = {hansen_p:.3f} with {n_instruments} instruments for "
            f"{n_units} units. A p-value this high is weak evidence of "
            "validity, not strong evidence — try collapse=True and check "
            "whether it survives."
        )
    if n_units and n_instruments >= n_units:
        notes.append(
            f"{n_instruments} instruments for {n_units} units: the "
            "over-identification test is unreliable at this ratio."
        )
    return {
        "test": "sargan",
        "label": "Over-identification tests",
        "rows": rows,
        "n_instruments": n_instruments,
        "n_units": n_units,
        "interpretation": " ".join(notes) if notes else "",
    }


def estat_difference_in_hansen(result: Any, alpha: float = 0.05) -> Dict[str, Any]:
    """Difference-in-Hansen (C) tests of instrument subsets.

    The only way to interrogate the *extra* assumptions a moment set brings
    — above all system GMM's level moments, which require each unit's
    deviation from its long-run mean to be uncorrelated with the fixed
    effect. Reporting a system-GMM estimate without this is reporting an
    untested identifying assumption.
    """
    info = _require(result, "difhansen")
    groups = info.get("difference_in_hansen") or {}
    rows = []
    for name, payload in groups.items():
        p = float(payload.get("pvalue", np.nan))
        rows.append(
            {
                "subset": name,
                "hansen_excluding": float(payload.get("hansen_excluding", np.nan)),
                "df_excluding": int(payload.get("df_excluding", 0)),
                "statistic": float(payload.get("statistic", np.nan)),
                "df": int(payload.get("df", 0)),
                "pvalue": p,
                "reject": bool(np.isfinite(p) and p < alpha),
                "note": payload.get("note", ""),
            }
        )
    rejected = [r["subset"] for r in rows if r["reject"]]
    return {
        "test": "difhansen",
        "label": "Difference-in-Hansen tests of instrument subsets",
        "rows": rows,
        "interpretation": (
            "Exogeneity rejected for: " + ", ".join(rejected)
            if rejected
            else "No instrument subset is rejected at the chosen level."
        ),
    }
