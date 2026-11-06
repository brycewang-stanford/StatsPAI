"""``sp.audit`` checks for causal forests.

A fitted :class:`~statspai.forest.causal_forest.CausalForest` carries no
``method`` / ``model_info``, so the audit used to route it to the
regression pool and ask about robust standard errors and Oster bounds,
none of which apply to a forest. The questions a referee asks of a causal
forest are different: is there overlap in the forest's own propensities,
does the best-linear-predictor calibration test (``grf::test_calibration``)
say the mean prediction is right and the heterogeneity real, were the
trees honest, and are the training-sample statistics out-of-bag.

Everything here reads the fitted forest. The calibration test is the one
quantity not stored at fit time; it is a no-intercept regression on the
forest's out-of-bag predictions (:func:`statspai.calibration_test`), so it
is computed here without refitting anything.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: The checks evaluate ``model_info``-shaped evidence built by
#: :func:`forest_evidence`; their names are the keys below.
FAMILY = "forest"


def is_causal_forest(result: Any) -> bool:
    """True for a fitted ``sp.causal_forest`` (``CausalForest``)."""
    from ..forest.causal_forest import CausalForest

    return isinstance(result, CausalForest) and bool(
        getattr(result, "fitted_", False) or getattr(result, "_forest", None)
    )


def forest_evidence(result: Any) -> Dict[str, Any]:
    """The audit's evidence for a fitted causal forest, as a flat dict.

    Keys: ``nuisance_overlap`` (the fit-time record), ``honest_splitting``,
    ``calibration`` (from :func:`statspai.calibration_test`) and
    ``oob_predictions``. A diagnostic that does not exist for the fit
    records why under ``_not_applicable``.
    """
    from ..exceptions import StatsPAIError
    from ..forest.forest_inference import calibration_test

    diag = dict(getattr(result, "diagnostics", None) or {})
    ev: Dict[str, Any] = {"model_type": "causal_forest", "_not_applicable": {}}
    na: Dict[str, str] = ev["_not_applicable"]

    overlap = diag.get("nuisance_overlap") or {}
    if overlap.get("applicable"):
        ev["nuisance_overlap"] = dict(overlap)
    else:
        na["overlap"] = str(
            overlap.get("reason") or "the forest recorded no propensity overlap"
        )

    honest = bool(getattr(result, "honest", False))
    ev["honest_splitting"] = {
        "honest": honest,
        "honesty_fraction": (
            float(getattr(result, "honesty_fraction", 0.0) or 0.0) if honest else 0.0
        ),
    }

    try:
        table = calibration_test(result)
    except (StatsPAIError, ValueError, TypeError, RuntimeError) as exc:
        reason = f"calibration test unavailable for this fit: {exc}"
        na["calibration_mean"] = reason
        na["heterogeneity_detected"] = reason
    else:
        mean = table.loc["mean_forest_prediction"]
        diff = table.loc["differential_forest_prediction"]
        half = 0.5 * float(mean["ci_high"] - mean["ci_low"])
        ev["calibration"] = {
            "beta_mean": float(mean["coef"]),
            "beta_mean_ci": (float(mean["ci_low"]), float(mean["ci_high"])),
            # |beta_mean - 1| in units of the CI half-width: < 1 means the
            # 95% interval covers 1 (the mean prediction is calibrated)
            "mean_distance_from_one": (
                abs(float(mean["coef"]) - 1.0) / half if half > 0 else float("inf")
            ),
            "beta_differential": float(diff["coef"]),
            "p_differential": float(diff["p"]),
        }

    n_no_oob = diag.get("n_rows_without_oob_prediction")
    if n_no_oob is not None:
        ev["oob_predictions"] = {"n_rows_without_oob_prediction": int(n_no_oob)}
    return ev


def not_applicable(name: str, model_info: Dict[str, Any]) -> Optional[str]:
    """Reason a forest check does not exist for this fit, else ``None``."""
    return (model_info.get("_not_applicable") or {}).get(name)


def overlap_max_share() -> float:
    from ..forest.causal_forest import CausalForest

    return float(CausalForest.NUISANCE_OVERLAP_MAX_SHARE)
