"""Covariate benchmarks for the DML omitted-variable-bias bound.

A benchmark answers "how strong is a confounder like ``X_k``?" by leaving
``X_k`` out, refitting the same DML model on the same rows and splits, and
comparing the short fit with the long one (Chernozhukov, Cinelli, Newey,
Sharma and Syrgkanis 2022, section 6). The gain statistics follow the
definitions ``doubleml``'s ``sensitivity_benchmark`` uses:

* ``cf_y = (R2_y,long - R2_y,short) / (1 - R2_y,long)`` with
  ``R2_y = 1 - sigma^2 / Var(Y)``, the share of the outcome variation left
  by the short model that the benchmark covariates explain;
* ``cf_d = (1 - R2_riesz) / R2_riesz`` with ``R2_riesz = nu^2_short /
  nu^2_long``, the relative gain in the variation of the Riesz representer;
* ``delta_theta = theta_short - theta_long``;
* ``rho = sign(delta_theta) * min(1, |delta_theta| / sqrt((sigma^2_short -
  sigma^2_long) (nu^2_long - nu^2_short)))``, the degree of adversity.

Each is computed per cross-fitting repetition and combined by the median.

A residual-correlation shortcut cannot replace the refit: ``X_k`` is one of
the regressors behind the residuals, so its correlation with them is zero
by construction for a linear learner and close to zero for any learner
that fits well.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

_FOLD_COL = "__fold__"
_CLUSTER_COL = "__cl__"


class ShortRefit:
    """What a fitted DML model needs to be refitted without some covariates.

    Holds the estimator (its learners and options) and the complete rows it
    was fitted on, with the realised fold labels and cluster labels as
    columns, so the short fit uses the same rows and the same splits.
    """

    def __init__(self, estimator: Any, clean: pd.DataFrame) -> None:
        self.estimator = estimator
        self.frame = clean.reset_index(drop=True)

    def __repr__(self) -> str:
        # Stable text: result.to_dict() stringifies what it cannot encode,
        # and a default repr would put a memory address there.
        return f"ShortRefit({type(self.estimator).__name__}, n={len(self.frame)})"

    def fit(self, drop: Sequence[str]) -> Any:
        est = self.estimator
        keep = [c for c in est.covariates if c not in set(drop)]
        if not keep:
            raise MethodIncompatibility(
                "dml_sensitivity: the benchmark set is every covariate of the "
                "fit; the short model would have no covariates.",
                recovery_hint="Benchmark a proper subset of the covariates.",
            )
        short = type(est)(
            data=self.frame,
            y=est.y,
            treat=est.treat,
            covariates=keep,
            ml_g=est.ml_g,
            ml_m=est.ml_m,
            n_folds=est.n_folds,
            n_rep=est.n_rep,
            alpha=est.alpha,
            random_state=est.random_state,
            fold_indices=_FOLD_COL if _FOLD_COL in self.frame.columns else None,
            score=est.score,
            normalize_ipw=est.normalize_ipw,
            trimming_threshold=est.trimming_threshold,
            cluster=_CLUSTER_COL if _CLUSTER_COL in self.frame.columns else None,
        )
        return short.fit()


def _per_rep(reps: List[Dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([float(r[key]) for r in reps], dtype=float)


def gain_statistics(
    long_reps: List[Dict[str, Any]],
    short_reps: List[Dict[str, Any]],
    var_y: float,
) -> Dict[str, float]:
    """Benchmark ``cf_y``, ``cf_d``, ``rho`` and ``delta_theta``.

    ``long_reps`` / ``short_reps`` are the per-repetition score elements a
    DML fit stores (``theta``, ``sigma2``, ``nu2``); ``var_y`` is the
    variance of the outcome on the estimation rows (``ddof=0``).
    """
    if len(long_reps) != len(short_reps):
        raise MethodIncompatibility(  # pragma: no cover - same n_rep by construction
            "dml_sensitivity: long and short fits have different n_rep."
        )
    s2_long, s2_short = _per_rep(long_reps, "sigma2"), _per_rep(short_reps, "sigma2")
    nu2_long, nu2_short = _per_rep(long_reps, "nu2"), _per_rep(short_reps, "nu2")
    th_long, th_short = _per_rep(long_reps, "theta"), _per_rep(short_reps, "theta")

    r2_y_long = 1.0 - s2_long / var_y
    r2_y_short = 1.0 - s2_short / var_y
    r2_riesz = nu2_short / nu2_long
    cf_y = np.clip((r2_y_long - r2_y_short) / (1.0 - r2_y_long), 0.0, 1.0)
    cf_d = np.clip((1.0 - r2_riesz) / r2_riesz, 0.0, 1.0)

    delta = th_short - th_long
    var_g = s2_short - s2_long
    var_riesz = nu2_long - nu2_short
    ok = (var_g > 0) & (var_riesz > 0)
    denom = np.where(ok, np.sqrt(np.where(ok, var_g * var_riesz, 1.0)), 0.0)
    ratio = np.ones_like(delta)
    np.divide(np.abs(delta), denom, out=ratio, where=denom != 0)
    rho = np.sign(delta) * np.clip(ratio, 0.0, 1.0)
    return {
        "cf_y": float(np.median(cf_y)),
        "cf_d": float(np.median(cf_d)),
        "rho": float(np.median(rho)),
        "delta_theta": float(np.median(delta)),
    }


def benchmark_table(
    info: Dict[str, Any],
    benchmark_covariates: Sequence[Any],
    *,
    theta: float,
    s: float,
    k_y: float,
    k_d: float,
) -> pd.DataFrame:
    """One row per benchmark: a covariate name, or a list of names left out
    together."""
    refit: Optional[ShortRefit] = info.get("_benchmark_refit")
    long_reps = info.get("_sens")
    cov_names = list(info.get("_covariate_names") or [])
    if isinstance(benchmark_covariates, str):
        benchmark_covariates = [benchmark_covariates]
    sets: List[List[str]] = [
        [b] if isinstance(b, str) else [str(x) for x in b] for b in benchmark_covariates
    ]
    unknown = sorted({c for group in sets for c in group if c not in cov_names})
    if unknown:
        raise MethodIncompatibility(
            f"dml_sensitivity: benchmark_covariates {unknown} are not "
            "covariates of the fit.",
            recovery_hint=f"Choose from the fit's covariates: {cov_names}.",
            diagnostics={"unknown": unknown, "covariates": cov_names},
        )
    if refit is None or not long_reps:
        raise MethodIncompatibility(
            "dml_sensitivity: benchmark_covariates needs to refit the model "
            "without the benchmark covariates, which this fit does not allow "
            "(sample weights, a survey design, two-way clustering or "
            "externally supplied predictions).",
            recovery_hint=(
                "Fit sp.dml(model='plr' or 'irm') without sample_weight=, "
                "design=, two-way cluster= or external_predictions=, or pass "
                "cf_y= and cf_d= directly."
            ),
        )
    var_y = float(np.var(np.asarray(info["_Y"], dtype=float)))
    rows: List[Dict[str, Any]] = []
    for group in sets:
        short = refit.fit(group)
        gain = gain_statistics(long_reps, short.model_info["_sens"], var_y)
        cf_y_b = float(np.clip(k_y * gain["cf_y"], 0.0, 0.999))
        cf_d_b = float(np.clip(k_d * gain["cf_d"], 0.0, 0.999))
        bias_b = float(np.sqrt(cf_y_b * cf_d_b / (1.0 - cf_d_b)) * s)
        rows.append(
            {
                "variable": "+".join(group),
                "k_y": k_y,
                "k_d": k_d,
                "cf_y_bench": cf_y_b,
                "cf_d_bench": cf_d_b,
                "rho": gain["rho"],
                "delta_theta": gain["delta_theta"],
                "bias_bound": bias_b,
                "adjusted_low": theta - bias_b,
                "adjusted_high": theta + bias_b,
            }
        )
    return pd.DataFrame(rows)
