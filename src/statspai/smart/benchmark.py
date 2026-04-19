"""
Calibration benchmark for the ``verify`` scoring.

``sp.verify_benchmark()`` runs the verify engine against several built-in
DGPs (data generating processes) with known true effects. The output is a
small calibration card telling the user, empirically:

- on a clean DGP where the recommended method is correct, what score
  range should they expect?
- how does score degrade when the method is misspecified relative to
  the DGP?

This is the "is my score of 78/100 good or bad?" answer.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .verify import _run_method

__all__ = ["verify_benchmark"]


_SCENARIOS = {
    "rct": {
        "dgp": "dgp_rct",
        "dgp_kwargs": {"n": 500, "effect": 0.5, "n_covariates": 2},
        "y": "y",
        "treatment": "treatment",
        "design": "rct",
        "true_effect": 0.5,
    },
    "did": {
        "dgp": "dgp_did",
        "dgp_kwargs": {"n_units": 200, "n_periods": 2, "effect": 0.5},
        "y": "y",
        "treatment": "treated",
        "id": "unit",
        "time": "time",
        "design": "did",
        "true_effect": 0.5,
    },
    "staggered_did": {
        "dgp": "dgp_did",
        "dgp_kwargs": {"n_units": 200, "n_periods": 5, "effect": 0.5,
                       "staggered": True},
        "y": "y",
        "treatment": "treated",
        "id": "unit",
        "time": "time",
        "design": "did",
        "true_effect": 0.5,
    },
    "rd": {
        "dgp": "dgp_rd",
        "dgp_kwargs": {"n": 1000, "effect": 0.5},
        "y": "y",
        "running_var": "x",
        "cutoff": 0.0,
        "design": "rd",
        "true_effect": 0.5,
    },
    "iv": {
        "dgp": "dgp_iv",
        "dgp_kwargs": {"n": 500, "effect": 0.5, "first_stage": 0.6},
        "y": "y",
        "treatment": "treatment",
        "instrument": "instrument",
        "design": "iv",
        "true_effect": 0.5,
    },
    "observational": {
        "dgp": "dgp_observational",
        "dgp_kwargs": {"n": 500, "effect": 0.5, "confounding": 0.2},
        "y": "y",
        "treatment": "treatment",
        "covariates": ["x1", "x2"],
        "design": "observational",
        "true_effect": 0.5,
    },
}


def verify_benchmark(
    scenarios: Optional[List[str]] = None,
    n_reps: int = 3,
    seed: int = 0,
    verify_B: int = 30,
    verify_budget_s: float = 15.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run verify against built-in DGPs with known true effects.

    Parameters
    ----------
    scenarios : list of str, optional
        Subset of {'rct', 'did', 'staggered_did', 'rd', 'iv',
        'observational'}. Defaults to all six.
    n_reps : int, default 3
        Number of independent DGP draws per scenario. Each uses a
        different seed to average out Monte Carlo noise.
    seed : int, default 0
        Base seed. Per-rep seeds are ``seed + offset``.
    verify_B : int, default 30
        Bootstrap replications per verification run.
    verify_budget_s : float, default 15.0
        Wall-clock budget per verification run.
    verbose : bool, default True
        Print per-scenario progress.

    Returns
    -------
    pd.DataFrame
        One row per (scenario, rep, recommendation) combination, with
        columns: scenario, rep, method, score, stability, placebo,
        subsample, point_estimate, true_effect, bias, elapsed_s.
    """
    import statspai as sp

    if scenarios is None:
        scenarios = list(_SCENARIOS.keys())

    rows: List[Dict] = []

    for name in scenarios:
        if name not in _SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{name}'. "
                f"Available: {list(_SCENARIOS.keys())}"
            )
        spec = _SCENARIOS[name]
        dgp_fn = getattr(sp, spec["dgp"])

        if verbose:
            print(f"[verify_benchmark] {name}: {spec['dgp']} "
                  f"(true effect = {spec['true_effect']}, n_reps = {n_reps})")

        for rep in range(n_reps):
            df = dgp_fn(**spec["dgp_kwargs"], seed=seed + rep)

            rec_kwargs = {
                "data": df, "y": spec["y"], "design": spec["design"],
                "verify": True, "verify_B": verify_B,
                "verify_budget_s": verify_budget_s, "verify_top_k": 2,
            }
            for k in ("treatment", "id", "time", "running_var", "cutoff",
                      "instrument", "covariates"):
                if k in spec:
                    rec_kwargs[k] = spec[k]

            try:
                rec = sp.recommend(**rec_kwargs)
            except Exception as e:
                rows.append({
                    "scenario": name, "rep": rep, "method": "ERROR",
                    "score": np.nan, "stability": np.nan,
                    "placebo": np.nan, "subsample": np.nan,
                    "point_estimate": np.nan,
                    "true_effect": spec["true_effect"],
                    "bias": np.nan, "elapsed_s": np.nan,
                    "error": str(e)[:80],
                })
                continue

            for r in rec.recommendations[:2]:
                v = r.get("verify") or {}
                score = v.get("score", np.nan)
                stab = (v.get("stability") or {}).get("score", np.nan)
                plac = (v.get("placebo") or {}).get("score", np.nan)
                subs = (v.get("subsample") or {}).get("score", np.nan)

                # Point estimate: run once on full data for bias check
                try:
                    point = _run_method(r, df)
                except Exception:
                    point = np.nan
                bias = (point - spec["true_effect"]
                        if isinstance(point, float) and np.isfinite(point)
                        else np.nan)

                rows.append({
                    "scenario": name, "rep": rep,
                    "method": r["method"],
                    "score": score, "stability": stab,
                    "placebo": plac, "subsample": subs,
                    "point_estimate": point,
                    "true_effect": spec["true_effect"],
                    "bias": bias,
                    "elapsed_s": v.get("elapsed_s", np.nan),
                    "error": v.get("error"),
                })

    df_out = pd.DataFrame(rows)

    if verbose:
        # Summary pivot: mean score per scenario for top-1 method
        top1 = df_out.groupby(["scenario", "rep"]).head(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg = top1.groupby("scenario").agg(
                score_mean=("score", lambda s: np.nanmean(s)),
                score_std=("score", lambda s: np.nanstd(s, ddof=1)
                           if np.sum(np.isfinite(s)) > 1 else np.nan),
                bias_mean=("bias", lambda s: np.nanmean(np.abs(s))),
            ).round(2)
        print()
        print("=" * 60)
        print("verify_benchmark — calibration summary (top-1 method)")
        print("=" * 60)
        print(agg.to_string())
        print("=" * 60)

    return df_out
