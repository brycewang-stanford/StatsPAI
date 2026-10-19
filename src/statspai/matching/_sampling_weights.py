"""Sampling (survey) weights in nearest-neighbour matching.

The ``MatchIt`` treatment of ``s.weights`` (R MatchIt 4.7.2 documentation
and vignettes "sampling-weights" / "estimating-effects"; GPL, used as a
black box):

1. the propensity score is a logistic regression weighted by the sampling
   weights (``glm(..., weights = s)``, quasi-binomial);
2. matching itself does not use the weights;
3. the effect is the coefficient of ``lm(y ~ treat, weights = w * s)`` on
   the matched sample, ``w`` the matching weights;
4. its standard error is cluster-robust on matched sets
   (``sandwich::vcovCL(fit, cluster = ~subclass)``, HC1 with the
   ``G / (G - 1)`` factor) without replacement, and ``HC3`` with
   replacement, where a control can sit in several sets.

The standard error does not account for the estimation of the propensity
score or for the matching itself; it is what MatchIt's documentation
prescribes and what ``tests/reference_parity/test_match_sweights_parity.py``
checks.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceWarning, MethodIncompatibility, NumericalInstability
from ..exceptions import warn as _sp_warn


def weighted_logit(X: np.ndarray, t: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Fitted probabilities of the sampling-weighted logit of ``t`` on
    ``[1, X]`` (the quasi-binomial ``glm(weights = s)`` estimate)."""
    sd = X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.column_stack([np.ones(len(t)), (X - X.mean(axis=0)) / sd])
    beta = np.zeros(Z.shape[1])
    for _ in range(100):
        eta = Z @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = Z.T @ (s * (t - p))
        if np.max(np.abs(grad)) <= 1e-11 * s.sum():
            break
        H = (Z * (s * p * (1 - p))[:, None]).T @ Z
        beta = beta + np.linalg.solve(H, grad)
        if np.max(np.abs(beta)) > 40:
            raise NumericalInstability(
                "sp.match(sampling_weights=): the weighted propensity model "
                "separates the treatment groups.",
                recovery_hint="Drop or coarsen the separating covariate.",
            )
    else:  # pragma: no cover - Newton on a concave likelihood
        _sp_warn(
            ConvergenceWarning,
            "sp.match(sampling_weights=): the weighted logit did not converge.",
            stacklevel=4,
        )
    return 1.0 / (1.0 + np.exp(-(Z @ beta)))


def match_sampling_weighted(
    data: pd.DataFrame, sampling_weights: str, spec: Dict[str, Any], fit: Any
) -> CausalResult:
    """Run :func:`sp.match` with MatchIt-style sampling weights.

    ``fit(data, spec)`` runs the ordinary matching estimator.
    """
    method = str(spec.get("method", "nearest")).lower()
    if method not in ("nearest", "psm", "nn", "nearest_neighbor"):
        raise MethodIncompatibility(
            "sp.match: sampling_weights= is implemented for nearest-neighbour "
            f"matching (MatchIt's method='nearest'); got method={method!r}.",
            recovery_hint="Use method='nearest', or a weighting estimator: "
            "sp.ipw / sp.aipw / sp.ebalance accept weights=.",
        )
    estimand = str(spec.get("estimand", "ATT")).upper()
    if estimand not in ("ATT", "ATC"):
        raise MethodIncompatibility(
            "sp.match: sampling_weights= supports estimand='ATT' or 'ATC' "
            "(nearest-neighbour matching keeps one arm whole).",
            recovery_hint="Use estimand='ATT'.",
        )
    if spec.get("bias_correction") or str(spec.get("se_method", "auto")) != "auto":
        raise MethodIncompatibility(
            "sp.match: with sampling_weights= the standard error is the "
            "MatchIt regression one; bias_correction / se_method do not apply.",
            recovery_hint="Drop bias_correction= and se_method=.",
        )
    if sampling_weights not in data.columns:
        raise MethodIncompatibility(
            f"sp.match: sampling_weights column {sampling_weights!r} not found."
        )
    s_all = data[sampling_weights].to_numpy(dtype=float)
    ok = np.isfinite(s_all)
    if (s_all[ok] <= 0).any():
        raise MethodIncompatibility(
            "sp.match: sampling weights must be strictly positive.",
            recovery_hint="Drop rows with zero weight before matching.",
        )

    work = data
    spec = dict(spec)
    distance = spec.get("distance")
    uses_ps = distance in (None, "propensity", "ps", "pscore", "logit")
    if uses_ps and spec.get("pscore") is None:
        if spec.get("ps_poly", 1) != 1 or str(spec.get("ps_model", "logit")) != "logit":
            raise MethodIncompatibility(
                "sp.match: with sampling_weights= the propensity score is the "
                "weighted logit on the covariates (ps_poly=1, ps_model='logit').",
                recovery_hint="Fit your own score and pass pscore=<column>.",
            )
        cols = [spec["treat"]] + list(spec["covariates"]) + [sampling_weights]
        rows = data[cols].notna().all(axis=1).to_numpy()
        X = data.loc[rows, list(spec["covariates"])].to_numpy(dtype=float)
        t = data.loc[rows, spec["treat"]].to_numpy(dtype=float)
        ps = np.full(len(data), np.nan)
        ps[rows] = weighted_logit(X, t, s_all[rows])
        work = data.copy()
        work["__ps_sampling__"] = ps
        spec["pscore"] = "__ps_sampling__"

    base = fit(work, spec)
    md = base.model_info.get("matched_data")
    if md is None:  # pragma: no cover - nearest matching always records it
        raise MethodIncompatibility("sp.match: no matched sample to weight.")
    md = md[md["_weight"].notna() & (md["_weight"] > 0)]
    y = md[spec["y"]].to_numpy(dtype=float)
    d = md["_treated"].to_numpy(dtype=float)
    w = md["_weight"].to_numpy(dtype=float) * md[sampling_weights].to_numpy(dtype=float)
    if estimand == "ATC":
        d = 1.0 - d
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtW = X.T * w
    bread = np.linalg.inv(XtW @ X)
    coef = bread @ (XtW @ y)
    tau = float(coef[1])
    e = y - X @ coef
    replace = bool(spec.get("replace", True))
    if replace:
        h = np.einsum("ij,jk,ik->i", X, bread, X) * w
        score = X * (w * e / (1.0 - h))[:, None]
        V = bread @ (score.T @ score) @ bread
        vce = "HC3"
        n_sets = None
    else:
        sets = _matched_sets(md)
        _, code = np.unique(sets, return_inverse=True)
        G = int(code.max()) + 1
        score = X * (w * e)[:, None]
        sums = np.zeros((G, 2))
        np.add.at(sums, code, score)
        V = bread @ (sums.T @ sums) @ bread
        V *= (G / (G - 1.0)) * ((n - 1.0) / (n - 2.0))
        vce = "cluster(matched set)"
        n_sets = G
    se = float(np.sqrt(V[1, 1]))
    alpha = float(spec.get("alpha", 0.05))
    zc = stats.norm.ppf(1 - alpha / 2)
    info = dict(base.model_info)
    info.update(
        {
            "sampling_weights": sampling_weights,
            "sampling_weight_vce": vce,
            "n_matched_sets": n_sets,
            "ps_uses_sampling_weights": uses_ps
            and spec.get("pscore") == "__ps_sampling__",
            "unweighted_estimate": float(base.estimate),
        }
    )
    return CausalResult(
        method=f"{base.method} (sampling-weighted)",
        estimand=estimand,
        estimate=tau,
        se=se,
        pvalue=float(2 * stats.norm.sf(abs(tau / se))) if se > 0 else float("nan"),
        ci=(tau - zc * se, tau + zc * se),
        alpha=alpha,
        n_obs=base.n_obs,
        detail=base.detail,
        model_info=info,
    )


def _matched_sets(md: pd.DataFrame) -> np.ndarray:
    """Matched-set label per row of the matched sample (without
    replacement every control belongs to exactly one treated unit)."""
    ids = md["_id"].to_numpy()
    set_of = {}
    ncols = [c for c in md.columns if c.startswith("_n") and c[2:].isdigit()]
    treated = md["_treated"].to_numpy() == 1
    for i, row in zip(ids[treated], md.loc[treated, ncols].to_numpy()):
        set_of[i] = i
        for c in row:
            if np.isfinite(c):
                set_of[int(c)] = i
    return np.array([set_of[i] for i in ids])
