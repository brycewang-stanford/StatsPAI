"""Data-generating processes shared by the SE screen.

Every design here has closed-form (or quadrature) population truths, so a
row never estimates its own target by simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

# ---------------------------------------------------------------------------
# Observational design with selection on two observed covariates.
#
#   x1, x2 ~ N(0, 1) independent
#   P(D = 1 | x) = expit(PS_B1 x1 + PS_B2 x2)      (logit, good overlap)
#   tau(x) = TAU0 + TAU1 x1
#   Y = 1 + x1 + 0.5 x2 + tau(x) D + e,  e ~ N(0, 1)
#
# Both the logit propensity model and the linear-with-interaction outcome
# model are correctly specified; a linear outcome model without the
# interaction is misspecified only in the treated arm's slope on x1, which
# per-arm regressions absorb.
# ---------------------------------------------------------------------------
PS_B1, PS_B2 = 0.5, -0.5
TAU0, TAU1 = 1.0, 0.5


def _gauss_hermite_2d(fn, deg: int = 80) -> float:
    """E[fn(x1, x2)] for independent standard normals, by quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(deg)
    weights = weights / weights.sum()
    a, b = np.meshgrid(nodes, nodes, indexing="ij")
    w = np.outer(weights, weights)
    return float((w * fn(a, b)).sum())


def obs_truths(intercept: float = 0.0) -> dict:
    """Population ATE / ATT / ATC / ATO of the observational design."""

    def ps(a, b):
        return expit(intercept + PS_B1 * a + PS_B2 * b)

    def tau(a, b):
        return TAU0 + TAU1 * a

    e_ps = _gauss_hermite_2d(ps)
    att = _gauss_hermite_2d(lambda a, b: tau(a, b) * ps(a, b)) / e_ps
    atc = _gauss_hermite_2d(lambda a, b: tau(a, b) * (1 - ps(a, b))) / (1 - e_ps)
    h = lambda a, b: ps(a, b) * (1 - ps(a, b))  # noqa: E731
    ato = _gauss_hermite_2d(lambda a, b: tau(a, b) * h(a, b)) / _gauss_hermite_2d(h)

    def tilted(tilt):
        return _gauss_hermite_2d(
            lambda a, b: tau(a, b) * tilt(ps(a, b))
        ) / _gauss_hermite_2d(lambda a, b: tilt(ps(a, b)))

    # Li, Morgan and Zaslavsky (2018) tilting functions
    matching = tilted(lambda e: np.minimum(e, 1 - e))
    entropy = tilted(lambda e: -e * np.log(e) - (1 - e) * np.log(1 - e))
    return {
        "ATE": TAU0,
        "ATT": att,
        "ATC": atc,
        "ATO": ato,
        "matching": matching,
        "entropy": entropy,
    }


OBS_TRUTH = obs_truths()


def obs(
    rng: np.random.Generator, n: int, hetero: bool = True, intercept: float = 0.0
) -> pd.DataFrame:
    """One draw of the observational design.

    Columns ``y, d, x1, x2`` plus ``tau`` (the unit's own effect, for the
    in-sample versions of the estimands) and ``ps`` (the true propensity).
    With ``hetero=False`` the effect is the constant ``TAU0``. ``intercept``
    shifts the propensity index (``-1.2`` gives about 26% treated).
    """
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    ps = expit(intercept + PS_B1 * x1 + PS_B2 * x2)
    d = (rng.uniform(size=n) < ps).astype(int)
    tau = TAU0 + (TAU1 * x1 if hetero else 0.0) * np.ones(n)
    y = 1 + x1 + 0.5 * x2 + tau * d + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "tau": tau, "ps": ps})


def rct(rng: np.random.Generator, n: int, p: float = 0.5) -> pd.DataFrame:
    """Bernoulli-randomised experiment with the same outcome equation."""
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    d = (rng.uniform(size=n) < p).astype(int)
    tau = TAU0 + TAU1 * x1
    y = 1 + x1 + 0.5 * x2 + tau * d + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "tau": tau})


def sample_truth(df: pd.DataFrame, estimand: str) -> float:
    """In-sample version of an estimand on a draw of :func:`obs`."""
    if estimand == "ATT":
        return float(df.loc[df["d"] == 1, "tau"].mean())
    if estimand == "ATC":
        return float(df.loc[df["d"] == 0, "tau"].mean())
    return float(df["tau"].mean())


def score_truth(df: pd.DataFrame, estimand: str, intercept: float = 0.0) -> float:
    """An estimand of a draw of :func:`obs`, given the propensity scores.

    Matching on the score compares units with equal scores, not equal
    covariates, so the estimator's conditional variance (Abadie and Imbens
    2006, with the score as the matching variable) is around the average
    of ``E[tau(X) | score]`` over the target units. With ``s = PS_B1 x1 +
    PS_B2 x2`` the regression of ``x1`` on ``s`` has slope ``PS_B1 /
    (PS_B1^2 + PS_B2^2)``, so ``E[tau | score] = TAU0 + TAU1 * slope * s``.
    """
    s = np.log(df["ps"] / (1 - df["ps"])) - intercept
    tau_s = TAU0 + TAU1 * PS_B1 / (PS_B1**2 + PS_B2**2) * s
    if estimand == "ATT":
        return float(tau_s[df["d"] == 1].mean())
    if estimand == "ATC":
        return float(tau_s[df["d"] == 0].mean())
    return float(tau_s.mean())


def obs_clustered(
    rng: np.random.Generator, n_clusters: int = 100, size: int = 10
) -> pd.DataFrame:
    """The observational design with within-cluster dependence.

    ``x1`` and the outcome error each carry a cluster component (half of
    the variance of ``x1``, intra-cluster correlation 1/3 in the error);
    the marginal law of ``(x1, x2)`` is unchanged, so the ATE is still
    ``TAU0``. Column ``g`` is the cluster.
    """
    n = n_clusters * size
    g = np.repeat(np.arange(n_clusters), size)
    x1 = np.sqrt(0.5) * rng.normal(size=n_clusters)[g] + np.sqrt(0.5) * rng.normal(
        size=n
    )
    x2 = rng.normal(size=n)
    ps = expit(PS_B1 * x1 + PS_B2 * x2)
    d = (rng.uniform(size=n) < ps).astype(int)
    tau = TAU0 + TAU1 * x1
    err = np.sqrt(0.5) * rng.normal(size=n_clusters)[g] + rng.normal(size=n)
    y = 1 + x1 + 0.5 * x2 + tau * d + err
    return pd.DataFrame(
        {"y": y, "d": d, "x1": x1, "x2": x2, "tau": tau, "ps": ps, "g": g}
    )
