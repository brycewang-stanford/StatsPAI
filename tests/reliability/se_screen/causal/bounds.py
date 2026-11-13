"""SE screen: partial identification (Lee, Manski, Nevo-Rosen).

The claim screened here is coverage of the reported interval for a
parameter that is only set-identified. Each row's ``truth`` is the
population identified set; coverage is the smaller of the two endpoint
coverages (an Imbens-Manski interval is at its nominal level at an
endpoint of a wide set, and above it in the interior).

Design constants (fixed before the first run, 2026-11-13):

* ``N = 2000``, ``B = 400``.
* Lee: randomised ``D``, selection ``S = 1[0.3 + 0.4 D + u > 0]``,
  ``Y = 1 + 0.5 D + 0.8 u + e`` observed when ``S = 1``; monotone
  selection holds. The population bounds are computed by one-dimensional
  quadrature from the skew-normal law of ``Y | S = 1`` (``lee_truth``).
* Manski: binary outcome, ``P(D = 1) = 0.5``, ``E[Y | D = 1] = 0.5``,
  ``E[Y | D = 0] = 0.3``. Worst-case set ``[-0.4, 0.6]``.
* Nevo-Rosen: ``(x, z, u)`` jointly normal with unit variances,
  ``corr(x, u) = 0.5``, ``corr(z, u) = 0.2``, ``corr(x, z) = -0.4``,
  ``y = x + u``. The two IV estimands are ``1 + 0.2 / (-0.4) = 0.5`` and
  ``1 + 0.3 / 1.4``, so the set is closed and contains the slope 1.

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/bounds.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _causal_harness import Row, main  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402

N = 2000


def lee_truth() -> tuple:
    """Population Lee bounds of the selection design, by quadrature."""
    c0, c1 = 0.3, 0.7  # selection thresholds: u > -c
    p0, p1 = stats.norm.cdf(c0), stats.norm.cdf(c1)
    sigma = np.sqrt(1 + 0.8**2)
    rho = 0.8 / sigma
    mean_control = 1.0 + 0.8 * stats.norm.pdf(c0) / p0
    # density of Y | D = 1, S = 1 with Y = 1.5 + 0.8 u + e and u > -c1
    y = np.linspace(1.5 - 10 * sigma, 1.5 + 10 * sigma, 400_001)
    zy = (y - 1.5) / sigma
    dens = (
        stats.norm.pdf(zy)
        / sigma
        * stats.norm.cdf((rho * zy + c1) / np.sqrt(1 - rho**2))
        / p1
    )
    dy = y[1] - y[0]
    cdf = np.cumsum(dens) * dy
    keep = p0 / p1  # share of the treated-and-selected kept after trimming
    lower_mask = cdf <= keep
    upper_mask = cdf >= 1 - keep
    low = np.sum(y[lower_mask] * dens[lower_mask]) * dy / keep
    high = np.sum(y[upper_mask] * dens[upper_mask]) * dy / keep
    return float(low - mean_control), float(high - mean_control)


LEE = lee_truth()
MANSKI = (-0.4, 0.6)
MANSKI_MTS = (-0.4, 0.2)
NEVO_ROSEN = (0.5, 1 + 0.3 / 1.4)


def _lee_draw(rng, n=N):
    d = rng.integers(0, 2, n)
    u = rng.normal(size=n)
    s = ((0.3 + 0.4 * d + u) > 0).astype(int)
    y = np.where(s == 1, 1 + 0.5 * d + 0.8 * u + rng.normal(size=n), np.nan)
    return pd.DataFrame({"y": y, "d": d, "s": s, "x": rng.integers(0, 2, n)})


def _causal_set(r):
    lo, hi = r.ci
    return {
        "lo": float(lo),
        "hi": float(hi),
        "b_lo": float(r.model_info["lower_bound"]),
        "b_hi": float(r.model_info["upper_bound"]),
    }


def _bounds_set(r):
    lo, hi = r.parameter_ci()
    return {"lo": lo, "hi": hi, "b_lo": float(r.lower), "b_hi": float(r.upper)}


def _lee(**kw):
    def fit(rng):
        df = _lee_draw(rng)
        if kw.get("se_method", "bootstrap") == "bootstrap":
            kw["random_state"] = int(rng.integers(2**31))
        return _causal_set(sp.lee_bounds(df, "y", "d", "s", **kw))

    return fit


def _selection(rng):
    df = _lee_draw(rng)
    r = sp.selection_bounds(
        df,
        "y",
        "d",
        "s",
        method="unconditional",
        random_state=int(rng.integers(2**31)),
    )
    return _bounds_set(r)


def _manski_draw(rng, n=N):
    u = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.4 + 0.2 * (u > 0)).astype(int)
    y = (rng.uniform(size=n) < 0.3 + 0.2 * d).astype(float)
    return pd.DataFrame({"y": y, "d": d, "x": rng.normal(size=n)})


def _manski(**kw):
    def fit(rng):
        df = _manski_draw(rng)
        r = sp.manski_bounds(
            df,
            "y",
            "d",
            y_lower=0.0,
            y_upper=1.0,
            random_state=int(rng.integers(2**31)),
            **kw,
        )
        return _causal_set(r)

    return fit


def _horowitz(rng):
    df = _manski_draw(rng)
    r = sp.horowitz_manski(
        df,
        "y",
        "d",
        ["x"],
        y_lower=0.0,
        y_upper=1.0,
        n_boot=200,
        random_state=int(rng.integers(2**31)),
    )
    return _bounds_set(r)


def _ml_bounds(rng):
    df = _manski_draw(rng, 1000)
    r = sp.ml_bounds(
        df,
        "y",
        "d",
        ["x"],
        y_min=0.0,
        y_max=1.0,
        n_bootstrap=100,
        random_state=int(rng.integers(2**31)),
    )
    return {
        "lo": float(r.lower),
        "hi": float(r.upper),
        "b_lo": float(r.adaptive_lower),
        "b_hi": float(r.adaptive_upper),
    }


def _nevo_rosen(rng):
    cov = np.array([[1.0, -0.4, 0.5], [-0.4, 1.0, 0.2], [0.5, 0.2, 1.0]])
    xzu = rng.multivariate_normal(np.zeros(3), cov, size=N)
    df = pd.DataFrame({"x": xzu[:, 0], "z": xzu[:, 1]})
    df["y"] = df["x"] + xzu[:, 2]
    r = sp.iv_bounds(df, "y", "x", "z", random_state=int(rng.integers(2**31)))
    return _bounds_set(r)


S = dict(kind="set", n=N)
ROWS = [
    Row(
        "lee_bounds.bootstrap",
        "lee_bounds",
        "sp.lee_bounds(df, y, d, s)",
        truth=LEE,
        fit=_lee(),
        design="Lee selection design",
        **S,
    ),
    Row(
        "lee_bounds.analytic",
        "lee_bounds",
        "sp.lee_bounds(..., se_method='analytic')",
        truth=LEE,
        fit=_lee(se_method="analytic"),
        design="Lee selection design",
        **S,
    ),
    Row(
        "lee_bounds.covariates",
        "lee_bounds",
        "sp.lee_bounds(..., covariates=['x'])",
        truth=LEE,
        fit=_lee(covariates=["x"]),
        design="Lee selection design; x is an independent binary covariate, "
        "so the tightened set equals the plain one",
        **S,
    ),
    Row(
        "selection_bounds",
        "selection_bounds",
        "sp.selection_bounds(df, y, d, s, method='unconditional').parameter_ci()",
        truth=LEE,
        fit=_selection,
        design="Lee selection design",
        **S,
    ),
    Row(
        "manski_bounds",
        "manski_bounds",
        "sp.manski_bounds(df, y, d, y_lower=0, y_upper=1)",
        truth=MANSKI,
        fit=_manski(),
        design="binary-outcome Manski design",
        **S,
    ),
    Row(
        "manski_bounds.mts",
        "manski_bounds",
        "sp.manski_bounds(..., assumption='mts')",
        truth=MANSKI_MTS,
        fit=_manski(assumption="mts"),
        design="binary-outcome Manski design",
        **S,
    ),
    Row(
        "horowitz_manski",
        "horowitz_manski",
        "sp.horowitz_manski(df, y, d, ['x'], n_boot=200).parameter_ci()",
        truth=MANSKI,
        fit=_horowitz,
        design="binary-outcome Manski design; x is independent noise",
        **S,
    ),
    Row(
        "ml_bounds",
        "ml_bounds",
        "sp.ml_bounds(df, y, d, ['x'], y_min=0, y_max=1, n_bootstrap=100), "
        "[lower, upper]",
        truth=MANSKI,
        fit=_ml_bounds,
        B=200,
        design="binary-outcome Manski design, n = 1000, B = 200",
        **{**S, "n": 1000},
    ),
    Row(
        "iv_bounds.nevo_rosen",
        "iv_bounds",
        "sp.iv_bounds(df, y, x, z).parameter_ci()",
        truth=NEVO_ROSEN,
        fit=_nevo_rosen,
        design="Nevo-Rosen design",
        **S,
    ),
]

MECHANISMS: dict = {}

# What a row measured before a fix that this screen led to.
HISTORY = {
    "ml_bounds": {
        "statspai_version": "1.43.1",
        "B": 200,
        "coverage": 0.685,
        "coverage_lower_endpoint": 0.685,
        "coverage_upper_endpoint": 0.695,
        "coverage_whole_set": 0.38,
        "mechanism": (
            "The bootstrap resampled the cross-fitted pointwise bounds "
            "L(x_i), U(x_i) with the nuisance fits held fixed. Those are "
            "predictions; their spread across units is the spread of the "
            "covariates' influence and holds none of the sampling noise "
            "of Y and A. With a covariate that is pure noise the band was "
            "about a quarter of the width it needed. The band now "
            "resamples each unit's own contribution to the bound, whose "
            "mean is the plain-means Manski bound."
        ),
    },
}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS, HISTORY)
