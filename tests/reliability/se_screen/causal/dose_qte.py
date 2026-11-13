"""SE screen: quantile and distributional effects, continuous treatments.

Design constants (fixed before the first run, 2026-11-13):

* ``N = 1000``, ``B = 400`` unless a row says otherwise.
* Quantile effects use ``_dgp.obs``. There ``Y(0) ~ N(1, 1.5^2)`` and
  ``Y(1) ~ N(2, 3.5)``, so the unconditional QTE at ``tau`` is
  ``1 + z_tau (sqrt(3.5) - 1.5)``. The quantile effect on the treated has
  no closed form (the covariates of the treated are tilted); it is
  computed by Gauss-Hermite quadrature and root finding (``qtt_truth``).
* Complier quantile effects: randomised instrument, 60% compliers,
  ``Y(0) = e + 0.5 * complier``, ``Y(1) = 1 + 1.5 e + 0.5 * complier``,
  so the complier QTE at ``tau`` is ``1 + 0.5 z_tau``.
* QDiD: four group-by-period cells of 250; the untreated outcome is
  ``0.5 g + 0.3 t + e`` and the treated-post outcome ``2 + 0.5 + 0.3 +
  1.5 e``, so the effect at ``tau`` is ``2 + 0.5 z_tau``.
* Dose response: ``T = 0.5 x1 + v``. The headline is the response at the
  sample 75th percentile of the dose minus that at the 25th, so each
  draw's own target is ``0.7 (q75 - q25)`` (``sample_target``).

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/dose_qte.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _dgp import PS_B1, PS_B2, obs  # noqa: E402
from _harness import Row, main  # noqa: E402
from scipy import optimize, stats  # noqa: E402
from scipy.special import expit  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
X = ["x1", "x2"]


def qte_truth(tau: float) -> float:
    return float(1 + stats.norm.ppf(tau) * (np.sqrt(3.5) - 1.5))


def qtt_truth(tau: float) -> float:
    """Quantile effect on the treated in ``_dgp.obs``, by quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(80)
    weights = weights / weights.sum()
    a, b = np.meshgrid(nodes, nodes, indexing="ij")
    w = np.outer(weights, weights) * expit(PS_B1 * a + PS_B2 * b)
    w = w / w.sum()
    mu0 = 1 + a + 0.5 * b
    mu1 = mu0 + 1 + 0.5 * a

    def quantile(mu):
        return optimize.brentq(
            lambda y: float((w * stats.norm.cdf(y - mu)).sum()) - tau, -20, 20
        )

    return float(quantile(mu1) - quantile(mu0))


def _qte(index, hetero=True, **kw):
    def fit(rng):
        df = obs(rng, N, hetero=hetero)
        r = sp.qte(
            df,
            "y",
            "d",
            quantiles=[0.25, 0.5, 0.75],
            controls=X,
            seed=int(rng.integers(2**31)),
            **kw,
        )
        return {
            "est": float(r.effects[index]),
            "se": float(r.se[index]),
            "lo": float(r.ci_lower[index]),
            "hi": float(r.ci_upper[index]),
        }

    return fit


def _qte_rct(rng):
    """No covariates: the analytic influence-function SE is the default."""
    n = N
    d = rng.integers(0, 2, n)
    y = np.where(d == 1, 1 + 1.5 * rng.normal(size=n), rng.normal(size=n))
    df = pd.DataFrame({"y": y, "d": d})
    r = sp.qte(df, "y", "d", quantiles=[0.25, 0.5, 0.75])
    return {
        "est": float(r.effects[2]),
        "se": float(r.se[2]),
        "lo": float(r.ci_lower[2]),
        "hi": float(r.ci_upper[2]),
    }


def _dte(rng):
    df = obs(rng, N)
    r = sp.distributional_te(
        df,
        "y",
        "d",
        x=X,
        quantiles=[0.25, 0.5, 0.75],
        n_boot=200,
        seed=int(rng.integers(2**31)),
    )
    est, se = float(r.qte_effects[1]), float(r.qte_se[1])
    z = stats.norm.ppf(0.975)
    return {"est": est, "se": se, "lo": est - z * se, "hi": est + z * se}


def _complier(fn, **kw):
    def fit(rng):
        n = 2000
        z = rng.integers(0, 2, n)
        comp = rng.uniform(size=n) < 0.6
        d = (z * comp).astype(int)
        e = rng.normal(size=n)
        y = np.where(d == 1, 1 + 1.5 * e, e) + 0.5 * comp
        df = pd.DataFrame({"y": y, "d": d, "z": z})
        r = fn(
            df,
            "y",
            "d",
            "z",
            quantiles=np.array([0.25, 0.5, 0.75]),
            seed=int(rng.integers(2**31)),
            **kw,
        )
        return {
            "est": float(r.late_q[2]),
            "se": float(r.se_q[2]),
            "lo": float(r.ci_low[2]),
            "hi": float(r.ci_high[2]),
        }

    return fit


def _qdid(rng):
    n = N
    g = np.repeat([0, 0, 1, 1], n // 4)
    t = np.tile(np.repeat([0, 1], n // 4), 2)
    e = rng.normal(size=n)
    y = 0.5 * g + 0.3 * t + np.where((g == 1) & (t == 1), 2 + 1.5 * e, e)
    df = pd.DataFrame({"y": y, "g": g, "t": t})
    r = sp.qdid(
        df,
        y="y",
        group="g",
        time="t",
        quantiles=[0.25, 0.5, 0.75],
        n_boot=200,
        seed=int(rng.integers(2**31)),
    )
    return {
        "est": float(r.effects[2]),
        "se": float(r.se[2]),
        "lo": float(r.ci_lower[2]),
        "hi": float(r.ci_upper[2]),
    }


def _dose(method, confounded, n=N, **kw):
    def fit(rng):
        from sklearn.linear_model import LinearRegression

        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        t = 0.5 * x1 + rng.normal(size=n)
        y = 1 + 0.7 * t + (x1 if confounded else 0.0) + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "t": t, "x1": x1, "x2": x2})
        r = sp.dose_response(
            df,
            "y",
            "t",
            X,
            method=method,
            treatment_model=LinearRegression(),
            outcome_model=LinearRegression(),
            random_state=int(rng.integers(2**31)),
            **kw,
        )
        lo, hi = r.ci
        mi = r.model_info
        return {
            "est": float(r.estimate),
            "se": float(r.se),
            "lo": float(lo),
            "hi": float(hi),
            "truth_s": 0.7 * (mi["dose_75"] - mi["dose_25"]),
        }

    return fit


IQR_T = 0.7 * 2 * stats.norm.ppf(0.75) * np.sqrt(1.25)
D = "_dgp.obs"
ROWS = [
    Row(
        "qte.firpo.q75",
        "qte",
        "sp.qte(df, y, d, controls=X), tau = 0.75",
        N,
        qte_truth(0.75),
        _qte(2),
        design=D,
    ),
    Row(
        "qte.firpo.q50",
        "qte",
        "sp.qte(df, y, d, controls=X), tau = 0.5",
        N,
        qte_truth(0.5),
        _qte(1),
        design=D,
    ),
    Row(
        "qte.firpo.q75.analytic",
        "qte",
        "sp.qte(..., controls=X, se='analytic'), tau = 0.75",
        N,
        qte_truth(0.75),
        _qte(2, se="analytic"),
        design=D,
    ),
    Row(
        "qte.firpo.no_covariates",
        "qte",
        "sp.qte(df, y, d), tau = 0.75",
        N,
        1 + 0.5 * stats.norm.ppf(0.75),
        _qte_rct,
        design="randomised d, Y(1) = 1 + 1.5 e, Y(0) = e",
    ),
    Row(
        "qte.firpo_qtt.q75",
        "qte",
        "sp.qte(..., method='firpo_qtt', controls=X), tau = 0.75",
        N,
        qtt_truth(0.75),
        _qte(2, method="firpo_qtt"),
        design=D,
    ),
    Row(
        "qte.distribution.q75",
        "qte",
        "sp.qte(..., method='distribution', controls=X), tau = 0.75",
        N,
        qtt_truth(0.75),
        _qte(2, method="distribution"),
        design=D,
    ),
    Row(
        "qte.conditional_qr.q75",
        "qte",
        "sp.qte(..., method='conditional_qr', controls=X, n_boot=200), tau = 0.75",
        N,
        1.0,
        _qte(2, hetero=False, method="conditional_qr", n_boot=200),
        B=200,
        design=D + " with a constant effect (location shift, so the "
        "conditional quantile effect is 1 at every tau; 200 bootstrap "
        "draws, B = 200)",
    ),
    Row(
        "distributional_te.q50",
        "distributional_te",
        "sp.distributional_te(df, y, d, x=X, n_boot=200).qte_effects[0.5] "
        "+/- 1.96 se",
        N,
        qtt_truth(0.5),
        _dte,
        design=D,
    ),
    Row(
        "dist_iv.q75",
        "dist_iv",
        "sp.dist_iv(df, y, d, z), tau = 0.75",
        2000,
        1 + 0.5 * stats.norm.ppf(0.75),
        _complier(sp.dist_iv),
        design="randomised instrument, 60% compliers, n = 2000",
    ),
    Row(
        "beyond_average_late.q75",
        "beyond_average_late",
        "sp.beyond_average_late(df, y, d, z), tau = 0.75",
        2000,
        1 + 0.5 * stats.norm.ppf(0.75),
        _complier(sp.beyond_average_late),
        design="as dist_iv.q75",
    ),
    Row(
        "qdid.q75",
        "qdid",
        "sp.qdid(df, y=, group=, time=, n_boot=200), tau = 0.75",
        N,
        2 + 0.5 * stats.norm.ppf(0.75),
        _qdid,
        design="four cells of 250; effect 2 + 0.5 z_tau",
    ),
    Row(
        "dose_response.gps",
        "dose_response",
        "sp.dose_response(df, y, t, X, linear nuisance models)",
        N,
        IQR_T,
        _dose("gps", confounded=False),
        design="T = 0.5 x1 + v, Y = 1 + 0.7 T + e (covariates move the dose "
        "only, so the Hirano-Imbens quadratic outcome model is correct)",
        sample_target=True,
    ),
    Row(
        "dose_response.gps.confounded",
        "dose_response",
        "sp.dose_response(df, y, t, X, linear nuisance models)",
        N,
        IQR_T,
        _dose("gps", confounded=True),
        design="T = 0.5 x1 + v, Y = 1 + 0.7 T + x1 + e",
        sample_target=True,
    ),
    Row(
        "dose_response.dr",
        "dose_response",
        "sp.dose_response(..., method='dr', linear nuisance models)",
        500,
        IQR_T,
        _dose("dr", confounded=True, n=500),
        B=200,
        design="T = 0.5 x1 + v, Y = 1 + 0.7 T + x1 + e (n = 500, B = 200)",
        sample_target=True,
    ),
]

MECHANISMS: dict = {
    "qte.firpo.q75.analytic": (
        "Documented option. With covariates the analytic influence-function "
        "standard error treats the propensity score as known, which is "
        "conservative when it is estimated; the docstring says so, and "
        "se='auto' therefore switches to the bootstrap when controls are "
        "given (qte.firpo.q75, in band). Without covariates the analytic "
        "form is exact and in band (qte.firpo.no_covariates)."
    ),
    "dose_response.gps.confounded": (
        "Known property of the method, documented in sp.dose_response. The "
        "Hirano-Imbens estimator regresses the outcome on the dose and the "
        "generalized propensity score and is consistent only if that "
        "regression is right. Here the outcome model is linear in (T, "
        "GPS). The GPS is a normal density of the treatment residual, an "
        "even function of it, so it is uncorrelated with a covariate that "
        "enters the outcome linearly and the fitted curve keeps the "
        "unadjusted slope 0.7 + 0.5 / 1.25 = 1.1 instead of 0.7. The "
        "standard error is right for that estimate (SE / SD 0.94); the "
        "interval is centred in the wrong place. The docstring reports "
        "the same bias for the Hirano-Imbens quadratic and the function "
        "warns on its default models; method='dr' is in band on this "
        "design (dose_response.dr). When the covariates move the dose "
        "only, the GPS path is in band (dose_response.gps)."
    ),
}

# What a row measured before a fix that this screen led to.
HISTORY = {
    "dose_response.dr": {
        "statspai_version": "1.43.1",
        "B": 200,
        "coverage": 0.94,
        "se_over_sd": 0.703,
        "median_se_over_iqr_sd": 0.925,
        "bias_over_sd": 0.025,
        "mechanism": (
            "The bandwidth minimises a leave-one-out risk over a grid. A "
            "dose with no neighbour inside a narrow bandwidth gave a risk "
            "of NaN (0 / 0), and np.argmin returns the first NaN, so the "
            "narrowest bandwidth of the grid was selected in those draws "
            "and the curve was undersmoothed. The 25th-to-75th percentile "
            "contrast then swung widely: the SD over draws was dominated "
            "by them while the median SE over the interquartile scale was "
            "0.93. The risk of such a bandwidth is now infinite (fixed on "
            "main by the ML probe line the same day, commit 5eaa11f3e; "
            "this screen found it independently)."
        ),
    },
}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS, HISTORY)
