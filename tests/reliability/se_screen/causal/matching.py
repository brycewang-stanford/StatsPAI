"""SE screen: matching estimators.

Design constants (fixed before the first run, 2026-11-13):

* ``N = 1000``, design ``_dgp.obs`` (two continuous covariates, logit
  propensity, effect ``1 + 0.5 x1``). Rows that match without replacement
  or in pairs use ``intercept = -1.2`` (about 26% treated) so that every
  treated unit has a comparison unit.
* ``B = 400`` draws per row; bootstrap rows use ``B = 200``.
* Target. The Abadie-Imbens (2006) variance that ``sp.match`` reports by
  default is conditional on the matching variable, so those rows are
  scored against each draw's own conditional ATT / ATE
  (``sample_target=True``): the average of ``tau(X_i)`` when matching on
  the covariates, and of ``E[tau(X) | score_i]`` when matching on the
  propensity score (``_dgp.score_truth``; with the true score). Rows whose
  documentation claims a population variance are scored against the
  population value.

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/matching.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _dgp import OBS_TRUTH, obs, obs_truths, sample_truth, score_truth  # noqa: E402
from _harness import Row, ci_from, main  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
X = ["x1", "x2"]
ATE, ATT = OBS_TRUTH["ATE"], OBS_TRUTH["ATT"]
LOW = -1.2
ATT_LOW = obs_truths(LOW)["ATT"]
Z = stats.norm.ppf(0.975)


def _match(target="ATT", intercept=0.0, n=N, seeded=False, **kw):
    """``truth_s`` is the draw's own target for a conditional variance:
    given the covariates when matching on them, given the propensity
    scores when matching on the score (see ``_dgp.score_truth``)."""
    on_score = kw.get("distance", "propensity") == "propensity" and kw.get(
        "method", "nearest"
    ) in ("nearest", "kernel", "radius", "llr", "stratify")

    def fit(rng):
        df = obs(rng, n, intercept=intercept)
        if seeded:
            kw["bootstrap_seed"] = int(rng.integers(2**31))
        out = ci_from(sp.match(df, "y", "d", X, **kw))
        if on_score:
            out["truth_s"] = score_truth(df, target, intercept)
        else:
            out["truth_s"] = sample_truth(df, target)
        return out

    return fit


def _true_score(target="ATT", strata=False, **kw):
    """Match on the true propensity score instead of an estimated one."""

    def fit(rng):
        df = obs(rng, N)
        if strata:
            # deciles of the true score, as given strata
            import pandas as pd

            df["block"] = pd.qcut(df["ps"], 10, labels=False)
            out = ci_from(
                sp.match(df, "y", "d", X, method="stratify", strata="block", **kw)
            )
        else:
            out = ci_from(sp.match(df, "y", "d", X, pscore="ps", **kw))
        out["truth_s"] = score_truth(df, target)
        return out

    return fit


def _psmatch2(intercept=0.0, **kw):
    def fit(rng):
        df = obs(rng, N, intercept=intercept)
        r = sp.psmatch2(df, treat="d", covariates=X, y="y", **kw)
        lo, hi = r.ci
        return {
            "est": float(r.att),
            "se": float(r.se),
            "lo": float(lo),
            "hi": float(hi),
            "truth_s": score_truth(df, "ATT", intercept),
        }

    return fit


def _full(target="ATT", **kw):
    def fit(rng):
        df = obs(rng, 600)
        out = ci_from(sp.full_match(df, "y", "d", X, **kw))
        out["truth_s"] = sample_truth(df, target)
        return out

    return fit


def _optimal(rng):
    df = obs(rng, 600, intercept=LOW)
    r = sp.optimal_match(df, "d", "y", X)
    est, se = float(r.estimate), float(r.se)
    # the result carries no interval; this is the normal one from its SE
    return {
        "est": est,
        "se": se,
        "lo": est - Z * se,
        "hi": est + Z * se,
        "truth_s": sample_truth(df, "ATT"),
    }


def _genmatch(rng):
    df = obs(rng, 500)
    r = sp.genmatch(
        df,
        "y",
        "d",
        X,
        population_size=20,
        generations=5,
        random_state=int(rng.integers(2**31)),
    )
    lo, hi = r.ci
    return {
        "est": float(r.att),
        "se": float(r.att_se),
        "lo": float(lo),
        "hi": float(hi),
        "truth_s": sample_truth(df, "ATT"),
    }


def _cardinality(rng):
    df = obs(rng, 600, hetero=False)
    r = sp.cardinality_match(df, "d", "y", X)
    est, se = float(r.ate), float(r.se)
    return {"est": est, "se": se, "lo": est - Z * se, "hi": est + Z * se}


D = "_dgp.obs"
DL = "_dgp.obs(intercept=-1.2)"
ROWS = [
    Row(
        "match.att.ps",
        "match",
        "sp.match(df, y, d, X)",
        N,
        ATT,
        _match(),
        design=D,
        sample_target=True,
    ),
    Row(
        "match.att.ps.ai2016",
        "match",
        "sp.match(..., se_method='abadie_imbens_2016')",
        N,
        ATT,
        _match(se_method="abadie_imbens_2016"),
        design=D,
    ),
    Row(
        "match.ate.ps",
        "match",
        "sp.match(..., estimand='ATE')",
        N,
        ATE,
        _match("ATE", estimand="ATE"),
        design=D,
        sample_target=True,
    ),
    Row(
        "match.ate.ps.ai2016",
        "match",
        "sp.match(..., estimand='ATE', se_method='abadie_imbens_2016')",
        N,
        ATE,
        _match("ATE", estimand="ATE", se_method="abadie_imbens_2016"),
        design=D,
    ),
    Row(
        "match.att.mahalanobis",
        "match",
        "sp.match(..., distance='mahalanobis', bias_correction=True)",
        N,
        ATT,
        _match(distance="mahalanobis", bias_correction=True),
        design=D,
        sample_target=True,
    ),
    Row(
        "match.att.mahalanobis.m3",
        "match",
        "sp.match(..., distance='mahalanobis', n_matches=3, bias_correction=True)",
        N,
        ATT,
        _match(distance="mahalanobis", n_matches=3, bias_correction=True),
        design=D,
        sample_target=True,
    ),
    Row(
        "match.ate.mahalanobis",
        "match",
        "sp.match(..., estimand='ATE', distance='mahalanobis', bias_correction=True)",
        N,
        ATE,
        _match("ATE", estimand="ATE", distance="mahalanobis", bias_correction=True),
        design=D,
        sample_target=True,
    ),
    Row(
        "match.ate.mahalanobis.pop",
        "match",
        "sp.match(..., estimand='ATE', distance='mahalanobis', "
        "bias_correction=True, se_method='abadie_imbens_pop')",
        N,
        ATE,
        _match(
            "ATE",
            estimand="ATE",
            distance="mahalanobis",
            bias_correction=True,
            se_method="abadie_imbens_pop",
        ),
        design=D,
    ),
    Row(
        "match.att.ps.no_replace",
        "match",
        "sp.match(..., replace=False)",
        N,
        ATT_LOW,
        _match(intercept=LOW, replace=False),
        design=DL,
        sample_target=True,
    ),
    Row(
        "match.att.ps.caliper",
        "match",
        "sp.match(..., caliper=0.2, caliper_scale='sd')",
        N,
        ATT,
        _match(caliper=0.2, caliper_scale="sd"),
        design=D,
        sample_target=True,
    ),
    Row(
        "match.att.ps.true_score",
        "match",
        "sp.match(..., pscore='ps')",
        N,
        ATT,
        _true_score(),
        design=D + ", matching on the true score",
        sample_target=True,
    ),
    Row(
        "match.ate.ps.true_score",
        "match",
        "sp.match(..., pscore='ps', estimand='ATE')",
        N,
        ATE,
        _true_score("ATE", estimand="ATE"),
        design=D + ", matching on the true score",
        sample_target=True,
    ),
    Row(
        "match.att.kernel.true_score",
        "match",
        "sp.match(..., method='kernel', pscore='ps')",
        N,
        ATT,
        _true_score(method="kernel"),
        design=D + ", matching on the true score",
    ),
    Row(
        "match.att.stratify.true_score",
        "match",
        "sp.match(..., method='stratify', strata=<deciles of the true score>)",
        N,
        ATT,
        _true_score(strata=True),
        design=D + ", strata are deciles of the true score",
    ),
    Row(
        "match.att.kernel",
        "match",
        "sp.match(..., method='kernel')",
        N,
        ATT,
        _match(method="kernel"),
        design=D,
    ),
    Row(
        "match.att.kernel.bootstrap",
        "match",
        "sp.match(..., method='kernel', se_method='bootstrap')",
        N,
        ATT,
        _match(method="kernel", se_method="bootstrap", seeded=True),
        B=200,
        design=D + " (B = 200, 200 bootstrap draws)",
    ),
    Row(
        "match.att.radius",
        "match",
        "sp.match(..., method='radius', caliper=0.05)",
        N,
        ATT,
        _match(method="radius", caliper=0.05),
        design=D,
    ),
    Row(
        "match.att.llr",
        "match",
        "sp.match(..., method='llr', kernel='tricube')",
        N,
        ATT,
        _match(method="llr", kernel="tricube", seeded=True),
        B=200,
        design=D + " (B = 200; bootstrap is the only SE for llr)",
    ),
    Row(
        "match.att.llr.wide",
        "match",
        "sp.match(..., method='llr', kernel='tricube', bwidth=0.2, bootstrap_reps=100)",
        N,
        ATT,
        _match(
            method="llr", kernel="tricube", bwidth=0.2, bootstrap_reps=100, seeded=True
        ),
        B=200,
        design=D + " (B = 200, 100 bootstrap draws)",
    ),
    Row(
        "match.att.stratify",
        "match",
        "sp.match(..., method='stratify', n_strata=10)",
        N,
        ATT,
        _match(method="stratify", n_strata=10),
        design=D,
    ),
    Row(
        "match.ate.stratify",
        "match",
        "sp.match(..., method='stratify', n_strata=10, estimand='ATE')",
        N,
        ATE,
        _match("ATE", method="stratify", n_strata=10, estimand="ATE"),
        design=D,
    ),
    Row(
        "match.att.cem",
        "match",
        "sp.match(..., method='cem')",
        N,
        ATT,
        _match(method="cem"),
        design=D,
        sample_target=True,
    ),
    Row(
        "match.ate.nnmatch",
        "match",
        "sp.match(..., method='nnmatch', estimand='ATE', bias_adjust=True)",
        N,
        ATE,
        _match("ATE", method="nnmatch", estimand="ATE", bias_adjust=True),
        design=D,
    ),
    Row(
        "match.att.nnmatch",
        "match",
        "sp.match(..., method='nnmatch', estimand='ATT', bias_adjust=True)",
        N,
        ATT,
        _match(method="nnmatch", estimand="ATT", bias_adjust=True),
        design=D,
    ),
    Row(
        "psmatch2.att",
        "psmatch2",
        "sp.psmatch2(df, treat=d, covariates=X, y=y)",
        N,
        ATT,
        _psmatch2(),
        design=D,
    ),
    Row(
        "psmatch2.att.ai",
        "psmatch2",
        "sp.psmatch2(..., ai=1)",
        N,
        ATT,
        _psmatch2(ai=1),
        design=D,
        sample_target=True,
    ),
    Row(
        "full_match.att",
        "full_match",
        "sp.full_match(df, y, d, X)",
        600,
        ATT,
        _full(),
        design=D + " (n = 600)",
    ),
    Row(
        "full_match.ate",
        "full_match",
        "sp.full_match(..., estimand='ATE')",
        600,
        ATE,
        _full("ATE", estimand="ATE"),
        design=D + " (n = 600)",
    ),
    Row(
        "optimal_match.att",
        "optimal_match",
        "sp.optimal_match(df, d, y, X); interval = estimate +/- 1.96 se",
        600,
        ATT_LOW,
        _optimal,
        design=DL + " (n = 600)",
    ),
    Row(
        "genmatch.att",
        "genmatch",
        "sp.genmatch(df, y, d, X, population_size=20, generations=5)",
        500,
        ATT,
        _genmatch,
        B=200,
        design=D + " (n = 500, B = 200, a short genetic search)",
    ),
    Row(
        "cardinality_match.ate",
        "cardinality_match",
        "sp.cardinality_match(df, d, y, X); interval = estimate +/- 1.96 se",
        600,
        ATE,
        _cardinality,
        design=D + " with a constant effect (the matched subset is "
        "chosen for balance, so a heterogeneous effect has no fixed target)",
    ),
]

_ESTIMATED_SCORE = (
    "Known property, documented in sp.match. The analytic variance takes "
    "the propensity score as given. The score is estimated on the same "
    "sample, and a fitted logit balances the covariates in the sample "
    "better than the true score does, which lowers the sampling variance "
    "of the matched contrast (Abadie and Imbens 2016); the more the "
    "covariates predict the outcome beyond the score, the larger the gap. "
    "Matching on the true score with the same formula is in band (the "
    "*.true_score rows), and se_method='abadie_imbens_2016', which adds "
    "the correction, is in band for nearest-neighbour matching "
    "(match.att.ps.ai2016, match.ate.ps.ai2016)."
)
_PSMATCH2 = (
    "Reference convention plus the estimated-score effect. The psmatch2 "
    "analytic variance, var_1 / N_1 + var_0 sum(w^2) / N_1^2, uses each "
    "arm's raw outcome variance (homoskedastic, score taken as given), so "
    "it also counts the outcome variation that matching on the score "
    "removes. It is kept because it reproduces Stata psmatch2 (1e-16); "
    "se_method='bootstrap' is in band for kernel matching "
    "(match.att.kernel.bootstrap) and ai=1 / se_method='abadie_imbens' "
    "is closer for nearest-neighbour matching. "
)
MECHANISMS: dict = {
    "match.att.ps": _ESTIMATED_SCORE,
    "match.ate.ps": _ESTIMATED_SCORE,
    "match.att.ps.no_replace": _ESTIMATED_SCORE,
    "match.att.ps.caliper": _ESTIMATED_SCORE,
    "match.att.stratify": _ESTIMATED_SCORE,
    "match.ate.stratify": _ESTIMATED_SCORE,
    "full_match.att": _ESTIMATED_SCORE,
    "full_match.ate": _ESTIMATED_SCORE,
    "psmatch2.att.ai": _ESTIMATED_SCORE,
    "match.att.kernel": _PSMATCH2 + _ESTIMATED_SCORE,
    "match.att.radius": _PSMATCH2 + _ESTIMATED_SCORE,
    "psmatch2.att": _PSMATCH2 + _ESTIMATED_SCORE,
    "match.att.llr": (
        "Known property of local linear matching at a narrow bandwidth. "
        "The local linear weights are unbounded when a treated unit has "
        "few distinct controls inside the bandwidth (0.06 on the score "
        "here), and a bootstrap resample, which repeats controls, makes "
        "that more likely. A handful of replications then land far from "
        "the rest and dominate the standard deviation of the bootstrap "
        "distribution: the mean reported SE is several times the sampling "
        "SD while the median SE over the interquartile scale of the "
        "estimate (median_se_over_iqr_sd) is near one. The estimator now "
        "warns when the bootstrap standard deviation is more than twice "
        "the scale its interquartile range implies and reports that scale "
        "in model_info['bootstrap_se_iqr']. A wider bandwidth removes the "
        "problem (match.att.llr.wide)."
    ),
    "genmatch.att": (
        "Known and warned about. att_se is the matched-pair standard "
        "error, which treats pairs as independent although matching with "
        "replacement reuses controls; sp.genmatch says so in a warning "
        "that quotes the same range (0.56 to 0.91 of the sampling SD). No "
        "analytic alternative is implemented for the genetic-matching "
        "metric; sp.match(distance='mahalanobis') with the Abadie-Imbens "
        "variance is in band."
    ),
    "match.att.cem": (
        "Bias only, a property of the method: CEM drops treated units "
        "whose cell holds no control and compares the rest within coarse "
        "cells, so its target is the effect on the matched treated units "
        "and a little within-cell imbalance remains. The standard error "
        "was the finding (see HISTORY): it is in band since the fix."
    ),
}

# What a row measured before a fix that this screen led to.
HISTORY = {
    "match.att.cem": {
        "statspai_version": "1.43.1",
        "B": 400,
        "coverage": 0.9925,
        "se_over_sd": 1.42,
        "bias_over_sd": -0.252,
        "mechanism": (
            "The ATT's standard error was built from each arm's variance "
            "around its overall mean over the matched sample, which "
            "includes the outcome variation between cells; exact matching "
            "on the cells removes that variation from the estimate. The "
            "documentation described a within-cell variance, and the ATE "
            "branch used one. The ATT now does too."
        ),
    },
}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS, HISTORY)
