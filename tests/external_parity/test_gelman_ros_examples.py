"""Gelman, Hill and Vehtari, *Regression and Other Stories* (2020), on the
book's own data.

The examples (https://github.com/avehtari/ROS-Examples, shared with
*Active Statistics*, Gelman and Vehtari 2024) are R programs around
``rstanarm::stan_glm``. The answer key is ``data/gelman_ros_R.json``, made
by ``gelman_ros_reference.R`` next to this file. Here every number is
recomputed with the ``sp.*`` call a user would reach for.

Neither the programs nor the data are redistributed. Point
``STATSPAI_ROS_DIR`` at a copy of the repository:

    STATSPAI_ROS_DIR=/path/to/ROS-Examples \\
        pytest tests/external_parity/test_gelman_ros_examples.py

It is skipped otherwise. What the pass found is in
``docs/dev/2026-10-07-gelman-vehtari-active-statistics-review.md``; the same
functions are tested on a committed synthetic file in
``tests/reference_parity/test_regression_stories_r_parity.py``.

Two kinds of comparison.

``GLM`` (1e-6 relative): maximum-likelihood fits, where both sides solve
the same convex problem.

Posterior summaries: two samplers, Stan's NUTS and ours, with the *same*
prior and likelihood, differ by Monte Carlo error only. That is a
stochastic screen, not parity. Tolerances are stated in posterior standard
deviations: medians within 0.1 sd, spreads within 10 percent, and ``elpd``
within 1.5 (the Monte Carlo error of each side is a few tenths).
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_ROS_DIR")
KEY = Path(__file__).parent / "data" / "gelman_ros_R.json"
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "KidIQ" / "data" / "kidiq.csv").is_file(),
    reason="set STATSPAI_ROS_DIR to a copy of avehtari/ROS-Examples",
)

GLM = 1e-6
KW = dict(prior="weakly_informative", chains=4, seed=20261007)


@pytest.fixture(scope="module")
def R():
    return json.loads(KEY.read_text(encoding="utf-8"))


def _read(*parts, **kw):
    return pd.read_csv(Path(ROOT).joinpath(*parts), **kw)


def _same_fit(fit, want, rtol=GLM):
    np.testing.assert_allclose(np.asarray(fit.params, float), want["est"], rtol=rtol, atol=1e-9)
    np.testing.assert_allclose(np.asarray(fit.std_errors, float), want["se"], rtol=rtol)


def _same_posterior(fit, want, rename=None, sd_rtol=0.10):
    """Posterior of ``fit`` against rstanarm's, parameter by parameter."""
    rename = rename or {}
    draws = fit.draws
    for name, med, sd in zip(want["names"], want["median"], want["sd"]):
        if name == "sigma":
            mine = np.sqrt(draws["sigma2"])
        elif name == "reciprocal_dispersion":
            mine = 1.0 / draws["alpha"]
        else:
            mine = draws[rename.get(name, name)]
        assert abs(float(np.median(mine)) - med) < 0.1 * sd, (name, np.median(mine), med)
        assert float(np.std(mine)) == pytest.approx(sd, rel=sd_rtol), name


def test_elections_and_the_economy(R):
    hibbs = _read("ElectionsEconomy", "data", "hibbs.dat", sep=r"\s+")
    _same_fit(sp.regress("vote ~ growth", hibbs), R["hibbs"]["lm"], rtol=1e-10)
    fit = sp.bayes_regress("vote ~ growth", hibbs, draws=10000, burnin=1000, **KW)
    want = R["hibbs"]["stan"]
    _same_posterior(fit, want, {"(Intercept)": "Intercept"})
    # the summary the book prints: median and MAD SD
    mad = sp.mad_sd(fit.draws)
    assert mad["growth"] == pytest.approx(want["mad_sd"][1], rel=0.08)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = sp.loo(fit)
    assert loo.elpd == pytest.approx(want["elpd_loo"], abs=1.5)
    # the fixed default prior is not vague for an intercept near 46
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="not vague"):
        vague = sp.bayes_regress("vote ~ growth", hibbs, draws=4000, seed=1)
    assert vague.params["Intercept"] < fit.params["Intercept"] - 0.1


def test_kidiq_workflow(R):
    kidiq = _read("KidIQ", "data", "kidiq.csv")
    fit = sp.bayes_regress("kid_score ~ mom_hs + mom_iq", kidiq, draws=5000, burnin=1000, **KW)
    small = sp.bayes_regress("kid_score ~ mom_hs", kidiq, draws=5000, burnin=1000, **KW)
    want = R["kidiq"]
    _same_posterior(fit, want["stan"], {"(Intercept)": "Intercept"})
    loo = sp.loo(fit)
    assert loo.pareto_k.max() < 0.7
    assert loo.elpd == pytest.approx(want["stan"]["elpd_loo"], abs=1.5)
    assert loo.se_elpd == pytest.approx(want["stan"]["se_elpd_loo"], rel=0.02)
    assert loo.p == pytest.approx(want["stan"]["p_loo"], abs=0.5)
    cmp = sp.loo_compare({"both": fit, "hs": small})
    assert list(cmp.index) == ["both", "hs"]
    assert cmp.loc["hs", "elpd_diff"] == pytest.approx(want["elpd_diff"], abs=1.5)
    assert cmp.loc["hs", "se_diff"] == pytest.approx(want["se_diff"], rel=0.05)
    assert sp.bayes_r2(fit).estimate == pytest.approx(want["stan"]["bayes_r2_median"], abs=0.01)
    assert sp.loo_r2(fit, seed=1).estimate == pytest.approx(want["stan"]["loo_r2_mean"], abs=0.01)
    # ten-fold cross-validation estimates the same quantity
    kf = sp.kfold(
        sp.bayes_regress("kid_score ~ mom_hs + mom_iq", kidiq, draws=1000, burnin=500, **KW),
        k=10,
        seed=1,
    )
    assert kf.elpd == pytest.approx(loo.elpd, abs=6.0)


def test_arsenic_wells(R):
    wells = _read("Arsenic", "data", "wells.csv")
    want = R["wells"]
    mle = sp.logit("switch ~ dist100 + arsenic + educ4", wells, tol=1e-12)
    _same_fit(mle, want["glm"])
    # average predictive comparisons of chapter 14
    for var, (hi, lo) in {"dist100": (1, 0), "arsenic": (1.0, 0.5), "educ4": (3, 0)}.items():
        at = sp.margins_at(mle, wells, at={var: [lo, hi]})
        col = "margin" if "margin" in at.columns else at.columns[1]
        diff = float(at[col].iloc[1] - at[col].iloc[0])
        assert diff == pytest.approx(want["apc"][var], rel=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "switch ~ dist100 + arsenic", wells, model="logit", draws=10000, burnin=2000, **KW
        )
    _same_posterior(fit, want["stan"], {"(Intercept)": "Intercept"})
    assert sp.loo(fit).elpd == pytest.approx(want["stan"]["elpd_loo"], abs=1.5)
    assert sp.bayes_r2(fit).estimate == pytest.approx(want["stan"]["bayes_r2_median"], abs=0.005)
    # binned residuals against distance show no trend left over
    binned = sp.binned_residuals(mle, by=wells["dist100"], n_bins=40)
    assert binned.attrs["share_outside"] < 0.2


def test_roaches_counts_with_exposure(R):
    roaches = _read("Roaches", "data", "roaches.csv")
    roaches["roach100"] = roaches["roach1"] / 100
    roaches["log_exposure"] = np.log(roaches["exposure2"])
    want = R["roaches"]
    qp = sp.glm(
        "y ~ roach100 + treatment + senior", roaches, family="quasipoisson",
        offset="log_exposure", tol=1e-12,
    )
    _same_fit(qp, want["quasipoisson"])
    formula = "y ~ roach100 + treatment + senior"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = sp.bayes_regress(
            formula, roaches, model="negbin", exposure="exposure2",
            draws=15000, burnin=3000, **KW,
        )
        pois = sp.bayes_regress(
            formula, roaches, model="poisson", exposure="exposure2",
            draws=10000, burnin=3000, **KW,
        )
        loo_nb, loo_pois = sp.loo(nb), sp.loo(pois)
    _same_posterior(nb, want["negbin"], {"(Intercept)": "Intercept"}, sd_rtol=0.12)
    _same_posterior(pois, want["poisson"], {"(Intercept)": "Intercept"}, sd_rtol=0.12)
    assert loo_nb.elpd == pytest.approx(want["negbin"]["elpd_loo"], abs=2.0)
    # the Poisson model is the book's example of unreliable importance
    # weights: both sides flag it, and the estimate is only roughly stable
    assert loo_pois.bad_observations().size > 5 and want["poisson"]["max_k"] > 0.7
    assert loo_pois.elpd == pytest.approx(want["poisson"]["elpd_loo"], rel=0.05)
    assert loo_nb.elpd - loo_pois.elpd > 4000
    # the Poisson model cannot produce the zeros in the data
    assert sp.ppc(pois, "prop_zero", seed=1).p_value < 0.01
    check = sp.ppc(nb, "prop_zero", seed=1)
    assert check.observed == pytest.approx(want["prop_zero"])
    assert 0.05 < check.p_value < 0.95


def test_grouped_and_logical_outcomes(R):
    golf = _read("Golf", "data", "golf.txt", sep=r"\s+", skiprows=2)
    _same_fit(sp.glm("cbind(y, n - y) ~ x", golf, family="binomial", tol=1e-13), R["golf"])
    earnings = _read("Earnings", "data", "earnings.csv")
    _same_fit(sp.logit("(earn > 0) ~ height + male", earnings, tol=1e-12), R["earnings"]["positive"])
    positive = earnings.loc[earnings["earn"] > 0]
    fit = sp.regress("log(earn) ~ height + male + height:male", positive)
    _same_fit(fit, R["earnings"]["log"], rtol=1e-9)


def test_poststratification(R):
    poll = _read("Poststrat", "data", "poll.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            "vote ~ factor(pid)", poll, model="logit", draws=10000, burnin=2000, **KW
        )
    cells = pd.DataFrame(
        {"pid": ["Republican", "Democrat", "Independent"], "N": [0.33, 0.36, 0.31]}
    )
    out = sp.poststratify(fit, cells)
    want = R["poststrat"]
    assert out.estimate == pytest.approx(want["mean"], abs=0.1 * want["sd"])
    assert out.sd == pytest.approx(want["sd"], rel=0.08)
    assert abs(out.estimate - want["raw"]) > 0.005


def test_child_care_propensity_score_with_redundant_indicators(R):
    cc2 = _read("Childcare", "data", "cc2.csv")
    cc2 = cc2.rename(columns={c: c.replace(".", "_") for c in cc2.columns})
    want = R["childcare"]
    covs = [c.replace(".", "_") for c in want["covs"]]
    with pytest.warns(UserWarning, match="omitted because of collinearity") as caught:
        ps = sp.logit("treat ~ " + " + ".join(covs), cc2, tol=1e-12)
    text = " ".join(str(w.message) for w in caught)
    assert want["aliased"] == ["white", "college"]
    for name in want["aliased"]:
        assert f"note: {name} omitted" in text
    assert [o["variable"] for o in ps.model_info["omitted"]] == want["aliased"]
    _same_fit(ps, want)
    p = np.asarray(ps.predict())
    cc2["w"] = np.where(cc2["treat"] == 1, 1.0, p / (1 - p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sv = sp.svydesign(cc2, weights="w").glm("ppvtr_36 ~ treat + " + " + ".join(covs))
    assert sv.estimate["treat"] == pytest.approx(want["svy_treat"], rel=1e-6)
    assert sv.std_error["treat"] == pytest.approx(want["svy_treat_se"], rel=1e-6)


def test_sesame_street_instrument(R):
    sesame = _read("Sesame", "data", "sesame.csv")
    fit = sp.ivreg("postlet ~ watched | encouraged", sesame)
    np.testing.assert_allclose(
        [fit.params["Intercept"], fit.params["watched"]], R["sesame"]["est"], rtol=1e-9
    )
    np.testing.assert_allclose(
        [fit.std_errors["Intercept"], fit.std_errors["watched"]], R["sesame"]["se"], rtol=1e-9
    )
