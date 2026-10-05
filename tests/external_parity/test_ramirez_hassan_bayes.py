"""Ramirez-Hassan, *Introduction to Bayesian Econometrics* (2026): the
examples of chapters 6 and 9 on the book's own data.

The book fits its models with ``MCMCpack`` and ``bayesm``. The answer key
is ``data/ramirez_hassan_bayes_R.json``, long runs of those samplers with
the book's priors (``ramirez_hassan_bayes_reference.R`` next to this file).
Here the same posteriors are sampled with ``sp.bayes_regress`` and
``sp.bayes_mixed``.

Neither the book's programs nor its data (GPL-3,
https://github.com/besmarter/BSTApp) are redistributed. Point
``STATSPAI_RAMIREZ_HASSAN_DIR`` at the ``DataApp`` folder:

    STATSPAI_RAMIREZ_HASSAN_DIR=/path/to/BSTApp/DataApp \\
        pytest tests/external_parity/test_ramirez_hassan_bayes.py

It is skipped otherwise. What the pass found is in
``docs/dev/2026-10-06-ramirez-hassan-bayesian-econometrics-review.md``.

What this is evidence of. Two correct samplers of one posterior differ by
Monte Carlo error, so this is a stochastic screen, not a parity claim: a
posterior mean must be within four combined Monte Carlo standard errors of
the reference (plus 2 percent of a posterior sd for the reference's own
numerical noise), a posterior sd within 6 percent. The samplers themselves
are verified against exact posteriors in
``tests/reference_parity/test_bayes_regress_exact_posterior.py`` and
``test_bayes_mixed_exact_posterior.py``.
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

ROOT = os.environ.get("STATSPAI_RAMIREZ_HASSAN_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "1ValueFootballPlayers.csv").is_file(),
    reason="set STATSPAI_RAMIREZ_HASSAN_DIR to the book's BSTApp/DataApp folder",
)

PLAYERS = "Perf + Age + Age2 + NatTeam + Goals + Exp + Exp2"
HEALTH = "SHI + Female + Age + Age2 + Est2 + Est3 + Fair + Good + Excellent"
ORDERED = HEALTH + " + PriEd + HighEd + VocEd + UnivEd"


@pytest.fixture(scope="module")
def R():
    path = Path(__file__).parent / "data" / "ramirez_hassan_bayes_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def players() -> pd.DataFrame:
    d = pd.read_csv(Path(ROOT) / "1ValueFootballPlayers.csv")
    d["lv"] = np.log(d["Value"])
    d["lvc"] = np.log(d["ValueCens"])
    return d


@pytest.fixture(scope="module")
def health() -> pd.DataFrame:
    return pd.read_csv(Path(ROOT) / "2HealthMed.csv")


def agree(fit, ref, ours=None, theirs=None, sd_tol=0.06):
    """Posterior means and sds of ``fit`` against a reference posterior."""
    theirs = list(ref["names"]) if theirs is None else theirs
    ours = theirs if ours is None else ours
    pos = [ref["names"].index(t) for t in theirs]
    r_mean = np.array(ref["mean"])[pos]
    r_sd = np.array(ref["sd"])[pos]
    r_mcse = np.array(ref["mcse"])[pos]
    t = fit.table.loc[ours]
    gap = np.abs(t["mean"].to_numpy() - r_mean)
    allow = 4 * np.sqrt(t["mcse"].to_numpy() ** 2 + r_mcse**2) + 0.02 * r_sd
    assert (gap <= allow).all(), pd.DataFrame(
        {"ours": t["mean"].to_numpy(), "ref": r_mean, "gap": gap, "allow": allow}, index=ours
    )
    ratio = t["sd"].to_numpy() / r_sd
    assert np.abs(ratio - 1).max() < sd_tol, dict(zip(ours, ratio.round(3)))


def test_linear_model_market_value_of_players(R, players):
    """Section 6.1, ``MCMCpack::MCMCregress``."""
    fit = sp.bayes_regress(
        f"lv ~ {PLAYERS}", players, prior_var=1000.0, sigma2_prior=(0.001, 0.001),
        draws=40000, burnin=2000, seed=1,
    )
    ref = R["linear"]
    names = ["Intercept"] + PLAYERS.split(" + ") + ["sigma2"]
    agree(fit, ref, ours=names, theirs=list(ref["names"]))
    # the book's reading: playing for the national team raises value by
    # exp(0.85) - 1, about 134 percent
    assert np.exp(fit.params["NatTeam"]) - 1 == pytest.approx(1.34, abs=0.06)


def test_tobit_censored_market_value(R, players):
    """Section 6.8, ``MCMCpack::MCMCtobit``, censored below one million."""
    fit = sp.bayes_regress(
        f"lvc ~ {PLAYERS}", players, model="tobit", lower=float(np.log(1e6)),
        prior_var=1000.0, sigma2_prior=(0.001, 0.001), draws=40000, burnin=2000, seed=2,
    )
    assert fit.model_info["n_left_censored"] == R["tobit_n_censored"]
    ref = R["tobit"]
    names = ["Intercept"] + PLAYERS.split(" + ") + ["sigma2"]
    agree(fit, ref, ours=names, theirs=list(ref["names"]))


@pytest.mark.parametrize("q", [50, 90])
def test_quantile_regression_market_value(R, players, q):
    """Section 6.9, ``MCMCpack::MCMCquantreg``: asymmetric Laplace, scale 1."""
    fit = sp.bayes_regress(
        f"lv ~ {PLAYERS}", players, model="quantile", quantile=q / 100, scale=1.0,
        prior_var=1000.0, draws=40000, burnin=2000, seed=3,
    )
    ref = R[f"quantile_{q}"]
    agree(fit, ref, ours=["Intercept"] + PLAYERS.split(" + "), theirs=list(ref["names"]))


def test_probit_hospitalisation(R, health):
    """Section 6.3, ``bayesm::rbprobitGibbs`` with a N(0, I) prior."""
    fit = sp.bayes_regress(
        f"Hosp ~ {HEALTH}", health, model="probit", prior_var=1.0,
        draws=20000, burnin=2000, seed=4,
    )
    agree(fit, R["probit"])
    # self-rated health is the dominant predictor in both
    assert fit.prob("Excellent < 0") > 0.999


def test_logit_hospitalisation(R, health):
    """Section 6.2's sampler on the same data, ``MCMCpack::MCMClogit``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            f"Hosp ~ {HEALTH}", health, model="logit", prior_var=100.0,
            draws=20000, burnin=2000, thin=5, seed=5,
        )
    ref = R["logit"]
    agree(fit, ref, ours=["Intercept"] + HEALTH.split(" + "), theirs=list(ref["names"]))


def test_ordered_probit_preventive_visits(R, health):
    """Section 6.6, ``bayesm::rordprobitGibbs``.

    The book passes a design without a constant to a sampler whose first
    cutpoint is fixed at zero, which forces the first threshold of the
    index to zero. With free cutpoints (a constant in bayesm, always in
    ``sp.bayes_regress``) the two samplers agree with each other and with
    maximum likelihood; the book's restricted version does not agree with
    maximum likelihood.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.bayes_regress(
            f"MedVisPrevOr ~ {ORDERED}", health, model="oprobit", prior_var=1000.0,
            draws=20000, burnin=3000, seed=6,
        )
    slopes = ORDERED.split(" + ")
    cuts = [f"cut{j}" for j in range(1, 6)]
    agree(fit, R["oprobit_constant"], ours=slopes, theirs=slopes, sd_tol=0.1)
    agree(fit, R["oprobit_cuts"], ours=cuts, theirs=cuts, sd_tol=0.1)
    mle = sp.oprobit(f"MedVisPrevOr ~ {ORDERED}", health)
    est = {k: float(v) for k, v in mle.params.items()}
    se = {k: float(v) for k, v in mle.std_errors.items()}
    cut_names = [k for k in est if k not in slopes]
    assert len(cut_names) == 5
    for name in slopes:
        assert abs(fit.params[name] - est[name]) < 0.35 * se[name], name
    for ours, theirs in zip(cuts, cut_names):
        assert abs(fit.params[ours] - est[theirs]) < 0.35 * se[theirs], ours
    # the book's specification is a different, restricted model
    book = dict(zip(R["oprobit_book"]["names"], R["oprobit_book"]["mean"]))
    worst = max(abs(book[name] - est[name]) / se[name] for name in slopes)
    assert worst > 3.0


def test_hierarchical_model_public_capital(R):
    """Section 9.1, ``MCMCpack::MCMChregress``, random intercept by state.

    Two things are wrong with the reference posterior, neither of them ours.

    The book's prior for the variance of the state effects,
    ``InvWishart(5, 5 * 1)``, is on the wrong scale for a log outcome. It
    supplies nine tenths of the sum of squares, and the variance component
    comes out fourteen times the restricted maximum likelihood estimate.
    Both samplers reproduce that; ours says so in a warning.

    ``MCMChregress`` reports a posterior sd of the intercept of 0.008. With
    48 states and a state-effect variance of 0.106 no estimator can know
    the intercept better than ``sqrt(0.106 / 48) = 0.047``. Its draws of
    the fixed effects have the spread of the conditional distribution
    given the state effects. Ours has the marginal spread, and with a prior
    on the right scale it agrees with ``lme4::lmer``.
    """
    d = pd.read_csv(Path(ROOT) / "8PublicCap.csv")
    for c in ("gsp", "pcap", "pc", "emp"):
        d[f"l{c}"] = np.log(d[c])
    formula = "lgsp ~ lpcap + lpc + lemp + unemp"
    ours = ["Intercept", "lpcap", "lpc", "lemp", "unemp"]
    ref = R["hier"]
    names = list(ref["names"])
    r_mean = dict(zip(names, ref["mean"]))
    r_sd = dict(zip(names, ref["sd"]))
    s2 = next(n for n in names if n.startswith("sigma2"))
    vcv = next(n for n in names if n.startswith("VCV"))

    # 1. the book's prior
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="driving the variance"):
        book = sp.bayes_mixed(
            formula, d, group="id", prior_var=1e6, re_prior=(5.0, 1.0),
            sigma2_prior=(0.002, 0.002), draws=20000, burnin=3000, seed=7,
        )
    assert book.model_info["re_prior_share"]["Intercept"] > 0.85
    assert book.params["sigma2"] == pytest.approx(r_mean[s2], rel=0.02)
    assert book.params["var(Intercept)"] == pytest.approx(r_mean[vcv], rel=0.05)
    assert book.std_errors["var(Intercept)"] == pytest.approx(r_sd[vcv], rel=0.1)
    for a, b in zip(ours, [n for n in names if n.startswith("beta.")]):
        # same centre ...
        assert abs(book.params[a] - r_mean[b]) < 0.1 * book.std_errors[a], a
    # ... but the reference's spread is below what the model allows
    floor = np.sqrt(r_mean[vcv] / 48)
    assert r_sd["beta.(Intercept)"] < 0.25 * floor
    assert book.std_errors["Intercept"] > floor

    # 2. a prior on the scale of the state effects, against lme4
    lmer = R["hier_lmer"]
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.exceptions.StatsPAIWarning)
        fit = sp.bayes_mixed(
            formula, d, group="id", prior_var=1e6, re_prior=(3.0, 0.01),
            draws=20000, burnin=3000, seed=7,
        )
    assert fit.model_info["re_prior_share"]["Intercept"] < 0.15
    est = np.array(lmer["est"])
    se = np.array(lmer["se"])
    assert np.abs(fit.params[ours].to_numpy() - est).max() < 0.1 * se.max()
    assert np.abs((fit.params[ours].to_numpy() - est) / se).max() < 0.1
    assert np.abs(fit.std_errors[ours].to_numpy() / se - 1).max() < 0.08
    assert fit.params["sigma2"] == pytest.approx(lmer["sigma2"], rel=0.02)
    assert fit.params["var(Intercept)"] == pytest.approx(lmer["var_id"], rel=0.15)
