"""Repeated calls on identical data must return identical estimates.

A stochastic estimator whose default seed is ``None`` is a silent
reproducibility hole: the user runs the same script twice, gets two
different numbers, and has no way to tell which one to put in the paper.
``sp.aipw`` shipped that way until 1.21.0 — its cross-fitting split came
from ``np.random.default_rng(None)``, which draws from OS entropy and
therefore ignores ``np.random.seed(...)`` too, so even pinning the global
RNG did not make it reproducible.

These tests pin the convention for every stochastic estimator we expose:
calling twice with default arguments returns the same number, and the
seed that produced it is recorded on the result.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp

COVARIATES = [
    "age",
    "educ",
    "black",
    "hispanic",
    "married",
    "nodegree",
    "re74",
    "re75",
]


@pytest.fixture(scope="module")
def lalonde():
    return sp.datasets.nsw_lalonde(simulated=False)


def _estimate(result) -> float:
    for attr in ("estimate", "att", "ate"):
        value = getattr(result, attr, None)
        if value is not None:
            return float(value)
    raise AttributeError(f"no estimate on {type(result).__name__}")


# --------------------------------------------------------------------- #
# sp.aipw — the estimator that actually regressed
# --------------------------------------------------------------------- #


def test_aipw_is_deterministic_by_default(lalonde):
    """Default sp.aipw must return the same estimate on every call."""
    runs = [
        _estimate(sp.aipw(lalonde, y="re78", treat="treat", covariates=COVARIATES))
        for _ in range(3)
    ]
    assert len(set(runs)) == 1, (
        f"sp.aipw returned {runs} across three identical calls; the "
        "cross-fitting seed is not pinned"
    )


def test_aipw_default_survives_a_perturbed_global_rng(lalonde):
    """The default must not depend on the caller's global numpy seed.

    ``np.random.default_rng(None)`` ignores ``np.random.seed`` entirely,
    so a seed that only *looks* pinned would still drift here.
    """
    np.random.seed(1)
    first = _estimate(sp.aipw(lalonde, y="re78", treat="treat", covariates=COVARIATES))
    np.random.seed(999)
    _ = np.random.random(50)  # advance the global stream
    second = _estimate(sp.aipw(lalonde, y="re78", treat="treat", covariates=COVARIATES))
    assert first == second


def test_aipw_records_the_seed_it_used(lalonde):
    """Provenance: the result says which seed produced it."""
    result = sp.aipw(lalonde, y="re78", treat="treat", covariates=COVARIATES)
    assert result.model_info["seed"] == 42

    explicit = sp.aipw(lalonde, y="re78", treat="treat", covariates=COVARIATES, seed=7)
    assert explicit.model_info["seed"] == 7


def test_aipw_seed_none_still_opts_into_randomness(lalonde):
    """seed=None remains available for fold-sensitivity studies."""
    runs = [
        _estimate(
            sp.aipw(
                lalonde,
                y="re78",
                treat="treat",
                covariates=COVARIATES,
                seed=None,
            )
        )
        for _ in range(4)
    ]
    assert len(set(runs)) > 1, (
        "seed=None should draw a fresh fold split per call; got "
        f"identical estimates {runs}"
    )
    result = sp.aipw(
        lalonde,
        y="re78",
        treat="treat",
        covariates=COVARIATES,
        seed=None,
    )
    assert result.model_info["seed"] is None


def test_aipw_distinct_seeds_give_distinct_folds(lalonde):
    """A pinned seed must actually reach the fold assignment."""
    a = _estimate(
        sp.aipw(lalonde, y="re78", treat="treat", covariates=COVARIATES, seed=1)
    )
    b = _estimate(
        sp.aipw(lalonde, y="re78", treat="treat", covariates=COVARIATES, seed=2)
    )
    assert a != b, "seed is accepted but ignored"


# --------------------------------------------------------------------- #
# The estimand-first DSL inherits whatever its plan dispatches to
# --------------------------------------------------------------------- #


def _question(data):
    q = sp.causal_question(
        treatment="treat",
        outcome="re78",
        data=data,
        population="Lalonde NSW treated + PSID comparison",
        estimand="ATT",
        design="selection_on_observables",
        covariates=COVARIATES,
    )
    q.identify()
    return q


def test_causal_question_estimate_is_reproducible(lalonde):
    """sp.causal_question(...).estimate() must not drift between runs.

    This is the user-visible symptom: the DSL resolves to cross-fitted
    AIPW, so an unpinned AIPW seed made the headline number of an
    entire pipeline irreproducible.
    """
    runs = [_estimate(_question(lalonde).estimate()) for _ in range(3)]
    assert len(set(runs)) == 1, (
        f"question.estimate() returned {runs} across three identical "
        "runs; the dispatched estimator's seed is not pinned"
    )


def test_causal_question_estimate_honours_an_explicit_seed(lalonde):
    """An explicit seed still flows through the dispatcher."""
    a = _estimate(_question(lalonde).estimate(seed=1))
    b = _estimate(_question(lalonde).estimate(seed=1))
    c = _estimate(_question(lalonde).estimate(seed=2))
    assert a == b
    assert a != c


# --------------------------------------------------------------------- #
# The rest of the stochastic surface — guard against future regressions
# --------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, call",
    [
        (
            "dml",
            lambda d: sp.dml(d, y="re78", treat="treat", covariates=COVARIATES),
        ),
        (
            "tmle",
            lambda d: sp.tmle(d, y="re78", treat="treat", covariates=COVARIATES),
        ),
        (
            "metalearner",
            lambda d: sp.metalearner(
                d,
                y="re78",
                treat="treat",
                covariates=COVARIATES,
                learner="dr",
            ),
        ),
    ],
)
def test_stochastic_estimators_are_deterministic_by_default(lalonde, name, call):
    """Every estimator with a random component defaults to reproducible."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = [_estimate(call(lalonde)) for _ in range(2)]
    assert len(set(runs)) == 1, f"sp.{name} returned {runs} across two identical calls"
