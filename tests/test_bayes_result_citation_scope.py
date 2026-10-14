"""A Bayesian fit cites the papers of the model it fitted.

``BayesRegressResult`` declared every key of its family as a class tuple,
and the result contract reports the first one. Every fit (a Bayesian
bootstrap, a horseshoe, a normal regression) therefore said
``citation_key: albert1993bayesian``, the probit sampler, while its own
``cite()`` gave the right answer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.filterwarnings(
    "ignore::statspai.exceptions.ConvergenceWarning"
)

import statspai as sp
from statspai.mcmc.regress import _CITATIONS, BayesRegressResult


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 1 + 0.5 * x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "z": z, "d": (y > 1).astype(int)})


FITS = {
    "bayes_bootstrap": lambda f: sp.bayes_bootstrap("y ~ x", f, draws=100, seed=1),
    "normal": lambda f: sp.bayes_regress(
        "y ~ x", f, draws=200, burnin=100, seed=1, prior="weakly_informative"
    ),
    "probit": lambda f: sp.bayes_regress(
        "d ~ x",
        f,
        model="probit",
        draws=200,
        burnin=100,
        seed=1,
        prior="weakly_informative",
    ),
    "horseshoe": lambda f: sp.bayes_shrink(
        "y ~ x + z", f, prior="horseshoe", draws=200, burnin=100, seed=1
    ),
}


@pytest.mark.parametrize("model", list(FITS))
def test_payload_cites_the_fitted_model(frame, model):
    fit = FITS[model](frame)
    assert fit.model == model
    expected = list(_CITATIONS[model])
    payload = fit.to_dict(detail="agent")
    assert payload["citation_key"] == expected[0]
    assert payload.get("citation_keys", expected) == expected
    assert fit.cite().split("\n") == expected
    assert payload["method"].startswith(f"Bayesian {model} regression")


def test_class_still_lists_the_whole_family():
    family = {k for keys in _CITATIONS.values() for k in keys}
    assert set(BayesRegressResult._citation_keys) == family


def test_subclass_with_its_own_keys_keeps_them():
    from statspai.mcmc.mixed import BayesMixedResult

    assert BayesMixedResult._citation_keys == ("chib1999mcmc",)
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(20), 5)
    df = pd.DataFrame({"x": rng.normal(size=100), "g": g})
    df["y"] = 1 + 0.5 * df["x"] + rng.normal(size=20)[g] + rng.normal(size=100)
    fit = sp.bayes_mixed("y ~ x", df, group="g", draws=200, burnin=100, seed=1)
    assert fit.to_dict(detail="minimal")["citation_key"] == "chib1999mcmc"


def test_every_key_is_in_paper_bib():
    from statspai._bibpath import master_bib_path

    bib = master_bib_path().read_text(encoding="utf-8")
    for keys in _CITATIONS.values():
        for key in keys:
            assert "{" + key + "," in bib, key
