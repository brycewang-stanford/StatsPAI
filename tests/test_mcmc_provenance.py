"""``statspai.mcmc`` samplers announce dropped rows and record seed + data.

``sp.bayes_regress`` dropped 5 NaN rows without a word and its result had
no provenance record, so a posterior could not be tied to its sample or
rerun from the result (rOpenSci audit probes P5, P14). The samplers now
carry the same record ``sp.dml`` does.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning
from statspai.output._lineage import compute_data_hash, records_provenance


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "g": np.repeat(np.arange(20), 10)})
    df["y"] = 1 + df.x + rng.normal(size=n)
    return df


SAMPLERS = {
    "bayes_regress": lambda d, **k: sp.bayes_regress(
        "y ~ x", d, draws=200, burnin=50, **k
    ),
    "bayes_mixed": lambda d, **k: sp.bayes_mixed(
        "y ~ x", d, group="g", draws=200, burnin=50, **k
    ),
    "bayes_shrink": lambda d, **k: sp.bayes_shrink(
        "y ~ x", d, draws=200, burnin=50, **k
    ),
    "bma": lambda d, **k: sp.bma("y ~ x", d, **k),
    "bayes_bootstrap": lambda d, **k: sp.bayes_bootstrap("y ~ x", d, **k),
}


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_dropped_rows_are_announced(frame, name):
    d = frame.copy()
    d.loc[:4, "x"] = np.nan
    with pytest.warns(AssumptionWarning, match="5 of 200 rows dropped") as rec:
        res = SAMPLERS[name](d, seed=1)
    msg = next(w.message for w in rec if isinstance(w.message, AssumptionWarning))
    assert msg.diagnostics["missing_by_column"] == {"x": 5}
    assert res.n_obs == 195


@pytest.mark.parametrize("name", sorted(SAMPLERS))
def test_seed_and_data_are_recorded(frame, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SAMPLERS[name](frame, seed=7)
    prov = res.result_card()["provenance"]
    assert prov["function"] == f"sp.{name}"
    assert prov["seed"] == 7 and prov["reproducible"] is True
    assert prov["data_hash"] == compute_data_hash(frame)
    assert prov["data_shape"] == [200, 3]


def test_default_seed_is_recorded_as_not_reproducible(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.bayes_regress("y ~ x", frame, draws=100, burnin=20)
    prov = res.result_card()["provenance"]
    assert prov["seed"] is None
    assert prov["reproducible"] is False
    assert prov["seed_source"] == "call"


def test_seeded_rerun_reproduces_the_draws(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.bayes_regress("y ~ x", frame, draws=100, burnin=20, seed=3)
        b = sp.bayes_regress("y ~ x", frame, draws=100, burnin=20, seed=3)
    np.testing.assert_array_equal(a.params.to_numpy(), b.params.to_numpy())


def test_record_defaults_only_fills_what_was_not_passed():
    class R:
        pass

    @records_provenance("sp.demo", record_defaults=("seed", "absent"))
    def fit(data, seed=None, k=1):
        return R()

    df = pd.DataFrame({"a": [1.0]})
    assert fit(df)._provenance.params["seed"] is None
    assert fit(df, seed=5)._provenance.params["seed"] == 5
    assert "k" not in fit(df)._provenance.params
    assert "absent" not in fit(df)._provenance.params
