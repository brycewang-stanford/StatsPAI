"""The weight study beyond OLS: reproducible, and what the warnings rest on.

``tests/reliability/extreme_weights_models.py`` simulates the coverage of
the 95% interval of a weighted fixed-effects slope and a weighted Poisson
slope as the weights grow more dispersed. This file checks the stored
design, that one cell of each model reproduces its stored prefix, that
the statements in ``tests/reliability/README.md`` hold with their Monte
Carlo error, and that the entry points raise the warnings the study
motivates, record the diagnostics, and stay silent on equal weights.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._agent_summary import (
    effective_n_clusters,
    kish_effective_n,
    weighted_effective_n_clusters,
)
from statspai.exceptions import AssumptionWarning

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "extreme_weights_models.py"
RESULTS = ROOT / "reliability" / "extreme_weights_models_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("extreme_weights_models", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cell(results, model, size, sigma):
    (cell,) = [
        c
        for c in results["cells"]
        if (c["model"], c["size"], c["sigma"]) == (model, size, sigma)
    ]
    return cell


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 2000 and results["level"] == 0.95
    keys = {(c["model"], c["size"], c["sigma"]) for c in results["cells"]}
    assert keys == {
        (m, n, s) for m in study.MODELS for n in study.SIZES[m] for s in study.SIGMAS
    }


@pytest.mark.parametrize("model,size", [("panel_fe", 50), ("poisson", 200)])
def test_a_cell_reproduces_its_stored_prefix(study, results, model, size):
    fresh = study.run_cell(model, size, 2.0, study.PREFIX)
    stored = _cell(results, model, size, 2.0)
    for v in study.VARIANCES[model]:
        assert fresh[v]["prefix_hits"] == stored[v]["prefix_hits"], v


def test_equal_weights_cover_at_the_nominal_level(study, results):
    for model in study.MODELS:
        for size in study.SIZES[model]:
            cell = _cell(results, model, size, 0.0)
            for v in study.VARIANCES[model]:
                cov, se = cell[v]["coverage"], cell[v]["mc_se"]
                assert abs(cov - 0.95) < 3 * se, (model, size, v, cov)


def test_classical_intervals_fail_under_sampling_weights(study, results):
    for model in study.MODELS:
        for size in study.SIZES[model]:
            for sigma, ceiling in ((1.0, 0.82), (2.0, 0.56)):
                cov = _cell(results, model, size, sigma)["classical"]["coverage"]
                assert cov < ceiling, (model, size, sigma, cov)


def test_robust_intervals_are_short_only_in_small_effective_samples(results):
    # Poisson: 84% at a Kish size of 17, 88% at 53, nominal at 382.
    small = _cell(results, "poisson", 200, 2.0)
    mid = _cell(results, "poisson", 1000, 2.0)
    large = _cell(results, "poisson", 1000, 1.0)
    assert small["kish_median"] < 20 and small["robust"]["coverage"] < 0.87
    assert mid["kish_median"] < 60 and mid["robust"]["coverage"] < 0.91
    assert large["kish_median"] > 300
    assert abs(large["robust"]["coverage"] - 0.95) < 3 * large["robust"]["mc_se"]


def test_cluster_intervals_shorten_with_the_weight_effective_cluster_count(results):
    by_kish = sorted(
        (
            (c["kish_median"], c["cluster"]["coverage"])
            for c in results["cells"]
            if c["model"] == "panel_fe"
        )
    )
    # 7 -> 78%, 17 -> 87%, 23 -> 91%, 50+ -> nominal within error
    assert [round(k) for k, _ in by_kish][:3] == [7, 17, 23]
    assert by_kish[0][1] < 0.80 and by_kish[1][1] < 0.89 and by_kish[2][1] < 0.925
    for kish, cov in by_kish[3:]:
        assert kish >= 50 and cov > 0.93, (kish, cov)


# --------------------------------------------------------------------- #
# The diagnostics in the entry points
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(1)
    G, T = 100, 5
    unit = np.repeat(np.arange(G), T)
    w_unit = np.exp(rng.normal(scale=2.0, size=G))
    x = rng.normal(size=G * T)
    return pd.DataFrame(
        {
            "y": 0.5 * x + rng.normal(size=G * T),
            "c": rng.poisson(np.exp(0.2 + 0.5 * x)),
            "x": x,
            "id": unit,
            "t": np.tile(np.arange(T), G),
            "w": w_unit[unit],
            "one": 1.0,
        }
    )


def _fit(call):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = call()
    msgs = [str(m.message) for m in rec if issubclass(m.category, AssumptionWarning)]
    if hasattr(res, "weight_info"):
        info = res.weight_info or {}
    else:
        info = res.model_info
    return info, msgs


CALLS = {
    "panel": lambda d, w, **kw: sp.panel(
        d, "y ~ x", entity="id", time="t", method="fe", weights=w, **kw
    ),
    "hdfe_ols": lambda d, w, **kw: sp.hdfe_ols("y ~ x | id", d, weights=w, **kw),
    "poisson": lambda d, w, **kw: sp.poisson("c ~ x", d, weights=w, **kw),
}


def test_weighted_cluster_count_generalises_the_size_based_one(frame):
    keys = frame["id"].to_numpy()
    assert weighted_effective_n_clusters(frame["one"], keys) == pytest.approx(
        effective_n_clusters(pd.Series(keys))
    )
    w_unit = frame.groupby("id")["w"].sum().to_numpy()
    assert weighted_effective_n_clusters(frame["w"], keys) == pytest.approx(
        kish_effective_n(w_unit)
    )


@pytest.mark.parametrize("name", sorted(CALLS))
def test_default_variance_warns_on_dispersed_weights(frame, name):
    info, msgs = _fit(lambda: CALLS[name](frame, "w"))
    assert info["n_effective_weights"] == pytest.approx(kish_effective_n(frame["w"]))
    assert any("weights are dispersed" in m for m in msgs), msgs


@pytest.mark.parametrize("name", sorted(CALLS))
def test_cluster_variance_warns_on_few_weight_effective_clusters(frame, name):
    info, msgs = _fit(lambda: CALLS[name](frame, "w", cluster="id"))
    expected = weighted_effective_n_clusters(frame["w"], frame["id"])
    assert expected < 30
    assert info["n_clusters_effective_weights"] == pytest.approx(expected)
    assert any("weights concentrate on a few" in m for m in msgs), msgs


@pytest.mark.parametrize("name", ["panel", "poisson"])
def test_robust_variance_warns_in_a_small_effective_sample(frame, name):
    _, msgs = _fit(lambda: CALLS[name](frame, "w", robust="robust"))
    assert any("carry most of the weight" in m for m in msgs), msgs


def test_ppmlhdfe_and_feols_carry_the_same_diagnostic(frame):
    info, msgs = _fit(lambda: sp.ppmlhdfe("c ~ x | id", frame, weights="w"))
    assert info["n_effective_weights"] == pytest.approx(kish_effective_n(frame["w"]))
    assert any("carry most of the weight" in m for m in msgs), msgs
    pytest.importorskip("pyfixest")
    info, msgs = _fit(
        lambda: sp.feols("y ~ x | id", frame, weights="w", vcov={"CRV1": "id"})
    )
    assert info["n_clusters_effective_weights"] < 30
    assert any("weights concentrate on a few" in m for m in msgs), msgs


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("kw", [{}, {"cluster": "id"}])
def test_equal_weights_are_silent(frame, name, kw):
    info, msgs = _fit(lambda: CALLS[name](frame, "one", **kw))
    assert info["n_effective_weights"] == pytest.approx(len(frame))
    assert not [m for m in msgs if "weight" in m.lower()], msgs


@pytest.mark.parametrize("name", sorted(CALLS))
def test_unweighted_fits_record_nothing(frame, name):
    info, msgs = _fit(lambda: CALLS[name](frame, None))
    assert "n_effective_weights" not in (info or {})
    assert not [m for m in msgs if "weight" in m.lower()], msgs
