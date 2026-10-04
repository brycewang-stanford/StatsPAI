"""The third weight study: reproducible, and what the warnings rest on.

``tests/reliability/extreme_weights_fast.py`` repeats the weight question
for ``sp.fast.feols``, ``sp.fast.fepois`` and ``sp.nbreg``. This file
checks the stored design, that one cell of each model reproduces its
stored prefix, that the statements in ``tests/reliability/README.md``
hold, and that the three entry points record the diagnostics and warn.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pytest

import statspai as sp
from statspai.core._agent_summary import kish_effective_n
from statspai.exceptions import AssumptionWarning

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "extreme_weights_fast.py"
RESULTS = ROOT / "reliability" / "extreme_weights_fast_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("extreme_weights_fast", SCRIPT)
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
    refused = sum(
        sum(c[v]["refused"].values())
        for c in results["cells"]
        for v in study.VARIANCES[c["model"]]
    )
    # one negative-binomial fit of 84,000 did not converge (one observation
    # held 80% of the weight); it is counted, not dropped
    assert refused == 1


@pytest.mark.parametrize(
    "model,size", [("fast_feols", 50), ("fast_fepois", 50), ("nbreg", 200)]
)
def test_a_cell_reproduces_its_stored_prefix(study, results, model, size):
    fresh = study.run_cell(model, size, 2.0, study.PREFIX)
    stored = _cell(results, model, size, 2.0)
    for v in study.VARIANCES[model]:
        assert fresh[v]["prefix_hits"] == stored[v]["prefix_hits"], v


def test_default_variance_fails_under_sampling_weights_in_all_three(study, results):
    for model in study.MODELS:
        default = study.VARIANCES[model][0]
        for size in study.SIZES[model]:
            for sigma, ceiling in ((1.0, 0.82), (2.0, 0.56)):
                cov = _cell(results, model, size, sigma)[default]["coverage"]
                assert cov < ceiling, (model, size, sigma, cov)


def test_robust_variance_is_short_only_in_small_effective_samples(results):
    for model, v in (("fast_feols", "hc1"), ("fast_fepois", "hc1")):
        assert _cell(results, model, 50, 2.0)[v]["coverage"] < 0.90
        big = _cell(results, model, 200, 1.0)[v]
        assert abs(big["coverage"] - 0.95) < 3 * big["mc_se"]
    assert _cell(results, "nbreg", 200, 2.0)["robust"]["coverage"] < 0.87
    assert _cell(results, "nbreg", 1000, 2.0)["robust"]["coverage"] < 0.90
    big = _cell(results, "nbreg", 1000, 1.0)["robust"]
    assert abs(big["coverage"] - 0.95) < 3 * big["mc_se"]


def test_cluster_variance_follows_the_weight_effective_cluster_count(results):
    for model in ("fast_feols", "fast_fepois"):
        rows = sorted(
            (c["kish_median"], c["cr1"]["coverage"])
            for c in results["cells"]
            if c["model"] == model
        )
        assert [round(k) for k, _ in rows][:3] == [7, 17, 23] or [
            round(k) for k, _ in rows
        ][:3] == [7, 18, 23]
        assert rows[0][1] < 0.81 and rows[1][1] < 0.88 and rows[2][1] < 0.925
        for kish, cov in rows[3:]:
            assert kish >= 50 and cov > 0.93, (model, kish, cov)


# --------------------------------------------------------------------- #
# The diagnostics in the entry points
# --------------------------------------------------------------------- #


def _fit(call):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = call()
    msgs = [str(m.message) for m in rec if issubclass(m.category, AssumptionWarning)]
    if hasattr(res, "weight_info"):
        return res.weight_info or {}, msgs
    return res.model_info, msgs


@pytest.mark.parametrize("model", ["fast_feols", "fast_fepois"])
def test_fast_entry_points_carry_the_diagnostic(study, model):
    df, _ = study.draw(model, 100, 2.0, 1)
    fn = sp.fast.feols if model == "fast_feols" else sp.fast.fepois
    info, msgs = _fit(lambda: fn("y ~ x + z | id", df, weights="w", vcov="iid"))
    assert info["n_effective_weights"] < 0.5 * len(df)
    assert any("weights are dispersed" in m for m in msgs), msgs
    info, msgs = _fit(lambda: fn("y ~ x + z | id", df, weights="w", vcov="hc1"))
    assert any("carry most of the weight" in m for m in msgs), msgs
    info, msgs = _fit(
        lambda: fn("y ~ x + z | id", df, weights="w", vcov="cr1", cluster="id")
    )
    assert info["n_clusters_effective_weights"] < 30
    assert any("weights concentrate on a few" in m for m in msgs), msgs
    # equal weights and no weights are silent
    df["one"] = 1.0
    info, msgs = _fit(lambda: fn("y ~ x + z | id", df, weights="one", vcov="iid"))
    assert not [m for m in msgs if "weight" in m.lower()], msgs
    info, msgs = _fit(lambda: fn("y ~ x + z | id", df, vcov="iid"))
    assert info == {} and not [m for m in msgs if "weight" in m.lower()]


def test_nbreg_carries_the_diagnostic(study):
    df, _ = study.draw("nbreg", 500, 2.0, 1)
    info, msgs = _fit(lambda: sp.nbreg("y ~ x + z", df, weights="w"))
    assert info["n_effective_weights"] == pytest.approx(kish_effective_n(df["w"]))
    assert any("frequencies" in m for m in msgs), msgs
    _, msgs = _fit(lambda: sp.nbreg("y ~ x + z", df, weights="w", robust="robust"))
    assert any("carry most of the weight" in m for m in msgs), msgs
    info, msgs = _fit(lambda: sp.nbreg("y ~ x + z", df))
    assert "n_effective_weights" not in info
    assert not [m for m in msgs if "weight" in m.lower()]
