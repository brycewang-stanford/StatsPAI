"""The fourth weight study: reproducible, and what the warnings rest on.

``tests/reliability/extreme_weights_binary_iv.py`` repeats the weight
question for ``sp.logit``, ``sp.probit``, ``sp.glm`` and ``sp.iv``. This
file checks the stored design, that one cell of each model reproduces
its stored prefix, the statements in ``tests/reliability/README.md``,
and that the four entry points record the diagnostics and warn.
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.core._agent_summary import kish_effective_n
from statspai.exceptions import AssumptionWarning

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "reliability" / "extreme_weights_binary_iv.py"
RESULTS = ROOT / "reliability" / "extreme_weights_binary_iv_results.json"


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("extreme_weights_binary_iv", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def _cell(results, model, n, sigma):
    (cell,) = [
        c
        for c in results["cells"]
        if (c["model"], c["n"], c["sigma"]) == (model, n, sigma)
    ]
    return cell


def test_file_has_the_declared_design(study, results):
    assert results["B"] == 2000 and results["level"] == 0.95
    keys = {(c["model"], c["n"], c["sigma"]) for c in results["cells"]}
    assert keys == {
        (m, n, s) for m in study.MODELS for n in study.N_VALUES for s in study.SIGMAS
    }
    for c in results["cells"]:
        for v in study.VARIANCES:
            assert c[v]["n_fitted"] == 2000 and not c[v]["refused"]


@pytest.mark.parametrize("model", ["logit", "probit", "glm", "iv"])
def test_a_cell_reproduces_its_stored_prefix(study, results, model):
    fresh = study.run_cell(model, 200, 2.0, study.PREFIX)
    stored = _cell(results, model, 200, 2.0)
    for v in study.VARIANCES:
        assert fresh[v]["prefix_hits"] == stored[v]["prefix_hits"], v


def test_equal_weights_cover_at_the_nominal_level(study, results):
    for model in study.MODELS:
        for n in study.N_VALUES:
            cell = _cell(results, model, n, 0.0)
            for v in study.VARIANCES:
                assert abs(cell[v]["coverage"] - 0.95) < 3 * cell[v]["mc_se"], (
                    model,
                    n,
                    v,
                )


def test_default_variance_fails_under_sampling_weights(study, results):
    for model in study.MODELS:
        for n in study.N_VALUES:
            mild = _cell(results, model, n, 1.0)["classical"]["coverage"]
            strong = _cell(results, model, n, 2.0)["classical"]["coverage"]
            assert mild < 0.80 and strong < 0.52, (model, n, mild, strong)
    # the three likelihood models are hit harder than the linear one
    for model in ("logit", "probit", "glm"):
        assert _cell(results, model, 1000, 2.0)["classical"]["coverage"] < 0.17


def test_robust_variance_is_short_only_in_small_effective_samples(study, results):
    for model in study.MODELS:
        small = _cell(results, model, 200, 2.0)
        mid = _cell(results, model, 1000, 2.0)
        large = _cell(results, model, 1000, 1.0)
        assert small["kish_median"] < 20 and small["robust"]["coverage"] < 0.86
        assert mid["kish_median"] < 60 and mid["robust"]["coverage"] < 0.90
        assert large["kish_median"] > 300
        assert abs(large["robust"]["coverage"] - 0.95) < 3 * large["robust"]["mc_se"]


# --------------------------------------------------------------------- #
# The diagnostics in the entry points
# --------------------------------------------------------------------- #


def _fit(call):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = call()
    msgs = [str(m.message) for m in rec if issubclass(m.category, AssumptionWarning)]
    return res.model_info, msgs


def _calls(study):
    binary, _ = study.draw("logit", 600, 2.0, 1)
    linear, _ = study.draw("iv", 600, 2.0, 1)
    for frame in (binary, linear):
        frame["g"] = np.arange(len(frame)) % 60
        frame["one"] = 1.0
    return {
        "logit": (binary, lambda d, **kw: sp.logit("y ~ x + z", d, **kw), "robust"),
        "probit": (binary, lambda d, **kw: sp.probit("y ~ x + z", d, **kw), "robust"),
        "glm": (
            binary,
            lambda d, **kw: sp.glm("y ~ x + z", d, family="binomial", **kw),
            "robust",
        ),
        "iv": (linear, lambda d, **kw: sp.iv("y ~ z + (d ~ q)", data=d, **kw), "hc1"),
    }


@pytest.mark.parametrize("name", ["logit", "probit", "glm", "iv"])
def test_entry_point_records_and_warns(study, name):
    frame, call, robust = _calls(study)[name]
    info, msgs = _fit(lambda: call(frame, weights="w"))
    assert info["n_effective_weights"] == pytest.approx(kish_effective_n(frame["w"]))
    assert any("weights are dispersed" in m for m in msgs), msgs
    # the reading the default variance assumes is named for what it is
    assumed = "precisions" if name == "iv" else "frequencies"
    assert any(assumed in m for m in msgs), msgs

    _, msgs = _fit(lambda: call(frame, weights="w", robust=robust))
    assert any("carry most of the weight" in m for m in msgs), msgs

    info, msgs = _fit(lambda: call(frame, weights="w", cluster="g"))
    assert info["n_clusters_effective_weights"] < 30
    assert any("weights concentrate on a few" in m for m in msgs), msgs


@pytest.mark.parametrize("name", ["logit", "probit", "glm", "iv"])
def test_equal_or_absent_weights_are_silent(study, name):
    frame, call, _ = _calls(study)[name]
    info, msgs = _fit(lambda: call(frame, weights="one"))
    assert info["n_effective_weights"] == pytest.approx(len(frame))
    assert not [m for m in msgs if "weight" in m.lower()], msgs
    info, msgs = _fit(lambda: call(frame))
    assert "n_effective_weights" not in info
    assert not [m for m in msgs if "weight" in m.lower()], msgs


def test_gaussian_glm_names_the_analytic_reading(study):
    frame, _, _ = _calls(study)["iv"]
    _, msgs = _fit(lambda: sp.glm("y ~ z", frame, family="gaussian", weights="w"))
    assert any("precisions" in m for m in msgs), msgs
