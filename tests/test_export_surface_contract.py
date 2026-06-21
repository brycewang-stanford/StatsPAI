"""Contract: every estimator's result is exportable through one path.

StatsPAI's export story rests on a universality claim — ``sp.regtable(r)``
accepts *any* fitted result (it duck-types ``params`` / ``std_errors`` or
``estimate`` / ``se``), so a user never has to learn a per-estimator export
API.  This suite pins that claim across a representative spread of result
classes (``EconometricResults`` / ``CausalResult`` / ``PanelResults`` /
``FrontierResult``): each must render to LaTeX + a JSON-safe ``to_dict`` that
round-trips through ``from_dict``.  If a future estimator returns a result the
table builder can't consume, this fails loudly rather than the gap going
unnoticed.

It also documents the *boundary*: results that are not coefficient tables
(forests' heterogeneous effects, partial-identification bounds, density
curves) are out of scope here — they carry their own visualisations/summaries.
"""

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output.regression_table import RegtableResult


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["d"] = (0.5 * df["x"] + 0.4 * df["z"] + rng.normal(size=n) > 0).astype(int)
    df["y"] = 1 + 2.0 * df["d"] + 0.5 * df["x"] - 0.3 * df["z"] + rng.normal(size=n)
    df["count"] = rng.poisson(np.exp(0.2 * df["x"]).clip(upper=20))
    df["entity"] = np.tile(np.arange(n // 4), 4)[:n]
    df["time"] = np.repeat(np.arange(4), n // 4)[:n]
    return df


def _fit(name, df):
    """Fit one estimator; importorskip heavy optional backends."""
    if name == "ols":
        return sp.regress("y ~ d + x + z", data=df)
    if name == "iv":
        return sp.iv("y ~ (d ~ z) + x", data=df)
    if name == "logit":
        return sp.logit("d ~ x + z", data=df)
    if name == "glm_poisson":
        return sp.glm("count ~ x + z", data=df, family="poisson")
    if name == "qreg":
        return sp.qreg(df, "y ~ d + x")
    if name == "panel":
        return sp.panel(df, "y ~ d + x", entity="entity", time="time")
    if name == "frontier":
        return sp.frontier(df, "y", ["x", "z"])
    if name == "dml":
        pytest.importorskip("sklearn")
        return sp.dml(df, "y", d="d", X=["x", "z"])
    if name == "tmle":
        pytest.importorskip("sklearn")
        return sp.tmle(df, "y", "d", ["x", "z"])
    raise AssertionError(name)


ESTIMATORS = [
    "ols",
    "iv",
    "logit",
    "glm_poisson",
    "qreg",
    "panel",
    "frontier",
    "dml",
    "tmle",
]


class TestUniversalExporter:
    """sp.regtable(result) must accept every estimator's result object."""

    @pytest.mark.parametrize("name", ESTIMATORS)
    def test_regtable_latex(self, name, data):
        r = _fit(name, data)
        tex = sp.regtable(r).to_latex()
        assert "\\begin{table}" in tex

    @pytest.mark.parametrize("name", ESTIMATORS)
    def test_regtable_dict_is_json_safe(self, name, data):
        r = _fit(name, data)
        payload = sp.regtable(r).to_dict()
        assert payload["kind"] == "regression_table"
        assert payload["n_models"] == 1
        flat = json.dumps(payload)  # strict JSON — no NaN/Inf tokens
        assert "NaN" not in flat and "Infinity" not in flat

    @pytest.mark.parametrize("name", ESTIMATORS)
    def test_regtable_from_dict_round_trips(self, name, data):
        r = _fit(name, data)
        t = sp.regtable(r)
        again = RegtableResult.from_dict(t.to_dict())
        assert again.to_latex() == t.to_latex()

    @pytest.mark.parametrize("name", ESTIMATORS)
    def test_siunitx_path_renders(self, name, data):
        """The journal-grade siunitx path must work for plain tables too."""
        r = sp.regtable(_fit(name, data))
        tex = r.to_latex(siunitx=True)
        assert "S[table-format=" in tex


class TestResultOwnExportMethods:
    """Where a result class advertises its own export methods, they work."""

    @pytest.mark.parametrize("name", ESTIMATORS)
    def test_own_to_latex_if_present(self, name, data):
        r = _fit(name, data)
        if hasattr(r, "to_latex"):
            assert isinstance(r.to_latex(), str)

    @pytest.mark.parametrize("name", ESTIMATORS)
    def test_own_to_dict_if_present(self, name, data):
        r = _fit(name, data)
        if hasattr(r, "to_dict"):
            json.dumps(r.to_dict())  # must be JSON-safe


class TestBoundaryDocumentation:
    """Pin the documented boundary: regtable consumes coefficient-like
    results (params/std_errors OR estimate/se). A bare object without either
    is correctly rejected rather than silently producing an empty table."""

    def test_non_model_object_rejected(self):
        with pytest.raises(Exception):
            sp.regtable(object())
