"""Tests for agent-native serialisation of ``RegtableResult``.

``sp.regtable(...)`` produces a publication table; the package is
agent-native (CLAUDE.md §1), so that table must serialise to a JSON-safe
dict an LLM tool loop can cache and reason over.  ``to_dict()`` carries three
layers — metadata, the rendered cell grid, and the numeric truth per model —
and ``to_json()`` must round-trip through ``json.dumps``.
"""

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture
def two_models():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 1.0 + 2.0 * df["x"] - 0.5 * df["z"] + rng.normal(size=n)
    df["y2"] = 0.5 + 1.5 * df["x"] + rng.normal(size=n)
    m1 = sp.regress("y ~ x + z", data=df)
    m2 = sp.regress("y2 ~ x", data=df)
    return m1, m2


@pytest.fixture
def table(two_models):
    m1, m2 = two_models
    return sp.regtable(m1, m2, template="aer", model_labels=["Wages", "Hours"])


class TestToDictStructure:

    def test_top_level_keys(self, table):
        d = table.to_dict()
        for key in (
            "kind",
            "n_models",
            "model_labels",
            "columns",
            "table",
            "models",
            "template",
            "se_type",
            "stars",
            "star_levels",
            "requested_stats",
        ):
            assert key in d, f"missing top-level key {key!r}"
        assert d["kind"] == "regression_table"
        assert d["n_models"] == 2
        assert d["model_labels"] == ["Wages", "Hours"]
        assert d["template"] == "aer"

    def test_rendered_cell_grid(self, table):
        d = table.to_dict()
        assert d["columns"][0] == "term"
        assert d["columns"][1:] == ["Wages", "Hours"]
        first = d["table"][0]
        # Intercept row with stars in the rendered cell
        assert first["term"] == "Intercept"
        assert "Wages" in first and "Hours" in first
        assert any(c.isdigit() for c in first["Wages"])

    def test_numeric_truth_layer(self, table):
        d = table.to_dict()
        assert len(d["models"]) == 2
        m0 = d["models"][0]
        assert m0["depvar"] == "y"
        xc = m0["coefficients"]["x"]
        # Known DGP coefficient ~ 2.0
        assert xc["estimate"] == pytest.approx(2.0, abs=0.25)
        assert xc["std_error"] > 0
        assert 0.0 <= xc["p_value"] <= 1.0
        assert xc["conf_low"] < xc["estimate"] < xc["conf_high"]
        assert m0["stats"]["N"] == 200

    def test_numeric_layer_is_floats_not_strings(self, table):
        """The 'models' layer is the machine-readable truth: numbers, not
        the formatted '2.067***' cells from the 'table' layer."""
        d = table.to_dict()
        est = d["models"][0]["coefficients"]["x"]["estimate"]
        assert isinstance(est, float)


class TestJsonSafety:

    def test_round_trips_through_json(self, table):
        s = table.to_json()
        back = json.loads(s)
        assert back["n_models"] == 2
        assert back["models"][0]["coefficients"]["x"]["estimate"] == pytest.approx(
            2.0, abs=0.25
        )

    def test_indent_passthrough(self, table):
        s = table.to_json(indent=2)
        assert "\n" in s

    def test_no_nan_inf_leak(self, two_models):
        """NaN / Inf must be coerced to null so the JSON is strict-valid."""
        m1, m2 = two_models
        d = sp.regtable(m1, m2).to_dict()
        flat = json.dumps(d)  # would raise/emit NaN tokens if not coerced
        assert "NaN" not in flat
        assert "Infinity" not in flat

    def test_jsonable_helper_handles_edge_scalars(self):
        from statspai.output.regression_table import RegtableResult

        j = RegtableResult._jsonable
        assert j(float("nan")) is None
        assert j(float("inf")) is None
        assert j(np.int64(7)) == 7
        assert isinstance(j(np.float64(1.5)), float)
        assert j(None) is None
        assert j("abc") == "abc"
        assert j(True) is True

    def test_summary_and_agent_summary_are_json_safe(self, table):
        assert table.summary() == table.to_text()

        payload = table.to_agent_summary(max_rows=3, max_terms=1)
        encoded = json.dumps(payload)
        assert "regression_table_summary" in encoded
        assert payload["kind"] == "regression_table_summary"
        assert payload["n_models"] == 2
        assert len(payload["table_head"]) == 3
        assert len(payload["models"]) == 2
        assert len(payload["models"][0]["coefficients"]) == 1


class TestRenders:

    def test_no_renders_by_default(self, table):
        assert "renders" not in table.to_dict()

    def test_renders_true_embeds_all(self, table):
        d = table.to_dict(renders=True)
        assert set(d["renders"]) == {"latex", "html", "markdown", "text"}
        assert "\\begin{table}" in d["renders"]["latex"]

    def test_renders_subset(self, table):
        d = table.to_dict(renders=["latex"])
        assert set(d["renders"]) == {"latex"}

    def test_renders_unknown_raises(self, table):
        with pytest.raises(ValueError):
            table.to_dict(renders=["bogus"])

    def test_renders_passthrough_in_to_json(self, table):
        s = table.to_json(renders=["markdown"])
        back = json.loads(s)
        assert "markdown" in back["renders"]


class TestJsonFileDispatch:

    def test_save_json_extension(self, table, tmp_path):
        p = tmp_path / "t.json"
        table.save(str(p))
        back = json.loads(p.read_text(encoding="utf-8"))
        assert back["kind"] == "regression_table"
        assert back["n_models"] == 2

    def test_regtable_filename_json(self, two_models, tmp_path):
        m1, m2 = two_models
        p = tmp_path / "t.json"
        sp.regtable(m1, m2, filename=str(p))
        back = json.loads(p.read_text(encoding="utf-8"))
        assert back["kind"] == "regression_table"


class TestConsistency:

    def test_table_layer_matches_to_dataframe(self, table):
        d = table.to_dict()
        df = table.to_dataframe()
        assert len(d["table"]) == len(df)
        assert d["columns"][1:] == [str(c) for c in df.columns]

    def test_n_models_matches_models_layer(self, table):
        d = table.to_dict()
        assert d["n_models"] == len(d["models"])


class TestValidationTaxonomy:

    def test_regtable_requires_at_least_one_model(self):
        with pytest.raises(DataInsufficient):
            sp.regtable()

    def test_invalid_output_reports_diagnostics(self, two_models):
        m1, _ = two_models
        with pytest.raises(MethodIncompatibility) as exc:
            sp.regtable(m1, output="quartoo")
        assert exc.value.diagnostics["output"] == "quartoo"

    def test_tests_length_reports_diagnostics(self, two_models):
        m1, m2 = two_models
        with pytest.raises(MethodIncompatibility) as exc:
            sp.regtable(m1, m2, tests={"Bad": [1.0]})
        assert exc.value.diagnostics == {
            "label": "Bad",
            "n_entries": 1,
            "n_models": 2,
        }
