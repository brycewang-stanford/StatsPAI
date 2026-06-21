"""Agent-native serialisation for multi-table containers.

``Collection`` (a document of tables + prose) and ``PaperTables`` (a
multi-panel bundle) gain ``to_dict()`` / ``to_json()`` mirroring
``RegtableResult`` — every regression-table item carries the full
three-layer payload, so an LLM tool loop can cache and reason over a whole
document without re-rendering.
"""

import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output.regression_table import RegtableResult


@pytest.fixture
def models():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["y"] = 1.0 + 2.0 * df["x"] - 0.5 * df["z"] + rng.normal(size=n)
    m1 = sp.regress("y ~ x + z", data=df)
    m2 = sp.regress("y ~ x", data=df)
    return df, m1, m2


class TestCollection:

    def test_to_dict_structure(self, models):
        df, m1, m2 = models
        c = sp.collect(title="My doc")
        c.add_regression(m1, m2, title="Table 1")
        c.add_summary(df, title="Descriptives")
        c.add_text("Some prose.") if hasattr(c, "add_text") else None
        d = c.to_dict()
        assert d["kind"] == "collection"
        assert d["title"] == "My doc"
        assert d["n_items"] == len(c.items)
        kinds = [it["item_kind"] for it in d["items"]]
        assert "regtable" in kinds and "summary" in kinds

    def test_regtable_item_carries_full_payload(self, models):
        _, m1, m2 = models
        c = sp.collect()
        c.add_regression(m1, m2)
        content = c.to_dict()["items"][0]["content"]
        assert content["kind"] == "regression_table"
        assert "models" in content and "table" in content
        # the nested payload is itself a valid regtable dict → round-trippable
        again = RegtableResult.from_dict(content)
        assert "\\begin{table}" in again.to_latex()

    def test_summary_item_has_rows(self, models):
        df, m1, _ = models
        c = sp.collect()
        c.add_summary(df, title="D")
        content = c.to_dict()["items"][0]["content"]
        assert "rows" in content and "columns" in content

    def test_to_json_is_strict(self, models):
        df, m1, m2 = models
        c = sp.collect()
        c.add_regression(m1, m2)
        c.add_summary(df)
        s = c.to_json()
        back = json.loads(s)
        assert back["kind"] == "collection"
        assert "NaN" not in s and "Infinity" not in s

    def test_empty_collection(self):
        c = sp.collect(title="Empty")
        d = c.to_dict()
        assert d["n_items"] == 0
        assert d["items"] == []


class TestPaperTables:

    def test_to_dict_structure(self, models):
        _, m1, m2 = models
        pt = sp.paper_tables(main=[m1, m2], robustness=[m2], template="qje")
        d = pt.to_dict()
        assert d["kind"] == "paper_tables"
        assert d["template"] == "qje"
        assert d["panel_names"] == ["main", "robustness"]

    def test_panels_carry_full_payload(self, models):
        _, m1, m2 = models
        pt = sp.paper_tables(main=[m1, m2])
        panel = pt.to_dict()["panels"]["main"]
        assert panel["kind"] == "regression_table"
        assert panel["n_models"] == 2
        assert RegtableResult.from_dict(panel).to_latex() == pt.main.to_latex()

    def test_to_json_is_strict(self, models):
        _, m1, m2 = models
        pt = sp.paper_tables(main=[m1], robustness=[m2])
        s = pt.to_json()
        assert json.loads(s)["kind"] == "paper_tables"
        assert "NaN" not in s and "Infinity" not in s

    def test_only_populated_panels(self, models):
        _, m1, _ = models
        pt = sp.paper_tables(main=[m1])
        d = pt.to_dict()
        assert list(d["panels"].keys()) == ["main"]
