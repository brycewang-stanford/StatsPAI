"""Round-trip tests for ``RegtableResult.from_dict()`` (inverse of to_dict).

The contract: for a table built without exotic options (multi_se / eform /
column_spanners / tests / apply_coef), ``from_dict(t.to_dict())`` must
re-render byte-identically to the original across every text format. This
makes the JSON payload a faithful cache of the table, not just a snapshot.
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
    n = 300
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "z": rng.normal(size=n), "w": rng.normal(size=n)}
    )
    df["y"] = 1.0 + 2.0 * df["x"] - 0.5 * df["z"] + rng.normal(size=n)
    df["y2"] = 0.5 + 1.5 * df["x"] + 0.3 * df["w"] + rng.normal(size=n)
    m1 = sp.regress("y ~ x + z", data=df)
    m2 = sp.regress("y2 ~ x + w", data=df)
    m3 = sp.regress("y ~ x", data=df)
    return m1, m2, m3


# Each entry builds a table from the (m1, m2, m3) fixture.
SCENARIOS = {
    "single": lambda m: sp.regtable(m[0]),
    "multi": lambda m: sp.regtable(m[0], m[1]),
    "template_aer": lambda m: sp.regtable(m[0], m[1], template="aer"),
    "labels": lambda m: sp.regtable(
        m[0], m[1], model_labels=["A", "B"], dep_var_labels=["Y", "Y2"]
    ),
    "keep": lambda m: sp.regtable(m[0], m[1], keep=["x"]),
    "drop": lambda m: sp.regtable(m[0], m[1], drop=["Intercept"]),
    "order": lambda m: sp.regtable(m[0], m[1], order=["z", "x", "Intercept"]),
    "coef_labels": lambda m: sp.regtable(m[0], m[1], coef_labels={"x": "Treat"}),
    "add_rows": lambda m: sp.regtable(m[0], m[1], add_rows={"Controls": ["No", "Yes"]}),
    "se_ci": lambda m: sp.regtable(m[0], m[1], se_type="ci"),
    "se_t": lambda m: sp.regtable(m[0], m[1], se_type="t"),
    "fmt4": lambda m: sp.regtable(m[0], m[1], fmt="%.4f"),
    "multipanel": lambda m: sp.regtable([m[0], m[1]], [m[2]], panel_labels=["A", "B"]),
    "notes_title": lambda m: sp.regtable(m[0], title="T", notes=["n1"]),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@pytest.mark.parametrize("fmt", ["to_latex", "to_markdown", "to_text", "to_html"])
def test_round_trip_rerender_is_identical(models, scenario, fmt):
    t = SCENARIOS[scenario](models)
    rt = RegtableResult.from_dict(t.to_dict())
    assert (
        getattr(rt, fmt)() == getattr(t, fmt)()
    ), f"{scenario} re-render diverged for {fmt}"


class TestJsonRoundTrip:

    def test_through_json_string(self, models):
        t = sp.regtable(models[0], models[1], template="qje")
        payload = json.loads(t.to_json())
        rt = RegtableResult.from_dict(payload)
        assert rt.to_latex() == t.to_latex()

    def test_save_json_then_from_dict(self, models, tmp_path):
        t = sp.regtable(models[0], models[1])
        p = tmp_path / "t.json"
        t.save(str(p))
        rt = RegtableResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
        assert rt.to_markdown() == t.to_markdown()


class TestNumericPreservation:

    def test_coefficients_survive(self, models):
        t = sp.regtable(models[0], models[1])
        rt = RegtableResult.from_dict(t.to_dict())
        a = t.to_dict()["models"][0]["coefficients"]["x"]
        b = rt.to_dict()["models"][0]["coefficients"]["x"]
        assert b["estimate"] == pytest.approx(a["estimate"])
        assert b["std_error"] == pytest.approx(a["std_error"])
        assert b["p_value"] == pytest.approx(a["p_value"])

    def test_idempotent_dict(self, models):
        """to_dict -> from_dict -> to_dict reproduces the same payload."""
        t = sp.regtable(models[0], models[1], template="aer")
        d1 = t.to_dict()
        d2 = RegtableResult.from_dict(d1).to_dict()
        assert d1 == d2


class TestErrors:

    def test_rejects_non_payload(self):
        with pytest.raises(ValueError):
            RegtableResult.from_dict({"kind": "not_a_table"})

    def test_rejects_non_dict(self):
        with pytest.raises(ValueError):
            RegtableResult.from_dict("nope")


class TestBackwardCompat:

    def test_missing_render_spec_best_effort(self, models):
        """An old payload without render_spec still reconstructs (single
        panel, default fmt) rather than crashing."""
        t = sp.regtable(models[0], models[1])
        d = t.to_dict()
        d.pop("render_spec", None)
        rt = RegtableResult.from_dict(d)
        # Renders a valid table; coefficient values preserved.
        assert "\\begin{table}" in rt.to_latex()
        assert rt.n_models == 2

    def test_mismatched_panel_sizes_falls_back(self, models):
        """A hand-edited payload with inconsistent panel_sizes must not drop
        models silently; it falls back to a single panel."""
        t = sp.regtable([models[0]], [models[1]], panel_labels=["A", "B"])
        d = t.to_dict()
        d["render_spec"]["panel_sizes"] = [5, 9]  # nonsense
        rt = RegtableResult.from_dict(d)
        assert rt.n_models == 2
