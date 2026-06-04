"""Coverage for previously-untested output / survey / dataset helpers:
``sp.svyglm`` (+ ``SurveyDesign``), the Stata-style estimate store
``sp.eststo`` / ``sp.estclear`` / ``sp.esttab``, the journal-template registry
``sp.list_journal_templates`` / ``sp.get_journal_template`` / ``sp.list_themes``
and the ``sp.chilean_households`` example dataset.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# --------------------------------------------------------------------------
# Survey-weighted GLM
# --------------------------------------------------------------------------
def test_svyglm_recovers_ols_coefficients():
    rng = np.random.RandomState(0)
    n = 600
    df = pd.DataFrame({"x": rng.randn(n)})
    df["y"] = 2.0 + 1.5 * df["x"] + rng.randn(n)
    df["w"] = rng.uniform(0.5, 2.0, n)

    design = sp.SurveyDesign(data=df, weights="w")
    res = sp.svyglm("y ~ x", design)

    est = np.asarray(res.estimate, dtype=float)
    # formula "y ~ x" -> [intercept, slope]
    assert est[0] == pytest.approx(2.0, abs=0.25)
    assert est[1] == pytest.approx(1.5, abs=0.25)
    assert np.all(np.asarray(res.std_error, dtype=float) > 0)
    assert np.all(np.isfinite(np.asarray(res.deff, dtype=float)))
    assert res.dof > 0


def test_svyglm_ci_brackets_estimate():
    rng = np.random.RandomState(1)
    n = 500
    df = pd.DataFrame({"x": rng.randn(n)})
    df["y"] = 1.0 + 0.8 * df["x"] + rng.randn(n)
    df["w"] = np.ones(n)  # equal weights -> design effect ~ 1

    res = sp.svyglm("y ~ x", sp.SurveyDesign(data=df, weights="w"))
    lo = np.asarray(res.ci_lower, dtype=float)
    hi = np.asarray(res.ci_upper, dtype=float)
    est = np.asarray(res.estimate, dtype=float)
    assert np.all(lo <= est) and np.all(est <= hi)


# --------------------------------------------------------------------------
# eststo / estclear / esttab — Stata-style estimate store
# --------------------------------------------------------------------------
def test_eststo_esttab_roundtrip_and_clear():
    rng = np.random.RandomState(0)
    n = 200
    df = pd.DataFrame({"x": rng.randn(n)})
    df["y"] = 2.0 + 1.5 * df["x"] + rng.randn(n)

    sp.estclear()
    model = sp.regress("y ~ x", df)
    sp.eststo(model, name="m1")
    sp.eststo(model, name="m2")

    table = sp.esttab()
    rendered = table.to_text() if hasattr(table, "to_text") else str(table)
    assert "m1" in rendered and "m2" in rendered
    # The result object exposes the agent-native export surface.
    for exporter in ("to_latex", "to_markdown", "to_dataframe"):
        assert hasattr(table, exporter)

    sp.estclear()
    # After clearing, the store no longer contains the two models.
    cleared = sp.esttab()
    cleared_text = cleared.to_text() if hasattr(cleared, "to_text") else str(cleared)
    assert "m1" not in cleared_text and "m2" not in cleared_text


# --------------------------------------------------------------------------
# Journal templates / plot themes
# --------------------------------------------------------------------------
def test_journal_template_registry():
    names = sp.list_journal_templates()
    assert "aer" in names and "qje" in names
    tpl = sp.get_journal_template("aer")
    # A template must carry the fields the table formatter relies on.
    assert {"label", "star_levels", "se_label"} <= set(tpl.keys())


def test_get_journal_template_unknown_raises():
    with pytest.raises((KeyError, ValueError)):
        sp.get_journal_template("not_a_real_journal")


def test_list_themes_contains_default():
    themes = sp.list_themes()
    assert "statspai" in themes


# --------------------------------------------------------------------------
# Example dataset
# --------------------------------------------------------------------------
def test_chilean_households_shape_and_reproducibility():
    d1 = sp.chilean_households(n=300, seed=0)
    d2 = sp.chilean_households(n=300, seed=0)
    assert len(d1) == 300
    assert "log_income" in d1.columns
    # Deterministic given the seed.
    pd.testing.assert_frame_equal(d1, d2)
