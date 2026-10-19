"""Shared argument spellings across the table functions (2026-10-19 audit).

``regtable`` says ``model_labels`` / ``coef_labels``, ``modelsummary`` and
``outreg2`` say ``model_names``, ``outreg2`` says ``variable_labels``,
``esttab`` says ``names`` / ``labels``. Each now also accepts the others'
spelling (and Stata's ``mtitles`` / ``coeflabels``); the output is identical
to the canonical call, and passing both spellings is an error.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def models():
    r = np.random.default_rng(0)
    df = pd.DataFrame({"x": r.normal(size=120), "z": r.normal(size=120)})
    df["y"] = 1 + df.x - df.z + r.normal(size=120)
    return sp.regress("y ~ x", data=df), sp.regress("y ~ x + z", data=df)


def _q(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("model_names", "model_labels"),
        ("mtitles", "model_labels"),
    ],
)
def test_regtable_model_label_aliases(models, alias, canonical):
    a = _q(sp.regtable, *models, **{alias: ["A", "B"]}).to_text()
    b = _q(sp.regtable, *models, **{canonical: ["A", "B"]}).to_text()
    assert a == b and "A" in a


@pytest.mark.parametrize("alias", ["coeflabels", "variable_labels"])
def test_regtable_coef_label_aliases(models, alias):
    a = _q(sp.regtable, *models, **{alias: {"x": "Treatment"}}).to_text()
    b = _q(sp.regtable, *models, coef_labels={"x": "Treatment"}).to_text()
    assert a == b and "Treatment" in a


def test_esttab_aliases(models):
    a = _q(sp.esttab, *models, model_labels=["A", "B"], coef_labels={"x": "T"})
    b = _q(sp.esttab, *models, names=["A", "B"], labels={"x": "T"})
    assert a.to_text() == b.to_text()


def test_modelsummary_alias(models):
    a = _q(sp.modelsummary, *models, model_labels=["A", "B"])
    b = _q(sp.modelsummary, *models, model_names=["A", "B"])
    assert a == b


def test_outreg2_aliases(models, tmp_path):
    a = _q(
        sp.outreg2,
        *models,
        filename=str(tmp_path / "a.tex"),
        model_labels=["A", "B"],
        coef_labels={"x": "T"},
        digits=2,
    )
    b = _q(
        sp.outreg2,
        *models,
        filename=str(tmp_path / "b.tex"),
        model_names=["A", "B"],
        variable_labels={"x": "T"},
        decimal_places=2,
    )
    assert a == b


def test_both_spellings_is_an_error(models):
    with pytest.raises(TypeError):
        _q(sp.regtable, *models, model_names=["A", "B"], model_labels=["A", "B"])
