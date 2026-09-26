"""Every regression-table entry point reports the same numbers.

``sp.esttab`` and ``sp.modelsummary`` delegate to ``sp.regtable``; ``sp.etable``
(native results) and ``sp.outreg2`` format coefficients themselves -- ``etable``
on purpose, since ``regtable``'s term extraction renames some terms (Tobit's
``const`` / ``sigma``). Five code paths for one table is exactly where a star
threshold or a rounding rule drifts, so pin the rendered coefficient, standard
error and significance marker of each path to the same strings.
"""

from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TERMS = ["Intercept", "x1", "x2"]


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(0, 40, n),
        }
    )
    df["y"] = 1 + 0.05 * df.x1 - 0.3 * df.x2 + rng.normal(size=n)
    df["yb"] = (df.y > 1).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            sp.regress("y ~ x1", data=df, vce="cluster g"),
            sp.logit("yb ~ x1 + x2", data=df),
        )


def _from_long_frame(frame: pd.DataFrame) -> dict:
    """{(term, column): (coef_cell, se_cell)} from a coef-row / se-row frame."""
    out = {}
    labels = list(frame.index)
    for i, label in enumerate(labels[:-1]):
        if label in TERMS and labels[i + 1] == "":
            for col in frame.columns:
                coef = str(frame.iloc[i][col]).strip()
                se = str(frame.iloc[i + 1][col]).strip()
                if coef:
                    out[(label, col)] = (coef, se)
    return out


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def test_limited_dependent_results_are_tabulated_term_by_term():
    """Tobit is a CausalResult with a full coefficient vector.

    regtable used to collapse it to the headline estimand: one row renamed
    ``beta_x1``, with ``const`` and ``sigma`` missing.
    """
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x1": rng.normal(size=400)})
    df["y"] = np.maximum(0.5 + df.x1 + rng.normal(size=400), 0)
    fit = _quiet(sp.tobit, df, y="y", x=["x1"], ll=0)
    table = _quiet(sp.regtable, fit).to_dataframe()
    labels = [i for i in table.index if i]
    assert labels[:3] == ["const", "x1", "sigma"]
    assert "beta_x1" not in table.index
    se_row = table.iloc[list(table.index).index("sigma") + 1, 0]
    assert se_row == f"({float(fit.std_errors['sigma']):.3f})"
    etable = _quiet(sp.etable, fit)
    assert list(etable.index) == ["const", "x1", "sigma"]
    coef = table.loc["x1"].iloc[0]
    assert etable.loc["x1", "(1)"].startswith(coef)


def test_all_table_paths_agree(models, tmp_path):
    reference = _from_long_frame(_quiet(sp.regtable, *models).to_dataframe())
    assert reference, "regtable produced no coefficient rows"
    assert reference[("x2", "(2)")][0].endswith("***")

    esttab = _from_long_frame(_quiet(sp.esttab, *models).to_dataframe())
    summary = _from_long_frame(_quiet(sp.modelsummary, *models, output="dataframe"))
    assert esttab == reference
    assert summary == reference

    etable = _quiet(sp.etable, *models)
    for (term, col), (coef, se) in reference.items():
        assert etable.loc[term, col] == f"{coef} {se}"

    path = tmp_path / "table.tex"
    _quiet(sp.outreg2, *models, filename=str(path), format="latex")
    tex = path.read_text(encoding="utf-8")
    for (term, col), (coef, se) in reference.items():
        j = int(col.strip("()"))
        row = re.search(rf"^{re.escape(term)} & (.*?) \\\\\n & (.*?) \\\\", tex, re.M)
        assert row, term
        coefs = [c.strip() for c in row.group(1).split("&")]
        ses = [c.strip() for c in row.group(2).split("&")]
        assert (coefs[j - 1], ses[j - 1]) == (coef, se)
