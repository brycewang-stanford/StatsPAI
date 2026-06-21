"""Shared NHEFS / *What If* helpers for the original-data parity harness.

Encodes the single confounder specification used throughout Hernán &
Robins, *Causal Inference: What If* (Part II), so that every chapter
module — and both the Python (StatsPAI) and R reference sides — fit the
*same* propensity / outcome model on the *same* CSV bytes.

Canonical confounder set (book Programs 12.x-15.x):

    categorical (indicator sets) : sex, race, education(5), exercise(3), active(3)
    continuous (with quadratics) : age, smokeintensity, smokeyrs, wt71

Treatment  : ``qsmk``    (quit smoking 1971-1982)
Outcome    : ``wt82_71`` (10-year weight change, kg; complete case n=1566)
"""

from __future__ import annotations

import pandas as pd
import statspai as sp

from _common import DATA_DIR

# Confounders, split by how they enter the model.
CAT_CONF = ["sex", "race", "education", "exercise", "active"]
CONT_CONF = ["age", "smokeintensity", "smokeyrs", "wt71"]


def nhefs_complete() -> pd.DataFrame:
    """The n=1566 complete-case weight sample (non-missing ``wt82_71``)."""
    return sp.datasets.nhefs(complete_case=True)


def dump_csv(module: str) -> pd.DataFrame:
    """Write the complete-case NHEFS to ``data/<module>.csv`` so the R
    reference reads identical bytes, and return the DataFrame."""
    df = nhefs_complete()
    df.to_csv(DATA_DIR / f"{module}.csv", index=False)
    return df


def book_design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """StatsPAI-friendly numeric encoding of the book's model.

    Returns ``(design_df, covariate_columns)`` where the continuous
    confounders carry quadratic terms and the categoricals are expanded
    to drop-first indicators — the exact linear predictor used by the
    book's logistic propensity model and linear outcome model.
    """
    d = df.copy()
    for c in CONT_CONF:
        d[f"{c}2"] = d[c] ** 2
    dd = pd.get_dummies(d, columns=CAT_CONF, drop_first=True)
    cat_cols = [c for c in dd.columns if any(c.startswith(p + "_") for p in CAT_CONF)]
    cont_cols = CONT_CONF + [f"{c}2" for c in CONT_CONF]
    covs = cat_cols + cont_cols
    for c in covs:
        dd[c] = dd[c].astype(float)
    return dd, covs


# R-side formula fragment that reproduces ``book_design`` exactly:
# categoricals as factors, continuous with quadratics.
R_CONF_FORMULA = (
    "factor(sex)+factor(race)+factor(education)+factor(exercise)+"
    "factor(active)+age+I(age^2)+smokeintensity+I(smokeintensity^2)+"
    "smokeyrs+I(smokeyrs^2)+wt71+I(wt71^2)"
)
