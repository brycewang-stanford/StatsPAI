"""sp.paper() folds the fitted result's automatic diagnostics into Robustness.

The auto-paper's Robustness section must state plainly what the estimator's
self-audit (``result.violations()``) found — a flagged assumption becomes a
line in the paper, with the ``sp.*`` a reviewer would ask the authors to try.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

import statspai as sp


def _obs_design(confounding: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(confounding * x1 + 0.6 * confounding * x2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 1 + 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


def _paper(df: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.paper(df, y="y", treatment="d", covariates=["x1", "x2"])


def test_robustness_section_surfaces_a_flagged_diagnostic():
    draft = _paper(_obs_design(confounding=1.5))  # strong → poor post-match balance
    rob = draft.sections.get("Robustness", "")
    assert "Automatic diagnostic checks" in rob
    # the flagged imbalance and its recommended remedy both make the paper
    assert "imbalance" in rob.lower()
    assert "sp.ebalance" in rob


def test_robustness_section_reports_a_clean_self_audit():
    draft = _paper(_obs_design(confounding=0.15))  # weak → matching balances well
    rob = draft.sections.get("Robustness", "")
    assert "Automatic diagnostic checks" in rob
    # a well-behaved fit must not manufacture a balance violation
    assert "notable residual" not in rob.lower()


def _iv_design(first_stage_coef: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = first_stage_coef * z + u + rng.normal(size=n)
    y = 1.0 * d + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


def test_liml_records_first_stage_f_so_weak_iv_is_flagged():
    """The workflow picks LIML for IV designs; it must still record the
    first-stage F so a weak instrument surfaces (previously silent)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weak = sp.liml("y ~ (d ~ z)", data=_iv_design(0.03))
        strong = sp.liml("y ~ (d ~ z)", data=_iv_design(2.0))
    assert weak.model_info["first_stage_f"] < 10
    assert "weak_instrument" in {v["test"] for v in weak.violations()}
    assert strong.model_info["first_stage_f"] > 10
    assert "weak_instrument" not in {v["test"] for v in strong.violations()}


def test_jive_records_first_stage_f_so_weak_iv_is_flagged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weak = sp.jive(data=_iv_design(0.03), y="y", x_endog=["d"], z=["z"])
        strong = sp.jive(data=_iv_design(2.0), y="y", x_endog=["d"], z=["z"])
    assert weak.model_info["first_stage_f"] < 10
    assert "weak_instrument" in {v["test"] for v in weak.violations()}
    assert strong.model_info["first_stage_f"] > 10
    assert "weak_instrument" not in {v["test"] for v in strong.violations()}


def test_paper_robustness_flags_weak_instrument():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        draft = sp.paper(_iv_design(0.03), y="y", treatment="d", instrument="z")
    rob = draft.sections.get("Robustness", "").lower()
    assert "weak instrument" in rob
