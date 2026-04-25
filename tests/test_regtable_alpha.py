"""
Tests for the ``alpha`` parameter on ``regtable`` / ``esttab`` CI display.
"""

import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def ols_models():
    rng = np.random.default_rng(2026)
    n = 600
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    y = 1.0 + 0.5 * x1 + 0.25 * x2 + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    m1 = sp.regress("y ~ x1", data=df)
    m2 = sp.regress("y ~ x1 + x2", data=df)
    return m1, m2


def _ci_from_text(txt, var):
    """Extract `[lo, hi]` immediately after the row label."""
    pattern = re.compile(rf"{re.escape(var)}.*?\[\s*(-?[\d.]+),\s*(-?[\d.]+)\s*\]", re.S)
    m = pattern.search(txt)
    assert m, f"could not locate CI for {var} in:\n{txt}"
    return float(m.group(1)), float(m.group(2))


def test_regtable_alpha_default_uses_95_ci_label(ols_models):
    m1, m2 = ols_models
    txt = sp.regtable(m1, m2, se_type="ci").to_text()
    assert "95% CI" in txt
    assert "90% CI" not in txt
    assert "99% CI" not in txt


def test_regtable_alpha_changes_label(ols_models):
    m1, m2 = ols_models
    assert "90% CI" in sp.regtable(m1, m2, se_type="ci", alpha=0.10).to_text()
    assert "99% CI" in sp.regtable(m1, m2, se_type="ci", alpha=0.01).to_text()


def test_regtable_alpha_changes_ci_width(ols_models):
    m1, _ = ols_models
    txt95 = sp.regtable(m1, se_type="ci", alpha=0.05).to_text()
    txt90 = sp.regtable(m1, se_type="ci", alpha=0.10).to_text()
    txt99 = sp.regtable(m1, se_type="ci", alpha=0.01).to_text()

    lo90, hi90 = _ci_from_text(txt90, "x1")
    lo95, hi95 = _ci_from_text(txt95, "x1")
    lo99, hi99 = _ci_from_text(txt99, "x1")

    w90, w95, w99 = hi90 - lo90, hi95 - lo95, hi99 - lo99
    # Stricter alpha (smaller alpha) ⇒ wider CI
    assert w90 < w95 < w99


def test_regtable_alpha_invalid_raises(ols_models):
    m1, _ = ols_models
    with pytest.raises(ValueError, match="alpha"):
        sp.regtable(m1, se_type="ci", alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        sp.regtable(m1, se_type="ci", alpha=1.0)


def test_esttab_alpha_changes_label_and_width(ols_models):
    m1, m2 = ols_models
    txt95 = sp.esttab(m1, m2, ci=True).to_text()
    txt90 = sp.esttab(m1, m2, ci=True, alpha=0.10).to_text()
    assert "95% CI" in txt95
    assert "90% CI" in txt90

    lo95, hi95 = _ci_from_text(txt95, "x1")
    lo90, hi90 = _ci_from_text(txt90, "x1")
    assert (hi90 - lo90) < (hi95 - lo95)


def test_alpha_recomputed_ci_matches_manual(ols_models):
    """For alpha != 0.05 we recompute b ± crit·se; verify against scipy."""
    from scipy import stats as sp_stats

    m1, _ = ols_models
    b = float(m1.params["x1"])
    se = float(m1.std_errors["x1"])
    df_resid = (m1.diagnostics.get("df_resid")
                or m1.data_info.get("df_resid")
                or (m1.diagnostics.get("N") or m1.data_info.get("nobs"))
                - len(m1.params))

    txt = sp.regtable(m1, se_type="ci", alpha=0.10, fmt="%.6f").to_text()
    lo, hi = _ci_from_text(txt, "x1")

    crit = sp_stats.t.ppf(0.95, df_resid)
    expected_lo = b - crit * se
    expected_hi = b + crit * se
    assert lo == pytest.approx(expected_lo, rel=1e-3, abs=1e-4)
    assert hi == pytest.approx(expected_hi, rel=1e-3, abs=1e-4)
