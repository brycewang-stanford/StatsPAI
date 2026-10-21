"""``sp.from_r`` additions of 2026-10-19: fixest ``sunab()``, rdrobust
``covs``, and the ``dropped`` / ``complete`` keys both translators share.

The fixest numbers are R 4.5 / fixest 0.14.0 on the panel built by
``_sunab_panel`` (written to CSV with ``%.17g`` and read back in R, the
never-treated cohort coded 10000 there and 0 here)::

    m  <- feols(y ~ sunab(cohort_r, year) | id + year, d, cluster = ~id)
    m2 <- feols(y ~ x + sunab(cohort_r, year) | id + year, d, cluster = ~id)
    summary(m, agg = "att")

printed with ``sprintf("%.15e")``. The tolerance 1e-8 sits above the
iterative within-transformation (agreement is about 5e-11) and far below
any difference in the estimator.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

RTOL = 1e-8

#: relative period -> (coefficient, clustered SE), fixest sunab
FIXEST = {
    -7: (3.957357949047033e-01, 3.008836829856485e-01),
    -2: (-2.238377218590448e-01, 2.019201086658202e-01),
    0: (2.962991827240037e-01, 1.889828269499818e-01),
    3: (1.562985491380235e00, 2.500538544646721e-01),
    6: (3.291012920889671e00, 3.864571430066321e-01),
}
FIXEST_ATT = (1.426857534130324e00, 1.614337130640106e-01)
FIXEST_COV = {
    -7: (3.787964533086265e-01, 2.753999516949661e-01),
    0: (2.578922082635514e-01, 1.843846718521196e-01),
    6: (3.354697744663826e00, 3.648896845767357e-01),
}


def _sunab_panel() -> pd.DataFrame:
    r = np.random.default_rng(3)
    rows = []
    for i in range(150):
        c = r.choice([0, 4, 6, 8])
        a = r.normal()
        for t in range(1, 11):
            te = (t - c + 1) * 0.5 if (c and t >= c) else 0
            x = r.normal()
            rows.append((i, t, c, x, a + 0.1 * t + 0.3 * x + te + r.normal()))
    return pd.DataFrame(rows, columns=["id", "year", "cohort", "x", "y"])


def _run(out: dict, data: pd.DataFrame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, out["tool"])(data=data, **out["arguments"])


def _event_study(result) -> dict:
    es = result.model_info["event_study"]
    return {
        int(t): (float(b), float(s))
        for t, b, s in zip(es["relative_time"], es["att"], es["se"])
    }


def test_sunab_translates_to_sun_abraham_and_reproduces_fixest():
    out = sp.from_r(
        "feols(y ~ sunab(cohort, year) | id + year, data = df, cluster = ~id)"
    )
    assert out["ok"] and out["complete"] and out["dropped"] == []
    assert out["tool"] == "sun_abraham"
    assert out["arguments"] == {
        "y": "y",
        "g": "cohort",
        "t": "year",
        "i": "id",
        "parity": "r",
    }
    result = _run(out, _sunab_panel())
    # fixest's conventions: fixed cohort shares, t(G - 1) with clustered SEs
    conv = result.model_info["conventions"]
    assert conv["share_variance"] is False and conv["crit_dist"] == "t"
    assert conv["crit_df"] == _sunab_panel()["id"].nunique() - 1
    es = _event_study(result)
    for k, (b, se) in FIXEST.items():
        assert np.isclose(es[k][0], b, rtol=RTOL, atol=0), k
        assert np.isclose(es[k][1], se, rtol=RTOL, atol=0), k
    att, att_se = FIXEST_ATT
    assert np.isclose(result.model_info["att_fixest_att"], att, rtol=RTOL)
    assert np.isclose(result.model_info["se_fixest_att"], att_se, rtol=RTOL)


def test_sunab_with_a_covariate():
    out = sp.from_r(
        "feols(y ~ x + sunab(cohort, year) | id + year, data = df, cluster = ~id)"
    )
    assert out["arguments"]["covariates"] == ["x"]
    es = _event_study(_run(out, _sunab_panel()))
    for k, (b, se) in FIXEST_COV.items():
        assert np.isclose(es[k][0], b, rtol=RTOL, atol=0), k
        assert np.isclose(es[k][1], se, rtol=RTOL, atol=0), k


def test_sunab_without_a_variance_is_incomplete():
    """fixest 0.14 reports IID SEs without vcov/cluster; sp.sun_abraham
    clusters, so the translation must not say it is complete."""
    out = sp.from_r("feols(y ~ sunab(cohort, year) | id + year, data = df)")
    assert out["ok"] and out["complete"] is False
    assert out["dropped"] == ["vcov"] == out["untranslated_arguments"]


@pytest.mark.parametrize(
    "line",
    [
        "feols(y ~ sunab(cohort, year, ref.p = c(-1, -5)) | id + year, data = df)",
        "feols(y ~ sunab(cohort, year) | id, data = df)",
        "feols(y ~ sunab(cohort, year) | id + year + region, data = df)",
        "feols(y ~ sunab(cohort, year) | id + year, data = df, vcov = 'hetero')",
        "feols(y ~ sunab(cohort, year) | id + year, data = df, cluster = ~id + year)",
    ],
)
def test_sunab_designs_outside_the_mapping_are_refused(line):
    out = sp.from_r(line)
    assert out["ok"] is False
    assert "sun_abraham" in out.get("suggestions", []) or "sunab" in out["error"]


@pytest.mark.parametrize(
    "covs",
    [
        "cbind(df$z1, df$z2)",
        'df[, c("z1", "z2")]',
        'as.matrix(df[c("z1", "z2")])',
    ],
)
def test_rdrobust_covs_name_columns(covs):
    out = sp.from_r(f"rdrobust(df$y, df$x, covs = {covs})")
    assert out["ok"] and out["complete"]
    assert out["arguments"]["covs"] == ["z1", "z2"]
    assert "covs=['z1', 'z2']" in out["python_code"]


def test_rdrobust_computed_covs_are_dropped_out_loud():
    out = sp.from_r("rdrobust(df$y, df$x, covs = df$z1 * 2)")
    assert out["ok"] and out["complete"] is False
    assert out["dropped"] == ["covs"]
    assert "covs" not in out["arguments"]


def test_dropped_and_complete_are_shared_by_both_translators():
    stata = sp.from_stata("regress y x if d == 1, hascons")
    assert stata["complete"] is False
    assert stata["dropped"] == ["hascons", "if d == 1"]
    assert stata["untranslated_options"] == ["hascons"]  # old key kept
    assert stata["unapplied_sample"] == "if d == 1"
    assert sp.from_stata("regress y x, vce(robust)")["complete"] is True
    # display-only options are not dropped pieces
    assert sp.from_stata("regress y x, noheader")["complete"] is True
    r = sp.from_r("lm(y ~ x, data = df, subset = x > 0)")
    assert r["complete"] is False and r["dropped"] == ["subset"]
    assert r["untranslated_arguments"] == ["subset"]  # old key kept
    assert sp.from_r("lm(y ~ x, data = df)")["complete"] is True
