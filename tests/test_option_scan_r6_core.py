"""Option scan round 6: ``core/`` (result objects, formula parsing).

Each test states the correct behaviour; the fix that made it pass is in the
same commit.

* ``CausalResult.tidy(conf_level=)`` recomputed the headline interval but
  left every event-study and group-time row at the fit-time (95%) width,
  labelled as the requested level.
* An event-study table without stored bounds fell back to ``att -/+ 1.96 se``
  whatever ``alpha`` the fit was run at.
* ``CausalResult.summary(alpha=0.1)`` printed the 95% interval under a
  ``[90% CI]`` label.
* ``y ~ x - 1 + 1`` and ``y ~ x - 0`` fitted a model without an intercept;
  R ``lm`` and patsy both keep it (a later ``+ 1`` re-adds it and ``- 0``
  means ``+ 1``).
* ``EconometricResults.coef(term)`` reported a 95% interval after a fit at
  ``alpha=0.1`` while ``tidy()``, which it claims to abbreviate, used 90%.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.core.results import CausalResult
from statspai.core.utils import parse_formula


def _cs_result():
    df = sp.dgp_did(n_units=60, n_periods=6, staggered=True, seed=1)
    df["g"] = df["first_treat"].fillna(0).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(df, y="y", g="g", t="time", i="unit")


# --------------------------------------------------------------------------
# CausalResult intervals at a level other than the fit-time one
# --------------------------------------------------------------------------


def test_tidy_conf_level_rescales_every_row() -> None:
    r = _cs_result()
    assert r.alpha == 0.05 and r.model_info.get("df_resid") is None
    t90 = r.tidy(conf_level=0.90)
    z90 = stats.norm.ppf(0.95)
    rows = t90[t90["type"] != "main"]
    assert {"event_study", "group_time"} <= set(rows["type"])
    # pointwise interval at the requested level: estimate -/+ z * se
    np.testing.assert_allclose(
        rows["conf_low"], rows["estimate"] - z90 * rows["std_error"], rtol=1e-12
    )
    np.testing.assert_allclose(
        rows["conf_high"], rows["estimate"] + z90 * rows["std_error"], rtol=1e-12
    )


def test_tidy_at_fit_level_keeps_stored_rows() -> None:
    r = _cs_result()
    es = r.model_info["event_study"]
    t = r.tidy()
    rows = t[t["type"] == "event_study"]
    np.testing.assert_array_equal(rows["conf_low"].to_numpy(), es["ci_lower"])
    np.testing.assert_array_equal(
        r.tidy(conf_level=0.95)["conf_low"].to_numpy(), t["conf_low"].to_numpy()
    )


def _bare_result(alpha: float) -> CausalResult:
    """A result whose event-study table carries no interval columns."""
    es = pd.DataFrame(
        {"relative_time": [-1, 0, 1], "att": [0.1, 1.0, 2.0], "se": [0.2, 0.3, 0.4]}
    )
    z = stats.norm.ppf(1 - alpha / 2)
    return CausalResult(
        method="test",
        estimand="ATT",
        estimate=1.5,
        se=0.5,
        pvalue=0.01,
        ci=(1.5 - z * 0.5, 1.5 + z * 0.5),
        alpha=alpha,
        n_obs=100,
        model_info={"event_study": es},
    )


def test_event_study_rows_without_bounds_use_the_fit_alpha() -> None:
    r = _bare_result(alpha=0.10)
    rows = r.tidy().query("type == 'event_study'")
    z90 = stats.norm.ppf(0.95)
    np.testing.assert_allclose(
        rows["conf_high"] - rows["conf_low"], 2 * z90 * rows["std_error"], rtol=1e-12
    )


def test_summary_alpha_prints_the_interval_it_labels() -> None:
    r = _bare_result(alpha=0.05)
    text = str(r.summary(alpha=0.10, digits=6))
    assert "[90% CI]" in text
    lo, hi = (
        r.tidy(conf_level=0.90)
        .query("type == 'main'")[["conf_low", "conf_high"]]
        .iloc[0]
    )
    line = next(s for s in text.splitlines() if "[90% CI]" in s)
    printed = [float(v) for v in line.split("[", 2)[2].strip(" ]").split(",")]
    np.testing.assert_allclose(printed, [lo, hi], atol=5e-7)
    # the fit-time level still prints the stored interval
    text95 = str(r.summary(digits=6))
    line = next(s for s in text95.splitlines() if "[95% CI]" in s)
    printed = [float(v) for v in line.split("[", 2)[2].strip(" ]").split(",")]
    np.testing.assert_allclose(printed, list(r.ci), atol=5e-7)


def test_summary_alpha_with_a_non_symmetric_interval_keeps_its_level() -> None:
    # a bootstrap / bias-corrected interval is not estimate -/+ crit * se,
    # so it cannot be rescaled; it is printed at its own level with a note
    r = _bare_result(alpha=0.05)
    r.ci = (0.2, 2.9)
    text = str(r.summary(alpha=0.10, digits=6))
    assert "[90% CI]" not in text
    line = next(s for s in text.splitlines() if "[95% CI]" in s)
    printed = [float(v) for v in line.split("[", 2)[2].strip(" ]").split(",")]
    np.testing.assert_allclose(printed, [0.2, 2.9], atol=5e-7)
    assert "alpha=0.1" in text


# --------------------------------------------------------------------------
# Intercept spellings in the fast formula path and parse_formula
# --------------------------------------------------------------------------


def _xy() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    x = rng.normal(size=80)
    return pd.DataFrame({"x": x, "y": 2.0 + 0.5 * x + rng.normal(size=80)})


@pytest.mark.parametrize(
    "formula, has_intercept",
    [
        # R lm: names(coef(lm(f))) has (Intercept) for the first three
        ("y ~ x - 1 + 1", True),
        ("y ~ x - 0", True),
        ("y ~ 0 + x + 1", True),
        ("y ~ 1 + x - 1", False),
        ("y ~ x + 0", False),
        ("y ~ x - 1", False),
    ],
)
def test_intercept_spelling_follows_r(formula: str, has_intercept: bool) -> None:
    df = _xy()
    fit = sp.regress(formula, data=df)
    ref = sp.regress("y ~ x" if has_intercept else "y ~ 0 + x", data=df)
    assert list(fit.params.index) == list(ref.params.index)
    np.testing.assert_array_equal(fit.params.to_numpy(), ref.params.to_numpy())
    assert parse_formula(formula)["has_constant"] is has_intercept


# --------------------------------------------------------------------------
# coef() defaults to the fit-time level, as tidy() does
# --------------------------------------------------------------------------


def test_coef_default_level_is_the_fit_level() -> None:
    df = _xy()
    df["d"] = (df["y"] > 2.0).astype(int)
    fit = sp.logit("d ~ x", data=df, alpha=0.10)
    row = fit.coef("x")
    want = fit.tidy().set_index("term").loc["x"]
    assert row["conf_low"] == want["conf_low"]
    assert row["conf_high"] == want["conf_high"]
    np.testing.assert_allclose(
        row["conf_high"] - row["conf_low"],
        2 * stats.norm.ppf(0.95) * row["std_error"],
        rtol=1e-12,
    )
    # an explicit level still wins
    row95 = fit.coef("x", conf_level=0.95)
    np.testing.assert_allclose(
        row95["conf_high"] - row95["conf_low"],
        2 * stats.norm.ppf(0.975) * row95["std_error"],
        rtol=1e-12,
    )


# --------------------------------------------------------------------------
# predict(what='confidence') uses the fit's reference distribution
# --------------------------------------------------------------------------


def _pred_data() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = 160
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "z": rng.normal(size=n), "h": np.arange(n) % 8}
    )
    df["y"] = 1.0 + df.x - 0.5 * df.z + rng.normal(size=n) * (1 + 0.3 * df.h)
    df["pos"] = np.exp(0.3 + 0.2 * df.x) * rng.gamma(4.0, 0.25, size=n)
    return df


@pytest.mark.parametrize(
    "fit",
    [
        # t(G - 1) under clustering, as regress, vce(cluster) and margins use
        lambda d: sp.regress("y ~ x + z", data=d, cluster="h"),
        # z, as glm and margins after glm use
        lambda d: sp.glm("y ~ x + z", data=d),
        lambda d: sp.glm("y ~ x + z", data=d, cluster="h"),
        lambda d: sp.regress("y ~ x + z", data=d),
        lambda d: sp.logit("d ~ x + z", data=d.assign(d=(d.y > 1).astype(int))),
    ],
    ids=["regress-cluster", "glm-gaussian", "glm-cluster", "regress", "logit"],
)
def test_prediction_at_origin_is_the_intercept_interval(fit) -> None:
    # at x = z = 0 the linear index is the intercept, with the same SE, so
    # its interval on the index scale is the intercept's own interval
    df = _pred_data()
    res = fit(df)
    new = pd.DataFrame({"x": [0.0], "z": [0.0]})
    eta = res.predict(new, what="link")
    lo, hi = res.conf_int().iloc[0]
    assert eta[0] == pytest.approx(res.params.iloc[0], rel=1e-12)
    out = res.predict(new, what="confidence")
    inverse = res._inverse_link() or (lambda v: v)
    np.testing.assert_allclose(
        [out["lower"].iloc[0], out["upper"].iloc[0]],
        [inverse(np.array(lo)), inverse(np.array(hi))],
        rtol=1e-10,
    )


def test_glm_log_link_prediction_interval_maps_the_intercept_interval() -> None:
    df = _pred_data()
    res = sp.glm("pos ~ x + z", data=df, family="gamma", link="log")
    out = res.predict(pd.DataFrame({"x": [0.0], "z": [0.0]}), what="confidence")
    lo, hi = res.conf_int().iloc[0]
    np.testing.assert_allclose(
        [out["lower"].iloc[0], out["upper"].iloc[0]], np.exp([lo, hi]), rtol=1e-10
    )


# --------------------------------------------------------------------------
# nonparametric
# --------------------------------------------------------------------------


def test_lpoly_alpha_sets_the_interval_and_is_validated() -> None:
    df = _pred_data()
    r95 = sp.lpoly(df, y="y", x="x", bandwidth=0.8, n_grid=5)
    r80 = sp.lpoly(df, y="y", x="x", bandwidth=0.8, n_grid=5, alpha=0.2)
    np.testing.assert_allclose(
        r80.ci_upper - r80.ci_lower, 2 * stats.norm.ppf(0.9) * r95.se, rtol=1e-12
    )
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.lpoly(df, y="y", x="x", bandwidth=0.8, alpha=5)
