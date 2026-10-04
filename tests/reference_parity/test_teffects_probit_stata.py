"""Propensity-score estimators against Stata 18 on the bundled Lalonde data.

The reference numbers were produced by Stata 18 (``teffects`` and
``psmatch2`` 4.0.12) on ``sp.datasets.nsw_lalonde()`` written to CSV, with
two covariate lists:

* ``DISCRETE`` -- age, education and four indicators. Many units share
  every covariate, so many controls tie at the smallest score distance.
  This is the case that separates a tie rule decided by rounding from one
  decided by the data: before 2026-10 the fitted score of two units with
  the same covariates could differ in the last place, the tie was broken
  by it, and ``teffects psmatch`` was missed (1199.67 for 1209.72).
* ``FULL`` -- the same plus earnings in 1974 and 1975.

``teffects`` prints seven significant digits; the tolerances are two units
of the last digit printed. ``psmatch2`` returns ``r(att)`` in full but
holds the score in single precision, so its ATT agrees to about 1e-7.

    teffects psmatch (re78) (treat <x>[, probit]), atet | ate
        (both with the Abadie-Imbens (2016) standard error)
    teffects ipw     (re78) (treat <x>[, probit]), atet | ate
    teffects aipw    (re78 <x>) (treat <x>), ate
    psmatch2 treat <x>, outcome(re78) [logit]
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

DISCRETE = ["age", "educ", "black", "hispanic", "married", "nodegree"]
FULL = DISCRETE + ["re74", "re75"]


@pytest.fixture(scope="module")
def lalonde() -> pd.DataFrame:
    return sp.datasets.nsw_lalonde()


def _close(ours: float, printed: float, digits: int = 7) -> bool:
    """Within two units of the last of ``digits`` significant digits."""
    unit = 10.0 ** (np.floor(np.log10(abs(printed))) - digits + 1)
    return bool(abs(ours - printed) <= 2.0 * unit)


def _match(data: pd.DataFrame, covariates: list, estimand: str, model: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(
            data, y="re78", treat="treat", covariates=covariates,
            distance="propensity", estimand=estimand, ties="all",
            ps_model=model, se_method="abadie_imbens_2016",
        )  # fmt: skip


# --------------------------------------------------------- teffects psmatch
@pytest.mark.parametrize(
    "covariates, model, effect, se",
    [
        (DISCRETE, "logit", 1209.718, 1261.376),
        (DISCRETE, "probit", 118.834, 1347.901),
        (FULL, "logit", 1968.8, 1126.321),
        (FULL, "probit", 1260.967, 828.5266),
    ],
)
def test_psmatch_atet_matches_stata(lalonde, covariates, model, effect, se):
    r = _match(lalonde, covariates, "ATT", model)
    assert _close(r.estimate, effect)
    # Abadie-Imbens (2016), which charges for the estimated score
    assert _close(r.se, se)


@pytest.mark.parametrize(
    "covariates, model, effect, se",
    [
        (DISCRETE, "logit", 320.1955, 973.734),
        (FULL, "logit", -304.6074, 1076.527),
        (FULL, "probit", -204.2756, 1029.85),
    ],
)
def test_psmatch_ate_matches_stata(lalonde, covariates, model, effect, se):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(
            lalonde, y="re78", treat="treat", covariates=covariates,
            distance="propensity", estimand="ATE", ties="all",
            se_method="abadie_imbens_2016", ps_model=model,
        )  # fmt: skip
    assert _close(r.estimate, effect)
    # sigma^2 - c'Vc: the Stata 18 manual prints the adjustment with a plus
    # sign, e(V) has the minus sign of Abadie and Imbens (2016)
    assert _close(r.se, se)
    comps = r.model_info["ai2016_components"]
    assert comps["c_V_c"] > 0 and r.se < comps["base_se"]
    assert comps["ate"] == pytest.approx(r.estimate, rel=1e-12)


def test_psmatch_ate_variance_needs_the_matches_stata_makes(lalonde):
    kw = dict(
        y="re78", treat="treat", covariates=FULL, distance="propensity",
        estimand="ATE", se_method="abadie_imbens_2016",
    )  # fmt: skip
    for extra, message in (
        ({}, "ties"),
        ({"ties": "all", "caliper": 0.1}, "caliper"),
    ):
        with pytest.raises(sp.exceptions.MethodIncompatibility, match=message):
            sp.match(lalonde, **kw, **extra)


# ------------------------------------------------------------- teffects ipw
@pytest.mark.parametrize(
    "covariates, model, estimand, effect, se",
    [
        (DISCRETE, "probit", "ATT", 963.3853, 839.7896),
        (DISCRETE, "probit", "ATE", 197.6861, 842.0226),
        (DISCRETE, "logit", "ATE", 237.8791, 801.3467),
        (FULL, "probit", "ATT", 1231.182, 795.4452),
        (FULL, "probit", "ATE", 30.04422, 933.1056),
        (FULL, "logit", "ATE", 224.6763, 876.1932),
        (FULL, "logit", "ATT", 1214.071, 798.1546),
    ],
)
def test_ipw_sandwich_matches_stata(lalonde, covariates, model, estimand, effect, se):
    r = sp.ipw(
        lalonde, y="re78", treat="treat", covariates=covariates,
        estimand=estimand, se_method="sandwich", ps_model=model,
    )  # fmt: skip
    # the point estimate is compared on the scale of its standard error: an
    # effect of 30 with a standard error of 933 is printed to 1e-5, and the
    # probit's last iterations move it by 1e-4
    assert abs(r.estimate - effect) < 2e-6 * se
    assert _close(r.se, se)
    assert r.model_info["ps_model"] == model


def test_ipw_probit_default_is_unchanged(lalonde):
    """``ps_model`` defaults to the logit of earlier releases."""
    kw = dict(y="re78", treat="treat", covariates=FULL, se_method="sandwich")
    a = sp.ipw(lalonde, **kw)
    b = sp.ipw(lalonde, ps_model="logit", **kw)
    assert a.estimate == b.estimate and a.se == b.se


def test_ipw_rejects_unknown_ps_model(lalonde):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="ps_model"):
        sp.ipw(lalonde, y="re78", treat="treat", covariates=FULL, ps_model="cloglog")


# ------------------------------------------------------------ teffects aipw
def test_aipw_without_cross_fitting_matches_stata(lalonde):
    r = sp.aipw(
        lalonde, y="re78", treat="treat", covariates=DISCRETE,
        estimand="ATE", cross_fit=False, se_method="sandwich",
    )  # fmt: skip
    assert _close(r.estimate, 316.2065)
    assert _close(r.se, 858.4882)


# ----------------------------------------------------------------- psmatch2
@pytest.mark.parametrize(
    "covariates, model, att, se",
    [
        (DISCRETE, "probit", -388.938201698, 1311.157400795),
        (DISCRETE, "logit", 691.966832135, 1248.398639539),
        (FULL, "probit", 1345.52125436, 1179.55161749),
        (FULL, "logit", 1967.93965850, 1056.48676175),
    ],
)
def test_psmatch2_matches_stata(lalonde, covariates, model, att, se):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.psmatch2(
            lalonde, treat="treat", covariates=covariates, outcome="re78",
            ps_model=model,
        )  # fmt: skip
    # psmatch2 keeps _pscore as a float; 2e-7 is that storage, not a search
    assert abs(r.att - att) < 2e-7 * abs(att)
    assert abs(r.se - se) < 2e-7 * se
    assert r.result.model_info["propensity_model"] == model


# -------------------------------------------------- ties decided by the data
def _grid_sample(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 30, n).astype(float),
            "educ": rng.integers(8, 14, n).astype(float),
            "black": rng.integers(0, 2, n).astype(float),
            "married": rng.integers(0, 2, n).astype(float),
        }
    )
    index = -1 + 0.05 * (df.age - 24) + 0.2 * (df.educ - 11) + 0.4 * df.black
    df["d"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-index))).astype(int)
    df["y"] = 1 + 0.3 * df.age + df.educ + 2 * df.d + rng.normal(size=n) * 3
    return df


@pytest.mark.parametrize("model", ["logit", "probit"])
def test_units_with_the_same_covariates_get_the_same_score(model):
    from statspai.matching.match import MatchEstimator

    df = _grid_sample()
    X = df[["age", "educ", "black", "married"]].to_numpy(dtype=float)
    fit = MatchEstimator._logit_propensity_fit(X, df.d.to_numpy(float), model=model)
    _, cell = np.unique(X, axis=0, return_inverse=True)
    spread = pd.Series(fit["p_raw"]).groupby(np.ravel(cell)).agg(np.ptp)
    assert spread.max() == 0.0
    assert cell.max() + 1 < len(df)  # the sample does have repeated rows


@pytest.mark.parametrize("model", ["logit", "probit"])
def test_matching_does_not_depend_on_row_order(model):
    """With every tied control kept, shuffling the rows changes nothing.
    It did while rounding in the fitted score broke the ties."""
    df = _grid_sample()
    covariates = ["age", "educ", "black", "married"]
    results = []
    for seed in range(4):
        d = df.sample(frac=1, random_state=seed).reset_index(drop=True) if seed else df
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.match(
                d, y="y", treat="d", covariates=covariates,
                distance="propensity", estimand="ATT", ties="all",
                se_method="abadie_imbens_2016", ps_model=model,
            )  # fmt: skip
        results.append((r.estimate, r.se))
    estimates, ses = np.array(results).T
    assert np.ptp(estimates) < 1e-10
    assert np.ptp(ses) < 1e-10


def test_the_stata_reference_separates_the_tie_rules(lalonde, monkeypatch):
    """The reference above is one the old score computation misses, so it
    guards the fix and not only the estimator."""
    module = sys.modules["statspai.matching.match"]
    monkeypatch.setattr(module, "_index_by_row", lambda X, beta: X @ beta)
    r = _match(lalonde, DISCRETE, "ATT", "logit")
    if _close(r.estimate, 1209.718):
        pytest.skip("this BLAS rounds identical rows identically")
    assert abs(r.estimate - 1209.718) > 1.0


def test_match_rejects_unknown_ps_model(lalonde):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="ps_model"):
        sp.match(
            lalonde, y="re78", treat="treat", covariates=FULL,
            distance="propensity", ps_model="cloglog",
        )  # fmt: skip
