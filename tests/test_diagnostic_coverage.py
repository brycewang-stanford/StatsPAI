"""Diagnostic coverage — no estimator may silently skip a diagnostic.

A ``result.violations()`` check only fires if the estimator populated the
``model_info`` key it reads. When one IV entry point stores ``first_stage_f``
and another does not, the weak-instrument warning silently vanishes for the
second — exactly the kind of gap that erodes trust (it was real for ``sp.liml``
and ``sp.jive`` until fixed). This suite pins the whole IV family: every
estimator must record the first-stage strength and flag a weak instrument, and
none may cry wolf on a strong one. A new IV estimator that forgets the
diagnostic fails here instead of shipping a blind spot.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _iv_df(first_stage_coef: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = first_stage_coef * z + u + rng.normal(size=n)
    y = 1.0 * d + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "z": z})


# (id, fitter) — every single-endogenous IV estimator StatsPAI exposes.
_IV_ESTIMATORS = [
    ("ivreg_2sls", lambda df: sp.ivreg("y ~ (d ~ z)", data=df)),
    ("iv_2sls", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="2sls")),
    ("iv_liml", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="liml")),
    ("iv_fuller", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="fuller")),
    ("iv_gmm", lambda df: sp.iv("y ~ (d ~ z)", data=df, method="gmm")),
    ("liml", lambda df: sp.liml("y ~ (d ~ z)", data=df)),
    ("jive", lambda df: sp.jive(df, y="y", x_endog=["d"], z=["z"])),
]


@pytest.mark.parametrize("name,fit", _IV_ESTIMATORS, ids=[e[0] for e in _IV_ESTIMATORS])
def test_iv_estimator_records_first_stage_and_flags_weak(name, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weak = fit(_iv_df(0.03))
        strong = fit(_iv_df(2.0))

    assert weak.model_info.get("first_stage_f") is not None, (
        f"{name}: model_info['first_stage_f'] is missing — weak IV would be "
        "silently skipped by result.violations()"
    )
    assert "weak_instrument" in {
        v["test"] for v in weak.violations()
    }, f"{name}: a weak first stage did not surface in violations()"
    assert "weak_instrument" not in {
        v["test"] for v in strong.violations()
    }, f"{name}: false-positive weak-instrument flag on a strong first stage"


# --------------------------------------------------------------------------- #
#  Panel — every clustered method records n_clusters
# --------------------------------------------------------------------------- #


def _panel_df(n_units: int, n_periods: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        a = rng.normal()
        for t in range(n_periods):
            x = rng.normal()
            rows.append({"id": i, "yr": t, "x": x, "y": x + a + rng.normal()})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("method", ["fe", "re", "twoway", "fd", "pooled"])
def test_panel_method_records_n_clusters_and_flags_few(method):
    def fit(df):
        return sp.panel(
            df, "y ~ x", entity="id", time="yr", method=method, cluster="entity"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        few = fit(_panel_df(12))
        many = fit(_panel_df(60))
    assert few.model_info.get("n_clusters") == 12, f"panel({method}) drops n_clusters"
    assert "few_clusters" in {v["test"] for v in few.violations()}
    assert "few_clusters" not in {v["test"] for v in many.violations()}


# --------------------------------------------------------------------------- #
#  Cluster-robust regressions — every ``cluster=`` estimator records n_clusters
# --------------------------------------------------------------------------- #


def _clustered_df(n_clusters: int, n: int = 1200, kind: str = "cont", seed: int = 1):
    """A clustered dataset with the requested number of clusters. ``kind``
    selects an outcome the estimator family accepts (continuous / binary /
    count)."""
    rng = np.random.default_rng(seed)
    g = rng.integers(0, n_clusters, n)
    x = rng.normal(size=n)
    if kind == "bin":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(int)
    elif kind == "count":
        y = rng.poisson(np.exp(0.4 + 0.3 * x))
    else:
        y = x + rng.normal(size=n)
    return pd.DataFrame(
        {"y": y, "x": x, "g": g, "d": x + rng.normal(size=n), "z": rng.normal(size=n)}
    )


# (id, fitter, outcome-kind) — every regression entry point that accepts a
# cluster spec. Each must record n_clusters so few-cluster CRV inference is
# flagged; feols routes clusters through pyfixest's ``_G`` instead of a
# ``cluster=`` kwarg, but must land in the same diagnostic.
_CLUSTER_ESTIMATORS = [
    ("regress", lambda df: sp.regress("y ~ x", data=df, cluster="g"), "cont"),
    ("ivreg", lambda df: sp.ivreg("y ~ (d ~ z)", data=df, cluster="g"), "cont"),
    ("logit", lambda df: sp.logit("y ~ x", data=df, cluster="g"), "bin"),
    ("probit", lambda df: sp.probit("y ~ x", data=df, cluster="g"), "bin"),
    ("poisson", lambda df: sp.poisson("y ~ x", data=df, cluster="g"), "count"),
    ("nbreg", lambda df: sp.nbreg("y ~ x", data=df, cluster="g"), "count"),
    ("feols", lambda df: sp.feols("y ~ x", data=df, vcov={"CRV1": "g"}), "cont"),
]


@pytest.mark.parametrize(
    "name,fit,kind",
    _CLUSTER_ESTIMATORS,
    ids=[e[0] for e in _CLUSTER_ESTIMATORS],
)
def test_cluster_estimator_records_n_clusters_and_flags_few(name, fit, kind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        few = fit(_clustered_df(12, kind=kind))
        many = fit(_clustered_df(60, kind=kind))
    assert few.model_info.get("n_clusters") == 12, (
        f"{name}: model_info['n_clusters'] missing — few-cluster CRV inference "
        "would be silently skipped by result.violations()"
    )
    assert "few_clusters" in {
        v["test"] for v in few.violations()
    }, f"{name}: 12 clusters did not surface as few_clusters in violations()"
    assert "few_clusters" not in {
        v["test"] for v in many.violations()
    }, f"{name}: false-positive few-cluster flag with 60 clusters"


# --------------------------------------------------------------------------- #
#  Matching — every method records the post-match balance table
# --------------------------------------------------------------------------- #


def _confounded(strength: float, n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(strength * x1 + 0.6 * strength * x2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 1 + 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


@pytest.mark.parametrize("method", ["psm", "nearest", "mahalanobis", "cem"])
def test_match_method_records_balance_and_flags_imbalance(method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(
            _confounded(1.5), y="y", treat="d", covariates=["x1", "x2"], method=method
        )
    assert isinstance(
        r.model_info.get("balance"), pd.DataFrame
    ), f"match({method}) does not record a balance table"
    assert "balance" in {v["test"] for v in r.violations()}


def test_cbps_reports_residual_imbalance():
    """CBPS stores balance under std_mean_diff_after (not `balance`) and is not
    tagged 'matching' — residual imbalance must still surface."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.cbps(_confounded(1.5), y="y", treat="d", covariates=["x1", "x2"])
    assert isinstance(r.model_info.get("std_mean_diff_after"), dict)
    assert "balance" in {v["test"] for v in r.violations()}


# --------------------------------------------------------------------------- #
#  Count — Poisson flags over-dispersion and excess zeros
# --------------------------------------------------------------------------- #


def test_poisson_flags_overdispersion_and_excess_zeros():
    rng = np.random.default_rng(0)
    x = rng.normal(size=600)
    lam = np.exp(0.5 + 0.8 * x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        over = sp.poisson(
            "y ~ x",
            data=pd.DataFrame({"y": rng.poisson(lam * rng.gamma(1, 1, 600)), "x": x}),
        )
        zinfl = pd.DataFrame({"y": rng.poisson(lam), "x": x})
        zinfl.loc[rng.uniform(size=600) < 0.4, "y"] = 0
        zi = sp.poisson("y ~ x", data=zinfl)
        clean = sp.poisson("y ~ x", data=pd.DataFrame({"y": rng.poisson(lam), "x": x}))
    assert "overdispersion" in {v["test"] for v in over.violations()}
    assert "excess_zeros" in {v["test"] for v in zi.violations()}
    clean_tests = {v["test"] for v in clean.violations()}
    assert "overdispersion" not in clean_tests and "excess_zeros" not in clean_tests


# --------------------------------------------------------------------------- #
#  DML / IPW / TMLE — propensity overlap
# --------------------------------------------------------------------------- #


def _confounded_overlap(strength: float, n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    ps = 1 / (1 + np.exp(-(strength * x1 + 0.6 * strength * x2)))
    d = (rng.uniform(size=n) < ps).astype(int)
    y = 1 + 2 * d + x1 + x2 + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})


_OVERLAP_ESTIMATORS = [
    (
        "dml_irm",
        lambda df: sp.dml(df, y="y", treat="d", covariates=["x1", "x2"], model="irm"),
    ),
    ("tmle", lambda df: sp.tmle(df, y="y", treat="d", covariates=["x1", "x2"])),
    ("ipw", lambda df: sp.ipw(df, y="y", treat="d", covariates=["x1", "x2"])),
]


@pytest.mark.parametrize(
    "name,fit", _OVERLAP_ESTIMATORS, ids=[e[0] for e in _OVERLAP_ESTIMATORS]
)
def test_propensity_estimator_flags_weak_overlap(name, fit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        strong = fit(_confounded_overlap(4.0))
        good = fit(_confounded_overlap(0.6))
    assert "dml_overlap" in {
        v["test"] for v in strong.violations()
    }, f"{name}: weak overlap did not surface in violations()"
    assert "dml_overlap" not in {
        v["test"] for v in good.violations()
    }, f"{name}: false-positive overlap flag on a well-overlapped design"


# --------------------------------------------------------------------------- #
#  Synthetic control — pre-fit diagnostics are recorded
# --------------------------------------------------------------------------- #


def _synth_bad_fit() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    rows = []
    for u in ["T"] + [f"D{i}" for i in range(8)]:
        base = 50 if u == "T" else rng.uniform(10, 30)
        slope = 5.0 if u == "T" else rng.uniform(-1, 1)
        for yr in range(1980, 1995):
            rows.append(
                {"u": u, "yr": yr, "y": base + slope * (yr - 1980) + rng.normal(0, 1)}
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "name,fit",
    [
        (
            "synth",
            lambda df: sp.synth(
                df,
                unit="u",
                time="yr",
                outcome="y",
                treated_unit="T",
                treatment_time=1990,
            ),
        ),
        (
            "augsynth",
            lambda df: sp.augsynth(
                df,
                unit="u",
                time="yr",
                outcome="y",
                treated_unit="T",
                treatment_time=1990,
            ),
        ),
        (
            "gsynth",
            lambda df: sp.gsynth(
                df,
                unit="u",
                time="yr",
                outcome="y",
                treated_unit="T",
                treatment_time=1990,
            ),
        ),
    ],
)
def test_synth_records_prefit_diagnostics(name, fit):
    """Every SCM variant must record the pre-fit inputs so synth_prefit can be
    assessed (augsynth/gsynth fit the pre-period well, so they need not fire —
    but they must not be blind)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fit(_synth_bad_fit())
    mi = r.model_info
    for key in ("pre_treatment_rmse", "Y_treated", "times", "treatment_time"):
        assert mi.get(key) is not None, f"{name}: model_info['{key}'] missing"


def test_plain_scm_flags_unmatchable_pre_trend():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.synth(
            _synth_bad_fit(),
            unit="u",
            time="yr",
            outcome="y",
            treated_unit="T",
            treatment_time=1990,
        )
    assert "synth_prefit" in {v["test"] for v in r.violations()}


# --------------------------------------------------------------------------- #
#  Limited-dependent / survival — assumption diagnostics
#
#  These estimators each carry a signature assumption whose violation quietly
#  invalidates the headline coefficient: Cox's proportional hazards, Tobit's
#  usable variation under censoring, Heckman's numerical identification, and a
#  logit's finiteness under separation. Each check only fires if the estimator
#  stored the statistic it reads (ph_test / censor_pct / rho / coefs) — pin the
#  storage AND the fire/clean behaviour so none silently regresses to a blind
#  spot the way the IV family once did.
# --------------------------------------------------------------------------- #


def test_cox_flags_nonproportional_hazards():
    """PH-violating data (covariate trends with failure time) must reject the
    proportional-hazards test; textbook proportional data must not."""
    rng = np.random.default_rng(11)
    n = 700
    t_bad = np.sort(rng.exponential(1.0, n)) + 0.01
    x_bad = np.linspace(-2, 2, n) + rng.normal(0, 0.3, n)  # x rises with time
    xg = np.random.default_rng(3).normal(size=n)
    t_good = -np.log(np.random.default_rng(3).uniform(size=n)) / np.exp(0.8 * xg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = sp.cox(
            data=pd.DataFrame({"t": t_bad, "d": np.ones(n, int), "x": x_bad}),
            duration="t",
            event="d",
            x=["x"],
        )
        good = sp.cox(
            data=pd.DataFrame({"t": t_good + 0.001, "d": np.ones(n, int), "x": xg}),
            duration="t",
            event="d",
            x=["x"],
        )
    assert bad.model_info.get("ph_test") is not None, (
        "cox: model_info['ph_test'] missing — the proportional-hazards check "
        "would be silently skipped by result.violations()"
    )
    assert "proportional_hazards" in {v["test"] for v in bad.violations()}
    assert "proportional_hazards" not in {v["test"] for v in good.violations()}


def test_tobit_flags_extreme_censoring():
    rng = np.random.default_rng(5)
    n = 800
    x = rng.normal(size=n)
    y_bad = np.maximum(-3.0 + 1.0 * x + rng.normal(size=n), 0.0)  # ~98% at floor
    y_good = np.maximum(1.0 + 2.0 * x + rng.normal(size=n), 0.0)  # ~33% censored
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bad = sp.tobit(pd.DataFrame({"y": y_bad, "x": x}), y="y", x=["x"], ll=0)
        good = sp.tobit(pd.DataFrame({"y": y_good, "x": x}), y="y", x=["x"], ll=0)
    assert bad.model_info.get("censor_pct") is not None, (
        "tobit: model_info['censor_pct'] missing — the extreme-censoring check "
        "would be silently skipped by result.violations()"
    )
    assert "extreme_censoring" in {v["test"] for v in bad.violations()}
    assert "extreme_censoring" not in {v["test"] for v in good.violations()}


def test_heckman_flags_rho_boundary():
    """When the outcome error is (near-)perfectly correlated with the selection
    error, rho hits the ±1 boundary — a numerical red flag the check must
    surface; a well-identified moderate-rho fit must not."""
    rng_b = np.random.default_rng(9)
    n = 600
    z = rng_b.normal(size=n)
    x = rng_b.normal(size=n)
    u = rng_b.normal(size=n)
    sel = 0.5 + 0.8 * z + u > 0
    y = 1 + 2 * x + 3 * u  # outcome error == selection error => rho -> 1
    boundary_df = pd.DataFrame(
        {"y": np.where(sel, y, np.nan), "x": x, "z": z, "sel": sel.astype(int)}
    )

    rng_c = np.random.default_rng(5)
    m = 2000
    zc = rng_c.normal(size=m)
    xc = rng_c.normal(size=m)
    uc = rng_c.normal(size=m)
    epsc = 0.6 * uc + np.sqrt(1 - 0.6**2) * rng_c.normal(size=m)  # rho ~ 0.6
    selc = 0.3 + 1.0 * zc + 0.5 * xc + uc > 0
    yc = 1 + 2 * xc + 3 * epsc
    clean_df = pd.DataFrame(
        {"y": np.where(selc, yc, np.nan), "x": xc, "z": zc, "sel": selc.astype(int)}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boundary = sp.heckman(boundary_df, y="y", x=["x"], select="sel", z=["z"])
        clean = sp.heckman(clean_df, y="y", x=["x"], select="sel", z=["z"])
    assert boundary.model_info.get("rho") is not None, (
        "heckman: model_info['rho'] missing — the rho-boundary check would be "
        "silently skipped by result.violations()"
    )
    assert "heckman_rho_boundary" in {v["test"] for v in boundary.violations()}
    assert "heckman_rho_boundary" not in {v["test"] for v in clean.violations()}


def test_logit_flags_separation():
    rng = np.random.default_rng(0)
    n = 400
    xs = rng.normal(size=n)
    y_sep = (xs > 0).astype(int)  # outcome perfectly predicted by x
    xn = rng.normal(size=n)
    y_clean = (rng.uniform(size=n) < 1 / (1 + np.exp(-xn))).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sep = sp.logit("y ~ x", data=pd.DataFrame({"y": y_sep, "x": xs}))
        clean = sp.logit("y ~ x", data=pd.DataFrame({"y": y_clean, "x": xn}))
    assert "separation" in {v["test"] for v in sep.violations()}
    assert "separation" not in {v["test"] for v in clean.violations()}
