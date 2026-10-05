"""The translated call must carry Stata's own small-sample convention.

Replaying the Stock & Watson (4th ed.) replication logs through ``sp.stata``
showed two places where the translation ran the right model with the wrong
finite-sample factor:

* ``probit`` / ``logit`` / ``poisson`` / ``nbreg`` with ``vce(robust)`` were
  mapped to HC1 (``N/(N-K)``). Stata's robust VCE for a maximum-likelihood
  command carries ``N/(N-1)``.
* ``ivreg2`` / ``ivregress`` with ``robust`` and no ``small`` were mapped to
  HC1. Without ``small`` Stata applies no degrees-of-freedom factor.

Each convention is checked here against an independent implementation
(statsmodels, linearmodels), not against StatsPAI's own estimator.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(20261002)
    n = 400
    z = rng.normal(size=n)
    w = rng.normal(size=n)
    u = rng.normal(size=n)
    x = z + 0.5 * u + rng.normal(size=n)
    out = pd.DataFrame({"z": z, "w": w, "x": x, "g": np.repeat(np.arange(40), 10)})
    out["y"] = 1 + x + w + u * (1 + 0.5 * np.abs(z))
    out["d"] = (0.4 * out.w - 0.3 * out.z + rng.normal(size=n) > 0).astype(int)
    out["c"] = rng.poisson(np.exp(0.2 * out.w + 0.1 * out.z))
    return out


def _run(line, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stata(line, data=data)


@pytest.mark.parametrize("command", ["probit", "logit", "poisson"])
@pytest.mark.parametrize("option", ["r", "robust", "vce(robust)"])
def test_ml_robust_carries_n_over_n_minus_one(df, command, option):
    sm = pytest.importorskip("statsmodels.api")
    y = "c" if command == "poisson" else "d"
    X = sm.add_constant(df[["w", "z"]])
    model = {"probit": sm.Probit, "logit": sm.Logit, "poisson": sm.Poisson}[command]
    ref = model(df[y], X).fit(disp=0, cov_type="HC0")
    n = len(df)
    want = ref.bse[["w", "z"]].to_numpy() * np.sqrt(n / (n - 1))

    got = _run(f"{command} {y} w z, {option}", df)
    # rtol: both sides are Newton fits of the same likelihood; the sandwich
    # agrees to optimiser precision.
    np.testing.assert_allclose(got.std_errors[["w", "z"]].to_numpy(), want, rtol=1e-6)
    # ... and it is distinguishable from HC1 at this sample size
    hc1 = ref.bse[["w", "z"]].to_numpy() * np.sqrt(n / (n - 3))
    assert np.all(np.abs(got.std_errors[["w", "z"]].to_numpy() - hc1) > 1e-6 * hc1)


def test_ml_robust_translation_names_the_convention():
    for command in ("probit", "logit", "poisson", "nbreg"):
        out = sp.from_stata(f"{command} y x, vce(robust)")
        assert out["arguments"]["robust"] == "robust", command
        assert "robust='robust'" in out["python_code"], command
    # regress keeps HC1: that is what Stata's regress, robust computes
    assert sp.from_stata("regress y x, vce(robust)")["arguments"]["robust"] == "hc1"


@pytest.mark.parametrize(
    "line, cov, debiased",
    [
        # no `small`: large-sample robust, no degrees-of-freedom factor
        ("ivreg2 y w (x = z), robust", "robust", False),
        ("ivregress 2sls y w (x = z), vce(robust)", "robust", False),
        # `small`, and the legacy command: N/(N-K)
        ("ivreg2 y w (x = z), robust small", "robust", True),
        ("ivregress 2sls y w (x = z), vce(robust) small", "robust", True),
        ("ivreg y w (x = z), r", "robust", True),
        ("ivreg y (x = z) w", "unadjusted", True),
    ],
)
def test_iv_small_sample_convention(df, line, cov, debiased):
    iv = pytest.importorskip("linearmodels.iv")
    ref = iv.IV2SLS.from_formula("y ~ 1 + w + [x ~ z]", df).fit(
        cov_type=cov, debiased=debiased
    )
    got = _run(line, df)
    np.testing.assert_allclose(got.params["x"], ref.params["x"], rtol=1e-10)
    np.testing.assert_allclose(got.std_errors["x"], ref.std_errors["x"], rtol=1e-10)


def test_legacy_ivreg_cluster_matches_small_sample_cluster(df):
    iv = pytest.importorskip("linearmodels.iv")
    ref = iv.IV2SLS.from_formula("y ~ 1 + w + [x ~ z]", df).fit(
        cov_type="clustered", clusters=df.g, debiased=True
    )
    got = _run("ivreg y w (x = z), cluster(g)", df)
    np.testing.assert_allclose(got.std_errors["x"], ref.std_errors["x"], rtol=1e-10)
    out = sp.from_stata("ivreg y w (x = z), cluster(g)")
    assert out["untranslated_options"] == []
    assert not any("sqrt(N/(N-K))" in note for note in out["notes"])


# ------------------------------------------------- commands and descriptives
@pytest.mark.parametrize(
    "short, full",
    [
        ("regr y w, r", "regress y w, r"),
        ("summ y w", "summarize y w"),
        ("cor y w z", "correlate y w z"),
        ("prob d w, r", "probit d w, r"),
        ("logi d w", "logit d w"),
        ("poi c w", "poisson c w"),
        ("tob y w, ll(0)", "tobit y w, ll(0)"),
        ("te w", "test w"),
    ],
)
def test_command_abbreviations_translate_like_the_full_name(short, full):
    a, b = sp.from_stata(short), sp.from_stata(full)
    assert a["ok"], a
    assert a["python_code"] == b["python_code"]


@pytest.mark.parametrize("line", ["re y w", "s y", "co y w", "pro d w", "t w"])
def test_too_short_an_abbreviation_is_not_guessed(line):
    assert sp.from_stata(line)["ok"] is False


def test_correlate_is_casewise_and_pwcorr_is_pairwise(df):
    holes = df[["y", "w", "z"]].copy()
    holes.loc[:9, "z"] = np.nan
    cor = _run("correlate y w z", holes)
    pw = _run("pwcorr y w z", holes)
    complete = holes.dropna()
    np.testing.assert_allclose(
        cor.loc["y", "w"], np.corrcoef(complete.y, complete.w)[0, 1], rtol=1e-12
    )
    np.testing.assert_allclose(
        pw.loc["y", "w"], np.corrcoef(holes.y, holes.w)[0, 1], rtol=1e-12
    )
    assert cor.loc["y", "w"] != pw.loc["y", "w"]
    # an option that changes what is computed is reported, not dropped
    assert sp.from_stata("correlate y w, covariance")["untranslated_options"] == [
        "covariance"
    ]


def test_descriptive_commands_keep_the_estimates_for_test(df):
    direct = _run("regress y w z, r; test w z", df)
    interleaved = _run("regress y w z, r; summarize w; correlate w z; test w z", df)
    assert interleaved == direct


def test_tobit_vce_is_carried(df):
    censored = df.assign(yc=df.y.clip(lower=0))
    for line, kwargs in (
        ("tobit yc w z, ll(0) vce(robust)", {"vce": "robust"}),
        ("tobit yc w z, ll(0) vce(cluster g)", {"cluster": "g"}),
    ):
        out = sp.from_stata(line)
        assert out["untranslated_options"] == [], out
        got = _run(line, censored)
        want = sp.tobit(censored, y="yc", x=["w", "z"], ll=0.0, **kwargs)
        np.testing.assert_array_equal(
            got.std_errors.to_numpy(), want.std_errors.to_numpy()
        )
    plain = sp.tobit(censored, y="yc", x=["w", "z"], ll=0.0)
    assert not np.allclose(plain.std_errors.to_numpy(), got.std_errors.to_numpy())


def test_xtreg_fe_says_the_constant_is_absorbed():
    out = sp.from_stata("xtreg y x, fe i(id)")
    assert any("_cons" in line for line in out["semantics"])


def test_summarize_detail_uses_statas_percentile_rule_and_moments():
    out = sp.from_stata("summarize y, detail")
    args = out["arguments"]
    assert args["percentile_method"] == "stata"
    for name in ("p1", "p5", "p95", "p99", "variance", "skewness", "kurtosis"):
        assert name in args["stats"]
    assert any("smallest and largest" in line for line in out["semantics"])
    plain = sp.from_stata("summarize y")
    assert plain["semantics"] == [] and "percentile_method" not in plain["arguments"]


def test_statas_percentile_rule_by_hand():
    # N = 8: h = N * p / 100. p25 -> h = 2, an integer: mean of the 2nd and
    # 3rd order statistics. p10 -> h = 0.8: the 1st. p90 -> h = 7.2: the 8th.
    data = pd.DataFrame({"v": [3.0, 1.0, 4.0, 1.5, 9.0, 2.5, 6.0, 5.0]})
    got = sp.stata("summarize v, detail", data=data).loc["v"]
    assert got["P25"] == (1.5 + 2.5) / 2
    assert got["Median"] == (3.0 + 4.0) / 2
    assert got["P75"] == (5.0 + 6.0) / 2
    assert got["P10"] == 1.0 and got["P90"] == 9.0
    linear = sp.sumstats(data, stats=["p25", "p90"], output="numeric").loc["v"]
    assert linear["P25"] == 2.25 and linear["P90"] != 9.0
    # moments with divisor N: Stata's skewness and kurtosis
    x = data.v.to_numpy()
    d = x - x.mean()
    m2 = np.mean(d**2)
    np.testing.assert_allclose(got["Skewness"], np.mean(d**3) / m2**1.5)
    np.testing.assert_allclose(got["Kurtosis"], np.mean(d**4) / m2**2)
    np.testing.assert_allclose(got["Variance"], x.var(ddof=1))


def test_tabstat_translates_statistics_and_by(df):
    out = sp.from_stata("tabstat y w, s(mean sd q n) by(g) nototal")
    assert out["ok"] and out["untranslated_options"] == [], out
    assert out["arguments"]["stats"] == ["mean", "sd", "p25", "median", "p75", "n"]
    assert out["arguments"]["by"] == "g"
    assert out["arguments"]["percentile_method"] == "stata"
    table = _run("tabstat y w, s(mean sd) c(s)", df)
    np.testing.assert_allclose(table.loc["y", "Mean"], df.y.mean())
    assert sp.from_stata("tabstat y")["arguments"]["stats"] == ["mean"]
    assert sp.from_stata("tabstat y, s(gmean)")["ok"] is False
    total = sp.from_stata("tabstat y, by(g)")
    assert total["arguments"]["total"] is True  # tabstat's Total row


def test_sumstats_refuses_an_unknown_statistic(df):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="unknown statistic"):
        sp.sumstats(df, vars=["y"], stats=["mean", "mode"])
    with pytest.raises(MethodIncompatibility, match="percentile_method"):
        sp.sumstats(df, vars=["y"], percentile_method="r6")
