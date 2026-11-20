"""``result.report()`` for IV: the interval to report under a weak first stage.

The rules are those of Lee, McCrary, Moreira and Porter (2022) for one
instrument and Montiel Olea and Pflueger (2013) for several; the tests check
them against the functions that implement those papers and against coverage
in a known DGP.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _draw(seed, pi, k, n=2000, beta=0.5, rho=0.8):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, k))
    u = rng.normal(size=n)
    v = rho * u + np.sqrt(1 - rho**2) * rng.normal(size=n)
    x1 = rng.normal(size=n)
    d = Z @ np.full(k, pi) + v
    df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(k)])
    return df.assign(d=d, x1=x1, y=beta * d + 0.3 * x1 + u)


def _fit(df, k, **kw):
    zs = " + ".join(f"z{i}" for i in range(k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.ivreg(f"y ~ x1 + (d ~ {zs})", data=df, robust="hc1", **kw)


def test_strong_instrument_reports_the_conventional_interval():
    r = _fit(_draw(0, 0.5, 1), 1)
    rep = r.report()
    assert rep["status"] == "conventional"
    assert rep["ci_method"] == "wald"
    assert rep["ci"] == rep["conventional"]["ci"]
    assert rep["estimate"] == pytest.approx(float(r.params["d"]))
    assert rep["basis"] == []


def test_weak_single_instrument_reports_the_anderson_rubin_set():
    df = _draw(1, 0.12, 1)
    r = _fit(df, 1)
    rep = r.report()
    assert 4 < rep["first_stage_F"] < 104.7
    assert rep["status"] == "adjusted" and rep["ci_method"] == "anderson_rubin"
    ref = sp.anderson_rubin_test(
        df, y="y", endog="d", instruments=["z0"], exog=["x1"], ar_vcov="HC1"
    )
    assert rep["ci"] == pytest.approx(list(ref["ar_ci"]), rel=1e-10)
    # the tF interval is computed alongside, from the fit's own t-ratio
    crit = sp.tF_critical_value(rep["first_stage_F"])
    est, se = float(r.params["d"]), float(r.std_errors["d"])
    assert rep["alternatives"]["tF"]["ci"] == pytest.approx(
        [est - crit * se, est + crit * se]
    )
    # the fit's own numbers are untouched
    assert rep["conventional"]["ci"] == pytest.approx(
        [float(r.conf_int_lower["d"]), float(r.conf_int_upper["d"])]
    )


def test_unbounded_robust_set_is_a_refusal():
    rep = _fit(_draw(3, 0.02, 1), 1).report()
    assert rep["first_stage_F"] < 3.84
    assert rep["status"] == "refuse"
    assert rep["estimate"] is None and rep["ci"] is None
    assert "unbounded" in rep["refuse"]
    assert np.isfinite(rep["conventional"]["estimate"])


def test_clustered_fit_falls_back_to_tF():
    df = _draw(1, 0.12, 1)
    df["g"] = np.arange(len(df)) % 200
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.ivreg("y ~ x1 + (d ~ z0)", data=df, cluster="g")
    rep = r.report()
    assert rep["ci_method"] == "tF"
    assert "anderson_rubin" not in rep["alternatives"]
    half = (rep["ci"][1] - rep["ci"][0]) / 2
    crit = sp.tF_critical_value(rep["first_stage_F"])
    assert half == pytest.approx(crit * float(r.std_errors["d"]))


def test_many_weak_instruments_report_liml_with_the_anderson_rubin_set():
    df = _draw(5, 0.03, 19)
    rep = _fit(df, 19).report()
    assert rep["first_stage_F"] < rep["weak_threshold"]
    assert rep["status"] in ("adjusted", "refuse")
    if rep["status"] == "adjusted":
        assert rep["ci_method"] == "anderson_rubin"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zs = " + ".join(f"z{i}" for i in range(19))
            liml = sp.ivreg(f"y ~ x1 + (d ~ {zs})", data=df, method="liml")
        assert rep["estimate"] == pytest.approx(float(liml.params["d"]), rel=1e-8)


def test_coverage_many_weak_instruments():
    # 2SLS with 19 weak instruments is biased toward OLS: its Wald interval
    # covers the truth far below 95%, the reported set at or above it.
    conv = rep_cover = n_rep = 0
    reps = 120
    for s in range(reps):
        rep = _fit(_draw(100 + s, 0.03, 19), 19).report()
        lo, hi = rep["conventional"]["ci"]
        conv += lo <= 0.5 <= hi
        if rep["status"] != "refuse":
            n_rep += 1
            rep_cover += rep["ci"][0] <= 0.5 <= rep["ci"][1]
    assert conv / reps < 0.6
    assert n_rep > 0.8 * reps
    # binomial 3-sigma band below 0.95 at n_rep draws
    assert rep_cover / n_rep > 0.95 - 3 * np.sqrt(0.95 * 0.05 / n_rep)


def test_agent_payload_carries_the_report():
    r = _fit(_draw(1, 0.12, 1), 1)
    d = r.to_dict(detail="agent")
    assert d["report"]["status"] == "adjusted"
    assert d["ci"] == pytest.approx(d["report"]["conventional"]["ci"])
    assert "report" not in r.to_dict(detail="standard")


def test_results_without_a_rule_return_none():
    df = _draw(0, 0.5, 1)
    assert sp.regress("y ~ d + x1", data=df).report() is None
