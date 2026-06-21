"""Coverage tests for statspai.did.pretrends (pretrends_test / pretrends_power /
sensitivity_rr / pretrends_summary).

Uses a real OLS event-study result from sp.event_study on a small synthetic
staggered panel. Asserts real properties (p-values in [0,1], power in [0,1],
CI ordering, breakdown behaviour, error raising on bad input) — no fabricated
"expected" numbers.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult
from statspai.did import pretrends as ptmod
from statspai.exceptions import MethodIncompatibility


def _staggered_event_study(seed=0, window=(-3, 3)):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(60):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        for t in range(1, 11):
            te = 1.0 * (t - g + 1) if (g > 0 and t >= g) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": fe + 0.3 * t + te + rng.normal(0, 0.4),
                    "g": g,
                }
            )
    df = pd.DataFrame(rows)
    df["ft"] = df["g"].replace(0, np.nan)
    return sp.event_study(
        df, y="y", treat_time="ft", time="time", unit="unit", window=window
    )


@pytest.fixture(scope="module")
def es_result():
    return _staggered_event_study()


# ----------------------------------------------------------------------
# pretrends_test
# ----------------------------------------------------------------------


def test_pretrends_test_wald(es_result):
    out = sp.pretrends_test(es_result, type="wald")
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["type"] == "wald"
    assert out["df"] >= 1
    assert out["statistic"] >= 0.0
    assert isinstance(out["reject"], (bool, np.bool_))
    assert "chi2" in out["stat_label"]
    assert "parallel" in out["interpretation"].lower()


def test_pretrends_test_f(es_result):
    out = sp.pretrends_test(es_result, type="f")
    assert 0.0 <= out["pvalue"] <= 1.0
    assert out["type"] == "f"
    assert out["stat_label"].startswith("F(")


def test_pretrends_test_bad_type(es_result):
    with pytest.raises(MethodIncompatibility):
        sp.pretrends_test(es_result, type="bogus")


def test_pretrends_test_rejects_invalid_alpha(es_result):
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.pretrends_test(es_result, alpha=1.0)


def test_pretrends_test_with_full_vcv(es_result):
    # Supply an explicit full-rank vcv_pre aligned with estimated pre-periods.
    es = es_result.model_info["event_study"]
    pre = es[es["relative_time"] < 0]
    k = int((pre["se"].values > 0).sum())
    rng = np.random.default_rng(1)
    A = rng.normal(size=(k, k))
    vcv = A @ A.T + np.eye(k)  # SPD
    r2 = CausalResult(
        method="es",
        estimand="ATT",
        estimate=es_result.estimate,
        se=es_result.se,
        pvalue=0.5,
        ci=(0, 1),
        alpha=0.05,
        n_obs=100,
        detail=es,
        model_info={"event_study": es, "vcv_pre": vcv},
    )
    out = sp.pretrends_test(r2)
    assert 0.0 <= out["pvalue"] <= 1.0


def test_pretrends_test_no_pre_raises():
    # event study restricted to non-negative relative time → no pre periods
    es = pd.DataFrame(
        {
            "relative_time": [0, 1, 2],
            "estimate": [0.0, 1.0, 2.0],
            "se": [0.0, 0.1, 0.2],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=1.0,
        se=0.1,
        pvalue=0.5,
        ci=(0, 1),
        alpha=0.05,
        n_obs=10,
        detail=es,
        model_info={"event_study": es},
    )
    with pytest.raises(ValueError):
        sp.pretrends_test(r)


def test_pretrends_test_all_zero_se_raises():
    es = pd.DataFrame(
        {
            "relative_time": [-2, -1, 1],
            "estimate": [0.0, 0.0, 1.0],
            "se": [0.0, 0.0, 0.1],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=1.0,
        se=0.1,
        pvalue=0.5,
        ci=(0, 1),
        alpha=0.05,
        n_obs=10,
        detail=es,
        model_info={"event_study": es},
    )
    with pytest.raises(ValueError):
        sp.pretrends_test(r)


# ----------------------------------------------------------------------
# pretrends_power
# ----------------------------------------------------------------------


def test_pretrends_power_default_delta(es_result):
    out = sp.pretrends_power(es_result)
    assert 0.0 <= out["power"] <= 1.0
    assert out["noncentrality"] >= 0.0
    assert out["df"] >= 1
    assert np.isfinite(out["critical_value"])
    assert len(out["delta"]) == out["df"]


def test_pretrends_power_custom_delta(es_result):
    out0 = sp.pretrends_power(es_result)
    k = out0["df"]
    # large delta → higher power than default
    out = sp.pretrends_power(es_result, delta=np.ones(k) * 5.0)
    assert 0.0 <= out["power"] <= 1.0
    assert out["power"] >= out0["power"] - 1e-9


def test_pretrends_power_full_length_delta(es_result):
    # delta of length K_all (incl. reference) gets aligned down to estimated K
    es = es_result.model_info["event_study"]
    pre = es[es["relative_time"] < 0]
    k_all = len(pre)
    out = sp.pretrends_power(es_result, delta=np.ones(k_all) * 2.0)
    assert 0.0 <= out["power"] <= 1.0


def test_pretrends_power_wrong_length_delta(es_result):
    with pytest.raises(MethodIncompatibility):
        sp.pretrends_power(
            es_result, delta=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        )


def test_pretrends_power_rejects_nonfinite_delta(es_result):
    with pytest.raises(MethodIncompatibility, match="delta"):
        sp.pretrends_power(es_result, delta=np.array([np.nan]))


def test_pretrends_power_rejects_invalid_alpha(es_result):
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.pretrends_power(es_result, alpha=0.0)


def test_pretrends_power_low_power_warning():
    # Tiny violation, large SEs → low power, triggers warning branch.
    es = pd.DataFrame(
        {
            "relative_time": [-3, -2, -1, 1, 2],
            "estimate": [0.0, 0.0, 0.0, 1.0, 1.0],
            "se": [10.0, 10.0, 0.0, 1.0, 1.0],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=1.0,
        se=1.0,
        pvalue=0.3,
        ci=(0, 2),
        alpha=0.05,
        n_obs=50,
        detail=es,
        model_info={"event_study": es},
    )
    out = sp.pretrends_power(r, delta=np.array([1e-6, 1e-6]))
    assert out["power"] < 0.5
    assert out["warning"] is not None
    assert "LOW POWER" in out["warning"]


# ----------------------------------------------------------------------
# sensitivity_rr
# ----------------------------------------------------------------------


def test_sensitivity_rr_default_grid(es_result):
    s = sp.sensitivity_rr(es_result)
    assert len(s.mbar_grid) == 20
    assert np.all(s.ci_upper >= s.ci_lower)
    assert s.method == "C-LF"
    assert s.att == pytest.approx(es_result.estimate)


def test_sensitivity_rr_custom_grid(es_result):
    s = sp.sensitivity_rr(es_result, Mbar=[0, 0.5, 1.0, 5.0])
    assert len(s.mbar_grid) == 4
    # Wider Mbar → wider (or equal) CI
    width = s.ci_upper - s.ci_lower
    assert np.all(np.diff(width) >= -1e-9)


def test_sensitivity_rr_breakdown_finite():
    # Construct a result whose ATT is small relative to SE so CI includes 0.
    es = pd.DataFrame(
        {
            "relative_time": [-3, -2, -1, 1, 2],
            "estimate": [0.0, 0.0, 0.0, 0.1, 0.2],
            "se": [0.05, 0.05, 0.0, 1.0, 1.0],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=0.1,
        se=1.0,
        pvalue=0.9,
        ci=(-1.9, 2.1),
        alpha=0.05,
        n_obs=50,
        detail=es,
        model_info={"event_study": es},
    )
    s = sp.sensitivity_rr(r, Mbar=[0.0, 0.1, 0.2])
    assert np.isfinite(s.breakdown_mbar)


def test_sensitivity_rr_bad_method(es_result):
    with pytest.raises(MethodIncompatibility):
        sp.sensitivity_rr(es_result, method="FLCI")


def test_sensitivity_rr_rejects_invalid_grid_controls(es_result):
    with pytest.raises(MethodIncompatibility, match="n_grid"):
        sp.sensitivity_rr(es_result, n_grid=0)
    with pytest.raises(MethodIncompatibility, match="Mbar"):
        sp.sensitivity_rr(es_result, Mbar=[])
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        sp.sensitivity_rr(es_result, Mbar=[0.0, -0.1])
    with pytest.raises(MethodIncompatibility, match="finite"):
        sp.sensitivity_rr(es_result, Mbar=[0.0, np.inf])


def test_sensitivity_rr_no_post_raises():
    es = pd.DataFrame(
        {
            "relative_time": [-3, -2, -1],
            "estimate": [0.0, 0.0, 0.0],
            "se": [0.1, 0.1, 0.0],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=0.0,
        se=0.1,
        pvalue=0.9,
        ci=(-0.2, 0.2),
        alpha=0.05,
        n_obs=30,
        detail=es,
        model_info={"event_study": es},
    )
    with pytest.raises(ValueError):
        sp.sensitivity_rr(r)


def test_sensitivity_rr_single_pre_period():
    # Only one pre period → slope computed from single estimate branch.
    es = pd.DataFrame(
        {
            "relative_time": [-1, 1, 2],
            "estimate": [0.0, 1.0, 2.0],
            "se": [0.0, 0.3, 0.3],
        }
    )
    # add an extra estimated pre period replaced by manipulating: use t=-2
    es = pd.DataFrame(
        {
            "relative_time": [-2, -1, 1, 2],
            "estimate": [0.5, 0.0, 1.0, 2.0],
            "se": [0.0, 0.0, 0.3, 0.3],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=1.0,
        se=0.3,
        pvalue=0.01,
        ci=(0.4, 1.6),
        alpha=0.05,
        n_obs=40,
        detail=es,
        model_info={"event_study": es},
    )
    # Only one nonzero-SE pre period? Actually both pre have se=0; but pre split
    # keeps all pre rows. The WLS path with >=2 pre rows is exercised.
    s = sp.sensitivity_rr(r, Mbar=[0.0, 1.0])
    assert len(s.mbar_grid) == 2


def test_sensitivity_result_summary_and_repr(es_result):
    s = sp.sensitivity_rr(es_result, Mbar=[0.0, 1.0, 2.0])
    txt = s.summary()
    assert "Rambachan" in txt
    assert "ATT" in txt
    assert repr(s) == txt
    html = s._repr_html_()
    assert "<table" in html


def test_sensitivity_result_summary_no_breakdown():
    # ATT large & SEs tiny → CI never includes zero → "No breakdown" branch.
    es = pd.DataFrame(
        {
            "relative_time": [-3, -2, -1, 1, 2],
            "estimate": [0.0, 0.0, 0.0, 10.0, 10.0],
            "se": [0.01, 0.01, 0.0, 0.01, 0.01],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=10.0,
        se=0.01,
        pvalue=0.0,
        ci=(9.9, 10.1),
        alpha=0.05,
        n_obs=50,
        detail=es,
        model_info={"event_study": es},
    )
    s = sp.sensitivity_rr(r, Mbar=[0.0, 0.001])
    assert not np.isfinite(s.breakdown_mbar)
    txt = s.summary()
    assert "No breakdown" in txt
    html = s._repr_html_()
    assert "None" in html


# ----------------------------------------------------------------------
# pretrends_summary
# ----------------------------------------------------------------------


def test_pretrends_summary(es_result, capsys):
    txt = sp.pretrends_summary(es_result)
    assert "Pre-Trends Analysis" in txt
    assert "Power" in txt


def test_pretrends_summary_low_power():
    es = pd.DataFrame(
        {
            "relative_time": [-3, -2, -1, 1, 2],
            "estimate": [0.001, 0.001, 0.0, 1.0, 1.0],
            "se": [10.0, 10.0, 0.0, 1.0, 1.0],
        }
    )
    r = CausalResult(
        method="es",
        estimand="ATT",
        estimate=1.0,
        se=1.0,
        pvalue=0.3,
        ci=(-1, 3),
        alpha=0.05,
        n_obs=50,
        detail=es,
        model_info={"event_study": es},
    )
    txt = sp.pretrends_summary(r, delta=np.array([1e-6, 1e-6]))
    assert "Power" in txt


# ----------------------------------------------------------------------
# helper-level error branches
# ----------------------------------------------------------------------


def test_extract_event_study_missing_raises():
    r = CausalResult(
        method="x",
        estimand="ATT",
        estimate=0.0,
        se=0.1,
        pvalue=0.5,
        ci=(0, 1),
        alpha=0.05,
        n_obs=10,
        detail=None,
        model_info={},
    )
    with pytest.raises(ValueError):
        ptmod._extract_event_study(r)


def test_extract_event_study_wrong_type_raises():
    r = CausalResult(
        method="x",
        estimand="ATT",
        estimate=0.0,
        se=0.1,
        pvalue=0.5,
        ci=(0, 1),
        alpha=0.05,
        n_obs=10,
        detail=None,
        model_info={"event_study": [1, 2, 3]},
    )
    with pytest.raises(MethodIncompatibility):
        ptmod._extract_event_study(r)


def test_resolve_columns_missing_time():
    df = pd.DataFrame({"estimate": [1.0], "se": [0.1]})
    with pytest.raises(MethodIncompatibility):
        ptmod._resolve_columns(df)


def test_resolve_columns_missing_estimate():
    df = pd.DataFrame({"relative_time": [-1], "se": [0.1]})
    with pytest.raises(MethodIncompatibility):
        ptmod._resolve_columns(df)


def test_resolve_columns_missing_se():
    df = pd.DataFrame({"relative_time": [-1], "estimate": [1.0]})
    with pytest.raises(MethodIncompatibility):
        ptmod._resolve_columns(df)
