"""sp.margins / sp.margins_at after limited-dependent-variable models, and
sp.heckprobit, against Stata 18 MP.

Before, ``sp.margins`` refused mlogit / ologit / ivprobit, failed on tobit
(it looked for a column called ``sigma``) and on heckman. Now each model's
Stata ``predict`` statistics are available through ``predict=``, with
Stata's defaults.

The reference is ``tests/fixtures/ldv_margins/stata_margins.do`` run on the
600 simulated rows of ``_data.py`` (``import delimited ..., asdouble``,
``set type double``). Point estimates agree to ~1e-10. Stata's ``margins``
differentiates numerically and its standard errors are accurate to a few
1e-7 for some statistics (tobit ``pr()``, ivprobit ``pr``, heckprobit
``p11``); the analytic checks below show ours are within 1e-8 of the
complex-step derivative there, so the Stata comparison uses rtol 2e-6.
"""

import importlib.util
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy import special

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "ldv_margins"
_spec = importlib.util.spec_from_file_location("_ldv_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_margins.json").read_text(encoding="utf-8"))
RTOL = 2e-6


@pytest.fixture(scope="module")
def df():
    return _mod.make_ldv_frame()


@pytest.fixture(scope="module")
def fits(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "mlogit": sp.mlogit("ym ~ x1 + x2 + C(g)", data=df),
            "ologit": sp.ologit("yo ~ x1 + x2 + C(g)", data=df),
            "oprobit": sp.oprobit("yo ~ x1 + x2 + C(g)", data=df),
            "tobit": sp.tobit(df, formula="yt ~ x1 + x2 + C(g)", ll=0),
            "tobit_robust": sp.tobit(
                df, formula="yt ~ x1 + x2 + C(g)", ll=0, vce="robust"
            ),
            "tobit_pw": sp.tobit(df, formula="yt ~ x1 + x2 + C(g)", ll=0, weights="w"),
            "heckman": sp.heckman(
                df,
                y="yh",
                x=["x1", "x2", "g2", "g3"],
                select="s",
                z=["z", "x1"],
                method="ml",
            ),
            "ivprobit": sp.ivprobit(
                df, y="yb", x=["x1", "g2", "g3"], endog="xe", instruments=["z"]
            ),
            "heckprobit": sp.heckprobit(
                df, y="ybh", x=["x1", "x2", "g2", "g3"], select="s", z=["z", "x1"]
            ),
        }


def _stata_key(name, categories):
    """``x1:2._predict`` / ``2.g`` / ``x1`` -> (variable, outcome)."""
    parts = name.split(":")
    out = None
    if len(parts) > 1:
        out = categories[int(re.match(r"(\d+)", parts[1]).group(1)) - 1]
    return parts[0], out


def _compare_dydx(table, tag, categories=None):
    ref = {
        _stata_key(n, categories): v
        for n, v in STATA[tag].items()
        if v[1] > 0  # base levels and omitted variables
    }
    ours = {
        (r["variable"], r["outcome"] if "outcome" in table else None): (
            r["dy/dx"],
            r["se"],
        )
        for _, r in table.iterrows()
    }
    assert set(ref) <= set(ours), sorted(set(ref) - set(ours))
    got = np.array([ours[k] for k in ref])
    want = np.array(list(ref.values()))
    np.testing.assert_allclose(got[:, 0], want[:, 0], rtol=RTOL, atol=1e-9)
    np.testing.assert_allclose(got[:, 1], want[:, 1], rtol=RTOL)
    # every row Stata does not report (z under predict(xb), the base
    # level) is an exact zero
    for k, (b, _) in ours.items():
        if k not in ref:
            assert abs(b) < 1e-12, k


DYDX_CASES = [
    ("mlogit", "mlogit_ame", {}, [0, 1, 2]),
    ("mlogit", "mlogit_mem", {"method": "mem"}, [0, 1, 2]),
    ("ologit", "ologit_ame", {}, [0, 1, 2, 3]),
    ("oprobit", "oprobit_ame", {}, [0, 1, 2, 3]),
    ("tobit", "tobit_xb", {}, None),
    ("tobit", "tobit_e0", {"predict": "e(0,.)"}, None),
    ("tobit", "tobit_ystar0", {"predict": "ystar(0,.)"}, None),
    ("tobit", "tobit_pr0", {"predict": "pr(0,.)"}, None),
    ("tobit_robust", "tobit_e0_robust", {"predict": "e"}, None),
    ("tobit_pw", "tobit_pw_e0", {"predict": "e(0,.)"}, None),
    ("heckman", "heckman_xb", {}, None),
    ("heckman", "heckman_ycond", {"predict": "ycond"}, None),
    ("heckman", "heckman_yexpected", {"predict": "yexpected"}, None),
    ("heckman", "heckman_psel", {"predict": "psel"}, None),
    ("ivprobit", "ivprobit_ame", {}, None),
    ("ivprobit", "ivprobit_xb", {"predict": "xb"}, None),
    ("ivprobit", "ivprobit_pr", {"predict": "pr"}, None),
    ("ivprobit", "ivprobit_fixedasf", {"predict": "pr fixedasf"}, None),
    ("heckprobit", "heckprobit_pmargin", {}, None),
    ("heckprobit", "heckprobit_pcond", {"predict": "pcond"}, None),
    ("heckprobit", "heckprobit_p11", {"predict": "p11"}, None),
    ("heckprobit", "heckprobit_p10", {"predict": "p10"}, None),
    ("heckprobit", "heckprobit_p01", {"predict": "p01"}, None),
    ("heckprobit", "heckprobit_p00", {"predict": "p00"}, None),
    ("heckprobit", "heckprobit_psel", {"predict": "psel"}, None),
]


@pytest.mark.parametrize("fit,tag,kw,cats", DYDX_CASES, ids=[c[1] for c in DYDX_CASES])
def test_dydx_matches_stata_margins(df, fits, fit, tag, kw, cats):
    table = sp.margins(fits[fit], data=df, **kw)
    _compare_dydx(table, tag, cats)


def test_default_order_follows_the_formula(df, fits):
    # patsy puts C(g) first; Stata's dydx(*) follows the command
    table = sp.margins(fits["ologit"], data=df)
    assert list(dict.fromkeys(table["variable"])) == ["x1", "x2", "2.g", "3.g"]


def _rows(tag):
    return np.array([v for v in STATA[tag].values() if v[1] > 0])


def test_predictive_margins_ologit_at(df, fits):
    m = sp.margins_at(fits["ologit"], data=df, at={"x1": [-1, 0, 1]})
    m = m.sort_values(["outcome", "x1"])
    want = _rows("ologit_at")  # Stata: outcome-major, then at()
    np.testing.assert_allclose(m["margin"], want[:, 0], rtol=RTOL)
    np.testing.assert_allclose(m["se"], want[:, 1], rtol=RTOL)
    # the probabilities of the four outcomes add up to one at each point
    np.testing.assert_allclose(m.groupby("x1")["margin"].sum(), 1.0, atol=1e-12)


def test_predictive_margins_mlogit_by_factor_level(df, fits):
    m = sp.margins_at(fits["mlogit"], data=df, at={"g": [1, 2, 3]})
    m = m.sort_values(["outcome", "g"])
    want = _rows("mlogit_g")
    np.testing.assert_allclose(m["margin"], want[:, 0], rtol=RTOL)
    np.testing.assert_allclose(m["se"], want[:, 1], rtol=RTOL)


@pytest.mark.parametrize(
    "fit,tag,at,predict",
    [
        ("tobit", "tobit_at_e0", {"x1": [0, 1]}, "e(0,.)"),
        ("heckman", "heckman_at_ycond", {"x1": [0, 1]}, "ycond"),
        # the ASF holds the reduced-form error at its observed value while
        # margins sets the endogenous regressor (or an exogenous one)
        ("ivprobit", "ivprobit_at_pr", {"xe": [0, 1]}, "pr"),
        ("ivprobit", "ivprobit_at_x1_pr", {"x1": [0, 1]}, "pr"),
    ],
)
def test_predictive_margins_at(df, fits, fit, tag, at, predict):
    m = sp.margins_at(fits[fit], data=df, at=at, predict=predict)
    want = _rows(tag)
    np.testing.assert_allclose(m["margin"], want[:, 0], rtol=RTOL)
    np.testing.assert_allclose(m["se"], want[:, 1], rtol=RTOL)


def test_at_contrast_matches_stata_atcontrast(df, fits):
    m = sp.margins_at(
        fits["tobit"], data=df, at={"x1": [0, 1]}, predict="e(0,.)", contrast="first"
    )
    ((b, se),) = STATA["tobit_at_e0_contrast"].values()
    np.testing.assert_allclose(m["contrast"], [b], rtol=RTOL)
    np.testing.assert_allclose(m["se"], [se], rtol=RTOL)


# --- analytic checks where Stata's numerical derivatives are the weak side --


def _complex_step_se(f, theta, V):
    J = np.array(
        [
            np.imag(f(theta + 1j * 1e-30 * np.eye(theta.size)[j])) / 1e-30
            for j in range(theta.size)
        ]
    )
    return float(np.real(f(theta))), float(np.sqrt(J @ V @ J))


def _npdf(a):
    return np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)


def test_tobit_pr_ame_equals_closed_form(df, fits):
    st = fits["tobit"]._ml_state
    terms = st["terms"]
    assert terms == ["const", "C(g)[T.2]", "C(g)[T.3]", "x1", "x2"]
    X = np.column_stack([np.ones(len(df)), df.g2, df.g3, df.x1, df.x2])

    def ame(t):  # d Pr(y > 0) / d x1 = phi(xb / s) b1 / s
        s = np.exp(t[-1])
        return np.mean(_npdf((X @ t[:-1]) / s)) * t[3] / s

    est, se = _complex_step_se(ame, st["theta"], st["vcov"])
    row = sp.margins(fits["tobit"], data=df, predict="pr(0,.)", variables=["x1"])
    np.testing.assert_allclose(row["dy/dx"], [est], rtol=1e-10)
    np.testing.assert_allclose(row["se"], [se], rtol=1e-8)


def test_ivprobit_asf_ame_equals_closed_form(df, fits):
    r = fits["ivprobit"]
    th = r.params.to_numpy()
    V = r.data_info["var_cov"]
    X = np.column_stack([df.xe, df.x1, df.g2, df.g3, np.ones(len(df))])
    R = np.column_stack([df.x1, df.g2, df.g3, df.z, np.ones(len(df))])

    def ame(t):  # Pr(y=1 | x, v) = Phi((xb + rho v / sigma) / sqrt(1 - rho^2))
        rho, s = np.tanh(t[10]), np.exp(t[11])
        v = df.xe.to_numpy() - R @ t[5:10]
        c = np.sqrt(1 - rho**2)
        return np.mean(_npdf((X @ t[:5] + rho / s * v) / c)) * t[0] / c

    est, se = _complex_step_se(ame, th, V)
    row = sp.margins(r, data=df, predict="pr", variables=["xe"])
    np.testing.assert_allclose(row["dy/dx"], [est], rtol=1e-10)
    np.testing.assert_allclose(row["se"], [se], rtol=1e-8)


def test_heckprobit_p11_ame_equals_closed_form(df, fits):
    st = fits["heckprobit"]._ml_state
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2, df.g2, df.g3])
    Z = np.column_stack([np.ones(len(df)), df.z, df.x1])

    def ame(t):  # d Phi2(xb, zg; rho) / d x2 = phi(xb) Phi((zg - rho xb)/c) b2
        rho = np.tanh(t[8])
        xb, zg = X @ t[:5], Z @ t[5:8]
        cdf = special.ndtr((zg - rho * xb) / np.sqrt(1 - rho * rho))
        return np.mean(_npdf(xb) * cdf) * t[2]

    est, se = _complex_step_se(ame, st["theta"], st["vcov"])
    row = sp.margins(fits["heckprobit"], data=df, predict="p11", variables=["x2"])
    np.testing.assert_allclose(row["dy/dx"], [est], rtol=1e-10)
    np.testing.assert_allclose(row["se"], [se], rtol=1e-8)


def test_ordered_probabilities_sum_to_one_so_effects_sum_to_zero(df, fits):
    table = sp.margins(fits["oprobit"], data=df)
    sums = table.groupby("variable")["dy/dx"].sum()
    np.testing.assert_allclose(sums.to_numpy(), 0.0, atol=1e-10)


# --- heckprobit itself ------------------------------------------------------

_HP_ORDER = [1, 2, 3, 4, 0, 6, 7, 5, 8]  # ours: const first; Stata: _cons last


@pytest.mark.parametrize(
    "vce,tag", [(None, "heckprobit_e"), ("robust", "heckprobit_rob_e")]
)
def test_heckprobit_matches_stata(df, vce, tag):
    r = sp.heckprobit(
        df, y="ybh", x=["x1", "x2", "g2", "g3"], select="s", z=["z", "x1"], vce=vce
    )
    want = np.array(list(STATA[tag].values()))
    assert list(STATA[tag])[-1] == "/athrho"
    np.testing.assert_allclose(
        r.params.to_numpy()[:9][_HP_ORDER], want[:, 0], rtol=1e-8
    )
    np.testing.assert_allclose(
        r.std_errors.to_numpy()[:9][_HP_ORDER], want[:, 1], rtol=1e-8
    )
    assert r.model_info["n_selected"] == 351
    assert np.isclose(r.model_info["rho"], np.tanh(want[-1, 0]), rtol=1e-8)


def test_heckprobit_select_defaults_to_observed_outcome(df):
    a = sp.heckprobit(df, y="ybh", x=["x1", "x2"], select="s", z=["z", "x1"])
    b = sp.heckprobit(df, y="ybh", x=["x1", "x2"], z=["z", "x1"])
    np.testing.assert_allclose(a.params, b.params, rtol=1e-10)


def test_heckprobit_rejects_continuous_outcome(df):
    with pytest.raises(MethodIncompatibility, match="must be 0/1"):
        sp.heckprobit(df, y="yh", x=["x1"], select="s", z=["z"])


def test_heckprobit_needs_z(df):
    with pytest.raises(MethodIncompatibility, match="z="):
        sp.heckprobit(df, y="ybh", x=["x1"], select="s", z=[])


def test_heckprobit_lr_test_is_twice_the_loglik_gain(df):
    r = sp.heckprobit(df, y="ybh", x=["x1", "x2"], select="s", z=["z", "x1"])
    sel = df.s == 1
    a = sp.probit("s ~ z + x1", data=df).diagnostics["Log-Likelihood"]
    b = sp.probit("ybh ~ x1 + x2", data=df[sel]).diagnostics["Log-Likelihood"]
    np.testing.assert_allclose(
        r.model_info["lr_rho0"], 2 * (r.model_info["log_likelihood"] - a - b), rtol=1e-7
    )


# --- refusals ---------------------------------------------------------------


def test_unknown_predict_is_refused(df, fits):
    with pytest.raises(MethodIncompatibility, match="ycond"):
        sp.margins(fits["tobit"], data=df, predict="ycond")


def test_category_restricts_outcomes(df, fits):
    t = sp.margins(fits["mlogit"], data=df, category=2)
    assert set(t["outcome"]) == {2}
    with pytest.raises(MethodIncompatibility, match="category=7"):
        sp.margins(fits["mlogit"], data=df, category=7)


def test_predict_on_a_single_index_model_is_refused(df):
    r = sp.regress("ys ~ x1 + x2", data=df)
    with pytest.raises(MethodIncompatibility, match="predict="):
        sp.margins(r, data=df, predict="pr")


def test_asf_at_means_is_refused(df, fits):
    with pytest.raises(MethodIncompatibility, match="observed reduced-form"):
        sp.margins(fits["ivprobit"], data=df, predict="pr", method="mem")


def test_ivprobit_twostep_is_refused(df):
    r = sp.ivprobit(
        df, y="yb", x=["x1"], endog="xe", instruments=["z"], method="twostep"
    )
    with pytest.raises(MethodIncompatibility, match="twostep"):
        sp.margins(r, data=df)


@pytest.mark.parametrize("fn", ["avg_slopes", "avg_predictions"])
def test_marginaleffects_functions_point_to_margins(df, fits, fn):
    with pytest.raises(MethodIncompatibility, match="sp.margins"):
        getattr(sp, fn)(fits["tobit"], newdata=df)


def test_contrast_points_to_margins_at(df, fits):
    with pytest.raises(MethodIncompatibility, match="margins_at"):
        sp.contrast(fits["ologit"], data=df, variable="g")


def test_vcov_false_drops_inference_after_ldv_models():
    # 2026-10-20 probe: vcov=False was not passed to the LDV branch, so the
    # se / z / p / CI columns came back after intreg / ologit / heckprobit.
    import numpy as np
    import pandas as pd

    import statspai as sp

    rng = np.random.default_rng(0)
    n = 1500
    e, x = rng.normal(size=(2, n))
    w = 2 + 0.5 * e + 0.3 * x + rng.normal(size=n)
    df = pd.DataFrame({"e": e, "x": x, "lo": np.floor(w), "hi": np.floor(w) + 1})
    r = sp.intreg(df, y_lower="lo", y_upper="hi", x=["e", "x"])
    full = sp.margins(r, data=df)
    bare = sp.margins(r, data=df, vcov=False)
    assert bare.columns.tolist() == ["variable", "dy/dx"]
    np.testing.assert_array_equal(bare["dy/dx"].to_numpy(), full["dy/dx"].to_numpy())
    at = sp.margins_at(r, data=df, at={"e": [0, 1]}, vcov=False)
    assert not {"se", "z", "pvalue", "ci_lower", "ci_upper"} & set(at.columns)


def test_heckprobit_refuses_an_outcome_that_does_not_vary():
    # 2026-10-20 probe: y = 1 on every selected row gave converged=True, a
    # 17.4 coefficient with SE 3e30; Stata stops ("outcome does not vary").
    import numpy as np
    import pandas as pd
    import pytest

    import statspai as sp
    from statspai.exceptions import DataInsufficient

    rng = np.random.default_rng(0)
    n = 500
    z, x = rng.normal(size=(2, n))
    s = (z + rng.normal(size=n) > 0).astype(int)
    df = pd.DataFrame({"x": x, "z": z, "s": s, "y": np.where(s == 1, 1.0, np.nan)})
    with pytest.raises(DataInsufficient, match="must vary"):
        sp.heckprobit(df, y="y", x=["x"], select="s", z=["z", "x"])
