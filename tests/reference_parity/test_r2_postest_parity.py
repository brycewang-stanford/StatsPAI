"""Cross-language parity: ``margins_at`` / ``contrast`` / ``pwcompare`` against Stata 18.

Fixtures
--------
* ``_fixtures/_generate_r2_postest_stata.do`` (Stata 18, official commands) on
  ``_fixtures/r2_postest_data.csv`` (written by ``_generate_r2_postest_data.py``,
  fixed seed, ``%.17g``) -> ``r2_postest_stata.json``: whole ``r(table)`` /
  ``r(table_vs)`` / ``r(at)`` matrices and ``e(b)`` at 17 significant digits.
* ``_generate_r2_postest_R.R`` -> ``r2_postest_R.json``: ``emmeans`` pairwise
  comparisons (none / holm / bonferroni / sidak) and
  ``marginaleffects::avg_predictions`` (default and ``df = df.residual``).

Stata command each StatsPAI call is held against
------------------------------------------------
=====================================================  =========================================
``sp.margins_at(r, d, at={"x": [...]})``               ``margins, at(x=(...))`` (``predict()``
                                                       = xb after ``regress``), incl. several
                                                       at-variables (Cartesian grid, first
                                                       variable slowest) and at() on a factor
``sp.contrast(r, d, "g", method="r"|"ar"|"gw")``       ``margins r.g`` / ``ar.g`` / ``gw.g``;
``reference=3``                                        ``margins rb3.g``
``sp.pwcompare(r, d, "g", adjust=...)``                ``margins g, pwcompare(effects)
                                                       mcompare(noadjust|bonferroni|sidak)``
``sp.margins(r, d, variables=["x"], at={...})``        ``margins, dydx(x) at(...)``
``sp.margins(..., method="mem")``                      ``margins, dydx(...) atmeans``
``sp.test(r, "C(g)[T.2] = C(g)[T.3] = C(g)[T.4] = 0")``  joint row of ``contrast r.g``
``sp.margins_at`` / ``sp.contrast`` after logit/probit  ``margins, at()`` / ``margins r.g`` (Pr)
=====================================================  =========================================

Conventions every number depends on
-----------------------------------
* ``margins`` (and ``sp.contrast`` / ``sp.pwcompare``) average predictions over
  the observed covariate distribution (*asobserved*). Stata's ``contrast`` and
  ``pwcompare`` *commands* instead balance over other factors (*asbalanced*) and
  evaluate the factor main-effect contrast at ``c.x = 0`` when the factor
  interacts with a continuous covariate. Those differ from ``margins`` exactly
  when ``g`` interacts with something; both mechanisms are asserted below as
  identities (class 3, documented convention, not a defect).
* Stata reports t with ``e(df_r)`` after ``regress`` (``G - 1`` under
  ``vce(cluster)``), z after ML commands; multiple-comparison adjustments are
  applied to the p-values and to the interval critical value.
* Delta-method SEs use the fit's full ``e(V)`` (OLS / HC1 / CR1).

Tolerances
----------
* Linear fits: point estimates and SEs are closed-form functionals of
  ``(b, V)``; observed agreement 1e-15 .. 7e-15, asserted at 1e-10.
* ``marginaleffects`` SEs come from a numerical Jacobian (default
  ``numderiv = "fdforward"``: 4.4e-8 gap; the generator uses ``"richardson"``:
  1.8e-10), asserted at 1e-9.
* ML fits at Stata's default convergence: ``logit`` coefficients themselves
  differ from StatsPAI by 1.1e-6 (Stata's default ``nrtolerance`` stops early;
  the same model refitted with tolerances 1e-14 matches StatsPAI to 4.6e-12 and
  its margins to 1.5e-12, asserted at 1e-9 in ``test_logit_gap_is_the_optimiser``).
  So the default-tolerance logit margins are asserted at 1e-6 (observed 2.4e-7
  est / 5.2e-8 SE), p-values on the log scale. Probit / poisson coefficient
  gaps are 1.8e-11 / 1.2e-8; their margins are asserted at 1e-7.

Former defects, fixed in 1.32 (the D-tests assert the Stata number)
-------------------------------------------------------------------
D1  ``margins_at`` / ``contrast`` / ``pwcompare`` use N(0, 1) for p-values and
    intervals whatever the fit (``stats.norm`` hard-coded in
    ``postestimation/margins.py``), while ``sp.margins`` in the same file and
    Stata use t(``inference_df(result)``). After ``vce(cluster)`` (df = 39)
    a Sidak-adjusted p is 7.5e-14 against Stata's 1.4e-8.
D2  Sidak-adjusted p computed as ``1 - (1 - p) ** m``: catastrophic
    cancellation returns exactly 0 for p < ~1e-17 (Stata: 7.0e-26).
D3  ``margins(method="mem")`` averages the raw factor column (mean of g =
    2.19) and then evaluates the dummies at that value (all zero), i.e. the
    base level, instead of at the dummy means as ``atmeans`` does.
D4  ``at=`` naming a variable that does not enter the model is silently
    ignored (Stata: error 322, "not found in list of covariates").
D5  Margins average over every row of ``data``; Stata averages over
    ``e(sample)``. With 25 missing outcomes the margin is off by 3.4e-3.

All five were strict xfails until 1.32, when margins / margins_at /
contrast / pwcompare moved onto one shared context (estimation sample,
fit weights, t(df) or z, atmeans as products of component means).
"""

from __future__ import annotations

import json
import pathlib
import re
import warnings
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
LIN = 1e-10  # closed-form functionals of an OLS fit
ML = 1e-6  # logit at Stata's default ML convergence (coefficient gap 1.1e-6)
ML_TIGHT = 1e-7  # probit / poisson (coefficient gaps 1.8e-11 / 1.2e-8)
# marginaleffects differentiates numerically (Richardson): SE gap 1.8e-10
R_NUMDERIV = 1e-9


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _stata() -> Dict[str, Any]:
    path = _FIX / "r2_postest_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _fixtures/_generate_r2_postest_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _r() -> Dict[str, Any]:
    path = _FIX / "r2_postest_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_r2_postest_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "r2_postest_data.csv")


def T(key: str) -> pd.DataFrame:
    """A Stata matrix from the fixture, rows x cols (missing -> NaN)."""
    m = _stata()[key]
    v = np.array([[np.nan if a is None else a for a in row] for row in m["v"]])
    return pd.DataFrame(v, index=m["rows"], columns=m["cols"])


def _fits():
    d = _data
    return {
        "ols": lambda: sp.regress("yl ~ C(g) + x + z", d()),
        "robust": lambda: sp.regress("yl ~ C(g) + x + z", d(), vce="robust"),
        "cluster": lambda: sp.regress("yl ~ C(g) + x + z", d(), vce="cluster cl"),
        "inter": lambda: sp.regress("yl ~ C(g)*x + z", d()),
        "ff": lambda: sp.regress("yl ~ C(g)*C(h) + x", d()),
        "miss": lambda: sp.regress("yl_m ~ C(g) + x + z", d()),
        "logit": lambda: sp.logit("yb ~ C(g) + x + z", d()),
        "probit": lambda: sp.probit("yp ~ C(g) + x + z", d()),
        "logit_nf": lambda: sp.logit("yb ~ x + z", d()),
        "probit_nf": lambda: sp.probit("yp ~ x + z", d()),
        "poisson": lambda: sp.poisson("yc ~ x + z", d(), vce="robust"),
    }


@lru_cache(maxsize=None)
def _fit(name: str) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fits()[name]()


def _close(got, want, rtol):
    np.testing.assert_allclose(
        np.asarray(got, dtype=float), np.asarray(want, dtype=float), rtol=rtol, atol=0
    )


def _pairs(cols):
    """Stata ``r2vs1.g`` / ``2vs1bn.g`` / ``ar3vs2.g`` -> ``"2 vs 1"``."""
    out = []
    for c in cols:
        m = re.match(r"^[a-z]*(\d+)vs(\d+)", c)
        out.append(f"{m.group(1)} vs {m.group(2)}")
    return out


LINEAR = ("ols", "robust", "cluster")


# ---------------------------------------------------------------------------
# margins_at: predictive margins at fixed covariate values
# ---------------------------------------------------------------------------
MARGINS_AT = [
    # (fixture key, fit, at)
    ("ols__at_x", "ols", {"x": [-1.0, 0.0, 1.5]}),
    ("robust__at_x", "robust", {"x": [-1.0, 0.0, 1.5]}),
    ("cluster__at_x", "cluster", {"x": [-1.0, 0.0, 1.5]}),
    ("ols__at_xz", "ols", {"x": [-1.0, 1.0], "z": [0.0, 2.0]}),
    ("ols__at_g", "ols", {"g": [1, 3]}),
    ("inter__at_x", "inter", {"x": [-1.0, 0.0, 1.5]}),
    ("inter__at_gx", "inter", {"g": [1, 3], "x": [0.0, 1.0]}),
    ("ff__at_h", "ff", {"h": [0, 1]}),
]


@pytest.mark.parametrize("key,fit,at", MARGINS_AT, ids=[k for k, _, _ in MARGINS_AT])
def test_margins_at_estimate_and_se(key, fit, at):
    ref = T(key)
    m = sp.margins_at(_fit(fit), _data(), at=at)
    _close(m["margin"], ref.loc["b"], LIN)
    _close(m["se"], ref.loc["se"], LIN)


def test_margins_at_grid_order_matches_stata():
    """Multiple at-variables: Cartesian grid with the first variable slowest."""
    at = T("ols__at_xz_at")
    m = sp.margins_at(_fit("ols"), _data(), at={"x": [-1.0, 1.0], "z": [0.0, 2.0]})
    _close(m["x"], at["x"], 0)
    _close(m["z"], at["z"], 0)
    gx = T("inter__at_gx_at")
    m = sp.margins_at(_fit("inter"), _data(), at={"g": [1, 3], "x": [0.0, 1.0]})
    stata_g = gx[["1b.g", "2.g", "3.g", "4.g"]].to_numpy().argmax(axis=1) + 1
    _close(m["g"], stata_g, 0)
    _close(m["x"], gx["x"], 0)


def test_margins_at_identity_linear_model():
    """Reference-free: margin(x=v) = b0 + sum_k b_gk * share_k + b_x v + b_z mean(z)."""
    d, r = _data(), _fit("ols")
    b = r.params
    share = {k: float((d["g"] == k).mean()) for k in (2, 3, 4)}
    m = sp.margins_at(r, d, at={"x": [0.5]})
    hand = (
        b["Intercept"]
        + sum(b[f"C(g)[T.{k}]"] * share[k] for k in share)
        + b["x"] * 0.5
        + b["z"] * d["z"].mean()
    )
    assert float(m["margin"].iloc[0]) == pytest.approx(hand, rel=1e-13)


# ---------------------------------------------------------------------------
# contrast: r. / rb#. / ar. / gw.
# ---------------------------------------------------------------------------
CONTRASTS = [
    (f"{blk}__{op}_g", blk, meth, None)
    for blk in LINEAR
    for op, meth in (("r", "r"), ("ar", "ar"))
] + [
    ("ols__gw_g", "ols", "gw", None),
    ("ols__rb3_g", "ols", "r", 3),
    ("inter__r_g", "inter", "r", None),
    ("inter__ar_g", "inter", "ar", None),
    ("inter__gw_g", "inter", "gw", None),
    ("ff__r_g", "ff", "r", None),
]


@pytest.mark.parametrize(
    "key,fit,method,reference", CONTRASTS, ids=[c[0] for c in CONTRASTS]
)
def test_contrast_estimate_and_se(key, fit, method, reference):
    ref = T(key)
    c = sp.contrast(_fit(fit), _data(), "g", method=method, reference=reference)
    if method != "gw":
        assert list(c["contrast_label"]) == _pairs(ref.columns)
    _close(c["contrast"], ref.loc["b"], LIN)
    _close(c["se"], ref.loc["se"], LIN)


def test_contrast_reference_identity():
    """Reference-free: without interactions r.g contrasts are the dummy coefficients."""
    r = _fit("ols")
    c = sp.contrast(r, _data(), "g", method="r")
    _close(c["contrast"], [r.params[f"C(g)[T.{k}]"] for k in (2, 3, 4)], 1e-12)


def test_contrast_command_is_asbalanced_at_zero():
    """Class 3 (documented convention): Stata's ``contrast`` command differs from
    ``margins r.g`` -- which sp.contrast reproduces -- by balancing over other
    factors and holding interacting continuous covariates at 0. Both mechanisms
    are reconstructed here from StatsPAI's own coefficients.
    """
    # g##c.x: contrast r.g = the g main-effect coefficients (x = 0).
    b = _fit("inter").params
    _close(
        T("inter__contrast_rg").loc["b"],
        [b[f"C(g)[T.{k}]"] for k in (2, 3, 4)],
        LIN,
    )
    # g##h: contrast r.g = differences of equally weighted (balanced) h cells.
    b = _fit("ff").params

    def cell(g, h):
        return (
            b["Intercept"]
            + b.get(f"C(g)[T.{g}]", 0.0)
            + h * b["C(h)[T.1]"]
            + h * b.get(f"C(g)[T.{g}]:C(h)[T.1]", 0.0)
        )

    balanced = [
        (cell(k, 0) + cell(k, 1)) / 2 - (cell(1, 0) + cell(1, 1)) / 2 for k in (2, 3, 4)
    ]
    _close(T("ff__contrast_rg").loc["b"], balanced, LIN)
    # ... and sp.contrast is the asobserved (margins) quantity, not this one.
    c = sp.contrast(_fit("ff"), _data(), "g", method="r")
    assert np.max(np.abs(c["contrast"].to_numpy() - np.asarray(balanced))) > 0.01
    # Without interactions the two coincide.
    _close(T("ols__contrast_rg").loc["b"], T("ols__r_g").loc["b"], LIN)


@pytest.mark.parametrize("blk", LINEAR)
def test_joint_contrast_test_is_sp_test(blk):
    """The Joint row of ``contrast r.g`` is the Wald test of all g dummies."""
    w = sp.test(_fit(blk), "C(g)[T.2] = C(g)[T.3] = C(g)[T.4] = 0")
    assert w["statistic"] == pytest.approx(T(f"{blk}__contrast_F").iloc[0, -1], rel=LIN)
    assert w["pvalue"] == pytest.approx(T(f"{blk}__contrast_p").iloc[0, -1], rel=1e-9)
    assert w["df"] == (
        int(T(f"{blk}__contrast_df").iloc[0, -1]),
        int(_stata()[f"{blk}__df_r"]),
    )


@pytest.mark.parametrize("blk,rtol", [("logit", ML), ("probit", ML_TIGHT)])
def test_joint_contrast_test_after_ml_is_chi2(blk, rtol):
    w = sp.test(_fit(blk), "C(g)[T.2] = C(g)[T.3] = C(g)[T.4] = 0")
    assert w["distribution"] == "chi2"
    assert w["statistic"] == pytest.approx(
        T(f"{blk}__contrast_chi2").iloc[0, -1], rel=rtol
    )
    assert np.log(w["pvalue"]) == pytest.approx(
        np.log(T(f"{blk}__contrast_p").iloc[0, -1]), rel=1e-5
    )


# ---------------------------------------------------------------------------
# pwcompare
# ---------------------------------------------------------------------------
PW = [
    (f"{blk}__pw_{adj}", blk, adj) for blk in LINEAR for adj in ("sidak", "bonferroni")
] + [
    ("ols__pw_none", "ols", "none"),
    ("inter__pw_bonferroni", "inter", "bonferroni"),
    ("ff__pw_sidak", "ff", "sidak"),
]


@pytest.mark.parametrize("key,fit,adjust", PW, ids=[p[0] for p in PW])
def test_pwcompare_estimate_and_se(key, fit, adjust):
    ref = T(key)
    p = sp.pwcompare(_fit(fit), _data(), "g", adjust=adjust)
    assert list(p["comparison"]) == _pairs(ref.columns)
    _close(p["diff"], ref.loc["b"], LIN)
    _close(p["se"], ref.loc["se"], LIN)


def test_pwcompare_command_equals_margins_pwcompare_without_interaction():
    _close(T("ols__pwcompare_sidak").loc["b"], T("ols__pw_sidak").loc["b"], LIN)
    _close(
        T("ols__pwcompare_sidak").loc["pvalue"], T("ols__pw_sidak").loc["pvalue"], 1e-9
    )


def test_pwcompare_matches_emmeans_estimates():
    """emmeans ``pairs(reverse = TRUE)`` labels ``gj - gk``; same differences / SEs."""
    ref = _r()["emmeans_pairs"]["none"]
    p = sp.pwcompare(_fit("ols"), _data(), "g").set_index("comparison")
    for lab, est, se in zip(ref["contrast"], ref["estimate"], ref["se"]):
        j, k = re.match(r"g(\d+) - g(\d+)", lab).groups()
        assert p.loc[f"{j} vs {k}", "diff"] == pytest.approx(est, rel=LIN)
        assert p.loc[f"{j} vs {k}", "se"] == pytest.approx(se, rel=LIN)


def test_emmeans_holm_interval_is_bonferroni():
    """R convention sp.pwcompare(adjust='holm') follows for its intervals."""
    em = _r()["emmeans_pairs"]
    _close(em["holm"]["lower"], em["bonferroni"]["lower"], 1e-14)


# ---------------------------------------------------------------------------
# margins, dydx() at() / atmeans
# ---------------------------------------------------------------------------
DYDX_AT = [
    # (fixture key, fit, variable, [at dicts in column order], rtol)
    ("ols__dydx_x_at_z", "ols", "x", [{"z": 0}, {"z": 1}], LIN),
    ("inter__dydx_x_at_g", "inter", "x", [{"g": k} for k in (1, 2, 3, 4)], LIN),
    ("logit__dydx_x_at_z", "logit", "x", [{"z": 0}, {"z": 1}], ML),
    ("logit__dydx_x_at_g", "logit", "x", [{"g": 1}, {"g": 3}], ML),
    ("probit__dydx_x_at_z", "probit", "x", [{"z": 0}, {"z": 1}], ML_TIGHT),
    ("probit__dydx_x_at_g", "probit", "x", [{"g": 1}, {"g": 3}], ML_TIGHT),
    ("poisson__dydx_x_at_z", "poisson", "x", [{"z": 0}, {"z": 1}], ML_TIGHT),
]


@pytest.mark.parametrize("key,fit,var,ats,rtol", DYDX_AT, ids=[k[0] for k in DYDX_AT])
def test_margins_dydx_at(key, fit, var, ats, rtol):
    ref = T(key)
    for j, at in enumerate(ats):
        m = sp.margins(_fit(fit), _data(), variables=[var], at=at)
        assert float(m["dy/dx"].iloc[0]) == pytest.approx(
            ref.loc["b"].iloc[j], rel=rtol
        )
        assert float(m["se"].iloc[0]) == pytest.approx(ref.loc["se"].iloc[j], rel=rtol)
        assert np.log(float(m["pvalue"].iloc[0])) == pytest.approx(
            np.log(ref.loc["pvalue"].iloc[j]), rel=1e-4
        )


def test_margins_dydx_average_with_interaction():
    ref = T("inter__dydx_x")
    m = sp.margins(_fit("inter"), _data(), variables=["x"])
    assert float(m["dy/dx"].iloc[0]) == pytest.approx(ref.loc["b"].iloc[0], rel=LIN)
    assert float(m["se"].iloc[0]) == pytest.approx(ref.loc["se"].iloc[0], rel=LIN)
    assert float(m["pvalue"].iloc[0]) == pytest.approx(
        ref.loc["pvalue"].iloc[0], rel=1e-9
    )
    # identity: AME of x = b_x + sum_k b_{gk:x} share_k
    b, d = _fit("inter").params, _data()
    hand = b["x"] + sum(b[f"C(g)[T.{k}]:x"] * (d["g"] == k).mean() for k in (2, 3, 4))
    assert float(m["dy/dx"].iloc[0]) == pytest.approx(hand, rel=1e-13)


@pytest.mark.parametrize(
    "key,fit,rtol",
    [
        ("logit_nf__dydx_atmeans", "logit_nf", ML),
        ("probit_nf__dydx_atmeans", "probit_nf", ML_TIGHT),
        ("poisson__dydx_atmeans", "poisson", ML_TIGHT),
    ],
)
def test_margins_atmeans_continuous_covariates(key, fit, rtol):
    ref = T(key)
    m = sp.margins(_fit(fit), _data(), method="mem")
    assert list(m["variable"]) == list(ref.columns)
    _close(m["dy/dx"], ref.loc["b"], rtol)
    _close(m["se"], ref.loc["se"], rtol)


def test_logit_gap_is_the_optimiser():
    """Stata's default ML convergence, not StatsPAI, sets the logit gap."""
    cols = ["1b.g", "2.g", "3.g", "4.g", "x", "z", "_cons"]
    order = ["_cons", "2.g", "3.g", "4.g", "x", "z"]

    def stata_b(key):
        row = T(key).iloc[0]
        row.index = [c.split(":")[-1] for c in row.index]
        assert list(row.index) == cols
        return row[order].to_numpy()

    b = _fit("logit").params.to_numpy()
    gap_default = np.max(np.abs(b - stata_b("logit__b")) / np.abs(stata_b("logit__b")))
    gap_tight = np.max(
        np.abs(b - stata_b("logit_tight__b")) / np.abs(stata_b("logit_tight__b"))
    )
    assert 1e-7 < gap_default < 2e-6
    assert gap_tight < 1e-10
    ref = T("logit_tight__dydx_x_at_z")
    m = sp.margins(_fit("logit"), _data(), variables=["x"], at={"z": 0})
    assert float(m["dy/dx"].iloc[0]) == pytest.approx(ref.loc["b"].iloc[0], rel=1e-9)
    assert float(m["se"].iloc[0]) == pytest.approx(ref.loc["se"].iloc[0], rel=1e-9)


# ---------------------------------------------------------------------------
# index models: margins_at / contrast / pwcompare on the response scale
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fit,rtol", [("logit", ML), ("probit", ML_TIGHT)])
def test_level_functions_on_response_scale_after_index_models(fit, rtol):
    """Stata ``margins, at(x=(0 1))`` and ``margins r.g`` after logit / probit
    report predict(pr) margins (fixture keys ``*_at_x_pr``, ``*_r_g_pr``).
    Until 1.32 StatsPAI refused these rather than mislabel index predictions;
    they now predict on Pr(y=1) with z inference, as Stata does."""
    r, d = _fit(fit), _data()
    m = sp.margins_at(r, d, at={"x": [0.0, 1.0]})
    ref = T(f"{fit}__at_x_pr")
    _close(m["margin"], ref.loc["b"], rtol)
    _close(m["se"], ref.loc["se"], rtol)
    _close(m["ci_lower"], ref.loc["ll"], rtol)
    c = sp.contrast(r, d, "g", method="r")
    ref = T(f"{fit}__r_g_pr")
    assert list(c["contrast_label"]) == _pairs(ref.columns)
    _close(c["contrast"], ref.loc["b"], rtol)
    _close(c["se"], ref.loc["se"], rtol)
    assert np.log(c["pvalue"].to_numpy()) == pytest.approx(
        np.log(ref.loc["pvalue"].to_numpy()), rel=1e-4
    )
    # pwcompare "k vs 1" rows are the same margins-scale differences
    p = sp.pwcompare(r, d, "g").set_index("comparison")
    for lab, diff in zip(c["contrast_label"], c["contrast"]):
        assert p.loc[lab, "diff"] == pytest.approx(diff, rel=1e-12)


# ---------------------------------------------------------------------------
# R: marginaleffects conventions (estimates independent of the reference law)
# ---------------------------------------------------------------------------
def test_margins_at_matches_marginaleffects_avg_predictions():
    ref = _r()["avg_predictions_default"]
    m = sp.margins_at(_fit("ols"), _data(), at={"x": ref["x"]})
    _close(m["margin"], ref["estimate"], LIN)
    _close(m["se"], ref["se"], R_NUMDERIV)


def test_marginaleffects_default_is_normal_and_df_is_stata():
    """marginaleffects defaults to N(0, 1) even after lm(); with df = df.residual
    its interval is Stata's. Asserted on the reference numbers themselves."""
    r = _r()
    z = stats.norm.ppf(0.975)
    t = stats.t.ppf(0.975, r["df_residual"])
    a0, a1 = r["avg_predictions_default"], r["avg_predictions_df"]
    _close(np.subtract(a0["estimate"], z * np.asarray(a0["se"])), a0["lower"], 1e-12)
    _close(np.subtract(a1["estimate"], t * np.asarray(a1["se"])), a1["lower"], 1e-12)
    _close(a1["lower"], T("ols__at_x").loc["ll"], R_NUMDERIV)


# ---------------------------------------------------------------------------
# Defects: xfail(strict=True), each asserting the Stata number
# ---------------------------------------------------------------------------
_D1 = (
    "D1: margins_at / contrast / pwcompare hard-code stats.norm for p-values and "
    "intervals (postestimation/margins.py: z_crit = stats.norm.ppf in margins_at "
    "and contrast; pv = 2*stats.norm.sf in contrast and pwcompare; "
    "z_crit = stats.norm.ppf(1 - alpha_adj/2) in pwcompare). Stata and sp.margins "
    "in the same file use t(inference_df(result)): df_r = 594 here, G-1 = 39 "
    "under vce(cluster). Fix: df = inference_df(result); use stats.t when finite, "
    "as sp.margins does."
)


@pytest.mark.parametrize("blk", LINEAR)
def test_D1_margins_at_uses_t_df_r(blk):
    ref = T(f"{blk}__at_x")
    m = sp.margins_at(_fit(blk), _data(), at={"x": [-1.0, 0.0, 1.5]})
    _close(m["ci_lower"], ref.loc["ll"], 1e-9)
    _close(m["ci_upper"], ref.loc["ul"], 1e-9)


@pytest.mark.parametrize("blk", LINEAR)
def test_D1_contrast_uses_t_df_r(blk):
    ref = T(f"{blk}__r_g")
    c = sp.contrast(_fit(blk), _data(), "g", method="r")
    _close(c["pvalue"], ref.loc["pvalue"], 1e-8)
    _close(c["ci_lower"], ref.loc["ll"], 1e-9)


@pytest.mark.parametrize("blk", LINEAR)
@pytest.mark.parametrize("adjust", ["sidak", "bonferroni"])
def test_D1_pwcompare_adjusted_inference_uses_t_df_r(blk, adjust):
    ref = T(f"{blk}__pw_{adjust}")
    p = sp.pwcompare(_fit(blk), _data(), "g", adjust=adjust)
    _close(p["pvalue_adj"], ref.loc["pvalue"], 1e-8)
    _close(p["ci_lower"], ref.loc["ll"], 1e-9)
    _close(p["ci_upper"], ref.loc["ul"], 1e-9)


def test_D1_pwcompare_holm_matches_emmeans():
    ref = _r()["emmeans_pairs"]["holm"]
    p = sp.pwcompare(_fit("ols"), _data(), "g", adjust="holm").set_index("comparison")
    for lab, pv in zip(ref["contrast"], ref["p"]):
        j, k = re.match(r"g(\d+) - g(\d+)", lab).groups()
        assert p.loc[f"{j} vs {k}", "pvalue_adj"] == pytest.approx(pv, rel=1e-8)


def test_D2_sidak_adjustment_keeps_small_p():
    p = sp.pwcompare(_fit("ols"), _data(), "g", adjust="sidak")
    m = len(p)
    exact = -np.expm1(m * np.log1p(-p["pvalue"].to_numpy()))
    assert (p["pvalue_adj"] > 0).all()
    _close(p["pvalue_adj"], exact, 1e-12)


_D3 = (
    "D3: margins(method='mem') builds the mean row with "
    "frame.mean(numeric_only=True) before the design (postestimation/margins.py, "
    "margins(): `frame = frame.mean(...).to_frame().T`), so a factor column g is "
    "averaged to 2.19 and every C(g)[T.k] dummy evaluates to 0 (the base level). "
    "Stata atmeans sets each dummy to its sample share (r(at): .323 .303 .235 "
    ".138). Fix: average the *parts* (numeric columns and factor indicators "
    "_part_values) and form design columns as products of those means."
)


@pytest.mark.parametrize("fit,rtol", [("logit", ML), ("probit", ML_TIGHT)])
def test_D3_atmeans_with_factor(fit, rtol):
    ref = T(f"{fit}__dydx_atmeans")
    # Stata: margins, dydx(x z) atmeans  (the factor g is held at its shares)
    m = sp.margins(_fit(fit), _data(), variables=["x", "z"], method="mem")
    _close(m["dy/dx"], ref.loc["b"], rtol)
    _close(m["se"], ref.loc["se"], rtol)


@pytest.mark.parametrize("fit,rtol", [("logit", ML), ("probit", ML_TIGHT)])
def test_D3_at_with_atmeans_with_factor(fit, rtol):
    ref = T(f"{fit}__dydx_x_at_z1_atmeans")
    m = sp.margins(_fit(fit), _data(), variables=["x"], at={"z": 1}, method="mem")
    assert float(m["dy/dx"].iloc[0]) == pytest.approx(ref.loc["b"].iloc[0], rel=rtol)


def test_D4_at_variable_not_in_model_is_an_error():
    assert _stata()["ols__at_notinmodel_rc"] == 322
    r, d = _fit("ols"), _data()
    for call in (
        lambda: sp.margins_at(r, d, at={"cl": [0, 1]}),
        lambda: sp.margins(r, d, variables=["x"], at={"cl": 0}),
    ):
        try:
            call()
        except MethodIncompatibility:
            continue
        raise AssertionError("at() variable outside the model was accepted")


def test_margins_on_estimation_sample_matches_stata():
    """With data restricted to e(sample) margins_at / contrast match Stata."""
    d = _data().dropna(subset=["yl_m"])
    r = _fit("miss")
    assert r.data_info["nobs"] == int(_stata()["miss__N"]) == len(d)
    m = sp.margins_at(r, d, at={"x": [0.0, 1.0]})
    _close(m["margin"], T("miss__at_x").loc["b"], LIN)
    _close(m["se"], T("miss__at_x").loc["se"], LIN)
    c = sp.contrast(r, d, "g", method="gw")
    _close(c["contrast"], T("miss__gw_g").loc["b"], LIN)


def test_D5_margins_use_estimation_sample():
    r, d = _fit("miss"), _data()
    m = sp.margins_at(r, d, at={"x": [0.0, 1.0]})
    _close(m["margin"], T("miss__at_x").loc["b"], 1e-9)
    c = sp.contrast(r, d, "g", method="gw")
    _close(c["contrast"], T("miss__gw_g").loc["b"], 1e-9)
