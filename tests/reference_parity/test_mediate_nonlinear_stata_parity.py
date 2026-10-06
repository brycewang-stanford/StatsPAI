"""``sp.mediate(inference='robust', outcome_model=...)`` against Stata 18 ``mediate``.

Data: ``_fixtures/mediate_nonlinear.csv`` (1,500 simulated rows; a binary and
a continuous treatment, a continuous and a binary mediator, binary and count
outcomes). The Stata numbers were captured from ``e(b)`` and ``e(V)`` with
``%20.14f`` after, for example::

    mediate (yb x1 x2, probit) (m x1 x2) (d), all
    mediate (yb2 x1, logit) (mb x1, logit) (d), all nointeraction
    mediate (yc x1 x2, poisson) (m x1 x2) (d), all
    mediate (ybc x1, probit) (mc x1) (dc, continuous(1 3)), all
    estat proportion

Each entry below names its command. Stata integrates a continuous mediator
out of a non-linear outcome model with the maximum-likelihood residual
variance of the mediator regression, held fixed in the covariance; that is
reproduced here. Stata refuses a logit outcome with a linear mediator; that
pair is checked against direct numerical integration instead.

Tolerance: both sides solve the same estimating equations. Stata's variance
uses numerical derivatives, as does this implementation for the non-linear
models; agreement is 1e-8 or better on every number.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"

# (kwargs, {effect: (estimate, se)})
STATA = {
    "(yb x1 x2, probit) (m x1 x2) (d), all": (
        dict(y="yb", treat="d", mediator="m", covariates=["x1", "x2"],
             outcome_model="probit", interaction=True),
        {
            "NIE": (0.09191637134857, 0.01223695178989),
            "NDE": (0.07437080180650, 0.02640782580682),
            "PNIE": (0.10278472595557, 0.01197421560612),
            "TNDE": (0.06350244719950, 0.02494574460537),
            "TE": (0.16628717315507, 0.02469249259481),
            "Prop. Mediated": (0.55275683388313, 0.10445779274132),
        },
    ),
    "(yb x1 x2, probit) (m x1 x2) (d), all nointeraction": (
        dict(y="yb", treat="d", mediator="m", covariates=["x1", "x2"],
             outcome_model="probit"),
        {
            "NIE": (0.09598529957056, 0.01022973872077),
            "NDE": (0.07032319893814, 0.02535930263581),
            "PNIE": (0.09889211029643, 0.01025944222240),
            "TNDE": (0.06741638821227, 0.02437296084718),
            "TE": (0.16630849850871, 0.02470262462065),
            "Prop. Mediated": (0.57715210245578, 0.09815817610901),
        },
    ),
    "(yb2 x1, logit) (mb x1, logit) (d), all": (
        dict(y="yb2", treat="d", mediator="mb", covariates=["x1"],
             outcome_model="logit", mediator_model="logit", interaction=True),
        {
            "NIE": (0.05429935455998, 0.01079265435126),
            "NDE": (0.11832661536436, 0.02693787312515),
            "PNIE": (0.05475943902626, 0.01106678722368),
            "TNDE": (0.11786653089808, 0.02700274484512),
            "TE": (0.17262596992434, 0.02546810817314),
            "Prop. Mediated": (0.31454916420616, 0.07513912674285),
        },
    ),
    "(yb2 x1, logit) (mb x1, logit) (d), all nointeraction": (
        dict(y="yb2", treat="d", mediator="mb", covariates=["x1"],
             outcome_model="logit", mediator_model="logit"),
        {
            "NIE": (0.05432160816391, 0.00863733131134),
            "NDE": (0.11830455075416, 0.02616333944599),
            "PNIE": (0.05473583123312, 0.00869122966034),
            "TNDE": (0.11789032768495, 0.02610388387269),
            "TE": (0.17262615891807, 0.02546803336453),
            "Prop. Mediated": (0.31467773195192, 0.06519277065059),
        },
    ),
    "(yb2 x1, probit) (mb x1, probit) (d), nointeraction": (
        dict(y="yb2", treat="d", mediator="mb", covariates=["x1"],
             outcome_model="probit", mediator_model="probit"),
        {
            "NIE": (0.05438636173367, 0.00864139736286),
            "NDE": (0.11848175747366, 0.02614277610324),
            "TE": (0.17286811920733, 0.02546227816827),
            "Prop. Mediated": (0.31461186702936, 0.06503688486292),
        },
    ),
    "(yb2 x1, logit) (mb x1, probit) (d), nointeraction": (
        dict(y="yb2", treat="d", mediator="mb", covariates=["x1"],
             outcome_model="logit", mediator_model="probit"),
        {
            "NIE": (0.05428647497889, 0.00863968657219),
            "NDE": (0.11830835687266, 0.02616398825499),
            "TE": (0.17259483185154, 0.02547445627404),
            "Prop. Mediated": (0.31453128924266, 0.06518244368976),
        },
    ),
    "(yc x1 x2, poisson) (m x1 x2) (d), all": (
        dict(y="yc", treat="d", mediator="m", covariates=["x1", "x2"],
             outcome_model="poisson", interaction=True),
        {
            "NIE": (0.31013867819218, 0.03809953004833),
            "NDE": (0.55153629394056, 0.06942226670450),
            "PNIE": (0.25953436616416, 0.03316163736425),
            "TNDE": (0.60214060596859, 0.07554001009188),
            "TE": (0.86167497213274, 0.07087267606737),
            "Prop. Mediated": (0.35992536422934, 0.04504561056239),
        },
    ),
    "(yc x1 x2, poisson) (m x1 x2) (d), all nointeraction": (
        dict(y="yc", treat="d", mediator="m", covariates=["x1", "x2"],
             outcome_model="poisson"),
        {
            "NIE": (0.32957078252790, 0.03433203126356),
            "NDE": (0.53154180323479, 0.06584682821978),
            "PNIE": (0.23417656006387, 0.02688529164951),
            "TNDE": (0.62693602569882, 0.07533512719414),
            "TE": (0.86111258576269, 0.07093100566823),
            "Prop. Mediated": (0.38272670493603, 0.04020127232571),
        },
    ),
    "(yc x1, poisson) (mb x1, logit) (d), all": (
        dict(y="yc", treat="d", mediator="mb", covariates=["x1"],
             outcome_model="poisson", mediator_model="logit", interaction=True),
        {
            "NIE": (0.04449337664396, 0.02874292041492),
            "NDE": (0.78715105314850, 0.07611465085210),
            "PNIE": (-0.00206143645392, 0.02420227495926),
            "TNDE": (0.83370586624638, 0.07754414802676),
            "TE": (0.83164442979246, 0.07252567285591),
            "Prop. Mediated": (0.05350048055401, 0.03454892714840),
        },
    ),
    "(ybc x1, probit) (mc x1) (dc, continuous(1 3)), all": (
        dict(y="ybc", treat="dc", mediator="mc", covariates=["x1"],
             outcome_model="probit", interaction=True, treat_values=(1, 3)),
        {
            "NIE": (0.12387325154946, 0.01607692056353),
            "NDE": (0.07639238673414, 0.02777161140102),
            "PNIE": (0.10145783862960, 0.01473726017620),
            "TNDE": (0.09880779965401, 0.02384431378405),
            "TE": (0.20026563828360, 0.02109647332452),
            "Prop. Mediated": (0.61854471196921, 0.10832767246798),
        },
    ),
}  # fmt: skip


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "mediate_nonlinear.csv")


@pytest.mark.parametrize("command", sorted(STATA))
def test_effects_and_standard_errors_equal_stata(data, command):
    kwargs, expected = STATA[command]
    fit = sp.mediate(data, inference="robust", **kwargs)
    detail = fit.detail.set_index("effect")
    assert list(detail.index) == ["NIE", "NDE", "PNIE", "TNDE", "TE", "Prop. Mediated"]
    for effect, (estimate, se) in expected.items():
        assert detail.loc[effect, "estimate"] == pytest.approx(estimate, rel=1e-8)
        assert detail.loc[effect, "se"] == pytest.approx(se, rel=1e-7), effect
    assert fit.model_info["outcome_model"] == kwargs["outcome_model"]


def test_potential_outcome_means_equal_stata(data):
    # mediate (yb x1 x2, probit) (m x1 x2) (d), pomeans
    fit = sp.mediate(
        data, y="yb", treat="d", mediator="m", covariates=["x1", "x2"],
        inference="robust", outcome_model="probit", interaction=True,
    )  # fmt: skip
    po = fit.model_info["po_means"].set_index("effect")
    expected = {
        "Y0M0": (0.48322438158708, 0.01817235847837),
        "Y1M0": (0.55759518339358, 0.02126611976173),
        "Y0M1": (0.58600910754265, 0.02002625546981),
        "Y1M1": (0.64951155474215, 0.01715484523633),
    }
    for name, (estimate, se) in expected.items():
        assert po.loc[name, "estimate"] == pytest.approx(estimate, rel=1e-10)
        assert po.loc[name, "se"] == pytest.approx(se, rel=1e-8)
    # outcome coefficient standard errors, e(V) of the yb equation
    se = fit.model_info["outcome_se"]
    assert se["d"] == pytest.approx(0.08675424128777, rel=1e-8)
    assert se["m"] == pytest.approx(0.04988245972106, rel=1e-8)


def test_logit_outcome_with_linear_mediator_is_the_integral(data):
    """The pair Stata refuses. Check the quadrature against scipy's adaptive
    integration of the same definition, one observation at a time."""
    fit = sp.mediate(
        data, y="yb", treat="d", mediator="m", covariates=["x1", "x2"],
        inference="robust", outcome_model="logit",
    )  # fmt: skip
    b = fit.model_info["outcome_coef"]
    a = fit.model_info["mediator_coef"]
    s2 = fit.model_info["mediator_variance"]
    rows = data.head(40)

    def po_mean(d, d_prime):
        total = 0.0
        for _, r in rows.iterrows():
            eta = b["_cons"] + b["d"] * d + b["x1"] * r.x1 + b["x2"] * r.x2
            mean_m = a["_cons"] + a["d"] * d_prime + a["x1"] * r.x1 + a["x2"] * r.x2
            val, _ = integrate.quad(
                lambda m: stats.norm.pdf(m, mean_m, np.sqrt(s2))
                / (1 + np.exp(-(eta + b["m"] * m))),
                mean_m - 12 * np.sqrt(s2),
                mean_m + 12 * np.sqrt(s2),
                epsabs=1e-13,
            )
            total += val
        return total / len(rows)

    sub = sp.mediate(
        data, y="yb", treat="d", mediator="m", covariates=["x1", "x2"],
        inference="robust", outcome_model="logit",
    )  # fmt: skip
    # the same coefficients evaluated on the first 40 rows by quadrature
    from statspai.mediation._po_means_nonlinear import _integrated_mean

    eta = (b["_cons"] + b["d"] * 1 + b["x1"] * rows.x1 + b["x2"] * rows.x2).to_numpy()
    mean_m = (
        a["_cons"] + a["d"] * 0 + a["x1"] * rows.x1 + a["x2"] * rows.x2
    ).to_numpy()
    gh = _integrated_mean(eta, b["m"], mean_m, s2, "logit").mean()
    assert gh == pytest.approx(po_mean(1, 0), rel=1e-10)
    assert sub.estimate == pytest.approx(fit.estimate)
    # and the effects are probabilities' differences of a sensible size
    assert 0.02 < fit.estimate < 0.2


def test_nonlinear_decomposition_adds_up(data):
    fit = sp.mediate(
        data, y="yc", treat="d", mediator="m", covariates=["x1", "x2"],
        inference="robust", outcome_model="poisson", interaction=True,
    )  # fmt: skip
    e = fit.detail.set_index("effect")["estimate"]
    assert e["NIE"] + e["NDE"] == pytest.approx(e["TE"], rel=1e-12)
    assert e["PNIE"] + e["TNDE"] == pytest.approx(e["TE"], rel=1e-12)


def test_linear_outcome_with_treat_values_is_the_coefficient_product(data):
    """Linear models, continuous treatment: the indirect effect of moving the
    treatment from 1 to 3 is 2 * a * b."""
    fit = sp.mediate(
        data.assign(yl=data["mc"] * 0.5 + data["dc"] * 0.2 + data["x1"]),
        y="yl", treat="dc", mediator="mc", covariates=["x1"],
        inference="robust", treat_values=(1, 3),
    )  # fmt: skip
    a = fit.model_info["mediator_coef"]["dc"]
    b = fit.model_info["outcome_coef"]["mc"]
    assert fit.estimate == pytest.approx(2 * a * b, rel=1e-9)
    assert fit.model_info["treat_values"] == (1.0, 3.0)


def test_requirements_are_stated(data):
    base = dict(y="yb", treat="d", mediator="m", covariates=["x1"])
    with pytest.raises(MethodIncompatibility, match="inference='robust'"):
        sp.mediate(data, outcome_model="logit", **base)
    with pytest.raises(MethodIncompatibility, match="outcome_model must be"):
        sp.mediate(data, inference="robust", outcome_model="cloglog", **base)
    with pytest.raises(MethodIncompatibility, match="0/1 outcome"):
        sp.mediate(
            data, y="yc", treat="d", mediator="m", inference="robust",
            outcome_model="logit",
        )  # fmt: skip
    with pytest.raises(MethodIncompatibility, match="treat_values"):
        sp.mediate(
            data, y="ybc", treat="dc", mediator="mc", inference="robust",
            outcome_model="probit",
        )  # fmt: skip


def test_bootstrap_reports_an_interval_for_the_proportion_mediated():
    rng = np.random.default_rng(0)
    n = 800
    t = rng.binomial(1, 0.5, n)
    m = 0.6 * t + rng.normal(size=n)
    y = 0.5 * t + 0.7 * m + rng.normal(size=n)
    fit = sp.mediate(
        pd.DataFrame({"t": t, "m": m, "y": y}), y="y", treat="t", mediator="m",
        n_boot=400,
    )  # fmt: skip
    row = fit.detail.set_index("effect").loc["Prop. Mediated"]
    truth = 0.42 / 0.92
    assert row["ci_lower"] < row["estimate"] < row["ci_upper"]
    assert row["ci_lower"] < truth < row["ci_upper"]
    assert fit.model_info["ci_prop_mediated"] == (row["ci_lower"], row["ci_upper"])
