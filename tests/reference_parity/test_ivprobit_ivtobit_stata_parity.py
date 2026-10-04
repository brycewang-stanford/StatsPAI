"""Cross-language parity: ``sp.ivprobit`` / ``sp.ivtobit`` against Stata 18.

Fixture: ``_fixtures/_generate_ivprobit_ivtobit_stata.do`` (official
``ivprobit`` and ``ivtobit``; nothing to install). The data are generated
in Stata and exported at ``%21.16e``, so both sides read the same bytes.

Every block is held to 1e-6 on coefficients, standard errors and the two
Wald statistics.

* Newey's two-step estimator is closed form after two probit / tobit fits
  and lands between 1e-13 and 2e-8.
* The maximum likelihood blocks land at 1e-8 or better on coefficients and
  8e-8 on standard errors (ours come from a numerical Hessian of exact
  scores). That needed one thing on the Stata side: ``ml`` stops at
  ``nrtolerance(1e-5)`` by default, which leaves coefficients up to 3e-6
  short of the optimum, so the do-file tightens the stopping rule. With
  Stata's defaults the same comparison shows gaps of 3e-6 that are the
  stopping rule and nothing else; ``sp`` always iterates to a gradient of
  1e-6 or less, asserted below.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
ONE = dict(endog="w1", instruments=["z1", "z2"])
TWO = dict(endog=["w1", "w2"], instruments=["z1", "z2", "z3"])

# block -> (function, keyword arguments)
BLOCKS = {
    "ivprobit_mle": ("ivprobit", dict(y="d", **ONE)),
    "ivprobit_mle_robust": ("ivprobit", dict(y="d", vce="robust", **ONE)),
    "ivprobit_mle_cluster": ("ivprobit", dict(y="d", cluster="clust", **ONE)),
    "ivprobit_twostep": ("ivprobit", dict(y="d", method="twostep", **ONE)),
    "ivprobit_mle_2endog": ("ivprobit", dict(y="d", **TWO)),
    "ivprobit_twostep_2endog": ("ivprobit", dict(y="d", method="twostep", **TWO)),
    "ivtobit_mle": ("ivtobit", dict(y="yc", **ONE)),
    "ivtobit_mle_robust": ("ivtobit", dict(y="yc", vce="robust", **ONE)),
    "ivtobit_mle_cluster": ("ivtobit", dict(y="yc", cluster="clust", **ONE)),
    "ivtobit_twostep": ("ivtobit", dict(y="yc", method="twostep", **ONE)),
    "ivtobit_mle_2endog": ("ivtobit", dict(y="yc", **TWO)),
    "ivtobit_twostep_2endog": ("ivtobit", dict(y="yc", method="twostep", **TWO)),
    "ivtobit_mle_twolimit": ("ivtobit", dict(y="yb", ll=-1, ul=2.5, **ONE)),
    "ivtobit_twostep_twolimit": (
        "ivtobit",
        dict(y="yb", ll=-1, ul=2.5, method="twostep", **ONE),
    ),
}
RTOL_MLE = 1e-6
RTOL_TWOSTEP = 1e-6


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "ivprobit_ivtobit_data.csv")


@pytest.fixture(scope="module")
def stata() -> dict:
    return json.loads(
        (_FIX / "ivprobit_ivtobit_stata.json").read_text(encoding="utf-8")
    )


def _fit(block: str, data: pd.DataFrame):
    fn, kw = BLOCKS[block]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, fn)(data, x=["x1", "x2"], **kw)


@pytest.mark.parametrize("block", sorted(BLOCKS))
def test_coefficients_and_standard_errors(block, data, stata):
    res, ref = _fit(block, data), stata[block]
    rtol = RTOL_TWOSTEP if "twostep" in block else RTOL_MLE
    if "twostep" in block:
        # Stata labels the two-step coefficients without the equation name.
        assert [n.split(":")[-1] for n in res.params.index] == ref["names"]
    else:
        assert list(res.params.index) == ref["names"]
    np.testing.assert_allclose(res.params.values, ref["b"], rtol=rtol, atol=1e-9)
    np.testing.assert_allclose(res.std_errors.values, ref["se"], rtol=rtol, atol=1e-9)


@pytest.mark.parametrize("block", sorted(BLOCKS))
def test_wald_tests(block, data, stata):
    res, ref = _fit(block, data), stata[block]
    rtol = RTOL_TWOSTEP if "twostep" in block else RTOL_MLE
    info = res.model_info
    assert info["exogeneity_chi2"] == pytest.approx(ref["chi2_exog"], rel=rtol)
    assert info["exogeneity_pvalue"] == pytest.approx(ref["p_exog"], rel=1e-4)
    assert info["wald_chi2"] == pytest.approx(ref["chi2"], rel=rtol)


@pytest.mark.parametrize("block", sorted(b for b in BLOCKS if "mle" in b))
def test_both_sides_reach_the_same_optimum(block, data, stata):
    info = _fit(block, data).model_info
    assert info["log_likelihood"] == pytest.approx(stata[block]["ll"], abs=1e-8)
    assert info["gradient_norm"] < 1e-6
    assert info["converged"]


def test_robust_and_cluster_change_only_the_standard_errors(data):
    plain = _fit("ivprobit_mle", data)
    for block in ("ivprobit_mle_robust", "ivprobit_mle_cluster"):
        other = _fit(block, data)
        np.testing.assert_allclose(other.params.values, plain.params.values)
        assert not np.allclose(other.std_errors.values, plain.std_errors.values)
    assert _fit("ivprobit_mle_cluster", data).model_info["n_clusters"] == 60


def test_twostep_and_mle_differ_by_the_documented_scale(data):
    """Newey normalises Var(u | v) = 1, maximum likelihood Var(u) = 1."""
    mle = _fit("ivprobit_mle", data)
    two = _fit("ivprobit_twostep", data)
    rho = mle.model_info["rho_w1"]
    ratio = two.params.values[:4] / mle.params.values[:4]
    np.testing.assert_allclose(ratio, 1 / np.sqrt(1 - rho**2), rtol=0.03)


def test_known_truth_is_recovered(data):
    """The do-file's DGP: 0.8 on w1, -0.5 on w2, 0.4 on x1, -0.6 on x2."""
    res = _fit("ivprobit_mle_2endog", data)
    truth = np.array([0.8, -0.5, 0.4, -0.6, 0.3])
    z = (res.params.values[:5] - truth) / res.std_errors.values[:5]
    assert np.all(np.abs(z) < 3.0)
    # A probit that ignores the endogeneity is far off on the same data.
    naive = sp.probit(data=data, y="d", x=["w1", "w2", "x1", "x2"])
    assert abs(naive.params["w1"] - 0.8) > 5 * naive.std_errors["w1"]


@pytest.mark.parametrize("fn", ["ivprobit", "ivtobit"])
def test_refusals(fn, data):
    f = getattr(sp, fn)
    y = "d" if fn == "ivprobit" else "yc"
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="endog"):
        f(data, y=y, x=["x1"], instruments=["z1"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="order condition"):
        f(data, y=y, x=["x1"], endog=["w1", "w2"], instruments=["z1"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="more than one"):
        f(data, y=y, x=["x1", "z1"], endog="w1", instruments=["z1", "z2"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="twostep"):
        f(
            data,
            y=y,
            x=["x1"],
            endog="w1",
            instruments=["z1"],
            method="twostep",
            vce="robust",
        )
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not found"):
        f(data, y=y, x=["nope"], endog="w1", instruments=["z1"])


def test_ivprobit_refuses_a_non_binary_outcome(data):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="binary"):
        sp.ivprobit(data, y="yc", x=["x1"], endog="w1", instruments=["z1"])


def test_ivtobit_without_limits_is_refused(data):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="linear IV"):
        sp.ivtobit(
            data, y="yc", x=["x1"], endog="w1", instruments=["z1"], ll=None, ul=None
        )


def test_from_stata_round_trip(data, stata):
    """``sp.from_stata`` writes a call that reproduces the Stata block."""
    cases = {
        "ivprobit_mle_cluster": "ivprobit d x1 x2 (w1 = z1 z2), vce(cluster clust)",
        "ivprobit_twostep_2endog": "ivprobit d x1 x2 (w1 w2 = z1 z2 z3), twostep",
        "ivtobit_mle_twolimit": "ivtobit yb x1 x2 (w1 = z1 z2), ll(-1) ul(2.5)",
    }
    for block, command in cases.items():
        out = sp.from_stata(command)
        assert out["ok"] and out["untranslated_options"] == [], out
        res = getattr(sp, out["tool"])(data, **out["arguments"])
        np.testing.assert_allclose(
            res.params.values, stata[block]["b"], rtol=RTOL_MLE, atol=1e-9
        )
    # ivtobit censors only where Stata was told to: no ll() means no lower limit.
    args = sp.from_stata("ivtobit yc x1 (w1 = z1), ul(5)")["arguments"]
    assert args["ll"] is None and args["ul"] == 5.0
    lost = sp.from_stata("ivtobit yc x1 (w1 = z1), ll")
    assert lost["untranslated_options"] == ["ll"]


def test_cite_returns_the_registered_reference(data):
    for block in ("ivprobit_twostep", "ivtobit_mle"):
        assert "newey1987efficient" in _fit(block, data).cite()
