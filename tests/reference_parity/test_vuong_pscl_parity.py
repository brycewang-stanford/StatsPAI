"""``sp.vuong`` and per-observation log-likelihoods against R ``pscl``.

Fixture: ``_fixtures/_generate_vuong_pscl.R`` simulates overdispersed,
zero-inflated counts, writes them at 17 digits and fits five models in R
(``glm`` Poisson, ``MASS::glm.nb``, ``pscl::zeroinfl`` Poisson and negative
binomial, ``pscl::hurdle``).

Three things are pinned.

* The per-observation log-likelihood every StatsPAI fit now carries, against
  R's own densities at R's own estimates: 1e-7 (observed 4e-14 for Poisson,
  7e-8 at worst, which is where R's optimiser stopped).
* The Vuong statistic, raw and with the AIC and BIC corrections: 1e-6
  (observed 6e-8 at worst).
* The statistic that ``sp.zip_model`` and ``sp.zinb`` report on their own.
  Before 1.39 it compared against a Poisson (negative binomial) density
  evaluated at the zero-inflated model's count coefficients, not at that
  model's own maximum, and came out as 13.4 where the test is 8.69.

``pscl::vuong`` prints its result and returns nothing, so the fixture
rebuilds the statistic from the per-observation log-likelihoods and stores
the printed lines too. The raw statistic is checked against the print in
every pair. The corrected ones are checked where pscl counts parameters as
we do; it leaves a negative binomial dispersion out of the count, which is
a documented difference and not a tolerance.
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
X = ["x1", "x2"]
PAIRS = ["zip_poisson", "zinb_nb2", "zip_nb2", "hurdle_zip", "nb2_hurdle"]
# Pairs in which exactly one model is negative binomial.
PSCL_COUNTS_DIFFERENTLY = {"zip_nb2", "nb2_hurdle"}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "vuong_data.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((_FIX / "vuong_pscl.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fits(data) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "poisson": sp.poisson(data=data, y="y", x=X),
            "nb2": sp.nbreg(data=data, y="y", x=X),
            "zip": sp.zip_model(data=data, y="y", x=X),
            "zinb": sp.zinb(data=data, y="y", x=X),
            "hurdle": sp.hurdle(data=data, y="y", x=X),
        }


@pytest.mark.parametrize("model", ["poisson", "nb2", "zip", "zinb", "hurdle"])
def test_per_observation_loglik(model, fits, ref):
    ll = fits[model].data_info["llobs"]
    np.testing.assert_allclose(ll, ref["llobs"][model], atol=1e-7)
    assert ll.sum() == pytest.approx(ref["loglik"][model], abs=1e-8)


@pytest.mark.parametrize("pair", PAIRS)
def test_vuong_statistics(pair, fits, ref):
    a, b = pair.split("_")
    out, want = sp.vuong(fits[a], fits[b]), ref["pairs"][pair]
    assert (out["k1"], out["k2"]) == (want["k1"], want["k2"])
    assert out["statistic"] == pytest.approx(want["z"]["raw"], rel=1e-6)
    assert out["aic"]["statistic"] == pytest.approx(want["z"]["aic"], rel=1e-6)
    assert out["bic"]["statistic"] == pytest.approx(want["z"]["bic"], rel=1e-6)


def _printed(lines, label):
    row = next(ln for ln in lines if ln.startswith(label))
    return float(row[len(label) :].split()[0])


@pytest.mark.parametrize("pair", PAIRS)
def test_against_what_pscl_prints(pair, fits, ref):
    a, b = pair.split("_")
    out, lines = sp.vuong(fits[a], fits[b]), ref["pairs"][pair]["printed"]
    assert out["statistic"] == pytest.approx(_printed(lines, "Raw"), abs=2e-6)
    aic, bic = _printed(lines, "AIC-corrected"), _printed(lines, "BIC-corrected")
    if pair in PSCL_COUNTS_DIFFERENTLY:
        # pscl's count is one short on the negative binomial side. Putting
        # its count into our formula reproduces its print.
        n = out["n_obs"]
        k = (out["k1"] - (a == "nb2")) - (out["k2"] - (b == "nb2"))
        m = fits[a].data_info["llobs"] - fits[b].data_info["llobs"]
        z = np.sqrt(n) * (m.mean() - k / n) / m.std(ddof=1)
        assert z == pytest.approx(aic, abs=2e-6)
        assert abs(out["aic"]["statistic"] - aic) > 0.05
    else:
        assert out["aic"]["statistic"] == pytest.approx(aic, abs=2e-6)
        assert out["bic"]["statistic"] == pytest.approx(bic, abs=2e-6)


def test_builtin_zero_inflation_statistic_uses_the_fitted_comparison(fits, ref):
    zip_stat = fits["zip"].diagnostics["vuong_stat"]
    zinb_stat = fits["zinb"].diagnostics["vuong_stat"]
    assert zip_stat == pytest.approx(ref["pairs"]["zip_poisson"]["z"]["raw"], rel=1e-6)
    assert zinb_stat == pytest.approx(ref["pairs"]["zinb_nb2"]["z"]["raw"], rel=1e-6)
    assert zip_stat == pytest.approx(
        sp.vuong(fits["zip"], fits["poisson"])["statistic"], rel=1e-9
    )


def test_sign_and_symmetry(fits):
    ab = sp.vuong(fits["zip"], fits["poisson"])
    ba = sp.vuong(fits["poisson"], fits["zip"])
    assert ab["statistic"] == pytest.approx(-ba["statistic"])
    assert ab["preferred"] == "model1" and ba["preferred"] == "model2"
    assert ab["loglik1"] > ab["loglik2"]


def test_equivalent_models_do_not_reject():
    """Probit against logit when the true link lies between the two."""
    rng = np.random.default_rng(20261004)
    reject = 0
    reps = 200
    for _ in range(reps):
        n = 500
        x = rng.normal(size=n)
        # A link halfway between the two: neither model is the truth.
        e = np.where(
            rng.uniform(size=n) < 0.5, rng.logistic(size=n) / 1.7, rng.normal(size=n)
        )
        df = pd.DataFrame({"y": (0.3 + 0.8 * x + e > 0).astype(float), "x": x})
        out = sp.vuong(
            sp.probit(data=df, y="y", x=["x"]), sp.logit(data=df, y="y", x=["x"])
        )
        reject += out["pvalue"] < 0.05
    assert reject / reps <= 0.10, reject / reps


def test_refusals(data, fits):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="same rows"):
        sp.vuong(fits["zip"], sp.poisson(data=data.iloc[:900], y="y", x=X))
    other = data.assign(y=data["y"].to_numpy()[::-1])
    with pytest.raises(bad, match="different outcome"):
        sp.vuong(fits["zip"], sp.poisson(data=other, y="y", x=X))
    with pytest.raises(bad, match="nothing to compare"):
        sp.vuong(fits["poisson"], fits["poisson"])
    with pytest.raises(bad, match="per-observation"):
        sp.vuong(fits["poisson"], sp.regress("y ~ x1", data=data))
    w = data.assign(w=1.0 + data["x2"])
    with pytest.raises(bad, match="weights"):
        sp.vuong(fits["zip"], sp.poisson(data=w, y="y", x=X, weights="w"))


def test_stata_forcevuong_agrees(fits):
    """Stata 18 `zip` / `zinb`, `forcevuong`: an independent second reference.

    Fixture: ``_fixtures/_generate_vuong_stata.do`` on the same CSV.
    """
    stata = json.loads((_FIX / "vuong_stata.json").read_text(encoding="utf-8"))
    for pair, key in (("zip_poisson", "zip"), ("zinb_nb2", "zinb")):
        assert fits[key].diagnostics["vuong_stat"] == pytest.approx(
            stata[pair], rel=1e-6
        )
    # Without `forcevuong` Stata refuses, for the reason the docstring gives.
    assert stata["rc_vuong_without_force"] == 498
