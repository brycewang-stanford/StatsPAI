"""Goodness-of-fit measures for probit / logit, and tobit by SCLS.

Fixture: ``_fixtures/_generate_binary_fit_and_scls.R`` on the data of the
conditional-moment fixture.

* ``model_info['r2']`` of ``sp.probit`` and ``sp.logit`` against
  ``DescTools::PseudoR2`` (McFadden, Cox-Snell, Nagelkerke, Efron,
  McKelvey-Zavoina, Tjur) and ``micsr::rsq`` (Estrella, and four of the
  others again), at 1e-6. Observed 1e-13 against DescTools at a tightly
  converged ``glm``, 8e-9 at worst against micsr.
* ``sp.tobit(method='scls')`` coefficients against
  ``micsr::tobit1(method = "trimmed")`` at 1e-8 (observed 5e-12).

Two things from micsr are deliberately not compared, and both are checked
another way.

* Its McKelvey-Zavoina value for the probit on these data is 2.3 times the
  DescTools one. The DescTools value is the one reproduced here, and micsr
  agrees with both for the logit.
* Its standard errors for the trimmed estimator are not usable (20 to 228
  for coefficients of order one on the book's own example). Powell's
  variance is checked by the coverage of its confidence interval instead.
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
X = ["x1", "x2", "x3"]
DESCTOOLS = {
    "mcfadden": "McFadden",
    "cox_snell": "CoxSnell",
    "nagelkerke": "Nagelkerke",
    "efron": "Efron",
    "mckelvey_zavoina": "McKelveyZavoina",
    "tjur": "Tjur",
}
MICSR = {
    "mcfadden": "mcfadden",
    "tjur": "tjur",
    "estrella": "estrella",
    "efron": "rss",
    "cox_snell": "lr",
}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "cmtest_data.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads((_FIX / "binary_fit_and_scls.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("link", ["probit", "logit"])
def test_goodness_of_fit_family(link, data, ref):
    r2 = getattr(sp, link)(data=data, y="yb", x=X).model_info["r2"]
    for ours, theirs in DESCTOOLS.items():
        assert r2[ours] == pytest.approx(ref[link]["desctools"][theirs], rel=1e-6), ours
    for ours, theirs in MICSR.items():
        assert r2[ours] == pytest.approx(ref[link]["micsr"][theirs], rel=1e-6), ours
    assert r2["mcfadden"] == pytest.approx(
        getattr(sp, link)(data=data, y="yb", x=X).model_info["pseudo_r2"]
    )
    assert all(0.0 < v < 1.0 for v in r2.values())


def test_micsr_mckelvey_zavoina_differs_for_probit_only(data, ref):
    ours = sp.probit(data=data, y="yb", x=X).model_info["r2"]["mckelvey_zavoina"]
    assert ours == pytest.approx(
        ref["probit"]["desctools"]["McKelveyZavoina"], rel=1e-6
    )
    assert abs(ref["probit"]["micsr"]["mckel_zavo"] / ours - 1.0) > 0.5
    logit = sp.logit(data=data, y="yb", x=X).model_info["r2"]["mckelvey_zavoina"]
    assert logit == pytest.approx(ref["logit"]["micsr"]["mckel_zavo"], rel=1e-6)


def test_weighted_fits_report_no_family(data):
    w = data.assign(w=1.0 + data["x2"])
    assert sp.probit(data=w, y="yb", x=X, weights="w").model_info["r2"] is None


def test_scls_coefficients(data, ref):
    res = sp.tobit(data, "yc", X, method="scls")
    assert ref["scls_names"] == ["(Intercept)"] + X
    np.testing.assert_allclose(res.params.values, ref["scls_coef"], atol=1e-8)
    assert "sigma" not in res.params.index
    assert "powell1986symmetrically" in res.cite()


def test_scls_is_a_fixed_point(data):
    """The estimate solves least squares on the symmetrically censored sample."""
    res = sp.tobit(data, "yc", X, method="scls")
    b = res.params.values
    Xm = np.column_stack([np.ones(len(data)), data[X].to_numpy()])
    xb = Xm @ b
    keep = xb > 0
    again = np.linalg.lstsq(
        Xm[keep], np.minimum(data["yc"].to_numpy()[keep], 2 * xb[keep]), rcond=None
    )[0]
    np.testing.assert_allclose(again, b, atol=1e-9)
    assert res.model_info["n_used"] == int(keep.sum())


def test_scls_is_consistent_where_maximum_likelihood_is_not():
    """Heteroskedastic, fat-tailed, symmetric errors; true slope 1."""
    rng = np.random.default_rng(1)
    reps, cover, slopes, mle = 200, 0, [], []
    for _ in range(reps):
        n = 1500
        x = rng.normal(size=n)
        e = np.exp(0.5 * x) * rng.standard_t(5, size=n)
        df = pd.DataFrame({"y": np.maximum(1 + x + e, 0.0), "x": x})
        a = sp.tobit(df, "y", ["x"], method="scls")
        slopes.append(a.params["x"])
        cover += abs(a.params["x"] - 1.0) < 1.96 * a.std_errors["x"]
        mle.append(sp.tobit(df, "y", ["x"]).params["x"])
    assert abs(np.mean(slopes) - 1.0) < 0.03
    # 95% nominal; 200 replications put a correct interval inside [0.91, 0.99].
    assert 0.91 <= cover / reps <= 0.99, cover / reps
    assert np.mean(mle) > 1.25


def test_scls_shifts_with_the_limit_and_clusters(data):
    base = sp.tobit(data, "yc", X, method="scls")
    shifted = sp.tobit(data.assign(yc=data["yc"] + 3.0), "yc", X, ll=3.0, method="scls")
    np.testing.assert_allclose(
        shifted.params.values[1:], base.params.values[1:], atol=1e-9
    )
    assert shifted.params["const"] == pytest.approx(base.params["const"] + 3.0)
    cl = sp.tobit(
        data.assign(g=np.arange(len(data)) % 40), "yc", X, method="scls", cluster="g"
    )
    np.testing.assert_allclose(cl.params.values, base.params.values)
    assert not np.allclose(cl.std_errors.values, base.std_errors.values)


def test_scls_refusals(data):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="lower limit only"):
        sp.tobit(data, "yc", X, ul=5.0, method="scls")
    with pytest.raises(bad, match="weights"):
        sp.tobit(data.assign(w=1.0), "yc", X, weights="w", method="scls")
    with pytest.raises(bad, match="not available"):
        sp.tobit(data, "yc", X, method="twostep")


def test_tobit_feeds_vuong(data):
    out = sp.vuong(sp.tobit(data, "yc", X), sp.tobit(data, "yc", ["x1", "x3"]))
    assert out["k1"] == 5 and out["k2"] == 4
    assert out["loglik1"] > out["loglik2"]
