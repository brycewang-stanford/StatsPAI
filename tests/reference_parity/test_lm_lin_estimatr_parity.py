"""``sp.lm_lin`` against R ``estimatr::lm_lin`` 2.0.0, and its population variance.

Lin's (2013) interacted regression adjustment is the estimator the
experiments chapter of *Applied Causal Inference Powered by ML and AI*
builds by hand. Two layers of evidence:

* **Reference parity (T2).** Estimate, standard error, degrees of freedom,
  interval and p-value for every variance estimator, unclustered and
  clustered, at 1e-9 relative: both sides are closed-form functions of the
  same least-squares fit. Fixture: ``_generate_lm_lin.R`` (simulated).
* **Known truth (T1).** With random covariates and heterogeneous effects
  the regression variance covers the *sample* average effect but not the
  population one; ``superpopulation=True`` restores nominal coverage of
  the population effect. 600 replications; a binomial 3-sd band around
  0.95 is +/- 0.027.

References
----------
[@lin2013agnostic]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
COVARIATES = ["x1", "x2", "g"]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(FIX / "lm_lin.csv")


@pytest.fixture(scope="module")
def reference() -> dict:
    return json.loads((FIX / "lm_lin_R.json").read_text(encoding="utf-8"))


def _check(fit, ref):
    assert fit.estimate == pytest.approx(ref["estimate"], rel=1e-9)
    assert fit.se == pytest.approx(ref["se"], rel=1e-9)
    assert fit.model_info["df"] == pytest.approx(ref["df"], rel=1e-9)
    assert fit.ci[0] == pytest.approx(ref["lower"], rel=1e-9)
    assert fit.ci[1] == pytest.approx(ref["upper"], rel=1e-9)
    assert fit.pvalue == pytest.approx(ref["p"], rel=1e-7)


@pytest.mark.parametrize("vce", ["hc2", "hc0", "hc1", "hc3", "classical"])
def test_matches_estimatr(data, reference, vce):
    _check(sp.lm_lin(data, "y", "d", COVARIATES, vce=vce), reference[vce])


@pytest.mark.parametrize("vce", ["cr2", "stata"])
def test_matches_estimatr_with_clusters(data, reference, vce):
    fit = sp.lm_lin(data, "y", "dc", COVARIATES, cluster="cl", vce=vce)
    _check(fit, reference[vce])
    assert fit.model_info["n_clusters"] == 50


def test_defaults_are_hc2_and_cr2(data, reference):
    _check(sp.lm_lin(data, "y", "d", COVARIATES), reference["hc2"])
    _check(sp.lm_lin(data, "y", "dc", COVARIATES, cluster="cl"), reference["cr2"])


def test_every_coefficient_matches(data, reference):
    fit = sp.lm_lin(data, "y", "d", COVARIATES)
    ours = np.sort(fit.detail["coef"].to_numpy())
    theirs = np.sort(np.array(list(reference["coefficients"].values()), dtype=float))
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12)
    assert "d:x1" in set(fit.detail["term"])


def test_collinear_covariates_are_dropped_and_reported(data):
    extra = data.assign(x1_copy=2 * data["x1"])
    base = sp.lm_lin(data, "y", "d", COVARIATES)
    fit = sp.lm_lin(extra, "y", "d", [*COVARIATES, "x1_copy"])
    assert fit.estimate == pytest.approx(base.estimate, rel=1e-10)
    assert fit.se == pytest.approx(base.se, rel=1e-10)
    assert fit.model_info["dropped_collinear"] == ["x1_copy", "d:x1_copy"]


def _population_draw(seed: int, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    d = rng.binomial(1, 0.5, size=n)
    y = d * (1.0 + 2.0 * x) + x + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "x": x})


def test_superpopulation_variance_covers_the_population_effect():
    """True population effect 1; the sample effect is 1 + 2 * mean(x)."""
    reps = 600
    fixed_hits = super_hits = sample_hits = 0
    for r in range(reps):
        df = _population_draw(5000 + r)
        fixed = sp.lm_lin(df, "y", "d", ["x"])
        both = sp.lm_lin(df, "y", "d", ["x"], superpopulation=True)
        assert both.estimate == fixed.estimate
        assert both.se > fixed.se
        fixed_hits += fixed.ci[0] <= 1.0 <= fixed.ci[1]
        super_hits += both.ci[0] <= 1.0 <= both.ci[1]
        sate = 1.0 + 2.0 * df["x"].mean()
        sample_hits += fixed.ci[0] <= sate <= fixed.ci[1]
    assert abs(super_hits / reps - 0.95) <= 0.027
    assert fixed_hits / reps < 0.90  # too narrow for the population effect
    assert abs(sample_hits / reps - 0.95) <= 0.027  # right for the sample effect


def test_superpopulation_term_is_gamma_var_x_gamma_over_n(data):
    fit = sp.lm_lin(data, "y", "d", ["x1"], vce="hc0", superpopulation=True)
    gamma = float(fit.detail.set_index("term").loc["d:x1", "coef"])
    extra = gamma**2 * data["x1"].var(ddof=1) / len(data)
    assert fit.model_info["var_covariate_means"] == pytest.approx(extra, rel=1e-12)
    assert fit.se**2 == pytest.approx(
        fit.model_info["se_regression"] ** 2 + extra, rel=1e-12
    )


def test_more_precise_than_the_difference_in_means(data):
    assert sp.lm_lin(data, "y", "d", COVARIATES).se < (
        sp.difference_in_means(data, "y", "d").se
    )


def test_refusals(data):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="two values"):
        sp.lm_lin(data, "y", "x1", ["x2"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="covariates"):
        sp.lm_lin(data, "y", "d", [])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="vce"):
        sp.lm_lin(data, "y", "d", ["x1"], vce="cr2")
    with pytest.raises(sp.exceptions.ColumnNotFound, match="nope"):
        sp.lm_lin(data, "y", "d", ["nope"])
    with pytest.raises(sp.exceptions.DataInsufficient, match="rows"):
        sp.lm_lin(data.head(5), "y", "d", ["x1", "x2"])
