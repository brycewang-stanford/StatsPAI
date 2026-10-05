"""``CoxResult.ph_test`` against ``survival::cox.zph`` and Stata ``estat phtest``.

The proportional-hazards test is a score test with a closed form, so both
references can be matched to rounding. ``method='score'`` (the default) is
the test of ``cox.zph`` in survival >= 3.0; ``method='approx'`` is the
Grambsch-Therneau (1994) approximation that Stata reports.

Fixtures: ``_fixtures/cox_ph_test.csv`` and ``cox_ph_test_R.json`` (written by
``_generate_cox_ph_test.R``, survival 3.8-3). The Stata numbers were produced
by Stata 18 on the same CSV::

    stset time, failure(event)
    stcox x1 x2 x3, efron nohr
    estat phtest, detail            // and with km / rank / log
    stset time_tied, failure(event)
    stcox x1 x2 x3, breslow nohr
    estat phtest, detail km         // r(chi2) = 26.820501592996717

and are the values Stata prints (five decimals for rho, two for chi2).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
X = ["x1", "x2", "x3"]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "cox_ph_test.csv")


@pytest.fixture(scope="module")
def r_ref():
    return json.loads((FIX / "cox_ph_test_R.json").read_text(encoding="utf-8"))


def test_score_test_equals_cox_zph(data, r_ref):
    n_checked = 0
    for key, ref in r_ref.items():
        if key == "version":
            continue
        time, ties, strata = key.split("|")
        fit = sp.cox(
            data=data,
            duration=time,
            event="event",
            x=X,
            ties=ties,
            strata="g" if strata == "strata" else None,
        )
        for transform in ("km", "identity", "rank", "log"):
            out = fit.ph_test(transform=transform)
            assert list(out["variable"]) == X + ["GLOBAL"]
            assert list(out["df"]) == [1, 1, 1, 3]
            # Tolerance: both sides evaluate the same closed form at their
            # own converged coefficients, which agree to about 1e-9.
            np.testing.assert_allclose(
                out["chi2"],
                [row["chisq"] for row in ref[transform]],
                rtol=1e-5,
                atol=1e-6,
                err_msg=f"{key} {transform}",
            )
            np.testing.assert_allclose(
                out["p_value"],
                [row["p"] for row in ref[transform]],
                rtol=1e-4,
                atol=1e-7,
            )
            n_checked += 1
    assert n_checked == 32


STATA_UNTIED_EFRON = {
    # transform: (rho x1..x3, chi2 x1..x3, global chi2)
    "identity": ([-0.04089, -0.24752, -0.21270], [0.41, 14.61, 9.02], 21.96),
    "km": ([-0.04397, -0.33914, -0.15809], [0.47, 27.43, 4.98], 30.65),
    "rank": ([-0.04206, -0.34954, -0.14465], [0.43, 29.14, 4.17], 31.65),
    "log": ([-0.02839, -0.36112, -0.13591], [0.20, 31.10, 3.68], 33.14),
}


def test_approximation_equals_stata_estat_phtest(data):
    fit = sp.cox(data=data, duration="time", event="event", x=X)
    for transform, (rho, chi2, chi2_global) in STATA_UNTIED_EFRON.items():
        out = fit.ph_test(transform=transform, method="approx")
        # Stata prints rho to five decimals and chi2 to two.
        np.testing.assert_allclose(out["rho"].iloc[:3], rho, atol=5.1e-6)
        np.testing.assert_allclose(out["chi2"].iloc[:3], chi2, atol=5.1e-3)
        assert out["chi2"].iloc[3] == pytest.approx(chi2_global, abs=5.1e-3)
    tied = sp.cox(data=data, duration="time_tied", event="event", x=X, ties="breslow")
    out = tied.ph_test(transform="km", method="approx")
    assert out["chi2"].iloc[3] == pytest.approx(26.820501592996717, rel=1e-7)
    np.testing.assert_allclose(
        out["rho"].iloc[:3], [-0.02187, -0.31671, -0.16374], atol=5.1e-6
    )


def test_the_non_proportional_covariate_is_the_one_flagged(data):
    fit = sp.cox(data=data, duration="time", event="event", x=X)
    out = fit.ph_test().set_index("variable")
    assert out.loc["x2", "p_value"] < 1e-5
    assert out.loc["x1", "p_value"] > 0.5
    assert fit.model_info["ph_test"]["worst_variable"] == "x2"


def test_size_under_proportional_hazards():
    """Rejection rate at 5% when the hazards are proportional."""
    rng = np.random.default_rng(20261006)
    reps, rejections = 300, 0
    for _ in range(reps):
        n = 200
        x = rng.standard_normal(n)
        t = rng.exponential(1.0 / np.exp(0.5 * x))
        c = rng.exponential(3.0, n)
        df = pd.DataFrame({"t": np.minimum(t, c), "d": (t <= c).astype(int), "x": x})
        fit = sp.cox(data=df, duration="t", event="d", x=["x"])
        rejections += fit.ph_test()["p_value"].iloc[0] < 0.05
    # 300 draws: a 5% test rejects between 2% and 9% of the time with
    # probability above 0.99.
    assert 0.02 <= rejections / reps <= 0.09


def test_options_are_validated(data):
    fit = sp.cox(data=data, duration="time", event="event", x=X)
    with pytest.raises(ValueError, match="transform"):
        fit.ph_test(transform="sqrt")
    with pytest.raises(ValueError, match="method"):
        fit.ph_test(method="exact")
    custom = fit.ph_test(transform=np.sqrt)
    assert np.isfinite(custom["chi2"]).all()
    zero = data.assign(time=np.where(data.index == 0, 0.0, data["time"]))
    zero.loc[0, "event"] = 1
    fit0 = sp.cox(data=zero, duration="time", event="event", x=X)
    with pytest.raises(ValueError, match="not finite"):
        fit0.ph_test(transform="log")
