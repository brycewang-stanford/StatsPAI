"""Examples of Wooldridge, *Introductory Econometrics*, against Stata 18.

The reference numbers below were printed by Stata 18 MP on the datasets
that accompany the book (``mroz``, ``barium``, ``nyse``, ``crime1``,
``hprice1``), with the commands quoted next to each test. They cover the
places where StatsPAI was wrong or silent before the October 2026 review
(``docs/dev/2026-10-05-wooldridge-review.md``): DFBETAS, the ARCH LM and
Durbin's alternative tests, the quasi-Poisson covariance, and the Stata
translations of ``heckman``, ``truncreg`` and ``glm``.

The datasets are not redistributed here. Point ``STATSPAI_WOOLDRIDGE_DIR``
at a folder holding the ``.dta`` files to run this; it is skipped otherwise.

    STATSPAI_WOOLDRIDGE_DIR=/path/to/dta \
        pytest tests/test_wooldridge_examples.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_WOOLDRIDGE_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not Path(ROOT).is_dir(),
    reason="set STATSPAI_WOOLDRIDGE_DIR to the Wooldridge .dta files",
)

# Stata prints 10 digits below; the .dta files store single precision, so
# agreement stops near 1e-7.
RTOL = 1e-6


def woo(name: str) -> pd.DataFrame:
    return pd.read_stata(Path(ROOT) / f"{name}.dta")


BARIUM = "lchnimp ~ lchempi + lgas + lrtwex + befile6 + affile6 + afdec6"


def test_example_12_4_serial_correlation_tests():
    # regress lchnimp lchempi lgas lrtwex befile6 affile6 afdec6
    fit = sp.regress(BARIUM, woo("barium"))
    # estat bgodfrey, lags(3)                -> chi2 14.768
    bg = sp.estat(fit, "bgodfrey", lags=3, print_results=False)
    assert round(bg["statistic"], 3) == 14.768
    # estat durbinalt, lags(3)               -> chi2 15.374
    alt = sp.estat(fit, "durbinalt", lags=3, print_results=False)
    assert round(alt["statistic"], 3) == 15.374
    # estat durbinalt, lags(3) small         -> F(3, 121) = 5.125, the test
    # of the three lagged residuals the book reports
    small = sp.estat(fit, "durbinalt", lags=3, version="fstat", print_results=False)
    assert round(small["statistic"], 3) == 5.125
    assert (small["df1"], small["df2"]) == (3, 121)


def test_example_12_5_cochrane_orcutt_and_prais_winsten():
    barium = woo("barium")
    # prais ..., corc
    corc = sp.prais(BARIUM, barium, method="corc")
    np.testing.assert_allclose(
        [
            corc.model_info["rho"],
            corc.params["lchempi"],
            corc.std_errors["lchempi"],
            corc.params["Intercept"],
            corc.std_errors["Intercept"],
        ],
        [0.2933586530, 2.9474447202, 0.6455564108, -37.3205733748, 23.2215215896],
        rtol=RTOL,
    )
    # prais ...
    pw = sp.prais(BARIUM, barium)
    np.testing.assert_allclose(
        [pw.model_info["rho"], pw.params["lchempi"], pw.std_errors["lchempi"]],
        [0.2932138675, 2.9409634460, 0.6328381034],
        rtol=RTOL,
    )


def test_example_12_9_arch_lm():
    nyse = woo("nyse")
    nyse["return_1"] = nyse["return"].shift(1)
    # regress return L.return
    fit = sp.regress('Q("return") ~ return_1', nyse)
    # estat archlm, lags(1) / lags(2)
    one = sp.estat(fit, "archlm", lags=1, print_results=False)
    two = sp.estat(fit, "archlm", lags=2, print_results=False)
    np.testing.assert_allclose(one["statistic"], 78.1611787218, rtol=1e-9)
    np.testing.assert_allclose(two["statistic"], 79.0538854896, rtol=1e-9)
    np.testing.assert_allclose(one["pvalue"], 9.496667477e-19, rtol=1e-6)


def test_example_17_3_quasi_poisson():
    # glm narr86 pcnv ... born60, family(poisson) scale(x2)
    fit = sp.glm(
        "narr86 ~ pcnv + avgsen + tottime + ptime86 + qemp86 + inc86 + black"
        " + hispan + born60",
        woo("crime1"),
        family="poisson",
        scale="x2",
    )
    np.testing.assert_allclose(
        [
            fit.std_errors["pcnv"],
            fit.std_errors["Intercept"],
            fit.model_info["vcov_scale"],
        ],
        [0.1046487782, 0.0828238476, 1.5167881618],
        rtol=RTOL,
    )


def test_influence_statistics():
    # regress price lotsize sqrft bdrms ; predict rs, rstudent ; dfbeta
    fit = sp.regress("price ~ lotsize + sqrft + bdrms", woo("hprice1"))
    out = sp.estat(fit, "leverage", print_results=False)
    np.testing.assert_allclose(
        [out["rstudent"][0], *out["dfbetas"][0][1:]],
        [-0.7686373591, 0.0361297503, -0.0498301201, -0.0137028527],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        [out["rstudent"].min(), out["rstudent"].max()],
        [-5.711375, 3.811597],
        rtol=1e-6,
    )
    # summarize _dfbeta_1 : min -13.02199
    np.testing.assert_allclose(out["dfbetas"][:, 1].min(), -13.02199, rtol=1e-6)


SELECT = "inlf = educ exper expersq nwifeinc age kidslt6 kidsge6"


def test_example_17_5_heckman_translation_follows_stata_default():
    mroz = woo("mroz")
    # heckman lwage educ exper expersq, select(...)          (ML)
    ml = sp.stata(f"heckman lwage educ exper expersq, select({SELECT})", data=mroz)
    np.testing.assert_allclose(
        [
            ml.params["educ"],
            ml.std_errors["educ"],
            ml.model_info["log_likelihood"],
        ],
        [0.1083502306, 0.0148607031, -832.88508646],
        rtol=RTOL,
    )
    # heckman ..., select(...) twostep
    two = sp.stata(
        f"heckman lwage educ exper expersq, select({SELECT}) twostep", data=mroz
    )
    np.testing.assert_allclose(
        [
            two.params["educ"],
            two.std_errors["educ"],
            two.params["lambda (IMR)"],
            two.std_errors["lambda (IMR)"],
        ],
        [0.1090655302, 0.0155229548, 0.0322618551, 0.1336246435],
        # the inverse Mills ratio comes from a probit that each program
        # iterates to its own convergence criterion; the standard error of
        # lambda agrees to 3e-6
        rtol=1e-5,
    )


def test_truncreg_translation():
    mroz = woo("mroz").dropna(subset=["lwage"])
    # truncreg hours educ exper, ll(0)
    fit = sp.stata("truncreg hours educ exper, ll(0)", data=mroz)
    np.testing.assert_allclose(
        [
            fit.params["educ"],
            fit.std_errors["educ"],
            fit.diagnostics["log_likelihood"],
        ],
        [-28.56457354, 22.46972998, -3403.486183],
        rtol=RTOL,
    )


def test_example_16_5_three_stage_least_squares():
    mroz = woo("mroz").dropna(subset=["lwage"])
    # reg3 (hours lwage educ age kidslt6 nwifeinc) (lwage hours educ exper expersq)
    fit = sp.three_sls(
        {
            "hours": ("hours", ["educ", "age", "kidslt6", "nwifeinc"], ["lwage"]),
            "lwage": ("lwage", ["educ", "exper", "expersq"], ["hours"]),
        },
        mroz,
    )
    hours, lwage = fit.equations["hours"], fit.equations["lwage"]
    np.testing.assert_allclose(
        [
            hours["params"]["lwage"],
            hours["params"]["educ"],
            lwage["params"]["hours"],
            lwage["params"]["educ"],
        ],
        [1781.81680036, -212.79250205, 0.00019094, 0.11274107],
        # Stata printed eight decimals: one unit of the last one
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [
            hours["se"]["lwage"],
            hours["se"]["educ"],
            lwage["se"]["hours"],
            lwage["se"]["educ"],
        ],
        [436.79003802, 53.34912557, 0.00024620, 0.01527884],
        rtol=0,
        atol=1e-8,
    )


# The three tests below take their reference from R on the same files:
# censReg 0.5 (Example 17.2), survival::survreg with a Gaussian distribution
# on log duration (Example 17.4) and quantreg::rq (the LAD fit of chapter 9).


def test_example_17_2_tobit():
    mroz = woo("mroz")
    mroz["expersq"] = mroz["exper"] ** 2
    x = ["nwifeinc", "educ", "exper", "expersq", "age", "kidslt6", "kidsge6"]
    fit = sp.tobit(mroz, "hours", x)
    np.testing.assert_allclose(
        fit.params[["const"] + x].values,
        [
            965.30540554755,
            -8.81424134412,
            80.64559102016,
            131.56428008403,
            -1.86415723260,
            -54.40500568417,
            -894.02161663678,
            -16.21800251761,
        ],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        fit.std_errors[["const"] + x].values,
        [
            446.44628365689,
            4.45909821835,
            21.58309994289,
            17.27939396762,
            0.53766279183,
            7.41871714975,
            111.87835695248,
            38.64153584524,
        ],
        rtol=1e-4,
    )
    np.testing.assert_allclose(np.log(fit.params["sigma"]), 7.02288733482, rtol=1e-6)


def test_example_17_4_censored_duration_regression():
    recid = woo("recid")
    recid["fail"] = 1 - recid["cens"]
    x = [
        "workprg",
        "priors",
        "tserved",
        "felon",
        "alcohol",
        "drugs",
        "black",
        "married",
        "educ",
        "age",
    ]
    # a normal regression for log duration, censored where the spell had not
    # ended: the log-normal accelerated failure time model
    fit = sp.survreg(data=recid, duration="durat", event="fail", x=x, dist="lognormal")
    np.testing.assert_allclose(
        fit.params[["_cons", "workprg", "priors", "age", "log(sigma)"]].values,
        [
            4.0993858941306,
            -0.0625715444548,
            -0.1372528912525,
            0.0039102855177,
            0.5935863781175,
        ],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        fit.std_errors[["_cons", "workprg", "priors", "age", "log(sigma)"]].values,
        [
            0.3475350436067,
            0.1200369185910,
            0.0214586614927,
            0.0006062049644,
            0.0344121767948,
        ],
        rtol=1e-4,
    )


def test_lad_regression_with_a_transformed_regressor():
    fit = sp.qreg(woo("rdchem"), formula="rdintens ~ I(sales/1000) + profmarg")
    np.testing.assert_allclose(
        fit.params.values,
        [1.6207403735869, 0.0186947707445, 0.1182513264859],
        rtol=1e-8,
    )
