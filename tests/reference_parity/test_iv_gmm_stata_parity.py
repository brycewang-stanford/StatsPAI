"""``sp.iv(method='gmm')`` reproduces Stata ``ivregress gmm``.

Card's schooling data, over-identified (``nearc4``, ``nearc2``), two-step
GMM under each weight matrix ``ivregress gmm`` offers, with and without
``small``. The reference file is written by
``_fixtures/_generate_iv_gmm_Stata.do`` (Stata 18).

The mapping, which the ``small`` entry of ``sp.iv``'s docstring states:

==============================================  ==================================
``ivregress gmm ..., <options>``                ``sp.iv(..., method='gmm', ...)``
==============================================  ==================================
``wmatrix(robust)`` (Stata's default)           ``robust='hc1', small=False``
``wmatrix(robust) small``                       ``robust='hc1'``
``wmatrix(cluster cl)``                         ``cluster='cl', small=False``
``wmatrix(cluster cl) small``                   ``cluster='cl'``
``wmatrix(unadjusted)``                         ``gmm_vcov='efficient', small=False``
``wmatrix(unadjusted) small``                   ``gmm_vcov='efficient'``
``wmatrix(unadjusted) vce(robust) small``       (the default)
==============================================  ==================================

Until 1.40.1 the unadjusted weight matrix carried no finite-sample factor
under ``small=True``, so ``wmatrix(unadjusted) small`` had no counterpart
and the default reported the large-sample variance with t / F inference
(reported by an external audit, 2026-10-23).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
FORMULA = "lwage ~ exper + expersq + black + south + smsa + (educ ~ nearc4 + nearc2)"

#: Stata prints the two-step estimate to its optimiser's tolerance: the
#: worst of the comparisons below is 3.7e-10 (a coefficient), measured
#: 2026-10-05. 5e-9 leaves a factor of ten and is far inside the 1e-6 gate.
RTOL = 5e-9

CASES = {
    "robust": {"robust": "hc1", "small": False},
    "robust_small": {"robust": "hc1"},
    "cluster": {"cluster": "cl", "small": False},
    "cluster_small": {"cluster": "cl"},
    "unadjusted": {"gmm_vcov": "efficient", "small": False},
    "unadjusted_small": {"gmm_vcov": "efficient"},
}


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "iv_gmm_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def card():
    return pd.read_csv(_FIX / "iv_card.csv")


@pytest.mark.parametrize("key", sorted(CASES))
def test_coefficients_and_standard_errors_match_ivregress_gmm(ref, card, key):
    res = sp.iv(FORMULA, data=card, method="gmm", **CASES[key])
    want = ref[key]
    assert int(res.data_info["nobs"]) == int(want["N"])
    for name, b, se in (("educ", "b", "se"), ("exper", "b_exper", "se_exper")):
        assert float(res.params[name]) == pytest.approx(want[b], rel=RTOL)
        assert float(res.std_errors[name]) == pytest.approx(want[se], rel=RTOL)


@pytest.mark.parametrize("key", sorted(CASES))
def test_hansen_j_matches_estat_overid(ref, card, key):
    res = sp.iv(FORMULA, data=card, method="gmm", **CASES[key])
    diag = res.diagnostics
    assert diag["Hansen J statistic"] == pytest.approx(ref[key]["J"], rel=RTOL)
    assert diag["Hansen J p-value"] == pytest.approx(ref[key]["J_p"], rel=RTOL)
    assert diag["Hansen J df"] == 1


@pytest.mark.parametrize("key", ["robust", "cluster"])
def test_without_small_the_p_value_is_from_the_normal(ref, card, key):
    res = sp.iv(FORMULA, data=card, method="gmm", **CASES[key])
    z = ref[key]["b"] / ref[key]["se"]
    assert float(res.pvalues["educ"]) == pytest.approx(
        2 * stats.norm.sf(abs(z)), rel=1e-7
    )
    assert res.model_info["small"] is False


def test_small_changes_the_variance_by_the_documented_factor(ref, card):
    n, k, g = 3010, 7, 50
    robust = (ref["robust_small"]["se"] / ref["robust"]["se"]) ** 2
    assert robust == pytest.approx(n / (n - k), rel=1e-9)
    cluster = (ref["cluster_small"]["se"] / ref["cluster"]["se"]) ** 2
    assert cluster == pytest.approx(g / (g - 1) * (n - 1) / (n - k), rel=1e-9)
    # the unadjusted weight matrix takes the same N / (N - K), and sp.iv
    # applies it there too (it did not until 1.40.1)
    gap = (ref["unadjusted_small"]["se"] / ref["unadjusted"]["se"]) ** 2
    assert gap == pytest.approx(n / (n - k), rel=1e-9)
    small = sp.iv(FORMULA, data=card, method="gmm", gmm_vcov="efficient")
    large = sp.iv(FORMULA, data=card, method="gmm", gmm_vcov="efficient", small=False)
    ratio = float(small.std_errors["educ"] / large.std_errors["educ"]) ** 2
    assert ratio == pytest.approx(n / (n - k), rel=1e-12)


def test_default_is_the_sandwich_around_the_unadjusted_weight_matrix(card):
    """``ivregress gmm, wmatrix(unadjusted) vce(robust)``: 0.04851397500143
    without ``small``; ``small`` multiplies the variance by N / (N - K)."""
    n, k = 3010, 7
    res = sp.iv(FORMULA, data=card, method="gmm", small=False)
    assert float(res.std_errors["educ"]) == pytest.approx(0.04851397500143, rel=1e-9)
    res = sp.iv(FORMULA, data=card, method="gmm")
    assert float(res.std_errors["educ"]) == pytest.approx(
        0.04851397500143 * np.sqrt(n / (n - k)), rel=1e-9
    )


# --------------------------------------------------------------------- #
# The same commands through the translator
# --------------------------------------------------------------------- #

STATA = "ivregress gmm lwage exper expersq black south smsa (educ = nearc4 nearc2)"

#: option string -> key in the reference file, or the (b, se) Stata 18
#: printed for a combination the generator does not loop over.
TRANSLATED = {
    "": "robust",
    ", robust": "robust",
    ", small": "robust_small",
    ", wmatrix(cluster cl)": "cluster",
    ", vce(cluster cl)": "cluster",
    ", wmatrix(cluster cl) small": "cluster_small",
    ", wmatrix(unadjusted)": "unadjusted",
    ", vce(unadjusted)": (0.158838655315, 0.048498277691),
    ", wmatrix(unadjusted) vce(robust)": (0.16084872834863, 0.04851397500143),
    ", wmatrix(unadjusted) small": "unadjusted_small",
}


@pytest.mark.parametrize("options", sorted(TRANSLATED))
def test_translated_command_reproduces_stata(ref, card, options):
    import warnings

    out = sp.from_stata(STATA + options)
    assert out["untranslated_options"] == [], out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata(STATA + options, data=card)
    want = TRANSLATED[options]
    b, se = (ref[want]["b"], ref[want]["se"]) if isinstance(want, str) else want
    assert float(res.params["educ"]) == pytest.approx(b, rel=1e-8)
    assert float(res.std_errors["educ"]) == pytest.approx(se, rel=1e-8)


@pytest.mark.parametrize(
    "options,lost",
    [
        (", wmatrix(cluster cl) vce(robust)", "vce"),
        (", wmatrix(hac bartlett 2)", "wmatrix"),
        (", igmm", "igmm"),
    ],
)
def test_combinations_without_a_counterpart_are_reported(card, options, lost):
    out = sp.from_stata(STATA + options)
    assert lost in out["untranslated_options"], out
    with pytest.raises(Exception, match="(?i)untranslated|not translated|" + lost):
        sp.stata(STATA + options, data=card)
