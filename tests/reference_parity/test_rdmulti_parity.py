"""Reference parity: ``sp.rdmc`` vs R rdmulti 2.0.0 (WP-5).

``sp.rdmc`` claimed to be "equivalent to ``rdmulti::rdmc()``" but could not
express that function's central argument. R's ``rdmc`` takes ``C``, a
**per-unit cutoff**: in the Cattaneo, Titiunik, Vazquez-Bare & Keele (2016)
multi-cutoff design each unit faces its own threshold, and cutoff ``c``'s
effect is identified only from the units assigned to ``c``. StatsPAI
offered only a shared-running-variable estimator in which every unit enters
every cutoff's local regression.

Those are different estimators, and the difference is not small. On the
fixture below -- effects of **2.0 / 5.0 / -3.0** at cutoffs -10 / 0 / 15 --
the shared path returned **0.22 / 0.51 / 0.66**: units belonging to one
cutoff were pooled into the others' windows, averaging the three effects
into noise and destroying even the sign of the third.

``cutoff_var=`` now selects the R-equivalent estimator, delegating each
cutoff to :func:`statspai.rdrobust` so it inherits the CCT cascade. Every
quantity matches R to <= 2.4e-08 (bandwidths) or ~1e-13 (everything else).

A note on R's field names, established by comparison rather than assumed:
``rdmc``'s ``B``/``V`` hold the **bias-corrected** coefficient and its
**variance** (``V`` is a variance -- ``r$CI`` half-widths are
``1.96*sqrt(V)``), while ``Coefs`` holds the **conventional** estimate. The
printed table pairs the conventional point estimate with the robust CI,
exactly as ``rdrobust`` does. Reading ``B`` as "the" coefficient and ``V``
as an SE would have produced a fixture that was wrong in two ways at once.

The design uses three deliberately *different* effects, including a sign
flip, so the suite can distinguish a per-cutoff estimator from one that
pools. Had all three been equal, the broken implementation would have
passed.
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

RTOL = 1e-9
# Bandwidths come back through model_info, which rounds to six decimals.
RTOL_BW = 1e-5


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rdmulti_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdmulti_R.R to build rdmulti_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def design():
    return pd.read_csv(_FIX / "rdmulti_design.csv")


@pytest.fixture(scope="module")
def fit(design):
    return sp.rdmc(design, y="y", x="x", cutoff_var="cvar")


# ── the fixture must be able to tell right from wrong ──────────────────── #


def test_design_is_discriminating(rjson):
    """The three cutoffs must carry genuinely different effects.

    If they were equal, an estimator that ignored the cutoff assignment
    entirely would still reproduce them, and this whole file would be
    vacuous. The third effect is negative on purpose.
    """
    taus = rjson["true_tau"]
    assert len(set(taus)) == len(taus), f"effects are not distinct: {taus}"
    assert min(taus) < 0 < max(taus), (
        f"no sign flip among {taus}; a pooled estimator could still land "
        "close to every cutoff by averaging"
    )
    est = rjson["coefs"]
    for got, want in zip(est, taus):
        assert abs(got - want) < 0.6, (
            f"R itself does not recover the design ({got} vs {want}); the "
            "fixture is too noisy to test anything"
        )


# ── per-cutoff parity ───────────────────────────────────────────────────── #


def test_cutoffs_are_recovered_from_the_assignment(fit, rjson):
    assert [cr["cutoff"] for cr in fit.cutoff_results] == rjson["cutoffs"]
    assert fit.n_cutoffs == len(rjson["cutoffs"])


@pytest.mark.parametrize("i", [0, 1, 2])
def test_conventional_coefficient_matches_r(fit, rjson, i):
    got = fit.cutoff_results[i]["estimate"]
    assert got == pytest.approx(rjson["coefs"][i], rel=RTOL)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_bias_corrected_coefficient_matches_r(fit, rjson, i):
    got = fit.cutoff_results[i]["estimate_robust"]
    assert got == pytest.approx(rjson["coefs_rb"][i], rel=RTOL)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_robust_standard_error_matches_r(fit, rjson, i):
    """``V`` is a variance, so the comparison is against its square root."""
    got = fit.cutoff_results[i]["se_robust"]
    assert got == pytest.approx(np.sqrt(rjson["var_rb"][i]), rel=RTOL)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_bandwidth_matches_r(fit, rjson, i):
    """Each cutoff gets its OWN CCT bandwidth, as in rdmulti.

    The old path used one Silverman rule-of-thumb bandwidth for all three.
    """
    got = fit.cutoff_results[i]["bandwidth"]
    assert got == pytest.approx(rjson["h_left"][i], rel=RTOL_BW)


@pytest.mark.parametrize("i", [0, 1, 2])
def test_effective_sample_size_matches_r(fit, rjson, i):
    """Right answer on the right window."""
    want = int(rjson["Nh_left"][i]) + int(rjson["Nh_right"][i])
    assert fit.cutoff_results[i]["n"] == want


def test_bandwidths_differ_across_cutoffs(fit):
    """Property test: a single shared bandwidth would be the old bug."""
    hs = [cr["bandwidth"] for cr in fit.cutoff_results]
    assert len(set(np.round(hs, 9))) == len(hs), f"shared bandwidth: {hs}"


# ── pooling ─────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("i", [0, 1, 2])
def test_pooling_weights_match_r(fit, rjson, i):
    """rdmulti weights by effective sample size, not inverse variance."""
    assert fit.cutoff_results[i]["weight"] == pytest.approx(
        rjson["weights"][i], rel=RTOL
    )


def test_weighted_estimate_matches_r(fit, rjson):
    assert fit.pooled_estimate == pytest.approx(rjson["weighted_coef"], rel=RTOL)


def test_weights_sum_to_one(fit):
    assert sum(cr["weight"] for cr in fit.cutoff_results) == pytest.approx(1.0)


# ── the defect this file exists to prevent ─────────────────────────────── #


def test_shared_running_variable_path_is_a_different_estimator(design, rjson):
    """The two modes must not be confused for one another.

    Documented rather than silently "fixed": the shared-running-variable
    estimator is a legitimate design, it is simply not rdmulti's. This test
    pins the fact that it gives a materially different answer here, so that
    nobody re-points ``cutoff_var=`` at it on the assumption they agree.
    """
    shared = sp.rdmc(design, y="y", x="x", cutoffs=[-10, 0, 15])
    for cr, want in zip(shared.cutoff_results, rjson["true_tau"]):
        assert abs(cr["estimate"] - want) > 0.5, (
            "the shared-running-variable path unexpectedly recovers the "
            "unit-specific design; if that is now true, the warning in "
            "rdmc's docstring is stale and should be revisited"
        )


def test_rdmc_requires_one_of_the_two_modes(design):
    with pytest.raises(ValueError, match="cutoff_var"):
        sp.rdmc(design, y="y", x="x")


def test_dataset_matches_the_r_side(rjson, design):
    assert len(design) == int(rjson["_meta"]["n"])
