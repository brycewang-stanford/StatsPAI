"""Reference parity: the AIPW *operator* vs ``grf``, forest held fixed.

What this pins, and why it is separate from ``test_grf_parity.py``
------------------------------------------------------------------
``test_grf_parity.py`` and Track A module 13 compare ``sp.causal_forest``
against ``grf::causal_forest`` end to end. Two independently grown
forests never share their CATE predictions, so that comparison can only
be graded against combined Monte Carlo error, and the AIPW *standard
error* — a function of the forest's own predictions — carries a 50%
relative band that is too wide to constitute validation.

The estimator factorises into two pieces with very different
verifiability:

1. **The forest.** Not pinnable across implementations. Its calibration
   is evidenced by the Track B coverage sweep, not by a relative band.
2. **The AIPW operator.** The closed-form map from
   ``(Y, W, tau.hat, Y.hat, W.hat)`` to the score vector, point estimate
   and standard error. This *is* exactly pinnable — and is what this
   module does.

The fixture ``grf_scores_R.json`` carries ``grf``'s own forest outputs,
its ``grf::get_scores()`` vector, and its reported ATE/ATT. Feeding
StatsPAI's operator ``grf``'s forest outputs must reproduce all of them
to the floating-point floor. Anything left over in the end-to-end
comparison is then attributable to the forest alone, which is exactly
the claim the module-13 disclosure note makes.

Regenerate with::

    Rscript tests/reference_parity/_fixtures/_generate_grf_scores.R

References
----------
[@athey2019generalized], [@robins1994estimation]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from statspai.forest.forest_inference import aipw_scores, grf_att_atc

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "grf_scores_R.json"

pytestmark = pytest.mark.skipif(
    not _FIXTURE.exists(), reason="grf_scores_R.json fixture is not materialized"
)


@pytest.fixture(scope="module")
def grf():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def inputs(grf):
    """grf's own forest outputs, the inputs the operator consumes."""
    return dict(
        tau=np.asarray(grf["tau_hat"], dtype=float),
        T=np.asarray(grf["W"], dtype=float),
        e_hat=np.asarray(grf["w_hat"], dtype=float),
        m_hat=np.asarray(grf["y_hat"], dtype=float),
        Y=np.asarray(grf["y"], dtype=float),
    )


def test_ate_scores_match_grf_get_scores_elementwise(grf, inputs):
    """Per-unit Gamma_i must equal ``grf::get_scores`` unit by unit.

    Matching only in the mean would leave the possibility of two
    different score vectors that happen to average alike.
    """
    psi = aipw_scores(target="all", **inputs)
    reference = np.asarray(grf["scores"], dtype=float)
    assert psi.shape == reference.shape
    np.testing.assert_allclose(psi, reference, rtol=0, atol=1e-12)


def test_ate_point_and_se_match_grf(grf, inputs):
    """mean(Gamma) and sd(Gamma)/sqrt(n) are grf's estimate and std.err."""
    psi = aipw_scores(target="all", **inputs)
    n = psi.size
    assert float(psi.mean()) == pytest.approx(grf["ate"]["estimate"], rel=1e-12)
    assert float(psi.std(ddof=1) / np.sqrt(n)) == pytest.approx(
        grf["ate"]["se"], rel=1e-12
    )


def test_att_point_and_se_match_grf(grf, inputs):
    """ATT uses grf's plug-in + Hajek-normalised-correction decomposition.

    Dividing a single Robins score by ``p1`` — StatsPAI's pre-v1.21
    route — agrees on the point estimate but inflates the standard error
    by about 12% on this fixture, with the forest held fixed.
    """
    estimate, se, dr = grf_att_atc(target="treated", **inputs)
    assert estimate == pytest.approx(grf["att"]["estimate"], rel=1e-12)
    assert se == pytest.approx(grf["att"]["se"], rel=1e-12)
    assert dr.shape == inputs["Y"].shape


def test_att_decomposition_reproduces_the_reported_estimate(grf, inputs):
    """The plug-in and correction terms must actually sum to the estimate."""
    estimate, _, dr = grf_att_atc(target="treated", **inputs)
    treated = inputs["T"] == 1
    plug_in = float(inputs["tau"][treated].mean())
    assert plug_in + float(dr.mean()) == pytest.approx(estimate, rel=1e-14)


def test_atc_is_the_mirror_image_of_att(inputs):
    """ATC must swap the arms, not merely reuse the ATT weighting."""
    att, att_se, _ = grf_att_atc(target="treated", **inputs)
    atc, atc_se, _ = grf_att_atc(target="control", **inputs)
    assert np.isfinite([att, att_se, atc, atc_se]).all()
    assert att != atc
    # ATC plug-in anchors on the control arm's CATE mean.
    control = inputs["T"] == 0
    assert atc == pytest.approx(
        float(inputs["tau"][control].mean())
        + float(grf_att_atc(target="control", **inputs)[2].mean()),
        rel=1e-14,
    )


def test_operator_rejects_unsupported_target(inputs):
    """ATT is not a single influence function; saying so is the contract."""
    with pytest.raises(Exception, match="target='all'"):
        aipw_scores(target="treated", **inputs)


def test_operator_rejects_ragged_inputs(inputs):
    bad = dict(inputs)
    bad["Y"] = bad["Y"][:-1]
    with pytest.raises(Exception, match="same length"):
        aipw_scores(target="all", **bad)


def test_fixture_provenance(grf):
    """The fixture must record the grf build that produced the numbers."""
    meta = grf["meta"]
    assert meta["num_trees"] == 2000
    assert meta["seed"] == 42
    assert meta["grf_version"]
    assert grf["n_obs"] == len(grf["scores"])
