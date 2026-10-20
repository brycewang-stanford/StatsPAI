"""``sp.breakdown_m`` under sign / shape / SDRM restrictions: known truths.

With a near-noiseless event study (``sd = 1e-3``) the conditional confidence
set is the identified set widened by about ``z sd``, and the identified sets
have closed forms. Pre-periods ``delta_{-3}, delta_{-2}`` with the reference
``delta_{-1} = 0``, on-impact estimate ``beta_0 = 0.1``:

* linear pre-trend (0.2, 0.1): under ``Delta^SD(M)`` the extrapolated
  violation is ``delta_0 = -0.1`` give or take ``M``, so ``theta_0 = beta_0 -
  delta_0`` reaches zero at ``M = 0.2``; a positive sign restriction
  (``delta_post >= 0``) leaves that end unchanged.
* curved pre-trend (0.3, 0.1): pre-period curvature 0.1, so under
  ``Delta^SDRM(Mbar)`` ``delta_0 in [-0.1 - 0.1 Mbar, -0.1 + 0.1 Mbar]`` and
  ``theta_0`` reaches zero at ``Mbar = 2``; the largest pre-period first
  difference is 0.2, so under ``Delta^RM(Mbar)`` it reaches zero at
  ``Mbar = 0.5``, and under ``Delta^RMNB`` (``delta_post <= 0``, so
  ``theta_0 >= beta_0``) never.

The estimated breakdown lies below the truth by the sampling allowance
(a few ``sd`` of the target and of the pre-period quantities).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.core.results import CausalResult
from statspai.did.honest_did import event_study_from_moments
from statspai.exceptions import MethodIncompatibility

V = np.eye(4) * 1e-6
TIMES = [-3, -2, 0, 1]


def _linear():
    return event_study_from_moments(
        np.array([0.2, 0.1, 0.1, 0.12]), V, event_times=TIMES
    )


def _curved():
    return event_study_from_moments(
        np.array([0.3, 0.1, 0.1, 0.12]), V, event_times=TIMES
    )


def _bd(res, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.breakdown_m(res, e=0, honestdid_method="Conditional", **kw)


def test_smoothness_with_sign_restriction_recovers_the_identified_crossing():
    m = _bd(_linear(), method="smoothness", bias_direction="positive")
    # theta_0 lower end: 0.1 + 0.1 - M - z * sd(beta_0 - 2 beta_-2 + beta_-3).
    assert 0.2 - 4 * np.sqrt(6) * 1e-3 <= m <= 0.2


def test_sdrm_recovers_the_identified_crossing():
    m = _bd(_curved(), method="smoothness_relative_magnitude")
    # (0.2 - slack) / (0.1 + slack): the target and the pre-period curvature
    # are each off by a few sd(curvature) = sqrt(6) * 1e-3.
    slack = 4 * np.sqrt(6) * 1e-3
    assert (0.2 - slack) / (0.1 + slack) <= m <= 2.0


@pytest.mark.slow
def test_relative_magnitudes_with_sign_restrictions():
    curved = _curved()
    slack = 4 * np.sqrt(2) * 1e-3
    m_pb = _bd(curved, method="relative_magnitude", bias_direction="positive")
    assert (0.1 - slack) / (0.2 + slack) <= m_pb <= 0.5
    assert math.isinf(
        _bd(curved, method="relative_magnitude", bias_direction="negative")
    )


def test_default_paths_are_unchanged_by_the_new_arguments():
    """``honestdid_method=None`` with no restriction is the FLCI / C-LF value."""
    res = _linear()
    assert sp.breakdown_m(res, e=0) == sp.breakdown_m(res, e=0, honestdid_method="FLCI")


def test_flci_with_a_restriction_warns_and_ignores_it():
    res = _linear()
    with pytest.warns(UserWarning, match="does not use the sign or shape"):
        m = sp.breakdown_m(res, e=0, honestdid_method="FLCI", bias_direction="positive")
    assert m == sp.breakdown_m(res, e=0)


def test_sign_and_shape_together_are_refused():
    with pytest.raises(MethodIncompatibility, match="not both") as err:
        sp.breakdown_m(
            _linear(),
            bias_direction="positive",
            monotonicity_direction="increasing",
        )
    assert err.value.recovery_hint


def test_restricted_sets_need_the_joint_covariance():
    es = __import__("pandas").DataFrame(
        {"relative_time": [-2, 0, 1], "att": [0.0, 0.3, 0.3], "se": [0.1] * 3}
    )
    res = CausalResult(
        method="x",
        estimand="ATT",
        estimate=0.3,
        se=0.1,
        pvalue=0.0,
        ci=(0.1, 0.5),
        alpha=0.05,
        n_obs=10,
        model_info={"event_study": es},
    )
    with pytest.raises(MethodIncompatibility, match="joint covariance") as err:
        sp.breakdown_m(res, e=0, bias_direction="positive")
    assert err.value.recovery_hint


def test_empty_set_below_the_crossing_warns():
    """A curved pre-trend rejects ``Delta^SD(M)`` for M below its curvature
    (0.1): the set is empty there, which is reported. Above it
    ``delta_0 in [-0.1 - M, -0.1 + M]``, so the set first contains zero at
    ``M = 0.2``."""
    with pytest.warns(UserWarning, match="confidence set is empty"):
        m = sp.breakdown_m(_curved(), e=0, honestdid_method="Conditional")
    assert 0.2 - 4 * np.sqrt(6) * 1e-3 <= m <= 0.2
