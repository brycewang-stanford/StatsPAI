"""Reference parity: ``sp.staggered_rollout`` vs R ``staggered``.

Roth & Sant'Anna (2023) efficient estimation for randomised staggered
rollouts. This is a **design-based** estimator: it identifies off random
adoption *timing*, not parallel trends, so it answers a different question
from every other DiD estimator in StatsPAI.

Reference generation (R 4.5.2, ``staggered`` 1.2.2) on canonical
``did::mpdta``::

    mpdta$g <- ifelse(mpdta$first.treat == 0, Inf, mpdta$first.treat)
    staggered(df = mpdta, i = "countyreal", t = "year", g = "g",
              y = "lemp", estimand = <estimand>, beta = <NULL | 1>)

``beta = NULL`` is the efficient estimator; ``beta = 1`` is the plug-in.

.. note::
   The never-treated coding matters enormously. R's ``staggered`` requires
   ``g = Inf``; passing ``g = 0`` makes it read never-treated units as a
   cohort treated before the sample and returns −0.3704 instead of −0.0471.
   ``sp.staggered_rollout`` accepts 0 / NaN / inf and normalises, so the trap
   is unreachable through the public API — ``test_never_treated_coding_is_normalised``
   pins that.

Interpretation guard: on ``mpdta`` the design-based estimate (−0.0471) differs
from Callaway-Sant'Anna (−0.0400) because mpdta's timing is *not* randomised.
That gap is the estimand difference, not an error, and
``test_differs_from_parallel_trends_estimator`` pins it so nobody "fixes" it.

References
----------
Roth, J. and Sant'Anna, P.H.C. (2023). "Efficient Estimation for Staggered
Rollout Designs." *Journal of Political Economy Microeconomics*, 1(4),
669-709. [@roth2023efficient]
"""

from __future__ import annotations

import hashlib
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._staggered_rollout import staggered_rollout_core

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)
_MPDTA_SHA256 = "1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661"

# (estimand, efficient) -> (estimate, conservative/Neyman SE) from R staggered
R_STAGGERED = {
    ("simple", True): (-0.0470539142, 0.0116138788),
    ("simple", False): (-0.0397636256, 0.0118272142),
    ("cohort", True): (-0.0298479506, 0.0125571289),
    ("cohort", False): (-0.0304622281, 0.0125590491),
    ("calendar", True): (-0.0579882830, 0.0144374235),
    ("calendar", False): (-0.0442670835, 0.0157172229),
}


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    if not _MPDTA.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"locked mpdta fixture missing: {_MPDTA}")
    digest = hashlib.sha256(_MPDTA.read_bytes()).hexdigest()
    assert (
        digest == _MPDTA_SHA256
    ), f"mpdta fixture changed; expected {_MPDTA_SHA256}, got {digest}"
    return pd.read_csv(_MPDTA)


@pytest.mark.parametrize("key", sorted(R_STAGGERED))
def test_estimate_matches_r_staggered(mpdta, key):
    estimand, efficient = key
    est_r, _ = R_STAGGERED[key]
    core = staggered_rollout_core(
        mpdta,
        i="countyreal",
        t="year",
        g="first_treat",
        y="lemp",
        estimand=estimand,
        efficient=efficient,
    )
    assert core.estimate == pytest.approx(est_r, abs=1e-8), (
        f"{estimand}/{'efficient' if efficient else 'plugin'}: "
        f"StatsPAI {core.estimate:.10f} vs R {est_r:.10f}"
    )


@pytest.mark.parametrize("key", sorted(R_STAGGERED))
def test_conservative_se_matches_r_staggered(mpdta, key):
    estimand, efficient = key
    _, se_r = R_STAGGERED[key]
    core = staggered_rollout_core(
        mpdta,
        i="countyreal",
        t="year",
        g="first_treat",
        y="lemp",
        estimand=estimand,
        efficient=efficient,
    )
    assert core.se == pytest.approx(se_r, abs=1e-8)


def test_public_entry_point_matches_core(mpdta):
    res = sp.staggered_rollout(
        mpdta, y="lemp", i="countyreal", t="year", g="first_treat"
    )
    est_r, se_r = R_STAGGERED[("simple", True)]
    assert res.estimate == pytest.approx(est_r, abs=1e-8)
    assert res.se == pytest.approx(se_r, abs=1e-8)
    assert "random adoption timing" in res.estimand


def test_efficient_beats_the_plugin_on_precision(mpdta):
    """The whole point of the efficient weights is a smaller variance."""
    eff = staggered_rollout_core(
        mpdta, i="countyreal", t="year", g="first_treat", y="lemp", efficient=True
    )
    plug = staggered_rollout_core(
        mpdta, i="countyreal", t="year", g="first_treat", y="lemp", efficient=False
    )
    assert eff.se < plug.se
    assert not np.allclose(eff.beta, 1.0)
    assert np.allclose(plug.beta, 1.0)


def test_never_treated_coding_is_normalised(mpdta):
    """0 / NaN / inf must all mean 'never treated'.

    R's ``staggered`` silently misreads ``g = 0`` as a pre-sample cohort and
    returns −0.3704; accepting all three codings closes that trap.
    """
    zero_coded = mpdta.copy()
    inf_coded = mpdta.copy()
    inf_coded["first_treat"] = np.where(
        inf_coded["first_treat"] == 0, np.inf, inf_coded["first_treat"]
    )
    nan_coded = mpdta.copy()
    nan_coded["first_treat"] = nan_coded["first_treat"].replace(0, np.nan)

    results = [
        staggered_rollout_core(
            df, i="countyreal", t="year", g="first_treat", y="lemp"
        ).estimate
        for df in (zero_coded, inf_coded, nan_coded)
    ]
    for got in results:
        assert got == pytest.approx(R_STAGGERED[("simple", True)][0], abs=1e-8)


def test_differs_from_parallel_trends_estimator(mpdta):
    """Design-based and parallel-trends estimands genuinely differ here.

    mpdta's adoption timing is not randomised, so the two answer different
    questions. Pinned so the gap is not mistaken for a bug.
    """
    design = sp.staggered_rollout(
        mpdta, y="lemp", i="countyreal", t="year", g="first_treat"
    ).estimate
    cs = sp.callaway_santanna(
        mpdta, y="lemp", g="first_treat", t="year", i="countyreal"
    ).estimate
    assert design == pytest.approx(-0.0470539142, abs=1e-8)
    assert cs == pytest.approx(-0.0399512752, abs=1e-8)
    assert abs(design - cs) > 5e-3


def test_unbalanced_panel_fails_loudly(mpdta):
    from statspai.exceptions import DataInsufficient

    holed = mpdta.drop(index=mpdta.index[:3])
    with pytest.raises(DataInsufficient, match="balanced panel"):
        staggered_rollout_core(
            holed, i="countyreal", t="year", g="first_treat", y="lemp"
        )


def test_bad_estimand_raises(mpdta):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="estimand"):
        staggered_rollout_core(
            mpdta,
            i="countyreal",
            t="year",
            g="first_treat",
            y="lemp",
            estimand="bogus",
        )
