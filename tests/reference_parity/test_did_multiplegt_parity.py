"""Reference parity: ``sp.did_multiplegt`` vs ``DIDmultiplegt`` 0.1.4.

The dCDH 2020 DID_M estimator was carried as having no cross-language
reference, on the grounds that the CRAN package was broken. That was half
right. The 2.x rewrite routes the classic estimator through ``mode="old"``
and that path returns ``NaN`` even on the package's own bundled
``wagepan_mgt`` example — but the archived **0.1.4** works, and it is the
release the estimator originally shipped in.

Reference generation (R 4.5.2, DIDmultiplegt 0.1.4 from the CRAN archive)::

    did_multiplegt(df = d, Y = "y", G = "id", T = "t", D = "d",
                   placebo = 1, dynamic = 1, brep = 0)

Fixture: ``_fixtures/didm_switching_panel.csv`` — 180 units x 6 periods with
treatment switching both ON and OFF, which is the design DID_M exists for and
the one that exercises the switch-off sign convention no cohort-based
estimator has.

Two defects this comparison found, both fixed and both pinned below:

* The **dynamic** effect at horizon ``l`` did not require switchers to hold
  their new treatment through the window. A unit that switched at ``t`` and
  switched again at ``t+1`` was still counted, which makes "the effect of
  having switched ``l`` periods ago" undefined. On this fixture it moved the
  horizon-1 effect from 0.9974 to 1.2146, and the switcher count from 175 to
  140 — the count is what makes the diagnosis unambiguous.
* The **placebo** ignored the same stability condition over the pre-window.

And one thing that turned out NOT to be a defect. The placebo's sign differs
between dCDH's own two implementations: on ``did::mpdta`` both report
``|placebo_1| = 0.024269`` and their three effects agree to six decimals, but
Stata's ``did_multiplegt_old`` reports ``+0.024269`` and ``DIDmultiplegt``
0.1.4 reports ``-0.024269``. An earlier read of this treated the R sign as
correct and flipped the default; that would have silently moved every
Stata-matching placebo StatsPAI had ever reported. ``placebo_sign`` selects
the convention instead, the default keeps Stata's, and the tests below pin
both.

References
----------
de Chaisemartin, C. and D'Haultfoeuille, X. (2020). "Two-Way Fixed Effects
Estimators with Heterogeneous Treatment Effects." *American Economic Review*,
110(9), 2964-2996. [@dechaisemartin2020two]
"""

from __future__ import annotations

import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "didm_switching_panel.csv"

# DIDmultiplegt 0.1.4, placebo = 1, dynamic = 1, brep = 0.
R_EFFECT = 1.187842536315
R_DYNAMIC_1 = 0.995455123564
R_PLACEBO_1 = 0.032431647759


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    return pd.read_csv(_FIXTURE)


@pytest.fixture(scope="module")
def fit(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_multiplegt(
            panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            placebo=1,
            dynamic=1,
            n_boot=0,
            placebo_sign="r",
        )


def _es(fit):
    return fit.model_info["event_study"].set_index("relative_time")["att"]


def test_static_effect_matches_reference(fit):
    assert float(_es(fit).loc[0]) == pytest.approx(R_EFFECT, abs=1e-9)
    assert fit.estimate == pytest.approx(R_EFFECT, abs=1e-9)


def test_dynamic_effect_matches_reference(fit):
    """⚠️ Changed: switchers must hold their new treatment through the
    horizon. Without that this returned a different number."""
    assert float(_es(fit).loc[1]) == pytest.approx(R_DYNAMIC_1, abs=1e-9)


def test_placebo_matches_reference(fit):
    """⚠️ Changed: the pre-window stability condition.

    Requested with ``placebo_sign="r"`` because this compares against the R
    package; see the module docstring on why the sign is a parameter.
    """
    assert float(_es(fit).loc[-1]) == pytest.approx(R_PLACEBO_1, abs=1e-9)


def test_the_two_placebo_conventions_are_exact_negatives(panel):
    """Pins the disagreement itself, so neither side can drift unnoticed."""
    out = {}
    for convention in ("stata", "r"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[convention] = float(
                _es(
                    sp.did_multiplegt(
                        panel,
                        y="y",
                        group="id",
                        time="t",
                        treatment="d",
                        placebo=1,
                        n_boot=0,
                        placebo_sign=convention,
                    )
                ).loc[-1]
            )
    assert out["stata"] == pytest.approx(-out["r"], abs=1e-12)
    assert out["r"] == pytest.approx(R_PLACEBO_1, abs=1e-9)


def test_default_placebo_convention_is_unchanged(panel):
    """Guards the default. Flipping it would move every placebo StatsPAI has
    reported, silently, on the strength of one implementation over another."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_multiplegt(
            panel, y="y", group="id", time="t", treatment="d", placebo=1, n_boot=0
        )
    assert float(_es(res).loc[-1]) == pytest.approx(-R_PLACEBO_1, abs=1e-9)


def test_placebo_conventions_on_a_planted_pre_trend(panel):
    """A planted upward pre-trend among switchers, read under BOTH
    conventions, so the test says what each one means rather than asserting
    a sign in the abstract."""
    import numpy as np

    rng = np.random.default_rng(4)
    rows = []
    for uid in range(300):
        switches = uid < 150
        fe = rng.normal()
        for t in range(1, 6):
            d = 1 if (switches and t >= 4) else 0
            # switchers drift upward before treatment: a pre-trend
            trend = 0.5 * t if switches else 0.0
            rows.append((uid, t, d, fe + trend + rng.normal(0, 0.2)))
    df = pd.DataFrame(rows, columns=["id", "t", "d", "y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_multiplegt(
            df, y="y", group="id", time="t", treatment="d", placebo=1, n_boot=0
        )
    stata_sign = float(_es(res).loc[-1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_r = sp.did_multiplegt(
            df,
            y="y",
            group="id",
            time="t",
            treatment="d",
            placebo=1,
            n_boot=0,
            placebo_sign="r",
        )
    r_sign = float(_es(res_r).loc[-1])
    assert abs(stata_sign) > 0.2, stata_sign
    assert stata_sign == pytest.approx(-r_sign, abs=1e-12)


def test_switch_off_cells_are_used(fit):
    """DID_M's reason for existing: it handles treatment turning off.

    The detail table must carry both directions, or the fixture is not
    testing what this file says it tests.
    """
    directions = set(fit.detail["direction"])
    assert directions == {"on", "off"}
    assert (fit.detail["n_switchers"] > 0).all()


def test_dynamic_excludes_units_that_switch_again(panel):
    """The horizon-1 effect must rest on fewer switchers than horizon 0,
    because some of the horizon-0 switchers move again inside the window."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_multiplegt(
            panel, y="y", group="id", time="t", treatment="d", dynamic=1, n_boot=0
        )
    dyn = res.model_info["dynamic"]
    assert len(dyn) == 2
    assert dyn[0]["estimate"] != pytest.approx(dyn[1]["estimate"], abs=1e-6)
