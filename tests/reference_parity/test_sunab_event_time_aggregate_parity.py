"""``sp.sun_abraham``'s default overall aggregate vs ``fixest::sunab``.

Track A module 05 pins the cohort-size-weighted ATT
(``aggregation="fixest_att"``) and every per-period effect. The headline
before 1.41 was ``aggregation="event_time"`` (since 1.41 it is
``"fixest_att"``; the event-time aggregate is still reported in
``model_info`` and selected by passing it): the equal-weighted average of
the post-treatment event-time effects. No row compared that aggregate with
a reference, so ``sp.validation_scope`` could not honestly call such a fit
covered. The fits below name ``aggregation="event_time"`` and the pre-1.41
``crit_dist="normal"`` explicitly.

Reference: the same linear combination of ``fixest``'s own event-time
coefficients, with SE ``sqrt(w' V w)`` from ``fixest``'s covariance
(cohort shares fixed). The generator first requires its aggregation map
to reproduce ``fixest``'s aggregated coefficients (1e-12) and SEs (1e-10).
Fixture: ``_fixtures/sunab_event_time_R.json`` from
``_generate_sunab_event_time_R.R`` (fixest 0.14.0), on module 05's bytes.
Observed agreement: 8e-12 relative on both.

The headline aggregate's SE treats cohort shares as fixed on both
``share_variance`` settings (the switch changes the per-period SEs only),
which the second test pins.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_ROOT = pathlib.Path(__file__).resolve().parents[2]
REF = json.loads(
    (pathlib.Path(__file__).parent / "_fixtures" / "sunab_event_time_R.json").read_text(
        encoding="utf-8"
    )
)
DATA = pd.read_csv(_ROOT / "tests" / "r_parity" / "data" / "05_sunab.csv")
RTOL = 1e-8


def _fit(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.sun_abraham(
            DATA,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            aggregation="event_time",
            crit_dist="normal",
            **kw,
        )


def test_event_time_aggregate_matches_fixest_linear_combination():
    fit = _fit()
    assert fit.model_info["summary_aggregation"] == "event_time"
    assert float(fit.estimate) == pytest.approx(REF["att_event_time"], rel=RTOL)
    assert float(fit.se) == pytest.approx(REF["se_event_time"], rel=RTOL)


@pytest.mark.parametrize("share_variance", [True, False])
def test_headline_aggregate_se_does_not_depend_on_share_variance(share_variance):
    fit = _fit(share_variance=share_variance)
    assert float(fit.se) == pytest.approx(REF["se_event_time"], rel=RTOL)


def test_scope_map_covers_the_event_time_fit():
    scope = sp.validation_scope(_fit())
    assert scope["configuration"]["aggregation"] == "event_time"
    assert scope["status"] == "covered"
