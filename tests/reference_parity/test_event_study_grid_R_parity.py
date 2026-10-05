"""``sp.event_study`` across its common options, against ``fixest``.

A 32-cell grid on one panel: adoption (single date, staggered) x window
((-4, 4), (-3, 5)) x reference period (-1, -2) x covariate (none, one
time-varying) x weights (none, unit-level). Each cell is a two-way
fixed-effects regression on event-time dummies clustered on the unit,
fitted by ``fixest::feols`` in ``_fixtures/_generate_event_study_grid_R.R``.

Four cells are registered in ``sp.validation_scope``: a single adoption
date at the default window and reference period, with and without the
covariate and the weights. The rest are checked here and deliberately not
registered. On a staggered panel this regression is the contaminated
estimator that the heterogeneity-robust methods replace, and agreeing
with a reference there shows the arithmetic is right, not that the
estimand is. And one other window and one other reference period are not
evidence for every other one, which is what the map's ``other`` value
would claim.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"

#: Worst of the 32 x (up to 8) x 2 comparisons, measured 2026-10-05:
#: 1.9e-12 on an estimate, 9.0e-14 on a standard error.
RTOL = 1e-10


@pytest.fixture(scope="module")
def ref():
    payload = json.loads((_FIX / "event_study_grid_R.json").read_text(encoding="utf-8"))
    return {k: v for k, v in payload.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def panel():
    d = pd.read_csv(_FIX / "event_study_grid.csv")
    d["treat_time"] = d["g"].where(d["g"] > 0, np.nan)
    return d


def _fit(panel, spec):
    d = panel[panel["g"].isin([0, 7])] if spec["adoption"] == "single" else panel
    kw = {}
    if spec["covariates"] == "x":
        kw["covariates"] = ["x"]
    if spec["weights"] == "w":
        kw["weights"] = "w"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.event_study(
            d,
            y="y",
            treat_time="treat_time",
            time="time",
            unit="unit",
            window=tuple(spec["window"]),
            ref_period=spec["ref"],
            **kw,
        )


def test_grid_is_the_declared_one(ref):
    assert len(ref) == 32
    assert {(s["adoption"], tuple(s["window"]), s["ref"]) for s in ref.values()} == {
        (a, w, r)
        for a in ("single", "staggered")
        for w in ((-4, 4), (-3, 5))
        for r in (-1, -2)
    }


@pytest.mark.parametrize("adoption", ["single", "staggered"])
def test_every_event_time_coefficient_and_se_matches_fixest(ref, panel, adoption):
    failures = []
    n_checked = 0
    for key, spec in ref.items():
        if spec["adoption"] != adoption:
            continue
        res = _fit(panel, spec)
        es = res.model_info["event_study"].set_index("relative_time")
        want = {int(k): v for k, v in spec["coefs"].items()}
        # the same event times, the reference period omitted on both sides
        assert set(es.index.astype(int)) - {spec["ref"]} == set(want), key
        for k, v in want.items():
            n_checked += 1
            for name in ("estimate", "se"):
                got = float(es.loc[k, name])
                if got != pytest.approx(v[name], rel=RTOL):
                    failures.append(f"{key} t={k} {name}: {got!r} vs {v[name]!r}")
    assert n_checked == 16 * 8
    assert not failures, "\n".join(failures[:10])


def test_weights_and_covariates_change_the_answer(ref):
    """The grid would be vacuous if an option were silently ignored."""
    base = ref["single|w-4_4|ref-1|cov_none|wt_none"]["coefs"]
    for other in (
        "single|w-4_4|ref-1|cov_x|wt_none",
        "single|w-4_4|ref-1|cov_none|wt_w",
        "single|w-4_4|ref-2|cov_none|wt_none",
    ):
        assert abs(ref[other]["coefs"]["0"]["estimate"] - base["0"]["estimate"]) > 1e-4
    # a window changes the coefficients at its binned endpoints, not inside
    narrow = ref["single|w-3_5|ref-1|cov_none|wt_none"]["coefs"]
    assert abs(narrow["-3"]["estimate"] - base["-3"]["estimate"]) > 1e-4
    assert narrow["0"]["estimate"] == pytest.approx(base["0"]["estimate"], rel=1e-10)
