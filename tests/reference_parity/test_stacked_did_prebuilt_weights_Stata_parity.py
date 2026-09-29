"""``sp.stacked_did`` on a pre-built stack, with weights, vs Stata ``reghdfe``.

Two gaps from the Minimum Wages replication (QJE 2019): ``stacked_did`` had
no ``weights=`` (Cengiz et al. weight by state population), and it only
accepted a first-treatment panel -- a stack built with a clean-control rule
had to be passed as a panel, where it was stacked again and the never-treated
controls reused in every sub-experiment. It now takes ``weights=`` and a
pre-built stack (``event_id=``, ``treated=``, ``event_time=``), and fits by
the HDFE kernel, so the standard errors follow ``reghdfe`` (they used to
differ by a constant degrees-of-freedom factor).

Reference: Stata 18, ``reghdfe y D_* [aw=pop], absorb(id#event year#event)
cluster(id)`` on ``_fixtures/stacked_did_prebuilt.csv`` (the CDLZ stack of
``stacked_did_panel.csv`` over [-3, 3] with never- and later-treated controls
and a population weight), from ``_generate_stacked_did_prebuilt_Stata.do``.
Coefficients, SEs and the post-period average are held to 1e-9 (observed
~1e-14 with the CSV imported as double on the Stata side).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "stacked_did_prebuilt_Stata.json").read_text(encoding="utf-8"))
NAMES = {-3: "m3", -2: "m2", 0: "p0", 1: "p1", 2: "p2", 3: "p3"}


@pytest.fixture(scope="module")
def stack():
    return pd.read_csv(_FIX / "stacked_did_prebuilt.csv")


def _fit(stack, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stacked_did(
            stack,
            y="y",
            group="id",
            time="year",
            event_id="event",
            treated="treated",
            event_time="rel",
            window=(-3, 3),
            **kw,
        )


@pytest.mark.parametrize("w", ["none", "pop"])
def test_prebuilt_stack_matches_reghdfe(stack, w):
    ref = R[w]
    r = _fit(stack, weights=None if w == "none" else "pop")
    es = r.model_info["event_study"].set_index("relative_time")
    assert r.n_obs == ref["N"]
    for k, nm in NAMES.items():
        assert es.loc[k, "att"] == pytest.approx(ref[f"b_{nm}"], rel=1e-9, abs=1e-10)
        assert es.loc[k, "se"] == pytest.approx(ref[f"se_{nm}"], rel=1e-9)
    assert r.estimate == pytest.approx(ref["att"], rel=1e-9)
    assert r.se == pytest.approx(ref["att_se"], rel=1e-9)


def test_prebuilt_equals_the_stack_it_would_build(stack):
    panel = pd.read_csv(_FIX / "stacked_did_panel.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        auto = sp.stacked_did(
            panel,
            y="y",
            group="id",
            time="year",
            first_treat="first_treat",
            window=(-3, 3),
            never_treated_only=False,
        )
    pre = _fit(stack)
    assert pre.estimate == pytest.approx(auto.estimate, abs=1e-12)
    assert pre.se == pytest.approx(auto.se, abs=1e-12)
    assert pre.model_info["prebuilt_stack"] is True


def test_prebuilt_needs_all_three_columns(stack):
    with pytest.raises(ValueError, match="event_time"):
        sp.stacked_did(
            stack, y="y", group="id", time="year", event_id="event", treated="treated"
        )
