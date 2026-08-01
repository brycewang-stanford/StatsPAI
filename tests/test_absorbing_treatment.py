"""Non-absorbing (reverting) treatment must not be silently mishandled.

Cohort-based DiD estimators — Callaway-Sant'Anna, Sun-Abraham,
Borusyak-Jaravel-Spiess, Wooldridge ETWFE, stacked DiD — represent treatment
by the period a unit is *first* treated. That is lossless only if treatment is
absorbing. When it reverts, the reversal is discarded and post-reversal
periods are treated as still-treated, biasing the estimate toward zero.

The bias is not subtle: on the panel built below (a third of units switch on
at t=4 and back off at t=7, true ATT 1.5) ``sp.callaway_santanna`` returns
about 0.71 — a 53% error, with no warning, because it never sees the
time-varying indicator.

These tests pin (a) that ``sp.check_absorbing`` detects the reversal, and
(b) that ``sp.recommend``, which *does* see the indicator, stops routing such
data to a cohort estimator.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_ATT = 1.5


def _panel(reverting: bool, seed: int = 3) -> pd.DataFrame:
    """150 units x 10 periods; a third treated, optionally reverting."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(150):
        ui = rng.normal(0, 0.5)
        for t in range(1, 11):
            if reverting:
                on = u % 3 == 0 and 4 <= t < 7
            else:
                on = u % 3 == 0 and t >= 4
            rows.append(
                {
                    "i": u,
                    "t": t,
                    "d": int(on),
                    "y": ui + 0.1 * t + TRUE_ATT * int(on) + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sp.check_absorbing
# ---------------------------------------------------------------------------


def test_detects_reversal():
    chk = sp.check_absorbing(_panel(True), unit="i", time="t", treat="d")
    assert chk.is_absorbing is False
    assert chk.n_reverting_units == 50
    assert chk.n_units == 150
    assert chk.share_reverting == pytest.approx(1 / 3, abs=1e-9)
    assert chk.first_reversal_period == 7
    assert "NOT absorbing" in chk.summary()


def test_absorbing_panel_is_clean():
    chk = sp.check_absorbing(_panel(False), unit="i", time="t", treat="d")
    assert chk.is_absorbing is True
    assert chk.n_reverting_units == 0
    assert chk.n_reversals == 0
    assert "no reversals" in chk.summary()


def test_never_treated_only_is_absorbing():
    df = pd.DataFrame(
        [{"i": u, "t": t, "d": 0} for u in range(10) for t in range(1, 5)]
    )
    assert sp.check_absorbing(df, "i", "t", "d").is_absorbing is True


def test_strict_mode_raises():
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="NOT absorbing"):
        sp.check_absorbing(_panel(True), unit="i", time="t", treat="d", strict=True)


def test_strict_mode_passes_on_absorbing():
    chk = sp.check_absorbing(_panel(False), unit="i", time="t", treat="d", strict=True)
    assert chk.is_absorbing is True


def test_missing_column_fails_loudly():
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.check_absorbing(_panel(True), "i", "t", "nope")


def test_non_numeric_treatment_fails_loudly():
    from statspai.exceptions import MethodIncompatibility

    df = _panel(True)
    df["d"] = df["d"].map({0: "off", 1: "on"})
    with pytest.raises(MethodIncompatibility, match="not numeric"):
        sp.check_absorbing(df, "i", "t", "d")


def test_unsorted_input_is_handled():
    """Reversal detection must not depend on row order."""
    shuffled = _panel(True).sample(frac=1.0, random_state=0)
    chk = sp.check_absorbing(shuffled, "i", "t", "d")
    assert chk.n_reverting_units == 50


# ---------------------------------------------------------------------------
# The bias this guards against, and the routing fix
# ---------------------------------------------------------------------------


def test_cohort_estimator_is_badly_biased_under_reversal():
    """Documents *why* the guard exists.

    Callaway-Sant'Anna cannot see the reversal from (y, g, t, i), so it cannot
    warn on its own — which is exactly why the check belongs upstream.
    """
    df = _panel(True)
    first = df[df.d == 1].groupby("i")["t"].min().rename("g")
    df = df.merge(first, on="i", how="left")
    df["g"] = df["g"].fillna(0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.callaway_santanna(df, y="y", g="g", t="t", i="i")

    assert abs(res.estimate - TRUE_ATT) > 0.5, (
        "the reversal bias vanished; if a cohort estimator learned to handle "
        "reverting treatment, this guard and the recommend routing should be "
        "revisited"
    )


def test_recommend_routes_away_from_cohort_estimators_on_reversal():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = sp.recommend(
            _panel(True), y="y", treatment="d", id="i", time="t", design="did"
        )

    assert rec.recommendations[0]["function"] == "did_multiplegt"
    assert any("NOT absorbing" in w for w in rec.warnings)


def test_recommend_still_prefers_callaway_when_absorbing():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rec = sp.recommend(
            _panel(False), y="y", treatment="d", id="i", time="t", design="did"
        )

    assert rec.recommendations[0]["function"] == "callaway_santanna"
    assert not any("absorbing" in w for w in rec.warnings)


def test_recommended_reversal_estimator_is_stable_not_experimental():
    """The suggested alternative must survive the default agent-safe filter.

    ``did_multiplegt_dyn`` is registered experimental, so recommending it here
    would be silently dropped and leave the user with nothing usable.
    """
    from statspai.registry import _REGISTRY

    spec = _REGISTRY.get("did_multiplegt")
    assert spec is not None
    assert getattr(spec, "stability", None) == "stable"


# ---------------------------------------------------------------------------
# `robust=` is boolean across the DiD family, a string HC-type elsewhere.
# ---------------------------------------------------------------------------


def _two_period_panel(n: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "i": np.repeat(np.arange(n // 2), 2),
            "t": np.tile([0, 1], n // 2),
            "treat": np.repeat(rng.integers(0, 2, n // 2), 2),
            "sub": np.repeat(rng.integers(0, 2, n // 2), 2),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 2.0 * df["treat"] * df["t"] + df["x"] + rng.normal(size=n)
    return df


_ROBUST_CALLS = {
    "ddd": lambda f, df, v: f(
        df, y="y", treat="treat", time="t", subgroup="sub", robust=v
    ),
    "did_2x2": lambda f, df, v: f(df, y="y", treat="treat", time="t", robust=v),
    "did_analysis": lambda f, df, v: f(df, y="y", treat="treat", time="t", robust=v),
}


@pytest.mark.parametrize("fn", sorted(_ROBUST_CALLS))
@pytest.mark.parametrize("bad", ["HC1", "HC3", "cluster", 1, 0, None])
def test_did_family_robust_rejects_non_boolean(fn, bad):
    """``robust="cluster"`` used to return unclustered SEs, silently.

    Clustering lives on a separate ``cluster=`` argument across this family,
    so the string was read as truthy and the request simply vanished. The
    numbers returned were correct for what was computed and were not what
    the caller asked for — the worst shape a defect can take, because
    nothing about the output looks wrong.

    ``sp.did`` has rejected non-booleans for a while; these three did not,
    which made the guarantee inconsistent *within one module*.
    """
    with pytest.raises(Exception, match="(?i)boolean"):
        _ROBUST_CALLS[fn](getattr(sp, fn), _two_period_panel(), bad)


@pytest.mark.parametrize("fn", sorted(_ROBUST_CALLS))
@pytest.mark.parametrize("good", [True, False, np.True_, np.False_])
def test_did_family_robust_accepts_booleans(fn, good):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _ROBUST_CALLS[fn](getattr(sp, fn), _two_period_panel(), good)


def test_require_bool_has_exactly_one_implementation():
    """§4: shared primitives live in ``_core``, not copied per file.

    Two byte-identical private copies existed in ``did/__init__.py`` and
    ``did/callaway_santanna.py``; both now alias the ``_core`` helper.
    """
    import importlib

    # ``statspai.did`` the attribute is the *function*; the subpackage has to
    # be fetched through importlib or this silently asserts nothing.
    did_pkg = importlib.import_module("statspai.did")
    core = importlib.import_module("statspai.did._core")
    cs = importlib.import_module("statspai.did.callaway_santanna")

    assert did_pkg._require_bool is core.require_bool
    assert cs._require_bool is core.require_bool
