"""sp.did_multiplegt's dynamic effects, placebos and controls against both of
dCDH's own implementations, on the same CSV bytes (point estimates).

Fixture: ``_fixtures/dcdh_dynamic_panel.csv`` (60 groups, 9 periods, 1-4
rows per cell; a count treatment ``dc`` from baselines 0 / 1 / 2 and a
binary ``db``, both switching on and off; a control ``x``; ``keep == 0``
marks the cells dropped in the unbalanced variant), from
``_generate_dcdh_dynamic_data.py``. References:

* ``dcdh_dynamic_Stata.csv``: Stata 18 ``did_multiplegt_old``, without
  ``robust_dynamic`` (tag ``_nr``, ``placebo(2)``), with it (``_rd``,
  ``dynamic(3) placebo(3)``, including ``e(effect_average)``) and with
  ``firstdiff_placebo`` added (``_fd``), from
  ``_generate_dcdh_dynamic_Stata.do``;
* ``dcdh_dynamic_R.csv``: R ``DIDmultiplegt`` 0.1.4 (``_r``, ``placebo = 2,
  dynamic = 3``), from ``_generate_dcdh_dynamic_R.R``.

Each runs on the individual rows (``indiv``) and on the cell means
(``cell``), balanced and unbalanced, with and without ``controls(x)``.

What the two references establish (black box, 2026-10-22):

* The dynamic effects differ by design. Stata computes them only with
  ``robust_dynamic`` (first-time switchers against not-yet-switched groups:
  ``robust_dynamic=True``); R 0.1.4 extends DID_M pair by pair (the
  default, which Track A module 81 certifies). They agree on a binary
  staggered treatment with one row per cell.
* Placebos with controls: R refits the slopes on each placebo's own stayers
  (default ``placebo_controls='refit'``); Stata reuses the DID_M slopes
  (``placebo_controls='main'``). Without controls they agree.
* Unbalanced panels: R takes the first difference between a group's
  consecutive *observed* periods, across a gap; Stata (and StatsPAI) leaves
  the difference over a gap missing. So every R ``unb`` row, DID_M
  included, differs from Stata. Those rows are kept in the fixture for the
  record and not asserted (did.md lists the difference); the Stata ``unb``
  rows are.

Tolerance 1e-10 relative: the same arithmetic on the same bytes (observed
worst case below 1e-13; regressions with period effects add a few ulps).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
RTOL = 1e-10
COMBOS = [
    (layout, d, ctrl, bal)
    for layout in ("indiv", "cell")
    for bal in ("bal", "unb")
    for ctrl in ("none", "x")
    for d in ("db", "dc")
]
Y = {"db": "yb", "dc": "y"}


def _ref(name):
    raw = pd.read_csv(FIX / name, skipinitialspace=True)
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    return {(t, n): float(v) for t, n, v in raw.itertuples(index=False)}


@pytest.fixture(scope="module")
def stata():
    return _ref("dcdh_dynamic_Stata.csv")


@pytest.fixture(scope="module")
def rref():
    return _ref("dcdh_dynamic_R.csv")


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(FIX / "dcdh_dynamic_panel.csv")


def _data(raw, layout, bal):
    df = raw if bal == "bal" else raw[raw["keep"] == 1]
    if layout == "cell":
        df = df.groupby(["g", "t"], as_index=False)[["y", "yb", "dc", "db", "x"]].mean()
    return df


def _fit(df, d, ctrl, **kw):
    return sp.did_multiplegt(
        df,
        y=Y[d],
        group="g",
        time="t",
        treatment=d,
        controls=["x"] if ctrl == "x" else None,
        n_boot=0,
        **kw,
    )


def _close(a, b):
    assert np.isclose(a, b, rtol=RTOL, atol=1e-13), (a, b)


def _est(r, key):
    return [x["estimate"] for x in r.model_info[key]]


@pytest.mark.parametrize("layout, d, ctrl, bal", COMBOS)
def test_robust_dynamic_matches_stata(raw, stata, layout, d, ctrl, bal):
    tag = f"{layout}_{d}_{ctrl}_{bal}_rd"
    r = _fit(
        _data(raw, layout, bal), d, ctrl, dynamic=3, placebo=3, robust_dynamic=True
    )
    _close(r.estimate, stata[(tag, "effect_0")])
    assert r.model_info["n_switchers"] == stata[(tag, "N_switchers_effect_0")]
    for row in r.model_info["dynamic"]:
        h = row["horizon"]
        _close(row["estimate"], stata[(tag, f"effect_{h}")])
        assert row["n_switchers"] == stata[(tag, f"N_switchers_effect_{h}")]
    for lag, est in enumerate(_est(r, "placebo"), start=1):
        _close(est, stata[(tag, f"placebo_{lag}")])
    ea = r.model_info["effect_average"]
    _close(ea["estimate"], stata[(tag, "effect_average")])
    assert ea["n_switchers"] == stata[(tag, "N_switchers_effect_average")]


@pytest.mark.parametrize("layout, d, ctrl, bal", COMBOS)
def test_firstdiff_placebo_matches_stata(raw, stata, layout, d, ctrl, bal):
    """Stata's firstdiff_placebo: Y_{F-l} - Y_{F-l-1} against DID_0's
    controls, groups observed from F-l-1 to F (the unbalanced rows pin the
    window: requiring only F-1 and F gives 30 switchers for placebo 3 on
    cell_db_none_unb where Stata has 28)."""
    tag = f"{layout}_{d}_{ctrl}_{bal}_fd"
    df = _data(raw, layout, bal)
    kw = dict(robust_dynamic=True, firstdiff_placebo=True, placebo=3)
    r = _fit(df, d, ctrl, dynamic=3, **kw)
    for row in r.model_info["placebo"]:
        lag = -row["lag"]
        _close(row["estimate"], stata[(tag, f"placebo_{lag}")])
        assert row["n_switchers"] == stata[(tag, f"N_switchers_placebo_{lag}")]
    assert r.model_info["firstdiff_placebo"] is True
    # Stata gives the same first-difference placebos without dynamic(); so
    # does StatsPAI (they use DID_0's comparison whatever the horizon)
    r0 = _fit(df, d, ctrl, **kw)
    assert _est(r0, "placebo") == _est(r, "placebo")
    # placebo 1 is the long-difference placebo 1 with the sign flipped
    long1 = _est(_fit(df, d, ctrl, robust_dynamic=True, placebo=1), "placebo")[0]
    _close(_est(r, "placebo")[0], -long1)


def test_effect_average_on_absorbing_binary_is_switcher_weighted_mean():
    """Identity: a binary staggered treatment keeps every first-time
    switcher treated, so each first stage DID^D_l is 1 and effect_average is
    sum_l N_l DID_l / sum_l N_l."""
    rng = np.random.default_rng(11)
    rows = []
    for g in range(40):
        first = int(rng.integers(3, 9)) if g >= 8 else 99
        for t in range(1, 9):
            d = int(t >= first)
            rows.append({"g": g, "t": t, "d": d, "y": 0.2 * t + 1.3 * d + rng.normal()})
    df = pd.DataFrame(rows)
    r = sp.did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        n_boot=0,
        dynamic=3,
        robust_dynamic=True,
    )
    dyn = r.model_info["dynamic"]
    n = np.array([x["n_switchers"] for x in dyn], dtype=float)
    est = np.array([x["estimate"] for x in dyn])
    ea = r.model_info["effect_average"]
    _close(ea["first_stage"], 1.0)
    _close(ea["estimate"], float(np.sum(n * est) / np.sum(n)))
    assert ea["n_switchers"] == int(n.sum())
    # the unweighted mean is a different number here (horizons differ in N)
    avg = r.model_info["avg_cumulative_effect"]["estimate"]
    _close(avg, float(est.mean()))
    assert abs(avg - ea["estimate"]) > 1e-6


def test_effect_average_only_with_robust_dynamic_and_horizons(raw):
    df = _data(raw, "cell", "bal")
    assert _fit(df, "db", "none", dynamic=2).model_info["effect_average"] is None
    rd0 = _fit(df, "db", "none", robust_dynamic=True)
    assert rd0.model_info["effect_average"] is None
    a = sp.did_multiplegt(
        df,
        y="yb",
        group="g",
        time="t",
        treatment="db",
        dynamic=2,
        robust_dynamic=True,
        n_boot=30,
        seed=4,
    )
    b = sp.did_multiplegt(
        df,
        y="yb",
        group="g",
        time="t",
        treatment="db",
        dynamic=2,
        robust_dynamic=True,
        n_boot=30,
        seed=4,
    )
    ea = a.model_info["effect_average"]
    assert np.isfinite(ea["se"]) and ea["se"] > 0
    assert ea["se"] == b.model_info["effect_average"]["se"]
    assert ea["ci_lower"] < ea["estimate"] < ea["ci_upper"]


def test_effect_average_with_zero_first_stage_is_missing_and_warns():
    """A group going 1 -> 2 -> 0 has sign -1 (summed deviation -2), so its
    first stage is -1 at l = 0 and +1 at l = 1: the sum is zero and the
    ratio is undefined, not infinite."""
    rows = []
    for g in range(6):
        path = [1, 1, 2, 0, 0, 0] if g == 0 else [1] * 6
        for t, d in enumerate(path, start=1):
            rows.append({"g": g, "t": t, "d": d, "y": 0.1 * t + 0.5 * d + 0.01 * g})
    df = pd.DataFrame(rows)
    with pytest.warns(UserWarning, match="effect_average"):
        r = sp.did_multiplegt(
            df,
            y="y",
            group="g",
            time="t",
            treatment="d",
            n_boot=0,
            dynamic=1,
            robust_dynamic=True,
        )
    ea = r.model_info["effect_average"]
    assert ea["first_stage"] == 0.0 and np.isnan(ea["estimate"])
    assert np.isfinite(r.estimate)


def test_firstdiff_placebo_requires_robust_dynamic(raw):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="firstdiff"):
        _fit(_data(raw, "cell", "bal"), "db", "none", placebo=1, firstdiff_placebo=True)


@pytest.mark.parametrize("layout, d, ctrl, bal", COMBOS)
def test_default_without_robust_dynamic_matches_stata(raw, stata, layout, d, ctrl, bal):
    """DID_M and its forward-difference placebos (sign 'r'); with controls
    the placebos reuse the DID_M slopes, as Stata does."""
    tag = f"{layout}_{d}_{ctrl}_{bal}_nr"
    extra = {"placebo_controls": "main"} if ctrl == "x" else {}
    r = _fit(_data(raw, layout, bal), d, ctrl, placebo=2, placebo_sign="r", **extra)
    _close(r.estimate, stata[(tag, "effect_0")])
    for lag, est in enumerate(_est(r, "placebo"), start=1):
        _close(est, stata[(tag, f"placebo_{lag}")])


@pytest.mark.parametrize("layout, d, ctrl", [c[:3] for c in COMBOS if c[3] == "bal"])
def test_default_matches_r_didmultiplegt(raw, rref, layout, d, ctrl):
    """The default dynamic effects and placebos, controls refitted per
    estimate, as R DIDmultiplegt 0.1.4 computes them (balanced panels)."""
    tag = f"{layout}_{d}_{ctrl}_bal_r"
    r = _fit(_data(raw, layout, "bal"), d, ctrl, dynamic=3, placebo=2, placebo_sign="r")
    _close(r.estimate, rref[(tag, "effect_0")])
    dyn = _est(r, "dynamic")
    for h in (1, 2, 3):
        _close(dyn[h], rref[(tag, f"dynamic_{h}")])
    for lag, est in enumerate(_est(r, "placebo"), start=1):
        _close(est, rref[(tag, f"placebo_{lag}")])
    assert r.model_info["controls_scope"] == ("all" if ctrl == "x" else None)


def test_r_and_stata_disagree_on_unbalanced_did_m(raw, stata, rref):
    """The documented difference, pinned so that it cannot drift silently:
    on the unbalanced panel StatsPAI's DID_M is Stata's, not R's."""
    df = _data(raw, "cell", "unb")
    r = _fit(df, "db", "none")
    _close(r.estimate, stata[("cell_db_none_unb_nr", "effect_0")])
    assert abs(r.estimate - rref[("cell_db_none_unb_r", "effect_0")]) > 1e-3


def test_robust_equals_default_on_absorbing_binary_cells():
    """Identity: with a binary staggered treatment and one row per cell,
    first-time switchers are the switchers and not-yet-switched groups are
    the robust stayers, so both rules give the same effects."""
    rng = np.random.default_rng(5)
    rows = []
    for g in range(30):
        first = int(rng.integers(2, 9)) if g >= 6 else 99
        for t in range(1, 9):
            d = int(t >= first)
            rows.append(
                {
                    "g": g,
                    "t": t,
                    "d": d,
                    "y": 0.3 * t + (1 + 0.2 * (t - first)) * d + rng.normal(),
                }
            )
    df = pd.DataFrame(rows)
    kw = dict(y="y", group="g", time="t", treatment="d", n_boot=0, dynamic=3, placebo=2)
    a = sp.did_multiplegt(df, **kw)
    b = sp.did_multiplegt(df, robust_dynamic=True, **kw)
    _close(a.estimate, b.estimate)
    for x, z in zip(_est(a, "dynamic"), _est(b, "dynamic")):
        _close(x, z)
    # placebo 1 is the same difference; from lag 2 on robust_dynamic's are
    # long differences (Y_{F-1-l} - Y_{F-1}), the default's first differences
    _close(_est(a, "placebo")[0], _est(b, "placebo")[0])
    assert b.model_info["robust_dynamic"] is True


def test_controls_no_longer_warn(raw):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        r = _fit(_data(raw, "cell", "bal"), "db", "x", placebo=1, dynamic=1)
    assert r.model_info["controls_scope"] == "all"
    assert r.model_info["placebo_controls"] == "refit"


@pytest.mark.parametrize(
    "kw",
    [
        {"robust_dynamic": True, "placebo_sign": "r"},
        {"robust_dynamic": True, "placebo_controls": "main"},
    ],
)
def test_robust_dynamic_refuses_pairwise_options(raw, kw):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="robust_dynamic"):
        _fit(_data(raw, "cell", "bal"), "db", "x", placebo=1, **kw)


def test_placebo_controls_value_is_checked(raw):
    with pytest.raises(ValueError, match="placebo_controls"):
        _fit(_data(raw, "cell", "bal"), "db", "x", placebo=1, placebo_controls="r")


def test_robust_bootstrap_is_reproducible(raw):
    df = _data(raw, "cell", "bal")
    a = _fit(df, "db", "none", dynamic=1, robust_dynamic=True)
    b = sp.did_multiplegt(
        df,
        y="yb",
        group="g",
        time="t",
        treatment="db",
        dynamic=1,
        robust_dynamic=True,
        n_boot=30,
        seed=7,
    )
    c = sp.did_multiplegt(
        df,
        y="yb",
        group="g",
        time="t",
        treatment="db",
        dynamic=1,
        robust_dynamic=True,
        n_boot=30,
        seed=7,
    )
    assert b.estimate == a.estimate
    assert np.isfinite(b.se) and b.se > 0 and b.se == c.se
