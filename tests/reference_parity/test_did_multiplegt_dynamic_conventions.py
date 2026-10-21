"""sp.did_multiplegt's dynamic effects, placebos and controls against both of
dCDH's own implementations, on the same CSV bytes (point estimates).

Fixture: ``_fixtures/dcdh_dynamic_panel.csv`` (60 groups, 9 periods, 1-4
rows per cell; a count treatment ``dc`` from baselines 0 / 1 / 2 and a
binary ``db``, both switching on and off; a control ``x``; ``keep == 0``
marks the cells dropped in the unbalanced variant), from
``_generate_dcdh_dynamic_data.py``. References:

* ``dcdh_dynamic_Stata.csv``: Stata 18 ``did_multiplegt_old``, without
  ``robust_dynamic`` (tag ``_nr``, ``placebo(2)``) and with it (``_rd``,
  ``dynamic(3) placebo(3)``), from ``_generate_dcdh_dynamic_Stata.do``;
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
