"""``sp.assign_placebo_dates``: distribution, reproducibility, event study.

The device (never-treated units get event dates drawn from the treated
units' dates) is right when the drawn dates follow the treated
distribution: then every event time mixes the same calendar periods in
both groups and calendar shocks cancel. The tests check the distribution
(exact frequencies for the proportional allocation, a binomial screen for
independent draws), invariance to row order, and that an event study of
treated versus placebo-dated units on a design with no effect and strong,
non-linear calendar effects is centred on zero, while the same event study
with dates drawn from a different distribution is not.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _units(n_treated, n_never, dates, probs, seed, strata=None):
    rng = np.random.default_rng(seed)
    ev = rng.choice(dates, size=n_treated, p=probs).astype(float)
    df = pd.DataFrame(
        {
            "id": np.arange(n_treated + n_never),
            "event": np.concatenate([ev, np.full(n_never, np.nan)]),
        }
    )
    if strata is not None:
        df["s"] = strata
    return df


def test_result_does_not_depend_on_row_order():
    df = _units(300, 700, [2003, 2004, 2005, 2006], [0.1, 0.2, 0.3, 0.4], 0)
    panel = df.loc[df.index.repeat(3)].assign(year=np.tile([2002, 2004, 2006], len(df)))
    panel["s"] = panel["id"] % 3
    kw = dict(time="year", strata="s", seed=42)
    a = sp.assign_placebo_dates(panel, "id", "event", **kw)
    shuffled = panel.sample(frac=1.0, random_state=7)
    b = sp.assign_placebo_dates(shuffled, "id", "event", **kw)
    ua = a.groupby("id")["placebo_event_time"].first()
    ub = b.groupby("id")["placebo_event_time"].first()
    pd.testing.assert_series_equal(ua, ub)
    # and with require_observed, which reads each unit's own periods
    a2 = sp.assign_placebo_dates(panel, "id", "event", require_observed=True, **kw)
    b2 = sp.assign_placebo_dates(shuffled, "id", "event", require_observed=True, **kw)
    pd.testing.assert_series_equal(
        a2.groupby("id")["placebo_event_time"].first(),
        b2.groupby("id")["placebo_event_time"].first(),
    )


def test_treated_rows_keep_their_date_and_flags():
    df = _units(50, 80, [1, 2, 3], [0.2, 0.3, 0.5], 1)
    out = sp.assign_placebo_dates(df, "id", "event", seed=0)
    treated = df["event"].notna()
    np.testing.assert_array_equal(
        out.loc[treated, "placebo_event_time"], df.loc[treated, "event"]
    )
    assert not out.loc[treated, "placebo"].any()
    assert out.loc[~treated, "placebo"].all()
    assert out.attrs["placebo_dates"]["n_placebo_units"] == 80


def test_proportional_allocation_reproduces_frequencies_exactly():
    dates = [2004, 2005, 2006, 2007, 2008]
    s_t = np.repeat([0, 1], [40, 60])
    rng = np.random.default_rng(3)
    ev = np.where(
        s_t == 0,
        rng.choice(dates, size=100, p=[0.1, 0.4, 0.2, 0.2, 0.1]),
        rng.choice(dates, size=100, p=[0.3, 0.1, 0.1, 0.1, 0.4]),
    ).astype(float)
    # balanced: three never-treated units per treated unit in each stratum
    s_n = np.repeat([0, 1], [120, 180])
    df = pd.DataFrame(
        {
            "id": np.arange(400),
            "event": np.concatenate([ev, np.full(300, np.nan)]),
            "s": np.concatenate([s_t, s_n]),
        }
    )
    out = sp.assign_placebo_dates(
        df, "id", "event", strata="s", method="proportional", seed=5
    )
    for s in (0, 1):
        tr = out[(out.s == s) & ~out.placebo]["placebo_event_time"].value_counts()
        pl = out[(out.s == s) & out.placebo]["placebo_event_time"].value_counts()
        pd.testing.assert_series_equal(
            pl.sort_index(), (3 * tr).sort_index(), check_names=False
        )


def test_empirical_draws_match_the_treated_frequencies():
    dates = np.arange(2000, 2010)
    probs = np.linspace(1, 3, 10)
    probs /= probs.sum()
    df = _units(3000, 30000, dates, probs, 2)
    out = sp.assign_placebo_dates(df, "id", "event", seed=8)
    tr = out.loc[~out.placebo, "placebo_event_time"].value_counts(normalize=True)
    pl = out.loc[out.placebo, "placebo_event_time"].value_counts(normalize=True)
    tr, pl = tr.reindex(dates, fill_value=0), pl.reindex(dates, fill_value=0)
    # draws are multinomial(30000, treated frequencies): each placebo share
    # is within 4 binomial SEs of its treated share
    se = np.sqrt(tr * (1 - tr) / 30000)
    assert np.all(np.abs(pl - tr) < 4 * se)


def test_strata_draw_only_their_own_dates():
    df = pd.DataFrame(
        {
            "id": range(8),
            "event": [1, 1, 5, 5, np.nan, np.nan, np.nan, np.nan],
            "s": ["a", "a", "b", "b", "a", "a", "b", "b"],
        }
    )
    out = sp.assign_placebo_dates(df, "id", "event", strata="s", seed=0)
    assert set(out.loc[out.placebo & (out.s == "a"), "placebo_event_time"]) == {1.0}
    assert set(out.loc[out.placebo & (out.s == "b"), "placebo_event_time"]) == {5.0}


def test_require_observed_draws_inside_the_unit_window():
    rows = []
    for i in range(200):
        years = range(2000, 2010) if i < 100 else range(2000 + i % 5, 2004 + i % 5)
        ev = 2001 + i % 8 if i < 100 else np.nan
        rows += [(i, y, ev) for y in years]
    df = pd.DataFrame(rows, columns=["id", "year", "event"])
    out = sp.assign_placebo_dates(
        df, "id", "event", time="year", require_observed=True, seed=1
    )
    pl = out[out.placebo]
    inside = pl.groupby("id")[["placebo_event_time", "year"]].apply(
        lambda g: g["placebo_event_time"].iloc[0] in set(g["year"])
    )
    assert inside.all() and len(inside) == 100
    np.testing.assert_array_equal(
        out["placebo_event_time_rel"], out["year"] - out["placebo_event_time"]
    )


def test_unit_with_no_eligible_date_is_reported():
    df = pd.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "year": [2000, 2001, 2005, 2006],
            "event": [2001, 2001, np.nan, np.nan],
        }
    )
    with pytest.warns(UserWarning, match="got no placebo date"):
        out = sp.assign_placebo_dates(
            df, "id", "event", time="year", require_observed=True, seed=0
        )
    assert out.loc[out.id == 2, "placebo_event_time"].isna().all()
    assert not out.loc[out.id == 2, "placebo"].any()
    assert out.attrs["placebo_dates"]["n_undated_units"] == 1


def test_stratum_without_treated_units_is_reported():
    df = pd.DataFrame({"id": [1, 2, 3], "event": [4, np.nan, np.nan], "s": [0, 0, 1]})
    with pytest.warns(UserWarning, match="without treated units"):
        out = sp.assign_placebo_dates(df, "id", "event", strata="s", seed=0)
    assert out.attrs["placebo_dates"]["strata_without_treated"] == [1]


# ---------------------------------------------------------------------------
# Event study with no effect
# ---------------------------------------------------------------------------


def _panel(seed=0, n_treated=1500, n_never=4500):
    rng = np.random.default_rng(seed)
    years = np.arange(2000, 2018)
    gamma = 0.08 * (years - 2000) ** 2  # strong, convex calendar effects
    dates = np.arange(2006, 2012)
    probs = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.3])
    n = n_treated + n_never
    ev = np.concatenate(
        [rng.choice(dates, size=n_treated, p=probs), np.full(n_never, np.nan)]
    )
    alpha = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(n), len(years)),
            "year": np.tile(years, n),
            "event": np.repeat(ev, len(years)),
        }
    )
    df["y"] = (
        np.repeat(alpha, len(years)) + np.tile(gamma, n) + rng.normal(size=len(df))
    )  # no treatment effect
    return df


def _event_study(df, rel_col, flag_col, ks):
    """est_k = mean_i [Y(e+k) - Y(e-1)] treated  minus  placebo-dated."""
    base = df[df[rel_col] == -1].set_index("id")["y"]
    out = {}
    for k in ks:
        cur = df[df[rel_col] == k].set_index("id")
        diff = (cur["y"] - base.reindex(cur.index)).dropna()
        grp = cur.loc[diff.index, flag_col]
        a, b = diff[~grp], diff[grp]
        est = a.mean() - b.mean()
        se = np.sqrt(a.var() / len(a) + b.var() / len(b))
        out[k] = (est, se)
    return out


def test_event_study_with_placebo_dates_is_centred_on_zero():
    df = _panel()
    out = sp.assign_placebo_dates(df, "id", "event", time="year", seed=3)
    ks = [k for k in range(-5, 6) if k != -1]
    es = _event_study(out, "placebo_event_time_rel", "placebo", ks)
    z = np.array([e / s for e, s in es.values()])
    # no effect: every coefficient within 4 SE of zero (10 coefficients)
    assert np.all(np.abs(z) < 4)


def test_placebo_dates_from_another_distribution_are_biased():
    """Same design, placebo dates uniform over 2002-2015 instead of the
    treated distribution: calendar effects no longer cancel."""
    df = _panel()
    rng = np.random.default_rng(4)
    never = df.loc[df["event"].isna(), "id"].unique()
    wrong = pd.Series(rng.integers(2002, 2016, len(never)).astype(float), index=never)
    df = df.assign(
        pe=df["event"].fillna(df["id"].map(wrong)),
        pl=df["event"].isna(),
    )
    df["rel"] = df["year"] - df["pe"]
    es = _event_study(df, "rel", "pl", [3, 5])
    z = np.array([e / s for e, s in es.values()])
    # convex calendar effects: Y(e+k) - Y(e-1) grows with e, and the wrong
    # dates are on average earlier -> large negative placebo differences
    assert np.all(np.abs(z) > 10)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_event_time_must_be_constant_within_unit():
    df = pd.DataFrame({"id": [1, 1, 2], "event": [3, 4, np.nan]})
    with pytest.raises(MethodIncompatibility, match="constant within each unit"):
        sp.assign_placebo_dates(df, "id", "event")


def test_no_treated_unit_is_refused():
    df = pd.DataFrame({"id": [1, 2], "event": [np.nan, np.nan]})
    with pytest.raises(DataInsufficient):
        sp.assign_placebo_dates(df, "id", "event")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"require_observed": True}, "time="),
        ({"method": "uniform"}, "method"),
        (
            {"time": "year", "require_observed": True, "method": "proportional"},
            "proportional",
        ),
        ({"column": "event"}, "already exists"),
    ],
)
def test_bad_options_are_refused(kwargs, match):
    df = pd.DataFrame({"id": [1, 2], "year": [1, 1], "event": [1, np.nan]})
    with pytest.raises(MethodIncompatibility, match=match):
        sp.assign_placebo_dates(df, "id", "event", **kwargs)


def test_seed_reproducible():
    df = _units(30, 60, [1, 2, 3], [0.3, 0.3, 0.4], 9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = sp.assign_placebo_dates(df, "id", "event", seed=12)
        b = sp.assign_placebo_dates(df, "id", "event", seed=12)
    pd.testing.assert_frame_equal(a, b)
