"""``sp.did_multiplegt_dyn``: the TWFE event-time axis, ``detail['se']`` and
the 1.42 default announcement.

The estimator indexes effects from 0 and placebos from -1, so placebo 1 sits
at relative time -1, the period every event-study regression normalises to
zero. ``twfe_relative_time`` puts each number where a base-(-1) event study
puts the same contrast. With one cohort and never-treated controls on a
balanced panel the fully saturated TWFE event-study coefficient at ``k`` is
the difference in means of ``Y_{F+k} - Y_{F-1}``, which is exactly effect
``k`` (``k >= 0``) and placebo ``l = -(k + 1)`` (``k <= -2``): the test
asserts that identity.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

F = 6


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        treated = i < 25
        a = rng.normal()
        for t in range(1, 11):
            d = int(treated and t >= F)
            y = a + 0.3 * t + (0.5 * (t - F + 1) if d else 0.0) + rng.normal(0, 0.5)
            rows.append({"i": i, "t": t, "d": d, "y": y, "g": F if treated else 0})
    return pd.DataFrame(rows)


def _fit(df: pd.DataFrame, **kw) -> sp.CausalResult:
    kw.setdefault("aggregation", "switchers")
    kw.setdefault("se_method", "analytic")
    kw.setdefault("placebo", 3)
    kw.setdefault("dynamic", 3)
    return sp.did_multiplegt_dyn(df, "y", group="i", time="t", treatment="d", **kw)


def test_twfe_relative_time_equals_saturated_event_study(panel):
    r = _fit(panel)
    es = r.model_info["event_study"]
    assert es["twfe_relative_time"].tolist() == [-4, -3, -2, 0, 1, 2, 3]
    assert es["relative_time"].tolist() == [-3, -2, -1, 0, 1, 2, 3]  # unchanged

    df = panel.copy()
    k = np.where(df["g"] > 0, df["t"] - F, np.nan)
    names = {}
    for e in range(-5, 5):
        if e == -1:
            continue
        names[e] = f"k{'m' if e < 0 else 'p'}{abs(e)}"
        df[names[e]] = (k == e).astype(int)
    fit = sp.feols("y ~ " + " + ".join(names.values()) + " | i + t", df)
    for row in es.itertuples(index=False):
        assert row.att == pytest.approx(
            float(fit.params[names[row.twfe_relative_time]]), abs=1e-10
        )


def test_detail_carries_twfe_axis_and_reported_se(panel):
    r = _fit(panel)
    det = r.detail.set_index("horizon")
    es = r.model_info["event_study"].set_index("relative_time")
    assert det["twfe_relative_time"].to_dict() == es["twfe_relative_time"].to_dict()
    # analytic: the public se is the private analytic one, horizon by horizon
    np.testing.assert_array_equal(det["se"].to_numpy(), det["_se_analytic"].to_numpy())
    np.testing.assert_array_equal(
        det["se"].to_numpy(), es.loc[det.index, "se"].to_numpy()
    )

    rb = _fit(panel, se_method="bootstrap", n_boot=40, seed=3)
    detb = rb.detail.set_index("horizon")
    esb = rb.model_info["event_study"].set_index("relative_time")
    # bootstrap: detail['se'] follows the requested method, not _se_analytic
    np.testing.assert_array_equal(
        detb["se"].to_numpy(), esb.loc[detb.index, "se"].to_numpy()
    )
    assert not np.allclose(detb["se"], detb["_se_analytic"])


def test_unset_defaults_warn_and_keep_their_numbers(panel):
    kw = dict(placebo=1, dynamic=2, n_boot=30, seed=7)
    args = (panel, "y")
    cols = dict(group="i", time="t", treatment="d")
    with pytest.warns(
        FutureWarning, match="aggregation='simple'.*se_method='bootstrap'"
    ):
        r_default = sp.did_multiplegt_dyn(*args, **cols, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        r_explicit = sp.did_multiplegt_dyn(
            *args, **cols, aggregation="simple", se_method="bootstrap", **kw
        )
    assert r_default.estimate == r_explicit.estimate
    assert r_default.se == r_explicit.se
    assert r_default.model_info["aggregation"] == "simple"
    assert r_default.model_info["se_method"] == "bootstrap"

    # only the argument left unset is named
    with pytest.warns(FutureWarning) as rec:
        sp.did_multiplegt_dyn(*args, **cols, aggregation="switchers", **kw)
    msg = " ".join(str(w.message) for w in rec if w.category is FutureWarning)
    assert "se_method='bootstrap'" in msg and "aggregation='simple'" not in msg


def test_switchers_is_the_weighted_mean_of_the_effects(panel):
    # Binary absorbing treatment: Av_tot_eff = sum_l N_l delta_l / sum_l N_l.
    r = _fit(panel, placebo=0)
    det = r.detail
    w = det["w_switchers"].to_numpy()
    assert r.estimate == pytest.approx(
        float(np.sum(w * det["delta_l"]) / np.sum(w)), rel=1e-12
    )


def test_translations_pin_both_arguments():
    for out in (
        sp.from_stata("did_multiplegt_dyn y g t d, effects(3) placebo(2)"),
        sp.from_r('did_multiplegt_dyn(df, "y", "g", "t", "d", effects = 3)'),
    ):
        assert out["arguments"]["aggregation"] == "switchers"
        assert out["arguments"]["se_method"] == "analytic"
        assert out["arguments"]["dynamic"] == 2
