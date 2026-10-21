"""``sp.did_summary`` puts one estimand on every row; event-study paths.

Up to 1.40.0 the SA and stacked rows were the equal-weight average of the
post-treatment event-time effects while CS / BJS / ETWFE were averages over
treated observations, so the cross-method spread mixed two estimands (on a
267-city staggered panel: SA 0.0170 and stacked 0.0162 against ~0.012 for
the other three). Every row now is the treated-observation ATT, and each
row equals the estimator called on its own with that aggregation.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402

KW = dict(y="y", time="time", first_treat="first_treat", group="unit")


@pytest.fixture(scope="module")
def panel():
    return sp.dgp_did(n_units=150, n_periods=9, staggered=True, seed=11)


@pytest.fixture(scope="module")
def summary(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_summary(panel, event_study=True, **KW)


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def test_sa_and_stacked_rows_are_treated_observation_atts(panel, summary):
    det = summary.detail.set_index("method")
    sa = _quiet(
        sp.sun_abraham,
        panel,
        y="y",
        g="first_treat",
        t="time",
        i="unit",
        aggregation="fixest_att",
        crit_dist="normal",
    )
    st = _quiet(
        sp.stacked_did,
        panel,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        aggregation="treated_obs",
    )
    assert det.loc["sa", "estimate"] == pytest.approx(sa.estimate, rel=1e-12)
    assert det.loc["sa", "se"] == pytest.approx(sa.se, rel=1e-12)
    assert det.loc["stacked", "estimate"] == pytest.approx(st.estimate, rel=1e-12)
    assert det.loc["stacked", "se"] == pytest.approx(st.se, rel=1e-12)
    # and not the equal-weight event-time average they used to report
    old = _quiet(
        sp.stacked_did,
        panel,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
    )
    assert not np.isclose(det.loc["stacked", "estimate"], old.estimate)


def test_estimand_column(summary):
    det = summary.detail
    assert list(det.columns[:4]) == ["method", "estimator", "estimand", "estimate"]
    assert det["estimand"].str.startswith("ATT").all()
    assert set(summary.model_info["estimands"]) == set(det["method"])


def test_paths_reproduce_each_estimator(panel, summary):
    paths = summary.event_study_paths
    assert list(paths.columns) == [
        "method",
        "relative_time",
        "att",
        "se",
        "ci_lower",
        "ci_upper",
        "pvalue",
        "lead_definition",
        "reference",
    ]
    assert paths["relative_time"].between(-5, 5).all()
    sa = _quiet(
        sp.sun_abraham,
        panel,
        y="y",
        g="first_treat",
        t="time",
        i="unit",
        aggregation="fixest_att",
        crit_dist="normal",
    ).event_study.set_index("relative_time")
    got = paths[(paths.method == "sa") & ~paths.reference].set_index("relative_time")
    for k, row in got.iterrows():
        assert row["att"] == pytest.approx(sa.loc[k, "att"], rel=1e-12)
        assert row["se"] == pytest.approx(sa.loc[k, "se"], rel=1e-12)
    em = sp.etwfe_emfx(_quiet(sp.etwfe, panel, **KW), type="event").event_study
    et = paths[paths.method == "etwfe"].set_index("relative_time")
    assert (et.index >= 0).all()
    for k, row in et.iterrows():
        assert row["att"] == pytest.approx(
            em.set_index("relative_time").loc[k, "att"], rel=1e-12
        )


def test_lead_definitions_and_reference_rows(summary):
    paths = summary.event_study_paths
    lead = paths.groupby("method")["lead_definition"].first().to_dict()
    assert lead == {
        "cs": "reference_period",
        "sa": "reference_period",
        "stacked": "reference_period",
        "bjs": "untreated_regression",
        "etwfe": "post_only",
    }
    ref = paths[paths.reference]
    assert sorted(ref["method"]) == ["cs", "sa", "stacked"]
    assert (ref["relative_time"] == -1).all() and (ref["att"] == 0).all()
    # BJS estimates its k = -1 lead; nothing there is a normalisation
    bjs = paths[paths.method == "bjs"].set_index("relative_time")
    assert -1 in bjs.index and not bjs.loc[-1, "reference"]
    assert bjs.loc[-1, "se"] > 0


def test_bjs_path_is_did_imputation_with_pretrends(panel, summary):
    paths = summary.event_study_paths
    bjs = paths[paths.method == "bjs"].set_index("relative_time")
    leads = sorted(k for k in bjs.index if k < 0)
    ref = _quiet(
        sp.did_imputation,
        panel,
        y="y",
        group="unit",
        time="time",
        first_treat="first_treat",
        horizon=[k for k in bjs.index if k >= 0],
        pretrends=len(leads),
    ).event_study.set_index("relative_time")
    for k, row in bjs.iterrows():
        assert row["att"] == pytest.approx(ref.loc[k, "att"], rel=1e-12)


def test_default_has_no_paths(panel):
    out = _quiet(sp.did_summary, panel, methods=["cs", "bjs"], **KW)
    assert out.model_info["event_study_paths"] is None
    assert not hasattr(out, "event_study_paths")
    with pytest.raises(ValueError, match="event_study=True"):
        sp.did_summary_plot(out, kind="event_study")


def test_event_window_trims_paths(panel):
    out = _quiet(
        sp.did_summary,
        panel,
        methods=["sa", "stacked"],
        event_study=True,
        event_window=(-2, 1),
        **KW,
    )
    assert set(out.event_study_paths["relative_time"]) == {-2, -1, 0, 1}


def test_event_study_plot_draws_hollow_bjs_leads(summary):
    fig, ax = sp.did_summary_plot(summary, kind="event_study")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "bjs (untreated_regression leads)" in labels
    assert {"cs", "sa", "stacked", "etwfe"} <= set(labels)
    with pytest.raises(ValueError, match="kind"):
        sp.did_summary_plot(summary, kind="forest")
    matplotlib.pyplot.close(fig)


# ---------------------------------------------------------------- bacon_plot


def _scatter_colors(ax):
    out = {}
    for coll in ax.collections:
        lab = coll.get_label()
        fc = coll.get_facecolor()
        if len(fc):
            out[lab] = mcolors.to_hex(fc[0], keep_alpha=False)
    return out


def test_bacon_plot_default_colors_match_the_type_names():
    df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=0)
    b = _quiet(
        sp.bacon_decomposition, df, y="y", treat="treated", time="time", id="unit"
    )
    fig, ax = sp.bacon_plot(b)
    cols = _scatter_colors(ax)
    types = set(b["decomposition"]["type"])
    assert types <= set(cols)
    assert all(cols[t] != "#7f8c8d" for t in types)  # not the grey fallback
    assert len({cols[t] for t in types}) == len(types)
    matplotlib.pyplot.close(fig)


def test_bacon_plot_partial_colors_override_one_type():
    df = sp.dgp_did(n_units=120, n_periods=8, staggered=True, seed=0)
    b = _quiet(
        sp.bacon_decomposition, df, y="y", treat="treated", time="time", id="unit"
    )
    fig, ax = sp.bacon_plot(b, colors={"Treated vs Untreated": "#000000"})
    cols = _scatter_colors(ax)
    assert cols["Treated vs Untreated"] == "#000000"
    assert all(c != "#7f8c8d" for c in cols.values())
    matplotlib.pyplot.close(fig)


def test_bacon_plot_colors_cover_every_type():
    # 'Both Treated' / 'Later vs Always Treated' come from the covariate and
    # always-treated paths; each type gets its own non-grey default.
    types = [
        "Treated vs Untreated",
        "Earlier vs Later Treated",
        "Later vs Earlier Treated",
        "Later vs Always Treated",
        "Both Treated",
    ]
    decomp = pd.DataFrame(
        {
            "type": types,
            "weight": [0.4, 0.2, 0.2, 0.1, 0.1],
            "estimate": [1, 2, 3, 4, 5],
        }
    )
    fig, ax = sp.bacon_plot({"decomposition": decomp, "beta_twfe": 2.0})
    cols = _scatter_colors(ax)
    assert set(types) <= set(cols)
    assert all(cols[t] != "#7f8c8d" for t in types)
    assert len({cols[t] for t in types}) == len(types)
    matplotlib.pyplot.close(fig)
