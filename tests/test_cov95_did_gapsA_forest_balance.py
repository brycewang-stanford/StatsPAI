"""Coverage gaps in ``sp.did_forest`` and ``sp.did_balance``: input
contracts, cells that cannot be estimated, non-integer calendars, the
event-study plot and the balance report's degenerate cases.
"""

import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import (  # noqa: E402
    AssumptionWarning,
    DataInsufficient,
    MethodIncompatibility,
)

# ---------------------------------------------------------------------------
#  did_forest
# ---------------------------------------------------------------------------

FK = dict(y="y", id="id", time="t", cohort="g", x="x1", n_estimators=60)
FK["min_group_size"] = 10


def _forest_panel(N=160, T=5, seed=0, cohorts=(3, 4)):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=N)
    a = rng.normal(size=N)
    g = np.array([0, *cohorts])[np.arange(N) % (len(cohorts) + 1)]
    frames = []
    for t in range(1, T + 1):
        d = (g > 0) & (t >= g)
        y = a + 0.5 * t + d * (1 + 0.5 * x1) + 0.5 * rng.normal(size=N)
        frames.append(
            pd.DataFrame(
                {"id": np.arange(N), "t": t, "g": g, "x1": x1, "cl": np.arange(N) % 20}
            ).assign(y=y)
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture(scope="module")
def fpanel():
    return _forest_panel()


@pytest.fixture(scope="module")
def fres(fpanel):
    return sp.did_forest(fpanel, **FK)


def test_forest_recovers_the_average_effect(fres):
    # tau = 1 + 0.5 x1 with E[x1] = 0.
    assert abs(fres.overall["estimate"] - 1.0) < 4 * fres.overall["se"]
    assert list(fres.event_study["event_time"]) == [-3, -2, 0, 1, 2]
    assert len(fres.dropped_cells) == 0


@pytest.mark.parametrize(
    "change, match",
    [
        ({"y": "zz"}, r"column\(s\) not in data: \['zz'\]"),
        ({"control_group": "all"}, "control_group must be 'notyettreated'"),
        ({"anticipation": -1}, "anticipation must be a non-negative integer"),
        ({"anticipation": True}, "anticipation must be a non-negative integer"),
        ({"min_group_size": 1}, "min_group_size must be an integer >= 2"),
    ],
)
def test_forest_argument_contracts(fpanel, change, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.did_forest(fpanel, **{**FK, **change})


def test_forest_data_contracts(fpanel):
    with pytest.raises(MethodIncompatibility, match="must be a pandas DataFrame"):
        sp.did_forest(fpanel.to_numpy(), **FK)
    labelled = fpanel.assign(t=fpanel["t"].map(lambda v: f"p{v}"))
    with pytest.raises(MethodIncompatibility, match="time column must be numeric"):
        sp.did_forest(labelled, **FK)
    moving = fpanel.copy()
    moving.loc[0, "cl"] = 99
    with pytest.raises(MethodIncompatibility, match="varies within unit"):
        sp.did_forest(moving, clusters="cl", **FK)
    with pytest.raises(DataInsufficient, match="at least two clusters"):
        sp.did_forest(fpanel.assign(cl=1), clusters="cl", **FK)
    with pytest.raises(DataInsufficient, match="no treated cohort found"):
        sp.did_forest(fpanel.assign(g=0), **FK)


def test_forest_with_no_estimable_cell(fpanel):
    with pytest.raises(DataInsufficient, match="no group-time cell") as err:
        sp.did_forest(fpanel, **{**FK, "min_group_size": 500})
    reasons = [c["reason"] for c in err.value.diagnostics["dropped_cells"]]
    assert reasons and all(r.startswith("too few units") for r in reasons)
    with pytest.raises(DataInsufficient, match="no post-treatment cell"):
        sp.did_forest(fpanel, event_window=(-3, -1), **FK)


def test_forest_too_few_trees_for_out_of_bag_predictions(fpanel):
    # Two trees leave rows that no tree held out; such a cell has no honest
    # ATT and is dropped rather than averaged over the rows that do.
    with pytest.raises(DataInsufficient, match="no group-time cell") as err:
        sp.did_forest(fpanel, **{**FK, "n_estimators": 2})
    reasons = {c["reason"] for c in err.value.diagnostics["dropped_cells"]}
    assert reasons == {"rows without out-of-bag prediction; increase n_estimators"}


def test_forest_cohort_treated_in_the_first_period_is_dropped():
    df = _forest_panel(cohorts=(1, 3, 4))
    with pytest.warns(AssumptionWarning, match="cell\\(s\\) were dropped"):
        res = sp.did_forest(df, **FK)
    first = res.dropped_cells[res.dropped_cells["group"] == 1]
    assert list(first["reason"]) == ["no pre-treatment base period"]
    assert first["time"].isna().all()
    assert 1 not in set(res.att_gt["group"])


def test_forest_is_invariant_to_a_fractional_calendar_shift(fpanel, fres):
    shifted = fpanel.assign(
        t=fpanel["t"] + 0.5, g=np.where(fpanel["g"] > 0, fpanel["g"] + 0.5, 0)
    )
    res = sp.did_forest(shifted, **FK)
    assert res.overall["estimate"] == pytest.approx(fres.overall["estimate"], abs=1e-12)
    assert res.overall["se"] == pytest.approx(fres.overall["se"], abs=1e-12)
    # Labels keep the user's calendar; event time is the same difference.
    assert set(res.att_gt["group"]) == {3.5, 4.5}
    assert sorted(res.att_gt["event_time"].unique()) == [-3.0, -2.0, 0.0, 1.0, 2.0]


def test_forest_never_treated_controls_window_and_clusters(fpanel, fres):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_forest(
            fpanel,
            control_group="nevertreated",
            event_window=(0, 1),
            clusters="cl",
            **FK,
        )
    assert set(res.att_gt["event_time"]) == {0, 1}
    # 54 never-treated units are the comparison group of every cell.
    n_never = fpanel.loc[fpanel["g"] == 0, "id"].nunique()
    assert (res.att_gt["n_control"] == n_never).all()
    assert abs(res.overall["estimate"] - 1.0) < 4 * res.overall["se"]


def test_forest_predict_cate_and_plot(fres):
    with pytest.raises(MethodIncompatibility, match="no estimated cells at event"):
        fres.predict_cate(np.zeros((2, 1)), 99)
    tau = fres.predict_cate(np.array([[-1.0], [1.0]]), 0)
    assert tau.shape == (2,)
    assert tau[1] > tau[0]  # the effect rises with x1
    ax = fres.plot()
    assert ax.get_xlabel() == "Event time" and ax.get_ylabel() == "ATT"
    assert ax.get_title() == "DiD causal forest: event study"
    plotted = ax.containers[0].lines[0]
    assert list(plotted.get_xdata()) == list(fres.event_study["event_time"])
    assert np.allclose(plotted.get_ydata(), fres.event_study["att"])
    import matplotlib.pyplot as plt

    fig, own = plt.subplots()
    assert fres.plot(ax=own) is own
    plt.close("all")


# ---------------------------------------------------------------------------
#  did_balance
# ---------------------------------------------------------------------------

BK = dict(g="g", t="t", i="i")


@pytest.fixture(scope="module")
def bpanel():
    rng = np.random.default_rng(0)
    N, T = 60, 5
    g = np.array([0, 3, 4])[np.arange(N) % 3]
    x1 = rng.normal(size=N)
    frames = []
    for t in range(1, T + 1):
        frames.append(
            pd.DataFrame(
                {
                    "i": np.arange(N),
                    "t": t,
                    "g": g,
                    "x1": x1 + 0.1 * t * rng.normal(size=N),
                    "x2": rng.normal(size=N),
                    "const": x1,
                    "sep": (g == 3).astype(float),
                    "w": 1.0 + np.arange(N) % 2,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_balance_report_for_a_balanced_design(bpanel):
    res = sp.did_balance(bpanel, ["x1", "x2", "const"], **BK)
    assert res.flagged == []
    assert res.max_abs_norm_diff == pytest.approx(
        float(res.table["abs_norm_diff"].max())
    )
    assert res.max_abs_norm_diff < 0.25
    assert res.diagnostics["constant_in_changes"] == ["const"]
    text = res.summary()
    assert "No covariate breaches |norm. diff| > 0.25 in either panel." in text
    assert "IMBALANCED" not in text
    tex = res.to_latex()
    assert "Variable & Comparison & Treated & Norm. diff. \\\\" in tex
    assert "Weighted" not in tex


def test_balance_complete_separation_prints_as_infinite(bpanel):
    res = sp.did_balance(bpanel, ["sep"], cohort=3, **BK)
    level = res.levels.iloc[0]
    assert level["mean_treated"] == 1.0 and level["mean_comparison"] == 0.0
    assert np.isposinf(level["norm_diff"])
    assert res.flagged == ["sep"]
    assert "+inf" in res.summary()


def test_balance_covariate_missing_at_the_comparison_period(bpanel):
    df = bpanel.copy()
    df.loc[df["t"] == 3, "x1"] = np.nan
    res = sp.did_balance(df, ["x1"], cohort=3, **BK)
    assert (res.base_period, res.comparison_period) == (2, 3)
    assert len(res.levels) == 1 and res.changes.empty
    assert "Covariate CHANGES" not in res.summary()
    assert "Covariate differences" not in res.to_latex()


def test_balance_not_yet_treated_comparison_group(bpanel):
    treated_only = bpanel[bpanel["g"] > 0]
    with pytest.raises(DataInsufficient, match=r"comparison \(0\) group"):
        sp.did_balance(treated_only, ["x1"], cohort=3, **BK)
    res = sp.did_balance(
        treated_only, ["x1"], cohort=3, control_group="notyettreated", **BK
    )
    assert (res.n_treated, res.n_comparison) == (20, 20)
    assert res.diagnostics["control_group"] == "notyettreated"


@pytest.mark.parametrize(
    "kwargs, exc, match",
    [
        ({"covariates": []}, MethodIncompatibility, "at least one covariate"),
        ({"weights": "zz"}, MethodIncompatibility, r"not found in data: \['zz'\]"),
        ({"control_group": "all"}, MethodIncompatibility, "control_group must be"),
        ({"cohort": 7}, MethodIncompatibility, "cohort=7 is not a treated cohort"),
        ({"base_period": 5}, DataInsufficient, "No period after the base period"),
    ],
)
def test_balance_argument_contracts(bpanel, kwargs, exc, match):
    kw = {"covariates": ["x1"], **BK, **kwargs}
    with pytest.raises(exc, match=match):
        sp.did_balance(bpanel, **kw)


def test_balance_data_contracts(bpanel):
    with pytest.raises(DataInsufficient, match="No treated cohorts found"):
        sp.did_balance(bpanel.assign(g=0), ["x1"], **BK)
    early = bpanel.assign(g=np.where(bpanel["g"] > 0, 1, 0))
    with pytest.raises(DataInsufficient, match="Cohort 1 has no pre-treatment"):
        sp.did_balance(early, ["x1"], **BK)
    with pytest.raises(DataInsufficient, match="enough non-missing observations"):
        sp.did_balance(bpanel.assign(x1=np.nan), ["x1"], **BK)
    holed = bpanel.copy()
    holed.loc[holed["i"] == 0, "w"] = np.nan
    with pytest.raises(MethodIncompatibility, match="missing/non-finite values"):
        sp.did_balance(holed, ["x1"], weights="w", **BK)
    with pytest.raises(MethodIncompatibility, match="contains negative values"):
        sp.did_balance(bpanel.assign(w=-1.0), ["x1"], weights="w", **BK)


def test_balance_group_with_zero_total_weight(bpanel):
    # Every treated unit carries weight zero: the weighted treated mean and
    # variance are undefined, the unweighted columns are untouched.
    df = bpanel.assign(w=np.where(bpanel["g"] == 3, 0.0, 1.0))
    res = sp.did_balance(df, ["x1"], cohort=3, weights="w", **BK)
    plain = sp.did_balance(bpanel, ["x1"], cohort=3, **BK)
    assert res.table["w_mean_treated"].isna().all()
    assert np.allclose(res.table["norm_diff"], plain.table["norm_diff"])
    assert np.allclose(res.table["w_mean_comparison"], plain.table["mean_comparison"])
