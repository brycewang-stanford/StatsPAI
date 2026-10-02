"""``sp.difference_in_means`` and the CR2 degrees of freedom, against R.

Reference: R ``estimatr`` 2.0.0 (and ``clubSandwich`` for the degrees of
freedom) on the experiment built by :func:`experiment` below, written to a
CSV with 17 significant digits::

    difference_in_means(y ~ d, data = d)
    difference_in_means(y ~ d, blocks = block, data = d)
    difference_in_means(yc ~ dc, clusters = cl, data = d)
    difference_in_means(ybc ~ dbc, blocks = block, clusters = bcl, data = d)
    difference_in_means(yp ~ dp, blocks = pair, data = d)
    difference_in_means(ycp ~ dcp, blocks = cpair, clusters = cl, data = d)
    lm_robust(yc ~ dc + x + factor(block), clusters = cl, se_type = "CR2")

The data are deterministic (no random numbers): 660 units in 11 blocks of
unequal size with unequal treated shares, 110 clusters of six units, 132
clusters nested in the blocks, 330 pairs of units and 55 pairs of clusters.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

# rtol: closed-form estimators on the same numbers. Loosest observed 9e-12.
RTOL = 1e-10


def experiment() -> pd.DataFrame:
    sizes = [20, 35, 50, 65, 80, 30, 45, 60, 75, 90, 110]
    shares = [0.2, 0.3, 0.4, 0.5, 0.6, 0.25, 0.35, 0.45, 0.55, 0.3, 0.5]
    rows = []
    i = 0
    for b, (nb, sh) in enumerate(zip(sizes, shares)):
        n1 = int(round(sh * nb))
        u = (np.sin((np.arange(nb) + 1 + 100 * b) * 12.9898) * 43758.5453) % 1
        rank = np.argsort(np.argsort(u))
        for j in range(nb):
            i += 1
            d = int(rank[j] < n1)
            x = np.cos(i * 0.37) + 0.1 * b
            y = (
                1
                + 0.5 * b
                + (1.0 + 0.2 * b) * d
                + 0.8 * x
                + 1.5 * np.sin(i * 78.233) * (1 + 0.5 * d)
            )
            rows.append((i, b, d, x, y))
    df = pd.DataFrame(rows, columns=["i", "block", "d", "x", "y"])
    df["cl"] = (df["i"] - 1) // 6
    ncl = int(df["cl"].max()) + 1
    cu = (np.sin(np.arange(ncl) * 4.1) * 1000) % 1
    df["dc"] = (cu[df["cl"]] < 0.45).astype(int)
    df["yc"] = (
        2
        + 1.2 * df["dc"]
        + 0.8 * df["x"]
        + np.sin(df["cl"] * 3.3)
        + 1.2 * np.sin(df["i"] * 78.233)
    )
    within = df.groupby("block").cumcount()
    df["bcl"] = df["block"] * 1000 + within // 5
    key = df["bcl"].to_numpy()
    cu2 = (np.sin(key * 1.7) * 4000) % 1
    med = pd.Series(cu2).groupby(df["block"].to_numpy()).transform("median").to_numpy()
    df["dbc"] = (cu2 < med).astype(int)
    df["ybc"] = (
        1
        + 0.4 * df["block"]
        + 1.5 * df["dbc"]
        + np.sin(key * 2.9)
        + 1.1 * np.sin(df["i"] * 12.345)
    )
    df["pair"] = (df["i"] - 1) // 2
    flip = ((np.sin(df["pair"] * 5.3) * 100) % 1 < 0.5).astype(int)
    df["dp"] = ((df["i"] - 1) % 2 == flip).astype(int)
    df["yp"] = 0.02 * df["pair"] + 0.9 * df["dp"] + np.sin(df["i"] * 3.21)
    df["cpair"] = df["cl"] // 2
    cflip = ((np.sin(df["cpair"] * 7.7) * 100) % 1 < 0.5).astype(int)
    df["dcp"] = (df["cl"] % 2 == cflip).astype(int)
    df["ycp"] = (
        0.05 * df["cpair"]
        + 1.1 * df["dcp"]
        + np.sin(df["cl"] * 3.3)
        + np.sin(df["i"] * 9.87)
    )
    return df


# label: (kwargs, design, estimate, se, df, ci lower, ci upper)
CASES = {
    "standard": (
        dict(y="y", treat="d"),
        "Standard",
        2.370179514384,
        0.202731087401,
        527.0478520589,
        1.971919317330,
        2.768439711437,
    ),
    "blocked": (
        dict(y="y", treat="d", blocks="block"),
        "Blocked",
        2.231553014807,
        0.118742625541,
        638.0,
        1.998379400841,
        2.464726628772,
    ),
    "clustered": (
        dict(y="yc", treat="dc", cluster="cl"),
        "Clustered",
        1.429560626163,
        0.170887530845,
        106.7025226603,
        1.090785226020,
        1.768336026307,
    ),
    "block_clustered": (
        dict(y="ybc", treat="dbc", blocks="block", cluster="bcl"),
        "Block-clustered",
        1.340416067313,
        0.187990135275,
        110.0,
        0.967863743843,
        1.712968390784,
    ),
    "matched_pair": (
        dict(y="yp", treat="dp", blocks="pair"),
        "Matched-pair",
        0.814777789478,
        0.077517522935,
        329.0,
        0.662285265389,
        0.967270313567,
    ),
    "matched_pair_clustered": (
        dict(y="ycp", treat="dcp", blocks="cpair", cluster="cl"),
        "Matched-pair clustered",
        1.395765214865,
        0.190513475384,
        54.0,
        1.013808693946,
        1.777721735784,
    ),
}


@pytest.fixture(scope="module")
def df():
    return experiment()


@pytest.mark.parametrize("label", sorted(CASES))
def test_every_design_matches_estimatr(df, label):
    kwargs, design, est, se, dof, lo, hi = CASES[label]
    res = sp.difference_in_means(df, **kwargs)
    assert res.model_info["design"] == design
    assert res.estimate == pytest.approx(est, rel=RTOL)
    assert res.se == pytest.approx(se, rel=RTOL)
    assert res.model_info["df"] == pytest.approx(dof, rel=RTOL)
    assert res.ci[0] == pytest.approx(lo, rel=RTOL)
    assert res.ci[1] == pytest.approx(hi, rel=RTOL)


def test_p_value_and_level(df):
    res = sp.difference_in_means(df, "y", "d", blocks="block")
    # estimatr: p = 4.86169292959e-63
    assert res.pvalue == pytest.approx(4.86169292959e-63, rel=1e-8)
    wide = sp.difference_in_means(df, "y", "d", alpha=0.01)
    # estimatr, alpha = 0.01
    assert wide.ci[0] == pytest.approx(1.846081204101, rel=RTOL)
    assert wide.ci[1] == pytest.approx(2.894277824666, rel=RTOL)


def test_a_subset_of_blocks(df):
    # difference_in_means(y ~ d, blocks = block, data = d, subset = block < 4)
    res = sp.difference_in_means(df[df["block"] < 4], "y", "d", blocks="block")
    assert res.estimate == pytest.approx(1.355460724987, rel=RTOL)
    assert res.se == pytest.approx(0.230180059573, rel=RTOL)
    assert res.model_info["df"] == 162.0


def test_standard_design_is_the_unequal_variance_t_test(df):
    res = sp.difference_in_means(df, "y", "d")
    tt = sp.ttest(df, "y", by="d", unequal=True)
    # sp.ttest reports mean(lower group) - mean(higher group).
    assert res.estimate == pytest.approx(-tt.estimate, rel=1e-12)
    assert res.se == pytest.approx(tt.se, rel=1e-12)
    assert res.model_info["df"] == pytest.approx(tt.df, rel=1e-12)


def test_att_weights_blocks_by_their_treated_units(df):
    res = sp.difference_in_means(df, "y", "d", blocks="block", estimand="ATT")
    est = var = 0.0
    n_treated = df["d"].sum()
    for _, g in df.groupby("block"):
        y1, y0 = g.loc[g.d == 1, "y"], g.loc[g.d == 0, "y"]
        w = len(y1) / n_treated
        est += w * (y1.mean() - y0.mean())
        var += w**2 * (y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    assert res.estimate == pytest.approx(est, rel=1e-12)
    assert res.se == pytest.approx(np.sqrt(var), rel=1e-12)
    assert res.estimand == "ATT"
    ate = sp.difference_in_means(df, "y", "d", blocks="block")
    # effects grow with the block index and so do the treated shares
    assert res.estimate != pytest.approx(ate.estimate, rel=1e-3)
    assert res.detail["weight"].sum() == pytest.approx(1.0, abs=1e-12)


def test_block_table(df):
    res = sp.difference_in_means(df, "y", "d", blocks="block")
    table = res.detail
    assert list(table["n"]) == [20, 35, 50, 65, 80, 30, 45, 60, 75, 90, 110]
    assert float(table["weight"] @ table["estimate"]) == pytest.approx(
        res.estimate, rel=1e-12
    )
    assert float(np.sqrt((table["weight"] ** 2 * table["se"] ** 2).sum())) == (
        pytest.approx(res.se, rel=1e-12)
    )


def test_treatment_labels_need_not_be_zero_and_one(df):
    labelled = df.assign(arm=np.where(df["d"] == 1, "small", "regular"))
    # 'small' sorts after 'regular', so it is the treated level
    res = sp.difference_in_means(labelled, "y", "arm", blocks="block")
    assert res.estimate == pytest.approx(CASES["blocked"][2], rel=RTOL)
    assert res.model_info["treat_levels"] == ("regular", "small")


def test_missing_rows_are_dropped_with_a_warning(df):
    holes = df.copy()
    holes.loc[holes.index[:7], "y"] = np.nan
    with pytest.warns(UserWarning, match="dropped 7 rows"):
        res = sp.difference_in_means(holes, "y", "d")
    assert res.n_obs == len(df) - 7
    assert res.model_info["n_dropped"] == 7


def test_designs_that_cannot_be_analysed_raise(df):
    with pytest.raises(MethodIncompatibility, match="exactly two values"):
        sp.difference_in_means(df, "y", "block")
    with pytest.raises(MethodIncompatibility, match="both treated and control"):
        sp.difference_in_means(df, "y", "d", cluster="cl")
    with pytest.raises(MethodIncompatibility, match="span more than one block"):
        sp.difference_in_means(df, "yc", "dc", blocks="block", cluster="cl")
    with pytest.raises(MethodIncompatibility, match="estimand"):
        sp.difference_in_means(df, "y", "d", estimand="LATE")
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.difference_in_means(df, "y", "nope")
    with pytest.raises(MethodIncompatibility, match="ATT"):
        sp.difference_in_means(
            df, "ycp", "dcp", blocks="cpair", cluster="cl", estimand="ATT"
        )
    # one block with a single treated unit is neither a pair nor estimable
    thin = df.copy()
    first = thin.index[thin["block"] == 0]
    thin.loc[first, "d"] = 0
    thin.loc[first[0], "d"] = 1
    with pytest.raises(DataInsufficient, match="fewer than two"):
        sp.difference_in_means(thin, "y", "d", blocks="block")
    thin.loc[first[0], "d"] = 0
    with pytest.raises(DataInsufficient, match="no treated or no control"):
        sp.difference_in_means(thin, "y", "d", blocks="block")


# lm_robust(yc ~ dc + x + factor(block), clusters = cl, se_type = "CR2"):
# name: (std.error, df, conf.low). clubSandwich::coef_test gives the same df.
CR2 = {
    "Intercept": (0.144982733792, 2.6464149911, 1.493252386093),
    "dc": (0.152543875087, 79.1058574304, 0.937585113898),
    "x": (0.084169509345, 92.5290485122, 0.641297389635),
    "C(block)[T.3]": (0.229738009425, 4.3527643374, -0.639434028986),
}


@pytest.mark.parametrize("clustered_base", [False, True])
def test_cr2_degrees_of_freedom_match_estimatr(df, clustered_base):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sp.regress(
            "yc ~ dc + x + C(block)", df, cluster="cl" if clustered_base else None
        )
        fit = sp.cr2_se(base, df, cluster="cl")
    lower = fit.conf_int().iloc[:, 0]
    for name, (se, dof, lo) in CR2.items():
        assert fit.std_errors[name] == pytest.approx(se, rel=RTOL)
        assert fit.diagnostics["satterthwaite_dof"][name] == pytest.approx(
            dof, rel=RTOL
        )
        # each coefficient is referred to its own t distribution
        assert lower[name] == pytest.approx(lo, rel=1e-9)
    assert fit.tidy().set_index("term").loc["dc", "conf_low"] == pytest.approx(
        CR2["dc"][2], rel=1e-9
    )


def test_cr2_two_arm_degrees_of_freedom(df):
    # lm_robust(yc ~ dc, clusters = cl, se_type = "CR2")$df: 51, 106.7025226603
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.cr2_se(sp.regress("yc ~ dc", df), df, cluster="cl")
    dof = fit.diagnostics["satterthwaite_dof"]
    assert dof["Intercept"] == pytest.approx(51.0, rel=RTOL)
    assert dof["dc"] == pytest.approx(106.7025226603, rel=RTOL)
