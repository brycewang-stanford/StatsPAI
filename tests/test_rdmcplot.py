"""``sp.rdmcplot``: one ``rdplot`` per cutoff on a shared axis."""

import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import DataInsufficient, MethodIncompatibility  # noqa: E402


@pytest.fixture(scope="module")
def design():
    rng = np.random.default_rng(11)
    n = 1500
    cut = rng.choice([30.0, 60.0, 90.0], size=n)
    x = cut + rng.uniform(-25, 25, n)
    y = 1 + 0.4 * (x >= cut) + 0.01 * x + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "x": x, "cut": cut})


def test_numbers_are_those_of_rdplot_on_each_subsample(design):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, ax = sp.rdmcplot(design, y="y", x="x", cutoff_var="cut", p=1)
        assert sorted(fig.rdmcplot_data) == [30.0, 60.0, 90.0]
        for c, got in fig.rdmcplot_data.items():
            one, _ = sp.rdplot(
                design[design["cut"] == c], y="y", x="x", c=c, p=1, hide_ci=True
            )
            want = one.rdplot_data
            assert got["J"] == want["J"]
            for key in ("rdplot_mean_bin", "rdplot_mean_y"):
                np.testing.assert_allclose(
                    got["vars_bins"][key], want["vars_bins"][key], rtol=1e-13
                )
            np.testing.assert_allclose(
                got["vars_poly"]["rdplot_y"], want["vars_poly"]["rdplot_y"], rtol=1e-13
            )
    # one dashed line per cutoff, one legend entry per cutoff
    assert len(ax.get_legend().get_texts()) == 3
    plt.close("all")


def test_options_can_differ_by_cutoff(design):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, _ = sp.rdmcplot(
            design,
            y="y",
            x="x",
            cutoff_var="cut",
            cutoffs=[30, 60],
            p=[1, 2],
            nbins=[5, 8],
            hide_ci=False,
        )
    assert sorted(fig.rdmcplot_data) == [30.0, 60.0]
    assert list(fig.rdmcplot_data[30.0]["J"]) == [5, 5]
    assert list(fig.rdmcplot_data[60.0]["J"]) == [8, 8]
    plt.close("all")


def test_option_lists_must_match_the_cutoffs(design):
    with pytest.raises(MethodIncompatibility, match="for 3 cutoffs"):
        sp.rdmcplot(design, y="y", x="x", cutoff_var="cut", p=[1, 2])
    plt.close("all")


def test_a_cutoff_with_one_empty_side_is_refused(design):
    bad = design[(design["cut"] != 30.0) | (design["x"] >= 30.0)]
    with pytest.raises(DataInsufficient, match="cutoff 30"):
        sp.rdmcplot(bad, y="y", x="x", cutoff_var="cut")
    plt.close("all")
