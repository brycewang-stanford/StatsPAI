"""``rdrandinf`` / ``rdwinselect`` / ``rdmc`` and ``rddensity``'s binomial
options in ``sp.from_stata``.

The commands and option spellings are those of the replication files of
Cattaneo, Idrobo and Titiunik (2024); the rules are the ones every
translation follows: an option is carried over, reported as untranslated,
or listed as display-only.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _t(command):
    return sp.from_stata(command)


def test_rdrandinf_window_and_cutoff_are_carried_over():
    out = _t("rdrandinf Y X, c(1) wl(0.2348) wr(1.7652) seed(50)")
    assert out["ok"] and out["tool"] == "rdrandinf"
    assert out["arguments"] == {
        "y": "Y",
        "x": "X",
        "c": 1.0,
        "wl": 0.2348,
        "wr": 1.7652,
        "seed": 50,
        "ci": False,
    }
    assert out["untranslated_options"] == []
    assert any("not Stata's" in n for n in out["notes"])


def test_rdrandinf_c_is_cutoff():
    a = _t("rdrandinf Y X, c(1) wl(0) wr(2)")
    b = _t("rdrandinf Y X, cutoff(1) wl(0) wr(2)")
    assert a["arguments"] == b["arguments"]


def test_rdrandinf_ci_numlist_becomes_alpha_and_a_grid():
    out = _t("rdrandinf Y X, wl(-1) wr(1) ci(0.10 -1(0.5)1) d(7.414)")
    assert out["arguments"]["alpha"] == 0.10
    assert out["arguments"]["ci"] == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert out["arguments"]["d"] == 7.414


def test_rdrandinf_without_ci_asks_for_no_interval():
    """Stata computes an interval only when ``ci()`` is given."""
    assert _t("rdrandinf Y X, wl(-1) wr(1)")["arguments"]["ci"] is False


@pytest.mark.parametrize(
    "spec, stat", [("D", "itt"), ("D ar", "itt"), ("D itt", "itt"), ("D tsls", "tsls")]
)
def test_rdrandinf_fuzzy_statistic(spec, stat):
    out = _t(f"rdrandinf Y X, wl(-1) wr(1) fuzzy({spec})")
    assert out["arguments"]["fuzzy"] == "D"
    assert out["arguments"]["fuzzy_stat"] == stat


def test_rdrandinf_bernoulli_and_options_keep_their_names():
    out = _t(
        "rdrandinf Y X, wl(-1) wr(1) bernoulli(pr) statistic(ranksum) reps(500) "
        "nulltau(2)"
    )
    args = out["arguments"]
    assert args["bernoulli"] == "pr"
    assert args["statistic"] == "ranksum"
    assert args["n_perms"] == 500
    assert args["nulltau"] == 2.0


def test_rdrandinf_unknown_option_is_reported():
    out = _t("rdrandinf Y X, wl(-1) wr(1) firststage")
    assert out["untranslated_options"] == ["firststage"]


@pytest.mark.parametrize(
    "command", ["rdrandinf Y X, wl(-1) wr(1)", "rdwinselect X Z, approx"]
)
def test_vce_is_carried_over_with_polynomial_adjustment(command):
    """rdlocrand 3.0: vce(hc1 | hc2 | hc3), any case, for p() > 0."""
    out = _t(command + " p(1) vce(HC2)")
    assert out["arguments"]["vce"] == "hc2" and out["arguments"]["p"] == 1
    assert out["untranslated_options"] == []
    assert not any("HC3" in n for n in out["notes"])


@pytest.mark.parametrize(
    "command", ["rdrandinf Y X, wl(-1) wr(1)", "rdwinselect X Z, approx"]
)
def test_p_without_vce_says_which_release_used_which_default(command):
    out = _t(command + " p(1)")
    assert "vce" not in out["arguments"]
    assert any("HC3" in n and "vce(hc2)" in n for n in out["notes"])
    assert not any("HC3" in n for n in _t(command + " wmin(1)")["notes"])


def test_vce_that_is_not_an_hc_estimator_is_reported():
    out = _t("rdrandinf Y X, wl(-1) wr(1) p(1) vce(cluster id)")
    assert out["untranslated_options"] == ["vce"]


def test_rdrandinf_interference_interval():
    out = _t("rdrandinf Y X, wl(-1) wr(1) interfci(0.05)")
    assert out["arguments"]["interfci"] == 0.05
    assert out["untranslated_options"] == []


def test_rdrandinf_without_a_window_is_refused():
    out = _t("rdrandinf Y X, covariates(a b)")
    assert not out["ok"]
    assert "rdwinselect" in (out.get("error") or out.get("message") or "")


def test_rdwinselect_covariates_and_renames():
    out = _t(
        "rdwinselect X a b, cutoff(0.00005) wmin(0.01) wstep(0.01) seed(50) "
        "level(0.135)"
    )
    assert out["arguments"] == {
        "x": "X",
        "covs": ["a", "b"],
        "c": 5e-05,
        "wmin": 0.01,
        "wstep": 0.01,
        "alpha": 0.135,
        "seed": 50,
    }


def test_rdwinselect_flags_and_display_options():
    out = _t("rdwinselect X a, seed(50) wobs(2) nwindows(200) plot approx")
    assert out["arguments"]["approx"] is True
    assert out["arguments"]["wobs"] == 2 and out["arguments"]["nwindows"] == 200
    assert out["ignored_display_options"] == ["plot"]
    assert out["untranslated_options"] == []


def test_rdwinselect_mass_point_windows():
    out = _t("rdwinselect X a, wmasspoints")
    assert out["arguments"]["wmasspoints"] is True
    assert out["untranslated_options"] == []
    assert _t("rdwinselect X a, evalat(means)")["untranslated_options"] == ["evalat"]


def test_rdms_needs_the_boundary_points_from_the_data():
    """``cvar()`` names variables; one line of Stata does not hold the points."""
    out = _t("rdms y a b tr, cvar(p1 p2)")
    assert not out["ok"]
    assert "sp.stata reads them from the data" in out["error"]
    out = _t("rdms y a b tr, c(p1 p2) xnorm(xn) cutoff1(0 30 0) cutoff2(0 0 50)")
    assert out["arguments"] == {
        "y": "y",
        "x1": "a",
        "x2": "b",
        "treat": "tr",
        "cutoff1": [0.0, 30.0, 0.0],
        "cutoff2": [0.0, 0.0, 50.0],
        "xnorm": "xn",
    }


def test_sp_stata_reads_rdms_boundary_points_from_the_data():
    rng = np.random.default_rng(5)
    n = 3000
    a, b = rng.uniform(-50, 80, n), rng.uniform(-60, 90, n)
    tr = ((a >= 0) & (b >= 0)).astype(float)
    df = pd.DataFrame(
        {"a": a, "b": b, "tr": tr, "y": 1 + 0.6 * tr + rng.normal(0, 0.5, n)}
    )
    df["p1"] = np.nan
    df["p2"] = np.nan
    df.loc[[0, 1], "p1"] = [0.0, 30.0]
    df.loc[[0, 1], "p2"] = [0.0, 0.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata("rdms y a b tr, cvar(p1 p2)", data=df)
        direct = sp.rdms(
            df, y="y", x1="a", x2="b", treat="tr", cutoff1=[0, 30], cutoff2=[0, 0]
        )
    assert [cr["cutoff"] for cr in res.cutoff_results] == [(0.0, 0.0), (30.0, 0.0)]
    assert [cr["estimate"] for cr in res.cutoff_results] == [
        cr["estimate"] for cr in direct.cutoff_results
    ]


def test_rdmc_c_is_the_cutoff_variable():
    out = _t("rdmc y x, c(cutoff)")
    assert out["arguments"] == {"y": "y", "x": "x", "cutoff_var": "cutoff"}
    assert _t("rdmc y x, cvar(cutoff)")["arguments"] == out["arguments"]
    assert _t("rdmc y x, c(cutoff) pooled_opt(h(20))")["untranslated_options"] == [
        "pooled_opt"
    ]
    assert not _t("rdmc y x")["ok"]


def test_rddensity_binomial_options():
    out = _t("rddensity X, bino_w(0.13) bino_nw(1)")
    assert out["arguments"]["bino_w"] == 0.13
    assert out["arguments"]["bino_nw"] == 1
    assert out["untranslated_options"] == []
    out = _t("rddensity X, nobinomial")
    assert out["ignored_display_options"] == ["nobinomial"]


def test_translated_commands_run():
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 3, 600)
    df = pd.DataFrame(
        {"x": x, "y": 1.0 * (x >= 1) + rng.normal(size=600), "z": rng.normal(size=600)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata("rdrandinf y x, c(1) wl(0.5) wr(1.5) seed(1)", data=df)
        win = sp.stata("rdwinselect x z, c(1) approx", data=df)
        with pytest.raises(MethodIncompatibility):
            sp.stata("rdrandinf y x, wl(0.5) wr(1.5) firststage", data=df)
    assert res.model_info["window"] == (0.5, 1.5)
    assert np.isnan(res.ci[0])
    assert {"variable", "binom_pvalue"} <= set(win.columns)


def test_rdms_one_score_cumulative_cutoffs():
    out = _t(
        "rdms y x, cvar(c) range(lo hi) cutoff1(33 66) range1(10 40) range2(50 90)"
    )
    assert out["arguments"] == {
        "y": "y",
        "x1": "x",
        "cutoff1": [33.0, 66.0],
        "ranges": [(10.0, 50.0), (40.0, 90.0)],
    }
    assert not _t("rdms y x, cvar(c)")["ok"]
    assert not _t("rdms y x z, cvar(c)")["ok"]


def test_sp_stata_reads_cumulative_cutoffs_and_ranges_from_the_data():
    rng = np.random.default_rng(7)
    n = 2500
    x = rng.uniform(0, 100, n)
    y = 1 + 0.5 * (x >= 33) + 0.8 * (x >= 66) + rng.normal(0, 0.4, n)
    df = pd.DataFrame({"y": y, "x": x})
    for name, values in (("c", [33, 66]), ("lo", [10, 40]), ("hi", [50, 90])):
        df[name] = np.nan
        df.loc[[0, 1], name] = values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata("rdms y x, cvar(c) range(lo hi)", data=df)
        direct = sp.rdms(
            df, y="y", x1="x", cutoff1=[33, 66], ranges=[(10, 50), (40, 90)]
        )
    assert [cr["estimate"] for cr in res.cutoff_results] == [
        cr["estimate"] for cr in direct.cutoff_results
    ]


def test_rdmcplot_per_cutoff_options_come_from_the_data():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = _t("rdmcplot y x, c(cut) pvar(p) nbinsvar(nl) nbinsrightvar(nr)")
    assert out["untranslated_options"] == ["pvar", "nbinsvar", "nbinsrightvar"]
    out = _t(
        "rdmcplot y x, c(cut) pvar(p) nbinsvar(nl) nbinsrightvar(nr) "
        "pvec(1 1) nbinsvec(6 7) nbinsrightvec(8 9) nodraw"
    )
    assert out["arguments"] == {
        "y": "y",
        "x": "x",
        "cutoff_var": "cut",
        "p": [1, 1],
        "nbins": [(6, 8), (7, 9)],
    }
    assert out["ignored_display_options"] == ["nodraw"]

    rng = np.random.default_rng(2)
    n = 1200
    cut = rng.choice([30.0, 60.0], size=n)
    x = cut + rng.uniform(-20, 20, n)
    df = pd.DataFrame(
        {"y": 1 + 0.4 * (x >= cut) + rng.normal(0, 0.3, n), "x": x, "cut": cut}
    )
    for name, values in (("p", [1, 2]), ("nl", [6, 7]), ("nr", [8, 9])):
        df[name] = np.nan
        df.loc[[0, 1], name] = values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, _ = sp.stata(
            "rdmcplot y x, c(cut) pvar(p) nbinsvar(nl) nbinsrightvar(nr)", data=df
        )
        with pytest.raises(MethodIncompatibility):
            sp.stata("rdmcplot y x, c(cut) genvars", data=df)
    assert list(fig.rdmcplot_data[30.0]["J"]) == [6, 8]
    assert list(fig.rdmcplot_data[60.0]["J"]) == [7, 9]
    plt.close("all")
