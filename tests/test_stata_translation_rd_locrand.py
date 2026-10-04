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
    out = _t("rdrandinf Y X, wl(-1) wr(1) interfci(0.05)")
    assert out["untranslated_options"] == ["interfci"]


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


def test_rdwinselect_mass_point_windows_are_not_translated():
    assert _t("rdwinselect X a, wmasspoints")["untranslated_options"] == ["wmasspoints"]


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
            sp.stata("rdrandinf y x, wl(0.5) wr(1.5) interfci(0.05)", data=df)
    assert res.model_info["window"] == (0.5, 1.5)
    assert np.isnan(res.ci[0])
    assert {"variable", "binom_pvalue"} <= set(win.columns)
