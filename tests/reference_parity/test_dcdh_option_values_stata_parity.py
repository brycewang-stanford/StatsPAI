"""``sp.did_multiplegt_dyn`` option by option against Stata ``did_multiplegt_dyn``.

Fixture: ``tests/stata_parity/option_parity/results/97_dcdh_options_Stata.json``,
written by ``97_dcdh_options.do`` (Stata 18 MP, ``did_multiplegt_dyn`` from
SSC, double precision) on ``data_97_dcdh_options.csv``
(``_generate_data_97_dcdh_options.py``). The panel has a binary treatment
that moves both ways, never-switchers at both baselines, groups that return
to their period-one treatment, a time-invariant trend cell, a time-varying
weight and covariate, a cluster coarser than the group, rows to drop for an
unbalanced panel, and a second treatment whose period-one level is
continuous.

Earlier fixtures pinned most of these options' point estimates
(``test_multiplegt_dyn_options_parity.py``, ``test_dcdh_options_parity.py``)
or pinned them only together with ``controls=`` or on a count treatment
(``test_dcdh_textbook_stata_parity.py``). Here every option value runs on
its own, and each fit is compared on every effect and placebo, its analytic
standard error, the switcher count, ``Av_tot_eff`` (StatsPAI's
``aggregation='switchers'`` headline) and its standard error, the Wald
p-values of the joint tests, and every entry of the joint covariance of the
horizons (Stata's ``e(V)``; StatsPAI's ``model_info['event_study_vcov']``).
``only_never_switchers`` (``control='never_treated'``) had no reference
before this file.

Index convention: Stata's ``effects(k)`` is ``dynamic=k-1``; its
``Effect_l`` is horizon ``l-1`` and ``Placebo_l`` is horizon ``-l``.

Tolerance: 1e-6 relative on estimates, standard errors and covariances
(T2); 1e-6 absolute on p-values. Observed agreement is about 1e-13.

The ``bootstrap(200, s)`` standard errors at five seeds are a stochastic
screen (S), not parity: Stata's random stream cannot be reproduced. The mean
over five seeds on each side must agree within four pooled Monte Carlo
standard errors, using ``se / sqrt(2 B)`` for one B-draw bootstrap SE.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_OPT = Path(__file__).resolve().parents[1] / "stata_parity" / "option_parity"
RTOL = 1e-6
P_ATOL = 1e-6


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(_OPT / "data_97_dcdh_options.csv")


@pytest.fixture(scope="module")
def ref() -> dict:
    return json.loads(
        (_OPT / "results" / "97_dcdh_options_Stata.json").read_text(encoding="utf-8")
    )


# tag -> (effects, placebos, weighted, keyword arguments)
CASES = {
    "base": (3, 2, False, {"effects_equal": True}),
    # the same fit without the equality test: Stata's base numbers again
    "plain": (3, 2, False, {}),
    "in": (3, 2, False, {"switchers": "in"}),
    "out": (3, 2, False, {"switchers": "out"}),
    "same": (3, 2, False, {"same_switchers": True}),
    "never": (3, 2, False, {"control": "never_treated"}),
    "tnp": (3, 2, False, {"trends_nonparam": ["reg"]}),
    "norm": (3, 2, False, {"normalized": True}),
    "wt": (3, 2, True, {"weights": "w"}),
    "cl": (3, 2, False, {"cluster": "cl"}),
    "ctl": (3, 2, False, {"controls": ["x"]}),
    "e1": (1, 1, False, {}),
    "combo": (
        3,
        2,
        True,
        {
            "trends_nonparam": ["reg"],
            "normalized": True,
            "weights": "w",
            "cluster": "cl",
            "switchers": "in",
        },
    ),
    "cont1": (3, 1, False, {"continuous": 1}),
    "cont2": (3, 1, False, {"continuous": 2}),
    "unb": (3, 2, False, {}),
}


def _fit(panel: pd.DataFrame, tag: str, **extra):
    effects, placebos, _, kwargs = CASES[tag]
    data = panel
    y, treat = "y", "d"
    if tag.startswith("cont"):
        y, treat = "yc", "dc"
    if tag == "unb":
        data = panel[panel["keep"] == 1]
    opts = dict(se_method="analytic", aggregation="switchers")
    opts.update(extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_multiplegt_dyn(
            data,
            y,
            group="g",
            time="t",
            treatment=treat,
            dynamic=effects - 1,
            placebo=placebos,
            **kwargs,
            **opts,
        )


# fixture key prefix of each case, where it is not the case's own tag
_REF_TAG = {"plain": "base"}


def _rt(tag: str) -> str:
    return _REF_TAG.get(tag, tag)


def _horizon(name: str) -> int:
    kind, k = name.rsplit("_", 1)
    return int(k) - 1 if kind == "Effect" else -int(k)


@pytest.fixture(scope="module")
def fits(panel):
    return {tag: _fit(panel, tag) for tag in CASES}


@pytest.mark.parametrize("tag", sorted(CASES))
def test_effects_placebos_and_their_ses_match_stata(fits, ref, tag):
    effects, placebos, weighted, _ = CASES[tag]
    res = fits[tag]
    es = res.model_info["event_study"].set_index("relative_time")
    if weighted:
        # with weight() the command reports the weighted number of switchers
        es["n_switchers"] = res.detail.set_index("horizon")["w_switchers"]
    for ell in range(1, effects + 1):
        row = es.loc[ell - 1]
        assert row["att"] == pytest.approx(ref[f"{_rt(tag)}_effect_{ell}"], rel=RTOL)
        assert row["se"] == pytest.approx(ref[f"{_rt(tag)}_se_effect_{ell}"], rel=RTOL)
        assert row["n_switchers"] == pytest.approx(
            ref[f"{_rt(tag)}_n_switchers_{ell}"], rel=RTOL
        )
    for ell in range(1, placebos + 1):
        row = es.loc[-ell]
        assert row["att"] == pytest.approx(ref[f"{_rt(tag)}_placebo_{ell}"], rel=RTOL)
        assert row["se"] == pytest.approx(ref[f"{_rt(tag)}_se_placebo_{ell}"], rel=RTOL)
        assert row["n_switchers"] == pytest.approx(
            ref[f"{_rt(tag)}_n_switchers_placebo_{ell}"], rel=RTOL
        )


@pytest.mark.parametrize("tag", sorted(CASES))
def test_av_tot_eff_and_its_se_match_stata(fits, ref, tag):
    res = fits[tag]
    assert res.estimate == pytest.approx(ref[f"{_rt(tag)}_av_tot"], rel=RTOL)
    assert res.se == pytest.approx(ref[f"{_rt(tag)}_se_av_tot"], rel=RTOL)


@pytest.mark.parametrize("tag", sorted(CASES))
def test_joint_covariance_matches_stata_e_v(fits, ref, tag):
    """Every entry of e(V) over the effects and placebos, off-diagonals too."""
    vcov = fits[tag].model_info["event_study_vcov"]
    prefix = f"{_rt(tag)}_V_"
    keys = [k for k in ref if k.startswith(prefix)]
    effects, placebos, _, _ = CASES[tag]
    k = effects + placebos
    assert len(keys) == k * (k + 1) // 2
    for key in keys:
        # e.g. "Effect_2_Placebo_1"
        kind_a, la, kind_b, lb = key[len(prefix) :].split("_")
        a, b = _horizon(f"{kind_a}_{la}"), _horizon(f"{kind_b}_{lb}")
        got = float(vcov.loc[a, b])
        assert got == pytest.approx(ref[key], rel=RTOL, abs=1e-15), key


@pytest.mark.parametrize("tag", sorted(CASES))
def test_simple_aggregate_is_built_from_statas_effects_and_e_v(panel, ref, tag):
    """aggregation='simple' has no reference counterpart of its own.

    Its headline is the equal-weight mean of the effects and its standard
    error the matching quadratic form in their joint covariance, so it is
    checked against those two quantities built from Stata's ``Effect_l`` and
    ``e(V)``.
    """
    effects, _, _, _ = CASES[tag]
    res = _fit(panel, tag, aggregation="simple")
    est = [ref[f"{_rt(tag)}_effect_{l}"] for l in range(1, effects + 1)]
    V = np.array(
        [
            [
                ref[f"{_rt(tag)}_V_Effect_{max(a, b)}_Effect_{min(a, b)}"]
                for b in range(1, effects + 1)
            ]
            for a in range(1, effects + 1)
        ]
    )
    w = np.full(effects, 1.0 / effects)
    assert res.estimate == pytest.approx(np.mean(est), rel=RTOL)
    assert res.se == pytest.approx(float(np.sqrt(w @ V @ w)), rel=RTOL)


@pytest.mark.parametrize("tag", sorted(CASES))
def test_joint_tests_match_stata(fits, ref, tag):
    res = fits[tag]
    effects, placebos, _, _ = CASES[tag]
    if effects > 1:
        assert res.model_info["joint_effects_test"]["pvalue"] == pytest.approx(
            ref[f"{_rt(tag)}_p_joint_effects"], abs=P_ATOL
        )
    if placebos > 1:
        assert res.model_info["joint_placebo_test"]["pvalue"] == pytest.approx(
            ref[f"{_rt(tag)}_p_joint_placebo"], abs=P_ATOL
        )


def test_effects_equal_matches_stata(fits, ref):
    test = fits["base"].model_info["effects_equal_test"]
    assert test["df"] == 2
    assert test["pvalue"] == pytest.approx(ref["base_p_equal"], abs=P_ATOL)


def test_the_options_select_different_samples(ref):
    """Guard against a fixture on which the options cannot be told apart."""
    # every option that changes the sample or the outcome moves the estimate;
    # cluster(cl) and effects(1) do not, by construction
    moved = ("in", "out", "same", "never", "tnp", "norm", "wt", "ctl", "unb")
    for tag in moved:
        gaps = [
            abs(ref[f"{tag}_effect_{l}"] - ref[f"base_effect_{l}"]) for l in (1, 2, 3)
        ]
        assert max(gaps) > 1e-6, tag
    assert ref["cl_effect_1"] == ref["base_effect_1"] == ref["e1_effect_1"]
    assert ref["cl_se_effect_1"] != ref["base_se_effect_1"]
    n = {tag: ref[f"{tag}_n_switchers_1"] for tag in ("base", "in", "out")}
    assert n["in"] + n["out"] == n["base"] and n["in"] > 0 and n["out"] > 0


def test_fits_record_their_configuration(fits):
    mi = fits["combo"].model_info
    assert mi["trends_nonparam"] == ["reg"] and mi["normalized"] is True
    assert mi["treatment_kind"] == "binary" and mi["balanced_panel"] is True
    assert fits["cont1"].model_info["treatment_kind"] == "continuous"
    assert fits["unb"].model_info["balanced_panel"] is False
    assert fits["ctl"].model_info["controls"] == ["x"]


# --------------------------------------------------------------------------- #
#  se_method='bootstrap': stochastic screen (S), not parity
# --------------------------------------------------------------------------- #

N_BOOT = 200
SEEDS = (1, 2, 3, 4, 5)


@pytest.fixture(scope="module")
def boot_ses(panel):
    out = []
    for s in SEEDS:
        res = _fit(panel, "base", se_method="bootstrap", n_boot=N_BOOT, seed=s)
        es = res.model_info["event_study"].set_index("relative_time")
        out.append(
            {
                **{f"se_effect_{l}": es.loc[l - 1, "se"] for l in (1, 2, 3)},
                **{f"se_placebo_{l}": es.loc[-l, "se"] for l in (1, 2)},
                "se_av_tot": res.se,
            }
        )
    return pd.DataFrame(out)


@pytest.mark.parametrize(
    "stat",
    [
        "se_effect_1",
        "se_effect_2",
        "se_effect_3",
        "se_placebo_1",
        "se_placebo_2",
        "se_av_tot",
    ],
)
def test_bootstrap_se_screen_against_stata(boot_ses, ref, stat):
    stata = np.array([ref[f"boot{s}_{stat}"] for s in SEEDS], dtype=float)
    ours = boot_ses[stat].to_numpy(dtype=float)
    assert np.isfinite(ours).all() and np.isfinite(stata).all()
    # one B-draw bootstrap SE has Monte Carlo SD of about se / sqrt(2 B)
    pooled = stata.mean() / np.sqrt(2 * N_BOOT) * np.sqrt(2 / len(SEEDS))
    assert abs(ours.mean() - stata.mean()) < 4 * pooled


def test_continuous_drops_cells_on_both_sides_of_the_baseline(panel):
    """Design Restriction 2 applies with continuous= too, as in the reference.

    A group pushed above its period-one treatment and later below it loses
    the cells from the crossing on; before 1.39.7 continuous= kept them.
    """
    twisted = panel.copy()
    path = twisted.pivot(index="g", columns="t", values="dc")
    up_by_6 = path.index[path[6] > path[1]]  # switched up by period 6
    g = int(up_by_6[0])
    rows = twisted["g"] == g
    d1 = float(path.loc[g, 1])
    crossed = rows & (twisted["t"] >= 7)
    twisted.loc[crossed, "dc"] = d1 - 0.25
    kw = dict(
        group="g",
        time="t",
        treatment="dc",
        dynamic=2,
        placebo=1,
        continuous=1,
        se_method="analytic",
        aggregation="switchers",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.did_multiplegt_dyn(twisted, "yc", **kw)
        dropped = sp.did_multiplegt_dyn(twisted[~crossed], "yc", **kw)
    assert res.model_info["n_dropped_bidirectional"] == int(crossed.sum())
    pd.testing.assert_frame_equal(
        res.model_info["event_study"], dropped.model_info["event_study"]
    )
    assert res.estimate == dropped.estimate and res.se == dropped.se
