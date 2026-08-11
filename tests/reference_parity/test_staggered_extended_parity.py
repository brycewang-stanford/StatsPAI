"""Reference parity: the extended ``sp.staggered_rollout`` surface vs R ``staggered``.

``test_staggered_rollout_parity.py`` pins the original three estimands. This
file pins everything added on top of them, all against R ``staggered`` 1.2.2:

* the **adjusted** (non-conservative) standard error — R's primary ``se``,
  which subtracts the part of the variance random adoption timing identifies;
* ``use_last_treated_only``, the Sun-Abraham comparison group;
* ``estimand='eventstudy'`` at every feasible event time, scalar and vector,
  including the joint covariance across event times;
* ``sp.staggered_cs`` / ``sp.staggered_sa``;
* the Fisher randomisation test;
* the two behavioural edge cases (singleton cohorts, units treated at the
  first period).

Reference generation
--------------------
``Rscript tests/reference_parity/_generate_staggered_extended_R.R`` writes
``_fixtures/staggered_extended_reference.json`` plus the panels it uses. Three
datasets appear:

``mpdta``
    The canonical locked fixture. Has never-treated units (``g = Inf``).
``rollout``
    A genuinely *randomised* rollout with **no** never-treated units, so
    ``max(g)`` is finite — a branch mpdta never reaches.
``nullpanel``
    The same design with the treatment effect switched off. A randomisation
    test on a panel with a real effect returns ``p = 0`` for any
    implementation and therefore pins nothing; the null panel puts the
    p-value in the interior where an error would show.

On the randomisation test
-------------------------
R draws its permutations with an internal ``set.seed(k)`` per draw, which no
other language reproduces bit-for-bit. Rather than pretend otherwise, parity
is pinned in two independent pieces:

1. ``test_fisher_matches_r_draw_for_draw`` replays 40 permutations *generated
   by R* through the Python estimator and requires agreement to 1e-9. This
   pins the estimator under permutation exactly — the part that can actually
   be wrong.
2. ``test_fisher_pvalue_matches_r_within_monte_carlo_error`` compares the
   end-to-end p-values, which are two independent Monte-Carlo estimates of
   the same quantity and so can only agree up to simulation error.

References
----------
Roth, J. and Sant'Anna, P.H.C. (2023). "Efficient Estimation for Staggered
Rollout Designs." *Journal of Political Economy Microeconomics*, 1(4),
669-709. [@roth2023efficient]
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._staggered_rollout import (
    _fit,
    _summarize,
    _weight_matrices,
    _wide_panel,
    staggered_rollout_core,
)

_HERE = pathlib.Path(__file__).resolve().parent
_FIX = _HERE / "_fixtures"
_REF_PATH = _FIX / "staggered_extended_reference.json"
_MPDTA = _HERE.parent / "orig_parity" / "data" / "02_mpdta_original.csv"

# Agreement demanded of every deterministic quantity. R prints ~15 significant
# digits; 1e-9 is far tighter than any numerically meaningful difference and
# far looser than the ~1e-16 the implementation actually achieves.
_ATOL = 1e-9


def _load_ref() -> dict:
    if not _REF_PATH.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing reference fixture: {_REF_PATH}")
    return json.loads(_REF_PATH.read_text(encoding="utf-8"))


_REF = _load_ref()


def _panel(name: str) -> dict:
    """Dataset kwargs for :func:`staggered_rollout_core` by fixture name."""
    if name == "mpdta":
        return dict(
            data=pd.read_csv(_MPDTA),
            i="countyreal",
            t="year",
            g="first_treat",
            y="lemp",
        )
    path = _FIX / f"staggered_{'null' if name == 'nullpanel' else name}_panel.csv"
    if not path.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing panel fixture: {path}")
    return dict(data=pd.read_csv(path), i="unit", t="time", g="first_treat", y="y")


@pytest.fixture(scope="module")
def panels() -> dict:
    return {n: _panel(n) for n in ("mpdta", "rollout", "nullpanel")}


# --------------------------------------------------------------------------
# estimand x beta x comparison group
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_REF["grid"]))
def test_grid_matches_r(panels, key):
    """Every estimand x efficient/plug-in x comparison-group cell."""
    dataset, estimand, tag, controls = key.split("|")
    ref = _REF["grid"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        core = staggered_rollout_core(
            **panels[dataset],
            estimand=estimand,
            efficient=(tag == "efficient"),
            use_last_treated_only=(controls == "last"),
        )
    assert core.estimate == pytest.approx(ref["estimate"], abs=_ATOL), key
    assert core.se_neyman == pytest.approx(ref["se_neyman"], abs=_ATOL), key
    # R names the adjusted (non-conservative) SE simply `se`.
    assert core.se_adjusted == pytest.approx(ref["se"], abs=_ATOL), key


def test_adjusted_se_never_exceeds_the_conservative_one(panels):
    """The randomisation adjustment can only remove variance, never add it."""
    for dataset in ("mpdta", "rollout", "nullpanel"):
        for estimand in ("simple", "cohort", "calendar"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                core = staggered_rollout_core(**panels[dataset], estimand=estimand)
            assert core.se_adjusted <= core.se_neyman + 1e-12, (dataset, estimand)


def test_se_type_selects_which_se_is_reported(panels):
    """``se_type`` moves ``.se`` between the two, leaving both available."""
    kwargs = dict(y="y", i="unit", t="time", g="first_treat", estimand="simple")
    data = panels["rollout"]["data"]
    neyman = sp.staggered_rollout(data, se_type="neyman", **kwargs)
    adjusted = sp.staggered_rollout(data, se_type="adjusted", **kwargs)

    assert neyman.se == pytest.approx(neyman.model_info["se_neyman"], abs=1e-15)
    assert adjusted.se == pytest.approx(adjusted.model_info["se_adjusted"], abs=1e-15)
    assert adjusted.se < neyman.se
    # Both are always carried, whichever one was asked for.
    for res in (neyman, adjusted):
        assert res.model_info["se_neyman"] > 0
        assert res.model_info["se_adjusted"] > 0


def test_se_type_is_validated(panels):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="se_type"):
        staggered_rollout_core(**panels["rollout"], se_type="robust")


# --------------------------------------------------------------------------
# the general control set (use_did_a0=False)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_REF.get("general_a0", {})))
def test_general_control_set_matches_r(panels, key):
    """Every pre-period as a control, not just the ``g - 1`` DiD contrast."""
    dataset, estimand, tag = key.split("|")
    ref = _REF["general_a0"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        core = staggered_rollout_core(
            **panels[dataset],
            estimand=estimand,
            efficient=(tag == "efficient"),
            use_did_a0=False,
        )
    assert core.estimate == pytest.approx(ref["estimate"], abs=_ATOL), key
    assert core.se_neyman == pytest.approx(ref["se_neyman"], abs=_ATOL), key
    assert core.se_adjusted == pytest.approx(ref["se"], abs=_ATOL), key


def test_general_control_set_fits_a_beta_vector(panels):
    """The DiD control set gives one weight; the general one gives many."""
    did = staggered_rollout_core(**panels["rollout"], estimand="simple")
    general = staggered_rollout_core(
        **panels["rollout"], estimand="simple", use_did_a0=False
    )
    assert did.beta.size == 1
    assert general.beta.size > 1
    # More controls cannot make the efficient estimator less precise.
    assert general.se_neyman <= did.se_neyman + 1e-12


def test_general_control_set_rejects_the_plug_in(panels):
    """``beta = 1`` is a single contrast and is undefined against a vector.

    R errors here too. Computing something anyway would invent an estimator
    with no reference implementation behind it.
    """
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="plug-in"):
        staggered_rollout_core(**panels["rollout"], efficient=False, use_did_a0=False)


# --------------------------------------------------------------------------
# event study
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_REF["eventstudy"]))
def test_event_study_matches_r(panels, key):
    """Every feasible event time, efficient and plug-in."""
    dataset, _, event_time, tag = key.split("|")
    ref = _REF["eventstudy"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        core = staggered_rollout_core(
            **panels[dataset],
            estimand="eventstudy",
            event_time=float(event_time),
            efficient=(tag == "efficient"),
        )
    assert core.estimate == pytest.approx(ref["estimate"], abs=_ATOL), key
    assert core.se_neyman == pytest.approx(ref["se_neyman"], abs=_ATOL), key
    assert core.se_adjusted == pytest.approx(ref["se"], abs=_ATOL), key


def test_event_time_minus_one_is_mechanically_zero(panels):
    """At ``e = -1`` outcome and control weights coincide, so the estimate is 0.

    The efficient weights then solve to ``beta = 1`` exactly, which cancels the
    estimate. It is a useful invariant: any drift in the weight construction
    breaks it immediately.
    """
    core = staggered_rollout_core(
        **panels["mpdta"], estimand="eventstudy", event_time=-1.0
    )
    assert core.estimate == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(core.beta, 1.0, atol=1e-9)


def test_infeasible_event_time_fails_loudly(panels):
    with pytest.raises(sp.exceptions.DataInsufficient):
        staggered_rollout_core(
            **panels["rollout"], estimand="eventstudy", event_time=25.0
        )


@pytest.mark.parametrize("key", sorted(_REF["eventstudy_vcv"]))
def test_event_study_vector_and_joint_vcv_match_r(panels, key):
    """A vector of event times reproduces R's per-row fit *and* its full vcv."""
    dataset, tag = key.split("|")
    ref = _REF["eventstudy_vcv"][key]
    event_time = [float(e) for e in ref["event_time"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.staggered_rollout(
            panels[dataset]["data"],
            y=panels[dataset]["y"],
            i=panels[dataset]["i"],
            t=panels[dataset]["t"],
            g=panels[dataset]["g"],
            estimand="eventstudy",
            event_time=event_time,
            efficient=(tag == "efficient"),
        )
    detail = res.detail
    assert list(detail["event_time"]) == event_time
    for ref_field, got_field in (
        ("estimate", "estimate"),
        ("se_neyman", "se_neyman"),
        ("se", "se_adjusted"),
    ):
        np.testing.assert_allclose(
            detail[got_field].to_numpy(),
            np.asarray(ref[ref_field], dtype=float),
            atol=_ATOL,
            err_msg=f"{key}:{ref_field}",
        )
    np.testing.assert_allclose(
        np.asarray(res.model_info["vcov_neyman"], dtype=float),
        np.asarray(ref["vcv_neyman"], dtype=float),
        atol=1e-11,
        err_msg=f"{key}:vcv_neyman",
    )
    np.testing.assert_allclose(
        np.asarray(res.model_info["vcov_adjusted"], dtype=float),
        np.asarray(ref["vcv"], dtype=float),
        atol=1e-11,
        err_msg=f"{key}:vcv",
    )


def test_vector_event_time_summary_uses_the_joint_covariance(panels):
    """``.estimate`` averages the non-negative event times; its SE is joint.

    Treating the event times as independent would understate the SE here, so
    the check is that the reported SE is *not* the naive diagonal one.
    """
    res = sp.staggered_rollout(
        panels["rollout"]["data"],
        y="y",
        i="unit",
        t="time",
        g="first_treat",
        estimand="eventstudy",
        event_time=[0, 1, 2],
    )
    detail = res.detail
    assert res.estimate == pytest.approx(float(detail["estimate"].mean()), abs=1e-12)

    vcov = np.asarray(res.model_info["vcov"], dtype=float)
    w = np.full(3, 1 / 3)
    assert res.se == pytest.approx(float(np.sqrt(w @ vcov @ w)), abs=1e-12)
    naive = float(np.sqrt((detail["se"].to_numpy() ** 2).sum()) / 3)
    assert res.se != pytest.approx(naive, abs=1e-6)


def test_vector_model_info_describes_the_aggregate_not_the_first_event_time(panels):
    """Every scalar in ``model_info`` must belong to the reported aggregate.

    Carrying the first event time's SE alongside an averaged ``.estimate``
    would read as the SE *of that estimate* and be wrong by a wide margin —
    on this panel the first event time's SE is ~20% below the aggregate's.
    """
    res = sp.staggered_rollout(
        panels["rollout"]["data"],
        y="y",
        i="unit",
        t="time",
        g="first_treat",
        estimand="eventstudy",
        event_time=[0, 1, 2],
    )
    w = np.full(3, 1 / 3)
    for key, matrix in (
        ("se_neyman", "vcov_neyman"),
        ("se_adjusted", "vcov_adjusted"),
    ):
        vcov = np.asarray(res.model_info[matrix], dtype=float)
        assert res.model_info[key] == pytest.approx(
            float(np.sqrt(w @ vcov @ w)), abs=1e-12
        ), key
        # ...and is emphatically not the first event time's value.
        assert res.model_info[key] != pytest.approx(
            float(res.detail[key].iloc[0]), abs=1e-6
        ), key


def test_vector_fisher_pvalues_stay_per_event_time(panels):
    """No scalar randomisation p-value is invented for an averaged estimate."""
    res = sp.staggered_rollout(
        panels["nullpanel"]["data"],
        y="y",
        i="unit",
        t="time",
        g="first_treat",
        estimand="eventstudy",
        event_time=[0, 1],
        fisher=True,
        n_fisher=50,
        random_state=3,
    )
    assert "fisher_pvalue" not in res.model_info
    by_event = res.model_info["fisher_pvalue_by_event_time"]
    assert set(by_event) == {0.0, 1.0}
    np.testing.assert_allclose(
        list(by_event.values()), res.detail["fisher_pvalue"].to_numpy()
    )


# --------------------------------------------------------------------------
# staggered_cs / staggered_sa
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", sorted(_REF["wrappers"]))
def test_cs_sa_wrappers_match_r(panels, key):
    dataset, estimand, which = key.split("|")
    ref = _REF["wrappers"][key]
    fn = sp.staggered_cs if which == "cs" else sp.staggered_sa
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = fn(
            panels[dataset]["data"],
            y=panels[dataset]["y"],
            i=panels[dataset]["i"],
            t=panels[dataset]["t"],
            g=panels[dataset]["g"],
            estimand=estimand,
        )
    assert res.estimate == pytest.approx(ref["estimate"], abs=_ATOL), key
    assert res.model_info["se_neyman"] == pytest.approx(ref["se_neyman"], abs=_ATOL)
    assert res.model_info["se_adjusted"] == pytest.approx(ref["se"], abs=_ATOL)


def test_cs_and_sa_differ_only_in_the_comparison_group(panels):
    """SA uses the last-treated cohort only; CS uses every not-yet-treated one."""
    kwargs = dict(y="y", i="unit", t="time", g="first_treat", estimand="simple")
    data = panels["rollout"]["data"]
    cs = sp.staggered_cs(data, **kwargs)
    sa = sp.staggered_sa(data, **kwargs)
    assert cs.model_info["use_last_treated_only"] is False
    assert sa.model_info["use_last_treated_only"] is True
    assert cs.estimate != pytest.approx(sa.estimate, abs=1e-6)


def test_wrappers_are_the_plug_in_not_the_efficient_estimator(panels):
    """Both reference wrappers fix ``beta = 1``; nothing may quietly optimise it."""
    for fn in (sp.staggered_cs, sp.staggered_sa):
        res = fn(
            panels["rollout"]["data"],
            y="y",
            i="unit",
            t="time",
            g="first_treat",
        )
        assert res.model_info["efficient"] is False
        assert np.allclose(res.model_info["beta"], 1.0)


# --------------------------------------------------------------------------
# Fisher randomisation test
# --------------------------------------------------------------------------


def test_fisher_matches_r_draw_for_draw():
    """Replay R's own permutations: the estimator must agree on every draw."""
    perm_path = _FIX / "staggered_fisher_permutations.csv"
    if not perm_path.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing permutation fixture: {perm_path}")
    perms = pd.read_csv(perm_path)
    cols = [c for c in perms.columns if c.startswith("p")]

    spec = _REF["fisher_explicit"]
    units = np.asarray(spec["units"], dtype=float)
    cohort_of_unit = np.asarray(spec["cohort_of_unit"], dtype=float)
    # The wide panel orders units ascending, which is how R exported them.
    assert np.array_equal(units, np.sort(units))

    data = _panel("nullpanel")
    panel = _wide_panel(data["data"], i="unit", t="time", g="first_treat", y="y")
    summaries = _summarize(panel)
    a_theta, a_zero = _weight_matrices(
        "simple", summaries.g_list, panel.t_list, summaries.sizes
    )

    for k in range(len(perms)):
        index = perms.loc[k, cols].to_numpy(dtype=int) - 1  # R is 1-based
        permuted = panel._replace(g_of_unit=cohort_of_unit[index])
        estimate, se_neyman, se_adjusted, _ = _fit(
            _summarize(permuted), a_theta, a_zero, panel.t_list, True
        )
        assert estimate == pytest.approx(perms.loc[k, "estimate"], abs=_ATOL), k
        assert se_neyman == pytest.approx(perms.loc[k, "se_neyman"], abs=_ATOL), k
        assert se_adjusted == pytest.approx(perms.loc[k, "se"], abs=_ATOL), k


def test_fisher_permutation_preserves_cohort_sizes():
    """Permuting adoption dates must not resize any cohort.

    This is what lets the weight matrices be built once and reused; if it ever
    stopped holding, the p-value would silently use the wrong weights.
    """
    data = _panel("nullpanel")
    panel = _wide_panel(data["data"], i="unit", t="time", g="first_treat", y="y")
    before = _summarize(panel)
    rng = np.random.default_rng(0)
    after = _summarize(panel._replace(g_of_unit=rng.permutation(panel.g_of_unit)))
    np.testing.assert_array_equal(before.g_list, after.g_list)
    np.testing.assert_array_equal(before.sizes, after.sizes)


def test_fisher_pvalue_matches_r_within_monte_carlo_error():
    """End-to-end p-value against R's, up to simulation error.

    Both sides draw 2000 permutations independently. At the reference value
    ``p ~ 0.07`` the standard error of each estimate is ~0.006, so the
    difference has a standard error of ~0.008; 0.03 is a hair under four of
    those. Wide enough never to flake, tight enough to catch a one-sided /
    two-sided mix-up or a wrong comparison direction, which would move the
    p-value by an order of magnitude.
    """
    ref = _REF["fisher_package"]["nullpanel"]
    data = _panel("nullpanel")
    res = sp.staggered_rollout(
        data["data"],
        y="y",
        i="unit",
        t="time",
        g="first_treat",
        estimand="simple",
        fisher=True,
        n_fisher=2000,
        random_state=0,
    )
    assert res.model_info["n_fisher"] == 2000
    assert res.model_info["fisher_pvalue_neyman"] == pytest.approx(
        ref["fisher_pval_se_neyman"], abs=0.03
    )
    assert res.model_info["fisher_pvalue_adjusted"] == pytest.approx(
        ref["fisher_pval"], abs=0.03
    )


def test_fisher_rejects_when_the_effect_is_real():
    """On the panel with a strong effect R returns p = 0; so must we."""
    data = _panel("rollout")
    res = sp.staggered_rollout(
        data["data"],
        y="y",
        i="unit",
        t="time",
        g="first_treat",
        fisher=True,
        n_fisher=200,
        random_state=1,
    )
    assert _REF["fisher_package"]["rollout"]["fisher_pval"] == 0
    assert res.model_info["fisher_pvalue"] == 0.0


def test_fisher_is_reproducible_and_seed_sensitive():
    data = _panel("nullpanel")
    kwargs = dict(y="y", i="unit", t="time", g="first_treat", fisher=True, n_fisher=100)
    a = sp.staggered_rollout(data["data"], random_state=7, **kwargs)
    b = sp.staggered_rollout(data["data"], random_state=7, **kwargs)
    assert a.model_info["fisher_pvalue"] == b.model_info["fisher_pvalue"]


def test_fisher_is_off_by_default(panels):
    res = sp.staggered_rollout(
        panels["rollout"]["data"], y="y", i="unit", t="time", g="first_treat"
    )
    assert "fisher_pvalue" not in res.model_info


# --------------------------------------------------------------------------
# behavioural edge cases
# --------------------------------------------------------------------------


def test_singleton_cohort_is_dropped_with_a_warning():
    """R drops single-unit cohorts and warns; the estimate must then agree."""
    data = _panel("singleton")
    with pytest.warns(UserWarning, match="single cross-sectional unit"):
        core = staggered_rollout_core(**data, estimand="simple")
    ref = _REF["singleton"]
    assert core.estimate == pytest.approx(ref["estimate"], abs=_ATOL)
    assert core.se_neyman == pytest.approx(ref["se_neyman"], abs=_ATOL)
    assert core.se_adjusted == pytest.approx(ref["se"], abs=_ATOL)


def test_units_treated_in_the_first_period_are_kept_by_the_plain_estimator():
    data = _panel("early")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        core = staggered_rollout_core(**data, estimand="simple", efficient=False)
    ref = _REF["early"]["plain"]
    assert core.estimate == pytest.approx(ref["estimate"], abs=_ATOL)
    assert core.se_neyman == pytest.approx(ref["se_neyman"], abs=_ATOL)


@pytest.mark.parametrize("which", ["cs", "sa"])
def test_wrappers_drop_units_treated_in_the_first_period(which):
    """ATT(g, t) is not identified for them, so both wrappers remove them."""
    data = _panel("early")
    fn = sp.staggered_cs if which == "cs" else sp.staggered_sa
    with pytest.warns(UserWarning, match="treated in the first period"):
        res = fn(data["data"], y="y", i="unit", t="time", g="first_treat")
    ref = _REF["early"][which]
    assert res.estimate == pytest.approx(ref["estimate"], abs=_ATOL)
    assert res.model_info["se_neyman"] == pytest.approx(ref["se_neyman"], abs=_ATOL)
    assert res.model_info["se_adjusted"] == pytest.approx(ref["se"], abs=_ATOL)


def test_reference_metadata_is_the_version_we_claim():
    """Guard against a fixture regenerated under a different R package."""
    assert _REF["meta"]["staggered_version"] == "1.2.2"
