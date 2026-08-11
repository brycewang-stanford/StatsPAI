"""Callaway-Sant'Anna parity for three behaviours fixed in 1.23.0.

All three were found by running StatsPAI head to head against the
``csdid`` / ``DRDIDpy`` Python ports and then reading the R sources they
port. Reference values here come from R ``did`` 2.3.0 itself, generated
by ``_generate_cs_gaps_R.R``.

A. ``anticipation > 0`` with ``base_period='varying'``
   R shifts the base period to ``g − 1 − δ`` only once a cell is
   *post*-treatment; a pre-treatment placebo keeps the period immediately
   before it (``compute.att_gt.R``, the ``pret`` block). StatsPAI shifted
   the placebos too, which moved every pre-treatment cell and dropped the
   earliest ones from the grid entirely. Post-treatment ATT(g, t) were
   never affected, so this is a pre-trend/event-study-lead bug, not an
   ATT bug.

B. ``allow_unbalanced_panel``
   Simply did not exist. R switches to the repeated-cross-section
   estimators (which never difference within unit, so no row is lost) and
   folds the influence functions to the unit level via
   ``.rowid <- idname``, keeping ``n`` at the number of units. Skipping
   that fold would treat a unit's pre and post rows as independent and
   understate every SE, so the SE column is the load-bearing assertion
   here — not the ATT.

C. clustered multiplier bootstrap on **unequal** cluster sizes
   StatsPAI aggregated the influence function to cluster *means* over
   ``n_clusters`` (``se = bSigma/sqrt(n_clusters)``). That faithfully
   mirrored **CRAN ``did`` 2.3.0**, which does the same — so this is not a
   transcription slip. It is nonetheless wrong for unequal clusters: a
   cluster enters with weight ``1/|c|``, so a singleton cluster dominates
   and the SEs inflate without bound as the size spread grows.

   Upstream ``did`` (GitHub master, post-2.3.0) switched to cluster
   *sums* with ``se = bSigma·sqrt(n_clusters)/n``, noting the old
   aggregation only coincides for equal-sized clusters; ``csdid`` tracks
   the corrected form. StatsPAI now does too.

   These tests therefore do **not** pin R 2.3.0's clustered numbers —
   that would freeze the superseded convention. They pin the statistical
   property the corrected aggregation has and the old one does not: the
   cluster-sum bootstrap must reproduce the closed-form cluster-robust
   variance ``Σ_c (Σ_{i∈c} ψ_i)² / n²``, and must collapse to the
   ordinary ``σ²/n`` when there is no within-cluster correlation.

   They use many clusters on purpose. With the ~9 clusters of the R
   fixture the multiplier distribution is a coarse sum of 9 atoms, its
   IQR-rescaled spread is nowhere near its own standard deviation, and no
   closed form applies — a real property is only testable once the
   bootstrap is near-normal.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURES = Path(__file__).parent / "_fixtures"
_PANEL = _FIXTURES / "cs_gaps_panel.csv"
_UNBALANCED = _FIXTURES / "cs_gaps_unbalanced_panel.csv"
_REFERENCE = _FIXTURES / "cs_gaps_reference.csv"

# Deterministic (bstrap=False) paths agree with R to solver tolerance:
# the logit and the WLS outcome model are the only sources of slack.
_RTOL = 1e-7
_ATOL = 1e-9


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(_PANEL)


@pytest.fixture(scope="module")
def unbalanced() -> pd.DataFrame:
    return pd.read_csv(_UNBALANCED)


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(_REFERENCE)


def _ref(reference: pd.DataFrame, case: str) -> pd.DataFrame:
    out = reference[reference["case"] == case]
    assert not out.empty, f"no reference rows for case {case!r}"
    # Under base_period='universal' R also emits the normalised reference
    # cell ATT(g, g−1−δ) ≡ 0 with se = NA. StatsPAI omits that row (it
    # carries no information and is not an estimate), so it is dropped
    # here rather than compared. Every estimated cell must still match,
    # and the grid check below still catches a genuinely missing one.
    out = out[np.isfinite(out["se"])]
    assert not out.empty, f"case {case!r} has no estimated cells"
    return out.reset_index(drop=True)


def _merge(got: pd.DataFrame, want: pd.DataFrame) -> pd.DataFrame:
    """Align on (group, time), asserting the two grids are identical.

    The grid itself is part of what regressed in case A, so an
    inner join would hide the bug.
    """
    merged = want.merge(
        got[["group", "time", "att", "se"]],
        on=["group", "time"],
        how="outer",
        suffixes=("_r", "_sp"),
        indicator=True,
    )
    mismatched = merged[merged["_merge"] != "both"]
    assert mismatched.empty, "the (g, t) grids differ from R:\n" + mismatched[
        ["group", "time", "_merge"]
    ].to_string(index=False)
    return merged


# ---------------------------------------------------------------------
# A. anticipation x base_period
# ---------------------------------------------------------------------


@pytest.mark.parametrize("base_period", ["varying", "universal"])
@pytest.mark.parametrize("anticipation", [0, 1, 2])
def test_anticipation_matches_r_grid_and_values(
    panel, reference, base_period, anticipation
):
    want = _ref(reference, f"anticipation:{base_period}:a{anticipation}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = sp.callaway_santanna(
            panel,
            y="y",
            g="g",
            t="t",
            i="i",
            estimator="dr",
            base_period=base_period,
            anticipation=anticipation,
            control_group="nevertreated",
            bstrap=False,
        )
    merged = _merge(got.detail, want)
    np.testing.assert_allclose(
        merged["att_sp"], merged["att_r"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(merged["se_sp"], merged["se_r"], rtol=_RTOL, atol=_ATOL)


def test_anticipation_leaves_varying_pretreatment_placebos_alone(panel, reference):
    """The regression itself: pre-treatment cells must not move with δ.

    Under ``base_period='varying'`` a placebo's base is the period before
    it, so raising ``anticipation`` cannot change its value — only which
    cells are classified post-treatment. The old code shifted the placebo
    base too and every pre-treatment cell moved.
    """
    a0 = _ref(reference, "anticipation:varying:a0")
    a1 = _ref(reference, "anticipation:varying:a1")
    shared = a0.merge(a1, on=["group", "time"], suffixes=("_0", "_1"))
    pre = shared[shared["time"] < shared["group"]]
    assert len(pre) >= 2, "fixture must retain shared pre-treatment cells"
    # R itself holds the placebos fixed; this pins the expectation the
    # StatsPAI grid is then checked against above.
    np.testing.assert_allclose(pre["att_0"], pre["att_1"], rtol=1e-12)


# ---------------------------------------------------------------------
# B. allow_unbalanced_panel
# ---------------------------------------------------------------------


@pytest.mark.parametrize("covariates", ["none", "x1"])
@pytest.mark.parametrize("estimator", ["dr", "ipw", "reg"])
def test_allow_unbalanced_panel_matches_r(unbalanced, reference, estimator, covariates):
    want = _ref(reference, f"unbalanced:{estimator}:{covariates}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = sp.callaway_santanna(
            unbalanced,
            y="y",
            g="g",
            t="t",
            i="i",
            x=None if covariates == "none" else ["x1"],
            estimator=estimator,
            base_period="varying",
            control_group="nevertreated",
            allow_unbalanced_panel=True,
            bstrap=False,
        )
    merged = _merge(got.detail, want)
    np.testing.assert_allclose(
        merged["att_sp"], merged["att_r"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(merged["se_sp"], merged["se_r"], rtol=_RTOL, atol=_ATOL)


def test_allow_unbalanced_panel_folds_influence_to_units(unbalanced):
    """``n`` is the unit count, not the row count.

    This is what separates the unbalanced-panel route from a true
    repeated cross-section: the same estimators, but influence functions
    indexed by unit. Getting it wrong is invisible in the ATT and shows
    up only in the SEs.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = sp.callaway_santanna(
            unbalanced,
            y="y",
            g="g",
            t="t",
            i="i",
            allow_unbalanced_panel=True,
            bstrap=False,
        )
    n_units = unbalanced["i"].nunique()
    assert got.model_info["n_units"] == n_units
    assert got.model_info["n_obs"] == len(unbalanced)
    assert got.model_info["allow_unbalanced_panel"] is True
    assert got.model_info["panel"] is True
    assert got._influence_funcs.shape[0] == n_units


def test_allow_unbalanced_panel_is_inert_on_a_balanced_panel(panel):
    """R resets the flag when the panel is balanced; so must StatsPAI.

    Otherwise the option would silently change the estimand on data that
    never needed it.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.callaway_santanna(panel, y="y", g="g", t="t", i="i", bstrap=False)
        flagged = sp.callaway_santanna(
            panel,
            y="y",
            g="g",
            t="t",
            i="i",
            allow_unbalanced_panel=True,
            bstrap=False,
        )
    pd.testing.assert_frame_equal(plain.detail, flagged.detail)
    assert plain.method == flagged.method


def test_allow_unbalanced_panel_differs_from_dropping_incomplete_units(unbalanced):
    """The two routes are different estimators, and must not coincide.

    A test that only checked "it runs" would pass even if the flag were
    wired to a no-op.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        folded = sp.callaway_santanna(
            unbalanced,
            y="y",
            g="g",
            t="t",
            i="i",
            allow_unbalanced_panel=True,
            bstrap=False,
        )
        dropped = sp.callaway_santanna(
            unbalanced, y="y", g="g", t="t", i="i", bstrap=False
        )
    assert not np.allclose(
        folded.detail["att"].to_numpy(), dropped.detail["att"].to_numpy()
    )


def test_unbalanced_panel_warns_when_the_flag_is_not_set(unbalanced):
    with pytest.warns(UserWarning, match="allow_unbalanced_panel=True"):
        sp.callaway_santanna(unbalanced, y="y", g="g", t="t", i="i", bstrap=False)


# ---------------------------------------------------------------------
# C. clustered multiplier bootstrap, unequal cluster sizes
# ---------------------------------------------------------------------

_BITERS = 200_000


def test_unclustered_bootstrap_still_matches_r(panel, reference):
    """Control arm: identical in did 2.3.0 and master, and already correct.

    The cluster fix must not have disturbed it.
    """
    want = _ref(reference, "cluster_boot:none")
    got = sp.callaway_santanna(
        panel,
        y="y",
        g="g",
        t="t",
        i="i",
        estimator="dr",
        base_period="varying",
        control_group="nevertreated",
        bstrap=True,
        biters=_BITERS,
        random_state=4242,
    )
    merged = _merge(got.detail, want)
    np.testing.assert_allclose(
        merged["att_sp"], merged["att_r"], rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(merged["se_sp"], merged["se_r"], rtol=0.05)


def _lopsided_clusters(rng, n_clusters=120):
    """Cluster sizes spanning 1 … 40, so means != sums by a wide margin."""
    sizes = rng.integers(1, 41, size=n_clusters)
    return np.repeat(np.arange(n_clusters), sizes)


def _cluster_sum_se(psi, codes, n):
    sums = np.zeros((codes.max() + 1, psi.shape[1]))
    np.add.at(sums, codes, psi)
    return np.sqrt((sums**2).sum(axis=0)) / n


def _cluster_mean_se(psi, codes):
    """The superseded did 2.3.0 aggregation, for contrast."""
    sums = np.zeros((codes.max() + 1, psi.shape[1]))
    np.add.at(sums, codes, psi)
    counts = np.bincount(codes).astype(float)
    n_c = len(counts)
    return np.sqrt(((sums / counts[:, None]) ** 2).sum(axis=0)) / n_c


def test_cluster_bootstrap_reproduces_the_cluster_robust_closed_form():
    """The load-bearing property: sums, not means.

    With enough clusters the multiplier distribution is near-normal, so
    the IQR-rescaled bootstrap SE must land on the closed-form
    cluster-robust SE ``sqrt(Σ_c (Σ_{i∈c} ψ_i)²) / n``.
    """
    from statspai.did._core import multiplier_bootstrap

    rng = np.random.default_rng(0)
    codes = _lopsided_clusters(rng)
    n = len(codes)
    # Genuine within-cluster correlation: a shared cluster shock.
    shock = rng.normal(size=(codes.max() + 1, 4))
    psi = rng.normal(size=(n, 4)) + shock[codes]
    psi -= psi.mean(axis=0)

    want = _cluster_sum_se(psi, codes, n)
    got, _ = multiplier_bootstrap(
        psi, n, 0.05, 60_000, random_state=3, cluster_ids=codes
    )
    np.testing.assert_allclose(got, want, rtol=0.06)

    # Not vacuous: the pre-1.23.0 aggregation lands somewhere else
    # entirely. Its *direction* is not fixed — it inflates when the small
    # clusters carry the noise and deflates under a strong common shock —
    # so the guard is on the size of the gap, not its sign.
    ratio = _cluster_mean_se(psi, codes) / want
    assert np.min(np.abs(ratio - 1.0)) > 0.05, ratio


def test_cluster_sum_aggregation_is_the_unbiased_one():
    """Why sums are right and means are not, with no bootstrap involved.

    With no within-cluster correlation the cluster-robust variance must
    reproduce the ordinary ``σ²/n``: ``E[Σ_c (Σ_{i∈c} ψ_i)²] = n σ²``
    however the units are partitioned. The cluster-mean aggregation gives
    each cluster ``σ²/|c|`` instead, so the small clusters dominate and it
    is biased upward by a factor that grows with the size spread — ~1.45x
    on cluster sizes 1…40 here, and ~5x on the 1…150 spread of a real
    state-clustered county panel.

    Averaged over draws so the assertion is about the estimators, not
    about one lucky realisation.
    """
    rng = np.random.default_rng(7)
    codes = _lopsided_clusters(rng)
    n = len(codes)
    iid_se = 1.0 / np.sqrt(n)  # psi ~ N(0, 1)

    sums, means = [], []
    for _ in range(200):
        psi = rng.normal(size=(n, 2))
        psi -= psi.mean(axis=0)
        sums.append(_cluster_sum_se(psi, codes, n))
        means.append(_cluster_mean_se(psi, codes))
    sums_mean = np.mean(sums, axis=0)
    means_mean = np.mean(means, axis=0)

    np.testing.assert_allclose(sums_mean / iid_se, 1.0, rtol=0.03)
    assert np.min(means_mean / iid_se) > 1.25


def test_equal_sized_clusters_are_where_the_two_conventions_agree():
    """Why this survived so long — and why balanced-cluster tests still pass."""
    from statspai.did._core import multiplier_bootstrap

    rng = np.random.default_rng(2)
    codes = np.repeat(np.arange(100), 20)  # 100 clusters, all size 20
    n = len(codes)
    shock = rng.normal(size=(100, 3))
    psi = rng.normal(size=(n, 3)) + shock[codes]
    psi -= psi.mean(axis=0)

    sums_se = _cluster_sum_se(psi, codes, n)
    np.testing.assert_allclose(_cluster_mean_se(psi, codes), sums_se, rtol=1e-12)

    got, _ = multiplier_bootstrap(
        psi, n, 0.05, 60_000, random_state=9, cluster_ids=codes
    )
    np.testing.assert_allclose(got, sums_se, rtol=0.06)


def test_clustering_on_the_unit_id_alone_is_a_no_op(panel):
    """``clustervars`` = the unit id must reproduce the unclustered SEs.

    Every cluster then has size 1, so sums and the raw influence
    functions coincide — a cheap end-to-end check that the fold is wired
    to the right axis.
    """
    from statspai.did._core import multiplier_bootstrap

    fit = sp.callaway_santanna(
        panel, y="y", g="g", t="t", i="i", estimator="dr", bstrap=False
    )
    psi = fit._influence_funcs
    n = psi.shape[0]
    plain, _ = multiplier_bootstrap(psi, n, 0.05, 20_000, random_state=11)
    singleton, _ = multiplier_bootstrap(
        psi, n, 0.05, 20_000, random_state=11, cluster_ids=np.arange(n)
    )
    np.testing.assert_allclose(singleton, plain, rtol=1e-12)
