"""Reference parity: ``sp.functional_form_test`` vs ``didFF::didFF``.

Roth & Sant'Anna (2023) ask when parallel trends is insensitive to the choice
of functional form, and give a testable answer: the counterfactual untreated
distribution the design implies for the treated group must be a real
distribution, so its density must be non-negative everywhere. ``sp.functional_form_test``
implements that; ``didFF`` 0.1.0 is Sant'Anna's own implementation.

Reference generation (R 4.5.2, didFF 0.1.0)::

    didFF(data = d, yname = "lemp", tname = "year", idname = "countyreal",
          gname = "first.treat", nbins = 6, seed = 0, numSims = 100000)

Fixtures:

* ``_fixtures/mpdta_did_package.csv`` -- ``did::mpdta`` exported from R.
  Note this is NOT ``sp.datasets.mpdta()``, which is a documented *simulated
  replica*: same shape, different numbers (``lemp`` spans 7.7-8.8 there
  against 1.1-10.4 in the real data). Parity has to run on the real one.
* ``_fixtures/functional_form_reject_panel.csv`` -- a multiplicative DGP read
  on the level scale, where the implied density goes negative and the test
  rejects.

Both designs are pinned deliberately. The p-value saturates at 1 whenever the
max-t statistic is negative, so an accept-only fixture would leave the
critical value and the whole rejection path unexercised -- the test would pass
with an arbitrarily wrong simulator.

Two binning details had to match R exactly and are worth recording, because
each one moved the answer by ~3e-4 per bin until it did:

1. Bins are built on the **untreated** observations only (``t < g`` or
   ``g == 0``), not the full panel.
2. R's ``cut(x, breaks = n)`` lays ``n + 1`` equally spaced edges from min to
   max and then pushes only the two **outer** edges out by ``dx/1000``. The
   interior edges stay on the unpadded grid, so the first and last bins come
   out slightly wider than the rest.

References
----------
Roth, J. and Sant'Anna, P. H. C. (2023). "When Is Parallel Trends Sensitive to
Functional Form?" *Econometrica*, 91(2), 737-747. DOI 10.3982/ECTA19402.
[@roth2023when]
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_MPDTA = _FIX / "mpdta_did_package.csv"
_REJECT = _FIX / "functional_form_reject_panel.csv"

# didFF on did::mpdta, nbins = 6 and nbins = 10.
R_MPDTA_6 = [
    0.0304139344957,
    0.0674359104695,
    0.3416865755096,
    0.3827411511547,
    0.1358376116166,
    0.0418848167539,
]
R_MPDTA_10 = [
    0.0106406411495,
    0.0145207475559,
    0.0373269625036,
    0.1192327894407,
    0.2578152798251,
    0.2642877717345,
    0.1382775038547,
    0.1070841593385,
    0.0376997238178,
    0.0131144207797,
]
R_MPDTA_EDGES_6 = [
    1.08926820442608,
    2.65595966233907,
    4.21330703601004,
    5.77065440968100,
    7.32800178335196,
    8.88534915702293,
    10.4520406149359,
]

# didFF on the rejecting fixture, nbins = 8.
R_REJECT = [
    -0.350746268657,
    0.307270181755,
    0.395634890872,
    0.336433410835,
    0.175779394485,
    0.085377134428,
    0.030150753769,
    0.020100502513,
]


def _load(path: pathlib.Path) -> pd.DataFrame:
    if not path.exists():  # pragma: no cover - fixtures ship with the repo
        pytest.skip(f"missing fixture: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    return _load(_MPDTA)


@pytest.fixture(scope="module")
def reject_panel() -> pd.DataFrame:
    return _load(_REJECT)


def _run(df, y, g, t, i, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.functional_form_test(df, y=y, g=g, t=t, i=i, **kwargs)


@pytest.fixture(scope="module")
def mpdta_fit(mpdta):
    return _run(
        mpdta, "lemp", "first_treat", "year", "countyreal", n_bins=6, n_sims=10_000
    )


@pytest.fixture(scope="module")
def reject_fit(reject_panel):
    return _run(reject_panel, "y", "g", "t", "id", n_bins=8, n_sims=100_000)


@pytest.mark.parametrize("k", range(6))
def test_mpdta_density_matches_didff(mpdta_fit, k):
    got = float(mpdta_fit.table["implied_density"].iloc[k])
    assert got == pytest.approx(
        R_MPDTA_6[k], abs=1e-9
    ), f"bin {k}: StatsPAI {got:.12f} vs didFF {R_MPDTA_6[k]:.12f}"


@pytest.mark.parametrize("k", range(7))
def test_bin_edges_match_r_cut(mpdta_fit, k):
    """The edges are the fiddly part -- pin them, not just the masses."""
    edges = list(mpdta_fit.table["bin_lower"]) + [mpdta_fit.table["bin_upper"].iloc[-1]]
    assert edges[k] == pytest.approx(R_MPDTA_EDGES_6[k], abs=1e-10)


def test_mpdta_ten_bins_match_didff(mpdta):
    res = _run(
        mpdta, "lemp", "first_treat", "year", "countyreal", n_bins=10, n_sims=10_000
    )
    got = res.table["implied_density"].to_numpy()
    assert np.max(np.abs(got - np.array(R_MPDTA_10))) < 1e-9


def test_mpdta_does_not_reject(mpdta_fit):
    assert mpdta_fit.pvalue > 0.9
    assert mpdta_fit.statistic < 0  # no bin is negative by even one SE
    assert mpdta_fit.negative_bins.empty


@pytest.mark.parametrize("k", range(8))
def test_reject_density_matches_didff(reject_fit, k):
    got = float(reject_fit.table["implied_density"].iloc[k])
    assert got == pytest.approx(
        R_REJECT[k], abs=1e-9
    ), f"bin {k}: StatsPAI {got:.12f} vs didFF {R_REJECT[k]:.12f}"


def test_reject_case_rejects(reject_fit):
    """didFF returns pval = 0 here; without this design the critical value
    would never be exercised, because a negative max-t saturates p at 1."""
    assert reject_fit.pvalue == 0.0
    assert reject_fit.statistic > 10.0
    assert len(reject_fit.negative_bins) == 1
    assert float(reject_fit.negative_bins["implied_density"].iloc[0]) < -0.3


def test_summary_names_the_consequence(reject_fit):
    text = reject_fit.summary()
    assert "REJECTED" in text
    assert "monotonic transformation" in text


def test_densities_sum_to_about_one(mpdta_fit, reject_fit):
    """The implied masses are a distribution over the binned support."""
    for fit in (mpdta_fit, reject_fit):
        assert float(fit.table["implied_density"].sum()) == pytest.approx(1.0, abs=0.05)


def test_too_few_bins_fails_loudly(mpdta):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="n_bins"):
        _run(mpdta, "lemp", "first_treat", "year", "countyreal", n_bins=1)


def test_unknown_aggregation_fails_loudly(mpdta):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="aggregation"):
        _run(
            mpdta,
            "lemp",
            "first_treat",
            "year",
            "countyreal",
            n_bins=6,
            aggregation="overall",
        )


def test_simulated_pvalue_is_reproducible(reject_panel):
    """`random_state` is fixed by default so a reported p-value can be
    re-derived; two calls must not disagree."""
    a = _run(reject_panel, "y", "g", "t", "id", n_bins=8, n_sims=5000)
    b = _run(reject_panel, "y", "g", "t", "id", n_bins=8, n_sims=5000)
    assert a.pvalue == b.pvalue


def test_the_shipped_mpdta_replica_is_not_the_real_data():
    """Guards the trap this file exists to avoid.

    ``sp.datasets.mpdta()`` is a documented simulated replica. If it ever
    silently became the real data (or vice versa) the fixture above would
    be redundant or wrong, and either way someone should notice.
    """
    replica = sp.datasets.mpdta()
    real = _load(_MPDTA)
    assert len(replica) == len(real)  # same shape by construction
    assert float(replica["lemp"].min()) > float(real["lemp"].min()) + 5
