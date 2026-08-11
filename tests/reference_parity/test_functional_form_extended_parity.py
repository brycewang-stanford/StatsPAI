"""Reference parity: the extended ``sp.functional_form_test`` surface vs ``didFF``.

``test_functional_form_parity.py`` pins the core test at fixed bin counts.
This file pins everything added on top of it, all against R ``didFF`` 0.1.0:

* ``weights`` — sampling weights, which reach both the implied density and
  the test;
* the automatic binning rule, including the discrete-outcome branch;
* ``binpoints`` padding when the supplied cut points fall short of the data;
* every ``aggregation`` type, and the dynamic event-time window
  (``balance_e`` / ``min_e`` / ``max_e``);
* the implied-density plot.

Reference generation
--------------------
``Rscript tests/reference_parity/_generate_didff_extended_R.R`` writes
``_fixtures/didff_extended_reference.json`` plus the panels it uses.

Why the reject panel appears here too
-------------------------------------
Every mpdta p-value in this file is 1: the implied density is comfortably
non-negative, so the max-t statistic is negative and the simulated critical
value never binds. A file built only on mpdta would therefore pass with an
arbitrarily broken simulator. The reject panel — a multiplicative DGP read on
the level scale — drives the p-value to 0, and is run weighted as well as
unweighted so that weights are pinned through the *test*, not merely through
the point estimates.

References
----------
Roth, J. and Sant'Anna, P. H. C. (2023). "When Is Parallel Trends Sensitive to
Functional Form?" *Econometrica*, 91(2), 737-747. DOI 10.3982/ECTA19402.
[@roth2023when]
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).resolve().parent / "_fixtures"
_REF_PATH = _FIX / "didff_extended_reference.json"

# didFF prints ~15 significant digits; the implementation lands within ~2e-15,
# so 1e-9 is a wide margin that still catches any real divergence.
_ATOL = 1e-9

_MPDTA_KW = dict(y="lemp", g="first_treat", t="year", i="countyreal")
_DISCRETE_KW = dict(y="y", g="first_treat", t="time", i="unit")
_REJECT_KW = dict(y="y", g="g", t="t", i="id")


def _load_ref() -> dict:
    if not _REF_PATH.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing reference fixture: {_REF_PATH}")
    return json.loads(_REF_PATH.read_text(encoding="utf-8"))


_REF = _load_ref()


def _read(name: str) -> pd.DataFrame:
    path = _FIX / name
    if not path.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing panel fixture: {path}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    return _read("mpdta_did_package_weighted.csv")


@pytest.fixture(scope="module")
def discrete() -> pd.DataFrame:
    return _read("didff_discrete_panel.csv")


@pytest.fixture(scope="module")
def reject() -> pd.DataFrame:
    return _read("functional_form_reject_weighted.csv")


def _densities(result) -> np.ndarray:
    return result.table["implied_density"].to_numpy(dtype=float)


def _assert_matches(result, ref, label: str) -> None:
    np.testing.assert_allclose(
        _densities(result),
        np.asarray(ref["implied_density"], dtype=float),
        atol=_ATOL,
        err_msg=f"{label}: implied density",
    )


# --------------------------------------------------------------------------
# sampling weights
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key,n_bins", [("nbins6", 6), ("nbins10", 10)])
def test_weighted_density_matches_didff(mpdta, key, n_bins):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=n_bins, weights="w")
    _assert_matches(res, _REF["weights"][key], key)
    assert res.diagnostics["weighted"] is True


def test_weights_actually_change_the_answer(mpdta):
    """Guard against a silently ignored ``weights`` argument.

    A passthrough that never arrives would make these two runs identical, and
    every weighted parity assertion above would still pass by accident.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        weighted = sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=6, weights="w")
        plain = sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=6)
    assert not np.allclose(_densities(weighted), _densities(plain), atol=1e-6)
    # ...and each still matches its own R counterpart.
    _assert_matches(weighted, _REF["weights"]["nbins6"], "weighted")
    _assert_matches(plain, _REF["weights"]["unweighted6"], "unweighted")
    assert plain.diagnostics["weighted"] is False


def test_weighted_rejection_matches_didff(reject):
    """Weights must reach the test statistic, not just the point estimates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(reject, **_REJECT_KW, n_bins=6, weights="w")
    _assert_matches(res, _REF["reject"]["weighted"], "reject-weighted")
    assert res.pvalue == _REF["reject"]["weighted"]["pval"] == 0


def test_unknown_weight_column_fails_loudly(mpdta):
    with pytest.raises(Exception):
        sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=6, weights="not_a_column")


# --------------------------------------------------------------------------
# automatic and discrete binning
# --------------------------------------------------------------------------


def test_auto_binning_matches_didff_default(mpdta):
    """With no ``n_bins``, a continuous outcome gets ``min(20, n_distinct)`` bins."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(mpdta, **_MPDTA_KW)
    ref = _REF["auto_default"]
    assert len(res.table) == len(ref["implied_density"]) == 20
    _assert_matches(res, ref, "auto")
    assert res.diagnostics["discrete_outcome"] is False


def test_discrete_outcome_gets_one_bin_per_value(discrete):
    """Few distinct untreated values: bin per value, with a warning."""
    with pytest.warns(UserWarning, match="discrete"):
        res = sp.functional_form_test(discrete, **_DISCRETE_KW)
    ref = _REF["discrete"]["auto"]
    assert len(res.table) == len(ref["implied_density"]) == _REF["discrete_n_distinct"]
    _assert_matches(res, ref, "discrete-auto")
    assert res.diagnostics["discrete_outcome"] is True
    # The reference labels a discrete bin by its own value.
    np.testing.assert_allclose(
        res.table["level"].to_numpy(dtype=float),
        np.asarray([float(x) for x in ref["level"]]),
        atol=_ATOL,
    )


def test_explicit_n_bins_overrides_the_discrete_rule(discrete):
    """An explicit ``n_bins`` always cuts, however few distinct values there are."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(discrete, **_DISCRETE_KW, n_bins=4)
    ref = _REF["discrete"]["forced_bins"]
    assert len(res.table) == 4
    _assert_matches(res, ref, "discrete-forced")
    assert res.diagnostics["discrete_outcome"] is False


def test_auto_is_the_default_and_none_is_its_alias(mpdta):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.functional_form_test(mpdta, **_MPDTA_KW)
        explicit = sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins="auto")
        as_none = sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=None)
    np.testing.assert_allclose(_densities(default), _densities(explicit), atol=1e-15)
    np.testing.assert_allclose(_densities(default), _densities(as_none), atol=1e-15)


def test_unknown_n_bins_string_fails_loudly(mpdta):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="n_bins"):
        sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins="sturges")


def test_auto_rejection_matches_didff(reject):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(reject, **_REJECT_KW)
    _assert_matches(res, _REF["reject"]["auto"], "reject-auto")
    assert res.pvalue == 0.0


# --------------------------------------------------------------------------
# binpoints
# --------------------------------------------------------------------------


def test_covering_binpoints_match_didff(mpdta):
    lo, hi = _REF["binpoints"]["range"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(mpdta, **_MPDTA_KW, binpoints=[lo, 5, 7, 9, hi])
    _assert_matches(res, _REF["binpoints"]["covering"], "binpoints-covering")


def test_short_binpoints_are_padded_with_a_warning(mpdta):
    """Cut points that stop short would drop mass; the reference pads instead."""
    with pytest.warns(UserWarning, match="do not cover the range"):
        res = sp.functional_form_test(mpdta, **_MPDTA_KW, binpoints=[5, 7, 9])
    _assert_matches(res, _REF["binpoints"]["short"], "binpoints-short")
    # Padding reproduces the covering case exactly — that is the whole point.
    np.testing.assert_allclose(
        _densities(res),
        np.asarray(_REF["binpoints"]["covering"]["implied_density"], dtype=float),
        atol=_ATOL,
    )


def test_binpoints_and_n_bins_together_fail_loudly(mpdta):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="only one"):
        sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=6, binpoints=[5, 7, 9])


def test_binpoints_collapsing_to_one_bin_fail_loudly(mpdta):
    lo, hi = _REF["binpoints"]["range"]
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="single bin"):
        sp.functional_form_test(mpdta, **_MPDTA_KW, binpoints=[lo - 1, hi + 1])


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("aggregation", ["simple", "group", "calendar", "dynamic"])
def test_every_aggregation_matches_didff(mpdta, aggregation):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(
            mpdta, **_MPDTA_KW, n_bins=6, aggregation=aggregation
        )
    _assert_matches(res, _REF["aggregation"][aggregation], aggregation)


@pytest.mark.parametrize(
    "key,kwargs",
    [
        ("min0", dict(min_e=0)),
        ("max1", dict(min_e=0, max_e=1)),
        ("balanced1", dict(balance_e=1)),
    ],
)
def test_dynamic_event_time_window_matches_didff(mpdta, key, kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(
            mpdta, **_MPDTA_KW, n_bins=6, aggregation="dynamic", **kwargs
        )
    _assert_matches(res, _REF["dynamic_window"][key], key)


def test_event_time_window_actually_bites(mpdta):
    """A restricted window must change the aggregate it is restricting."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = sp.functional_form_test(
            mpdta, **_MPDTA_KW, n_bins=6, aggregation="dynamic"
        )
        windowed = sp.functional_form_test(
            mpdta, **_MPDTA_KW, n_bins=6, aggregation="dynamic", min_e=0, max_e=1
        )
    assert not np.allclose(_densities(full), _densities(windowed), atol=1e-8)


# --------------------------------------------------------------------------
# the plot
# --------------------------------------------------------------------------


def test_plot_draws_one_bar_per_bin_and_flags_negatives(reject):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(reject, **_REJECT_KW, n_bins=6)
    assert len(res.negative_bins) > 0, "reject panel should have negative bins"

    ax = res.plot()
    heights = sorted(patch.get_height() for patch in ax.patches)
    assert len(heights) == len(res.table)
    np.testing.assert_allclose(heights, sorted(_densities(res)), atol=1e-12)
    # Negative bars are drawn in their own colour so the violation is visible.
    colours = {patch.get_facecolor() for patch in ax.patches}
    assert len(colours) == 2
    labels = {text.get_text() for text in ax.get_legend().get_texts()}
    assert labels == {"Negative", "Non-negative"}


def test_plot_window_restricts_the_bars(mpdta):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=6)
    full = res.plot()
    windowed = res.plot(lb=5.0, ub=9.0)
    assert len(windowed.patches) < len(full.patches)


def test_plot_with_an_empty_window_fails_loudly(mpdta):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(mpdta, **_MPDTA_KW, n_bins=6)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="nothing to draw"):
        res.plot(lb=1e6)


# --------------------------------------------------------------------------
# distributional DiD (didFF::distDD)
# --------------------------------------------------------------------------


def _dist_case(name, mpdta, discrete):
    kw = dict(n_bins=6)
    if name == "mpdta6":
        return sp.distributional_did(mpdta, **_MPDTA_KW, **kw)
    if name == "mpdta_weighted":
        return sp.distributional_did(mpdta, **_MPDTA_KW, weights="w", **kw)
    if name == "discrete":
        return sp.distributional_did(discrete, **_DISCRETE_KW)
    if name == "simple_agg":
        return sp.distributional_did(mpdta, **_MPDTA_KW, aggregation="simple", **kw)
    raise AssertionError(name)


@pytest.mark.parametrize("name", ["mpdta6", "mpdta_weighted", "discrete", "simple_agg"])
def test_distributional_did_matches_distdd(mpdta, discrete, name):
    """Point estimates *and* standard errors.

    ``distDD`` exposes standard errors on this path; the functional-form
    entry point does not, so these rows pin something the other file cannot.
    """
    ref = _REF["distdd"][name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = _dist_case(name, mpdta, discrete)
    np.testing.assert_allclose(
        res.table["estimate"].to_numpy(dtype=float),
        np.asarray(ref["estimates"], dtype=float),
        atol=_ATOL,
        err_msg=f"{name}: estimates",
    )
    np.testing.assert_allclose(
        res.table["se"].to_numpy(dtype=float),
        np.asarray(ref["se"], dtype=float),
        atol=_ATOL,
        err_msg=f"{name}: standard errors",
    )


def test_distributional_effects_sum_to_zero(mpdta):
    """Treatment moves probability mass between bins; it cannot create it.

    An implementation that dropped the zero-sum structure — by binning over
    the untreated rows only, say — would break this without breaking any
    single-bin comparison.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.distributional_did(mpdta, **_MPDTA_KW, n_bins=6)
    assert res.diagnostics["effect_sum"] == pytest.approx(0.0, abs=1e-10)


def test_distributional_did_survives_an_empty_bin(mpdta):
    """The automatic binning leaves an empty bin here, and didFF 0.1.0 dies.

    ``didFF`` builds its table from ``levels(droplevels(bins))`` (19 entries)
    alongside 20 point estimates and raises "arguments imply differing number
    of rows: 20, 19". There is therefore no reference row to compare against;
    what is pinned is that StatsPAI reports every bin and flags the empty one
    rather than crashing.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.distributional_did(mpdta, **_MPDTA_KW)
    assert len(res.table) == 20
    assert int(res.table["used"].sum()) == 19
    assert res.diagnostics["n_bins_dropped"] == 1
    assert res.table.loc[~res.table["used"], "se"].isna().all()


def test_distributional_did_bins_over_the_whole_panel():
    """Its bins must span treated post-values; the FF test's deliberately do not.

    The two functions ask different questions of the same data, and this is
    the structural difference between them: the functional-form test bins the
    *untreated* rows, so post-treatment outcomes cannot move the edges of the
    density it is testing, while the distributional estimand is precisely
    about where treated mass ended up.

    On ``mpdta`` the two binnings coincide, because the outcome's extremes are
    attained on untreated rows — so the panel here is built to put treated
    post-values outside the untreated support, which is the only situation
    where the difference is observable.
    """
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(1, 201):
        g = 3 if uid % 2 == 0 else 0
        base = rng.normal(5.0, 0.4)
        for period in range(1, 5):
            lift = 6.0 if (g > 0 and period >= g) else 0.0
            rows.append(
                {"id": uid, "t": period, "g": g, "y": base + 0.1 * period + lift}
            )
    panel = pd.DataFrame(rows)
    keys = dict(y="y", g="g", t="t", i="id")

    untreated_max = panel.loc[(panel["t"] < panel["g"]) | (panel["g"] == 0), "y"].max()
    assert panel["y"].max() > untreated_max + 1.0, "fixture must have treated overshoot"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dist = sp.distributional_did(panel, **keys, n_bins=6)
        ff = sp.functional_form_test(panel, **keys, n_bins=6)

    assert float(dist.table["bin_upper"].max()) > float(ff.table["bin_upper"].max())
    assert not np.allclose(
        dist.table["bin_lower"].to_numpy(dtype=float),
        ff.table["bin_lower"].to_numpy(dtype=float),
    )


def test_distributional_did_plot_draws_bins_with_error_bars(mpdta):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.distributional_did(mpdta, **_MPDTA_KW, n_bins=6)
    ax = res.plot()
    assert len(ax.patches) == int(res.table["used"].sum())
    assert len(ax.collections) >= 1  # the error bars


def test_reference_metadata_is_the_version_we_claim():
    assert _REF["meta"]["didff_version"] == "0.1.0"
