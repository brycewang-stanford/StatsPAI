"""Coverage round-2 — ``statspai.rd.rdmulti`` (rdmc / rdms).

Multi-cutoff RD (``sp.rdmc``) and multi-score / geographic RD (``sp.rdms``).
Exercises both pooling methods, both kernels, the forest-plot path, and the
degenerate branches (too-few-obs warning, all-invalid pooling fallback).

Real synthetic RD data with a known jump at each cutoff; assertions check
effect signs/magnitudes, positive SEs / bandwidths, and CI ordering — never
fabricated numbers.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.core.results import CausalResult  # noqa: E402


def _multi_cutoff_df(seed=7, n=3000, cutoffs=(0.0, 2.0, 4.0), tau=3.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 5.0, n)
    # each unit assigned to nearest cutoff; jump tau at whichever cutoff applies
    y = 0.5 * x + rng.normal(0, 0.4, n)
    for c in cutoffs:
        near = np.abs(x - c) < 1.0
        y[near] += tau * (x[near] >= c)
    return pd.DataFrame({"y": y, "x": x})


def _geo_df(seed=3, n=2500, tau=2.5):
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    y = 0.3 * x1 + 0.2 * x2 + tau * (x1 >= 0) + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def test_rdmc_ivw_pooling_and_summary():
    df = _multi_cutoff_df()
    res = sp.rdmc(df, y="y", x="x", cutoffs=[0.0, 2.0, 4.0], pooling="ivw")
    assert res.n_cutoffs == 3
    assert res.pooled_se > 0
    assert res.pooled_ci[0] < res.pooled_ci[1]
    # pooled effect should be in the neighborhood of the +3 jump
    assert 1.0 < res.pooled_estimate < 5.0
    txt = res.summary()
    assert "Multi-Cutoff" in txt and "Pooled" in txt


def test_rdmc_equal_pooling():
    df = _multi_cutoff_df()
    res = sp.rdmc(df, y="y", x="x", cutoffs=[0.0, 2.0, 4.0], pooling="equal")
    assert np.isfinite(res.pooled_estimate)
    assert res.pooled_se > 0


def test_rdmc_manual_bandwidth_and_uniform_kernel():
    df = _multi_cutoff_df()
    res = sp.rdmc(df, y="y", x="x", cutoffs=[0.0, 2.0], bandwidth=0.6, kernel="uniform")
    for cr in res.cutoff_results:
        assert cr["bandwidth"] == 0.6


def test_rdmc_plot_forest():
    df = _multi_cutoff_df()
    res = sp.rdmc(df, y="y", x="x", cutoffs=[0.0, 2.0])
    fig, ax = plt.subplots()
    out = res.plot(ax=ax)
    assert out is not None
    plt.close("all")


def test_rdmc_plot_creates_own_axes():
    df = _multi_cutoff_df()
    res = sp.rdmc(df, y="y", x="x", cutoffs=[0.0, 2.0, 4.0])
    out = res.plot()  # no ax -> internal subplots() branch
    assert out is not None
    plt.close("all")


def test_rdmc_all_invalid_pooling_returns_nan():
    # cutoffs far outside the data support -> no observations -> NaN pools
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=200), "x": rng.uniform(-1, 1, 200)})
    res = sp.rdmc(df, y="y", x="x", cutoffs=[50.0, 80.0], bandwidth=0.2)
    assert np.isnan(res.pooled_estimate)
    assert np.isnan(res.pooled_se)


def test_rdms_geographic():
    df = _geo_df()
    res = sp.rdms(df, y="y", x1="x1", x2="x2", bandwidth=0.8)
    assert isinstance(res, CausalResult)
    assert res.se > 0
    assert res.ci[0] < res.ci[1]
    assert np.isfinite(res.estimate)
    # `n_local` is gone with the Euclidean-window implementation. The
    # rewritten estimator delegates to sp.rdrobust on a signed
    # distance-to-boundary score, so the sample actually used is reported
    # the way every other RD entry point reports it -- per side.
    assert res.model_info["n_effective_left"] > 0
    assert res.model_info["n_effective_right"] > 0


def test_rdms_uniform_kernel_and_manual_bw():
    df = _geo_df()
    res = sp.rdms(df, y="y", x1="x1", x2="x2", bandwidth=0.5, kernel="uniform")
    assert np.isfinite(res.estimate)
    assert res.model_info["kernel"] == "uniform"


def test_rdms_supports_every_kernel_rdrobust_supports():
    """Epanechnikov is a real kernel here now, not a fallback.

    The retired implementation recognised only "triangular" and
    "uniform" and quietly treated anything else as triangular, so this
    test used to assert a *fallback*. Delegating to sp.rdrobust means the
    kernel argument reaches an estimator that implements all three, and
    the three now give genuinely different answers -- which is what the
    argument is for.
    """
    df = _geo_df().assign(z=lambda d: (d["x1"] >= 0).astype(float))
    ests = {}
    for kern in ("triangular", "uniform", "epanechnikov"):
        res = sp.rdms(
            df, y="y", x1="x1", x2="x2", treat="z", bandwidth=0.8, kernel=kern
        )
        assert np.isfinite(res.estimate)
        assert res.model_info["kernel"] == kern
        ests[kern] = float(res.estimate)
    assert (
        ests["epanechnikov"] != ests["triangular"]
    ), "epanechnikov is being silently treated as triangular again"


def test_rdms_requires_or_assumes_a_treatment_indicator():
    """Treatment on a 2D boundary is not implied by the coordinates.

    The retired implementation hard-coded ``x1 >= cutoff1`` behind a
    comment reading "Convention" -- an assumption about the design
    presented as a fact about the data. ``rdmulti::rdms`` requires the
    indicator (its ``zvar``) for exactly this reason. Omitting it still
    works, so existing callers are not broken, but it says so.
    """
    df = _geo_df()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assumed = sp.rdms(df, y="y", x1="x1", x2="x2")
    msgs = " ".join(str(x.message) for x in w).lower()
    assert "treat=" in msgs and "assumption" in msgs
    assert assumed.model_info["treat_assumed"] is True

    df = df.assign(z=(df["x1"] >= 0).astype(float))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stated = sp.rdms(df, y="y", x1="x1", x2="x2", treat="z")
    assert not [m for m in w if "treat=" in str(m.message)]
    assert stated.model_info["treat_assumed"] is False
    # Same design, stated rather than guessed: identical numbers.
    assert stated.estimate == pytest.approx(assumed.estimate, rel=1e-12)


def test_rdms_rejects_a_non_binary_treatment_column():
    df = _geo_df().assign(bad=2.0)
    with pytest.raises(ValueError, match="must be 0/1"):
        sp.rdms(df, y="y", x1="x1", x2="x2", treat="bad")
