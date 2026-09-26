"""Coverage campaign — rdrobust option branches and result exports.

Part of the core-module ≥95% coverage initiative
(see ``.coverage_campaign/CAMPAIGN.md``). Complements the parallel-agent
``test_cov95_rd_rdrobust.py`` (sharp/fuzzy/RKD/covariate/cluster) by driving the
remaining option branches: the ``bwselect='cct'`` R-parity delegation and the
other bandwidth selectors, RBC bootstrap inference, manual bandwidths (h/b/rho),
the not-yet-supported observation-weight guard, and the result-object exporters
(``summary`` / ``to_latex`` / ``tidy`` / ``to_dict`` / ``plot``).

File name uses the campaign's own ``test_rd_cov_*`` convention to avoid colliding
with the parallel agent's ``test_cov95_rd_*`` files.
"""

from __future__ import annotations

import importlib.util

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402


@pytest.fixture(scope="module")
def sharp():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + 0.8 * (x >= 0) + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "x": x, "w": rng.uniform(0.5, 1.5, n)})


# ``bwselect='cct'`` delegates to the official rdrobust package, an *optional*
# dependency (the ``rd-cct`` extra). When rdrobust is not installed,
# ``sp.rdrobust(..., bwselect='cct')`` deliberately raises ImportError, so the
# parametrization skips that case rather than hard-failing — keeping the suite
# green on minimal installs. The other selectors are native StatsPAI code and
# always run.
_HAS_RDROBUST = importlib.util.find_spec("rdrobust") is not None
_cct = pytest.param(
    "cct",
    marks=pytest.mark.skipif(
        not _HAS_RDROBUST,
        reason="optional dependency 'rdrobust' (rd-cct extra) not installed",
    ),
)


@pytest.mark.parametrize("bwselect", [_cct, "mserd", "msetwo", "cerrd", "certwo"])
def test_rdrobust_bwselect_variants(sharp, bwselect):
    res = sp.rdrobust(sharp, y="y", x="x", c=0.0, bwselect=bwselect)
    assert res is not None


def test_rdrobust_rbc_bootstrap(sharp):
    res = sp.rdrobust(sharp, y="y", x="x", c=0.0, bootstrap="rbc", n_boot=199)
    assert res is not None


def test_rdrobust_manual_bandwidths(sharp):
    # h + b explicitly (b and rho are mutually exclusive)
    res = sp.rdrobust(sharp, y="y", x="x", c=0.0, h=0.3, b=0.5)
    assert res is not None


def test_rdrobust_rho(sharp):
    res = sp.rdrobust(sharp, y="y", x="x", c=0.0, h=0.3, rho=0.8)
    assert res is not None


def test_rdrobust_weights_are_used(sharp):
    # weights= is implemented since 1.32 (R rdrobust parity in
    # tests/reference_parity/test_rd_weights_parity.py): unit weights leave
    # the fit unchanged, non-constant ones change it.
    base = sp.rdrobust(sharp, y="y", x="x", c=0.0)
    unit = sp.rdrobust(sharp.assign(w1=1.0), y="y", x="x", c=0.0, weights="w1")
    assert unit.estimate == base.estimate
    rng = np.random.default_rng(0)
    varied = sp.rdrobust(
        sharp.assign(wv=rng.uniform(0.5, 2.0, len(sharp))),
        y="y",
        x="x",
        c=0.0,
        weights="wv",
    )
    assert varied.estimate != base.estimate


def test_rdrobust_result_exports(sharp):
    res = sp.rdrobust(sharp, y="y", x="x", c=0.0)
    assert isinstance(res.summary(), str)
    assert isinstance(res.to_latex(), str)
    assert len(res.tidy()) > 0
    assert isinstance(res.to_dict(), dict)
    res.plot()
    plt.close("all")


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_rdrobust_kernels_with_donut(sharp, kernel):
    res = sp.rdrobust(sharp, y="y", x="x", c=0.0, kernel=kernel, donut=0.02)
    assert res is not None
