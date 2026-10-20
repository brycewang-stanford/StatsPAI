"""Input checks of ``sp.iv_wild_bootstrap`` (2026-10-20 probe).

* A shuffled ``data`` (same length, other row order) paired every
  residual with another observation's cluster and moved the p-value
  silently; only the length was checked.
* One cluster raised a raw ``ZeroDivisionError``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    n, g = 400, 12
    firm = rng.integers(0, g, n)
    z = rng.normal(size=n)
    u = rng.normal(size=g)[firm] + rng.normal(size=n)
    d = 0.6 * z + 0.5 * u + rng.normal(size=n)
    df = pd.DataFrame({"y": 1 + 0.3 * d + u, "d": d, "z": z, "firm": firm})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.ivreg("y ~ (d ~ z)", data=df, cluster="firm")
    return fit, df


def test_same_frame_runs_and_is_deterministic(fitted):
    fit, df = fitted
    a = sp.iv_wild_bootstrap(fit, df, "firm", "d", n_boot=199, seed=1)
    b = sp.iv_wild_bootstrap(fit, df.copy(), "firm", "d", n_boot=199, seed=1)
    assert a["p_boot"] == b["p_boot"]


def test_reordered_rows_are_refused(fitted):
    fit, df = fitted
    with pytest.raises(MethodIncompatibility, match="rows of `data`"):
        sp.iv_wild_bootstrap(
            fit, df.sample(frac=1, random_state=1), "firm", "d", n_boot=199, seed=1
        )


def test_one_cluster_is_refused(fitted):
    fit, df = fitted
    with pytest.raises(DataInsufficient, match="one cluster"):
        sp.iv_wild_bootstrap(fit, df.assign(firm=0), "firm", "d", n_boot=99, seed=1)
