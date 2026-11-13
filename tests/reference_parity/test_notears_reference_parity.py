"""``sp.notears`` against the authors' implementation of NOTEARS.

Reference: ``notears.linear.notears_linear`` (github.com/xunzheng/notears,
Apache-2.0) with ``lambda1 = 0.1`` and the least-squares loss, on the
standardised data ``sp.notears`` builds
(``_fixtures/_generate_notears_data.py``; weights in
``notears_reference.json`` from ``_generate_notears_reference.py``).

What is compared:

* the thresholded graph (``|w| >= 0.3``): exact, every edge with its
  direction, on every design of the fixture;
* the weights of those edges: 5e-2 absolute.

Why the weights are not held to 1e-6. The problem is not convex and each
inner problem is stopped by L-BFGS-B at its default tolerance, so the
last digits of the weights depend on the optimiser's path. The reference
does not reproduce itself more closely than that: run under scipy 1.13
and under scipy 1.17 (which replaced the L-BFGS-B routine) its own
weights move by up to 1.3e-2 on these designs, and by less than 1e-10 on
the designs where the path is stable. ``sp.notears`` shows the same
pattern against it. The fixture keeps the designs where every reference
weight is at least 0.1 away from the threshold, so the graph does not
hinge on those digits.

Through 1.43.1 ``sp.notears`` took the dual step after every inner solve,
with the penalty already raised, instead of first raising the penalty
until the constraint had fallen (Algorithm 1 of Zheng et al. 2018). On
the first twelve candidate designs it ended at the reference's graph in
eight; it now does in all twelve.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "notears_reference.json").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    "_generate_notears_data", _FIX / "_generate_notears_data.py"
)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
SEEDS = sorted(REF["cases"], key=int)


def _fit(seed):
    X, _ = _gen.make(int(seed))
    ref = REF["cases"][seed]
    assert list(X.shape) == ref["shape"]
    assert float(np.abs(X).sum()) == pytest.approx(ref["checksum"], rel=1e-12)
    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(X.shape[1])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return sp.notears(df), np.asarray(ref["W"])


def test_fixture_designs_are_clear_of_the_threshold():
    assert len(SEEDS) >= 12
    n_edges = 0
    for seed in SEEDS:
        W = np.asarray(REF["cases"][seed]["W"])
        assert np.min(np.abs(np.abs(W) - 0.3)) >= 0.1
        n_edges += int((np.abs(W) >= 0.3).sum())
    assert n_edges >= 40


@pytest.mark.parametrize("seed", SEEDS)
def test_graph_is_the_reference_graph_and_weights_are_close(seed):
    res, W = _fit(seed)
    ours = res["adjacency"].to_numpy()
    want = np.where(np.abs(W) >= 0.3, W, 0.0)
    assert np.array_equal(ours != 0, want != 0)
    np.testing.assert_allclose(ours, want, rtol=0, atol=5e-2)
    assert res["is_dag"]
    assert res["n_edges"] == int((want != 0).sum())
