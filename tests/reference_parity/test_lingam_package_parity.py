"""``sp.lingam`` against the ``lingam`` package (DirectLiNGAM).

Reference: ``lingam`` 1.13.0 (MIT), ``DirectLiNGAM()`` with its default
pairwise likelihood-ratio measure, on 40 simulated linear non-Gaussian
structural equation models (``_fixtures/_generate_lingam_data.py``; orders
in ``_fixtures/lingam_package.json`` from ``_generate_lingam_package.py``).

What is compared: the causal order, exactly, on all 40. It is a
permutation, so there is no tolerance.

What is not: the coefficient matrix. ``sp.lingam`` regresses each variable
on all of its predecessors by least squares; the package prunes that
regression with an adaptive Lasso. Given the same order those are two
estimators of the same coefficients, not one computation. The test checks
instead that the matrix is the least-squares fit it is documented to be.

Through 1.43.1 the search used a different independence score and agreed
with the package on 17 of these 40 orders; on 300 further draws of the
same design 87.3% of its orders respected the true graph, against 98.3%
now.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "lingam_package.json").read_text(encoding="utf-8"))
_spec = importlib.util.spec_from_file_location(
    "_generate_lingam_data", _FIX / "_generate_lingam_data.py"
)
_gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gen)
SEEDS = sorted(REF["cases"], key=int)


def test_fixture_is_the_package_output():
    assert REF["versions"]["lingam"] == "1.13.0"
    assert len(SEEDS) == _gen.N_CASES == 40


@pytest.mark.parametrize("seed", SEEDS)
def test_causal_order_is_the_packages(seed):
    ref = REF["cases"][seed]
    X, _ = _gen.make(int(seed))
    assert list(X.shape) == ref["shape"]
    # same data on both sides: the generator's stream has not moved
    assert float(np.abs(X).sum()) == pytest.approx(ref["checksum"], rel=1e-12)
    assert list(sp.lingam(X).order) == ref["order"]
    # the search standardises every pair, so the scale flag cannot matter
    assert list(sp.lingam(X, standardize=False).order) == ref["order"]


def test_orders_respect_the_true_graph_where_the_reference_does():
    # Every reference order on these 40 is a topological order of the DAG
    # that generated the data, so the matched orders are too.
    for seed in SEEDS:
        X, B = _gen.make(int(seed))
        pos = {v: k for k, v in enumerate(REF["cases"][seed]["order"])}
        for i, j in zip(*np.nonzero(B)):
            assert pos[j] < pos[i], (seed, i, j)


@pytest.mark.parametrize("seed", ["0", "7", "20"])
def test_adjacency_is_least_squares_on_the_predecessors(seed):
    X, _ = _gen.make(int(seed))
    res = sp.lingam(X)
    Xc = X - X.mean(axis=0)
    for rank, i in enumerate(res.order):
        before = list(res.order[:rank])
        want = np.zeros(X.shape[1])
        if before:
            want[before] = np.linalg.lstsq(Xc[:, before], Xc[:, i], rcond=None)[0]
        np.testing.assert_allclose(res.adjacency[i], want, rtol=1e-9, atol=1e-12)


def test_constant_and_collinear_columns_are_refused():
    X, _ = _gen.make(0)
    with pytest.raises(DataInsufficient):
        sp.lingam(np.column_stack([X, np.ones(len(X))]))
    with pytest.raises(DataInsufficient):
        sp.lingam(np.column_stack([X, 2.0 * X[:, 0]]))
