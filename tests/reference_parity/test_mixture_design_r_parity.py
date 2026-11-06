"""``sp.mixture_design`` against R ``mixexp`` 1.2.7 (``SLD``, ``SCD``).

The reference designs are in ``_fixtures/mixture_design_R.json``, written by
``_fixtures/_generate_mixture_design_R.R``. mixexp is GPL and archived on
CRAN; it was run as a black box.

Simplex-lattice and simplex-centroid designs are fixed sets of blends, so
the comparison is the set of runs: both sides are sorted the same way and
then compared element by element. The run order is not part of the design
(the two sides list the lattice blends in different orders).
Tolerance 1e-12: shares such as 1/3 are rounding in both languages.

Not compared: the ``'space_filling'`` kind is a stochastic search (support
points), and mixexp's ``Xvert`` cannot serve for ``lower=``: on a region
cut by lower bounds alone it returns some edge centroids as rows of zeros
and its vertices carry errors of order 1e-8.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp

REF = json.loads(
    (Path(__file__).parent / "_fixtures" / "mixture_design_R.json").read_text(
        encoding="utf-8"
    )
)


def _sorted_rows(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    keys = np.round(a, 10)
    order = np.lexsort(keys.T[::-1])
    return a[order]


@pytest.mark.parametrize("key", sorted(REF["lattice"]))
def test_simplex_lattice_equals_mixexp_sld(key):
    q, m = (int(v) for v in key.split("_"))
    d = sp.mixture_design(q, kind="simplex_lattice", degree=m)
    ref = np.array(REF["lattice"][key])
    assert d.design.shape == ref.shape
    np.testing.assert_allclose(
        _sorted_rows(d.design.to_numpy()), _sorted_rows(ref), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("key", sorted(REF["centroid"]))
def test_simplex_centroid_equals_mixexp_scd(key):
    q = int(key)
    d = sp.mixture_design(q, kind="simplex_centroid")
    ref = np.array(REF["centroid"][key])
    assert d.n_runs == 2**q - 1 == ref.shape[0]
    np.testing.assert_allclose(
        _sorted_rows(d.design.to_numpy()), _sorted_rows(ref), rtol=0, atol=1e-12
    )


def test_percentages_are_the_lattice_scaled():
    """``total=100`` is the same design in percent."""
    d = sp.mixture_design(4, kind="simplex_lattice", degree=3, total=100)
    ref = np.array(REF["lattice"]["4_3"]) * 100
    np.testing.assert_allclose(
        _sorted_rows(d.design.to_numpy()), _sorted_rows(ref), rtol=0, atol=1e-10
    )
