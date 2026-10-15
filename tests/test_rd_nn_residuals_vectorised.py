"""The run-vectorised nearest-neighbour residuals equal the old loop bit for bit.

_nn_residuals feeds rdrobust's bandwidth selector and its vce='nn'
standard errors, so the vectorised search must reproduce the loop exactly,
ties included: R rdrobust is pinned against it at 1e-9.
"""

import zlib

import numpy as np
import pytest

from statspai.rd._cct_bandwidth import _nn_residuals


def _loop_reference(
    x: np.ndarray,
    y: np.ndarray,
    dups: np.ndarray,
    dupsid: np.ndarray,
    matches: int,
) -> np.ndarray:
    """The per-observation loop this module used through 1.39.6."""
    n = len(y)
    res = np.empty(n)
    limit = min(matches, n - 1)
    for pos in range(n):
        rpos = dups[pos] - dupsid[pos]
        lpos = dupsid[pos] - 1
        while lpos + rpos < limit:
            if pos - lpos - 1 < 0:
                rpos += dups[pos + rpos + 1]
            elif pos + rpos + 1 > n - 1:
                lpos += dups[pos - lpos - 1]
            else:
                dl = x[pos] - x[pos - lpos - 1]
                dr = x[pos + rpos + 1] - x[pos]
                if dl > dr:
                    rpos += dups[pos + rpos + 1]
                elif dl < dr:
                    lpos += dups[pos - lpos - 1]
                else:
                    rpos += dups[pos + rpos + 1]
                    lpos += dups[pos - lpos - 1]
        lo = max(0, pos - lpos)
        hi = min(n - 1, pos + rpos)
        idx = np.arange(lo, hi + 1)
        y_j = y[idx].sum() - y[pos]
        ji = len(idx) - 1
        res[pos] = np.sqrt(ji / (ji + 1.0)) * (y[pos] - y_j / ji)
    return res


def _tie_arrays(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    dupsid = np.empty(len(x), dtype=int)
    for k in range(len(cnt)):
        rows = np.flatnonzero(inv == k)
        dupsid[rows] = np.arange(1, len(rows) + 1)
    return cnt[inv], dupsid


@pytest.mark.parametrize(
    "design", ["continuous", "many_ties", "three_values", "rounded"]
)
@pytest.mark.parametrize("matches", [1, 3, 5])
def test_bit_identical_to_the_loop(design, matches):
    rng = np.random.default_rng(zlib.crc32(f"{design}-{matches}".encode()))
    for n in (2, 3, 9, 57, 400):
        if design == "continuous":
            x = rng.normal(size=n)
        elif design == "many_ties":
            x = rng.integers(0, max(2, n // 3), n).astype(float)
        elif design == "three_values":
            x = rng.integers(0, 3, n).astype(float)
        else:
            x = np.round(rng.normal(size=n), 1)
        x = np.sort(x)
        y = rng.normal(size=n) * 10.0 ** rng.integers(-3, 4)
        dups, dupsid = _tie_arrays(x)
        expected = _loop_reference(x, y, dups, dupsid, matches)
        got = _nn_residuals(x, y, dups, dupsid, matches)
        np.testing.assert_array_equal(got, expected)
