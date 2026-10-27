"""``sp.sharpened_qvalues`` vs Anderson's Stata do-file.

Reference: ``fdr_sharpened_qvalues.do`` ("Last modified: M. Anderson,
11/20/07"; sha256 43e4b411...8b44, Wayback copy of 2025-02-16), run
unchanged in Stata 18 under the default ``set type float`` by
``_fixtures/_generate_sharpened_qvalues_Stata.do`` on the 16 p-vectors of
``_fixtures/sharpened_qvalues_input.csv`` (2,474 p-values, M from 1 to
2,000; ties, p = 0 and 1, threshold hits, missing values).

Tolerance: none. The do-file stores ``bky06_qval`` as float, and every
q-value here, rounded to float, equals Stata's float bit for bit. That is
the claim the default makes, so the test asserts equality.
"""

from __future__ import annotations

import pathlib
import re

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
INPUT = pd.read_csv(_FIX / "sharpened_qvalues_input.csv", float_precision="round_trip")
REF = pd.read_csv(_FIX / "sharpened_qvalues_Stata.csv", dtype=str)


def _stata_hex(s: str) -> float:
    """Parse Stata's ``%21x`` (``+1.8000000000000X-003``); missing -> NaN."""
    m = re.fullmatch(r"([+-])1\.([0-9a-f]+)X([+-][0-9a-f]+)", s.strip())
    if m is None:
        if s.strip().startswith(("+0.", "-0.")):
            return 0.0
        raise ValueError(s)
    sign, mant, exp = m.groups()
    if exp == "+3ff":  # Stata's system missing
        return float("nan")
    return float.fromhex(f"{sign}0x1.{mant}p{int(exp, 16)}")


def _case(name: str):
    p = INPUT.loc[INPUT["case"] == name, "pval"].to_numpy()
    r = REF[REF["case"] == name].copy()
    r["pos"] = r["pos"].astype(int)
    r = r.sort_values("pos")
    return (
        p,
        np.array([_stata_hex(v) for v in r["pval_hex"]]),
        np.array([_stata_hex(v) for v in r["bky06_qval_hex"]]),
    )


CASES = list(dict.fromkeys(INPUT["case"]))


def test_fixture_covers_every_case():
    assert set(CASES) == set(REF["case"])
    assert len(INPUT) == len(REF) == 2474


@pytest.mark.parametrize("name", CASES)
def test_grid_qvalues_equal_the_dofile(name):
    p, p_stata, q_stata = _case(name)
    # Stata held the same p-values (as float, as the do-file declares pval)
    np.testing.assert_array_equal(
        np.where(np.isnan(p), -1, p.astype(np.float32)),
        np.where(np.isnan(p_stata), -1, p_stata.astype(np.float32)),
    )
    q = sp.sharpened_qvalues(p, missing="drop")
    np.testing.assert_array_equal(np.isnan(q), np.isnan(q_stata))
    ok = ~np.isnan(q)
    np.testing.assert_array_equal(
        q[ok].astype(np.float32), q_stata[ok].astype(np.float32)
    )
    # reported as the decimal grid level
    np.testing.assert_allclose(q[ok] * 1000, np.round(q[ok] * 1000), atol=1e-9)


def test_adjust_pvalues_bky_is_the_same_function():
    p, _, q_stata = _case("anderson_like")
    np.testing.assert_array_equal(
        sp.adjust_pvalues(p, method="bky").astype(np.float32),
        q_stata.astype(np.float32),
    )


def test_missing_count_treats_nan_as_p_one():
    # missing='count' (sp.adjust_pvalues' convention) equals the do-file run
    # on the vector with each NaN replaced by 1, with the NaN put back.
    p, _, _ = _case("with_nan")
    _, _, q_one = _case("with_nan_as_one")
    q = sp.sharpened_qvalues(p)
    assert np.isnan(q[np.isnan(p)]).all()
    ok = ~np.isnan(p)
    np.testing.assert_array_equal(
        q[ok].astype(np.float32), q_one[ok].astype(np.float32)
    )
    # and it is not the do-file's own rule, which shrinks the family
    assert not np.array_equal(q[ok], sp.sharpened_qvalues(p, missing="drop")[ok])


def _rejected(p: np.ndarray, q: float) -> np.ndarray:
    """Which hypotheses the two-stage procedure rejects at level q."""
    M = p.size
    s = np.sort(p)
    r = np.arange(1, M + 1)
    qa = q / (1 + q)
    hit = np.flatnonzero(s <= qa * r / M)
    r1 = hit[-1] + 1 if hit.size else 0
    if r1 == M:
        k = M
    else:
        hit = np.flatnonzero(s <= qa * M / (M - r1) * r / M)
        k = hit[-1] + 1 if hit.size else 0
    return p <= s[k - 1] if k else np.zeros(M, dtype=bool)


def test_exact_mode_is_the_infimum_rejection_level():
    rng = np.random.default_rng(7)
    for t in range(300):
        M = int(rng.integers(1, 25))
        p = np.concatenate([rng.beta(0.3, 6, M // 2), rng.uniform(size=M - M // 2)])
        if t % 5 == 0:
            p = np.round(p, 2)  # ties and round thresholds
        q = sp.sharpened_qvalues(p, exact=True)
        for i in range(M):
            if q[i] < 1:
                assert _rejected(p, q[i] * (1 + 1e-10))[i]
                if q[i] > 0:
                    assert not _rejected(p, q[i] * (1 - 1e-10))[i]
            else:
                assert not _rejected(p, 1 - 1e-10)[i]


@pytest.mark.parametrize("name", ["mixed40", "big300", "anderson_like", "ties"])
def test_grid_is_exact_rounded_up_to_the_grid(name):
    p, _, _ = _case(name)
    exact = sp.sharpened_qvalues(p, exact=True)
    grid = sp.sharpened_qvalues(p)
    expected = np.clip(np.ceil(np.round(exact * 1000, 9)) / 1000, 0.001, 1.0)
    np.testing.assert_allclose(grid, expected, atol=1e-12)


def test_single_pvalue_closed_form():
    # M = 1: stage 1 rejects iff q/(1+q) >= p, i.e. q >= p / (1 - p).
    for p in [0.0, 0.01, 0.2, 1 / 3, 0.5]:
        exact = sp.sharpened_qvalues([p], exact=True)[0]
        assert exact == pytest.approx(p / (1 - p), abs=1e-15)
    assert sp.sharpened_qvalues([0.6], exact=True)[0] == 1.0
    assert sp.sharpened_qvalues([0.0])[0] == 0.001  # the grid's floor


def test_ties_permutation_and_monotonicity():
    p = np.array([0.04, 0.01, 0.04, 0.3, 0.002, 0.01, 0.6])
    q = sp.sharpened_qvalues(p)
    assert q[0] == q[2] and q[1] == q[5]
    perm = np.random.default_rng(1).permutation(p.size)
    np.testing.assert_array_equal(sp.sharpened_qvalues(p[perm]), q[perm])
    order = np.argsort(p, kind="stable")
    assert np.all(np.diff(q[order]) >= 0)


def test_edge_inputs():
    assert sp.sharpened_qvalues([]).size == 0
    assert np.isnan(sp.sharpened_qvalues([np.nan, np.nan])).all()
    with pytest.raises(MethodIncompatibility):
        sp.sharpened_qvalues([0.2, 1.5])
    with pytest.raises(MethodIncompatibility):
        sp.sharpened_qvalues([[0.1, 0.2]])
    with pytest.raises(MethodIncompatibility):
        sp.sharpened_qvalues([0.1], missing="ignore")
    # existing methods keep their numbers
    np.testing.assert_allclose(
        sp.adjust_pvalues([0.01, 0.04, 0.03, 0.20], method="holm"),
        [0.04, 0.09, 0.09, 0.2],
    )
