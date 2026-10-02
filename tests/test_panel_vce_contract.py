"""``sp.panel(vce=...)`` refuses what it does not implement.

Until 2026-10 any ``vce=`` outside the extended menu fell through and
returned the classical standard errors: ``vce='robust'`` gave the same
numbers as no option, and so did ``vce='nonsense'``. Found while checking
whether Stata's ``xtreg, re vce(robust)`` could be translated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    rows = []
    for i in range(40):
        a = rng.normal()
        for t in range(1, 7):
            x = rng.normal() + 0.3 * a
            rows.append((i, t, 1 + 0.9 * x + a + rng.normal(), x))
    return pd.DataFrame(rows, columns=["id", "t", "y", "x"])


def _se(frame, **kw):
    return float(sp.panel(frame, "y ~ x", entity="id", time="t", **kw).std_errors["x"])


@pytest.mark.parametrize("method", ["fe", "re"])
@pytest.mark.parametrize("vce", ["robust", "cluster", "hc1", "nonsense"])
def test_unknown_vce_is_refused_not_ignored(frame, method, vce):
    with pytest.raises(MethodIncompatibility, match="not a variance option"):
        sp.panel(frame, "y ~ x", entity="id", time="t", method=method, vce=vce)


def test_the_refusal_names_the_working_spelling(frame):
    with pytest.raises(MethodIncompatibility) as exc:
        sp.panel(frame, "y ~ x", entity="id", time="t", vce="robust")
    hint = str(exc.value)
    assert "robust='robust'" in hint and "cluster=" in hint


def test_supported_spellings_still_change_the_standard_error(frame):
    classical = _se(frame, method="fe")
    robust = _se(frame, method="fe", robust="robust")
    clustered = _se(frame, method="fe", cluster="id")
    cr2 = _se(frame, method="fe", vce="CR2", cluster="id")
    assert len({round(v, 10) for v in (classical, robust, clustered, cr2)}) == 4
    # Omitting vce= is still the classical SE.
    assert _se(frame, method="fe", vce=None) == classical
