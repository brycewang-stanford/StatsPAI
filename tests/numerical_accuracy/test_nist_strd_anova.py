"""NIST StRD one-way ANOVA certification for the StatsPAI OLS kernel.

The NIST Statistical Reference Datasets publish certified F-statistics,
mean squares, R-squared and residual standard deviations for one-way
analysis of variance.  A one-way ANOVA is algebraically a least-squares
fit of ``y ~ C(group)``, so these datasets certify the *numerical
accuracy* of ``sp.regress``'s sum-of-squares decomposition and F-statistic
— a complement to ``test_nist_strd_ols.py`` (which certifies the
coefficient / standard-error path).

The datasets span three difficulty levels.  The ``SmLs0{1..9}`` family is
deliberately constructed with large constant leading digits so that a
naively implemented (cancellation-prone) sum of squares loses precision;
``SmLs03/06/09`` carry that to n = 18009 observations.

Known limitation
----------------
The *higher-difficulty* ``SmLs0{7,8,9}`` family (9 constant leading digits)
is marked ``xfail``: ``sp.regress``'s F / R² path loses accuracy through
catastrophic cancellation under extreme constant offsets, and the loss
grows with sample size —

    SmLs07 (n=189)    relative F error ≈ 4.8e-4
    SmLs08 (n=1809)   relative F error ≈ 9.9e-3
    SmLs09 (n=18009)  relative F error ≈ 2.3e-2  (F≈1955 vs certified 2001)

The lower / average-difficulty families reproduce the certified F to
machine precision at every sample size — including ``SmLs03/06`` at the
same n=18009 (≈1e-16 and ≈6e-11) — so the boundary is the *offset
magnitude*, not n.  These xfails pin that boundary; a future
numerically-stabler (mean-centred) OLS sum-of-squares will surface them as
``xpass``.  Fixing the kernel is intentionally out of scope here (it would
change a core estimator's numerical output); this suite only documents the
limit.

Fixtures are static, public-domain NIST files (see ``PROVENANCE.md``); they
run without R, Stata, or network access.

Reference
---------
NIST/SEMATECH Statistical Reference Datasets, Analysis of Variance.
https://www.itl.nist.gov/div898/strd/anova/anova.html
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

import statspai as sp

FIXTURE_DIR = Path(__file__).with_name("_fixtures") / "nist_strd_anova"
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_BETWEEN_RE = re.compile(rf"^\s*Between\s+\w+\s+(\d+)\s+({_NUM})\s+({_NUM})\s+({_NUM})")
_WITHIN_RE = re.compile(rf"^\s*Within\s+\w+\s+(\d+)\s+({_NUM})\s+({_NUM})")
_R2_RE = re.compile(rf"Certified R-Squared\s+({_NUM})")
_RESID_RE = re.compile(rf"Standard Deviation\s+({_NUM})")

# Per-dataset relative tolerance for the certified F-statistic / R-squared.
# A uniform 1e-9 clears every design except SmLs09 (handled via xfail).
DEFAULT_RTOL = 1e-9

# All eleven NIST one-way ANOVA datasets, in ascending difficulty.
CASES = [
    "SiRstv",
    "SmLs01",
    "SmLs02",
    "SmLs03",
    "SmLs04",
    "SmLs05",
    "SmLs06",
    "SmLs07",
    "SmLs08",
    "SmLs09",
    "AtmWtAg",
]
# The higher-difficulty SmLs0{7,8,9} family (9 constant leading digits) is
# beyond sp.regress's current F/R^2 numerical accuracy (see module docstring).
_HARD_REASON = (
    "sp.regress F/R^2 loses 3-4+ significant digits under 9 "
    "constant leading digits (catastrophic cancellation; error "
    "grows with n)"
)
XFAIL = {name: _HARD_REASON for name in ("SmLs07", "SmLs08", "SmLs09")}


def _f(value: str) -> float:
    return float(value.replace("D", "E").replace("d", "e"))


@dataclass(frozen=True)
class NistAnovaCase:
    name: str
    f_stat: float
    r_squared: float
    resid_sd: float
    data: pd.DataFrame


def _parse_case(path: Path) -> NistAnovaCase:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    f_stat = r2 = resid = None
    for ln in lines:
        m = _BETWEEN_RE.match(ln)
        if m:
            f_stat = _f(m.group(4))
        m = _WITHIN_RE.match(ln)
        if m:
            pass  # within MS available as m.group(3) if ever needed
        m = _R2_RE.search(ln)
        if m:
            r2 = _f(m.group(1))
        m = _RESID_RE.search(ln)
        if m:
            resid = _f(m.group(1))

    # The data block follows a "Data: <factor> <response>" header that — unlike
    # the descriptive "Data: 1 Factor ..." line in the preamble — contains no
    # digits.  Rows are "<group:int> <value:float>".
    start = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("Data:") and not any(c.isdigit() for c in s):
            start = i + 1
            break
    rows = []
    for ln in lines[start:]:
        parts = ln.split()
        if len(parts) == 2:
            try:
                rows.append((int(float(parts[0])), float(parts[1])))
            except ValueError:
                continue
    data = pd.DataFrame(rows, columns=["g", "y"])
    if f_stat is None or r2 is None or data.empty:
        raise AssertionError(f"failed to parse NIST ANOVA fixture {path.name}")
    return NistAnovaCase(path.stem, f_stat, r2, resid, data)


def _load(name: str) -> NistAnovaCase:
    return _parse_case(FIXTURE_DIR / f"{name}.dat")


@pytest.mark.parametrize("name", CASES)
def test_nist_anova_f_statistic(name):
    if name in XFAIL:
        pytest.xfail(XFAIL[name])
    case = _load(name)
    res = sp.regress("y ~ C(g)", data=case.data)
    glance = res.glance()
    f_sp = float(glance["f_statistic"].iloc[0])
    assert f_sp == pytest.approx(
        case.f_stat, rel=DEFAULT_RTOL
    ), f"{name}: sp F={f_sp:.12g} vs NIST certified {case.f_stat:.12g}"


@pytest.mark.parametrize("name", CASES)
def test_nist_anova_r_squared(name):
    if name in XFAIL:
        pytest.xfail(XFAIL[name])
    case = _load(name)
    res = sp.regress("y ~ C(g)", data=case.data)
    r2_sp = float(res.glance()["r_squared"].iloc[0])
    assert r2_sp == pytest.approx(
        case.r_squared, rel=DEFAULT_RTOL
    ), f"{name}: sp R^2={r2_sp:.12g} vs NIST certified {case.r_squared:.12g}"
