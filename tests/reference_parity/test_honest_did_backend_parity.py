"""Reference parity: ``sp.honest_did`` native backend vs R ``HonestDiD``.

``sp.honest_did`` ships two backends and they do **not** compute the same
object:

* ``backend='native'`` returns a *worst-case-bias* interval,
  ``θ̂ ± bias_bound ± z_{α/2}·SE`` — the worst-case bias added to an ordinary
  Wald interval.
* ``backend='r'`` delegates to the R ``HonestDiD`` package, which solves the
  Rambachan-Roth partial-identification problem (FLCI for the smoothness
  restriction, C-LF for relative magnitudes) using the full pre-period
  covariance structure.

They agree closely under ``method='relative_magnitude'`` and diverge
materially under ``method='smoothness'``, where the native interval can be
**narrower** than the reference and therefore overstate robustness. That is a
real trap for anyone reporting native output as "honest DiD", so
``backend='native'`` now warns, and this module pins the relationship in both
directions: agreement where it exists, divergence where it does not.

Implementing the true FLCI natively is tracked as future work; until then the
R backend is the publication-grade path.

Data provenance
---------------
``tests/orig_parity/data/02_mpdta_original.csv``, SHA256
``1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661`` —
verified byte-identical to a rebuild of ``did::mpdta`` from the R package.

References
----------
- Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel
  Trends." *Review of Economic Studies*, 90(5), 2555-2591.
  [@rambachan2023more]
"""

from __future__ import annotations

import hashlib
import pathlib
import shutil
import subprocess
import warnings

import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)
_MPDTA_SHA256 = "1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661"

_M_GRID = [0.0, 0.01, 0.02]


def _has_r_honestdid() -> bool:
    if shutil.which("Rscript") is None:
        return False
    try:
        out = subprocess.run(
            ["Rscript", "-e", 'cat("HonestDiD" %in% rownames(installed.packages()))'],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.SubprocessError, OSError):  # pragma: no cover
        return False
    return "TRUE" in out.stdout


requires_r = pytest.mark.skipif(
    not _has_r_honestdid(),
    reason="R with the HonestDiD package is required for this parity check",
)


@pytest.fixture(scope="module")
def cs_result():
    if not _MPDTA.exists():  # pragma: no cover - fixture shipped with the repo
        pytest.skip(f"locked mpdta fixture missing: {_MPDTA}")
    digest = hashlib.sha256(_MPDTA.read_bytes()).hexdigest()
    assert (
        digest == _MPDTA_SHA256
    ), f"mpdta fixture changed; expected {_MPDTA_SHA256}, got {digest}"
    mp = pd.read_csv(_MPDTA)
    return sp.callaway_santanna(mp, y="lemp", g="first_treat", t="year", i="countyreal")


def _native(cs_result, method):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.honest_did(
            cs_result, e=0, method=method, backend="native", m_grid=_M_GRID
        )


def _r(cs_result, method):
    return sp.honest_did(cs_result, e=0, method=method, backend="r", m_grid=_M_GRID)


def test_native_backend_warns_it_is_not_the_rr_confidence_set(cs_result):
    """The approximation must announce itself — silent output reads as HonestDiD."""
    with pytest.warns(UserWarning, match="worst-case-bias"):
        sp.honest_did(cs_result, e=0, method="smoothness", m_grid=_M_GRID)


def test_native_smoothness_is_additive_in_m(cs_result):
    """Pin the native formula: the interval widens by exactly M per unit of M.

    This is the signature of ``θ̂ ± M·n_drift ± z·SE`` and is what separates it
    from a real FLCI, whose width is non-linear in M.
    """
    out = _native(cs_result, "smoothness").set_index("M")
    # e=0 -> n_drift = 1, so each 0.01 step widens each side by exactly 0.01.
    assert out.loc[0.01, "ci_lower"] == pytest.approx(
        out.loc[0.0, "ci_lower"] - 0.01, abs=1e-6
    )
    assert out.loc[0.02, "ci_upper"] == pytest.approx(
        out.loc[0.0, "ci_upper"] + 0.02, abs=1e-6
    )


@requires_r
def test_relative_magnitude_native_tracks_r(cs_result):
    """Under relative magnitudes the two backends agree to a few percent."""
    native = _native(cs_result, "relative_magnitude").set_index("M")
    ref = _r(cs_result, "relative_magnitude").set_index("M")

    for m in _M_GRID:
        for col in ("ci_lower", "ci_upper"):
            assert native.loc[m, col] == pytest.approx(ref.loc[m, col], abs=5e-3), (
                f"M={m} {col}: native {native.loc[m, col]:.6f} "
                f"vs HonestDiD {ref.loc[m, col]:.6f}"
            )


@requires_r
def test_smoothness_native_diverges_from_r_and_can_be_narrower(cs_result):
    """Pin the known divergence, including its dangerous direction.

    If a future native implementation actually solves the FLCI this test should
    be replaced by an equality check — it is deliberately written to fail if
    the gap silently changes character.
    """
    native = _native(cs_result, "smoothness").set_index("M")
    ref = _r(cs_result, "smoothness").set_index("M")

    nat_w = native["ci_upper"] - native["ci_lower"]
    ref_w = ref["ci_upper"] - ref["ci_lower"]

    # On this panel the native interval is narrower than HonestDiD at *every*
    # M on the grid — i.e. it uniformly understates the uncertainty a
    # Rambachan-Roth smoothness restriction actually implies.  That is the
    # direction that matters: a user reading native output concludes their
    # result is more robust to parallel-trends violations than it is.
    for m in _M_GRID:
        assert nat_w.loc[m] < ref_w.loc[m], (
            f"M={m}: native width {nat_w.loc[m]:.6f} is no longer below "
            f"HonestDiD's {ref_w.loc[m]:.6f} — if the native FLCI was "
            "implemented, replace this test with an equality check"
        )

    # The gap grows with M, because additive widening cannot track the FLCI.
    assert (ref_w.loc[0.02] - nat_w.loc[0.02]) > (ref_w.loc[0.0] - nat_w.loc[0.0])
