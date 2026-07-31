"""Reference parity: ``sp.honest_did`` native backend vs R ``HonestDiD``.

``sp.honest_did`` ships two backends and they do **not** compute the same
object:

* ``method='smoothness'`` is solved natively as the true Rambachan-Roth
  fixed-length confidence interval (``statspai.did._flci``): a convex program
  for the optimal affine estimator over the event-study covariance.
* ``method='relative_magnitude'`` is still a worst-case-bias approximation,
  ``θ̂ ± M̄·max|δ_pre| ± z·SE``, and warns.

Both backends are now handed the **same full event-study covariance**,
recovered from the Callaway-Sant'Anna influence functions. That matters: the R
backend used to build ``sigma <- diag(ses^2)`` itself, discarding the
cross-period covariance, so the two were solving different problems and could
not be compared. With matched inputs the native FLCI and R ``HonestDiD`` agree
to ~7e-5 on the interval width, the residual being ``HonestDiD``'s Monte-Carlo
folded-normal quantile (10^6 draws, ~2e-3 of quantile error; StatsPAI inverts
the CDF exactly).

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


def test_smoothness_does_not_warn(cs_result):
    """The FLCI is exact, so it must not carry an approximation warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        sp.honest_did(cs_result, e=0, method="smoothness", m_grid=_M_GRID)


def test_relative_magnitude_still_warns(cs_result):
    """That path is still an approximation and must announce itself."""
    with pytest.warns(UserWarning, match="worst-case-bias"):
        sp.honest_did(cs_result, e=0, method="relative_magnitude", m_grid=_M_GRID)


def test_native_smoothness_is_not_additive_in_m(cs_result):
    """A real FLCI is non-linear in M; the old approximation was additive.

    The old path widened each side by exactly M, which is the signature this
    guards against coming back.
    """
    out = _native(cs_result, "smoothness").set_index("M")
    additive_lower = out.loc[0.0, "ci_lower"] - 0.01
    assert out.loc[0.01, "ci_lower"] != pytest.approx(additive_lower, abs=1e-6)


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
def test_smoothness_native_flci_matches_r(cs_result):
    """The native FLCI must reproduce R HonestDiD, not merely approximate it.

    Tolerance is set by ``HonestDiD``'s Monte-Carlo folded-normal quantile
    (10^6 draws), not by anything StatsPAI does — the native quantile is exact.
    """
    native = _native(cs_result, "smoothness").set_index("M")
    ref = _r(cs_result, "smoothness").set_index("M")

    for m in _M_GRID:
        for col in ("ci_lower", "ci_upper"):
            assert native.loc[m, col] == pytest.approx(ref.loc[m, col], abs=1e-3), (
                f"M={m} {col}: native {native.loc[m, col]:.6f} "
                f"vs HonestDiD {ref.loc[m, col]:.6f}"
            )


@requires_r
def test_both_backends_receive_the_same_covariance(cs_result):
    """Widths must agree tightly, which only happens with matched sigma.

    The R backend used to build diag(se^2) itself; with the cross-period
    covariance dropped the two backends solved different problems and their
    widths differed by ~10%.
    """
    native = _native(cs_result, "smoothness").set_index("M")
    ref = _r(cs_result, "smoothness").set_index("M")
    nat_w = native["ci_upper"] - native["ci_lower"]
    ref_w = ref["ci_upper"] - ref["ci_lower"]
    for m in _M_GRID:
        assert abs(nat_w.loc[m] - ref_w.loc[m]) < 1e-3
