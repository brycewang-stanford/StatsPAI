"""Tests for the optional HonestDiD R backend."""

import subprocess

import numpy as np
import pandas as pd
import pytest

from statspai.core.results import CausalResult
from statspai.did import honest_did
from statspai.did.honest_did import _find_rscript


def _parity_result():
    es = pd.DataFrame(
        {
            "relative_time": [-3, -2, -1, 0, 1, 2],
            "att": [0.01, -0.02, 0.0, 0.5, 0.4, 0.3],
            "se": [0.05, 0.05, 0.05, 0.10, 0.10, 0.10],
        }
    )
    return CausalResult(
        method="ParityHonestDiDRelMags",
        estimand="ATT(0)",
        estimate=0.5,
        se=0.10,
        pvalue=0.0,
        ci=(0.30, 0.70),
        alpha=0.05,
        n_obs=1000,
        model_info={"event_study": es},
    )


def _skip_unless_honestdid_available():
    rscript = _find_rscript()
    if rscript is None:
        pytest.skip("Rscript is not installed")
    probe = subprocess.run(
        [
            rscript,
            "-e",
            (
                "quit(status = as.integer("
                "!requireNamespace('HonestDiD', quietly=TRUE) || "
                "!requireNamespace('jsonlite', quietly=TRUE)))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("R packages HonestDiD/jsonlite are not installed")


def test_relative_magnitude_honestdid_backend_matches_reference_fixture():
    _skip_unless_honestdid_available()
    table = honest_did(
        _parity_result(),
        e=0,
        m_grid=[0.0, 2.0],
        method="relative_magnitude",
        backend="honestdid",
    )
    got = table.set_index("M")
    assert np.isclose(got.loc[0.0, "ci_lower"], 0.31031031031031)
    assert np.isclose(got.loc[0.0, "ci_upper"], 0.69069069069069)
    assert np.isclose(got.loc[2.0, "ci_lower"], 0.154154154154154)
    assert np.isclose(got.loc[2.0, "ci_upper"], 0.842842842842843)


def test_relative_magnitude_honestdid_backend_allows_conditional_method():
    _skip_unless_honestdid_available()
    table = honest_did(
        _parity_result(),
        e=0,
        m_grid=[0.0, 2.0],
        method="relative_magnitude",
        backend="honestdid",
        honestdid_method="Conditional",
    )
    got = table.set_index("M")
    assert np.isclose(got.loc[0.0, "ci_lower"], 0.3063063063063063)
    assert np.isclose(got.loc[0.0, "ci_upper"], 0.6946946946946948)
    assert np.isclose(got.loc[2.0, "ci_lower"], 0.1581581581581579)
    assert np.isclose(got.loc[2.0, "ci_upper"], 0.8428428428428427)


def test_relative_magnitude_honestdid_backend_rejects_incompatible_method():
    with pytest.raises(ValueError, match="honestdid_method"):
        honest_did(
            _parity_result(),
            e=0,
            m_grid=[0.0],
            method="relative_magnitude",
            backend="honestdid",
            honestdid_method="FLCI",
        )


def test_honest_did_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        honest_did(_parity_result(), backend="unknown")


def _rm_result(shift):
    """Post coefficients 2.0 / 2.1 minus ``shift``; pre-periods fixed."""
    es = pd.DataFrame(
        {
            "relative_time": [-3, -2, 0, 1],
            "att": [0.05, -0.03, 2.0 - shift, 2.1 - shift],
            "se": [0.08, 0.08, 0.08, 0.08],
        }
    )
    return CausalResult(
        method="RMShift",
        estimand="ATT(0)",
        estimate=2.0 - shift,
        se=0.08,
        pvalue=0.0,
        ci=(1.84 - shift, 2.16 - shift),
        alpha=0.05,
        n_obs=1000,
        model_info={"event_study": es},
    )


def test_r_backend_empty_grid_is_a_typed_error_with_a_grid_that_works():
    """An estimate 25 SE from zero lies outside HonestDiD's default RM grid
    (-/+ 20 sd around 0), so R accepts no grid point; this raised
    KeyError('ci_lower') before 2026-10-20 (jsonlite dropped R's lb = Inf /
    ub = -Inf). With the suggested grid the set is the one at estimate 0 on
    the default grid moved by 2: shifting the post coefficients by c moves
    the relative-magnitude set by c (the pre-periods are unchanged)."""
    from statspai.exceptions import ConvergenceFailure

    _skip_unless_honestdid_available()
    kw = dict(
        e=0,
        m_grid=[0.0, 0.5],
        method="relative_magnitude",
        backend="r",
        honestdid_method="Conditional",
    )
    with pytest.raises(ConvergenceFailure, match="misses the estimate") as info:
        honest_did(_rm_result(0.0), **kw)
    lb, ub = info.value.diagnostics["suggested_grid"]
    np.testing.assert_allclose([lb, ub], [2.0 - 1.6, 2.0 + 1.6], rtol=1e-12)
    far = honest_did(_rm_result(0.0), grid_lb=lb, grid_ub=ub, **kw)
    at_zero = honest_did(_rm_result(2.0), **kw)  # default grid covers it
    np.testing.assert_allclose(
        far[["ci_lower", "ci_upper"]].to_numpy(),
        at_zero[["ci_lower", "ci_upper"]].to_numpy() + 2.0,
        rtol=0,
        atol=1e-9,
    )
    assert far["ci_lower"].iloc[0] < 2.0 < far["ci_upper"].iloc[0]
