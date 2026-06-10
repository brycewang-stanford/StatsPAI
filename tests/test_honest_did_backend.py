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
