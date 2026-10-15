"""``sp.honest_did`` fails loudly instead of returning a look-alike interval.

* An event study without any pre-treatment coefficient cannot calibrate
  either restriction. The native backend used to fall back to
  ``theta +/- z*SE`` (relative magnitudes) under a warning that said the
  covariance was unavailable, a table that reads like a robustness result.
  It now raises ``DataInsufficient``, as the R backend does.
* ``backend='r'`` read HonestDiD's answer from R's stdout; the C-F path's
  solver prints to stdout, so the parse raised a bare ``JSONDecodeError``.
  The answer now goes through a file, and an unreadable answer raises
  ``ConvergenceFailure`` carrying R's stdout / stderr.
"""

from __future__ import annotations

import importlib
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult
from statspai.exceptions import ConvergenceFailure, DataInsufficient

hd = importlib.import_module("statspai.did.honest_did")


def _result(times, att, se):
    es = pd.DataFrame({"relative_time": times, "att": att, "se": se})
    vcov = pd.DataFrame(np.diag(np.asarray(se) ** 2), index=times, columns=times)
    return CausalResult(
        method="x",
        estimand="ATT",
        estimate=float(att[list(times).index(0)]),
        se=float(se[0]),
        pvalue=0.0,
        ci=(0.0, 1.0),
        alpha=0.05,
        n_obs=100,
        model_info={"event_study": es, "event_study_vcov": vcov},
    )


@pytest.mark.parametrize("method", ["smoothness", "relative_magnitude"])
def test_no_pre_periods_raises(method):
    res = _result([0, 1, 2], [0.4, 0.3, 0.2], [0.1, 0.1, 0.1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no fallback warning either
        with pytest.raises(DataInsufficient, match="pre-treatment") as err:
            sp.honest_did(res, e=0, method=method)
    assert err.value.recovery_hint
    assert err.value.diagnostics["available"] == [0, 1, 2]


def test_one_pre_period_still_runs():
    res = _result([-2, 0, 1], [0.01, 0.4, 0.3], [0.05, 0.1, 0.1])
    out = sp.honest_did(res, e=0, m_grid=[0.0, 0.5], method="relative_magnitude")
    assert out.attrs["interval"] == "arp_c_lf"


def test_unreadable_r_answer_is_a_typed_error(monkeypatch):
    """R exits 0 but leaves no readable table (simulated, no R needed)."""
    res = _result([-2, 0, 1], [0.01, 0.4, 0.3], [0.05, 0.1, 0.1])
    monkeypatch.setattr(hd, "_find_rscript", lambda: "Rscript")

    def fake_run(args, **kw):
        return subprocess.CompletedProcess(
            args, 0, stdout="Polishing not needed\n", stderr="a warning\n"
        )

    monkeypatch.setattr(hd.subprocess, "run", fake_run)
    with pytest.raises(ConvergenceFailure, match="Polishing not needed") as err:
        sp.honest_did(res, e=0, backend="r", honestdid_method="C-F", m_grid=[0.1])
    assert "a warning" in str(err.value)
    assert err.value.diagnostics["honestdid_method"] == "C-F"


def _has_r_honestdid() -> bool:
    rscript = hd._find_rscript()
    if rscript is None:
        return False
    ok = subprocess.run(
        [
            rscript,
            "-e",
            "quit(status = !(requireNamespace('HonestDiD', quietly=TRUE) && "
            "requireNamespace('jsonlite', quietly=TRUE)))",
        ],
        capture_output=True,
    )
    return ok.returncode == 0


@pytest.mark.slow  # Rscript round trip
def test_r_backend_c_f_returns_a_table():
    """HonestDiD's C-F solver prints to stdout; the answer must still parse."""
    if not _has_r_honestdid():
        pytest.skip("R with the HonestDiD and jsonlite packages is required")
    res = _result(
        [-3, -2, 0, 1, 2],
        [0.01, -0.02, 0.5, 0.4, 0.3],
        [0.05, 0.05, 0.1, 0.1, 0.1],
    )
    out = sp.honest_did(res, e=0, backend="r", honestdid_method="C-F", m_grid=[0.1])
    assert np.isfinite(out[["ci_lower", "ci_upper"]].to_numpy()).all()
    assert out.attrs["interval"] == "arp_c_f"
    assert out.attrs["code_path"] == "r"
