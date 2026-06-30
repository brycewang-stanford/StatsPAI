"""Cross-backend correctness gate (via the honest HDFE benchmark harness).

The benchmark in ``benchmarks/cross_library/`` only makes a defensible *speed*
claim if every backend computes the *same* estimator. This test promotes that
precondition to a CI gate: StatsPAI's two HDFE paths (``sp.feols`` and the
native ``sp.absorb_ols`` kernel) must agree with ``pyfixest`` on the slope of a
two-way fixed-effects fit, to a tight tolerance. R ``fixest`` is excluded here
(subprocess; not assumed present in CI) — it is exercised by the harness itself.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HARNESS = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "cross_library"
    / "hdfe_benchmark.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location("_xlib_bench", _HARNESS)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def harness():
    return _load_harness()


def test_python_backends_agree(harness):
    """sp.feols, sp.absorb_ols and pyfixest must compute the same slope."""
    pytest.importorskip("pyfixest")
    df = harness.make_panel(4000)
    rows = [
        harness._bench_sp_feols(df, repeats=1),
        harness._bench_sp_native(df, repeats=1),
        harness._bench_pyfixest(df, repeats=1),
    ]
    ok = [r for r in rows if r.status == "ok"]
    assert len(ok) >= 2, f"need ≥2 runnable backends, got {[r.status for r in rows]}"
    agree = harness.coefficient_agreement(ok, reference="pyfixest", rtol=1e-6)
    assert agree["agree"], (
        f"HDFE backends disagree: max rel diff {agree['max_rel_diff']:.2e} "
        f"vs {agree['reference']}"
    )


def test_harness_refuses_to_write_outside_its_directory(harness):
    """Safety: the harness must never write into Paper-JSS / Track-C."""
    with pytest.raises(ValueError):
        harness._assert_under_write_root(Path("Paper-JSS/manuscript/evil.json"))
    # its own directory is allowed
    harness._assert_under_write_root(_HARNESS.parent / "ok.json")
