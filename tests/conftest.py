"""pytest conftest — project-wide test configuration.

Defensive imports
-----------------
``scipy.optimize`` is pre-imported at session scope to stabilise the PyO3
type-registry table.  scipy ≥ 1.14 ships ``_highspy._core`` (a PyO3 shared
library) which registers the C++ type ``ObjSense`` at import time via
``pyo3::generic_type``.  If this module is ever removed from ``sys.modules``
and re-imported within the same process, PyO3's global type table rejects
the duplicate registration with::

    ImportError: generic_type: type "ObjSense" is already registered!

This can fire during large pytest sessions (300+ files) when coverage
tracing or the module-import machinery transiently unloads a dependency
chain that includes ``_highspy._core``.

By loading ``scipy.optimize`` (hence ``_highspy._core``) once at conftest
parse time — well before any test-file collection begins — we ensure the
PyO3 type stays registered under a stable ``sys.modules`` key for the
entire process lifetime.
"""

import importlib.util  # noqa: E402

import pytest  # noqa: E402
import scipy.optimize  # noqa: F401 — stabilise PyO3 type registry

_HAS_PYFIXEST = importlib.util.find_spec("pyfixest") is not None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):  # type: ignore[no-untyped-def]
    """Skip, rather than fail, a test that needs pyfixest where it cannot be.

    pyfixest ships no wheel for Python 3.9, so the ``fixest`` extra resolves
    to nothing there (see pyproject.toml) and every pyfixest-backed test is
    meant to skip. Tests written since forgot the ``importorskip`` guard: a
    full run on 3.9 failed 40 of them on ``MissingDependencyError``. This
    applies the rule in one place.

    It fires only when pyfixest is truly absent and the error names it, so
    it cannot hide a failure on an interpreter that has the package, and a
    test that asserts on the missing-dependency error itself (by mocking
    pyfixest away) is untouched: its error is caught inside the test.
    """
    outcome = yield
    if _HAS_PYFIXEST or outcome.excinfo is None:
        return
    exc = outcome.excinfo[1]
    if isinstance(exc, ImportError) and "pyfixest" in str(exc):
        pytest.skip(f"pyfixest is not installed: {exc}")
