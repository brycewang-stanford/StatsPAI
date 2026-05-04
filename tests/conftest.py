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

import scipy.optimize  # noqa: F401 — stabilise PyO3 type registry
