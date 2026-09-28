"""Central import helper for optional dependencies.

Optional extras (``fixest``, ``bayes``, ``neural``, ``plotting`` ...) must be
imported lazily (CLAUDE.md §4).  When one is missing, callers should get a
:class:`~statspai.exceptions.MissingDependencyError` — an ``ImportError`` *and*
a ``StatsPAIError`` — carrying the exact ``pip install`` command, instead of a
bare ``ImportError`` an agent cannot act on.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Optional

from .exceptions import MissingDependencyError

__all__ = ["require_optional"]


def require_optional(
    module: str,
    *,
    extra: Optional[str] = None,
    pip_name: Optional[str] = None,
    purpose: str = "",
) -> ModuleType:
    """Import ``module`` or raise a structured ``MissingDependencyError``.

    Parameters
    ----------
    module : str
        Importable module name, e.g. ``"pyfixest"``.
    extra : str, optional
        StatsPAI extra that installs it (``pip install "statspai[<extra>]"``).
    pip_name : str, optional
        PyPI distribution name when it differs from ``module``.
    purpose : str, optional
        Short description of what the module is needed for.

    Returns
    -------
    module
        The imported module.

    Raises
    ------
    MissingDependencyError
        If the import fails. The original ``ImportError`` is chained as
        ``__cause__``.

    Examples
    --------
    >>> from statspai._optional_deps import require_optional
    >>> require_optional("json").__name__
    'json'
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        err = MissingDependencyError.for_package(
            module.split(".")[0], extra=extra, pip_name=pip_name, purpose=purpose
        )
        err.diagnostics["import_error"] = f"{type(exc).__name__}: {exc}"
        raise err from exc
