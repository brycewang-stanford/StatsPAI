"""Turn named diagnostics' warnings off for a block of code.

StatsPAI reports what it notices about a fit as warnings (rows dropped for
missing values, standard errors below the level of assignment, a count of
specifications tried). In a simulation, or once a diagnostic has been read
and answered in the text, the same warning on every fit is noise. Filtering
it with :mod:`warnings` means matching the wording of the message, which
changes between releases. :func:`silence` names the diagnostic instead.

Only the warning is withheld. The diagnostic still runs and is still
recorded on the result (``result.violations()``, ``model_info``), so a
silenced fit can be audited afterwards.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import ContextManager, Dict, FrozenSet, Iterator, Sequence, Union

__all__ = ["silence", "is_silenced", "DIAGNOSTICS"]

#: The diagnostics that can be silenced, by the name they carry in
#: ``warning.diagnostics["test"]`` and in ``result.violations()``.
DIAGNOSTICS: Dict[str, str] = {
    "listwise_deletion": ("rows with a missing value in a model variable were dropped"),
    "specification_search": (
        "many distinct specifications of one outcome fitted on the same data"
    ),
    "treatment_cluster_level": (
        "standard errors computed below the level the treatment is assigned at"
    ),
    "staggered_twfe": (
        "two-way fixed effects with staggered adoption and forbidden comparisons"
    ),
}

_SILENCED: ContextVar[FrozenSet[str]] = ContextVar(
    "statspai_silenced_diagnostics", default=frozenset()
)


def is_silenced(test: object) -> bool:
    """Whether the warning of diagnostic ``test`` is withheld right now."""
    return isinstance(test, str) and test in _SILENCED.get()


@contextmanager
def _withheld(names: FrozenSet[str]) -> Iterator[None]:
    token = _SILENCED.set(_SILENCED.get() | names)
    try:
        yield
    finally:
        _SILENCED.reset(token)


def silence(diagnostics: Union[str, Sequence[str]]) -> ContextManager[None]:
    """Withhold the warnings of named diagnostics inside a ``with`` block.

    The diagnostics still run and stay on each result
    (``result.violations()``, ``model_info``); only ``warnings.warn`` is
    skipped. Use it where a diagnostic has been read and answered, or in a
    simulation loop, instead of filtering on the text of the message.

    Parameters
    ----------
    diagnostics : str or sequence of str
        One or more of ``'listwise_deletion'``, ``'specification_search'``,
        ``'treatment_cluster_level'``, ``'staggered_twfe'``. The name of a
        diagnostic is ``warning.diagnostics['test']`` on the warning it
        raises and ``test`` in ``result.violations()``.

    Returns
    -------
    context manager
        Also usable as a decorator. Blocks nest, and the setting is local
        to the current thread or task.

    Raises
    ------
    MethodIncompatibility
        For a name that is not a diagnostic, raised by the call itself, so
        that a typo cannot silence nothing unnoticed.

    Examples
    --------
    >>> import numpy as np, pandas as pd, warnings
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=50)})
    >>> df["y"] = df.x + rng.normal(size=50)
    >>> df.loc[:4, "x"] = np.nan
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("error")       # any warning would raise
    ...     with sp.silence("listwise_deletion"):
    ...         fit = sp.regress("y ~ x", data=df)
    >>> fit.model_info["listwise_deletion"]["n_rows_dropped_missing"]
    5
    """
    names = [diagnostics] if isinstance(diagnostics, str) else list(diagnostics)
    unknown = [n for n in names if n not in DIAGNOSTICS]
    if unknown or not names:
        from ..exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"silence: {unknown or names!r} is not a diagnostic that can be "
            f"silenced. Known: {sorted(DIAGNOSTICS)}.",
            recovery_hint="Use the name in warning.diagnostics['test'] (or "
            "'test' in result.violations()). Other warnings are filtered "
            "with warnings.filterwarnings.",
            diagnostics={"unknown": unknown, "known": sorted(DIAGNOSTICS)},
        )
    return _withheld(frozenset(names))
