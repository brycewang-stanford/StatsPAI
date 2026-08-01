"""Dependency-free argument validators shared across estimator families.

Deliberately importing nothing beyond numpy and ``..exceptions``. The
first home for ``require_bool_flag`` was ``core/_vcov.py``, which reads
as the natural place for a standard-error validator — but ``_vcov``
imports ``._numba_kernels`` at module level, so routing five estimator
modules through it dragged numba into a plain ``import statspai`` and
tripped the cold-import budget. A validator has no business costing an
import; hence this module.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import MethodIncompatibility


def require_bool_flag(value, *, argument: str = "robust") -> bool:
    """Reject a non-boolean where the signature promises an on/off flag.

    ``robust`` is a *boolean* switch on eight estimators and a *string*
    HC-type selector on the regression family (``robust="HC1"``).
    ``statspai._house_style.ROBUST_BOOL_HINTS`` names that split the
    highest-impact hazard in the signature surface. It was also a silent
    one: every bool-typed site accepted the string form, read it as truthy,
    and returned its default sandwich.

    ``robust="cluster"`` is the case that costs someone a result. Clustering
    is a separate ``cluster=`` argument on all of these estimators, so the
    call returned *unclustered* standard errors and gave no sign the request
    had been dropped — output that is correct for what was computed and is
    not what was asked for.

    One implementation, per CLAUDE.md §4; the DiD and dynamic-panel families
    alias this rather than carrying their own copies.
    """
    if not isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{argument}` must be boolean.",
            recovery_hint=(
                f"Pass `{argument}=True` or `{argument}=False`. A standard-error "
                f"*type* string such as 'HC1' is not accepted here; for clustered "
                f"standard errors pass `cluster=<column>` instead."
            ),
            diagnostics={"argument": argument, "type": type(value).__name__},
        )
    return bool(value)
