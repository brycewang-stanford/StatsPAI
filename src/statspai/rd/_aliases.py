"""User-friendly RD aliases (Chinese-market & agent discoverability).

These thin wrappers re-export the canonical estimators under names the v3
methodology document uses, so that ``sp.geographic_rd(...)`` and
``sp.multi_cutoff_rd(...)`` work out-of-the-box alongside the R/Stata-style
``sp.rdms`` and ``sp.rdmc``.
"""

from __future__ import annotations

from typing import Any

from ..core.results import CausalResult
from .rdmulti import rdmc, rdms, RDMultiResult
from .rd2d import rd2d
from .multi_score import rd_multi_score, MultiScoreRDResult

__all__ = [
    "multi_cutoff_rd",
    "geographic_rd",
    "boundary_rd",
    "multi_score_rd",
]


def multi_cutoff_rd(*args: Any, **kwargs: Any) -> RDMultiResult:
    """User-friendly alias for :func:`sp.rdmc` (multi-cutoff RD).

    See :func:`statspai.rd.rdmulti.rdmc` for full documentation.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 600
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.8 * (x >= 0) + 0.5 * x + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> res = sp.multi_cutoff_rd(
    ...     df, y="y", x="x", cutoffs=[-0.4, 0.0, 0.4]
    ... )
    >>> res.n_cutoffs
    3
    >>> len(res.cutoff_results)
    3
    >>> round(float(res.pooled_estimate), 3)
    0.258
    """
    return rdmc(*args, **kwargs)


def geographic_rd(*args: Any, **kwargs: Any) -> CausalResult:
    """User-friendly alias for :func:`sp.rdms` (multi-score RD).

    Geographic RD is the most common multi-score special case: the running
    score is a signed distance to a political boundary.  Cattaneo et al.
    (2024) dispatch it via ``rdms``; this alias makes the intent explicit.

    See :func:`statspai.rd.rdmulti.rdms` for the full signature.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> x1 = rng.uniform(-1, 1, n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> treat = ((x1 >= 0) & (x2 >= 0)).astype(float)
    >>> y = 0.8 * treat + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    >>> res = sp.geographic_rd(df, y="y", x1="x1", x2="x2", bandwidth=0.5)
    >>> type(res).__name__
    'CausalResult'
    >>> bool(hasattr(res, "estimate"))
    True

    References
    ----------
    keele2015geographic
    """
    return rdms(*args, **kwargs)


def boundary_rd(*args: Any, **kwargs: Any) -> CausalResult:
    """User-friendly alias for :func:`sp.rd2d` (boundary discontinuity design).

    Cattaneo, Titiunik & Yu (2025) boundary discontinuity design for 2D
    running variables (lat/long, for example).

    See :func:`statspai.rd.rd2d.rd2d` for the full signature.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 1500
    >>> x1 = rng.uniform(-1, 1, n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> treat = (x1 >= 0).astype(int)
    >>> y = 0.7 * treat + 0.4 * x1 + 0.2 * x2 + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({"x1": x1, "x2": x2, "treat": treat, "y": y})
    >>> res = sp.boundary_rd(df, y="y", x1="x1", x2="x2", treatment="treat")
    >>> type(res).__name__
    'CausalResult'
    >>> bool(hasattr(res, "estimate"))
    True

    References
    ----------
    cattaneo2025boundary
    """
    return rd2d(*args, **kwargs)


def multi_score_rd(*args: Any, **kwargs: Any) -> MultiScoreRDResult:
    """User-friendly alias for :func:`sp.rd_multi_score`.

    Multi-score RD when eligibility depends on more than one discontinuous
    rule (e.g. income AND age thresholds).  Separate from boundary RDD in
    that the rules are axis-aligned.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 800
    >>> x1 = rng.uniform(-1, 1, n)
    >>> x2 = rng.uniform(-1, 1, n)
    >>> treat = (x1 >= 0) & (x2 >= 0)
    >>> y = (0.8 * treat + 0.5 * x1 + 0.3 * x2
    ...      + rng.normal(0, 0.3, n))
    >>> df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    >>> res = sp.multi_score_rd(
    ...     df, y="y", running_vars=["x1", "x2"], cutoffs=[0.0, 0.0]
    ... )
    >>> res.n_obs
    800
    >>> round(float(res.boundary_effect), 3)
    0.137
    """
    return rd_multi_score(*args, **kwargs)
