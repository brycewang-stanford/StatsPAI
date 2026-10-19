"""Nonparametric bootstrap of the potential-outcome mediation estimator.

``sp.mediate(inference='bootstrap')`` (the default) used to refuse
``interaction=True``, a non-linear mediator / outcome model and
``treat_values=``: those are estimated from the four potential-outcome
means (:mod:`._po_means`, :mod:`._po_means_nonlinear`), which only had the
sandwich variance of ``inference='robust'``. Here the same point estimator
is re-run on case-resampled data, as R ``mediation::mediate(boot = TRUE)``
does: standard errors are the bootstrap standard deviations and intervals
are percentile intervals. Point estimates are those of ``'robust'``.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd

from ..exceptions import ConvergenceFailure, DataInsufficient


def bootstrap_po_effects(
    data: pd.DataFrame,
    columns: List[str],
    fit_once: Callable[[pd.DataFrame], Dict[str, Any]],
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Re-fit ``fit_once`` on ``n_boot`` case resamples of the complete rows.

    Returns the ``(n_boot, n_effects)`` draws of every effect reported by
    the full-sample fit (the effects table plus the proportion mediated),
    their names, and how many replications failed to fit.
    """
    df = data.dropna(subset=columns).reset_index(drop=True)
    n = len(df)
    full = fit_once(df)
    names = list(full["effects"]["effect"]) + ["Prop. Mediated"]
    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, len(names)), np.nan)
    failed = 0
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fb = fit_once(df.iloc[idx].reset_index(drop=True))
        except (
            np.linalg.LinAlgError,
            ConvergenceFailure,
            DataInsufficient,
            ValueError,
        ):
            # a resample without variation in a 0/1 mediator or with a
            # separated logit cannot be fitted; counted and reported
            failed += 1
            continue
        est = fb["effects"].set_index("effect")["estimate"]
        draws[b, :-1] = est.reindex(names[:-1]).to_numpy(dtype=float)
        draws[b, -1] = float(fb["prop_mediated"]["estimate"].iloc[0])
    return {"full": full, "names": names, "draws": draws, "n_failed": failed}
