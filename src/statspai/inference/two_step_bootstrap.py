"""Bootstrap of a two-step estimator with a generated regressor.

``sp.two_step_bootstrap(step1, step2, data, cluster=...)`` re-runs *both*
steps on every resample: ``step1`` builds the generated regressor (worker
fixed effects, a first-stage prediction, a peer mean) from the resampled
data, ``step2`` fits the model of interest on its output. The spread of the
step-2 coefficients across resamples then includes the estimation noise of
step 1, which the standard errors step 2 reports on its own leave out.

It is ``sp.bootstrap(data, lambda d: step2(step1(d)), cluster=...)`` with
the two steps named, the step-2 standard errors on the original data kept
next to the bootstrap ones (``model_info['naive_se']`` and the ratio), and
the same resampling engine: pairs cluster bootstrap by default when
``cluster`` is given, ``idcluster`` to give every drawn copy of a cluster
its own id.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from ..exceptions import ColumnNotFound, MethodIncompatibility
from .basic_tests import _check_alpha
from .bootstrap import _as_statistic_vector, _bootstrap_vector

__all__ = ["two_step_bootstrap"]


def _naive_se(result: Any, names: list) -> Optional[pd.Series]:
    for attr in ("std_errors", "se", "bse"):
        se = getattr(result, attr, None)
        if isinstance(se, pd.Series):
            return se.reindex(names).astype(float)
    return None


def two_step_bootstrap(
    step1: Callable[[pd.DataFrame], pd.DataFrame],
    step2: Callable[[pd.DataFrame], Any],
    data: pd.DataFrame,
    *,
    cluster: Optional[str] = None,
    n_boot: int = 999,
    seed: Optional[int] = None,
    alpha: float = 0.05,
    idcluster: Optional[str] = None,
) -> Any:
    """Cluster (or row) bootstrap of a generated-regressor estimator.

    Parameters
    ----------
    step1 : callable
        ``DataFrame -> DataFrame``: builds the generated regressor(s) on the
        data it is given (e.g. estimates ``sp.akm`` and merges the worker
        effects back, or adds a first-stage fitted value).
    step2 : callable
        ``DataFrame -> result``: the model of interest on ``step1``'s
        output. It returns a fitted result with ``.params`` (any StatsPAI
        regression), a Series, or an array.
    data : DataFrame
        The original data, resampled as a whole.
    cluster : str, optional
        Resample whole clusters (pairs cluster bootstrap) instead of rows.
        Choose the coarsest level at which step 1 and step 2 errors are
        correlated (e.g. the firm when step 1 pools a firm's workers).
    n_boot : int, default 999
        Replications.
    seed : int, optional
        Seed of the resampling (``numpy.random.RandomState``).
    alpha : float, default 0.05
        Level of the percentile intervals in ``model_info``.
    idcluster : str, optional
        Name of a new column with a distinct id for each drawn cluster
        (Stata ``idcluster()``); use it in step 1 / step 2 when the cluster
        is also a fixed effect, so that a cluster drawn twice counts twice.

    Returns
    -------
    EconometricResults
        Step-2 point estimates on the original data with bootstrap
        standard errors and covariance. ``model_info`` adds ``naive_se``
        (step 2's own standard errors on the original data, when it reports
        them), ``se_ratio`` (bootstrap / naive), the percentile intervals,
        the replicates and the count of failed replications.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(60), 5)
    >>> z = rng.normal(size=300); d = z + rng.normal(size=300)
    >>> df = pd.DataFrame({"g": g, "z": z, "d": d, "y": 0.5 * d + rng.normal(size=300)})
    >>> def first(frame):
    ...     b = np.polyfit(frame.z, frame.d, 1)
    ...     return frame.assign(dhat=np.polyval(b, frame.z))
    >>> res = sp.two_step_bootstrap(first, lambda f: sp.regress("y ~ dhat", data=f),
    ...                             df, cluster="g", n_boot=99, seed=1)
    >>> "naive_se" in res.model_info
    True

    References
    ----------
    cameron2008bootstrap
    """
    if not callable(step1) or not callable(step2):
        raise MethodIncompatibility(
            "two_step_bootstrap: step1 and step2 must be callables "
            "(DataFrame -> DataFrame and DataFrame -> result).",
            recovery_hint="Pass functions, e.g. step1=lambda d: d.assign(...), "
            "step2=lambda d: sp.regress('y ~ x', data=d).",
        )
    _check_alpha(alpha, "two_step_bootstrap")
    if int(n_boot) < 10:
        raise MethodIncompatibility(
            f"two_step_bootstrap: n_boot={n_boot}; at least 10 replications "
            "are needed for a standard error.",
            recovery_hint="Use a few hundred replications or more.",
        )
    if idcluster is not None and cluster is None:
        raise MethodIncompatibility(
            "two_step_bootstrap: idcluster= requires cluster=.",
            recovery_hint="Pass cluster= or drop idcluster=.",
        )
    if cluster is not None:
        if cluster not in data.columns:
            raise ColumnNotFound(
                f"two_step_bootstrap: cluster={cluster!r} is not a column.",
                recovery_hint="Pass an existing cluster column.",
            )
        if data[cluster].isna().any():
            raise MethodIncompatibility(
                f"two_step_bootstrap: cluster={cluster!r} has missing ids.",
                recovery_hint="Drop or fill those rows first.",
                diagnostics={"n_missing": int(data[cluster].isna().sum())},
            )
        if data[cluster].nunique() < 2:
            raise MethodIncompatibility(
                "two_step_bootstrap: one cluster; resampling clusters needs "
                "at least two.",
                recovery_hint="Use a finer cluster or drop cluster=.",
            )

    def statistic(frame: pd.DataFrame) -> Any:
        stage1 = step1(frame)
        if not isinstance(stage1, pd.DataFrame):
            raise MethodIncompatibility(
                "two_step_bootstrap: step1 must return a DataFrame, got "
                f"{type(stage1).__name__}.",
                recovery_hint="Return the data with the generated column(s) " "added.",
            )
        return step2(stage1)

    base = data
    if idcluster is not None:
        base = data.assign(**{idcluster: pd.factorize(data[cluster])[0]})
    fitted = statistic(base)
    names, theta = _as_statistic_vector(fitted)
    if names is None:
        names, theta = ["stat"], np.atleast_1d(np.asarray(theta, dtype=float))
    rng = np.random.RandomState(seed)
    res = _bootstrap_vector(
        data,
        statistic,
        names,
        np.asarray(theta, dtype=float),
        n_boot=int(n_boot),
        cluster=cluster,
        block=None,
        idcluster=idcluster,
        alpha=alpha,
        rng=rng,
        seed=seed,
    )
    naive = _naive_se(fitted, names)
    res.model_info["model_type"] = "Two-step bootstrap"
    res.model_info["method"] = "two-step bootstrap" + (
        f", cluster({cluster})" if cluster else ""
    )
    res.model_info["naive_se"] = naive
    res.model_info["se_ratio"] = res.std_errors / naive if naive is not None else None
    res.model_info["step2_result"] = fitted
    return res
