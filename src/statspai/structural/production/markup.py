"""
De Loecker & Warzynski (2012) markup estimator on top of a fitted
production function.

Markup formula
--------------
For a flexible input ``v`` (typically materials), the firm-time markup is

    mu_it = theta_v_it * (P_it Q_it) / (P_v_it V_it)

where ``theta_v`` is the output elasticity of ``v`` and the second factor
is the inverse of the cost share of ``v``.  In logs and Cobb-Douglas,
``theta_v`` is just the elasticity beta_v from the production function;
for translog, ``theta_v`` is firm-time-varying.

This implementation accepts a fitted :class:`ProductionResult`, a
revenue column and a cost-of-input column, and computes ``mu_it``
firm-by-firm with the standard η-correction (subtract the i.i.d. shock
``eta`` from log revenue before forming the cost share).

Reference
---------
De Loecker, J. & Warzynski, F. (2012). Markups and firm-level export
status. American Economic Review, 102(6), 2437-2471.
[@deloecker2012markups]
"""

from __future__ import annotations


import numpy as np
import pandas as pd

from ._result import ProductionResult


def markup(
    result: ProductionResult,
    revenue: str,
    input_cost: str,
    flexible_input: str = "m",
    correct_eta: bool = True,
) -> pd.Series:
    """Compute De Loecker-Warzynski (2012) firm-time markups.

    Parameters
    ----------
    result : ProductionResult
        Fitted production function (output of :func:`olley_pakes`,
        :func:`levinsohn_petrin`, :func:`ackerberg_caves_frazer`, or
        :func:`wooldridge_prod`). Must have a coefficient for
        ``flexible_input``.
    revenue : str
        Column in ``result.sample`` with **log** firm-time revenue
        (P*Q in levels, log-transformed).
    input_cost : str
        Column with **log** firm-time expenditure on the flexible input
        (P_v * V in levels, log-transformed).
    flexible_input : str, default ``"m"``
        Name of the flexible input. Must exist in ``result.coef``.
    correct_eta : bool, default True
        If True, subtract the stage-1 i.i.d. shock ``eta`` from log
        revenue before forming the cost share, as recommended by
        De Loecker & Warzynski (2012, eq. 6).

    Returns
    -------
    pd.Series
        Firm-time markups ``mu_it`` aligned to ``result.sample.index``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for fid in range(120):
    ...     omega, k = rng.normal(0.0, 0.28), rng.normal(0.0, 0.5)
    ...     for t in range(8):
    ...         omega = 0.7 * omega + rng.normal(0.0, 0.20)
    ...         l = 0.5 * omega + 0.3 * k + rng.normal(0.0, 0.10)
    ...         m = 0.8 * omega + 0.5 * k + rng.normal(0.0, 0.05)
    ...         y = 0.6 * l + 0.35 * k + omega + rng.normal(0.0, 0.10)
    ...         rows.append({"id": fid, "year": t, "y": y, "l": l, "k": k, "m": m})
    >>> df = pd.DataFrame(rows)
    >>> df["log_revenue"] = df["y"] + np.log(1.20) + rng.normal(0, 0.05, len(df))
    >>> df["log_mat_cost"] = df["m"] + np.log(0.80) + rng.normal(0, 0.05, len(df))
    >>> # The flexible input must be a production-function coefficient, so fit
    >>> # ACF with materials ``m`` as a free input to expose its elasticity.
    >>> res = sp.acf(df, output="y", free=["l", "m"], state="k", proxy="m",
    ...              panel_id="id", time="year")
    >>> res.sample["log_revenue"] = df.loc[res.sample.index, "log_revenue"].to_numpy()
    >>> res.sample["log_mat_cost"] = df.loc[res.sample.index, "log_mat_cost"].to_numpy()
    >>> mu = sp.markup(res, revenue="log_revenue", input_cost="log_mat_cost",
    ...                flexible_input="m")
    >>> bool((mu > 0).all())
    True

    Notes
    -----
    For Cobb-Douglas, ``theta_v`` is the constant elasticity ``beta_v``.
    For translog (when the production fit was run with
    ``functional_form="translog"``), ``theta_v_it`` is firm-time-varying
    and read from ``result.model_info["elasticities"]``. The
    eta-correction in the markup formula transfers cleanly to translog
    (cf. De Loecker, Goldberg, Khandelwal & Pavcnik 2016, AER §III.B).

    References
    ----------
    De Loecker, J. & Warzynski, F. (2012). Markups and firm-level
    export status. American Economic Review, 102(6), 2437-2471.
    """
    sample = result.sample
    fform = (
        result.model_info.get("functional_form", "cobb-douglas")
        .lower()
        .replace("_", "-")
    )

    # Resolve theta_v_it: scalar for Cobb-Douglas, firm-time vector for translog.
    if fform in ("cobb-douglas", "cd"):
        if flexible_input not in result.coef:
            raise KeyError(
                f"Flexible input {flexible_input!r} not found in coef. "
                f"Available: {list(result.coef)}. "
                "If your flexible input is a state input or proxy, refit "
                "the production function with it as a free input."
            )
        theta_v = np.full(len(sample), float(result.coef[flexible_input]))
    elif fform == "translog":
        # Firm-time elasticities are pre-computed and stashed on the result.
        elasts = result.model_info.get("elasticities")
        if elasts is None:
            raise RuntimeError(
                "Translog ProductionResult is missing pre-computed "
                "elasticities. Refit with sp.acf / sp.olley_pakes / "
                "sp.levinsohn_petrin (functional_form='translog')."
            )
        if flexible_input not in elasts.columns:
            raise KeyError(
                f"Flexible input {flexible_input!r} not in elasticity "
                f"panel. Available raw inputs: {list(elasts.columns)}."
            )
        theta_v = elasts[flexible_input].to_numpy(dtype=float)
    else:
        raise NotImplementedError(
            f"Markup not supported for functional_form={fform!r}. "
            "Use 'cobb-douglas' or 'translog'."
        )

    for col in (revenue, input_cost):
        if col not in sample.columns:
            raise KeyError(
                f"Column {col!r} not in result.sample (have "
                f"{list(sample.columns)}). Pass a DataFrame whose "
                "columns include log revenue and log input expenditure."
            )

    # Cost share = (P_v * V) / (P * Q) in levels.
    # In logs: log_cost - log_revenue. Levels via exp.
    log_rev = sample[revenue].to_numpy(dtype=float)
    if correct_eta:
        log_rev = log_rev - sample["eta"].to_numpy(dtype=float)
    log_cost = sample[input_cost].to_numpy(dtype=float)
    cost_share = np.exp(log_cost - log_rev)

    mu = theta_v / cost_share
    return pd.Series(mu, index=sample.index, name="markup")
