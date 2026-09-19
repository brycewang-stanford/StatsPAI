"""de Chaisemartin-D'Haultfoeuille weights of a two-way fixed-effects regression.

Private primitive for the ``feTR`` decomposition of de Chaisemartin and
D'Haultfoeuille (2020): the TWFE coefficient on a binary (or group x period
level) treatment ``D`` in ``Y ~ D | G + T`` equals

    beta_fe = sum_{(g,t): D_gt != 0} w_gt * ATT_gt,
    w_gt    = P_gt * D_gt * eps_gt / sum_{(g,t)} P_gt * D_gt * eps_gt,

where ``P_gt`` is the (weighted) share of observations in cell ``(g, t)`` and
``eps_gt`` is the residual of the regression of ``D`` on group and period
fixed effects. Negative ``w_gt`` are the "negative weights" the paper warns
about. The two summary measures (``sensibility``: the smallest standard
deviation of cell effects compatible with ``beta_fe`` and ``ATT = 0``;
``sensibility2``: the smallest one compatible with ``beta_fe`` and cell
effects of the opposite sign) follow the same paper.

This module reproduces R ``TwoWayFEWeights::twowayfeweights(type = "feTR")``
(the authors' own package) line for line; every convention below is read from
``TwoWayFEWeights:::twowayfeweights_calculate`` and
``TwoWayFEWeights:::twowayfeweights_result`` (version 2.1.0):

* a treatment that varies within a ``(G, T)`` cell is replaced by its
  *unweighted* cell mean before anything else -- even when observation
  weights are supplied, because ``twowayfeweights_normalize_var`` runs before
  the weights are attached (``mean(D)`` by ``(G, T)``);
* weights below ``1e-10`` in absolute value are set to exactly zero before
  they are counted;
* the standard deviation of the ``W`` ratios uses the ``sqrt(M / (M - 1))``
  small-sample factor with ``M`` the number of treated cells.

The cross-language test is
``tests/reference_parity/test_did_synth_R_parity.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

__all__ = ["dcdh_fe_weights"]

_ZERO_BELOW = 1e-10


def _two_way_residual(
    v: np.ndarray, g: np.ndarray, t: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Residual of a weighted regression of ``v`` on group and period dummies.

    Exact least squares on the dummy design (``lstsq``), not an iterative
    within transformation, so the residual carries no convergence error. The
    design has ``n_groups + n_periods - 1`` free columns, which is small for
    the cell-level problems this module solves.
    """
    gi = pd.factorize(g, sort=True)[0]
    ti = pd.factorize(t, sort=True)[0]
    n_g, n_t = gi.max() + 1, ti.max() + 1
    X = np.zeros((v.shape[0], n_g + n_t - 1))
    X[np.arange(v.shape[0]), gi] = 1.0
    keep_t = ti > 0
    X[np.where(keep_t)[0], n_g + ti[keep_t] - 1] = 1.0
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
    return v - X @ coef


def dcdh_fe_weights(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    d: str,
    weights: Optional[str] = None,
) -> Dict[str, Any]:
    """Cell weights and summary measures of the TWFE coefficient (``feTR``).

    Parameters
    ----------
    data : pd.DataFrame
    y, group, time, d : str
        Outcome, group, period and treatment columns. ``d`` may be binary or
        a group x period level intensity.
    weights : str, optional
        Observation weights column.

    Returns
    -------
    dict
        ``cells`` (DataFrame with ``group``, ``time``, ``D``, ``weight``,
        ``W``, ``nat_weight``), ``beta`` (the TWFE coefficient),
        ``nr_plus``, ``nr_minus``, ``nr_weights``, ``sum_plus``,
        ``sum_minus``, ``sensibility``, ``sensibility2`` (``nan`` when no
        weight is negative, where the measure is undefined), ``tot_cells``.
    """
    cols = [y, group, time, d] + ([weights] if weights else [])
    df = data[cols].dropna().copy()
    df["_w"] = df[weights].astype(float) if weights else 1.0
    # A treatment varying within a cell is replaced by its UNWEIGHTED cell
    # mean (TwoWayFEWeights:::twowayfeweights_normalize_var runs before the
    # observation weights are attached).
    df["_D"] = df.groupby([group, time], sort=True)[d].transform("mean").astype(float)

    w = df["_w"].to_numpy(dtype=float)
    D = df["_D"].to_numpy(dtype=float)
    Y = df[y].to_numpy(dtype=float)
    g = df[group].to_numpy()
    t = df[time].to_numpy()

    obs = w.sum()
    mean_D = float(np.sum(w * D) / obs)
    eps = _two_way_residual(D, g, t, w)
    # beta: FWL on the same design (Y and D residualised on both FEs).
    y_res = _two_way_residual(Y, g, t, w)
    beta = float(np.sum(w * eps * y_res) / np.sum(w * eps * eps))

    denom_W = float(np.sum(w * eps * D) / obs)
    df["_eps"] = eps
    df["_P"] = df.groupby([group, time])["_w"].transform("sum") / obs
    first = df.groupby([group, time], sort=True).head(1)
    P = first["_P"].to_numpy(dtype=float)
    Dc = first["_D"].to_numpy(dtype=float)
    Wr = first["_eps"].to_numpy(dtype=float) * mean_D / denom_W
    nat = P * Dc / mean_D
    wt = Wr * nat
    wt = np.where(np.abs(wt) < _ZERO_BELOW, 0.0, wt)

    cells = pd.DataFrame(
        {
            "group": first[group].to_numpy(),
            "time": first[time].to_numpy(),
            "D": Dc,
            "weight": wt,
            "W": Wr,
            "nat_weight": nat,
        }
    )

    plus = wt[wt > 0]
    minus = wt[wt < 0]
    M = int(np.sum(nat != 0))
    W_mean = float(np.sum(nat * Wr) / np.sum(nat))
    W_sd = float(np.sqrt(np.sum(nat * (Wr - W_mean) ** 2)) * np.sqrt(M / (M - 1)))
    sensibility = abs(beta) / W_sd

    sensibility2 = float("nan")
    if minus.size:
        s = cells.loc[cells["weight"] != 0].copy()
        # Ascending W (ties: descending G, descending T) for the cumulants...
        s = s.sort_values(
            ["W", "group", "time"], ascending=[True, False, False], kind="mergesort"
        )
        s["P_k"] = s["nat_weight"].cumsum()
        s["S_k"] = s["weight"].cumsum()
        s["T_k"] = (s["nat_weight"] * s["W"] ** 2).cumsum()
        # ...then descending W (ties: ascending G, ascending T) for the scan.
        s = s.sort_values(
            ["W", "group", "time"], ascending=[False, True, True], kind="mergesort"
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            one_minus_p = 1.0 - s["P_k"].to_numpy()
            s_k = s["S_k"].to_numpy()
            sens2 = abs(beta) / np.sqrt(s["T_k"].to_numpy() + s_k**2 / one_minus_p)
            ind = (s["W"].to_numpy() < -s_k / one_minus_p).astype(float)
        ind[0] = 0.0
        lag = np.r_[-1.0, ind[:-1]]
        ind = np.maximum(ind, lag)
        n = len(s)
        sensibility2 = float(sens2[n - int(ind.sum())])

    return {
        "cells": cells,
        "beta": beta,
        "nr_plus": int(plus.size),
        "nr_minus": int(minus.size),
        "nr_weights": int(plus.size + minus.size),
        "sum_plus": float(plus.sum()),
        "sum_minus": float(minus.sum()),
        "sensibility": float(sensibility),
        "sensibility2": sensibility2,
        "tot_cells": M,
    }
