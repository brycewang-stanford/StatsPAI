"""Design-based difference in means for randomized experiments.

The estimator is the one the design calls for, and so is its variance:

=======================  ==============================================
design                   variance and degrees of freedom
=======================  ==============================================
complete randomization   Neyman, ``s1^2/n1 + s0^2/n0``; Satterthwaite df
blocked                  block estimates weighted by block size, Neyman
                         variance within each block; ``N - 2J`` df
matched pairs            variance of the pair differences; ``J - 1`` df
clustered                CR2 (``bell2002bias``) with the Bell-McCaffrey df
blocked and clustered    CR2 within each block; ``G - 2J`` df
matched pairs of         Imai-King-Nall variance of the size-weighted pair
clusters                 differences; ``J - 1`` df
=======================  ==============================================

``N`` is the number of units, ``J`` of blocks and ``G`` of clusters. These
are the designs and the formulas of R ``estimatr::difference_in_means``,
which is the numerical reference (``imbens2015causal`` chapters 6, 9 and 10
for the first three rows).
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["difference_in_means"]

# Verbatim from paper.bib (imbens2015causal).
CausalResult._CITATIONS["neyman_difference_in_means"] = (
    "@book{imbens2015causal,\n"
    "  title={Causal Inference for Statistics, Social, and Biomedical "
    "Sciences: An Introduction},\n"
    "  author={Imbens, Guido W. and Rubin, Donald B.},\n"
    "  year={2015},\n"
    "  publisher={Cambridge University Press},\n"
    "  isbn={978-0-521-88588-1},\n"
    "  doi={10.1017/CBO9781139025751}\n"
    "}"
)


def _welch(y1: np.ndarray, y0: np.ndarray) -> Tuple[float, float, float]:
    """Difference in means, Neyman variance and Satterthwaite df."""
    n1, n0 = len(y1), len(y0)
    v1 = float(np.var(y1, ddof=1)) / n1
    v0 = float(np.var(y0, ddof=1)) / n0
    var = v1 + v0
    denom = v1**2 / (n1 - 1) + v0**2 / (n0 - 1)
    df = var**2 / denom if denom > 0 else float(n1 + n0 - 2)
    return float(y1.mean() - y0.mean()), var, float(df)


def _cr2_two_arms(
    y: np.ndarray, d: np.ndarray, cl: np.ndarray, *, want_df: bool
) -> Tuple[float, float, float]:
    """CR2 variance of the coefficient on ``d`` in ``y ~ 1 + d``.

    Clusters are nested in arms, so the two arms are separate regressions
    on a constant and the leverage adjustment has a closed form: the
    adjusted cluster total is ``sum(e_g) / sqrt(1 - n_g / n_arm)``.
    """
    var = 0.0
    means = {}
    for arm in (0, 1):
        keep = d == arm
        ya, ca = y[keep], cl[keep]
        n_arm = len(ya)
        means[arm] = float(ya.mean())
        resid = ya - means[arm]
        codes, _ = pd.factorize(ca)
        totals = np.bincount(codes, weights=resid)
        sizes = np.bincount(codes).astype(float)
        var += float((totals**2 / (1.0 - sizes / n_arm)).sum()) / n_arm**2
    df = float("nan")
    if want_df:
        from .jackknife import _satterthwaite_dof

        X = np.column_stack([np.ones(len(y)), d.astype(float)])
        df = float(
            _satterthwaite_dof(X, cl, np.unique(cl), np.linalg.inv(X.T @ X), 2)[1]
        )
    return means[1] - means[0], var, df


def difference_in_means(
    data: pd.DataFrame,
    y: str,
    treat: str,
    *,
    blocks: Optional[str] = None,
    cluster: Optional[str] = None,
    estimand: str = "ATE",
    alpha: float = 0.05,
) -> CausalResult:
    """Difference in means with the variance the experimental design implies.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Outcome.
    treat : str
        Treatment, with exactly two values. The estimate is the mean under
        the larger value minus the mean under the smaller one.
    blocks : str, optional
        Blocks (strata) within which treatment was randomized. Blocks of two
        units, or of two clusters, are analysed as matched pairs.
    cluster : str, optional
        Clusters that were assigned to treatment together.
    estimand : {'ATE', 'ATT'}, default 'ATE'
        With blocks, ``'ATE'`` weights each block by its number of units and
        ``'ATT'`` by its number of treated units. Without blocks the two
        coincide under randomization and the estimate is the same.
    alpha : float, default 0.05
        One minus the confidence level.

    Returns
    -------
    CausalResult
        ``estimate``, ``se``, ``pvalue`` and ``ci`` from a t distribution
        with ``model_info['df']`` degrees of freedom.
        ``model_info['design']`` names the design that was used. ``detail``
        has one row per block (estimate, standard error, weight) or, without
        blocks, one row per arm.

    Raises
    ------
    MethodIncompatibility
        Treatment is not two-valued, a cluster spans both arms or two
        blocks, or ``estimand='ATT'`` is asked of matched pairs of clusters.
    DataInsufficient
        A block that is not a matched pair has fewer than two treated or two
        control units (or clusters), so its variance is not identified.

    Notes
    -----
    Rows with a missing value in any of the columns used are dropped with a
    warning. The clustered variance is the CR2 estimator of the regression
    of ``y`` on a constant and ``treat``; its degrees of freedom are those
    of ``sp.cr2_se``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"school": np.repeat(np.arange(8), 30)})
    >>> df["small"] = rng.permutation(np.tile([0, 1], 120))
    >>> df["score"] = 0.3 * df.school + 2.0 * df.small + rng.normal(size=240)
    >>> res = sp.difference_in_means(df, "score", "small", blocks="school")
    >>> res.model_info["design"]
    'Blocked'
    >>> bool(1.0 < res.estimate < 3.0)
    True

    References
    ----------
    imbens2015causal, bell2002bias, pustejovsky2018small
    """
    context = "difference_in_means"
    estimand_key = str(estimand).upper()
    if estimand_key == "ATET":
        estimand_key = "ATT"
    if estimand_key not in ("ATE", "ATT"):
        raise MethodIncompatibility(
            f"{context}: estimand must be 'ATE' or 'ATT', got {estimand!r}."
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(f"{context}: alpha must lie in (0, 1).")
    columns: List[str] = [y, treat]
    for extra in (blocks, cluster):
        if extra is not None:
            columns.append(extra)
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"{context}: columns not in data: {missing}.")

    clean = data[columns].dropna()
    n_dropped = len(data) - len(clean)
    if n_dropped:
        warnings.warn(
            f"{context}: dropped {n_dropped} rows with missing values.",
            UserWarning,
            stacklevel=2,
        )
    levels = np.sort(pd.unique(clean[treat]))
    if len(levels) != 2:
        raise MethodIncompatibility(
            f"{context}: treat must take exactly two values, found "
            f"{len(levels)}: {list(levels[:6])}."
        )
    yv = clean[y].to_numpy(dtype=float)
    dv = (clean[treat].to_numpy() == levels[1]).astype(int)
    n = len(clean)
    cl = clean[cluster].to_numpy() if cluster is not None else None
    bl = clean[blocks].to_numpy() if blocks is not None else None

    if cl is not None:
        per_cluster = pd.Series(dv).groupby(cl).nunique()
        if (per_cluster > 1).any():
            raise MethodIncompatibility(
                f"{context}: {int((per_cluster > 1).sum())} clusters contain "
                "both treated and control units. A cluster is assigned as a "
                "whole; use blocks= if treatment varies inside these groups."
            )
        if bl is not None:
            spans = pd.Series(bl).groupby(cl).nunique()
            if (spans > 1).any():
                raise MethodIncompatibility(
                    f"{context}: {int((spans > 1).sum())} clusters span more "
                    "than one block; clusters must be nested in blocks."
                )

    model_info: Dict[str, object] = {
        "treat_levels": (levels[0], levels[1]),
        "n_dropped": int(n_dropped),
        "n_treated": int(dv.sum()),
        "n_control": int(n - dv.sum()),
    }

    if bl is None:
        if min(dv.sum(), n - dv.sum()) < 2:
            raise DataInsufficient(
                f"{context}: each arm needs at least two units.",
                diagnostics={
                    "n_treated": int(dv.sum()),
                    "n_control": int(n - dv.sum()),
                },
            )
        if cl is None:
            est, var, df = _welch(yv[dv == 1], yv[dv == 0])
            design = "Standard"
        else:
            n_cl = pd.Series(cl).groupby(dv).nunique()
            if n_cl.min() < 2:
                raise DataInsufficient(
                    f"{context}: each arm needs at least two clusters.",
                    diagnostics={"clusters_per_arm": n_cl.to_dict()},
                )
            est, var, df = _cr2_two_arms(yv, dv, cl, want_df=True)
            design = "Clustered"
            model_info["n_clusters"] = int(len(np.unique(cl)))
        detail = pd.DataFrame(
            {
                "arm": [levels[0], levels[1]],
                "n": [int((dv == 0).sum()), int((dv == 1).sum())],
                "mean": [float(yv[dv == 0].mean()), float(yv[dv == 1].mean())],
                "sd": [
                    float(np.std(yv[dv == 0], ddof=1)),
                    float(np.std(yv[dv == 1], ddof=1)),
                ],
            }
        )
    else:
        est, var, df, design, detail = _blocked(
            yv, dv, bl, cl, estimand_key, context, model_info
        )

    se = float(np.sqrt(var))
    tstat = est / se if se > 0 else float("nan")
    pvalue = float(2 * stats.t.sf(abs(tstat), df)) if se > 0 else float("nan")
    crit = float(stats.t.ppf(1 - alpha / 2, df))
    model_info.update(
        {
            "design": design,
            "df": float(df),
            "df_inference": float(df),
            "statistic": tstat,
        }
    )
    return CausalResult(
        method=f"Difference in means ({design.lower()} design)",
        estimand=estimand_key,
        estimate=float(est),
        se=se,
        pvalue=pvalue,
        ci=(float(est - crit * se), float(est + crit * se)),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="neyman_difference_in_means",
    )


def _blocked(
    yv: np.ndarray,
    dv: np.ndarray,
    bl: np.ndarray,
    cl: Optional[np.ndarray],
    estimand_key: str,
    context: str,
    model_info: Dict[str, object],
) -> Tuple[float, float, float, str, pd.DataFrame]:
    codes, block_labels = pd.factorize(bl, sort=True)
    n_blocks = len(block_labels)
    n = len(yv)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(n_blocks + 1))
    groups = [order[bounds[b] : bounds[b + 1]] for b in range(n_blocks)]

    def units(idx: np.ndarray, arm: int) -> int:
        """Randomization units (clusters, or units) of one arm in a block."""
        keep = idx[dv[idx] == arm]
        return len(np.unique(cl[keep])) if cl is not None else len(keep)

    n_units = np.array([[units(idx, 0), units(idx, 1)] for idx in groups])
    if (n_units.min(axis=1) < 1).any():
        bad = [block_labels[b] for b in np.flatnonzero(n_units.min(axis=1) < 1)]
        raise DataInsufficient(
            f"{context}: {len(bad)} blocks have no treated or no control "
            f"units, for example {bad[:5]}. Their effect is not identified.",
            diagnostics={"blocks": bad},
        )
    pairs = bool((n_units == 1).all())
    if not pairs and (n_units.min(axis=1) < 2).any():
        bad = [block_labels[b] for b in np.flatnonzero(n_units.min(axis=1) < 2)]
        what = "clusters" if cl is not None else "units"
        raise DataInsufficient(
            f"{context}: {len(bad)} blocks have fewer than two treated or "
            f"two control {what}, for example {bad[:5]}, so the variance "
            "within them is not identified. Matched pairs need every block "
            "to be a pair.",
            diagnostics={"blocks": bad},
        )

    sizes = np.array([len(idx) for idx in groups], dtype=float)
    treated = np.array([int(dv[idx].sum()) for idx in groups], dtype=float)
    tau = np.array(
        [yv[idx][dv[idx] == 1].mean() - yv[idx][dv[idx] == 0].mean() for idx in groups]
    )
    if estimand_key == "ATT":
        weights = treated / treated.sum()
    else:
        weights = sizes / n
    est = float(weights @ tau)
    block_var = np.full(n_blocks, np.nan)

    if pairs and cl is None:
        var = float(((tau - est) ** 2).sum()) / (n_blocks * (n_blocks - 1))
        df = float(n_blocks - 1)
        design = "Matched-pair"
    elif pairs:
        if estimand_key == "ATT":
            raise MethodIncompatibility(
                f"{context}: estimand='ATT' is not defined for matched pairs "
                "of clusters here; the pair variance is for the size-weighted "
                "average effect."
            )
        var = float(
            n_blocks
            / ((n_blocks - 1) * n**2)
            * ((sizes * tau - n * est / n_blocks) ** 2).sum()
        )
        df = float(n_blocks - 1)
        design = "Matched-pair clustered"
    else:
        for b, idx in enumerate(groups):
            if cl is None:
                _, block_var[b], _ = _welch(
                    yv[idx][dv[idx] == 1], yv[idx][dv[idx] == 0]
                )
            else:
                _, block_var[b], _ = _cr2_two_arms(
                    yv[idx], dv[idx], cl[idx], want_df=False
                )
        var = float((weights**2 * block_var).sum())
        if cl is None:
            df = float(n - 2 * n_blocks)
            design = "Blocked"
        else:
            df = float(len(np.unique(cl)) - 2 * n_blocks)
            design = "Block-clustered"
    if df <= 0:
        raise DataInsufficient(
            f"{context}: no degrees of freedom left ({df:g}).",
            diagnostics={"df": df, "n_blocks": n_blocks},
        )

    model_info["n_blocks"] = int(n_blocks)
    if cl is not None:
        model_info["n_clusters"] = int(len(np.unique(cl)))
    detail = pd.DataFrame(
        {
            "block": block_labels,
            "n": sizes.astype(int),
            "n_treated": treated.astype(int),
            "estimate": tau,
            "se": np.sqrt(block_var),
            "weight": weights,
        }
    )
    return est, var, df, design, detail
