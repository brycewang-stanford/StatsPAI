"""
Bandwidth selection for local polynomial RD estimation.

This module is the public entry point ``sp.rdbwselect``. The arithmetic
lives in :mod:`statspai.rd._cct_bandwidth`, which is the same
Calonico-Cattaneo-Titiunik three-stage cascade ``sp.rdrobust`` uses to
pick its own default bandwidth, so the two cannot disagree.

They did until 1.27.0. This file used to carry a complete second
implementation -- a single-step rule of thumb, roughly 600 lines of it --
and ``sp.rdbwselect`` called that one while the estimator called the
cascade. The rule of thumb returned bandwidths 2.8x to 4.8x too narrow and
did not vary with the polynomial order, because its exponent 1/5 equals
CCT's 1/(2p+3) only at p == 1. It has been removed rather than deprecated:
keeping a known-wrong implementation of a quantity that already has a
right one in the same subpackage is what produced the defect.

All ten selectors ``rdrobust`` offers are supported, including the
``sum`` forms that ``comb1`` / ``comb2`` are defined in terms of. Sharp
RD, covariate adjustment and cluster-robust variance are supported;
covariates collinear with the running variable are refused rather than
silently absorbed (see :func:`statspai.rd._core._check_covariate_rank`).

References
----------
Calonico, S., Cattaneo, M.D. and Farrell, M.H. (2020).
"Optimal Bandwidth Choice for Robust Bias-Corrected Inference in
Regression Discontinuity Designs." *Econometrics Journal*, 23(2),
192-210. [@calonico2020optimal]

Calonico, S., Cattaneo, M.D. and Titiunik, R. (2014).
"Robust Nonparametric Confidence Intervals for Regression-Discontinuity
Designs." *Econometrica*, 82(6), 2295-2326. [@calonico2014robust]
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ._cct_bandwidth import BW_SELECTORS, cct_bandwidth
from ._core import _check_covariate_rank

# ======================================================================
# Public API
# ======================================================================

#: The ten selectors ``rdrobust::rdbwselect`` offers. ``msesum`` and
#: ``cersum`` were absent until the Track A module 88 sweep: they are the
#: sum-form cascades that ``*comb1`` / ``*comb2`` are *built from*, so the
#: public entry point could not reach a bandwidth its own combination rules
#: depend on.
_VALID_METHODS = set(BW_SELECTORS)


def rdbwselect(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    fuzzy: Optional[str] = None,
    deriv: int = 0,
    p: int = 1,
    q: Optional[int] = None,
    covs: Optional[List[str]] = None,
    kernel: str = "triangular",
    bwselect: str = "mserd",
    cluster: Optional[str] = None,
    all: bool = False,
) -> pd.DataFrame:
    """
    Bandwidth selection for local polynomial RD estimation.

    Runs the Calonico-Cattaneo-Titiunik three-stage cascade and returns all
    ten MSE-optimal and CER-optimal selectors ``rdrobust`` offers.
    MSE-optimal bandwidths minimize the mean squared error of the RD point
    estimator, while CER-optimal bandwidths minimize the coverage error
    rate of robust bias-corrected confidence intervals.

    This is the same code path ``sp.rdrobust`` uses to pick its own default
    bandwidth, so a value read here and passed back as ``h=`` reproduces
    ``sp.rdrobust``'s default fit exactly. Pinned against
    ``rdrobust::rdbwselect`` (R) and the ``rdbwselect`` ado (Stata) by Track
    A module ``88_rdbwselect`` to 1.8e-12 and 3.7e-9 respectively, across
    all ten selectors, polynomial orders 1-3, three kernels, covariate
    adjustment, clustering and the regression-kink derivative.

    .. versionchanged:: 1.27.0
       Both properties above are new. This function previously ran a
       separate single-step rule of thumb and returned bandwidths 2.8x-4.8x
       too narrow, and the four ``comb`` selectors resolved to the plain
       ``rd`` cascade. See MIGRATION.md — results should be recomputed.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    y : str
        Outcome variable column name.
    x : str
        Running variable column name.
    c : float, default 0
        RD cutoff value.
    fuzzy : str, optional
        Treatment variable name for fuzzy RD. When provided, bandwidth
        accounts for first-stage variance in the Wald / IV estimator.
    deriv : int, default 0
        Derivative order. 0 = standard RD (jump in level),
        1 = regression kink design (change in slope).
    p : int, default 1
        Polynomial order for point estimation (1 = local linear).
    q : int, optional
        Polynomial order for bias correction. Default is p + 1.
    covs : list of str, optional
        Covariate column names. When provided, the variance estimates
        used in bandwidth selection account for covariate adjustment,
        typically yielding narrower bandwidths.
    kernel : str, default 'triangular'
        Kernel function: 'triangular', 'uniform', or 'epanechnikov'.
    bwselect : str, default 'mserd'
        Bandwidth selection method. One of:

        - ``'mserd'`` : MSE-optimal common bandwidth (default)
        - ``'msetwo'`` : MSE-optimal separate left/right bandwidths
        - ``'msesum'`` : MSE-optimal for the sum of the two intercepts
        - ``'msecomb1'`` : ``min(mserd, msesum)``, per side
        - ``'msecomb2'`` : ``median(msetwo, mserd, msesum)``, per side
        - ``'cerrd'``, ``'certwo'``, ``'cersum'``, ``'cercomb1'``,
          ``'cercomb2'`` : the CER-optimal counterparts of the above

        The combination rules are applied to the finished ``h`` and ``b``
        of each cascade, element-wise per side -- not stage by stage.
    cluster : str, optional
        Cluster variable name for cluster-robust variance estimation.
    all : bool, default False
        If True, compute and return all ten bandwidth types.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``[method, h_left, h_right, b_left, b_right,
        n_left, n_right]``. When ``all=False``, contains a single row for
        the selected method. When ``all=True``, contains ten rows, one
        per method.

    Notes
    -----
    The CER-optimal bandwidth is related to the MSE-optimal bandwidth by:

        h_CER = h_MSE * n^{-1/((2p+3)(2p+5))}

    For p=1 (local linear), this gives h_CER ~ h_MSE * n^{-1/35}, which
    is strictly narrower than h_MSE. The narrower bandwidth yields
    confidence intervals with better coverage properties at the cost of
    slightly wider intervals.

    The MSE-optimal bandwidth has rate n^{-1/(2p+3)} while the CER rate
    is n^{-1/(2p+3) - 1/((2p+3)(2p+5))}. For large samples the
    difference is meaningful: CER bandwidths produce robust CIs that
    achieve their nominal coverage rate, whereas MSE bandwidths can
    exhibit substantial coverage distortion.

    References
    ----------
    Calonico, S., Cattaneo, M.D. and Farrell, M.H. (2020).
    "Optimal Bandwidth Choice for Robust Bias-Corrected Inference in
    Regression Discontinuity Designs." *Econometrics Journal*, 23(2),
    192-210. [@calonico2020optimal]

    Calonico, S., Cattaneo, M.D. and Titiunik, R. (2014).
    "Robust Nonparametric Confidence Intervals for Regression-Discontinuity
    Designs." *Econometrica*, 82(6), 2295-2326. [@calonico2014robust]

    Examples
    --------
    Basic MSE-optimal bandwidth:

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> x = rng.uniform(-1, 1, n)
    >>> y = 0.5 * x + 3.0 * (x >= 0) + rng.normal(0, 0.3, n)
    >>> df = pd.DataFrame({'y': y, 'x': x})
    >>> bw = sp.rdbwselect(df, y='y', x='x', c=0)
    >>> bool('h_left' in bw.columns)  # DataFrame with bandwidth info
    True

    Compare all eight bandwidth methods:

    >>> bw_all = sp.rdbwselect(df, y='y', x='x', c=0, all=True)
    >>> bool(len(bw_all) >= 1)
    True

    CER-optimal bandwidth for better coverage:

    >>> bw_cer = sp.rdbwselect(df, y='y', x='x', c=0, bwselect='cerrd')
    >>> bool(bw_cer is not None)
    True

    Fuzzy RD with covariates:

    >>> treated = ((x >= 0) & (rng.uniform(0, 1, n) < 0.85)).astype(float)
    >>> df['treated'] = treated
    >>> df['age'] = rng.normal(40, 10, n)
    >>> df['gender'] = rng.integers(0, 2, n).astype(float)
    >>> bw_fuzzy = sp.rdbwselect(df, y='y', x='x', c=0,
    ...                          fuzzy='treated', covs=['age', 'gender'])
    >>> bool(bw_fuzzy is not None)
    True
    """
    # --- Validate inputs ---
    if kernel not in ("triangular", "uniform", "epanechnikov"):
        raise ValueError(  # pragma: no cover
            f"kernel must be 'triangular', 'uniform', or 'epanechnikov', "
            f"got '{kernel}'"
        )
    if bwselect not in _VALID_METHODS:
        raise ValueError(  # pragma: no cover
            f"bwselect must be one of {sorted(_VALID_METHODS)}, " f"got '{bwselect}'"
        )
    if deriv < 0:
        raise ValueError(f"deriv must be non-negative, got {deriv}")  # pragma: no cover
    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")  # pragma: no cover
    if deriv > 0 and p < deriv + 1:
        p = deriv + 1
    if q is None:
        q = p + 1
    if q <= p:
        raise ValueError(f"q must be > p, got q={q}, p={p}")  # pragma: no cover

    # --- Parse data ---
    Y = data[y].values.astype(float)
    X_raw = data[x].values.astype(float)
    X_c = X_raw - c

    # Drop missing
    valid = np.isfinite(Y) & np.isfinite(X_c)
    if fuzzy is not None:
        D = data[fuzzy].values.astype(float)
        valid &= np.isfinite(D)
    else:
        D = None

    if covs is not None:
        covs_data = data[covs].values.astype(float)
        valid &= np.all(np.isfinite(covs_data), axis=1)
    else:
        covs_data = None

    if cluster is not None:
        cluster_vals = data[cluster].values
    else:
        cluster_vals = None

    # Apply valid mask
    Y = Y[valid]
    X_c = X_c[valid]
    if D is not None:
        D = D[valid]
    if covs_data is not None:
        covs_data = covs_data[valid]
    if cluster_vals is not None:
        cluster_vals = cluster_vals[valid]

    n = len(Y)
    if n < 20:
        raise ValueError(f"Need at least 20 observations, got {n}.")  # pragma: no cover

    left = X_c < 0
    right = X_c >= 0
    n_left = int(left.sum())
    n_right = int(right.sum())

    if n_left < p + 2 or n_right < p + 2:
        raise ValueError(  # pragma: no cover
            f"Not enough observations on each side of the cutoff "
            f"(left={n_left}, right={n_right}, need >= {p + 2})."
        )

    # --- Compute bandwidths ---
    # The CCT three-stage cascade in ``rd/_cct_bandwidth.py``, which is the
    # same code path ``sp.rdrobust`` uses and which is pinned against
    # ``rdrobust::rdbwselect`` by Track A module 88.
    #
    # This entry point previously ran a single-step rule of thumb of its own
    # (``_compute_all_bandwidths``, now retired). That formula's exponent
    # 1/5 equals CCT's 1/(2p+3) only at p == 1, and it produced no separate
    # bias bandwidth b, so the published function returned h between 2.8x
    # and 4.8x too narrow -- 4.63 against R's 17.75 on the Lee 2008 senate
    # replica -- while its docstring advertised Calonico, Cattaneo and
    # Farrell (2020). A user who took a bandwidth from sp.rdbwselect and
    # passed it to sp.rdrobust(h=...) got a materially different estimate
    # from sp.rdrobust's own default, with nothing to signal the mismatch.
    _check_covariate_rank(X_c, covs_data, p, names=covs, where="rdbwselect")

    def _bw_for(method: str) -> Tuple[float, float, float, float]:
        out = cct_bandwidth(
            Y,
            X_c,
            c=0.0,  # X_c is already centred at the cutoff
            p=p,
            q=q,
            deriv=deriv,
            kernel=kernel,
            bwselect=method,
            covs=covs_data,
            cluster=cluster_vals,
        )
        return out["h_left"], out["h_right"], out["b_left"], out["b_right"]

    # --- Count effective observations for each bandwidth ---
    def _count_effective(h_l: float, h_r: float) -> Tuple[int, int]:
        n_eff_l = int(np.sum((X_c[left] >= -h_l)))
        n_eff_r = int(np.sum((X_c[right] <= h_r)))
        return n_eff_l, n_eff_r

    # --- Build output ---
    # Bandwidths are returned at full precision. They used to be rounded to
    # six decimals here, which capped any downstream agreement at ~1e-6
    # relative and silently perturbed sp.rdrobust(h=...) when a user fed one
    # back in -- a rounded selector output is not the selector's answer.
    methods = list(BW_SELECTORS) if all else [bwselect]
    rows = []
    for method in methods:
        h_l, h_r, b_l, b_r = _bw_for(method)
        n_eff_l, n_eff_r = _count_effective(h_l, h_r)
        rows.append(
            {
                "method": method,
                "h_left": h_l,
                "h_right": h_r,
                "b_left": b_l,
                "b_right": b_r,
                "n_left": n_eff_l,
                "n_right": n_eff_r,
            }
        )
    return pd.DataFrame(rows)
