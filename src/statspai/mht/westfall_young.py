"""Westfall-Young (1993) permutation stepdown for multiple outcomes.

Randomization-based family-wise error control for experiments: re-randomize
the treatment exactly as the design did (completely, within strata, by
cluster), recompute every outcome's statistic under each re-randomization,
and adjust with the stepdown maxT algorithm. Unlike ``romano_wolf``'s
bootstrap, the reference distribution is the design's own randomization
distribution, so the adjusted p-values are exact under the sharp null.

Stata ``wyoung, permute()`` (Jones, Molitor and Reif 2019) and R
``multtest::mt.maxT`` implement the same algorithm.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import MethodIncompatibility
from ..inference.randomization import _enumerate_assignments, _ols_stat_factory


def _stepdown_maxt(t_obs: np.ndarray, t_perm: np.ndarray) -> np.ndarray:
    """Westfall-Young stepdown maxT adjusted p-values (two-sided).

    ``t_obs`` (m,) observed statistics; ``t_perm`` (B, m) statistics under
    the re-randomizations. Hypotheses are ordered by ``|t|``; for each draw
    the successive maxima of ``|t*|`` from the least significant upward give
    the reference for each step, and monotonicity is enforced.
    """
    a = np.abs(np.asarray(t_obs, dtype=float))
    ab = np.abs(np.asarray(t_perm, dtype=float))
    order = np.argsort(-a)  # most significant first
    m = a.size
    q = np.empty_like(ab)
    q[:, order[m - 1]] = ab[:, order[m - 1]]
    for j in range(m - 2, -1, -1):
        q[:, order[j]] = np.maximum(q[:, order[j + 1]], ab[:, order[j]])
    raw = np.array([np.mean(q[:, h] >= a[h] * (1 - 1e-12)) for h in range(m)])
    adj = np.empty(m)
    prev = 0.0
    for h in order:
        prev = max(prev, raw[h])
        adj[h] = prev
    return adj


def westfall_young(
    data: pd.DataFrame,
    y: Sequence[str],
    treat: str,
    covariates: Optional[List[str]] = None,
    strata: Optional[str] = None,
    cluster: Optional[str] = None,
    n_perms: int = 10_000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Westfall-Young permutation stepdown p-values across outcomes.

    For each outcome in ``y`` the statistic is the t-statistic of ``treat``
    in the OLS regression of the outcome on ``treat``, ``covariates`` and
    (with ``strata``) stratum fixed effects -- HC1, or CR1 by ``cluster``.
    The treatment is re-randomized as the experiment assigned it: within
    ``strata`` when given, at the ``cluster`` level when given, keeping the
    number of treated units (clusters) per stratum. The same re-randomization
    applies to every outcome, which is what lets the adjustment use the
    dependence between them. Each outcome is estimated on its own non-missing
    rows. When the design admits no more than ``n_perms`` assignments they
    are enumerated and the p-values are exact.

    Parameters
    ----------
    data : pd.DataFrame
    y : sequence of str
        Outcomes (one hypothesis each).
    treat : str
        Binary (0/1) treatment.
    covariates : list of str, optional
        Controls in every regression.
    strata : str, optional
        Randomization strata (blocks).
    cluster : str, optional
        Unit of randomization when treatment was assigned to groups; also the
        cluster of the CR1 standard error.
    n_perms : int, default 10000
        Re-randomizations (all of them when there are no more).
    seed : int, optional

    Returns
    -------
    pd.DataFrame
        One row per outcome: ``outcome``, ``coef``, ``t``, ``n_obs``,
        ``p_analytic`` (t-distribution), ``p_perm`` (per-outcome
        randomization p-value) and ``p_wy`` (Westfall-Young adjusted);
        ``.attrs`` carries ``n_perms`` and ``exact``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> d = pd.DataFrame({"Z": rng.permutation([0, 1] * 40)})
    >>> d["y1"] = 0.8 * d.Z + rng.normal(size=80)
    >>> d["y2"] = rng.normal(size=80)
    >>> out = sp.westfall_young(d, y=["y1", "y2"], treat="Z", n_perms=1000, seed=1)
    >>> list(out.columns)[:3]
    ['outcome', 'coef', 't']
    >>> bool(out.set_index("outcome").loc["y1", "p_wy"] < 0.05)
    True

    References
    ----------
    Westfall, P. H. and Young, S. S. (1993). *Resampling-Based Multiple
    Testing*. Wiley.
    """
    ys = [y] if isinstance(y, str) else list(y)
    covariates = list(covariates or [])
    base_cols = [treat] + covariates + [c for c in (strata, cluster) if c]
    missing = [c for c in ys + base_cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"Columns not found in data: {missing}")
    df = data[list(dict.fromkeys(ys + base_cols))]
    df = df.dropna(subset=base_cols).reset_index(drop=True)
    D = df[treat].to_numpy(dtype=float)
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise MethodIncompatibility(f"'{treat}' must be binary 0/1.")
    st = df[strata].to_numpy() if strata else None
    cl = df[cluster].to_numpy() if cluster else None
    n = len(df)

    # one statistic function per outcome, on the rows where it is observed
    fns, rows, coefs, ts, n_obs, p_an = [], [], [], [], [], []
    for o in ys:
        r = df[o].notna().to_numpy()
        sub = df.loc[r]
        fn_t = _ols_stat_factory(
            sub,
            covariates,
            None if st is None else st[r],
            None if cl is None else cl[r],
            t_stat=True,
        )
        fn_b = _ols_stat_factory(
            sub,
            covariates,
            None if st is None else st[r],
            None,
            t_stat=False,
        )
        yv = sub[o].to_numpy(dtype=float)
        fns.append((fn_t, yv))
        rows.append(r)
        t0 = fn_t(yv, D[r])
        coefs.append(fn_b(yv, D[r]))
        ts.append(t0)
        n_obs.append(int(r.sum()))
        if cl is not None:
            dfree = len(np.unique(cl[r])) - 1
        else:
            k = (
                2
                + len(covariates)
                + (len(np.unique(st[r])) - 1 if st is not None else 0)
            )
            dfree = int(r.sum()) - k
        p_an.append(float(2 * stats.t.sf(abs(t0), max(dfree, 1))))
    t_obs = np.array(ts)

    assignments = _enumerate_assignments(D, clusters=cl, strata=st, max_count=n_perms)
    exact = assignments is not None
    if exact:
        draws = assignments
    else:
        rng = np.random.default_rng(seed)
        draws = np.empty((n_perms, n))
        if cl is not None:
            ucl, inv = np.unique(cl, return_inverse=True)
            cl_D = np.array([D[inv == g][0] for g in range(ucl.size)])
            cl_st = (
                np.array([st[inv == g][0] for g in range(ucl.size)])
                if st is not None
                else np.zeros(ucl.size)
            )
            groups = [np.flatnonzero(cl_st == s_) for s_ in np.unique(cl_st)]
            for b in range(n_perms):
                perm = cl_D.copy()
                for gi in groups:
                    perm[gi] = rng.permutation(cl_D[gi])
                draws[b] = perm[inv]
        else:
            groups = (
                [np.flatnonzero(st == s_) for s_ in np.unique(st)]
                if st is not None
                else [np.arange(n)]
            )
            for b in range(n_perms):
                perm = D.copy()
                for gi in groups:
                    perm[gi] = rng.permutation(D[gi])
                draws[b] = perm

    t_perm = np.empty((draws.shape[0], len(ys)))
    for b, d_b in enumerate(draws):
        for j, ((fn_t, yv), r) in enumerate(zip(fns, rows)):
            t_perm[b, j] = fn_t(yv, d_b[r])

    p_perm = np.mean(np.abs(t_perm) >= np.abs(t_obs) * (1 - 1e-12), axis=0)
    p_wy = _stepdown_maxt(t_obs, t_perm)
    out = pd.DataFrame(
        {
            "outcome": ys,
            "coef": coefs,
            "t": t_obs,
            "n_obs": n_obs,
            "p_analytic": p_an,
            "p_perm": p_perm,
            "p_wy": p_wy,
        }
    )
    out.attrs.update({"n_perms": int(draws.shape[0]), "exact": exact})
    if not exact and n_perms < 1000:
        warnings.warn(
            f"westfall_young: {n_perms} re-randomizations; use 10,000+ for "
            "reported p-values.",
            UserWarning,
            stacklevel=2,
        )
    return out
