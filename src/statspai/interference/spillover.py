"""
Spillover contrasts by peer exposure under partial interference.

Units sit in clusters; interference runs within but not between clusters.
Each unit's peer exposure is a summary of its cluster-mates' treatments
(``exposure_fn``), and units are split at a threshold ``c`` (the median of
the positive exposures). With ``H = 1{exposure >= c}`` the function reports
three differences in means:

- **Direct**: ``E[Y | D=1, H=1] - E[Y | D=0, H=1]``
- **Spillover**: ``E[Y | D=0, H=1] - E[Y | D=0, H=0]``
- **Total**: ``E[Y | D=1, H=1] - E[Y | D=0, H=0]`` (= direct + spillover)

with cluster-bootstrap standard errors.

These are descriptive contrasts. They have a causal reading when own
treatment is randomised independently of the cluster-mates' (Bernoulli
assignment), so that ``D`` is independent of the exposure: the direct
contrast is then an average effect of own treatment among units with high
exposure, and the spillover contrast compares untreated units across
exposure strata. Both depend on the threshold, and the spillover contrast
is confounded by anything that differs between high- and low-exposure
units (with ``'count'``, cluster size).

This is **not** the Hudgens-Halloran (2008) estimator. Their direct,
indirect, total and overall effects are defined for a two-stage design in
which clusters are randomised to treatment saturations (allocation
strategies) and individuals within clusters; estimating them needs the
saturation each cluster was assigned, which this function does not take.
Until 2026-10-19 this module was labelled as that estimator (and as IPW,
which it never was), silently ignored ``covariates`` and returned 0 for
any contrast whose cell was empty. For exposure mappings with design-based
inference on a network see ``sp.network_exposure``.

References
----------
[@hudgens2008toward] (the two-stage estimands this function does not
compute).
"""

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from .._input_validation import missing_named_columns
from ..core.results import CausalResult
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility

_EXPOSURE_FNS = ("fraction", "any", "count")


@accepts_aliases(seed="random_state")
def spillover(
    data: pd.DataFrame,
    y: str,
    treat: str,
    cluster: str,
    covariates: Optional[List[str]] = None,
    exposure_fn: str = "fraction",
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """
    Direct and spillover contrasts by peer exposure (median split).

    See the module docstring for the three estimands. They are differences
    in means between treatment x exposure cells, not the Hudgens-Halloran
    two-stage estimands.

    Parameters
    ----------
    data : pd.DataFrame
        Input data with cluster structure.
    y : str
        Outcome variable.
    treat : str
        Binary individual treatment variable (0/1).
    cluster : str
        Cluster/group identifier.
    covariates : list of str, optional
        Not supported: the contrasts are unadjusted. Passing covariates
        raises ``MethodIncompatibility`` (they used to be accepted and
        ignored).
    exposure_fn : {'fraction', 'any', 'count'}, default 'fraction'
        How to measure peer exposure:
        - 'fraction': fraction of cluster-mates treated
        - 'any': any cluster-mate treated (binary)
        - 'count': count of treated cluster-mates
        Units are split at the median of the positive exposures.
    n_bootstrap : int, default 500
    alpha : float, default 0.05
    random_state : int, default 42

    Returns
    -------
    CausalResult
        estimate = total contrast.
        detail DataFrame has columns 'effect_type', 'estimate', 'se'.
        model_info contains 'direct_effect', 'spillover_effect',
        'total_effect', the threshold ('median_exposure'), the cell sizes
        ('cell_sizes') and 'n_boot_dropped' (bootstrap replicates in which a
        cell was empty).

    Raises
    ------
    DataInsufficient
        A treatment x exposure cell needed by a contrast is empty.
    MethodIncompatibility
        ``covariates`` were passed, or ``exposure_fn`` is unknown.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(40):
    ...     treated = rng.integers(0, 2, size=5)
    ...     frac = treated.mean()
    ...     for i in range(5):
    ...         y = 1.0 + 0.8 * treated[i] + 0.5 * frac + rng.normal()
    ...         rows.append({"household": c, "vaccinated": int(treated[i]),
    ...                      "outcome": y})
    >>> df = pd.DataFrame(rows)
    >>> result = sp.spillover(df, y='outcome', treat='vaccinated',
    ...                       cluster='household', n_bootstrap=50,
    ...                       random_state=0)
    >>> bool('direct_effect' in result.model_info)
    True
    >>> bool('spillover_effect' in result.model_info)
    True
    """
    est = SpilloverEstimator(
        data=data,
        y=y,
        treat=treat,
        cluster=cluster,
        covariates=covariates,
        exposure_fn=exposure_fn,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )
    return est.fit()


class SpilloverEstimator:
    """Spillover / interference effects estimator.

    The engine behind :func:`sp.spillover`. Construct it with the data
    and column names, then call :meth:`fit` to obtain a ``CausalResult``
    whose ``model_info`` carries ``direct_effect``, ``spillover_effect``
    and ``total_effect``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for c in range(40):
    ...     treated = rng.integers(0, 2, size=5)
    ...     frac = treated.mean()
    ...     for i in range(5):
    ...         y = 1.0 + 0.8 * treated[i] + 0.5 * frac + rng.normal()
    ...         rows.append({"hh": c, "vaccinated": int(treated[i]),
    ...                      "outcome": y})
    >>> df = pd.DataFrame(rows)
    >>> est = sp.SpilloverEstimator(
    ...     data=df, y="outcome", treat="vaccinated", cluster="hh",
    ...     n_bootstrap=50, random_state=0,
    ... )
    >>> res = est.fit()
    >>> sorted(k for k in res.model_info if k.endswith("_effect"))
    ['direct_effect', 'spillover_effect', 'total_effect']
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        cluster: str,
        covariates: Optional[List[str]] = None,
        exposure_fn: str = "fraction",
        n_bootstrap: int = 500,
        alpha: float = 0.05,
        random_state: int = 42,
    ) -> None:
        self.data = data
        self.y = y
        self.treat = treat
        self.cluster = cluster
        self.covariates = covariates
        self.exposure_fn = exposure_fn
        self.n_bootstrap = n_bootstrap
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Estimate direct and spillover effects."""
        cols = [self.y, self.treat, self.cluster]
        if self.covariates:
            cols += self.covariates
        missing = [c for c in cols if c not in self.data.columns]
        if missing:
            raise ColumnNotFound.from_missing(
                missing_named_columns(
                    {
                        "y": self.y,
                        "treat": self.treat,
                        "cluster": self.cluster,
                        "covariates": self.covariates,
                    },
                    self.data.columns,
                ),
                self.data.columns,
                message=f"Columns not found in data: {missing}",
            )

        if self.covariates:
            raise MethodIncompatibility(
                "sp.spillover: the contrasts are unadjusted differences in "
                "means; covariates are not supported (they used to be "
                "accepted and silently ignored).",
                recovery_hint=(
                    "Drop covariates=, or regress the outcome on own "
                    "treatment, exposure and covariates with cluster-robust "
                    "standard errors (sp.regress(..., cluster=...))."
                ),
                diagnostics={"covariates": list(self.covariates)},
            )
        if self.exposure_fn not in _EXPOSURE_FNS:
            raise MethodIncompatibility(
                f"sp.spillover: exposure_fn={self.exposure_fn!r} is not one "
                f"of {list(_EXPOSURE_FNS)}.",
                recovery_hint="Use 'fraction', 'any' or 'count'.",
                diagnostics={"exposure_fn": self.exposure_fn},
            )

        full = self.data[cols].copy()
        # Peer exposure needs every unit's treatment: a unit whose treatment
        # or cluster is missing cannot be placed. Rows were dropped silently
        # before exposure was computed, so a missing outcome also removed the
        # unit's treatment from its cluster-mates' exposure (2026-10-20
        # probe: the threshold moved from 0.5 to 0.667).
        n_bad = int(full[[self.treat, self.cluster]].isna().any(axis=1).sum())
        if n_bad:
            raise MethodIncompatibility(
                f"sp.spillover: {n_bad} row(s) have a missing treatment or "
                "cluster, so their peers' exposure is undefined.",
                recovery_hint="Fill or drop those rows deliberately before "
                "calling (dropping a unit changes its peers' exposure).",
                diagnostics={"n_rows": n_bad},
            )
        D_all = full[self.treat].values.astype(np.float64)
        G_all = full[self.cluster].values
        exposure_all = self._compute_exposure(full, D_all, G_all)
        keep = full[self.y].notna().to_numpy()
        if not keep.all():
            warnings.warn(
                f"sp.spillover: {int((~keep).sum())} row(s) with a missing "
                "outcome are left out of the contrasts; their treatment still "
                "counts in their cluster-mates' exposure.",
                UserWarning,
                stacklevel=3,
            )
        df = full.loc[keep]
        n = len(df)

        Y = df[self.y].values.astype(np.float64)
        D = D_all[keep]
        G = G_all[keep]
        exposure = exposure_all[keep]

        # Classify units into 4 groups:
        # (own_treat=1, high_exposure) vs (own_treat=0, low_exposure) etc.
        median_exp = np.median(exposure[exposure > 0]) if np.any(exposure > 0) else 0.5
        high_exp = exposure >= median_exp

        # Direct effect: E[Y|D=1, high_exp] - E[Y|D=0, high_exp]
        mask_d1_hi = (D == 1) & high_exp
        mask_d0_hi = (D == 0) & high_exp

        # Spillover effect: E[Y|D=0, high_exp] - E[Y|D=0, low_exp]
        mask_d0_lo = (D == 0) & (~high_exp)

        cell_sizes = {
            "treated_high": int(mask_d1_hi.sum()),
            "untreated_high": int(mask_d0_hi.sum()),
            "untreated_low": int(mask_d0_lo.sum()),
        }
        empty = [k for k, v in cell_sizes.items() if v == 0]
        if empty:
            raise DataInsufficient(
                f"sp.spillover: no units in cell(s) {empty} at exposure "
                f"threshold {median_exp:g}; the contrasts need treated and "
                "untreated units with high exposure and untreated units "
                "with low exposure.",
                recovery_hint=(
                    "Treatment or exposure does not vary enough within "
                    "clusters; try another exposure_fn, or a cluster-level "
                    "analysis."
                ),
                diagnostics={"cell_sizes": cell_sizes, "threshold": median_exp},
            )
        direct = _safe_diff(Y, mask_d1_hi, mask_d0_hi)
        spillover_eff = _safe_diff(Y, mask_d0_hi, mask_d0_lo)
        total = direct + spillover_eff

        # Bootstrap
        rng = np.random.RandomState(self.random_state)
        clusters = np.unique(G)
        n_clusters = len(clusters)

        boot_direct = np.zeros(self.n_bootstrap)
        boot_spill = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            # Cluster bootstrap
            cl_idx = rng.choice(n_clusters, size=n_clusters, replace=True)
            selected_clusters = clusters[cl_idx]

            idx_list = []
            for cl in selected_clusters:
                idx_list.append(np.where(G == cl)[0])
            if len(idx_list) == 0:
                boot_direct[b] = direct
                boot_spill[b] = spillover_eff
                continue

            idx = np.concatenate(idx_list)
            Y_b = Y[idx]
            D_b = D[idx]
            exp_b = exposure[idx]

            hi_b = exp_b >= median_exp

            m_d1h = (D_b == 1) & hi_b
            m_d0h = (D_b == 0) & hi_b
            m_d0l = (D_b == 0) & (~hi_b)

            boot_direct[b] = _safe_diff(Y_b, m_d1h, m_d0h)
            boot_spill[b] = _safe_diff(Y_b, m_d0h, m_d0l)

        ok = np.isfinite(boot_direct) & np.isfinite(boot_spill)
        n_dropped = int((~ok).sum())
        if ok.sum() < 2:
            raise DataInsufficient(
                "sp.spillover: fewer than two cluster-bootstrap replicates "
                "had every cell populated.",
                recovery_hint="Add clusters or raise n_bootstrap.",
                diagnostics={"n_bootstrap": self.n_bootstrap, "dropped": n_dropped},
            )
        boot_direct, boot_spill = boot_direct[ok], boot_spill[ok]
        se_direct = float(np.std(boot_direct, ddof=1))
        se_spill = float(np.std(boot_spill, ddof=1))
        se_total = float(np.std(boot_direct + boot_spill, ddof=1))

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)

        if se_total > 0:
            pvalue = float(2 * sp_stats.norm.sf(abs(total / se_total)))
        else:
            pvalue = 0.0
        ci = (total - z_crit * se_total, total + z_crit * se_total)

        detail = pd.DataFrame(
            {
                "effect_type": ["Direct", "Spillover", "Total"],
                "estimate": [direct, spillover_eff, total],
                "se": [se_direct, se_spill, se_total],
            }
        )

        model_info = {
            "direct_effect": float(direct),
            "direct_se": se_direct,
            "spillover_effect": float(spillover_eff),
            "spillover_se": se_spill,
            "total_effect": float(total),
            "total_se": se_total,
            "n_clusters": n_clusters,
            "exposure_fn": self.exposure_fn,
            "median_exposure": float(median_exp),
            "cell_sizes": cell_sizes,
            "n_boot_dropped": n_dropped,
            "estimands": {
                "direct": "E[Y|D=1,H=1] - E[Y|D=0,H=1]",
                "spillover": "E[Y|D=0,H=1] - E[Y|D=0,H=0]",
                "total": "E[Y|D=1,H=1] - E[Y|D=0,H=0]",
                "H": "1{peer exposure >= median of positive exposures}",
            },
        }

        return CausalResult(
            method="Spillover contrasts by peer exposure (median split)",
            estimand="Total contrast E[Y|D=1,high] - E[Y|D=0,low]",
            estimate=float(total),
            se=se_total,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=detail,
            model_info=model_info,
            _citation_key="spillover",
        )

    def _compute_exposure(
        self,
        df: pd.DataFrame,
        D: np.ndarray,
        G: np.ndarray,
    ) -> np.ndarray:
        """Compute peer treatment exposure for each unit."""
        exposure = np.zeros(len(D))
        clusters = np.unique(G)

        for cl in clusters:
            mask = G == cl
            cl_size = mask.sum()
            cl_treated = D[mask].sum()

            for i in np.where(mask)[0]:
                # Exclude own treatment
                peer_treated = cl_treated - D[i]
                peer_size = cl_size - 1

                if peer_size <= 0:
                    exposure[i] = 0
                elif self.exposure_fn == "fraction":
                    exposure[i] = peer_treated / peer_size
                elif self.exposure_fn == "any":
                    exposure[i] = float(peer_treated > 0)
                elif self.exposure_fn == "count":
                    exposure[i] = peer_treated
                else:
                    exposure[i] = peer_treated / max(peer_size, 1)

        return exposure


def _safe_diff(
    Y: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
) -> float:
    """Mean difference; NaN (never 0) when a cell is empty."""
    if mask_a.sum() > 0 and mask_b.sum() > 0:
        return float(np.mean(Y[mask_a]) - np.mean(Y[mask_b]))
    return float("nan")


CausalResult._CITATIONS["spillover"] = (
    "@article{hudgens2008toward,\n"
    "  title={Toward Causal Inference with Interference},\n"
    "  author={Hudgens, Michael G and Halloran, M Elizabeth},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={103},\n"
    "  number={482},\n"
    "  pages={832--842},\n"
    "  year={2008}\n"
    "}"
)
