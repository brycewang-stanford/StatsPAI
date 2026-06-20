"""
Experimental design tools for randomized controlled trials.

Provides stratified randomization, cluster randomization, balance checking,
and re-randomization for RCTs.

Equivalent to Stata's ``randtreat`` and R's ``randomizr``.

References
----------
Bruhn, M. & McKenzie, D. (2009).
"In Pursuit of Balance: Randomization in Practice in
Development Field Experiments." *AEJ: Applied*, 1(4), 200-232.
[@bruhn2009pursuit]
"""

import warnings
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import StatsPAIWarning


class RandomizationResult:
    """Results from randomization.

    Produced by :func:`randomize`. Carries the assigned data frame
    (``.data``), arm counts and an optional balance summary, with a
    formatted ``.summary()``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({"age": rng.normal(40, 10, n),
    ...                    "income": rng.normal(50, 15, n),
    ...                    "district": rng.integers(0, 4, n)})
    >>> res = sp.randomize(df, strata="district",
    ...                    balance_vars=["age", "income"], seed=1)
    >>> type(res).__name__
    'RandomizationResult'
    >>> int(res.n_treated + res.n_control)
    200
    >>> isinstance(res.summary(), str)
    True
    """

    def __init__(
        self,
        data: pd.DataFrame,
        treatment_col: str,
        n_treated: Optional[int],
        n_control: Optional[int],
        strata_col: Optional[str],
        method: str,
        balance: Optional[Dict[str, Any]],
        seed: Optional[int],
    ) -> None:
        self.data = data
        self.treatment_col = treatment_col
        self.n_treated = n_treated
        self.n_control = n_control
        self.strata_col = strata_col
        self.method = method
        self.balance = balance
        self.seed = seed

    def summary(self) -> str:
        lines: List[str] = [
            "Randomization Summary",
            "=" * 50,
            f"Method: {self.method}",
        ]
        if self.n_treated is not None and self.n_control is not None:
            lines.extend(
                [
                    f"Treated: {self.n_treated}   Control: {self.n_control}",
                    f"Total: {self.n_treated + self.n_control}",
                ]
            )
        else:
            counts = self.data[self.treatment_col].value_counts().sort_index()
            count_text = ", ".join(f"{arm}: {count}" for arm, count in counts.items())
            lines.append(f"Arm counts: {count_text}")
            lines.append(f"Total: {len(self.data)}")
        if self.strata_col:
            lines.append(f"Stratified by: {self.strata_col}")
        if self.seed is not None:
            lines.append(f"Seed: {self.seed}")
        if self.balance is not None:
            lines.append(f"\nBalance (F-test p-value): {self.balance['omnibus_p']:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)


class BalanceResult:
    """Results from balance check.

    Produced by :func:`balance_check`. Exposes the per-covariate balance
    ``.table`` (means, raw and normalized differences, t-test p-values),
    the omnibus F-test, and ``.summary()`` / ``.plot()`` (a love plot of
    normalized differences).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> treated = rng.integers(0, 2, n)
    >>> df = pd.DataFrame({"treated": treated,
    ...                    "age": rng.normal(40, 10, n),
    ...                    "income": rng.normal(50, 15, n)})
    >>> bal = sp.balance_check(df, treatment="treated",
    ...                        covariates=["age", "income"])
    >>> type(bal).__name__
    'BalanceResult'
    >>> bal.n_treat + bal.n_control
    200
    >>> list(bal.table["variable"])
    ['age', 'income']
    >>> isinstance(bal.summary(), str)
    True
    """

    def __init__(
        self,
        table: pd.DataFrame,
        omnibus_f: float,
        omnibus_p: float,
        normalized_diffs: Dict[str, float],
        n_treat: int,
        n_control: int,
    ) -> None:
        self.table = table
        self.omnibus_f = omnibus_f
        self.omnibus_p = omnibus_p
        self.normalized_diffs = normalized_diffs
        self.n_treat = n_treat
        self.n_control = n_control

    def summary(self) -> str:
        lines: List[str] = [
            "Balance Check",
            "=" * 70,
            f"Treated: {self.n_treat}   Control: {self.n_control}",
            "",
            (
                f"{'Variable':<20s} {'Mean(T)':>10s} "
                f"{'Mean(C)':>10s} {'Diff':>10s} "
                f"{'NormDiff':>10s} {'p-value':>8s}"
            ),
            "-" * 70,
        ]
        for _, row in self.table.iterrows():
            norm_diff = row.get("norm_diff", 0)
            flag = " ***" if abs(norm_diff) > 0.25 else ""
            lines.append(
                f"{row['variable']:<20s} "
                f"{row['mean_treat']:>10.4f} "
                f"{row['mean_control']:>10.4f} "
                f"{row['diff']:>10.4f} "
                f"{norm_diff:>10.4f} "
                f"{row['p_value']:>8.4f}{flag}"
            )
        lines.append("-" * 70)
        lines.append(
            "Omnibus F-test: " f"F = {self.omnibus_f:.3f}, p = {self.omnibus_p:.4f}"
        )
        lines.append("Note: *** indicates |normalized difference| > 0.25")
        return "\n".join(lines)

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Love plot of normalized differences."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, max(5, len(self.table) * 0.4)))

        nd = self.normalized_diffs
        y_pos = range(len(nd))
        vars_ = list(nd.keys())
        vals = list(nd.values())

        plot_kwargs: Dict[str, Any] = {
            "color": ["red" if abs(v) > 0.25 else "steelblue" for v in vals],
        }
        plot_kwargs.update(kwargs)
        ax.barh(y_pos, vals, **plot_kwargs)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(vars_)
        ax.axvline(0, color="black", lw=0.5)
        ax.axvline(0.25, color="red", ls="--", lw=0.5, alpha=0.5)
        ax.axvline(-0.25, color="red", ls="--", lw=0.5, alpha=0.5)
        ax.set_xlabel("Normalized Difference")
        ax.set_title("Balance: Normalized Differences")
        plt.tight_layout()
        return ax


def randomize(
    data: pd.DataFrame,
    n_arms: int = 2,
    prob: Optional[List[float]] = None,
    strata: Optional[str] = None,
    cluster: Optional[str] = None,
    method: str = "simple",
    balance_vars: Optional[List[str]] = None,
    n_rerand: int = 0,
    rerand_threshold: float = 0.001,
    seed: Optional[int] = None,
    treatment_col: str = "treatment",
) -> RandomizationResult:
    """
    Randomize units to treatment and control.

    Equivalent to R's ``randomizr::complete_ra()`` / ``block_ra()`` /
    ``cluster_ra()``.

    Parameters
    ----------
    data : pd.DataFrame
        Data with units to randomize.
    n_arms : int, default 2
        Number of treatment arms.
    prob : list of float, optional
        Probability of each arm. Default: equal.
    strata : str, optional
        Stratification variable for block randomization.
    cluster : str, optional
        Cluster variable for cluster randomization.
    method : str, default 'simple'
        'simple', 'complete', 'stratified', 'cluster'.
    balance_vars : list of str, optional
        Variables to check balance on (for re-randomization).
    n_rerand : int, default 0
        Number of re-randomization iterations (0 = no re-randomization).
    rerand_threshold : float, default 0.001
        Mahalanobis distance threshold for re-randomization.
    seed : int, optional
        Random seed for reproducibility.
    treatment_col : str, default 'treatment'
        Name of treatment column to create.

    Returns
    -------
    RandomizationResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({"district": rng.integers(0, 4, n),
    ...                    "age": rng.normal(40, 10, n),
    ...                    "income": rng.normal(50, 15, n)})
    >>> result = sp.randomize(df, strata="district",
    ...                       balance_vars=["age", "income"], seed=1)
    >>> type(result).__name__
    'RandomizationResult'
    >>> int(result.n_treated + result.n_control)
    200
    >>> bool(isinstance(result.summary(), str))
    True
    >>> df_randomized = result.data
    """
    rng = np.random.default_rng(seed)
    df = data.copy()
    n = len(df)

    if prob is None:
        prob = [1.0 / n_arms] * n_arms

    assignments: np.ndarray
    if method == "simple" and strata is None and cluster is None:
        # Complete randomization
        assignments = rng.choice(n_arms, size=n, p=prob)

    elif strata is not None or method == "stratified":
        # Stratified (block) randomization
        assignments = np.empty(n, dtype=int)
        strata_col = cast(str, strata)
        for _, group in df.groupby(strata_col):
            idx = group.index
            g_n = len(idx)
            # Within each stratum, do complete randomization
            counts = [int(round(p * g_n)) for p in prob]
            # Adjust rounding — ensure no negative counts
            while sum(counts) < g_n:
                counts[0] += 1
            while sum(counts) > g_n:
                # Find largest count to decrement
                max_idx = max(range(len(counts)), key=lambda i: counts[i])
                counts[max_idx] -= 1
            arm_labels = np.concatenate(
                [np.full(max(c, 0), arm) for arm, c in enumerate(counts)]
            )
            rng.shuffle(arm_labels)
            assignments[idx] = arm_labels[:g_n]

    elif cluster is not None or method == "cluster":
        # Cluster randomization
        cluster_col = cast(str, cluster)
        clusters = df[cluster_col].unique()
        n_clusters = len(clusters)
        cluster_assignments = rng.choice(n_arms, size=n_clusters, p=prob)
        cluster_map = dict(zip(clusters, cluster_assignments))
        assignments = df[cluster_col].map(cluster_map).to_numpy(dtype=int)

    else:
        assignments = rng.choice(n_arms, size=n, p=prob)

    # Re-randomization
    if n_rerand > 0 and balance_vars is not None:
        best_assignments = assignments.copy()
        best_distance = _mahalanobis_distance(df, balance_vars, assignments)

        for _ in range(n_rerand):
            new_assignments = rng.choice(n_arms, size=n, p=prob)
            d = _mahalanobis_distance(df, balance_vars, new_assignments)
            if d < best_distance:
                best_distance = d
                best_assignments = new_assignments.copy()
                if d < rerand_threshold:
                    break

        assignments = best_assignments

    df[treatment_col] = assignments

    # Check balance
    bal: Optional[BalanceResult] = None
    if balance_vars is not None and n_arms == 2:
        bal = balance_check(df, treatment=treatment_col, covariates=balance_vars)

    n_treated = int((assignments == 1).sum()) if n_arms == 2 else None
    n_control = int((assignments == 0).sum()) if n_arms == 2 else None

    return RandomizationResult(
        data=df,
        treatment_col=treatment_col,
        n_treated=n_treated,
        n_control=n_control,
        strata_col=strata,
        method=method,
        balance=bal.__dict__ if bal else None,
        seed=seed,
    )


def _mahalanobis_distance(
    df: pd.DataFrame,
    balance_vars: List[str],
    assignments: np.ndarray,
) -> float:
    """Compute Mahalanobis distance between treatment and control means."""
    treat = df.loc[assignments == 1, balance_vars].values
    control = df.loc[assignments == 0, balance_vars].values

    if len(treat) == 0 or len(control) == 0:
        return np.inf

    diff = treat.mean(axis=0) - control.mean(axis=0)
    pooled = np.cov(df[balance_vars].values, rowvar=False)
    try:
        inv_cov = np.linalg.inv(pooled)
        return float(diff @ inv_cov @ diff)
    except np.linalg.LinAlgError:
        return float(np.sum(diff**2))


def balance_check(
    data: pd.DataFrame,
    treatment: str,
    covariates: List[str],
    alpha: float = 0.05,
) -> BalanceResult:
    """
    Check covariate balance between treatment and control.

    Computes normalized differences, t-tests, and omnibus F-test.

    Equivalent to Stata's ``iebaltab`` and R's ``cobalt::bal.tab()``.

    Parameters
    ----------
    data : pd.DataFrame
    treatment : str
        Binary treatment variable (0/1).
    covariates : list of str
        Covariates to check balance on.
    alpha : float, default 0.05

    Returns
    -------
    BalanceResult

    Notes
    -----
    If the omnibus F-test cannot be computed (e.g. non-numeric or perfectly
    collinear covariates), ``omnibus_f`` and ``omnibus_p`` are reported as
    NaN and a ``StatsPAIWarning`` is emitted.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> df = pd.DataFrame({"treated": rng.integers(0, 2, n),
    ...                    "age": rng.normal(40, 10, n),
    ...                    "income": rng.normal(50, 15, n),
    ...                    "education": rng.normal(12, 3, n)})
    >>> bal = sp.balance_check(df, treatment="treated",
    ...                        covariates=["age", "income", "education"])
    >>> type(bal).__name__
    'BalanceResult'
    >>> list(bal.table["variable"])
    ['age', 'income', 'education']
    >>> bool(isinstance(bal.summary(), str))
    True
    """
    treat = data[data[treatment] == 1]
    control = data[data[treatment] == 0]
    n_t, n_c = len(treat), len(control)

    rows = []
    norm_diffs = {}

    for var in covariates:
        mt = treat[var].mean()
        mc = control[var].mean()
        st = treat[var].std(ddof=1)
        sc = control[var].std(ddof=1)
        diff = mt - mc

        # Normalized difference (Imbens & Rubin 2015)
        pooled_sd = np.sqrt((st**2 + sc**2) / 2)
        nd = diff / pooled_sd if pooled_sd > 0 else 0

        # t-test
        t_stat, p_val = stats.ttest_ind(
            treat[var].dropna(), control[var].dropna(), equal_var=False
        )

        rows.append(
            {
                "variable": var,
                "mean_treat": mt,
                "mean_control": mc,
                "diff": diff,
                "norm_diff": nd,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )
        norm_diffs[var] = nd

    table = pd.DataFrame(rows)

    # Omnibus F-test: regress treatment on all covariates
    try:
        y_treat = data[treatment].values.astype(float)
        X_bal = data[covariates].values.astype(float)
        X_bal = np.column_stack([np.ones(len(X_bal)), X_bal])
        valid = np.all(np.isfinite(X_bal), axis=1) & np.isfinite(y_treat)
        X_bal, y_treat = X_bal[valid], y_treat[valid]

        beta = np.linalg.lstsq(X_bal, y_treat, rcond=None)[0]
        resid = y_treat - X_bal @ beta
        rss = np.sum(resid**2)
        tss = np.sum((y_treat - y_treat.mean()) ** 2)
        k = X_bal.shape[1] - 1
        n_total = len(y_treat)
        f_stat = ((tss - rss) / k) / (rss / (n_total - k - 1))
        f_p = 1 - stats.f.cdf(f_stat, k, n_total - k - 1)
    except Exception as e:
        warnings.warn(
            f"balance_check: omnibus F-test could not be computed "
            f"({type(e).__name__}: {e}); omnibus_f/omnibus_p reported as NaN",
            StatsPAIWarning,
            stacklevel=2,
        )
        f_stat, f_p = np.nan, np.nan

    return BalanceResult(
        table=table,
        omnibus_f=f_stat,
        omnibus_p=f_p,
        normalized_diffs=norm_diffs,
        n_treat=n_t,
        n_control=n_c,
    )
