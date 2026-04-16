"""
Data Generating Processes (DGPs) for causal inference.

Ready-made simulation functions for teaching, testing, and Monte Carlo studies.
Every major causal inference design has a corresponding DGP.

Usage:
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=100, n_periods=10, effect=0.5, seed=42)
    >>> df.attrs['true_effect']
    0.5
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Difference-in-Differences
# ---------------------------------------------------------------------------

def dgp_did(
    n_units: int = 100,
    n_periods: int = 10,
    effect: float = 0.5,
    staggered: bool = False,
    n_groups: int = 4,
    heterogeneous: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate Difference-in-Differences panel data.

    Parameters
    ----------
    n_units : int
        Number of cross-sectional units.
    n_periods : int
        Number of time periods.
    effect : float
        Average treatment effect on the treated.
    staggered : bool
        If True, treatment adoption is staggered across groups.
    n_groups : int
        Number of treatment-timing groups (used when ``staggered=True``).
    heterogeneous : bool
        If True, the treatment effect varies by unit.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``unit``, ``time``, ``y``, ``treated``, ``first_treat``, ``group``.
        ``df.attrs['true_effect']`` stores the average treatment effect.

    Examples
    --------
    >>> df = dgp_did(n_units=50, n_periods=8, effect=1.0, seed=0)
    >>> df.shape
    (400, 6)
    """
    rng = np.random.default_rng(seed)

    unit_fe = rng.normal(0, 1, size=n_units)
    time_fe = rng.normal(0, 0.5, size=n_periods)

    # Assign groups and first-treat times
    group = np.zeros(n_units, dtype=int)
    first_treat = np.full(n_units, np.inf)

    if staggered:
        # Group 0 = never-treated (~20% of units); groups 1..n_groups = treated
        group = rng.integers(0, n_groups + 1, size=n_units)
        treat_times = np.linspace(
            int(n_periods * 0.4), n_periods - 1, n_groups, dtype=int
        )
        for g in range(n_groups):
            first_treat[group == g + 1] = treat_times[g]
        # group 0 stays never-treated (first_treat = inf)
    else:
        # Classic 2x2: half treated at mid-point
        treated_mask = rng.choice(n_units, size=n_units // 2, replace=False)
        group[treated_mask] = 1
        first_treat[treated_mask] = n_periods // 2

    # Unit-level heterogeneous effects
    if heterogeneous:
        unit_effects = effect + rng.normal(0, 0.3, size=n_units)
    else:
        unit_effects = np.full(n_units, effect)

    rows = []
    for i in range(n_units):
        for t in range(n_periods):
            d = 1.0 if t >= first_treat[i] else 0.0
            y = (
                unit_fe[i]
                + time_fe[t]
                + unit_effects[i] * d
                + rng.normal(0, 0.5)
            )
            rows.append((i, t, y, d, first_treat[i] if first_treat[i] < np.inf else np.nan, group[i]))

    df = pd.DataFrame(rows, columns=["unit", "time", "y", "treated", "first_treat", "group"])
    df["unit"] = df["unit"].astype(int)
    df["time"] = df["time"].astype(int)
    df["group"] = df["group"].astype(int)
    df.attrs["true_effect"] = effect
    return df


# ---------------------------------------------------------------------------
# Regression Discontinuity
# ---------------------------------------------------------------------------

def dgp_rd(
    n: int = 1000,
    effect: float = 0.3,
    cutoff: float = 0.0,
    fuzzy: bool = False,
    bandwidth_relevant: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate Regression Discontinuity data (sharp or fuzzy).

    Parameters
    ----------
    n : int
        Sample size.
    effect : float
        Treatment effect at the cutoff.
    cutoff : float
        RD cutoff value.
    fuzzy : bool
        If True, generate a fuzzy RD design.
    bandwidth_relevant : float
        Standard deviation of the running variable (controls spread).
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``x``, ``treatment``.

    Examples
    --------
    >>> df = dgp_rd(n=500, effect=0.5, seed=0)
    >>> 'treatment' in df.columns
    True
    """
    rng = np.random.default_rng(seed)

    x = rng.uniform(-1, 1, size=n)

    if fuzzy:
        prob = 1.0 / (1.0 + np.exp(-3.0 * (x - cutoff)))
        treatment = rng.binomial(1, prob).astype(float)
    else:
        treatment = (x >= cutoff).astype(float)

    # Smooth control function
    f_x = 0.5 * x + 0.3 * x ** 2
    y = f_x + effect * treatment + rng.normal(0, 0.3, size=n)

    df = pd.DataFrame({"y": y, "x": x, "treatment": treatment})
    df.attrs["true_effect"] = effect
    return df


def dgp_rd_kink(
    n: int = 2000,
    kink: float = 0.8,
    cutoff: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate Regression Kink Design data (Card et al. 2015).

    The treatment function has a kink (change in slope) at the cutoff.
    The true kink effect on the outcome equals ``kink``.

    DGP: T = X + kink_T * max(X, 0), Y = 0.4*T + noise
    True RKD = 0.4 * kink_T / kink_T = kink (simplified)

    Parameters
    ----------
    n : int
        Sample size.
    kink : float
        True kink in slope at cutoff.
    cutoff : float
        Kink point location.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``x``, ``treatment``.

    Examples
    --------
    >>> df = dgp_rd_kink(n=2000, kink=0.8, seed=42)
    >>> df.attrs['true_kink']
    0.8
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    x_c = x - cutoff
    # Slope changes by `kink` at cutoff
    y = 0.5 * x_c + kink * np.maximum(x_c, 0) + rng.normal(0, 0.5, n)
    treatment = np.maximum(x_c, 0)  # treatment intensity

    df = pd.DataFrame({"y": y, "x": x, "treatment": treatment})
    df.attrs["true_kink"] = kink
    return df


def dgp_rd_multi(
    n: int = 3000,
    cutoffs: list | None = None,
    effects: list | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate multi-cutoff RD data.

    Parameters
    ----------
    n : int
        Sample size.
    cutoffs : list of float
        Cutoff values. Default [0.0, 1.0].
    effects : list of float
        Treatment effects at each cutoff. Default [2.0, 3.0].
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``x``, ``z`` (covariate).

    Examples
    --------
    >>> df = dgp_rd_multi(n=3000, seed=42)
    >>> df.attrs['true_effects']
    {0.0: 2.0, 1.0: 3.0}
    """
    if cutoffs is None:
        cutoffs = [0.0, 1.0]
    if effects is None:
        effects = [2.0, 3.0]

    rng = np.random.default_rng(seed)
    x = rng.uniform(cutoffs[0] - 2, cutoffs[-1] + 2, n)
    z = rng.normal(0, 1, n)

    y = 0.5 * x + 0.2 * z + rng.normal(0, 0.5, n)
    for c, tau in zip(cutoffs, effects):
        y += tau * (x >= c)

    df = pd.DataFrame({"y": y, "x": x, "z": z})
    df.attrs["true_effects"] = dict(zip(cutoffs, effects))
    df.attrs["cutoffs"] = cutoffs
    return df


def dgp_rd_hte(
    n: int = 3000,
    ate: float = 2.0,
    hte_coef: float = 1.5,
    cutoff: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate RD data with heterogeneous treatment effects.

    CATE(z) = ate + hte_coef * z, where z ~ N(0,1).
    So ATE = ate and the slope of heterogeneity = hte_coef.

    Parameters
    ----------
    n : int
        Sample size.
    ate : float
        Average treatment effect at cutoff.
    hte_coef : float
        Slope of heterogeneity w.r.t. covariate z.
    cutoff : float
        RD cutoff.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``x``, ``z``.

    Examples
    --------
    >>> df = dgp_rd_hte(n=3000, ate=2.0, hte_coef=1.5, seed=42)
    >>> df.attrs['true_ate']
    2.0
    >>> df.attrs['true_hte_coef']
    1.5
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    z = rng.normal(0, 1, n)
    tau = ate + hte_coef * z
    d = (x >= cutoff).astype(float)
    y = 0.5 * x + tau * d + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"y": y, "x": x, "z": z})
    df.attrs["true_ate"] = ate
    df.attrs["true_hte_coef"] = hte_coef
    df.attrs["true_cate_fn"] = lambda z_val: ate + hte_coef * z_val
    return df


def dgp_rd_2d(
    n: int = 2000,
    effect: float = 2.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate 2D boundary RD data.

    Treatment is assigned when x1 >= 0 (vertical boundary).
    True effect = ``effect``.

    Parameters
    ----------
    n : int
        Sample size.
    effect : float
        Treatment effect at boundary.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``x1``, ``x2``, ``d`` (treatment).

    Examples
    --------
    >>> df = dgp_rd_2d(n=2000, effect=2.0, seed=42)
    >>> df.attrs['true_effect']
    2.0
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1, 1, n)
    x2 = rng.uniform(-1, 1, n)
    d = (x1 >= 0).astype(float)
    y = 0.3 * x1 + 0.2 * x2 + effect * d + rng.normal(0, 0.5, n)

    df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "d": d})
    df.attrs["true_effect"] = effect
    return df


def dgp_rdit(
    n_periods: int = 200,
    effect: float = 2.0,
    cutoff_period: int = 100,
    seasonality: bool = True,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate Regression Discontinuity in Time data (Hausman-Rapson 2018).

    Parameters
    ----------
    n_periods : int
        Number of time periods.
    effect : float
        Treatment effect at the policy change date.
    cutoff_period : int
        Period of the policy change.
    seasonality : bool
        Include monthly seasonality.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``time``, ``date``.

    Examples
    --------
    >>> df = dgp_rdit(n_periods=200, effect=2.0, seed=42)
    >>> df.attrs['true_effect']
    2.0
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_periods)
    dates = pd.date_range("2010-01-01", periods=n_periods, freq="MS")

    # Trend + treatment + noise
    y = 0.01 * t + effect * (t >= cutoff_period) + rng.normal(0, 0.5, n_periods)

    # Seasonality
    if seasonality:
        month = dates.month
        y += 0.5 * np.sin(2 * np.pi * month / 12)

    # Autocorrelation (AR(1))
    for i in range(1, n_periods):
        y[i] += 0.3 * (y[i - 1] - 0.01 * (i - 1))

    df = pd.DataFrame({"y": y, "time": t, "date": dates})
    df.attrs["true_effect"] = effect
    df.attrs["cutoff_period"] = cutoff_period
    df.attrs["cutoff_date"] = str(dates[cutoff_period])
    return df


# ---------------------------------------------------------------------------
# Instrumental Variables
# ---------------------------------------------------------------------------

def dgp_iv(
    n: int = 500,
    effect: float = 0.5,
    first_stage: float = 0.4,
    n_instruments: int = 1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate Instrumental Variables data with endogeneity.

    Parameters
    ----------
    n : int
        Sample size.
    effect : float
        True causal effect of treatment on outcome.
    first_stage : float
        Strength of the instrument in the first stage.
    n_instruments : int
        Number of instruments.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``treatment``, ``instrument`` (or ``instrument_1``, ...),
        ``x1``, ``x2``.

    Examples
    --------
    >>> df = dgp_iv(n=300, effect=0.5, seed=0)
    >>> df.attrs['true_effect']
    0.5
    """
    rng = np.random.default_rng(seed)

    u = rng.normal(0, 1, size=n)  # unobserved confounder
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)

    # Instruments
    z = rng.normal(0, 1, size=(n, n_instruments))
    z_effect = z @ rng.uniform(0.8, 1.2, size=n_instruments) * first_stage

    eps_d = rng.normal(0, 0.5, size=n)
    treatment = z_effect + 0.5 * u + 0.2 * x1 + eps_d

    eps_y = rng.normal(0, 0.5, size=n)
    y = effect * treatment + 0.5 * u + 0.3 * x1 + 0.2 * x2 + eps_y

    data = {"y": y, "treatment": treatment, "x1": x1, "x2": x2}
    if n_instruments == 1:
        data["instrument"] = z[:, 0]
    else:
        for j in range(n_instruments):
            data[f"instrument_{j + 1}"] = z[:, j]

    df = pd.DataFrame(data)
    df.attrs["true_effect"] = effect
    return df


# ---------------------------------------------------------------------------
# Randomised Controlled Trial
# ---------------------------------------------------------------------------

def dgp_rct(
    n: int = 500,
    effect: float = 0.3,
    p_treat: float = 0.5,
    heterogeneous: bool = False,
    n_covariates: int = 3,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate Randomised Controlled Trial data.

    Parameters
    ----------
    n : int
        Sample size.
    effect : float
        Average treatment effect.
    p_treat : float
        Probability of assignment to treatment.
    heterogeneous : bool
        If True, the treatment effect varies with the first covariate.
    n_covariates : int
        Number of pre-treatment covariates.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``treatment``, ``x1``, ``x2``, ...

    Examples
    --------
    >>> df = dgp_rct(n=200, effect=1.0, seed=0)
    >>> df['treatment'].mean()  # approximately 0.5
    0.525
    """
    rng = np.random.default_rng(seed)

    X = rng.normal(0, 1, size=(n, n_covariates))
    beta = rng.normal(0, 0.5, size=n_covariates)
    treatment = rng.binomial(1, p_treat, size=n).astype(float)

    if heterogeneous:
        tau = effect * (1 + 0.5 * X[:, 0])
    else:
        tau = np.full(n, effect)

    y = X @ beta + tau * treatment + rng.normal(0, 1, size=n)

    data = {"y": y, "treatment": treatment}
    for j in range(n_covariates):
        data[f"x{j + 1}"] = X[:, j]

    df = pd.DataFrame(data)
    df.attrs["true_effect"] = effect
    return df


# ---------------------------------------------------------------------------
# Panel Data
# ---------------------------------------------------------------------------

def dgp_panel(
    n_units: int = 100,
    n_periods: int = 20,
    fe_sd: float = 1.0,
    te_sd: float = 0.5,
    ar1_coef: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate panel data with unit/time fixed effects and AR(1) covariate.

    Parameters
    ----------
    n_units : int
        Number of units.
    n_periods : int
        Number of time periods.
    fe_sd : float
        Standard deviation of unit fixed effects.
    te_sd : float
        Standard deviation of time fixed effects.
    ar1_coef : float
        AR(1) coefficient for the covariate process.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``unit``, ``time``, ``y``, ``x``.

    Examples
    --------
    >>> df = dgp_panel(n_units=20, n_periods=5, seed=0)
    >>> df.shape
    (100, 4)
    """
    rng = np.random.default_rng(seed)

    alpha = rng.normal(0, fe_sd, size=n_units)
    lam = rng.normal(0, te_sd, size=n_periods)
    beta = 1.0

    # AR(1) covariate per unit
    X = np.zeros((n_units, n_periods))
    X[:, 0] = rng.normal(0, 1, size=n_units)
    for t in range(1, n_periods):
        X[:, t] = ar1_coef * X[:, t - 1] + rng.normal(0, 1, size=n_units)

    eps = rng.normal(0, 1, size=(n_units, n_periods))
    Y = alpha[:, None] + lam[None, :] + beta * X + eps

    unit = np.repeat(np.arange(n_units), n_periods)
    time = np.tile(np.arange(n_periods), n_units)

    df = pd.DataFrame({
        "unit": unit,
        "time": time,
        "y": Y.ravel(),
        "x": X.ravel(),
    })
    df.attrs["true_effect"] = beta
    return df


# ---------------------------------------------------------------------------
# Observational / Matching Data
# ---------------------------------------------------------------------------

def dgp_observational(
    n: int = 1000,
    effect: float = 0.5,
    confounding: float = 0.3,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate observational data with selection on observables.

    Parameters
    ----------
    n : int
        Sample size.
    effect : float
        True treatment effect.
    confounding : float
        Strength of confounding (higher = more selection bias).
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``treatment``, ``x1``, ``x2``, ``propensity_score``.

    Examples
    --------
    >>> df = dgp_observational(n=500, effect=1.0, seed=0)
    >>> df.attrs['true_effect']
    1.0
    """
    rng = np.random.default_rng(seed)

    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)

    logit = confounding * x1 + 0.2 * x2
    propensity = 1.0 / (1.0 + np.exp(-logit))
    treatment = rng.binomial(1, propensity).astype(float)

    y = effect * treatment + 0.5 * x1 + 0.3 * x2 + rng.normal(0, 1, size=n)

    df = pd.DataFrame({
        "y": y,
        "treatment": treatment,
        "x1": x1,
        "x2": x2,
        "propensity_score": propensity,
    })
    df.attrs["true_effect"] = effect
    return df


# ---------------------------------------------------------------------------
# Cluster-Randomised Controlled Trial
# ---------------------------------------------------------------------------

def dgp_cluster_rct(
    n_clusters: int = 50,
    cluster_size: int = 20,
    effect: float = 0.3,
    icc: float = 0.1,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate cluster-randomised trial data.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    cluster_size : int
        Number of individuals per cluster.
    effect : float
        Treatment effect.
    icc : float
        Intra-cluster correlation coefficient (0-1).
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``y``, ``treatment``, ``cluster_id``, ``unit_id``.

    Examples
    --------
    >>> df = dgp_cluster_rct(n_clusters=10, cluster_size=5, effect=0.5, seed=0)
    >>> df.shape
    (50, 4)
    """
    rng = np.random.default_rng(seed)

    total_var = 1.0
    sigma_b2 = icc * total_var
    sigma_w2 = (1 - icc) * total_var

    sigma_b = np.sqrt(sigma_b2)
    sigma_w = np.sqrt(sigma_w2)

    treatment_cluster = rng.binomial(1, 0.5, size=n_clusters).astype(float)

    rows = []
    for g in range(n_clusters):
        u_g = rng.normal(0, sigma_b)
        for j in range(cluster_size):
            eps = rng.normal(0, sigma_w)
            y = effect * treatment_cluster[g] + u_g + eps
            rows.append((y, treatment_cluster[g], g, j))

    df = pd.DataFrame(rows, columns=["y", "treatment", "cluster_id", "unit_id"])
    df["cluster_id"] = df["cluster_id"].astype(int)
    df["unit_id"] = df["unit_id"].astype(int)
    df.attrs["true_effect"] = effect
    return df


# ---------------------------------------------------------------------------
# Bunching
# ---------------------------------------------------------------------------

def dgp_bunching(
    n: int = 10000,
    kink_point: float = 50000.0,
    elasticity: float = 0.3,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate bunching data around a kink point.

    Parameters
    ----------
    n : int
        Sample size.
    kink_point : float
        Location of the kink (e.g., tax threshold).
    elasticity : float
        Behavioral elasticity governing bunching intensity.
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``income``, ``counterfactual_income``.

    Examples
    --------
    >>> df = dgp_bunching(n=5000, kink_point=50000, elasticity=0.2, seed=0)
    >>> (df['income'] <= 50000).mean() > (df['counterfactual_income'] <= 50000).mean()
    True
    """
    rng = np.random.default_rng(seed)

    # Counterfactual income from log-normal centred near the kink
    log_mean = np.log(kink_point) - 0.5 * 0.3 ** 2
    z_star = rng.lognormal(mean=log_mean, sigma=0.3, size=n)

    # Behavioural response: those above the kink reduce income
    income = z_star.copy()
    above = z_star > kink_point
    income[above] = kink_point + (z_star[above] - kink_point) * (1 - elasticity)

    df = pd.DataFrame({
        "income": income,
        "counterfactual_income": z_star,
    })
    df.attrs["true_effect"] = elasticity
    return df


# ---------------------------------------------------------------------------
# Synthetic Control
# ---------------------------------------------------------------------------

def dgp_synth(
    n_units: int = 20,
    n_periods: int = 30,
    treated_unit: int = 0,
    treatment_time: int = 20,
    effect: float = 0.5,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic control data with a factor model.

    Parameters
    ----------
    n_units : int
        Number of units (including the treated unit).
    n_periods : int
        Number of time periods.
    treated_unit : int
        Index of the treated unit.
    treatment_time : int
        Period when treatment begins.
    effect : float
        Treatment effect (additive, constant post-treatment).
    seed : int or None
        Random seed.

    Returns
    -------
    pd.DataFrame
        Columns: ``unit``, ``time``, ``y``, ``treated``.

    Examples
    --------
    >>> df = dgp_synth(n_units=10, n_periods=20, effect=1.0, seed=0)
    >>> df.loc[(df['unit'] == 0) & (df['time'] >= 20), 'treated'].unique()
    array([1.])
    """
    rng = np.random.default_rng(seed)

    # Factor model: Y_it = mu_i + lambda_t * f_i + effect * D_it + eps
    mu = rng.normal(0, 1, size=n_units)
    f = rng.normal(0, 1, size=n_units)  # unit factor loadings
    lam = np.cumsum(rng.normal(0, 0.1, size=n_periods))  # common factor (trend)

    rows = []
    for i in range(n_units):
        for t in range(n_periods):
            d = 1.0 if (i == treated_unit and t >= treatment_time) else 0.0
            y = mu[i] + lam[t] * f[i] + effect * d + rng.normal(0, 0.3)
            rows.append((i, t, y, d))

    df = pd.DataFrame(rows, columns=["unit", "time", "y", "treated"])
    df["unit"] = df["unit"].astype(int)
    df["time"] = df["time"].astype(int)
    df.attrs["true_effect"] = effect
    return df


# ---------------------------------------------------------------------------
# Shift-Share / Bartik
# ---------------------------------------------------------------------------

def dgp_bartik(
    n_regions: int = 50,
    n_industries: int = 10,
    effect: float = 1.0,
    seed: int | None = None,
) -> dict:
    """Generate shift-share (Bartik) instrument data.

    Parameters
    ----------
    n_regions : int
        Number of regions.
    n_industries : int
        Number of industries.
    effect : float
        True effect of the Bartik instrument on the outcome.
    seed : int or None
        Random seed.

    Returns
    -------
    dict
        ``'data'``: DataFrame with ``y``, ``bartik``, ``region``;
        ``'shares'``: (n_regions, n_industries) array;
        ``'shocks'``: (n_industries,) array.

    Examples
    --------
    >>> result = dgp_bartik(n_regions=20, n_industries=5, seed=0)
    >>> result['data'].shape
    (20, 3)
    """
    rng = np.random.default_rng(seed)

    # Industry shares per region (Dirichlet -> rows sum to 1)
    shares = rng.dirichlet(np.ones(n_industries), size=n_regions)

    # National industry-level shocks
    shocks = rng.normal(0, 1, size=n_industries)

    # Bartik instrument
    bartik = shares @ shocks

    y = effect * bartik + rng.normal(0, 1, size=n_regions)

    data = pd.DataFrame({
        "y": y,
        "bartik": bartik,
        "region": np.arange(n_regions),
    })
    data.attrs["true_effect"] = effect

    return {"data": data, "shares": shares, "shocks": shocks}
