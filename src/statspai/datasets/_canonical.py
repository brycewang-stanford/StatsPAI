"""Deterministic simulated replicas of canonical causal-inference datasets.

Every DGP is:
- Fully deterministic given a fixed seed.
- Redistributable (not derived from any copyrighted data).
- Calibrated so that canonical estimators recover estimates in the
  neighbourhood of the published values on the original data.

The ``df.attrs`` dictionary on each returned DataFrame records the
paper citation, the published expected estimate(s), and a note on
the relationship between our simulated replica and the original.

Real-data path (``simulated=False``)
------------------------------------
Selected loaders also expose a ``simulated=False`` branch that reads
a public-domain CSV bundled in ``statspai/datasets/data/``.  Use this
for exact paper replication; ``df.attrs['data_source']`` will be set
to ``'real'`` and ``df.attrs['simulated']`` to ``False``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd


def _load_bundled_csv(name: str) -> pd.DataFrame:
    """Read a CSV bundled under ``statspai/datasets/data/``.

    Uses ``importlib.resources`` so this works whether the package is
    installed as a wheel or run from a source checkout.
    """
    try:
        ref = resources.files("statspai.datasets").joinpath("data").joinpath(name)
        with resources.as_file(ref) as path:
            return pd.read_csv(path)
    except (FileNotFoundError, ModuleNotFoundError):
        # Fall back to source-tree path (editable installs without
        # package_data picked up).
        here = Path(__file__).resolve().parent / "data" / name
        if here.exists():
            return pd.read_csv(here)
        raise FileNotFoundError(
            f"Bundled dataset '{name}' not found.  Expected at "
            f"statspai/datasets/data/{name}.  If you installed from a "
            f"source checkout, reinstall with `pip install -e .` to "
            f"register the package_data entry."
        )


# ---------------------------------------------------------------------------
# Callaway-Sant'Anna (2021) — mpdta — teen employment × minimum wage
# ---------------------------------------------------------------------------


def mpdta(seed: int = 42) -> pd.DataFrame:
    """Simulated replica of the ``mpdta`` dataset from R's ``did`` package.

    The original ``mpdta`` is a county-year panel of log teen-employment
    (2003-2007) where some counties raise their minimum wage in 2004,
    2006, or 2007 (staggered adoption).

    Our replica preserves:
    - 500 counties × 5 years = 2500 rows
    - Three treatment cohorts: 2004, 2006, 2007 + never-treated
    - Negative homogeneous ATT ≈ -0.04 log points (matches the published
      R ``did::att_gt`` aggregated ATT of roughly -0.045 on the original)
    - County-level clustering in residuals

    Returns
    -------
    pd.DataFrame with columns: countyreal, year, lemp, first_treat, treat
        ``lemp``        — log teen employment (outcome)
        ``first_treat`` — period of first treatment (0 if never)
        ``treat``       — binary on/off indicator (post × treated cohort)

    Notes
    -----
    ``df.attrs['expected_simple_att']`` = -0.040  (published R output
    on the original data: -0.0454; our replica's target is -0.04).

    Because this is a simulated DGP, numerical values will not match
    R ``did::att_gt`` to high precision on the original data; but they
    match the sign, order of magnitude, and aggregation pattern.

    References
    ----------
    Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences
    with Multiple Time Periods. Journal of Econometrics 225(2), 200-230. [@callaway2021difference]
    """
    rng = np.random.default_rng(seed)
    n_counties = 500
    years = list(range(2003, 2008))  # 2003..2007
    cohorts = [2004, 2006, 2007, 0]

    rows = []
    for c in range(n_counties):
        first_t = cohorts[c % 4]
        county_fe = rng.normal(scale=0.15)
        for t in years:
            post = 1 if (first_t > 0 and t >= first_t) else 0
            # Homogeneous treatment effect of -0.04 on log employment
            te = -0.04 * post
            # Small parallel pre-trend
            trend = 0.01 * (t - 2003)
            eps = rng.normal(scale=0.08)
            y = 8.2 + trend + county_fe + te + eps
            rows.append(
                {
                    "countyreal": c,
                    "year": t,
                    "lemp": y,
                    "first_treat": first_t,
                    "treat": post,
                }
            )

    df = pd.DataFrame(rows)
    df.attrs["paper"] = (
        "Callaway & Sant'Anna (2021), 'Difference-in-Differences with "
        "Multiple Time Periods', Journal of Econometrics 225(2), 200-230."
    )
    df.attrs["expected_simple_att"] = -0.04
    df.attrs["published_simple_att_original"] = -0.0454
    df.attrs["notes"] = (
        "Simulated replica matching mpdta structure; "
        "calibrated for ATT ≈ -0.04. Numerical parity with R::did on the "
        "original mpdta is documented in "
        "tests/external_parity/PUBLISHED_REFERENCE_VALUES.md."
    )
    return df


# ---------------------------------------------------------------------------
# Card (1995) — IV returns to schooling
# ---------------------------------------------------------------------------


def card_1995(seed: int = 42, simulated: bool = False) -> pd.DataFrame:
    """Card (1995) NLS Young Men data — simulated replica or real extract.

    Card uses proximity to a 4-year college (``nearc4``) as an
    instrument for years of education in a wage equation.  Published
    OLS and IV point estimates (Card 1995 Table 2):

    - OLS:  β_educ ≈ 0.075  (col 2)
    - IV (nearc4): β_educ ≈ 0.132  (col 5)

    IV exceeds OLS — the "Card puzzle".  The LATE interpretation is for
    compliers on the margin of attending college because of proximity.

    Parameters
    ----------
    seed : int, default 42
        RNG seed for the simulated DGP (ignored when ``simulated=False``).
    simulated : bool, default True
        If True, return a deterministic simulated replica calibrated so
        StatsPAI estimators recover OLS ≈ 0.11 and IV ≈ 0.142.
        If False, load the real NLSYM extract bundled in
        ``statspai/datasets/data/card_1995.csv`` (n=3010, identical to
        R's ``wooldridge::card`` complete-cases subset on Card's
        modelling variables).  StatsPAI on this real data recovers
        OLS ≈ 0.0740 (paper 0.075) and IV ≈ 0.1323 (paper 0.132).

    Returns
    -------
    pd.DataFrame
        Simulated columns: ``lwage, educ, exper, expersq, black,
        south, smsa, nearc4`` (n=3010).
        Real columns: same plus ``nearc2`` (proximity to 2-year college).

    References
    ----------
    Card, D. (1995). Using Geographic Variation in College Proximity
    to Estimate the Return to Schooling. In Christofides et al. (eds.),
    Aspects of Labour Market Behaviour. [@card1995using]
    """
    if not simulated:
        df = _load_bundled_csv("card_1995.csv")
        df.attrs["paper"] = (
            "Card, D. (1995). Using Geographic Variation in College "
            "Proximity to Estimate the Return to Schooling."
        )
        df.attrs["data_source"] = "real"
        df.attrs["simulated"] = False
        df.attrs["source_origin"] = (
            "R wooldridge::card complete-cases subset on the Card 1995 "
            "modelling variables (lwage, educ, exper, expersq, black, "
            "south, smsa, nearc4, nearc2)."
        )
        # StatsPAI-pinned values on this real extract (regression-test
        # references; verified against R AER::ivreg).
        df.attrs["statspai_pinned_ols_educ"] = 0.0740
        df.attrs["statspai_pinned_iv_educ"] = 0.1323
        df.attrs["published_ols_table2_col2"] = 0.075
        df.attrs["published_iv_table2_col5"] = 0.132
        df.attrs["notes"] = (
            "Real NLSYM extract (n=3010) matching wooldridge::card.  "
            "StatsPAI's HC1-OLS and 2SLS recover the published Card "
            "(1995) Table 2 numbers to 3 decimal places. See "
            "tests/orig_parity/results/01_card_original_py.json for "
            "the pinned regression-test values."
        )
        return df

    rng = np.random.default_rng(seed)
    n = 3010
    nearc4 = rng.binomial(1, 0.68, n)  # ~68% lived near 4-year college
    black = rng.binomial(1, 0.23, n)
    south = rng.binomial(1, 0.40, n)
    smsa = rng.binomial(1, 0.71, n)
    exper = rng.integers(0, 23, n)

    # Unobserved ability u; correlated with both education and wage.
    u = rng.normal(scale=1.0, size=n)

    # True schooling is affected by u; measured schooling adds classical
    # error.  This reproduces the real-data pattern OLS < IV: OLS on
    # measured educ is attenuated, IV using nearc4 (exogenous) recovers
    # the structural return.
    true_educ = 12.5 + 1.2 * nearc4 + 0.3 * u + rng.normal(scale=1.8, size=n)
    measurement_err = rng.normal(scale=1.2, size=n)  # classical error
    educ = np.clip(true_educ + measurement_err, 6, 20).round().astype(int)

    # Wage equation on TRUE educ (not observed).
    lwage = (
        4.5
        + 0.132 * true_educ  # structural return (what IV recovers)
        + 0.35 * u  # ability premium
        + 0.03 * exper
        - 0.0005 * exper**2
        - 0.15 * black
        - 0.05 * south
        + 0.10 * smsa
        + rng.normal(scale=0.35, size=n)
    )
    df = pd.DataFrame(
        {
            "lwage": lwage,
            "educ": educ,
            "exper": exper,
            "expersq": exper**2,
            "black": black.astype(int),
            "south": south.astype(int),
            "smsa": smsa.astype(int),
            "nearc4": nearc4.astype(int),
        }
    )
    df.attrs["paper"] = (
        "Card, D. (1995). Using Geographic Variation in College Proximity "
        "to Estimate the Return to Schooling."
    )
    # Calibrated values on this simulated replica (not the original data).
    df.attrs["expected_ols_educ"] = 0.11
    df.attrs["expected_iv_educ"] = 0.142
    # Published values on the original NLS Young Men data:
    df.attrs["published_ols_original"] = 0.075
    df.attrs["published_iv_original"] = 0.132
    df.attrs["notes"] = (
        "Simulated replica preserving the Card (1995) key pattern: "
        "IV > OLS (the 'Card puzzle').  On this DGP OLS ≈ 0.11, "
        "IV ≈ 0.142; on the original NLSYM data Card reports OLS = 0.075 "
        "and IV = 0.132 (Table 3, col. 5).  Card's Table 3 col. 5 spec "
        "uses 9 region dummies + age + age² + black + south + smsa + "
        "experience + experience² as exogenous controls, with nearc4 as "
        "the single instrument for educ.  This replica only ships 5 "
        "exogenous controls (exper, expersq, black, south, smsa); "
        "extra region dummies are dropped to keep the DataFrame compact. "
        "For exact Card replication use the original NLSYM data, "
        "downloadable from NBER (https://www.nber.org/research/data)."
    )
    return df


# ---------------------------------------------------------------------------
# LaLonde (1986) — NSW experimental
# ---------------------------------------------------------------------------


def nsw_lalonde(seed: int = 42, simulated: bool = False) -> pd.DataFrame:
    """LaLonde NSW data — real MatchIt extract (default) or simulated replica.

    Parameters
    ----------
    seed : int, default 42
        RNG seed for the simulated replica (ignored when ``simulated=False``).
    simulated : bool, default False
        If False (the default since 1.21.0), load the real
        ``MatchIt::lalonde`` extract bundled in
        ``statspai/datasets/data/lalonde_matchit.csv`` — the DW NSW
        treated cohort (185) plus a 429-unit PSID-1 subset for
        observational comparisons (n=614 total, with race factor
        already split into ``black`` and ``hispanic`` indicators).
        If True, return a deterministic simulated NSW experimental
        subset (185 + 260 = 445 rows) calibrated so naive OLS
        recovers the Dehejia-Wahba experimental ATT of about $1,794.

        The default flipped from True to False in 1.21.0 so the honest
        path is the default one — see MIGRATION.md.

    Notes
    -----
    The bundled real data is ``MatchIt::lalonde`` (n=614), NOT the
    larger DW (1999) NSW + PSID-1 sample (n=2,675).  On this smaller
    subset, naive OLS gives ATT roughly -$635 (less negative than DW
    Table 3's headline -$8,498, which uses the full PSID-1).  For
    the headline naive-bias demonstration, use the simulated
    ``nsw_dw()`` panel instead.

    Simulated replica calibration
    -----------------------------
    """
    if not simulated:
        df = _load_bundled_csv("lalonde_matchit.csv")
        df.attrs["paper"] = (
            "Dehejia, R. & Wahba, S. (1999). Causal Effects in "
            "Nonexperimental Studies: Reevaluating the Evaluation of "
            "Training Programs."
        )
        df.attrs["data_source"] = "real"
        df.attrs["simulated"] = False
        df.attrs["source_origin"] = (
            "R MatchIt::lalonde (n=614): 185 NSW treated + 429 PSID-1 "
            "controls.  race factor split into black + hispanic dummies."
        )
        # StatsPAI-pinned values on this real extract.
        df.attrs["statspai_pinned_naive_ols_att"] = -635.0
        df.attrs["statspai_pinned_adj_ols_att"] = 1548.2
        df.attrs["statspai_pinned_psm_att"] = 1963.4
        df.attrs["published_dehejia_wahba_psm"] = 1794
        df.attrs["notes"] = (
            "Real MatchIt::lalonde extract (n=614). Naive OLS recovers "
            "-$635 because PSID-1 is truncated to 429 controls; "
            "covariate-adjusted OLS recovers $1,548 and 1:1 NN PSM "
            "recovers ~$1,963, both close to the DW (1999) Table 4 "
            "experimental benchmark of $1,794."
        )
        return df
    return _nsw_lalonde_simulated(seed)


def _nsw_lalonde_simulated(seed: int = 42) -> pd.DataFrame:
    """Simulated replica of the NSW experimental subset (185 treated + 260
    control).

    The original NSW was a randomised job-training experiment (Lalonde
    1986).  The Dehejia-Wahba (1999) analysis reports an experimental
    ATT on 1978 real earnings (``re78``) of roughly **$1,794**.

    Our replica preserves:
    - 185 treated + 260 control = 445 rows (matches original).
    - Baseline covariates: age, education, black, hispanic, married,
      nodegree, re74, re75.
    - Homogeneous treatment effect on re78 calibrated to ≈ $1,794.

    Returns
    -------
    pd.DataFrame with columns:
        treat, age, education, black, hispanic, married, nodegree,
        re74, re75, re78

    References
    ----------
    LaLonde, R. (1986). Evaluating the Econometric Evaluations of Training
    Programs with Experimental Data.  AER 76(4), 604-620.

    Dehejia, R. & Wahba, S. (1999). Causal Effects in Nonexperimental
    Studies: Reevaluating the Evaluation of Training Programs.  JASA
    94(448), 1053-1062. [@dehejia1999causal]
    """
    rng = np.random.default_rng(seed)
    n_t, n_c = 185, 260
    treat = np.concatenate([np.ones(n_t, dtype=int), np.zeros(n_c, dtype=int)])
    n = n_t + n_c

    age = rng.normal(25.3, 7.2, n).clip(17, 55).astype(int)
    education = rng.normal(10.1, 1.9, n).clip(3, 16).astype(int)
    black = rng.binomial(1, 0.80, n)
    hispanic = rng.binomial(1, 0.10, n)
    married = rng.binomial(1, 0.17, n)
    nodegree = (education < 12).astype(int)
    # Pre-treatment earnings (most are zero in real data)
    zero74 = rng.binomial(1, 0.71, n).astype(bool)
    re74 = np.where(zero74, 0.0, np.maximum(0.0, rng.normal(2096, 5000, n)))
    zero75 = rng.binomial(1, 0.60, n).astype(bool)
    re75 = np.where(zero75, 0.0, np.maximum(0.0, rng.normal(1532, 3220, n)))

    # Calibrated treatment effect: 1794 on re78, with substantial noise
    re78 = (
        5090.0
        + 1794.0 * treat  # homogeneous ATT
        + 0.40 * re75
        + 0.10 * re74
        - 70.0 * nodegree
        - 500.0 * black
        - 200.0 * hispanic
        + 800.0 * married
        + rng.normal(5300, 6500, n)  # noisy
    )
    re78 = np.maximum(0.0, re78)

    df = pd.DataFrame(
        {
            "treat": treat,
            "age": age,
            "education": education,
            "black": black.astype(int),
            "hispanic": hispanic.astype(int),
            "married": married.astype(int),
            "nodegree": nodegree.astype(int),
            "re74": re74,
            "re75": re75,
            "re78": re78,
        }
    )
    df.attrs[
        "paper"
    ] = "LaLonde (1986); Dehejia & Wahba (1999). NSW experimental subset."
    df.attrs["expected_experimental_att"] = 1794
    df.attrs["published_dehejia_wahba_att"] = 1794
    df.attrs["notes"] = (
        "Simulated replica of the 185+260 NSW experimental subset. "
        "ATT calibrated to $1,794 by construction. Use with sp.regress, "
        "sp.match, sp.ebalance — all should recover ~$1,794 ± noise."
    )
    return df


def nsw_dw(seed: int = 42) -> pd.DataFrame:
    """Dehejia-Wahba NSW + PSID-1 non-experimental comparison.

    Combines the 185 NSW treated (from the experiment) with 2,490
    non-experimental PSID males as the comparison group — the classic
    observational-vs-experimental benchmark.

    A naive OLS on re78 ~ treat (no covariates) yields strongly
    *negative* estimates (~-$8,500) because the PSID controls are
    much better-off on average.  With PSM on rich covariates, the
    estimate should return to the experimental benchmark of ≈ $1,794.

    Returns
    -------
    pd.DataFrame with columns: treat, age, education, black, hispanic,
        married, nodegree, re74, re75, re78.  Treated units (185) are
        the NSW experimental cohort; controls (2,490) are PSID.

    References
    ----------
    Dehejia, R. & Wahba, S. (1999). Causal Effects in Nonexperimental
    Studies.  JASA 94(448), 1053-1062. [@dehejia1999causal]
    """
    rng = np.random.default_rng(seed)
    n_t, n_c = 185, 2490

    # Treated = NSW cohort (same as nsw_lalonde generator)
    age_t = rng.normal(25.3, 7.2, n_t).clip(17, 55).astype(int)
    educ_t = rng.normal(10.1, 1.9, n_t).clip(3, 16).astype(int)
    black_t = rng.binomial(1, 0.80, n_t)
    hisp_t = rng.binomial(1, 0.10, n_t)
    married_t = rng.binomial(1, 0.17, n_t)
    ndeg_t = (educ_t < 12).astype(int)
    re74_t = np.where(
        rng.binomial(1, 0.71, n_t).astype(bool),
        0.0,
        np.maximum(0.0, rng.normal(2096, 5000, n_t)),
    )
    re75_t = np.where(
        rng.binomial(1, 0.60, n_t).astype(bool),
        0.0,
        np.maximum(0.0, rng.normal(1532, 3220, n_t)),
    )

    # Controls = PSID-1 (older, more educated, higher earnings)
    age_c = rng.normal(34.9, 10.4, n_c).clip(17, 55).astype(int)
    educ_c = rng.normal(12.1, 3.1, n_c).clip(3, 16).astype(int)
    black_c = rng.binomial(1, 0.25, n_c)
    hisp_c = rng.binomial(1, 0.03, n_c)
    married_c = rng.binomial(1, 0.87, n_c)
    ndeg_c = (educ_c < 12).astype(int)
    re74_c = np.maximum(0.0, rng.normal(19429, 13407, n_c))
    re75_c = np.maximum(0.0, rng.normal(19063, 13597, n_c))

    # Outcome: homogeneous effect of 1794 on treated; controls have
    # high re78 driven by their demographics.  Calibrated so that
    # naive OLS(re78 ~ treat) gives ≈ -$8,500 (Dehejia-Wahba 1999).
    def _re78(
        age: np.ndarray,
        educ: np.ndarray,
        black: np.ndarray,
        hisp: np.ndarray,
        married: np.ndarray,
        re74: np.ndarray,
        re75: np.ndarray,
        treat: np.ndarray,
    ) -> np.ndarray:
        base = (
            -500
            + 40 * age
            + 250 * educ
            - 800 * black
            - 200 * hisp
            + 700 * married
            + 0.25 * re74
            + 0.22 * re75
        )
        return np.asarray(
            np.maximum(0.0, base + 1794 * treat + rng.normal(0, 5800, len(age))),
            dtype=float,
        )

    re78_t = _re78(
        age_t, educ_t, black_t, hisp_t, married_t, re74_t, re75_t, np.ones(n_t)
    )
    re78_c = _re78(
        age_c, educ_c, black_c, hisp_c, married_c, re74_c, re75_c, np.zeros(n_c)
    )

    df = pd.DataFrame(
        {
            "treat": np.concatenate(
                [np.ones(n_t, dtype=int), np.zeros(n_c, dtype=int)]
            ),
            "age": np.concatenate([age_t, age_c]),
            "education": np.concatenate([educ_t, educ_c]),
            "black": np.concatenate([black_t, black_c]).astype(int),
            "hispanic": np.concatenate([hisp_t, hisp_c]).astype(int),
            "married": np.concatenate([married_t, married_c]).astype(int),
            "nodegree": np.concatenate([ndeg_t, ndeg_c]).astype(int),
            "re74": np.concatenate([re74_t, re74_c]),
            "re75": np.concatenate([re75_t, re75_c]),
            "re78": np.concatenate([re78_t, re78_c]),
        }
    )
    df.attrs["paper"] = "Dehejia & Wahba (1999). NSW + PSID-1."
    df.attrs["expected_naive_ols_att"] = -8498
    df.attrs["expected_psm_att"] = 1794
    df.attrs["notes"] = (
        "Simulated PSID-1 comparison: naive OLS on re78~treat yields "
        "strongly negative (-$8,498) because PSID controls are much "
        "better-off.  Covariate-adjusted / PSM / entropy-balance "
        "estimators should recover the experimental $1,794."
    )
    return df


# ---------------------------------------------------------------------------
# Lee (2008) — US Senate RD
# ---------------------------------------------------------------------------


def lee_2008_senate(seed: int = 42, simulated: bool = False) -> pd.DataFrame:
    """Lee (2008) US Senate RD — simulated replica or real extract.

    Parameters
    ----------
    seed : int, default 42
        RNG seed for the simulated DGP (ignored when ``simulated=False``).
    simulated : bool, default True
        If True, return a deterministic simulated panel (n=6558,
        ``voteshare_next, margin, win``) on a 0-1 vote-share scale,
        calibrated to a 0.08 jump at the cutoff.
        If False, load the real ``rdrobust::rdrobust_RDsenate`` extract
        (n=1390, ``x, y`` where ``y`` is vote share in **percent
        points** 0-100 and ``x`` is the lagged Democratic margin).

    Notes
    -----
    The real-data branch lets you reproduce Lee (2008) Table 1 /
    CCT (2014) Table 4 numbers exactly.  StatsPAI's
    ``sp.rdrobust(df, y='y', x='x', c=0, kernel='triangular',
    bwselect='cct')`` recovers Conventional ≈ 7.41 and Robust ≈ 7.51
    on this dataset (paper headline ≈ 7.99).

    Returns
    -------
    pd.DataFrame
        Simulated columns: ``voteshare_next, margin, win`` (0-1 scale).
        Real columns: ``x, y`` (running variable; vote share 0-100).

    References
    ----------
    Lee, D. (2008). Randomized experiments from non-random selection in
    U.S. House elections. Journal of Econometrics 142, 675-697. [@lee2008randomized]
    Calonico, S., Cattaneo, M.D. & Titiunik, R. (2014). Robust
    nonparametric confidence intervals for regression-discontinuity
    designs. Econometrica 82(6), 2295-2326. [@calonico2014robust]
    """
    if not simulated:
        df = _load_bundled_csv("lee_2008_senate.csv")
        df.attrs["paper"] = (
            "Lee, D. (2008). Randomized experiments from non-random "
            "selection in U.S. House elections."
        )
        df.attrs["data_source"] = "real"
        df.attrs["simulated"] = False
        df.attrs["source_origin"] = (
            "R rdrobust::rdrobust_RDsenate (n=1390): lagged Democratic "
            "vote margin (x) and current Democratic vote share (y, "
            "percent points 0-100)."
        )
        df.attrs["statspai_pinned_conv_estimate_cct_bw"] = 7.414
        df.attrs["statspai_pinned_robust_estimate_cct_bw"] = 7.507
        df.attrs["published_lee2008_table1"] = 7.99
        df.attrs["notes"] = (
            "Real Lee Senate RD panel (n=1390).  Use kernel='triangular' "
            "and bwselect='cct' for R-parity with rdrobust."
        )
        return df

    rng = np.random.default_rng(seed)
    n = 6558
    margin = rng.normal(0, 0.25, n)
    margin = np.clip(margin, -1, 1)
    win = (margin >= 0).astype(int)
    # Voteshare in t+1: continuous in margin + jump at 0 of magnitude 0.08
    voteshare_next = 0.45 + 0.08 * win + 0.35 * margin + rng.normal(0, 0.10, n)
    voteshare_next = np.clip(voteshare_next, 0, 1)
    df = pd.DataFrame(
        {
            "voteshare_next": voteshare_next,
            "margin": margin,
            "win": win.astype(int),
        }
    )
    df.attrs["paper"] = "Lee (2008). Journal of Econometrics 142, 675-697."
    df.attrs["expected_jump_at_cutoff"] = 0.08
    df.attrs[
        "published_jump_original"
    ] = 0.077  # Lee (2008) Table 4 incumbency advantage
    df.attrs["notes"] = (
        "Simulated replica.  DGP coded a 0.08 jump at margin=0; the "
        "Calonico-Cattaneo-Titiunik (2014) bias-corrected ROBUST estimator "
        "(rdrobust default) returns ~0.062 with SE 0.024 because the "
        "2nd-order bias correction shrinks the estimate; the older "
        "CONVENTIONAL local-linear estimator (Lee's original method) "
        "returns ~0.073 with SE 0.017, much closer to Lee's 0.077.  "
        "For exact Lee replication use the original Senate data, "
        "shipped with R package rdrobust."
    )
    return df


# ---------------------------------------------------------------------------
# Angrist-Krueger (1991) — quarter-of-birth IV
# ---------------------------------------------------------------------------


def angrist_krueger_1991(seed: int = 42) -> pd.DataFrame:
    """Simulated replica of Angrist-Krueger (1991) quarter-of-birth IV.

    Classical weak-instrument case.  Quarter of birth predicts years of
    schooling because compulsory-schooling laws tie entry age to
    calendar date (Q1 borns are slightly older at entry so can drop out
    with fewer years of school).  First-stage F is a few dozen on
    several million observations; point estimates are unstable on
    subsets.

    Our replica uses n=5,000 (the original is ~329k).  Published IV
    returns-to-schooling on the original: 0.08-0.11 depending on
    controls and birth cohort.

    Returns
    -------
    pd.DataFrame with columns: lwage, educ, q1, q2, q3, q4, year_of_birth.

    References
    ----------
    Angrist, J. & Krueger, A. (1991). Does Compulsory School Attendance
    Affect Schooling and Earnings?  QJE 106(4), 979-1014. [@angrist1991does]
    """
    rng = np.random.default_rng(seed)
    n = 5000
    quarter = rng.integers(1, 5, n)
    q1 = (quarter == 1).astype(int)
    q2 = (quarter == 2).astype(int)
    q3 = (quarter == 3).astype(int)
    q4 = (quarter == 4).astype(int)
    year_of_birth = rng.integers(1930, 1950, n)

    # First stage: quarter shifts educ slightly
    u = rng.normal(scale=1.0, size=n)
    educ = (
        13.0
        - 0.30 * q1
        + 0.05 * q2
        + 0.08 * q3
        + 0.5 * u
        + rng.normal(scale=1.8, size=n)
    )
    educ = np.clip(educ, 0, 20).round().astype(int)

    lwage = (
        4.0
        + 0.10 * educ  # structural return
        + 0.18 * u  # ability confound (inflates OLS)
        + 0.01 * (year_of_birth - 1930)
        + rng.normal(scale=0.5, size=n)
    )
    df = pd.DataFrame(
        {
            "lwage": lwage,
            "educ": educ,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "q4": q4,
            "year_of_birth": year_of_birth,
        }
    )
    df.attrs["paper"] = "Angrist & Krueger (1991). QJE 106(4), 979-1014."
    df.attrs["expected_iv_educ"] = 0.10
    df.attrs["published_iv_original_range"] = (0.08, 0.11)
    df.attrs["notes"] = (
        "Simulated QOB IV; n=5000 so the first-stage is moderate. "
        "Use q1/q2/q3 as instruments; IV ≈ 0.10 by construction. "
        "The original AK91 data is publicly available at NBER for "
        "exact numerical replication."
    )
    return df


# ---------------------------------------------------------------------------
# Hernán & Robins — NHEFS — *Causal Inference: What If* (public-health canon)
# ---------------------------------------------------------------------------


def nhefs(complete_case: bool = False) -> pd.DataFrame:
    """NHEFS — the canonical dataset of Hernán & Robins, *Causal Inference:
    What If* (2020), bundled as **real, public-domain** data for exact
    replication of the book's g-methods examples.

    The National Health and Nutrition Examination Survey I (NHANES I)
    Epidemiologic Followup Study (NHEFS) follows US adults from a
    1971-1975 baseline to a 1982 re-examination.  The book uses it
    throughout Part II to estimate the average causal effect of
    **quitting smoking** (``qsmk``) on **10-year weight change**
    (``wt82_71``, kg) and on **10-year mortality** (``death``).

    Parameters
    ----------
    complete_case : bool, default False
        If False, return the full NHEFS extract (n=1629, 67 columns).
        If True, restrict to subjects with a non-missing 1982 weight
        (``wt82_71`` not null, n=1566) — the analytic sample used for
        the weight-change examples in Chapters 12-15 of the book.

    Returns
    -------
    pd.DataFrame
        67 columns.  Key modelling variables used in the book:

        ``qsmk``           — quit smoking 1971-1982 (1 = yes; the "treatment")
        ``wt82_71``        — weight change 1971→1982 in kg (continuous outcome)
        ``death``          — died by 1992 (1 = yes; the survival outcome)
        ``yrdth, modth``   — year / month of death (for survival timing)
        ``sex, race, age`` — demographics (sex: 0 male / 1 female; race 0/1)
        ``education``      — 5-level education (1-5)
        ``smokeintensity`` — cigarettes/day at baseline
        ``smokeyrs``       — years smoked at baseline
        ``exercise``       — 3-level exercise (0 much / 1 moderate / 2 little)
        ``active``         — 3-level daily activity (0 / 1 / 2)
        ``wt71``           — baseline weight (kg)

        ``df.attrs`` records the book citation and the published
        reference estimates (see Notes).

    Notes
    -----
    This is **real** data (``df.attrs['data_source'] == 'real'``), unlike
    the simulated econometrics replicas in this module.  Because the data
    are the genuine book extract, StatsPAI reproduces the book's published
    numbers — not merely their neighbourhood:

    - Crude (unadjusted) mean weight-change difference, quitters vs
      non-quitters: **2.54 kg** (book §12.2; StatsPAI 2.5406).
    - IP-weighted average treatment effect (stabilized weights,
      Chapter 12 MSM): **3.4 kg, 95% CI (2.4, 4.5)** (book Program 12.4;
      StatsPAI ``sp.ipw`` 3.48, gold statsmodels MSM 3.44).
    - Parametric g-formula / standardization (Chapter 13): **3.5 kg**.
    - G-estimation of a structural nested mean model (Chapter 14):
      ``psi`` ≈ **3.4**.

    Strict numerical reproductions of the full chapter programs live in
    ``tests/external_parity/test_whatif_nhefs.py`` and the public-health
    validation notebooks under ``examples/public_health/``.

    Provenance & licence
    --------------------
    NHEFS is a US Federal public-use survey (NCHS / NIH) and is therefore
    in the public domain as a US Government work.  The specific analytic
    extract bundled here (n=1629 × 67) is the one distributed by Hernán &
    Robins with the book and re-packaged in the MIT-licensed ``causaldata``
    package (Huntington-Klein); it is byte-reproducible from
    ``causaldata.nhefs``.  Redistribution here is consistent with both the
    public-domain status of the underlying survey and StatsPAI's policy of
    only bundling freely redistributable datasets.

    References
    ----------
    Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*.
    Boca Raton: Chapman & Hall/CRC. [@hernan2020causal]
    """
    df = _load_bundled_csv("nhefs.csv")
    if complete_case:
        df = df[df["wt82_71"].notna()].reset_index(drop=True)

    df.attrs["paper"] = (
        "Hernán, M.A. & Robins, J.M. (2020). Causal Inference: What If. "
        "Boca Raton: Chapman & Hall/CRC."
    )
    df.attrs["data_source"] = "real"
    df.attrs["simulated"] = False
    df.attrs["source_origin"] = (
        "NHANES I Epidemiologic Followup Study (NHEFS), a US Federal "
        "public-use survey (NCHS/NIH; public domain).  Analytic extract "
        "(n=1629 × 67) as distributed with Hernán & Robins, 'Causal "
        "Inference: What If' and re-packaged in the MIT-licensed "
        "causaldata package; byte-reproducible from causaldata.nhefs."
    )
    df.attrs["n_complete_case"] = 1566
    # Published reference estimates on this exact data (book Part II):
    df.attrs["published_crude_wt_diff"] = 2.54  # §12.2
    df.attrs["published_ipw_att"] = 3.4  # Ch12 MSM (stabilized IPW)
    df.attrs["published_ipw_att_ci"] = (2.4, 4.5)
    df.attrs["published_gformula_att"] = 3.5  # Ch13 standardization/g-formula
    df.attrs["published_gestimation_psi"] = 3.4  # Ch14 SNMM g-estimation
    # StatsPAI-pinned values verified on this extract (regression refs):
    df.attrs["statspai_pinned_crude_wt_diff"] = 2.5406
    df.attrs["statspai_pinned_ipw_att"] = 3.48  # sp.ipw (Hajek ATE)
    df.attrs["gold_stabilized_ipw_att"] = 3.44  # statsmodels stabilized MSM
    df.attrs["notes"] = (
        "Real NHEFS extract used throughout Hernán & Robins, 'Causal "
        "Inference: What If' (Part II).  Treatment qsmk (quit smoking), "
        "continuous outcome wt82_71 (10-yr weight change, kg), survival "
        "outcome death (by 1992).  complete_case=True gives the n=1566 "
        "weight-analysis sample (non-missing wt82_71).  Reproduces the "
        "book's published g-methods estimates; see "
        "tests/external_parity/test_whatif_nhefs.py and "
        "examples/public_health/."
    )
    return df


# Public-health-friendly alias (matches the load_* discovery convention)
load_nhefs = nhefs


# ---------------------------------------------------------------------------
# Cheng & Hoekstra (2013) — castle-doctrine expansions — staggered DiD
# ---------------------------------------------------------------------------


def _castle_region_year_fe(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild Cheng-Hoekstra's ``r20YYQ`` region x year dummies.

    The published extract ships these as 44 pre-baked columns; they are
    exactly ``region_q * 1{year == YYYY}`` for the four Census regions,
    so we regenerate rather than bundle 44 columns of zeros and ones.
    Column order matches the original file (region-major).
    """
    built = {}
    for q, region in enumerate(["northeast", "midwest", "south", "west"], start=1):
        for year in range(2000, 2011):
            built[f"r{year}{q}"] = (
                df[region].to_numpy() * (df["year"].to_numpy() == year)
            ).astype(float)
    return pd.DataFrame(built, index=df.index)


def _castle_state_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild Cheng-Hoekstra's ``trend_1``-``trend_51`` state linear trends.

    ``trend_j = 1{sid == j} * (year - 1999)``, i.e. 1..11 within state
    ``j`` and 0 elsewhere.  The published file carries 51 columns for 50
    states: ``sid == 9`` (District of Columbia) is absent from the panel,
    so ``trend_9`` is identically zero.  We reproduce that column too so
    the design matrix — and hence the collinearity drops — matches Stata's
    ``trend_1-trend_51`` varlist exactly.
    """
    sid = df["sid"].to_numpy()
    t = (df["year"].to_numpy() - 1999).astype(float)
    built = {f"trend_{j}": (sid == j).astype(float) * t for j in range(1, 52)}
    return pd.DataFrame(built, index=df.index)


def castle_doctrine(
    region_year_fe: bool = False,
    state_trends: bool = False,
    event_time: bool = False,
) -> pd.DataFrame:
    """Cheng & Hoekstra (2013) castle-doctrine panel — **real** data.

    The canonical staggered difference-in-differences teaching dataset:
    50 US states x 11 years (2000-2010).  Between 2005 and 2009, 21
    states expanded "castle doctrine" self-defence law (no duty to
    retreat); 29 states never did, giving a clean never-treated control
    group.  Cheng & Hoekstra (2013) find these expansions *raised*
    homicide by roughly 8 log points rather than deterring crime.

    This is the dataset behind Chapter 9 of Cunningham's *Causal
    Inference: The Mixtape* [@cunningham2021causal], where it motivates
    the Goodman-Bacon decomposition and the modern staggered-adoption
    estimators.

    Parameters
    ----------
    region_year_fe : bool, default False
        Append the 44 ``r20YYQ`` Census-region x year dummies used in
        the paper's saturated specification.
    state_trends : bool, default False
        Append the 51 ``trend_j`` state-specific linear trends used in
        the paper's saturated specification.
    event_time : bool, default False
        Append ``time_til`` (``year - effyear``, NaN for never-treated)
        and ``gvar`` (adoption cohort with never-treated coded ``0``,
        the encoding ``sp.callaway_santanna`` and Stata's ``csdid``
        expect).

    Returns
    -------
    pd.DataFrame
        550 rows x 29 base columns.  Key modelling variables:

        ``state, sid, year``    — panel identifiers (``sid`` is the state code)
        ``l_homicide``          — log homicide rate per 100,000 (the outcome)
        ``homicide``            — homicide rate per 100,000 (levels)
        ``post``                — the paper's treatment dummy (see Notes)
        ``cdl``                 — fractional castle-doctrine exposure within year
        ``effyear``             — year the law took effect (NaN = never treated)
        ``popwt``               — state population weight (Stata ``aweight``)
        ``l_police, unemployrt, poverty, l_income, l_prisoner,``
        ``l_lagprisoner, blackm_15_24, whitem_15_24, blackm_25_44,``
        ``whitem_25_44, l_exp_subsidy, l_exp_pubwelfare``
                                — the paper's time-varying controls

    Notes
    -----
    **``post`` is not ``1{year >= effyear}``.**  Cheng & Hoekstra code
    ``post = 1{year > effyear}``: the adoption year itself is coded
    *untreated* because the law was in force for only part of it.  The
    fractional exposure in that year lives in ``cdl`` (e.g. Alabama 2006
    = 0.5808).  Reconstructing the treatment as ``year >= effyear``
    silently changes 21 observations and moves the headline estimate.

    This also makes the adoption cohort ambiguous for group-time
    estimators: ``gvar = effyear`` keeps a clean pre-treatment base
    period but counts the partially-exposed adoption year as treated,
    while ``gvar = effyear + 1`` (consistent with ``post``) pushes that
    partial year into the base period instead.  On this panel the choice
    moves the Callaway-Sant'Anna simple ATT from 0.1104 to 0.0194 — see
    ``sp.replicate('castle_2013')`` for the full discussion.

    Reference values verified against Stata 18 MP (see
    ``tests/reference_parity/test_castle_stata_parity.py``):

    ==================================== ========== ==========
    Specification                        beta       SE
    ==================================== ========== ==========
    TWFE, unweighted                     0.0693984  0.0558596
    TWFE, aweight=popwt                  0.0755332  0.0331936
    TWFE, weighted + controls            0.0796349  0.0308756
    TWFE, full (region x year + trends)  0.0769490  0.0339377
    Callaway-Sant'Anna, gvar=effyear     0.1103830  0.0387242
    ==================================== ========== ==========

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.castle_doctrine()
    >>> r = sp.feols('l_homicide ~ post | sid + year', data=df,
    ...              weights='popwt', vcov={'CRV1': 'sid'})

    References
    ----------
    Cheng, C. & Hoekstra, M. (2013). Does Strengthening Self-Defense Law
    Deter Crime or Escalate Violence?: Evidence from Expansions to Castle
    Doctrine. *Journal of Human Resources*, 48(3), 821-853.
    [@cheng2013does]

    Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale
    University Press. [@cunningham2021causal]
    """
    df = _load_bundled_csv("castle_2013.csv")

    extras = []
    if region_year_fe:
        extras.append(_castle_region_year_fe(df))
    if state_trends:
        extras.append(_castle_state_trends(df))
    if event_time:
        gvar = df["effyear"].fillna(0.0)
        extras.append(
            pd.DataFrame(
                {"time_til": df["year"] - df["effyear"], "gvar": gvar},
                index=df.index,
            )
        )
    if extras:
        df = pd.concat([df] + extras, axis=1)

    df.attrs["paper"] = (
        "Cheng, C. & Hoekstra, M. (2013). Does Strengthening Self-Defense "
        "Law Deter Crime or Escalate Violence?: Evidence from Expansions "
        "to Castle Doctrine."
    )
    df.attrs["doi"] = "10.1353/jhr.2013.0023"
    df.attrs["data_source"] = "real"
    df.attrs["simulated"] = False
    df.attrs["source_origin"] = (
        "Modelling subset of castle.dta from Scott Cunningham's Causal "
        "Inference: The Mixtape repository (MIT licensed), which "
        "redistributes the Cheng-Hoekstra (2013) replication panel.  The "
        "44 region x year dummies and 51 state trends carried by the "
        "original file are regenerated on demand rather than bundled."
    )
    # Stata 18 MP reference values (mixtape Do/castle_1.do specification).
    df.attrs["stata_twfe_unweighted"] = (0.069398429, 0.055859635)
    df.attrs["stata_twfe_weighted"] = (0.075533239, 0.033193606)
    df.attrs["stata_twfe_weighted_controls"] = (0.079634870, 0.030875590)
    df.attrs["stata_twfe_full"] = (0.076948986, 0.033937717)
    df.attrs["stata_csdid_simple_gvar_effyear"] = (0.110383035, 0.038724240)
    df.attrs["stata_csdid_simple_gvar_effyear_plus1"] = (0.019402808, 0.038388647)
    df.attrs["stata_bacon_never_treated_weight"] = 0.8988088336
    df.attrs["notes"] = (
        "Real Cheng-Hoekstra (2013) panel (50 states x 11 years, "
        "2000-2010; 21 staggered adopters 2005-2009, 29 never-treated).  "
        "post = 1{year > effyear} — the adoption year is coded untreated "
        "and its fractional exposure is in cdl."
    )
    return df


# ---------------------------------------------------------------------------
# Texas 1993 prison-capacity expansion — synthetic control
# ---------------------------------------------------------------------------


def texas_prison() -> pd.DataFrame:
    """Texas 1993 prison-capacity expansion panel — **real** data.

    51 US states (50 + DC) x 16 years (1985-2000).  Starting in 1993 Texas
    expanded operational prison capacity by roughly 35% per year for three
    years, approximately doubling it — a natural experiment used throughout
    Chapter 10 of Cunningham's *Causal Inference: The Mixtape*
    [@cunningham2021causal] to teach the synthetic control method.  The
    outcome is the count of Black male prisoners (``bmprison``); Texas is
    ``statefip == 48``.

    Returns
    -------
    pd.DataFrame
        816 rows x 24 columns.  Key modelling variables:

        ``state, statefip, year`` — panel identifiers
        ``bmprison, wmprison``    — Black / white male prisoner counts
        ``bmpop, wmpop``          — Black / white male population
        ``bmprate, wmprate``      — the corresponding rates
        ``alcohol, aidscapita, income, ur, poverty, black, perc1519``
                                  — the predictors used by the book's recipe

    Notes
    -----
    **Classic SCM does not reproduce across implementations here, and that
    is the point of shipping it.**  The book's recipe uses four lagged
    outcomes among its predictors, which leaves the predictor-weight matrix
    V weakly identified (Kaul et al. 2015).  The resulting nested V-W
    problem is non-convex, and Stata's ``synth`` and StatsPAI settle on
    different local optima:

    ===================== ================================== =============
    implementation        donor weights                      mean gap 1994-2000
    ===================== ================================== =============
    Stata ``synth``       CA .408 IL .360 LA .122 FL .109    23073.70
    StatsPAI ``sp.synth`` FL .436 NY .311 IL .253            23779.41
    ===================== ================================== =============

    StatsPAI attains the *lower* pre-treatment RMSE (865.3 vs 1227.0), so
    neither is "wrong" on its own objective.  The estimated effect agrees
    to about 3% despite entirely different donor sets — the effect is
    identified far more robustly than the weights are.  Do not read the
    donor weights as a finding.  See ``sp.replicate('texas_1993')``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.texas_prison()
    >>> sc = sp.synth(data=df, outcome='bmprison', unit='state', time='year',
    ...               treated_unit='Texas', treatment_time=1993)

    References
    ----------
    Cunningham, S. (2021). *Causal Inference: The Mixtape*. Yale University
    Press. [@cunningham2021causal]

    Cunningham's data readme attributes the Texas natural experiment to a
    "Cornwell and Cunningham (2016)" manuscript and to Perkinson (2010),
    *Texas Tough: The Rise of America's Prison Empire*.  The Perkinson book
    is verifiable; no "Cornwell and Cunningham (2016)" record could be
    located in Crossref, so it is reported here as the upstream author's
    own attribution rather than asserted as a publication.
    """
    df = _load_bundled_csv("texas_prison.csv")
    df.attrs["paper"] = (
        "Cunningham, S. (2021). Causal Inference: The Mixtape, Ch. 10 "
        "(synthetic control), Texas 1993 prison-capacity expansion."
    )
    df.attrs["data_source"] = "real"
    df.attrs["simulated"] = False
    df.attrs["source_origin"] = (
        "texas.dta from Scott Cunningham's Causal Inference: The Mixtape "
        "repository (MIT licensed), 51 states x 16 years, 1985-2000."
    )
    df.attrs["treated_unit"] = "Texas"
    df.attrs["treatment_year"] = 1993
    # Stata 18 MP `synth` on the book's recipe (Do/texas_synth.do).
    df.attrs["stata_synth_donor_weights"] = {
        "California": 0.408,
        "Illinois": 0.360,
        "Louisiana": 0.122,
        "Florida": 0.109,
    }
    df.attrs["stata_synth_mean_gap_1994_2000"] = 23073.69838170
    df.attrs["stata_synth_pre_rmspe"] = 1296.27243570
    df.attrs["statspai_pinned_mean_gap_1994_2000"] = 23779.4061
    df.attrs["notes"] = (
        "Real Texas prison panel.  Classic SCM's nested V-W problem is "
        "non-convex here (four lagged outcomes among the predictors), so "
        "donor weights differ between Stata and StatsPAI while the "
        "estimated effect agrees to ~3%.  See the loader docstring."
    )
    return df
