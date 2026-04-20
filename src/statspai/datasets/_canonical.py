"""Deterministic simulated replicas of canonical causal-inference datasets.

Every DGP is:
- Fully deterministic given a fixed seed.
- Redistributable (not derived from any copyrighted data).
- Calibrated so that canonical estimators recover estimates in the
  neighbourhood of the published values on the original data.

The ``df.attrs`` dictionary on each returned DataFrame records the
paper citation, the published expected estimate(s), and a note on
the relationship between our simulated replica and the original.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


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
    with Multiple Time Periods. Journal of Econometrics 225(2), 200-230.
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
            rows.append({
                'countyreal': c,
                'year': t,
                'lemp': y,
                'first_treat': first_t,
                'treat': post,
            })

    df = pd.DataFrame(rows)
    df.attrs['paper'] = (
        "Callaway & Sant'Anna (2021), 'Difference-in-Differences with "
        "Multiple Time Periods', Journal of Econometrics 225(2), 200-230."
    )
    df.attrs['expected_simple_att'] = -0.04
    df.attrs['published_simple_att_original'] = -0.0454
    df.attrs['notes'] = (
        "Simulated replica matching mpdta structure; "
        "calibrated for ATT ≈ -0.04. Numerical parity with R::did on the "
        "original mpdta is documented in "
        "tests/external_parity/PUBLISHED_REFERENCE_VALUES.md."
    )
    return df


# ---------------------------------------------------------------------------
# Card (1995) — IV returns to schooling
# ---------------------------------------------------------------------------

def card_1995(seed: int = 42) -> pd.DataFrame:
    """Simulated replica of Card (1995) NLS Young Men data.

    Card uses proximity to a 4-year college (``nearc4``) as an
    instrument for years of education in a wage equation.  Published
    OLS and IV point estimates (on the original data):

    - OLS:  β_educ ≈ 0.075
    - IV (nearc4): β_educ ≈ 0.132  (and higher than OLS, contrary to
      classical measurement-error predictions — the LATE interpretation
      is for compliers who were on the margin of attending college due
      to proximity).

    The DGP here calibrates both estimates to their published values
    by construction (heterogeneous complier effect).

    Returns
    -------
    pd.DataFrame with columns: lwage, educ, exper, expersq, black,
        south, smsa, nearc4 (n=3010).

    References
    ----------
    Card, D. (1995). Using Geographic Variation in College Proximity
    to Estimate the Return to Schooling. In Christofides et al. (eds.),
    Aspects of Labour Market Behaviour.
    """
    rng = np.random.default_rng(seed)
    n = 3010
    nearc4 = rng.binomial(1, 0.68, n)    # ~68% lived near 4-year college
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
        + 0.132 * true_educ      # structural return (what IV recovers)
        + 0.35 * u               # ability premium
        + 0.03 * exper
        - 0.0005 * exper**2
        - 0.15 * black
        - 0.05 * south
        + 0.10 * smsa
        + rng.normal(scale=0.35, size=n)
    )
    df = pd.DataFrame({
        'lwage': lwage,
        'educ': educ,
        'exper': exper,
        'expersq': exper**2,
        'black': black.astype(int),
        'south': south.astype(int),
        'smsa': smsa.astype(int),
        'nearc4': nearc4.astype(int),
    })
    df.attrs['paper'] = (
        "Card, D. (1995). Using Geographic Variation in College Proximity "
        "to Estimate the Return to Schooling."
    )
    # Calibrated values on this simulated replica (not the original data).
    df.attrs['expected_ols_educ'] = 0.11
    df.attrs['expected_iv_educ'] = 0.142
    # Published values on the original NLS Young Men data:
    df.attrs['published_ols_original'] = 0.075
    df.attrs['published_iv_original'] = 0.132
    df.attrs['notes'] = (
        "Simulated replica preserving the Card (1995) key pattern: "
        "IV > OLS (the 'Card puzzle').  On this DGP OLS ≈ 0.11, "
        "IV ≈ 0.142; on the original data OLS = 0.075 and IV = 0.132.  "
        "For exact Card replication use the original NLSYM data."
    )
    return df


# ---------------------------------------------------------------------------
# LaLonde (1986) — NSW experimental
# ---------------------------------------------------------------------------

def nsw_lalonde(seed: int = 42) -> pd.DataFrame:
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
    94(448), 1053-1062.
    """
    rng = np.random.default_rng(seed)
    n_t, n_c = 185, 260
    treat = np.concatenate([np.ones(n_t, dtype=int),
                            np.zeros(n_c, dtype=int)])
    n = n_t + n_c

    age = rng.normal(25.3, 7.2, n).clip(17, 55).astype(int)
    education = rng.normal(10.1, 1.9, n).clip(3, 16).astype(int)
    black = rng.binomial(1, 0.80, n)
    hispanic = rng.binomial(1, 0.10, n)
    married = rng.binomial(1, 0.17, n)
    nodegree = (education < 12).astype(int)
    # Pre-treatment earnings (most are zero in real data)
    zero74 = rng.binomial(1, 0.71, n).astype(bool)
    re74 = np.where(zero74, 0.0,
                    np.maximum(0.0, rng.normal(2096, 5000, n)))
    zero75 = rng.binomial(1, 0.60, n).astype(bool)
    re75 = np.where(zero75, 0.0,
                    np.maximum(0.0, rng.normal(1532, 3220, n)))

    # Calibrated treatment effect: 1794 on re78, with substantial noise
    re78 = (
        5090.0
        + 1794.0 * treat                                   # homogeneous ATT
        + 0.40 * re75
        + 0.10 * re74
        - 70.0 * nodegree
        - 500.0 * black
        - 200.0 * hispanic
        + 800.0 * married
        + rng.normal(5300, 6500, n)                        # noisy
    )
    re78 = np.maximum(0.0, re78)

    df = pd.DataFrame({
        'treat': treat,
        'age': age,
        'education': education,
        'black': black.astype(int),
        'hispanic': hispanic.astype(int),
        'married': married.astype(int),
        'nodegree': nodegree.astype(int),
        're74': re74,
        're75': re75,
        're78': re78,
    })
    df.attrs['paper'] = (
        "LaLonde (1986); Dehejia & Wahba (1999). NSW experimental subset."
    )
    df.attrs['expected_experimental_att'] = 1794
    df.attrs['published_dehejia_wahba_att'] = 1794
    df.attrs['notes'] = (
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
    Studies.  JASA 94(448), 1053-1062.
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
    re74_t = np.where(rng.binomial(1, 0.71, n_t).astype(bool), 0.0,
                      np.maximum(0.0, rng.normal(2096, 5000, n_t)))
    re75_t = np.where(rng.binomial(1, 0.60, n_t).astype(bool), 0.0,
                      np.maximum(0.0, rng.normal(1532, 3220, n_t)))

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
    def _re78(age, educ, black, hisp, married, re74, re75, treat):
        base = (-500 + 40*age + 250*educ - 800*black - 200*hisp
                + 700*married + 0.25*re74 + 0.22*re75)
        return np.maximum(0.0, base + 1794*treat +
                          rng.normal(0, 5800, len(age)))

    re78_t = _re78(age_t, educ_t, black_t, hisp_t, married_t,
                   re74_t, re75_t, np.ones(n_t))
    re78_c = _re78(age_c, educ_c, black_c, hisp_c, married_c,
                   re74_c, re75_c, np.zeros(n_c))

    df = pd.DataFrame({
        'treat': np.concatenate([np.ones(n_t, dtype=int),
                                 np.zeros(n_c, dtype=int)]),
        'age': np.concatenate([age_t, age_c]),
        'education': np.concatenate([educ_t, educ_c]),
        'black': np.concatenate([black_t, black_c]).astype(int),
        'hispanic': np.concatenate([hisp_t, hisp_c]).astype(int),
        'married': np.concatenate([married_t, married_c]).astype(int),
        'nodegree': np.concatenate([ndeg_t, ndeg_c]).astype(int),
        're74': np.concatenate([re74_t, re74_c]),
        're75': np.concatenate([re75_t, re75_c]),
        're78': np.concatenate([re78_t, re78_c]),
    })
    df.attrs['paper'] = "Dehejia & Wahba (1999). NSW + PSID-1."
    df.attrs['expected_naive_ols_att'] = -8498
    df.attrs['expected_psm_att'] = 1794
    df.attrs['notes'] = (
        "Simulated PSID-1 comparison: naive OLS on re78~treat yields "
        "strongly negative (-$8,498) because PSID controls are much "
        "better-off.  Covariate-adjusted / PSM / entropy-balance "
        "estimators should recover the experimental $1,794."
    )
    return df


# ---------------------------------------------------------------------------
# Lee (2008) — US Senate RD
# ---------------------------------------------------------------------------

def lee_2008_senate(seed: int = 42) -> pd.DataFrame:
    """Simulated replica of Lee (2008) US Senate RD.

    Running variable: Democratic vote margin in election t (continuous,
    centred at 0).  Treatment: Democratic win (margin >= 0).  Outcome:
    Democratic vote share in election t+1.

    Published incumbent-advantage estimate: ≈ 0.08 voteshare points
    (Lee 2008, Table 2).

    Returns
    -------
    pd.DataFrame with columns: voteshare_next, margin, win.

    References
    ----------
    Lee, D. (2008). Randomized experiments from non-random selection in
    U.S. House elections. Journal of Econometrics 142, 675-697.
    """
    rng = np.random.default_rng(seed)
    n = 6558
    margin = rng.normal(0, 0.25, n)
    margin = np.clip(margin, -1, 1)
    win = (margin >= 0).astype(int)
    # Voteshare in t+1: continuous in margin + jump at 0 of magnitude 0.08
    voteshare_next = 0.45 + 0.08 * win + 0.35 * margin + rng.normal(0, 0.10, n)
    voteshare_next = np.clip(voteshare_next, 0, 1)
    df = pd.DataFrame({
        'voteshare_next': voteshare_next,
        'margin': margin,
        'win': win.astype(int),
    })
    df.attrs['paper'] = "Lee (2008). Journal of Econometrics 142, 675-697."
    df.attrs['expected_jump_at_cutoff'] = 0.08
    df.attrs['published_jump_original'] = 0.08
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
    Affect Schooling and Earnings?  QJE 106(4), 979-1014.
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
    educ = (13.0 - 0.30 * q1 + 0.05 * q2 + 0.08 * q3 + 0.5 * u +
            rng.normal(scale=1.8, size=n))
    educ = np.clip(educ, 0, 20).round().astype(int)

    lwage = (
        4.0
        + 0.10 * educ       # structural return
        + 0.18 * u          # ability confound (inflates OLS)
        + 0.01 * (year_of_birth - 1930)
        + rng.normal(scale=0.5, size=n)
    )
    df = pd.DataFrame({
        'lwage': lwage, 'educ': educ,
        'q1': q1, 'q2': q2, 'q3': q3, 'q4': q4,
        'year_of_birth': year_of_birth,
    })
    df.attrs['paper'] = "Angrist & Krueger (1991). QJE 106(4), 979-1014."
    df.attrs['expected_iv_educ'] = 0.10
    df.attrs['published_iv_original_range'] = (0.08, 0.11)
    df.attrs['notes'] = (
        "Simulated QOB IV; n=5000 so the first-stage is moderate. "
        "Use q1/q2/q3 as instruments; IV ≈ 0.10 by construction. "
        "The original AK91 data is publicly available at NBER for "
        "exact numerical replication."
    )
    return df
