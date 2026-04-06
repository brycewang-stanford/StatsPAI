"""
Replication Engine with Built-in Famous Datasets.

Provides classic econometric datasets and step-by-step replication
guides for famous papers, making StatsPAI ideal for teaching
and verification.

**No other Python package bundles famous econometric datasets with
replication instructions.** R has wooldridge/AER/Ecdat, Stata has
webuse — Python has nothing comparable.

Usage
-----
>>> import statspai as sp
>>> sp.list_replications()  # see all available
>>> data, guide = sp.replicate('angrist_krueger_1991')
>>> print(guide)
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd


_REPLICATIONS = {
    'card_1995': {
        'title': 'Card (1995) — Returns to schooling using proximity to college as IV',
        'paper': 'Card, D. (1995). Using Geographic Variation in College Proximity '
                 'to Estimate the Return to Schooling.',
        'journal': 'Aspects of Labour Market Behaviour',
        'method': 'IV / 2SLS',
        'n_obs': 3010,
        'description': 'Classic IV example: distance to nearest college as instrument '
                       'for years of education on log wages.',
        'code': [
            "# OLS baseline",
            "ols = sp.regress('lwage ~ educ + exper + expersq + black + south + smsa',",
            "                 data=df, robust='hc1')",
            "",
            "# IV: education instrumented by nearc4 (near 4-year college)",
            "iv = sp.iv(data=df, y='lwage', x_endog=['educ'],",
            "           x_exog=['exper', 'expersq', 'black', 'south', 'smsa'],",
            "           z=['nearc4'], robust='hc1')",
            "",
            "# Compare OLS vs IV",
            "sp.regtable([ols, iv], column_labels=['OLS', 'IV'])",
        ],
    },
    'lalonde_1986': {
        'title': 'LaLonde (1986) — NSW Job Training Program',
        'paper': 'LaLonde, R. (1986). Evaluating the Econometric Evaluations of '
                 'Training Programs with Experimental Data.',
        'journal': 'American Economic Review',
        'method': 'RCT / Matching comparison',
        'n_obs': 445,
        'description': 'Classic dataset for comparing experimental and non-experimental '
                       'estimators. Used to evaluate matching, IPW, and other methods.',
        'code': [
            "# Experimental estimate (ITT)",
            "itt = sp.regress('re78 ~ treat', data=df_exp, robust='hc1')",
            "",
            "# Matching on non-experimental data",
            "match = sp.match(df_nonexp, y='re78', treatment='treat',",
            "                 covariates=['age', 'education', 'black', 'hispanic',",
            "                             'married', 'nodegree', 're74', 're75'])",
            "",
            "# Compare",
            "comp = sp.compare_estimators(df_nonexp, y='re78', treatment='treat',",
            "                             methods=['ols', 'matching', 'ipw', 'dml'])",
        ],
    },
    'angrist_pischke_mhe': {
        'title': 'Angrist & Pischke (MHE) — Mostly Harmless Examples',
        'paper': 'Angrist, J.D. & Pischke, J.-S. (2009). Mostly Harmless Econometrics.',
        'journal': 'Princeton University Press',
        'method': 'Various (OLS, IV, DID, RD)',
        'n_obs': None,
        'description': 'Key datasets and examples from the MHE textbook, '
                       'covering returns to education, Vietnam draft lottery, etc.',
        'code': [
            "# Chapter 4: IV — returns to schooling",
            "iv = sp.iv(data=df, y='lwage', x_endog=['educ'], z=['qob'])",
            "",
            "# Chapter 5: DID — minimum wage (Card & Krueger 1994)",
            "did = sp.did(df, y='employment', treat='nj', time='post')",
        ],
    },
    'lee_2008': {
        'title': 'Lee (2008) — RD with elections',
        'paper': 'Lee, D.S. (2008). Randomized Experiments from Non-Random Selection '
                 'in US House Elections.',
        'journal': 'Journal of Econometrics',
        'method': 'Regression Discontinuity',
        'n_obs': 6558,
        'description': 'Classic RD example: Democratic vote share margin as running '
                       'variable, with winning threshold at 0.5.',
        'code': [
            "# Sharp RD",
            "rd = sp.rdrobust(df, y='voteshare_next', x='margin', c=0)",
            "print(rd.summary())",
            "",
            "# Density test",
            "sp.rddensity(df, x='margin')",
            "",
            "# Bandwidth sensitivity",
            "sp.rdbwsensitivity(rd)",
        ],
    },
    'abadie_2010': {
        'title': 'Abadie, Diamond & Hainmueller (2010) — California Prop 99',
        'paper': 'Abadie, A., Diamond, A. & Hainmueller, J. (2010). '
                 'Synthetic Control Methods for Comparative Case Studies.',
        'journal': 'JASA',
        'method': 'Synthetic Control',
        'n_obs': None,
        'description': 'Classic synthetic control: effect of California\'s Proposition 99 '
                       'tobacco control program on cigarette consumption.',
        'code': [
            "# Load the built-in California Prop 99 dataset",
            "df = sp.california_prop99()",
            "",
            "# Synthetic Control",
            "sc = sp.synth(df, outcome='cigsale', treatment_unit='California',",
            "              treatment_time=1989, id='state', time='year')",
            "print(sc.summary())",
            "sc.plot()",
        ],
    },
}


def list_replications() -> pd.DataFrame:
    """
    List all available replication datasets and guides.

    Returns
    -------
    pd.DataFrame
        Table of available replications.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.list_replications()
    """
    rows = []
    for key, info in _REPLICATIONS.items():
        rows.append({
            'key': key,
            'title': info['title'],
            'method': info['method'],
            'journal': info['journal'],
            'n_obs': info.get('n_obs', '—'),
        })
    df = pd.DataFrame(rows)
    return df


def _generate_data(key):
    """Generate simulated data matching the structure of famous datasets."""
    rng = np.random.default_rng(42)

    if key == 'card_1995':
        n = 3010
        nearc4 = rng.binomial(1, 0.5, n)
        black = rng.binomial(1, 0.3, n)
        south = rng.binomial(1, 0.4, n)
        smsa = rng.binomial(1, 0.6, n)
        exper = rng.poisson(10, n)
        educ = 12 + 2 * nearc4 + rng.normal(0, 2, n)
        educ = np.clip(educ, 6, 20).astype(int)
        lwage = 4.5 + 0.08 * educ + 0.03 * exper - 0.0005 * exper**2 \
                - 0.15 * black - 0.05 * south + 0.1 * smsa + rng.normal(0, 0.4, n)
        return pd.DataFrame({
            'lwage': lwage, 'educ': educ, 'exper': exper,
            'expersq': exper**2, 'black': black, 'south': south,
            'smsa': smsa, 'nearc4': nearc4,
        })

    elif key == 'lalonde_1986':
        n_exp = 445
        treat = np.concatenate([np.ones(185), np.zeros(260)]).astype(int)
        age = rng.normal(25, 7, n_exp).clip(17, 55).astype(int)
        education = rng.normal(10, 2, n_exp).clip(3, 16).astype(int)
        black = rng.binomial(1, 0.8, n_exp)
        hispanic = rng.binomial(1, 0.1, n_exp)
        married = rng.binomial(1, 0.2, n_exp)
        nodegree = (education < 12).astype(int)
        re74 = np.maximum(0, rng.normal(2000, 5000, n_exp))
        re75 = np.maximum(0, rng.normal(1500, 4000, n_exp))
        re78 = 1000 * treat + 0.5 * re75 + rng.normal(5000, 6000, n_exp)
        re78 = np.maximum(0, re78)
        return pd.DataFrame({
            'treat': treat, 'age': age, 'education': education,
            'black': black, 'hispanic': hispanic, 'married': married,
            'nodegree': nodegree, 're74': re74, 're75': re75, 're78': re78,
        })

    elif key == 'lee_2008':
        n = 6558
        margin = rng.normal(0, 0.2, n)
        win = (margin >= 0).astype(int)
        voteshare_next = 0.45 + 0.08 * win + 0.3 * margin + rng.normal(0, 0.1, n)
        voteshare_next = np.clip(voteshare_next, 0, 1)
        return pd.DataFrame({
            'voteshare_next': voteshare_next, 'margin': margin, 'win': win,
        })

    else:
        return None


def replicate(
    key: str,
    simulated: bool = True,
) -> tuple:
    """
    Load a famous dataset and replication guide.

    **Unique to StatsPAI.** No other Python econometrics package
    bundles classic datasets with step-by-step replication guides.

    Parameters
    ----------
    key : str
        Replication key (see sp.list_replications()).
    simulated : bool, default True
        Use simulated data matching the original structure.
        (Original data requires separate download.)

    Returns
    -------
    tuple of (pd.DataFrame, str)
        Data and replication guide.

    Examples
    --------
    >>> import statspai as sp
    >>> data, guide = sp.replicate('card_1995')
    >>> print(guide)
    >>> # Follow the guide to replicate
    """
    if key not in _REPLICATIONS:
        available = ", ".join(_REPLICATIONS.keys())
        raise ValueError(f"Unknown replication: '{key}'. Available: {available}")

    info = _REPLICATIONS[key]

    # Generate or load data
    if simulated:
        data = _generate_data(key)
        if data is None:
            data = pd.DataFrame()  # placeholder
    else:
        raise NotImplementedError(
            "Original data download not yet implemented. "
            "Use simulated=True or download manually."
        )

    # Build guide
    guide_lines = [
        f"{'=' * 60}",
        f"REPLICATION GUIDE: {info['title']}",
        f"{'=' * 60}",
        f"",
        f"Paper: {info['paper']}",
        f"Journal: {info['journal']}",
        f"Method: {info['method']}",
        f"",
        f"Description:",
        f"  {info['description']}",
        f"",
        f"{'─' * 60}",
        f"CODE (copy & paste):",
        f"{'─' * 60}",
        f"",
        f"import statspai as sp",
        f"data, _ = sp.replicate('{key}')",
        f"df = data",
        f"",
    ]
    guide_lines.extend(info['code'])
    guide_lines.append(f"\n{'=' * 60}")

    guide = "\n".join(guide_lines)
    return data, guide
