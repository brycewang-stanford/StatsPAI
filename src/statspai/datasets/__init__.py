"""Canonical econometrics datasets with documented expected estimates.

This subpackage provides deterministic, reproducible datasets used
throughout the causal-inference literature, consolidated under a
single import path ``sp.datasets``:

>>> import statspai as sp
>>> df = sp.datasets.nsw_lalonde()
>>> df.attrs['expected_experimental_att']
1794

Each function returns a ``pd.DataFrame`` with:

- A fully-deterministic DGP (fixed seed, BSD/MIT redistributable).
- ``df.attrs`` containing:
  - ``'paper'`` — original paper citation
  - ``'expected_*'`` — theoretically-anchored estimates (from the
    published paper on the ORIGINAL data, not necessarily the
    simulated replica)
  - ``'notes'`` — what to be careful about when using this replica

The simulated replicas are designed to match the *structure* and
*summary statistics* of the original datasets.  For numerical R /
Stata parity against the original data, see
``tests/external_parity/PUBLISHED_REFERENCE_VALUES.md``.

Available datasets
------------------

DID / panel
    ``mpdta()``                 — Callaway-Sant'Anna teen employment
    ``teen_employment()``       — alias of ``mpdta()``

RD
    ``lee_2008_senate()``       — US Senate RD (Lee 2008)

IV
    ``card_1995()``             — IV returns-to-schooling
    ``angrist_krueger_1991()``  — quarter-of-birth IV

Matching / SOO
    ``nsw_lalonde()``           — LaLonde NSW job training (experimental subset)
    ``nsw_dw()``                — Dehejia-Wahba NSW + PSID comparison

Synthetic control
    ``california_prop99()``     — ADH tobacco (re-exported from synth)
    ``basque_terrorism()``      — Abadie-Gardeazabal (re-exported)
    ``german_reunification()``  — ADH 2015 (re-exported)
"""
from __future__ import annotations

from typing import Dict, Any, List
import pandas as pd

from ._canonical import (
    mpdta,
    card_1995,
    nsw_lalonde,
    nsw_dw,
    lee_2008_senate,
    angrist_krueger_1991,
)

# Re-export synth-shipped datasets (unchanged DGPs; this is the
# consolidated namespace)
from ..synth.datasets import (
    california_tobacco as california_prop99,
    basque_terrorism,
    german_reunification,
)

# Convenience alias
teen_employment = mpdta


def list_datasets() -> pd.DataFrame:
    """Return a DataFrame describing all available datasets.

    Columns: name, design, n_obs, paper, expected_main.
    """
    registry = [
        ('mpdta', 'DID', 2500,
         "Callaway-Sant'Anna (2021)",
         "Simple ATT ≈ -0.040 (teen employment effect of min-wage)"),
        ('card_1995', 'IV', 3010,
         "Card (1995)",
         "IV returns-to-schooling ≈ 0.132 (OLS ≈ 0.075)"),
        ('nsw_lalonde', 'RCT / matching', 445,
         "LaLonde (1986) / Dehejia-Wahba (1999)",
         "Experimental ATT ≈ $1,794 (re78)"),
        ('nsw_dw', 'SOO', 2675,
         "Dehejia-Wahba (1999)",
         "Naive OLS ≈ -$8,498; PSM ≈ $1,794"),
        ('lee_2008_senate', 'RD', 6558,
         "Lee (2008)",
         "Incumbent advantage ≈ 0.08 voteshare points"),
        ('angrist_krueger_1991', 'IV', 5000,
         "Angrist-Krueger (1991)",
         "QOB IV returns-to-schooling ≈ 0.08–0.11"),
        ('california_prop99', 'SCM', 1200,
         "Abadie-Diamond-Hainmueller (2010)",
         "ATT ≈ -15 packs/capita (1988-2000)"),
        ('basque_terrorism', 'SCM', 774,
         "Abadie-Gardeazabal (2003)",
         "GDP gap ≈ -0.855 (mean 1975-1997)"),
        ('german_reunification', 'SCM', 748,
         "Abadie-Diamond-Hainmueller (2015)",
         "West Germany GDPpc gap ≈ -1,500 (post-1990)"),
    ]
    return pd.DataFrame(registry,
                        columns=['name', 'design', 'n_obs',
                                 'paper', 'expected_main'])


__all__ = [
    'mpdta', 'teen_employment',
    'card_1995',
    'nsw_lalonde', 'nsw_dw',
    'lee_2008_senate',
    'angrist_krueger_1991',
    'california_prop99', 'basque_terrorism', 'german_reunification',
    'list_datasets',
]
