"""Canonical econometrics datasets with documented expected estimates.

This subpackage provides deterministic, reproducible datasets used
throughout the causal-inference literature, consolidated under a
single import path ``sp.datasets``:

>>> import statspai as sp
>>> df = sp.datasets.nsw_lalonde()          # real extract, no network
>>> df.shape
(614, 11)
>>> sp.datasets.nsw_lalonde(simulated=True).attrs['expected_experimental_att']
1794

**One rule:** if StatsPAI ships the real published data for a dataset,
a bare call returns it. ``simulated=True`` asks for the calibrated
replica instead. ``list_datasets()['source']`` says which you get.
Nothing here touches the network — the whole catalogue works offline.

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
    ``nsw_lalonde()``           — LaLonde NSW job training (real MatchIt
                                  extract n=614)
    ``nsw_dw()``                — Dehejia-Wahba NSW + PSID comparison

Synthetic control
    ``california_prop99()``     — ADH tobacco (re-exported from synth)
    ``basque_terrorism()``      — Abadie-Gardeazabal (re-exported)
    ``german_reunification()``  — ADH 2015 (re-exported)

Public health / epidemiology (REAL data)
    ``nhefs()``                 — Hernán-Robins *What If* NHEFS (g-methods
                                  canon: quit-smoking → weight / mortality)
    ``load_nhefs()``            — alias of ``nhefs()``
"""

from __future__ import annotations

import pandas as pd

# Re-export synth-shipped datasets (unchanged DGPs; this is the
# consolidated namespace)
from ..synth.datasets import basque_terrorism
from ..synth.datasets import california_tobacco as _california_tobacco_simulated
from ..synth.datasets import german_reunification
from ._canonical import (
    _load_bundled_csv,
    angrist_krueger_1991,
    card_1995,
    castle_doctrine,
    lee_2008_senate,
    load_nhefs,
    mpdta,
    nhefs,
    nsw_dw,
    nsw_lalonde,
)

# Data-source ingestion normalisers (World Bank / FRED / OECD-Eurostat SDMX).
# These reshape payloads a data MCP already fetched into tidy StatsPAI frames;
# they do not hit the network.
from .ingest import from_fred, from_sdmx, from_worldbank


def california_prop99(simulated: bool = False) -> pd.DataFrame:
    """California Proposition 99 panel (Abadie-Diamond-Hainmueller 2010).

    Parameters
    ----------
    simulated : bool, default True
        If True, return the simulated covariate-rich replica from
        ``synth.california_tobacco`` (39 states × 31 years, 1970-2000,
        ADH-shaped DGP).  Default for backward compatibility.
        If False, load the real ADH (2010) panel bundled in
        ``statspai/datasets/data/california_prop99.csv`` (39 states ×
        31 years, with covariates ``cigsale, retprice, lnincome,
        age15to24, beer``; identical to tidysynth's smoking dataset).
        Use this for exact paper replication.

    Returns
    -------
    pd.DataFrame
        Columns (both branches): ``state, year, cigsale, retprice,
        lnincome, age15to24, beer``.  The simulated branch additionally
        provides ``treated``; on the real branch we derive it as
        ``(state == 'California') & (year >= 1989)``.

    References
    ----------
    Abadie, A., Diamond, A. & Hainmueller, J. (2010).
    Synthetic Control Methods for Comparative Case Studies.
    Journal of the American Statistical Association 105(490), 493-505.
    [@abadie2010synthetic]
    """
    if simulated:
        return _california_tobacco_simulated()

    df = _load_bundled_csv("california_prop99.csv")
    # The bundled real CSV does not carry a 'treated' indicator; derive
    # it so downstream callers (synth, synthdid, plotting) work uniformly.
    if "treated" not in df.columns:
        df = df.copy()
        df["treated"] = ((df["state"] == "California") & (df["year"] >= 1989)).astype(
            int
        )
    df.attrs["paper"] = (
        "Abadie, A., Diamond, A. & Hainmueller, J. (2010). "
        "Synthetic Control Methods for Comparative Case Studies. "
        "JASA 105(490), 493-505."
    )
    df.attrs["data_source"] = "real"
    df.attrs["simulated"] = False
    df.attrs["source_origin"] = (
        "Public-domain ADH (2010) California Prop 99 panel; "
        "byte-identical to tidysynth's smoking dataset (1970-2000)."
    )
    df.attrs["notes"] = (
        "Real ADH panel for exact paper replication.  Use the full "
        "ADH (2010) predictor recipe via sp.synth(method='classic', "
        "special_predictors=...) for canonical numbers; the headline "
        "1989-2000 average gap is roughly -19 packs/capita per ADH "
        "(2010) Figure 2."
    )
    return df


# Convenience alias
teen_employment = mpdta


#: Which variant a bare ``name()`` call returns. Loaders absent from this
#: map return a simulated replica.
_DEFAULT_SOURCE = {
    "card_1995": "bundled CSV",
    "lee_2008_senate": "bundled CSV",
    "california_prop99": "bundled CSV",
    "nsw_lalonde": "bundled CSV",
    "nhefs": "bundled CSV",
    "castle_doctrine": "bundled CSV",
}


def list_datasets() -> pd.DataFrame:
    """Return a DataFrame describing all available datasets.

    Columns: name, design, n_obs, paper, paper_original, expected_main,
    source.

    Every dataset here loads from disk — none of them touches the
    network, so the whole catalogue works offline. ``source`` says which
    variant a bare ``name()`` call returns: ``"bundled CSV"`` is the real
    extract shipped in ``datasets/data/``, ``"simulated"`` a
    deterministic replica calibrated near the published estimate.

    - ``paper_original`` is the headline number from the published paper on the
      ORIGINAL data (what readers expect to see).
    - ``expected_main`` is what the canonical estimator recovers on this
      simulated replica (what users will actually observe). The two differ
      because the bundled replicas are deterministic DGPs calibrated to the
      neighbourhood of the published values, not the original data.

    For the strict numerical neighbourhood proofs see
    ``tests/external_parity/test_published_replications.py`` and
    ``tests/external_parity/PUBLISHED_REFERENCE_VALUES.md``.
    """
    registry = [
        # (name, design, n_obs, paper, paper_original, expected_main)
        (
            "mpdta",
            "DID",
            2500,
            "Callaway-Sant'Anna (2021)",
            "Simple ATT ≈ -0.0454 (R did::att_gt on original mpdta)",
            "Simple ATT ≈ -0.033, dynamic ATT ≈ -0.034 on this replica",
        ),
        (
            "card_1995",
            "IV",
            3010,
            "Card (1995)",
            "IV β_educ ≈ 0.132, OLS ≈ 0.075 (Table 3, NLSYM)",
            "IV β_educ ≈ 0.132, OLS ≈ 0.074 on the real NLSYM extract",
        ),
        (
            "nsw_lalonde",
            "SOO / matching",
            614,
            "LaLonde (1986) / Dehejia-Wahba (1999)",
            "Naive OLS ≈ -$8,498; PSM ≈ $1,794 (DW 1999, full PSID-1)",
            "Naive OLS ≈ -$635; PSM ≈ $1,963 (real MatchIt extract, n=614)",
        ),
        (
            "nsw_dw",
            "SOO",
            2675,
            "Dehejia-Wahba (1999)",
            "Naive OLS ≈ -$8,498; PSM ≈ $1,794 (DW 1999)",
            "Naive OLS ≈ -$8,387; covariate-adjusted ≈ $2,313 on replica",
        ),
        (
            "lee_2008_senate",
            "RD",
            1390,
            "Lee (2008)",
            "Incumbent advantage ≈ 0.077 voteshare pts (Table 4)",
            "Conventional ≈ 0.073, CCT robust ≈ 0.062 on this replica",
        ),
        (
            "angrist_krueger_1991",
            "IV",
            5000,
            "Angrist-Krueger (1991)",
            "QOB IV β_educ ≈ 0.08–0.11 (Table V, range)",
            "IV β_educ ≈ 0.10 by construction on this replica",
        ),
        (
            "california_prop99",
            "SCM",
            1209,
            "Abadie-Diamond-Hainmueller (2010)",
            "Mean 1989-2000 ATT ≈ -19 packs/capita (JASA Fig. 2)",
            "Classic ADH ≈ -13.1, ASCM ≈ -13.3 packs/capita on this replica",
        ),
        (
            "basque_terrorism",
            "SCM",
            731,
            "Abadie-Gardeazabal (2003)",
            "GDP gap ≈ -0.855 (mean 1975-1997)",
            "GDP gap ≈ -0.855 on this replica (calibrated)",
        ),
        (
            "german_reunification",
            "SCM",
            748,
            "Abadie-Diamond-Hainmueller (2015)",
            "West Germany GDPpc gap ≈ -1,500 (post-1990)",
            "GDPpc gap ≈ -1,500 on this replica (calibrated)",
        ),
        (
            "castle_doctrine",
            "staggered DID (real)",
            550,
            "Cheng & Hoekstra (2013), JHR 48(3)",
            "TWFE log-homicide effect ≈ 0.08 (Table 4, weighted, clustered)",
            "REAL data: TWFE 0.0769 = Stata to 1e-9; Callaway-Sant'Anna 0.1104",
        ),
        (
            "nhefs",
            "g-methods (real)",
            1629,
            "Hernán & Robins (2020), Causal Inference: What If",
            "Quit-smoking IP-weighted ATT ≈ 3.4 kg, 95% CI (2.4, 4.5) (Ch12)",
            "REAL data: StatsPAI reproduces 3.4-3.5 kg across Ch12-14 g-methods",
        ),
    ]
    table = pd.DataFrame(
        registry,
        columns=["name", "design", "n_obs", "paper", "paper_original", "expected_main"],
    )
    # What a bare ``name()`` call hands back. Every loader reads from disk
    # — nothing here touches the network — but "bundled CSV" is the real
    # extract shipped in ``datasets/data/``, while "simulated" is a
    # deterministic replica calibrated near the published estimate. Four
    # loaders (card_1995, lee_2008_senate, california_prop99,
    # nsw_lalonde) also accept ``simulated=`` to pick the other one.
    table["source"] = table["name"].map(_DEFAULT_SOURCE).fillna("simulated")
    return table


__all__ = [
    "mpdta",
    "teen_employment",
    "card_1995",
    "nsw_lalonde",
    "nsw_dw",
    "lee_2008_senate",
    "angrist_krueger_1991",
    "castle_doctrine",
    "california_prop99",
    "basque_terrorism",
    "german_reunification",
    "nhefs",
    "load_nhefs",
    "list_datasets",
    "from_worldbank",
    "from_fred",
    "from_sdmx",
]
