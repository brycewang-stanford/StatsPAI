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
The one exception is ``nsw_dw()``, whose bare call is the replica
(parity fixtures are built on it); ``nsw_dw(simulated=False)`` is the
real sample.

Shipped datasets never touch the network. A few real datasets that are
too large for the wheel, or whose licence does not clearly allow
redistribution, are *fetched*: the first call downloads the file from
its canonical public URL, checks a pinned SHA-256 and caches it under
``$STATSPAI_DATA_DIR`` (default ``~/.cache/statspai/datasets``); later
calls work offline. ``list_datasets()['delivery']`` says which is which.
Nothing downloads at import time.

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
                                  (replica; ``simulated=False`` fetches
                                  the original)
    ``teen_employment()``       — alias of ``mpdta()``
    ``castle_doctrine()``       — Cheng-Hoekstra castle-doctrine panel (real)
    ``bacon_divorce()``         — Goodman-Bacon divorce-reform panel
                                  (real, fetched)
    ``jel_medicaid()``          — Baker et al. JEL county mortality
                                  (real, fetched)
    ``lalonde_panel()``         — NSW + CPS + PSID 1975/1978 panel of
                                  Sant'Anna-Zhao (real, fetched)

Structural / dynamic discrete choice
    ``rust_bus()``              — Rust (1987) bus engine replacements
                                  (real, fetched)

RD
    ``lee_2008_senate()``       — US Senate RD (Lee 2008 design; CFT 2015 data)

IV
    ``card_1995()``             — IV returns-to-schooling
    ``angrist_krueger_1991()``  — quarter-of-birth IV

Matching / SOO
    ``nsw_lalonde()``           — LaLonde NSW job training (real MatchIt
                                  extract n=614)
    ``nsw_dw()``                — simulated replica of the Dehejia-Wahba
                                  NSW + PSID comparison; the real sample
                                  with ``simulated=False``

Synthetic control
    ``california_prop99()``     — ADH tobacco (re-exported from synth)
    ``basque_terrorism()``      — Abadie-Gardeazabal (re-exported)
    ``german_reunification()``  — ADH 2015 (re-exported)

Heterogeneous effects in panels (simulated)
    ``currency_union_panel()``  — dyadic trade panel with staggered
                                  currency-union adoption and known
                                  pair-level effects (layout after
                                  Aytug 2026)

Public health / epidemiology (REAL data)
    ``nhefs()``                 — Hernán-Robins *What If* NHEFS (g-methods
                                  canon: quit-smoking → weight / mortality)
    ``load_nhefs()``            — alias of ``nhefs()``
"""

from __future__ import annotations

import functools
import inspect
import warnings
from typing import Any, Callable, Optional

import pandas as pd

# Re-export synth-shipped datasets (unchanged DGPs; this is the
# consolidated namespace)
from ..synth.datasets import basque_terrorism
from ..synth.datasets import california_tobacco as _california_tobacco_simulated
from ..synth.datasets import german_reunification
from ._canonical import (
    SASP_COVARIATES,
    SASP_TIME_INVARIANT,
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
    sasp_panel,
    texas_prison,
    thornton_hiv,
)
from ._currency_union import currency_union_panel
from ._did_gallery import REMOTE as _DID_REMOTE
from ._did_gallery import bacon_divorce, jel_medicaid, lalonde_panel
from ._fetch import is_cached as _is_cached
from ._rust_bus import COLUMNS as _RUST_BUS_COLUMNS
from ._rust_bus import RUST_BUS as _RUST_BUS
from ._rust_bus import rust_bus

# Data-source ingestion normalisers (World Bank / FRED / OECD-Eurostat SDMX).
# These reshape payloads a data MCP already fetched into tidy StatsPAI frames;
# they do not hit the network.
from .ingest import from_fred, from_sdmx, from_worldbank

#: Remote files by loader name (for list_datasets and the fetch tests).
_REMOTE = {**_DID_REMOTE, "rust_bus": _RUST_BUS}


def california_prop99(simulated: bool = False) -> pd.DataFrame:
    """California Proposition 99 panel (Abadie-Diamond-Hainmueller 2010).

    Parameters
    ----------
    simulated : bool, default False
        If True, return the simulated covariate-rich replica from
        ``synth.california_tobacco`` (39 states × 31 years, 1970-2000,
        ADH-shaped DGP).
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
    "texas_prison": "bundled CSV",
    "sasp_panel": "bundled CSV",
    "thornton_hiv": "bundled CSV",
    "jel_medicaid": "fetched",
    "bacon_divorce": "fetched",
    "lalonde_panel": "fetched",
    "rust_bus": "fetched",
}

#: Loaders whose bare call downloads (then caches) the real file. Their
#: columns are listed here so that ``list_datasets()`` never downloads.
_FETCHED_COLUMNS = {
    "jel_medicaid": [
        "state",
        "stfips",
        "county",
        "county_code",
        "year",
        "year_code",
        "deaths",
        "population_20_64",
        "crude_rate_20_64",
        "population_total",
        "population_20_64_hispanic",
        "population_20_64_female",
        "population_20_64_white",
        "unemployed",
        "labor_force",
        "unemp_rate",
        "poverty_rate",
        "median_income",
        "Expansion Status",
        "Description",
        "yaca",
        "maca",
    ],
    "bacon_divorce": [
        "stfips",
        "year",
        "_nfd",
        "post",
        "asmrs",
        "pcinc",
        "asmrh",
        "cases",
        "weight",
        "copop",
        "first_treat",
    ],
    "lalonde_panel": [
        "id",
        "year",
        "treated",
        "age",
        "educ",
        "black",
        "married",
        "nodegree",
        "dwincl",
        "re74",
        "hisp",
        "early_ra",
        "sample",
        "experimental",
        "re",
    ],
}

_FETCHED_COLUMNS["rust_bus"] = list(_RUST_BUS_COLUMNS)

#: The real variant of a loader whose bare call returns a replica.
_REAL_VARIANT = {
    "mpdta": "mpdta(simulated=False): original panel, fetched",
    "nsw_dw": "nsw_dw(simulated=False): real sample, shipped",
}


def list_datasets() -> pd.DataFrame:
    """Return a DataFrame describing all available datasets.

    Columns: name, design, n_obs, paper, paper_original, expected_main,
    source, simulated, delivery, available_offline, real_variant, columns.

    ``source`` says which variant a bare ``name()`` call returns:
    ``"bundled CSV"`` is the real extract shipped in ``datasets/data/``,
    ``"simulated"`` a deterministic replica calibrated near the published
    estimate, ``"fetched"`` real data downloaded from its public source
    on first use. ``delivery`` is ``"shipped"`` (in the wheel, works
    offline) or ``"fetched"``; ``available_offline`` is True for shipped
    data and, for fetched data, whether a verified copy is already in the
    cache (``$STATSPAI_DATA_DIR``). ``real_variant`` names the call that
    returns the real data when a bare call returns a replica. Listing the
    catalogue never downloads anything.

    - ``paper_original`` is the headline number from the published paper on the
      ORIGINAL data (what readers expect to see).
    - ``expected_main`` is what the canonical estimator recovers from the
      variant a bare ``name()`` call returns (what users will actually
      observe), and it is prefixed ``REAL data:`` wherever that default is
      the bundled extract rather than a replica. For a ``"simulated"`` row
      the two columns differ because the replica is a deterministic DGP
      calibrated to the neighbourhood of the published value, not the
      original data. For a ``"bundled CSV"`` row any gap is an estimator or
      extract difference and is named as such.

    For the strict numerical neighbourhood proofs see
    ``tests/external_parity/test_published_replications.py`` and
    ``tests/external_parity/PUBLISHED_REFERENCE_VALUES.md``.

    Examples
    --------
    >>> import statspai as sp
    >>> table = sp.list_datasets()
    >>> bool(table.set_index("name").loc["mpdta", "simulated"])
    True
    >>> table.set_index("name").loc["card_1995", "source"]
    'bundled CSV'
    """
    registry = [
        # (name, design, n_obs, paper, paper_original, expected_main)
        (
            "mpdta",
            "DID",
            2500,
            "Callaway-Sant'Anna (2021)",
            "Simple ATT ≈ -0.03995 (R did::att_gt on original mpdta)",
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
            "Naive difference -$15,205; experimental $1,794 (DW 1999)",
            "Naive OLS ≈ -$8,387; covariate-adjusted ≈ $2,313 on replica; "
            "simulated=False returns the real sample",
        ),
        (
            "lee_2008_senate",
            "RD",
            1390,
            "Lee (2008)",
            "Incumbent advantage ≈ 0.077 voteshare pts (Table 4)",
            "REAL data: conventional 7.414, robust 7.507 pp = R rdrobust to ~1e-12",
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
            "REAL data: SDID -15.60 (= Stata sdid, -15.60383); default "
            "sp.synth classic SCM -19.51. (-17.90, the R synthdid parity "
            "number, is on the simulated sp.california_prop99() panel.)",
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
            "texas_prison",
            "synthetic control (real)",
            816,
            "Cunningham (2021), Mixtape Ch. 10",
            "Texas 1993 prison expansion ~doubles Black male incarceration",
            "REAL data: mean 1994-2000 gap 23,779 vs Stata synth 23,074 (~3%)",
        ),
        (
            "sasp_panel",
            "panel FE (real)",
            1787,
            "Cunningham & Kendall (2011), JUE 69(3)",
            "Within-provider unsafe-sex premium; Mixtape Ch. 8",
            "REAL data: pooled 0.0134 vs within 0.0510 = Stata to ~5e-10",
        ),
        (
            "thornton_hiv",
            "RCT / randomization inference (real)",
            4820,
            "Thornton (2008), AER 98(5)",
            "Cash incentive raises HIV-result collection by ~45 pts",
            "REAL data: SDO 0.450552 = Stata exactly; RI p = 0",
        ),
        (
            "nhefs",
            "g-methods (real)",
            1629,
            "Hernán & Robins (2020), Causal Inference: What If",
            "Quit-smoking IP-weighted ATT ≈ 3.4 kg, 95% CI (2.4, 4.5) (Ch12)",
            "REAL data: StatsPAI reproduces 3.4-3.5 kg across Ch12-14 g-methods",
        ),
        (
            "currency_union_panel",
            "panel HTE / causal forest with FE (simulated dyadic)",
            2205,
            "Aytug (2026), arXiv:2601.19664 (layout only)",
            "CFFE euro trade ATT 0.133 log points (+14.2%), Table 6, on "
            "Eurostat EU15 data (not used here)",
            "Simulated: true ATT on treated rows 0.133 by construction "
            "(tau_true); FE-forest imputation ATT with controls='auto' "
            "recovers it, the pooled forest is biased upward",
        ),
        (
            "jel_medicaid",
            "DID 2x2 / staggered (real, fetched)",
            31843,
            "Baker, Callaway, Cunningham, Goodman-Bacon & Sant'Anna, JEL",
            "Weighted 2x2 -2.56 (Table 5 col 4); DR with 2013 covariates "
            "-3.76 (Table 7); TWFE with time-varying X -1.37 (Table 5 col 6)",
            "REAL data: -2.56 / -3.76 / -1.37 reproduced (= Stata to 1e-5); "
            "sample='2x2' / 'staggered' apply the authors' cleaning",
        ),
        (
            "bacon_divorce",
            "staggered DID / Bacon decomposition (real, fetched)",
            1617,
            "Goodman-Bacon (2021), JoE 225(2), Section 6",
            "TWFE -3.08; 156 2x2 comparisons; later-vs-always weight 0.38",
            "REAL data: TWFE -3.0799 and all 156 2x2s = Stata bacondecomp",
        ),
        (
            "lalonde_panel",
            "DR-DID panel (real, fetched)",
            38408,
            "Sant'Anna & Zhao (2020), JoE 219(1), Table 3",
            "Evaluation bias, LaLonde sample, linear X: DR -871, improved "
            "DR -901, OR -1301 (truth 0)",
            "REAL data: all 54 cells of Table 3 reproduced (= Stata drdid)",
        ),
        (
            "rust_bus",
            "dynamic discrete choice / NFXP (real, fetched)",
            8260,
            "Rust (1987), Econometrica 55(5), Tables V-X",
            "Groups 1-4, beta=.9999, 90 bins: RC 9.7558 (1.227), "
            "theta_11 2.6275 (0.618), LL -6055.250 (Table IX)",
            "REAL data: sp.ddc with Rust's increments and vce='opg' gives "
            "9.7557 (1.2265), 2.6276 (0.6173); Tables V, VI, VIII-X reproduced",
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
    table["simulated"] = table["source"] == "simulated"
    table["delivery"] = [
        "fetched" if s == "fetched" else "shipped" for s in table["source"]
    ]
    table["available_offline"] = [
        _is_cached(_REMOTE[n]) if n in _REMOTE else True for n in table["name"]
    ]
    table["real_variant"] = table["name"].map(_REAL_VARIANT)
    table["columns"] = [
        (
            list(_FETCHED_COLUMNS[name])
            if name in _FETCHED_COLUMNS
            else list(_quiet_load(name).columns)
        )
        for name in table["name"]
    ]
    return table


# ----------------------------------------------------------------------
# Provenance flags and the simulated-data warning
# ----------------------------------------------------------------------

#: Datasets whose simulated-replica warning this process already issued.
_WARNED: set = set()


def _quiet_load(name: str) -> pd.DataFrame:
    """Load ``name`` for the catalogue, leaving the once-per-process
    warning for the first real use of the data."""
    already = name in _WARNED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = globals()[name]()
    if not already:
        _WARNED.discard(name)
    return df


def _flag_source(
    name: str,
    df: pd.DataFrame,
    simulated_arg: Any,
    call: Optional[str] = None,
    real_alternative: Optional[str] = None,
) -> pd.DataFrame:
    """Write ``attrs['simulated']`` / ``attrs['data_source']`` and warn once.

    Several replicas carry the names of famous real datasets (``mpdta``,
    ``nsw_dw``, ``basque_terrorism``). An estimate on them is a property
    of the simulation, and an agent that reads only the name reports it
    as a finding about the real data. The flag is on every frame; the
    warning fires the first time a process loads each replica.
    """
    sim = df.attrs.get("simulated")
    if sim is None:
        if isinstance(simulated_arg, bool):
            sim = simulated_arg
        else:
            sim = _DEFAULT_SOURCE.get(name) != "bundled CSV"
    df.attrs["simulated"] = bool(sim)
    df.attrs.setdefault("data_source", "simulated" if sim else "real")
    key = call or name
    if sim and key not in _WARNED:
        _WARNED.add(key)
        from ..exceptions import StatsPAIWarning

        paper = df.attrs.get("paper") or ""
        about = f" ({paper})" if isinstance(paper, str) and paper else ""
        real = f" The real data: {real_alternative}." if real_alternative else ""
        warnings.warn(
            f"{call or f'sp.datasets.{name}'}() returns a SIMULATED "
            f"replica{about}, not the original data: estimates on it describe "
            "the simulation and must not be reported as findings about the "
            f"real data.{real} df.attrs['simulated'] is True; "
            "sp.list_datasets() shows which datasets are real extracts.",
            StatsPAIWarning,
            stacklevel=3,
        )
    return df


def _flagged(
    name: str,
    fn: Callable[..., pd.DataFrame],
    call: Optional[str] = None,
    real_alternative: Optional[str] = None,
) -> Callable[..., pd.DataFrame]:
    sig = inspect.signature(fn)

    @functools.wraps(fn)
    def loader(*args: Any, **kwargs: Any) -> pd.DataFrame:
        df = fn(*args, **kwargs)
        simulated_arg = None
        if "simulated" in sig.parameters:
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            simulated_arg = bound.arguments.get("simulated")
        return _flag_source(name, df, simulated_arg, call, real_alternative)

    return loader


_LOADERS = (
    "mpdta",
    "card_1995",
    "nsw_lalonde",
    "nsw_dw",
    "lee_2008_senate",
    "angrist_krueger_1991",
    "castle_doctrine",
    "texas_prison",
    "sasp_panel",
    "thornton_hiv",
    "california_prop99",
    "basque_terrorism",
    "german_reunification",
    "nhefs",
    "currency_union_panel",
    "jel_medicaid",
    "bacon_divorce",
    "lalonde_panel",
    "rust_bus",
)
for _name in _LOADERS:
    globals()[_name] = _flagged(
        _name,
        globals()[_name],
        real_alternative=(
            "sp.datasets.mpdta(simulated=False) downloads the original panel"
            if _name == "mpdta"
            else None
        ),
    )
teen_employment = mpdta
load_nhefs = nhefs
del _name


def load_dataset(name: str, **kwargs: Any) -> pd.DataFrame:
    """Load a bundled example dataset by name.

    Parameters
    ----------
    name : str
        One of the names in :func:`list_datasets` (``"mpdta"``,
        ``"card_1995"``, ``"nsw_dw"``, ``"lee_2008_senate"``, ...).
    **kwargs
        Passed to the dataset's loader (e.g. ``simulated=False`` for
        ``nsw_dw`` or ``mpdta``, ``seed=`` for the simulated replicas,
        ``download=False`` to refuse a download for a fetched dataset).

    Returns
    -------
    pd.DataFrame
        ``df.attrs['simulated']`` says whether it is a simulated replica
        or the real extract; ``df.attrs['paper']`` names the source study.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.load_dataset("card_1995")
    >>> df.attrs["simulated"]
    False
    """
    if name not in _LOADERS and name not in ("teen_employment", "load_nhefs"):
        import difflib

        from ..exceptions import MethodIncompatibility

        close = difflib.get_close_matches(str(name), _LOADERS, n=3, cutoff=0.5)
        hint = f"Did you mean {close[0]!r}? " if close else ""
        raise MethodIncompatibility(
            f"Unknown dataset {name!r}.",
            recovery_hint=hint + "sp.list_datasets() lists the available names.",
            diagnostics={
                "name": name,
                "available": list(_LOADERS),
                "did_you_mean": close,
            },
            alternative_functions=["sp.list_datasets"],
        )
    return globals()[name](**kwargs)


__all__ = [
    "mpdta",
    "teen_employment",
    "card_1995",
    "nsw_lalonde",
    "nsw_dw",
    "lee_2008_senate",
    "angrist_krueger_1991",
    "castle_doctrine",
    "texas_prison",
    "sasp_panel",
    "SASP_COVARIATES",
    "SASP_TIME_INVARIANT",
    "thornton_hiv",
    "california_prop99",
    "basque_terrorism",
    "german_reunification",
    "nhefs",
    "load_nhefs",
    "currency_union_panel",
    "jel_medicaid",
    "bacon_divorce",
    "lalonde_panel",
    "rust_bus",
    "list_datasets",
    "load_dataset",
    "from_worldbank",
    "from_fred",
    "from_sdmx",
]
