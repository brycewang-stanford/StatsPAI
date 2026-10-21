"""Real data for the DiD paper gallery, fetched on demand.

Four of the five datasets behind ``docs/guides/did_paper_gallery.md`` are
not shipped in the wheel (the fifth, the castle-doctrine panel, is:
``castle_doctrine()``). Each loader here downloads its file once from the
canonical public URL, checks a pinned SHA-256, and caches it
(:mod:`statspai.datasets._fetch`). Why each one is fetched rather than
shipped:

==================  ==========================================  =============
loader              source and licence                          size
==================  ==========================================  =============
``jel_medicaid``    pedrohcgs/JEL-DiD (MIT), Baker et al. JEL   6.3 MB, too
                                                                large to ship
``bacon_divorce``   pped.org ``bacon_example.dta`` (Stata       81 kB, no
                    ``bacondecomp`` example); no licence stated licence
``lalonde_panel``   Stata ``drdid`` example ``lalonde.dta``     0.9 MB, no
                    (friosavila.github.io); no licence stated   licence
``mpdta(simulated   Stata ``csdid`` example ``mpdta.dta``, the  67 kB; the R
=False)``           same bytes as R ``did``'s ``mpdta``         copy is GPL-3
==================  ==========================================  =============

Every frame carries ``attrs['data_source'] == 'real'``,
``attrs['simulated'] is False``, the URL, the SHA-256 and the licence note.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ._fetch import RemoteFile, fetch

JEL_MEDICAID = RemoteFile(
    filename="jel_county_mortality_data.csv",
    url=(
        "https://raw.githubusercontent.com/pedrohcgs/JEL-DiD/"
        "affb8f69f6975652445adbc727a50ea04b2c7bd8/data/county_mortality_data.csv"
    ),
    sha256="06beff83aac89236ab851316394574eba973ec762e54a3e98f830ecbb692684f",
    size=6260985,
    licence="MIT (pedrohcgs/JEL-DiD LICENSE, (c) 2025 Baker, Callaway, "
    "Cunningham, Goodman-Bacon and Sant'Anna)",
    origin="pedrohcgs/JEL-DiD data/county_mortality_data.csv at commit affb8f6 "
    "(the 'Public Release' commit, 2025-12-09)",
)

BACON_DIVORCE = RemoteFile(
    filename="bacon_example.dta",
    # pped.org's HTTPS certificate has expired; the pinned SHA-256 is what
    # makes the plain-HTTP download safe to use.
    url="http://pped.org/bacon_example.dta",
    sha256="5a555448e605f75603160be20f56ca343600ec81863f3060994bac1285c29094",
    size=80668,
    licence="no licence stated; fetched, not redistributed",
    origin="Stata bacondecomp help-file example (pped.org/bacon_example.dta)",
)

LALONDE_PANEL = RemoteFile(
    filename="drdid_lalonde.dta",
    url="https://friosavila.github.io/playingwithstata/drdid/lalonde.dta",
    sha256="fec39246a10f1d6430d674cb62b4987473ba07cfc0bbcc68590cfecfd45ecd00",
    size=895309,
    licence="no licence stated; fetched, not redistributed",
    origin="Stata drdid example data (friosavila/playingwithstata, drdid/lalonde.dta)",
)

MPDTA_ORIGINAL = RemoteFile(
    filename="csdid_mpdta.dta",
    url="https://friosavila.github.io/playingwithstata/drdid/mpdta.dta",
    sha256="114e742ed979adb595e5642eda0262f00bd27f3570caa1ceecffa92d898326ae",
    size=66922,
    licence="no licence stated for the .dta; R did's copy is GPL-3; fetched, "
    "not redistributed",
    origin="Stata csdid example data (friosavila/playingwithstata, "
    "drdid/mpdta.dta); values equal R did 2.5.1 mpdta to 4e-14",
)

#: Remote files by loader name (for list_datasets and the gallery tests).
REMOTE = {
    "jel_medicaid": JEL_MEDICAID,
    "bacon_divorce": BACON_DIVORCE,
    "lalonde_panel": LALONDE_PANEL,
    "mpdta_original": MPDTA_ORIGINAL,
}

#: JEL-DiD covariates as the authors' scripts construct them.
JEL_COVARIATES = (
    "perc_female",
    "perc_white",
    "perc_hispanic",
    "unemp_rate",
    "poverty_rate",
    "median_income",
)

#: States that expanded Medicaid before 2014; the JEL scripts drop them.
_JEL_EARLY_STATES = ("DC", "DE", "MA", "NY", "VT")


def _flag(df: pd.DataFrame, spec: RemoteFile, paper: str) -> pd.DataFrame:
    df.attrs["paper"] = paper
    df.attrs["data_source"] = "real"
    df.attrs["simulated"] = False
    df.attrs["source_url"] = spec.url
    df.attrs["sha256"] = spec.sha256
    df.attrs["licence"] = spec.licence
    df.attrs["source_origin"] = spec.origin
    return df


def jel_medicaid(
    sample: Literal["raw", "2x2", "staggered"] = "raw",
    download: bool = True,
) -> pd.DataFrame:
    """County mortality and the ACA Medicaid expansion (Baker et al., JEL) — real data.

    The running example of Baker, Callaway, Cunningham, Goodman-Bacon and
    Sant'Anna's practitioner's guide to difference-in-differences
    [@baker2026difference]: US counties 2009-2019, the outcome is the
    crude mortality rate of adults aged 20-64 per 100,000. The file is
    fetched from the authors' replication repository on first use
    (6.3 MB, MIT licence) and cached; see :mod:`statspai.datasets._fetch`.

    Parameters
    ----------
    sample : {'raw', '2x2', 'staggered'}, default 'raw'
        ``'raw'`` is the published file unchanged (31,843 rows).
        ``'2x2'`` and ``'staggered'`` apply the cleaning of the authors'
        scripts: drop the five states that expanded before 2014 (DC, DE,
        MA, NY, VT), build the covariates (``perc_female``, ``perc_white``,
        ``perc_hispanic`` and ``unemp_rate`` in percent, ``median_income``
        in thousands), drop counties with a missing value, keep counties
        observed in all 11 years, and add ``set_wt`` (2013 population aged
        20-64, the paper's weight) and ``treat_year`` (expansion year, 0 =
        not expanded by 2019). ``'2x2'`` keeps 2014 expanders and counties
        not expanded by 2019 (2,200 counties, 24,200 rows: Tables 2-7 and
        the 2 x T event study); ``'staggered'`` keeps every cohort (2014,
        2015, 2016, 2019 and never; 2,604 counties).
    download : bool, default True
        Download the file when it is not cached. With ``False`` a missing
        file raises instead of touching the network.

    Returns
    -------
    pd.DataFrame
        ``attrs`` carries the source URL, the pinned SHA-256 and the licence.

    Raises
    ------
    DataInsufficient
        Offline and not cached, or the download does not match the pinned
        SHA-256. The message gives the URL and the path to save it at.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.jel_medicaid(sample="2x2")  # doctest: +SKIP
    >>> df.county_code.nunique()  # doctest: +SKIP
    2200

    References
    ----------
    [@baker2026difference]
    """
    if sample not in ("raw", "2x2", "staggered"):
        from ..exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"jel_medicaid: sample must be 'raw', '2x2' or 'staggered', got {sample!r}",
            recovery_hint="Use sample='2x2' for the 2014-vs-never design, "
            "'staggered' for all cohorts, 'raw' for the file as published.",
            diagnostics={"sample": sample},
        )
    path = fetch(JEL_MEDICAID, download=download)
    df = pd.read_csv(path)
    paper = (
        "Baker, Callaway, Cunningham, Goodman-Bacon & Sant'Anna, "
        "'Difference-in-Differences Designs: A Practitioner's Guide', "
        "Journal of Economic Literature."
    )
    if sample == "raw":
        return _flag(df, JEL_MEDICAID, paper)

    df = df.copy()
    df["state_abbr"] = df["county"].str[-2:]
    df = df[~df["state_abbr"].isin(_JEL_EARLY_STATES)]
    if sample == "2x2":
        df = df[(df["yaca"] == 2014) | df["yaca"].isna() | (df["yaca"] > 2019)]
    df = df.assign(
        perc_white=df["population_20_64_white"] / df["population_20_64"] * 100,
        perc_hispanic=df["population_20_64_hispanic"] / df["population_20_64"] * 100,
        perc_female=df["population_20_64_female"] / df["population_20_64"] * 100,
        unemp_rate=df["unemp_rate"] * 100,
        median_income=df["median_income"] / 1000,
    )
    cols = [
        "state_abbr",
        "county",
        "county_code",
        "year",
        "population_20_64",
        "yaca",
        "crude_rate_20_64",
        *JEL_COVARIATES,
    ]
    df = df[cols].dropna(subset=[c for c in cols if c != "yaca"])
    has_1314 = df.groupby("county_code")["year"].transform(
        lambda s: int(s.isin([2013, 2014]).sum()) == 2
    )
    df = df[has_1314.astype(bool)]
    df = df[df.groupby("county_code")["year"].transform("size") == 11]
    w13 = df[df["year"] == 2013].set_index("county_code")["population_20_64"]
    df = df.assign(set_wt=df["county_code"].map(w13))
    expanded = df["yaca"].notna() & (df["yaca"] <= 2019)
    df["treat_year"] = np.where(expanded, df["yaca"], 0).astype(int)
    if sample == "2x2":
        df["treat_year"] = np.where(df["yaca"] == 2014, 2014, 0)
    df["county_code"] = df["county_code"].astype(int)
    df = df.rename(columns={"state_abbr": "state"}).reset_index(drop=True)
    df.attrs["sample"] = sample
    return _flag(df, JEL_MEDICAID, paper)


def bacon_divorce(download: bool = True) -> pd.DataFrame:
    """Unilateral divorce reform and female suicide, 1964-1996 — real data.

    The application of Goodman-Bacon (2021) [@goodmanbacon2021difference]
    and the example data of Stata's ``bacondecomp``
    [@goodmanbacon2019bacondecomp]: 49 states x 33 years (1,617 rows).
    ``asmrs`` is the age-adjusted female suicide rate per million and
    ``post`` the reform indicator. Twelve reform cohorts (1969-1985), 5
    never-reformed states and 8 states that reformed before 1964 (treated
    in every year of the panel). Fetched on first use (81 kB) and cached;
    no licence is stated at the source, so it is not shipped.

    Parameters
    ----------
    download : bool, default True
        Download the file when it is not cached.

    Returns
    -------
    pd.DataFrame
        The published columns (``stfips``, ``year``, ``_nfd``, ``post``,
        ``asmrs``, ``pcinc``, ``asmrh``, ``cases``, ``weight``, ``copop``)
        in float64 / int64, plus ``first_treat``: the first year with
        ``post == 1`` (0 = never reformed; 1964 marks the 8 states already
        treated when the panel starts).

    Raises
    ------
    DataInsufficient
        Offline and not cached, or a SHA-256 mismatch.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.bacon_divorce()  # doctest: +SKIP
    >>> bd = sp.bacon_decomposition(  # doctest: +SKIP
    ...     df, y="asmrs", treat="post", time="year", id="stfips"
    ... )
    >>> round(bd["beta_twfe"], 4)  # doctest: +SKIP
    -3.0799
    """
    path = fetch(BACON_DIVORCE, download=download)
    df = pd.read_stata(path)
    for col in ("stfips", "year", "post"):
        df[col] = df[col].astype("int64")
    for col in df.columns:
        if col not in ("stfips", "year", "post"):
            df[col] = df[col].astype("float64")
    first = df[df["post"] == 1].groupby("stfips")["year"].min()
    df["first_treat"] = df["stfips"].map(first).fillna(0).astype("int64")
    return _flag(
        df,
        BACON_DIVORCE,
        "Goodman-Bacon (2021), 'Difference-in-differences with variation in "
        "treatment timing', Journal of Econometrics 225(2), Section 6.",
    )


def lalonde_panel(download: bool = True) -> pd.DataFrame:
    """NSW experiment with CPS and PSID comparison groups, 1975 and 1978 — real data.

    The long panel behind Sant'Anna and Zhao (2020) [@santanna2020doubly],
    Table 3, and the example data of Stata's ``drdid``: 19,204 people
    observed in 1975 and 1978 (38,408 rows). ``sample`` is 1 for the NSW
    experiment (LaLonde sample, 722 people), 2 for CPS (15,992) and 3 for
    PSID (2,490); ``treated`` is the NSW random assignment, ``experimental``
    marks NSW rows, ``dwincl`` the Dehejia-Wahba subsample and
    ``early_ra`` the early-random-assignment subsample; ``re`` is real
    earnings in the row's year. Fetched on first use (0.9 MB) and cached;
    no licence is stated at the source, so it is not shipped.

    Unlike ``nsw_lalonde()`` (a 614-row cross-section from MatchIt, with
    ``re74`` / ``re75`` / ``re78`` columns), this is the two-period panel
    used by the DR-DiD papers.

    Parameters
    ----------
    download : bool, default True
        Download the file when it is not cached.

    Returns
    -------
    pd.DataFrame
        Covariates are float64. The ``.dta`` stores ``age`` and ``educ`` as
        int8, where ``age ** 3`` silently wraps around; the cast removes
        that trap.

    Raises
    ------
    DataInsufficient
        Offline and not cached, or a SHA-256 mismatch.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.lalonde_panel()  # doctest: +SKIP
    >>> nsw_control = (df["sample"] == 1) & (df["treated"] == 0)  # doctest: +SKIP
    >>> d = df[nsw_control | (df["sample"] == 2)].copy()  # doctest: +SKIP
    >>> d["post"] = (d["year"] == 1978).astype(int)  # doctest: +SKIP
    >>> x = ["age", "educ", "black", "married", "nodegree", "hisp", "re74"]
    >>> r = sp.drdid(  # doctest: +SKIP
    ...     d, y="re", treat="experimental", time="post", id="id", covariates=x
    ... )
    >>> round(r.estimate, 2)  # doctest: +SKIP
    -871.33
    """
    path = fetch(LALONDE_PANEL, download=download)
    df = pd.read_stata(path)
    for col in ("id", "year", "sample", "experimental"):
        df[col] = df[col].astype("int64")
    for col in df.columns:
        if col not in ("id", "year", "sample", "experimental"):
            df[col] = df[col].astype("float64")
    return _flag(
        df,
        LALONDE_PANEL,
        "Sant'Anna & Zhao (2020), 'Doubly robust difference-in-differences "
        "estimators', Journal of Econometrics 219(1), Table 3; data after "
        "LaLonde (1986) and Dehejia & Wahba (1999).",
    )


def mpdta_original(download: bool = True) -> pd.DataFrame:
    """The original ``mpdta`` panel (500 counties, 2003-2007), fetched.

    Reached through ``sp.datasets.mpdta(simulated=False)``.
    """
    path = fetch(MPDTA_ORIGINAL, download=download)
    df = pd.read_stata(path)
    for col in ("year", "countyreal", "first_treat", "treat"):
        df[col] = df[col].astype("int64")
    for col in ("lpop", "lemp"):
        df[col] = df[col].astype("float64")
    df = df[["countyreal", "year", "lpop", "lemp", "first_treat", "treat"]]
    df.attrs["expected_simple_att"] = -0.0400
    df.attrs["notes"] = (
        "Original mpdta: ``treat`` is the ever-treated indicator (the "
        "simulated replica's ``treat`` is the post-treatment dummy). R did "
        "and Stata csdid: simple ATT -0.0400, group -0.0310, calendar "
        "-0.0417, dynamic -0.0772 (DR, never-treated, varying base)."
    )
    return _flag(
        df,
        MPDTA_ORIGINAL,
        "Callaway & Sant'Anna (2021), 'Difference-in-Differences with "
        "Multiple Time Periods', Journal of Econometrics 225(2), 200-230.",
    )
