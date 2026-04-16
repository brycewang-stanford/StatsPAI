"""
Canonical synthetic control datasets for teaching and benchmarking.

Three classic datasets from the SCM literature, generated as realistic
simulations that match the structure and key properties of the originals.

Functions
---------
- ``german_reunification()`` -- Abadie, Diamond & Hainmueller (2015)
- ``basque_terrorism()`` -- Abadie & Gardeazabal (2003)
- ``california_tobacco()`` -- Abadie, Diamond & Hainmueller (2010)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def german_reunification() -> pd.DataFrame:
    """
    German reunification dataset (simulated).

    Returns a balanced panel of GDP per capita for 17 OECD countries,
    1960--2003.  West Germany is the treated unit; treatment begins in
    1990 (reunification).

    The simulated trajectories reproduce the key stylised facts:
    Luxembourg has the highest GDP per capita (~40 000), Portugal the
    lowest (~10 000), and all countries share a common upward growth
    trend.  Post-1990, West Germany exhibits an approximately 1 500
    GDP-per-capita decline relative to its synthetic counterfactual.

    References
    ----------
    Abadie, A., Diamond, A. & Hainmueller, J. (2015).
    "Comparative Politics and the Synthetic Control Method."
    *American Journal of Political Science*, 59(2), 495--510.

    Returns
    -------
    pd.DataFrame
        Columns: ``country``, ``year``, ``gdppc``, ``treated``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.synth.german_reunification()
    >>> result = sp.synth.synth(df, y='gdppc', unit='country',
    ...                         time='year', treat_unit='West Germany',
    ...                         treat_time=1990)
    """
    rng = np.random.default_rng(1990)

    countries = [
        "West Germany", "USA", "UK", "Austria", "Belgium",
        "Denmark", "France", "Greece", "Italy", "Japan",
        "Luxembourg", "Netherlands", "Norway", "Portugal",
        "Spain", "Sweden", "Switzerland",
    ]

    years = list(range(1960, 2004))
    treat_year = 1990
    n_countries = len(countries)
    n_years = len(years)

    # Country-specific base GDP per capita circa 1960 (thousands, then *1000)
    base_map = {
        "West Germany": 12000, "USA": 15000, "UK": 11500,
        "Austria": 10500, "Belgium": 11000, "Denmark": 12500,
        "France": 11000, "Greece": 6000, "Italy": 8500,
        "Japan": 5500, "Luxembourg": 16000, "Netherlands": 12000,
        "Norway": 12500, "Portugal": 4000, "Spain": 6500,
        "Sweden": 13000, "Switzerland": 16500,
    }

    # Annual growth rates (percent, roughly calibrated)
    growth_map = {
        "West Germany": 2.8, "USA": 2.3, "UK": 2.2,
        "Austria": 3.1, "Belgium": 2.7, "Denmark": 2.4,
        "France": 2.6, "Greece": 3.5, "Italy": 3.0,
        "Japan": 4.5, "Luxembourg": 3.0, "Netherlands": 2.5,
        "Norway": 2.9, "Portugal": 3.6, "Spain": 3.4,
        "Sweden": 2.3, "Switzerland": 1.8,
    }

    rows: list[dict] = []
    for country in countries:
        base = base_map[country]
        g = growth_map[country] / 100.0
        # Small country-specific noise process
        noise = rng.normal(0, 1, n_years)
        cumulative_noise = np.cumsum(noise) * 80  # persistent shocks

        for j, yr in enumerate(years):
            t = yr - 1960
            gdp = base * (1 + g) ** t + cumulative_noise[j]

            # Treatment effect: ~1500 decline post-1990 for West Germany
            if country == "West Germany" and yr >= treat_year:
                post_t = yr - treat_year
                # Gradual divergence that stabilises around -1500
                gdp -= 1500 * (1 - np.exp(-0.4 * (post_t + 1)))

            rows.append({
                "country": country,
                "year": yr,
                "gdppc": round(max(gdp, 1000), 1),
                "treated": 1 if country == "West Germany" and yr >= treat_year else 0,
            })

    return pd.DataFrame(rows)


def basque_terrorism() -> pd.DataFrame:
    """
    Basque Country terrorism dataset (simulated).

    Returns a balanced panel of GDP per capita (thousands of 1986 USD)
    for 17 Spanish regions, 1955--1997.  The Basque Country is the
    treated unit; treatment begins in 1970 (onset of ETA terrorism).

    The simulated data reproduce the gradual widening of an approximately
    10 % GDP gap between the Basque Country and its synthetic
    counterfactual after 1970.

    References
    ----------
    Abadie, A. & Gardeazabal, J. (2003).
    "The Economic Costs of Conflict: A Case Study of the Basque Country."
    *American Economic Review*, 93(1), 113--132.

    Returns
    -------
    pd.DataFrame
        Columns: ``region``, ``year``, ``gdppc``, ``treated``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.synth.basque_terrorism()
    >>> result = sp.synth.synth(df, y='gdppc', unit='region',
    ...                         time='year', treat_unit='Basque Country',
    ...                         treat_time=1970)
    """
    rng = np.random.default_rng(1970)

    regions = [
        "Basque Country", "Cataluna", "Madrid", "Andalucia",
        "Aragon", "Asturias", "Baleares", "Canarias",
        "Cantabria", "Castilla y Leon", "Castilla-La Mancha",
        "Extremadura", "Galicia", "Murcia", "Navarra",
        "Rioja", "Valencia",
    ]

    years = list(range(1955, 1998))
    treat_year = 1970
    n_regions = len(regions)
    n_years = len(years)

    # Base GDP per capita circa 1955 (thousands of 1986 USD)
    base_map = {
        "Basque Country": 3.8, "Cataluna": 3.6, "Madrid": 4.0,
        "Andalucia": 2.0, "Aragon": 2.8, "Asturias": 2.9,
        "Baleares": 3.2, "Canarias": 2.3, "Cantabria": 2.7,
        "Castilla y Leon": 2.2, "Castilla-La Mancha": 1.8,
        "Extremadura": 1.5, "Galicia": 2.0, "Murcia": 2.1,
        "Navarra": 3.0, "Rioja": 2.8, "Valencia": 2.7,
    }

    # Growth rates (percent per year, roughly calibrated to Spanish miracle)
    growth_map = {
        "Basque Country": 4.0, "Cataluna": 3.9, "Madrid": 4.2,
        "Andalucia": 3.5, "Aragon": 3.7, "Asturias": 3.3,
        "Baleares": 4.1, "Canarias": 3.8, "Cantabria": 3.5,
        "Castilla y Leon": 3.4, "Castilla-La Mancha": 3.3,
        "Extremadura": 3.2, "Galicia": 3.4, "Murcia": 3.6,
        "Navarra": 3.9, "Rioja": 3.7, "Valencia": 3.8,
    }

    rows: list[dict] = []
    for region in regions:
        base = base_map[region]
        g = growth_map[region] / 100.0
        noise = rng.normal(0, 1, n_years)
        cumulative_noise = np.cumsum(noise) * 0.04

        for j, yr in enumerate(years):
            t = yr - 1955
            gdp = base * (1 + g) ** t + cumulative_noise[j]

            # Treatment effect: gradual ~10% GDP gap post-1970
            if region == "Basque Country" and yr >= treat_year:
                post_t = yr - treat_year
                counterfactual = base * (1 + g) ** t
                # Slowly widening gap that reaches ~10% of counterfactual
                gap_frac = 0.10 * (1 - np.exp(-0.08 * (post_t + 1)))
                gdp -= counterfactual * gap_frac

            rows.append({
                "region": region,
                "year": yr,
                "gdppc": round(max(gdp, 0.5), 3),
                "treated": (
                    1 if region == "Basque Country" and yr >= treat_year else 0
                ),
            })

    return pd.DataFrame(rows)


def california_tobacco() -> pd.DataFrame:
    """
    California Proposition 99 tobacco dataset (simulated, extended).

    Returns a balanced panel of per-capita cigarette sales and covariates
    for 39 US states, 1970--2000.  California is the treated unit;
    treatment begins in 1989 (Proposition 99).

    This dataset extends the simpler ``california_prop99()`` panel with
    additional covariates (retail price, log income, youth population
    share, beer consumption), enabling covariate-matching SCM analyses.

    References
    ----------
    Abadie, A., Diamond, A. & Hainmueller, J. (2010).
    "Synthetic Control Methods for Comparative Case Studies: Estimating
    the Effect of California's Tobacco Control Program."
    *Journal of the American Statistical Association*, 105(490), 493--505.

    Returns
    -------
    pd.DataFrame
        Columns: ``state``, ``year``, ``cigsale``, ``retprice``,
        ``lnincome``, ``age15to24``, ``beer``, ``treated``.

    Notes
    -----
    - ``cigsale`` : per-capita cigarette sales (packs).
    - ``retprice`` : average retail price per pack (cents, real).
    - ``lnincome`` : log of real per-capita personal income.
    - ``age15to24`` : share of population aged 15--24 (percent).
    - ``beer`` : per-capita beer consumption (gallons).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.synth.california_tobacco()
    >>> result = sp.synth.synth(df, y='cigsale', unit='state',
    ...                         time='year', treat_unit='California',
    ...                         treat_time=1989)
    """
    rng = np.random.default_rng(1989)

    states = [
        "Alabama", "Arkansas", "Colorado", "Connecticut", "Delaware",
        "Georgia", "Idaho", "Illinois", "Indiana", "Iowa",
        "Kansas", "Kentucky", "Louisiana", "Maine", "Minnesota",
        "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada",
        "New Hampshire", "New Mexico", "North Carolina", "North Dakota",
        "Ohio", "Oklahoma", "Pennsylvania", "Rhode Island",
        "South Carolina", "South Dakota", "Tennessee", "Texas",
        "Utah", "Vermont", "Virginia", "West Virginia",
        "Wisconsin", "Wyoming", "California",
    ]

    years = list(range(1970, 2001))
    treat_year = 1989
    n_states = len(states)
    n_years = len(years)

    # --- Cigarette sales ---
    # Base levels circa 1970: ~100-160 packs, national average ~130
    base_cig = rng.uniform(95, 165, n_states)
    # National declining trend (packs per year)
    nat_trend_cig = -2.0
    state_trend_cig = rng.normal(0, 0.25, n_states)

    # --- Retail price ---
    # Base circa 1970: ~35-55 cents per pack
    base_price = rng.uniform(35, 55, n_states)
    nat_trend_price = 2.5  # rising prices over time (cents/year)
    state_trend_price = rng.normal(0, 0.3, n_states)

    # --- Log income ---
    # Base circa 1970: ~9.2-9.8 (log of ~10k-18k 1982 dollars)
    base_lninc = rng.uniform(9.2, 9.8, n_states)
    nat_trend_lninc = 0.018  # ~1.8% real income growth per year

    # --- Age 15-24 share ---
    # Baby boom peak in early 1980s, then decline
    base_youth = rng.uniform(16.0, 20.0, n_states)

    # --- Beer consumption ---
    # Base circa 1970: ~22-32 gallons per capita
    base_beer = rng.uniform(22, 32, n_states)
    nat_trend_beer = -0.15  # slight decline

    rows: list[dict] = []
    for i, st in enumerate(states):
        # State-specific persistent shocks
        cig_noise = rng.normal(0, 1, n_years)
        cig_cum = np.cumsum(cig_noise) * 1.0

        price_noise = rng.normal(0, 1, n_years)
        price_cum = np.cumsum(price_noise) * 0.8

        inc_noise = rng.normal(0, 0.003, n_years)
        inc_cum = np.cumsum(inc_noise)

        for j, yr in enumerate(years):
            t = yr - 1970

            # Cigarette sales
            cig = (
                base_cig[i]
                + (nat_trend_cig + state_trend_cig[i]) * t
                + cig_cum[j]
            )

            # California treatment: ~25 pack additional decline post-1989
            if st == "California" and yr >= treat_year:
                cig -= 25 * (1 - np.exp(-0.3 * (yr - treat_year + 1)))

            # Retail price
            price = (
                base_price[i]
                + (nat_trend_price + state_trend_price[i]) * t
                + price_cum[j]
            )

            # Log income
            lninc = base_lninc[i] + nat_trend_lninc * t + inc_cum[j]

            # Youth share: hump around 1980, then decline
            youth = (
                base_youth[i]
                + 2.5 * np.exp(-0.5 * ((yr - 1980) / 5) ** 2)
                - 0.05 * t
                + rng.normal(0, 0.15)
            )

            # Beer consumption
            beer = (
                base_beer[i]
                + nat_trend_beer * t
                + rng.normal(0, 0.4)
            )

            rows.append({
                "state": st,
                "year": yr,
                "cigsale": round(max(cig, 5.0), 1),
                "retprice": round(max(price, 20.0), 1),
                "lnincome": round(lninc, 4),
                "age15to24": round(max(youth, 8.0), 2),
                "beer": round(max(beer, 10.0), 2),
                "treated": 1 if st == "California" and yr >= treat_year else 0,
            })

    return pd.DataFrame(rows)
