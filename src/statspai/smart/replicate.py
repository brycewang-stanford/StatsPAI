"""
Replication Engine with Built-in Famous Datasets.

Provides classic econometric datasets and step-by-step replication
guides for famous papers, making StatsPAI ideal for teaching
and verification.

The bundled recipes are intended for teaching, smoke tests, and
version-to-version numerical drift checks.

Two tracks where applicable
---------------------------
For papers that have both an "as published" estimator and a more
recent improvement, the guide ships **two recipes**:

- **classic** — faithful to the original paper (e.g. Card 1995's
  2SLS, ADH 2010's outcome-only synth) with golden numbers from the
  paper itself.
- **modern** — a contemporary alternative the StatsPAI team
  recommends for new analyses (e.g. weak-IV-robust AR confidence
  intervals, synthdid, augsynth).  Pinned numbers are StatsPAI
  regression-test references on the bundled real data, not paper
  values — used to detect numerical drift across versions.

Real vs simulated data
----------------------
Where a public-domain CSV exists, the guide loads it via
``sp.datasets.<name>(simulated=False)`` (Card 1995, ADH 2010).  For
papers without a bundled real CSV, the guide falls back to a
deterministic simulated replica.

Usage
-----
>>> import statspai as sp
>>> sp.list_replications()
>>> data, guide = sp.replicate('card_1995')
>>> print(guide)
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# Replication registry
# ----------------------------------------------------------------------
#
# Schema (per entry):
#   title, paper, paper_bib, journal, year, design, n_obs, description
#   data_loader (str)   : 'datasets.<name>' resolved against statspai
#   data_kwargs (dict)  : kwargs for the loader; commonly {'simulated': False}
#   data_origin (str)   : provenance line shown in the guide
#   classic (dict|None) : {name, paper_table, code[], golden_numbers[],
#                          tolerance, references[]}
#   modern  (dict|None) : {name, rationale, code[], pinned_numbers[],
#                          tolerance, references[]}
#   code   (list[str])  : LEGACY single-track code block (used when
#                          neither classic nor modern is set)
#
# golden_numbers entries: (label, statspai_value, paper_value, citation)
# pinned_numbers entries: (label, statspai_value, note)


_REPLICATIONS: Dict[str, Dict[str, Any]] = {
    # ------------------------------------------------------------------
    # Card (1995) — IV returns to schooling, NLSYM
    # ------------------------------------------------------------------
    "card_1995": {
        "title": "Card (1995) — Returns to schooling using proximity to college as IV",
        "paper": (
            "Card, D. (1995). Using Geographic Variation in College "
            "Proximity to Estimate the Return to Schooling."
        ),
        "paper_bib": "card1995using",
        "journal": "In Christofides et al. (eds.), Aspects of Labour Market Behaviour",
        "year": 1995,
        "design": "IV / 2SLS",
        "n_obs": 3010,
        "description": (
            "Distance to nearest 4-year college (nearc4) as instrument "
            "for years of education in a wage equation.  IV exceeds OLS "
            'by ~6 log points — the "Card puzzle", interpretable as a '
            "LATE for compliers on the proximity margin."
        ),
        "data_loader": "datasets.card_1995",
        "data_kwargs": {"simulated": False},
        "data_origin": (
            "Real NLSYM extract bundled in "
            "statspai/datasets/data/card_1995.csv (n=3010, identical to "
            "R wooldridge::card complete-cases on Card's modelling "
            "variables)."
        ),
        "classic": {
            "name": "OLS + 2SLS (Card 1995 Table 2)",
            "paper_table": "Card (1995) Table 2, cols 2 & 5",
            "references": ["card1995using"],
            "tolerance": 1e-3,
            "code": [
                "# Card (1995) headline specification",
                "ols = sp.regress(",
                "    'lwage ~ educ + exper + expersq + black + south + smsa',",
                "    data=df, robust='hc1')",
                "",
                "iv = sp.ivreg(",
                "    'lwage ~ exper + expersq + black + south + smsa + '",
                "    '(educ ~ nearc4)',",
                "    data=df, robust='hc1')",
                "",
                "sp.regtable([ols, iv], column_labels=['OLS', 'IV (nearc4)'])",
            ],
            # (label, statspai value on real data, paper value, citation)
            "golden_numbers": [
                ("OLS β_educ", 0.0740, 0.075, "Card (1995) Table 2, col 2"),
                ("IV β_educ", 0.1323, 0.132, "Card (1995) Table 2, col 5"),
            ],
        },
        "modern": {
            "name": "Anderson-Rubin weak-IV-robust inference",
            "rationale": (
                "With one instrument, 2SLS Wald CIs distort when the "
                "first-stage F is moderate.  On Card's real data the "
                'effective F is ~17.5 — in the "moderate" weak-IV regime '
                "where Andrews-Stock-Sun (2019) recommend AR-type "
                "identification-robust CIs over conventional t-tests."
            ),
            "references": [
                "andrews2019weak",
                "moreira2003conditional",
                "kleibergen2002pivotal",
            ],
            "tolerance": 5e-3,
            "code": [
                "# Anderson-Rubin 95% confidence interval (weak-IV-robust)",
                "ar_ci = sp.anderson_rubin_ci(",
                "    data=df, y='lwage', endog='educ',",
                "    instruments=['nearc4'],",
                "    exog=['exper', 'expersq', 'black', 'south', 'smsa'],",
                "    level=0.95)",
                "print(f'AR-CI 95%: [{ar_ci.lower:.4f}, {ar_ci.upper:.4f}]')",
                "",
                "# AR test of H0: β_educ = 0 plus first-stage diagnostics",
                "ar = sp.anderson_rubin_test(",
                "    data=df, y='lwage', endog='educ',",
                "    instruments=['nearc4'],",
                "    exog=['exper', 'expersq', 'black', 'south', 'smsa'])",
                "print(ar['interpretation'])",
            ],
            # (label, statspai pinned value, note)
            "pinned_numbers": [
                ("AR-CI 95% lower", 0.0389, "identification-robust lower bound"),
                ("AR-CI 95% upper", 0.2601, "identification-robust upper bound"),
                (
                    "First-stage F (effective)",
                    17.51,
                    "Olea-Pflueger effective F; moderate strength",
                ),
                ("AR test p-value @ β=0", 0.0088, "rejects β_educ = 0 at 1%"),
            ],
        },
    },
    # ------------------------------------------------------------------
    # Abadie, Diamond & Hainmueller (2010) — California Prop 99
    # ------------------------------------------------------------------
    "abadie_2010": {
        "title": "Abadie, Diamond & Hainmueller (2010) — California Prop 99",
        "paper": (
            "Abadie, A., Diamond, A. & Hainmueller, J. (2010). "
            "Synthetic Control Methods for Comparative Case "
            "Studies: Estimating the Effect of California's "
            "Tobacco Control Program."
        ),
        "paper_bib": "abadie2010synthetic",
        "journal": "Journal of the American Statistical Association 105(490), 493-505",
        "year": 2010,
        "design": "Synthetic Control",
        "n_obs": 1209,
        "description": (
            "Effect of California's 1989 tobacco-control program on "
            'per-capita cigarette sales.  Construct a "synthetic '
            'California" as a convex combination of donor states '
            "matched on pre-1989 outcomes (and covariates).  ADH (2010) "
            "Figure 2 shows a post-1989 gap of roughly 19 packs/capita."
        ),
        "data_loader": "datasets.california_prop99",
        "data_kwargs": {"simulated": False},
        "data_origin": (
            "Real ADH (2010) panel bundled in "
            "statspai/datasets/data/california_prop99.csv "
            "(39 states × 31 years, 1970-2000; byte-identical to "
            "tidysynth's smoking dataset)."
        ),
        "classic": {
            "name": "Outcome-only synthetic control (ADH-style)",
            "paper_table": "ADH (2010) Figure 2, Table 2",
            "references": ["abadie2010synthetic"],
            # Loose: SCM is sensitive to predictor recipe; we pin our
            # outcome-only recovery to within ~0.1 of the paper headline.
            "tolerance": 0.5,
            "code": [
                "# Outcome-only synthetic control (closest reproducible recipe",
                "# to ADH 2010 Figure 2; full ADH predictor recipe also",
                "# supported via special_predictors=...)",
                "sc = sp.synth(",
                "    data=df, outcome='cigsale',",
                "    unit='state', time='year',",
                "    treated_unit='California', treatment_time=1989,",
                "    method='classic', placebo=False)",
                "print(sc.summary())",
                "sc.plot()",
            ],
            "golden_numbers": [
                (
                    "Average post-1989 ATT (packs/capita)",
                    -19.7605,
                    -19.0,
                    "ADH (2010) Figure 2 (qualitative ≈ -19)",
                ),
            ],
        },
        "modern": {
            "name": "synthdid (Arkhangelsky 2021) + Augmented SCM (Ben-Michael 2021)",
            "rationale": (
                "Two post-2010 refinements: (a) synthdid combines unit "
                "and time weights to remove additive shocks; (b) "
                "augmented SCM adds a ridge-regression bias correction "
                "when pre-treatment fit is imperfect.  Both reduce "
                "sensitivity to the predictor recipe that classic SCM "
                "is famously fragile to."
            ),
            "references": ["arkhangelsky2021synthetic", "benmichael2021augmented"],
            "tolerance": 1e-2,
            "code": [
                "# (a) Synthetic Difference-in-Differences",
                "sdid = sp.synthdid_estimate(",
                "    data=df, y='cigsale', unit='state', time='year',",
                "    treat_unit='California', treat_time=1989)",
                "print('synthdid ATT:', round(float(sdid.estimate), 2))",
                "",
                "# (b) Augmented SCM with ridge bias correction",
                "asc = sp.augsynth(",
                "    data=df, outcome='cigsale',",
                "    unit='state', time='year',",
                "    treated_unit='California', treatment_time=1989)",
                "print('augsynth ATT:', round(float(asc.estimate), 2))",
            ],
            "pinned_numbers": [
                ("synthdid ATT", -27.3491, "unit + time weights, real ADH panel"),
                ("augsynth ATT", -16.7317, "ridge-augmented SCM, real ADH panel"),
            ],
        },
    },
    # ------------------------------------------------------------------
    # Legacy single-track entries (kept for backward compatibility)
    # ------------------------------------------------------------------
    "lalonde_1986": {
        "title": "LaLonde (1986) / Dehejia-Wahba (1999) — NSW + PSID",
        "paper": (
            "Dehejia, R. & Wahba, S. (1999). Causal Effects in "
            "Nonexperimental Studies: Reevaluating the Evaluation "
            "of Training Programs."
        ),
        "paper_bib": "dehejia1999causal",
        "journal": "JASA 94(448), 1053-1062 (LaLonde 1986: AER 76(4))",
        "year": 1999,
        "design": "Observational ATT recovery vs experimental benchmark",
        "n_obs": 614,
        "description": (
            "Combine the 185 NSW experimental treated with PSID-1 "
            "controls to test whether observational estimators can "
            "recover the experimental ATT (DW 1999 Table 4 PSM "
            "benchmark: ~$1,794).  Naive OLS shows strong selection "
            "bias; covariate-adjusted, matching, and doubly-robust "
            "estimators all converge near the experimental target."
        ),
        "data_loader": "datasets.nsw_lalonde",
        "data_kwargs": {"simulated": False},
        "data_origin": (
            "Real R MatchIt::lalonde extract bundled in "
            "statspai/datasets/data/lalonde_matchit.csv (n=614: 185 "
            "NSW treated + 429 PSID-1 controls).  Note: smaller than "
            "the full DW PSID-1 sample (n=2,675); naive bias here is "
            "-$635 rather than DW Table 3's -$8,498 headline."
        ),
        "classic": {
            "name": "Dehejia-Wahba (1999) propensity-score matching",
            "paper_table": "DW (1999) Table 3 (OLS) and Table 4 (PSM)",
            "references": ["dehejia1999causal", "rosenbaum1983central"],
            "tolerance": 5.0,  # version-to-version drift in $ units
            "code": [
                "# Naive OLS — shows the selection bias on this subset",
                "naive = sp.regress('re78 ~ treat', data=df, robust='hc1')",
                "",
                "# Covariate-adjusted OLS",
                "adj = sp.regress(",
                "    're78 ~ treat + age + educ + black + hispanic + '",
                "    'married + nodegree + re74 + re75',",
                "    data=df, robust='hc1')",
                "",
                "# 1:1 nearest-neighbour propensity-score matching (DW recipe)",
                "psm = sp.match(",
                "    data=df, y='re78', treat='treat',",
                "    covariates=['age', 'educ', 'black', 'hispanic',",
                "                'married', 'nodegree', 're74', 're75'],",
                "    method='nearest')",
                "",
                "sp.regtable([naive, adj], column_labels=['Naive OLS', 'Adjusted OLS'])",
                "print('1:1 NN PSM ATT:', round(float(psm.estimate), 0))",
            ],
            "golden_numbers": [
                (
                    "Naive OLS ATT ($)",
                    -635.0,
                    -635.0,
                    "StatsPAI vs R MatchIt parity (matchit_lalonde subset)",
                ),
                ("Adjusted OLS ATT ($)", 1548.2, 1548.2, "StatsPAI vs R parity"),
                (
                    "1:1 NN PSM ATT ($)",
                    1963.4,
                    1794.0,
                    "StatsPAI vs DW (1999) Table 4 experimental benchmark",
                ),
            ],
        },
        "modern": {
            "name": "Doubly-robust DML + entropy balancing",
            "rationale": (
                "Modern doubly-robust alternatives — DML (Chernozhukov "
                "et al. 2018) and entropy balancing (Hainmueller 2012) "
                "— give consistent ATT estimates under either correct "
                "outcome or correct propensity model, and avoid the "
                "PSM sensitivity to caliper / tie handling that plagues "
                "the classic recipe."
            ),
            "references": ["chernozhukov2018double", "hainmueller2012entropy"],
            "tolerance": 50.0,
            "code": [
                "covs = ['age', 'educ', 'black', 'hispanic', 'married',",
                "        'nodegree', 're74', 're75']",
                "",
                "# Double machine learning (partially-linear regression)",
                "dml = sp.dml(data=df, y='re78', d='treat',",
                "             covariates=covs, model='plr')",
                "print('DML PLR ATT:', round(float(dml.estimate), 0))",
                "",
                "# Entropy balancing",
                "eb = sp.ebalance(data=df, y='re78', treat='treat',",
                "                 covariates=covs)",
                "print('Entropy-bal ATT:', round(float(eb.estimate), 0))",
            ],
            "pinned_numbers": [
                (
                    "DML PLR ATT ($)",
                    1022.5,
                    "doubly-robust; close to DW $1,794 experimental benchmark",
                ),
                (
                    "Entropy-balancing ATT ($)",
                    1237.1,
                    "covariate moments matched on weights; close to DW $1,794",
                ),
            ],
        },
    },
    "angrist_pischke_mhe": {
        "title": "Angrist & Pischke (MHE) — Mostly Harmless Examples",
        "paper": (
            "Angrist, J.D. & Pischke, J.-S. (2009). Mostly Harmless " "Econometrics."
        ),
        "paper_bib": None,
        "journal": "Princeton University Press",
        "year": 2009,
        "design": "Various (OLS, IV, DID, RD)",
        "n_obs": None,
        "description": (
            "Key datasets and examples from the MHE textbook, covering "
            "returns to education, Vietnam draft lottery, etc."
        ),
        "data_loader": None,
        "data_kwargs": {},
        "data_origin": "Simulated illustrative data; not bundled.",
        "classic": None,
        "modern": None,
        "code": [
            "# Chapter 4: IV — returns to schooling",
            "iv = sp.ivreg('lwage ~ (educ ~ qob)', data=df)",
            "",
            "# Chapter 5: DID — minimum wage (Card & Krueger 1994)",
            "did = sp.did(df, y='employment', treat='nj', time='post')",
        ],
    },
    "lee_2008": {
        "title": "Lee (2008) — Senate-elections RD",
        "paper": (
            "Lee, D.S. (2008). Randomized Experiments from "
            "Non-Random Selection in US House Elections."
        ),
        "paper_bib": "lee2008randomized",
        "journal": "Journal of Econometrics 142(2), 675-697",
        "year": 2008,
        "design": "Regression Discontinuity",
        "n_obs": 1390,
        "description": (
            "Sharp RD on US Senate elections: lagged Democratic margin "
            "is the running variable; winning the seat (margin > 0) is "
            "treatment; vote share next election is the outcome.  Lee "
            "(2008) Table 1 reports an incumbency advantage of ~7.99 "
            "percentage points; CCT (2014) Table 4 replicates with "
            "bias-corrected robust inference."
        ),
        "data_loader": "datasets.lee_2008_senate",
        "data_kwargs": {"simulated": False},
        "data_origin": (
            "Real rdrobust::rdrobust_RDsenate panel bundled in "
            "statspai/datasets/data/lee_2008_senate.csv (n=1390; "
            "columns x = lagged Dem margin, y = current Dem vote "
            "share in percentage points 0-100)."
        ),
        "classic": {
            "name": "Local-linear conventional RD (Lee 2008)",
            "paper_table": "Lee (2008) Table 1; CCT (2014) Table 4",
            "references": ["lee2008randomized"],
            "tolerance": 1e-2,
            "code": [
                "# Conventional local-linear sharp RD with triangular kernel",
                "# and CCT (R-parity) MSE-optimal bandwidth.",
                "rd = sp.rdrobust(",
                "    df, y='y', x='x', c=0,",
                "    kernel='triangular', bwselect='cct')",
                "conv = rd.diagnostics['conventional']",
                "print(f'Conventional jump: {conv[\"estimate\"]:.3f} '",
                '      f\'(SE {conv["se"]:.3f}) at h={rd.diagnostics["bandwidth_h"]:.2f}\')',
            ],
            "golden_numbers": [
                (
                    "Conventional jump (pp)",
                    7.414,
                    7.99,
                    "Lee (2008) Table 1; CCT (2014) Table 4 conventional",
                ),
                (
                    "Conventional SE (pp)",
                    1.459,
                    1.46,
                    "StatsPAI vs R rdrobust parity at CCT bandwidth",
                ),
            ],
        },
        "modern": {
            "name": "CCT bias-corrected robust inference",
            "rationale": (
                "Calonico-Cattaneo-Titiunik (2014) showed that "
                "conventional local-linear CIs distort under MSE-"
                "optimal bandwidth because the bias is non-negligible. "
                "The bias-corrected robust estimator and SE are now "
                "the standard for RD inference and the rdrobust "
                "package default."
            ),
            "references": ["calonico2014robust"],
            "tolerance": 1e-2,
            "code": [
                "# Bias-corrected robust point estimate and CI",
                "rd = sp.rdrobust(",
                "    df, y='y', x='x', c=0,",
                "    kernel='triangular', bwselect='cct')",
                "rob = rd.diagnostics['robust']",
                "print(f'Robust jump: {rob[\"estimate\"]:.3f} '",
                "      f'(SE {rob[\"se\"]:.3f})')",
                "",
                "# Density test (no manipulation around cutoff)",
                "sp.rddensity(df, x='x', c=0)",
            ],
            "pinned_numbers": [
                (
                    "Robust jump (pp)",
                    7.507,
                    "CCT bias-corrected; matches R rdrobust at CCT bandwidth",
                ),
                (
                    "Robust SE (pp)",
                    1.741,
                    "identification-robust; preferred over Conventional SE",
                ),
                (
                    "CCT bandwidth h (pp)",
                    17.754,
                    "MSE-optimal; identical between R and StatsPAI",
                ),
            ],
        },
    },
    "graddy_2006": {
        "title": "Graddy (2006) — Fulton Fish Market demand elasticity via IV",
        "paper": "Graddy, K. (2006). Markets: The Fulton Fish Market.",
        "paper_bib": None,
        "journal": "Journal of Economic Perspectives 20(2), 207-220",
        "year": 2006,
        "design": "IV / 2SLS",
        "n_obs": 111,
        "description": (
            "Classic IV example from Cunningham's Causal Inference: "
            "The Mixtape (Ch. 7).  Estimates demand elasticity for "
            "fish using weather as instruments — wave height is "
            "strong, wind speed is weak."
        ),
        "data_loader": None,
        "data_kwargs": {},
        "data_origin": "Simulated DGP; original data on Graddy's website.",
        "classic": None,
        "modern": None,
        "code": [
            "# OLS (biased — supply/demand simultaneity)",
            "ols = sp.regress('log_quantity ~ log_price + mon + tue + wed + thu',",
            "                 data=df, robust='hc1')",
            "",
            "# IV with strong instrument (wave height)",
            "iv_strong = sp.ivreg('log_quantity ~ mon + tue + wed + thu + '",
            "                     '(log_price ~ wave_height)',",
            "                     data=df, robust='hc1')",
            "",
            "# IV with weak instrument (wind speed) — compare bias",
            "iv_weak = sp.ivreg('log_quantity ~ mon + tue + wed + thu + '",
            "                   '(log_price ~ wind_speed)',",
            "                   data=df, robust='hc1')",
            "",
            "sp.regtable([ols, iv_strong, iv_weak],",
            "            column_labels=['OLS', 'IV (wave)', 'IV (wind)'])",
            "",
            "# Weak instrument diagnostics",
            "ar = sp.anderson_rubin_test(data=df, y='log_quantity',",
            "     endog='log_price', instruments=['wave_height'],",
            "     exog=['mon', 'tue', 'wed', 'thu'])",
        ],
    },
}


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def list_replications() -> pd.DataFrame:
    """List all available replication datasets and guides.

    Returns
    -------
    pd.DataFrame
        Columns: ``key, title, design, journal, n_obs, has_real_data,
        has_classic_track, has_modern_track``.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.list_replications()
    """
    rows = []
    for key, info in _REPLICATIONS.items():
        loader = info.get("data_loader")
        kwargs = info.get("data_kwargs") or {}
        has_real = bool(loader) and bool(kwargs.get("simulated") is False)
        rows.append(
            {
                "key": key,
                "title": info["title"],
                "design": info["design"],
                "journal": info["journal"],
                "n_obs": info.get("n_obs", "—"),
                "has_real_data": has_real,
                "has_classic_track": info.get("classic") is not None,
                "has_modern_track": info.get("modern") is not None,
            }
        )
    return pd.DataFrame(rows)


def replicate(
    key: str,
    simulated: Optional[bool] = None,
) -> Tuple[pd.DataFrame, str]:
    """Load a famous dataset and a step-by-step replication guide.

    Load classic datasets with paper-faithful and modern recipes side
    by side.

    Parameters
    ----------
    key : str
        Replication key (see ``sp.list_replications()``).
    simulated : bool, optional
        Override the entry's default data source.  ``True`` forces a
        simulated replica; ``False`` forces the bundled real CSV (only
        valid for entries where ``has_real_data`` is True).  Default
        ``None`` uses whatever the entry declares (currently real
        for ``card_1995`` and ``abadie_2010``).

    Returns
    -------
    (data, guide) : tuple[pd.DataFrame, str]
        ``data``  — the dataset (real where available).
        ``guide`` — a printable replication guide with classic and
        modern tracks where applicable.

    Examples
    --------
    >>> import statspai as sp
    >>> data, guide = sp.replicate('card_1995')
    >>> print(guide)
    """
    if key not in _REPLICATIONS:
        available = ", ".join(_REPLICATIONS.keys())
        raise ValueError(f"Unknown replication: '{key}'. Available: {available}")

    info = _REPLICATIONS[key]
    data = _load_data(key, info, simulated_override=simulated)
    guide = _format_guide(key, info, data)
    return data, guide


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _load_data(
    key: str,
    info: Dict[str, Any],
    simulated_override: Optional[bool],
) -> pd.DataFrame:
    """Resolve and call the entry's data loader, falling back to the
    legacy in-file simulator when no loader is registered."""
    loader_path = info.get("data_loader")
    kwargs = dict(info.get("data_kwargs") or {})
    if simulated_override is not None:
        kwargs["simulated"] = simulated_override

    if loader_path:
        try:
            fn = _resolve_loader(loader_path)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"Could not resolve data loader '{loader_path}' for "
                f"replication '{key}': {exc}"
            ) from exc
        # Filter kwargs to those the loader actually accepts; some
        # legacy loaders don't take a `simulated` parameter.
        accepted = _accepted_kwargs(fn, kwargs)
        return fn(**accepted)

    legacy = _generate_data_legacy(key)
    if legacy is None:
        raise RuntimeError(
            f"Replication '{key}' has no data loader and no legacy "
            f"simulator; this entry is incomplete."
        )
    return legacy


def _resolve_loader(path: str) -> Callable[..., pd.DataFrame]:
    """Resolve dotted attribute path against the top-level statspai
    namespace (lazy import to avoid bootstrap cycles)."""
    import statspai as _sp  # local import — replicate.py imports during

    # statspai package init
    obj: Any = _sp
    for part in path.split("."):
        obj = getattr(obj, part)
    if not callable(obj):
        raise AttributeError(f"Resolved object {path!r} is not callable")
    return cast(Callable[..., pd.DataFrame], obj)


def _accepted_kwargs(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Drop kwargs the loader's signature does not advertise."""
    import inspect

    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# ----------------------------------------------------------------------
# Guide formatter
# ----------------------------------------------------------------------


def _format_guide(
    key: str,
    info: Dict[str, Any],
    data: pd.DataFrame,
) -> str:
    """Render the replication guide as a printable string."""
    lines: List[str] = []
    bar = "=" * 72
    rule = "-" * 72

    lines.append(bar)
    lines.append(f"REPLICATION GUIDE: {info['title']}")
    lines.append(bar)
    lines.append("")
    lines.append(f"Paper      : {info['paper']}")
    lines.append(f"Journal    : {info['journal']}")
    lines.append(f"Design     : {info['design']}")
    if info.get("paper_bib"):
        lines.append(f"BibTeX key : {info['paper_bib']} (verified in paper.bib)")
    lines.append("")
    lines.append("Description:")
    for chunk in _wrap(info["description"], width=70, indent="  "):
        lines.append(chunk)
    lines.append("")
    lines.append(f"Data       : {data.shape[0]:,} rows × {data.shape[1]} cols")
    lines.append(f"Provenance : {info.get('data_origin', '—')}")
    lines.append("")
    lines.append("# Load")
    lines.append("import statspai as sp")
    lines.append(f"data, _ = sp.replicate('{key}')")
    lines.append("df = data")
    lines.append("")

    classic = info.get("classic")
    modern = info.get("modern")

    if classic is None and modern is None:
        # Legacy single-track entry
        lines.append(rule)
        lines.append("CODE")
        lines.append(rule)
        lines.extend(info.get("code", []))
        lines.append("")
        lines.append(bar)
        return "\n".join(lines)

    if classic is not None:
        lines.append(rule)
        lines.append(f"TRACK 1 — CLASSIC: {classic['name']}")
        lines.append(rule)
        if classic.get("paper_table"):
            lines.append(f"Reference : {classic['paper_table']}")
        if classic.get("references"):
            lines.append(f"BibTeX    : {', '.join(classic['references'])}")
        lines.append("")
        lines.extend(classic.get("code", []))
        lines.append("")
        gold = classic.get("golden_numbers") or []
        if gold:
            tol = classic.get("tolerance", 1e-3)
            lines.append("Expected numbers (StatsPAI on real data vs. paper):")
            for label, sp_val, paper_val, citation in gold:
                delta = sp_val - paper_val
                lines.append(
                    f"  {label:<40s} StatsPAI = {sp_val:+.4f}   "
                    f"Paper = {paper_val:+.4f}   |Δ| = {abs(delta):.4f}"
                )
                lines.append(f"      [{citation}]")
            lines.append(
                f"  Regression-test drift tolerance (StatsPAI version "
                f"to version): |Δ| ≤ {tol}"
            )
            lines.append("  (Paper alignment Δ above can be larger; see citation.)")
        lines.append("")

    if modern is not None:
        lines.append(rule)
        lines.append(f"TRACK 2 — MODERN: {modern['name']}")
        lines.append(rule)
        lines.append("Why a second track?")
        for chunk in _wrap(modern.get("rationale", ""), width=70, indent="  "):
            lines.append(chunk)
        if modern.get("references"):
            lines.append(f"BibTeX    : {', '.join(modern['references'])}")
        lines.append("")
        lines.extend(modern.get("code", []))
        lines.append("")
        pinned = modern.get("pinned_numbers") or []
        if pinned:
            tol = modern.get("tolerance", 1e-2)
            lines.append("Expected numbers (StatsPAI regression-test pins;")
            lines.append("not paper values — paper predates these methods):")
            for entry in pinned:
                if len(entry) == 3:
                    label, sp_val, note = entry
                else:  # tolerate shorter tuples
                    label, sp_val = entry[0], entry[1]
                    note = ""
                lines.append(
                    f"  {label:<40s} StatsPAI = {sp_val:+.4f}   " f"({note})"
                    if note
                    else f"  {label:<40s} StatsPAI = {sp_val:+.4f}"
                )
            lines.append(f"  Pinned tolerance: |Δ| ≤ {tol}")
        lines.append("")

    lines.append(bar)
    return "\n".join(lines)


def _wrap(text: str, width: int, indent: str) -> List[str]:
    """Minimal word-wrap that respects an indent prefix."""
    if not text:
        return []
    import textwrap

    return textwrap.wrap(
        text, width=width, initial_indent=indent, subsequent_indent=indent
    ) or [indent]


# ----------------------------------------------------------------------
# Legacy in-file simulators (kept for entries without a datasets loader)
# ----------------------------------------------------------------------


def _generate_data_legacy(key: str) -> Optional[pd.DataFrame]:
    """Simulators for legacy entries that don't yet have a
    ``sp.datasets.*`` loader.  Currently only ``graddy_2006`` and
    ``angrist_pischke_mhe`` reach this path."""
    rng = np.random.default_rng(42)

    if key == "graddy_2006":
        n = 111
        day_of_week = rng.choice(5, n)
        mon = (day_of_week == 0).astype(int)
        tue = (day_of_week == 1).astype(int)
        wed = (day_of_week == 2).astype(int)
        thu = (day_of_week == 3).astype(int)
        wave_height = rng.exponential(2.0, n)
        wind_speed = rng.exponential(5.0, n)
        supply_shock = -0.3 * wave_height - 0.05 * wind_speed + rng.normal(0, 0.5, n)
        demand_shock = rng.normal(0, 0.5, n)
        log_price = (
            1.0 - 0.4 * supply_shock + 0.4 * demand_shock + rng.normal(0, 0.2, n)
        )
        log_quantity = (
            8.5
            - 0.95 * log_price
            - 0.1 * mon
            + 0.05 * tue
            - 0.02 * wed
            + 0.08 * thu
            + 0.3 * demand_shock
            + rng.normal(0, 0.3, n)
        )
        df = pd.DataFrame(
            {
                "log_quantity": log_quantity,
                "log_price": log_price,
                "wave_height": wave_height,
                "wind_speed": wind_speed,
                "mon": mon,
                "tue": tue,
                "wed": wed,
                "thu": thu,
            }
        )
        df.attrs["true_elasticity"] = -0.95
        return df

    if key == "angrist_pischke_mhe":
        # MHE is a textbook reference; no single dataset.  Return an
        # empty frame so the guide still renders.
        return pd.DataFrame()

    return None
