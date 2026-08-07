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
                "sp.regtable([ols, iv], model_labels=['OLS', 'IV (nearc4)'])",
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
    # Cheng & Hoekstra (2013) — castle doctrine — staggered DiD
    # ------------------------------------------------------------------
    "castle_2013": {
        "title": "Cheng & Hoekstra (2013) — Castle-doctrine expansions and homicide",
        "paper": (
            "Cheng, C. & Hoekstra, M. (2013). Does Strengthening "
            "Self-Defense Law Deter Crime or Escalate Violence?: "
            "Evidence from Expansions to Castle Doctrine."
        ),
        "paper_bib": "cheng2013does",
        "journal": "Journal of Human Resources 48(3), 821-853",
        "year": 2013,
        "design": "Staggered DiD",
        "n_obs": 550,
        "description": (
            "50 US states x 11 years (2000-2010).  Between 2005 and 2009, "
            "21 states expanded castle-doctrine self-defence law; 29 never "
            "did, giving a clean never-treated control group.  Contrary to "
            "a deterrence story, the expansions *raised* log homicide by "
            "roughly 8 log points.  This is the Chapter 9 dataset of "
            "Cunningham's Causal Inference: The Mixtape, where it motivates "
            "the Goodman-Bacon decomposition and modern staggered-adoption "
            "estimators."
        ),
        "data_loader": "datasets.castle_doctrine",
        "data_kwargs": {"simulated": False},
        "data_origin": (
            "Real Cheng-Hoekstra (2013) replication panel, modelling subset "
            "bundled in statspai/datasets/data/castle_2013.csv (550 rows x "
            "29 cols) from the MIT-licensed mixtape repository.  The 44 "
            "region x year dummies and 51 state trends are regenerated on "
            "demand via castle_doctrine(region_year_fe=True, "
            "state_trends=True)."
        ),
        "classic": {
            "name": "Two-way fixed effects (Cheng-Hoekstra Table 4)",
            "paper_table": "Cheng & Hoekstra (2013) Table 4; Mixtape Ch. 9",
            "references": ["cheng2013does", "cunningham2021causal"],
            "tolerance": 1e-5,
            "code": [
                "# The paper's ladder of TWFE specifications.  Weights are",
                "# state population (Stata aweight); SEs cluster on state.",
                "df = sp.datasets.castle_doctrine(",
                "    region_year_fe=True, state_trends=True)",
                "",
                "xvar = ['l_police', 'unemployrt', 'poverty', 'l_income',",
                "        'l_prisoner', 'l_lagprisoner', 'blackm_15_24',",
                "        'whitem_15_24', 'blackm_25_44', 'whitem_25_44',",
                "        'l_exp_subsidy', 'l_exp_pubwelfare']",
                "region = [c for c in df.columns if c.startswith('r20')]",
                "trends = [c for c in df.columns if c.startswith('trend_')]",
                "",
                "bare = sp.feols('l_homicide ~ post | sid + year',",
                "                data=df, vcov={'CRV1': 'sid'})",
                "wtd  = sp.feols('l_homicide ~ post | sid + year', data=df,",
                "                weights='popwt', vcov={'CRV1': 'sid'})",
                "full = sp.feols(",
                "    'l_homicide ~ post + ' + ' + '.join(xvar + region + trends)",
                "    + ' | sid + year',",
                "    data=df, weights='popwt', vcov={'CRV1': 'sid'})",
                "",
                "sp.regtable([bare, wtd, full],",
                "            model_labels=['TWFE', '+ weights', '+ full controls'])",
            ],
            "golden_numbers": [
                (
                    "TWFE unweighted beta_post",
                    0.069398429,
                    0.069398429,
                    "Stata 18 MP xtreg fe vce(cluster sid)",
                ),
                (
                    "TWFE weighted beta_post",
                    0.075533239,
                    0.075533239,
                    "Stata 18 MP, [aweight=popwt]",
                ),
                (
                    "TWFE weighted + controls",
                    0.079634870,
                    0.079634870,
                    "Stata 18 MP, mixtape Do/castle_1.do covariates",
                ),
                (
                    "TWFE full (region x year + trends)",
                    0.076948986,
                    0.076948986,
                    "Stata 18 MP, mixtape Do/castle_1.do headline spec",
                ),
            ],
        },
        "modern": {
            "name": "Goodman-Bacon decomposition + Callaway-Sant'Anna",
            "rationale": (
                "With staggered adoption, TWFE is a weighted average of all "
                "2x2 DiDs, including 'forbidden' comparisons that use "
                "already-treated units as controls.  The Bacon "
                "decomposition shows 89.9% of the weight here falls on "
                "clean never-treated comparisons — unusually benign — yet "
                "the remaining 10.1% still pulls the estimate down: "
                "Callaway-Sant'Anna returns 0.110 against TWFE's 0.069.  "
                "Note the cohort-coding trap documented below."
            ),
            "references": [
                "goodmanbacon2021difference",
                "callaway2021difference",
                "cunningham2021causal",
            ],
            "tolerance": 1e-5,
            "code": [
                "df = sp.datasets.castle_doctrine(event_time=True)",
                "",
                "# (a) Why is TWFE what it is?  Decompose it.",
                "bacon = sp.bacon_decomposition(",
                "    df, y='l_homicide', treat='post', time='year', id='sid')",
                "dec = bacon['decomposition']",
                "clean = dec[dec['type'] == 'Treated vs Untreated']['weight'].sum()",
                "print(f'TWFE = {bacon[\"beta_twfe\"]:.4f}; '",
                "      f'never-treated weight = {clean:.1%}')",
                "",
                "# (b) Group-time ATT robust to treatment-effect heterogeneity.",
                "#     gvar codes never-treated as 0 (event_time=True adds it).",
                "cs = sp.callaway_santanna(",
                "    df, y='l_homicide', g='gvar', t='year', i='sid',",
                "    control_group='nevertreated')",
                "print('CS simple ATT:', float(sp.aggte(cs, type='simple').estimate))",
                "sp.aggte(cs, type='dynamic').plot()",
            ],
            "pinned_numbers": [
                (
                    "Bacon TWFE beta",
                    0.069398429,
                    "matches Stata bacondecomp; all 25 2x2 cells agree",
                ),
                (
                    "Bacon never-treated weight",
                    0.898808834,
                    "share of TWFE from clean never-treated comparisons",
                ),
                (
                    "Callaway-Sant'Anna simple ATT (gvar=effyear)",
                    0.110383035,
                    "matches R did::aggte and Stata csdid to 1e-9",
                ),
                (
                    "Callaway-Sant'Anna simple ATT (gvar=effyear+1)",
                    0.019402808,
                    "cohort-coding trap — see the note below, NOT a bug",
                ),
            ],
        },
        "caveats": [
            "post is NOT 1{year >= effyear}.  Cheng & Hoekstra code "
            "post = 1{year > effyear}: the adoption year is treated as "
            "UNTREATED because the law was in force for only part of it, "
            "and the fractional exposure lives in `cdl` (Alabama 2006 = "
            "0.5808).  Rebuilding treatment as year >= effyear silently "
            "changes 21 observations.",
            "That makes the adoption cohort ambiguous for group-time "
            "estimators.  gvar = effyear keeps a clean pre-treatment base "
            "period but counts the partially-exposed adoption year as fully "
            "treated (ATT = 0.1104).  gvar = effyear + 1 is consistent with "
            "`post` but pushes that partially-treated year into the BASE "
            "period, contaminating the baseline (ATT = 0.0194).  Neither is "
            "unambiguously right; report both, or drop the adoption year.",
            "This replication is what caught the aggte standard-error bug: "
            "StatsPAI's Callaway-Sant'Anna SEs were up to 8% smaller than R "
            "did / Stata csdid because the estimated cohort-share weights "
            "were treated as fixed.  Fixed as of the current release — SEs "
            "now match R and Stata exactly.  Numbers quoted from an earlier "
            "version should be re-run; see MIGRATION.md.",
        ],
    },
    # ------------------------------------------------------------------
    # Texas 1993 prison expansion — synthetic control (Mixtape Ch. 10)
    # ------------------------------------------------------------------
    "texas_1993": {
        "title": "Texas 1993 prison expansion — synthetic control (Mixtape Ch. 10)",
        "paper": (
            "Cunningham, S. (2021). Causal Inference: The Mixtape, "
            "Chapter 10 (Synthetic Control)."
        ),
        "paper_bib": "cunningham2021causal",
        "journal": "Yale University Press",
        "year": 2021,
        "design": "Synthetic Control",
        "n_obs": 816,
        "description": (
            "Texas roughly doubled prison operational capacity over "
            "1993-1995 (about +35%/year).  Build a synthetic Texas from "
            "the other 50 states and read the gap in Black male prisoner "
            "counts.  Ships as a deliberate *non-parity* case: the book's "
            "recipe puts four lagged outcomes among the predictors, which "
            "leaves the predictor-weight matrix V weakly identified and "
            "the nested V-W problem non-convex, so Stata and StatsPAI "
            "settle on different donor sets while agreeing on the effect."
        ),
        "data_loader": "datasets.texas_prison",
        "data_kwargs": {"simulated": False},
        "data_origin": (
            "Real texas.dta from the MIT-licensed mixtape repository, "
            "bundled in statspai/datasets/data/texas_prison.csv "
            "(816 rows x 24 cols; 51 states x 16 years, 1985-2000)."
        ),
        "classic": {
            "name": "Abadie-style SCM on the book's predictor recipe",
            "paper_table": "Mixtape Ch. 10; Cunningham's texas_synth.do",
            "references": ["cunningham2021causal", "abadie2010synthetic"],
            # Loose on purpose: this is a local-optimum comparison across
            # two optimisers, not a bit-parity claim. See caveats.
            "tolerance": 1.0,
            "code": [
                "df = sp.datasets.texas_prison()",
                "",
                "# The book's recipe: period-specific predictors plus three",
                "# covariates averaged over the pre-period.",
                "lags = (1988, 1990, 1991, 1992)",
                "special = ([('bmprison', y, 'mean') for y in lags]",
                "           + [('alcohol', 1990, 'mean'),",
                "              ('aidscapita', 1990, 'mean'),",
                "              ('aidscapita', 1991, 'mean')]",
                "           + [('black', y, 'mean') for y in (1990, 1991, 1992)]",
                "           + [('perc1519', 1990, 'mean')])",
                "",
                "sc = sp.synth(",
                "    data=df, outcome='bmprison', unit='state', time='year',",
                "    treated_unit='Texas', treatment_time=1993,",
                "    covariates=['income', 'ur', 'poverty'],",
                "    special_predictors=special,",
                "    backend='native', placebo=False)",
                "",
                "print(sc.weights[sc.weights['weight'] > 1e-3])",
                "gap = sc.model_info['gap_table']",
                "print('mean 1994-2000 gap:',",
                "      round(gap[gap['time'] >= 1994]['gap'].mean(), 1))",
            ],
            "golden_numbers": [
                (
                    "Mean gap 1994-2000 (Black male prisoners)",
                    23779.4061,
                    23073.6984,
                    "StatsPAI vs Stata 18 MP `synth`: different local "
                    "optima, ~3% apart",
                ),
                (
                    "Pre-treatment RMSE",
                    865.3084,
                    1227.0256,
                    "StatsPAI attains the better pre-fit on its own objective",
                ),
            ],
        },
        "modern": {
            "name": "synthdid + augmented SCM (no V to optimise)",
            "rationale": (
                "The classic estimator's fragility here is entirely in the "
                "predictor-weight matrix V.  synthdid (Arkhangelsky et al. "
                "2021) and augmented SCM (Ben-Michael et al. 2021) do not "
                "estimate a V at all — synthdid uses unit and time weights "
                "on the outcome, ASCM adds a ridge bias correction to an "
                "outcome-only fit.  Both are reproducible across "
                "implementations on this panel, which the classic recipe "
                "is not."
            ),
            "references": ["arkhangelsky2021synthetic", "benmichael2021augmented"],
            "tolerance": 1.0,
            "code": [
                "df = sp.datasets.texas_prison()",
                "",
                "sdid = sp.synthdid_estimate(",
                "    data=df, y='bmprison', unit='state', time='year',",
                "    treat_unit='Texas', treat_time=1993)",
                "print('synthdid ATT:', round(float(sdid.estimate), 1))",
                "",
                "# Outcome-only classic SCM: V is fixed to the identity, so the",
                "# remaining problem in the donor weights is convex and has a",
                "# unique solution — the reproducible fallback.",
                "sc = sp.synth(",
                "    data=df, outcome='bmprison', unit='state', time='year',",
                "    treated_unit='Texas', treatment_time=1993, placebo=False)",
                "print('outcome-only ATT:', round(float(sc.estimate), 1))",
            ],
            "pinned_numbers": [
                (
                    "synthdid ATT",
                    19478.6,
                    "unit + time weights; no V to optimise",
                ),
                (
                    "Outcome-only classic ATT",
                    21482.1,
                    "V fixed to identity -> convex in W -> unique solution",
                ),
                (
                    "Outcome-only donor weights",
                    0.476,
                    "New York .476, Illinois .340, Florida .184 (reproducible)",
                ),
            ],
        },
        "caveats": [
            "This entry is shipped as a NON-PARITY case on purpose.  Stata "
            "`synth` picks California .408 / Illinois .360 / Louisiana .122 "
            "/ Florida .109; StatsPAI picks Florida .436 / New York .311 / "
            "Illinois .253.  Neither is a bug: the book's recipe includes "
            "four lagged outcomes among the predictors, which per Kaul, "
            "Klossner, Pfeifer & Schieler (2015) leaves V weakly identified, "
            "and the nested V-W optimisation is non-convex with several "
            "local optima.  StatsPAI reaches the LOWER pre-treatment RMSE "
            "(865.3 vs 1227.0), i.e. the better fit on the stated objective, "
            "and it is not a lucky draw: raising n_random_starts from 4 to "
            "40 returns the identical optimum.",
            "The lesson worth carrying: the estimated effect is far more "
            "robust than the weights.  Mean 1994-2000 gap is 23074 (Stata, "
            "book spec), 23343 (Stata, default MSPE window) and 23779 "
            "(StatsPAI) — a ~3% spread across entirely disjoint donor sets. "
            "Report the effect; do not interpret the donor weights as a "
            "finding, and do not tune the recipe until the weights look "
            "familiar.",
            "If you need an SCM number that reproduces across software, use "
            "the modern track, or the outcome-only classic recipe (drop "
            "covariates and special predictors) where V is fixed to the "
            "identity and the donor-weight problem becomes convex.",
            "Attribution: Cunningham's own data readme credits the natural "
            "experiment to a 'Cornwell and Cunningham (2016)' manuscript "
            "and to Perkinson (2010), Texas Tough.  Perkinson is "
            "verifiable; no Cornwell & Cunningham (2016) record exists in "
            "Crossref, so StatsPAI cites the Mixtape itself and reports "
            "that attribution as the upstream author's, not as a "
            "publication we could verify.",
        ],
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
                "sp.regtable([naive, adj], model_labels=['Naive OLS', 'Adjusted OLS'])",
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
            "            model_labels=['OLS', 'IV (wave)', 'IV (wind)'])",
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

    caveats = info.get("caveats") or []
    if caveats:
        lines.append(rule)
        lines.append("CAVEATS — read before quoting any number from this panel")
        lines.append(rule)
        for i, caveat in enumerate(caveats, start=1):
            wrapped = _wrap(caveat, width=68, indent="     ")
            if wrapped:
                # Re-prefix the first line with the item number.
                first = wrapped[0].lstrip()
                lines.append(f"  {i}. {first}")
                lines.extend(wrapped[1:])
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
