# Published reference values — external parity anchors

Each simulated replica in `sp.datasets` is calibrated so that the
canonical estimator recovers output in the neighbourhood of the
**published numbers on the ORIGINAL data**.  This document is the
primary-source index for every such claim.

## Callaway-Sant'Anna mpdta

| Statistic | Value | Source |
| --- | --- | --- |
| Simple ATT | -0.0454 (SE 0.0113) | `did` R package vignette, aggte type="simple" |
| Group 2004 ATT | -0.0727 (SE 0.0292) | Callaway-Sant'Anna (2021), Table 5 |
| Group 2006 ATT | -0.0029 (SE 0.0207) | Callaway-Sant'Anna (2021), Table 5 |
| Group 2007 ATT | -0.0310 (SE 0.0124) | Callaway-Sant'Anna (2021), Table 5 |

Citation: Callaway, B. & Sant'Anna, P.H.C. (2021). Difference-in-Differences
with Multiple Time Periods. *Journal of Econometrics*, 225(2), 200-230.

`sp.datasets.mpdta()` replica target: simple ATT ≈ -0.04 (sign and
order of magnitude match; exact numerical R parity not claimed
because the data is simulated).

## Card (1995) returns to schooling

| Statistic | Value | Source |
| --- | --- | --- |
| OLS β(educ) | 0.075 | Card (1995) Table 2, col 2 |
| IV β(educ), nearc4 | 0.132 | Card (1995) Table 2, col 5 |
| IV β(educ), nearc2 | 0.097 | Card (1995) Table 2, col 6 |

Citation: Card, D. (1995). Using Geographic Variation in College Proximity
to Estimate the Return to Schooling. In Christofides et al. (eds.),
*Aspects of Labour Market Behaviour*.

`sp.datasets.card_1995()` replica target: OLS ≈ 0.11, IV ≈ 0.142
(IV > OLS, the "Card puzzle").  Exact numerical values differ
because the DGP is simulated, not the original NLSYM data.

## LaLonde (1986) / Dehejia-Wahba (1999) NSW

| Statistic | Value | Source |
| --- | --- | --- |
| Experimental ATT on re78 (NSW only) | $1,794 (SE $633) | Dehejia-Wahba (1999) Table 2 |
| Naive OLS on NSW+PSID-1 | -$8,498 | Dehejia-Wahba (1999) Table 3 |
| PSM ATT on NSW+PSID-1 | $1,794 | Dehejia-Wahba (1999) Table 4 |
| Covariate-adj OLS on NSW+PSID-1 | $218 (bias remains) | Dehejia-Wahba (1999) Table 3 |

Citations:

- LaLonde, R. (1986). Evaluating the Econometric Evaluations of Training
  Programs with Experimental Data. *AER* 76(4), 604-620.
- Dehejia, R. & Wahba, S. (1999). Causal Effects in Nonexperimental
  Studies: Reevaluating the Evaluation of Training Programs. *JASA*
  94(448), 1053-1062.

`sp.datasets.nsw_lalonde()` and `nsw_dw()` replica targets: match
published numbers within $500 on the experimental subset; the
naive-OLS bias on the NSW+PSID combination matches within $500.

## Lee (2008) US Senate RD

| Statistic | Value | Source |
| --- | --- | --- |
| RD jump at margin = 0 | 0.080 (SE 0.008) | Lee (2008) Table 2, col 1 |

Citation: Lee, D. (2008). Randomized experiments from non-random
selection in U.S. House elections. *Journal of Econometrics* 142,
675-697.

`sp.datasets.lee_2008_senate()` replica target: 0.08.

## Angrist-Krueger (1991) quarter-of-birth

| Statistic | Value | Source |
| --- | --- | --- |
| OLS β(educ), cohort 1930-1939 | 0.063 (SE 0.00035) | Angrist-Krueger (1991) Table V |
| 2SLS β(educ), QOB×YOB | 0.089 (SE 0.016) | Angrist-Krueger (1991) Table V |
| 2SLS β(educ), cohort 1920-1929 | 0.102 (SE 0.018) | Angrist-Krueger (1991) Table IV |

Citation: Angrist, J. & Krueger, A. (1991). Does Compulsory School
Attendance Affect Schooling and Earnings? *QJE* 106(4), 979-1014.

`sp.datasets.angrist_krueger_1991()` replica target: IV ≈ 0.10 by
construction.  The original NBER public data is needed for exact
numerical replication.

## California Prop 99 (Abadie-Diamond-Hainmueller 2010)

| Statistic | Value | Source |
| --- | --- | --- |
| Average post-1988 gap (packs/capita) | -14.4 to -24.5 | Abadie-Diamond-Hainmueller (2010), Table 2 |

Citation: Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic
Control Methods for Comparative Case Studies: Estimating the Effect of
California's Tobacco Control Program. *JASA* 105(490), 493-505.

`sp.datasets.california_prop99()` replica target: ATT ≈ -15 packs/capita.

## Basque Country Terrorism (Abadie-Gardeazabal 2003)

| Statistic | Value | Source |
| --- | --- | --- |
| Avg GDPpc gap 1975-1997 | -0.855 (10%) | Abadie-Gardeazabal (2003) Figure 2 |

Citation: Abadie, A. & Gardeazabal, J. (2003). The Economic Costs of
Conflict: A Case Study of the Basque Country. *AER* 93(1), 113-132.

`sp.datasets.basque_terrorism()` replica target: gap ≈ -0.855.

## German Reunification (Abadie-Diamond-Hainmueller 2015)

| Statistic | Value | Source |
| --- | --- | --- |
| Avg GDPpc gap 1990-2003 | ≈ -1,600 (USD 2002) | Abadie-Diamond-Hainmueller (2015) Figure 4 |

Citation: Abadie, A., Diamond, A. & Hainmueller, J. (2015). Comparative
Politics and the Synthetic Control Method. *American Journal of
Political Science* 59(2), 495-510.

`sp.datasets.german_reunification()` replica target: gap ≈ -1,500.

---

## Hernán & Robins, *Causal Inference: What If* — NHEFS (REAL data)

Unlike every anchor above, `sp.datasets.nhefs()` ships the **real,
public-domain** NHEFS extract that Hernán & Robins use throughout Part II
of the textbook (NHANES I follow-up; re-packaged from the MIT-licensed
`causaldata`). Because the data are genuine, StatsPAI is held to the
book's **actual published figures**, not merely their neighbourhood, and
each statistic carries a same-bytes R gold reference (paired scripts
`tests/orig_parity/06–11_nhefs_*.{py,R}`). Effect of quitting smoking
(`qsmk`) on 10-year weight change (`wt82_71`, kg) and mortality (`death`).

| Ch | Statistic | Book value | StatsPAI | R gold | Source |
| --- | --- | ---: | ---: | ---: | --- |
| 12 | Crude weight-change diff (confounded) | 2.54 | 2.54 | 2.54 | §12.2 |
| 12 | IP-weighted ATE (MSM) | 3.4 (CI 2.4–4.5) | 3.47 | 3.44 | Program 12.4 |
| 13 | Standardization / parametric g-formula | 3.5 | 3.46 | 3.46 | Program 13.3 |
| 14 | G-estimation of SNMM (ψ) | 3.4 | 3.46 | 3.46 | Program 14.2 |
| 15 | Outcome-reg `qsmk` main coef | 2.56 | 2.56 | 2.56 | Program 15.1 |
| 15 | Outcome-reg `qsmk × smokeintensity` | 0.0467 | 0.0467 | 0.0467 | Program 15.1 |
| 15 | Effect at smokeintensity = 5 | 2.79 | 2.79 | 2.79 | Program 15.1 |
| 15 | Effect at smokeintensity = 40 | 4.43 | 4.43 | 4.43 | Program 15.1 |
| 15 | Propensity-score-adjusted ATE | ≈ 3.5 | 3.45 | 3.45 | Program 15.3 |
| 17 | Conditional (unweighted) hazard ratio | ≈ 1.4 | 1.39 | 1.39 | §17 |
| 17 | IP-weighted hazard ratio | ≈ 1.0 | 1.00 | 1.00 | Program 17.4 |
| 17 | IP-weighted 120-mo survival difference | ≈ 0 | 0.002 | 0.002 | Program 17.4 |
| — | E-value, crude qsmk→death RR (=1.33) | 1.98 | 1.98 | 1.98 | VanderWeele-Ding (2017) |

Citation: Hernán, M.A. & Robins, J.M. (2020). *Causal Inference: What If*.
Boca Raton: Chapman & Hall/CRC.

Headline: across the textbook's g-methods, **StatsPAI reproduces the book
to the printed precision** (the Chapter 15 regression coefficients match
to four decimals), and **StatsPAI agrees with an independent base-R /
`survival` / `EValue` reference on the same CSV bytes** — usually to 3–4
decimals. The Chapter 15 closed-form regressions are an exact match; the
IP-weighted, g-formula, g-estimation, and survival quantities embed minor
documented convention choices (Hájek vs stabilized weights; additive
encoding) that keep them within ~2% of the book and the R gold. This is
the first epidemiology-stack (vs econometrics-stack) parity certification
in the repository. Pinned regression tests live in
`tests/external_parity/test_whatif_nhefs.py`.

---

## How to add a new parity anchor

1. Add a DGP to `src/statspai/datasets/_canonical.py` that calibrates
   the canonical estimator's output to the published value ± 50%.
2. Expose via `sp.datasets.<name>()` and register in `list_datasets()`.
3. Add a pinned test in `tests/external_parity/test_published_replications.py`:
   - One test pinning the output to 4 decimals (drift guard).
   - One test asserting the output is in a reasonable neighbourhood
     of the published value on the original data (calibration check).
4. Document the primary-source citation in this file.
