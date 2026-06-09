# NIST StRD — Analysis of Variance certification datasets

These `.dat` files are the **unmodified** one-way Analysis of Variance
datasets from the U.S. National Institute of Standards and Technology
*Statistical Reference Datasets* (StRD) collection.

- Source: <https://www.itl.nist.gov/div898/strd/anova/anova.html>
- Per-dataset files: `https://www.itl.nist.gov/div898/strd/anova/<Name>.dat`
- Retrieved: 2026-06-08 (via `curl`, byte-for-byte; never hand-transcribed).
- Each file embeds, in its ASCII header, the **certified** between/within sums
  of squares, mean squares, F-statistic, R², and residual standard deviation,
  computed by NIST in multiple-precision arithmetic and rounded to 15
  significant digits. The `<group> <value>` data rows follow. The certified
  values are the single source of truth — `test_nist_strd_anova.py` reads them
  straight out of these headers.

NIST is a U.S. Government work and is not subject to copyright in the United
States.

## The 11 datasets (by difficulty)

A one-way ANOVA is algebraically OLS of `y ~ C(group)`, so these certify the
numerical accuracy of `sp.regress`'s sum-of-squares / F-statistic path.

| File        | n     | Treatments | Difficulty | Notes |
|-------------|-------|------------|------------|-------|
| SiRstv      | 25    | 5          | lower      | silicon resistivity |
| SmLs01      | 189   | 9          | lower      | certified F = 21 exactly |
| SmLs02      | 1809  | 9          | lower      | certified F = 21 exactly |
| SmLs03      | 18009 | 9          | lower      | certified F = 2001 exactly |
| SmLs04      | 189   | 9          | average    | ~6 constant leading digits |
| SmLs05      | 1809  | 9          | average    | ~6 constant leading digits |
| SmLs06      | 18009 | 9          | average    | ~6 constant leading digits |
| SmLs07      | 189   | 9          | higher     | ~9 constant leading digits |
| SmLs08      | 1809  | 9          | higher     | ~9 constant leading digits |
| SmLs09      | 18009 | 9          | higher     | ~9 constant leading digits |
| AtmWtAg     | 48    | 2          | average    | atomic weight of silver |

The `SmLs0{1..9}` family is built so naive (cancellation-prone) sums of squares
lose precision: difficulty rises with the number of constant leading digits
(01–03 = ~3, 04–06 = ~6, 07–09 = ~9). `sp.regress` reproduces the certified F
to machine precision through the average-difficulty family (including SmLs06 at
n=18009) but loses 3–4+ significant digits on the higher-difficulty `SmLs07/08/09`
— those three are recorded as `xfail` in the test (see its module docstring).
