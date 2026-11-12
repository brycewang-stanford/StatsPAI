---
description: Estimate instrumental variables (2SLS) in Python with first-stage F, robust standard errors and a weak-instrument-robust Anderson-Rubin interval, on the Card (1995) data.
---

# Instrumental variables and 2SLS in Python

**Short answer.** `sp.ivreg("y ~ (d ~ z) + controls", data=df)` runs two-stage
least squares and prints the first-stage F, partial R² and a Hausman test with
the coefficients. When the first stage is not strong, add a
weak-instrument-robust interval with `sp.anderson_rubin_ci`. On StatsPAI's
cross-language suite `sp.ivreg` matches R's `AER::ivreg` and Stata's
`ivregress 2sls, small` (`sp.parity_status("ivreg")`).

## When this applies

- **Estimand**: with a binary instrument and heterogeneous effects, 2SLS
  identifies a local average treatment effect for compliers, not the ATE.
- **Assumptions**: relevance (check the first stage), exclusion (the
  instrument affects the outcome only through the treatment) and, for the
  LATE reading, monotonicity. Only relevance can be tested from the data.
- **Not the right tool** when the instrument plausibly has a direct effect, or
  when you have many weak instruments (consider `sp.liml` or `sp.jive`).

## Example: returns to schooling (Card 1995)

The instrument is growing up near a four-year college (`nearc4`); the data are
the real NLSYM extract.

```python
import statspai as sp

card = sp.datasets.card_1995()
iv = sp.ivreg("lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa",
              data=card)
print(iv.summary())
```

```text
educ 0.1323 0.0492 2.6870 0.0072 0.0358 0.2288
...
First-stage F (educ) : 16.7176
...
Hausman p-value : 0.2149
```

The IV estimate (0.132) is larger than OLS (0.074) and far less precise. The
first-stage F of 16.7 is above the old rule of thumb of 10 but not
comfortably so, so report an interval that stays valid under weak
instruments:

```python
ar = sp.anderson_rubin_ci(y="lwage", endog="educ", instruments=["nearc4"],
                          exog=["exper", "expersq", "black", "south", "smsa"],
                          data=card)
print(ar.summary())
```

## OLS and IV side by side, robust standard errors

```python
ols = sp.regress("lwage ~ educ + exper + expersq + black + south + smsa",
                 data=card, vce="robust")
iv_r = sp.ivreg("lwage ~ (educ ~ nearc4) + exper + expersq + black + south + smsa",
                data=card, robust="hc1")
print(sp.regtable([ols, iv_r], model_labels=["OLS", "2SLS"], keep=["educ"]))
```

```text
OLS 2SLS
educ 0.0740*** 0.1323***
(0.0036) (0.0486)
First-stage F 17.51
N 3,010 3,010
```

With `robust="hc1"` the first-stage F uses the same robust variance, as
Stata's `estat firststage` does after `vce(robust)`.

## Check before you report

- First-stage strength (F, effective F via `sp.effective_f_test`) and a
  weak-IV-robust interval (`sp.anderson_rubin_ci`).
- An argument for exclusion: it cannot be tested, so it has to be made.
- `sp.iv_diag` collects the post-2022 reporting checklist (first stage,
  effective F, AR test, OLS comparison) in one call; the
  [IV chooser](../guides/choosing_iv_estimator.md) covers LIML, JIVE, shift-share
  and judge designs.

## Stata and R equivalents

| Task | Stata | R | StatsPAI |
| --- | --- | --- | --- |
| 2SLS | `ivregress 2sls lwage (educ = nearc4) exper ..., small` | `AER::ivreg(lwage ~ educ + exper ... \| nearc4 + exper ...)` | `sp.ivreg("lwage ~ (educ ~ nearc4) + exper ...", data=df)` |
| Robust SE | `, vce(robust)` | `sandwich::vcovHC(fit, "HC1")` | `robust="hc1"` |
| First stage | `estat firststage` | `summary(fit, diagnostics = TRUE)` | printed by `.summary()` |
| AR interval | `weakiv` | `ivmodel::AR.test()` | `sp.anderson_rubin_ci()` |

## Other Python options

`linearmodels` has the most complete set of IV estimators and first-stage
diagnostics in the Python ecosystem besides StatsPAI (2SLS, LIML, GMM, CUE,
Kleibergen–Paap); `pyfixest` supports IV with fixed effects and an effective
F. See the [library comparison](python-econometrics-libraries.md).

---

*Tested with StatsPAI 1.43.1 on 2026-11-12. The code and numbers on this page
are run by `tests/test_docs_task_pages.py`.
[Report a problem](https://github.com/brycewang-stanford/StatsPAI/issues).*
