# Stock & Watson, 4th edition, in StatsPAI

Stock and Watson's *Introduction to Econometrics* is where many people first
meet a robust standard error. Its authors publish the replication files for
the fourth edition, with the Stata do-files and the logs they produced for
chapters 2 to 13 and a RATS program with its output for chapter 15.

This guide does three things with them.

1. It runs the book's do-files in Python, unchanged apart from the `use`
   line.
2. It shows how to check that the numbers are the book's, down to the last
   digit Stata printed.
3. It lists, chapter by chapter, what a paper written today would add to the
   book's analysis, and the StatsPAI call for it.

The files are at <https://www.princeton.edu/~mwatson/Stock-Watson_4E/>. They
are not redistributed with StatsPAI. Unzip the chapter folders and
`SW_4E_Replication_Data` into one folder; the examples below call it
`files/`.

## Paste the do-file

`sp.stata` takes the lines as they stand in the do-file. It runs the data
steps on a private copy of the DataFrame, applies `if` and `in` the way
Stata does, and returns the last result.

```python
import pandas as pd
import statspai as sp

fatality = pd.read_stata(
    "files/SW_4E_Replication_Data/fatality.dta", convert_categoricals=False
)

fe = sp.stata("""
    gen vfrall = 10000*mrall
    xtset state year
    xtreg vfrall beertax, fe vce(cluster state)
""", data=fatality)

fe.params["beertax"], fe.std_errors["beertax"]     # -0.6559, 0.2919
```

Those are the coefficient and the clustered standard error of equation
(10.15). Three things to know.

- **`use` is refused.** It replaces the data, so pass the DataFrame in.
  `set more off`, `log using`, `label` and `describe` are skipped.
- **`generate` stores single precision**, as Stata does unless the line says
  `double`. That is why the digits match. In your own work, compute the
  variable in pandas and you get double precision.
- **`convert_categoricals=False`** keeps Stata's numeric codes, which is
  what the do-files compare against (`s7==3`).

A longer example from chapter 11, with the marginal effect the book
computes by hand:

```python
hmda = pd.read_stata(
    "files/SW_4E_Replication_Data/hmda_sw.dta", convert_categoricals=False
)

effect = sp.stata("""
    gen deny = (s7==3)
    gen pi_rat = s46/100
    gen black = (s13==3)
    probit deny pi_rat black, r
    margins, dydx(black)
""", data=hmda)
```

## Check the numbers

A Stata log holds the commands that ran and what they printed, so it is its
own answer key. The replay script runs every logged command through one
`sp.stata` session and compares each coefficient, standard error, test
statistic, p-value, summary statistic and displayed scalar.

```bash
python scripts/stata_log_replay.py files --data files/SW_4E_Replication_Data
```

On the thirteen logs of chapters 2 to 13 it reproduces 1,300 printed numbers
and differs on none. The same check runs as a test when the files are
present:

```bash
STATSPAI_SW4E_DIR=files pytest tests/external_parity/test_stock_watson_4e_logs.py
STATSPAI_SW4E_DIR=files pytest tests/external_parity/test_stock_watson_4e_ch15.py
```

The second test covers chapter 15, whose program is in RATS. It compares
`sp.ardl`, `sp.unitroot` and `sp.structural_break` with the RATS output.

The replay works on any Stata log. It is the fastest way to find out whether
StatsPAI reproduces a project you already ran in Stata.

## What to add, chapter by chapter

The book's methods are still the right starting point. What has changed
since 2018 is what a referee expects next to them. The references for each
method are on the function's docstring (`sp.help("effective_f_test")`).

| Chapter | In the book | In StatsPAI | What a paper adds now |
| --- | --- | --- | --- |
| 3 | Difference in means, t test | `sp.ttest(df, "ahe", by="female", unequal=True)` | |
| 4 to 7 | OLS, robust standard errors, F tests | `sp.regress(..., robust="hc1")`, `sp.test` | |
| 8 | Polynomials, logs, interactions | formula terms in `sp.regress` | `sp.binscatter` before choosing a form; `sp.interflex` for an interaction |
| 9 | Threats to validity, as a checklist | | `sp.sensemakr`, `sp.oster_bounds` to put a number on omitted-variable bias |
| 10 | Entity and time fixed effects, clustered errors | `sp.feols("y ~ x \| state + year", cluster="state")` | Wild cluster bootstrap with few clusters; a staggered-adoption estimator when a policy starts at different dates |
| 11 | Probit and logit, effects at chosen values | `sp.probit`, `sp.logit` | `sp.margins` for average marginal effects |
| 12 | 2SLS, first-stage F above 10, J test | `sp.ivreg` | `sp.iv_diag`: effective F, Anderson-Rubin interval, tF interval |
| 13 | Experiments, difference-in-differences, regression discontinuity | `sp.did`, `sp.rdrobust` | Event-study estimators robust to heterogeneous effects, `sp.honest_did`, `sp.rddensity` |
| 14 | Ridge, lasso, principal components | `sp.lasso_select` | The same learners as nuisance models in `sp.dml` |
| 15 | AR and ADL forecasts, BIC, QLR, pseudo out-of-sample | `sp.ardl`, `sp.structural_break`, `sp.unitroot` | |
| 16 | Distributed lags with HAC errors | `sp.regress(..., robust="hac", hac_lags=m)` | `sp.local_projections` |
| 17 | VAR, DF-GLS, cointegration, GARCH | `sp.var`, `sp.unitroot(test="dfgls")`, `sp.engle_granger`, `sp.garch` | |

### Chapter 10: clustered errors with 48 states

The book clusters by state. With 48 clusters that is usually fine, and the
wild cluster bootstrap is the check.

```python
fatality["vfrall"] = 10000 * fatality.mrall
for v in ("vfrall", "beertax"):                       # within transformation
    fatality[v + "_w"] = fatality[v] - fatality.groupby("state")[v].transform("mean")

boot = sp.wild_cluster_bootstrap(
    fatality, y="vfrall_w", x=["beertax_w"], cluster="state",
    test_var="beertax_w", seed=1,
)
boot["p_cluster"], boot["p_boot"]                     # 0.029, 0.058
```

The clustered p-value is 0.029 and the bootstrap p-value 0.058. The beer-tax
coefficient is less sharply estimated than the conventional interval says.

### Chapter 12: is the instrument strong enough

The book's rule is a first-stage F above 10. That rule was derived for
homoskedastic errors. `sp.iv_diag` reports the measures that hold with
robust errors, and an interval that stays valid when the instrument is weak.

```python
import numpy as np

cig = pd.read_stata("files/SW_4E_Replication_Data/cig_ch12.dta")
c95 = cig[cig.year == 1995].copy()
c95["lpackpc"] = np.log(c95.packpc)
c95["lravgprs"] = np.log(c95.avgprs / c95.cpi)
c95["rtaxso"] = (c95.taxs - c95.tax) / c95.cpi
c95["lperinc"] = np.log(c95.income / (c95["pop"] * c95.cpi))

report = sp.iv_diag(c95, "lpackpc", "lravgprs", ["rtaxso"], exog=["lperinc"])
print(report.summary())
```

On the cigarette data the 2SLS elasticity is -1.14 (0.37), the effective F
is 44.7, and the Anderson-Rubin 95% set is [-1.85, -0.33], close to the Wald
interval. The instrument is strong here, and now there is a number that
says so.

### Chapter 15: forecasting GDP growth

`sp.ardl` fits the forecasting regression and carries the tools the chapter
uses around it.

```python
# macro: one row per quarter, indexed by a pandas PeriodIndex, with
# ygrowth = 400 * diff(log(GDPC1)) and rspread = GS10 - TB3MS
adl = sp.ardl(
    macro, "ygrowth", "rspread", lags=2, x_lags=2,
    sample=("1962Q1", "2017Q3"), vce="hc0",
)
adl.forecast()                  # 2.93 for 2017:Q4
adl.granger()["statistic"]      # F = 4.06: the term spread helps predict growth

sp.ardl(macro, "ygrowth", lags="bic", max_lags=6, sample=("1962Q1", "2017Q3")).lags   # 2

oos = sp.ardl(macro, "ygrowth", lags=1, sample=("1962Q1", "2017Q3")).poos("2007Q1")
oos.attrs["rmsfe"], oos.attrs["bias"]                 # 2.60, -1.09
```

The pseudo out-of-sample bias of -1.09 is three standard errors from zero.
The AR(1) over-forecast growth after 2007, which is the chapter's point
about breaks. The QLR test puts a date on one:

```python
lagged = macro.assign(
    yg1=macro.ygrowth.shift(1), yg2=macro.ygrowth.shift(2),
    rs1=macro.rspread.shift(1), rs2=macro.rspread.shift(2),
).loc["1962Q1":"2017Q3"]

qlr = sp.structural_break(
    lagged, y="ygrowth", x=["yg1", "yg2", "rs1", "rs2"], method="sup-f",
    break_vars=["const", "rs1", "rs2"], vce="hc1",
)
qlr.f_stats, lagged.index[qlr.sup_break - 1]          # 6.47, 1980Q4
```

`break_vars` holds the lags of growth fixed and lets the intercept and the
term-spread coefficients change, which is the test the book runs.

For a unit root, `sp.unitroot(macro.y.loc["1961Q2":"2017Q3"], trend="ct",
lags=2)` is the book's augmented Dickey-Fuller regression of log GDP
(statistic -1.95, far from the 5% critical value of -3.43). `test="dfgls"` is the
more powerful test of chapter 17. Its critical values here depend on the
sample size and the lag order; the asymptotic ones reject too often in a
few decades of quarterly data.

## What does not carry over

- **Chapters 16 and 17** have no Stata files in the bundle.
- **`xtreg, fe` prints `_cons`**, the average fixed effect. `sp.feols`
  absorbs it, so the result has slopes only.
- **Loops, `egen`, `merge`, `reshape`** are not run by `sp.stata`. Do those
  steps in pandas and pass the prepared DataFrame.
- **`pctile`** is not translated; `df.quantile` does the same job.
