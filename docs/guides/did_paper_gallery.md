# DiD paper gallery: five published applications, reproduced

Five difference-in-differences papers, each rerun in a few lines of
`import statspai as sp`. Every number below is checked in CI by
`tests/external_parity/test_did_paper_gallery.py` against two references:

- what the paper (or the authors' replication output) prints, at the
  decimals it prints;
- Stata 18 MP run on the same data bytes, at full precision.

The reference numbers and their provenance are in
`tests/external_parity/data/did_paper_gallery.json`. Every call passes its
conventions explicitly (base period, estimator, control group, share
variance), so the numbers do not depend on StatsPAI's defaults.

## Getting the data

The castle-doctrine panel ships with StatsPAI. The other four datasets are
real public files that are not in the wheel: one is too large (6.3 MB) and
three come without a licence that clearly allows redistribution. Their
loaders download the file from its canonical URL on first use, check a
pinned SHA-256, and cache it.

| Loader | Paper | Delivery | Source |
| --- | --- | --- | --- |
| `sp.datasets.jel_medicaid()` | Baker et al. [@baker2026difference] | fetched, 6.3 MB | authors' repository `pedrohcgs/JEL-DiD` (MIT) |
| `sp.datasets.lalonde_panel()` | Sant'Anna and Zhao [@santanna2020doubly] | fetched, 0.9 MB | Stata `drdid` example data |
| `sp.datasets.bacon_divorce()` | Goodman-Bacon [@goodmanbacon2021difference] | fetched, 81 kB | Stata `bacondecomp` example data |
| `sp.datasets.mpdta(simulated=False)` | Callaway and Sant'Anna [@callaway2021difference] | fetched, 67 kB | Stata `csdid` example data (same values as R `did`'s `mpdta`) |
| `sp.datasets.castle_doctrine()` | Cheng and Hoekstra [@cheng2013does] | shipped | Mixtape repository [@cunningham2021causal] |

The cache is `~/.cache/statspai/datasets` (`%LOCALAPPDATA%\statspai\datasets`
on Windows); set `STATSPAI_DATA_DIR` to move it. Nothing downloads at
import time, and `sp.list_datasets()` never downloads. Its `delivery`
column says `shipped` or `fetched`, and `available_offline` says whether a
fetched file is already cached. Offline, a fetched loader raises
`DataInsufficient` with the URL, the SHA-256 and the path to save the file
at, so it can be placed there by hand. Pass `download=False` to forbid the
network.

Note that `sp.datasets.mpdta()` without arguments is still the simulated
replica (it warns once per process), and `sp.datasets.nsw_lalonde()` is
the 614-row MatchIt cross-section, not the two-period panel used below.

## 1. Baker, Callaway, Cunningham, Goodman-Bacon and Sant'Anna: Medicaid and mortality

County mortality of adults aged 20-64, 2009-2019; 2014 expansion states
against states that had not expanded by 2019 [@baker2026difference].

```python
import statspai as sp

d = sp.datasets.jel_medicaid(sample="2x2")      # the authors' cleaning
s = d[d.year.isin([2013, 2014])].assign(
    Treat=lambda x: (x.treat_year == 2014).astype(int),
    Post=lambda x: (x.year == 2014).astype(int),
)
s["D"] = s.Treat * s.Post
sp.feols("crude_rate_20_64 ~ D + Treat + Post", s, weights="set_wt",
         vcov={"CRV1": "county_code"})           # -2.56 (1.49)

# Table 7: covariates frozen at 2013, population weights
covs = ["perc_female", "perc_white", "perc_hispanic",
        "unemp_rate", "poverty_rate", "median_income"]
x13 = s[s.year == 2013].set_index("county_code")[covs]
for c in covs:
    s[c] = s.county_code.map(x13[c])
sp.callaway_santanna(s, y="crude_rate_20_64", g="treat_year", t="year",
                     i="county_code", x=covs, weights="set_wt",
                     estimator="dr", control_group="nevertreated",
                     base_period="varying")      # -3.76
```

| Quantity | Paper | Stata | StatsPAI |
| --- | --- | --- | --- |
| 2x2, unweighted / weighted (Table 5 cols 1, 4) | 0.12 (3.75) / -2.56 (1.49) | 0.121630 / -2.562875 | same to 6 decimals |
| Regression / IPW / DR, weighted (Table 7) | -3.46 / -3.84 / -3.76 | -3.459222 / -3.841699 / -3.756107 | -3.459220 / -3.841697 / -3.756105 |
| Regression / IPW / DR, unweighted (Table 7) | -1.62 / -0.86 / -1.23 | not run | rounds to the paper |
| TWFE with time-varying covariates (Table 5 col 6) | -1.37 | -1.369975 | -1.369979 |
| Staggered TWFE, weighted / unweighted | | -0.935153 / -1.011909 | same to 6 decimals |

The last digits of the covariate rows differ because the Stata do-file
creates the covariates with `gen`, which stores them as float (about 7
significant digits). The Table 5 standard error of column 6 (1.62) comes
from R; Stata `reghdfe` and StatsPAI both give 1.6149.

## 2. Sant'Anna and Zhao (2020): doubly robust DiD on LaLonde

NSW randomized-out controls against CPS comparison units, so the true
effect is zero and every estimate is an evaluation bias
[@santanna2020doubly; @lalonde1986evaluating; @dehejia1999causal].

```python
import statspai as sp

df = sp.datasets.lalonde_panel()
df["post"] = (df.year == 1978).astype(int)
d = df[((df["sample"] == 1) & (df.treated == 0)) | (df["sample"] == 2)]
x = ["age", "educ", "black", "married", "nodegree", "hisp", "re74"]
sp.drdid(d, y="re", treat="experimental", time="post", id="id",
         covariates=x, method="trad")            # -871.33 (396.02)
sp.drdid(d, y="re", treat="experimental", time="post", id="id",
         covariates=x, method="imp")             # -901.27 (393.61)
```

All 54 cells of Table 3 (three samples, three covariate specifications,
six estimators) round to the published dollars, point estimates and
standard errors. Against Stata `drdid` on all 81 estimates of the
replication the largest gap is $0.016 on an ATT near $1,000 (the logit
stopping rule in the propensity score) and under $0.01 on the SE.

## 3. Goodman-Bacon (2021): divorce reform and female suicide

Unilateral divorce reform and the age-adjusted female suicide rate,
49 states, 1964-1996 [@goodmanbacon2021difference; @goodmanbacon2019bacondecomp].

```python
import statspai as sp

df = sp.datasets.bacon_divorce()
bd = sp.bacon_decomposition(df, y="asmrs", treat="post", time="year",
                            id="stfips")
bd["beta_twfe"]          # -3.0799
bd["summary"]            # four comparison types, 156 2x2s
```

| Quantity | Paper | Stata `bacondecomp` | StatsPAI |
| --- | --- | --- | --- |
| TWFE DD | -3.08 (robust s.e. 1.27) | -3.079926 | -3.079926 |
| Later vs always treated: weight / DD | 0.38 / -7.04 | 0.384432 / -7.043689 | same |
| Treated vs never treated | 0.24 / -5.33 | 0.240270 / -5.330907 | same |
| Later vs earlier treated | 0.26 / 3.51 | 0.264644 / 3.511967 | same |
| Earlier vs later treated | 0.11 / -0.19 | 0.110654 / -0.186770 | same |

All 156 two-by-two estimates and weights agree with `bacondecomp`
(estimates to 1e-4, the float precision of Stata's output; weights to
1e-6). Without the eight states reformed before 1964, the
Callaway-Sant'Anna simple ATT is -10.22 (3.09) in both Stata `csdid` and
StatsPAI.

## 4. Callaway and Sant'Anna (2021): minimum wage and teen employment

The `mpdta` panel: 500 counties, 2003-2007 [@callaway2021difference].

```python
import statspai as sp

df = sp.datasets.mpdta(simulated=False)
cs = sp.callaway_santanna(df, y="lemp", g="first_treat", t="year",
                          i="countyreal", estimator="dr",
                          control_group="nevertreated",
                          base_period="varying")
for agg in ["simple", "group", "calendar", "dynamic"]:
    print(agg, sp.aggte(cs, type=agg).estimate)
```

| Aggregation | R `did` 2.5.1 | Stata `csdid` | StatsPAI |
| --- | --- | --- | --- |
| simple | -0.0400 (0.0120) | -0.0400 (0.0120) | -0.0400 (0.0120) |
| group | -0.0310 (0.01245) | -0.0310 (0.01239) | -0.0310 (0.01245) |
| calendar | -0.0417 (0.0160) | -0.0417 (0.0160) | -0.0417 (0.0160) |
| dynamic | -0.0772 (0.0200) | -0.0772 (0.0200) | -0.0772 (0.0200) |

Point estimates and the 12 ATT(g,t) agree with both references to
machine precision; analytic SEs agree with R (`bstrap=FALSE`) to 1e-8.
The group SE differs only in Stata: `csdid` treats the cohort shares as
known, R and StatsPAI add their sampling variance.
`sp.aggte(cs, type="group", share_variance=False)` gives Stata's 0.01239.
The pre-trend Wald statistic is 7.79 on 5 degrees of freedom everywhere,
with a chi-squared p-value of 0.1681.

## 5. Cheng and Hoekstra (2013): castle doctrine and homicide

Population-weighted log homicide, 50 states, 2000-2010 [@cheng2013does],
in the version reprinted as Table 9.7 of the Mixtape [@cunningham2021causal].

```python
import statspai as sp

d = sp.datasets.castle_doctrine()
d["region"] = d.northeast + 2 * d.midwest + 3 * d.south + 4 * d.west
sp.feols("l_homicide ~ cdl | sid + year", d, weights="popwt",
         vcov={"CRV1": "sid"})                    # 0.0801 (0.0342)
sp.feols("l_homicide ~ cdl | sid + region^year", d, weights="popwt",
         vcov={"CRV1": "sid"})                    # 0.0946 (0.0279)
```

Both columns match the published table to its four decimals and Stata
`reghdfe` to 1e-8 relative. The shipped CSV is the Mixtape `castle.dta`
with `popwt` written as float64 decimals, which moves nothing above 1e-7.
With `gvar = effyear`, never-treated controls and a universal base
period, the Callaway-Sant'Anna simple ATT is 0.1082 (0.0245) in both
Stata `csdid` and StatsPAI.

## Running the checks

```bash
STATSPAI_FETCH_DATASETS=1 pytest tests/external_parity/test_did_paper_gallery.py
```

Without `STATSPAI_FETCH_DATASETS=1` the tests that need a fetched file run
only if it is already cached, and otherwise skip with the reason. The
nightly CI job sets the variable.
