# A design-based econometrics textbook in StatsPAI

Zhao Xiliang's 《基于设计的计量经济学》 (*Design-Based Econometrics*, second
edition) teaches causal inference by research design. It moves from
randomized experiments to matching, instrumental variables,
difference-in-differences, imputation and synthetic control, regression
discontinuity and mediation. The book ships Stata do-files for chapters 4 to
10 and an appendix.

Those do-files were run line by line in Stata 18 and in StatsPAI on the same
data. This guide records the result. It has three parts.

1. A chapter map from the Stata commands the book uses to the StatsPAI call.
2. The same analyses on datasets that ship with StatsPAI, with the numbers
   Stata gives.
3. The places where two correct programs print different numbers, and why.

The book's data are not redistributed with StatsPAI. Every number below comes
from a bundled dataset, and each is pinned by a test under
`tests/reference_parity/`.

## Chapter map

| Chapter | Stata in the do-files | StatsPAI |
| --- | --- | --- |
| 4 Randomized experiments | `ttest`, `reg, vce(hc2)`, hand-built stratified Neyman estimates, `ritest` | `sp.ttest`, `sp.regress(robust='hc2')`, `sp.difference_in_means(blocks=, cluster=)`, `sp.ri_test` |
| 5 Unconfoundedness | `teffects nnmatch`, `teffects psmatch`, `tebalance summarize`, `teffects ipw` / `aipw` | `sp.match(method='nnmatch')` and its `detail` table, `sp.match(method='psm')`, `sp.ipw`, `sp.aipw` |
| 6 Instrumental variables | `ivregress 2sls` / `liml`, `ivreg2`, `estat firststage`, `estat endogenous`, `weakivtest` | `sp.iv`, `sp.iv(method='liml')`, `sp.estat`, `sp.effective_f_test(y=)`, `sp.iv_diag` |
| 7 Difference-in-differences | `xtreg, fe`, `reghdfe`, `didregress`, `bacondecomp`, `csdid`, `xthdidregress`, `did_imputation`, `pretrends`, `honestdid` | `sp.panel`, `sp.feols`, `sp.didregress`, `sp.estat(result, 'ptrends')`, `sp.bacon_decomposition`, `sp.callaway_santanna`, `sp.etwfe`, `sp.did_imputation`, `sp.pretrends_power`, `sp.honest_did` |
| 8 Imputation and synthetic control | `synth`, `synth_runner`, `sdid`, `fect` | `sp.synth(v_method='regression')`, `sp.sdid`, `sp.fect` |
| 9 Regression discontinuity | `rdrobust`, `rdbwselect`, `rdplot`, `rddensity`, kink designs with `deriv(1)` and `scalepar()` | `sp.rdrobust`, `sp.rdbwselect`, `sp.rdplot`, `sp.rddensity`, `sp.rdrobust(deriv=1, scalepar=)` |
| 10 Causal mediation | `mediate` (Stata 18) | `sp.mediate(inference='robust')` |

## Chapter 4: experiments

A randomized experiment is analysed by its design. `sp.difference_in_means`
picks the variance from `blocks=` and `cluster=`.

```python
import numpy as np
import pandas as pd
import statspai as sp

rng = np.random.default_rng(0)
df = pd.DataFrame({"school": np.repeat(np.arange(8), 30)})
df["small"] = rng.permutation(np.tile([0, 1], 120))
df["score"] = 0.3 * df.school + 2.0 * df.small + rng.normal(size=240)

res = sp.difference_in_means(df, "score", "small", blocks="school")
res.model_info["design"], res.model_info["df"]      # 'Blocked', 224.0
res.detail                                          # one row per school

att = sp.difference_in_means(df, "score", "small", blocks="school",
                             estimand="ATT")
```

| Design | Arguments | Variance | Degrees of freedom |
| --- | --- | --- | --- |
| Complete randomization | none | `s1²/n1 + s0²/n0` | Satterthwaite |
| Blocked | `blocks=` | block estimates weighted by size | `N - 2J` |
| Matched pairs | `blocks=`, two units each | variance of pair differences | `J - 1` |
| Clustered | `cluster=` | CR2 | Bell-McCaffrey |
| Blocked and clustered | both | CR2 within each block | `G - 2J` |
| Matched pairs of clusters | both, two clusters each | size-weighted pair differences | `J - 1` |

All six agree with R `estimatr::difference_in_means` to 1e-11. The book's
stratified estimates for the class-size experiment, which its do-file builds
by hand, come out of the `blocks=` call with `estimand='ATE'` and `'ATT'` to
the last digit Stata prints.

Without blocks the standard error equals the HC2 one from
`sp.regress("score ~ small", df, robust="hc2")`. That identity is the reason
the book reports HC2.

## Chapter 5: matching

`teffects nnmatch` is `sp.match(method='nnmatch')`.

```python
nsw = sp.datasets.nsw_dw()
X = ["age", "education", "black", "hispanic", "married", "nodegree",
     "re74", "re75"]

m = sp.match(nsw, y="re78", treat="treat", covariates=X,
             method="nnmatch", estimand="ATT")
m.estimate, m.se            # 367.7272, 1165.5767
m.detail                    # tebalance summarize

m2 = sp.match(nsw, y="re78", treat="treat", covariates=X,
              method="nnmatch", estimand="ATT",
              exact=["black"], bias_adjust=["age", "re74"])
m2.estimate, m2.se          # 1968.9053, 1206.4973
```

| Stata option | Argument |
| --- | --- |
| `atet` | `estimand='ATT'` |
| `nn(k)` | `n_matches=k` |
| `metric(ivariance)` / `metric(euclidean)` | `metric=` |
| `ematch(varlist)` | `exact=[...]` |
| `biasadj(varlist)` | `bias_adjust=[...]` |
| `caliper(#)` | `caliper=` |
| `vce(robust, nn(h))` / `vce(iid)` | `vce_nn=h` / `vce='iid'` |

Nine configurations agree with Stata 18 to 5e-13 in the estimate and the
standard error. Ties are kept, as in Stata. The older
`sp.match(distance='mahalanobis')` is a different estimator and gives
different numbers.

## Chapter 6: instrumental variables

```python
card = sp.datasets.card_1995()
formula = ("lwage ~ exper + expersq + black + south + smsa"
           " + (educ ~ nearc4 + nearc2)")

iv = sp.iv(formula, card, robust="robust", small=False)
iv.params["educ"], iv.std_errors["educ"]        # 0.1608, 0.0485

liml = sp.iv(formula, card, method="liml")

f = sp.effective_f_test(card, "educ", ["nearc4", "nearc2"],
                        ["exper", "expersq", "black", "south", "smsa"],
                        y="lwage")
f["F_eff"]                                      # 9.6428
f["critical_values"]["tsls"][0.10]              # 4.0586
f["critical_values"]["liml"][0.10]              # 12.5459

sp.estat(iv, "endogenous")                      # robust F(1, 3002) = 3.978
```

Three things to know.

- **`small=False` is `ivregress` without `small`.** Stata's default divides
  by `N` and reports z statistics. `sp.iv` on its own reports the
  small-sample version, which is `ivregress ..., small`.
- **The effective F has no single threshold.** The Montiel Olea-Pflueger
  critical value depends on the estimator and on the data. Here TSLS passes
  at a 10% bias tolerance and LIML does not. The familiar 23.1 is the value
  for one instrument.
- **Factor terms work in IV formulas.** `C(qob)` and interactions can be
  exogenous regressors or instruments, which is what the
  quarter-of-birth example needs.

## Chapter 7: difference-in-differences

```python
mp = sp.datasets.mpdta()

cs = sp.callaway_santanna(mp, y="lemp", g="first_treat", t="year",
                          i="countyreal")
et = sp.etwfe(mp, y="lemp", group="countyreal", time="year",
              first_treat="first_treat")
bjs = sp.did_imputation(mp, y="lemp", group="countyreal", time="year",
                        first_treat="first_treat")
```

`didregress` and `xtdidregress` are `sp.didregress`, and the two tests the
book runs after them are `sp.estat`.

```python
two = mp[mp.first_treat.isin([0, 2006])].copy()
two["d"] = ((two.first_treat > 0) & (two.year >= two.first_treat)).astype(int)

did = sp.didregress(two, "lemp", "d", group="countyreal", time="year")
did.estimate, did.se                    # -0.0300, 0.0103
sp.estat(did, "ptrends")                # F(1, 249) = 1.61
sp.estat(did, "granger")                # F(2, 249) = 0.86

xt = sp.didregress(two, "lemp", "d", group="countyreal", time="year",
                   id="countyreal")     # xtdidregress: se 0.0092
```

The book's simulated staggered panel has no never-treated unit. From the
last cohort's adoption on there is nothing to compare with. `sp.etwfe` now
drops those periods, takes the last cohort as the reference and says so in a
warning. That is what R `etwfe` does, and the numbers agree to 1e-11.

A lead and lag regression written by hand can go straight into the
pre-trend tools. Stata's `pretrends` and `honestdid` read `e(b)` and `e(V)`,
and so do these.

```python
names = ["lead4", "lead3", "lead2", "lag0", "lag1", "lag2", "lag3"]
fit = sp.feols("y ~ " + " + ".join(names) + " | id + t", panel,
               vcov={"CRV1": "id"})
times = dict(zip(names, [-4, -3, -2, 0, 1, 2, 3]))

sp.pretrends_power(fit, slope=0.03, event_times=times)
sp.pretrends_slope_for_power(fit, event_times=times)

beta = fit.params[names].to_numpy()
sigma = fit.vcov().loc[names, names].to_numpy()
sp.honest_did_from_moments(beta, sigma, num_pre_periods=3,
                           method="smoothness", m_grid=[0, 0.1, 0.2])
```

## Chapter 8: synthetic control

Stata's `synth` without `nested` does not search for the predictor weights.
It takes them from a regression. That is `v_method='regression'`.

```python
prop99 = sp.california_prop99()
spec = [("packspercapita", 1975, "mean"),
        ("packspercapita", 1980, "mean"),
        ("packspercapita", 1988, "mean"),
        ("packspercapita", slice(1970, 1974), "mean")]

sc = sp.synth(prop99, "packspercapita", "state", "year", "California", 1989,
              method="classic", special_predictors=spec,
              v_method="regression")
sc.estimate, sc.pvalue          # -18.806, 0.0256
```

The predictor weights agree with `e(V_matrix)` to 1e-10 and the
pre-treatment RMSPE to 2e-10. The fit with its 38 placebo fits takes about a
second.

## Chapter 9: regression discontinuity

```python
lee = sp.datasets.lee_2008_senate()

rd = sp.rdrobust(lee, "y", "x")
sp.rdbwselect(lee, "y", "x", all=True)
den = sp.rddensity(lee, "x")
den.model_info["conventional"]          # rddensity, all
den.model_info["binomial_tests"]        # the binomial table Stata prints
sp.rdplot(lee, "y", "x")
```

A kink design is `deriv=1`, and `scalepar=` rescales the estimate by the
kink in the policy rule, as Stata's `scalepar()` does.

## Chapter 10: mediation

```python
med = sp.mediate(df, y="y", treat="d", mediator="m", covariates=["x"],
                 inference="robust", interaction=True)
```

This is Stata 18's `mediate`. It reports the indirect and direct effects
(NIE, NDE, PNIE, TNDE), the total effect and the proportion mediated, with
standard errors from the stacked estimating equations.
`mediator_model='logit'` or `'probit'` handles a binary mediator.

## Pasting the do-file

`sp.stata` runs Stata lines on a DataFrame, and `sp.from_stata` shows the
StatsPAI call without running it.

```python
sp.from_stata("teffects nnmatch (re78 age education black) (treat), "
              "atet nn(3) ematch(black)")["python_code"]

sp.stata("""
    tsset sid year
    synth packspercapita packspercapita(1975) packspercapita(1988), ///
          trunit(3) trperiod(1989)
""", data=prop99_with_sid)
```

On the book's 13 do-files `sp.stata` runs 88% of the estimation commands as
written. The rest are refused with a reason. They are loops over macros,
`xthdidregress`, `synth_runner` and one factor term. A command with no
line-by-line translation, such as `ritest`, `honestdid`, `pretrends`,
`weakivtest` or `fect`, comes back with the name of the function to call.

## Where two correct programs differ

| What you see | Why |
| --- | --- |
| `ivregress` standard errors smaller than `sp.iv` | Stata's default is the large-sample variance. Pass `small=False`. |
| `synth` post-treatment gap of -18.70 against -18.81 | Stata stores donor weights rounded to three decimals and builds its synthetic path from them. They sum to 0.999 here. Rounding our weights reproduces its path to 1e-9. |
| `mediate` standard errors differ in the fourth digit | Stata's variance uses numerical derivatives. Its standard errors move by 1e-4 when a covariate is centred, which is an equivalent model. The analytic ones here do not move. |
| `weakivtest` 30% critical values differ by 6e-4 | It evaluates the non-centrality at 3.33 where the definition is 1/0.3. |
| `sdid ..., covariates(x)` differs by 3e-4 | Stata's default `optimized` method and R `synthdid` both stop a gradient iteration at 10,000 steps, at different points. `covariate_method='optimized'` follows R. `'projected'` is a regression and agrees with Stata. |
| `estat ptrends` differs by 3e-6 | The trend on raw years is nearly collinear with the treatment dummy. Stata's own `areg` with centred time gives the value here. |
| `pretrends` power differs by 2e-4 | Both sides integrate a multivariate normal probability numerically. The likelihood ratio, which is closed form, agrees to 1e-13. |
| `teffects ra` and `ipwra` are refused by `sp.stata` | They have no StatsPAI call with the same estimator. Translating them to a regression would change the estimand. |

## Fixes this exercise produced

Running a textbook is a cheap way to find bugs, because the book says what
the answer should be. Six came out of this one.

- `sp.etwfe` returned arbitrary numbers when every unit is eventually
  treated.
- Data-driven RD bandwidths were off by about 1e-4 when the interquartile
  range set the pilot bandwidth.
- The LIML `kappa` lost digits when it was close to 1.
- `sp.cr2_se` used approximate degrees of freedom, and the smallest one for
  every coefficient.
- `sp.estat(result, 'endogenous')` returned the homoskedastic test after a
  robust fit.
- `didregress` was translated to a 2x2 on collapsed periods.

Each is in the changelog with the reference value and the test that pins it.
