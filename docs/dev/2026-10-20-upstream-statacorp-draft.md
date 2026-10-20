# Draft notes to StataCorp technical support (four items)

Status: draft, not sent. Bryce sends these himself, one message per item.
Each item stands alone and needs only Stata 18.

Before sending, run `update all` and rerun the snippets. The machine used
here runs Stata/MP 18.0, revision 07 Jun 2023, which is old. If an item no
longer reproduces after the update, drop it.

The evidence behind each item in this repository is listed below.

| Item | Committed evidence | Module note |
| --- | --- | --- |
| 1. `stteffects ipwra, atet` | `tests/reference_parity/test_survival_causal_parity.py`, fixtures `_fixtures/stteffects_data.csv`, `_fixtures/stteffects_Stata.csv` | `docs/dev/modules/survival.md` |
| 2. `xtintreg` standard errors | `tests/test_xttobit_stata.py`, `tests/fixtures/xttobit/` | `docs/dev/modules/panel_multilevel.md` |
| 3. `gsdesign` ESS, nonbinding futility | `tests/reference_parity/test_group_sequential_stata_parity.py`, `_fixtures/gsdesign_ess_Stata.json` | `docs/dev/modules/experiments_interference.md` |
| 4. `xtregar, fe` `_cons` SE | `tests/test_xtregar_stata.py`, `tests/fixtures/xtregar/` | `docs/dev/modules/panel_multilevel.md` |

The self-contained snippets below were run once on 2026-10-20 to check that
they execute and show the pattern. The numbers quoted from those runs are
Stata's own output under the stated seed.

---

## Item 1

Subject: stteffects ipwra, atet appears to average over the whole sample

Stata/MP 18.0, macOS.

`stteffects ipwra ..., atet` seems to average the treated-weighted outcome
models' predictions over all observations, not over the treated. When the
outcome model is correct this estimates the ATE. `stteffects ra ..., atet`
and `stteffects ipw ..., atet` do average over the treated.

Reproduction. Weibull outcomes with a treatment effect that varies with
`x1`, and treatment selected on `x1`, so the ATE and the ATET differ.

```stata
version 18
clear all
set seed 20261020
set obs 4000
gen double x1 = rnormal()
gen byte   x2 = runiform() < 0.5
gen byte   a  = runiform() < invlogit(1.2*x1)
* Weibull AFT outcome, shape s = 1.5; mean exp(xb)*Gamma(1 + 1/s)
local s = 1.5
gen double xb0 = 0.0*x1 - 0.2*x2 + 1.0
gen double xb1 = 0.6*x1 - 0.2*x2 + 1.5
gen double m0 = exp(xb0 + lngamma(1 + 1/`s'))
gen double m1 = exp(xb1 + lngamma(1 + 1/`s'))
gen double T = exp(cond(a, xb1, xb0)) * (-ln(runiform()))^(1/`s')
gen double C = exp(3) * (-ln(runiform()))^(1/1.5)
gen double time  = min(T, C)
gen byte   event = T <= C
* sample estimands from the true potential-outcome means
gen double te = m1 - m0
quietly summarize te
display "true sample ATE  = " %9.4f r(mean)
quietly summarize te if a == 1
display "true sample ATET = " %9.4f r(mean)
stset time, failure(event)
stteffects ra    (x1 x2) (a), atet
stteffects ipwra (x1 x2) (a x1 x2)
stteffects ipwra (x1 x2) (a x1 x2), atet
```

Output on our machine.

| Quantity | Value |
| --- | --- |
| true sample ATE | 2.2506 |
| true sample ATET | 3.5395 |
| `stteffects ra, atet` | 3.4056 (SE 0.1432) |
| `stteffects ipwra` (ATE) | 2.1766 (SE 0.1017) |
| `stteffects ipwra, atet` | 2.1948 (SE 0.0972) |

The RA estimate of the ATET is close to the ATET. The IPWRA estimate of
the ATET is close to the ATE and about 14 standard errors from the ATET.

A second data set gives the same picture. On 800 observations
([`stteffects_data.csv`](https://raw.githubusercontent.com/brycewang-stanford/StatsPAI/6fdc22bef90d03defabfc45b5b26efd9cde60e8f/tests/reference_parity/_fixtures/stteffects_data.csv),
variables `time event a x1 x2`, `stset time, failure(event)`) Stata reports 2.3146 for `ra, atet` and 1.7575 for
`ipwra, atet`, against 1.8010 for the IPWRA ATE. We took the coefficients
of the two outcome models that `ipwra, atet` fits and averaged their
predicted means ourselves. Averaging over all 800 rows gives 1.7575, which
matches Stata to 1e-9. Averaging over the 320 treated rows gives 2.2684.
The same holds with a censoring model, `stteffects ipwra (x1 x2) (a x1 x2) (x1), atet`.
Stata reports 1.4122, the all-rows average reproduces it, and the treated
average is 1.6542.

As we read Methods and formulas for `stteffects`, the moment conditions
for the potential-outcome means under `ipwra` with `atet` (equations 15e
and 15f) carry no treatment indicator. That would explain the numbers.

Is this intended? If so, the ATET label seems misleading. If not, the
averages should run over the treated observations.

---

## Item 2

Subject: xtintreg standard errors under adaptive quadrature differ from the curvature of e(ll)

Stata/MP 18.0, macOS.

With the default adaptive quadrature (`mvaghermite`), the standard errors
that `xtintreg` reports differ by about 0.5% from the inverse curvature of
the log likelihood it reports. With non-adaptive quadrature
(`intmethod(ghermite)`) they agree. `e(user)` is `xtintreg_d2`, so the
variance comes from the analytic Hessian. Our guess is that this Hessian
does not account for the adaptive change of variables. We have not
checked that guess.

Reproduction. A simulated unbalanced panel, then a central-difference
Hessian of `e(ll)`, each evaluation from `from(b, copy) iterate(0)`.

```stata
version 18
clear all
set seed 20261019
* 120 unbalanced panels (2 to 7 periods), normal panel effect,
* outcome coded into unit-width intervals, a fifth exact, left-censored at -1
set obs 120
gen int id = _n
gen double nu = 0.8*rnormal()
gen int Ti = 2 + floor(6*runiform())
expand Ti
bysort id: gen int t = _n
gen double x1 = rnormal()
gen double x2 = rnormal()
gen double ys = 0.7 + 1.0*x1 - 0.6*x2 + nu + rnormal()
gen double lo = floor(ys)
gen double hi = lo + 1
gen byte exact = runiform() < 0.2
replace lo = ys if exact
replace hi = ys if exact
replace lo = . if ys < -1
replace hi = -1 if ys < -1
xtset id t

capture program drop curv_se
program define curv_se
    * central-difference Hessian of e(ll) at b0, from() iterate(0)
    args b0 opts
    tempname b H V f0 fpp fpm fmp fmm
    local k = colsof(`b0')
    matrix `H' = J(`k', `k', 0)
    forvalues i = 1/`k' {
        local hi_ = 1e-4*max(1, abs(`b0'[1,`i']))
        forvalues j = `i'/`k' {
            local hj_ = 1e-4*max(1, abs(`b0'[1,`j']))
            foreach s in pp pm mp mm {
                matrix `b' = `b0'
                local si = cond(substr("`s'",1,1)=="p", 1, -1)
                local sj = cond(substr("`s'",2,1)=="p", 1, -1)
                matrix `b'[1,`i'] = `b'[1,`i'] + `si'*`hi_'
                matrix `b'[1,`j'] = `b'[1,`j'] + `sj'*`hj_'
                quietly xtintreg lo hi x1 x2, from(`b', copy) iterate(0) `opts'
                scalar `f`s'' = e(ll)
            }
            matrix `H'[`i',`j'] = (`fpp' - `fpm' - `fmp' + `fmm')/(4*`hi_'*`hj_')
            matrix `H'[`j',`i'] = `H'[`i',`j']
        }
    }
    matrix `V' = invsym(-1*`H')
    forvalues i = 1/`k' {
        display "  curvature se, parameter `i': " %10.7f sqrt(`V'[`i',`i'])
    }
end

foreach opts in "intpoints(12)" "intpoints(40)" "intmethod(ghermite) intpoints(100)" {
    display _n "=== `opts'"
    quietly xtintreg lo hi x1 x2, `opts'
    matrix b0 = e(b)
    matrix V0 = e(V)
    display "  e(user) = `e(user)'   e(ll) = " %15.8f e(ll)
    forvalues i = 1/5 {
        display "  reported  se, parameter `i': " %10.7f sqrt(V0[`i',`i'])
    }
    curv_se b0 "`opts'"
}
```

Output for `/sigma_e` (parameter 5).

| Quadrature | Reported SE | Curvature SE |
| --- | --- | --- |
| `intpoints(12)`, adaptive (default) | 0.0428876 | 0.0431161 |
| `intpoints(40)`, adaptive | 0.0428876 | 0.0431161 |
| `intmethod(ghermite) intpoints(100)` | 0.0431161 | 0.0431161 |

The point estimates and `e(ll)` agree across the three fits. The other
four parameters show the same pattern with smaller gaps (4e-7 to 8e-6).

We saw the same on a second panel (540 observations). There the default
fit reports se(sigma_e) = 0.0387655, while the curvature of `e(ll)` and the
non-adaptive fit both give 0.0385837.

Is the analytic Hessian meant to be exact under adaptive quadrature?

---

## Item 3

Subject: gsdesign with nonbinding futility reports an expected sample size above the maximum

Stata/MP 18.0, macOS.

```stata
gsdesign onemean 10 15, sd(12) onesided power(0.9) futility(errpocock) ///
    nfractional intpointsscale(60) infotol(1e-12)
return list
```

This returns `r(N_max) = 59.1659` and `r(ESS1) = 59.4171`. The default
tolerances give the same picture (59.1656 and 59.4168). An expected sample
size cannot exceed the maximum sample size.

We can reproduce Stata's `r(ESS0)` and `r(ESS1)` to 1e-7, for this design
and for eight other designs with nonbinding futility bounds, with this
rule. The weight on look k is the efficacy-crossing probability computed
with the futility bounds ignored, plus the futility-crossing probability
computed with the futility bounds followed. A path that crosses a futility
bound and would later cross the efficacy bound then counts twice, and the
weights sum to more than one.

For efficacy-only designs and for designs with binding futility, our
expected sample sizes match Stata's to 1e-7. For nonbinding designs we
compute the expected sample size when the futility rule is followed.
Those stopping probabilities sum to one. On one of these designs a
simulation of 200,000 Brownian paths agrees with it within 4 Monte Carlo
standard errors. For the design
above that gives ESS1 = 57.33 and ESS0 = 36.77 (Stata 59.42 and 37.23).

As we read Methods and formulas, it describes a third rule. The futility
bound is dropped when delta = 0 and kept otherwise. That rule reproduces
neither of Stata's two numbers.

Which quantity is `r(ESS1)` meant to be under nonbinding futility?

---

## Item 4

Subject: xtregar, fe reports a standard error of _cons that is too small by a factor 1 - rho

Stata/MP 18.0, macOS.

After `xtregar, fe`, `_b[_cons]` is the intercept of the within regression
on the transformed data divided by `1 - rho`. The reported `_se[_cons]`
appears not to be divided. It is smaller than the sampling standard
deviation of `_b[_cons]` by about `1 - rho`.

Reproduction. A fixed design (regressor and panel effects drawn once),
AR(1) errors with rho = 0.6, and `rhof(0.6)` so that rho is not estimated.

```stata
version 18
capture program drop xtar_draw
clear all
set seed 1
local G 40
local T 8
local rho 0.6
* fixed design: regressor and panel effects drawn once
set obs `G'
gen int id = _n
gen double nu = rnormal()
expand `T'
bysort id: gen int t = _n
gen double x = rnormal()
tempfile base
save `base'
program define xtar_draw, rclass
    args base rho
    use `base', clear
    xtset id t
    gen double e = rnormal()/sqrt(1 - `rho'^2) if t == 1
    by id: replace e = `rho'*e[_n-1] + rnormal() if t > 1
    gen double y = 1 + 2*x + nu + e
    xtregar y x, fe rhof(`rho')
    return scalar b  = _b[_cons]
    return scalar se = _se[_cons]
end
simulate b=r(b) se=r(se), reps(300) nodots: xtar_draw `base' `rho'
quietly summarize b
local sd = r(sd)
quietly summarize se
display "mean reported se(_cons) = " %8.4f r(mean)
display "Monte Carlo sd(_cons)   = " %8.4f `sd'
display "ratio                   = " %8.4f r(mean)/`sd'
display "1 - rho                 = " %8.4f 1 - `rho'
```

It printed a Monte Carlo SD of 0.1464 and a ratio of 0.4098, against
`1 - rho = 0.4`. With 300 replications the Monte Carlo error of an SD is
about 4%.

On a second, unbalanced panel with gaps we matched every other `xtregar`
result to about 1e-15 across 29 runs. These cover all rho types, `twostep`, `rhof()`,
`lbi`, FE and RE. The FE `_cons` SE is the one exception, and there
Stata's value equals the divided SE times `1 - rho` to 1e-10.

Is the undivided standard error intended?
