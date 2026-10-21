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
