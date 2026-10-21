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
