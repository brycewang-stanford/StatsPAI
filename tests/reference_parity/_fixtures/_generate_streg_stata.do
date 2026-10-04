* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_survreg_streg_stata_parity.py
*
* Requires: Stata 18 (streg is official).
* Run:      stata -b do _generate_streg_stata.do   (from this directory)
*
* `streg ..., time` is the accelerated failure-time metric sp.survreg uses.
* Recorded: the four distributions, vce(oim), vce(robust), vce(cluster), and
* gamma frailty with the likelihood-ratio test of theta = 0.
*
* Durations come from a Weibull with gamma heterogeneity, so the frailty
* variance is away from its boundary and the plain Weibull is misspecified.
* Data is generated and exported HERE so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20261006
set obs 1500

gen double x1 = rnormal()
gen byte   x2 = runiformint(0, 1)
gen int    clust = mod(_n, 75) + 1
gen double v  = rgamma(1/0.6, 0.6)
* Weibull AFT: ln t = 0.5 + 0.4 x1 - 0.3 x2 + sigma * w, sigma = 0.7, with
* the hazard multiplied by v.
gen double tstar = exp(0.5 + 0.4*x1 - 0.3*x2) * (-ln(runiform()) / v)^0.7
gen double cens  = rexponential(6)
gen double time  = min(tstar, cens)
gen byte   event = tstar <= cens
drop v tstar cens

format x1 time %21.16e
export delimited time event x1 x2 clust using "streg_data.csv", replace datafmt

capture program drop wr
program define wr
    args fh tag last
    tempname b V
    matrix `b' = e(b)
    matrix `V' = e(V)
    local names : colfullnames `b'
    local k = colsof(`b')
    file write `fh' `"  "`tag'": {"' _n
    file write `fh' `"    "names": ["'
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        file write `fh' `""`nm'""'
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]," _n `"    "b": ["'
    forvalues j = 1/`k' {
        file write `fh' %21.16e (`b'[1, `j'])
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]," _n `"    "se": ["'
    forvalues j = 1/`k' {
        file write `fh' %21.16e (sqrt(`V'[`j', `j']))
        if `j' < `k' file write `fh' ", "
    }
    file write `fh' "]," _n
    if "`e(ll)'" != "" file write `fh' `"    "ll": "' %21.16e (e(ll)) "," _n
    if "`e(N_clust)'" != "" file write `fh' `"    "n_clusters": "' %21.16e (e(N_clust)) "," _n
    if "`e(theta)'" != "" file write `fh' `"    "theta": "' %21.16e (e(theta)) "," _n
    if "`e(chi2_c)'" != "" file write `fh' `"    "chi2_c": "' %21.16e (e(chi2_c)) "," _n
    if "`e(p_c)'" != "" file write `fh' `"    "p_c": "' %21.16e (e(p_c)) "," _n
    file write `fh' `"    "N": "' %21.16e (e(N)) _n
    file write `fh' "  }," _n
end


* ml stops at nrtolerance(1e-5) by default; tighten it so the comparison is
* not limited by the stopping rule.
local tight "tolerance(1e-12) ltolerance(1e-14) nrtolerance(1e-13)"

stset time, failure(event)

tempname fh
file open `fh' using "streg_stata.json", write replace text
file write `fh' "{" _n

foreach d in weibull exponential lognormal loglogistic {
    quietly streg x1 x2, dist(`d') time `tight'
    wr `fh' "`d'"
    quietly streg x1 x2, dist(`d') time vce(robust) `tight'
    wr `fh' "`d'_robust"
    quietly streg x1 x2, dist(`d') time vce(cluster clust) `tight'
    wr `fh' "`d'_cluster"
}
quietly streg x1 x2, dist(weibull) time frailty(gamma) `tight'
wr `fh' "weibull_frailty"
* On these data the frailty variance of the other two distributions goes to
* its boundary (theta -> 0, LR statistic 0). ml cannot meet the tight rule
* on a flat likelihood, so these blocks use Stata's defaults; only the
* log-likelihood and the LR test are compared.
foreach d in lognormal loglogistic {
    quietly streg x1 x2, dist(`d') time frailty(gamma)
    wr `fh' "`d'_frailty_boundary"
}
quietly streg x1 x2, dist(weibull) time frailty(gamma) vce(robust) `tight'
wr `fh' "weibull_frailty_robust"

file write `fh' `"  "_meta": {"' _n
file write `fh' `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write `fh' `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write `fh' `"  }"' _n
file write `fh' "}" _n
file close `fh'
