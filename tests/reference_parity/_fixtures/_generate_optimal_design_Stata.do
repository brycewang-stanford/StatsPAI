* ---------------------------------------------------------------------------
* Reference numbers for tests/reference_parity/test_optimal_design_stata_parity.py
*
* Stata 18 -power twomeans, knownsds- (two-sample z test) and its -cluster-
* option. Each line of optimal_design_Stata.csv is one call: an id, the
* returned N (or N1 / N2 / K1 / K2) and the returned delta. nfractional
* gives the unrounded sample size, so the test checks the rounding rule
* separately from the formula.
* Run (from this directory): stata-mp -b do _generate_optimal_design_Stata.do
* ---------------------------------------------------------------------------
version 18
clear all
set more off
tempname fh
file open `fh' using "optimal_design_Stata.csv", write replace
file write `fh' "id,N,N1,N2,K1,K2,delta,N_frac,N1_frac,N2_frac,K1_frac,K2_frac" _n

capture program drop onecall
program define onecall
    args fh id
    gettoken id spec : 0
    gettoken id spec : spec
    * spec is everything after the id
    quietly power twomeans `spec'
    local N = r(N)
    local N1 = r(N1)
    local N2 = r(N2)
    local K1 = cond(missing(r(K1)), ., r(K1))
    local K2 = cond(missing(r(K2)), ., r(K2))
    local d = r(delta)
    quietly power twomeans `spec' nfractional
    local K1f = cond(missing(r(K1)), ., r(K1))
    local K2f = cond(missing(r(K2)), ., r(K2))
    file write `fh' "`id'," %21.17g (`N') "," %21.17g (`N1') "," %21.17g (`N2') "," ///
        %21.17g (`K1') "," %21.17g (`K2') "," %21.17g (`d') "," ///
        %21.17g (r(N)) "," %21.17g (r(N1)) "," %21.17g (r(N2)) "," ///
        %21.17g (`K1f') "," %21.17g (`K2f') _n
end

* sample size (equal and unequal allocation, sd scaled by sqrt(1 - R2),
* cluster designs), then the minimum detectable difference given n
onecall `fh' a 0 0.2, sd(1) knownsds power(0.8) alpha(0.05)
onecall `fh' b 0 0.35, sd(2) knownsds power(0.9) alpha(0.01)
onecall `fh' c 0 0.2, sd(1) knownsds power(0.8) alpha(0.05) nratio(0.4285714285714286)
onecall `fh' d 0 0.2, sd(0.894427190999916) knownsds power(0.8) alpha(0.05)
onecall `fh' e 0 0.2, sd(1) knownsds power(0.8) alpha(0.05) cluster m1(20) m2(20) rho(0.05)
onecall `fh' f 0 0.15, sd(1.5) knownsds power(0.85) alpha(0.05) cluster m1(12) m2(12) rho(0.1)
onecall `fh' g 0, n(800) sd(1) knownsds power(0.8) alpha(0.05)
onecall `fh' h 0, n(500) sd(2) knownsds power(0.9) alpha(0.05) nratio(1.5)
file close `fh'
file open `fh' using "optimal_design_Stata_version.txt", write replace
file write `fh' "Stata `c(stata_version)' `c(edition_real)' `c(born_date)'" _n
file close `fh'
