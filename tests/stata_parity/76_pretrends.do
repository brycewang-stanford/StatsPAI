* tests/stata_parity/76_pretrends.do
*
* Module 76: Roth (2022) pre-trends power.
*   StatsPAI:  sp.pretrends_power / sp.pretrends_slope_for_power
*   R:         pretrends::pretrends / pretrends::slope_for_power (0.1.0)
*   Stata:     pretrends (Caceres Bravo's Stata port)  <-- this file
*
* This module has no panel: the estimator is a function of (betahat, sigma)
* alone. Both other sides build the same covariance from the same literal
* standard errors and an AR(1) correlation, so this script rebuilds the
* identical inputs in Mata rather than reading a CSV. Any edit to
* 76_pretrends.py's SE / BETA / RHO literals must be mirrored here; the
* assertion block below pins the resulting sigma so a one-sided edit fails
* loudly instead of silently grading a different fixture.
*
* Period indexing. The Python/R sides use tVec = (-4,-3,-2, 0,1,2) with
* reference period -1. The Stata port's numpre(3) declares the first three
* entries pre-treatment and the rest post, which places the same
* consecutive-integer grid around the omitted reference. Both therefore
* build delta = slope * (t - t_ref) = slope * (-3,-2,-1, 1,2,3). numpre()
* is used rather than time()/ref() because the port's time()/ref() parser
* expects the reference period to appear as a zero row inside b, which the
* R convention (reference period dropped) does not supply.
*
* Tolerance: rel < 1e-3, as registered. The power and Bayes factor go
* through a randomised multivariate-normal integrator on both sides
* (mvtnorm::pmvnorm in R), and the Stata port's slope_for_power search
* terminates on a dyadic bisection grid, so its slope lands on a
* 2^-k lattice point (e.g. 0.027911376953125) rather than on R's
* root-finder value. The likelihood ratio is the closed-form density ratio
* and agrees to ~1e-15.

version 18
clear all

do _common.do
stata_parity_init, module(76_pretrends)
stata_parity_open, module(76_pretrends)

* ------------------------------------------------------------------
* Rebuild the shared (betahat, sigma) inputs. Mirrors 76_pretrends.py:
*   SE   = [0.050, 0.045, 0.040, 0.100, 0.110, 0.120]
*   BETA = [0.012, -0.008, 0.021, 0.180, 0.240, 0.310]
*   RHO  = 0.5,  sigma = corr(|i-j|) * outer(SE, SE)
* ------------------------------------------------------------------
mata:
se   = (0.050, 0.045, 0.040, 0.100, 0.110, 0.120)
bet  = (0.012, -0.008, 0.021, 0.180, 0.240, 0.310)
idx  = (1..6)
D    = abs((idx' * J(1, 6, 1)) - (J(6, 1, 1) * idx))
S    = (0.5:^D) :* (se' * se)
st_matrix("beta", bet)
st_matrix("sigma", S)
end

* Guard the rebuilt fixture: sigma[1,1] = 0.05^2, sigma[1,2] = 0.5*0.05*0.045.
if abs(sigma[1,1] - 0.0025) > 1e-15 | abs(sigma[1,2] - 0.001125) > 1e-15 {
    display as error "fixture drift: rebuilt sigma does not match 76_pretrends.py"
    exit 459
}

local nobs = 1000

foreach s in 0.02 0.05 {
    local tag = subinstr("`s'", ".", "p", .)
    qui pretrends, numpre(3) b(beta) vcov(sigma) slope(`s') nocoefplot
    * Capture before anything else touches r(); r-class scalars have to be
    * assigned rather than macro-expanded to survive into the row writer.
    local pw = r(Power)
    local bf = r(Bayes)
    local lr = r(LR)
    stata_parity_row, stat("power_slope_`tag'")            est(`pw') nob(`nobs')
    stata_parity_row, stat("bayes_factor_slope_`tag'")     est(`bf') nob(`nobs')
    stata_parity_row, stat("likelihood_ratio_slope_`tag'") est(`lr') nob(`nobs')
}

foreach p in 0.5 0.8 {
    local tag = subinstr("`p'", ".", "p", .)
    qui pretrends power `p', numpre(3) b(beta) vcov(sigma)
    local sl = r(slope)
    stata_parity_row, stat("slope_for_power_`tag'") est(`sl') nob(`nobs')
}

stata_parity_extra, key(stata_command) val("pretrends, numpre(3) b(beta) vcov(sigma) slope(#) / pretrends power #, numpre(3) b(beta) vcov(sigma)")
stata_parity_extra, key(period_convention) val("numpre(3) reproduces tVec=(-4,-3,-2,0,1,2) with reference -1; time()/ref() expects the reference period inside b")
stata_parity_extra, key(slope_search) val("the Stata port terminates slope_for_power on a dyadic bisection grid, so its slope is a 2^-k lattice point rather than R's root-finder value")
stata_parity_extra, key(integrator) val("power and Bayes factor use a randomised multivariate-normal integrator on both sides; the likelihood ratio is closed form and agrees to ~1e-15")
stata_parity_extra, key(stata_bridge_status) val("materialized 2026-08-06 with licensed Stata 18")

stata_parity_close, module(76_pretrends)
