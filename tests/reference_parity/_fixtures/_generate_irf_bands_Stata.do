* Stata 18 reference for the asymptotic standard errors of sp.irf.
* Run from tests/reference_parity/_fixtures; writes irf_bands_Stata.csv.
clear all
import delimited "irf_bands.csv", clear
tsset t
quietly var y1 y2 y3, lags(1/2)
tempfile irfs
irf create base, set("`irfs'", replace) step(8)
quietly var y1 y2 y3, lags(1/2) dfk
irf create dfk, step(8)
use "`irfs'.irf", clear
keep irfname impulse response step irf stdirf oirf stdoirf cirf stdcirf coirf stdcoirf
format irf stdirf oirf stdoirf cirf stdcirf coirf stdcoirf %24.16e
export delimited using "irf_bands_Stata.csv", replace
