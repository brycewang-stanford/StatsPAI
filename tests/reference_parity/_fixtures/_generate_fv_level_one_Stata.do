* Reference numbers for tests/reference_parity/test_fv_level_one_parity.py.
* Stata 18 MP, reghdfe 6.13.1. Run from the repo root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_fv_level_one_Stata.do
set type double
import delimited using "tests/reference_parity/_fixtures/fv_level_one.csv", clear
reghdfe savings ibn.visit#1.treat ibn.visit#c.x1, absorb(visit week) vce(cluster hhid)
foreach v in 2 3 4 {
    di "RHD b`v' " %21.15g _b[`v'.visit#1.treat] " se " %21.15g _se[`v'.visit#1.treat]
    di "RHDX b`v' " %21.15g _b[`v'.visit#c.x1] " se " %21.15g _se[`v'.visit#c.x1]
}
regress savings i.visit#1.treat x1 i.visit, vce(cluster hhid)
foreach v in 2 3 4 {
    di "REG b`v' " %21.15g _b[`v'.visit#1.treat] " se " %21.15g _se[`v'.visit#1.treat]
}
