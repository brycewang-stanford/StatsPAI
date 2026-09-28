* Reference for sp.honest_did(l_vec=, window=): Stata honestdid 1.3.0
* (Caceres Bravo; Rambachan-Roth) on the jwdid, never event study of
* _fixtures/etwfe_poisson_jwdid_hettype.csv (the fixture of
* test_etwfe_poisson_jwdid_parity.py), restricted to event times -4..3
* (estat event, window(-4 3)): leads -4, -3, -2 and horizons 0..3.
* Targets: the average of the four horizons (l_vec = 1/4 each) and the
* single horizon e = 1.  Methods: FLCI under Delta^SD(M) and the
* deterministic Conditional test under Delta^RM(Mbar).
*
* Run from the repository root with Stata 18, jwdid v2.201 / ppmlhdfe 2.3.0
* and honestdid 1.3.0 on the adopath:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_honest_did_lvec_window_Stata.do
version 18
set type double
clear all
qui mata: mata mlib index

import delimited "tests/reference_parity/_fixtures/etwfe_poisson_jwdid_hettype.csv", clear asdouble
quietly jwdid y, ivar(id) tvar(year) gvar(g) method(ppmlhdfe) never
quietly estat event, predict(xb) window(-4 3)
matrix b0 = r(b)
matrix V0 = r(V)
* drop the reference period -1 (4th column): leads -4 -3 -2, horizons 0..3
matrix b = b0[1, 1..3], b0[1, 5..8]
matrix V = (V0[1..3, 1..3], V0[1..3, 5..8] \ V0[5..8, 1..3], V0[5..8, 5..8])
matrix lavg = (0.25, 0.25, 0.25, 0.25)
matrix le1  = (0, 1, 0, 0)

tempname fh
file open `fh' using "tests/reference_parity/_fixtures/honest_did_lvec_window_Stata.json", write replace text
file write `fh' "{" _n
file write `fh' `"  "b": ["'
forvalues j = 1/7 {
    file write `fh' %24.16e (b[1, `j'])
    if `j' < 7 file write `fh' ", "
}
file write `fh' "]," _n

capture program drop _hd
program define _hd
    args fh name lvec delta mvec method last
    quietly honestdid, b(b) vcov(V) numpre(3) l_vec(`lvec') mvec(`mvec') delta(`delta') method(`method')
    mata: st_matrix("CI", HonestEventStudy.CI)
    local k = rowsof(CI)
    file write `fh' `"  "`name'": {"M": ["'
    forvalues i = 2/`k' {
        file write `fh' %24.16e (CI[`i', 1])
        if `i' < `k' file write `fh' ", "
    }
    file write `fh' `"], "lb": ["'
    forvalues i = 2/`k' {
        file write `fh' %24.16e (CI[`i', 2])
        if `i' < `k' file write `fh' ", "
    }
    file write `fh' `"], "ub": ["'
    forvalues i = 2/`k' {
        file write `fh' %24.16e (CI[`i', 3])
        if `i' < `k' file write `fh' ", "
    }
    file write `fh' "]}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

_hd `fh' sd_avg lavg sd "0 0.01 0.05" FLCI
_hd `fh' sd_e1  le1  sd "0 0.01 0.05" FLCI
_hd `fh' rm_avg lavg rm "0 0.5 1" Conditional
_hd `fh' rm_e1  le1  rm "0 0.5 1" Conditional last
file write `fh' "}" _n
file close `fh'
