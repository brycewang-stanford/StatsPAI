* Stata 18 MP reference for tests/test_margins_fe_delta.py: margins after
* Poisson with the fixed effects as explicit parameters (i.g / i.g i.h).
* From this directory: stata-mp -b do stata_margins_fe_delta.do
version 18
set more off
set type double
tempname fh
file open `fh' using "stata_margins_fe_delta.json", write replace
file write `fh' "{" _n
local first 1
program drop _all
program define dumprow
    args fh key
    matrix b = r(b)
    matrix V = r(V)
    local nm : colfullnames b
    file write `fh' `"  "`key'": ["'
    forvalues j = 1/`=colsof(b)' {
        if `j' > 1 file write `fh' ", "
        file write `fh' "[" %23.16e (b[1,`j']) ", " %23.16e (sqrt(V[`j',`j'])) "]"
    }
    file write `fh' "]"
end
foreach v in oim robust cluster {
    local vce = cond("`v'" == "cluster", "vce(cluster g)", "vce(`v')")
    * one-way fixture
    import delimited using "fep.csv", clear asdouble
    qui poisson y x i.g, `vce' iterate(100) tolerance(1e-12) ltolerance(1e-14)
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "fep_`v'_bx": ["' %23.16e (_b[x]) ", " %23.16e (_se[x]) "]," _n
    qui margins, dydx(x)
    dumprow `fh' "fep_`v'_dydx"
    file write `fh' "," _n
    qui margins, at(x=(0 1))
    dumprow `fh' "fep_`v'_at"
    file write `fh' "," _n
    qui margins, at(x=(0 1)) contrast(atcontrast(r))
    dumprow `fh' "fep_`v'_atcontrast"
    file write `fh' "," _n
    * two-way fixture
    import delimited using "fep2.csv", clear asdouble
    qui poisson y x1 x2 d i.g i.h, `vce' iterate(100) tolerance(1e-12) ltolerance(1e-14)
    file write `fh' `"  "fep2_`v'_b": [["' %23.16e (_b[x1]) ", " %23.16e (_se[x1]) "], [" %23.16e (_b[x2]) ", " %23.16e (_se[x2]) "], [" %23.16e (_b[d]) ", " %23.16e (_se[d]) "]]," _n
    qui margins, dydx(x1 x2 d)
    dumprow `fh' "fep2_`v'_dydx"
    file write `fh' "," _n
    qui margins, at(d=(0 1))
    dumprow `fh' "fep2_`v'_at"
    file write `fh' "," _n
    qui margins, at(d=(0 1)) contrast(atcontrast(r))
    dumprow `fh' "fep2_`v'_atcontrast"
}
* weighted, clustered
import delimited using "fep2.csv", clear asdouble
qui poisson y x1 x2 d i.g i.h [pw=w], vce(cluster g) iterate(100) tolerance(1e-12) ltolerance(1e-14)
file write `fh' "," _n `"  "fep2_w_b": [["' %23.16e (_b[x1]) ", " %23.16e (_se[x1]) "], [" %23.16e (_b[x2]) ", " %23.16e (_se[x2]) "], [" %23.16e (_b[d]) ", " %23.16e (_se[d]) "]]," _n
qui margins, dydx(x1 x2 d)
dumprow `fh' "fep2_w_dydx"
file write `fh' "," _n
qui margins, at(d=(0 1)) contrast(atcontrast(r))
dumprow `fh' "fep2_w_atcontrast"
* ppmlhdfe for comparison (two-way, cluster)
import delimited using "fep2.csv", clear asdouble
qui ppmlhdfe y x1 x2 d, a(g h) d vce(cluster g) tolerance(1e-12)
file write `fh' "," _n `"  "fep2_ppml_b": [["' %23.16e (_b[x1]) ", " %23.16e (_se[x1]) "]]," _n
qui margins, dydx(x1 x2 d)
dumprow `fh' "fep2_ppml_dydx"
file write `fh' _n "}" _n
file close `fh'
