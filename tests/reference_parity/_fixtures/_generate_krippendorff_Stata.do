* Reference for sp.krippendorff_alpha: Stata kalpha 1.3.0 (29dec2017,
* D. Klein and R. Reckmann) and kappaetc 2.1.0 (11aug2022, D. Klein),
* both from SSC. Data: krippendorff_data.csv (80 units x 5 coders, missing
* codings) and krippendorff_small.csv (the kalpha help-file example).
* Units are rows, so kalpha needs its transpose option.
*
* Install into a private ado tree first (never the user's PLUS):
*   sysdir set PLUS "<scratch>/ado/plus"
*   ssc install kalpha
*   ssc install kappaetc
* Run from the repo root:
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_krippendorff_Stata.do
version 18
clear all
set type double
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/krippendorff_Stata.json", write replace
file write `fh' "{" _n
local firstds 1
foreach ds in small data {
    import delimited using "tests/reference_parity/_fixtures/krippendorff_`ds'.csv", clear asdouble
    unab coders : c*
    if !`firstds' file write `fh' "," _n
    local firstds 0
    file write `fh' `"  "`ds'": {"' _n
    local first 1
    foreach lev in nominal ordinal interval ratio {
        qui kalpha `coders', scale(`lev') transpose
        if !`first' file write `fh' "," _n
        local first 0
        file write `fh' `"    "kalpha_`lev'": {"alpha": "' %24.16e (r(kalpha)) ///
            `", "Do": "' %24.16e (r(Do)) `", "De": "' %24.16e (r(De)) ///
            `", "n": "' %24.16e (r(n)) `", "units": "' %24.16e (r(units)) "}"
    }
    * kappaetc: Krippendorff's alpha is the 6th coefficient of r(b)
    foreach w in identity quadratic ratio {
        qui kappaetc `coders', wgt(`w')
        matrix b = r(b)
        file write `fh' "," _n `"    "kappaetc_`w'": "' %24.16e (b[1, 6])
    }
    qui kappaetc `coders', wgt(ordinal, krippendorff)
    matrix b = r(b)
    file write `fh' "," _n `"    "kappaetc_ordinal_krippendorff": "' %24.16e (b[1, 6])
    file write `fh' _n "  }"
}
* bootstrap (Krippendorff 2013 algorithm in kalpha), 20000 reps
import delimited using "tests/reference_parity/_fixtures/krippendorff_data.csv", clear asdouble
unab coders : c*
foreach lev in nominal interval {
    qui kalpha `coders', scale(`lev') transpose bootstrap(reps(20000) seed(20261028) minalpha(0.667 0.8) nodots)
    matrix q = r(q)
    file write `fh' "," _n `"  "boot_`lev'": {"ci_lb": "' %24.16e (r(ci_lb)) ///
        `", "ci_ub": "' %24.16e (r(ci_ub)) `", "q667": "' %24.16e (q[1, 2]) ///
        `", "q800": "' %24.16e (q[2, 2]) `", "reps": "' %24.16e (r(reps)) "}"
}
file write `fh' _n "}" _n
file close `fh'
