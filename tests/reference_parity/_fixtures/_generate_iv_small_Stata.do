* ivregress with and without -small- on the ssc_presets data: SE and
* p-value of x, for sp.iv(small=).
*   python tests/reference_parity/_fixtures/_generate_ssc_presets_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_iv_small_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/ssc_presets.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/iv_small_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach est in 2sls liml {
    foreach v in "unadjusted" "robust" "cluster ind" {
        foreach sm in "" "small" {
            qui ivregress `est' y w (x = z), vce(`v') `sm'
            local key = "`est'_" + word("`v'", 1) + cond("`sm'" == "", "", "_small")
            local p = cond("`sm'" == "", 2 * normal(-abs(_b[x] / _se[x])), 2 * ttail(e(df_r), abs(_b[x] / _se[x])))
            if !`first' file write `fh' "," _n
            local first 0
            file write `fh' `"  "`key'": {"b": "' %24.16e (_b[x]) `", "se": "' %24.16e (_se[x]) `", "p": "' %24.16e (`p') "}"
        }
    }
}
file write `fh' _n "}" _n
file close `fh'
