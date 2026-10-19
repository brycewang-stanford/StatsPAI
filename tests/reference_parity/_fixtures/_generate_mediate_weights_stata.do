* Stata 18 mediate with probability weights: reference for
* sp.mediate(weights=, inference="robust").
*   python tests/reference_parity/_fixtures/_generate_mediate_weights_data.py
*   cd tests/reference_parity/_fixtures && stata-mp -b do _generate_mediate_weights_stata.do
version 18
clear all
import delimited using "mediate_weights.csv", clear asdouble
tempname fh
file open `fh' using "mediate_weights_Stata.json", write replace
file write `fh' "{" _n
local c1 "(y x) (m x) (d) [pw=w], all"
local c2 "(y x) (m x) (d) [pw=w], nointeraction"
local c3 "(yb x, logit) (mb x, logit) (d) [pw=w], all"
local c4 "(y x) (mb x, probit) (d) [pw=w], all"
forvalues k = 1/4 {
    qui mediate `c`k''
    matrix b = e(b)
    matrix V = e(V)
    local names : colfullnames b
    if `k' > 1 file write `fh' "," _n
    file write `fh' `"  "`k'": {"N": "' (e(N)) `", "b": {"'
    local j = 0
    local first = 1
    foreach nm of local names {
        local ++j
        local eq = substr("`nm'", 1, strpos("`nm'", ":") - 1)
        if inlist("`eq'", "NIE", "NDE", "TE", "PNIE", "TNDE") {
            if `first' == 0 file write `fh' ", "
            local first = 0
            file write `fh' `""`eq'": ["' %23.16e (b[1,`j']) ", " %23.16e (sqrt(V[`j',`j'])) "]"
        }
    }
    file write `fh' "}}"
}
file write `fh' _n "}" _n
file close `fh'
