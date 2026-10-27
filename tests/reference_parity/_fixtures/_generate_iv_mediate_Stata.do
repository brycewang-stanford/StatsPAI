* Reference numbers for tests/reference_parity/test_iv_mediate_stata_parity.py.
* Needs ivmediate (ssc install ivmediate; Dippel, Ferrara and Heblich) and
* ranktest (ssc install ranktest). Run from tests/reference_parity/_fixtures/ ;
* writes iv_mediate_Stata.json.
version 18
clear all
set type double
import delimited "iv_mediate.csv", clear asdouble

tempname fh
file open `fh' using "iv_mediate_Stata.json", write replace
file write `fh' "{" _n
file write `fh' `" "_provenance": "Stata/MP `c(stata_version)', ivmediate 1.0.4 (SSC), run `c(current_date)' by _generate_iv_mediate_Stata.do on iv_mediate.csv","' _n

* JSON helpers: numbers with a leading zero, string lists, matrix rows.
capture program drop _jfmt
program define _jfmt, rclass
    args val
    local s = strtrim(string(`val', "%21.17g"))
    if substr("`s'", 1, 1) == "." local s "0`s'"
    if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
end
capture program drop _jnum
program define _jnum
    args fh key val
    _jfmt `val'
    file write `fh' `"  "`key'": `r(s)',"' _n
end
capture program drop _jlist
program define _jlist
    args fh key list
    local out
    local sep
    foreach w of local list {
        local out `"`out'`sep'"`w'""'
        local sep ", "
    }
    file write `fh' `"  "`key'": [`out'],"' _n
end
capture program drop _jrow
program define _jrow
    args fh key mat
    local k = colsof(`mat')
    local out
    local sep
    forvalues j = 1/`k' {
        _jfmt `mat'[1,`j']
        local out "`out'`sep'`r(s)'"
        local sep ", "
    }
    file write `fh' `"  "`key'": [`out'],"' _n
end
capture program drop _jspec
program define _jspec
    * b (total, direct, indirect), V row by row, F statistics, share
    args fh key
    tempname b V r
    mat `b' = e(b)
    mat `V' = e(V)
    _jrow `fh' `key'_b `b'
    forvalues i = 1/3 {
        mat `r' = `V'[`i', 1..3]
        _jrow `fh' `key'_V`i' `r'
    }
    _jnum `fh' `key'_fstat1 e(fstat1)
    _jnum `fh' `key'_fstat2 e(fstat2)
    _jnum `fh' `key'_mepct e(mepct)
    _jnum `fh' `key'_N e(N)
    if "`e(N_clust)'" != "" _jnum `fh' `key'_N_clust e(N_clust)
end

ivmediate y, mediator(m) treatment(t) instrument(z)
_jspec `fh' plain
ivmediate y w1 w2, mediator(m) treatment(t) instrument(z)
_jspec `fh' cov
ivmediate y w1 w2, mediator(m) treatment(t) instrument(z) vce(robust)
_jspec `fh' cov_robust
ivmediate y w1 w2, mediator(m) treatment(t) instrument(z) vce(cluster cl)
_jspec `fh' cov_cluster
ivmediate y w1 w2, mediator(m) treatment(t) instrument(z) absorb(g)
_jspec `fh' fe
ivmediate y w1 w2, mediator(m) treatment(t) instrument(z) absorb(g) vce(cluster g)
_jspec `fh' fe_cluster
ivmediate y w1 w2, mediator(m) treatment(t) instrument(z) absorb(g) vce(robust) full
_jspec `fh' fe_robust

file write `fh' `"  "N": `=_N'"' _n "}" _n
file close `fh'
