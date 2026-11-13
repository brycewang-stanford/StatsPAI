* Reference numbers for tests/reference_parity/test_iv_mediate_joint_stata_parity.py.
* The three IV equations of ivmediate estimated as ONE just-identified GMM
* system with official Stata's gmm, then nlcom for (total, direct, a * d).
* onestep with an identity weight matrix leaves the estimates at the three
* IV fits; vce(robust) / vce(cluster) is the sandwich of the stacked
* moments, with no small-sample factor. No user-written command is needed.
* Run from tests/reference_parity/_fixtures/ ; writes iv_mediate_joint_Stata.json.
version 18
clear all
set type double
import delimited "iv_mediate.csv", clear asdouble

tempname fh
file open `fh' using "iv_mediate_joint_Stata.json", write replace
file write `fh' "{" _n
file write `fh' `" "_provenance": "Stata/MP `c(stata_version)', gmm (onestep, winitial(identity)) + nlcom, run `c(current_date)' by _generate_iv_mediate_joint_Stata.do on iv_mediate.csv","' _n

capture program drop _jfmt
program define _jfmt, rclass
    args val
    local s = strtrim(string(`val', "%21.17g"))
    if substr("`s'", 1, 1) == "." local s "0`s'"
    if substr("`s'", 1, 2) == "-." local s = "-0" + substr("`s'", 2, .)
    return local s "`s'"
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
    args fh key
    tempname b V r
    mat `b' = e(b)
    mat `V' = e(V)
    _jrow `fh' `key'_b `b'
    forvalues i = 1/3 {
        mat `r' = `V'[`i', 1..3]
        _jrow `fh' `key'_V`i' `r'
    }
end

capture program drop _joint
program define _joint
    * controls enter every equation and instrument their own columns
    syntax , fh(string) key(string) vce(string) [controls(varlist)]
    local x1
    local x2
    local x3
    foreach v of local controls {
        local x1 "`x1' - {g1_`v'}*`v'"
        local x2 "`x2' - {g2_`v'}*`v'"
        local x3 "`x3' - {g3_`v'}*`v'"
    }
    gmm (eq1: y - {b1}*t `x1' - {c1}) ///
        (eq2: m - {a}*t `x2' - {c2}) ///
        (eq3: y - {dir}*t - {d}*m `x3' - {c3}), ///
        instruments(eq1: z `controls') instruments(eq2: z `controls') ///
        instruments(eq3: z t `controls') ///
        onestep winitial(identity) vce(`vce') conv_maxiter(500)
    nlcom (total: _b[b1:_cons]) (direct: _b[dir:_cons]) ///
        (indirect: _b[a:_cons] * _b[d:_cons]), post
    _jspec `fh' `key'
end

_joint, fh(`fh') key(plain_robust) vce(robust)
_joint, fh(`fh') key(cov_robust) vce(robust) controls(w1 w2)
_joint, fh(`fh') key(cov_cluster) vce(cluster cl) controls(w1 w2)

file write `fh' `" "_end": 0"' _n
file write `fh' "}" _n
file close `fh'
