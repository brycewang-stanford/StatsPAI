* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_prodest_parity.py.
* Requires Stata 18 and prodest (SSC, Rovigatti & Mollisi) installed into a
* private ado directory -- never PLUS:
*
*   net set ado "<dir>" ; adopath ++ "<dir>" ; ssc install prodest
*
* Run _generate_prodest_data.py first (writes prodest_panel.csv), then
*   do _generate_prodest_stata.do     (from this directory)
* Writes prodest_Stata.json (numbers as %23.16e).
*
* Conventions pinned here
* -----------------------
* * valueadded: the proxy does not enter the production function, as in R
*   prodest and StatsPAI.
* * op / lp: e(b) is the first-stage free-input coefficient and the
*   second-stage state coefficient (Mata foplp: cubic g in lagged omega,
*   Nelder-Mead to tolerance 1e-5 from OLS + N(0, 0.01) noise). reps(3), the
*   fewest prodest's bootstrap covariance accepts; the first replication is
*   the full sample.
* * lp, acf: the same stage 1, instruments (L.free, state), W = (Z'Z)^-1/n.
* * wrdg: stacked ivregress gmm, wmatrix(unadjusted) (= 2SLS point
*   estimates), equation-2 intercept cons2, default robust e(V).
* * set type double: prodest builds its polynomial terms with generate,
*   which otherwise stores them as float and moves e(b) by up to 3e-7.
* ---------------------------------------------------------------------------
version 18
set more off
set type double
set seed 20260913
import delimited using "prodest_panel.csv", clear asdouble
xtset id year

tempname fh
file open `fh' using "prodest_Stata.json", write replace text
file write `fh' "{" _n

capture program drop writeb
program define writeb
    args fh key withv last
    matrix b = e(b)
    local names : colnames b
    local k : word count `names'
    file write `fh' `"  "`key'": {"b": {"'
    forvalues j = 1/`k' {
        local nm : word `j' of `names'
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'": "' %23.16e (b[1,`j'])
    }
    file write `fh' "}"
    capture matrix V = e(V)
    if _rc == 0 & "`withv'" == "v" {
        file write `fh' `", "V_diag": {"'
        forvalues j = 1/`k' {
            local nm : word `j' of `names'
            if `j' > 1 file write `fh' ", "
            file write `fh' `""`nm'": "' %23.16e (V[`j',`j'])
        }
        file write `fh' "}"
    }
    file write `fh' `", "N": "' %12.0f (e(N)) "}"
    if "`last'" == "" file write `fh' ","
    file write `fh' _n
end

foreach p in 2 3 {
    quietly prodest y, free(l) state(k) proxy(lninv) method(op) valueadded ///
        poly(`p') reps(3) id(id) t(year)
    writeb `fh' op_poly`p' nov
    quietly prodest y, free(l) state(k) proxy(m) method(lp) valueadded ///
        poly(`p') reps(3) id(id) t(year)
    writeb `fh' lp_poly`p' nov
    quietly prodest y, free(l) state(k) proxy(m) method(lp) acf valueadded ///
        poly(`p') reps(3) id(id) t(year)
    writeb `fh' acf_poly`p' nov
    quietly prodest y, free(l) state(k) proxy(m) method(wrdg) valueadded ///
        poly(`p') id(id) t(year)
    if `p' == 3 writeb `fh' wrdg_poly`p' v last
    else writeb `fh' wrdg_poly`p' v
}

file write `fh' "}" _n
file close `fh'
