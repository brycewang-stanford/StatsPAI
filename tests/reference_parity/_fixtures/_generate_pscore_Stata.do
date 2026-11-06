* Stata reference for sp.pscore (tests/reference_parity/test_pscore_stata_parity.py).
*
* pscore of Becker and Ichino (Stata Journal st0026_2), used as a black box:
* outputs only. It is installed into the PRIVATE ado directory
* _ado_pscore/ (git-ignored by the _ado_*/ rule), never into PLUS.
*
* Data: matching_lalonde.csv (MatchIt::lalonde, already committed for
* test_matching_r_parity.py), read with full double precision.
*
* For every specification the file records the treatment model (e(b), the
* standard errors, e(ll)), each row's score, block and common-support flag,
* the "Variable ... is not balanced in block ..." lines pscore prints, and,
* for every covariate in every final block, the p-value of the
* equal-variance t test of Stata's ttest (the test pscore reports on).
*
* Run: cd tests/reference_parity/_fixtures && stata-mp -b do _generate_pscore_Stata.do
version 18
clear all
set more off
set type double
local here "`c(pwd)'"
local ado "`here'/_ado_pscore"
capture mkdir "`ado'"
sysdir set PLUS "`ado'"
sysdir set PERSONAL "`ado'"
capture which pscore
if _rc net install st0026_2, from("http://www.stata-journal.com/software/sj5-3") replace

local X8 "age educ black hispan married nodegree re74 re75"
local s1 "treat `X8', logit comsup"
local x1 "`X8'"
local s2 "treat `X8',"
local x2 "`X8'"
local s3 "treat age educ married re74, logit numblo(3) level(0.05)"
local x3 "age educ married re74"
local s4 "treat `X8', comsup level(0.05)"
local x4 "`X8'"
local s5 "treat age educ black married re74, logit numblo(8) level(0.1)"
local x5 "age educ black married re74"
local K 5

tempname fh lh
file open `fh' using "pscore_Stata.json", write replace
file write `fh' "{" _n `"  "_versions": {"Stata": "`c(stata_version)'", "pscore": "st0026_2"},"' _n
forvalues k = 1/`K' {
    import delimited using "matching_lalonde.csv", clear asdouble
    gen long row = _n
    tempfile lg
    log using "`lg'", text replace name(pslog)
    pscore `s`k'' pscore(ps) blockid(blk)
    local nb = r(r)
    log close pslog
    matrix b = e(b)
    matrix V = e(V)
    local names : colnames b
    local ll = e(ll)
    capture confirm variable comsup
    if _rc gen byte comsup = 1
    if `k' > 1 file write `fh' "," _n
    file write `fh' `"  "`k'": {"spec": "`s`k''", "ll": "' %23.16e (`ll') `", "n_blocks": `nb', "b": {"'
    local j = 0
    foreach nm of local names {
        local ++j
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'": "' %23.16e (b[1,`j'])
    }
    file write `fh' `"}, "se": {"'
    local j = 0
    foreach nm of local names {
        local ++j
        if `j' > 1 file write `fh' ", "
        file write `fh' `""`nm'": "' %23.16e (sqrt(V[`j',`j']))
    }
    * per row: score (missing -> null), block, common support
    file write `fh' `"}, "ps": ["'
    forvalues i = 1/`=_N' {
        if `i' > 1 file write `fh' ", "
        if missing(ps[`i']) file write `fh' "null"
        else file write `fh' %23.16e (ps[`i'])
    }
    file write `fh' `"], "block": ["'
    forvalues i = 1/`=_N' {
        if `i' > 1 file write `fh' ", "
        if missing(blk[`i']) file write `fh' "null"
        else file write `fh' %4.0f (blk[`i'])
    }
    file write `fh' `"], "comsup": ["'
    forvalues i = 1/`=_N' {
        if `i' > 1 file write `fh' ", "
        file write `fh' %1.0f (comsup[`i'])
    }
    * the balance lines pscore printed
    file write `fh' `"], "not_balanced": ["'
    file open `lh' using "`lg'", read text
    file read `lh' line
    local first 1
    while r(eof) == 0 {
        if regexm(`"`line'"', "^Variable ([A-Za-z0-9_]+) is not balanced in block ([0-9]+)") {
            if !`first' file write `fh' ", "
            file write `fh' `"["`=regexs(1)'", `=regexs(2)']"'
            local first 0
        }
        file read `lh' line
    }
    file close `lh'
    * equal-variance t test of every covariate in every block with both arms
    file write `fh' `"], "ttest": ["'
    local first 1
    levelsof blk, local(blocks)
    foreach g of local blocks {
        qui count if blk == `g' & treat == 1
        local n1 = r(N)
        qui count if blk == `g' & treat == 0
        if `n1' == 0 | r(N) == 0 continue
        foreach v of local x`k' {
            capture ttest `v' if blk == `g', by(treat)
            if _rc continue
            if missing(r(p)) continue
            if !`first' file write `fh' ", "
            file write `fh' `"["`v'", `g', "' %23.16e (r(p)) "]"
            local first 0
        }
    }
    file write `fh' "]}"
}
file write `fh' _n "}" _n
file close `fh'
