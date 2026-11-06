* ---------------------------------------------------------------------------
* Round-2 Stata reference for tests/reference_parity/test_r2_ts_parity.py.
*
* Stata 18 built-ins: estat sbsingle (supremum Wald and Hansen 1997 p-value),
* its Mata p-value function pvalsup() on a grid, and xtunitroot fisher
* (the inverse-logit L* constant). SSC xtbreak (Ditzen, Karavias &
* Westerlund) + moremata for the F(l+1|l) statistics, installed into a
* private ado directory -- never PLUS:
*
*   _ado_r2_ts/   (gitignored; created by this script)
*
* xtbreak's Python helpers need a Python that Stata 18 can embed; Python 3.13
* fails to load them (IndentationError in the python: block), so the script
* points python_exec at the repository venv (Python 3.10) for this session,
* or at $STATSPAI_PYTHON when run from a worktree.
*
* Run _generate_r2_ts_data.py first, then from this directory:
*   stata-mp -b do _generate_r2_ts_stata.do
* Writes r2_ts_Stata.json (numbers as %23.16e).
* ---------------------------------------------------------------------------
version 18
set more off
set type double

local ADO "`c(pwd)'/_ado_r2_ts"
capture mkdir "`ADO'"
sysdir set PLUS "`ADO'"
net set ado "`ADO'"
adopath ++ "`ADO'"
capture which xtbreak
if _rc ssc install xtbreak
capture mata: mata which mm_quantile()
if _rc ssc install moremata
* The repository venv, or (from a git worktree, which has no .venv of its
* own) the interpreter named by the STATSPAI_PYTHON environment variable.
local VENVPY "`c(pwd)'/../../../.venv/bin/python"
capture confirm file "`VENVPY'"
if _rc {
    local VENVPY : env STATSPAI_PYTHON
    capture confirm file "`VENVPY'"
    if _rc {
        display as error "No Python for xtbreak: run from the main checkout" ///
            " (with .venv) or set STATSPAI_PYTHON to a Python 3.10-3.12 path."
        exit 601
    }
}
set python_exec "`VENVPY'"

tempname fh
file open `fh' using "r2_ts_Stata.json", write replace text
file write `fh' "{" _n

capture program drop jnum
program define jnum
    args fh key val last
    local sep = cond("`last'" == "last", "", ",")
    if missing(`val') file write `fh' `"  "`key'": null`sep'"' _n
    else file write `fh' `"  "`key'": "' %23.16e (`val') "`sep'" _n
end

capture program drop jmat
program define jmat
    args fh key M last
    local sep = cond("`last'" == "last", "", ",")
    local r = rowsof(`M')
    local c = colsof(`M')
    file write `fh' `"  "`key'": ["'
    forvalues i = 1/`r' {
        forvalues j = 1/`c' {
            local comma = cond(`i' == `r' & `j' == `c', "", ", ")
            local v = `M'[`i', `j']
            if missing(`v') file write `fh' "null`comma'"
            else file write `fh' %23.16e (`v') "`comma'"
        }
    }
    file write `fh' "]`sep'" _n
end

file write `fh' `"  "stata_version": "`c(stata_version)'","' _n
quietly which xtbreak
file write `fh' `"  "xtbreak_version": "2.2 - 03.12.2025","' _n

* ---- estat sbsingle: sup-Wald, p-value, trimmed range -----------------------
foreach case in ts_ym ts_yx r2_y3 r2_yx r2_yx2 r2_wn {
    if substr("`case'", 1, 2) == "ts" import delimited using "ts_break.csv", clear asdouble
    else import delimited using "r2_ts_break.csv", clear asdouble
    tsset t
    if "`case'" == "ts_ym"  quietly regress ym
    if "`case'" == "ts_yx"  quietly regress y x
    if "`case'" == "r2_y3"  quietly regress y3
    if "`case'" == "r2_yx"  quietly regress yx x
    if "`case'" == "r2_yx2" quietly regress yx2 x
    if "`case'" == "r2_wn"  quietly regress wn
    quietly estat sbsingle, trim(15) swald
    jnum `fh' "sbsingle_`case'_swald" r(chi2_swald)
    jnum `fh' "sbsingle_`case'_p" r(p_swald)
    local bd = real("`r(breakdate)'")
    local lt = real("`r(ltrim)'")
    local rt = real("`r(rtrim)'")
    jnum `fh' "sbsingle_`case'_breakdate" `bd'
    jnum `fh' "sbsingle_`case'_ltrim" `lt'
    jnum `fh' "sbsingle_`case'_rtrim" `rt'
}

* ---- pvalsup() on a grid: x = Wald statistic, k, pi0 -------------------------
local g = 0
foreach k in 1 2 3 5 10 40 {
    foreach pi0 in 0.005 0.05 0.10 0.15 0.20 0.25 0.33 0.495 0.5 {
        local g = `g' + 1
        matrix P`g' = J(7, 4, .)
        local i = 0
        foreach c in 0.5 2 5 8 12 20 35 {
            local i = `i' + 1
            local x = `k' + `c' * sqrt(`k')
            mata: pvalsup(`x', `k', `pi0')
            matrix P`g'[`i', 1] = `k'
            matrix P`g'[`i', 2] = `pi0'
            matrix P`g'[`i', 3] = `x'
            matrix P`g'[`i', 4] = r(p)
        }
        if `g' == 1 matrix PV = P`g'
        else matrix PV = PV \ P`g'
    }
}
jmat `fh' "pvalsup_grid" PV

* ---- xtbreak: F(l+1|l) statistics -------------------------------------------
import delimited using "ts_break.csv", clear asdouble
tsset t
quietly xtbreak test ym, breakconstant hypothesis(3) breaks(5) trim(0.15) sequential
matrix F = r(f)
jmat `fh' "xtbreak_ts_ym_f" F
import delimited using "r2_ts_break.csv", clear asdouble
tsset t
quietly xtbreak test y3, breakconstant hypothesis(3) breaks(5) trim(0.15) sequential
matrix F = r(f)
jmat `fh' "xtbreak_r2_y3_f" F

file write `fh' `"  "done": 1"' _n
file write `fh' "}" _n
file close `fh'
