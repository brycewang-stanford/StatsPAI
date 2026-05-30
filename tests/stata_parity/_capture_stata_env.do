* tests/stata_parity/_capture_stata_env.do
*
* Capture the Stata reference environment for the Track A parity harness:
*   - core Stata version / edition / OS  -> results/_stata_core.txt
*   - community ado package versions     -> results/_stata_ado_versions.tsv
*
* The ado "version" is the verbatim first `*!' banner line of each command's
* .ado file (Stata has no packageVersion() primitive). We record it exactly
* as written, or "(no *! version line)" / "NOT INSTALLED" -- never a guessed
* version (CLAUDE.md s10 zero-fabrication rule).
*
* The human-readable STATA_ENVIRONMENT.md is assembled from these two files
* by _gen_stata_env.py.
*
*   stata-mp -q -b do _capture_stata_env.do

version 18

* Community ado commands exercised by the .do panel (and their key deps).
local cmds ///
    csdid reghdfe rdrobust rddensity sdid honestdid did_imputation jwdid ///
    bacondecomp eventstudyinteract sensemakr drdid oaxaca ivreg2 ftools ///
    gtools estout

tempname fh adin
file open `fh' using "results/_stata_ado_versions.tsv", write replace
file write `fh' "command" _tab "version_line" _tab "path" _n

foreach c of local cmds {
    capture findfile `c'.ado
    if _rc {
        file write `fh' "`c'" _tab "NOT INSTALLED" _tab "" _n
    }
    else {
        local p "`r(fn)'"
        local vline ""
        local nread 0
        file open `adin' using "`p'", read
        file read `adin' line
        while r(eof) == 0 & `nread' < 60 & "`vline'" == "" {
            local t = trim(`"`line'"')
            if substr("`t'", 1, 2) == "*!" {
                local vline "`t'"
            }
            local nread = `nread' + 1
            file read `adin' line
        }
        file close `adin'
        if "`vline'" == "" local vline "(no *! version line)"
        file write `fh' "`c'" _tab `"`vline'"' _tab "`p'" _n
    }
}
file close `fh'

tempname fc
file open `fc' using "results/_stata_core.txt", write replace
file write `fc' "stata_version=`c(stata_version)'" _n
file write `fc' "edition=`c(edition_real)'" _n
file write `fc' "os=`c(os)'" _n
file write `fc' "machine_type=`c(machine_type)'" _n
file write `fc' "born_date=`c(born_date)'" _n
file close `fc'

display "OK -- wrote results/_stata_ado_versions.tsv + results/_stata_core.txt"
