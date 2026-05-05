* tests/stata_parity/_common.do
*
* Shared helpers for the Stata side of the StatsPAI Track A parity
* harness. Mirrors tests/r_parity/_common.{py,R}.
*
* Each NN_<name>.do script:
*   1. Reads ../r_parity/data/<name>.csv (the CSV dumped by the
*      Python side of the R parity harness -- same bytes, no drift).
*   2. Runs the canonical Stata reference implementation.
*   3. Calls stata_parity_open / stata_parity_row / stata_parity_extra
*      / stata_parity_close to write results/<module>_Stata.json.
*
* JSON precision: %24.17g (full IEEE-754).
* Storage default: double (forces import delimited to read full
*   precision; otherwise we lose 4-5 orders of magnitude vs R/Python).
*
* Implementation note: row state is accumulated to disk under
* logs/<module>.rows.tmp / logs/<module>.extras.tmp instead of being
* held in a Mata global. Reason: several Stata commands (rdrobust,
* csdid, others) internally call `mata mata clear` which would wipe
* a Mata-resident accumulator. Disk-resident state survives.

version 18

capture program drop stata_parity_init
program define stata_parity_init
    syntax , Module(string)
    global STATA_PARITY_MODULE  "`module'"
    global STATA_PARITY_HERE    "`c(pwd)'"
    global STATA_PARITY_DATA    "../r_parity/data"
    global STATA_PARITY_RESULTS "results"
    global STATA_PARITY_LOGS    "logs"
    global STATA_PARITY_ROWS    "logs/`module'.rows.tmp"
    global STATA_PARITY_EXTRAS  "logs/`module'.extras.tmp"
    capture mkdir "${STATA_PARITY_RESULTS}"
    capture mkdir "${STATA_PARITY_LOGS}"
    if c(matsize) < 400 {
        display as error "matsize=`c(matsize)' too small"
        exit 459
    }
    set type double
    set seed 42
    display "stata_parity_init OK module=`module' edition=`c(edition)' matsize=`c(matsize)'"
end


capture program drop stata_parity_open
program define stata_parity_open
    syntax , Module(string)
    capture erase "${STATA_PARITY_ROWS}"
    capture erase "${STATA_PARITY_EXTRAS}"
end


capture program drop _stata_parity_jsonnum
program define _stata_parity_jsonnum, rclass
    args x sentinel
    if `x' == `sentinel' {
        return local s "null"
        exit
    }
    if `x' >= . {
        return local s "null"
        exit
    }
    local v = trim(string(`x', "%24.17g"))
    if substr("`v'", 1, 1) == "."  local v = "0" + "`v'"
    if substr("`v'", 1, 2) == "-." local v = "-0" + substr("`v'", 2, .)
    return local s "`v'"
end

capture program drop _stata_parity_jsonint
program define _stata_parity_jsonint, rclass
    args x sentinel
    if `x' == `sentinel' {
        return local s "null"
        exit
    }
    if `x' >= . {
        return local s "null"
        exit
    }
    return local s = trim(string(`x', "%18.0f"))
end


capture program drop stata_parity_row
program define stata_parity_row
    syntax , STatname(string) ESTimate(real) ///
        [ STDerr(real -99e9) CIlo(real -99e9) CIhi(real -99e9) Nobs(real -99e9) ]

    local stat_esc : subinstr local statname `"""' `"\""', all
    local stat_esc : subinstr local stat_esc "\" "\\", all

    _stata_parity_jsonnum `estimate' -99e9
    local v_est = "`r(s)'"
    _stata_parity_jsonnum `stderr'   -99e9
    local v_se  = "`r(s)'"
    _stata_parity_jsonnum `cilo'     -99e9
    local v_lo  = "`r(s)'"
    _stata_parity_jsonnum `cihi'     -99e9
    local v_hi  = "`r(s)'"
    _stata_parity_jsonint `nobs'     -99e9
    local v_n   = "`r(s)'"

    file open _spjr using "${STATA_PARITY_ROWS}", write append
    file write _spjr "    {" _n
    file write _spjr `"      "module": ""' "${STATA_PARITY_MODULE}" `"","' _n
    file write _spjr `"      "side": "Stata","' _n
    file write _spjr `"      "statistic": ""' "`stat_esc'" `"","' _n
    file write _spjr `"      "estimate": "' "`v_est'" "," _n
    file write _spjr `"      "se": "'       "`v_se'"  "," _n
    file write _spjr `"      "ci_lo": "'    "`v_lo'"  "," _n
    file write _spjr `"      "ci_hi": "'    "`v_hi'"  "," _n
    file write _spjr `"      "n": "'        "`v_n'"   "," _n
    file write _spjr `"      "extra": {}"' _n
    file write _spjr "    }" _n
    file close _spjr
end


capture program drop stata_parity_extra
program define stata_parity_extra
    syntax , Key(string) Val(string)
    local key_esc : subinstr local key `"""' `"\""', all
    local key_esc : subinstr local key_esc "\" "\\", all
    local val_esc : subinstr local val `"""' `"\""', all
    local val_esc : subinstr local val_esc "\" "\\", all
    file open _spje using "${STATA_PARITY_EXTRAS}", write append
    file write _spje `"    ""' "`key_esc'" `"": ""' "`val_esc'" `"""' _n
    file close _spje
end


capture program drop stata_parity_extra_num
program define stata_parity_extra_num
    syntax , Key(string) Val(real)
    local key_esc : subinstr local key `"""' `"\""', all
    local key_esc : subinstr local key_esc "\" "\\", all
    _stata_parity_jsonnum `val' -99e9
    file open _spje using "${STATA_PARITY_EXTRAS}", write append
    file write _spje `"    ""' "`key_esc'" `"": "' "`r(s)'" _n
    file close _spje
end


capture program drop stata_parity_close
program define stata_parity_close
    syntax , Module(string)
    local outpath "${STATA_PARITY_RESULTS}/`module'_Stata.json"

    * Read row + extra fragments and assemble the final JSON, taking
    * care to (a) prefix each row by all-but-the-last with a comma and
    * (b) the same for extras, and (c) post-process the row file to
    * insert a leading "0" before any leading "." in numeric values
    * (Stata's %g formatter emits ".11..." which is invalid JSON).

    tempname inrows inextras out
    tempfile rows_fixed extras_fixed

    * Pass 1: rewrite rows file fixing leading dots in numbers.
    file open `inrows' using "${STATA_PARITY_ROWS}", read
    file open `out'    using `rows_fixed', write replace
    file read `inrows' line
    while r(eof) == 0 {
        local fixed : subinstr local line `": ."' `": 0."', all
        local fixed : subinstr local fixed `": -."' `": -0."', all
        file write `out' `"`fixed'"' _n
        file read `inrows' line
    }
    file close `inrows'
    file close `out'

    * Pass 2: same for extras (numbers may appear without quotes).
    file open `out' using `extras_fixed', write replace
    capture file open `inextras' using "${STATA_PARITY_EXTRAS}", read
    if !_rc {
        file read `inextras' line
        while r(eof) == 0 {
            local fixed : subinstr local line `": ."' `": 0."', all
            local fixed : subinstr local fixed `": -."' `": -0."', all
            file write `out' `"`fixed'"' _n
            file read `inextras' line
        }
        file close `inextras'
    }
    file close `out'

    * Now build the final JSON envelope.
    tempname final
    file open `final' using "`outpath'", write replace
    file write `final' "{" _n
    file write `final' `"  "module": ""' "`module'" `"","' _n
    file write `final' `"  "side": "Stata","' _n
    file write `final' `"  "rows": ["' _n

    * Read rows back; each row object is bracketed by lines starting
    * with "    {" / "    }". Inject a comma before every "    {"
    * line except the first.
    file open `inrows' using `rows_fixed', read
    local first 1
    file read `inrows' line
    while r(eof) == 0 {
        if substr(`"`line'"', 1, 5) == "    {" {
            if `first' == 0 {
                file write `final' "," _n
            }
            local first 0
        }
        file write `final' `"`line'"' _n
        file read `inrows' line
    }
    file close `inrows'

    file write `final' "  ]," _n

    file write `final' `"  "extra": {"' _n
    capture confirm file `extras_fixed'
    if !_rc {
        file open `inextras' using `extras_fixed', read
        local n_ex 0
        file read `inextras' line
        while r(eof) == 0 {
            local trimmed = trim(`"`line'"')
            if `"`trimmed'"' != "" {
                local n_ex = `n_ex' + 1
            }
            file read `inextras' line
        }
        file close `inextras'
        file open `inextras' using `extras_fixed', read
        local i 0
        file read `inextras' line
        while r(eof) == 0 {
            local trimmed = trim(`"`line'"')
            if `"`trimmed'"' != "" {
                local i = `i' + 1
                if `i' < `n_ex' {
                    file write `final' `"`line'"' "," _n
                }
                else {
                    file write `final' `"`line'"' _n
                }
            }
            file read `inextras' line
        }
        file close `inextras'
    }

    file write `final' "  }" _n
    file write `final' "}" _n
    file close `final'

    display "OK -- wrote `outpath'"

    * Cleanup tmp files.
    capture erase "${STATA_PARITY_ROWS}"
    capture erase "${STATA_PARITY_EXTRAS}"
end


global STATA_PARITY_Z95 = 1.959963984540054
