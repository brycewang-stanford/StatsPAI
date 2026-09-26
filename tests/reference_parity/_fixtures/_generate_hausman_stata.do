* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_hausman_stata_parity.py
*
* Requires: Stata 18 (official commands only).
* Run:      stata -b do _generate_hausman_stata.do   (from this directory)
*
* What this fixture pins
* ----------------------
* `xtreg, fe` / `xtreg, re` / `hausman fe re` (and `, sigmamore`) on two
* panels of 100 units x 8 periods:
*   a  unit effect independent of x1 (RE consistent)
*   b  x1 loads on the unit effect (RE inconsistent)
* On panel b the classical statistic is negative: Stata prints it with a
* warning and no p-value, and `sigmamore` yields a usable test.  StatsPAI's
* `sp.hausman_test` used its own RE variance estimate (about 50x off on panel
* a) and both StatsPAI implementations clamped a negative statistic to 0,
* reporting p = 1 and "use RE".
*
* Data is generated and exported HERE so both sides read the same bytes.
* ---------------------------------------------------------------------------
version 18
clear all
set type double
set seed 20260917
set obs 100

gen long   id = _n
gen double u  = rnormal()
expand 8
bysort id: gen int year = _n
sort id year
gen double e   = rnormal(0, 0.5)
gen double x2  = rnormal()
gen double x1a = rnormal()
gen double x1b = rnormal() + 0.4*u
gen double ya  = 1 + 0.5*x1a - 0.3*x2 + u + e
gen double yb  = 1 + 0.5*x1b - 0.3*x2 + u + e

format x2 x1a x1b ya yb %21.16e
export delimited id year x1a x1b x2 ya yb using "hausman_data.csv", replace

xtset id year

capture file close fh
file open fh using "hausman_stata.json", write replace text
file write fh "{" _n

foreach p in a b {
    quietly xtreg y`p' x1`p' x2, fe
    estimates store fe
    quietly xtreg y`p' x1`p' x2, re
    estimates store re
    quietly hausman fe re
    local chi2 = r(chi2)
    local df = r(df)
    quietly hausman fe re, sigmamore
    local chi2_more = r(chi2)
    local p_more = r(p)

    quietly estimates restore fe
    file write fh `"  "`p'": {"' _n
    file write fh `"    "b_fe_x1": "' %21.16e (_b[x1`p']) `","' _n
    file write fh `"    "b_fe_x2": "' %21.16e (_b[x2]) `","' _n
    quietly estimates restore re
    file write fh `"    "b_re_x1": "' %21.16e (_b[x1`p']) `","' _n
    file write fh `"    "b_re_x2": "' %21.16e (_b[x2]) `","' _n
    file write fh `"    "chi2": "' %21.16e (`chi2') `","' _n
    file write fh `"    "df": "' %3.0f (`df') `","' _n
    file write fh `"    "chi2_sigmamore": "' %21.16e (`chi2_more') `","' _n
    file write fh `"    "p_sigmamore": "' %21.16e (`p_more') _n
    if "`p'" == "a" {
        file write fh "  }," _n
    }
    else {
        file write fh "  }" _n
    }
}
file write fh "}" _n
file close fh
