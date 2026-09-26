* ---------------------------------------------------------------------------
* Stata reference for tests/reference_parity/test_match_fw_stata_parity.py
*
* Requires: Stata 18 (official teffects only; nothing to install).
* Run:      stata -b do _generate_match_fw_stata.do   (from this directory)
*
* What this fixture pins
* ----------------------
* teffects psmatch with frequency weights [fw=fw] -- the only weight
* teffects' matching estimators accept ([pw=], [iw=], [aw=] give rc 101 and
* vce(cluster) rc 198) -- and the same fit on the physically expanded data,
* which Stata reports identically. sp.match(weights=) implements [fw=] as
* that expansion.
*
* Data: the NSW-DW sample written by tests/r_parity/data/11_psm.csv, plus a
* seeded frequency weight in {1, 2, 3}. ATET only: the PSID controls give
* propensity scores below 1e-5, so teffects refuses the ATE (rc 459).
* ---------------------------------------------------------------------------
version 18
clear all
set seed 20260926
import delimited using "../../r_parity/data/11_psm.csv", clear asdouble case(preserve)
gen int fw = 1 + floor(3*runiform())
export delimited using "match_fw_data.csv", replace

local X age education black hispanic married re74 re75
file open fh using "match_fw_stata.json", write replace text
file write fh "{" _n
foreach stat in atet {
    local eq = upper("`stat'")
    quietly teffects psmatch (re78) (treat `X', logit) [fw=fw], `stat' nneighbor(1)
    file write fh `"  "psmatch_`stat'_fw": {"est": "' %23.16e (_b[`eq':r1vs0.treat]) `", "se": "' %23.16e (_se[`eq':r1vs0.treat]) `", "N": "' %12.0f (e(N)) `"},"' _n
}
preserve
expand fw
foreach stat in atet {
    local eq = upper("`stat'")
    quietly teffects psmatch (re78) (treat `X', logit), `stat' nneighbor(1)
    file write fh `"  "psmatch_`stat'_expanded": {"est": "' %23.16e (_b[`eq':r1vs0.treat]) `", "se": "' %23.16e (_se[`eq':r1vs0.treat]) `"},"' _n
}
restore
file write fh `"  "_meta": {"' _n
file write fh `"    "stata_version": "' `"""' "`c(stata_version)'" `"""' `","' _n
file write fh `"    "flavor": "' `"""' "`c(flavor)' `c(edition_real)'" `"""' `","' _n
file write fh `"    "seed": 20260926,"' _n
file write fh `"    "generated": "' `"""' "`c(current_date)'" `"""' _n
file write fh `"  }"' _n
file write fh "}" _n
file close fh
display "wrote match_fw_stata.json and match_fw_data.csv"
