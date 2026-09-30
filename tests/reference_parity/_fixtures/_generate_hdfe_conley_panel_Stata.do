* acreg spatial + time HAC with absorbed unit and period effects, for
* sp.hdfe_ols(vce='conley', conley_time=, conley_unit=, conley_lag=).
*   python tests/reference_parity/_fixtures/_generate_hdfe_conley_panel_data.py
*   stata-mp -b do tests/reference_parity/_fixtures/_generate_hdfe_conley_panel_Stata.do
version 18
clear all
import delimited using "tests/reference_parity/_fixtures/hdfe_conley_panel.csv", clear asdouble
tempname fh
file open `fh' using "tests/reference_parity/_fixtures/hdfe_conley_panel_Stata.json", write replace
file write `fh' "{" _n
local first 1
foreach spec in "lag(2) hac bartlett" "lag(0)" "lag(3) lagdist(3) hac bartlett" {
    qui acreg y x1 x2, spatial latitude(lat) longitude(lon) dist(300) id(id) time(t) `spec' pfe1(id) pfe2(t)
    local key = subinstr(subinstr(subinstr("`spec'", " ", "_", .), "(", "", .), ")", "", .)
    if !`first' file write `fh' "," _n
    local first 0
    file write `fh' `"  "`key'": {"b": ["' %24.16e (_b[x1]) ", " %24.16e (_b[x2]) `"], "se": ["' %24.16e (_se[x1]) ", " %24.16e (_se[x2]) `"], "N": "' (e(N)) "}"
}
file write `fh' _n "}" _n
file close `fh'
