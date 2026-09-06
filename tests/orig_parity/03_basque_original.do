* Original-data parity for Synth::basque (Abadie-Gardeazabal 2003) --
* Stata side. Mirrors 03_basque_original.R: the canonical ADH
* special-predictor specification (one gdppc mean per pre-1970 year,
* nested V optimization) on the same CSV bytes exported by the R script.
*
* Run from tests/orig_parity/:  stata -b do 03_basque_original.do
* Writes results/03_basque_original_Stata.json.
*
* The point of this artifact: on the *original* Synth::basque bytes the
* three implementations sit within ~4e-4 of one another
* (sp -0.8945886, R Synth -0.8944272, Stata synth -0.8942570), whereas
* on the calibrated replica of r_parity module 07 the R and Stata
* references split by 2.3e-2. The SCM non-uniqueness disclosure in the
* JSS paper cites both facts.

version 18
clear all

capture cd "`c(pwd)'"
import delimited "data/03_basque_original.csv", clear case(preserve)
capture confirm numeric variable gdppc
if _rc destring gdppc, replace force
keep region_id region year gdppc
xtset region_id year

* Treated unit: Basque Country (Pais Vasco), region_id 17.
synth gdppc gdppc(1955) gdppc(1956) gdppc(1957) gdppc(1958) gdppc(1959) ///
            gdppc(1960) gdppc(1961) gdppc(1962) gdppc(1963) gdppc(1964) ///
            gdppc(1965) gdppc(1966) gdppc(1967) gdppc(1968) gdppc(1969), ///
    trunit(17) trperiod(1970) nested

matrix Yt = e(Y_treated)
matrix Ys = e(Y_synthetic)
local n_rows = rowsof(Yt)
local sum_post = 0
local n_post = 0
forvalues r = 1/`n_rows' {
    local y = Yt[`r', 1]
    local s = Ys[`r', 1]
    local gap = `y' - `s'
    local rname : word `r' of `:rownames Yt'
    if `rname' >= 1970 {
        local sum_post = `sum_post' + `gap'
        local n_post = `n_post' + 1
    }
}
scalar avg_post_gap = `sum_post' / `n_post'
display %20.15f avg_post_gap

quietly count
local n_obs = r(N)

* Emit the sibling JSON artifact by hand (matches _common.py schema).
tempname fh
file open `fh' using "results/03_basque_original_Stata.json", write replace
file write `fh' "{" _n
file write `fh' `"  "module": "03_basque_original","' _n
file write `fh' `"  "side": "Stata","' _n
file write `fh' `"  "rows": ["' _n
file write `fh' "    {" _n
file write `fh' `"      "module": "03_basque_original","' _n
file write `fh' `"      "side": "Stata","' _n
file write `fh' `"      "statistic": "avg_post_gap","' _n
local est : display %20.15f avg_post_gap
file write `fh' `"      "estimate": `=trim("`est'")',"' _n
file write `fh' `"      "se": null,"' _n
file write `fh' `"      "n": `n_obs',"' _n
file write `fh' `"      "published": -0.855,"' _n
file write `fh' `"      "citation": "Abadie-Gardeazabal (2003) Figure 2 / Synth vignette","' _n
file write `fh' `"      "extra": {}"' _n
file write `fh' "    }" _n
file write `fh' "  ]," _n
file write `fh' `"  "extra": {"data_source": "Synth::basque", "n_obs": `n_obs', "engine": "Stata `c(stata_version)' `c(edition_real)'"}"' _n
file write `fh' "}" _n
file close `fh'
display "OK -- wrote results/03_basque_original_Stata.json"
