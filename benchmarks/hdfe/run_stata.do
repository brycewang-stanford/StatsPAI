// HDFE Poisson baseline — Stata `ppmlhdfe` reference run.
//
// Stata is NOT installed on the dev machine, so this `.do` file is a
// **template**: not executed by `run_baseline.py`. It is committed so
// that any contributor with Stata can reproduce the comparison and
// drop the resulting JSON into `results/<dataset>_stata.json`.
//
// Output JSON schema must match the one written by run_python.py /
// run_r.R so `run_baseline.py` can ingest it:
//
//   {
//     "dataset": "small",
//     "backend": "ppmlhdfe",
//     "warmup": 1,
//     "repeats": 3,
//     "n_rows": 100000,
//     "wall": { "wall_runs": [...], "wall_min": ..., "wall_mean": ..., "wall_max": ... },
//     "coefs": { "x1": ..., "x2": ... },
//     "se":    { "x1": ..., "x2": ... },
//     "iterations": <int>
//   }
//
// Usage (Stata 17+):
//
//   ssc install ppmlhdfe, replace
//   ssc install reghdfe,  replace
//   ssc install ftools,   replace
//   cd "<repo>/benchmarks/hdfe"
//   do run_stata.do small        // or medium / large
//
// CSV.GZ is not directly readable by Stata. Decompress first, e.g.:
//
//   !gunzip -k data/small.csv.gz
//   import delimited using data/small.csv, clear

args dataset
if "`dataset'" == "" local dataset "small"

local csv "data/`dataset'.csv"
capture confirm file "`csv'"
if _rc {
    di as error "Decompress data/`dataset'.csv.gz first (Stata cannot read .gz):"
    di as error "    gunzip -k data/`dataset'.csv.gz"
    exit 601
}

import delimited using "`csv'", clear
gen long _row = _n
qui count
local n = r(N)

// --- 1 warmup + 3 timed runs ------------------------------------------------
local repeats = 3
local warmup  = 1

forvalues i = 1/`warmup' {
    qui ppmlhdfe y x1 x2, absorb(fe1 fe2) iter(25) tol(1e-8)
}

local runs ""
forvalues i = 1/`repeats' {
    timer clear 1
    timer on  1
    qui ppmlhdfe y x1 x2, absorb(fe1 fe2) iter(25) tol(1e-8)
    timer off 1
    qui timer list 1
    local t = r(t1)
    if "`runs'" == "" {
        local runs "`t'"
    }
    else {
        local runs "`runs',`t'"
    }
}

// Final run for coef extraction
qui ppmlhdfe y x1 x2, absorb(fe1 fe2) iter(25) tol(1e-8)
matrix b = e(b)
matrix V = e(V)
local b_x1 = b[1, "x1"]
local b_x2 = b[1, "x2"]
local se_x1 = sqrt(V["x1", "x1"])
local se_x2 = sqrt(V["x2", "x2"])
// ppmlhdfe stores iteration count differently across versions; try a few.
local iter = e(ic)
if "`iter'" == "" local iter = e(N_ic)
if "`iter'" == "" local iter = .

// --- emit JSON to results/<dataset>_stata.json ------------------------------
// `wall_min/mean/max` are intentionally written as `null` so that
// `run_baseline.py::load_stata_result` recomputes them from `wall_runs`.
// Stata's missing dot (`.`) is NOT valid JSON.
tempname fh
file open `fh' using "results/`dataset'_ppmlhdfe.json", write replace
file write `fh' "{" _n
file write `fh' `"  "dataset": "`dataset'","' _n
file write `fh' `"  "backend": "ppmlhdfe","' _n
file write `fh' "  \"warmup\": `warmup'," _n
file write `fh' "  \"repeats\": `repeats'," _n
file write `fh' "  \"n_rows\": `n'," _n
file write `fh' "  \"wall\": { \"wall_runs\": [`runs'], \"wall_min\": null, \"wall_mean\": null, \"wall_max\": null }," _n
file write `fh' "  \"coefs\": { \"x1\": `b_x1', \"x2\": `b_x2' }," _n
file write `fh' "  \"se\":    { \"x1\": `se_x1', \"x2\": `se_x2' }," _n
if `iter' < . {
    file write `fh' "  \"iterations\": `iter'" _n
}
else {
    file write `fh' "  \"iterations\": null" _n
}
file write `fh' "}" _n
file close `fh'

di as result "wrote results/`dataset'_ppmlhdfe.json"
