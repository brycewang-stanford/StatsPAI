* ---------------------------------------------------------------------------
* Generate the Stata reference fixture for the PSM-DID weight semantics in
* tests/reference_parity/test_psmdid_weight_parity.py
*
* Requires: Stata 18 + psmatch2 (ssc install psmatch2).
* Run:      stata -b do _generate_psmdid_weights.do   (from this directory)
*
* Why this fixture exists
* -----------------------
* The canonical Stata PSM-DID recipe is
*
*     psmatch2 d x1 x2, neighbor(1) logit      // creates _weight, _support
*     reg y d post did [fweight=_weight] if _support==1
*
* Stata's `fweight` means "this row stands for _weight identical rows": the
* residual degrees of freedom are sum(_weight) - k, NOT the row count - k.
* `aweight` instead keeps df = N_rows - k.  The point estimate is the same
* under both; the standard errors are not.  This fixture pins BOTH so the
* Python side can prove which one it reproduces.
*
* Produces
* --------
*   psmdid_baseline.csv  id x1 x2 d y0 _pscore _treated _support _weight
*   psmdid_panel.csv     id period post d y
*   (scalars are echoed to the log and recorded in psmdid_weights_stata.json)
* ---------------------------------------------------------------------------
clear all
set seed 20260806
set obs 300

gen id = _n
gen x1 = rnormal()
gen x2 = rnormal()
gen ps_true = invlogit(0.9*x1 - 0.6*x2 - 0.2)
gen d  = rbinomial(1, ps_true)
gen y0 = 1 + 0.7*x1 - 0.3*x2 + rnormal()

* --- 1:1 nearest-neighbour with replacement (psmatch2 default) -------------
* neighbor(1) keeps _weight integer, which is what Stata's fweight requires.
quietly psmatch2 d x1 x2, outcome(y0) neighbor(1) logit common
di "psmatch2 att   = " %21.16e r(att)
di "psmatch2 seatt = " %21.16e r(seatt)

keep id x1 x2 d y0 _pscore _treated _support _weight
format x1 x2 y0 _pscore _weight %21.16e
export delimited using "psmdid_baseline.csv", replace datafmt
tempfile base
save `base', replace

* --- build the 2-period panel ---------------------------------------------
* y = y0 + 1.5*d*post + noise.  The unit-level y0 carries into both periods,
* so the DiD is identified off the post-period jump.
expand 2
bysort id: gen period = _n - 1
gen post = period
set seed 777
gen y = y0 + 1.5*d*post + 0.4*rnormal()
gen did = d*post

keep id period post d y did _weight _support
order id period post d y did
format y %21.16e
export delimited id period post d y using "psmdid_panel.csv", replace datafmt

* --- the five weight regimes ----------------------------------------------
* Every regime uses the same matched sample: _weight non-missing AND on
* support.  Only the weighting of the regression differs.
keep if !missing(_weight) & _support == 1

quietly summarize _weight
local sumw = r(sum)
local nrows = r(N)
di "matched rows  = " `nrows'
di "sum of weight = " `sumw'

* (1) fweight, iid
reg y d post did [fweight=_weight]
di "FW_IID   b = " %21.16e _b[did] "  se = " %21.16e _se[did] ///
   "  N = " e(N) "  df_r = " e(df_r)

* (2) aweight, iid
reg y d post did [aweight=_weight]
di "AW_IID   b = " %21.16e _b[did] "  se = " %21.16e _se[did] ///
   "  N = " e(N) "  df_r = " e(df_r)

* (3) fweight, clustered on id
reg y d post did [fweight=_weight], cluster(id)
di "FW_CLUS  b = " %21.16e _b[did] "  se = " %21.16e _se[did] ///
   "  N = " e(N) "  df_r = " e(df_r) "  G = " e(N_clust)

* (4) aweight, clustered on id
reg y d post did [aweight=_weight], cluster(id)
di "AW_CLUS  b = " %21.16e _b[did] "  se = " %21.16e _se[did] ///
   "  N = " e(N) "  df_r = " e(df_r) "  G = " e(N_clust)

* (5) unweighted, iid
reg y d post did
di "NOW_IID  b = " %21.16e _b[did] "  se = " %21.16e _se[did] ///
   "  N = " e(N) "  df_r = " e(df_r)

* --- the identity that makes row expansion the definition of fweight ------
* Physically replicating each row _weight times and running plain OLS must
* reproduce regime (1) exactly.  If this holds, the Python implementation is
* free to expand rows instead of hand-deriving Stata's df corrections.
expand _weight
reg y d post did
di "EXPANDED b = " %21.16e _b[did] "  se = " %21.16e _se[did] ///
   "  N = " e(N) "  df_r = " e(df_r)
