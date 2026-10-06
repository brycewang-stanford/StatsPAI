* Reference values for tests/reference_parity/test_xcorr_parity.py
*
* Runs on the committed synthetic file xcorr.csv and writes xcorr_Stata.csv
* (lag, xc): the cross-correlations of `xcorr x y, lags(10)`.
*   cd tests/reference_parity/_fixtures
*   stata-mp -b do _generate_xcorr_Stata.do
version 18
clear all
import delimited using "xcorr.csv", clear asdouble
tsset t
xcorr x y, lags(10) generate(xc) nodraw
generate lag = _n - 11 if _n <= 21
keep if _n <= 21
keep lag xc
format xc %21.17g
export delimited using "xcorr_Stata.csv", replace
