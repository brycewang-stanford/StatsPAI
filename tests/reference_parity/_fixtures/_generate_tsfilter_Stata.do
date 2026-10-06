* Reference values for tests/reference_parity/test_tsfilter_parity.py
* Stata 18 (official command tsfilter).
* Run from this folder:
*     do _generate_tsfilter_Stata.do
* Reads tsfilter.csv; writes tsfilter_Stata.csv, one row per observation,
* a cycle column c_<case> and a trend column t_<case> per case.
clear all
set more off
* trend() takes the default storage type; float would lose nine digits
set type double
import delimited using "tsfilter.csv", clear case(preserve) asdouble
tsset t
tsfilter hp double c_hp1600 = y, smooth(1600) trend(t_hp1600)
tsfilter hp double c_hp100 = y, smooth(6.25) trend(t_hp100)
tsfilter bk double c_bk = y, minperiod(6) maxperiod(32) smaorder(12) trend(t_bk)
tsfilter bk double c_bk_st = y, minperiod(6) maxperiod(32) smaorder(12) stationary trend(t_bk_st)
tsfilter bk double c_bk_2_8_3 = y, minperiod(2) maxperiod(8) smaorder(3) trend(t_bk_2_8_3)
tsfilter cf double c_cf = y, minperiod(6) maxperiod(32) trend(t_cf)
tsfilter cf double c_cf_drift = y, minperiod(6) maxperiod(32) drift trend(t_cf_drift)
tsfilter cf double c_cf_st = y, minperiod(6) maxperiod(32) stationary trend(t_cf_st)
tsfilter cf double c_cf_sma = y, minperiod(6) maxperiod(32) smaorder(12) trend(t_cf_sma)
tsfilter cf double c_cf_sma_dr = y, minperiod(6) maxperiod(32) smaorder(12) drift trend(t_cf_sma_dr)
tsfilter cf double c_cf_st_sma = y, minperiod(6) maxperiod(32) smaorder(12) stationary trend(t_cf_st_sma)
tsfilter bw double c_bw = y, maxperiod(32) order(2) trend(t_bw)
tsfilter bw double c_bw_o4 = y, maxperiod(12) order(4) trend(t_bw_o4)
* added 2026-10-06, after the columns above so that their order is kept
tsfilter cf double c_cf_st_dr = y, minperiod(6) maxperiod(32) stationary drift trend(t_cf_st_dr)
tsfilter cf double c_cf_st_sma_dr = y, minperiod(6) maxperiod(32) smaorder(12) stationary drift trend(t_cf_st_sma_dr)
tsfilter cf double c_cf_sma5 = y, minperiod(2) maxperiod(8) smaorder(5) trend(t_cf_sma5)
drop y
format c_* t_* %24.17g
export delimited using "tsfilter_Stata.csv", replace
