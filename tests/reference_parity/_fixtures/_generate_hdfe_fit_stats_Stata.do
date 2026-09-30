* Reference values for tests/reference_parity/test_hdfe_fit_stats_reghdfe.py
* Stata 18 MP, reghdfe 6.13.1 (10Jan2026). The CSV was drawn once with
* numpy.random.default_rng(7) (firm nested in ind; city#q and ind#q absorbed).
import delimited "hdfe_fit_stats.csv", clear asdouble
capture program drop show
program show
  di %21.16g e(r2) %21.16g e(r2_a) %21.16g e(r2_within) %21.16g e(r2_a_within) ///
     %21.16g e(rmse) " dfr=" e(df_r) " dfa=" e(df_a) " nested=" e(df_a_nested) %21.16g _se[x]
end
qui reghdfe y x z, absorb(firm occ city#q ind#q) vce(cluster ind)
show
qui reghdfe y x z, absorb(firm occ city#q) vce(cluster occ)
show
qui reghdfe y x z, absorb(firm occ city#q ind#q) vce(robust)
show
qui reghdfe y x z, absorb(occ city#q) vce(cluster ind city)
show
