* Reference values for tests/reference_parity/test_hdfe_iv_ivreghdfe.py
* Stata 18 MP, ivreghdfe 1.1.4 (29nov2025), ivreg2 4.1.11, reghdfe 6.13.1.
* hdfe_iv.csv: numpy.random.default_rng(11), firm nested in ind, rounded to 6dp.
import delimited "hdfe_iv.csv", clear asdouble
ivreghdfe y w (d = z1 z2), absorb(firm city#q) cluster(ind) first
foreach s in N N_clust df_r rkf cdf idstat arf archi2 j r2 r2_a rmse {
  di "`s' " %24.17g e(`s')
}
matrix list e(b), format(%24.17g)
matrix list e(V), format(%24.17g)
matrix list e(first), format(%24.17g)
ivreghdfe y (d = z1), absorb(firm city#q) cluster(ind)
di %24.17g _b[d] %24.17g _se[d] %24.17g e(rkf) %24.17g e(cdf) %24.17g e(idstat)
ivreghdfe y w (d = z1 z2), absorb(firm city#q) robust
di %24.17g _se[d] %24.17g _se[w] %24.17g e(rkf) %24.17g e(cdf) %24.17g e(idstat) %24.17g e(j) " " e(df_r)
ivreghdfe y w (d = z1 z2), absorb(firm city#q)
di %24.17g _se[d] %24.17g _se[w] %24.17g e(cdf) %24.17g e(idstat) %24.17g e(sargan)
