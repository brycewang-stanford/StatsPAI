* Reference values for tests/reference_parity/test_hdfe_iv_nested_sdof.py:
* ivreghdfe when every absorbed fixed effect is nested in the cluster.
* Stata 18 MP, ivreghdfe 1.1.4, ivreg2 4.1.11, reghdfe 6.13.1, on the
* committed hdfe_iv.csv (firm nested in ind). Run from this directory:
*   stata-mp -b do _generate_hdfe_iv_nested_Stata.do
version 18
clear all
set type double
import delimited "hdfe_iv.csv", clear asdouble
foreach a in "firm" "ind" "firm ind" "firm city#q" {
  ivreghdfe y w (d = z1 z2), absorb(`a') cluster(ind)
  di "absorb(`a')"
  foreach s in N N_clust df_r df_a sdofminus rkf cdf idstat j {
    di "`s' " %24.17g e(`s')
  }
  di "b_d " %24.17g _b[d]
  di "se_d " %24.17g _se[d]
  di "se_w " %24.17g _se[w]
}
