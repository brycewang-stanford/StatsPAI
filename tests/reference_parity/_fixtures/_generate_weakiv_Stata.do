* Reference values for tests/reference_parity/test_weakiv_stata.py
* Stata 18 MP, weakiv 2.4.07 (1Oct2015), ivreg2 4.1.11, avar.
* weakiv_fx.csv: 400 obs, 40 clusters g, instruments z1-z3, control x1,
* endogenous d1 d2; numpy.random.default_rng(20261019), rounded to 6dp
* (DGP in test_weakiv_stata.py's docstring).
import delimited "weakiv_fx.csv", clear asdouble
program define dump
  foreach s in N ar_chi2 ar_p k_chi2 k_p j_chi2 j_p rk rk_p clr_stat clr_p kj_p {
    di "`s' " %24.17g e(`s')
  }
  foreach s in ar_cset k_cset clr_cset {
    di "`s' " e(`s')
  }
end
* one endogenous regressor, null 0.5
foreach v in "" "robust" "cluster(g)" {
  qui weakiv ivreg2 y x1 (d1 = z1 z2 z3), null(0.5) `v'
  dump
}
* two endogenous regressors, joint nulls
foreach b in "0.5 -0.3" "0.1 0.1" {
  foreach v in "" "robust" "cluster(g)" {
    qui weakiv ivreg2 y x1 (d1 d2 = z1 z2 z3), null(`b') `v' clrsims(400000)
    dump
  }
}
* subset AR
foreach b in 0.2 0.5 1.5 {
  qui weakiv ivreg2 y x1 (d1 d2 = z1 z2 z3), subset(d1) null(`b')
  di %24.17g e(ar_chi2) " " e(ar_df)
}
qui weakiv ivreg2 y x1 (d1 d2 = z1 z2 z3), subset(d2) null(0.1)
di %24.17g e(ar_chi2) " " e(ar_df)
* Stata's own p-values at the StatsPAI set endpoints (all 0.05; CLR to the
* Simpson-rule error of weakiv's p-value, ~1.5e-7), e.g.
foreach x in 0.8412897662787029 454.3508781183183 {
  qui weakiv ivreg2 y x1 (d1 = z1 z2 z3), null(`x') cluster(g)
  di "`x' " %20.15g e(ar_p)
}
foreach x in -1.0974187706394274 0.6377454878116505 1.704569132792757 1.8766640289962833 {
  qui weakiv ivreg2 y x1 (d1 = z1 z2 z3), null(`x') robust
  di "`x' " %20.15g e(k_p)
}
foreach x in 2.116846691048375 65.70888769155566 {
  qui weakiv ivreg2 y x1 (d1 d2 = z1 z2 z3), subset(d1) null(`x')
  di "`x' " %20.15g e(ar_p)
}
* i.i.d. closed-form intervals: the AR p-value at Stata's own endpoint
foreach x in .7814155 -2.428634 .78647930954 -2.515302179 {
  qui weakiv ivreg2 y x1 (d1 = z1 z2 z3), null(`x')
  di "`x' " %20.15g e(ar_p)
}
