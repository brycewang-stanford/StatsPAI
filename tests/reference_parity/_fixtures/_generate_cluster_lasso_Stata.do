* Reference numbers for tests/reference_parity/test_cluster_lasso_stata.py.
* Needs lassopack and pdslasso (ssc install lassopack / pdslasso).
* Run from tests/reference_parity/_fixtures/ ; prints the numbers that are
* stored in cluster_lasso_Stata.json.
import delimited "cluster_lasso_panel.csv", clear asdouble

* (1) cluster-Lasso first stage: d on [z, x]
rlasso d z1-z16 x1-x24, cluster(id) nocons
di "lambda0=" %20.14g e(lambda0) " gamma=" %20.14g e(gamma)
di "selected: `e(selected)'"
mat b = e(betaOLS)
mat list b, format(%18.12g)

* (2) the same without clustering (heteroskedastic loadings)
rlasso d z1-z16 x1-x24, robust nocons
di "selected: `e(selected)'"

* (3) post-double-selection and partialling-out, clustered
pdslasso y d (x1-x24), cluster(id) nocons
foreach m in beta_pds V_pds beta_plasso V_plasso {
    mat list e(`m'), format(%18.12g)
}

* (4) IV with selection among instruments and controls, clustered
ivlasso y (x1-x24) (d = z1-z16), cluster(id) nocons
foreach m in beta_plasso V_plasso {
    mat list e(`m'), format(%18.12g)
}

* (5) IV with selection among instruments only, clustered
ivlasso y (d = z1-z16), cluster(id) nocons
foreach m in beta_plasso V_plasso {
    mat list e(`m'), format(%18.12g)
}

* (6) a panel on which the iterated loadings have two fixed points; lassopack
* and hdm's iteration reach different ones (see the test).
import delimited "cluster_lasso_panel_path.csv", clear asdouble
rlasso d z1-z16 x1-x24, cluster(id) nocons
di "selected: `e(selected)'"
mat b = e(betaOLS)
mat list b, format(%18.12g)
