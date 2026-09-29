* Reference values for tests/reference_parity/test_gelbach_absorb_cluster_b1x2.py
* Stata 18 MP, b1x2 4.1.0 (Gelbach, SSC) installed into a private ado dir.
* gelbach_fe_cluster.csv: numpy.random.default_rng(5); fe (20 levels) is not
* nested in cl (15 clusters), so the dummies' dof enter both conventions alike.
import delimited "gelbach_fe_cluster.csv", clear asdouble
b1x2 y, x1all(x) x2all(a1 a2) x1only(x) x2delta(g1=a1 : g2=a2) cluster(cl)
matrix list e(b), format(%24.17g)
matrix list e(V), format(%24.17g)
qui tab fe, gen(D)
foreach v in "cluster(cl)" "robust" "" {
  b1x2 y, x1all(x D2-D20) x2all(a1 a2) x1only(x) x2delta(g1=a1 : g2=a2) `v'
  matrix list e(b), format(%24.17g)
  matrix list e(V), format(%24.17g)
}
