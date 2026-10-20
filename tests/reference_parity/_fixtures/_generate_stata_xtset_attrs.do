* Stata 18 reference for sp.stata after sp.xtset (tests/test_stata_xtset_attrs.py).
* Input: the columns id t x y of stata_xtset_attrs.csv, in file order (the
* rows are deliberately unsorted; six panels, each skipping periods).
* Run from the fixtures directory on a copy holding id t x y only:
*   stata-mp -b do _generate_stata_xtset_attrs.do
* The stata_* columns of the fixture are lx fx dx l2x n below, merged back
* on (id, t); the regression is printed:
*   b(L.x) = .85196664655777432, b(D.x) = .9694796762372746,
*   _cons = -.17361526055852408, se(L.x) = .36220711434893554,
*   se(D.x) = .23182111618179088, N = 26.
version 18
clear all
set type double
import delimited using xt.csv, clear asdouble
xtset id t
gen double lx = L.x
gen double fx = F.x
gen double dx = D.x
gen double l2x = L2.x
gen n = _n
regress y L.x D.x
matrix b = e(b)
matrix V = e(V)
display %21.17g b[1,1] " " %21.17g b[1,2] " " %21.17g b[1,3]
display %21.17g sqrt(V[1,1]) " " %21.17g sqrt(V[2,2]) " " %21.17g e(N)
format lx fx dx l2x %21.17g
export delimited id t lx fx dx l2x n using xt_out.csv, replace datafmt
