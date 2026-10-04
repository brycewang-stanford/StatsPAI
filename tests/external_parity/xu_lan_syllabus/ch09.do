use senate, clear
drop if y >= .
generate d = x >= 0
generate x2 = x^2
generate x3 = x^3
generate dx = d*x
generate dx2 = d*x2
regress y d x, robust
regress y d x dx, robust
regress y d x x2 dx dx2, robust
regress y d x dx if abs(x) <= 10, robust
regress y d x dx if abs(x) <= 20, robust
rdrobust y x
rdrobust y x, all
rdrobust y x, h(10)
rdrobust y x, p(2)
rdrobust y x, kernel(uniform)
rdrobust y x, bwselect(msetwo)
rdrobust y x, bwselect(cerrd)
rdbwselect y x, all
rddensity x
rdrobust y x, c(5)
rdrobust y x, c(-5)
generate t = d
replace t = 1 - d if mod(_n, 7) == 0
rdrobust y x, fuzzy(t)
ivregress 2sls y x dx (t = d) if abs(x) <= 15, robust
