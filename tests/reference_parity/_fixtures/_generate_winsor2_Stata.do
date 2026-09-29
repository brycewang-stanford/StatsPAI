* winsor2 reference for tests/reference_parity/test_winsor_winsor2_Stata_parity.py
version 18
clear all
import delimited using "winsor2_data.csv", clear asdouble
winsor2 x z, cuts(1 99) suffix(_a)
winsor2 x z if year >= 2007 & year <= 2020, cuts(5 95) suffix(_b)
winsor2 x z, cuts(2.5 97.5) by(g) suffix(_c)
keep x_a z_a x_b z_b x_c z_c
format x_a-z_c %24.17g
export delimited using "winsor2_Stata.csv", replace datafmt
