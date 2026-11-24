* Percentile reference for tests/reference_parity/test_winsor_pctile_parity.py
* Run from this folder:  stata -b do _generate_winsor_pctile_Stata.do
* Stata's _pctile (the routine winsor2 calls) on x = 1..n for n = 2..120 at
* every whole percentile and four fractional ones, and winsor2 itself at
* two percentile pairs where n p / 100 is a whole number that
* n * (p / 100) misses in double precision.
version 18
clear all
tempname h
postfile `h' int n double p double pctile using winsor_pctile_grid, replace
forvalues n = 2/120 {
    quietly {
        clear
        set obs `n'
        gen double x = _n
        _pctile x, p(0.5 2.5 97.5 99.5)
        local k = 0
        foreach p in 0.5 2.5 97.5 99.5 {
            local ++k
            post `h' (`n') (`p') (r(r`k'))
        }
        _pctile x, p(1(1)99)
        forvalues p = 1/99 {
            post `h' (`n') (`p') (r(r`p'))
        }
    }
}
postclose `h'
use winsor_pctile_grid, clear
* every value is a whole number or a half
format pctile %6.1f
export delimited using "winsor_pctile_Stata.csv", replace datafmt
erase winsor_pctile_grid.dta

clear
set obs 100
gen double x = _n
winsor2 x, cuts(7 93) suffix(_a)
winsor2 x, cuts(29 57) suffix(_b)
winsor2 x, cuts(7 93) trim suffix(_c)
format x_a-x_c %24.17g
export delimited using "winsor_pctile_winsor2_Stata.csv", replace datafmt
