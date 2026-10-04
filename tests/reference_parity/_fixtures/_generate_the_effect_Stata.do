* Reference values for tests/reference_parity/test_the_effect_stata_parity.py
* (the commands of Huntington-Klein, The Effect, 2nd ed., ch. 13-21, run on
* a dataset that ships with StatsPAI).
* Data: sp.datasets.nsw_dw() written with DataFrame.to_stata; Stata 18.0 MP.
* ebalance (Hainmueller & Xu), cem (Blackwell, Iacus, King & Porro) and
* sensemakr (Cinelli, Ferwerda & Hazlett) from SSC.
set linesize 200
use nsw_dw.dta, clear
local X age education black hispanic married nodegree re74 re75

* --- entropy balancing, solved to convergence
local E age education black married nodegree
foreach t in "1" "2 2 1 1 1" "3 2 1 1 1" {
    cap drop w
    ebalance treat `E', targets(`t') generate(w) tolerance(1e-10) maxiter(500)
    reg re78 treat [pw = w]
    di "REF ebalance [`t'] " %20.12f _b[treat] " " %20.12f _se[treat]
}

* --- coarsened exact matching
cem age education black(#2) re74(#6) re75(0 1000 5000 20000), treatment(treat)
count if cem_matched == 1 & treat == 1
di "REF cem matched_t " r(N)
count if cem_matched == 1 & treat == 0
di "REF cem matched_c " r(N)
reg re78 treat [iweight = cem_weights]
di "REF cem reg " %20.12f _b[treat] " " %20.12f _se[treat]
cem age education, treatment(treat)
count if cem_matched == 1
di "REF cem2 matched " r(N)
reg re78 treat [iweight = cem_weights]
di "REF cem2 reg " %20.12f _b[treat] " " %20.12f _se[treat]

* --- sensemakr with a group benchmark
sensemakr re78 treat `X', treat(treat) gbenchmark(black hispanic) gname(race) kd(1 2)
di "REF sense rv " %20.14f e(rv_q) " " %20.14f e(rv_qa) " " %20.14f e(r2yd_x)
matrix list e(bounds), format(%20.14f)

* --- factor-variable notation
reg re78 black##c.age##c.age
di "REF fv3 " %20.12f _b[1.black#c.age#c.age] " " %20.12f _se[1.black#c.age#c.age] " " %20.12f _b[1.black#c.age] " " %20.12f _b[c.age#c.age]
g byte agegrp = (age > 25) + (age > 35)
reg re78 agegrp##c.education
di "REF fvfac " %20.12f _b[2.agegrp#c.education] " " %20.12f _se[2.agegrp#c.education] " " %20.12f _b[1.agegrp]
reghdfe re78 treat##ib1.agegrp, absorb(education) vce(robust)
di "REF hdfe " %20.12f _b[1.treat#2.agegrp] " " %20.12f _se[1.treat#2.agegrp] " " %20.12f _b[1.treat#0.agegrp]
reg re78 treat age if education > 8 [aw = re74]
di "REF zerow " %20.12f _b[treat] " " %20.12f _se[treat] " " e(N)
collapse (mean) re78 age treat (sd) s = re75, by(education)
reg re78 age treat
di "REF collapse " %20.12f _b[age] " " %20.12f _se[age]

* --- the same collapse when the integers are stored as byte: a mean of a
* byte is stored as a float
use nsw_dw.dta, clear
compress treat age education
collapse (mean) re78 age treat (sd) s = re75, by(education)
reg re78 age treat
di "REF collapse_float " %20.12f _b[age] " " %20.12f _se[age]
