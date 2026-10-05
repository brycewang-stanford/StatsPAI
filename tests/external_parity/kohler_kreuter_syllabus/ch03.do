* Chapter 3: the grammar of Stata
use data1, clear
* 3.1.2 variable lists
summarize wor*
summarize ybirth-voc
summarize wor0?
summarize _all
* 3.1.4 / 3.1.5 in and if
summarize income in 1/10
summarize income in -5/l
summarize income if sex == 1 & ybirth < 1979
summarize income if (sex == 1 | sex == 2) & !missing(income)
summarize income if state != 11 & income > 0 & income < .
summarize income if inrange(ybirth,1950,1970)
summarize income if inlist(state,1,2,3)
* 3.1.6 expressions
display 42 + 5
display 2^10
display sqrt(2)
display exp(1)
display ln(10)
display log10(1000)
display abs(-3.5)
display int(5.9)
display round(5.567, .01)
display round(5.5)
display floor(-2.5)
display ceil(-2.5)
display mod(17, 5)
display max(1, 2, .)
display min(1, 2, .)
display 5 == 5
display 3 > .
display . > 100000
display .a > .
display normal(1.96)
display invnormal(.975)
display ttail(100, 1.96)
display invttail(100, .025)
display chi2tail(2, 5.99)
display Ftail(2, 100, 3)
display binomial(10, 3, .5)
display comb(10, 3)
display exp(lnfactorial(5))
generate hhinc_eq = hhinc/sqrt(hhsize)
summarize hhinc_eq
generate lninc = ln(income)
summarize lninc
* 3.1.7 number lists
forvalues i = 1(2)9 {
    display `i'
}
foreach n of numlist 1/3 10(10)30 {
    display `n'
}
* 3.2.1 by
sort sex
by sex: summarize income
bysort sex edu: summarize income
by sex, sort: generate n_sex = _N
summarize n_sex
* 3.2.2 foreach (foreachkkh.do)
foreach X of varlist wor01-wor03 {
    tabulate `X' sex
}
foreach var of varlist ybirth income {
    summarize `var', meanonly
    generate `var'_c = `var' - r(mean)
    label variable `var'_c "`var' (centered)"
}
summarize ybirth_c income_c
local i 1
foreach var of varlist eqp* {
    generate equip`i' = `var' - 1
    local i = `i' + 1
}
summarize equip*
* 3.2.3 forvalues
forvalues y = 1(1)5 {
    generate byte d`y' = edu == `y' if edu < .
}
summarize d1-d5
* 3.3 weights
use freqwe, clear
summarize ybirth [fweight = n]
summarize ybirth [fweight = n], detail
tabulate ybirth [fweight = n] if ybirth > 1990
use analwe, clear
summarize ybirth [aweight = n]
summarize ybirth
use data1, clear
summarize income [aweight = xweights]
mean income [pweight = xweights]
tabulate sex [aweight = xweights]
tabulate sex [iweight = xweights]
tabstat income [aweight = xweights], statistics(mean sd p50 n) by(sex)
regress income yedu [pweight = xweights]
regress income yedu [aweight = xweights]
