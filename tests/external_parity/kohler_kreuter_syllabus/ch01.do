* Chapter 1: the first time (an1cmdkkh.do, an2kkh.do)
use data1, clear
describe, short
drop ymove - xweights
summarize income
summarize
summarize income if sex == 1
summarize income if sex == 2
mvdecode income, mv(0=.c)
sort sex
by sex: summarize income
summarize income, detail
by sex: summarize income if edu == 4, detail
tabulate sex
tabulate emp sex
tabulate emp sex, column nofreq
generate men = 1 if sex == 1
replace men = 0 if sex == 2
generate emp3 = emp if emp <= 2
replace emp3 = 3 if emp == 4
tabulate emp emp3, missing
label variable emp3 "Status of employment (3 categories)"
label define emp3 1 "Full time" 2 "Part time" 3 "Irregular"
label values emp3 emp3
regress income men i.emp3
* an2kkh.do
use data1, clear
drop ymove - xweights
mvdecode income, mv(0=.c)
generate men = sex == 1
generate emp3 = emp if emp!=5
regress income men i.emp3
generate age = 2020 - ybirth
summarize age if !mi(income,emp), meanonly
generate c_age = age - r(mean) if !mi(income,emp)
summarize c_age
regress income i.emp c.men##c.c_age
