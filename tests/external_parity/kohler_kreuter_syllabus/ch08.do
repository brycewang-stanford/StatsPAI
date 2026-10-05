* Chapter 8: statistical inference
* 8.1 random numbers and samples
clear
set obs 1000
set seed 42
generate u = runiform()
generate z = rnormal()
generate z2 = rnormal(5, 2)
generate coin = rbinomial(1, .5)
generate die = ceil(6*runiform())
generate die2 = runiformint(1, 6)
summarize u z z2 coin die die2
use data1, clear
set seed 42
sample 10
count
use data1, clear
sample 100, count
count
use data1, clear
bsample 500
count
use data1, clear
bsample, cluster(psu)
count
* 8.1.4 the sampling distribution (grci.do)
use berlin, clear
summarize ybirth
set seed 42
sample 100, count
mean ybirth
* 8.2.1 standard errors for simple random samples
use data1, clear
mean ybirth
mean income
mean income, over(sex)
mean income if emp == 1, over(edu)
proportion pia
total hhsize
ratio (income/hhsize)
* 8.2.2 complex samples
svyset psu [pweight = xweights], strata(strata)
svydescribe
svy: mean ybirth
svy: mean income
svyset psu [pweight = xweights], strata(strata) singleunit(certainty)
svy: mean income
svy: mean income, over(sex)
svy: proportion pia
svy: ratio (income/hhsize)
svy: total hhsize
svy: total hhsize, over(sex)
svy: tabulate sex pia
svy: tabulate sex pia, row se ci
svy: tabulate sex pia, count format(%12.0f)
svy: tabulate edu sex
svy: tabulate edu
svy: regress income yedu i.sex
svy: logit pia yedu
svy, subpop(if sex == 2): mean income
svyset psu [pweight = xweights], strata(strata) singleunit(scaled)
svy: mean income
svyset psu [pweight = xweights], strata(strata) singleunit(centered)
svy: mean income
svyset psu [pweight = xweights]
svy: mean income
estat effects
svyset [pweight = xweights]
svy: mean income
svyset psu
svy: mean income
mean income [pweight = xweights]
mean income [pweight = xweights], vce(cluster psu)
mean income, vce(cluster psu)
mean income [pweight = xweights], over(sex)
mean income, over(sex) vce(cluster psu)
mean income [aweight = xweights]
mean income [fweight = hhsize]
proportion pia [pweight = xweights]
proportion pia, vce(cluster psu)
proportion pia, citype(wald)
proportion edu, over(sex)
ratio (income/hhsize), over(sex)
total hhsize, over(sex)
* 8.2.3 nonresponse: poststratification
generate age = 2020 - ybirth
generate agegr = irecode(age, 39, 59) + 1
generate poststr = sex*10 + agegr
bysort poststr: generate npost = _N
generate popsize = npost * 17000
svyset psu [pweight = dweight], strata(strata) poststrata(poststr) postweight(popsize) singleunit(certainty)
svy: mean income
svy: proportion pia
* item nonresponse: multiple imputation
use data1, clear
generate age = 2020 - ybirth
misstable summarize income yedu age sex
misstable patterns income yedu age sex
mi set mlong
mi register imputed income yedu
mi register regular age sex
mi impute chained (regress) income yedu = age sex, add(5) rseed(42)
mi estimate: mean income
mi estimate: regress income yedu age i.sex
* 8.2.4 uses of standard errors
use data1, clear
ci means income
ci means income ybirth, level(90)
generate byte supp = pia == 1 if pia < .
ci proportions supp
ci proportions supp, wald
ci proportions supp, wilson
ci proportions supp, agresti
ci proportions supp, jeffreys
ci variances income
cii means 100 25000 15000
cii proportions 100 40
cii proportions 100 40, wilson
ttest income == 30000
ttest income == 30000, level(90)
ttest income, by(sex)
ttest income, by(sex) unequal
ttest income, by(sex) unequal welch
ttest wor01 == wor02
ttesti 100 25000 15000 30000
ttesti 100 25000 15000 120 30000 20000
ttesti 100 25000 15000 120 30000 20000, unequal
prtest supp == .5
prtest supp, by(sex)
prtesti 100 .4 120 .5
bitest supp == .5
bitesti 100 40 .5
ztest income == 30000, sd(30000)
mean income, over(sex)
test _b[c.income@1.sex] = _b[c.income@2.sex]
lincom _b[c.income@1.sex] - _b[c.income@2.sex]
display invnormal(.975)
display invttail(4000, .025)
display 2*ttail(4000, 1.96)
display 2*(1 - normal(1.96))
* 8.3 causal inference: the effect of third-class tickets
use titanic2, clear
tabulate class survived, row
generate third = class == 3
tabulate third survived, row chi2
ttest survived, by(third)
regress survived third
regress survived third men age
regress survived i.class men age
regress survived third men age, vce(robust)
regress survived i.third##i.men age
margins third
margins third, dydx(men)
margins, dydx(third)
teffects ra (survived men age) (third)
teffects ra (survived men age) (third), atet
teffects ra (survived men age) (third), pomeans
teffects ipw (survived) (third men age)
teffects ipw (survived) (third men age), atet
teffects ipwra (survived men age) (third men age)
teffects aipw (survived men age) (third men age)
teffects nnmatch (survived men age) (third)
teffects nnmatch (survived men age) (third), ematch(men)
teffects nnmatch (survived men age) (third), atet nneighbor(3)
teffects nnmatch (survived men age) (third), biasadj(age)
teffects psmatch (survived) (third men age)
teffects psmatch (survived) (third men age), atet
tebalance summarize
teffects ipw (survived) (third men age)
tebalance summarize
teffects overlap, nodraw
* 8.4 predictive inference: the best split (ml_step1.do)
use sex ybirth yedu emp income using data1, clear
drop if mi(sex,ybirth,yedu,emp,income)
summarize income
gen tss = (income - r(mean))^2
summarize tss, meanonly
local tss = r(sum)
display `tss'
summarize income if yedu <= 12, meanonly
gen rss = (income - r(mean))^2 if yedu <= 12
summarize income if yedu > 12, meanonly
replace rss = (income - r(mean))^2 if yedu > 12
summarize rss, meanonly
display (`tss' - r(sum))/`tss'
drop rss
local i 1
gen rss = .
gen splitname = ""
gen r2 = .
foreach var of varlist sex ybirth yedu emp {
	levelsof `var', local(K)
	foreach k of local K {
		summarize income if `var' == `k', meanonly
		replace rss = (income - r(mean))^2 if `var' == `k'
		summarize income if `var' != `k', meanonly
		replace rss = (income - r(mean))^2 if `var' != `k'
		summarize rss, meanonly
		replace splitname = "`var'==`k'" in `i'
		replace r2 = (`tss' - r(sum))/`tss' in `i++'
	}
}
summarize r2
count if splitname != ""
gsort -r2
list splitname r2 in 1/5
* 8.4.5 training and test
use sex ybirth yedu emp income using data1, clear
drop if mi(sex,ybirth,yedu,emp,income)
generate train = mod(_n, 2)
regress income yedu ybirth i.sex i.emp if train
predict yhat
generate se = (income - yhat)^2
summarize se if train
display "RMSE train = " sqrt(r(mean))
summarize se if !train
display "RMSE test = " sqrt(r(mean))
correlate income yhat if !train
display "R2 test = " r(rho)^2
set seed 42
splitsample, generate(svar) split(.7 .3)
tabulate svar
