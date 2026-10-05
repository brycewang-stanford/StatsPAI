* Chapter 10: regression models for categorical dependent variables
use data1, clear
generate owner = renttype == 1 if renttype < .
generate age = 2020 - ybirth
generate east = state >= 11 & state <= 16 if state < .
generate hhinc_k = hhinc/1000
* 10.1 linear probability model
regress owner yedu
regress owner yedu, vce(robust)
* 10.2 odds
tabulate owner east, column
tabulate owner east, chi2
cc owner east
cs owner east
tabodds owner yedu
* 10.3 logistic regression
logit owner yedu
summarize age, meanonly
generate age_c = age - r(mean)
logit owner age_c hhinc_k
logit owner age_c hhinc_k east
display e(ll)
display e(ll_0)
display e(chi2)
display e(r2_p)
display 1 - e(ll)/e(ll_0)
display exp(_b[east])
logit owner age_c hhinc_k east, or
logistic owner age_c hhinc_k east
logit owner age_c hhinc_k east, nolog
predict phat
predict xb, xb
predict stdp, stdp
summarize phat xb stdp
margins
margins, at(age_c = (-30(15)30))
margins, at(east = (0 1))
margins, dydx(*)
margins, dydx(*) atmeans
margins, dydx(age_c) at(east = (0 1))
margins, at(hhinc_k = (20(20)100)) atmeans
logit owner age_c hhinc_k i.east
margins east
margins, dydx(east)
margins east, pwcompare
* 10.3.3 model fit
logit owner age_c hhinc_k east
estat classification
estat classification, cutoff(.3)
estat gof
estat gof, group(10)
lroc, nograph
lsens, nograph
estat ic
* 10.4 diagnostics
predict p2
predict dx2, dx2
predict ddev, ddeviance
predict dbeta, dbeta
predict rsta, rstandard
predict resp, residuals
predict dev, deviance
predict hat, hat
predict num, number
summarize p2 dx2 ddev dbeta rsta resp dev hat num
linktest
* 10.5 likelihood-ratio test
logit owner age_c hhinc_k east
estimates store full
logit owner age_c hhinc_k if e(sample)
estimates store restricted
lrtest full restricted
logit owner age_c hhinc_k east
test east
test age_c hhinc_k
* 10.6 refined models
logit owner c.age_c##c.age_c hhinc_k east
margins, at(age_c = (-30(15)30))
logit owner c.age_c##i.east hhinc_k
margins east, at(age_c = (-30(15)30))
margins, dydx(age_c) over(east)
logit owner i.edu hhinc_k
testparm i.edu
logit owner age_c hhinc_k east, vce(robust)
logit owner age_c hhinc_k east, vce(cluster hid2020)
capture noisily logit owner age_c hhinc_k east [pweight = xweights]
glm owner age_c hhinc_k east [pweight = xweights], family(binomial) link(logit)
* 10.7.1 probit
probit owner age_c hhinc_k east
margins, dydx(*)
predict pp
summarize pp
estat classification
cloglog owner age_c hhinc_k east
* 10.7.2 multinomial logit
generate party = pib if pib <= 6
mlogit party yedu age_c
mlogit party yedu age_c, baseoutcome(2)
mlogit party yedu age_c, rrr
margins, dydx(yedu) predict(outcome(1))
margins, dydx(yedu)
predict pm1 pm2 pm3 pm4 pm5 pm6
summarize pm1-pm6
test [1]yedu = [4]yedu
* 10.7.3 ordinal
ologit polint yedu age_c
ologit polint yedu age_c, or
margins, dydx(yedu)
margins, dydx(yedu) predict(outcome(1))
predict po1 po2 po3 po4
summarize po1-po4
oprobit polint yedu age_c
ologit heval yedu age_c i.sex
* titanic
use titanic2, clear
logit survived i.class men age
logit survived i.class##i.men age
margins class#men
logistic survived i.class men age
estat classification
poisson age i.class men
