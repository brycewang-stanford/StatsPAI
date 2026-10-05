* Chapter 9: introduction to linear regression
use anscombe, clear
regress y1 x1
regress y2 x2
regress y3 x3
regress y4 x4
correlate y1 x1
use data1, clear
* 9.1 simple regression
regress rent size
predict rent_hat
predict res, residuals
summarize rent_hat res
display _b[_cons] + _b[size]*1000
display e(r2)
display e(rmse)
display e(mss)/(e(mss) + e(rss))
display e(F)
display e(df_r)
display _se[size]
display _b[size]/_se[size]
correlate rent size
display r(rho)^2
regress rent size, level(90)
regress rent size, noconstant
regress rent size, noheader
* 9.2 multiple regression
generate age = 2020 - ybirth
regress rent size hhinc hhsize
regress rent size hhinc hhsize, beta
display e(r2_a)
regress rent size i.area1
regress rent size ib3.area1
regress rent size i.renttype i.state
testparm i.state
regress rent c.size##i.sex
regress rent c.size##c.hhinc
regress rent c.size c.size#c.size
regress rent c.age##c.age size
margins, at(age = (20(20)80))
margins, dydx(age) at(age = (20(20)80))
margins, dydx(size)
generate lrent = ln(rent)
generate lsize = ln(size)
regress lrent lsize
regress lrent size
regress rent lsize
generate men = sex == 1
regress income men yedu age
regress income i.sex##c.yedu age
margins sex
margins sex, at(yedu = (9 12 18))
margins, dydx(sex) at(yedu = (9 12 18))
margins, dydx(*)
contrast sex
test yedu = 2000
test yedu age
test 2.sex#c.yedu
lincom yedu + 2.sex#c.yedu
* 9.3 diagnostics
regress rent size hhinc hhsize
predict yh
predict rs, rstandard
predict rst, rstudent
predict lev, leverage
predict cook, cooksd
predict dfit, dfits
predict stdp, stdp
predict stdf, stdf
predict stdr, stdr
predict cvr, covratio
predict wd, welsch
dfbeta
summarize yh rs rst lev cook dfit stdp stdf stdr cvr wd _dfbeta_1 _dfbeta_2 _dfbeta_3
count if cook > 4/e(N) & cook < .
count if abs(_dfbeta_1) > 2/sqrt(e(N)) & _dfbeta_1 < .
estat vif
estat hettest
estat hettest, rhs
estat hettest size hhinc, iid
estat hettest, fstat
estat imtest
estat imtest, white
estat ovtest
estat ovtest, rhs
estat ic
estat summarize
estat vce
linktest
regress rent size hhinc hhsize, vce(robust)
regress rent size hhinc hhsize, vce(hc2)
regress rent size hhinc hhsize, vce(hc3)
regress rent size hhinc hhsize, vce(cluster hid2020)
regress rent size hhinc hhsize, vce(cluster psu)
* 9.3.3 autocorrelation
use data2agg, clear
tsset wave
regress lsat hhinc
estat dwatson
estat bgodfrey
estat durbinalt
regress lsat hhinc L.lsat
newey lsat hhinc, lag(2)
prais lsat hhinc
* 9.4 reporting
use data1, clear
generate age = 2020 - ybirth
generate men = sex == 1
regress income men age
estimates store m1
regress income men age yedu
estimates store m2
regress income men age yedu i.emp
estimates store m3
estimates table m1 m2 m3, se stats(N r2 r2_a)
estimates table m1 m2 m3, b(%9.3f) star
estimates restore m2
display _b[yedu]
nestreg: regress income (men age) (yedu) (i.emp)
regress income men age yedu
predict inc_hat
margins, at(age = (20(10)60) men = (0 1))
* 9.5.2 panel data (beatles)
use beatles, clear
regress lsat age
regress lsat age i.persnr
xtset persnr time
xtreg lsat age, fe
xtreg lsat age, be
xtreg lsat age, re
egen mlsat = mean(lsat), by(persnr)
egen mage = mean(age), by(persnr)
generate wlsat = lsat - mlsat
generate wage = age - mage
regress wlsat wage
regress wlsat wage, noconstant
regress mlsat mage
regress D.lsat D.age
regress D.lsat D.age, noconstant
areg lsat age, absorb(persnr)
* from wide to long
use data2w, clear
reshape long hhinc lsat mar whours, i(pid) j(wave)
summarize hhinc lsat mar whours wave
xtset pid wave
xtdescribe
xtsum hhinc lsat
generate lhhinc = ln(hhinc) if hhinc > 0
regress lsat lhhinc
regress lsat lhhinc, vce(cluster pid)
xtreg lsat lhhinc, fe
xtreg lsat lhhinc, fe vce(cluster pid)
xtreg lsat lhhinc, be
xtreg lsat lhhinc, re
xtreg lsat lhhinc i.wave, fe
xtreg lsat lhhinc i.wave, fe vce(cluster pid)
testparm i.wave
regress D.lsat D.lhhinc
regress D.lsat D.lhhinc, vce(cluster pid)
xtreg lsat lhhinc whours i.mar, fe
estimates store fe
xtreg lsat lhhinc whours i.mar, re
estimates store re
hausman fe re
areg lsat lhhinc, absorb(pid)
generate married = mar == 1 if mar < .
xtreg lsat married i.wave, fe vce(cluster pid)
* 9.5.3 instrumental variables
use data1, clear
generate age = 2020 - ybirth
generate lninc = ln(income) if income > 0
ivregress 2sls lninc age (yedu = i.edu)
estat firststage
estat endogenous
estat overid
ivregress 2sls lninc age (yedu = i.edu), vce(robust)
ivregress 2sls lninc age (yedu = i.edu), first
ivregress liml lninc age (yedu = i.edu)
ivregress gmm lninc age (yedu = i.edu)
regress lninc age yedu
