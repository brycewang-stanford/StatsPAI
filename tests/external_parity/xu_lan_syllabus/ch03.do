use auto, clear
regress price mpg weight length foreign
estat ic
estat vif
test mpg weight
test mpg = weight
estat hettest
estat hettest, rhs
estat hettest, iid
estat hettest mpg weight, fstat
estat imtest, white
estat imtest
estat szroeter, rhs
estat ovtest
regress price mpg weight length foreign, vce(robust)
regress price mpg weight length foreign, vce(hc3)
regress price mpg weight length foreign [aweight=1/weight]
vwls price mpg weight, sd(length)
stepwise, pr(.2): regress price mpg weight length foreign headroom trunk turn
stepwise, pe(.1): regress price mpg weight length foreign headroom trunk turn
use klein, clear
tsset yr
regress consump wagepriv wagegovt
estat dwatson
estat durbinalt
estat durbinalt, lags(2)
estat bgodfrey
estat bgodfrey, lags(2)
estat archlm
estat archlm, lags(2)
prais consump wagepriv wagegovt
prais consump wagepriv wagegovt, corc
prais consump wagepriv wagegovt, twostep
newey consump wagepriv wagegovt, lag(2)
regress consump wagepriv wagegovt L.consump
estat durbinalt
predict e, residuals
wntestq e
wntestq e, lags(4)
