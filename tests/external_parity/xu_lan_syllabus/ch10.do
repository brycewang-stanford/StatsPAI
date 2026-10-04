use nlswork, clear
xtset idcode year
xtdescribe
xtsum ln_wage ttl_exp tenure
regress ln_wage ttl_exp tenure union south, vce(cluster idcode)
xtreg ln_wage ttl_exp tenure union south, be
xtreg ln_wage ttl_exp tenure union south, fe
estimates store fe
xtreg ln_wage ttl_exp tenure union south, re
estimates store re
hausman fe re
hausman fe re, sigmamore
xttest0
xtreg ln_wage ttl_exp tenure union south, fe vce(cluster idcode)
xtreg ln_wage ttl_exp tenure union south, re vce(cluster idcode)
xtreg ln_wage ttl_exp tenure union south i.year, fe vce(cluster idcode)
testparm i.year
areg ln_wage ttl_exp tenure union south, absorb(idcode)
regress D.(ln_wage ttl_exp tenure union south), noconstant
xtreg ln_wage ttl_exp tenure union south, mle
use grunfeld, clear
xtset company year
xtreg invest mvalue kstock, fe
xtreg invest mvalue kstock, re
xttest0
xtreg invest mvalue kstock i.year, fe
regress invest mvalue kstock i.company
generate treated = company <= 5
generate post = year >= 1945
regress invest i.treated##i.post
regress invest i.treated##i.post mvalue kstock, vce(cluster company)
xtreg invest c.treated#c.post mvalue kstock i.year, fe vce(cluster company)
generate big = mvalue > 1000 if year == 1944
bysort company (big): replace big = big[1]
regress invest i.treated##i.post##i.big, vce(cluster company)
