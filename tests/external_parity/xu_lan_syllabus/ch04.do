use nlsw88, clear
generate lnwage = ln(wage)
generate exp2 = ttl_exp^2
regress lnwage grade ttl_exp exp2 tenure
regress lnwage grade ttl_exp exp2 tenure i.race
testparm i.race
regress lnwage grade ttl_exp tenure union
regress lnwage c.grade##i.union ttl_exp tenure
test 1.union 1.union#c.grade
lincom grade + 1.union#c.grade
regress lnwage grade ttl_exp tenure
test grade + ttl_exp = 0.1
test (grade = 0.07) (ttl_exp = 0.03)
nlcom _b[grade]/_b[ttl_exp]
regress lnwage grade ttl_exp tenure if union==1
regress lnwage grade ttl_exp tenure if union==0
regress lnwage grade ttl_exp tenure if union<.
probit union grade ttl_exp tenure south
margins, dydx(*)
margins, dydx(*) atmeans
estat classification
logit union grade ttl_exp tenure south
logit union grade ttl_exp tenure south, or
margins, dydx(*)
regress union grade ttl_exp tenure south, robust
use mroz, clear
tobit hours nwifeinc educ exper expersq age kidslt6 kidsge6, ll(0)
margins, dydx(*) predict(ystar(0,.))
margins, dydx(*) predict(e(0,.))
margins, dydx(*) predict(pr(0,.))
truncreg hours nwifeinc educ exper expersq age kidslt6 kidsge6 if hours>0, ll(0)
regress hours nwifeinc educ exper expersq age kidslt6 kidsge6
heckman lwage educ exper expersq, select(inlf = nwifeinc educ exper expersq age kidslt6 kidsge6) twostep
