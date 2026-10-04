use womenwk, clear
regress wage education age
heckman wage education age, select(married children education age) twostep
heckman wage education age, select(married children education age)
heckman wage education age, select(married children education age) vce(robust)
heckman wage education age, select(married children education age) twostep mills(imr)
summarize imr
generate work = wage < .
probit work married children education age
predict xb, xb
generate lambda = normalden(xb)/normal(xb)
regress wage education age lambda
use union3, clear
regress wage age grade smsa black tenure union
etregress wage age grade smsa black tenure, treat(union = south black tenure)
etregress wage age grade smsa black tenure, treat(union = south black tenure) twostep
etregress wage age grade smsa black tenure, treat(union = south black tenure) vce(robust)
etregress wage age grade smsa black tenure, treat(union = south black tenure) poutcomes
margins r.union, vce(unconditional)
