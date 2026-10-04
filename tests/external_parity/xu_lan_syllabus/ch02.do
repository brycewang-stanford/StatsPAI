use auto, clear
regress price weight
predict yhat
predict ehat, residuals
summarize yhat ehat
regress price weight, noconstant
regress price weight, robust
regress price weight, level(90)
qreg price weight
qreg price weight length foreign, quantile(.25)
qreg price weight length foreign, quantile(.75)
sqreg price weight length foreign, quantiles(.25 .5 .75) reps(50)
bsqreg price weight length, reps(50)
iqreg price weight length, quantiles(.25 .75) reps(50)
