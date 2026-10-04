use mroz, clear
regress lwage educ exper expersq
ivregress 2sls lwage exper expersq (educ = motheduc fatheduc)
estat endogenous
estat firststage
estat overid
ivregress 2sls lwage exper expersq (educ = motheduc fatheduc), vce(robust)
estat endogenous
estat firststage
estat overid
ivregress 2sls lwage exper expersq (educ = fatheduc), first
ivregress liml lwage exper expersq (educ = motheduc fatheduc)
ivregress gmm lwage exper expersq (educ = motheduc fatheduc)
estat overid
regress educ exper expersq motheduc fatheduc
test motheduc fatheduc
predict vhat, residuals
regress lwage educ exper expersq vhat
ivregress 2sls lwage exper expersq (educ = motheduc fatheduc), small
quietly regress lwage educ exper expersq
estimates store ols
quietly ivregress 2sls lwage exper expersq (educ = motheduc fatheduc)
estimates store iv
hausman iv ols, sigmamore
hausman iv ols
