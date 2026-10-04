use auto, clear
summarize price mpg weight
summarize price, detail
tabstat price mpg weight, statistics(mean sd skewness kurtosis min max)
correlate price mpg weight
pwcorr price mpg weight, sig
ttest mpg == 20
ttest mpg, by(foreign)
ttest mpg, by(foreign) unequal
sdtest mpg == 5
sdtest mpg, by(foreign)
ztest mpg == 20, sd(6)
ci means mpg price
ci variances mpg
sktest mpg price
swilk mpg price
tabulate rep78 foreign, chi2
prtest foreign == 0.4
bitest foreign == 0.4
ttesti 10 88 1.14 85
sdtesti 10 88 1.14 2
ztesti 10 88 0.7071 85
