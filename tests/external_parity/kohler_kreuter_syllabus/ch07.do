* Chapter 7: describing and comparing distributions
use data1, clear
* 7.2.1 tables
tabulate pib
tabulate pib, missing
tabulate pib, sort
tabulate pib, nolabel
tab1 sex emp edu
tabulate sex pib
tabulate sex pib, row
tabulate sex pib, column nofreq
tabulate sex pib, cell
tabulate sex pib, chi2
tabulate sex pib, chi2 lrchi2 V
tabulate sex pib, expected
tabulate edu heval, gamma taub
tabulate sex pia, exact
tabulate sex pia, chi2 V
tab2 sex emp mar, chi2
tabulate sex, summarize(income)
tabulate edu sex, summarize(income) means
tabulate edu, summarize(income) nostandard nofreq
bysort sex: tabulate edu pia, chi2
table edu sex, statistic(mean income)
table edu, statistic(frequency) statistic(mean income) statistic(sd income)
* 7.3.1 grouped data
generate age = 2020 - ybirth
generate age_g = recode(age, 30, 50, 70, 110)
tabulate age_g
egen inc_g = cut(income), group(5)
tabulate inc_g, summarize(income)
xtile inc_q = income, nquantiles(4)
tabulate inc_q, summarize(income)
xtile inc_d = income if income > 0, nquantiles(10)
tabulate inc_d
pctile pct = income, nquantiles(10)
list pct in 1/9
_pctile income, percentiles(10 25 50 75 90)
return list
centile income, centile(25 50 75)
* 7.3.2 statistics
summarize income
summarize income, detail
summarize income if income > 0, detail
summarize income size rent hhinc, detail
tabstat income, statistics(count mean sd min max)
tabstat income, statistics(q)
tabstat income, statistics(p5 p10 p25 p50 p75 p90 p95 p99)
tabstat income, statistics(mean median sd var cv semean skewness kurtosis iqr range sum)
tabstat income size rent, statistics(mean sd p50 iqr) columns(statistics)
tabstat income, statistics(count mean sd p25 p50 p75) by(edu)
tabstat income size, statistics(mean p50) by(sex) nototal
tabstat income, statistics(mean sd) by(state) missing
mean income
mean income, over(sex)
mean income size rent
proportion sex
proportion edu
ameans income if income > 0
correlate income yedu age hhinc
correlate income yedu, covariance
pwcorr income yedu age hhinc, obs sig
spearman income yedu
ktau heval edu
* 7.3.3 comparing distributions
ttest income, by(sex)
ranksum income, by(sex)
signrank wor01 = wor02
kwallis income, by(edu)
ksmirnov income, by(sex)
median income, by(sex)
sdtest income, by(sex)
robvar income, by(sex)
oneway income edu, tabulate
oneway income edu, bonferroni
anova income edu
sktest income
swilk rent
kdensity income if income < 100000, generate(kx kd) n(50) nodraw
summarize kx kd
lv income
codebook sex emp, compact
inspect income
