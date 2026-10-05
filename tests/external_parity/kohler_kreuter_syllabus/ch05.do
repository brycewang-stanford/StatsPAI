* Chapter 5: creating and changing variables
use data1, clear
* 5.1 generate and replace
generate age = 2020 - ybirth
summarize age
generate age2 = age^2
generate lsize = ln(size)
generate rent_sqm = rent/size
summarize age2 lsize rent_sqm
generate byte men = sex == 1
generate byte old = age >= 65 if !missing(age)
tabulate men old
generate minc = income/12
replace minc = . if minc == 0
summarize minc
* 5.1.3 useful functions
generate age_g1 = recode(age, 30, 45, 60, 100)
tabulate age_g1
generate age_g2 = irecode(age, 30, 45, 60)
tabulate age_g2
generate age_g3 = autocode(age, 4, 17, 100)
tabulate age_g3
generate inc_cat = cond(income < 20000, 1, cond(income < 50000, 2, 3)) if !missing(income)
tabulate inc_cat
generate west = inrange(state, 0, 10) if !missing(state)
generate city = inlist(state, 2, 4, 11) if !missing(state)
tabulate west city
generate r1 = round(rent, 100)
generate r2 = int(rent/100)
generate r3 = floor(size/10)*10
generate r4 = ceil(size/10)*10
summarize r1 r2 r3 r4
generate m = max(wor01, wor02, wor03)
generate mi = min(wor01, wor02, wor03)
generate nmiss = missing(wor01) + missing(wor02) + missing(wor03)
summarize m mi nmiss
generate sgn = sign(income - 30000)
generate ab = abs(income - 30000)
generate lg = log10(hhinc) if hhinc > 0
generate ex = exp(yedu/10)
generate sq = sqrt(hhinc) if hhinc >= 0
generate md = mod(ybirth, 10)
summarize sgn ab lg ex sq md
* 5.2 missing values
mvdecode income, mv(0=.c)
summarize income
count if income == .c
count if missing(income)
count if income >= .
mvencode income, mv(.c=0)
summarize income
replace income = .a if income == 0
generate inc_obs = !missing(income)
tabulate inc_obs
mvdecode rooms size, mv(-1)
summarize rooms size
* 5.3 labels
label variable age "Age in years"
label define yesno 0 "no" 1 "yes"
label values men yesno
tabulate men
* 5.4.1 recode
recode edu (1 2 = 1) (3 = 2) (4 5 = 3), generate(edu3)
tabulate edu edu3, missing
recode ybirth (min/1949 = 1) (1950/1969 = 2) (1970/max = 3), generate(cohort)
tabulate cohort
recode emp (5 = .) (else = 1), generate(working)
tabulate working, missing
recode wor01 wor02 (1 2 = 1) (3 = 0) (missing = .), prefix(c_)
tabulate c_wor01 c_wor02, missing
recode hhsize (1 = 1) (2 = 2) (3/max = 3), generate(hh3)
tabulate hh3
* 5.4.2 egen
egen inc_mean = mean(income)
egen inc_sd = sd(income)
egen inc_std = std(income)
egen inc_med = median(income)
egen inc_max = max(income), by(state)
egen inc_min = min(income), by(state)
egen inc_n = count(income), by(state)
egen inc_tot = total(income), by(state)
egen inc_p25 = pctile(income), p(25) by(sex)
egen inc_iqr = iqr(income), by(sex)
egen inc_rank = rank(income)
egen inc_rank_f = rank(income), field
egen inc_rank_u = rank(income), unique
summarize inc_*
egen worries = anycount(wor*), values(1)
tabulate worries
egen wor_mean = rowmean(wor01-wor12)
egen wor_miss = rowmiss(wor01-wor12)
egen wor_nonmiss = rownonmiss(wor01-wor12)
egen wor_tot = rowtotal(wor01-wor12)
egen wor_max = rowmax(wor01-wor12)
egen wor_min = rowmin(wor01-wor12)
egen wor_sd = rowsd(wor01-wor12)
summarize wor_*
egen grp = group(sex edu)
tabulate grp
egen tagst = tag(state)
tabulate tagst
egen agecut = cut(age), at(17 30 45 60 110)
tabulate agecut
egen agecut4 = cut(age), group(4)
tabulate agecut4
egen seq3 = seq(), from(1) to(3)
tabulate seq3
* 5.4.3 by, _n, _N
bysort hid2020: generate hhn = _N
bysort hid2020 (ybirth): generate hhrank = _n
bysort hid2020 (ybirth): generate oldest = ybirth[1]
bysort hid2020 (ybirth): generate youngest = ybirth[_N]
bysort hid2020 (ybirth): generate agegap = ybirth - ybirth[_n-1]
bysort hid2020 (ybirth): generate cuminc = sum(income)
bysort hid2020: generate first = _n == 1
summarize hhn hhrank oldest youngest agegap cuminc first
tabulate hhn if first
by hid2020: generate nmen = sum(men)
by hid2020: replace nmen = nmen[_N]
tabulate nmen if first
sort pid
generate id = _n
generate lagy = ybirth[_n-1]
generate leady = ybirth[_n+1]
generate runsum = sum(hhsize)
summarize id lagy leady runsum
* 5.5 strings
use mdb, clear
describe, short
generate lname = strlower(name)
generate uname = strupper(name)
generate len = strlen(name)
generate comma = strpos(name, ",")
generate lastname = substr(name, 1, comma - 1) if comma > 0
generate firstname = strtrim(substr(name, comma + 1, .)) if comma > 0
generate hasdr = strpos(name, "Dr.") > 0
generate initial = substr(name, 1, 1)
generate w2 = word(name, 2)
generate nw = wordcount(name)
generate rev = strreverse(initial)
generate sub1 = subinstr(name, "Dr. ", "", .)
generate rx = regexm(name, "^[A-M]")
generate von = ustrregexm(name, " von ")
summarize len comma hasdr nw rx von
tabulate initial if inlist(initial, "A", "B", "C")
count if lastname == "Adenauer"
encode party, generate(party_n)
tabulate party_n
decode party_n, generate(party_s)
count if party_s == party
generate str3 bys = string(birthyear)
tostring birthyear, generate(by_s)
destring by_s, generate(by_n)
summarize by_n birthyear
generate by2 = real(by_s)
summarize by2
* 5.6 dates and time
generate bdate = mdy(birthmonth, birthday, birthyear)
format bdate %td
summarize bdate
generate byr = year(bdate)
generate bmo = month(bdate)
generate bdy = day(bdate)
generate bdow = dow(bdate)
generate bq = quarter(bdate)
generate bdoy = doy(bdate)
generate bwk = week(bdate)
generate bhalf = halfyear(bdate)
summarize byr bmo bdy bdow bq bdoy bwk bhalf
generate ym = mofd(bdate)
generate yq = qofd(bdate)
generate yy = yofd(bdate)
summarize ym yq yy
generate agestart = (pstart - bdate)/365.25
summarize agestart
generate dur = pend - pstart
summarize dur
display mdy(1, 1, 1960)
display mdy(12, 31, 1999)
display date("2020-03-15", "YMD")
display date("15mar2020", "DMY")
display date("March 15, 2020", "MDY")
display td(15mar2020)
display year(td(15mar2020))
display ym(2020, 3)
display yq(2020, 1)
display dofm(ym(2020, 3))
display %td 21989
display %tdCCYY-NN-DD 21989
use diary, clear
generate double btime = hms(bhour, bmin, 0)
summarize btime
generate double etime = clock(estring, "hm")
summarize etime
generate dmin = (etime - btime)/60000
summarize dmin
display clock("2020-03-15 14:30:00", "YMDhms")
display hh(clock("14:30", "hm"))
display mm(clock("14:30", "hm"))
display hours(3600000)
display minutes(3600000)
display dofc(clock("2020-03-15 14:30:00", "YMDhms"))
display msofhours(1)
display mdyhms(3, 15, 2020, 14, 30, 0)
display dhms(21989, 14, 30, 0)
* 5.7 storage types
use data1, clear
generate x1 = .1
count if x1 == .1
count if x1 == float(.1)
generate double x2 = .1
count if x2 == .1
generate long big = 123456789
generate bigf = 123456789
summarize big bigf
display %15.0f bigf[1]
display float(16777217)
display 0.1 + 0.2 == 0.3
display float(0.1 + 0.2) == float(0.3)
generate byte b = 100
replace b = 200 in 1
summarize b
generate int i16 = 32000
replace i16 = 40000 in 1
summarize i16
display c(maxbyte)
display c(maxint)
display c(maxlong)
display c(epsfloat)
display c(pi)
