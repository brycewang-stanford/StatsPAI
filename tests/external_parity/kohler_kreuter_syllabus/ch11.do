* Chapter 11: reading and writing data
* 11.2.1 system files
import excel popst1.xls, firstrow clear
describe, short
summarize
import spss popst1.sav, clear
summarize
import sasxport8 popst1.v8xpt, clear
summarize
* 11.2.2 text files
import delimited popst1.raw, clear encoding(utf8)
summarize
import delimited state_name area pop_total pop_male pop_female pop_dens using popst1.raw, clear encoding(utf8)
summarize
import delimited popst2.raw, clear varnames(1) encoding(utf8)
summarize
import delimited popst5.raw, clear delimiters("\t") encoding(utf8)
summarize
infile str22 state_name area pop_total pop_male pop_female pop_dens using popst4.raw, clear
summarize
infile using popst5kkh.dct, clear
summarize
infix str22 state 1-22 area 23-27 pop_total 28-35 pop_male 36-42 pop_female 43-49 pop_dens 50-53 using popst6.raw, clear
summarize
infile using popst6kkh.dct, clear
summarize
* 11.3.2 input
clear
input persnr time lsat age
1 1968 8 28
1 1969 6 29
1 1970 5 30
2 1968 5 26
2 1969 2 27
2 1970 1 28
end
summarize
list
clear
input str10 name score
"John" 8
"Paul" 5
"George" 4
"Ringo" 9
end
summarize score
count if name == "Paul"
* 11.5.2 merge
use data1, clear
keep pid hid2020 ybirth sex
sort pid
save _m_person, replace
use data1, clear
keep pid income yedu emp
keep if income < .
sort pid
save _m_income, replace
use data1, clear
bysort hid2020: keep if _n == 1
keep hid2020 hhinc hhsize state
save _m_hh, replace
use _m_person, clear
merge 1:1 pid using _m_income
tabulate _merge
summarize income yedu if _merge == 3
drop _merge
merge m:1 hid2020 using _m_hh
tabulate _merge
summarize hhinc hhsize
drop _merge
merge 1:1 pid using _m_income, keep(match) nogenerate keepusing(yedu) update
count
use _m_person, clear
merge 1:1 pid using _m_income, keep(master match) generate(src)
tabulate src
use _m_person, clear
merge 1:1 pid using _m_income, assert(master match) keepusing(income)
summarize income
use _m_hh, clear
merge 1:m hid2020 using _m_person
tabulate _merge
count
use _m_person, clear
joinby hid2020 using _m_hh
count
* 11.5.3 append
use _m_person, clear
keep if sex == 1
save _m_men, replace
use _m_person, clear
keep if sex == 2
append using _m_men
count
tabulate sex
append using _m_income, generate(fromincome)
tabulate fromincome
summarize
* 11.5.4 frames
frame reset
use data1, clear
frame create hh
frame change hh
use _m_hh, clear
summarize hhinc
frame change default
frlink m:1 hid2020, frame(hh)
frget hhinc2 = hhinc, from(hh)
summarize hhinc hhinc2
frame hh: summarize hhsize
frame create results str10 name mean sd
foreach v of varlist income rent size {
    summarize `v'
    frame post results ("`v'") (r(mean)) (r(sd))
}
frame results: list
frame results: summarize mean sd
frame dir
frame drop hh
* collapse / contract / reshape as alternatives
use data1, clear
collapse (mean) income hhinc (sd) sd_income = income (count) n = income (median) med = income, by(state)
list
summarize
use data1, clear
collapse (mean) income [aweight = xweights], by(sex edu)
list
use data1, clear
contract sex edu
list
use data1, clear
statsby mean = r(mean) sd = r(sd), by(state) clear: summarize income
list
use data1, clear
keep pid wor01-wor05
reshape long wor, i(pid) j(item) string
tabulate item, summarize(wor)
reshape wide wor, i(pid) j(item) string
summarize
use data1, clear
keep sex edu income
drop if missing(edu)
collapse (mean) income, by(sex edu)
reshape wide income, i(edu) j(sex)
list
expand 2
count
duplicates report
duplicates drop
count
* 11.6 saving and exporting
use data1, clear
keep in 1/20
keep pid sex ybirth income
export delimited using _out.csv, replace
import delimited _out.csv, clear
summarize
export excel using _out.xlsx, firstrow(variables) replace
import excel _out.xlsx, firstrow clear
summarize
outfile using _out.raw, replace comma
compress
describe, short
order income, first
describe, simple
rename income inc
rename (sex ybirth) (gender birthyear)
describe, simple
