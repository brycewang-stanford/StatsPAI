* Regenerates extmiss118.dta. Run from this directory with a real Stata:
*     stata-mp -b do make_fixture.do
* The file is written by Stata itself so that the extended missing values
* (.a, .b) and the labels attached to them are stored the way Stata stores
* them, which no Python writer reproduces.  Never edit the .dta by hand.
version 14
clear
set obs 12
generate long id = _n
generate byte region = mod(_n, 4) + 1
replace region = .a in 3
replace region = .a in 7
replace region = .b in 9
replace region = .  in 11
generate double income = 1000 * _n
replace income = .b in 2
replace income = .  in 5
generate byte female = mod(_n, 2)
generate byte agree = mod(_n, 3)
label define regionlbl 1 "North" 2 "South" 3 "East" 4 "West" ///
    .a "Refused" .b "Don't know"
label values region regionlbl
label define yn 0 "No" 1 "Yes"
label values female yn
label values agree yn
label variable region "Census region"
label variable income "Annual income"
label variable female "Female"
label variable agree "Agrees"
label data "Extended missing fixture"
note: built by make_fixture.do
note region: .a refused, .b did not know
save extmiss118.dta, replace
