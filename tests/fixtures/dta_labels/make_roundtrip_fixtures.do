* Regenerates meta118.dta, meta117.dta, meta115.dta and alias120.dta.
* Run from this directory with a real Stata (18 or later for the alias file):
*     stata-mp -b do make_roundtrip_fixtures.do
* These hold what a .dta file stores beyond its rows and plain labels: a
* value-label set shared by several variables under its own name, a set
* that is defined and attached to nothing, notes, characteristics, an xtset
* declaration, display formats, and extended missing values in every
* numeric storage type.  Never edit the .dta files by hand.
version 14
clear all
set obs 8
generate long id = ceil(_n / 2)
generate int year = 2019 + mod(_n, 2)
generate byte q1 = mod(_n, 2)
generate byte q2 = mod(_n, 3) > 0
generate int score = 100 * _n
generate long big = 100000 * _n
generate float ratio = _n / 4
generate double wage = 1234.5 * _n
generate double day = td(1jan2020) + 31 * _n
generate str8 name = "p" + string(_n)
replace q1 = .a in 2
replace q1 = .z in 5
replace q2 = .  in 3
replace score = .b in 4
replace big = .c in 6
replace ratio = .d in 7
replace wage = .e in 1
replace day = .a in 8
label define yesno 0 "No" 1 "Yes" .a "Refused"
label values q1 yesno
label values q2 yesno
label define unused 1 "Kept by Stata, attached to nothing"
label variable q1 "Owns a car"
label variable q2 "Owns a house"
label variable wage "Monthly wage"
label data "Metadata fixture"
format wage %12.2fc
format day %tdCCYY-NN-DD
format name %-8s
note: first dataset note
note: second dataset note
note wage: top-coded
char _dta[source] "make_roundtrip_fixtures.do"
char score[unit] "points"
xtset id year
save meta118.dta, replace
saveold meta117.dta, version(13) replace
saveold meta115.dta, version(12) replace

* --- format 120: an alias variable (Stata 18+) ---------------------------
clear all
frame create other
frame other: set obs 3
frame other: generate long id = _n
frame other: generate double val = _n * 2.5
set obs 3
generate long id = _n
generate byte own = 7
label define seven 7 "Seven"
label values own seven
note own: kept beside an alias
frlink 1:1 id, frame(other)
fralias add val, from(other)
sort own id
save alias120.dta, replace
