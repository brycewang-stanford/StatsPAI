* Chapter 12: do-files for advanced users and user-written programs
use data1, clear
* 12.2.1 local macros
local a income
summarize `a'
local b sex emp
tabulate `b'
local i = 5 + 3
display `i'
local j 5 + 3
display `j'
display "`j'"
local k = `i' * 2
display `k'
local ++k
display `k'
local --k
display `k'
local a1 2 + 2
local a2 = 2 + 2
display "`a1' and `a2'"
display `a1' * 2
local x1 a
local x2 b
local x `x1'`x2'
display "`x'"
local x `x1' `x2'
display "`x'"
local list a b c
local list `list' d e
display "`list'"
local n: word count `list'
display `n'
local third: word 3 of `list'
display "`third'"
local lbl: variable label income
display "`lbl'"
local vl: value label sex
display "`vl'"
local l1: label (sex) 1
display "`l1'"
local l2: label sex 2
display "`l2'"
local typ: type income
display "`typ'"
local fmt: format income
display "`fmt'"
local len: length local list
display `len'
local s: subinstr local list "b" "B"
display "`s'"
local sa: subinstr local list " " "", all
display "`sa'"
local p: list posof "c" in list
display `p'
local u: list uniq list
display "`u'"
local srt: list sort list
display "`srt'"
local l3 a b c x
local both: list list & l3
display "`both'"
local either: list list | l3
display "`either'"
local minus: list list - l3
display "`minus'"
local sz: list sizeof list
display `sz'
local d: display %9.2f 3.14159
display "`d'"
local up = upper("abc")
display "`up'"
local pr = proper("hello world")
display "`pr'"
local now = "`c(os)'" != ""
display `now'
local pwdok = "`c(pwd)'" != ""
display `pwdok'
display `=2+2'
display "`=sqrt(16)'"
display `=_N'
display "`:word 2 of `list''"
display "`: word count `list''"
global root "mydir"
display "$root/data"
display "${root}_x"
local nums
forvalues q = 1/5 {
    local nums `nums' `q'
}
display "`nums'"
local sum = 0
foreach q of local nums {
    local sum = `sum' + `q'
}
display `sum'
local i = 1
while `i' <= 3 {
    display `i'
    local ++i
}
if `sum' > 10 {
    display "big"
}
else {
    display "small"
}
if `sum' > 100 display "huge"
else display "not huge"
levelsof edu, local(K)
display "`K'"
foreach k of local K {
    summarize income if edu == `k', meanonly
    display "`k': " r(mean)
}
unab wl: wor*
display "`wl'"
local nw: word count `wl'
display `nw'
ds wor*
display "`r(varlist)'"
ds, has(type string)
quietly summarize income
return list
display r(mean)
local m = r(mean)
display `m'
quietly regress income yedu
ereturn list
display e(N)
display "`e(cmd)'"
display "`e(depvar)'"
matrix list e(b)
matrix b = e(b)
display b[1,1]
matrix V = e(V)
display sqrt(V[1,1])
display colsof(b)
scalar s1 = 5
scalar s2 = s1^2
display s2
scalar drop s1
tempvar t
generate `t' = income/1000
summarize `t'
tempname sc
scalar `sc' = 3
display `sc'
* 12.2.3 programs
capture program drop hello
program hello
    display "hello, world"
end
hello
capture program drop hello2
program hello2
    display "hello, `1' and `2'"
    display "all: `0'"
end
hello2 John Paul
capture program drop mymean
program mymean, rclass
    summarize `1', meanonly
    return scalar mean = r(mean)
    return local var `1'
end
mymean income
return list
display r(mean)
capture program drop denscomp2
program denscomp2
    syntax varname(numeric) [if] [in], by(varname) [ at(integer 50) ]
    marksample touse
    summarize `varlist' if `touse', meanonly
    display "`varlist' by `by' at `at': " r(mean)
    count if `touse'
end
denscomp2 income, by(sex)
denscomp2 income if sex == 1, by(sex) at(20)
denscomp2 income in 1/100, by(sex)
capture program drop mysum
program mysum, rclass
    syntax varlist(min=1) [if] [in] [, Detail noMEAN Level(real 95) GENerate(name) *]
    marksample touse, novarlist
    foreach v of local varlist {
        quietly summarize `v' if `touse', `detail'
        display "`v': N = " r(N) " mean = " r(mean)
    }
    display "level `level' mean `mean' detail `detail' gen `generate' rest `options'"
    return scalar k = `: word count `varlist''
end
mysum income rent
mysum income rent, detail level(90)
mysum income if sex == 2, nomean gen(newv) foo(bar)
display r(k)
capture program drop cnt
program cnt
    args v val
    count if `v' == `val'
end
cnt sex 1
capture program drop shifter
program shifter
    while "`1'" != "" {
        display "`1'"
        macro shift
    }
end
shifter a b c
capture program drop tok
program tok
    tokenize `0'
    display "`2' `1'"
    gettoken first rest : 0
    display "`first' | `rest'"
end
tok alpha beta gamma
capture program drop chk
program chk
    capture confirm variable `1'
    if _rc {
        display "no variable `1'"
        exit
    }
    confirm numeric variable `1'
    display "`1' ok"
end
chk income
chk nosuchvar
capture noisily summarize nosuchvar
display _rc
capture confirm new variable income
display _rc
capture assert income > 0 if !missing(income)
display _rc
assert sex == 1 | sex == 2
generate owner = renttype == 1 if renttype < .
quietly logit owner yedu
p2
preserve
keep if sex == 1
count
restore
count
* p2.ado: Aldrich-Nelson
display e(chi2)/(e(chi2) + e(N))
* 12.3.6 unknown number of variables
levelsof edu, local(K)
foreach k of local K {
    generate byte edu_`k' = edu == `k' if !missing(edu)
    label variable edu_`k' "`: label (edu) `k''"
}
summarize edu_*
tabulate edu, generate(ed)
summarize ed1-ed5
separate income, by(sex)
summarize income1 income2
xi i.edu, noomit
summarize _Iedu*
